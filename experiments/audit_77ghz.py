"""Milestone 2 — the minimal 77 GHz audit, on ONE real file.

Confirms (or refutes) the 77 GHz facts the milestone-5 freeze depends on and records
everything in a provenance-complete JSON. It deliberately does NOT create the
production `loader_77ghz.py` (milestone 9 builds that against these confirmed facts),
extracts no features, and reads bounded slabs of a single file.

    uv run python experiments/audit_77ghz.py --config configs/exp_a_regression.yaml

**Nothing here selects a threshold or a rule.** The audited subject is a future
outer-test subject, so every 77 GHz QC constant this exercises is frozen a priori in
plans/MILESTONE_2_PLAN.md; the audit validates the frozen choices and characterises
the data. A degenerate or contradictory result is a stop-and-report, never a silent
revision.

Numerical helpers are pure and parameterised so tests/test_audit_77ghz.py can drive
them on small synthetic fixtures; only `main` applies the frozen 77 GHz constants.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import hann

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import SPEED_OF_LIGHT_M_S, load_config  # noqa: E402
from dehyd.data.loader_77ghz import reverse_axes, to_numeric  # noqa: E402
from dehyd.provenance import _git_info, sha256_file  # noqa: E402
from dehyd.qc.axis_check_77 import (  # noqa: E402
    AXIS_DC_HALFWIDTH,
    AXIS_DOMINANCE_FACTOR,
    AXIS_MIN_DC_FRACTION,
    AXIS_MIN_GATE_FRACTION,
    axis_metrics,
    axis_verdict,
    range_gate_bins,
)
from dehyd.qc.screens_77 import qc_in_band_mask_77, qc_smoke_frame  # noqa: E402

RADAR_VAR = "framesRadar"

# ---------------------------------------------------------------- frozen constants
# From matlab/77ghz_code/{filter_gpt_butterworth77,chirpavg_and_fuse_batch}.m — NOT
# the 10 GHz values.
FS_HZ = 500e3
BANDWIDTH_HZ = 2e9
CHIRP_TIME_S = 512e-6
PRF_HZ = 1.0 / CHIRP_TIME_S  # 1953.125 — slow-time sample rate
RANGE_GATE_M = (2.0, 4.0)
EXPECTED_SHAPE = (16, 256, 256, 125)  # (Nrx, Nchirps, Nfast, Nframes) as h5py presents it

# QC rules frozen a priori (MILESTONE_2_PLAN §2.6 step 4; implementation_plan.md Exp G).
QC_HISTOGRAM_BINS = 128
QC_FLATLINE_MAX_BIN_FRACTION = 0.25
QC_MIN_IN_BAND_RATIO = 0.30
QC_IN_BAND_MARGIN_HZ = FS_HZ / 256  # one FFT bin = 1953.125 Hz

# The axis-check constants (AXIS_DC_HALFWIDTH, AXIS_MIN_DC_FRACTION, AXIS_MIN_GATE_FRACTION,
# AXIS_DOMINANCE_FACTOR) are now defined once in dehyd.qc.axis_check_77 and imported above.
CHAIN_MIN_ENERGY_RATIO = 1e-9
QC_SMOKE_MIN_MEDIAN_RATIO = 0.01

DEFAULT_N_FRAMES = 10
MIN_FINITE_FRAME_FRACTION = 0.5


class AuditError(RuntimeError):
    """Raised when the audit cannot proceed on structural grounds."""


# ------------------------------------------------------------------------ structure


def describe_storage(dset) -> dict:
    """Observed HDF5 metadata plus the H1-storage verdict.

    Accepted representations, fixed in advance:
      * a plain real float dtype (float32/float64) -- what MATLAB writes for a
        real-sampled ADC capture; complex I/Q then arises only after the range FFT;
      * an HDF5 compound with exactly two float fields `real` and `imag` of equal
        width -- the MAT v7.3 convention for a complex array (as the 10 GHz files use).
    Anything else -- integer fields, other field names, native HDF5 complex -- is
    REJECTED rather than silently coerced. Supporting native complex would be a
    deliberate implementation, and it is deliberately not implemented here.
    """
    dtype = dset.dtype
    info = {
        "dtype_str": dtype.str,
        "dtype_name": str(dtype),
        "field_names": list(dtype.names) if dtype.names else None,
        "byteorder": dtype.byteorder,
        "itemsize": int(dtype.itemsize),
        "shape": list(dset.shape),
        "chunks": list(dset.chunks) if dset.chunks else None,
        "compression": dset.compression,
    }

    if dtype.names is None:
        accepted = dtype.kind == "f"
        info["representation"] = "real_float" if accepted else "unsupported"
    else:
        fields = list(dtype.names)
        widths = {dtype[n].kind for n in fields}
        sizes = {dtype[n].itemsize for n in fields}
        accepted = fields == ["real", "imag"] and widths == {"f"} and len(sizes) == 1
        info["representation"] = "compound_complex" if accepted else "unsupported"
        info["field_dtypes"] = {n: str(dtype[n]) for n in fields}

    info["verdict"] = "ACCEPTED" if accepted else "REJECTED"
    return info


def read_frames(dset, n_frames: int, block_size: int | None = None) -> np.ndarray:
    """Read the first `n_frames` along the LAST (frame) axis, in bounded slabs.

    The whole dataset is never materialised. Every request is an explicit 4-tuple with
    full slices on the non-frame axes (Rx/chirp/fast-time must be read completely for
    each slab) and a BOUNDED slice on the frame axis -- never an ellipsis, never the
    full frame extent. tests/test_audit_77ghz.py drives this with a spy dataset that
    raises if the contract is broken.
    """
    n_total = dset.shape[-1]
    if not 1 <= n_frames <= n_total:
        raise AuditError(f"--n-frames must be in [1, {n_total}], got {n_frames}")

    block_size = block_size or n_frames
    blocks = []
    for start in range(0, n_frames, block_size):
        stop = min(start + block_size, n_frames)
        blocks.append(dset[:, :, :, start:stop])
    return np.concatenate(blocks, axis=-1)


# reverse_axes / to_numeric are now defined once in dehyd.data.loader_77ghz and imported
# above (the milestone-5 promotion). tests/test_audit_77ghz.py still exercises them via
# `audit.reverse_axes` / `audit.to_numeric`.


def finite_frame_mask(cube: np.ndarray) -> np.ndarray:
    """True for frames with no non-finite sample. Frame-level, like the QC NaN/Inf screen."""
    return np.array([bool(np.all(np.isfinite(frame))) for frame in cube])


# The semantic axis check (range_gate_bins, axis_metrics, axis_verdict) is now defined once
# in dehyd.qc.axis_check_77, and the 77 GHz QC smoke (qc_in_band_mask_77, qc_smoke_frame) in
# dehyd.qc.screens_77 — both imported above (the milestone-5 promotion). This module keeps
# the chain/energy accounting and the provenance-bearing report, which have no production home.


# ------------------------------------------------- proposed chain + energy accounting


def normalized_energy(x: np.ndarray, transform_length: int = 1) -> float:
    """Total energy under the Parseval convention.

    numpy's FFT is unnormalised (a length-N transform scales total spectral energy by
    N), so spectral energies are divided by the CUMULATIVE transform length. Without
    this, stage-to-stage ratios are not comparable and the 1e-9 floor is meaningless.
    """
    return float(np.sum(np.abs(x) ** 2) / transform_length)


def proposed_chain(cube: np.ndarray, gate_bins: np.ndarray, *, fs_hz: float,
                   bandwidth_hz: float, chirp_time_s: float, gate_m,
                   butter_order: int = 4) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """The operations later code will actually use (Exp G chain steps 1-5 + RD map).

    (a) primary pre-WST gated range cube: MTI -> fast-time Butterworth -> Hann(fast)
        -> range FFT -> crop to the gate;
    (b) range-Doppler diagnostic: (a) -> Hann(slow) -> Doppler FFT -> fftshift.

    Symmetric Hann here (MATLAB's `hann(N)` default, which is what
    chirpavg_and_fuse_batch.m calls) -- deliberately NOT the periodic window the QC
    screens use.
    """
    n_fast, n_chirps = cube.shape[1], cube.shape[2]
    stages = [{"stage": "raw_slab", "energy": normalized_energy(cube)}]

    # (1) MTI: subtract the per-fast-bin mean over chirps (static clutter removal).
    mti = cube - cube.mean(axis=2, keepdims=True)
    stages.append({"stage": "mti", "energy": normalized_energy(mti)})

    # (2) fast-time zero-phase Butterworth over the gate's beat band.
    hz_per_m = 2.0 * (bandwidth_hz / chirp_time_s) / SPEED_OF_LIGHT_M_S
    wn = [hz_per_m * gate_m[0] / (fs_hz / 2), hz_per_m * gate_m[1] / (fs_hz / 2)]
    sos = butter(butter_order, wn, btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, mti, axis=1)
    stages.append({"stage": "bandpass", "energy": normalized_energy(filtered)})

    # (3-4) Hann(fast) + range FFT.
    windowed = filtered * hann(n_fast, sym=True).reshape(1, n_fast, 1, 1)
    range_fft = np.fft.fft(windowed, axis=1)
    stages.append({"stage": "range_fft", "energy": normalized_energy(range_fft, n_fast)})

    # (5) crop to the gate.
    gated = range_fft[:, gate_bins, :, :]
    stages.append({"stage": "range_gate_crop", "energy": normalized_energy(gated, n_fast)})

    # (b) Hann(slow) + Doppler FFT -> cumulative normalisation n_fast * n_chirps.
    slow_windowed = gated * hann(n_chirps, sym=True).reshape(1, 1, n_chirps, 1)
    rd = np.fft.fftshift(np.fft.fft(slow_windowed, axis=2), axes=2)
    stages.append(
        {"stage": "range_doppler", "energy": normalized_energy(rd, n_fast * n_chirps)}
    )
    return gated, rd, stages


# ------------------------------------------------------------------------- reporting


def json_safe(value):
    """Encode unavailable/non-finite floats as null so the JSON stays standard.

    Written with allow_nan=False, so anything unhandled fails loudly at write time
    rather than producing an artifact no strict parser can read.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _summary(values) -> dict:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(finite),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--file", default="data/77ghz/subject_1_8am.mat")
    parser.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES)
    parser.add_argument(
        "--out", default="audit_77ghz.json",
        help="bare filename; always resolved under <results_dir>/qc",
    )
    args = parser.parse_args(argv)

    if Path(args.out).name != args.out:
        parser.error("--out must be a bare filename; the config owns the output path")

    config = load_config(*args.config)
    out_dir = Path(config.paths.results_dir) / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out

    path = Path(args.file)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path

    findings: dict = {
        "audit": "77ghz-minimal",
        "milestone": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": args.file,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "git": _git_info(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
        },
        "constants": {
            "fs_hz": FS_HZ, "bandwidth_hz": BANDWIDTH_HZ, "chirp_time_s": CHIRP_TIME_S,
            "prf_hz": PRF_HZ, "range_gate_m": list(RANGE_GATE_M),
            "range_bin_spacing_m": SPEED_OF_LIGHT_M_S / (2 * BANDWIDTH_HZ),
            "expected_shape": list(EXPECTED_SHAPE),
            "qc_histogram_bins": QC_HISTOGRAM_BINS,
            "qc_flatline_max_bin_fraction": QC_FLATLINE_MAX_BIN_FRACTION,
            "qc_min_in_band_ratio": QC_MIN_IN_BAND_RATIO,
            "qc_in_band_margin_hz": QC_IN_BAND_MARGIN_HZ,
        },
        "audit_thresholds": {
            "A1_min_D_chirp": AXIS_MIN_DC_FRACTION,
            "A2_min_G_fast": AXIS_MIN_GATE_FRACTION,
            "A3_dominance_factor": AXIS_DOMINANCE_FACTOR,
            "dc_halfwidth_bins": AXIS_DC_HALFWIDTH,
            "chain_min_energy_ratio": CHAIN_MIN_ENERGY_RATIO,
            "qc_smoke_min_median_ratio": QC_SMOKE_MIN_MEDIAN_RATIO,
        },
        "conventions": {
            "qc_and_semantic_window": "periodic Hann (scipy hann sym=False)",
            "chain_window": "symmetric Hann (MATLAB hann(N) default)",
            "semantic_spectrum": "full spectrum, shared denominator for both axes",
            "semantic_G": "unshifted FFT bins over the range gate",
            "semantic_D": f"fftshifted, +-{AXIS_DC_HALFWIDTH} bins around zero",
            "qc_spectrum": "non-negative half-spectrum, DC included, Nyquist excluded",
            "energy_normalization": "Parseval; spectral energy / cumulative transform length",
        },
        "verdicts": {},
    }

    def write_and_exit(code: int) -> int:
        out_path.write_text(
            json.dumps(json_safe(findings), indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"findings     : {out_path}")
        for name, verdict in findings["verdicts"].items():
            print(f"  {name:<12}: {verdict}")
        return code

    with h5py.File(path, "r") as handle:
        if RADAR_VAR not in handle:
            findings["error"] = f"no '{RADAR_VAR}' variable (found {list(handle)})"
            findings["verdicts"] = {
                "H1_shape": "REJECTED", "H1_storage": "REJECTED",
                "H1_axes": "NOT_RUN", "qc_smoke": "NOT_RUN", "chain": "NOT_RUN",
            }
            return write_and_exit(1)

        dset = handle[RADAR_VAR]
        storage = describe_storage(dset)
        findings["storage"] = storage
        findings["verdicts"]["H1_shape"] = (
            "ACCEPTED" if tuple(dset.shape) == EXPECTED_SHAPE else "REJECTED"
        )
        findings["verdicts"]["H1_storage"] = storage["verdict"]

        if "REJECTED" in (findings["verdicts"]["H1_shape"], findings["verdicts"]["H1_storage"]):
            findings["verdicts"].update(
                {"H1_axes": "NOT_RUN", "qc_smoke": "NOT_RUN", "chain": "NOT_RUN"}
            )
            return write_and_exit(1)

        n_frames = args.n_frames
        n_total = dset.shape[-1]
        if not 1 <= n_frames <= n_total:
            parser.error(f"--n-frames must be in [1, {n_total}], got {n_frames}")
        print(f"reading {n_frames}/{n_total} frames (bounded slabs)...")
        raw = read_frames(dset, n_frames)

    cube = reverse_axes(to_numeric(raw))
    del raw

    min_finite = math.ceil(MIN_FINITE_FRAME_FRACTION * n_frames)
    finite = finite_frame_mask(cube)
    findings["slab"] = {
        "requested_frame_count": n_frames,
        "min_finite_frames": min_finite,
        "selected_frame_indices": list(range(n_frames)),
        "excluded_frame_indices": np.flatnonzero(~finite).tolist(),
        "effective_frame_indices": np.flatnonzero(finite).tolist(),
        "reversed_shape": list(cube.shape),
        "dtype": str(cube.dtype),
        "abs_min": float(np.abs(cube).min()),
        "abs_median": float(np.median(np.abs(cube))),
        "abs_max": float(np.abs(cube).max()),
        "n_non_finite_samples": int(np.count_nonzero(~np.isfinite(cube))),
    }

    gate_bins = range_gate_bins(cube.shape[1], BANDWIDTH_HZ, RANGE_GATE_M)
    findings["constants"]["range_gate_bins"] = [int(gate_bins[0]), int(gate_bins[-1])]
    qc_mask = qc_in_band_mask_77(
        cube.shape[1], FS_HZ, BANDWIDTH_HZ, CHIRP_TIME_S, RANGE_GATE_M, QC_IN_BAND_MARGIN_HZ
    )
    mask_bins = np.flatnonzero(qc_mask)
    findings["constants"]["qc_mask_bins"] = [int(mask_bins[0]), int(mask_bins[-1])]

    # --- QC smoke runs on ALL slab frames: reporting non-finite frames is its job ----
    qc_results = [
        qc_smoke_frame(cube[i], qc_mask, bins=QC_HISTOGRAM_BINS,
                       max_bin_fraction=QC_FLATLINE_MAX_BIN_FRACTION,
                       min_ratio=QC_MIN_IN_BAND_RATIO)
        for i in range(cube.shape[0])
    ]
    ratios = [r["in_band_ratio"] for r in qc_results]
    n_pass = sum(r["passed"] for r in qc_results)
    per_rx_total = np.sum([r["per_rx_flatline"] for r in qc_results], axis=0).tolist()
    median_ratio = _summary(ratios)["median"]
    findings["qc_smoke"] = {
        "n_frames": len(qc_results),
        "n_pass": n_pass,
        "n_nan_inf": sum(r["nan_inf"] for r in qc_results),
        "n_flatline": sum(r["flatline"] for r in qc_results),
        "n_low_in_band": sum(r["low_in_band"] for r in qc_results),
        "in_band_ratio": _summary(ratios),
        "per_rx_flatline_trace_counts": per_rx_total,
        "flatline_multiplicity_vs_10ghz": (16 * 256) / 20,
    }
    findings["verdicts"]["qc_smoke"] = (
        "NON_DEGENERATE"
        if n_pass >= 1 and median_ratio is not None and median_ratio >= QC_SMOKE_MIN_MEDIAN_RATIO
        else "DEGENERATE"
    )

    if int(finite.sum()) < min_finite:
        findings["verdicts"]["H1_axes"] = "NOT_RUN"
        findings["verdicts"]["chain"] = "NOT_RUN"
        return write_and_exit(1)

    effective = cube[finite]

    # --- H1-axes: RAW data, BEFORE any clutter subtraction --------------------------
    # MTI subtracts the per-fast-bin mean over chirps; a seated subject is quasi-static,
    # so nearly all of its energy sits at near-zero Doppler and MTI would remove exactly
    # the structure this check looks for. Hence: raw slab, and MTI lives only in
    # proposed_chain() below, which runs after.
    metrics = axis_metrics(effective, fast_axis=1, chirp_axis=2,
                           gate_bins=gate_bins, dc_halfwidth=AXIS_DC_HALFWIDTH)
    findings["axis_metrics"] = metrics
    findings["verdicts"]["H1_axes"] = axis_verdict(metrics)

    # --- proposed chain -------------------------------------------------------------
    _, _, stages = proposed_chain(
        effective, gate_bins, fs_hz=FS_HZ, bandwidth_hz=BANDWIDTH_HZ,
        chirp_time_s=CHIRP_TIME_S, gate_m=RANGE_GATE_M,
    )
    raw_energy = stages[0]["energy"]
    for stage in stages:
        stage["ratio_to_raw"] = stage["energy"] / raw_energy if raw_energy else float("nan")
    findings["chain_stages"] = stages
    final = [s for s in stages if s["stage"] in ("range_gate_crop", "range_doppler")]
    findings["verdicts"]["chain"] = (
        "NON_DEGENERATE"
        if all(math.isfinite(s["energy"]) and s["ratio_to_raw"] >= CHAIN_MIN_ENERGY_RATIO
               for s in final)
        else "DEGENERATE"
    )

    print(f"  G_fast={metrics['G_fast']:.4f}  G_chirp={metrics['G_chirp']:.4g}")
    print(f"  D_chirp={metrics['D_chirp']:.4f}  D_fast={metrics['D_fast']:.4f}")
    print(f"  qc smoke: {n_pass}/{len(qc_results)} pass, median in-band {median_ratio:.4f}")
    for stage in stages:
        print(f"  energy {stage['stage']:<16} ratio_to_raw={stage['ratio_to_raw']:.3e}")

    ok = all(
        findings["verdicts"][k] in ("ACCEPTED", "NON_DEGENERATE")
        for k in ("H1_shape", "H1_storage", "H1_axes", "qc_smoke", "chain")
    )
    return write_and_exit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
