"""The semantic axis check for 77 GHz — certifies fast-time vs chirp axis assignment.

The on-disk cube has two size-256 axes (fast-time and chirps) that are indistinguishable
by shape, so the loader can only *assume* a mapping; a fast<->chirp interchange would pass
every shape assertion and silently ruin every downstream result. This module is the guard:
it looks at where power actually sits — range-gate energy along the assumed fast axis, and
near-zero-Doppler concentration along the assumed chirp axis (a quasi-static seated subject)
— and returns ACCEPTED only when that structure appears, on the RAW pre-MTI cube (MTI would
strip the near-zero-Doppler energy the check keys on).

`range_gate_bins`, `axis_metrics`, `axis_verdict` are promoted from experiments/audit_77ghz.py
(which imports them back). New here: `certify_axis` (the per-file entrypoint) and
`axis_spec_hash` (the stable certification key that a path-only config overlay must not change).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.signal.windows import hann

from pathlib import Path

from ..config import SPEED_OF_LIGHT_M_S
from ..data.loader_77ghz import N_CHIRPS, N_FAST, N_RX, load_77ghz_file

# Frozen axis-check thresholds (M2 audit; audit-only diagnostics that never enter CV).
AXIS_DC_HALFWIDTH = 3  # +-3 bins around zero Doppler
AXIS_MIN_DC_FRACTION = 0.5  # A1: min D(chirp)
AXIS_MIN_GATE_FRACTION = 0.05  # A2: min G(fast)
AXIS_DOMINANCE_FACTOR = 10.0  # A3: G(fast) must dominate G(chirp)

# Bumping this invalidates every stored certificate on purpose (the algorithm changed).
AXIS_ALGO_VERSION = "1"

# The loaded-cube axis assignment this check certifies: [frame, fast, chirp, rx].
FAST_AXIS = 1
CHIRP_AXIS = 2


def range_gate_bins(n_fast: int, bandwidth_hz: float, gate_m) -> np.ndarray:
    """Range-FFT bin indices covering the gate. dr = c / (2B) per the reference."""
    dr = SPEED_OF_LIGHT_M_S / (2.0 * bandwidth_hz)
    ranges = np.arange(n_fast) * dr
    return np.flatnonzero((ranges >= gate_m[0]) & (ranges <= gate_m[1]))


def _mean_power_spectrum(cube: np.ndarray, axis: int) -> np.ndarray:
    """Power spectrum along `axis`, averaged over every other axis.

    Periodic Hann on the transformed axis (the QC-reference convention). The FULL
    spectrum is kept: these signals may be complex, so there is no Hermitian symmetry to
    exploit, and a shared full-spectrum denominator keeps the two candidate axes comparable.
    """
    n = cube.shape[axis]
    shape = [1] * cube.ndim
    shape[axis] = n
    windowed = cube * hann(n, sym=False).reshape(shape)
    power = np.abs(np.fft.fft(windowed, axis=axis)) ** 2
    other = tuple(a for a in range(cube.ndim) if a != axis)
    return power.mean(axis=other)


def axis_metrics(cube: np.ndarray, fast_axis: int, chirp_axis: int,
                 gate_bins: np.ndarray, dc_halfwidth: int) -> dict:
    """G and D for BOTH candidate axis assignments, computed identically.

    G(X) = fraction of power in the range-gate bins of the unshifted FFT along X.
    D(X) = fraction of power within +-dc_halfwidth bins of zero in the fftshifted
           spectrum along X.
    The proposed assignment expects G(fast) material, D(chirp) dominant, G(chirp) ~ 0.
    D(fast) is recorded but is NOT a standalone discriminator (close-in TX leakage can
    legitimately concentrate range power near bin 0); it only participates in the mirrored
    swapped-axis hypothesis.
    """
    out = {}
    for label, axis in (("fast", fast_axis), ("chirp", chirp_axis)):
        spectrum = _mean_power_spectrum(cube, axis)
        total = float(spectrum.sum())
        n = spectrum.shape[0]
        centre = n // 2
        shifted = np.fft.fftshift(spectrum)
        dc = shifted[max(0, centre - dc_halfwidth) : centre + dc_halfwidth + 1]
        out[f"G_{label}"] = float(spectrum[gate_bins].sum() / total) if total > 0 else 0.0
        out[f"D_{label}"] = float(dc.sum() / total) if total > 0 else 0.0
    return out


def axis_verdict(metrics: dict, *, min_dc=AXIS_MIN_DC_FRACTION,
                 min_gate=AXIS_MIN_GATE_FRACTION, dominance=AXIS_DOMINANCE_FACTOR) -> str:
    """ACCEPTED / REJECTED / INCONCLUSIVE, evaluating both assignments symmetrically.

    Failing the proposed assignment's criteria does not by itself prove the axes are
    swapped — low SNR or an unrepresentative file would look the same — so REJECTED is
    reserved for positive evidence favouring the swap.
    """
    proposed = (
        metrics["D_chirp"] >= min_dc
        and metrics["G_fast"] >= min_gate
        and metrics["G_fast"] >= dominance * metrics["G_chirp"]
    )
    swapped = (
        metrics["D_fast"] >= min_dc
        and metrics["G_chirp"] >= min_gate
        and metrics["G_chirp"] >= dominance * metrics["G_fast"]
    )
    if proposed and not swapped:
        return "ACCEPTED"
    if swapped and not proposed:
        return "REJECTED"
    return "INCONCLUSIVE"


def certify_axis(cube: np.ndarray, pre77) -> tuple[str, dict]:
    """Per-file semantic certification of the loaded cube [frame, fast, chirp, rx].

    Runs on the RAW pre-MTI cube. Returns (verdict, metrics). The caller (run_qc77 / the
    per-entrypoint guard) fails closed on any verdict other than 'ACCEPTED' — REJECTED and
    INCONCLUSIVE both fail: an inconclusive shape-indistinguishable mapping is not a
    certification.
    """
    gate_bins = range_gate_bins(cube.shape[FAST_AXIS], pre77.bandwidth_hz, pre77.gate_m)
    metrics = axis_metrics(cube, FAST_AXIS, CHIRP_AXIS, gate_bins, AXIS_DC_HALFWIDTH)
    return axis_verdict(metrics), metrics


def axis_spec_hash(config) -> str:
    """A stable key over exactly the inputs that change the axis verdict.

    Includes the algorithm version, the verdict thresholds, the expected shape +
    representation, and the gate/bandwidth used to place the range-gate bins. EXCLUDES
    environment-specific fields (paths.*, results_dir, device), so a path-only overlay
    (e.g. ibex.yaml) leaves a certificate valid after rsync, while changing any
    axis-relevant constant invalidates it.
    """
    pre77 = config.preprocess77
    spec = {
        "algo_version": AXIS_ALGO_VERSION,
        "thresholds": {
            "AXIS_MIN_DC_FRACTION": AXIS_MIN_DC_FRACTION,
            "AXIS_MIN_GATE_FRACTION": AXIS_MIN_GATE_FRACTION,
            "AXIS_DOMINANCE_FACTOR": AXIS_DOMINANCE_FACTOR,
            "AXIS_DC_HALFWIDTH": AXIS_DC_HALFWIDTH,
        },
        "expected_shape": [N_RX, N_CHIRPS, N_FAST],
        "representation": "real_float64",
        "gate_m": list(pre77.gate_m),
        "bandwidth_hz": pre77.bandwidth_hz,
        "fs_hz": pre77.fs_hz,
    }
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AxisCertError(RuntimeError):
    """Raised when a file has no ACCEPTED axis certificate and fails the inline check."""


def require_accepted_axis(path, config, *, survival_csv=None, load=load_77ghz_file) -> dict:
    """Hard guard: this file must carry an ACCEPTED axis mapping before any feature is written.

    Every extraction / preprocessing entrypoint (and the smoke) calls this per file, so a
    file that reached a CLI before or without the cohort QC artifact can never be processed
    from an uncertified axis mapping. It resolves in two ways (C5-08):

      (i) a matching ACCEPTED record in `survival_csv` keyed to the file's raw sha256 AND the
          config's axis_spec_hash — so a path-only overlay keeps the certificate valid but any
          axis-relevant change invalidates it; OR
      (ii) the raw semantic check run inline on the loaded cube (one whole-cube FFT).

    Any non-ACCEPTED verdict, or a record whose sha256/axis_spec_hash does not match, aborts.
    """
    from ..provenance import sha256_file

    path = Path(path)
    sha = sha256_file(path)
    spec_hash = axis_spec_hash(config)

    if survival_csv is not None and Path(survival_csv).exists():
        import pandas as pd

        df = pd.read_csv(survival_csv)
        if {"sha256", "axis_spec_hash", "axis_verdict"} <= set(df.columns):
            match = df[(df["sha256"] == sha) & (df["axis_spec_hash"] == spec_hash)]
            if not match.empty:
                if (match["axis_verdict"] == "ACCEPTED").all():
                    return {"source": "record", "verdict": "ACCEPTED", "sha256": sha,
                            "axis_spec_hash": spec_hash}
                raise AxisCertError(
                    f"{path.name}: axis certificate in {survival_csv} is not ACCEPTED "
                    f"({sorted(match['axis_verdict'].unique())}) — refusing to extract"
                )

    cube = load(path)
    verdict, metrics = certify_axis(cube, config.preprocess77)
    if verdict != "ACCEPTED":
        raise AxisCertError(
            f"{path.name}: inline axis check returned {verdict} (not ACCEPTED) — the "
            "fast/chirp mapping is uncertified; refusing to extract"
        )
    return {"source": "inline", "verdict": verdict, "sha256": sha,
            "axis_spec_hash": spec_hash, "axis_metrics": metrics}
