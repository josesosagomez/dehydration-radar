"""Milestone 3 — run the frozen preprocessing sequence over the eligible 10 GHz cohort.

Writes the per-session diagnostics that characterise what preprocessing actually does
to this dataset (where the dominant beat sits, how much energy the band gate keeps,
how concentrated the detected peak is), plus a provenance record.

    uv run python experiments/run_preprocess.py --config configs/exp_a_regression.yaml

**Diagnostic only — this script selects nothing.** Every constant is frozen before it
runs. A surprising distribution is a finding for HISTORY.md and the owner, never a
licence to retune (the M2 doctrine).

The correctness-critical logic lives in pure helpers so it is testable without a cohort
run (the M2 audit pattern); `main()` only sequences them.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import PreprocessConfig  # noqa: E402
from dehyd.config import load_config  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.loader_10ghz import load_10ghz_file  # noqa: E402
from dehyd.data.manifest import (  # noqa: E402
    apply_qc,
    build_manifest,
    eligible_frames,
    resolve_path,
)
from dehyd.preprocess.filters import apply_band_gate, filter_spec  # noqa: E402
from dehyd.preprocess.pipeline import preprocess_frame  # noqa: E402
from dehyd.preprocess.reduce import detect_option_b_peak, option_b_roi_bins  # noqa: E402
from dehyd.provenance import record_run  # noqa: E402

REPORT_NAME = "preprocess_diagnostics_10ghz.csv"

# The one configuration this script is allowed to characterise (see check_primary_spec).
CANONICAL_PRIMARY = PreprocessConfig()


class NotThePrimarySpec(ValueError):
    """Raised when a non-canonical preprocessing config would write the primary CSV."""


def check_primary_spec(pre: PreprocessConfig) -> None:
    """Refuse to run unless `pre` is exactly the canonical primary specification.

    The curated CSV is the PRIMARY artifact and every non-primary result must be
    explicitly labelled, so a run that is not the primary must not be able to overwrite
    it. Checking only the ablation switches would not be enough: `model_gate_m` of
    0.9-3.0 m is an inner-CV *candidate*, not the primary, and it changes both the ROI
    and the filter -- so it would silently replace the primary diagnostics with
    different numbers under a "primary" label.

    Comparing the whole frozen dataclass makes that one check, and names whatever
    deviates.
    """
    deviations = [
        f"{field.name}={getattr(pre, field.name)!r} (primary: {getattr(CANONICAL_PRIMARY, field.name)!r})"
        for field in dataclasses.fields(PreprocessConfig)
        if getattr(pre, field.name) != getattr(CANONICAL_PRIMARY, field.name)
    ]
    if deviations:
        raise NotThePrimarySpec(
            "run_preprocess.py writes the PRIMARY curated artifact and refuses a "
            "non-canonical preprocessing config; deviating fields: "
            + "; ".join(deviations)
        )


def energy_retention(raw_frame: np.ndarray, gated_frame: np.ndarray) -> float:
    """E_post / E_pre over the full complex cube, in the TIME domain.

    Parseval makes a spectral convention unnecessary here, so no FFT and no window
    enters this number -- which is exactly why it is comparable across implementations.
    Measured immediately after the band gate, before reduction and trim.

    An eligible frame cannot have zero energy (QC requires an in-band ratio >= 0.30,
    impossible at zero total power), so a zero denominator is an input-contract
    violation and raises rather than being silently guarded to 0.
    """
    e_pre = float(np.sum(np.abs(raw_frame) ** 2))
    if e_pre == 0.0:
        raise ValueError("raw frame has zero energy — not a QC-passing frame")
    return float(np.sum(np.abs(gated_frame) ** 2)) / e_pre


def concentration(detection) -> tuple[float, float]:
    """(roi_to_total, peak_share) from ONE detection result.

    Option-B's peak is inside the ROI by construction, so bin membership alone is no
    evidence that a meaningful dominant peak exists -- these two are.

      roi_to_total = sum(P[roi]) / sum(P)   -- how much of the spectrum is in the gate
      peak_share   = P[peak] / sum(P[roi])  -- how dominant the peak is within it

    `peak_share` is NaN when the ROI carries no power (the documented zero-ROI case):
    there is no detected peak to describe, and a fabricated 0 would be a lie.
    """
    total = float(detection.power.sum())
    roi_power = float(detection.power[detection.roi_bins].sum())
    roi_to_total = roi_power / total if total > 0 else 0.0
    peak_share = float(detection.power[detection.peak_bin] / roi_power) if roi_power > 0 else np.nan
    return roi_to_total, peak_share


def median_skipping_missing(values) -> float:
    """Median over the defined values; NaN only when every value is missing.

    `np.nanmedian` of an all-NaN array is NaN but warns, so the all-missing case is
    handled explicitly. A NaN here reaches the CSV as an empty cell -- never a
    fabricated 0.

    In practice an all-missing session should not occur: `peak_share` is undefined only
    when the ROI carries exactly zero power, and a frame with any energy at all leaves
    float-positive power there (while a zero-energy frame is rejected outright by
    `energy_retention`). The path is kept because "undefined" must stay distinguishable
    from "zero", not because it is expected.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmedian(values))


def lowest_mode(values) -> int:
    """The most frequent value, ties broken toward the LOWEST.

    Written out rather than delegated to pandas/scipy `mode`, whose tie behaviour can
    differ across versions -- the curated artifact must not depend on that.
    `np.unique` returns sorted values, so the first argmax of the counts is the lowest
    tied value by construction.
    """
    uniques, counts = np.unique(np.asarray(values), return_counts=True)
    return int(uniques[int(np.argmax(counts))])


def session_diagnostics(cube: np.ndarray, frame_indices, pre: PreprocessConfig) -> dict:
    """Every per-session number, from one pass over that session's eligible frames."""
    df_hz = pre.fs_hz / cube.shape[0]
    peaks: list[int] = []
    retentions: list[float] = []
    roi_fractions: list[float] = []
    peak_shares: list[float] = []
    all_finite = True

    for index in frame_indices:
        frame = cube[:, :, index]
        gated = apply_band_gate(frame, pre, axis=0)

        retentions.append(energy_retention(frame, gated))

        detection = detect_option_b_peak(gated, pre)
        peaks.append(detection.peak_bin)
        roi_fraction, peak_share = concentration(detection)
        roi_fractions.append(roi_fraction)
        peak_shares.append(peak_share)

        for reduction in ("a", "b"):
            for channel in ("mag", "iq"):
                out = preprocess_frame(frame, pre, reduction=reduction, channel=channel)
                all_finite &= bool(np.all(np.isfinite(out))) and out.shape[1] == (
                    cube.shape[0] - 2 * pre.edge_trim
                )

    shares = np.asarray(peak_shares, dtype=float)
    return {
        "n_eligible_frames": len(frame_indices),
        "peak_bin_mode": lowest_mode(peaks),
        "peak_bin_min": int(np.min(peaks)),
        "peak_bin_max": int(np.max(peaks)),
        "peak_hz_median": float(np.median(peaks) * df_hz),
        "energy_retention_median": float(np.median(retentions)),
        "roi_to_total_median": float(np.median(roi_fractions)),
        # Frames without a defined peak_share are skipped; an all-missing session
        # yields NaN, which pandas writes as an empty cell.
        "peak_share_median": median_skipping_missing(shares),
        "n_peak_share_missing": int(np.count_nonzero(np.isnan(shares))),
        "all_variants_finite": all_finite,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="PATH",
        help="config YAML; repeatable, later files win (use for per-machine overlays)",
    )
    args = parser.parse_args(argv)

    config = load_config(*args.config)
    pre = config.preprocess
    check_primary_spec(pre)  # before any I/O

    spec = filter_spec(pre)
    print(f"config       : {', '.join(args.config)}")
    print(
        f"band gate    : {spec['gate_method']} order {spec['butter_order']} "
        f"{spec['f_lo_hz']:.1f}-{spec['f_hi_hz']:.1f} Hz "
        f"(gate {tuple(pre.model_gate_m)} m, padlen {spec['padlen']})"
    )
    print(
        f"reduction    : options A/B; option-B ROI bins "
        f"{list(option_b_roi_bins(pre, 534))}, nb={pre.peak_neighbors}, "
        f"taper={pre.mask_taper}"
    )
    print(f"trim/standard: edge_trim {pre.edge_trim} -> 470 samples; {pre.standardize} z")

    gt = load_ground_truth(config.paths.weight_xlsx)
    manifest = build_manifest(config.paths, gt)
    manifest_qc = apply_qc(manifest, config.paths, config)
    population = eligible_frames(manifest_qc)
    print(
        f"population   : {len(population)} eligible frames, "
        f"{population.groupby(['subject', 'session_idx']).ngroups} sessions, "
        f"{population.subject.nunique()} subjects"
    )

    rows = []
    for (subject, session_idx), group in population.groupby(["subject", "session_idx"]):
        rel_path = group["rel_path"].iloc[0]
        cube = load_10ghz_file(resolve_path(config.paths, rel_path))
        diagnostics = session_diagnostics(cube, group["frame_idx"].tolist(), pre)
        rows.append(
            {
                "subject": int(subject),
                "session_idx": int(session_idx),
                "session_name": group["session_name"].iloc[0],
                **diagnostics,
            }
        )
        print(
            f"  s{subject:<2} {group['session_name'].iloc[0]:<5} "
            f"n={diagnostics['n_eligible_frames']:<3} "
            f"peak bin {diagnostics['peak_bin_mode']} "
            f"({diagnostics['peak_hz_median']:.0f} Hz) "
            f"retention {diagnostics['energy_retention_median']:.3f} "
            f"roi/total {diagnostics['roi_to_total_median']:.3f} "
            f"peak_share {diagnostics['peak_share_median']:.3f}"
        )

    report = pd.DataFrame(rows)

    hz_per_m = 2.0 * (pre.bandwidth_hz / pre.chirp_time_s) / 299_792_458.0
    print("\ncohort summary")
    print(f"  sessions            : {len(report)}")
    print(f"  peak bin (mode)     : {lowest_mode(report['peak_bin_mode'])}")
    print(
        f"  peak Hz median      : {report['peak_hz_median'].median():.1f} "
        f"(= {report['peak_hz_median'].median() / hz_per_m:.2f} m)"
    )
    print(
        f"  energy retention    : median {report['energy_retention_median'].median():.3f} "
        f"[{report['energy_retention_median'].min():.3f}, "
        f"{report['energy_retention_median'].max():.3f}]"
    )
    print(f"  roi/total median    : {report['roi_to_total_median'].median():.3f}")
    print(f"  peak_share median   : {report['peak_share_median'].median():.3f}")
    print(f"  all variants finite : {bool(report['all_variants_finite'].all())}")

    # The config is the single output-path authority — never a literal 'results/'.
    out_dir = Path(config.paths.results_dir) / "preprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_NAME
    report.to_csv(report_path, index=False)

    # Verify what was written rather than trusting the write.
    written = pd.read_csv(report_path)
    assert len(written) == len(report), "diagnostics report lost rows on write"
    assert bool(written["all_variants_finite"].all()), "a variant produced non-finite output"
    print(f"\nreport       : {report_path}")

    provenance_path = record_run(
        config,
        manifest_qc,
        folds=None,
        extra={
            "stage": "milestone-3-preprocess",
            "analysis_role": "primary",
            "filter_spec": spec,
            "option_b_roi_bins": [int(b) for b in option_b_roi_bins(pre, 534)],
            "n_sessions": len(report),
            "n_eligible_frames": int(report["n_eligible_frames"].sum()),
            "peak_hz_median": float(report["peak_hz_median"].median()),
            "energy_retention_median": float(report["energy_retention_median"].median()),
        },
    )
    print(f"provenance   : {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
