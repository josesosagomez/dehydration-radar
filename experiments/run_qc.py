"""Milestone 2 — run the frozen QC screens over the whole 10 GHz cohort.

Writes the per-subject/session survival report that the protocol requires to be
reported (implementation_plan.md, "Session eligibility after QC"), plus a provenance
record of the run.

    uv run python experiments/run_qc.py --config configs/exp_a_regression.yaml

QC is frozen and data-independent, so this runs ONCE, before and outside any
cross-validation. Nothing here selects a threshold; if the survival numbers look
surprising that is a finding for HISTORY.md, not a reason to move a threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import load_config  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.manifest import (  # noqa: E402
    apply_qc,
    build_manifest,
    eligible_frames,
    evaluable_subjects,
    session_qc_report,
)
from dehyd.provenance import record_run  # noqa: E402

REPORT_NAME = "qc_survival_10ghz.csv"


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
    print(f"config       : {', '.join(args.config)}")
    print(
        f"qc gate      : {config.qc.qc_gate_m} m  margin {config.qc.in_band_margin_hz} Hz  "
        f"min in-band ratio {config.qc.min_in_band_energy_ratio}"
    )

    gt = load_ground_truth(config.paths.weight_xlsx)
    manifest = build_manifest(config.paths, gt)
    print(f"manifest     : {len(manifest)} frames, {manifest.subject.nunique()} subjects")

    manifest_qc = apply_qc(manifest, config.paths, config)
    report = session_qc_report(manifest_qc)

    n_frames = len(manifest_qc)
    n_pass = int(manifest_qc["qc_pass"].sum())
    print(
        f"frames       : {n_pass} pass / {n_frames - n_pass} fail of {n_frames} "
        f"({100 * n_pass / n_frames:.1f}% survive)"
    )
    # Per-reason counts are non-additive: one frame can fail several screens.
    print(
        "  reasons    : "
        f"nan/inf {int(manifest_qc['qc_nan_inf'].sum())}, "
        f"flatline {int(manifest_qc['qc_flatline'].sum())}, "
        f"low in-band {int(manifest_qc['qc_low_in_band'].sum())} "
        f"(non-additive; rms flagged {int(manifest_qc['qc_rms_flag'].sum())}, diagnostic)"
    )

    n_eligible = int(report["eligible"].sum())
    print(f"sessions     : {n_eligible} eligible / {len(report) - n_eligible} dropped "
          f"of {len(report)}")
    if n_eligible < len(report):
        dropped = report[~report["eligible"]]
        for row in dropped.itertuples():
            print(
                f"  DROPPED    : subject {row.subject} {row.session_name} — "
                f"{row.n_pass}/{row.n_frames} survived, needed {row.min_pass}"
            )

    subjects = evaluable_subjects(manifest_qc)
    print(f"N_eval       : {len(subjects)} evaluable subjects {list(subjects)}")
    print(f"analysis pop : {len(eligible_frames(manifest_qc))} frames")

    # The config is the single output-path authority — never a literal 'results/'.
    out_dir = Path(config.paths.results_dir) / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_NAME
    report.to_csv(report_path, index=False)

    # Verify what was written rather than trusting the write.
    import pandas as pd

    written = pd.read_csv(report_path)
    assert len(written) == len(report), "survival report lost rows on write"
    assert (
        written["n_pass"] + written["n_fail_any"] == written["n_frames"]
    ).all(), "survival report does not reconcile"
    print(f"report       : {report_path}")

    provenance_path = record_run(
        config,
        manifest_qc,
        folds=None,
        extra={
            "stage": "milestone-2-qc",
            "n_frames": n_frames,
            "n_frames_pass": n_pass,
            "n_sessions": len(report),
            "n_sessions_eligible": n_eligible,
            "n_evaluable_subjects": len(subjects),
        },
    )
    print(f"provenance   : {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
