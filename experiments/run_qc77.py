"""Milestone 5 — run the frozen 77 GHz QC screens + semantic axis check over the cohort.

Under the owner's step-6 outcome (b) this is the AUTHORITATIVE survival/eligibility pass:
it writes results/qc/qc_survival_77ghz.csv (session-level qc_pass/eligibility + the per-file
axis certificate: verdict, raw sha256, axis_spec_hash), results/qc/qc_frames_77ghz.csv
(per-frame diagnostics), a label-blind flatline-diagnostics summary, and a provenance record.

    uv run python experiments/run_qc77.py --config configs/exp_77ghz.yaml

The semantic axis check runs once per file on the raw pre-MTI cube the QC pass already
decompressed, and the run FAILS CLOSED on any non-ACCEPTED verdict (REJECTED and INCONCLUSIVE
both abort). QC is frozen and data-independent, so this runs ONCE, outside any CV; surprising
survival numbers are a finding for HISTORY.md, never a reason to move a threshold.

--subject/--session subset the QC loop (the full manifest is still built and validated) for a
fast local smoke; the full authoritative run passes no subset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import load_config, require_77ghz_dir  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.loader_77ghz import load_77ghz_file  # noqa: E402
from dehyd.data.manifest_77 import (  # noqa: E402
    _finalize_qc_77,
    build_manifest_77,
    eligible_frames,
    evaluable_subjects,
    resolve_path_77,
    session_qc_report_77,
)
from dehyd.data.manifest import _join_qc  # noqa: E402
from dehyd.data.sessions import SESSION_NAMES  # noqa: E402
from dehyd.provenance import record_run, sha256_file  # noqa: E402
from dehyd.qc.axis_check_77 import AxisCertError, axis_spec_hash, certify_axis  # noqa: E402
from dehyd.qc.screens_77 import run_qc_cube_77  # noqa: E402

SURVIVAL_NAME = "qc_survival_77ghz.csv"
FRAMES_NAME = "qc_frames_77ghz.csv"


def _subset(manifest, subject, session):
    if subject is not None:
        manifest = manifest[manifest["subject"] == subject]
    if session is not None:
        manifest = manifest[manifest["session_name"] == session]
    if manifest.empty:
        raise SystemExit(f"no files match --subject {subject} --session {session}")
    return manifest.reset_index(drop=True)


def qc_and_axis(manifest, config):
    """Loop the files once: axis-certify the raw cube, then run the frozen QC screens.

    Fails closed on any non-ACCEPTED axis verdict. Returns (qc_rows, axis_records).
    """
    spec_hash = axis_spec_hash(config)
    qc_rows, axis_records = [], []
    for rel_path in manifest["rel_path"].drop_duplicates():
        path = resolve_path_77(config.paths, rel_path)
        cube = load_77ghz_file(path)  # whole-file read (one decompress)

        verdict, metrics = certify_axis(cube, config.preprocess77)  # RAW pre-MTI cube
        if verdict != "ACCEPTED":
            raise AxisCertError(
                f"{rel_path}: semantic axis check returned {verdict} (not ACCEPTED); "
                "the fast/chirp mapping is uncertified — the run fails closed"
            )
        axis_records.append({
            "rel_path": rel_path,
            "sha256": sha256_file(path),
            "axis_spec_hash": spec_hash,
            "axis_verdict": verdict,
            "axis_G_fast": metrics["G_fast"],
            "axis_G_chirp": metrics["G_chirp"],
            "axis_D_chirp": metrics["D_chirp"],
            "axis_D_fast": metrics["D_fast"],
        })

        for frame_idx, r in enumerate(run_qc_cube_77(cube, config.qc77, config.preprocess77)):
            qc_rows.append({
                "rel_path": rel_path, "frame_idx": frame_idx,
                "qc_nan_inf": r.nan_inf, "qc_flatline": r.flatline,
                "qc_low_in_band": r.low_in_band, "qc_pass": r.passed,
                "qc_in_band_ratio": r.in_band_ratio,
                "qc_n_flatline_traces": r.n_flatline_traces,
                "qc_rx_max_flatline": max(r.per_rx_flatline) if r.per_rx_flatline else 0,
            })
        print(f"  {rel_path}: axis {verdict}, "
              f"{sum(1 for row in qc_rows if row['rel_path'] == rel_path and row['qc_pass'])} pass")
    return pd.DataFrame(qc_rows), pd.DataFrame(axis_records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH",
                        help="config YAML; repeatable, later files win")
    parser.add_argument("--subject", type=int, default=None, help="restrict the QC loop")
    parser.add_argument("--session", choices=SESSION_NAMES, default=None)
    args = parser.parse_args(argv)

    config = load_config(*args.config)
    require_77ghz_dir(config)
    print(f"config       : {', '.join(args.config)}")
    print(f"qc77 gate    : {config.preprocess77.gate_m} m  "
          f"flatline skip_leading {config.qc77.flatline_skip_leading_bins}  "
          f"min in-band {config.qc77.min_in_band_energy_ratio}")

    gt = load_ground_truth(config.paths.weight_xlsx)
    full_manifest = build_manifest_77(config.paths, gt)  # validates the whole cohort
    manifest = _subset(full_manifest, args.subject, args.session)
    print(f"manifest     : {len(manifest)} frames, {manifest['rel_path'].nunique()} files "
          f"(of {full_manifest['rel_path'].nunique()} in the cohort)")

    qc_rows, axis_records = qc_and_axis(manifest, config)
    merged = _finalize_qc_77(_join_qc(manifest, qc_rows), config.qc77.min_frame_fraction)

    n_frames = len(merged)
    n_pass = int(merged["qc_pass"].sum())
    print(f"frames       : {n_pass} pass / {n_frames - n_pass} fail of {n_frames} "
          f"({100 * n_pass / n_frames:.1f}% survive)")
    print(f"  reasons    : nan/inf {int(merged['qc_nan_inf'].sum())}, "
          f"flatline {int(merged['qc_flatline'].sum())}, "
          f"low in-band {int(merged['qc_low_in_band'].sum())} (non-additive)")
    print(f"  flatline traces/frame: median {int(merged['qc_n_flatline_traces'].median())}, "
          f"max {int(merged['qc_n_flatline_traces'].max())} (of 4096)")

    # Session survival report + the per-file axis certificate columns.
    report = session_qc_report_77(merged).merge(axis_records, on="rel_path", how="left")
    n_eligible = int(report["eligible"].sum())
    print(f"sessions     : {n_eligible} eligible / {len(report) - n_eligible} dropped "
          f"of {len(report)}")
    for row in report[~report["eligible"]].itertuples():
        print(f"  DROPPED    : subject {row.subject} {row.session_name} — "
              f"{row.n_pass}/{row.n_frames} survived, needed {row.min_pass}")

    subjects = evaluable_subjects(merged)
    print(f"N_eval       : {len(subjects)} evaluable subjects {list(subjects)}")
    print(f"analysis pop : {len(eligible_frames(merged))} frames")

    out_dir = Path(config.paths.results_dir) / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    survival_path = out_dir / SURVIVAL_NAME
    frames_path = out_dir / FRAMES_NAME
    report.to_csv(survival_path, index=False)
    merged.to_csv(frames_path, index=False)

    # Verify what was written rather than trusting the write.
    written = pd.read_csv(survival_path)
    assert len(written) == len(report), "survival report lost rows on write"
    assert (written["n_pass"] + written["n_fail_any"] == written["n_frames"]).all(), \
        "survival report does not reconcile"
    assert (written["axis_verdict"] == "ACCEPTED").all(), "a non-ACCEPTED axis slipped through"
    print(f"survival     : {survival_path}")
    print(f"frames       : {frames_path}")

    provenance_path = record_run(
        config, merged, folds=None, data_dir=require_77ghz_dir(config),
        extra={
            "stage": "milestone-5-qc77", "eligibility_authoritative": True,
            "flatline_rule": "exclude-bin-0 (skip_leading=1)",
            "n_frames": n_frames, "n_frames_pass": n_pass,
            "n_sessions": len(report), "n_sessions_eligible": n_eligible,
            "n_evaluable_subjects": len(subjects),
            "subset": {"subject": args.subject, "session": args.session},
        },
    )
    print(f"provenance   : {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
