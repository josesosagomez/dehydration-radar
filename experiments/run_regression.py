"""Experiment A — fluid-loss regression entry point.

MILESTONE 2 SCOPE: this runs the data spine end to end — config -> ground truth ->
manifest -> QC -> nested-LOSO folds over the evaluable subjects -> provenance — and
stops there. Modeling (feature extraction, selection, fitting, scoring) arrives at
milestone 6, on top of exactly these folds. Nothing here constructs a split: folds
come only from eval/splits.py.

    uv run python experiments/run_regression.py --config configs/exp_a_regression.yaml

Per-machine roots are supplied by appending an overlay rather than editing the
canonical config in place:

    ... --config configs/exp_a_regression.yaml --config configs/ibex.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python experiments/run_regression.py) without
# requiring the package to be installed in editable mode.
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
from dehyd.eval.splits import nested_loso_splits  # noqa: E402
from dehyd.provenance import record_run  # noqa: E402


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
    print(f"device       : {config.run.device}   seed: {config.run.seed}")

    gt = load_ground_truth(config.paths.weight_xlsx)
    print(f"ground truth : {len(gt.subjects)} subjects, {len(gt.sessions)} sessions")
    print(
        "               Delta m% range "
        f"{gt.sessions.delta_m_pct.min():.2f} .. {gt.sessions.delta_m_pct.max():.2f}"
    )

    manifest = build_manifest(config.paths, gt)
    n_sessions = manifest.groupby(["subject", "session_idx"]).ngroups
    print(
        f"manifest     : {len(manifest)} frames, "
        f"{manifest.subject.nunique()} subjects, {n_sessions} sessions"
    )

    # QC is frozen and data-independent, so it runs ONCE here, before any split is
    # constructed — the folds are built over the post-QC evaluable population.
    manifest_qc = apply_qc(manifest, config.paths, config)
    report = session_qc_report(manifest_qc)
    n_pass = int(manifest_qc["qc_pass"].sum())
    print(
        f"qc           : {n_pass}/{len(manifest_qc)} frames pass, "
        f"{int(report['eligible'].sum())}/{len(report)} sessions eligible, "
        f"{len(eligible_frames(manifest_qc))} frames in the analysis population"
    )

    # Exp A rule: a subject is evaluable iff it has >= 1 eligible session. Subjects
    # with none are dropped BEFORE outer splitting so they never form an empty fold.
    subjects = list(evaluable_subjects(manifest_qc))
    folds = nested_loso_splits(
        subjects,
        n_inner_max=config.split.n_inner_max,
        min_train_subjects=config.split.min_train_subjects,
    )
    selectable = [f for f in folds if f.selectable]
    print(
        f"folds        : {len(folds)} outer ({len(selectable)} selectable), "
        f"{len(selectable[0].inner_folds) if selectable else 0} inner each"
    )

    provenance_path = record_run(
        config, manifest_qc, folds, extra={"stage": "milestone-2-smoke", "n_eval": len(subjects)}
    )
    print(f"provenance   : {provenance_path}")
    print("\nmilestone 2: data spine + QC OK. Modeling lands at milestone 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
