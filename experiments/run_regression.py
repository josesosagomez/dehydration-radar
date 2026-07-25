"""Experiment A — fluid-loss (Δm%) regression, session-level nested LOSO (milestone 7).

Runs the staged search on a persistent feature store (built by extract_features.py), for one
band, against the session-index-only baseline.

    # mechanism-only smoke (6 lowest evaluable subjects) — no performance value surfaced
    uv run python experiments/run_regression.py --config configs/exp_a_regression.yaml --subset 6subjects

    # 77 GHz smoke
    uv run python experiments/run_regression.py --config configs/exp_a_regression_77ghz.yaml --band 77ghz --subset 6subjects

    # THE OWNER GATE — the full-cohort run spends the config freeze (first visible real scores)
    uv run python experiments/run_regression.py --config configs/exp_a_regression.yaml --full-cohort

`--subset 6subjects` XOR `--full-cohort` is REQUIRED. The smoke runs the identical
search/scoring path but suppresses every performance value (mechanism-only); only the
owner-gated `--full-cohort` run writes metrics / predictions / scatter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import load_config  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.manifest import apply_qc, build_manifest  # noqa: E402
from dehyd.data.manifest_77 import apply_qc_77, build_manifest_77  # noqa: E402
from dehyd.eval import exp_a  # noqa: E402
from dehyd.eval.splits import nested_loso_splits  # noqa: E402
from dehyd.features.protocol_freeze import protocol_freeze_guard  # noqa: E402
from dehyd.provenance import _git_info, record_run  # noqa: E402


def _validate_flags(args, parser):
    if bool(args.subset) == bool(args.full_cohort):
        parser.error(
            "exactly one of --subset 6subjects or --full-cohort is required. The full-cohort "
            "Exp A run spends the config freeze — owner go-ahead required."
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--band", choices=("10ghz", "77ghz"), default="10ghz")
    parser.add_argument("--subset", metavar="6subjects", help="mechanism-only smoke on the 6 lowest subjects")
    parser.add_argument("--full-cohort", action="store_true", help="OWNER GATE: the full run (spends the freeze)")
    args = parser.parse_args(argv)
    _validate_flags(args, parser)

    config = load_config(*args.config)
    protocol_freeze_guard(config)  # config-level pre-flight (before any I/O), incl. the K=1 baseline path

    sessions = exp_a.build_sessions(config, args.band)
    mode = "full" if args.full_cohort else "smoke"

    if mode == "smoke":
        keep = set(exp_a.select_subset_subjects([s["subject"] for s in sessions], k=6))
        sessions = [s for s in sessions if s["subject"] in keep]

    print(f"config : {', '.join(args.config)}  band {args.band}  mode {mode}")
    print(f"sessions: {len(sessions)}  subjects: {len({s['subject'] for s in sessions})}")

    # Provenance over the eligible manifest + the LOSO folds.
    gt = load_ground_truth(config.paths.weight_xlsx)
    if args.band == "10ghz":
        manifest_qc = apply_qc(build_manifest(config.paths, gt), config.paths, config)
    else:
        manifest_qc = apply_qc_77(build_manifest_77(config.paths, gt), config.paths, config)
    subjects = sorted({s["subject"] for s in sessions})
    folds = nested_loso_splits(subjects)
    run_path = record_run(config, manifest_qc, folds, extra={"stage": f"exp-a-{mode}", "band": args.band,
                                                             "n_eval": len(subjects), "n_sessions": len(sessions)})
    run_dir = run_path.parent
    print(f"provenance: {run_path}")

    outputs = exp_a.run_and_report(
        config, args.band, sessions, config.paths.results_dir, run_dir,
        mode=mode, analysis_commit=_git_info()["commit"],
    )
    for name, path in outputs.items():
        print(f"  {name:16s}: {path}")
    if mode == "smoke":
        print("\nmechanism-only smoke OK — no performance value surfaced. STOP: owner checkpoint.")
    else:
        print("\nfull-cohort Exp A complete — the config freeze is now spent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
