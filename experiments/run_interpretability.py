"""Experiment E — LOSO path-group ablation of the fixed interpretability model (milestone 10).

For one band, refits the pre-registered Exp-E ridge on Exp-B residual targets inside every
outer LOSO fold, then deletes one WST path group at a time and refits, so each path's
contribution is measured as the change in held-out-subject residual MAE.

    # mechanism-only smoke (6 lowest evaluable subjects) — no importance value surfaced
    uv run python experiments/run_interpretability.py --config configs/exp_a_regression.yaml \\
        --config configs/exp_e.yaml --config configs/stats.yaml --band 10ghz --subset 6subjects

    # 77 GHz smoke
    uv run python experiments/run_interpretability.py --config configs/exp_a_regression_77ghz.yaml \\
        --config configs/exp_e.yaml --config configs/stats.yaml --band 77ghz --subset 6subjects

    # the full-cohort run (IBEX; see scripts/ibex/run_exp_e.sbatch)
    uv run python experiments/run_interpretability.py --config configs/exp_a_regression.yaml \\
        --config configs/exp_e.yaml --config configs/stats.yaml --config configs/ibex.yaml \\
        --band 10ghz --full-cohort

`--subset 6subjects` XOR `--full-cohort` is REQUIRED.

Unlike Exp G, E takes ONE `--config` list and one `--band`: the two bands are independent
analyses of two different filter banks, run separately, and there is nothing to hold in common
between them at run time — no shared population, no shared model, no cross-band estimand.

This is the cheap milestone-10 job. There is no inner CV and no search: per outer fold it is
`1 + n_paths` deterministic ridge fits on ~50-60 rows (measured: ~12 s per 10 GHz fold at 742
paths, ~7 s per 77 GHz fold at 424). Fold-parallel across 16 folds it is seconds of compute;
the run is dominated by reading the store.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import load_config, require_77ghz_dir  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.manifest import apply_qc, build_manifest  # noqa: E402
from dehyd.data.manifest_77 import apply_qc_77, build_manifest_77  # noqa: E402
from dehyd.eval import exp_a, exp_b, exp_e  # noqa: E402
from dehyd.eval.splits import nested_loso_splits  # noqa: E402
from dehyd.features.protocol_freeze import protocol_freeze_guard  # noqa: E402
from dehyd.provenance import _git_info, record_run  # noqa: E402


def _build_manifest_qc(config, band):
    gt = load_ground_truth(config.paths.weight_xlsx)
    if band == "10ghz":
        return apply_qc(build_manifest(config.paths, gt), config.paths, config)
    return apply_qc_77(build_manifest_77(config.paths, gt), config.paths, config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--band", choices=("10ghz", "77ghz"), default="10ghz")
    parser.add_argument("--subset", metavar="6subjects",
                        help="mechanism-only smoke on the 6 lowest evaluable subjects")
    parser.add_argument("--full-cohort", action="store_true", help="the full-cohort run")
    args = parser.parse_args(argv)
    if bool(args.subset) == bool(args.full_cohort):
        parser.error("exactly one of --subset 6subjects or --full-cohort is required")

    config = load_config(*args.config)
    protocol_freeze_guard(config)      # config-level pre-flight, before any I/O
    if args.band == "77ghz":
        require_77ghz_dir(config)

    # E reads Exp B's S0-excluded spine and reuses its train-only residualization code; it
    # consumes no Exp B result artifact, so there is no run to point at and nothing to stale.
    sessions = exp_b.build_sessions_b(config, args.band)
    mode = "full" if args.full_cohort else "smoke"
    if mode == "smoke":
        keep = set(exp_a.select_subset_subjects(exp_b.evaluable_subjects_b(sessions), k=6))
        sessions = [s for s in sessions if s["subject"] in keep]

    subjects = exp_b.evaluable_subjects_b(sessions)
    candidate = exp_e.fixed_candidate(config, args.band)
    print(f"config  : {', '.join(args.config)}  band {args.band}  mode {mode}")
    print(f"sessions: {len(sessions)}  subjects: {len(subjects)}")
    print(f"model   : FIXED {dict(candidate.active)} alpha={config.exp_e.ridge_alpha} "
          f"(pre-registered anchor, never the best A/B model)")

    run_path = record_run(
        config, _build_manifest_qc(config, args.band), nested_loso_splits(subjects),
        data_dir=require_77ghz_dir(config) if args.band == "77ghz" else None,
        extra={"stage": f"exp-e-{mode}", "band": args.band,
               "n_eval": len(subjects), "n_sessions": len(sessions)},
    )
    run_dir = run_path.parent
    print(f"provenance: {run_path}")

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"workers   : {n_workers} (folds run in parallel; result is order-independent)")

    outputs = exp_e.run_and_report_e(
        config, args.band, sessions, config.paths.results_dir, run_dir,
        mode=mode, analysis_commit=_git_info()["commit"], n_workers=n_workers,
    )
    for name, path in outputs.items():
        print(f"  {name:20s}: {path}")
    if mode == "smoke":
        print("\nmechanism-only smoke OK — no importance value surfaced.")
    else:
        print("\nfull-cohort Exp E complete — the path table is reported as measured. "
              "Attribution is model reliance, not causality, and no sign was required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
