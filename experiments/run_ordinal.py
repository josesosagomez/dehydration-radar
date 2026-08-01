"""Experiment C — the ordinal 5-class (S0-S4) secondary task, both arms (milestone 9).

Runs the staged two-arm ordinal search on the SAME persistent feature store Exp A uses, for
one band: Stage 1 (feature axes at the family-(a) ridge anchor, scored ordinally) feeds ONE
winning feature key to both Stage-2 arms — (a) the five base families x frozen grids as
thresholded ordinal regressors, (b) Frank-Hall over the frozen C grid (A-M9-1).

    # mechanism-only smoke (6 lowest evaluable subjects) — no performance value surfaced
    uv run python experiments/run_ordinal.py --config configs/exp_a_regression.yaml \\
        --config configs/exp_c.yaml --subset 6subjects

    # 77 GHz smoke
    uv run python experiments/run_ordinal.py --config configs/exp_a_regression_77ghz.yaml \\
        --config configs/exp_c.yaml --band 77ghz --subset 6subjects

    # the full-cohort run (no owner pause: the C design was frozen before Exp A was seen)
    uv run python experiments/run_ordinal.py --config configs/exp_a_regression.yaml \\
        --config configs/exp_c.yaml --full-cohort

`--subset 6subjects` XOR `--full-cohort` is REQUIRED. Exp C reports its ordinal metrics
ABSOLUTELY and registers no baseline comparison: the session-index baseline predicts the
class perfectly (the class IS the session index), so any radar-vs-baseline framing here is
degenerate and writing one would be an undisclosed protocol invention (plan §5 trap 16).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import FROZEN_SEED_SET, load_config, require_77ghz_dir  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.manifest import apply_qc, build_manifest  # noqa: E402
from dehyd.data.manifest_77 import apply_qc_77, build_manifest_77  # noqa: E402
from dehyd.eval import exp_a, exp_b, exp_c  # noqa: E402
from dehyd.eval.splits import nested_loso_splits  # noqa: E402
from dehyd.features.protocol_freeze import protocol_freeze_guard  # noqa: E402
from dehyd.provenance import _git_info, record_run  # noqa: E402


def _require_frozen_seed_set(config, mode) -> None:
    """Outside the mechanism-only smoke, the seed set is not a free run-level knob.

    `configs/smoke.yaml`'s `seed_set: [1]` became expressible at milestone 9 step 10, so the
    loader alone no longer protects a REPORTING run from it — this is the same guard
    `run_baselines._require_frozen_run_protocol` carries, applied to Exp C.
    """
    if mode == "smoke":
        return
    if tuple(config.run.seed_set) != FROZEN_SEED_SET:
        raise SystemExit(
            f"run.seed_set is {tuple(config.run.seed_set)} but a full Exp C run is frozen at "
            f"{FROZEN_SEED_SET} — the reduced seed set is a SMOKE-only overlay"
        )


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
                        help="mechanism-only smoke on the 6 lowest subjects")
    parser.add_argument("--full-cohort", action="store_true", help="the full-cohort run")
    args = parser.parse_args(argv)
    if bool(args.subset) == bool(args.full_cohort):
        parser.error("exactly one of --subset 6subjects or --full-cohort is required")

    config = load_config(*args.config)
    protocol_freeze_guard(config)   # config-level pre-flight (before any I/O)

    sessions = exp_c.build_sessions_c(config, args.band)
    mode = "full" if args.full_cohort else "smoke"
    _require_frozen_seed_set(config, mode)
    if mode == "smoke":
        keep = set(exp_a.select_subset_subjects(exp_c.evaluable_subjects_c(sessions), k=6))
        sessions = [s for s in sessions if s["subject"] in keep]

    print(f"config : {', '.join(args.config)}  band {args.band}  mode {mode}")
    print(f"sessions: {len(sessions)}  subjects: {len({s['subject'] for s in sessions})}")

    subjects = exp_c.evaluable_subjects_c(sessions)
    run_path = record_run(
        config, _build_manifest_qc(config, args.band), nested_loso_splits(subjects),
        data_dir=require_77ghz_dir(config) if args.band == "77ghz" else None,
        # `config_hash` is what the sanctioned exploratory frame split validates its source
        # artifact's lineage against (plan §2.10, C24) — the SAME named helper the Exp B
        # variant and the Exp D run groups use, never a second hashing recipe.
        extra={"stage": f"exp-c-{mode}", "band": args.band, "n_eval": len(subjects),
               "n_sessions": len(sessions), "config_hash": exp_b.config_fingerprint(config)},
    )
    run_dir = run_path.parent
    print(f"provenance: {run_path}")

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"workers   : {n_workers} (folds run in parallel; result is order-independent)")
    outputs = exp_c.run_and_report_c(
        config, args.band, sessions, config.paths.results_dir, run_dir,
        mode=mode, analysis_commit=_git_info()["commit"], n_workers=n_workers,
    )
    for name, path in outputs.items():
        print(f"  {name:18s}: {path}")
    if mode == "smoke":
        print("\nmechanism-only smoke OK — no performance value surfaced.")
    else:
        print("\nfull-cohort Exp C complete (ordinal metrics only; no baseline comparison).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
