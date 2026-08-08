"""Experiment H — the selection-variance robustness bootstrap for one experiment/band.

`R` full-procedure subject resamples of Experiment A, B or C: each replicate draws N subject
IDs with replacement and re-runs the COMPLETE procedure (candidate enumeration, inner-CV
selection, refit, scoring) on that drawn cohort, so the reported spread is the variance of the
*procedure*, not of a fixed model. The output is an empirical percentile range, deliberately
NOT a BCa interval (A-M10-5).

    # mechanism-only smoke — 8 replicates on the 6 lowest subjects. MUST come back
    # "inconclusive": min_successful stays at the frozen 100 and is never scaled.
    uv run python experiments/run_robustness.py --config configs/exp_a_regression.yaml \\
        --config configs/stats.yaml --experiment a --band 10ghz --replicates 8 --subset 6subjects

    # the full-cohort job in one allocation (IBEX; see scripts/ibex/run_robustness.sbatch)
    uv run python experiments/run_robustness.py --config configs/exp_a_regression.yaml \\
        --config configs/stats.yaml --experiment a --band 10ghz --replicates 200 --full-cohort

    # ...or the same job SHARDED across a SLURM array, then merged
    #    (see scripts/ibex/submit_robustness_sharded.sh)
    uv run python experiments/run_robustness.py ... --replicates 200 --full-cohort \\
        --replicate-start 1 --replicate-stop 10 --shard-out results/milestone10/robustness_shards/a_10ghz
    uv run python experiments/run_robustness.py ... --replicates 200 --full-cohort \\
        --merge-shards results/milestone10/robustness_shards/a_10ghz

`--subset 6subjects` XOR `--full-cohort` is REQUIRED. The smoke runs the identical resampling
and refit path but surfaces no estimate — only counts, skip reasons and the conclusive/
inconclusive status, matching the Exp A/B/C mechanism-only doctrine.

This is heavy CPU work: one replicate is one complete run of the chosen experiment on a cohort
of ~10 distinct subjects, and there are `R` of them plus one full-cohort point estimate — order
1,200 core-hours per (experiment, band) at `R=200`, ~2,000+ for Exp C. Always submit it to
Slurm; never run it on a login shell. Sharding is safe because each replicate's cohort is a pure
function of its own seed tuple, so a contiguous replicate range is a complete unit of work; the
merge refuses a shard set with a gap, an overlap, or a different commit/config/cohort.
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
from dehyd.eval import exp_a, robustness  # noqa: E402
from dehyd.eval.splits import nested_loso_splits  # noqa: E402
from dehyd.features.protocol_freeze import protocol_freeze_guard  # noqa: E402
from dehyd import provenance  # noqa: E402
from dehyd.provenance import _git_info, record_run  # noqa: E402


def _require_frozen_seed_set(config, mode) -> None:
    """Outside the mechanism-only smoke the model seed set is not a run-level knob.

    Same guard `run_ordinal.py` carries: `configs/smoke.yaml`'s `seed_set: [1]` is expressible,
    so the loader alone no longer protects a reporting run from it. Model seeds are the
    configured seeds and are never derived from the resampling seed (plan §2.4), which makes
    this the only place they can go wrong.
    """
    if mode == "smoke":
        return
    if tuple(config.run.seed_set) != FROZEN_SEED_SET:
        raise SystemExit(
            f"run.seed_set is {tuple(config.run.seed_set)} but a full robustness run is frozen "
            f"at {FROZEN_SEED_SET} — the reduced seed set is a SMOKE-only overlay"
        )


def _build_manifest_qc(config, band):
    gt = load_ground_truth(config.paths.weight_xlsx)
    if band == "10ghz":
        return apply_qc(build_manifest(config.paths, gt), config.paths, config)
    return apply_qc_77(build_manifest_77(config.paths, gt), config.paths, config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--experiment", choices=robustness.EXPERIMENTS, required=True)
    parser.add_argument("--band", choices=robustness.BANDS, default="10ghz")
    parser.add_argument("--replicates", type=int, default=None,
                        help="R (default: StatsConfig.robustness_replicates_r = 200)")
    parser.add_argument("--subset", metavar="6subjects",
                        help="mechanism-only smoke on the 6 lowest subjects")
    parser.add_argument("--full-cohort", action="store_true", help="the full-cohort run")
    parser.add_argument("--replicate-start", type=int, default=None,
                        help="SHARD MODE: first replicate of this array task's range (1-based)")
    parser.add_argument("--replicate-stop", type=int, default=None,
                        help="SHARD MODE: last replicate of this array task's range (inclusive)")
    parser.add_argument("--shard-out", metavar="DIR", default=None,
                        help="SHARD MODE: directory every shard of this job writes into")
    parser.add_argument("--merge-shards", metavar="DIR", default=None,
                        help="MERGE MODE: read that directory's shards, validate, and report")
    parser.add_argument("--run-dir-out", metavar="PATH",
                        help="after a SUCCESSFUL run, atomically write the absolute run "
                             "directory here for milestone-10 manifest construction")
    args = parser.parse_args(argv)
    if bool(args.subset) == bool(args.full_cohort):
        parser.error("exactly one of --subset 6subjects or --full-cohort is required")

    sharding = args.replicate_start is not None or args.replicate_stop is not None or \
        bool(args.shard_out)
    if sharding and args.merge_shards:
        parser.error("--merge-shards is a separate stage from --replicate-start/--shard-out")
    if sharding and not (args.replicate_start and args.replicate_stop and args.shard_out):
        parser.error("shard mode needs all three of --replicate-start, --replicate-stop and "
                     "--shard-out (a shard that does not declare its own range cannot be "
                     "validated for gaps or overlaps at merge time)")

    config = load_config(*args.config)
    protocol_freeze_guard(config)   # config-level pre-flight (before any I/O)

    mode = "full" if args.full_cohort else "smoke"
    _require_frozen_seed_set(config, mode)
    replicates = args.replicates if args.replicates is not None else config.stats.robustness_replicates_r
    if replicates < 1:
        parser.error(f"--replicates must be >= 1, got {replicates}")

    sessions = robustness.build_spine(config, args.experiment, args.band)
    if mode == "smoke":
        keep = set(exp_a.select_subset_subjects(
            robustness.spine_subjects(args.experiment, sessions), k=6
        ))
        sessions = [s for s in sessions if s["subject"] in keep]

    subjects = robustness.spine_subjects(args.experiment, sessions)
    first = args.replicate_start or 1
    seed_tuple = robustness.replicate_seed_tuple(config, args.experiment, args.band, first)
    print(f"config     : {', '.join(args.config)}  exp {args.experiment}  band {args.band}  mode {mode}")
    print(f"sessions   : {len(sessions)}  subjects: {len(subjects)}")
    print(f"replicates : R={replicates}  min_successful="
          f"{config.stats.robustness_min_successful_replicates} (frozen; never scaled)")
    print(f"seed tuple : {list(seed_tuple)} at replicate {first}  "
          f"(robustness_seed = config.run.seed = {robustness.robustness_seed(config)})")

    # Replicates are the parallel unit (each runs its own folds serially inside), so this is
    # replicate-level concurrency, not fold-level.
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"workers    : {n_workers} (replicates run in parallel; each is frozen by its own seed tuple)")
    analysis_commit = _git_info()["commit"]

    if sharding:
        # A shard deliberately does NOT call `record_run`: that hashes every raw file (tens of
        # GB at 77 GHz), and twenty array tasks doing it would be twenty times the I/O for one
        # run's worth of provenance. The MERGE writes the authoritative run directory; the
        # shard self-attests a lineage block the merge validates instead.
        print(f"shard      : replicates {args.replicate_start}..{args.replicate_stop} "
              f"of R={replicates} -> {args.shard_out}")
        outputs = robustness.run_and_report_shard(
            config, args.experiment, args.band, sessions, config.paths.results_dir,
            args.shard_out, analysis_commit=analysis_commit, replicates=replicates,
            start=args.replicate_start, stop=args.replicate_stop, n_workers=n_workers,
        )
        for name, path in outputs.items():
            print(f"  {name:18s}: {path}")
        print("\nshard complete — no summary and no range written here, by design: a range over "
              "a sub-range of replicates is not the estimand.")
        return 0

    run_path = record_run(
        config, _build_manifest_qc(config, args.band), nested_loso_splits(subjects),
        data_dir=require_77ghz_dir(config) if args.band == "77ghz" else None,
        extra={"stage": f"robustness-{args.experiment}-{mode}", "band": args.band,
               "experiment": args.experiment, "n_eval": len(subjects),
               "n_sessions": len(sessions), "replicates": int(replicates),
               "robustness_seed": robustness.robustness_seed(config),
               "shard_source": args.merge_shards},
    )
    run_dir = run_path.parent
    print(f"provenance : {run_path}")

    if args.merge_shards:
        print(f"merge      : reading shards from {args.merge_shards}")
        outputs = robustness.run_and_report_merge(
            config, args.experiment, args.band, sessions, config.paths.results_dir,
            args.merge_shards, run_dir, mode=mode, analysis_commit=analysis_commit,
            replicates=replicates, n_workers=n_workers,
        )
    else:
        outputs = robustness.run_and_report_robustness(
            config, args.experiment, args.band, sessions, config.paths.results_dir, run_dir,
            mode=mode, analysis_commit=analysis_commit, replicates=replicates,
            n_workers=n_workers,
        )
    for name, path in outputs.items():
        print(f"  {name:18s}: {path}")
    if mode == "smoke":
        print("\nmechanism-only smoke OK — no estimate surfaced; the status above is the "
              "min_successful rule doing its job.")
    else:
        print("\nfull-cohort robustness complete — the range is "
              f"{robustness.RANGE_LABEL}, NOT a BCa interval (A-M10-5).")
    # Written only here, at the end of a successful run: a crashed job must leave no pointer,
    # so manifest construction fails closed rather than registering a half-written directory.
    if args.run_dir_out:
        print(f"run dir   : {provenance.write_run_dir_pointer(args.run_dir_out, run_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
