"""Experiment B — clock-decoupling: session-mean-residualized fluid-loss regression (milestone 8).

Runs the residualized-target staged search on the SAME persistent feature store Exp A uses,
for one band, against the session-mean-only baseline (zero residual, by construction). Reuses
Exp A's exact search space (A-M6-3) under Exp B's own equal-session-weighted objective.

    # mechanism-only smoke (6 lowest evaluable subjects) — no performance value surfaced
    uv run python experiments/run_clock_decoupling.py --config configs/exp_a_regression.yaml \\
        --config configs/exp_b.yaml --subset 6subjects

    # 77 GHz smoke
    uv run python experiments/run_clock_decoupling.py --config configs/exp_a_regression_77ghz.yaml \\
        --config configs/exp_b.yaml --band 77ghz --subset 6subjects

    # the full-cohort run — NO owner pause for Exp B (Step 0 item 2): the core design was
    # frozen before Exp A was seen, so there is no freeze left to spend on this compute step.
    uv run python experiments/run_clock_decoupling.py --config configs/exp_a_regression.yaml \\
        --config configs/exp_b.yaml --full-cohort

`--subset 6subjects` XOR `--full-cohort` is REQUIRED, unless `--session-specific` (the
secondary, thin robustness variant — implementation_plan.md:722-724) is given, in which case
exactly one of `--init-run-group` / `--session {1,2,3,4} --run-dir PATH` /
`--merge-sessions --run-dir PATH` is required instead. See MILESTONE_8_PLAN.md §2.5 for the
full flow description of each session-specific sub-mode; these are meant to be driven by
`scripts/ibex/submit_exp_b_variant.sh`, not run ad hoc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import load_config, require_77ghz_dir  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.manifest import apply_qc, build_manifest  # noqa: E402
from dehyd.data.manifest_77 import apply_qc_77, build_manifest_77  # noqa: E402
from dehyd.eval import exp_a, exp_b  # noqa: E402
from dehyd.eval.splits import nested_loso_splits  # noqa: E402
from dehyd.features import store as store_mod  # noqa: E402
from dehyd.features.protocol_freeze import protocol_freeze_guard  # noqa: E402
from dehyd import provenance  # noqa: E402
from dehyd.provenance import _git_info, record_run  # noqa: E402


def _validate_flags(args, parser):
    if args.session_specific:
        if args.subset or args.full_cohort:
            parser.error("--session-specific is mutually exclusive with --subset/--full-cohort")
        sub_modes = [args.init_run_group, args.session is not None, args.merge_sessions]
        if sum(bool(m) for m in sub_modes) != 1:
            parser.error(
                "--session-specific requires exactly one of --init-run-group, "
                "--session {1,2,3,4} --run-dir PATH, or --merge-sessions --run-dir PATH"
            )
        if args.session is not None and args.run_dir is None:
            parser.error("--session requires --run-dir")
        if args.merge_sessions and args.run_dir is None:
            parser.error("--merge-sessions requires --run-dir")
    elif bool(args.subset) == bool(args.full_cohort):
        parser.error("exactly one of --subset 6subjects or --full-cohort is required")


def _build_manifest_qc(config, band):
    gt = load_ground_truth(config.paths.weight_xlsx)
    if band == "10ghz":
        return apply_qc(build_manifest(config.paths, gt), config.paths, config)
    return apply_qc_77(build_manifest_77(config.paths, gt), config.paths, config)


def _main_primary(args, config) -> int:
    sessions = exp_b.build_sessions_b(config, args.band)
    mode = "full" if args.full_cohort else "smoke"

    if mode == "smoke":
        keep = set(exp_a.select_subset_subjects(exp_b.evaluable_subjects_b(sessions), k=6))
        sessions = [s for s in sessions if s["subject"] in keep]

    print(f"config : {', '.join(args.config)}  band {args.band}  mode {mode}")
    print(f"sessions: {len(sessions)}  subjects: {len({s['subject'] for s in sessions})}")

    manifest_qc = _build_manifest_qc(config, args.band)
    subjects = exp_b.evaluable_subjects_b(sessions)
    folds = nested_loso_splits(subjects)
    run_path = record_run(
        config, manifest_qc, folds,
        data_dir=require_77ghz_dir(config) if args.band == "77ghz" else None,
        extra={"stage": f"exp-b-{mode}", "band": args.band, "n_eval": len(subjects), "n_sessions": len(sessions)},
    )
    run_dir = run_path.parent
    print(f"provenance: {run_path}")

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"workers   : {n_workers} (folds run in parallel; result is order-independent)")
    outputs = exp_b.run_and_report_b(
        config, args.band, sessions, config.paths.results_dir, run_dir,
        mode=mode, analysis_commit=_git_info()["commit"], n_workers=n_workers,
    )
    for name, path in outputs.items():
        print(f"  {name:16s}: {path}")
    if mode == "smoke":
        print("\nmechanism-only smoke OK — no performance value surfaced.")
    else:
        print("\nfull-cohort Exp B complete — no owner pause spent (core design frozen before Exp A).")
    # Only the PRIMARY pooled run writes a pointer. The session-specific variant is a separate
    # call tree producing shards, not one authoritative directory, so it has nothing to hand to
    # manifest construction. Written at the end of a successful run so a crashed job leaves
    # none, and manifest construction fails closed rather than registering a partial directory.
    if args.run_dir_out:
        print(f"run dir   : {provenance.write_run_dir_pointer(args.run_dir_out, run_dir)}")
    return 0


def _main_session_specific(args, config) -> int:
    sessions = exp_b.build_sessions_b(config, args.band)
    store_dir = config.paths.results_dir
    data_dir = require_77ghz_dir(config) if args.band == "77ghz" else None
    analysis_commit = _git_info()["commit"]

    if args.init_run_group:
        manifest_qc = _build_manifest_qc(config, args.band)
        # fail fast, before the array is ever submitted, on a stale/wrong store (C17).
        store_mod.validate_store(
            args.band, store_dir, exp_a.expected_fingerprints(config, args.band, sessions),
            analysis_commit=analysis_commit,
        )
        config_hash = exp_b.config_fingerprint(config)
        expected_subjects_by_session = {
            str(s): exp_b.eligible_subjects_for_session(sessions, s) for s in (1, 2, 3, 4)
        }
        folds_by_session = {
            str(s): provenance.fold_manifest(nested_loso_splits(expected_subjects_by_session[str(s)]))
            for s in (1, 2, 3, 4)
        }
        run_path = record_run(
            config, manifest_qc, folds=None, data_dir=data_dir,
            extra={
                "stage": "exp-b-session-specific-group", "band": args.band, "config_hash": config_hash,
                "expected_subjects_by_session": expected_subjects_by_session,
                "folds_by_session": folds_by_session,
            },
        )
        run_dir = run_path.parent
        print(run_dir)
        return 0

    if args.session is not None:
        run_dir = Path(args.run_dir)
        # defense in depth (C17): re-validated independently per task, never assumed still
        # true from --init-run-group's earlier check.
        store_mod.validate_store(
            args.band, store_dir, exp_a.expected_fingerprints(config, args.band, sessions),
            analysis_commit=analysis_commit,
        )
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        results_s = exp_b.run_exp_b_one_session(
            config, args.band, sessions, store_dir, args.session,
            seeds=config.run.seed_set, n_workers=n_workers,
        )
        exp_b._assert_mechanism_ok_b(results_s, sessions)   # C18, unchanged from the primary path

        run_dir.mkdir(parents=True, exist_ok=True)
        band, session = args.band, args.session
        exp_b._write_predictions_csv(
            results_s, run_dir / f"session_specific_predictions_{band}_s{session}.csv")
        exp_b._write_selection_table_csv(
            results_s, run_dir / f"session_specific_selection_table_{band}_s{session}.csv")
        exp_b._write_dropped_folds_csv(
            results_s, run_dir / f"session_specific_dropped_folds_{band}_s{session}.csv")

        summary = exp_b.summarize_variant_session(results_s, session, config)
        shard = {
            "run_group_id": run_dir.name,
            "band": band,
            "session": session,
            "analysis_commit": analysis_commit,
            "config_hash": exp_b.config_fingerprint(config),   # SAME named helper as --init-run-group (C20)
            "seed_set": list(config.run.seed_set),
            "n_eval_subjects": sorted({r.test_subject for r in results_s if r.reason is None}),
            "summary": summary,
        }
        shard_path = run_dir / f"session_specific_{band}_s{session}.json"
        shard_path.write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {shard_path}")
        return 0

    # args.merge_sessions
    run_dir = Path(args.run_dir)
    merged = exp_b.merge_session_specific_reports(args.band, run_dir)
    out_path = run_dir / f"session_specific_{args.band}.json"
    out_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}  completed_sessions={merged['completed_sessions']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--band", choices=("10ghz", "77ghz"), default="10ghz")
    parser.add_argument("--subset", metavar="6subjects", help="mechanism-only smoke on the 6 lowest subjects")
    parser.add_argument("--full-cohort", action="store_true", help="the full-cohort run")
    parser.add_argument("--session-specific", action="store_true",
                        help="the secondary session-specific variant's entirely separate code path")
    parser.add_argument("--init-run-group", action="store_true",
                        help="[--session-specific] run once: create the shared run-group provenance")
    parser.add_argument("--session", type=int, choices=(1, 2, 3, 4),
                        help="[--session-specific] run one session's independent search")
    parser.add_argument("--run-dir", metavar="PATH", help="[--session-specific] the shared run-group directory")
    parser.add_argument("--merge-sessions", action="store_true",
                        help="[--session-specific] combine whichever per-session shards exist under --run-dir")
    parser.add_argument("--run-dir-out", metavar="PATH",
                        help="after a SUCCESSFUL primary run, atomically write the absolute run "
                             "directory here for milestone-10 manifest construction")
    args = parser.parse_args(argv)
    _validate_flags(args, parser)

    config = load_config(*args.config)
    protocol_freeze_guard(config)   # config-level pre-flight (before any I/O)

    if args.session_specific:
        return _main_session_specific(args, config)
    return _main_primary(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
