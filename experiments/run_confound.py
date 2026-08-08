"""Experiment F — the unavailable heart-rate question plus the available-covariate sensitivity.

Records that the registered HR question is NOT ESTIMABLE from the delivered data (A-M10-2),
with the repository/workbook inventory that establishes it, and separately runs four nested
ridge models (clock / +covariates / +radar / +both) under outer LOSO in three variants.

    # mechanism-only smoke (6 lowest evaluable subjects) — no contrast value surfaced
    uv run python experiments/run_confound.py --config configs/exp_a_regression.yaml \\
        --config configs/exp_f.yaml --config configs/stats.yaml \\
        --exp-a-sources results/milestone10/exp_a_sources.json --band 10ghz --subset 6subjects

    # the full-cohort run (IBEX; see scripts/ibex/run_exp_f.sbatch)
    uv run python experiments/run_confound.py --config configs/exp_a_regression.yaml \\
        --config configs/exp_f.yaml --config configs/stats.yaml --config configs/ibex.yaml \\
        --exp-a-sources results/milestone10/exp_a_sources.json --band 10ghz --full-cohort

`--subset 6subjects` XOR `--full-cohort` is REQUIRED, and so is `--exp-a-sources`.

**`--exp-a-sources` is required and is never discovered.** Models 3 and 4 take their radar
feature key from an Exp-A run, and §1.3 makes that an explicit, approved pointer: no glob, no
"latest" directory, no fallback. `reference_gate.load_approved_sources` then refuses any band
the reference gate left `not_approved`, so an unvalidated Exp-A table cannot reach F by any
path. Until step 12 writes that file, this entrypoint has nothing to consume — which is the
intended failure, not a gap.

The HR inventory is written in BOTH modes. It is not a performance value; it is the answer to
the registered question, and a smoke that suppressed it would be hiding the one thing F can
state with certainty.
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
from dehyd.eval import exp_a, exp_f  # noqa: E402
from dehyd.eval.splits import nested_loso_splits  # noqa: E402
from dehyd.features.protocol_freeze import protocol_freeze_guard  # noqa: E402
from dehyd import provenance  # noqa: E402
from dehyd.provenance import _git_info, record_run  # noqa: E402


def _build_manifest_qc(config, band):
    gt = load_ground_truth(config.paths.weight_xlsx)
    if band == "10ghz":
        return apply_qc(build_manifest(config.paths, gt), config.paths, config)
    return apply_qc_77(build_manifest_77(config.paths, gt), config.paths, config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--exp-a-sources", required=True, metavar="PATH",
                        help="the approved exp_a_sources.json (explicit; never globbed)")
    parser.add_argument("--band", choices=("10ghz", "77ghz"), default="10ghz")
    parser.add_argument("--subset", metavar="6subjects",
                        help="mechanism-only smoke on the 6 lowest evaluable subjects")
    parser.add_argument("--full-cohort", action="store_true", help="the full-cohort run")
    parser.add_argument("--run-dir-out", metavar="PATH",
                        help="after a SUCCESSFUL run, atomically write the absolute run "
                             "directory here for milestone-10 manifest construction")
    args = parser.parse_args(argv)
    if bool(args.subset) == bool(args.full_cohort):
        parser.error("exactly one of --subset 6subjects or --full-cohort is required")

    config = load_config(*args.config)
    protocol_freeze_guard(config)      # config-level pre-flight, before any I/O
    if args.band == "77ghz":
        require_77ghz_dir(config)

    # F keeps S0: its question is about the clock across the whole day, and the session-index
    # one-hot is what makes S0 informative rather than a free perfectly-predicted row.
    sessions = exp_f.build_sessions_f(config, args.band)
    mode = "full" if args.full_cohort else "smoke"
    if mode == "smoke":
        keep = set(exp_a.select_subset_subjects(exp_f.evaluable_subjects_f(sessions), k=6))
        sessions = [s for s in sessions if s["subject"] in keep]

    subjects = exp_f.evaluable_subjects_f(sessions)
    print(f"config    : {', '.join(args.config)}  band {args.band}  mode {mode}")
    print(f"sessions  : {len(sessions)}  subjects: {len(subjects)}")
    print(f"exp-a src : {args.exp_a_sources} (explicit; approved bands only)")
    print(f"models    : {', '.join(exp_f.MODEL_IDS)}")
    print(f"variants  : {', '.join(exp_f.VARIANTS)}")
    print(f"heart rate: {exp_f.HR_STATUS} — the covariate models are NOT an HR adjustment")

    run_path = record_run(
        config, _build_manifest_qc(config, args.band), nested_loso_splits(subjects),
        data_dir=require_77ghz_dir(config) if args.band == "77ghz" else None,
        extra={"stage": f"exp-f-{mode}", "band": args.band, "n_eval": len(subjects),
               "n_sessions": len(sessions), "exp_a_sources": str(args.exp_a_sources),
               "heart_rate_status": exp_f.HR_STATUS},
    )
    run_dir = run_path.parent
    print(f"provenance: {run_path}")

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"workers   : {n_workers} (folds run in parallel; result is order-independent)")

    outputs = exp_f.run_and_report_f(
        config, args.band, sessions, config.paths.results_dir, run_dir,
        mode=mode, analysis_commit=_git_info()["commit"],
        exp_a_sources=args.exp_a_sources, n_workers=n_workers,
    )
    for name, path in outputs.items():
        print(f"  {name:14s}: {path}")
    if mode == "smoke":
        print("\nmechanism-only smoke OK — no contrast value surfaced.")
    else:
        print("\nfull-cohort Exp F complete — a limited clock/static-covariate sensitivity "
              "result. Heart rate is not estimable; temperature and glucose stay uncontrolled.")
    # Written only here, at the end of a successful run: a crashed job must leave no pointer,
    # so manifest construction fails closed rather than registering a half-written directory.
    if args.run_dir_out:
        print(f"run dir   : {provenance.write_run_dir_pointer(args.run_dir_out, run_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
