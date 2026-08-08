"""Experiment G — matched-session decision-level 10 GHz + 77 GHz fusion (milestone 10).

Builds both band spines, inner-joins them on `(subject, session_idx)`, and on that matched cell
population fits the frozen convex combiner `alpha * pred_10 + (1 - alpha) * pred_77` — with
`alpha` selected inside each outer fold from selection-honest nested cross-fitted out-of-fold
predictions (A-M10-3), never from anything the held-out subject touched.

    # mechanism-only smoke — the 6 lowest matched subjects; no performance value surfaced
    uv run python experiments/run_fusion.py --config-10 configs/exp_a_regression.yaml \\
        --config-77 configs/exp_a_regression_77ghz.yaml \\
        --shared-config configs/exp_g_fusion.yaml --shared-config configs/stats.yaml \\
        --subset 6subjects

    # the full-cohort job (IBEX; see scripts/ibex/run_exp_g.sbatch)
    uv run python experiments/run_fusion.py --config-10 configs/exp_a_regression.yaml \\
        --config-77 configs/exp_a_regression_77ghz.yaml \\
        --shared-config configs/exp_g_fusion.yaml --shared-config configs/stats.yaml \\
        --shared-config configs/ibex.yaml --full-cohort

`--subset 6subjects` XOR `--full-cohort` is REQUIRED.

**The two band configs are loaded SEPARATELY and never merged.** They describe different front
ends, different WST sections and different search spaces, so a single merged top-level config
would describe neither band; `--shared-config` overlays (the Exp G rule, the statistical
protocol, the machine paths) are applied to both, and `exp_g.assert_shared_protocol` then refuses
any pair that disagrees on the run seeds, the split constants, the fusion rule, the statistics or
the weight workbook the target is read from.

This is heavy CPU work: a complete Exp-A staged selection runs at THREE levels — per (outer fold
x meta fold x band) and per (outer fold x band) for the outer-final winner — so at 16 matched
subjects it is order 200 staged selections against Experiment A's 16 per band. Always submit it
to Slurm; never run the full cohort on a login shell.
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
from dehyd.eval import exp_a, exp_g  # noqa: E402
from dehyd.eval.exp_b import config_fingerprint  # noqa: E402
from dehyd.eval.splits import nested_loso_splits  # noqa: E402
from dehyd.features.protocol_freeze import protocol_freeze_guard  # noqa: E402
from dehyd import provenance  # noqa: E402
from dehyd.provenance import _git_info, record_run  # noqa: E402


def _require_frozen_seed_set(config, mode) -> None:
    """Outside the mechanism-only smoke the model seed set is not a run-level knob.

    The same guard `run_ordinal.py` and `run_robustness.py` carry: `configs/smoke.yaml`'s
    `seed_set: [1]` is expressible, so the loader alone no longer protects a reporting run from
    it. Exp G is doubly exposed because `ExpGConfig.seed_pairing` pairs seed label `k` across the
    two bands — a reduced seed set would silently change what "the five paired seed labels" means.
    """
    if mode == "smoke":
        return
    if tuple(config.run.seed_set) != FROZEN_SEED_SET:
        raise SystemExit(
            f"run.seed_set is {tuple(config.run.seed_set)} but a full Exp G run is frozen at "
            f"{FROZEN_SEED_SET} — the reduced seed set is a SMOKE-only overlay"
        )


def _build_manifest_qc(config):
    """The QC'd 10 GHz manifest `record_run` hashes. Provenance is recorded against the primary
    band — the one the headline contrast is against — with the 77 GHz config pinned by hash."""
    gt = load_ground_truth(config.paths.weight_xlsx)
    return apply_qc(build_manifest(config.paths, gt), config.paths, config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config-10", action="append", required=True, metavar="PATH",
                        help="the 10 GHz band config (repeatable; merged in order)")
    parser.add_argument("--config-77", action="append", required=True, metavar="PATH",
                        help="the 77 GHz band config (repeatable; merged in order)")
    parser.add_argument("--shared-config", action="append", default=[], metavar="PATH",
                        help="overlay applied to BOTH bands, last (exp_g / stats / machine paths)")
    parser.add_argument("--subset", metavar="6subjects",
                        help="mechanism-only smoke on the 6 lowest matched subjects")
    parser.add_argument("--full-cohort", action="store_true", help="the full-cohort run")
    parser.add_argument("--run-dir-out", metavar="PATH",
                        help="after a SUCCESSFUL run, atomically write the absolute run "
                             "directory here for milestone-10 manifest construction")
    args = parser.parse_args(argv)
    if bool(args.subset) == bool(args.full_cohort):
        parser.error("exactly one of --subset 6subjects or --full-cohort is required")

    # Loaded separately, overlaid identically. Never merged through the loader.
    config_10 = load_config(*args.config_10, *args.shared_config)
    config_77 = load_config(*args.config_77, *args.shared_config)
    protocol_freeze_guard(config_10)      # config-level pre-flight, per band, before any I/O
    protocol_freeze_guard(config_77)
    exp_g.assert_shared_protocol(config_10, config_77)
    require_77ghz_dir(config_77)          # G cannot run without the 77 GHz data root

    mode = "full" if args.full_cohort else "smoke"
    _require_frozen_seed_set(config_10, mode)

    matched, sessions_10, sessions_77, unmatched = exp_g.build_matched_population(
        config_10, config_77
    )
    if mode == "smoke":
        # The subset applies to every view of the population, the unmatched ledger included, so
        # the smoke's counts describe the cohort it actually ran on rather than a mixture.
        keep = set(exp_a.select_subset_subjects([c["subject"] for c in matched], k=6))
        matched = [c for c in matched if c["subject"] in keep]
        sessions_10 = [s for s in sessions_10 if s["subject"] in keep]
        sessions_77 = [s for s in sessions_77 if s["subject"] in keep]
        unmatched = [u for u in unmatched if u["subject"] in keep]

    population = exp_g.population_summary(matched, unmatched)
    subjects = sorted({c["subject"] for c in matched})
    print(f"config 10  : {', '.join(args.config_10 + args.shared_config)}")
    print(f"config 77  : {', '.join(args.config_77 + args.shared_config)}")
    print(f"mode       : {mode}")
    print(f"matched    : {len(matched)} cells over {len(subjects)} subjects "
          f"({population['n_unmatched_cells']} unmatched cells)")
    print(f"alpha grid : {len(config_10.exp_g.alpha_grid)} points, tie-break "
          f"{config_10.exp_g.alpha_tie_break}")

    # The 77 GHz config is pinned by hash in `extra`, using the same named `config_fingerprint`
    # helper every other multi-config stage uses rather than a second hashing recipe.
    run_path = record_run(
        config_10, _build_manifest_qc(config_10), nested_loso_splits(subjects),
        extra={"stage": f"exp-g-{mode}", "band": "10ghz+77ghz",
               "config_77_sha256": config_fingerprint(config_77),
               "n_eval": len(subjects), "n_sessions": len(matched),
               "population": population},
    )
    run_dir = run_path.parent
    print(f"provenance : {run_path}")

    # The parallel unit is (outer fold, band): the two bands' work is independent given the
    # fold, and alpha is arithmetic over their saved prediction tables afterwards.
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"workers    : {n_workers} (fold-bands run in parallel; result is order-independent)")

    outputs = exp_g.run_and_report(
        config_10, config_77, matched, sessions_10, sessions_77, unmatched,
        config_10.paths.results_dir, run_dir, mode=mode,
        analysis_commit=_git_info()["commit"], n_workers=n_workers,
    )
    for name, path in outputs.items():
        print(f"  {name:20s}: {path}")
    if mode == "smoke":
        print("\nmechanism-only smoke OK — no performance value surfaced.")
    else:
        print("\nfull-cohort Exp G complete — fusion is reported as observed and is NOT "
              "required to beat 10 GHz.")
    # Written only here, at the end of a successful run: a crashed job must leave no pointer,
    # so manifest construction fails closed rather than registering a half-written directory.
    if args.run_dir_out:
        print(f"run dir   : {provenance.write_run_dir_pointer(args.run_dir_out, run_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
