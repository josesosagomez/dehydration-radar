"""Experiment D — the baselines, under Experiment A's identical LOSO harness (milestone 9).

One entrypoint, one `--family`, four ways to run it:

  * **CNN families** (`cnn1d_raw`, `cnn1d_matched`, `spec2d_raw`, `spec2d_matched`) use the
    M8 run-group pattern, because a single job iterating 16 outer folds sequentially has
    wall-time equal to their sum (the C11 mistake). `--init-run-group` writes the shared
    provenance + the authoritative per-fold row census; `--fold N --run-dir P` is ONE array
    task; `--merge-folds --run-dir P` combines whatever shards exist, fail-closed.
    `--subset 6subjects` is the local mechanism-only smoke, which runs the same code in one
    process over the 6-subject subset.
  * **Cheap families** (`physics`, `session_index`) and `comparisons` take
    `--subset 6subjects` XOR `--full-cohort`.

    # local smokes (mechanism only — no performance value surfaced)
    uv run python experiments/run_baselines.py --config configs/exp_a_regression.yaml \\
        --config configs/baselines.yaml --family physics --subset 6subjects
    uv run python experiments/run_baselines.py --config configs/exp_a_regression.yaml \\
        --config configs/baselines.yaml --family spec2d_raw --subset 6subjects

    # the GPU fold array (driven by scripts/ibex/submit_exp_d_cnn.sh, not by hand)
    ... --family cnn1d_raw --init-run-group
    ... --family cnn1d_raw --fold 3 --run-dir results/runs/<group>
    ... --family cnn1d_raw --merge-folds --run-dir results/runs/<group>

    # the comparison stage, once every family has merged (O-M9-5 gates the radar side)
    ... --family comparisons --full-cohort --exp-a-run-dir R --m7-reference-dir M \\
        --family-run-dir physics=P --family-run-dir cnn1d_raw=Q ...

Outside smoke mode the seed set must be the frozen `(1,2,3,4,5)`, and `run.device` may name a
GPU only for the four CNN families — the WST/numpy canonical-backend policy is untouched and
the DL baselines are the one authorized GPU path (`implementation_plan.md:1326-1329`).
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
from dehyd.eval import exp_a, exp_b, exp_d  # noqa: E402
from dehyd.eval.splits import nested_loso_splits  # noqa: E402
from dehyd.features import store as store_mod  # noqa: E402
from dehyd.features.protocol_freeze import protocol_freeze_guard  # noqa: E402
from dehyd.provenance import _git_info, record_run  # noqa: E402

CNN_FAMILIES = exp_d.CNN_FAMILIES
CHEAP_FAMILIES = exp_d.DETERMINISTIC_FAMILIES
FAMILIES = CNN_FAMILIES + CHEAP_FAMILIES + ("comparisons",)
FROZEN_SEED_SET = (1, 2, 3, 4, 5)


def _validate_flags(args, parser) -> None:
    cnn_modes = [bool(args.init_run_group), args.fold is not None, bool(args.merge_folds)]
    selection = [bool(args.subset), bool(args.full_cohort)]

    if args.family in CNN_FAMILIES:
        if args.full_cohort:
            parser.error(
                "a CNN family has no single-process full-cohort mode: 16 outer folds run "
                "sequentially cost their sum. Use --init-run-group, then the --fold array, "
                "then --merge-folds (scripts/ibex/submit_exp_d_cnn.sh drives all three)."
            )
        if sum(cnn_modes) + int(bool(args.subset)) != 1:
            parser.error(
                f"--family {args.family} requires exactly one of --subset 6subjects, "
                "--init-run-group, --fold N --run-dir PATH, or --merge-folds --run-dir PATH"
            )
        if (args.fold is not None or args.merge_folds) and args.run_dir is None:
            parser.error("--fold and --merge-folds both require --run-dir PATH")
        if args.fold is not None and args.fold < 0:
            # a fold id is a POSITION in the selectable-fold list; a negative one would index
            # from the end and silently run the wrong fold under the right shard name
            parser.error(f"--fold must be a non-negative fold position, got {args.fold}")
        return

    if any(cnn_modes):
        parser.error(
            f"--init-run-group/--fold/--merge-folds belong to the CNN families "
            f"{list(CNN_FAMILIES)}, not to --family {args.family}"
        )
    if sum(selection) != 1:
        parser.error("exactly one of --subset 6subjects or --full-cohort is required")
    if args.family == "comparisons":
        if not args.exp_a_run_dir or not args.m7_reference_dir:
            parser.error("--family comparisons requires --exp-a-run-dir and --m7-reference-dir")
        missing = [f for f in exp_d.EXPD_FAMILIES if f not in dict(args.family_run_dir or {})]
        if missing:
            parser.error(
                f"--family comparisons needs a --family-run-dir NAME=PATH for every Exp D "
                f"family; missing {missing}"
            )


def _require_frozen_run_protocol(config, family, mode) -> None:
    """Outside the mechanism-only smoke, the run-level knobs are not free (§2.11)."""
    if mode == "smoke":
        return
    if tuple(config.run.seed_set) != FROZEN_SEED_SET:
        raise SystemExit(
            f"run.seed_set is {tuple(config.run.seed_set)} but a full Exp D run is frozen at "
            f"{FROZEN_SEED_SET} — the reduced seed set is a SMOKE-only overlay"
        )
    if str(config.run.device) != "cpu" and family not in CNN_FAMILIES:
        raise SystemExit(
            f"run.device={config.run.device!r} with --family {family}: the deep-learning "
            "baselines are the one authorized GPU path; every other family runs on CPU"
        )


def _family_run_dir(text):
    name, _, path = text.partition("=")
    if not path or name not in exp_d.EXPD_FAMILIES:
        raise argparse.ArgumentTypeError(
            f"--family-run-dir expects NAME=PATH with NAME in {list(exp_d.EXPD_FAMILIES)}, "
            f"got {text!r}"
        )
    return name, path


def _build_manifest_qc(config, band):
    gt = load_ground_truth(config.paths.weight_xlsx)
    if band == "10ghz":
        return apply_qc(build_manifest(config.paths, gt), config.paths, config)
    return apply_qc_77(build_manifest_77(config.paths, gt), config.paths, config)


def _sessions_for(config, band, mode):
    sessions = exp_a.build_sessions(config, band)
    if mode == "smoke":
        keep = set(exp_a.select_subset_subjects({s["subject"] for s in sessions}, k=6))
        sessions = [s for s in sessions if s["subject"] in keep]
    return sessions


def _validate_group_lineage(run_dir, *, band, family, analysis_commit, config_hash) -> dict:
    """Refuse an array task or a merge that points at somebody else's run group — BEFORE the
    GPU work, not after it (the M8 C19/C20 doctrine)."""
    provenance = json.loads((Path(run_dir) / "provenance.json").read_text(encoding="utf-8"))
    extra = provenance.get("extra") or {}
    for field, found, expected in (
        ("band", extra.get("band"), band),
        ("family", extra.get("family"), family),
        ("config_hash", extra.get("config_hash"), config_hash),
        ("analysis_commit", (provenance.get("git") or {}).get("commit"), analysis_commit),
    ):
        if found != expected:
            raise SystemExit(
                f"run group {Path(run_dir).name}: {field} is {found!r}, expected {expected!r} "
                "— refusing to write into a run group this task does not belong to"
            )
    return provenance


def _main_cnn(args, config) -> int:
    band, family = args.band, args.family
    store_dir = config.paths.results_dir
    analysis_commit = _git_info()["commit"]
    config_hash = exp_b.config_fingerprint(config)
    mode = "smoke" if args.subset else "full"
    _require_frozen_run_protocol(config, family, mode)

    sessions = _sessions_for(config, band, mode)
    subjects = sorted({int(s["subject"]) for s in sessions})

    if args.init_run_group:
        # only the stages that write provenance need a data root to hash against
        data_dir = require_77ghz_dir(config) if band == "77ghz" else None
        # fail fast, before the array is ever submitted, on a stale/wrong store (C17)
        store_mod.validate_store(band, store_dir,
                                 exp_a.expected_fingerprints(config, band, sessions),
                                 analysis_commit=analysis_commit)
        frames = exp_d.build_frames_d(config, band, family, sessions, store_dir)
        run_path = record_run(
            config, _build_manifest_qc(config, band), nested_loso_splits(subjects),
            data_dir=data_dir,
            extra={
                "stage": "exp-d-cnn-group", "band": band, "family": family,
                "config_hash": config_hash,
                "seed_set": [int(s) for s in config.run.seed_set],
                "n_eval": len(subjects), "n_sessions": len(sessions),
                # the authoritative per-fold row census every shard is validated against
                "expected_test_rows_by_fold": exp_d.expected_test_rows_by_fold(
                    frames, config.run.seed_set
                ),
            },
        )
        print(run_path.parent)      # the machine-readable handoff: LAST line of stdout
        return 0

    if args.merge_folds:
        merged = exp_d.merge_exp_d_folds(band, family, Path(args.run_dir))
        out_path = Path(args.run_dir) / f"exp_d_{family}_{band}_merged.json"
        out_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out_path}  state={merged['state']}  "
              f"completed_folds={merged['completed_folds']}")
        return 0

    folds = exp_d.selectable_folds(subjects)
    if args.fold is not None:
        run_dir = Path(args.run_dir)
        _validate_group_lineage(run_dir, band=band, family=family,
                                analysis_commit=analysis_commit, config_hash=config_hash)
        # defense in depth (C17): re-validated per task, never assumed still true from init
        store_mod.validate_store(band, store_dir,
                                 exp_a.expected_fingerprints(config, band, sessions),
                                 analysis_commit=analysis_commit)
        if args.fold >= len(folds):
            # the array is a fixed 16 tasks while N_eval can be smaller (§5 trap 14): a named
            # no-op, never an error and never a silent absence the merge cannot read
            path = exp_d.write_noop_marker(
                run_dir, band=band, family=family, fold_id=args.fold,
                reason=f"fold index {args.fold} is beyond the {len(folds)} selectable folds",
            )
            print(f"no-op: {path}")
            return 0
        fold = folds[args.fold]
        frames = exp_d.build_frames_d(config, band, family, sessions, store_dir)
        result = exp_d.run_cnn_family(config, band, family, fold, config.run.seed_set, frames,
                                      device=config.run.device)
        json_path, csv_path = exp_d.write_fold_shard(
            result, frames, fold, args.fold, run_dir, band=band, family=family,
            seeds=config.run.seed_set, run_group_id=run_dir.name,
            analysis_commit=analysis_commit, config_hash=config_hash,
        )
        print(f"wrote {json_path}\nwrote {csv_path}")
        return 0

    # --subset: the local mechanism-only smoke, same code path, one process
    store_mod.validate_store(band, store_dir,
                             exp_a.expected_fingerprints(config, band, sessions),
                             analysis_commit=analysis_commit)
    frames = exp_d.build_frames_d(config, band, family, sessions, store_dir)
    results = [exp_d.run_cnn_family(config, band, family, fold, config.run.seed_set, frames,
                                    device=config.run.device)
               for fold in folds]
    for result, fold in zip(results, folds, strict=True):
        assert result.test_subject == fold.test_subject
        for record in result.final_fits:
            assert result.test_subject not in record.subjects
    run_dir = record_run(
        config, _build_manifest_qc(config, band), nested_loso_splits(subjects),
        data_dir=require_77ghz_dir(config) if band == "77ghz" else None,
        extra={"stage": f"exp-d-{family}-smoke", "band": band, "family": family,
               "config_hash": config_hash},
    ).parent
    return _write_smoke_log(run_dir, band, family, len(results), len(sessions))


def _write_smoke_log(run_dir, band, family, n_folds, n_sessions) -> int:
    """MECHANISM-ONLY: structural counts and nothing else. No metric, no selected
    configuration, no epoch budget — a smoke must not surface a performance value."""
    log = Path(run_dir) / f"run_log_{family}_{band}.json"
    log.write_text(json.dumps({
        "stage": f"exp-d-{family}-smoke", "band": band, "family": family,
        "mode": "mechanism-only", "n_folds": n_folds, "n_sessions": n_sessions,
        "note": "performance values suppressed -- mechanism-only smoke",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  run_log : {log}")
    print("\nmechanism-only smoke OK — no performance value surfaced.")
    return 0


def _main_cheap(args, config) -> int:
    band, family = args.band, args.family
    store_dir = config.paths.results_dir
    mode = "full" if args.full_cohort else "smoke"
    _require_frozen_run_protocol(config, family, mode)
    analysis_commit = _git_info()["commit"]

    sessions = _sessions_for(config, band, mode)
    subjects = sorted({int(s["subject"]) for s in sessions})
    store_mod.validate_store(band, store_dir,
                             exp_a.expected_fingerprints(config, band, sessions),
                             analysis_commit=analysis_commit)
    run_dir = record_run(
        config, _build_manifest_qc(config, band), nested_loso_splits(subjects),
        data_dir=require_77ghz_dir(config) if band == "77ghz" else None,
        extra={"stage": f"exp-d-{family}-{mode}", "band": band, "family": family,
               "config_hash": exp_b.config_fingerprint(config),
               "n_eval": len(subjects), "n_sessions": len(sessions)},
    ).parent

    results = (exp_d.run_physics(config, band, sessions, store_dir) if family == "physics"
               else exp_d.run_session_index(config, band, sessions))
    exp_d.assert_mechanism_ok_d(results, subjects)

    if mode == "smoke":
        return _write_smoke_log(run_dir, band, family, len(results), len(sessions))

    prediction_rows = [row for r in results for row in exp_d.cheap_prediction_rows(r)]
    paths = exp_d.write_family_artifacts(
        band, family, run_dir,
        prediction_rows=prediction_rows,
        selection_rows=[exp_d.cheap_selection_row(r) for r in results],
        deterministic=True,
        bootstrap_b=config.stats.bootstrap_b,
        rng_seed=config.run.seed,
        skip_threshold_pct=config.stats.undefined_metric_skip_threshold_pct,
        lineage={"analysis_commit": analysis_commit,
                 "config_hash": exp_b.config_fingerprint(config),
                 "run_group_id": run_dir.name},
    )
    for name, path in paths.items():
        print(f"  {name:12s}: {path}")
    return 0


def _main_comparisons(args, config) -> int:
    band = args.band
    mode = "full" if args.full_cohort else "smoke"
    _require_frozen_run_protocol(config, "comparisons", mode)
    analysis_commit = _git_info()["commit"]
    config_hash = exp_b.config_fingerprint(config)
    family_runs = dict(args.family_run_dir)

    # every input family must come from THIS code and THIS config, named individually on
    # failure so a wrong directory is a one-line diagnosis rather than a hunt
    for family, run_dir in sorted(family_runs.items()):
        metrics = json.loads(
            (Path(run_dir) / f"metrics_{family}_{band}.json").read_text(encoding="utf-8")
        )
        lineage = metrics.get("lineage") or {}
        for field, expected in (("analysis_commit", analysis_commit),
                                ("config_hash", config_hash)):
            if lineage.get(field) != expected:
                raise SystemExit(
                    f"Exp D family {family} at {run_dir}: lineage.{field} is "
                    f"{lineage.get(field)!r}, expected {expected!r} — refusing to compare "
                    "against artifacts from a different revision or configuration"
                )

    radar = exp_d.load_exp_a_radar(band, args.exp_a_run_dir, args.m7_reference_dir,
                                   analysis_commit=analysis_commit)
    summary = exp_d.summarize_exp_d(band, config, family_runs, radar)

    run_dir = record_run(
        config, _build_manifest_qc(config, band), folds=None,
        data_dir=require_77ghz_dir(config) if band == "77ghz" else None,
        extra={"stage": f"exp-d-comparisons-{mode}", "band": band, "config_hash": config_hash,
               "exp_a_run_dir": str(args.exp_a_run_dir),
               "family_run_dirs": {f: str(p) for f, p in family_runs.items()}},
    ).parent
    paths = exp_d.write_exp_d_comparison_reports(summary, run_dir, band)
    for name, path in paths.items():
        print(f"  {name:12s}: {path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--band", choices=("10ghz", "77ghz"), default="10ghz")
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--subset", metavar="6subjects",
                        help="mechanism-only smoke on the 6 lowest subjects")
    parser.add_argument("--full-cohort", action="store_true", help="the full-cohort run")
    parser.add_argument("--init-run-group", action="store_true",
                        help="[CNN] run once: create the shared run group + its row census")
    parser.add_argument("--fold", type=int, help="[CNN] run ONE outer fold (an array task)")
    parser.add_argument("--merge-folds", action="store_true",
                        help="[CNN] combine whichever fold shards exist under --run-dir")
    parser.add_argument("--run-dir", metavar="PATH", help="[CNN] the shared run-group directory")
    parser.add_argument("--exp-a-run-dir", metavar="PATH",
                        help="[comparisons] the Exp A re-run whose predictions are the radar side")
    parser.add_argument("--m7-reference-dir", metavar="PATH",
                        help="[comparisons] the M7 artifacts the re-run must be bit-identical to")
    parser.add_argument("--family-run-dir", action="append", type=_family_run_dir,
                        metavar="NAME=PATH", help="[comparisons] one per Exp D family")
    args = parser.parse_args(argv)
    _validate_flags(args, parser)

    config = load_config(*args.config)
    protocol_freeze_guard(config)   # config-level pre-flight (before any I/O)

    print(f"config : {', '.join(args.config)}  band {args.band}  family {args.family}")
    if os.environ.get("SLURM_JOB_ID"):
        print(f"slurm  : job {os.environ['SLURM_JOB_ID']}")

    if args.family in CNN_FAMILIES:
        return _main_cnn(args, config)
    if args.family == "comparisons":
        return _main_comparisons(args, config)
    return _main_cheap(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
