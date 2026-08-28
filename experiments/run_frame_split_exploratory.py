"""The owner's sanctioned EXPLORATORY frame-level random split — NEVER REPORTED.

A thin wrapper over `dehyd.eval.frame_split`. One invocation = one sanctioned run: both
Exp C arms, all six Exp D baselines, or the separately owner-authorized full-WST Exp A
regressor (`regression/radar_wst`), each on either band.

    uv run python experiments/run_frame_split_exploratory.py \\
        --config configs/exp_a_regression.yaml --config configs/exp_c.yaml \\
        --band 10ghz --task ordinal --unit arm_a --source-run-dir results/runs/<the Exp C run>

**This entrypoint must not call `provenance.record_run`**: that unconditionally creates
`results/runs/<stamp>_<rev>/provenance.json`, and D11 forbids this path from adding anything
under `results/runs/` at all. It writes its own provenance — the same payload, from the same
public builder — under the exploratory root instead.

The store is not re-validated here. Every artifact this path reads was produced by a LOSO
run that already validated the store fail-closed at its own commit, and `load_source_run`
refuses a source artifact whose `analysis_commit`/`config_hash` differ from this run's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import load_config, require_77ghz_dir  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.manifest import apply_qc, build_manifest  # noqa: E402
from dehyd.data.manifest_77 import apply_qc_77, build_manifest_77  # noqa: E402
from dehyd.eval import exp_a, exp_b, exp_c, frame_split  # noqa: E402
from dehyd.features.protocol_freeze import protocol_freeze_guard  # noqa: E402
from dehyd.provenance import _git_info  # noqa: E402

TASKS = tuple(frame_split.TASK_UNITS)


def _build_manifest_qc(config, band):
    gt = load_ground_truth(config.paths.weight_xlsx)
    if band == "10ghz":
        return apply_qc(build_manifest(config.paths, gt), config.paths, config)
    return apply_qc_77(build_manifest_77(config.paths, gt), config.paths, config)


def _build_sessions(config, band, task):
    return (exp_c.build_sessions_c(config, band) if task == "ordinal"
            else exp_a.build_sessions(config, band))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--band", choices=frame_split.BANDS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--unit",
        required=True,
        help="ordinal -> arm_a|arm_b; regression -> six Exp D families or radar_wst",
    )
    parser.add_argument("--source-run-dir", required=True, metavar="PATH",
                        help="the LOSO run dir whose selected configuration is refit here")
    args = parser.parse_args(argv)

    allowed = frame_split.TASK_UNITS[args.task]
    if args.unit not in allowed:
        parser.error(
            f"--unit {args.unit!r} is not a sanctioned {args.task} unit (expected one of "
            f"{list(allowed)}). The matrix covers Exp C, every Exp D baseline, and the "
            "separately authorized full-WST Exp A refit."
        )

    config = load_config(*args.config)
    protocol_freeze_guard(config)   # config-level pre-flight (before any I/O)

    print(f"EXPLORATORY frame split (LEAKY, never reported): band={args.band} "
          f"task={args.task} unit={args.unit}")
    sessions = _build_sessions(config, args.band, args.task)
    manifest_qc = _build_manifest_qc(config, args.band)
    data_dir = require_77ghz_dir(config) if args.band == "77ghz" else None

    result = frame_split.run_frame_split(
        config, args.band, args.task, args.unit,
        source_run_dir=args.source_run_dir, sessions=sessions,
        store_dir=config.paths.results_dir,
        analysis_commit=_git_info()["commit"],
        config_hash=exp_b.config_fingerprint(config),
    )
    outputs = frame_split.write_frame_split_reports(result, config)
    outputs["provenance"] = frame_split.write_exploratory_provenance(
        config, args.band, args.task, args.unit, manifest_qc,
        frame_split.exploratory_dir(config, args.band), data_dir=data_dir, result=result,
    )
    for name, path in outputs.items():
        print(f"  {name:12s}: {path}")
    print("\nEXPLORATORY only: these numbers are leaky by construction (subjects appear on "
          "both sides of the split) and appear in NO report, figure, or chapter section.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
