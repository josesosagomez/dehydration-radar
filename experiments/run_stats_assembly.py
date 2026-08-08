"""Milestone-10 assembly — build the explicit run map, then write the final tables.

    # 1. build the manifest from explicitly named run directories (or pointer files the
    #    experiment wrappers wrote with --run-dir-out), and VALIDATE only
    uv run python experiments/run_stats_assembly.py \\
        --run-manifest results/milestone10/run_manifest.json \\
        --run a:10ghz=results/runs/<exp_a_10> --run a:77ghz=results/runs/<exp_a_77> \\
        --run-from-file b:10ghz=results/milestone10/sources/exp_b_10.txt \\
        --run g=results/runs/<exp_g> \\
        --validate-only

    # 2. the same manifest, this time writing the final tables
    uv run python experiments/run_stats_assembly.py \\
        --run-manifest results/milestone10/run_manifest.json --out results/milestone10

**Nothing is ever discovered.** Every source is named on the command line, either as a run
directory or as a pointer file a completed job wrote atomically. There is no glob, no "latest",
and no fallback: a milestone's final tables are lineage-bearing, and a headline number that came
from whichever directory happened to sort last is not traceable.

Re-running with `--run`/`--run-from-file` REBUILDS the manifest and rewrites it. Running with
neither reads the manifest already at `--run-manifest` and re-validates every registered
artifact's SHA-256, so a source that changed after registration stops the milestone instead of
quietly producing different final tables.

`--validate-only` performs the schema/SHA-256/lineage checks and writes NO numerical summary —
it is the first of the two calls in the plan's launch matrix, and its whole job is to fail
before anything is published.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.eval import assembly  # noqa: E402
from dehyd.provenance import read_run_dir_pointer  # noqa: E402


def _parse_source(spec: str, *, from_file: bool) -> assembly.SourceRun:
    """`EXPERIMENT[:BAND]=PATH` -> a SourceRun. The band is omitted only for cross-band G."""
    if "=" not in spec:
        raise SystemExit(f"--run/--run-from-file needs EXPERIMENT[:BAND]=PATH, got {spec!r}")
    key, _, path = spec.partition("=")
    experiment, _, band = key.partition(":")
    experiment = experiment.strip().lower()
    band = band.strip() or None
    if experiment not in assembly.EXPERIMENTS:
        raise SystemExit(
            f"unknown experiment {experiment!r} in {spec!r}; expected one of "
            f"{', '.join(assembly.EXPERIMENTS)}")
    run_dir = read_run_dir_pointer(path) if from_file else Path(path)
    return assembly.SourceRun(experiment=experiment, band=band, run_dir=run_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-manifest", required=True, metavar="PATH",
                        help="where the explicit run map is written (and re-read)")
    parser.add_argument("--run", action="append", default=[], metavar="EXP[:BAND]=DIR",
                        help="an explicitly named authoritative run directory (repeatable)")
    parser.add_argument("--run-from-file", action="append", default=[],
                        metavar="EXP[:BAND]=POINTER",
                        help="a pointer file a completed job wrote with --run-dir-out (repeatable)")
    parser.add_argument("--out", metavar="DIR",
                        help="where the final tables are written (required unless --validate-only)")
    parser.add_argument("--validate-only", action="store_true",
                        help="schema/SHA-256/lineage checks only; writes no numerical summary")
    args = parser.parse_args(argv)
    if not args.validate_only and not args.out:
        parser.error("--out DIR is required unless --validate-only is given")

    manifest_path = Path(args.run_manifest)
    sources = ([_parse_source(spec, from_file=False) for spec in args.run]
               + [_parse_source(spec, from_file=True) for spec in args.run_from_file])

    if sources:
        manifest = assembly.build_run_manifest(sources)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        print(f"manifest  : built from {len(sources)} explicit source(s) -> {manifest_path}")
    else:
        if not manifest_path.is_file():
            raise SystemExit(
                f"{manifest_path} does not exist and no --run/--run-from-file was given — "
                "assembly never discovers runs, so there is nothing to assemble from")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"manifest  : read {manifest_path}")

    assembly.validate_manifest(manifest)
    for key, entry in sorted(manifest["runs"].items()):
        print(f"  {key:16s} {entry['source_commit'][:12]}  "
              f"{len(entry['artifacts'])} artifact(s)  {entry['run_dir']}")

    if args.validate_only:
        print("\nvalidate-only OK — schema, SHA-256 and lineage checked; "
              "no numerical summary written.")
        return 0

    tables = assembly.assemble(manifest)
    summary = assembly.milestone_summary(manifest, tables)
    outputs = assembly.write_assembly_reports(manifest, tables, summary, args.out)
    for name, path in outputs.items():
        print(f"  {name:14s}: {path}")
    counts = summary["counts"]
    print(f"\nassembly complete — {counts['headline']} headline, {counts['per_subject']} "
          f"per-subject, {counts['paired']} paired, {counts['exclusions']} exclusion rows. "
          "Every row traces to an explicitly named run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
