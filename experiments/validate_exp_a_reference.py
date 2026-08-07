"""The Experiment-A reference gate — snapshot before the store rebuild, compare after.

Milestone 10 rebuilds both feature stores, and `store._check_match` binds a store to one
git commit by strict equality. So the evidence behind the M9 Exp-A results — the stored
session vectors, the raw pre-log tensors, the fold-local tuned ε reconstructed from them —
stops being recomputable the moment the stores move. This tool freezes that evidence first,
and afterwards proves the rebuilt stores still give Exp A the same answers before Experiment
F is allowed to consume Exp A's per-fold feature selection.

    # BEFORE any structural edit or store rebuild — from the tree/store state that
    # produced the reference runs (plan §4.2 step 1, §6):
    uv run python experiments/validate_exp_a_reference.py --snapshot \
      --reference-10 results/runs/20260803T143704568296Z_f0a46aa6 \
      --reference-77 results/runs/20260803T151715023672Z_f0a46aa6 \
      --output results/milestone10/reference_exp_a_manifest.json

    # AFTER the final store rebuild and the final-commit Exp A reruns (plan §6):
    uv run python experiments/validate_exp_a_reference.py --compare \
      --reference-manifest results/milestone10/reference_exp_a_manifest.json \
      --final-10-file results/milestone10/sources/exp_a_10.txt \
      --final-77-file results/milestone10/sources/exp_a_77.txt \
      --output results/milestone10/exp_a_sources.json

Band configs default to the canonical Exp-A configs so the commands above are literal; add
`--config configs/ibex.yaml` (applied to both bands) for the paths overlay when running on
IBEX, where the authoritative stores live.

Exit status is 0 only when every evidence class passed. Anything else is a
milestone-stopping event for scientific review, never a byte-neutral drift to wave through.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import load_config  # noqa: E402
from dehyd.eval import reference_gate as gate  # noqa: E402

DEFAULT_CONFIG = {
    "10ghz": ("configs/exp_a_regression.yaml",),
    "77ghz": ("configs/exp_a_regression_77ghz.yaml",),
}


def _configs(args, bands) -> dict:
    """One loaded Config per requested band: the band's own config(s) plus any shared overlay.

    Only the requested bands are loaded — a 10 GHz-only snapshot must not fail because
    `paths.data_77ghz_dir` is absent on the machine holding the 10 GHz store.
    """
    shared = tuple(args.config or ())
    per_band = {"10ghz": args.config_10, "77ghz": args.config_77}
    return {
        band: load_config(*(tuple(per_band[band] or DEFAULT_CONFIG[band]) + shared))
        for band in bands
    }


def _read_pointer(path) -> Path:
    """An absolute run directory from a pointer file written by a successful run.

    A file, never a glob: the plan forbids discovering a "latest" directory, because that
    silently picks up whatever ran most recently rather than the run that was validated.
    """
    path = Path(path)
    if not path.is_file():
        raise gate.ReferenceGateError(f"missing run-directory pointer file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise gate.ReferenceGateError(f"run-directory pointer file is empty: {path}")
    run_dir = Path(text.splitlines()[0].strip())
    if not run_dir.is_dir():
        raise gate.ReferenceGateError(f"{path} points at {run_dir}, which is not a directory")
    return run_dir


def _print_snapshot(manifest) -> None:
    print(f"grade         : {manifest['reference_grade']}")
    for band in manifest["bands_covered"]:
        evidence = manifest["bands"][band]
        print(f"  {band}")
        print(f"    run       : {evidence['run']['name']}  commit {evidence['run']['commit'][:8]}")
        print(f"    store     : {evidence['store']['n_sessions']} sessions, "
              f"built at {[c[:8] for c in evidence['store']['build_commits']]}")
        print(f"    population: {evidence['population']['n_subjects']} subjects, "
              f"{evidence['population']['n_sessions']} sessions")
        print(f"    folds     : {evidence['folds']['n_selectable']}/{evidence['folds']['n_folds']} selectable")
        print(f"    stage 1   : {evidence['stage1_candidates']['n_candidates']} candidates")
        print(f"    features  : {len(evidence['feature_inputs'])} distinct selected feature keys")
        for superseded in evidence.get("superseded_runs", []):
            verdict = superseded["vs_reference"]
            detail = verdict["fault"] or (
                f"max|dy_pred| = {verdict['max_abs_pred_delta']:.3e}"
                if verdict["max_abs_pred_delta"] is not None else "n/a"
            )
            print(f"    superseded: {superseded['name']} ({superseded['commit'][:8]}) "
                  f"-> {verdict['status']}; selection table identical="
                  f"{verdict['selection_table_byte_identical']}; {detail}")


def _print_comparison(sources) -> None:
    print(f"status: {sources['status']}")
    for band, record in sources["bands"].items():
        print(f"  {band}: {record['status']}")
        for row in record["comparisons"]:
            mark = "ok  " if row["status"] == "match" else "FAIL"
            detail = f"  ({row['detail']})" if row.get("detail") else ""
            print(f"    {mark} {row['evidence_class']}{detail}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", action="store_true", help="freeze the pre-rebuild reference evidence")
    mode.add_argument("--compare", action="store_true", help="check the rebuilt stores against a snapshot")

    parser.add_argument("--config", action="append", metavar="PATH",
                        help="overlay applied to BOTH bands (e.g. configs/ibex.yaml)")
    parser.add_argument("--config-10", action="append", metavar="PATH",
                        help=f"10 GHz config (default {DEFAULT_CONFIG['10ghz'][0]})")
    parser.add_argument("--config-77", action="append", metavar="PATH",
                        help=f"77 GHz config (default {DEFAULT_CONFIG['77ghz'][0]})")

    parser.add_argument("--reference-10", metavar="RUN_DIR", help="M9 Exp A 10 GHz run directory")
    parser.add_argument("--reference-77", metavar="RUN_DIR", help="M9 Exp A 77 GHz run directory")
    parser.add_argument("--superseded-10", action="append", metavar="RUN_DIR",
                        help="an earlier 10 GHz Exp A run whose store is already gone; recorded "
                             "artifact-only and compared to the reference (repeatable)")
    parser.add_argument("--superseded-77", action="append", metavar="RUN_DIR",
                        help="same, for 77 GHz")
    parser.add_argument("--reference-manifest", metavar="PATH", help="the snapshot to compare against")
    parser.add_argument("--final-10-file", metavar="PATH", help="pointer file naming the final 10 GHz run dir")
    parser.add_argument("--final-77-file", metavar="PATH", help="pointer file naming the final 77 GHz run dir")
    parser.add_argument("--output", required=True, metavar="PATH")

    parser.add_argument("--no-hash-npz", action="store_true",
                        help="skip hashing the store .npz files (faster; weaker tamper evidence)")
    parser.add_argument("--allow-store-commit-divergence", action="store_true",
                        help="write a DEGRADED, non-authoritative snapshot when the store's build "
                             "commit differs from the reference run's — a mechanism check only; "
                             "--compare refuses such a manifest")
    args = parser.parse_args(argv)
    hash_npz = not args.no_hash_npz

    if args.snapshot:
        run_dirs = {}
        if args.reference_10:
            run_dirs["10ghz"] = args.reference_10
        if args.reference_77:
            run_dirs["77ghz"] = args.reference_77
        if not run_dirs:
            parser.error("--snapshot needs --reference-10 and/or --reference-77")

        superseded = {"10ghz": args.superseded_10 or [], "77ghz": args.superseded_77 or []}
        configs = _configs(args, sorted(run_dirs))
        manifest = gate.snapshot(
            configs, run_dirs, hash_npz=hash_npz,
            allow_store_commit_divergence=args.allow_store_commit_divergence,
            superseded_run_dirs=superseded,
        )
        path = gate.write_json(args.output, manifest)
        _print_snapshot(manifest)
        print(f"\nwrote {path}")
        if manifest["reference_grade"] != gate.GRADE_AUTHORITATIVE:
            print(
                "\nDEGRADED snapshot — this is a mechanism check, NOT the milestone reference.\n"
                "The store it was taken from did not back the reference runs, so --compare will\n"
                "refuse it. Re-run against the store that produced them before any structural\n"
                "edit or store rebuild."
            )
            return 3
        return 0

    for required in ("reference_manifest", "final_10_file", "final_77_file"):
        if getattr(args, required) is None:
            parser.error(f"--compare needs --{required.replace('_', '-')}")

    manifest = json.loads(Path(args.reference_manifest).read_text(encoding="utf-8"))
    final_run_dirs = {
        "10ghz": _read_pointer(args.final_10_file),
        "77ghz": _read_pointer(args.final_77_file),
    }
    configs = _configs(args, sorted(final_run_dirs))
    sources = gate.compare(manifest, configs, final_run_dirs, hash_npz=hash_npz)
    path = gate.write_json(args.output, sources)
    _print_comparison(sources)
    print(f"\nwrote {path}")
    if sources["status"] != "approved":
        print(
            "\nMISMATCH — milestone-stopping. Experiment F and the assembly must not run:\n"
            "the rebuilt store does not reproduce the reference Exp A analysis, and that is a\n"
            "scientific finding to explain, not drift to excuse."
        )
        return 4
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except gate.ReferenceGateError as error:
        print(f"reference gate REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2) from None
