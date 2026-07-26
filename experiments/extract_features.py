"""Build the per-session WST feature store for a band (milestone 7).

The store is what Exp A's harness consumes: data-independent session vectors (log off /
on_frozen_eps), pre-log scales, and the RAW pre-log tensors the fold-local tuned-ε branch
reconstructs from. It is built ONCE, from a CLEAN committed revision, and validated
fail-closed before any analysis run reads it.

    # local (10 GHz, whole eligible cohort)
    uv run python experiments/extract_features.py --config configs/exp_a_regression.yaml --band 10ghz

    # one IBEX shard (77 GHz job array)
    uv run python experiments/extract_features.py --config configs/exp_a_regression_77ghz.yaml \
        --config configs/ibex.yaml --band 77ghz --subject 3 --session 12pm

    # validate a built store against what THIS config/commit would produce (fail-closed)
    uv run python experiments/extract_features.py --config ... --band 10ghz --validate

Both producers REFUSE a dirty tree (a store must be attributable to a clean revision, C7/C16).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import load_config, require_77ghz_dir  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.loader_10ghz import load_10ghz_file  # noqa: E402
from dehyd.data.loader_77ghz import load_77ghz_file  # noqa: E402
from dehyd.data.manifest import apply_qc, build_manifest, eligible_frames, resolve_path  # noqa: E402
from dehyd.data.manifest_77 import apply_qc_77, build_manifest_77, resolve_path_77  # noqa: E402
from dehyd.features import store as S  # noqa: E402
from dehyd.provenance import _git_info  # noqa: E402


def eligible_sessions(config, band: str, *, subject=None, session=None) -> list[dict]:
    """One record per eligible session: subject, session_idx, session_name, rel_path, frame_ids.

    In SHARD mode (`subject`/`session` given — one IBEX array task) the manifest is filtered to
    that cell BEFORE QC, so a task loads and QC-screens ONLY its own file instead of re-running
    the full-cohort QC over all 80 files. Eligibility of a session depends solely on that
    session's own frames, so this is correct as well as ~80x cheaper across the array."""
    gt = load_ground_truth(config.paths.weight_xlsx)
    if band == "10ghz":
        manifest = build_manifest(config.paths, gt)
    else:
        require_77ghz_dir(config)
        manifest = build_manifest_77(config.paths, gt)
    if subject is not None:
        manifest = manifest[manifest["subject"] == subject]
    if session is not None:
        manifest = manifest[manifest["session_name"] == session]
    apply = apply_qc if band == "10ghz" else apply_qc_77
    manifest_qc = apply(manifest, config.paths, config)
    pop = eligible_frames(manifest_qc)
    records = []
    for (subject, session_idx), group in pop.groupby(["subject", "session_idx"]):
        records.append(
            {
                "subject": int(subject),
                "session_idx": int(session_idx),
                "session_name": group["session_name"].iloc[0],
                "rel_path": group["rel_path"].iloc[0],
                "frame_ids": group["frame_idx"].tolist(),
            }
        )
    return records


def _raw_path(config, band, rel_path):
    return resolve_path(config.paths, rel_path) if band == "10ghz" else resolve_path_77(config.paths, rel_path)


def _build_npz(config, band, cube, frame_ids):
    if band == "10ghz":
        return S.build_session_npz_10ghz(cube, frame_ids, config)
    # 77 GHz: QC/eligibility already selected the frames; pass the eligible-frame cube.
    return S.build_session_npz_77ghz(cube[frame_ids], config)


def _load_cube(config, band, rel_path):
    path = _raw_path(config, band, rel_path)
    return load_10ghz_file(path) if band == "10ghz" else load_77ghz_file(path)


def _expected_fingerprint(config, band, sess):
    return S.compute_fingerprint(
        config, band,
        frame_ids=sess["frame_ids"],
        raw_path=_raw_path(config, band, sess["rel_path"]),
        session_eligible=True,
    )


def build_one(config, band, sess, store_dir) -> None:
    cube = _load_cube(config, band, sess["rel_path"])
    npz = _build_npz(config, band, cube, sess["frame_ids"])
    fp = _expected_fingerprint(config, band, sess)
    path = S.write_session_store(band, sess["subject"], sess["session_name"], npz, fp, store_dir)
    print(f"  wrote s{sess['subject']} {sess['session_name']:<5} -> {path.name} ({len(npz)} arrays)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--band", choices=("10ghz", "77ghz"), required=True)
    parser.add_argument("--subject", type=int, help="build only this subject (shard mode)")
    parser.add_argument("--session", help="build only this session name (shard mode)")
    parser.add_argument("--validate", action="store_true", help="validate the built store, fail-closed")
    args = parser.parse_args(argv)

    config = load_config(*args.config)
    store_dir = config.paths.results_dir
    # Pass the shard filter INTO eligibility so a task QC-loads only its own file(s), not the
    # whole cohort (a --validate over all sessions still QCs the full cohort once, correctly).
    sessions = eligible_sessions(config, args.band, subject=args.subject, session=args.session)

    if args.validate:
        expected = {(s["subject"], s["session_name"]): _expected_fingerprint(config, args.band, s) for s in sessions}
        S.validate_store(args.band, store_dir, expected, analysis_commit=_git_info()["commit"])
        print(f"validate : {args.band} store OK — {len(expected)} sessions match this config/commit")
        return 0

    S.assert_clean_tree()  # both producers refuse a dirty tree (C7/C16)
    print(f"building : {args.band} feature store, {len(sessions)} sessions -> {store_dir}/features/{args.band}")
    for sess in sessions:
        build_one(config, args.band, sess, store_dir)
    print(f"done     : {len(sessions)} sessions written. Run with --validate to verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
