"""Milestone 5 — 77 GHz WST feature diagnostics (band 2).

Three modes over the SAME library code (features/extraction_77.py):

  * default / array  : curated cohort feature diagnostics. Per eligible (subject, session)
      cell it extracts every variant and records dims, finiteness, the (tiling,fusion)
      pre-log scales, and timing -> results/wst/wst_diagnostics_77ghz.csv. With
      --subject/--session it writes ONE deterministic shard
      results/wst/shards/wst77_s<subj>_<sess>.csv (+ a fingerprint sidecar) — the IBEX
      job-array unit. Requires authoritative eligibility, so it fails closed if the
      cohort QC survival CSV is absent.
  * --merge-shards   : verify exactly the eligible shards (no duplicates, fingerprints
      agree) and write the curated CSV.
  * --smoke          : a NON-curated functional smoke (D5 / T-R77), outcome-independent.
      Fixed frame indices of one file, only the NaN/Inf + axis-cert guards (never the
      flatline rule / eligibility), one tiling, finite + expected-dims check, writes NO
      shard or curated artifact.

    uv run python experiments/run_wst77.py --config configs/exp_77ghz.yaml --smoke --subject 1 --session 8am

canonical_spec_guard_77 runs first (curated modes); every file is axis-certified before any
feature is written; numpy backs all reported features.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import load_config, require_77ghz_dir  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.loader_77ghz import load_77ghz_file  # noqa: E402
from dehyd.data.manifest_77 import build_manifest_77, resolve_path_77  # noqa: E402
from dehyd.data.sessions import SESSION_INDEX, SESSION_NAMES  # noqa: E402
from dehyd.features.extraction_77 import (  # noqa: E402
    canonical_spec_guard_77,
    extract_session_features_77,
    extract_session_variants_77,
    wst77_spec,
)
from dehyd.provenance import _git_info, record_run, sha256_file  # noqa: E402
from dehyd.qc.axis_check_77 import axis_spec_hash, require_accepted_axis  # noqa: E402

DIAG_NAME = "wst_diagnostics_77ghz.csv"
SURVIVAL_CSV = "qc_survival_77ghz.csv"
FRAMES_CSV = "qc_frames_77ghz.csv"
N_SMOKE_FRAMES = 3


def _first_finite_frames(cube, k):
    """The first k frames with no non-finite sample (the smoke's fixed, flatline-free set)."""
    idx = [i for i in range(cube.shape[0]) if np.all(np.isfinite(cube[i]))][:k]
    if len(idx) < 1:
        raise SystemExit("no finite frames in the file")
    return cube[idx], idx


def run_smoke(config, subject, session):
    """Non-curated functional smoke — outcome-independent, writes nothing curated."""
    session = session or "8am"
    rel = f"subject_{subject}_{session}.mat"
    path = resolve_path_77(config.paths, rel)
    survival = Path(config.paths.results_dir) / "qc" / SURVIVAL_CSV
    require_accepted_axis(path, config, survival_csv=survival if survival.exists() else None)

    cube = load_77ghz_file(path)
    small, idx = _first_finite_frames(cube, N_SMOKE_FRAMES)
    tiling = config.wst77.tilings[0]
    t0 = time.time()
    vec = extract_session_features_77(
        small, config.preprocess77, config.wst77,
        tiling=tiling, log_branch="on_frozen_eps", fusion="mean", family="pooled",
    )
    dt = time.time() - t0
    ok = bool(np.all(np.isfinite(vec)))
    print(f"smoke        : {rel} frames {idx} tiling Q={tiling.q}")
    print(f"  features   : D={vec.shape[0]}  finite={ok}  {dt:.1f}s  (analysis_role=smoke)")
    if not ok:
        raise SystemExit("smoke produced non-finite features")
    return 0


def _fingerprint(config, path):
    """Everything a shard's validity depends on — run/config, code rev, QC + axis, raw file.

    `frame_selection` is part of the fingerprint because it determines WHICH frames entered
    the features; a shard built before the eligible-frame filter must not merge with one
    built after it.
    """
    return {
        "git": _git_info(),
        "wst77_backend": config.wst77.backend,
        "axis_spec_hash": axis_spec_hash(config),
        "frame_selection": "qc_pass_frames_of_eligible_sessions",
        "raw_sha256": sha256_file(path),
    }


def _session_diag_rows(cube, config, subject, session_idx, rel_path):
    """One session's variant extraction -> diagnostic rows (one per (tiling,fusion)).

    `cube` must ALREADY be subset to the session's QC-passing frames — the analysis
    population is `eligible_frames` (passing frames of eligible sessions), not every frame
    of an eligible session.
    """
    t0 = time.time()
    result = extract_session_variants_77(cube, config.preprocess77, config.wst77)
    dt = time.time() - t0
    rows = []
    for (ti, fusion), scale in result.prelog_scale.items():
        n_paths, n_time = result.shapes[ti]
        rows.append({
            "subject": subject, "session_idx": session_idx,
            "session_name": SESSION_NAMES[session_idx], "rel_path": rel_path,
            "n_eligible_frames": cube.shape[0], "tiling_idx": ti, "fusion": fusion,
            "n_paths": n_paths, "n_time": n_time,
            "prelog_v0": scale[0], "prelog_v1": scale[1], "prelog_v2": scale[2],
            "all_finite": result.all_finite, "extract_seconds": round(dt, 2),
        })
    return rows


def run_curated(config, args):
    """Curated cohort / array mode. Requires the authoritative survival CSV (outcome b)."""
    canonical_spec_guard_77(config)
    survival = Path(config.paths.results_dir) / "qc" / SURVIVAL_CSV
    if not survival.exists():
        raise SystemExit(
            f"{survival} not found — the curated feature run requires authoritative "
            "eligibility from run_qc77.py (outcome b); run the cohort QC first."
        )
    survival_df = pd.read_csv(survival)
    eligible = survival_df[survival_df["eligible"]]

    # The analysis population is eligible_frames: the QC-PASSING frames of eligible sessions.
    # Session-level eligibility alone is not enough — a session can be eligible while still
    # containing failing frames, and those must not reach the features (they would contaminate
    # the frame-mean/median session vector and diverge from the 10 GHz arm).
    frames_path = Path(config.paths.results_dir) / "qc" / FRAMES_CSV
    if not frames_path.exists():
        raise SystemExit(
            f"{frames_path} not found — the curated feature run needs the per-frame QC to "
            "select the passing frames of each eligible session; run run_qc77.py first."
        )
    frames_df = pd.read_csv(frames_path)
    passing = {
        rel: sorted(int(i) for i in g.loc[g["qc_pass"], "frame_idx"])
        for rel, g in frames_df.groupby("rel_path")
    }

    gt = load_ground_truth(config.paths.weight_xlsx)
    manifest = build_manifest_77(config.paths, gt)

    cells = eligible[["subject", "session_name"]].itertuples(index=False)
    if args.subject is not None:
        cells = [c for c in cells if c.subject == args.subject
                 and (args.session is None or c.session_name == args.session)]
    else:
        cells = list(cells)

    out_dir = Path(config.paths.results_dir) / "wst"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for cell in cells:
        session_idx = SESSION_INDEX[cell.session_name]
        rel = f"subject_{cell.subject}_{cell.session_name}.mat"
        path = resolve_path_77(config.paths, rel)
        require_accepted_axis(path, config, survival_csv=survival)  # per-file guard
        keep = passing.get(rel)
        if not keep:
            raise SystemExit(f"{rel}: no QC-passing frames found in {FRAMES_CSV}")
        cube = load_77ghz_file(path)[keep]  # <- the analysis population, not all frames
        rows = _session_diag_rows(cube, config, cell.subject, session_idx, rel)
        all_rows.extend(rows)
        print(f"  {rel}: {len(keep)}/125 eligible frames, {len(rows)} variant rows, "
              f"finite={rows[0]['all_finite']}")

    frame = pd.DataFrame(all_rows)
    if args.subject is not None:  # a single-cell SHARD
        if args.session is None:
            raise SystemExit("--subject requires --session for a deterministic shard")
        shard_dir = out_dir / "shards"
        shard_dir.mkdir(exist_ok=True)
        sess = args.session
        shard = shard_dir / f"wst77_s{args.subject}_{sess}.csv"
        frame.to_csv(shard, index=False)
        (shard.with_suffix(".fingerprint.json")).write_text(
            json.dumps(_fingerprint(config, resolve_path_77(config.paths,
                       f"subject_{args.subject}_{sess}.mat")), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"shard        : {shard}")
    else:  # the whole eligible cohort in one job
        diag = out_dir / DIAG_NAME
        frame.to_csv(diag, index=False)
        assert len(pd.read_csv(diag)) == len(frame), "diagnostics lost rows on write"
        print(f"diagnostics  : {diag}")
        record_run(config, manifest, folds=None, data_dir=require_77ghz_dir(config),
                   extra={"stage": "milestone-5-wst77", "analysis_role": "primary",
                          "n_cells": len(cells), "wst77_spec": wst77_spec(config.wst77, config.preprocess77)})
    return 0


def run_merge(config):
    """Verify the eligible shards agree and write the curated CSV."""
    out_dir = Path(config.paths.results_dir) / "wst"
    shard_dir = out_dir / "shards"
    survival = pd.read_csv(Path(config.paths.results_dir) / "qc" / SURVIVAL_CSV)
    eligible = survival[survival["eligible"]]
    expected = {(int(r.subject), r.session_name) for r in eligible.itertuples()}

    shards, fingerprints = [], []
    for subject, session in sorted(expected):
        shard = shard_dir / f"wst77_s{subject}_{session}.csv"
        fp = shard.with_suffix(".fingerprint.json")
        if not shard.exists() or not fp.exists():
            raise SystemExit(f"missing shard or fingerprint for subject {subject} {session}")
        shards.append(pd.read_csv(shard))
        fingerprints.append(json.loads(fp.read_text()))

    # Two independent checks. (1) The shards must agree with EACH OTHER (no stale retry mixed
    # into a fresh set). (2) They must also agree with what THIS config/code would produce --
    # without it, a wholly-stale set is self-consistent and would merge silently. `git` and
    # `raw_sha256` are excluded from (2): git is unreadable on the compute nodes (recorded but
    # not comparable across machines) and raw_sha256 is per-file by construction.
    SEMANTIC = ("wst77_backend", "axis_spec_hash", "frame_selection")
    ref = {k: v for k, v in fingerprints[0].items() if k != "raw_sha256"}
    for fp in fingerprints[1:]:
        if {k: v for k, v in fp.items() if k != "raw_sha256"} != ref:
            raise SystemExit("shard fingerprints disagree (code/config/QC-rule mismatch)")

    expected = _fingerprint(config, resolve_path_77(config.paths, eligible.iloc[0].rel_path))
    for key in SEMANTIC:
        got, want = fingerprints[0].get(key), expected.get(key)
        if got != want:
            raise SystemExit(
                f"shards are STALE: fingerprint {key}={got!r} but this config/code produces "
                f"{want!r}. Re-run the array; do not merge shards built under a different rule."
            )

    merged = pd.concat(shards, ignore_index=True)
    n_cells = merged[["subject", "session_idx"]].drop_duplicates().shape[0]
    if n_cells != len(expected):
        raise SystemExit(f"merged {n_cells} cells but expected {len(expected)} eligible")
    diag = out_dir / DIAG_NAME
    merged.to_csv(diag, index=False)
    print(f"merged {len(shards)} shards -> {diag} ({n_cells} cells)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--session", choices=SESSION_NAMES, default=None)
    parser.add_argument("--smoke", action="store_true", help="non-curated functional smoke")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(*args.config)
    require_77ghz_dir(config)
    print(f"config       : {', '.join(args.config)}  backend {config.wst77.backend}")

    if args.smoke:
        return run_smoke(config, args.subject or 1, args.session)
    if args.merge_shards:
        return run_merge(config)
    return run_curated(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
