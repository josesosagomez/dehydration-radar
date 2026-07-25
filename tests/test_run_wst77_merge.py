"""run_wst77.py --merge-shards, exercised to completion (no private data).

No test previously ran run_merge end to end, which is how two bugs slipped through: a
self-consistent STALE shard set merging silently, and a variable-shadowing error that made the
final cell-count assertion compare against a fingerprint dict. This drives the real function on
synthetic shards: a clean merge writes the curated CSV, a stale fingerprint aborts, and a
disagreement between shards aborts.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import run_wst77  # noqa: E402

NAMES = ["8am", "10am", "12pm", "2pm", "4pm"]


def _config(tmp_path):
    d10 = tmp_path / "d10"; d10.mkdir()
    d77 = tmp_path / "d77"; d77.mkdir()
    xlsx = tmp_path / "w.xlsx"; xlsx.write_bytes(b"")
    results = tmp_path / "results"
    (results / "qc").mkdir(parents=True)
    (results / "wst" / "shards").mkdir(parents=True)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "paths": {"data_10ghz_dir": str(d10), "weight_xlsx": str(xlsx),
                  "results_dir": str(results), "data_77ghz_dir": str(d77)},
        "run": {"seed": 1, "seed_set": [1, 2, 3, 4, 5], "device": "cpu"},
    }), encoding="utf-8")
    from dehyd.config import load_config
    return load_config(cfg), d77, results


def _make_cohort(tmp_path, eligible_cells, ineligible_cells=()):
    """Survival CSV + one shard (+ matching fingerprint) per eligible cell + a raw .mat each."""
    config, d77, results = _config(tmp_path)

    rows = []
    for subj, sess in list(eligible_cells) + list(ineligible_cells):
        idx = NAMES.index(sess)
        elig = (subj, sess) in eligible_cells
        rows.append({"subject": subj, "session_idx": idx, "session_name": sess,
                     "rel_path": f"subject_{subj}_{sess}.mat", "n_frames": 125,
                     "n_pass": 125 if elig else 0, "n_fail_any": 0 if elig else 125,
                     "min_pass": 63, "eligible": elig})
        (d77 / f"subject_{subj}_{sess}.mat").write_bytes(f"raw-{subj}-{sess}".encode())
    pd.DataFrame(rows).to_csv(results / "qc" / "qc_survival_77ghz.csv", index=False)

    shard_dir = results / "wst" / "shards"
    for subj, sess in eligible_cells:
        idx = NAMES.index(sess)
        pd.DataFrame([{"subject": subj, "session_idx": idx, "session_name": sess,
                       "rel_path": f"subject_{subj}_{sess}.mat", "n_eligible_frames": 125,
                       "tiling_idx": t, "fusion": f, "n_paths": 424, "n_time": 8,
                       "prelog_v0": 0.0, "prelog_v1": 0.0, "prelog_v2": 0.0,
                       "all_finite": True, "extract_seconds": 1.0}
                      for t in range(3) for f in ("mean", "median")]).to_csv(
            shard_dir / f"wst77_s{subj}_{sess}.csv", index=False)
        fp = run_wst77._fingerprint(config, d77 / f"subject_{subj}_{sess}.mat")
        (shard_dir / f"wst77_s{subj}_{sess}.fingerprint.json").write_text(
            json.dumps(fp), encoding="utf-8")
    return config, results, shard_dir


def test_merge_writes_curated_csv_with_the_right_cell_count(tmp_path):
    cells = [(1, "8am"), (1, "10am"), (2, "8am")]
    config, results, _ = _make_cohort(tmp_path, cells, ineligible_cells=[(2, "10am")])
    assert run_wst77.run_merge(config) == 0
    out = pd.read_csv(results / "wst" / "wst_diagnostics_77ghz.csv")
    assert out.groupby(["subject", "session_idx"]).ngroups == len(cells)  # not len(a dict)
    assert len(out) == len(cells) * 6  # 3 tilings x 2 fusions


def test_merge_rejects_a_stale_fingerprint(tmp_path):
    """A wholly-stale-but-self-consistent set must NOT merge silently."""
    cells = [(1, "8am"), (1, "10am")]
    config, _, shard_dir = _make_cohort(tmp_path, cells)
    for fp_path in shard_dir.glob("*.fingerprint.json"):
        fp = json.loads(fp_path.read_text())
        fp["frame_selection"] = "all_frames_of_eligible_sessions"  # the pre-fix rule
        fp_path.write_text(json.dumps(fp), encoding="utf-8")
    with pytest.raises(SystemExit, match="STALE"):
        run_wst77.run_merge(config)


def test_merge_rejects_shards_that_disagree_with_each_other(tmp_path):
    cells = [(1, "8am"), (1, "10am"), (2, "8am")]
    config, _, shard_dir = _make_cohort(tmp_path, cells)
    odd = next(shard_dir.glob("*.fingerprint.json"))
    fp = json.loads(odd.read_text())
    fp["axis_spec_hash"] = "a-different-hash"
    odd.write_text(json.dumps(fp), encoding="utf-8")
    with pytest.raises(SystemExit, match="disagree"):
        run_wst77.run_merge(config)


def test_merge_aborts_on_a_missing_eligible_shard(tmp_path):
    cells = [(1, "8am"), (1, "10am")]
    config, _, shard_dir = _make_cohort(tmp_path, cells)
    next(shard_dir.glob("wst77_*.csv")).unlink()
    with pytest.raises(SystemExit, match="missing shard"):
        run_wst77.run_merge(config)
