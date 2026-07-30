"""T-M9-entrypoints (Exp C half): `experiments/run_ordinal.py`.

Flag validation, the mechanism-only reporting boundary, and the wiring the milestone's LATER
steps depend on — the run's `config_hash` (which the sanctioned exploratory frame split
validates its source artifact's lineage against, C24) and the band-correct `data_dir`
(without which a 77 GHz run hashes 10 GHz bytes under a 77 GHz label, C22).

The staged two-arm search itself is covered by test_exp_c.py; here `run_and_report_c` is
monkeypatched so the CLI's OWN wiring is tested fast, against a REAL `record_run` call.
"""

import json

import pandas as pd
import pytest

from dehyd.config import load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_b, exp_c

import experiments.run_ordinal as ro


@pytest.fixture(scope="module")
def config():
    return load_config("configs/exp_a_regression.yaml", "configs/exp_c.yaml")


CONFIG_ARGS = ["--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_c.yaml"]


def _sessions(n_subjects=8):
    return [
        {"subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
         "rel_path": f"subject_{s}_{SESSION_NAMES[i]}.mat", "frame_ids": [0],
         "delta_m_pct": -0.3 * i, "loss_l": 0.3 * i, "class_idx": i}
        for s in range(1, n_subjects + 1)
        for i in range(5)
    ]


def _manifest(data_dir, sessions):
    rows = []
    for s in sessions:
        (data_dir / s["rel_path"]).write_bytes(f"radar {s['rel_path']}".encode())
        rows.append({"subject": s["subject"], "session_idx": s["session_idx"],
                     "rel_path": s["rel_path"]})
    return pd.DataFrame(rows)


def _overlay(tmp_path, *, with_77=False):
    data_10 = tmp_path / "data10"
    results = tmp_path / "results"
    for directory in (data_10, results):
        directory.mkdir(exist_ok=True)
    (tmp_path / "weights.xlsx").write_bytes(b"pretend workbook")
    lines = ["paths:",
             f"  data_10ghz_dir: {data_10.as_posix()}",
             f"  weight_xlsx: {(tmp_path / 'weights.xlsx').as_posix()}",
             f"  results_dir: {results.as_posix()}"]
    if with_77:
        data_77 = tmp_path / "data77"
        data_77.mkdir(exist_ok=True)
        lines.append(f"  data_77ghz_dir: {data_77.as_posix()}")
    path = tmp_path / "paths_overlay.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, data_10, results


def _patch(monkeypatch, sessions, data_dir, captured):
    monkeypatch.setattr(ro.exp_c, "build_sessions_c", lambda *a, **k: sessions)
    monkeypatch.setattr(ro, "_build_manifest_qc", lambda config, band: _manifest(data_dir, sessions))

    def fake_run_and_report(config, band, sess, store_dir, run_dir, *, mode, analysis_commit,
                            n_workers=1):
        captured.update({"band": band, "mode": mode, "n_sessions": len(sess),
                         "run_dir": run_dir})
        return {"run_log": run_dir / f"run_log_{band}.json"}

    monkeypatch.setattr(ro.exp_c, "run_and_report_c", fake_run_and_report)


# ------------------------------------------------------------------- flag validation


def test_exactly_one_of_subset_or_full_cohort_is_required():
    with pytest.raises(SystemExit):
        ro.main(CONFIG_ARGS)
    with pytest.raises(SystemExit):
        ro.main(CONFIG_ARGS + ["--subset", "6subjects", "--full-cohort"])


def test_unknown_band_is_rejected():
    with pytest.raises(SystemExit):
        ro.main(CONFIG_ARGS + ["--band", "24ghz", "--full-cohort"])


# ------------------------------------------------------- the real record_run contract


def test_full_cohort_records_the_config_hash_the_frame_split_validates(tmp_path, monkeypatch,
                                                                       config):
    """(C24) The exploratory frame split refuses a source artifact whose `config_hash`
    differs from its own run's, so the Exp C run has to record one — and it must be the SAME
    named helper every other run group uses, never a second hashing recipe."""
    overlay, data_10, results = _overlay(tmp_path)
    sessions = _sessions()
    captured = {}
    _patch(monkeypatch, sessions, data_10, captured)

    assert ro.main(CONFIG_ARGS + ["--config", str(overlay), "--full-cohort"]) == 0

    run_dirs = sorted((results / "runs").iterdir())
    payload = json.loads((run_dirs[0] / "provenance.json").read_text(encoding="utf-8"))
    loaded = load_config("configs/exp_a_regression.yaml", "configs/exp_c.yaml", str(overlay))
    assert payload["extra"]["config_hash"] == exp_b.config_fingerprint(loaded)
    assert payload["extra"]["stage"] == "exp-c-full"
    assert payload["extra"]["band"] == "10ghz"
    assert payload["extra"]["n_eval"] == 8
    assert len(payload["folds"]) == 8            # a real fold-role manifest, not an empty slot
    assert captured["mode"] == "full" and captured["n_sessions"] == 40
    assert captured["run_dir"] == run_dirs[0]


def test_smoke_keeps_the_six_lowest_subjects_and_surfaces_no_performance_value(tmp_path,
                                                                               monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    sessions = _sessions()
    captured = {}
    _patch(monkeypatch, sessions, data_10, captured)

    assert ro.main(CONFIG_ARGS + ["--config", str(overlay), "--subset", "6subjects"]) == 0

    assert captured["mode"] == "smoke"
    assert captured["n_sessions"] == 30          # the 6 lowest subjects x 5 sessions
    payload = json.loads(
        (sorted((results / "runs").iterdir())[0] / "provenance.json").read_text(encoding="utf-8")
    )
    assert payload["extra"]["stage"] == "exp-c-smoke"
    # the CLI itself writes no metric file; the reporting boundary lives in run_and_report_c
    assert not list(results.rglob("metrics_exp_c_*"))


def test_77ghz_run_hashes_against_the_77ghz_root(tmp_path, monkeypatch):
    """(C22) `_hash_inputs` defaults to the 10 GHz root; the entrypoint must pass
    `require_77ghz_dir(config)`. Two roots holding same-named files with different bytes make
    the difference observable."""
    overlay, data_10, results = _overlay(tmp_path, with_77=True)
    sessions = _sessions(n_subjects=4)
    _patch(monkeypatch, sessions, data_10, {})
    data_77 = tmp_path / "data77"
    for s in sessions:
        (data_77 / s["rel_path"]).write_bytes(f"SEVENTY-SEVEN {s['rel_path']}".encode())

    assert ro.main(CONFIG_ARGS + ["--config", str(overlay), "--band", "77ghz",
                                  "--subset", "6subjects"]) == 0

    payload = json.loads(
        (sorted((results / "runs").iterdir())[0] / "provenance.json").read_text(encoding="utf-8")
    )
    hashes = {e["rel_path"]: e["sha256"] for e in payload["inputs"]["radar_files"]}
    from dehyd.provenance import sha256_file

    for rel_path, digest in hashes.items():
        assert digest == sha256_file(data_77 / rel_path)
        assert digest != sha256_file(data_10 / rel_path)


def test_exp_c_fits_no_baseline_estimator():
    """(§5 trap 16) The session-index baseline predicts the class PERFECTLY — the class IS the
    session index — so any radar-vs-baseline framing for Exp C is degenerate and the freeze
    registers none. Checked structurally: neither the entrypoint nor Exp C's composition
    imports the baseline estimators at all, so no comparison can be computed by accident."""
    import ast
    from pathlib import Path

    for module in (ro, exp_c):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "baselines" not in (node.module or ""), module.__name__
                assert all(alias.name != "baselines" for alias in node.names), module.__name__
            if isinstance(node, ast.Import):
                assert all("baselines" not in alias.name for alias in node.names)
