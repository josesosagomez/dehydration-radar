"""T-M8-entrypoint / T-M8-variant (CLI subset): the Exp B entrypoint — flag validation, the
mechanism-only reporting boundary (C9/C14), and the `--session-specific` sub-modes' wiring
against the REAL `provenance.record_run` contract (C19/C20/C21/C22) and the REAL
`merge_session_specific_reports`/`run_exp_b_one_session` functions.

The expensive staged search itself is covered by test_exp_b.py; here the primary path's
`run_and_report_b` (like `test_run_regression.py` does for Exp A) and the session-specific
sub-modes' heavier dependencies are monkeypatched so the CLI's OWN wiring is tested fast and
in isolation -- except `--init-run-group`, which is deliberately exercised against a REAL
`record_run` call (real file hashing, real provenance.json) so the C19-C22 contract bugs the
plan's own review caught from-memory are actually caught here, not just described in prose.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dehyd import provenance as provenance_mod
from dehyd.config import load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_a, exp_b
from dehyd.eval.harness import FitRecord, SeedOutcome
from dehyd.eval.splits import nested_loso_splits
from dehyd.features.store import write_session_store


@pytest.fixture(scope="module")
def config():
    return load_config("configs/exp_a_regression.yaml", "configs/exp_b.yaml")


def _sessions(n_subjects=4, session_indices=(1, 2, 3, 4)):
    return [
        {
            "subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
            "delta_m_pct": -0.3 * i - 0.05 * s,
            "rel_path": f"subject_{s}_{SESSION_NAMES[i]}.mat", "frame_ids": [0],
        }
        for s in range(1, n_subjects + 1)
        for i in session_indices
    ]


def _minimal_store(store_dir, sessions):
    for s in sessions:
        write_session_store("10ghz", s["subject"], s["session_name"], {"x": np.zeros(1)}, {"n_frames": 1}, store_dir)


def _fake_results_b(sessions):
    subjects = sorted({s["subject"] for s in sessions})
    out = []
    for test_subj in subjects:
        train = frozenset(x for x in subjects if x != test_subj)
        rows = [s for s in sessions if s["subject"] == test_subj]
        y = np.array([s["delta_m_pct"] for s in rows])
        session_idx = np.array([s["session_idx"] for s in rows])
        pred = y + 0.1
        out.append(exp_b.ExpBFoldResult(
            test_subject=test_subj,
            selected_feature_key=(0, "A", "mag", 0, "off"),
            selected_family="ridge",
            selected_params={"alpha": 1.0},
            test_predictions=pred,
            test_targets=y,
            test_session_idx=session_idx,
            seed_outcomes=[SeedOutcome(0, pred, pred, 0.1)],
            baseline_predictions=np.zeros_like(y),
            final_fits=[FitRecord("scaler", "outer_train", train, {"mean_": np.zeros(1)})],
            dropped_sessions_outer=(),
            dropped_sessions_inner=(),
            reason=None,
        ))
    return out


def _patch_primary(monkeypatch, sessions):
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda *a, **k: {})
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)
    monkeypatch.setattr(exp_b, "run_exp_b", lambda *a, **k: _fake_results_b(sessions))


def _config_with_tmp_dirs(config, tmp_path, *, with_77ghz=False):
    """A real Config, `dataclasses.replace`d to point paths at tmp directories, so a real
    `record_run`/`validate_store` call never touches the real project's data/ or results/."""
    data_10 = tmp_path / "data10"
    data_10.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    paths = dataclasses.replace(config.paths, data_10ghz_dir=data_10, results_dir=results)
    if with_77ghz:
        data_77 = tmp_path / "data77"
        data_77.mkdir()
        paths = dataclasses.replace(paths, data_77ghz_dir=data_77)
    return dataclasses.replace(config, paths=paths), data_10, results


def _minimal_manifest_qc(data_dir, sessions):
    """A minimal but REAL manifest DataFrame: one dummy raw file per session, hashed for real
    by `record_run`'s `_hash_inputs`. Bypasses the QC/ground-truth pipeline entirely (out of
    scope for M8), but exercises the REAL file-hashing/payload-construction code `record_run`
    itself runs -- the part C19-C22 actually caught from-memory bugs in."""
    rows = []
    for s in sessions:
        (data_dir / s["rel_path"]).write_bytes(b"dummy radar bytes")
        rows.append({"subject": s["subject"], "session_idx": s["session_idx"], "rel_path": s["rel_path"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- flag validation


def test_full_cohort_refused_without_flag(config):
    import experiments.run_clock_decoupling as rc

    with pytest.raises(SystemExit):
        rc.main(["--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml"])


def test_subset_and_full_cohort_are_mutually_exclusive():
    import experiments.run_clock_decoupling as rc

    with pytest.raises(SystemExit):
        rc.main([
            "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml",
            "--subset", "6subjects", "--full-cohort",
        ])


def test_session_specific_mutually_exclusive_with_subset():
    import experiments.run_clock_decoupling as rc

    with pytest.raises(SystemExit):
        rc.main([
            "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml",
            "--session-specific", "--subset", "6subjects",
        ])


def test_session_specific_requires_exactly_one_sub_mode():
    import experiments.run_clock_decoupling as rc

    with pytest.raises(SystemExit):
        rc.main(["--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml", "--session-specific"])
    with pytest.raises(SystemExit):
        rc.main([
            "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml",
            "--session-specific", "--init-run-group", "--merge-sessions", "--run-dir", "x",
        ])


def test_session_flag_requires_run_dir():
    import experiments.run_clock_decoupling as rc

    with pytest.raises(SystemExit):
        rc.main([
            "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml",
            "--session-specific", "--session", "1",
        ])


# -------------------------------------------------------- mechanism-only reporting (C9/C14)


def test_smoke_writes_no_performance_value(tmp_path, config, monkeypatch):
    sessions = _sessions()
    _minimal_store(tmp_path, sessions)
    _patch_primary(monkeypatch, sessions)
    run_dir = tmp_path / "run"

    outputs = exp_b.run_and_report_b(config, "10ghz", sessions, tmp_path, run_dir, mode="smoke", analysis_commit="x")

    assert set(outputs) == {"run_log"}
    files = {p.name for p in run_dir.iterdir()}
    assert not any(n.startswith(("metrics_", "predictions_b_", "scatter_b_", "selection_table_b_")) for n in files)
    log = json.loads(outputs["run_log"].read_text())
    assert log["mode"] == "mechanism-only" and "note" in log


def test_full_run_writes_metrics_and_scatter(tmp_path, config, monkeypatch):
    sessions = _sessions()
    _minimal_store(tmp_path, sessions)
    _patch_primary(monkeypatch, sessions)
    run_dir = tmp_path / "run"

    outputs = exp_b.run_and_report_b(config, "10ghz", sessions, tmp_path, run_dir, mode="full", analysis_commit="x")

    assert "metrics" in outputs and "scatter" in outputs
    assert outputs["metrics"].exists() and outputs["scatter"].exists()
    summary = json.loads(outputs["metrics"].read_text())
    assert summary["conditional_exploratory"] is True
    assert "primary_aggregate" in summary


# ------------------------------------------------- --init-run-group: the REAL record_run contract


def test_init_run_group_real_record_run_contract(tmp_path, config, monkeypatch, capsys):
    sessions = _sessions(n_subjects=3, session_indices=(1, 2, 3, 4))
    cfg, data_10, results_dir = _config_with_tmp_dirs(config, tmp_path)

    import experiments.run_clock_decoupling as rc

    monkeypatch.setattr(rc, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(exp_b, "build_sessions_b", lambda c, band: sessions)
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda c, band, s: {})   # store check not this test's job
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)
    monkeypatch.setattr(rc, "_build_manifest_qc", lambda c, band: _minimal_manifest_qc(data_10, sessions))

    rc.main([
        "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml",
        "--session-specific", "--init-run-group",
    ])

    printed = capsys.readouterr().out.strip().splitlines()[-1]
    run_dir = Path(printed)
    assert run_dir.is_dir()                                   # C19: a DIRECTORY, not the provenance file itself
    prov_path = run_dir / "provenance.json"
    assert prov_path.exists()
    provenance = json.loads(prov_path.read_text())

    assert "config_hash" not in provenance                     # C20: never at the top level
    assert "expected_subjects_by_session" not in provenance
    assert provenance["extra"]["config_hash"] == exp_b.config_fingerprint(cfg)
    for s in (1, 2, 3, 4):
        assert provenance["extra"]["expected_subjects_by_session"][str(s)] == \
            exp_b.eligible_subjects_for_session(sessions, s)
        expected_folds = provenance_mod.fold_manifest(
            nested_loso_splits(exp_b.eligible_subjects_for_session(sessions, s))
        )
        assert provenance["extra"]["folds_by_session"][str(s)] == expected_folds   # C21


def test_init_run_group_two_distinct_roots_hashes_from_77ghz_when_data_dir_passed(tmp_path, config, monkeypatch):
    """(C22) A band="77ghz" run must hash from the 77 GHz root, not the 10 GHz default."""
    sessions = _sessions(n_subjects=2, session_indices=(1, 2, 3, 4))
    cfg, data_10, results_dir = _config_with_tmp_dirs(config, tmp_path, with_77ghz=True)
    data_77 = cfg.paths.data_77ghz_dir

    import experiments.run_clock_decoupling as rc

    monkeypatch.setattr(rc, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(exp_b, "build_sessions_b", lambda c, band: sessions)
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda c, band, s: {})
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)
    # write the dummy raw files under the 77 GHz root ONLY -- if record_run hashed from the
    # 10 GHz root instead, sha256_file would fail closed (FileNotFoundError).
    monkeypatch.setattr(rc, "_build_manifest_qc", lambda c, band: _minimal_manifest_qc(data_77, sessions))

    rc.main([
        "--config", "configs/exp_a_regression_77ghz.yaml", "--config", "configs/exp_b.yaml",
        "--band", "77ghz", "--session-specific", "--init-run-group",
    ])
    # no exception -> record_run's sha256_file successfully read every rel_path from data_77,
    # which only happens if data_dir=require_77ghz_dir(config) was actually passed through.


def test_init_run_group_fails_closed_without_data_dir_when_omitted(tmp_path, config, monkeypatch):
    """(C22) The negative half: if `data_dir` were NOT passed for a 77 GHz run, hashing
    against the (wrong) 10 GHz default must fail closed -- proven directly against
    `provenance.record_run` itself, the exact call `--init-run-group` makes."""
    from dehyd.provenance import record_run

    sessions = _sessions(n_subjects=1, session_indices=(1,))
    cfg, data_10, results_dir = _config_with_tmp_dirs(config, tmp_path, with_77ghz=True)
    data_77 = cfg.paths.data_77ghz_dir
    manifest_qc = _minimal_manifest_qc(data_77, sessions)   # files live ONLY under data_77

    with pytest.raises(FileNotFoundError):
        record_run(cfg, manifest_qc, folds=None)   # no data_dir=... -> defaults to data_10ghz_dir


def test_init_run_group_failure_exits_nonzero(tmp_path, config, monkeypatch):
    """(C23) The Python-testable subset: if --init-run-group itself raises (here, because no
    matching feature store exists), the process exits nonzero, proving
    submit_exp_b_variant.sh's `sbatch --wait` + `set -e` chain would never reach array
    submission."""
    sessions = _sessions(n_subjects=2, session_indices=(1, 2))
    cfg, data_10, results_dir = _config_with_tmp_dirs(config, tmp_path)
    # deliberately do NOT write a matching feature store under results_dir.

    import experiments.run_clock_decoupling as rc

    monkeypatch.setattr(rc, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(exp_b, "build_sessions_b", lambda c, band: sessions)
    monkeypatch.setattr(rc, "_build_manifest_qc", lambda c, band: _minimal_manifest_qc(data_10, sessions))
    # expected_fingerprints and validate_store run FOR REAL here (not mocked) -- the missing
    # store must fail closed before record_run is ever called.

    with pytest.raises(Exception):
        rc.main([
            "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml",
            "--session-specific", "--init-run-group",
        ])


# ------------------------------------------------------ --session-specific --session (wiring)


def test_session_flag_writes_shard_via_shared_helpers(tmp_path, config, monkeypatch):
    sessions = _sessions(n_subjects=4)
    run_dir = tmp_path / "20260728T000000000000Z_abcd1234"
    run_dir.mkdir()

    import experiments.run_clock_decoupling as rc

    monkeypatch.setattr(exp_b, "build_sessions_b", lambda c, band: sessions)
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda c, band, s: {})
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)
    monkeypatch.setattr(
        exp_b, "run_exp_b_one_session",
        lambda *a, **k: _fake_results_b([s for s in sessions if s["session_idx"] == 1]),
    )
    monkeypatch.setattr(exp_b, "_assert_mechanism_ok_b", lambda *a, **k: None)

    rc.main([
        "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml",
        "--session-specific", "--session", "1", "--run-dir", str(run_dir),
    ])

    shard_path = run_dir / "session_specific_10ghz_s1.json"
    assert shard_path.exists()
    shard = json.loads(shard_path.read_text())
    assert shard["band"] == "10ghz" and shard["session"] == 1
    assert shard["run_group_id"] == run_dir.name
    assert "selection_frequency" in shard["summary"]
    assert "wilcoxon_p" not in shard["summary"]

    for name in ("predictions", "selection_table", "dropped_folds"):
        assert (run_dir / f"session_specific_{name}_10ghz_s1.csv").exists()


def test_session_flag_does_not_call_record_run(tmp_path, config, monkeypatch):
    """(C14) The per-session task never calls record_run itself -- only --init-run-group does."""
    sessions = _sessions(n_subjects=2)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    import experiments.run_clock_decoupling as rc

    monkeypatch.setattr(exp_b, "build_sessions_b", lambda c, band: sessions)
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda c, band, s: {})
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)
    monkeypatch.setattr(
        exp_b, "run_exp_b_one_session",
        lambda *a, **k: _fake_results_b([s for s in sessions if s["session_idx"] == 1]),
    )
    monkeypatch.setattr(exp_b, "_assert_mechanism_ok_b", lambda *a, **k: None)

    def fail_if_called(*a, **k):
        raise AssertionError("record_run must not be called on the --session path")

    monkeypatch.setattr(rc, "record_run", fail_if_called)

    rc.main([
        "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml",
        "--session-specific", "--session", "1", "--run-dir", str(run_dir),
    ])


def test_session_flag_unexpected_exception_propagates_to_nonzero_exit(tmp_path, config, monkeypatch):
    sessions = _sessions(n_subjects=4)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    import experiments.run_clock_decoupling as rc

    monkeypatch.setattr(exp_b, "build_sessions_b", lambda c, band: sessions)
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda c, band, s: {})
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("boom, simulating an unexpected failure inside the search")

    monkeypatch.setattr(exp_b, "run_exp_b_one_session", boom)

    with pytest.raises(RuntimeError, match="boom"):
        rc.main([
            "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml",
            "--session-specific", "--session", "1", "--run-dir", str(run_dir),
        ])


# --------------------------------------------------- --session-specific --merge-sessions (wiring)


def test_merge_sessions_writes_combined_report(tmp_path, monkeypatch):
    import experiments.run_clock_decoupling as rc

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake_merged = {"conditional_exploratory": True, "completed_sessions": [1, 2], "1": {}, "2": {}}
    monkeypatch.setattr(exp_b, "merge_session_specific_reports", lambda band, rd: fake_merged)

    rc.main([
        "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_b.yaml",
        "--session-specific", "--merge-sessions", "--run-dir", str(run_dir),
    ])

    out_path = run_dir / "session_specific_10ghz.json"
    assert out_path.exists()
    assert json.loads(out_path.read_text()) == fake_merged
