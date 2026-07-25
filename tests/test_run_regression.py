"""T-M7-entrypoint: the Exp A entrypoint — flag validation, deterministic subset selection,
and the mechanism-only reporting boundary (C9/C14: a smoke run surfaces NO performance value).

The expensive staged search itself is covered by test_exp_a; here `run_exp_a` is stubbed with
fabricated fold results so the reporting boundary is tested fast and in isolation.
"""

import json

import numpy as np
import pytest

from dehyd.config import load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_a
from dehyd.eval.exp_a import ExpAFoldResult, select_subset_subjects
from dehyd.eval.harness import FitRecord, SeedOutcome
from dehyd.features.store import write_session_store


@pytest.fixture(scope="module")
def config():
    return load_config("configs/exp_a_regression.yaml")


def _sessions(n_subjects=4, sessions=2):
    return [
        {"subject": s, "session_idx": i, "session_name": SESSION_NAMES[i], "delta_m_pct": -0.3 * i - 0.05 * s}
        for s in range(1, n_subjects + 1)
        for i in range(sessions)
    ]


def _minimal_store(tmp_path, sessions):
    for s in sessions:
        write_session_store("10ghz", s["subject"], s["session_name"], {"x": np.zeros(1)}, {"n_frames": 1}, tmp_path)


def _fake_results(sessions):
    subjects = sorted({s["subject"] for s in sessions})
    out = []
    for test_subj in subjects:
        train = frozenset(x for x in subjects if x != test_subj)
        rows = [s for s in sessions if s["subject"] == test_subj]
        y = np.array([s["delta_m_pct"] for s in rows])
        pred = y + 0.1
        out.append(ExpAFoldResult(
            test_subject=test_subj,
            selected_feature_key=(0, "A", "mag", 0, "off"),
            selected_family="ridge",
            selected_params={"alpha": 1.0},
            test_predictions=pred,
            test_targets=y,
            seed_outcomes=[SeedOutcome(0, pred, pred, 0.1)],
            baseline_predictions=y + 0.5,
            final_fits=[FitRecord("scaler", "outer_train", train, {"mean_": np.zeros(1)})],
        ))
    return out


def _patch(monkeypatch, sessions):
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda *a, **k: {})
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)
    monkeypatch.setattr(exp_a, "run_exp_a", lambda *a, **k: _fake_results(sessions))


# ----------------------------------------------------------------- flag validation


def test_select_subset_subjects_is_deterministic_lowest_six():
    subs = [12, 3, 8, 1, 5, 9, 4, 7, 2, 16]
    assert select_subset_subjects(subs, k=6) == [1, 2, 3, 4, 5, 7]


def test_full_cohort_refused_without_flag(config, tmp_path):
    import experiments.run_regression as rr  # noqa

    with pytest.raises(SystemExit):
        rr.main(["--config", "configs/exp_a_regression.yaml", "--band", "10ghz"])  # neither flag


def test_subset_and_full_cohort_are_mutually_exclusive():
    import experiments.run_regression as rr

    with pytest.raises(SystemExit):
        rr.main(["--config", "configs/exp_a_regression.yaml", "--subset", "6subjects", "--full-cohort"])


# ------------------------------------------------------ mechanism-only reporting (C9/C14)


def test_smoke_writes_no_performance_value(tmp_path, config, monkeypatch):
    sessions = _sessions()
    _minimal_store(tmp_path, sessions)
    _patch(monkeypatch, sessions)
    run_dir = tmp_path / "run"

    outputs = exp_a.run_and_report(config, "10ghz", sessions, tmp_path, run_dir, mode="smoke", analysis_commit="x")

    assert set(outputs) == {"run_log"}
    files = {p.name for p in run_dir.iterdir()}
    # NO metrics / predictions / scatter / selection table leaves the process.
    assert not any(n.startswith(("metrics_", "predictions_", "scatter_", "selection_table_")) for n in files)
    log = json.loads(outputs["run_log"].read_text())
    assert log["mode"] == "mechanism-only" and "note" in log


def test_full_run_writes_metrics_and_scatter(tmp_path, config, monkeypatch):
    sessions = _sessions()
    _minimal_store(tmp_path, sessions)
    _patch(monkeypatch, sessions)
    run_dir = tmp_path / "run"

    outputs = exp_a.run_and_report(config, "10ghz", sessions, tmp_path, run_dir, mode="full", analysis_commit="x")

    assert "metrics" in outputs and "scatter" in outputs
    assert outputs["metrics"].exists() and outputs["scatter"].exists()
    summary = json.loads(outputs["metrics"].read_text())
    assert summary["conditional_exploratory"] is True
    assert "subject_balanced_mae" in summary and "baseline_session_index_only" in summary
