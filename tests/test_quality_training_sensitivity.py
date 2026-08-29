from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.run_quality_training_sensitivity import _build_sensitivity_provenance
from dehyd.eval.harness import Candidate
from dehyd.eval.quality_training_sensitivity import (
    EXPECTED_NEGATIVE_KEYS,
    TREATMENTS,
    _selection_metrics_from_rows,
    ArrayFeatureSource,
    QualityTrainingError,
    align_sessions,
    assert_identical_test_keys,
    authenticate_reference,
    canonical_keys_from_reference,
    fit_selected_candidate,
    load_quality_margin,
    load_quality_training_config,
    make_subject_overlap_splits,
    publish_atomically,
    require_full_model_seeds,
    subject_overlap_census,
    synthetic_mechanism_smoke,
    validate_quality_table,
    verify_loso_regression_baseline_replay,
    write_protocol_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "quality_training_sensitivity_10ghz.yaml"


def _synthetic_problem():
    subjects = np.repeat(np.arange(1, 7), 5)
    classes = np.tile(np.arange(5), 6)
    keys = list(zip(subjects.tolist(), classes.tolist(), strict=True))
    quality = np.full(len(keys), 0.2)
    quality[0] = -0.5
    values = np.column_stack([np.arange(len(keys), dtype=float), classes.astype(float)])
    targets = -0.2 * classes.astype(float)
    source = ArrayFeatureSource(values, quality, keys)
    candidate = Candidate(
        "ridge", "ridge", (("alpha", 1.0),), (0, "A", "mag", 0, "tuned"), None
    )
    ordinal = Candidate(
        "ordinal", "ord_a_ridge", (("alpha", 1.0),),
        (0, "A", "mag", 0, "tuned"), None,
    )
    return subjects, classes, keys, quality, values, targets, source, candidate, ordinal


def test_strict_config_and_forbidden_error_source(tmp_path):
    config = load_quality_training_config(CONFIG)
    assert config.n_outer_splits == 5
    assert config.n_inner_splits == 4
    assert config.treatments == TREATMENTS

    raw = CONFIG.read_text(encoding="utf-8").replace(
        "../results/quality_10ghz/session_quality_all_80.csv",
        "../results/quality_error_10ghz/session_error.csv",
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text(raw, encoding="utf-8")
    with pytest.raises(QualityTrainingError, match="forbidden outcome/error"):
        load_quality_training_config(bad)


def test_config_pins_split_seed_output_root_and_model_seeds(tmp_path):
    raw = CONFIG.read_text(encoding="utf-8")
    bad_seed = tmp_path / "bad_seed.yaml"
    bad_seed.write_text(raw.replace("seed: 20260829", "seed: 7"), encoding="utf-8")
    with pytest.raises(QualityTrainingError, match="split.seed must be exactly"):
        load_quality_training_config(bad_seed)

    bad_output = tmp_path / "bad_output.yaml"
    bad_output.write_text(
        raw.replace(
            "../results/quality_training_sensitivity_10ghz",
            str((tmp_path / "wrong-results").resolve()),
        ),
        encoding="utf-8",
    )
    with pytest.raises(QualityTrainingError, match="must resolve exactly"):
        load_quality_training_config(bad_output)

    assert require_full_model_seeds([1, 2, 3, 4, 5]) == (1, 2, 3, 4, 5)
    with pytest.raises(QualityTrainingError, match="seed_set must be exactly"):
        require_full_model_seeds([1])


def test_authenticated_quality_join_is_exact_73_and_five_negative():
    config = load_quality_training_config(CONFIG)
    reference = authenticate_reference(config)
    keys = canonical_keys_from_reference(reference)
    quality = load_quality_margin(config, keys)
    negative = frozenset(
        (int(row.subject), int(row.session_idx))
        for row in quality.itertuples()
        if row.in_band_ratio_p10_margin < 0.0
    )
    assert len(quality) == 73
    assert negative == EXPECTED_NEGATIVE_KEYS
    assert np.isfinite(quality.in_band_ratio_p10_margin.to_numpy(float)).all()


def test_coordinated_quality_csv_and_provenance_mutation_still_fails(tmp_path):
    config = load_quality_training_config(CONFIG)
    reference = authenticate_reference(config)
    keys = canonical_keys_from_reference(reference)
    mutated_csv = tmp_path / "session_quality.csv"
    text = config.session_quality.read_text(encoding="utf-8")
    mutated_csv.write_text(text.replace("0.13334285631580112", "0.13334285631580113", 1), encoding="utf-8")
    mutated_hash = hashlib.sha256(mutated_csv.read_bytes()).hexdigest()
    mutated_provenance = tmp_path / "provenance.json"
    provenance = json.loads(config.quality_provenance.read_text(encoding="utf-8"))
    provenance["outputs"]["session_quality"]["sha256"] = mutated_hash
    mutated_provenance.write_text(json.dumps(provenance), encoding="utf-8")
    mutated_config = replace(
        config, session_quality=mutated_csv, quality_provenance=mutated_provenance
    )
    with pytest.raises(QualityTrainingError, match="does not match its provenance hash"):
        load_quality_margin(mutated_config, keys)


def test_quality_table_rejects_status_eligibility_and_margin_formula_mutations():
    frame = pd.read_csv(ROOT / "results" / "quality_10ghz" / "session_quality_all_80.csv")
    assert len(validate_quality_table(frame)) == 73

    wrong_status = frame.copy()
    wrong_status.loc[0, "audit_status"] = "REPEATABILITY_ANALYSABLE"
    with pytest.raises(QualityTrainingError, match="audit-status census changed"):
        validate_quality_table(wrong_status)

    wrong_eligibility = frame.copy()
    wrong_eligibility.loc[0, "eligible_existing_qc"] = True
    with pytest.raises(QualityTrainingError, match="eligibility flags disagree"):
        validate_quality_table(wrong_eligibility)

    wrong_margin = frame.copy()
    wrong_margin.loc[0, "in_band_ratio_p10_margin"] += 1e-5
    with pytest.raises(QualityTrainingError, match="not exactly"):
        validate_quality_table(wrong_margin)


def test_current_session_relation_authenticates_name_and_target():
    reference = {
        "population": {"sessions": [
            {"subject": 1, "session_idx": 0, "session_name": "8am", "delta_m_pct": 0.0},
            {"subject": 1, "session_idx": 1, "session_name": "10am", "delta_m_pct": -0.2},
        ]}
    }
    sessions = [
        {"subject": 1, "session_idx": 0, "session_name": "8am", "delta_m_pct": 0.0},
        {"subject": 1, "session_idx": 1, "session_name": "10am", "delta_m_pct": -0.2},
    ]
    assert align_sessions(sessions, reference) == sessions
    changed_name = copy.deepcopy(sessions)
    changed_name[1]["session_name"] = "noon"
    with pytest.raises(QualityTrainingError, match="session name changed"):
        align_sessions(changed_name, reference)
    changed_target = copy.deepcopy(sessions)
    changed_target[1]["delta_m_pct"] = -0.2000000001
    with pytest.raises(QualityTrainingError, match="target changed"):
        align_sessions(changed_target, reference)
    with pytest.raises(QualityTrainingError, match="ordered canonical keys"):
        align_sessions(list(reversed(sessions)), reference)


def test_session_folds_are_deterministic_disjoint_and_cover_every_row():
    _, classes, *_ = _synthetic_problem()
    first = make_subject_overlap_splits(classes, seed=20260829)
    second = make_subject_overlap_splits(classes, seed=20260829)
    assert first == second
    coverage = np.zeros(len(classes), dtype=int)
    for split in first:
        assert set(split.train_indices).isdisjoint(split.test_indices)
        coverage[list(split.test_indices)] += 1
    assert np.array_equal(coverage, np.ones(len(classes), dtype=int))


def test_subject_overlap_is_explicit_and_nonzero():
    subjects, classes, *_ = _synthetic_problem()
    for split in make_subject_overlap_splits(classes, seed=20260829):
        census = subject_overlap_census(split, subjects)
        assert census["n_overlapping_subjects"] > 0
        assert census["overlapping_subjects"]


def test_held_out_mutations_do_not_change_any_fitted_quantity():
    subjects, classes, keys, quality, values, targets, source, candidate, _ = _synthetic_problem()
    train = np.flatnonzero(subjects != 6)
    test = np.flatnonzero(subjects == 6)
    _, audit_before = fit_selected_candidate(
        candidate=candidate, source=source, keys=keys, quality_margin=quality,
        train_indices=train, test_indices=test, regression_target=targets, classes=classes,
        task="regression", treatment="baseline", seeds=(1,), authorize=None,
    )
    changed_values = values.copy()
    changed_values[test] += 10000.0
    changed_targets = targets.copy()
    changed_targets[test] += 10000.0
    changed_source = ArrayFeatureSource(changed_values, quality, keys)
    _, audit_after = fit_selected_candidate(
        candidate=candidate, source=changed_source, keys=keys, quality_margin=quality,
        train_indices=train, test_indices=test, regression_target=changed_targets, classes=classes,
        task="regression", treatment="baseline", seeds=(1,), authorize=None,
    )
    assert audit_before == audit_after


def test_fit_rejects_duplicate_alias_keys_across_different_indices():
    subjects, classes, keys, quality, values, targets, _, candidate, _ = _synthetic_problem()
    duplicate_keys = keys.copy()
    duplicate_keys[1] = duplicate_keys[0]
    source = ArrayFeatureSource(values, quality, duplicate_keys)
    with pytest.raises(QualityTrainingError, match="canonical row keys contain duplicates"):
        fit_selected_candidate(
            candidate=candidate, source=source, keys=duplicate_keys, quality_margin=quality,
            train_indices=[0, 2, 3, 4, 5], test_indices=[1], regression_target=targets,
            classes=classes, task="regression", treatment="baseline", seeds=(1,), authorize=None,
        )


def test_filter_changes_fitted_rows_and_fold_tuned_epsilon():
    subjects, classes, keys, quality, _, targets, source, candidate, _ = _synthetic_problem()
    train = np.flatnonzero(subjects != 6)
    test = np.flatnonzero(subjects == 6)
    _, baseline = fit_selected_candidate(
        candidate=candidate, source=source, keys=keys, quality_margin=quality,
        train_indices=train, test_indices=test, regression_target=targets, classes=classes,
        task="regression", treatment="baseline", seeds=(1,), authorize=None,
    )
    _, filtered = fit_selected_candidate(
        candidate=candidate, source=source, keys=keys, quality_margin=quality,
        train_indices=train, test_indices=test, regression_target=targets, classes=classes,
        task="regression", treatment="filter_negative_margin", seeds=(1,), authorize=None,
    )
    assert filtered[0]["n_fitted_rows"] == baseline[0]["n_fitted_rows"] - 1
    assert [1, 0] not in filtered[0]["fitted_row_keys"]
    assert filtered[0]["tuned_epsilon"] != baseline[0]["tuned_epsilon"]


def test_quality_feature_is_scaled_inside_training_fit():
    subjects, classes, keys, quality, _, targets, source, candidate, _ = _synthetic_problem()
    train = np.flatnonzero(subjects != 6)
    test = np.flatnonzero(subjects == 6)
    _, baseline = fit_selected_candidate(
        candidate=candidate, source=source, keys=keys, quality_margin=quality,
        train_indices=train, test_indices=test, regression_target=targets, classes=classes,
        task="regression", treatment="baseline", seeds=(1,), authorize=None,
    )
    _, appended = fit_selected_candidate(
        candidate=candidate, source=source, keys=keys, quality_margin=quality,
        train_indices=train, test_indices=test, regression_target=targets, classes=classes,
        task="regression", treatment="append_margin_feature", seeds=(1,), authorize=None,
    )
    assert appended[0]["quality_feature_appended"] is True
    assert appended[0]["scaler_mean_sha256"] != baseline[0]["scaler_mean_sha256"]


def test_classification_viability_is_checked_after_filtering():
    subjects, classes, keys, quality, _, targets, source, _, ordinal = _synthetic_problem()
    # Every class-0 training row is marked negative, so the filtered fit must refuse it.
    quality = quality.copy()
    quality[classes == 0] = -0.1
    source = ArrayFeatureSource(source.values, quality, keys)
    train = np.flatnonzero(subjects != 6)
    test = np.flatnonzero(subjects == 6)
    with pytest.raises(QualityTrainingError, match=r"missing classes \[0\]"):
        fit_selected_candidate(
            candidate=ordinal, source=source, keys=keys, quality_margin=quality,
            train_indices=train, test_indices=test, regression_target=targets, classes=classes,
            task="ordinal_classification", treatment="filter_negative_margin", seeds=(1,),
            authorize=None,
        )


def test_all_treatments_have_identical_test_keys_and_mismatch_fails():
    subjects, classes, keys, quality, _, targets, source, candidate, _ = _synthetic_problem()
    split = make_subject_overlap_splits(classes, seed=20260829)[0]
    rows = []
    for treatment in TREATMENTS:
        pred, _ = fit_selected_candidate(
            candidate=candidate, source=source, keys=keys, quality_margin=quality,
            train_indices=split.train_indices, test_indices=split.test_indices,
            regression_target=targets, classes=classes, task="regression",
            treatment=treatment, seeds=(1,), authorize=None,
        )
        for row in pred:
            row.update(protocol="p", task="regression", split_id="f0", arm="a", treatment=treatment)
        rows.extend(pred)
    assert_identical_test_keys(rows)
    broken = copy.deepcopy(rows)
    broken.pop()
    with pytest.raises(QualityTrainingError, match="different held-out keys"):
        assert_identical_test_keys(broken)

    reordered = copy.deepcopy(rows)
    appended = [row for row in reordered if row["treatment"] == "append_margin_feature"]
    others = [row for row in reordered if row["treatment"] != "append_margin_feature"]
    with pytest.raises(QualityTrainingError, match="different held-out keys"):
        assert_identical_test_keys(others + list(reversed(appended)))

    duplicated = copy.deepcopy(rows)
    duplicated.append(copy.deepcopy(duplicated[-1]))
    with pytest.raises(QualityTrainingError, match="duplicate held-out keys"):
        assert_identical_test_keys(duplicated)


def test_ordinal_session_selection_averages_mae_by_seed_but_qwk_uses_first_seed():
    truth = [0, 1, 2, 3, 4]
    rows = []
    for seed, predictions in ((11, truth), (12, [4, 3, 2, 1, 0])):
        for session_idx, (actual, predicted) in enumerate(zip(truth, predictions, strict=True)):
            rows.append({
                "seed": seed, "subject": 1, "session_idx": session_idx,
                "y_true": actual, "y_pred": predicted,
            })
    primary, qwk = _selection_metrics_from_rows(rows, "ordinal_classification")
    assert primary == pytest.approx((0.0 + 2.4) / 2.0)
    assert qwk == pytest.approx(1.0)


def test_authenticated_baseline_replay_gate_fails_on_prediction_or_epsilon_drift():
    reference = {
        "population": {"sessions": [
            {"subject": 1, "session_idx": 0}, {"subject": 1, "session_idx": 1},
        ]},
        "selected_folds": [{
            "test_subject": 1,
            "feature_key": [0, "A", "mag", 0, "tuned"],
            "family": "ridge", "params": {"alpha": 1.0},
            "tuned_epsilon": {"order_1": 0.01, "order_2": 0.001},
            "predictions": {
                "n_sessions": 2, "seeds": [1], "y_true": [0.0, -0.2],
                "y_pred_by_seed": {"1": [-0.01, -0.19]},
            },
        }],
    }
    predictions = [
        {"protocol": "loso", "task": "regression", "arm": "regression",
         "treatment": "baseline", "split_id": "subject_1", "seed": 1,
         "subject": 1, "session_idx": session, "y_true": truth, "y_pred": prediction}
        for session, truth, prediction in ((0, 0.0, -0.01), (1, -0.2, -0.19))
    ]
    audits = [{
        "protocol": "loso", "task": "regression", "arm": "regression",
        "treatment": "baseline", "split_id": "subject_1", "seed": 1,
        "tuned_epsilon": {1: 0.01, 2: 0.001},
    }]
    selections = [{
        "protocol": "loso", "task": "regression", "arm": "regression",
        "split_id": "subject_1", "family": "ridge", "params": {"alpha": 1.0},
        "feature_key": [0, "A", "mag", 0, "tuned"],
    }]
    assert verify_loso_regression_baseline_replay(
        predictions, audits, selections, reference
    )["status"] == "passed"
    changed_predictions = copy.deepcopy(predictions)
    changed_predictions[0]["y_pred"] += 1e-6
    with pytest.raises(QualityTrainingError, match="exceeded tolerance"):
        verify_loso_regression_baseline_replay(
            changed_predictions, audits, selections, reference
        )
    changed_audits = copy.deepcopy(audits)
    changed_audits[0]["tuned_epsilon"][1] += 1e-6
    with pytest.raises(QualityTrainingError, match="exceeded tolerance"):
        verify_loso_regression_baseline_replay(
            predictions, changed_audits, selections, reference
        )


def test_outputs_are_deterministic_and_isolated(tmp_path):
    subjects, classes, keys, quality, _, targets, source, candidate, _ = _synthetic_problem()
    split = make_subject_overlap_splits(classes, seed=20260829)[0]
    rows, audits = [], []
    for treatment in TREATMENTS:
        pred, audit = fit_selected_candidate(
            candidate=candidate, source=source, keys=keys, quality_margin=quality,
            train_indices=split.train_indices, test_indices=split.test_indices,
            regression_target=targets, classes=classes, task="regression", treatment=treatment,
            seeds=(1,), authorize=None,
        )
        for row in pred:
            row.update(protocol="loso", task="regression", split_id="f0", arm="regression", treatment=treatment)
        for row in audit:
            row.update(protocol="loso", task="regression", split_id="f0", arm="regression", treatment=treatment)
        rows.extend(pred)
        audits.extend(audit)
    selection = [{"candidate": "ridge"}]
    splits = [{"split_id": "f0"}]
    first = write_protocol_outputs(tmp_path / "one", "loso", rows, audits, selection, splits)
    second = write_protocol_outputs(tmp_path / "two", "loso", rows, audits, selection, splits)
    assert {name: path.read_bytes() for name, path in first.items()} == {
        name: path.read_bytes() for name, path in second.items()
    }
    assert not (tmp_path / "one" / "subject_overlap_session_cv").exists()


def test_atomic_publish_cleans_late_failure_and_never_overwrites(tmp_path):
    final = tmp_path / "final"

    def fail_late(staging):
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("late failure")

    with pytest.raises(RuntimeError, match="late failure"):
        publish_atomically(final, fail_late)
    assert not final.exists()
    assert list(tmp_path.glob(".final.staging-*")) == []

    final.mkdir()
    sentinel = final / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(QualityTrainingError, match="refusing to overwrite"):
        publish_atomically(final, lambda staging: None)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_atomic_publish_success_moves_complete_staging_tree(tmp_path):
    final = tmp_path / "final"

    def write_success(staging):
        nested = staging / "loso"
        nested.mkdir()
        (nested / "metrics.json").write_text('{"ok": true}\n', encoding="utf-8")
        return "complete"

    assert publish_atomically(final, write_success) == "complete"
    assert (final / "loso" / "metrics.json").read_text(encoding="utf-8") == '{"ok": true}\n'
    assert list(tmp_path.glob(".final.staging-*")) == []


def test_production_provenance_helper_contains_and_serializes_every_required_class():
    config_hashes = [
        {"path": "sensitivity.yaml", "sha256": "a" * 64},
        {"path": "exp_a.yaml", "sha256": "b" * 64},
        {"path": "ibex.yaml", "sha256": "c" * 64},
    ]
    output_hashes = {"loso/metrics.json": "d" * 64, "subject_overlap_session_cv/metrics.json": "e" * 64}
    store_lineage = {"n_fingerprints": 73, "fingerprints_sha256": "f" * 64}
    payload = _build_sensitivity_provenance(
        timestamp_utc="2026-08-30T00:00:00+00:00",
        git={"commit": "1" * 40, "dirty": False, "branch": "test"},
        config_hashes=config_hashes,
        resolved_config_fingerprint="2" * 64,
        resolved_config={"run": {"seed_set": [1, 2, 3, 4, 5]}},
        model_seed_set=(1, 2, 3, 4, 5),
        session_split_seed=20260829,
        replay_gate={"status": "passed", "tolerance": 1e-10},
        source_hashes={"session_quality": "3" * 64},
        store_lineage=store_lineage,
        packages={"numpy": "2.4.6"},
        platform_info={"system": "Linux", "cpu_model": "test cpu"},
        output_hashes=output_hashes,
    )
    assert payload["configs"] == config_hashes
    assert payload["resolved_config_fingerprint"] == "2" * 64
    assert payload["resolved_config"]["run"]["seed_set"] == [1, 2, 3, 4, 5]
    assert payload["seeds"]["model_seed_set"] == [1, 2, 3, 4, 5]
    assert payload["seeds"]["session_split_seed"] == 20260829
    assert payload["store_lineage"] == store_lineage
    assert payload["packages"]["numpy"] == "2.4.6"
    assert payload["platform"]["system"] == "Linux"
    assert payload["output_sha256"] == output_hashes
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert json.loads(serialized) == payload


def test_synthetic_smoke_is_deterministic_and_suppresses_scores():
    first = synthetic_mechanism_smoke()
    assert first == synthetic_mechanism_smoke()
    assert first["performance_values_suppressed"] is True
    assert "mae" not in json.dumps(first).lower()
    align_sessions,
    publish_atomically,
    require_full_model_seeds,
