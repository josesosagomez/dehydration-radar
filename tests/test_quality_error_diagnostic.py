"""Scientific and provenance contracts for quality-versus-LOSO-error analysis."""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dehyd.quality.error_diagnostic as diagnostic
from dehyd.quality.error_diagnostic import (
    EXPECTED_PREDICTION_SHA256,
    OBSOLETE_PREDICTION_SHA256,
    QualityErrorDiagnosticError,
    assemble_analysis_table,
    authenticate_exp_a_sources,
    build_session_prediction_table,
    cluster_bootstrap,
    fit_fixed_effect_association,
    load_quality_error_config,
    reconstruct_prediction_bytes,
    validate_canonical_population,
    relabel_cluster_bootstrap_sample,
    sha256_file,
    verify_and_load_quality,
    write_csv_deterministic,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "quality_error_10ghz.yaml"


@pytest.fixture(scope="module")
def source_records():
    reference = json.loads((ROOT / "results/milestone10/reference_exp_a_manifest.json").read_text())
    sources = json.loads((ROOT / "results/milestone10/exp_a_sources.json").read_text())
    return reference, sources


def test_authoritative_source_reconstructs_exact_prediction_bytes(source_records):
    reference, sources = source_records
    band = authenticate_exp_a_sources(reference, sources)
    payload = reconstruct_prediction_bytes(band)
    assert __import__("hashlib").sha256(payload).hexdigest() == EXPECTED_PREDICTION_SHA256
    assert OBSOLETE_PREDICTION_SHA256 != EXPECTED_PREDICTION_SHA256


@pytest.mark.parametrize("record", ["reference", "sources"])
def test_both_source_schema_versions_are_exact(source_records, record):
    reference, sources = copy.deepcopy(source_records)
    if record == "reference":
        reference["schema_version"] = "reference_exp_a_manifest_v2"
    else:
        sources["schema_version"] = "exp_a_sources_v2"
    with pytest.raises(QualityErrorDiagnosticError, match="schema_version"):
        authenticate_exp_a_sources(reference, sources)


@pytest.mark.parametrize("damage", ["missing", "duplicate"])
def test_each_exact_comparison_evidence_class_must_occur_once(source_records, damage):
    reference, sources = copy.deepcopy(source_records)
    comparisons = sources["bands"]["10ghz"]["comparisons"]
    if damage == "missing":
        comparisons.pop(0)
    else:
        comparisons[-1] = copy.deepcopy(comparisons[0])
    with pytest.raises(QualityErrorDiagnosticError, match="each exact evidence class once"):
        authenticate_exp_a_sources(reference, sources)


@pytest.mark.parametrize(
    "mutation",
    ["top_status", "band_status", "grade", "approval", "comparison", "delta", "bytes", "run", "commit", "config_hash", "prediction_hash"],
)
def test_source_approval_authority_and_comparison_gates_fail_closed(source_records, mutation):
    reference, sources = copy.deepcopy(source_records)
    if mutation == "top_status": sources["status"] = "pending"
    elif mutation == "band_status": sources["bands"]["10ghz"]["status"] = "pending"
    elif mutation == "grade": reference["reference_grade"] = "informative"
    elif mutation == "approval": sources["bands_approved"] = ["77ghz"]
    elif mutation == "comparison": sources["bands"]["10ghz"]["comparisons"][0]["status"] = "mismatch"
    elif mutation == "delta":
        next(r for r in sources["bands"]["10ghz"]["comparisons"] if r["evidence_class"] == "predictions")["max_abs_pred_delta"] = 1e-12
    elif mutation == "bytes":
        next(r for r in sources["bands"]["10ghz"]["comparisons"] if r["evidence_class"] == "predictions")["byte_identical"] = False
    elif mutation == "run": sources["bands"]["10ghz"]["final_run"]["path"] = "/wrong/run"
    elif mutation == "commit": sources["bands"]["10ghz"]["final_run"]["commit"] = "a" * 40
    elif mutation == "config_hash": sources["bands"]["10ghz"]["final_config_sha256"] = "b" * 64
    elif mutation == "prediction_hash":
        sources["bands"]["10ghz"]["final_run"]["artifacts"]["predictions"]["sha256"] = OBSOLETE_PREDICTION_SHA256
    with pytest.raises(QualityErrorDiagnosticError):
        authenticate_exp_a_sources(reference, sources)


def test_population_mapping_is_exact_and_seed_rows_are_not_independent(source_records):
    reference, sources = source_records
    band = authenticate_exp_a_sources(reference, sources)
    table = build_session_prediction_table(band)
    assert len(table) == 73
    assert not table.duplicated(["subject", "session_idx"]).any()
    assert set(table["n_seeds"]) <= {1, 5}
    assert len(table) != sum(table["n_seeds"])

    five_seed_fold = next(f for f in band["selected_folds"] if len(f["predictions"]["seeds"]) == 5)
    subject = five_seed_fold["test_subject"]
    row = table.loc[table.subject == subject].iloc[0]
    predictions = five_seed_fold["predictions"]
    per_seed = np.asarray([predictions["y_pred_by_seed"][str(seed)][0] for seed in predictions["seeds"]])
    truth = predictions["y_true"][0]
    assert row.mean_absolute_seed_error_pct_points == pytest.approx(np.mean(np.abs(per_seed - truth)))
    assert row.absolute_ensemble_error_pct_points == pytest.approx(abs(per_seed.mean() - truth))


@pytest.mark.parametrize("damage", ["reorder", "missing", "duplicate", "target"])
def test_reordered_missing_duplicate_or_retruthed_population_is_rejected(source_records, damage):
    reference, sources = copy.deepcopy(source_records)
    band = authenticate_exp_a_sources(reference, sources)
    sessions = band["population"]["sessions"]
    if damage == "reorder":
        same_subject = [i for i, row in enumerate(sessions) if row["subject"] == 1]
        sessions[same_subject[0]], sessions[same_subject[1]] = sessions[same_subject[1]], sessions[same_subject[0]]
    elif damage == "missing": sessions.pop(0)
    elif damage == "duplicate": sessions[1] = copy.deepcopy(sessions[0])
    elif damage == "target": sessions[0]["delta_m_pct"] += 0.01
    with pytest.raises(QualityErrorDiagnosticError):
        build_session_prediction_table(band)


def test_equal_target_subject_10_session_reorder_is_rejected_by_ordered_key_hash(source_records):
    reference, sources = copy.deepcopy(source_records)
    band = authenticate_exp_a_sources(reference, sources)
    sessions = band["population"]["sessions"]
    indices = [
        index for index, row in enumerate(sessions)
        if row["subject"] == 10 and row["session_idx"] in (0, 1)
    ]
    assert len(indices) == 2
    assert sessions[indices[0]]["delta_m_pct"] == sessions[indices[1]]["delta_m_pct"]
    sessions[indices[0]], sessions[indices[1]] = sessions[indices[1]], sessions[indices[0]]
    with pytest.raises(QualityErrorDiagnosticError, match="session_keys_sha256"):
        build_session_prediction_table(band)


@pytest.mark.parametrize(
    "field",
    ["session_keys_sha256", "targets_sha256", "frame_population_sha256"],
)
def test_corrupted_recorded_population_hashes_are_rejected(source_records, field):
    reference, _sources = copy.deepcopy(source_records)
    population = reference["bands"]["10ghz"]["population"]
    population[field] = "0" * 64
    with pytest.raises(QualityErrorDiagnosticError, match=field):
        validate_canonical_population(population)


def test_corrupted_frame_relation_is_rejected_even_if_recorded_hash_is_unchanged(source_records):
    reference, _sources = copy.deepcopy(source_records)
    population = reference["bands"]["10ghz"]["population"]
    population["sessions"][0]["n_frames"] += 1
    with pytest.raises(QualityErrorDiagnosticError, match="frame_population_sha256"):
        validate_canonical_population(population)


def test_real_quality_join_has_exact_73_and_71_populations(source_records):
    config = load_quality_error_config(CONFIG_PATH)
    band = authenticate_exp_a_sources(*source_records)
    predictions = build_session_prediction_table(band)
    cards, repeatability, _ = verify_and_load_quality(config)
    table, flow = assemble_analysis_table(predictions, cards, repeatability)
    assert len(table) == 73
    for tiling in range(3):
        assert table[f"wst_tiling_{tiling}_block_distance_maximum"].notna().sum() == 71
    assert flow["n_sessions"].tolist() == [80, 73, 73, 71]


def test_quality_hash_mutation_is_rejected(tmp_path):
    config = load_quality_error_config(CONFIG_PATH)
    mutated = tmp_path / "session.csv"
    mutated.write_bytes(config.session_quality.read_bytes() + b"\n")
    with pytest.raises(QualityErrorDiagnosticError, match="hash mismatch"):
        verify_and_load_quality(replace(config, session_quality=mutated))


def test_dirty_quality_provenance_is_rejected(tmp_path):
    config = load_quality_error_config(CONFIG_PATH)
    provenance = json.loads(config.quality_provenance.read_text(encoding="utf-8"))
    provenance["git"]["dirty"] = True
    dirty = tmp_path / "provenance.json"
    dirty.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(QualityErrorDiagnosticError, match="clean committed provenance"):
        verify_and_load_quality(replace(config, quality_provenance=dirty))


def test_wrong_wst_cell_or_orientation_cannot_sneak_into_join(source_records):
    config = load_quality_error_config(CONFIG_PATH)
    band = authenticate_exp_a_sources(*source_records)
    predictions = build_session_prediction_table(band)
    cards, repeatability, _ = verify_and_load_quality(config)
    wrong = repeatability.copy()
    wrong.loc[wrong.diagnostic_channel == "mag", "diagnostic_channel"] = "iq"
    with pytest.raises(QualityErrorDiagnosticError, match="cell"):
        assemble_analysis_table(predictions, cards, wrong)

    table, _ = assemble_analysis_table(predictions, cards, repeatability)
    assert np.allclose(table.one_minus_pass_fraction, 1 - table.pass_fraction)
    assert np.allclose(table.negative_in_band_ratio_p10_margin, -table.in_band_ratio_p10_margin)


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    for subject in range(1, 17):
        for session in range(5):
            within = ((subject * 7 + session * 3) % 11) / 10
            metric = 0.2 * subject + 0.4 * session + within
            rows.append({"subject": subject, "session_idx": session, "metric": metric})
    frame = pd.DataFrame(rows)
    mean = frame.metric.mean()
    sd = frame.metric.std(ddof=1)
    frame["mean_absolute_seed_error_pct_points"] = (
        1.0 + 0.03 * frame.subject + 0.05 * frame.session_idx + 2.0 * ((frame.metric - mean) / sd)
    )
    return frame


def test_fixed_effect_slope_and_partial_correlation_known_synthetic():
    fit = fit_fixed_effect_association(_synthetic_frame(), "metric")
    assert fit["coefficient_pct_points_per_sd_worse"] == pytest.approx(2.0, abs=1e-10)
    assert fit["partial_correlation"] == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("damage", ["nan", "inf", "zero_sd", "rank"])
def test_nonfinite_zero_sd_and_rank_deficiency_fail(damage):
    frame = _synthetic_frame()
    if damage == "nan": frame.loc[0, "metric"] = np.nan
    elif damage == "inf": frame.loc[0, "metric"] = np.inf
    elif damage == "zero_sd": frame["metric"] = 1.0
    elif damage == "rank": frame["metric"] = frame["session_idx"].astype(float)
    with pytest.raises(QualityErrorDiagnosticError):
        fit_fixed_effect_association(frame, "metric")


def test_duplicate_bootstrap_clusters_receive_unique_labels():
    frame = _synthetic_frame()
    sampled = np.asarray([1, 1, 2, 2] + list(range(3, 15)))
    boot = relabel_cluster_bootstrap_sample(frame, sampled)
    subject_one = boot.loc[boot.original_subject == 1]
    assert subject_one.bootstrap_cluster_id.nunique() == 2
    assert boot.bootstrap_cluster_id.nunique() == 16


def test_prescribed_duplicate_draw_uses_unique_clusters_in_production_bootstrap_path():
    frame = _synthetic_frame()
    original_mean = 3.0
    original_sd = 1.1062537992497157
    sampled = np.asarray([1, 1, *range(3, 17)])
    manual = relabel_cluster_bootstrap_sample(frame, sampled)
    manual_fit = fit_fixed_effect_association(
        manual,
        "metric",
        metric_mean=original_mean,
        metric_sd=original_sd,
        subject_column="bootstrap_cluster_id",
    )
    result = cluster_bootstrap(
        frame,
        "metric",
        metric_mean=original_mean,
        metric_sd=original_sd,
        attempted_draws=1,
        seed=999,  # ignored only for this explicit deterministic test draw
        maximum_invalid_fraction=0.05,
        strict=True,
        prescribed_draws=[sampled],
    )
    expected = manual_fit["coefficient_pct_points_per_sd_worse"]
    assert expected == pytest.approx(2.0, abs=1e-12)
    assert result["bootstrap_valid"] == 1
    assert result["bootstrap_ci95_low"] == pytest.approx(expected, abs=1e-12)
    assert result["bootstrap_ci95_high"] == pytest.approx(expected, abs=1e-12)


def test_more_than_five_percent_invalid_bootstrap_fails_strictly():
    rows = []
    for subject in range(1, 17):
        for session in range(2):
            metric = float(session) if subject == 1 else 0.0
            rows.append({
                "subject": subject, "session_idx": session, "metric": metric,
                "mean_absolute_seed_error_pct_points": metric + 0.01 * subject,
            })
    frame = pd.DataFrame(rows)
    with pytest.raises(QualityErrorDiagnosticError, match="invalid fraction"):
        cluster_bootstrap(
            frame, "metric", metric_mean=frame.metric.mean(), metric_sd=frame.metric.std(ddof=1),
            attempted_draws=200, seed=7, maximum_invalid_fraction=0.05, strict=True,
        )


def test_loso_influence_refits_after_each_original_subject():
    frame = _synthetic_frame()
    full = fit_fixed_effect_association(frame, "metric")
    coefficients = []
    for subject in range(1, 17):
        reduced = frame.loc[frame.subject != subject]
        fit = fit_fixed_effect_association(
            reduced, "metric", metric_mean=full["metric_mean_original"],
            metric_sd=full["metric_sd_original_ddof1"],
        )
        coefficients.append(fit["coefficient_pct_points_per_sd_worse"])
    assert len(coefficients) == 16
    assert np.isfinite(coefficients).all()


def test_deterministic_csv_bytes(tmp_path):
    frame = _synthetic_frame().iloc[:4]
    first = write_csv_deterministic(frame, tmp_path / "a.csv")
    second = write_csv_deterministic(frame, tmp_path / "b.csv")
    assert first.read_bytes() == second.read_bytes()


def test_diagnostic_path_does_not_import_training_preprocessing_or_ground_truth():
    source = inspect.getsource(diagnostic).lower()
    forbidden = ("dehyd.models", "dehyd.preprocess", "data.ground_truth", "run_regression", "load_ground_truth")
    assert not any(name in source for name in forbidden)
