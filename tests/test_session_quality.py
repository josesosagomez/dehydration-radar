"""Pure contracts for the descriptive 10 GHz session-quality audit."""

from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dehyd.quality.config import load_quality_audit_config
from dehyd.quality.radar_inventory_10ghz import assert_target_free_columns
from dehyd.quality.session_audit import (
    SessionAuditError,
    _concentration,
    _dimensionless_views,
    _energy_retention,
    assign_stored_index_blocks,
    build_diagnostic_scattering,
    build_recorded_equal_mass_table,
    build_subject_relative_table,
    compare_adjacent_session_geometry,
    plot_recorded_equal_mass,
    recorded_equal_mass_pairs,
    select_recorded_equal_mass_rows,
    summarize_repeatability_vectors,
)


@pytest.fixture(scope="module")
def audit_config():
    return load_quality_audit_config(
        Path(__file__).resolve().parents[1] / "configs" / "quality_audit_10ghz.yaml"
    ).radar


def _geometry(vectors, audit_config, *, n_paths=2, normalized=None, active=None):
    active = np.ones(n_paths, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    return summarize_repeatability_vectors(
        np.asarray(vectors, dtype=np.float64),
        np.arange(100, dtype=np.int64),
        audit_config,
        normalized_path_blocks=normalized,
        active_paths=active,
        n_near_zero_path_blocks=0,
        n_inactive_paths=int((~active).sum()),
        epsilon_path=1e-12,
    )


def test_block_identity_is_by_stored_index_not_row_position(audit_config):
    indices = np.arange(100, dtype=np.int64)
    blocks = assign_stored_index_blocks(
        indices,
        expected_frames=100,
        frames_per_block=20,
        n_blocks=5,
    )
    assert np.array_equal(blocks, np.repeat(np.arange(5), 20))

    permutation = np.random.default_rng(7).permutation(100)
    permuted_blocks = assign_stored_index_blocks(
        permutation, expected_frames=100, frames_per_block=20, n_blocks=5
    )
    assert np.array_equal(permuted_blocks, permutation // 20)


def test_block_declaration_fails_closed():
    with pytest.raises(SessionAuditError, match="exactly cover"):
        assign_stored_index_blocks(np.arange(90), expected_frames=100, frames_per_block=20, n_blocks=4)
    with pytest.raises(SessionAuditError, match="outside"):
        assign_stored_index_blocks(np.array([100]), expected_frames=100, frames_per_block=20, n_blocks=5)


def test_five_identical_blocks_have_zero_separation_and_influence(audit_config):
    vectors = np.tile(np.arange(1.0, 8.0), (100, 1))
    row, _ = _geometry(vectors, audit_config)
    assert row["separation_to_wobble_maximum"] == 0.0
    assert row["block_to_session_distance_maximum"] == 0.0
    assert row["max_leave_one_block_influence"] == 0.0
    assert row["pair_exactly_identical"] == "[true,true,true,true,true,true,true,true,true,true]"


@pytest.mark.parametrize("perturbed_block", range(5))
def test_each_isolated_bad_block_changes_a_worst_block_metric(audit_config, perturbed_block):
    vectors = np.tile(np.arange(1.0, 8.0), (100, 1))
    vectors[perturbed_block * 20 : (perturbed_block + 1) * 20, 2] += 3.0
    row, _ = _geometry(vectors, audit_config)
    assert row["block_to_session_distance_maximum"] > 0.0
    assert row["cosine_similarity_minimum"] < 1.0


def test_dimensionless_views_are_unit_invariant_and_path_sensitive():
    rng = np.random.default_rng(11)
    blocks = rng.normal(size=(20, 3, 6))
    views, common = _dimensionless_views(blocks, 1e-12)
    scaled_views, scaled_common = _dimensionless_views(37.0 * blocks, 1e-12)
    for view in ("within_path_shape", "path_energy_composition"):
        assert np.allclose(views[view][0], scaled_views[view][0], rtol=1e-14, atol=1e-14)
    assert common["n_inactive_paths"] == scaled_common["n_inactive_paths"]

    one_path_scaled = blocks.copy()
    one_path_scaled[:, 1, :] *= 5.0
    changed, _ = _dimensionless_views(one_path_scaled, 1e-12)
    assert np.allclose(
        views["within_path_shape"][0], changed["within_path_shape"][0], rtol=1e-14, atol=1e-14
    )
    assert not np.allclose(
        views["path_energy_composition"][0], changed["path_energy_composition"][0]
    )


def test_near_zero_counts_distinguish_frame_path_blocks_from_inactive_paths():
    blocks = np.ones((4, 3, 2), dtype=np.float64)
    blocks[:, 2] = 0.0  # one path inactive in every frame
    blocks[0, 1] = 0.0  # one additional frame-path block only
    _, common = _dimensionless_views(blocks, 1e-12)
    assert common["n_near_zero_path_blocks"] == 5
    assert common["n_inactive_paths"] == 1
    assert common["active_paths"].tolist() == [True, True, False]


def test_all_zero_paths_are_named_missing_with_complete_counts():
    views, common = _dimensionless_views(np.zeros((4, 3, 2)), 1e-12)
    assert views == {}
    assert common["missing_reason"] == "no_positive_path_norm"
    assert common["n_near_zero_path_blocks"] == 12
    assert common["n_inactive_paths"] == 3
    assert common["active_paths"].tolist() == [False, False, False]


def test_zero_centroid_cosine_is_missing_not_fabricated(audit_config):
    vectors = np.zeros((100, 4), dtype=np.float64)
    row, _ = _geometry(vectors, audit_config)
    assert np.isnan(row["cosine_similarity_median"])
    assert "zero_centroid_norm" in row["missing_reason"]


def test_preprocessing_formulas_reconcile_with_existing_authorities():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from run_preprocess import concentration, energy_retention

    rng = np.random.default_rng(13)
    raw = rng.normal(size=(10, 4)) + 1j * rng.normal(size=(10, 4))
    gated = 0.7 * raw
    assert _energy_retention(raw, gated) == energy_retention(raw, gated)

    class Detection:
        power = np.array([1.0, 2.0, 4.0, 3.0])
        roi_bins = np.array([1, 2])
        peak_bin = 2

    assert _concentration(Detection()) == concentration(Detection())


def test_subject_relative_dense_ranks_never_mix_wst_cells():
    cards = pd.DataFrame(
        [
            {"subject": 1, "session_idx": i, "session_name": str(i), "rel_path": str(i),
             "audit_status": "REPEATABILITY_ANALYSABLE", "repeatability_missing_reason": "",
             "pass_fraction": value}
            for i, value in enumerate([0.8, 0.8, 1.0])
        ]
    )
    repeatability = pd.DataFrame(
        [
            {"subject": 1, "session_idx": i, "session_name": str(i), "diagnostic_channel": channel,
             "tiling_idx": 0, "scattering_order": 1, "view": "within_path_shape",
             "missing_reason": "", "cosine_similarity_median": value,
             "cosine_similarity_minimum": value, "separation_to_wobble_median": 1 - value,
             "separation_to_wobble_maximum": 1 - value,
             "block_to_session_distance_maximum": 1 - value,
             "max_leave_one_block_influence": 1 - value,
             "n_near_zero_path_blocks": 0, "n_inactive_paths": 0}
            for channel, values in (("mag", [0.8, 0.9, 1.0]), ("iq", [1.0, 0.9, 0.8]))
            for i, value in enumerate(values)
        ]
    )
    table = build_subject_relative_table(cards, repeatability)
    core = table[(table["metric"] == "pass_fraction") & (table["diagnostic_channel"] == "")]
    assert core["within_subject_dense_rank"].tolist() == [2.0, 2.0, 1.0]
    mag = table[(table["metric"] == "cosine_similarity_median") & (table["diagnostic_channel"] == "mag")]
    iq = table[(table["metric"] == "cosine_similarity_median") & (table["diagnostic_channel"] == "iq")]
    assert mag["within_subject_dense_rank"].tolist() == [3.0, 2.0, 1.0]
    assert iq["within_subject_dense_rank"].tolist() == [1.0, 2.0, 3.0]


def test_recorded_equal_mass_identification_is_adjacent_and_exact():
    masses = pd.DataFrame(
        {
            "subject": [10] * 5,
            "session_idx": range(5),
            "session_name": ["8am", "10am", "12pm", "2pm", "4pm"],
            "mass_kg": [88.2, 88.2, 87.7, 87.7 + 1e-12, 87.0],
        }
    )
    pairs = recorded_equal_mass_pairs(masses)
    assert [(p["session_a_idx"], p["session_b_idx"]) for p in pairs] == [(0, 1)]


def test_equal_mass_shape_distance_uses_only_common_session_active_paths(audit_config):
    n_frames, n_paths, n_statistics = 100, 3, 2
    normalized_a = np.zeros((n_frames, n_paths, n_statistics), dtype=np.float64)
    normalized_b = np.zeros_like(normalized_a)
    normalized_a[:, 0] = [1.0, 0.0]
    normalized_a[:, 1] = [0.0, 1.0]
    normalized_b[:, 1] = [1.0, 0.0]
    normalized_b[:, 2] = [0.0, 1.0]
    vectors_a = normalized_a.reshape(n_frames, -1) / np.sqrt(n_paths)
    vectors_b = normalized_b.reshape(n_frames, -1) / np.sqrt(n_paths)
    _, geometry_a = _geometry(
        vectors_a,
        audit_config,
        n_paths=n_paths,
        normalized=normalized_a,
        active=[True, True, False],
    )
    _, geometry_b = _geometry(
        vectors_b,
        audit_config,
        n_paths=n_paths,
        normalized=normalized_b,
        active=[False, True, True],
    )
    masses = pd.DataFrame(
        {
            "subject": [10, 10],
            "session_idx": [0, 1],
            "session_name": ["8am", "10am"],
            "mass_kg": [88.2, 88.2],
        }
    )
    geometry = {
        (10, 0): {("mag", 0, 1, "within_path_shape"): geometry_a},
        (10, 1): {("mag", 0, 1, "within_path_shape"): geometry_b},
    }
    table = build_recorded_equal_mass_table(masses, geometry, audit_config)
    row = table[
        (table["diagnostic_channel"] == "mag")
        & (table["tiling_idx"] == 0)
        & (table["scattering_order"] == 1)
        & (table["view"] == "within_path_shape")
    ].iloc[0]
    assert row["common_path_count"] == 1
    assert row["common_vector_dimension"] == n_statistics
    assert row["n_inactive_paths_a"] == 1
    assert row["n_inactive_paths_b"] == 1
    assert row["epsilon_distance_a"] == 0.0
    assert row["epsilon_distance_b"] == 0.0


def test_mass_mutation_changes_only_separate_table_and_empty_table_is_writable(
    tmp_path, audit_config
):
    first_card = {"subject": 10, "session_idx": 0, "session_name": "8am"}
    second_card = {"subject": 10, "session_idx": 1, "session_name": "10am"}
    adjacent = pd.DataFrame(
        compare_adjacent_session_geometry(first_card, second_card, {}, {}, audit_config)
    )
    adjacent_before = adjacent.copy(deep=True)
    adjacent_csv_before = adjacent.to_csv(index=False).encode("utf-8")
    equal_masses = pd.DataFrame(
        {
            "subject": [10, 10],
            "session_idx": [0, 1],
            "session_name": ["8am", "10am"],
            "mass_kg": [88.2, 88.2],
        }
    )
    unequal_masses = equal_masses.copy()
    unequal_masses.loc[1, "mass_kg"] = 88.1
    equal_table = select_recorded_equal_mass_rows(equal_masses, adjacent)
    table = select_recorded_equal_mass_rows(unequal_masses, adjacent)
    assert len(equal_table) == 36
    assert table.empty
    assert adjacent.equals(adjacent_before)
    assert adjacent.to_csv(index=False).encode("utf-8") == adjacent_csv_before
    assert not any("mass" in column for column in adjacent.columns)

    config = dataclasses.replace(
        audit_config,
        output_results_dir=tmp_path / "results",
        output_figures_dir=tmp_path / "figures",
    )
    output = plot_recorded_equal_mass(table, config.output_figures_dir / "empty.png", config)
    assert output.is_file()


def test_target_free_modules_do_not_import_ground_truth_or_manifest():
    import dehyd.quality.radar_inventory_10ghz as inventory
    import dehyd.quality.session_audit as audit

    source = inspect.getsource(inventory) + inspect.getsource(audit)
    assert "data.ground_truth" not in source
    assert "data.manifest" not in source
    assert_target_free_columns(["subject", "session_idx", "raw_rms", "peak_range_m"])
    with pytest.raises(Exception, match="target-like"):
        assert_target_free_columns(["delta_m_pct"])


@pytest.mark.realdata
def test_one_real_session_wst_smoke(real_data_paths, audit_config):
    from dehyd.data.loader_10ghz import load_10ghz_file
    from dehyd.qc.screens import run_qc_cube
    from dehyd.quality.session_audit import compute_session_wst_repeatability

    cube = load_10ghz_file(real_data_paths["data_10ghz_dir"] / "subject_10_12pm.mat")
    verdicts = run_qc_cube(cube, audit_config.qc, audit_config.preprocess)
    passing = [i for i, verdict in enumerate(verdicts) if verdict.passed]
    card = {
        "subject": 10, "session_idx": 2, "session_name": "12pm",
        "rel_path": "subject_10_12pm.mat", "audit_status": "REPEATABILITY_ANALYSABLE",
        "repeatability_missing_reason": "",
    }
    rows, geometry = compute_session_wst_repeatability(
        cube, passing, card, audit_config, build_diagnostic_scattering(audit_config)
    )
    assert len(rows) == 36
    assert len(geometry) == 30  # order-0 composition is explicitly not applicable
    assert all(row["missing_reason"] == "not_applicable_single_path" for row in rows if row["view"] == "path_energy_composition" and row["scattering_order"] == 0)
