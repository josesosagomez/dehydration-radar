"""Configuration, output-scope, and CLI-boundary tests for the quality audit."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
import yaml

import experiments.run_session_quality_audit as audit_runner
from dehyd.config import config_to_dict, load_config
from dehyd.quality.config import QualityAuditConfigError, load_quality_audit_config, radar_config_record
import dehyd.quality.session_audit as session_audit
from dehyd.quality.session_audit import (
    SessionAuditError,
    build_radar_provenance,
    capture_clean_git_provenance,
    compare_adjacent_session_geometry,
    ensure_write_within,
    require_fresh_output_roots,
    snapshot_existing_results,
    verify_frozen_results,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT_YAML = ROOT / "configs" / "quality_audit_10ghz.yaml"


def test_audit_config_is_strict_and_base_config_is_byte_neutral(tmp_path):
    base_before = config_to_dict(load_config(ROOT / "configs" / "exp_a_regression.yaml"))
    loaded = load_quality_audit_config(AUDIT_YAML)
    base_after = config_to_dict(load_config(ROOT / "configs" / "exp_a_regression.yaml"))
    assert base_before == base_after
    assert loaded.radar.n_blocks == 5
    assert loaded.radar.min_passing_frames_per_block == 10

    raw = yaml.safe_load(AUDIT_YAML.read_text(encoding="utf-8"))
    raw["surprise"] = 1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(QualityAuditConfigError, match="unknown field"):
        load_quality_audit_config(bad)


def test_radar_config_projection_has_no_workbook_or_target():
    record = radar_config_record(load_quality_audit_config(AUDIT_YAML).radar)
    text = str(record).lower()
    assert "weight" not in text
    assert "workbook" not in text
    assert "delta_m" not in text


@pytest.mark.parametrize(
    ("git_record", "message"),
    [
        ({"commit": None, "dirty": False, "branch": "main"}, "real 40-character"),
        ({"commit": "abc", "dirty": False, "branch": "main"}, "real 40-character"),
        ({"commit": "a" * 40, "dirty": True, "branch": "main"}, "clean source tree"),
        ({"commit": "a" * 40, "dirty": None, "branch": "main"}, "clean source tree"),
    ],
)
def test_git_capture_requires_a_real_commit_and_clean_tree(monkeypatch, git_record, message):
    monkeypatch.setattr(session_audit, "_git_info", lambda: git_record)
    with pytest.raises(SessionAuditError, match=message):
        capture_clean_git_provenance()


def test_run_captures_git_before_loading_config(monkeypatch):
    events = []
    captured = {"commit": "a" * 40, "dirty": False, "branch": "audit"}

    def capture():
        events.append("git")
        return captured

    def stop_during_config(_path):
        assert events == ["git"]
        raise RuntimeError("stop after ordering check")

    monkeypatch.setattr(audit_runner, "capture_clean_git_provenance", capture)
    monkeypatch.setattr(audit_runner, "load_quality_audit_config", stop_during_config)
    with pytest.raises(RuntimeError, match="ordering check"):
        audit_runner.run(AUDIT_YAML)


def test_radar_provenance_uses_the_start_of_run_git_record():
    config = load_quality_audit_config(AUDIT_YAML).radar
    captured = {"commit": "b" * 40, "dirty": False, "branch": "audit"}
    provenance = build_radar_provenance(
        config,
        raw_sessions=[],
        source_artifacts={},
        radar_artifacts={},
        census={},
        git_record=captured,
    )
    assert provenance["git"] == captured


def test_output_guard_allows_only_the_two_audit_roots(tmp_path):
    loaded = load_quality_audit_config(AUDIT_YAML)
    radar = dataclasses.replace(
        loaded.radar,
        output_results_dir=tmp_path / "results" / "quality_10ghz",
        output_figures_dir=tmp_path / "figures" / "quality_10ghz",
    )
    allowed = ensure_write_within(
        radar.output_results_dir / "table.csv",
        (radar.output_results_dir, radar.output_figures_dir),
    )
    assert allowed.name == "table.csv"
    with pytest.raises(SessionAuditError, match="outside approved"):
        ensure_write_within(tmp_path / "results" / "old.csv", (radar.output_results_dir, radar.output_figures_dir))


def test_production_output_root_must_be_fresh(tmp_path):
    loaded = load_quality_audit_config(AUDIT_YAML)
    radar = dataclasses.replace(
        loaded.radar,
        output_results_dir=tmp_path / "quality_results",
        output_figures_dir=tmp_path / "quality_figures",
    )
    require_fresh_output_roots(radar)
    radar.output_results_dir.mkdir()
    with pytest.raises(SessionAuditError, match="not fresh"):
        require_fresh_output_roots(radar)


def test_csv_serialization_is_byte_identical_across_fresh_roots(tmp_path):
    loaded = load_quality_audit_config(AUDIT_YAML)
    frame = pd.DataFrame({"subject": [1, 2], "value": [1.0 / 3.0, float("nan")]})
    outputs = []
    for index in (1, 2):
        radar = dataclasses.replace(
            loaded.radar,
            output_results_dir=tmp_path / f"r{index}",
            output_figures_dir=tmp_path / f"f{index}",
        )
        outputs.append(write_csv(frame, radar.output_results_dir / "same.csv", radar))
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_frozen_results_guard_rejects_new_files(tmp_path):
    loaded = load_quality_audit_config(AUDIT_YAML)
    existing = tmp_path / "results"
    existing.mkdir()
    (existing / "original.txt").write_text("frozen", encoding="utf-8")
    radar = dataclasses.replace(
        loaded.radar,
        existing_results_dir=existing,
        output_results_dir=existing / "quality_10ghz",
        output_figures_dir=tmp_path / "figures" / "quality_10ghz",
    )
    snapshot = snapshot_existing_results(radar)
    (existing / "unexpected.txt").write_text("new", encoding="utf-8")
    with pytest.raises(SessionAuditError, match="unexpected new file"):
        verify_frozen_results(snapshot, radar)


def test_mass_mutation_can_change_only_the_three_separate_sanity_artifacts(tmp_path):
    loaded = load_quality_audit_config(AUDIT_YAML)
    radar = dataclasses.replace(
        loaded.radar,
        output_results_dir=tmp_path / "results" / "quality_10ghz",
        output_figures_dir=tmp_path / "figures" / "quality_10ghz",
    )
    radar_csv_names = (
        "session_quality_all_80.csv",
        "block_quality.csv",
        "wst_block_repeatability.csv",
        "subject_relative_quality.csv",
    )
    radar_png_names = ("session_component_heatmap.png", "subject_10_quality_card.png")
    for name in radar_csv_names:
        write_csv(
            pd.DataFrame({"subject": [10], "fixed_radar_value": [1.0]}),
            radar.output_results_dir / name,
            radar,
        )
    radar.output_figures_dir.mkdir(parents=True)
    for name in radar_png_names:
        (radar.output_figures_dir / name).write_bytes(b"fixed-radar-png")
    write_json(
        {"schema_version": "test", "git": {"commit": "c" * 40, "dirty": False}},
        radar.output_results_dir / "provenance_radar.json",
        radar,
    )

    first_card = {"subject": 10, "session_idx": 0, "session_name": "8am"}
    second_card = {"subject": 10, "session_idx": 1, "session_name": "10am"}
    adjacent = pd.DataFrame(
        compare_adjacent_session_geometry(first_card, second_card, {}, {}, radar)
    )
    masses = pd.DataFrame(
        {
            "subject": [10, 10],
            "session_idx": [0, 1],
            "session_name": ["8am", "10am"],
            "mass_kg": [88.2, 88.2],
        }
    )
    workbook = tmp_path / "weights.xlsx"
    workbook.write_bytes(b"equal-mass-fixture")
    finalized_radar_hashes = {"fixed": {"sha256": "d" * 64}}

    def all_output_bytes():
        paths = list(radar.output_results_dir.rglob("*")) + list(
            radar.output_figures_dir.rglob("*")
        )
        return {
            path.relative_to(tmp_path).as_posix(): path.read_bytes()
            for path in paths
            if path.is_file()
        }

    audit_runner._write_recorded_equal_mass_artifacts(
        masses, adjacent, finalized_radar_hashes, workbook, radar
    )
    before = all_output_bytes()
    masses.loc[1, "mass_kg"] = 88.1
    workbook.write_bytes(b"unequal-mass-fixture")
    audit_runner._write_recorded_equal_mass_artifacts(
        masses, adjacent, finalized_radar_hashes, workbook, radar
    )
    after = all_output_bytes()

    radar_only = {
        *(f"results/quality_10ghz/{name}" for name in radar_csv_names),
        *(f"figures/quality_10ghz/{name}" for name in radar_png_names),
        "results/quality_10ghz/provenance_radar.json",
    }
    for rel_path in radar_only:
        assert before[rel_path] == after[rel_path]
    changed = {path for path in before if before[path] != after[path]}
    allowed = {
        "results/quality_10ghz/recorded_equal_mass_sanity.csv",
        "figures/quality_10ghz/recorded_equal_mass_sanity.png",
        "results/quality_10ghz/provenance_recorded_equal_mass.json",
    }
    assert changed
    assert changed <= allowed


def test_no_leakage_test_was_not_repurposed():
    source = (ROOT / "tests" / "test_no_leakage.py").read_text(encoding="utf-8")
    assert "quality_10ghz" not in source
    assert "session_quality" not in source
