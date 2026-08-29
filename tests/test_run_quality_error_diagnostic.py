"""Runner ordering and output-isolation tests for the quality/error diagnostic."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run_quality_error_diagnostic as runner
from dehyd.quality.error_diagnostic import (
    QualityErrorDiagnosticError,
    ensure_fresh_output_roots,
    load_quality_error_config,
    require_production_output_roots,
    snapshot_existing_files,
    verify_existing_files,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "quality_error_10ghz.yaml"


def test_runner_captures_git_before_config(monkeypatch):
    events = []
    monkeypatch.setattr(runner, "capture_clean_git_state", lambda _root: events.append("git") or {"commit": "a" * 40, "branch": "test", "dirty": False})

    def stop(_path):
        assert events == ["git"]
        raise RuntimeError("ordered stop")

    monkeypatch.setattr(runner, "load_quality_error_config", stop)
    with pytest.raises(RuntimeError, match="ordered stop"):
        runner.run(CONFIG)


def test_output_roots_must_be_fresh(tmp_path):
    config = replace(
        load_quality_error_config(CONFIG),
        output_results_dir=tmp_path / "results" / "quality_error_10ghz",
        output_figures_dir=tmp_path / "figures" / "quality_error_10ghz",
    )
    ensure_fresh_output_roots(config)
    config.output_results_dir.mkdir(parents=True)
    with pytest.raises(QualityErrorDiagnosticError, match="fresh"):
        ensure_fresh_output_roots(config)


def test_production_outputs_are_pinned_to_repository_roots(tmp_path):
    config = load_quality_error_config(CONFIG)
    require_production_output_roots(config, ROOT)
    redirected = replace(config, output_results_dir=tmp_path / "quality_error_10ghz")
    with pytest.raises(QualityErrorDiagnosticError, match="production results root"):
        require_production_output_roots(redirected, ROOT)


def test_preexisting_results_and_figures_are_byte_protected(tmp_path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    figures.mkdir()
    (results / "frozen.csv").write_bytes(b"frozen")
    (figures / "frozen.png").write_bytes(b"image")
    config = replace(
        load_quality_error_config(CONFIG),
        output_results_dir=results / "quality_error_10ghz",
        output_figures_dir=figures / "quality_error_10ghz",
    )
    snapshot = snapshot_existing_files(config)
    config.output_results_dir.mkdir()
    (config.output_results_dir / "new.csv").write_text("allowed", encoding="utf-8")
    verify_existing_files(snapshot, config)
    (results / "frozen.csv").write_bytes(b"changed")
    with pytest.raises(QualityErrorDiagnosticError, match="changed"):
        verify_existing_files(snapshot, config)


def test_no_leakage_file_is_untouched_and_not_repurposed():
    source = (ROOT / "tests" / "test_no_leakage.py").read_text(encoding="utf-8")
    assert "quality_error_10ghz" not in source
