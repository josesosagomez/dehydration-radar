"""Relate frozen 10 GHz session quality to authoritative Experiment-A LOSO error."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.quality.error_diagnostic import (  # noqa: E402
    analyze_metrics,
    assemble_analysis_table,
    authenticate_exp_a_sources,
    build_session_prediction_table,
    capture_clean_git_state,
    ensure_fresh_output_roots,
    load_quality_error_config,
    metric_catalog,
    plot_association_forest,
    plot_population_flow,
    plot_residual_panels,
    provenance_record,
    reconstruct_prediction_bytes,
    require_production_output_roots,
    sha256_file,
    snapshot_existing_files,
    verify_and_load_quality,
    verify_existing_files,
    write_csv_deterministic,
    write_json_deterministic,
    _read_json,
)


def run(config_path: str | Path) -> dict[str, Path]:
    # Capture source state before even loading config or creating output directories.
    repo_root = Path(__file__).resolve().parents[1]
    git_state = capture_clean_git_state(repo_root)
    config = load_quality_error_config(config_path)
    require_production_output_roots(config, repo_root)
    ensure_fresh_output_roots(config)
    frozen_snapshot = snapshot_existing_files(config)

    reference = _read_json(config.reference_manifest)
    sources = _read_json(config.exp_a_sources)
    reference_band = authenticate_exp_a_sources(reference, sources)
    reconstructed_bytes = reconstruct_prediction_bytes(reference_band)
    predictions = build_session_prediction_table(reference_band)
    cards, repeatability, _quality_provenance = verify_and_load_quality(config)
    analysis_table, population_flow = assemble_analysis_table(predictions, cards, repeatability)
    association_summary, influence, residuals = analyze_metrics(analysis_table, config)
    catalog = metric_catalog()

    config.output_results_dir.mkdir(parents=True, exist_ok=False)
    config.output_figures_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "session_table": write_csv_deterministic(
            analysis_table, config.output_results_dir / "session_quality_vs_loso_error.csv"
        ),
        "metric_catalog": write_csv_deterministic(
            catalog, config.output_results_dir / "quality_metric_catalog.csv"
        ),
        "association_summary": write_csv_deterministic(
            association_summary, config.output_results_dir / "association_summary.csv"
        ),
        "influence": write_csv_deterministic(
            influence, config.output_results_dir / "leave_one_subject_out_influence.csv"
        ),
        "population_flow": write_csv_deterministic(
            population_flow, config.output_results_dir / "population_flow.csv"
        ),
        "association_forest": plot_association_forest(
            association_summary, config.output_figures_dir / "association_forest.png"
        ),
        "residual_panels": plot_residual_panels(
            residuals, config.output_figures_dir / "quality_error_residual_panels.png"
        ),
        "population_flow_figure": plot_population_flow(
            population_flow, config.output_figures_dir / "population_flow.png"
        ),
    }
    source_hashes = {
        "reference_manifest": {
            "path": str(config.reference_manifest), "sha256": sha256_file(config.reference_manifest)
        },
        "exp_a_sources": {
            "path": str(config.exp_a_sources), "sha256": sha256_file(config.exp_a_sources)
        },
        "quality_provenance": {
            "path": str(config.quality_provenance), "sha256": sha256_file(config.quality_provenance)
        },
        "session_quality": {
            "path": str(config.session_quality), "sha256": sha256_file(config.session_quality)
        },
        "wst_repeatability": {
            "path": str(config.wst_repeatability), "sha256": sha256_file(config.wst_repeatability)
        },
        "reconstructed_predictions": {
            "path": "in-memory authoritative manifest reconstruction",
            "sha256": __import__("hashlib").sha256(reconstructed_bytes).hexdigest(),
        },
    }
    provenance = provenance_record(
        config, git_state, source_hashes, paths, association_summary
    )
    paths["provenance"] = write_json_deterministic(
        provenance, config.output_results_dir / "provenance.json"
    )
    verify_existing_files(frozen_snapshot, config)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="quality-error YAML declaration")
    args = parser.parse_args(argv)
    paths = run(args.config)
    print(f"complete: {paths['association_summary'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
