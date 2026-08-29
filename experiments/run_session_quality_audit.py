"""Run the approved descriptive 10 GHz session-quality and repeatability audit.

This command never changes sample eligibility, retrains a model, or uses body mass in
the radar-only stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.provenance import sha256_file  # noqa: E402
from dehyd.quality.config import load_quality_audit_config  # noqa: E402
from dehyd.quality.radar_inventory_10ghz import (  # noqa: E402
    assert_target_free_columns,
    build_radar_inventory,
    load_session_and_qc,
)
from dehyd.quality.session_audit import (  # noqa: E402
    SessionAuditError,
    build_diagnostic_scattering,
    build_radar_provenance,
    build_subject_relative_table,
    capture_clean_git_provenance,
    compare_adjacent_session_geometry,
    compute_frame_components,
    compute_session_wst_repeatability,
    plot_recorded_equal_mass,
    plot_session_component_heatmap,
    plot_subject_10_card,
    require_fresh_output_roots,
    select_recorded_equal_mass_rows,
    snapshot_existing_results,
    summarize_session_components,
    verify_frozen_results,
    write_csv,
    write_json,
)  # noqa: E402


def _reconcile_existing_qc(cards: pd.DataFrame, source_path: Path) -> None:
    if not source_path.is_file():
        raise SessionAuditError(f"existing frozen QC report is missing: {source_path}")
    source = pd.read_csv(source_path).sort_values(["subject", "session_idx"]).reset_index(drop=True)
    current = cards.sort_values(["subject", "session_idx"]).reset_index(drop=True)
    if len(source) != 80 or len(current) != 80:
        raise SessionAuditError("QC reconciliation requires exactly 80 session rows")
    identity_columns = ["subject", "session_idx"]
    if not source[identity_columns].equals(current[identity_columns]):
        raise SessionAuditError("target-free QC session identities disagree with frozen report")
    mapping = {
        "n_frames": "n_raw",
        "n_pass": "n_pass",
        "n_fail_any": "n_fail_any",
        "n_nan_inf": "n_nan_inf",
        "n_flatline": "n_flatline",
        "n_low_in_band": "n_low_in_band",
        "n_rms_flagged": "n_rms_flagged",
        "eligible": "eligible_existing_qc",
    }
    for source_column, current_column in mapping.items():
        if source[source_column].tolist() != current[current_column].tolist():
            raise SessionAuditError(
                f"target-free QC disagrees with frozen report column {source_column!r}"
            )
    if int(current["eligible_existing_qc"].sum()) != 73:
        raise SessionAuditError("frozen eligibility did not reconcile to 73 sessions")


def _artifact_hashes(paths: dict[str, Path]) -> dict:
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _write_recorded_equal_mass_artifacts(
    mass_sessions: pd.DataFrame,
    adjacent_geometry: pd.DataFrame,
    finalized_radar_hashes: dict,
    weight_path: Path,
    config,
) -> dict[str, Path]:
    """Write the only three artifacts allowed to depend on recorded body mass."""
    paths: dict[str, Path] = {}
    equal_mass = select_recorded_equal_mass_rows(mass_sessions, adjacent_geometry)
    paths["equal_mass"] = write_csv(
        equal_mass, config.output_results_dir / "recorded_equal_mass_sanity.csv", config
    )
    paths["equal_mass_figure"] = plot_recorded_equal_mass(
        equal_mass, config.output_figures_dir / "recorded_equal_mass_sanity.png", config
    )
    equal_provenance = {
        "schema_version": "quality_10ghz_recorded_equal_mass_provenance_v1",
        "analysis_role": "separate_recorded_equal_mass_sanity_view",
        "warning": (
            "Recorded equality is rounded body mass, not proof of unchanged physiology. "
            "This output cannot set QC, exclusion, WST choice, or training weight."
        ),
        "ground_truth": {
            "rel_path": weight_path.name,
            "sha256": sha256_file(weight_path),
        },
        "radar_artifacts": finalized_radar_hashes,
        "n_recorded_equal_mass_pairs": int(
            equal_mass[["subject", "session_a_idx", "session_b_idx"]]
            .drop_duplicates()
            .shape[0]
        ),
        "outputs": _artifact_hashes(
            {"table": paths["equal_mass"], "figure": paths["equal_mass_figure"]}
        ),
    }
    paths["provenance_equal_mass"] = write_json(
        equal_provenance,
        config.output_results_dir / "provenance_recorded_equal_mass.json",
        config,
    )
    return paths


def run(config_path: str | Path) -> dict[str, Path]:
    git_record = capture_clean_git_provenance()
    loaded = load_quality_audit_config(config_path)
    config = loaded.radar
    require_fresh_output_roots(config)

    print("snapshot     : hashing existing local results (including feature arrays)")
    frozen_snapshot = snapshot_existing_results(config)
    config.output_results_dir.mkdir(parents=True, exist_ok=False)
    config.output_figures_dir.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {}
    paths["frozen_inputs"] = write_json(
        frozen_snapshot, config.output_results_dir / "frozen_inputs_before.json", config
    )

    sessions, inventory = build_radar_inventory(config)
    if len(inventory) != 8000 or len(sessions) != 80:
        raise SessionAuditError("target-free census is not 80 sessions / 8,000 frames")
    scattering_banks = build_diagnostic_scattering(config)

    cards, blocks, repeatability_rows = [], [], []
    adjacent_geometry_rows = []
    previous_card = None
    previous_geometry = None
    for position, session in enumerate(sessions, start=1):
        cube, frame_qc = load_session_and_qc(session, config)
        frame_components = compute_frame_components(cube, frame_qc, config)
        card, block_rows = summarize_session_components(frame_components, config)
        cards.append(card)
        blocks.extend(block_rows)
        passing = frame_components.loc[frame_components["qc_pass"], "frame_idx"].astype(int).tolist()
        rows, geometry = compute_session_wst_repeatability(
            cube, passing, card, config, scattering_banks
        )
        repeatability_rows.extend(rows)
        if previous_card is not None and previous_card["subject"] == card["subject"]:
            adjacent_geometry_rows.extend(
                compare_adjacent_session_geometry(
                    previous_card,
                    card,
                    previous_geometry,
                    geometry,
                    config,
                )
            )
        # Only one previous session's high-dimensional shape geometry remains live.
        # The compact adjacent-pair results have no target and stay in memory until
        # the radar artifacts are finalized and the separate mass filter is allowed.
        previous_card = card
        previous_geometry = geometry
        print(
            f"  [{position:02d}/80] subject {card['subject']:>2} {card['session_name']:<5} "
            f"{card['n_pass']:>3}/100 {card['audit_status']}"
        )

    cards_frame = pd.DataFrame(cards).sort_values(["subject", "session_idx"]).reset_index(drop=True)
    blocks_frame = pd.DataFrame(blocks).sort_values(["subject", "session_idx", "block_idx"]).reset_index(drop=True)
    repeatability = pd.DataFrame(repeatability_rows).sort_values(
        ["subject", "session_idx", "diagnostic_channel", "tiling_idx", "scattering_order", "view"]
    ).reset_index(drop=True)
    adjacent_geometry = pd.DataFrame(adjacent_geometry_rows)
    if len(adjacent_geometry) != 16 * 4 * 36:
        raise SessionAuditError(
            f"adjacent radar geometry has {len(adjacent_geometry)} rows, expected 2304"
        )
    if len(cards_frame) != 80 or cards_frame.duplicated(["subject", "session_idx"]).any():
        raise SessionAuditError("session card lost or duplicated a census cell")
    if len(blocks_frame) != 400:
        raise SessionAuditError(f"block table has {len(blocks_frame)} rows, expected 400")
    if len(repeatability) != 80 * 36:
        raise SessionAuditError(f"repeatability table has {len(repeatability)} rows, expected 2880")
    _reconcile_existing_qc(cards_frame, config.existing_qc_report)

    relative = build_subject_relative_table(cards_frame, repeatability)
    for radar_frame in (cards_frame, blocks_frame, repeatability, relative):
        assert_target_free_columns(radar_frame.columns)

    paths["session_quality"] = write_csv(
        cards_frame, config.output_results_dir / "session_quality_all_80.csv", config
    )
    paths["block_quality"] = write_csv(
        blocks_frame, config.output_results_dir / "block_quality.csv", config
    )
    paths["repeatability"] = write_csv(
        repeatability, config.output_results_dir / "wst_block_repeatability.csv", config
    )
    paths["subject_relative"] = write_csv(
        relative, config.output_results_dir / "subject_relative_quality.csv", config
    )
    paths["component_heatmap"] = plot_session_component_heatmap(
        cards_frame, config.output_figures_dir / "session_component_heatmap.png", config
    )
    paths["subject_10"] = plot_subject_10_card(
        cards_frame, config.output_figures_dir / "subject_10_quality_card.png", config
    )

    radar_preprovenance = _artifact_hashes(paths)
    provenance = build_radar_provenance(
        config,
        raw_sessions=sessions,
        source_artifacts={
            "existing_qc_report": {
                "path": str(config.existing_qc_report),
                "sha256": sha256_file(config.existing_qc_report),
            },
            "base_config": {
                "path": str(loaded.base_config_path),
                "sha256": sha256_file(loaded.base_config_path),
                "resolved_mapping_sha256": loaded.base_config_sha256,
            },
        },
        radar_artifacts=radar_preprovenance,
        census={
            "n_subjects": 16,
            "n_sessions": 80,
            "n_frames": 8000,
            "n_eligible_sessions": int(cards_frame["eligible_existing_qc"].sum()),
            "n_repeatability_analysable": int(
                (cards_frame["audit_status"] == "REPEATABILITY_ANALYSABLE").sum()
            ),
        },
        git_record=git_record,
    )
    paths["provenance_radar"] = write_json(
        provenance, config.output_results_dir / "provenance_radar.json", config
    )
    finalized_radar_hashes = _artifact_hashes(paths)

    # Outcome-bearing work begins only after every radar-only artifact is finalized.
    from dehyd.data.ground_truth import load_ground_truth

    ground_truth = load_ground_truth(loaded.weight_xlsx)
    paths.update(
        _write_recorded_equal_mass_artifacts(
            ground_truth.sessions,
            adjacent_geometry,
            finalized_radar_hashes,
            loaded.weight_xlsx,
            config,
        )
    )

    verify_frozen_results(frozen_snapshot, config)
    print(f"complete     : {config.output_results_dir}")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="quality-audit YAML")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
