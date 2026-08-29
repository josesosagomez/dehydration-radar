import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dehyd.eval.path40_exploratory import (
    Path40Dataset,
    Path40ExploratoryError,
    _canonical_json_sha256,
    _fit_fold,
    evaluate_path40,
    load_path40_dataset,
    write_outputs,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(n_subjects=3, n_sessions=6):
    return {
        "source": {
            "band": "10ghz",
            "population": "A65",
            "diagnostic_git_commit": "a" * 40,
            "analysis_sha256": None,
        },
        "feature": {
            "frontend": "complex10_fasttime_twosided",
            "bank_id": "primary",
            "channel": "",
            "position_pooling": "all_positions",
            "order": 1,
            "path_id": 40,
            "value_column": "value",
            "target_column": "delta_m_pct",
        },
        "model": {"ridge_alpha": 1.0},
        "evaluation": {
            "random_session_n_splits": 3,
            "random_session_seed": 20260821,
        },
        "expected_census": {"subjects": n_subjects, "sessions": n_sessions},
    }


def _diagnostic_fixture(root: Path):
    protocol = _protocol()
    band_root = root / "results" / "wst_order_trajectories" / "10ghz"
    keys = [(subject, session) for subject in range(1, 4) for session in range(2)]

    population = band_root / "population_frames_10ghz.csv"
    session = band_root / "wst_order_session_values_10ghz.csv"
    metadata = band_root / "wst_path_metadata_10ghz.csv"
    group = band_root / "wst_order_group_summary_10ghz.csv"
    _write_csv(
        population,
        [
            {
                "band": "10ghz",
                "population": "A65",
                "subject": str(subject),
                "session_idx": str(session_idx),
            }
            for subject, session_idx in keys
        ],
    )
    _write_csv(
        session,
        [
            {
                "band": "10ghz",
                "frontend": "complex10_fasttime_twosided",
                "bank_id": "primary",
                "channel": "",
                "position_pooling": "all_positions",
                "subject": str(subject),
                "session_idx": str(session_idx),
                "order": "1",
                "path_id": "40",
                "value": str(subject + 0.25 * session_idx),
                "delta_m_pct": str(-0.2 * session_idx),
            }
            for subject, session_idx in keys
        ],
    )
    _write_csv(
        metadata,
        [
            {
                "band": "10ghz",
                "frontend": "complex10_fasttime_twosided",
                "bank_id": "primary",
                "channel": "",
                "order": "1",
                "path_id": "40",
                "scientific_output": "true",
            }
        ],
    )
    _write_csv(
        group,
        [
            {
                "band": "10ghz",
                "population": "A65",
                "frontend": "complex10_fasttime_twosided",
                "bank_id": "primary",
                "channel": "",
                "position_pooling": "all_positions",
                "order": "1",
                "path_id": "40",
                "effect_axis": "session_index",
                "shared_primary_candidate": "true",
                "n_subjects": "3",
                "majority_sign": "1",
                "majority_fraction": "1.0",
                "holm_adjusted_p": "1.0",
                "multiplicity_supported": "false",
                "sensitivity_direction_persists": "true",
                "boundary_direction_persists": "true",
            }
        ],
    )
    (band_root / "freeze_manifest_10ghz.json").write_text(
        json.dumps(
            {
                "status": "frozen_no_real_data",
                "band": "10ghz",
                "diagnostic_git_commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    scientific = {
        "row_hashes": {
            path.name: _sha256(path)
            for path in (population, session, metadata, group)
        }
    }
    analysis_hash = _canonical_json_sha256(scientific)
    protocol["source"]["analysis_sha256"] = analysis_hash
    (band_root / "analysis_status_10ghz.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "band": "10ghz",
                "analysis_sha256": analysis_hash,
                "scientific_content": scientific,
            }
        ),
        encoding="utf-8",
    )
    return protocol, session


def test_loader_authenticates_snapshot_and_extracts_one_path_row_per_session(tmp_path):
    protocol, session_path = _diagnostic_fixture(tmp_path)
    dataset = load_path40_dataset(tmp_path, protocol)

    assert dataset.features.shape == (6, 1)
    assert dataset.subjects.tolist() == [1, 1, 2, 2, 3, 3]
    assert dataset.source["candidate_summary"]["holm_adjusted_p"] == "1.0"

    session_path.write_text(session_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(Path40ExploratoryError, match="hash mismatch"):
        load_path40_dataset(tmp_path, protocol)


def _synthetic_dataset() -> Path40Dataset:
    subjects = np.repeat(np.arange(1, 7), 4)
    sessions = np.tile(np.arange(4), 6)
    feature = subjects * 0.5 + sessions * 0.2
    target = -(sessions * 0.3) + subjects * 0.01
    return Path40Dataset(
        subjects=subjects,
        session_indices=sessions,
        features=feature.reshape(-1, 1),
        targets=target,
        source={},
    )


def test_both_protocols_cover_every_session_and_only_random_split_overlaps_subjects():
    dataset = _synthetic_dataset()
    protocol = _protocol(n_subjects=6, n_sessions=24)
    result, rows = evaluate_path40(dataset, protocol)

    loso = result["evaluations"]["loso"]
    random = result["evaluations"]["random_session_split"]
    assert loso["n_folds"] == 6
    assert loso["folds_with_subject_overlap"] == 0
    assert random["n_folds"] == 3
    assert random["folds_with_subject_overlap"] == 3
    assert loso["subject_isolated"] is True and loso["leaky_protocol"] is False
    assert random["subject_isolated"] is False and random["leaky_protocol"] is True
    assert len(rows) == 48
    assert all(row["post_selection_exploratory"] for row in rows)
    assert not any(row["leaky_protocol"] for row in rows[:24])
    assert all(row["leaky_protocol"] for row in rows[24:])


def test_held_out_targets_cannot_change_fold_fit_or_train_mean_reference():
    dataset = _synthetic_dataset()
    train = np.flatnonzero(dataset.subjects != 6)
    test = np.flatnonzero(dataset.subjects == 6)
    path_before, mean_before = _fit_fold(dataset, train, test, alpha=1.0)

    mutated = Path40Dataset(
        subjects=dataset.subjects.copy(),
        session_indices=dataset.session_indices.copy(),
        features=dataset.features.copy(),
        targets=dataset.targets.copy(),
        source={},
    )
    mutated.targets[test] += 1000.0
    path_after, mean_after = _fit_fold(mutated, train, test, alpha=1.0)
    np.testing.assert_array_equal(path_before, path_after)
    np.testing.assert_array_equal(mean_before, mean_after)


def test_outputs_are_machine_labelled_and_isolated(tmp_path):
    dataset = _synthetic_dataset()
    protocol = _protocol(n_subjects=6, n_sessions=24)
    result, rows = evaluate_path40(dataset, protocol)
    config = tmp_path / "protocol.yaml"
    config.write_text("schema_version: test\n", encoding="utf-8")

    outputs = write_outputs(tmp_path, result, rows, config_path=config)
    assert outputs["summary"].parent == tmp_path / "results" / "exploratory_path40"
    assert "postSelection_exploratory" in outputs["summary"].name
    assert "randomSessionSplit_leaked" in outputs["summary"].name
    assert "randomSessionSplit_leaked" in outputs["predictions"].name
    payload = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert payload["never_report_as_confirmatory"] is True
    assert payload["evaluations"]["loso"]["folds_with_subject_overlap"] == 0
