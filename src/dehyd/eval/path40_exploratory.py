"""Exploratory path-40 regression with and without subject isolation.

This module intentionally keeps two different caveats visible:

* the LOSO fit has no train/test subject overlap, but path 40 was identified using
  the same cohort before this model was proposed, so its score is post-selection and
  is not an unbiased confirmatory estimate;
* the random-session split deliberately places subjects on both sides of a split and
  is therefore leaky.  It exists only to answer the owner's private curiosity.

Both evaluations use the same single raw WST coefficient, train-fold-only scaling,
fixed Ridge model, and train-fold-only mean reference.  Only the split changes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import pooled_pearson_r, session_rmse, subject_balanced_mae


SCHEMA_VERSION = "path40_post_selection_exploratory_v1"
OUTPUT_DIRECTORY = "exploratory_path40"
OUTPUT_TAG = "path40_postSelection_exploratory"
RANDOM_TAG = "randomSessionSplit_leaked"


class Path40ExploratoryError(ValueError):
    """The frozen source, requested path, split, or output contract is invalid."""


@dataclass(frozen=True)
class Path40Dataset:
    subjects: np.ndarray
    session_indices: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    source: dict[str, Any]


def _canonical_json_sha256(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: normalize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        if isinstance(item, float) and item == 0.0:
            return 0.0
        return item

    encoded = json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Path40ExploratoryError(f"cannot read valid JSON from {path}") from error
    if not isinstance(value, dict):
        raise Path40ExploratoryError(f"expected a JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise Path40ExploratoryError(f"cannot read CSV {path}") from error


def _require_row_hash(status: Mapping[str, Any], path: Path) -> str:
    scientific = status.get("scientific_content")
    row_hashes = scientific.get("row_hashes") if isinstance(scientific, dict) else None
    expected = row_hashes.get(path.name) if isinstance(row_hashes, dict) else None
    if not isinstance(expected, str) or len(expected) != 64:
        raise Path40ExploratoryError(f"analysis status lacks a hash for {path.name}")
    observed = _sha256_file(path)
    if observed != expected:
        raise Path40ExploratoryError(
            f"frozen diagnostic hash mismatch for {path.name}: {observed} != {expected}"
        )
    return observed


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"


def load_path40_dataset(diagnostic_root: str | Path, protocol: Mapping[str, Any]) -> Path40Dataset:
    """Authenticate the completed diagnostic and load the exact A65 path-40 rows."""
    source_cfg = protocol["source"]
    feature_cfg = protocol["feature"]
    expected_cfg = protocol["expected_census"]

    repository = Path(diagnostic_root).expanduser().resolve(strict=True)
    band = str(source_cfg["band"])
    band_root = repository / "results" / "wst_order_trajectories" / band
    paths = {
        "status": band_root / f"analysis_status_{band}.json",
        "freeze": band_root / f"freeze_manifest_{band}.json",
        "population": band_root / f"population_frames_{band}.csv",
        "session": band_root / f"wst_order_session_values_{band}.csv",
        "metadata": band_root / f"wst_path_metadata_{band}.csv",
        "group": band_root / f"wst_order_group_summary_{band}.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise Path40ExploratoryError(f"frozen diagnostic inputs are missing: {missing}")

    status = _read_json(paths["status"])
    scientific = status.get("scientific_content")
    if (
        status.get("status") != "complete"
        or status.get("band") != band
        or not isinstance(scientific, dict)
        or _canonical_json_sha256(scientific) != status.get("analysis_sha256")
        or status.get("analysis_sha256") != source_cfg["analysis_sha256"]
    ):
        raise Path40ExploratoryError("analysis status is not the authorized completed snapshot")

    freeze = _read_json(paths["freeze"])
    if (
        freeze.get("status") != "frozen_no_real_data"
        or freeze.get("band") != band
        or freeze.get("diagnostic_git_commit") != source_cfg["diagnostic_git_commit"]
    ):
        raise Path40ExploratoryError("freeze manifest is not the authorized diagnostic commit")

    hashes = {
        name: _require_row_hash(status, paths[name])
        for name in ("population", "session", "metadata", "group")
    }

    population = str(source_cfg["population"])
    population_rows = _read_csv(paths["population"])
    member_keys = {
        (int(row["subject"]), int(row["session_idx"]))
        for row in population_rows
        if row["band"] == band and row["population"] == population
    }
    if len(member_keys) != int(expected_cfg["sessions"]):
        raise Path40ExploratoryError(
            f"{population} has {len(member_keys)} sessions, expected {expected_cfg['sessions']}"
        )

    filters = {
        "band": band,
        "frontend": str(feature_cfg["frontend"]),
        "bank_id": str(feature_cfg["bank_id"]),
        "channel": str(feature_cfg["channel"]),
        "position_pooling": str(feature_cfg["position_pooling"]),
        "order": str(feature_cfg["order"]),
        "path_id": str(feature_cfg["path_id"]),
    }
    selected = []
    for row in _read_csv(paths["session"]):
        key = (int(row["subject"]), int(row["session_idx"]))
        if key in member_keys and all(row[name] == value for name, value in filters.items()):
            selected.append(row)
    selected.sort(key=lambda row: (int(row["subject"]), int(row["session_idx"])))

    selected_keys = [(int(row["subject"]), int(row["session_idx"])) for row in selected]
    if len(selected_keys) != len(set(selected_keys)) or set(selected_keys) != member_keys:
        raise Path40ExploratoryError(
            "path 40 does not provide exactly one coefficient for every A65 session"
        )

    metadata_rows = [
        row for row in _read_csv(paths["metadata"])
        if all(row[name] == value for name, value in filters.items() if name != "position_pooling")
    ]
    if len(metadata_rows) != 1 or not _truth(metadata_rows[0]["scientific_output"]):
        raise Path40ExploratoryError("path 40 is absent or not a scientific-output path")

    candidate_rows = [
        row for row in _read_csv(paths["group"])
        if row["band"] == band
        and row["population"] == population
        and row["frontend"] == filters["frontend"]
        and row["bank_id"] == filters["bank_id"]
        and row["channel"] == filters["channel"]
        and row["position_pooling"] == filters["position_pooling"]
        and row["order"] == filters["order"]
        and row["path_id"] == filters["path_id"]
        and row["effect_axis"] == "session_index"
    ]
    if len(candidate_rows) != 1 or not _truth(candidate_rows[0]["shared_primary_candidate"]):
        raise Path40ExploratoryError("the requested row is not the frozen path-40 candidate")

    subjects = np.asarray([int(row["subject"]) for row in selected], dtype=np.int64)
    sessions = np.asarray([int(row["session_idx"]) for row in selected], dtype=np.int64)
    values = np.asarray([float(row[feature_cfg["value_column"]]) for row in selected])
    targets = np.asarray([float(row[feature_cfg["target_column"]]) for row in selected])
    if (
        len(set(subjects.tolist())) != int(expected_cfg["subjects"])
        or values.shape != targets.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(targets))
    ):
        raise Path40ExploratoryError("path-40 dataset census or finite-value check failed")

    return Path40Dataset(
        subjects=subjects,
        session_indices=sessions,
        features=values.reshape(-1, 1),
        targets=targets,
        source={
            "diagnostic_root": str(repository),
            "diagnostic_git_commit": freeze["diagnostic_git_commit"],
            "analysis_sha256": status["analysis_sha256"],
            "input_sha256": hashes,
            "candidate_summary": {
                key: candidate_rows[0][key]
                for key in (
                    "n_subjects", "majority_sign", "majority_fraction",
                    "holm_adjusted_p", "multiplicity_supported",
                    "sensitivity_direction_persists", "boundary_direction_persists",
                )
            },
        },
    )


def _fit_fold(
    dataset: Path40Dataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, solver="cholesky")),
        ]
    )
    model.fit(dataset.features[train_indices], dataset.targets[train_indices])
    path_predictions = model.predict(dataset.features[test_indices])
    mean_predictions = np.full(test_indices.size, dataset.targets[train_indices].mean())
    return path_predictions, mean_predictions


def _summarize(subjects: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    pearson = pooled_pearson_r(truth, prediction)
    return {
        "subject_balanced_mae_pct": subject_balanced_mae(subjects, truth, prediction),
        "session_mae_pct": float(np.mean(np.abs(truth - prediction))),
        "session_rmse_pct": session_rmse(truth, prediction),
        "pooled_pearson_r": pearson if math.isfinite(pearson) else None,
    }


def _evaluate_splits(
    dataset: Path40Dataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    protocol_name: str,
    alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = dataset.targets.size
    path_predictions = np.full(n, np.nan)
    mean_predictions = np.full(n, np.nan)
    fold_ids = np.full(n, -1, dtype=np.int64)
    manifests = []

    for fold_id, (train_indices, test_indices) in enumerate(splits):
        train_subjects = set(dataset.subjects[train_indices].tolist())
        test_subjects = set(dataset.subjects[test_indices].tolist())
        overlap = sorted(train_subjects & test_subjects)
        path_pred, mean_pred = _fit_fold(
            dataset, train_indices, test_indices, alpha=alpha
        )
        if np.any(fold_ids[test_indices] != -1):
            raise Path40ExploratoryError("a session is tested in more than one fold")
        path_predictions[test_indices] = path_pred
        mean_predictions[test_indices] = mean_pred
        fold_ids[test_indices] = fold_id
        manifests.append(
            {
                "fold": fold_id,
                "n_train_sessions": int(train_indices.size),
                "n_test_sessions": int(test_indices.size),
                "train_subjects": sorted(train_subjects),
                "test_subjects": sorted(test_subjects),
                "overlap_subjects": overlap,
            }
        )

    if np.any(fold_ids < 0) or not np.all(np.isfinite(path_predictions)):
        raise Path40ExploratoryError("out-of-fold predictions are incomplete")
    overlap_count = sum(bool(fold["overlap_subjects"]) for fold in manifests)
    if protocol_name == "loso" and overlap_count:
        raise Path40ExploratoryError("LOSO split contains train/test subject overlap")
    if protocol_name == "random_session_split" and overlap_count == 0:
        raise Path40ExploratoryError("random session split unexpectedly has no subject overlap")

    path_metrics = _summarize(dataset.subjects, dataset.targets, path_predictions)
    baseline_metrics = _summarize(dataset.subjects, dataset.targets, mean_predictions)
    summary = {
        "protocol": protocol_name,
        "subject_isolated": protocol_name == "loso",
        "leaky_protocol": protocol_name == "random_session_split",
        "post_selection_exploratory": True,
        "n_sessions": int(n),
        "n_subjects": int(len(set(dataset.subjects.tolist()))),
        "n_folds": len(splits),
        "folds_with_subject_overlap": overlap_count,
        "path40_ridge": path_metrics,
        "train_mean_reference": baseline_metrics,
        "path_minus_reference_subject_balanced_mae_pct": (
            path_metrics["subject_balanced_mae_pct"]
            - baseline_metrics["subject_balanced_mae_pct"]
        ),
        "fold_manifest": manifests,
    }
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": protocol_name,
            "fold": int(fold_ids[index]),
            "subject": int(dataset.subjects[index]),
            "session_idx": int(dataset.session_indices[index]),
            "path40_value": float(dataset.features[index, 0]),
            "signed_delta_m_pct": float(dataset.targets[index]),
            "path40_prediction_pct": float(path_predictions[index]),
            "train_mean_prediction_pct": float(mean_predictions[index]),
            "post_selection_exploratory": True,
            "leaky_protocol": protocol_name == "random_session_split",
            "never_report_as_confirmatory": True,
        }
        for index in range(n)
    ]
    return summary, rows


def evaluate_path40(dataset: Path40Dataset, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the same fixed model under LOSO and random session K-fold."""
    model_cfg = protocol["model"]
    evaluation_cfg = protocol["evaluation"]
    alpha = float(model_cfg["ridge_alpha"])

    subjects = dataset.subjects
    loso_splits = [
        (np.flatnonzero(subjects != subject), np.flatnonzero(subjects == subject))
        for subject in sorted(set(subjects.tolist()))
    ]
    kfold = KFold(
        n_splits=int(evaluation_cfg["random_session_n_splits"]),
        shuffle=True,
        random_state=int(evaluation_cfg["random_session_seed"]),
    )
    random_splits = [(train, test) for train, test in kfold.split(dataset.features)]

    loso, loso_rows = _evaluate_splits(
        dataset, loso_splits, protocol_name="loso", alpha=alpha
    )
    random, random_rows = _evaluate_splits(
        dataset, random_splits, protocol_name="random_session_split", alpha=alpha
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scientific_status": "exploratory_only",
        "post_selection_exploratory": True,
        "contains_leaky_protocol": True,
        "never_report_as_confirmatory": True,
        "warnings": {
            "loso": (
                "Subject isolation is valid, but path 40 was selected on this same cohort; "
                "the score is post-selection and not an unbiased confirmatory estimate."
            ),
            "random_session_split": (
                "Subjects occur in both training and test sets; this score is deliberately "
                "leaky and must not be used as evidence of generalization."
            ),
        },
        "feature": dict(protocol["feature"]),
        "model": {
            "family": "standard_scaler_then_ridge",
            "ridge_alpha": alpha,
            "ridge_solver": "cholesky",
            "fitting": "every fitted quantity uses training rows only",
        },
        "source": dataset.source,
        "evaluations": {"loso": loso, "random_session_split": random},
    }
    return result, loso_rows + random_rows


def _git_info(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                arguments, cwd=repository, capture_output=True, text=True, timeout=20
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD") or os.environ.get("DEHYD_GIT_COMMIT"),
        "branch": run("git", "branch", "--show-current") or os.environ.get("DEHYD_GIT_BRANCH"),
        "dirty": None if status is None else bool(status),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + f".partial.{os.getpid()}")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    partial = path.with_name(path.name + f".partial.{os.getpid()}")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def write_outputs(
    repository: str | Path,
    result: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    *,
    config_path: str | Path,
) -> dict[str, Path]:
    """Write only to the machine-labelled exploratory allowlist directory."""
    repository = Path(repository).resolve(strict=True)
    output_root = (repository / "results" / OUTPUT_DIRECTORY).resolve()
    if not output_root.is_relative_to((repository / "results").resolve()):
        raise Path40ExploratoryError("exploratory output escaped the repository results root")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"summary_{OUTPUT_TAG}_{RANDOM_TAG}.json"
    predictions_path = output_root / f"predictions_{OUTPUT_TAG}_{RANDOM_TAG}.csv"
    if OUTPUT_TAG not in summary_path.name or OUTPUT_TAG not in predictions_path.name:
        raise Path40ExploratoryError("exploratory output filename lacks its mandatory tag")

    config_path = Path(config_path).resolve(strict=True)
    result["provenance"] = {
        "analysis_repository": _git_info(repository),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: metadata.version(name)
            for name in ("numpy", "scikit-learn", "PyYAML")
        },
    }
    _write_csv_atomic(predictions_path, prediction_rows)
    result["prediction_csv"] = {
        "path": str(predictions_path),
        "sha256": _sha256_file(predictions_path),
    }
    _write_json_atomic(summary_path, result)
    return {"summary": summary_path, "predictions": predictions_path}
