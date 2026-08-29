"""Exploratory association between 10 GHz session quality and frozen LOSO error.

This module reads already-finalized diagnostic artifacts.  It never imports or runs
training, preprocessing, feature extraction, or ground-truth loading code.  The unit of
analysis is one subject-session; repeated model seeds are summarized within that unit.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


EXPECTED_PREDICTION_SHA256 = "78d6076c5c5fcd79cf7b994c5f7ad508832228f6cf9baca70bbdc39cf1cebf9e"
OBSOLETE_PREDICTION_SHA256 = "4bd21201cb87a62aed32b19e7f5fbb478fd7354a6a2c08040cfad6a377145c57"
EXPECTED_FINAL_RUN = "20260810T153739562215Z_04dc9521"
EXPECTED_FINAL_COMMIT = "04dc9521346215cc20a8402f0d00f63c36cf3b42"
EXPECTED_REFERENCE_SCHEMA = "reference_exp_a_manifest_v1"
EXPECTED_SOURCES_SCHEMA = "exp_a_sources_v1"
EXPECTED_EVIDENCE_CLASSES = (
    "population", "folds", "stage1_candidates", "selected_feature_keys",
    "stage2_candidates", "feature_inputs", "tuned_epsilon", "feature_matrices",
    "selection_table", "predictions", "scores",
)
EXPECTED_SESSION_KEYS_SHA256 = "ef937439d976ba5dc5e3c42d7ac08fb6c78996987ef1fdb80dacd1d23e24de54"
EXPECTED_TARGETS_SHA256 = "1f0530dd490513df2e1638345ae505e9f6d9bfb84455c1a804d9ac0aa38f3e51"
EXPECTED_FRAME_POPULATION_SHA256 = "e2d0ea2ca79c4eaa8cd24b63f19bee40cc4f04dcbdcfabc575d4422ea6379c8a"


class QualityErrorDiagnosticError(ValueError):
    """Raised when a frozen source or an analytic contract does not reconcile."""


@dataclass(frozen=True)
class QualityErrorConfig:
    reference_manifest: Path
    exp_a_sources: Path
    quality_provenance: Path
    session_quality: Path
    wst_repeatability: Path
    output_results_dir: Path
    output_figures_dir: Path
    bootstrap_draws: int
    bootstrap_seed: int
    confidence_level: float
    maximum_invalid_fraction: float
    strict: bool


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    source_table: str
    source_key: str
    units: str
    orientation: str
    analytic_population: str
    tiling_idx: int | None = None


METRICS = (
    MetricDefinition(
        "one_minus_pass_fraction", "session_quality_all_80.csv", "pass_fraction",
        "fraction", "1 - source; larger means worse", "Exp-A eligible sessions (N=73)",
    ),
    MetricDefinition(
        "twenty_minus_minimum_block_n_pass", "session_quality_all_80.csv",
        "minimum_block_n_pass", "frames", "20 - source; larger means worse",
        "Exp-A eligible sessions (N=73)",
    ),
    MetricDefinition(
        "negative_in_band_ratio_p10_margin", "session_quality_all_80.csv",
        "in_band_ratio_p10_margin", "dimensionless ratio",
        "negative source; larger means worse", "Exp-A eligible sessions (N=73)",
    ),
    MetricDefinition(
        "peak_bin_iqr", "session_quality_all_80.csv", "peak_bin_iqr", "FFT bins",
        "identity; larger means worse", "Exp-A eligible sessions (N=73)",
    ),
    *tuple(
        MetricDefinition(
            f"wst_tiling_{tiling}_block_distance_maximum",
            "wst_block_repeatability.csv", "block_to_session_distance_maximum",
            "dimensionless Euclidean distance", "identity; larger means worse",
            "repeatability-analysable Exp-A sessions (N=71)", tiling,
        )
        for tiling in range(3)
    ),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from_config(config_path: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise QualityErrorDiagnosticError(f"{name} must be a non-empty path string")
    path = Path(value)
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def load_quality_error_config(path: str | Path) -> QualityErrorConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "quality_error_10ghz_v1":
        raise QualityErrorDiagnosticError("wrong or missing quality-error schema_version")
    if set(raw) != {"schema_version", "sources", "outputs", "bootstrap", "strict"}:
        raise QualityErrorDiagnosticError("quality-error config has missing or unknown fields")
    sources, outputs, bootstrap = raw["sources"], raw["outputs"], raw["bootstrap"]
    if set(sources) != {
        "reference_manifest", "exp_a_sources", "quality_provenance",
        "session_quality", "wst_repeatability",
    }:
        raise QualityErrorDiagnosticError("sources declaration has missing or unknown fields")
    if set(outputs) != {"results", "figures"}:
        raise QualityErrorDiagnosticError("outputs declaration has missing or unknown fields")
    if set(bootstrap) != {
        "attempted_draws", "seed", "confidence_level", "maximum_invalid_fraction"
    }:
        raise QualityErrorDiagnosticError("bootstrap declaration has missing or unknown fields")
    if bootstrap["attempted_draws"] != 10_000:
        raise QualityErrorDiagnosticError("bootstrap.attempted_draws must be exactly 10000")
    if not isinstance(bootstrap["seed"], int):
        raise QualityErrorDiagnosticError("bootstrap.seed must be an integer")
    if bootstrap["confidence_level"] != 0.95 or bootstrap["maximum_invalid_fraction"] != 0.05:
        raise QualityErrorDiagnosticError("bootstrap confidence and invalid-fraction rules are frozen")
    if raw["strict"] is not True:
        raise QualityErrorDiagnosticError("production diagnostic must be strict")
    return QualityErrorConfig(
        reference_manifest=_resolve_from_config(path, sources["reference_manifest"], "reference_manifest"),
        exp_a_sources=_resolve_from_config(path, sources["exp_a_sources"], "exp_a_sources"),
        quality_provenance=_resolve_from_config(path, sources["quality_provenance"], "quality_provenance"),
        session_quality=_resolve_from_config(path, sources["session_quality"], "session_quality"),
        wst_repeatability=_resolve_from_config(path, sources["wst_repeatability"], "wst_repeatability"),
        output_results_dir=_resolve_from_config(path, outputs["results"], "outputs.results"),
        output_figures_dir=_resolve_from_config(path, outputs["figures"], "outputs.figures"),
        bootstrap_draws=10_000,
        bootstrap_seed=int(bootstrap["seed"]),
        confidence_level=0.95,
        maximum_invalid_fraction=0.05,
        strict=True,
    )


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise QualityErrorDiagnosticError(f"required source is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualityErrorDiagnosticError(f"JSON source is not an object: {path}")
    return value


def _reference_json_sha256(value: object) -> str:
    """Use the exact canonical JSON contract from ``reference_gate.json_sha256``."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_canonical_population(population: dict) -> None:
    """Recompute the three ordered population hashes before prediction mapping."""
    sessions = population.get("sessions", [])
    if not isinstance(sessions, list):
        raise QualityErrorDiagnosticError("canonical population sessions must be a list")
    session_key_rows = []
    target_rows = []
    frame_rows = []
    for row in sessions:
        try:
            subject = int(row["subject"])
            session_idx = int(row["session_idx"])
            session_name = row["session_name"]
            n_frames = int(row["n_frames"])
            frame_ids_sha256 = row["frame_ids_sha256"]
            delta_m_pct = float(row["delta_m_pct"])
        except (KeyError, TypeError, ValueError) as error:
            raise QualityErrorDiagnosticError("canonical population row is malformed") from error
        session_key_rows.append([subject, session_idx, session_name])
        target_rows.append([subject, session_idx, delta_m_pct])
        frame_rows.append([subject, session_idx, n_frames, frame_ids_sha256])
    checks = (
        ("session_keys_sha256", session_key_rows, EXPECTED_SESSION_KEYS_SHA256),
        ("targets_sha256", target_rows, EXPECTED_TARGETS_SHA256),
        ("frame_population_sha256", frame_rows, EXPECTED_FRAME_POPULATION_SHA256),
    )
    for field, rows, pinned_hash in checks:
        recorded_hash = population.get(field)
        recomputed_hash = _reference_json_sha256(rows)
        if recorded_hash != pinned_hash or recomputed_hash != pinned_hash:
            raise QualityErrorDiagnosticError(
                f"canonical population {field} does not match its pinned ordered relation"
            )


def authenticate_exp_a_sources(reference: dict, sources: dict) -> dict:
    """Fail closed unless the exact authoritative 10 GHz source chain is approved."""
    if reference.get("schema_version") != EXPECTED_REFERENCE_SCHEMA:
        raise QualityErrorDiagnosticError("unexpected reference manifest schema_version")
    if sources.get("schema_version") != EXPECTED_SOURCES_SCHEMA:
        raise QualityErrorDiagnosticError("unexpected Exp-A sources schema_version")
    if reference.get("reference_grade") != "authoritative":
        raise QualityErrorDiagnosticError("reference manifest is not authoritative")
    if sources.get("status") != "approved" or sources.get("reference_manifest_grade") != "authoritative":
        raise QualityErrorDiagnosticError("Exp-A source ledger is not approved/authoritative")
    if "10ghz" not in sources.get("bands_approved", []):
        raise QualityErrorDiagnosticError("10ghz is not in bands_approved")
    reference_band = reference.get("bands", {}).get("10ghz")
    source_band = sources.get("bands", {}).get("10ghz")
    if not isinstance(reference_band, dict) or not isinstance(source_band, dict):
        raise QualityErrorDiagnosticError("10 GHz band is missing from source records")
    if reference_band.get("reference_grade") != "authoritative" or source_band.get("status") != "approved":
        raise QualityErrorDiagnosticError("10 GHz reference/source grade is not authoritative/approved")
    final_run = source_band.get("final_run", {})
    if Path(str(final_run.get("path", ""))).name != EXPECTED_FINAL_RUN:
        raise QualityErrorDiagnosticError("unexpected final 10 GHz run name")
    if final_run.get("commit") != EXPECTED_FINAL_COMMIT:
        raise QualityErrorDiagnosticError("unexpected final 10 GHz commit")
    comparisons = source_band.get("comparisons", [])
    evidence_classes = [row.get("evidence_class") for row in comparisons]
    if (
        len(evidence_classes) != len(EXPECTED_EVIDENCE_CLASSES)
        or set(evidence_classes) != set(EXPECTED_EVIDENCE_CLASSES)
        or any(evidence_classes.count(name) != 1 for name in EXPECTED_EVIDENCE_CLASSES)
    ):
        raise QualityErrorDiagnosticError("comparison ledger must contain each exact evidence class once")
    if any(row.get("status") != "match" for row in comparisons):
        raise QualityErrorDiagnosticError("not every 10 GHz source comparison is an exact match")
    prediction_rows = [row for row in comparisons if row.get("evidence_class") == "predictions"]
    if len(prediction_rows) != 1:
        raise QualityErrorDiagnosticError("prediction comparison must occur exactly once")
    prediction_comparison = prediction_rows[0]
    if prediction_comparison.get("max_abs_pred_delta") != 0.0 or prediction_comparison.get("byte_identical") is not True:
        raise QualityErrorDiagnosticError("prediction comparison is not exactly byte-identical with zero delta")

    expected_hashes = {
        reference_band.get("predictions_sha256"),
        reference_band.get("run", {}).get("artifacts", {}).get("predictions", {}).get("sha256"),
        source_band.get("final_run", {}).get("artifacts", {}).get("predictions", {}).get("sha256"),
        source_band.get("reference_run", {}).get("artifacts", {}).get("predictions", {}).get("sha256"),
    }
    if expected_hashes != {EXPECTED_PREDICTION_SHA256}:
        raise QualityErrorDiagnosticError("prediction hashes do not agree with the canonical hash")
    if OBSOLETE_PREDICTION_SHA256 in expected_hashes:
        raise QualityErrorDiagnosticError("obsolete 4bd212 prediction artifact was selected")
    config_hashes = {
        reference_band.get("config_sha256"), source_band.get("final_config_sha256"),
        source_band.get("reference_config_sha256"),
    }
    if len(config_hashes) != 1 or None in config_hashes:
        raise QualityErrorDiagnosticError("manifest/source/config hashes disagree")
    if source_band.get("mismatched_evidence_classes") != []:
        raise QualityErrorDiagnosticError("source ledger records mismatched evidence classes")
    validate_canonical_population(reference_band.get("population", {}))
    return reference_band


def reconstruct_prediction_bytes(reference_band: dict) -> bytes:
    """Reproduce ``exp_a.write_exp_a_reports`` CSV serialization from manifest values."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["subject", "seed", "y_true", "y_pred"])
    for fold in reference_band.get("selected_folds", []):
        predictions = fold.get("predictions", {})
        y_true = predictions.get("y_true", [])
        seeds = predictions.get("seeds", [])
        by_seed = predictions.get("y_pred_by_seed", {})
        for seed in seeds:
            y_pred = by_seed.get(str(seed))
            if not isinstance(y_pred, list) or len(y_pred) != len(y_true):
                raise QualityErrorDiagnosticError("prediction seed length disagrees with y_true")
            for truth, prediction in zip(y_true, y_pred):
                writer.writerow([fold.get("test_subject"), seed, truth, prediction])
    payload = stream.getvalue().encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if digest == OBSOLETE_PREDICTION_SHA256:
        raise QualityErrorDiagnosticError("reconstruction selected the obsolete 4bd212 predictions")
    if digest != EXPECTED_PREDICTION_SHA256:
        raise QualityErrorDiagnosticError(
            f"reconstructed predictions hash {digest} does not match {EXPECTED_PREDICTION_SHA256}"
        )
    return payload


def build_session_prediction_table(reference_band: dict) -> pd.DataFrame:
    """Map every fold prediction to the manifest's canonical subject-session order."""
    population = reference_band.get("population", {})
    validate_canonical_population(population)
    sessions = population.get("sessions", [])
    if population.get("n_sessions") != 73 or population.get("n_subjects") != 16 or len(sessions) != 73:
        raise QualityErrorDiagnosticError("canonical population is not 16 subjects / 73 sessions")
    keys = [(row.get("subject"), row.get("session_idx")) for row in sessions]
    if len(set(keys)) != 73:
        raise QualityErrorDiagnosticError("canonical population has missing or duplicate keys")
    folds = reference_band.get("selected_folds", [])
    if len(folds) != 16 or len({fold.get("test_subject") for fold in folds}) != 16:
        raise QualityErrorDiagnosticError("selected folds are missing or duplicated")

    output = []
    for fold in folds:
        subject = int(fold["test_subject"])
        subject_sessions = [row for row in sessions if int(row["subject"]) == subject]
        predictions = fold["predictions"]
        truths = predictions["y_true"]
        seeds = predictions["seeds"]
        if predictions.get("n_sessions") != len(subject_sessions) or len(truths) != len(subject_sessions):
            raise QualityErrorDiagnosticError(f"subject {subject} fold/session count mismatch")
        population_truths = [float(row["delta_m_pct"]) for row in subject_sessions]
        if not np.array_equal(np.asarray(population_truths), np.asarray(truths, dtype=float)):
            raise QualityErrorDiagnosticError(f"subject {subject} y_true is not elementwise population-equal")
        seed_predictions = np.asarray(
            [predictions["y_pred_by_seed"][str(seed)] for seed in seeds], dtype=np.float64
        )
        if seed_predictions.shape != (len(seeds), len(subject_sessions)):
            raise QualityErrorDiagnosticError(f"subject {subject} realized seed shape mismatch")
        if not np.isfinite(seed_predictions).all() or not np.isfinite(truths).all():
            raise QualityErrorDiagnosticError("canonical predictions contain NaN or Inf")
        for index, population_row in enumerate(subject_sessions):
            per_seed = seed_predictions[:, index]
            truth = float(truths[index])
            mean_prediction = float(per_seed.mean())
            output.append(
                {
                    "subject": subject,
                    "session_idx": int(population_row["session_idx"]),
                    "session_name": population_row["session_name"],
                    "delta_m_pct": truth,
                    "n_seeds": int(len(seeds)),
                    "mean_prediction_delta_m_pct": mean_prediction,
                    "signed_residual_pct_points": mean_prediction - truth,
                    "prediction_seed_sd_pct_points": (
                        float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else 0.0
                    ),
                    # Repeated seeds describe model uncertainty for this one session.  The
                    # session error is their mean absolute error, not an extra 149-row sample.
                    "mean_absolute_seed_error_pct_points": float(np.mean(np.abs(per_seed - truth))),
                    "absolute_ensemble_error_pct_points": abs(mean_prediction - truth),
                }
            )
    frame = pd.DataFrame(output)
    if len(frame) != 73 or frame.duplicated(["subject", "session_idx"]).any():
        raise QualityErrorDiagnosticError("prediction mapping did not produce 73 unique session keys")
    if set(zip(frame.subject, frame.session_idx)) != set(keys):
        raise QualityErrorDiagnosticError("prediction mapping changed the canonical population")
    return frame.sort_values(["subject", "session_idx"]).reset_index(drop=True)


def verify_and_load_quality(config: QualityErrorConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    provenance = _read_json(config.quality_provenance)
    quality_git = provenance.get("git", {})
    quality_commit = str(quality_git.get("commit", ""))
    if (
        quality_git.get("dirty") is not False
        or len(quality_commit) != 40
        or any(character not in "0123456789abcdef" for character in quality_commit.lower())
    ):
        raise QualityErrorDiagnosticError("quality artifacts do not have clean committed provenance")
    census = provenance.get("census", {})
    if (
        census.get("n_subjects") != 16
        or census.get("n_sessions") != 80
        or census.get("n_eligible_sessions") != 73
        or census.get("n_repeatability_analysable") != 71
    ):
        raise QualityErrorDiagnosticError("quality provenance census is not 16 / 80 / 73 / 71")
    outputs = provenance.get("outputs", {})
    expected = {
        "session_quality": (config.session_quality, "session_quality_all_80.csv"),
        "repeatability": (config.wst_repeatability, "wst_block_repeatability.csv"),
    }
    for artifact, (path, filename) in expected.items():
        record = outputs.get(artifact, {})
        if Path(str(record.get("path", ""))).name != filename:
            raise QualityErrorDiagnosticError(f"quality provenance does not name {filename}")
        if sha256_file(path) != record.get("sha256"):
            raise QualityErrorDiagnosticError(f"quality artifact hash mismatch: {filename}")
    cards = pd.read_csv(config.session_quality)
    repeatability = pd.read_csv(config.wst_repeatability)
    if len(cards) != 80 or cards.duplicated(["subject", "session_idx"]).any():
        raise QualityErrorDiagnosticError("quality card is not the exact 80-cell census")
    status_counts = cards["audit_status"].value_counts().to_dict()
    expected_counts = {
        "REPEATABILITY_ANALYSABLE": 71,
        "REVIEW_BLOCK_COVERAGE": 2,
        "INELIGIBLE_EXISTING_QC": 7,
    }
    if status_counts != expected_counts or int(cards["eligible_existing_qc"].sum()) != 73:
        raise QualityErrorDiagnosticError("quality status census is not 71 + 2 + 7")
    return cards, repeatability, provenance


def metric_catalog() -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        rows.append(
            {
                "metric_id": metric.metric_id,
                "source_table": metric.source_table,
                "source_key": metric.source_key,
                "units": metric.units,
                "worse_orientation": metric.orientation,
                "analytic_population": metric.analytic_population,
                "diagnostic_channel": "mag" if metric.tiling_idx is not None else "",
                "tiling_idx": metric.tiling_idx,
                "scattering_order": 2 if metric.tiling_idx is not None else np.nan,
                "view": "within_path_shape" if metric.tiling_idx is not None else "",
                "role": "exploratory association; no exclusion rule",
            }
        )
    return pd.DataFrame(rows)


def assemble_analysis_table(
    predictions: pd.DataFrame, cards: pd.DataFrame, repeatability: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_keys = set(zip(predictions.subject, predictions.session_idx))
    eligible = cards.loc[cards["eligible_existing_qc"].astype(bool)].copy()
    eligible_keys = set(zip(eligible.subject, eligible.session_idx))
    if prediction_keys != eligible_keys or len(eligible) != 73:
        raise QualityErrorDiagnosticError("Exp-A and quality eligible keys are not exactly equal")
    card_columns = [
        "subject", "session_idx", "session_name", "audit_status", "pass_fraction",
        "minimum_block_n_pass", "in_band_ratio_p10_margin", "peak_bin_iqr",
    ]
    table = predictions.merge(
        eligible[card_columns], on=["subject", "session_idx", "session_name"],
        how="left", validate="one_to_one",
    )
    table["one_minus_pass_fraction"] = 1.0 - table["pass_fraction"]
    table["twenty_minus_minimum_block_n_pass"] = 20.0 - table["minimum_block_n_pass"]
    table["negative_in_band_ratio_p10_margin"] = -table["in_band_ratio_p10_margin"]

    for tiling in range(3):
        cell = repeatability.loc[
            (repeatability["diagnostic_channel"] == "mag")
            & (repeatability["tiling_idx"] == tiling)
            & (repeatability["scattering_order"] == 2)
            & (repeatability["view"] == "within_path_shape")
        ]
        if len(cell) != 80 or cell.duplicated(["subject", "session_idx"]).any():
            raise QualityErrorDiagnosticError(f"WST tiling {tiling} cell is not exactly the 80-card census")
        column = f"wst_tiling_{tiling}_block_distance_maximum"
        values = cell[["subject", "session_idx", "block_to_session_distance_maximum"]].rename(
            columns={"block_to_session_distance_maximum": column}
        )
        table = table.merge(values, on=["subject", "session_idx"], how="left", validate="one_to_one")
        missing = set(
            zip(table.loc[~np.isfinite(table[column]), "subject"], table.loc[~np.isfinite(table[column]), "session_idx"])
        )
        review = set(
            zip(table.loc[table.audit_status == "REVIEW_BLOCK_COVERAGE", "subject"],
                table.loc[table.audit_status == "REVIEW_BLOCK_COVERAGE", "session_idx"])
        )
        if len(table) != 73 or int(np.isfinite(table[column]).sum()) != 71 or missing != review:
            raise QualityErrorDiagnosticError(f"WST tiling {tiling} population is not exact N=71 review-complete-case")

    flow = pd.DataFrame(
        [
            {"stage": "raw quality-card census", "n_sessions": 80, "n_subjects": 16, "reason": "all acquired sessions"},
            {"stage": "Exp-A eligible joined sessions", "n_sessions": 73, "n_subjects": 16, "reason": "frozen existing QC"},
            {"stage": "card-metric analytic population", "n_sessions": 73, "n_subjects": 16, "reason": "four finite card metrics"},
            {"stage": "WST-metric analytic population", "n_sessions": 71, "n_subjects": int(table.loc[table.audit_status == "REPEATABILITY_ANALYSABLE", "subject"].nunique()), "reason": "two review-block-coverage sessions unavailable; no imputation"},
        ]
    )
    return table.sort_values(["subject", "session_idx"]).reset_index(drop=True), flow


def _fixed_effect_design(subjects: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    subjects = np.asarray(subjects)
    sessions = np.asarray(sessions)
    columns = [np.ones(len(subjects), dtype=np.float64)]
    for level in sorted(np.unique(subjects))[1:]:
        columns.append((subjects == level).astype(np.float64))
    for level in sorted(np.unique(sessions))[1:]:
        columns.append((sessions == level).astype(np.float64))
    return np.column_stack(columns)


def _require_full_rank(design: np.ndarray, name: str) -> None:
    if not np.isfinite(design).all():
        raise QualityErrorDiagnosticError(f"{name} contains NaN or Inf")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise QualityErrorDiagnosticError(f"{name} is rank-deficient")


def fit_fixed_effect_association(
    frame: pd.DataFrame,
    metric_column: str,
    *,
    metric_mean: float | None = None,
    metric_sd: float | None = None,
    subject_column: str = "subject",
) -> dict:
    required = [
        subject_column, "session_idx", metric_column,
        "mean_absolute_seed_error_pct_points",
    ]
    if any(column not in frame for column in required):
        raise QualityErrorDiagnosticError("association input is missing required columns")
    values = frame[required].to_numpy()
    if not np.isfinite(values.astype(float)).all():
        raise QualityErrorDiagnosticError("association input contains NaN or Inf")
    metric = frame[metric_column].to_numpy(dtype=np.float64)
    outcome = frame["mean_absolute_seed_error_pct_points"].to_numpy(dtype=np.float64)
    if metric_mean is None:
        metric_mean = float(np.mean(metric))
    if metric_sd is None:
        metric_sd = float(np.std(metric, ddof=1))
    if not math.isfinite(metric_mean) or not math.isfinite(metric_sd) or metric_sd <= 0.0:
        raise QualityErrorDiagnosticError("metric mean/SD is nonfinite or sample SD is zero")
    standardized_worse = (metric - metric_mean) / metric_sd
    fe_design = _fixed_effect_design(
        frame[subject_column].to_numpy(), frame["session_idx"].to_numpy()
    )
    full_design = np.column_stack([fe_design, standardized_worse])
    _require_full_rank(fe_design, "fixed-effect design")
    _require_full_rank(full_design, "full association design")
    coefficients = np.linalg.lstsq(full_design, outcome, rcond=None)[0]

    metric_residual = standardized_worse - fe_design @ np.linalg.lstsq(
        fe_design, standardized_worse, rcond=None
    )[0]
    outcome_residual = outcome - fe_design @ np.linalg.lstsq(fe_design, outcome, rcond=None)[0]
    if np.std(metric_residual, ddof=1) <= 0.0 or np.std(outcome_residual, ddof=1) <= 0.0:
        raise QualityErrorDiagnosticError("partial-correlation residual variance is zero")
    partial_correlation = float(np.corrcoef(metric_residual, outcome_residual)[0, 1])
    if not math.isfinite(partial_correlation):
        raise QualityErrorDiagnosticError("partial correlation is nonfinite")
    return {
        "coefficient_pct_points_per_sd_worse": float(coefficients[-1]),
        "partial_correlation": partial_correlation,
        "metric_mean_original": float(metric_mean),
        "metric_sd_original_ddof1": float(metric_sd),
        "n_sessions": int(len(frame)),
        "n_subjects": int(frame[subject_column].nunique()),
        "design_columns": int(full_design.shape[1]),
        "metric_residual": metric_residual,
        "outcome_residual": outcome_residual,
    }


def relabel_cluster_bootstrap_sample(
    frame: pd.DataFrame, sampled_subjects: np.ndarray
) -> pd.DataFrame:
    """Carry whole clusters and distinguish duplicate draws of one original subject."""
    pieces = []
    for bootstrap_cluster_id, original_subject in enumerate(sampled_subjects):
        rows = frame.loc[frame["subject"] == original_subject].copy()
        if rows.empty:
            raise QualityErrorDiagnosticError("bootstrap sampled an absent original subject")
        rows["bootstrap_cluster_id"] = bootstrap_cluster_id
        rows["original_subject"] = original_subject
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True)


def _bootstrap_coefficient_arrays(
    bootstrap_clusters: np.ndarray,
    sessions: np.ndarray,
    metric_values: np.ndarray,
    outcomes: np.ndarray,
    metric_mean: float,
    metric_sd: float,
) -> float:
    """Fit the bootstrap coefficient without recomputing unused correlations.

    Frisch-Waugh-Lovell residualization gives the same metric coefficient as the
    full OLS design.  Solving metric and outcome residuals together is materially
    faster for the required 70,000 refits and keeps every rank/variance guard.
    """
    metric = (metric_values - metric_mean) / metric_sd
    outcome = outcomes
    if not np.isfinite(metric).all() or not np.isfinite(outcome).all():
        raise QualityErrorDiagnosticError("bootstrap input contains NaN or Inf")
    fixed_effects = _fixed_effect_design(
        bootstrap_clusters, sessions
    )
    fitted_coefficients, _residual_sum, rank, _singular_values = np.linalg.lstsq(
        fixed_effects, np.column_stack([metric, outcome]), rcond=None
    )
    if rank != fixed_effects.shape[1]:
        raise QualityErrorDiagnosticError("bootstrap fixed-effect design is rank-deficient")
    residuals = np.column_stack([metric, outcome]) - fixed_effects @ fitted_coefficients
    metric_residual = residuals[:, 0]
    outcome_residual = residuals[:, 1]
    denominator = float(metric_residual @ metric_residual)
    if denominator <= np.finfo(np.float64).eps * max(1, len(metric_residual)):
        raise QualityErrorDiagnosticError("partial-correlation residual variance is zero")
    coefficient = float((metric_residual @ outcome_residual) / denominator)
    if not math.isfinite(coefficient):
        raise QualityErrorDiagnosticError("nonfinite coefficient")
    return coefficient


def cluster_bootstrap(
    frame: pd.DataFrame,
    metric_column: str,
    *,
    metric_mean: float,
    metric_sd: float,
    attempted_draws: int,
    seed: int,
    maximum_invalid_fraction: float,
    strict: bool,
    prescribed_draws: list[np.ndarray] | None = None,
) -> dict:
    original_subjects = np.asarray(sorted(frame["subject"].unique()))
    if len(original_subjects) != 16:
        raise QualityErrorDiagnosticError("cluster bootstrap requires the original 16 subjects")
    rng = np.random.default_rng(seed)
    if prescribed_draws is not None and len(prescribed_draws) != attempted_draws:
        raise QualityErrorDiagnosticError("prescribed bootstrap draws must equal attempted_draws")
    coefficients = []
    invalid_reasons: dict[str, int] = {}
    # Cache each whole subject as arrays.  Pandas concatenation inside 70,000 draws
    # would dominate runtime without changing the statistical procedure.
    subject_arrays = {}
    for subject in original_subjects:
        rows = frame.loc[frame["subject"] == subject]
        subject_arrays[subject] = (
            rows["session_idx"].to_numpy(dtype=np.int64),
            rows[metric_column].to_numpy(dtype=np.float64),
            rows["mean_absolute_seed_error_pct_points"].to_numpy(dtype=np.float64),
        )
    for draw_index in range(attempted_draws):
        sampled = (
            np.asarray(prescribed_draws[draw_index])
            if prescribed_draws is not None
            else rng.choice(original_subjects, size=len(original_subjects), replace=True)
        )
        if sampled.shape != (len(original_subjects),):
            raise QualityErrorDiagnosticError("each prescribed bootstrap draw must contain 16 subjects")
        if any(subject not in subject_arrays for subject in sampled):
            raise QualityErrorDiagnosticError("prescribed bootstrap draw names an absent subject")
        sessions = np.concatenate([subject_arrays[subject][0] for subject in sampled])
        metric_values = np.concatenate([subject_arrays[subject][1] for subject in sampled])
        outcomes = np.concatenate([subject_arrays[subject][2] for subject in sampled])
        bootstrap_clusters = np.concatenate(
            [np.full(len(subject_arrays[subject][0]), cluster_id, dtype=np.int64)
             for cluster_id, subject in enumerate(sampled)]
        )
        try:
            coefficient = _bootstrap_coefficient_arrays(
                bootstrap_clusters, sessions, metric_values, outcomes, metric_mean, metric_sd
            )
            coefficients.append(coefficient)
        except QualityErrorDiagnosticError as error:
            text = str(error)
            if "rank-deficient" in text:
                reason = "rank_deficient"
            elif "residual variance is zero" in text:
                reason = "zero_residual_variance"
            else:
                reason = "nonfinite_or_other_invalid"
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    n_invalid = attempted_draws - len(coefficients)
    invalid_fraction = n_invalid / attempted_draws
    unreliable = invalid_fraction > maximum_invalid_fraction
    if not coefficients:
        raise QualityErrorDiagnosticError("all cluster-bootstrap draws were invalid")
    if unreliable and strict:
        raise QualityErrorDiagnosticError(
            f"bootstrap invalid fraction {invalid_fraction:.3%} exceeds {maximum_invalid_fraction:.1%}"
        )
    low, high = np.quantile(coefficients, [0.025, 0.975])
    return {
        "bootstrap_attempted": int(attempted_draws),
        "bootstrap_valid": int(len(coefficients)),
        "bootstrap_invalid": int(n_invalid),
        "bootstrap_invalid_fraction": float(invalid_fraction),
        "bootstrap_invalid_reasons": json.dumps(invalid_reasons, sort_keys=True, separators=(",", ":")),
        "bootstrap_ci95_low": float(low),
        "bootstrap_ci95_high": float(high),
        "bootstrap_unreliable": bool(unreliable),
    }


def analyze_metrics(
    table: pd.DataFrame, config: QualityErrorConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    summaries, influence_rows = [], []
    residuals = {}
    for metric in METRICS:
        analytic = table.loc[np.isfinite(table[metric.metric_id])].copy()
        expected_n = 71 if metric.tiling_idx is not None else 73
        if len(analytic) != expected_n:
            raise QualityErrorDiagnosticError(f"{metric.metric_id} has N={len(analytic)}, expected {expected_n}")
        fit = fit_fixed_effect_association(analytic, metric.metric_id)
        bootstrap = cluster_bootstrap(
            analytic,
            metric.metric_id,
            metric_mean=fit["metric_mean_original"],
            metric_sd=fit["metric_sd_original_ddof1"],
            attempted_draws=config.bootstrap_draws,
            seed=config.bootstrap_seed,
            maximum_invalid_fraction=config.maximum_invalid_fraction,
            strict=config.strict,
        )
        unique_values = int(analytic[metric.metric_id].nunique(dropna=True))
        best = float(analytic[metric.metric_id].min())
        non_best = int((analytic[metric.metric_id] > best).sum())
        metric_influence = []
        for subject in sorted(table["subject"].unique()):
            reduced = analytic.loc[analytic["subject"] != subject]
            reduced_fit = fit_fixed_effect_association(
                reduced,
                metric.metric_id,
                metric_mean=fit["metric_mean_original"],
                metric_sd=fit["metric_sd_original_ddof1"],
            )
            coefficient = reduced_fit["coefficient_pct_points_per_sd_worse"]
            metric_influence.append(coefficient)
            influence_rows.append(
                {
                    "metric_id": metric.metric_id,
                    "left_out_original_subject": int(subject),
                    "n_sessions": int(len(reduced)),
                    "coefficient_pct_points_per_sd_worse": coefficient,
                }
            )
        summaries.append(
            {
                "metric_id": metric.metric_id,
                "coefficient_pct_points_per_sd_worse": fit["coefficient_pct_points_per_sd_worse"],
                "partial_correlation_same_fe": fit["partial_correlation"],
                "metric_mean_original": fit["metric_mean_original"],
                "metric_sd_original_ddof1": fit["metric_sd_original_ddof1"],
                "n_finite": int(len(analytic)),
                "n_subjects": int(analytic["subject"].nunique()),
                "n_unique_values": unique_values,
                "n_non_best_values": non_best,
                "loso_influence_min": float(min(metric_influence)),
                "loso_influence_max": float(max(metric_influence)),
                "interpretation": "exploratory association; no exclusion rule",
                **bootstrap,
            }
        )
        residuals[metric.metric_id] = (fit["metric_residual"], fit["outcome_residual"])
    return pd.DataFrame(summaries), pd.DataFrame(influence_rows), residuals


def capture_clean_git_state(repo_root: Path) -> dict:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, text=True, capture_output=True, check=True
        )
        return result.stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        branch = git("branch", "--show-current")
        dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    except (OSError, subprocess.CalledProcessError) as error:
        raise QualityErrorDiagnosticError("could not capture Git provenance") from error
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit.lower()):
        raise QualityErrorDiagnosticError("Git provenance requires a valid 40-character commit")
    if dirty:
        raise QualityErrorDiagnosticError("production diagnostic requires a clean source tree")
    return {"commit": commit, "branch": branch, "dirty": False}


def ensure_fresh_output_roots(config: QualityErrorConfig) -> None:
    if config.output_results_dir.exists() or config.output_figures_dir.exists():
        raise QualityErrorDiagnosticError("quality-error output roots must be fresh")


def require_production_output_roots(config: QualityErrorConfig, repo_root: Path) -> None:
    expected_results = (repo_root / "results" / "quality_error_10ghz").resolve()
    expected_figures = (repo_root / "figures" / "quality_error_10ghz").resolve()
    if config.output_results_dir.resolve() != expected_results:
        raise QualityErrorDiagnosticError("production results root must be results/quality_error_10ghz")
    if config.output_figures_dir.resolve() != expected_figures:
        raise QualityErrorDiagnosticError("production figures root must be figures/quality_error_10ghz")


def snapshot_existing_files(config: QualityErrorConfig) -> dict[str, dict]:
    repo_root = config.output_results_dir.parents[1]
    records = {}
    for root_name in ("results", "figures"):
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if config.output_results_dir in path.parents or config.output_figures_dir in path.parents:
                continue
            rel_path = path.relative_to(repo_root).as_posix()
            records[rel_path] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return records


def verify_existing_files(snapshot: dict[str, dict], config: QualityErrorConfig) -> None:
    after = snapshot_existing_files(config)
    if after != snapshot:
        raise QualityErrorDiagnosticError("pre-existing results/figures changed during diagnostic")


def write_csv_deterministic(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")
    return path


def write_json_deterministic(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def plot_association_forest(summary: pd.DataFrame, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = summary.iloc[::-1]
    y = np.arange(len(ordered))
    estimate = ordered["coefficient_pct_points_per_sd_worse"].to_numpy()
    low = ordered["bootstrap_ci95_low"].to_numpy()
    high = ordered["bootstrap_ci95_high"].to_numpy()
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.errorbar(estimate, y, xerr=[estimate - low, high - estimate], fmt="o", capsize=3)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(y, ordered["metric_id"])
    axis.set_xlabel("MAE percentage-body-mass points per 1-SD worse quality")
    axis.set_title("Exploratory association — no exclusion rule")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, metadata={"Software": "dehydration_radar"})
    plt.close(figure)
    return path


def plot_residual_panels(
    residuals: dict[str, tuple[np.ndarray, np.ndarray]], path: Path
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(10, 9))
    for axis, metric in zip(axes.flat, METRICS):
        x, y = residuals[metric.metric_id]
        axis.scatter(x, y, s=15, alpha=0.7)
        axis.axhline(0.0, color="0.7", linewidth=0.8)
        axis.axvline(0.0, color="0.7", linewidth=0.8)
        axis.set_title(metric.metric_id, fontsize=8)
        axis.set_xlabel("quality residual")
        axis.set_ylabel("MAE residual")
    for axis in axes.flat[len(METRICS):]:
        axis.axis("off")
    figure.suptitle("Exploratory association — no exclusion rule")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, metadata={"Software": "dehydration_radar"})
    plt.close(figure)
    return path


def plot_population_flow(flow: pd.DataFrame, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    bars = axis.barh(np.arange(len(flow)), flow["n_sessions"], color="#4C78A8")
    axis.set_yticks(np.arange(len(flow)), flow["stage"])
    axis.invert_yaxis()
    axis.bar_label(bars)
    axis.set_xlabel("subject-session cells")
    axis.set_title("Exploratory association population — no exclusion rule")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, metadata={"Software": "dehydration_radar"})
    plt.close(figure)
    return path


def provenance_record(
    config: QualityErrorConfig,
    git_state: dict,
    source_hashes: dict,
    output_paths: dict[str, Path],
    summary: pd.DataFrame,
) -> dict:
    return {
        "schema_version": "quality_error_10ghz_provenance_v1",
        "analysis_role": "exploratory_quality_vs_frozen_loso_error_no_exclusion_rule",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state,
        "prediction_contract": {
            "source": "reference_exp_a_manifest.json bands.10ghz.selected_folds[*].predictions",
            "reconstructed_sha256": EXPECTED_PREDICTION_SHA256,
            "obsolete_sha256_rejected": OBSOLETE_PREDICTION_SHA256,
            "analysis_unit": "subject-session; repeated seeds summarized within session",
        },
        "sources": source_hashes,
        "bootstrap": {
            "attempted_draws_per_metric": config.bootstrap_draws,
            "seed": config.bootstrap_seed,
            "confidence_level": config.confidence_level,
            "maximum_invalid_fraction": config.maximum_invalid_fraction,
        },
        "population": {"card_metric_n": 73, "wst_metric_n": 71, "n_subjects": 16},
        "metrics": summary["metric_id"].tolist(),
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(output_paths.items())
        },
        "warning": "Exploratory association only; no threshold, exclusion, weighting, retraining, or causal claim.",
    }
