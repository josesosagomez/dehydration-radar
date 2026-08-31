"""Exploratory quality-aware training sensitivity for the 10 GHz cohort.

This experiment asks one narrow, post-hoc question: does either removing the five
sessions whose radar-only in-band margin is negative, or appending that raw margin as
one feature, improve a fixed learner's held-out predictions?

The three treatments always evaluate identical, unfiltered test rows.  Filtering is
training-only and happens before every fitted quantity, including a tuned WST epsilon.
LOSO is the primary protocol.  A session-level, subject-overlapping split is written to
a separate diagnostic directory and must never be interpreted as new-subject evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from ..features import store as store_mod
from ..features.pooling import aggregate_session, pool_stats_batch
from ..features.protocol_freeze import protocol_freeze_guard
from ..features.wst import apply_order_log
from ..models import regressors
from . import exp_a, exp_c, metrics as M
from .harness import Candidate, tuned_epsilons
from .selection import (
    CandidateScore,
    OrdinalCandidateScore,
    SIMPLICITY_RANK,
    select_candidate,
    select_candidate_ordinal,
)


SCHEMA_VERSION = "quality_training_sensitivity_10ghz_v1"
QUALITY_COLUMN = "in_band_ratio_p10_margin"
QUALITY_THRESHOLD = 0.0
TREATMENTS = ("baseline", "filter_negative_margin", "append_margin_feature")
PROTOCOLS = ("loso", "subject_overlap_session_cv")
TASKS = ("regression", "ordinal_classification")
EXPECTED_N_SESSIONS = 73
EXPECTED_N_SUBJECTS = 16
EXPECTED_NEGATIVE_KEYS = frozenset({(4, 2), (8, 0), (8, 2), (12, 0), (16, 3)})
EXPECTED_SPLIT_SEED = 20260829
EXPECTED_MODEL_SEEDS = (1, 2, 3, 4, 5)
EXPECTED_QUALITY_COMMIT = "bc5832b582e2d705c97bf7f445ba48fd38a4b2d3"
EXPECTED_QUALITY_CSV_SHA256 = "1f75a61601ac7e2b9d7debad5ad8a67afd418c66b7652365622152f40df085da"
BASELINE_REPLAY_TOLERANCE = 1e-10
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_OUTPUT_ROOT = (REPO_ROOT / "results" / "quality_training_sensitivity_10ghz").resolve()


class QualityTrainingError(ValueError):
    """Raised when source lineage, row alignment, or a split contract is broken."""


@dataclass(frozen=True)
class QualityTrainingConfig:
    base_configs: tuple[Path, ...]
    reference_manifest: Path
    exp_a_sources: Path
    quality_provenance: Path
    session_quality: Path
    results_dir: Path
    n_outer_splits: int
    n_inner_splits: int
    split_seed: int
    treatments: tuple[str, ...]
    strict: bool


@dataclass(frozen=True)
class RowSplit:
    split_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class FeatureMatrix:
    values: np.ndarray
    tuned_epsilon: dict[int, float] | None


@dataclass(frozen=True)
class SelectedCandidate:
    candidate: Candidate
    selection_source: str


def _resolve(config_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise QualityTrainingError(f"{field} must be a non-empty path string")
    path = Path(value)
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def require_full_model_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    realized = tuple(int(seed) for seed in seeds)
    if realized != EXPECTED_MODEL_SEEDS:
        raise QualityTrainingError(
            f"full run seed_set must be exactly {EXPECTED_MODEL_SEEDS}, got {realized}"
        )
    return realized


def load_quality_training_config(path: str | Path) -> QualityTrainingConfig:
    """Load the small, strict experiment config without extending the frozen global schema."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_top = {"schema_version", "base_configs", "sources", "outputs", "split", "treatments", "strict"}
    if not isinstance(raw, dict) or set(raw) != expected_top:
        raise QualityTrainingError("quality-training config has missing or unknown top-level fields")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise QualityTrainingError(f"schema_version must be {SCHEMA_VERSION!r}")
    if raw["strict"] is not True:
        raise QualityTrainingError("strict must be true")
    if not isinstance(raw["base_configs"], list) or not raw["base_configs"]:
        raise QualityTrainingError("base_configs must be a non-empty list")
    sources = raw["sources"]
    if set(sources) != {"reference_manifest", "exp_a_sources", "quality_provenance", "session_quality"}:
        raise QualityTrainingError("sources has missing or unknown fields")
    # Prediction-error tables and ground-truth-derived quality scores are deliberately absent.
    forbidden_fragments = ("quality_error", "prediction", "residual", "absolute_error")
    for name in ("quality_provenance", "session_quality"):
        source = str(sources[name]).lower()
        if any(fragment in source for fragment in forbidden_fragments):
            raise QualityTrainingError(f"{name} points to a forbidden outcome/error source")
    if set(raw["outputs"]) != {"results"}:
        raise QualityTrainingError("outputs must contain only results")
    split = raw["split"]
    if set(split) != {"n_outer_splits", "n_inner_splits", "seed"}:
        raise QualityTrainingError("split has missing or unknown fields")
    if split["n_outer_splits"] != 5 or split["n_inner_splits"] != 4:
        raise QualityTrainingError("the diagnostic is fixed at 5 outer and 4 inner session folds")
    if split["seed"] != EXPECTED_SPLIT_SEED:
        raise QualityTrainingError(f"split.seed must be exactly {EXPECTED_SPLIT_SEED}")
    treatments = tuple(raw["treatments"])
    if treatments != TREATMENTS:
        raise QualityTrainingError(f"treatments must be exactly {TREATMENTS}")
    resolved_output_root = _resolve(path, raw["outputs"]["results"], "outputs.results")
    if resolved_output_root != EXPECTED_OUTPUT_ROOT:
        raise QualityTrainingError(
            f"outputs.results must resolve exactly to {EXPECTED_OUTPUT_ROOT}"
        )
    return QualityTrainingConfig(
        base_configs=tuple(_resolve(path, value, "base_configs") for value in raw["base_configs"]),
        reference_manifest=_resolve(path, sources["reference_manifest"], "reference_manifest"),
        exp_a_sources=_resolve(path, sources["exp_a_sources"], "exp_a_sources"),
        quality_provenance=_resolve(path, sources["quality_provenance"], "quality_provenance"),
        session_quality=_resolve(path, sources["session_quality"], "session_quality"),
        results_dir=resolved_output_root,
        n_outer_splits=5,
        n_inner_splits=4,
        split_seed=EXPECTED_SPLIT_SEED,
        treatments=treatments,
        strict=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise QualityTrainingError(f"required source is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualityTrainingError(f"JSON source is not an object: {path}")
    return value


def authenticate_reference(config: QualityTrainingConfig) -> dict:
    """Authenticate the authoritative Exp-A relation used only for population and selection."""
    # Reuse the strict source gate already used by the preceding quality/error diagnostic.
    # That gate authenticates the ordered 73-row population rather than trusting a stored hash.
    from ..quality.error_diagnostic import authenticate_exp_a_sources

    reference = _read_json(config.reference_manifest)
    sources = _read_json(config.exp_a_sources)
    return authenticate_exp_a_sources(reference, sources)


def load_quality_margin(config: QualityTrainingConfig, canonical_keys: Iterable[tuple[int, int]]) -> pd.DataFrame:
    """Authenticate and align the single target-free quality signal to the 73 session keys."""
    provenance = _read_json(config.quality_provenance)
    if provenance.get("schema_version") != "quality_10ghz_radar_provenance_v1":
        raise QualityTrainingError("unexpected quality provenance schema")
    if provenance.get("analysis_role") != "descriptive_radar_only_quality_audit":
        raise QualityTrainingError("quality source is not the radar-only descriptive audit")
    git = provenance.get("git", {})
    quality_commit = str(git.get("commit", ""))
    if (
        git.get("dirty") is not False
        or len(quality_commit) != 40
        or quality_commit != quality_commit.lower()
        or any(character not in "0123456789abcdef" for character in quality_commit)
        or quality_commit != EXPECTED_QUALITY_COMMIT
    ):
        raise QualityTrainingError("quality source is not the pinned clean approved audit commit")
    census = provenance.get("census", {})
    if (census.get("n_sessions"), census.get("n_eligible_sessions"), census.get("n_subjects")) != (80, 73, 16):
        raise QualityTrainingError("quality provenance census is not 80/73/16")
    recorded = (provenance.get("outputs") or {}).get("session_quality", {})
    actual_csv_hash = _sha256_file(config.session_quality)
    if recorded.get("sha256") != EXPECTED_QUALITY_CSV_SHA256 or actual_csv_hash != EXPECTED_QUALITY_CSV_SHA256:
        raise QualityTrainingError("session-quality CSV does not match its provenance hash")

    frame = pd.read_csv(config.session_quality)
    eligible = validate_quality_table(frame)
    keys = list(zip(eligible.subject, eligible.session_idx, strict=True))
    canonical = list(canonical_keys)
    if len(keys) != EXPECTED_N_SESSIONS or len(set(keys)) != EXPECTED_N_SESSIONS:
        raise QualityTrainingError("quality join is not exactly 73 unique eligible session keys")
    if keys != canonical:
        raise QualityTrainingError("quality rows do not exactly match the canonical ordered 73 keys")
    margins = eligible[QUALITY_COLUMN].to_numpy(dtype=float)
    negative_keys = frozenset(key for key, value in zip(keys, margins, strict=True) if value < QUALITY_THRESHOLD)
    if negative_keys != EXPECTED_NEGATIVE_KEYS:
        raise QualityTrainingError(
            f"negative-margin keys changed: expected {sorted(EXPECTED_NEGATIVE_KEYS)}, got {sorted(negative_keys)}"
        )
    return eligible


def validate_quality_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the authenticated CSV's scientific relation after its byte hash is checked."""
    required = {"subject", "session_idx", "session_name", "eligible_existing_qc", QUALITY_COLUMN}
    if not required.issubset(frame.columns):
        raise QualityTrainingError(f"session-quality CSV is missing {sorted(required - set(frame.columns))}")
    if len(frame) != 80 or frame.duplicated(["subject", "session_idx"]).any():
        raise QualityTrainingError("session-quality CSV must contain exactly 80 unique session rows")
    all_keys = set(zip(frame["subject"].astype(int), frame["session_idx"].astype(int), strict=True))
    expected_all_keys = {(subject, session) for subject in range(1, 17) for session in range(5)}
    if all_keys != expected_all_keys:
        raise QualityTrainingError("session-quality CSV does not cover the complete 16 x 5 census")
    status_counts = frame["audit_status"].value_counts().to_dict()
    if status_counts != {
        "REPEATABILITY_ANALYSABLE": 71,
        "REVIEW_BLOCK_COVERAGE": 2,
        "INELIGIBLE_EXISTING_QC": 7,
    }:
        raise QualityTrainingError(f"session-quality audit-status census changed: {status_counts}")
    raw_flags = frame["eligible_existing_qc"]
    if raw_flags.dtype == bool:
        eligible_flags = raw_flags.to_numpy(dtype=bool)
    else:
        normalized = raw_flags.astype(str).map({"True": True, "False": False})
        if normalized.isna().any():
            raise QualityTrainingError("quality eligibility contains a non-boolean value")
        eligible_flags = normalized.to_numpy(dtype=bool)
    expected_flags = frame["audit_status"].ne("INELIGIBLE_EXISTING_QC").to_numpy()
    if not np.array_equal(eligible_flags, expected_flags) or int(eligible_flags.sum()) != 73:
        raise QualityTrainingError("quality eligibility flags disagree with the audit statuses")
    p10 = pd.to_numeric(frame["in_band_ratio_p10"], errors="raise").to_numpy(dtype=float)
    all_margins = pd.to_numeric(frame[QUALITY_COLUMN], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(p10).all() or not np.isfinite(all_margins).all():
        raise QualityTrainingError("quality p10 or margin contains NaN or Inf")
    # CSV decimal parsing can round the two serialized columns on opposite sides of one
    # float64 ulp.  Four machine epsilons is still an exact-formula reconciliation bound,
    # not a scientific tolerance or a tunable threshold.
    if not np.allclose(all_margins, p10 - 0.3, rtol=0.0, atol=4 * np.finfo(float).eps):
        raise QualityTrainingError("quality margin is not exactly in_band_ratio_p10 - 0.3")

    eligible = frame.loc[eligible_flags, list(required)].copy()
    eligible["subject"] = eligible["subject"].astype(int)
    eligible["session_idx"] = eligible["session_idx"].astype(int)
    eligible[QUALITY_COLUMN] = pd.to_numeric(eligible[QUALITY_COLUMN], errors="raise")
    margins = eligible[QUALITY_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(margins).all():
        raise QualityTrainingError("quality margin contains NaN or Inf")
    return eligible.sort_values(["subject", "session_idx"]).reset_index(drop=True)


def canonical_keys_from_reference(reference_band: dict) -> list[tuple[int, int]]:
    population = reference_band.get("population", {})
    sessions = population.get("sessions", [])
    keys = [(int(row["subject"]), int(row["session_idx"])) for row in sessions]
    if population.get("n_sessions") != 73 or population.get("n_subjects") != 16:
        raise QualityTrainingError("reference population is not 73 sessions / 16 subjects")
    if len(keys) != 73 or len(set(keys)) != 73:
        raise QualityTrainingError("reference population keys are missing or duplicated")
    return keys


def align_sessions(sessions: list[dict], reference_band: dict) -> list[dict]:
    population_rows = reference_band.get("population", {}).get("sessions", [])
    canonical_keys = [
        (int(row["subject"]), int(row["session_idx"])) for row in population_rows
    ]
    current_keys = [(int(row["subject"]), int(row["session_idx"])) for row in sessions]
    if current_keys != canonical_keys or len(set(current_keys)) != len(current_keys):
        raise QualityTrainingError("feature-store sessions do not match the ordered canonical keys")
    ordered = list(sessions)
    for row, reference_row, key in zip(ordered, population_rows, canonical_keys, strict=True):
        target = float(row["delta_m_pct"])
        if not math.isfinite(target):
            raise QualityTrainingError(f"non-finite target for {key}")
        if str(row["session_name"]) != str(reference_row["session_name"]):
            raise QualityTrainingError(f"session name changed for canonical key {key}")
        if target != float(reference_row["delta_m_pct"]):
            raise QualityTrainingError(f"target changed for canonical key {key}")
    return ordered


def make_loso_splits(subjects: np.ndarray) -> list[RowSplit]:
    splits = []
    for subject in sorted(set(np.asarray(subjects, dtype=int).tolist())):
        test = np.flatnonzero(subjects == subject)
        train = np.flatnonzero(subjects != subject)
        splits.append(RowSplit(f"subject_{subject}", tuple(train.tolist()), tuple(test.tolist())))
    return splits


def make_subject_overlap_splits(
    session_classes: np.ndarray, *, n_splits: int = 5, seed: int = 20260829
) -> list[RowSplit]:
    """Deterministic session-level folds; each row appears in test exactly once."""
    classes = np.asarray(session_classes, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    coverage = np.zeros(classes.size, dtype=int)
    for fold_idx, (train, test) in enumerate(splitter.split(np.zeros(classes.size), classes)):
        if np.intersect1d(train, test).size:
            raise QualityTrainingError("subject-overlap split has row overlap")
        coverage[test] += 1
        splits.append(RowSplit(f"fold_{fold_idx}", tuple(train.tolist()), tuple(test.tolist())))
    if not np.array_equal(coverage, np.ones(classes.size, dtype=int)):
        raise QualityTrainingError("subject-overlap folds do not give each session one test role")
    return splits


def subject_overlap_census(split: RowSplit, subjects: np.ndarray) -> dict:
    train_subjects = set(np.asarray(subjects)[list(split.train_indices)].astype(int).tolist())
    test_subjects = set(np.asarray(subjects)[list(split.test_indices)].astype(int).tolist())
    overlap = sorted(train_subjects & test_subjects)
    return {
        "n_train_subjects": len(train_subjects),
        "n_test_subjects": len(test_subjects),
        "n_overlapping_subjects": len(overlap),
        "overlapping_subjects": overlap,
    }


def make_inner_session_splits(
    outer_train_indices: Iterable[int], session_classes: np.ndarray, *, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    outer = np.asarray(tuple(outer_train_indices), dtype=int)
    labels = np.asarray(session_classes, dtype=int)[outer]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output = []
    for local_train, local_val in splitter.split(np.zeros(outer.size), labels):
        train, val = outer[local_train], outer[local_val]
        if np.intersect1d(train, val).size:
            raise QualityTrainingError("inner session split has row overlap")
        output.append((train, val))
    return output


class ExactRowFeatureSource:
    """Store-backed WST features whose fitted epsilon consumes explicit session keys."""

    def __init__(self, sessions: list[dict], store_dir: Path, config, quality_margin: np.ndarray):
        self.sessions = sessions
        self.config = config
        self.keys = [(int(s["subject"]), int(s["session_idx"])) for s in sessions]
        self.key_to_index = {key: index for index, key in enumerate(self.keys)}
        self.subjects = np.asarray([int(s["subject"]) for s in sessions], dtype=int)
        self.quality_margin = np.asarray(quality_margin, dtype=float)
        self._stores = [
            store_mod.read_session_store("10ghz", s["subject"], s["session_name"], store_dir)
            for s in sessions
        ]
        self._vector_cache: dict[tuple, np.ndarray] = {}
        self._raw_cache: dict[tuple, list] = {}
        self._tuned_cache: dict[tuple, FeatureMatrix] = {}

    @staticmethod
    def _vector_key(feature_key: tuple, branch: str) -> str:
        gate, reduction, channel, tiling, _ = feature_key
        return store_mod.vec_key(gate, reduction, channel, tiling, branch)

    @staticmethod
    def _prelog_key(feature_key: tuple) -> str:
        gate, reduction, channel, tiling, _ = feature_key
        return store_mod.prelog_key(gate, reduction, channel, tiling)

    @staticmethod
    def _raw_key(feature_key: tuple) -> str:
        gate, reduction, channel, tiling, _ = feature_key
        return store_mod.raw_key(gate, reduction, channel, tiling)

    def _fixed_matrix(self, feature_key: tuple, branch: str) -> np.ndarray:
        cache_key = (feature_key[:-1], branch)
        if cache_key not in self._vector_cache:
            store_key = self._vector_key(feature_key, branch)
            self._vector_cache[cache_key] = np.stack([store[store_key] for store in self._stores])
        return self._vector_cache[cache_key]

    def _epsilon(self, feature_key: tuple, fit_keys: tuple[tuple[int, int], ...]) -> dict[int, float]:
        prelog_key = self._prelog_key(feature_key)
        by_subject: dict[int, list[tuple[float, float, float]]] = {}
        for key in fit_keys:
            if key not in self.key_to_index:
                raise QualityTrainingError(f"unknown fitted row key {key}")
            index = self.key_to_index[key]
            values = tuple(np.asarray(self._stores[index][prelog_key], dtype=float).tolist())
            by_subject.setdefault(key[0], []).append(values)
        return tuned_epsilons(
            by_subject,
            sorted(by_subject),
            k=self.config.search_10ghz.tuned_eps_k,
            fallback=self.config.wst.log_epsilon,
        )

    def _tuned_matrix(self, feature_key: tuple, epsilon: dict[int, float]) -> np.ndarray:
        geometry = feature_key[:-1]
        if geometry not in self._raw_cache:
            raw_key = self._raw_key(feature_key)
            order_key = store_mod.order_key(feature_key[3])
            self._raw_cache[geometry] = [
                (store[raw_key], np.asarray(store[order_key])) for store in self._stores
            ]
        rows = []
        for scattering, order in self._raw_cache[geometry]:
            meta = {"order": order}
            logged = apply_order_log(
                scattering, meta, self.config.wst, log_on=True, epsilon_by_order=epsilon
            )
            rows.append(aggregate_session(pool_stats_batch(logged, meta)))
        return np.stack(rows)

    def matrix_for(
        self, candidate: Candidate, fit_keys: tuple[tuple[int, int], ...], *, append_quality: bool
    ) -> FeatureMatrix:
        branch = candidate.feature_key[-1]
        if branch in ("off", "frozen"):
            matrix = self._fixed_matrix(candidate.feature_key, branch)
            epsilon = None
        elif branch == "tuned":
            cache_key = (candidate.feature_key, fit_keys)
            if cache_key not in self._tuned_cache:
                epsilon = self._epsilon(candidate.feature_key, fit_keys)
                matrix = self._tuned_matrix(candidate.feature_key, epsilon)
                self._tuned_cache[cache_key] = FeatureMatrix(matrix, epsilon)
            cached = self._tuned_cache[cache_key]
            epsilon, matrix = cached.tuned_epsilon, cached.values
        else:
            raise QualityTrainingError(f"unknown WST log branch {branch!r}")
        if append_quality:
            matrix = np.column_stack([matrix, self.quality_margin])
        if not np.isfinite(matrix).all():
            raise QualityTrainingError("feature matrix contains NaN or Inf")
        return FeatureMatrix(np.asarray(matrix, dtype=float), epsilon)


class ArrayFeatureSource:
    """Small deterministic source used by unit tests and the local mechanism smoke."""

    def __init__(self, values: np.ndarray, quality_margin: np.ndarray, keys: list[tuple[int, int]]):
        self.values = np.asarray(values, dtype=float)
        self.quality_margin = np.asarray(quality_margin, dtype=float)
        self.keys = list(keys)

    def matrix_for(self, candidate, fit_keys, *, append_quality):
        # The scalar stands in for fold-tuned epsilon and proves its row dependency in tests.
        positions = [self.keys.index(key) for key in fit_keys]
        epsilon = {1: float(np.median(self.values[positions, 0])), 2: 1e-6}
        values = self.values.copy()
        if append_quality:
            values = np.column_stack([values, self.quality_margin])
        return FeatureMatrix(values, epsilon)


def _candidate_seed_list(candidate: Candidate, seeds: tuple[int, ...]) -> tuple[int, ...]:
    return seeds if candidate.family in regressors.SEED_SENSITIVE else (seeds[0],)


def _authorize_candidate(candidate: Candidate, config, task: str, arm: str | None) -> None:
    if task == "regression":
        exp_a.require_complete_active(dict(candidate.active))
        protocol_freeze_guard(config, active=dict(candidate.active))
        return
    exp_c._before_fit_c(config, arm or "a")(candidate)


def _quality_filtered_train_indices(
    train_indices: Iterable[int], quality_margin: np.ndarray, treatment: str
) -> np.ndarray:
    train = np.asarray(tuple(train_indices), dtype=int)
    if treatment == "filter_negative_margin":
        train = train[np.asarray(quality_margin, dtype=float)[train] >= QUALITY_THRESHOLD]
    if treatment not in TREATMENTS:
        raise QualityTrainingError(f"unknown treatment {treatment!r}")
    if train.size == 0:
        raise QualityTrainingError("quality filtering removed every training row")
    return train


def _task_target(task: str, regression_target: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if task == "regression":
        return np.asarray(regression_target, dtype=float)
    if task == "ordinal_classification":
        loss = -np.asarray(regression_target, dtype=float)
        return np.column_stack([loss, np.asarray(classes, dtype=float)])
    raise QualityTrainingError(f"unknown task {task!r}")


def _require_class_viability(classes: np.ndarray, train_indices: np.ndarray) -> None:
    present = set(np.asarray(classes, dtype=int)[train_indices].tolist())
    missing = sorted(set(range(5)) - present)
    if missing:
        raise QualityTrainingError(f"ordinal training rows are missing classes {missing}")


def fit_selected_candidate(
    *,
    candidate: Candidate,
    source,
    keys: list[tuple[int, int]],
    quality_margin: np.ndarray,
    train_indices: Iterable[int],
    test_indices: Iterable[int],
    regression_target: np.ndarray,
    classes: np.ndarray,
    task: str,
    treatment: str,
    seeds: tuple[int, ...],
    authorize: Callable[[Candidate], None] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Fit one already-selected learner under one treatment and return predictions/audit."""
    if len(keys) != len(set(keys)):
        raise QualityTrainingError("canonical row keys contain duplicates")
    raw_train = tuple(int(index) for index in train_indices)
    raw_test = tuple(int(index) for index in test_indices)
    if len(raw_train) != len(set(raw_train)) or len(raw_test) != len(set(raw_test)):
        raise QualityTrainingError("fitted/test row indices contain duplicates")
    test = np.asarray(raw_test, dtype=int)
    fitted = _quality_filtered_train_indices(raw_train, quality_margin, treatment)
    if np.intersect1d(fitted, test).size:
        raise QualityTrainingError("fitted and held-out row keys overlap")
    if task == "ordinal_classification":
        _require_class_viability(classes, fitted)
    fit_keys = tuple(keys[index] for index in fitted)
    test_keys = tuple(keys[index] for index in test)
    if len(fit_keys) != len(set(fit_keys)) or len(test_keys) != len(set(test_keys)):
        raise QualityTrainingError("fitted/test row keys contain duplicates")
    if set(fit_keys).intersection(test_keys):
        raise QualityTrainingError("fitted and held-out row keys overlap")
    feature_result = source.matrix_for(
        candidate, fit_keys, append_quality=treatment == "append_margin_feature"
    )
    X = feature_result.values
    y = _task_target(task, regression_target, classes)
    if authorize is not None:
        authorize(candidate)

    predictions, audits = [], []
    with threadpool_limits(1):
        for seed in _candidate_seed_list(candidate, seeds):
            estimator = regressors.build_estimator(candidate.family, candidate.params(), seed=seed)
            estimator.fit(X[fitted], y[fitted])
            predicted = estimator.predict(X[test])
            scaler = estimator.named_steps["scaler"]
            model = estimator.named_steps["model"]
            state = regressors.fitted_state_params(candidate.family, model)
            state_digest = hashlib.sha256()
            state_component_hashes = {}
            for name in sorted(state):
                state_digest.update(name.encode("utf-8"))
                state_digest.update(np.ascontiguousarray(state[name]).tobytes())
                state_component_hashes[name] = _sha256_array(state[name])
            for row_index, prediction in zip(test, predicted, strict=True):
                predictions.append(
                    {
                        "seed": int(seed),
                        "row_index": int(row_index),
                        "subject": int(keys[row_index][0]),
                        "session_idx": int(keys[row_index][1]),
                        "y_true": float(classes[row_index] if task == "ordinal_classification" else regression_target[row_index]),
                        "y_pred": float(prediction),
                    }
                )
            audits.append(
                {
                    "seed": int(seed),
                    "n_fitted_rows": int(fitted.size),
                    "fitted_row_keys": [list(key) for key in fit_keys],
                    "test_row_keys": [list(key) for key in test_keys],
                    "quality_threshold": QUALITY_THRESHOLD,
                    "quality_filter_applied": treatment == "filter_negative_margin",
                    "quality_feature_appended": treatment == "append_margin_feature",
                    "tuned_epsilon": feature_result.tuned_epsilon,
                    "scaler_mean_sha256": _sha256_array(scaler.mean_),
                    "scaler_scale_sha256": _sha256_array(scaler.scale_),
                    "model_state_sha256": state_digest.hexdigest(),
                    "model_state_component_sha256": state_component_hashes,
                    "ordinal_cutpoints": (
                        np.asarray(state["cutpoints_"], dtype=float).tolist()
                        if "cutpoints_" in state else None
                    ),
                    "ordinal_class_weights": (
                        np.asarray(state["class_weights_"], dtype=float).tolist()
                        if "class_weights_" in state else None
                    ),
                }
            )
    return predictions, audits


def _selection_metrics_from_rows(rows: list[dict], task: str) -> tuple[float, float]:
    """Frozen inner-selection seed semantics for an explicit-row validation fold.

    Primary MAE is computed independently for each effective seed and then averaged.
    Experiment C's QWK tie-break reads the first effective seed only, matching its existing
    ``_cell_qwk`` contract.  Concatenating seed rows would incorrectly treat model repeats
    as additional validation samples.
    """
    rows_by_seed: dict[int, list[dict]] = {}
    for row in rows:
        rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    if not rows_by_seed:
        raise QualityTrainingError("selection received no validation predictions")
    primary_by_seed = []
    secondary_by_seed = []
    for seed_rows in rows_by_seed.values():
        truth = np.asarray([row["y_true"] for row in seed_rows], dtype=float)
        predicted = np.asarray([row["y_pred"] for row in seed_rows], dtype=float)
        if task == "regression":
            subjects = np.asarray([row["subject"] for row in seed_rows], dtype=int)
            primary_by_seed.append(M.subject_balanced_mae(subjects, truth, predicted))
            secondary_by_seed.append(float(np.sqrt(np.mean((truth - predicted) ** 2))))
        else:
            class_pred = np.rint(predicted).astype(int)
            true_class = truth.astype(int)
            primary_by_seed.append(M.class_unit_mae(true_class, class_pred))
            secondary_by_seed.append(M.quadratic_weighted_kappa(true_class, class_pred))
    primary = float(np.mean(primary_by_seed))
    secondary = float(np.mean(secondary_by_seed)) if task == "regression" else float(secondary_by_seed[0])
    return primary, secondary


def score_candidates_on_session_splits(
    *, candidates: list[Candidate], source, keys, quality_margin, inner_splits,
    regression_target, classes, task, seeds, authorize_for: Callable[[Candidate], None]
) -> Candidate:
    """Baseline-only nested selection with explicit row masks for the overlap diagnostic."""
    scores = []
    for candidate in candidates:
        fold_primary, fold_secondary = [], []
        feature_dimension = None
        for train, val in inner_splits:
            if task == "ordinal_classification":
                try:
                    _require_class_viability(classes, train)
                except QualityTrainingError:
                    continue
            rows, _ = fit_selected_candidate(
                candidate=candidate, source=source, keys=keys, quality_margin=quality_margin,
                train_indices=train, test_indices=val, regression_target=regression_target,
                classes=classes, task=task, treatment="baseline", seeds=seeds,
                authorize=authorize_for,
            )
            primary, secondary = _selection_metrics_from_rows(rows, task)
            fold_primary.append(primary)
            fold_secondary.append(secondary)
            if feature_dimension is None:
                result = source.matrix_for(candidate, tuple(keys[i] for i in train), append_quality=False)
                feature_dimension = int(result.values.shape[1])
        if task == "regression":
            scores.append(CandidateScore(
                candidate.candidate_id,
                float(np.mean(fold_primary)) if fold_primary else float("nan"),
                SIMPLICITY_RANK[candidate.family], int(feature_dimension or 0),
                float(np.std(fold_primary, ddof=0)) if fold_primary else float("nan"),
            ))
        else:
            scores.append(OrdinalCandidateScore(
                candidate.candidate_id,
                float(np.mean(fold_primary)) if fold_primary else float("nan"),
                float(np.nanmean(fold_secondary)) if fold_secondary and np.isfinite(fold_secondary).any() else float("nan"),
                SIMPLICITY_RANK[candidate.family], int(feature_dimension or 0),
                float(np.std(fold_primary, ddof=0)) if fold_primary else float("nan"),
                len(fold_primary),
            ))
    winner_score = select_candidate(scores) if task == "regression" else select_candidate_ordinal(scores)
    return {candidate.candidate_id: candidate for candidate in candidates}[winner_score.candidate_id]


def _find_candidate(candidates: Iterable[Candidate], family: str, params: dict) -> Candidate:
    matches = [c for c in candidates if c.family == family and c.params() == params]
    if len(matches) != 1:
        raise QualityTrainingError(f"could not resolve one candidate for family={family}, params={params}")
    return matches[0]


def authoritative_regression_candidates(config, reference_band: dict) -> dict[int, SelectedCandidate]:
    """Resolve each authenticated Exp-A fold selection back into the frozen candidate list."""
    output = {}
    for row in reference_band["selected_folds"]:
        feature_key = tuple(row["feature_key"])
        stage1 = exp_a.stage1_candidates(
            config, "10ghz", config.search_10ghz.stage1_anchor_ridge_alpha
        )
        stage1_match = [candidate for candidate in stage1 if candidate.feature_key == feature_key]
        if len(stage1_match) != 1:
            raise QualityTrainingError(f"authenticated feature key is outside frozen Stage 1: {feature_key}")
        candidates = exp_a.stage2_candidates(config, "10ghz", feature_key, dict(stage1_match[0].active))
        candidate = _find_candidate(candidates, row["family"], dict(row["params"]))
        output[int(row["test_subject"])] = SelectedCandidate(candidate, "authenticated_exp_a_outer_fold")
    if set(output) != set(range(1, 17)):
        raise QualityTrainingError("authenticated Exp-A selections do not cover subjects 1..16")
    return output


def verify_loso_regression_baseline_replay(
    predictions: list[dict], audits: list[dict], selections: list[dict], reference_band: dict,
    *, tolerance: float = BASELINE_REPLAY_TOLERANCE,
) -> dict:
    """Fail closed unless the new baseline refit reproduces authenticated Exp A.

    Deterministic learners have one canonical effective seed in both artifacts; RF/GBM
    retain all five.  No synthetic seed replication is introduced merely to make shapes
    agree.  Selection, target/key order, tuned epsilon, and predictions are checked before
    the sensitivity runner writes any output.
    """
    baseline = [
        row for row in predictions
        if row["protocol"] == "loso" and row["task"] == "regression"
        and row["arm"] == "regression" and row["treatment"] == "baseline"
    ]
    baseline_audits = [
        row for row in audits
        if row["protocol"] == "loso" and row["task"] == "regression"
        and row["arm"] == "regression" and row["treatment"] == "baseline"
    ]
    selection_by_split = {
        row["split_id"]: row for row in selections
        if row["protocol"] == "loso" and row["task"] == "regression"
    }
    max_prediction_delta = 0.0
    max_epsilon_delta = 0.0
    checked_prediction_rows = 0
    for reference_fold in reference_band.get("selected_folds", []):
        subject = int(reference_fold["test_subject"])
        split_id = f"subject_{subject}"
        selection = selection_by_split.get(split_id)
        if selection is None:
            raise QualityTrainingError(f"baseline replay is missing selection for {split_id}")
        if (
            selection["family"] != reference_fold["family"]
            or selection["params"] != reference_fold["params"]
            or selection["feature_key"] != reference_fold["feature_key"]
        ):
            raise QualityTrainingError(f"baseline replay selection differs for {split_id}")
        reference_predictions = reference_fold["predictions"]
        expected_seeds = [int(seed) for seed in reference_predictions["seeds"]]
        actual_seeds = sorted({int(row["seed"]) for row in baseline if row["split_id"] == split_id})
        if actual_seeds != expected_seeds:
            raise QualityTrainingError(
                f"baseline replay effective seeds differ for {split_id}: {actual_seeds} != {expected_seeds}"
            )
        for seed in expected_seeds:
            actual = [
                row for row in baseline if row["split_id"] == split_id and int(row["seed"]) == seed
            ]
            expected_truth = np.asarray(reference_predictions["y_true"], dtype=float)
            expected_pred = np.asarray(reference_predictions["y_pred_by_seed"][str(seed)], dtype=float)
            actual_truth = np.asarray([row["y_true"] for row in actual], dtype=float)
            actual_pred = np.asarray([row["y_pred"] for row in actual], dtype=float)
            actual_keys = [(int(row["subject"]), int(row["session_idx"])) for row in actual]
            population_rows = [
                row for row in reference_band["population"]["sessions"]
                if int(row["subject"]) == subject
            ]
            expected_keys = [(subject, int(row["session_idx"])) for row in population_rows]
            if actual_keys != expected_keys or not np.array_equal(actual_truth, expected_truth):
                raise QualityTrainingError(f"baseline replay key/target order differs for {split_id}, seed {seed}")
            if actual_pred.shape != expected_pred.shape or not np.isfinite(actual_pred).all():
                raise QualityTrainingError(f"baseline replay prediction shape/finite check failed for {split_id}")
            delta = float(np.max(np.abs(actual_pred - expected_pred))) if actual_pred.size else 0.0
            max_prediction_delta = max(max_prediction_delta, delta)
            checked_prediction_rows += int(actual_pred.size)
        expected_epsilon = reference_fold.get("tuned_epsilon")
        fold_audits = [row for row in baseline_audits if row["split_id"] == split_id]
        if not fold_audits:
            raise QualityTrainingError(f"baseline replay is missing fit audit for {split_id}")
        for audit in fold_audits:
            actual_epsilon = audit["tuned_epsilon"]
            if expected_epsilon is None:
                if actual_epsilon is not None:
                    raise QualityTrainingError(f"baseline replay introduced tuned epsilon for {split_id}")
                continue
            if actual_epsilon is None:
                raise QualityTrainingError(f"baseline replay lost tuned epsilon for {split_id}")
            for order in (1, 2):
                delta = abs(float(actual_epsilon[order]) - float(expected_epsilon[f"order_{order}"]))
                max_epsilon_delta = max(max_epsilon_delta, delta)
    # Count against the manifest rather than assuming 73: seed-sensitive folds legitimately
    # have one row per realized model seed while deterministic folds have one.
    expected_count = sum(
        int(row["predictions"]["n_sessions"]) * len(row["predictions"]["seeds"])
        for row in reference_band["selected_folds"]
    )
    if checked_prediction_rows != expected_count:
        raise QualityTrainingError(
            f"baseline replay checked {checked_prediction_rows} prediction rows, expected {expected_count}"
        )
    if max_prediction_delta > tolerance or max_epsilon_delta > tolerance:
        raise QualityTrainingError(
            "baseline replay exceeded tolerance: "
            f"prediction={max_prediction_delta:.3e}, epsilon={max_epsilon_delta:.3e}, "
            f"tolerance={tolerance:.1e}"
        )
    return {
        "status": "passed",
        "tolerance": tolerance,
        "n_prediction_rows": checked_prediction_rows,
        "max_abs_prediction_delta": max_prediction_delta,
        "max_abs_tuned_epsilon_delta": max_epsilon_delta,
        "canonical_seed_semantics": "deterministic_one_seed; seed_sensitive_all_realized_seeds",
    }


def _nested_select_session_fold(config, source, keys, quality_margin, split, regression_target, classes, task, seeds, seed):
    inner = make_inner_session_splits(split.train_indices, classes, n_splits=4, seed=seed)
    anchor = config.search_10ghz.stage1_anchor_ridge_alpha
    if task == "regression":
        stage1 = exp_a.stage1_candidates(config, "10ghz", anchor)
        authorize = lambda c: _authorize_candidate(c, config, task, None)
        w1 = score_candidates_on_session_splits(
            candidates=stage1, source=source, keys=keys, quality_margin=quality_margin,
            inner_splits=inner, regression_target=regression_target, classes=classes,
            task=task, seeds=seeds, authorize_for=authorize,
        )
        stage2 = exp_a.stage2_candidates(config, "10ghz", w1.feature_key, dict(w1.active))
        winner = score_candidates_on_session_splits(
            candidates=stage2, source=source, keys=keys, quality_margin=quality_margin,
            inner_splits=inner, regression_target=regression_target, classes=classes,
            task=task, seeds=seeds, authorize_for=authorize,
        )
        return {"regression": SelectedCandidate(winner, "baseline_nested_session_cv")}

    stage1 = exp_c.stage1_candidates_c(config, "10ghz", anchor)
    w1 = score_candidates_on_session_splits(
        candidates=stage1, source=source, keys=keys, quality_margin=quality_margin,
        inner_splits=inner, regression_target=regression_target, classes=classes,
        task=task, seeds=seeds, authorize_for=lambda c: _authorize_candidate(c, config, task, "stage1"),
    )
    output = {}
    for arm, builder in (("a", exp_c.stage2_candidates_a), ("b", exp_c.stage2_candidates_b)):
        candidates = builder(config, "10ghz", w1.feature_key, dict(w1.active))
        winner = score_candidates_on_session_splits(
            candidates=candidates, source=source, keys=keys, quality_margin=quality_margin,
            inner_splits=inner, regression_target=regression_target, classes=classes,
            task=task, seeds=seeds,
            authorize_for=lambda c, arm=arm: _authorize_candidate(c, config, task, arm),
        )
        output[arm] = SelectedCandidate(winner, "baseline_nested_session_cv")
    return output


def _loso_classification_candidates(
    config, sessions, store_dir, seeds, *, n_workers: int = 1
) -> dict[int, dict[str, SelectedCandidate]]:
    """Run the unchanged Exp-C baseline selection, then expose its two fixed winners."""
    output = {}
    # The sensitivity runner's canonical spine comes from Exp A.  Exp C uses the same rows
    # and continuous target, but additionally expects the frozen ordinal target columns.
    # Copy each row so this adapter cannot mutate the authenticated 73-session relation.
    ordinal_sessions = [
        {
            **session,
            "loss_l": -float(session["delta_m_pct"]),
            "class_idx": int(session["session_idx"]),
        }
        for session in sessions
    ]
    results = exp_c.run_exp_c(
        config, "10ghz", ordinal_sessions, store_dir, seeds=seeds, n_workers=n_workers
    )
    stage1_candidates = exp_c.stage1_candidates_c(
        config, "10ghz", config.search_10ghz.stage1_anchor_ridge_alpha
    )
    for result in results:
        if result.reason is not None:
            raise QualityTrainingError(
                f"Exp-C baseline selection failed for subject {result.test_subject}: {result.reason}"
            )
        stage1_matches = [
            candidate for candidate in stage1_candidates
            if candidate.feature_key == result.stage1_feature_key
        ]
        if len(stage1_matches) != 1:
            raise QualityTrainingError(
                f"Exp-C Stage-1 feature key did not resolve once: {result.stage1_feature_key}"
            )
        output[result.test_subject] = {}
        for arm in exp_c.ARMS:
            selected = result.arm_result(arm)
            builder = exp_c.stage2_candidates_a if arm == "a" else exp_c.stage2_candidates_b
            candidates = builder(
                config, "10ghz", result.stage1_feature_key, dict(stage1_matches[0].active)
            )
            winner = _find_candidate(candidates, selected.selected_family, selected.selected_params)
            output[result.test_subject][arm] = SelectedCandidate(
                winner, "frozen_exp_c_nested_loso_baseline"
            )
    return output


def run_protocol(
    *, protocol: str, task: str, config, sessions, source, quality_margin,
    reference_band: dict, split_seed: int, seeds: tuple[int, ...], n_workers: int = 1
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    keys = [(int(s["subject"]), int(s["session_idx"])) for s in sessions]
    subjects = np.asarray([key[0] for key in keys], dtype=int)
    classes = np.asarray([key[1] for key in keys], dtype=int)
    target = np.asarray([float(s["delta_m_pct"]) for s in sessions], dtype=float)
    splits = make_loso_splits(subjects) if protocol == "loso" else make_subject_overlap_splits(classes, seed=split_seed)

    regression_loso = authoritative_regression_candidates(config, reference_band) if protocol == "loso" and task == "regression" else None
    classification_loso = (
        _loso_classification_candidates(
            config, sessions, config.paths.results_dir, seeds, n_workers=n_workers
        )
        if protocol == "loso" and task == "ordinal_classification" else None
    )
    predictions, audits, selections, split_rows = [], [], [], []
    for fold_number, split in enumerate(splits):
        if protocol == "loso":
            test_subject = int(subjects[list(split.test_indices)][0])
            selected_by_arm = (
                {"regression": regression_loso[test_subject]}
                if task == "regression" else classification_loso[test_subject]
            )
        else:
            selected_by_arm = _nested_select_session_fold(
                config, source, keys, quality_margin, split, target, classes, task, seeds,
                split_seed + 1000 + fold_number,
            )
        overlap = subject_overlap_census(split, subjects)
        split_rows.append({
            "protocol": protocol, "task": task, "split_id": split.split_id,
            "train_row_keys": [list(keys[i]) for i in split.train_indices],
            "test_row_keys": [list(keys[i]) for i in split.test_indices], **overlap,
        })
        for arm, selection in selected_by_arm.items():
            selections.append({
                "protocol": protocol, "task": task, "split_id": split.split_id, "arm": arm,
                "candidate_id": selection.candidate.candidate_id,
                "family": selection.candidate.family, "params": selection.candidate.params(),
                "feature_key": list(selection.candidate.feature_key),
                "selection_source": selection.selection_source,
            })
            for treatment in TREATMENTS:
                pred, audit = fit_selected_candidate(
                    candidate=selection.candidate, source=source, keys=keys,
                    quality_margin=quality_margin, train_indices=split.train_indices,
                    test_indices=split.test_indices, regression_target=target, classes=classes,
                    task=task, treatment=treatment, seeds=seeds,
                    authorize=lambda c, arm=arm: _authorize_candidate(c, config, task, None if task == "regression" else arm),
                )
                for row in pred:
                    row.update({"protocol": protocol, "task": task, "split_id": split.split_id,
                                "arm": arm, "treatment": treatment})
                for row in audit:
                    row.update({"protocol": protocol, "task": task, "split_id": split.split_id,
                                "arm": arm, "treatment": treatment})
                predictions.extend(pred)
                audits.extend(audit)
    assert_identical_test_keys(predictions)
    return predictions, audits, selections, split_rows


def assert_identical_test_keys(predictions: list[dict]) -> None:
    """Every paired treatment must contain exactly the baseline's held-out rows per seed."""
    grouped: dict[tuple, dict[str, list]] = {}
    for row in predictions:
        group = (row["protocol"], row["task"], row["split_id"], row["arm"], row["seed"])
        grouped.setdefault(group, {}).setdefault(row["treatment"], []).append(
            (row["subject"], row["session_idx"])
        )
    for group, by_treatment in grouped.items():
        if set(by_treatment) != set(TREATMENTS):
            raise QualityTrainingError(f"{group}: missing treatment prediction rows")
        baseline = by_treatment["baseline"]
        for treatment, keys in by_treatment.items():
            if len(keys) != len(set(keys)):
                raise QualityTrainingError(f"{group}/{treatment}: duplicate held-out keys")
        if any(keys != baseline for keys in by_treatment.values()):
            raise QualityTrainingError(f"{group}: treatments have different held-out keys")


def expand_deterministic_seed_predictions(predictions: list[dict]) -> list[dict]:
    """Give every split the same realized seed axis before computing metrics.

    Experiment A and Experiment C realize one prediction for deterministic learners and
    all configured seeds for stochastic learners.  Their canonical reporters copy a
    deterministic fold across the realized seed axis so every held-out session has equal
    weight.  This function applies that same reporting-only rule to sensitivity rows.

    Seed 1 is the canonical deterministic outcome.  Within each protocol/task/arm/
    treatment group, a split must contain either seed 1 alone or the complete seed set
    realized by that group.  Session order and truth must be identical across a split's
    realized seeds; malformed prediction tables fail instead of being scored.
    """
    if not predictions:
        raise QualityTrainingError("cannot summarize an empty prediction table")

    frame = pd.DataFrame(predictions)
    required_columns = {
        "protocol", "task", "split_id", "arm", "treatment", "seed",
        "subject", "session_idx", "y_true", "y_pred",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise QualityTrainingError(f"prediction table is missing columns: {missing}")

    group_columns = ["protocol", "task", "arm", "treatment"]
    expanded: list[dict] = []
    for group, group_rows in frame.groupby(group_columns, sort=False):
        seed_values = group_rows["seed"].to_numpy()
        if any(not isinstance(seed, (int, np.integer)) for seed in seed_values):
            raise QualityTrainingError(f"{group}: seed IDs must be integers")
        realized_seeds = tuple(sorted({int(seed) for seed in seed_values}))
        if not realized_seeds or realized_seeds[0] != 1:
            raise QualityTrainingError(f"{group}: realized seed IDs must include seed 1")

        for split_id, split_rows in group_rows.groupby("split_id", sort=False):
            split_seed_ids = tuple(sorted({int(seed) for seed in split_rows["seed"]}))
            if split_seed_ids not in ((1,), realized_seeds):
                raise QualityTrainingError(
                    f"{group}/{split_id}: seed IDs {split_seed_ids} are neither deterministic "
                    f"(1,) nor the complete realized set {realized_seeds}"
                )

            rows_by_seed: dict[int, list[dict]] = {}
            reference_keys: list[tuple[int, int]] | None = None
            reference_truth: np.ndarray | None = None
            for seed, seed_rows in split_rows.groupby("seed", sort=False):
                records = seed_rows.to_dict(orient="records")
                keys = [
                    (int(row["subject"]), int(row["session_idx"])) for row in records
                ]
                if len(keys) != len(set(keys)):
                    raise QualityTrainingError(
                        f"{group}/{split_id}/seed_{int(seed)}: duplicate session keys"
                    )
                truth = np.asarray([row["y_true"] for row in records], dtype=float)
                predicted = np.asarray([row["y_pred"] for row in records], dtype=float)
                if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
                    raise QualityTrainingError(
                        f"{group}/{split_id}/seed_{int(seed)}: non-finite truth or prediction"
                    )
                if reference_keys is None:
                    reference_keys = keys
                    reference_truth = truth
                elif keys != reference_keys:
                    raise QualityTrainingError(
                        f"{group}/{split_id}: ordered session keys differ across seeds"
                    )
                elif not np.array_equal(truth, reference_truth):
                    raise QualityTrainingError(
                        f"{group}/{split_id}: truth differs across seeds"
                    )
                rows_by_seed[int(seed)] = records

            if split_seed_ids == (1,):
                source_rows = rows_by_seed[1]
                for seed in realized_seeds:
                    for row in source_rows:
                        copied = dict(row)
                        copied["seed"] = seed
                        expanded.append(copied)
            else:
                for seed in realized_seeds:
                    expanded.extend(rows_by_seed[seed])

    return expanded


def summarize_predictions(predictions: list[dict]) -> dict:
    """Ordinal-first classification metrics and subject-balanced regression metrics."""
    observed_frame = pd.DataFrame(predictions)
    frame = pd.DataFrame(expand_deterministic_seed_predictions(predictions))
    summary = {"exploratory_post_hoc": True, "groups": []}
    group_columns = ["protocol", "task", "arm", "treatment"]
    observed_counts = observed_frame.groupby(group_columns, sort=True).size().to_dict()
    per_subject_rows = []
    for group, rows in frame.groupby(group_columns, sort=True):
        protocol, task, arm, treatment = group
        seed_metrics = []
        for seed, seed_rows in rows.groupby("seed", sort=True):
            truth = seed_rows.y_true.to_numpy(dtype=float)
            predicted = seed_rows.y_pred.to_numpy(dtype=float)
            subjects = seed_rows.subject.to_numpy(dtype=int)
            if task == "regression":
                subject_ids = sorted(set(subjects.tolist()))
                subject_mae = [np.mean(np.abs(truth[subjects == s] - predicted[subjects == s])) for s in subject_ids]
                subject_rmse = [np.sqrt(np.mean((truth[subjects == s] - predicted[subjects == s]) ** 2)) for s in subject_ids]
                seed_metrics.append({"seed": int(seed), "subject_balanced_mae_pct_points": float(np.mean(subject_mae)),
                                     "subject_balanced_rmse_pct_points": float(np.mean(subject_rmse))})
                for s, mae, rmse in zip(subject_ids, subject_mae, subject_rmse, strict=True):
                    per_subject_rows.append({"protocol": protocol, "task": task, "arm": arm,
                                             "treatment": treatment, "seed": int(seed), "subject": int(s),
                                             "mae": float(mae), "rmse": float(rmse)})
            else:
                true_class = truth.astype(int)
                pred_class = np.rint(predicted).astype(int)
                seed_metrics.append({
                    "seed": int(seed),
                    "class_unit_mae": M.class_unit_mae(true_class, pred_class),
                    "adjacent_accuracy": M.adjacent_accuracy(true_class, pred_class),
                    "quadratic_weighted_kappa": M.quadratic_weighted_kappa(true_class, pred_class),
                    "exact_accuracy_secondary": float(np.mean(true_class == pred_class)),
                })
                for s in sorted(set(subjects.tolist())):
                    mask = subjects == s
                    per_subject_rows.append({"protocol": protocol, "task": task, "arm": arm,
                                             "treatment": treatment, "seed": int(seed), "subject": int(s),
                                             "class_unit_mae": M.class_unit_mae(true_class[mask], pred_class[mask])})
        metric_names = [name for name in seed_metrics[0] if name != "seed"]
        summary["groups"].append({
            "protocol": protocol, "task": task, "arm": arm, "treatment": treatment,
            "n_prediction_rows": int(observed_counts[group]),
            "n_seed_expanded_scoring_rows": int(len(rows)),
            "n_realized_seeds": int(rows["seed"].nunique()),
            "n_unique_sessions": int(rows[["subject", "session_idx"]].drop_duplicates().shape[0]),
            **{name: float(np.nanmean([item[name] for item in seed_metrics])) for name in metric_names},
        })

    per_subject = pd.DataFrame(per_subject_rows)
    deltas = []
    index_columns = ["protocol", "task", "arm", "seed", "subject"]
    for metric in ("mae", "rmse", "class_unit_mae"):
        if metric not in per_subject.columns:
            continue
        table = per_subject.dropna(subset=[metric]).pivot(index=index_columns, columns="treatment", values=metric)
        if "baseline" not in table:
            continue
        for treatment in TREATMENTS[1:]:
            if treatment not in table:
                continue
            for index, value in (table[treatment] - table["baseline"]).items():
                deltas.append(dict(zip(index_columns, index, strict=True), treatment=treatment,
                                   metric=metric, delta_treatment_minus_baseline=float(value)))
    summary["paired_per_subject_deltas"] = deltas
    summary["interpretation"] = {
        "loso": "Primary estimate for a completely unseen subject.",
        "subject_overlap_session_cv": (
            "Optimistic diagnostic only: train and test contain sessions from many of the same subjects; "
            "this is not new-subject generalization and is post-hoc."
        ),
        "negative_delta": "For error metrics, a negative treatment-minus-baseline delta is better.",
    }
    return summary


def write_protocol_outputs(root: Path, protocol: str, predictions, audits, selections, splits) -> dict[str, Path]:
    out = root / protocol
    if out.exists() and any(out.iterdir()):
        raise QualityTrainingError(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": out / "predictions.csv",
        "fit_audit": out / "fit_audit.jsonl",
        "selections": out / "selections.json",
        "splits": out / "split_manifest.json",
        "metrics": out / "metrics.json",
    }
    fields = ["protocol", "task", "split_id", "arm", "treatment", "seed", "row_index",
              "subject", "session_idx", "y_true", "y_pred"]
    with paths["predictions"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{name: row[name] for name in fields} for row in predictions])
    with paths["fit_audit"].open("w", encoding="utf-8", newline="\n") as handle:
        for row in audits:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    paths["selections"].write_text(json.dumps(selections, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["splits"].write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics"].write_text(json.dumps(summarize_predictions(predictions), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def publish_atomically(output_root: Path, writer: Callable[[Path], object]) -> object:
    """Build a complete run beside the final root, then publish with one rename.

    A late exception removes only the uniquely-created staging directory.  An existing
    final target, even an empty one, is never overwritten or removed.
    """
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise QualityTrainingError(f"refusing to overwrite existing output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.staging-", dir=output_root.parent
    )).resolve()
    try:
        result = writer(staging)
        if output_root.exists():
            raise QualityTrainingError(f"output root appeared during staging: {output_root}")
        staging.replace(output_root)
        return result
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def synthetic_mechanism_smoke(seed: int = 20260829) -> dict:
    """Exercise both tasks and both split protocols locally without publishing scores."""
    rng = np.random.default_rng(seed)
    subjects = np.repeat(np.arange(1, 13), 5)
    classes = np.tile(np.arange(5), 12)
    keys = list(zip(subjects.tolist(), classes.tolist(), strict=True))
    margin = rng.normal(0.1, 0.04, len(keys))
    margin[[2, 17, 41]] = -0.02
    X = np.column_stack([classes + rng.normal(0, 0.2, len(keys)), rng.normal(size=len(keys))])
    target = -0.25 * classes + rng.normal(0, 0.05, len(keys))
    source = ArrayFeatureSource(X, margin, keys)
    candidate_reg = Candidate("smoke_ridge", "ridge", (("alpha", 1.0),), (0, "A", "mag", 0, "tuned"), None)
    candidate_ord = Candidate("smoke_ord", "ord_a_ridge", (("alpha", 1.0),), (0, "A", "mag", 0, "tuned"), None)
    exercised = []
    split_sets = {
        "loso": make_loso_splits(subjects)[:2],
        "subject_overlap_session_cv": make_subject_overlap_splits(classes, seed=seed)[:2],
    }
    for protocol, splits in split_sets.items():
        for task, candidate in (("regression", candidate_reg), ("ordinal_classification", candidate_ord)):
            for split in splits:
                for treatment in TREATMENTS:
                    predictions, audits = fit_selected_candidate(
                        candidate=candidate, source=source, keys=keys, quality_margin=margin,
                        train_indices=split.train_indices, test_indices=split.test_indices,
                        regression_target=target, classes=classes, task=task, treatment=treatment,
                        seeds=(1,), authorize=None,
                    )
                    exercised.append((protocol, task, split.split_id, treatment, len(predictions), audits[0]["n_fitted_rows"]))
    return {
        "stage": "quality-training-sensitivity-smoke",
        "mode": "mechanism-only",
        "seed": seed,
        "n_cells_exercised": len(exercised),
        "treatments": list(TREATMENTS),
        "protocols": list(split_sets),
        "tasks": list(TASKS),
        "performance_values_suppressed": True,
    }
