"""Strict, audit-specific configuration for the 10 GHz quality audit.

The project-wide :class:`dehyd.config.Config` is loaded only to reuse its already
validated radar, QC, preprocessing, and WST constants.  The audit then projects those
values into a smaller target-free record: the radar path never receives the workbook
path or any outcome-bearing object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import (
    REPO_ROOT,
    PreprocessConfig,
    QCConfig,
    WSTConfig,
    config_to_dict,
    load_config,
)

SCHEMA_VERSION = "quality_audit_10ghz_v1"


class QualityAuditConfigError(ValueError):
    """Raised when the audit declaration is incomplete or not the approved design."""


@dataclass(frozen=True)
class RadarAuditConfig:
    """Only values the target-free radar computation is allowed to receive."""

    data_10ghz_dir: Path
    existing_results_dir: Path
    output_results_dir: Path
    output_figures_dir: Path
    existing_qc_report: Path
    qc: QCConfig
    preprocess: PreprocessConfig
    wst: WSTConfig
    expected_subjects: tuple[int, ...]
    expected_frames_per_session: int
    n_blocks: int
    frames_per_block: int
    min_passing_frames_per_block: int
    epsilon_path_factor: float
    epsilon_distance_factor: float
    absent_remote_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class QualityAuditConfig:
    radar: RadarAuditConfig
    weight_xlsx: Path
    base_config_path: Path
    base_config_sha256: str


def _reject_unknown(mapping: dict, allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise QualityAuditConfigError(f"unknown field(s) in {name}: {', '.join(unknown)}")


def _require_mapping(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise QualityAuditConfigError(f"{key} must be a mapping")
    return value


def _repo_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise QualityAuditConfigError(f"{name} must be a non-empty path string")
    path = Path(value)
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _canonical_sha(mapping: dict) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def radar_config_record(config: RadarAuditConfig) -> dict:
    """Serializable radar-only declaration; intentionally has no workbook field."""
    return {
        "data_10ghz_dir": str(config.data_10ghz_dir),
        "existing_results_dir": str(config.existing_results_dir),
        "output_results_dir": str(config.output_results_dir),
        "output_figures_dir": str(config.output_figures_dir),
        "existing_qc_report": str(config.existing_qc_report),
        "qc": config_to_dict(config.qc),
        "preprocess": config_to_dict(config.preprocess),
        "wst": config_to_dict(config.wst),
        "expected_subjects": list(config.expected_subjects),
        "expected_frames_per_session": config.expected_frames_per_session,
        "n_blocks": config.n_blocks,
        "frames_per_block": config.frames_per_block,
        "min_passing_frames_per_block": config.min_passing_frames_per_block,
        "epsilon_path_factor": config.epsilon_path_factor,
        "epsilon_distance_factor": config.epsilon_distance_factor,
        "absent_remote_artifacts": list(config.absent_remote_artifacts),
    }


def load_quality_audit_config(path: str | Path) -> QualityAuditConfig:
    """Load the approved audit YAML without extending the global config schema."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise QualityAuditConfigError("quality-audit YAML must contain a mapping")
    _reject_unknown(
        raw,
        {"schema_version", "base_config", "outputs", "audit", "diagnostic", "sources"},
        "quality audit config",
    )
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise QualityAuditConfigError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {raw.get('schema_version')!r}"
        )

    base_value = raw.get("base_config")
    if not isinstance(base_value, str):
        raise QualityAuditConfigError("base_config must be a path string")
    base_path = (path.parent / base_value).resolve()
    canonical_base_path = (REPO_ROOT / "configs" / "exp_a_regression.yaml").resolve()
    if base_path != canonical_base_path:
        raise QualityAuditConfigError(
            f"base_config must name the unchanged canonical config {canonical_base_path}"
        )
    base = load_config(base_path)
    base_mapping_before = config_to_dict(base)

    outputs = _require_mapping(raw, "outputs")
    _reject_unknown(outputs, {"results", "figures"}, "outputs")
    results_output = _repo_path(outputs.get("results"), "outputs.results")
    figures_output = _repo_path(outputs.get("figures"), "outputs.figures")
    if results_output != (REPO_ROOT / "results" / "quality_10ghz").resolve():
        raise QualityAuditConfigError("outputs.results must be results/quality_10ghz")
    if figures_output != (REPO_ROOT / "figures" / "quality_10ghz").resolve():
        raise QualityAuditConfigError("outputs.figures must be figures/quality_10ghz")

    audit = _require_mapping(raw, "audit")
    _reject_unknown(
        audit,
        {
            "expected_subjects",
            "expected_frames_per_session",
            "n_blocks",
            "frames_per_block",
            "min_passing_frames_per_block",
            "epsilon_path_factor",
            "epsilon_distance_factor",
        },
        "audit",
    )
    expected_subjects = audit.get("expected_subjects")
    if expected_subjects != list(range(1, 17)):
        raise QualityAuditConfigError("audit.expected_subjects must be exactly [1, ..., 16]")
    integer_expected = {
        "expected_frames_per_session": 100,
        "n_blocks": 5,
        "frames_per_block": 20,
        "min_passing_frames_per_block": 10,
    }
    for key, expected in integer_expected.items():
        if audit.get(key) != expected:
            raise QualityAuditConfigError(f"audit.{key} must be the approved value {expected}")
    for key in ("epsilon_path_factor", "epsilon_distance_factor"):
        if audit.get(key) != 1.0e-12:
            raise QualityAuditConfigError(f"audit.{key} must be the approved value 1e-12")

    diagnostic = _require_mapping(raw, "diagnostic")
    _reject_unknown(diagnostic, {"reduction", "channels", "family", "log_on", "tilings"}, "diagnostic")
    approved_diagnostic = {
        "reduction": "a",
        "channels": ["mag", "iq"],
        "family": "pooled",
        "log_on": False,
        "tilings": "all_frozen",
    }
    if diagnostic != approved_diagnostic:
        raise QualityAuditConfigError(
            f"diagnostic declaration differs from the approved fixed family: {diagnostic!r}"
        )
    if base.qc != QCConfig() or base.preprocess != PreprocessConfig() or base.wst != WSTConfig():
        raise QualityAuditConfigError("base config does not carry the frozen canonical 10 GHz front end")

    sources = _require_mapping(raw, "sources")
    _reject_unknown(sources, {"existing_qc_report", "absent_remote_artifacts"}, "sources")
    absent = sources.get("absent_remote_artifacts")
    if absent != ["experiment_c_authoritative", "experiment_d_authoritative"]:
        raise QualityAuditConfigError("sources.absent_remote_artifacts must name authoritative C and D")

    # Loading/projecting a frozen dataclass must not mutate global serialization.
    if config_to_dict(base) != base_mapping_before:
        raise QualityAuditConfigError("loading the audit changed the base project config")

    radar = RadarAuditConfig(
        data_10ghz_dir=base.paths.data_10ghz_dir,
        existing_results_dir=base.paths.results_dir,
        output_results_dir=results_output,
        output_figures_dir=figures_output,
        existing_qc_report=_repo_path(sources.get("existing_qc_report"), "sources.existing_qc_report"),
        qc=base.qc,
        preprocess=base.preprocess,
        wst=base.wst,
        expected_subjects=tuple(expected_subjects),
        expected_frames_per_session=audit["expected_frames_per_session"],
        n_blocks=audit["n_blocks"],
        frames_per_block=audit["frames_per_block"],
        min_passing_frames_per_block=audit["min_passing_frames_per_block"],
        epsilon_path_factor=float(audit["epsilon_path_factor"]),
        epsilon_distance_factor=float(audit["epsilon_distance_factor"]),
        absent_remote_artifacts=tuple(absent),
    )
    return QualityAuditConfig(
        radar=radar,
        weight_xlsx=base.paths.weight_xlsx,
        base_config_path=base_path,
        base_config_sha256=_canonical_sha(base_mapping_before),
    )
