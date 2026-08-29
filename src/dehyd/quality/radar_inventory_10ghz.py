"""Target-free 10 GHz file inventory and per-frame QC wiring.

This module deliberately imports neither ``GroundTruth`` nor the target-bearing
manifest.  Its table schema contains only radar identity, stored frame index, and QC
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..data.loader_10ghz import (
    FileInfo,
    LoaderError,
    inspect_10ghz_file,
    load_10ghz_file,
    parse_10ghz_filename,
)
from ..data.sessions import SESSION_NAMES
from ..qc.screens import run_qc_cube
from .config import RadarAuditConfig

FORBIDDEN_RADAR_COLUMNS = (
    "mass",
    "delta_m",
    "class_label",
    "target",
    "prediction",
    "model_error",
)


class RadarInventoryError(ValueError):
    """Raised when the target-free raw census is not the fixed 16 x 5 cohort."""


@dataclass(frozen=True)
class RadarSession:
    info: FileInfo
    rel_path: str


def assert_target_free_columns(columns) -> None:
    lowered = [str(column).lower() for column in columns]
    offenders = [column for column in lowered if any(token in column for token in FORBIDDEN_RADAR_COLUMNS)]
    if offenders:
        raise RadarInventoryError(f"target-like columns are forbidden in radar-only data: {offenders}")


def build_radar_inventory(config: RadarAuditConfig) -> tuple[list[RadarSession], pd.DataFrame]:
    """Inspect every file and return the exact 8,000-row stored-index census."""
    data_dir = config.data_10ghz_dir
    if not data_dir.is_dir():
        raise RadarInventoryError(f"10 GHz data directory does not exist: {data_dir}")

    sessions: dict[tuple[int, int], RadarSession] = {}
    problems: list[str] = []
    for path in sorted(data_dir.glob("*.mat")):
        try:
            subject, session_idx = parse_10ghz_filename(path)
            info = inspect_10ghz_file(path)
        except LoaderError as exc:
            problems.append(str(exc))
            continue
        key = (subject, session_idx)
        if key in sessions:
            problems.append(f"duplicate file for subject/session {key}")
            continue
        sessions[key] = RadarSession(info=info, rel_path=path.relative_to(data_dir).as_posix())

    expected = {(subject, session) for subject in config.expected_subjects for session in range(len(SESSION_NAMES))}
    missing = sorted(expected - set(sessions))
    unexpected = sorted(set(sessions) - expected)
    if missing:
        problems.append(f"missing subject/session cells: {missing}")
    if unexpected:
        problems.append(f"unexpected subject/session cells: {unexpected}")
    for key, session in sessions.items():
        if session.info.n_frames != config.expected_frames_per_session:
            problems.append(
                f"subject/session {key} has {session.info.n_frames} frames; "
                f"expected {config.expected_frames_per_session}"
            )
    if problems:
        raise RadarInventoryError("target-free radar inventory failed:\n  " + "\n  ".join(problems))

    rows = []
    ordered_sessions = [sessions[key] for key in sorted(sessions)]
    for session in ordered_sessions:
        info = session.info
        for frame_idx in range(info.n_frames):
            rows.append(
                {
                    "subject": info.subject,
                    "session_idx": info.session_idx,
                    "session_name": SESSION_NAMES[info.session_idx],
                    "rel_path": session.rel_path,
                    "n_frames_in_file": info.n_frames,
                    "frame_idx": frame_idx,
                }
            )
    inventory = pd.DataFrame(rows).sort_values(["subject", "session_idx", "frame_idx"]).reset_index(drop=True)
    assert_target_free_columns(inventory.columns)
    if inventory.duplicated(["rel_path", "frame_idx"]).any():
        raise RadarInventoryError("duplicate (rel_path, frame_idx) in target-free inventory")
    return ordered_sessions, inventory


def load_session_and_qc(session: RadarSession, config: RadarAuditConfig):
    """Load one cube and return it with its one-row-per-frame frozen QC table."""
    cube = load_10ghz_file(session.info.path)
    verdicts = run_qc_cube(cube, config.qc, config.preprocess)
    rows = []
    for frame_idx, verdict in enumerate(verdicts):
        rows.append(
            {
                "subject": session.info.subject,
                "session_idx": session.info.session_idx,
                "session_name": SESSION_NAMES[session.info.session_idx],
                "rel_path": session.rel_path,
                "frame_idx": frame_idx,
                "qc_nan_inf": verdict.nan_inf,
                "qc_flatline": verdict.flatline,
                "qc_low_in_band": verdict.low_in_band,
                "qc_rms_flag": verdict.rms_flag,
                "qc_pass": verdict.passed,
                "qc_in_band_ratio": verdict.in_band_ratio,
                "qc_max_rms_z": verdict.max_rms_z,
                "qc_n_flatline_chirps": verdict.n_flatline_chirps,
                "qc_n_rms_outlier_chirps": verdict.n_rms_outlier_chirps,
            }
        )
    frame_qc = pd.DataFrame(rows)
    assert_target_free_columns(frame_qc.columns)
    return cube, frame_qc

