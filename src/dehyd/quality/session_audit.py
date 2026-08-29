"""Plain scientific computations for the 10 GHz session-quality audit.

Nothing in this module fits a population model or sees a hydration target.  A session
is summarized independently, and WST comparisons are made only within that session.
The separate recorded-equal-mass helper accepts an already parsed mass table only after
the caller has finalized the radar-only artifacts.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import platform
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.pooling import pool_stats_batch
from ..features.wst import build_scattering, scatter_frames, scattering_shape
from ..preprocess.filters import apply_band_gate
from ..preprocess.pipeline import preprocess_cube
from ..preprocess.reduce import detect_option_b_peak
from ..provenance import _cpu_model, _git_info, _package_versions, sha256_file
from .config import RadarAuditConfig, radar_config_record

VIEWS = ("within_path_shape", "path_energy_composition")
CHANNELS = ("mag", "iq")


class SessionAuditError(ValueError):
    """Raised when a session or dimensionless representation violates its contract."""


def capture_clean_git_provenance() -> dict:
    """Capture the committed implementation before this audit creates any outputs."""
    git_record = dict(_git_info())
    commit = git_record.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise SessionAuditError("quality audit requires a real 40-character Git commit")
    if git_record.get("dirty") is not False:
        raise SessionAuditError("quality audit requires a clean source tree before output creation")
    return git_record


@dataclass
class ViewGeometry:
    """Radar-only geometry retained in memory for the separated pair analysis."""

    frame_indices: np.ndarray
    block_ids: np.ndarray
    session_centroid: np.ndarray
    block_centroids: np.ndarray
    block_wobbles: np.ndarray
    epsilon_distance: float
    # For within-path shape only: unweighted unit-normalized blocks [N, P, K].
    normalized_path_blocks: np.ndarray | None
    active_paths: np.ndarray
    n_near_zero_path_blocks: int
    n_inactive_paths: int
    epsilon_path: float


def unscaled_mad(values) -> float:
    """Median absolute deviation without the Gaussian 1.4826 scale factor."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def assign_stored_index_blocks(
    frame_indices, *, expected_frames: int, frames_per_block: int, n_blocks: int
) -> np.ndarray:
    """Map generated stored-axis indices to five fixed blocks, independent of row order."""
    indices = np.asarray(frame_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise SessionAuditError("frame_indices must be a one-dimensional integer array")
    if expected_frames != frames_per_block * n_blocks:
        raise SessionAuditError("block declaration does not exactly cover the expected frame count")
    if indices.size and (indices.min() < 0 or indices.max() >= expected_frames):
        raise SessionAuditError("frame index lies outside the fixed stored-index census")
    return indices.astype(np.int64) // frames_per_block


def _energy_retention(raw_frame: np.ndarray, gated_frame: np.ndarray) -> float:
    """Same E_post/E_pre definition as ``experiments/run_preprocess.py``."""
    raw_energy = float(np.sum(np.abs(raw_frame) ** 2))
    if raw_energy == 0.0:
        raise SessionAuditError("QC-passing frame has zero raw energy")
    return float(np.sum(np.abs(gated_frame) ** 2)) / raw_energy


def _concentration(detection) -> tuple[float, float]:
    """Same ROI/total and peak/ROI definitions as the frozen preprocessing diagnostic."""
    total_power = float(detection.power.sum())
    roi_power = float(detection.power[detection.roi_bins].sum())
    roi_to_total = roi_power / total_power if total_power > 0.0 else 0.0
    peak_share = (
        float(detection.power[detection.peak_bin] / roi_power)
        if roi_power > 0.0
        else float("nan")
    )
    return roi_to_total, peak_share


def compute_frame_components(
    cube: np.ndarray, frame_qc: pd.DataFrame, config: RadarAuditConfig
) -> pd.DataFrame:
    """Attach raw-level and preprocessing measurements to one session's QC rows."""
    if cube.ndim != 3 or cube.shape[2] != len(frame_qc):
        raise SessionAuditError("cube and frame-QC row counts disagree")
    rows = frame_qc.sort_values("frame_idx").copy()
    expected = np.arange(cube.shape[2], dtype=np.int64)
    if not np.array_equal(rows["frame_idx"].to_numpy(dtype=np.int64), expected):
        raise SessionAuditError("frame_idx must be unique and contiguous before component work")
    rows["block_idx"] = assign_stored_index_blocks(
        expected,
        expected_frames=config.expected_frames_per_session,
        frames_per_block=config.frames_per_block,
        n_blocks=config.n_blocks,
    )

    finite_raw = np.all(np.isfinite(cube), axis=(0, 1))
    raw_rms = np.full(cube.shape[2], np.nan, dtype=np.float64)
    if finite_raw.any():
        finite_cube = cube[:, :, finite_raw]
        raw_rms[finite_raw] = np.sqrt(np.mean(np.abs(finite_cube) ** 2, axis=(0, 1)))
    rows["raw_rms"] = raw_rms

    for column in (
        "energy_retention",
        "roi_to_total",
        "peak_share",
        "peak_bin",
        "peak_range_m",
    ):
        rows[column] = np.nan

    hz_per_m = 2.0 * (config.preprocess.bandwidth_hz / config.preprocess.chirp_time_s) / 299_792_458.0
    df_hz = config.preprocess.fs_hz / cube.shape[0]
    for frame_idx in rows.loc[rows["qc_pass"], "frame_idx"].astype(int):
        frame = cube[:, :, frame_idx]
        gated = apply_band_gate(frame, config.preprocess, axis=0)
        detection = detect_option_b_peak(gated, config.preprocess)
        roi_to_total, peak_share = _concentration(detection)
        rows.loc[rows["frame_idx"] == frame_idx, "energy_retention"] = _energy_retention(frame, gated)
        rows.loc[rows["frame_idx"] == frame_idx, "roi_to_total"] = roi_to_total
        rows.loc[rows["frame_idx"] == frame_idx, "peak_share"] = peak_share
        rows.loc[rows["frame_idx"] == frame_idx, "peak_bin"] = detection.peak_bin
        rows.loc[rows["frame_idx"] == frame_idx, "peak_range_m"] = detection.peak_bin * df_hz / hz_per_m
    return rows.sort_values("frame_idx").reset_index(drop=True)


def _finite_stat(values, operation: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    if operation == "median":
        return float(np.median(array))
    if operation == "p10":
        return float(np.percentile(array, 10))
    if operation == "iqr":
        return float(np.percentile(array, 75) - np.percentile(array, 25))
    raise SessionAuditError(f"unknown summary operation {operation!r}")


def _mode_share(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    _, counts = np.unique(array, return_counts=True)
    return float(counts.max() / array.size)


def _lowest_mode(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    uniques, counts = np.unique(array, return_counts=True)
    return float(uniques[int(np.argmax(counts))])


def _component_summary(frame_table: pd.DataFrame) -> dict:
    in_band = frame_table["qc_in_band_ratio"]
    p10 = _finite_stat(in_band, "p10")
    return {
        "raw_rms_median": _finite_stat(frame_table["raw_rms"], "median"),
        "raw_rms_mad": unscaled_mad(frame_table["raw_rms"]),
        "in_band_ratio_median": _finite_stat(in_band, "median"),
        "in_band_ratio_p10": p10,
        "in_band_ratio_p10_margin": p10,
        "energy_retention_median": _finite_stat(frame_table["energy_retention"], "median"),
        "energy_retention_mad": unscaled_mad(frame_table["energy_retention"]),
        "roi_to_total_median": _finite_stat(frame_table["roi_to_total"], "median"),
        "roi_to_total_mad": unscaled_mad(frame_table["roi_to_total"]),
        "peak_share_median": _finite_stat(frame_table["peak_share"], "median"),
        "peak_share_mad": unscaled_mad(frame_table["peak_share"]),
        "peak_bin_median": _finite_stat(frame_table["peak_bin"], "median"),
        "peak_bin_mad": unscaled_mad(frame_table["peak_bin"]),
        "peak_bin_iqr": _finite_stat(frame_table["peak_bin"], "iqr"),
        "peak_bin_mode": _lowest_mode(frame_table["peak_bin"]),
        "peak_bin_mode_share": _mode_share(frame_table["peak_bin"]),
        "peak_range_m_median": _finite_stat(frame_table["peak_range_m"], "median"),
        "peak_range_m_mad": unscaled_mad(frame_table["peak_range_m"]),
    }


def _component_missing_reasons(summary: dict, *, n_pass: int, n_finite_raw: int) -> str:
    """Name every undefined component instead of leaving an unexplained CSV blank."""
    reasons = {}
    for column, value in summary.items():
        if pd.notna(value):
            continue
        if column.startswith("raw_rms"):
            reason = "no_finite_raw_frames" if n_finite_raw == 0 else "undefined_raw_summary"
        elif column.startswith("in_band"):
            reason = "qc_diagnostic_undefined"
        elif n_pass == 0:
            reason = "no_qc_passing_frames"
        elif column.startswith("peak_share"):
            reason = "zero_roi_power_or_undefined_peak_share"
        else:
            reason = "undefined_preprocessing_component"
        reasons[column] = reason
    return json.dumps(reasons, sort_keys=True, separators=(",", ":"))


def summarize_session_components(
    frame_table: pd.DataFrame, config: RadarAuditConfig
) -> tuple[dict, list[dict]]:
    """Return the one-row session card and five stored-index block rows."""
    identity = {
        "subject": int(frame_table["subject"].iloc[0]),
        "session_idx": int(frame_table["session_idx"].iloc[0]),
        "session_name": str(frame_table["session_name"].iloc[0]),
        "rel_path": str(frame_table["rel_path"].iloc[0]),
    }
    n_raw = len(frame_table)
    n_pass = int(frame_table["qc_pass"].sum())
    min_pass = math.ceil(config.qc.min_frame_fraction * n_raw)
    eligible = n_pass >= min_pass
    block_counts = [
        int(frame_table.loc[frame_table["block_idx"] == block, "qc_pass"].sum())
        for block in range(config.n_blocks)
    ]
    min_block = min(block_counts)
    if not eligible:
        status = "INELIGIBLE_EXISTING_QC"
        repeatability_reason = "ineligible_existing_qc"
    elif min_block < config.min_passing_frames_per_block:
        status = "REVIEW_BLOCK_COVERAGE"
        repeatability_reason = "insufficient_block_coverage"
    else:
        status = "REPEATABILITY_ANALYSABLE"
        repeatability_reason = ""

    defined_qc = frame_table["qc_in_band_ratio"].notna()
    component_summary = _component_summary(frame_table)
    card = {
        **identity,
        "n_raw": n_raw,
        "n_pass": n_pass,
        "pass_fraction": n_pass / n_raw,
        "fail_fraction": (n_raw - n_pass) / n_raw,
        "min_pass_existing_qc": min_pass,
        "eligible_existing_qc": eligible,
        "n_fail_any": n_raw - n_pass,
        "n_nan_inf": int(frame_table["qc_nan_inf"].sum()),
        "n_flatline": int(frame_table["qc_flatline"].sum()),
        "n_low_in_band": int(frame_table["qc_low_in_band"].sum()),
        "n_rms_flagged": int(frame_table["qc_rms_flag"].sum()),
        "rms_flag_fraction_defined": (
            float(frame_table.loc[defined_qc, "qc_rms_flag"].mean()) if defined_qc.any() else np.nan
        ),
        "n_nonfinite_raw": int((~frame_table["raw_rms"].notna()).sum()),
        **{f"block_{block}_n_pass": count for block, count in enumerate(block_counts)},
        "minimum_block_n_pass": min_block,
        **component_summary,
        "component_missing_reasons": _component_missing_reasons(
            component_summary,
            n_pass=n_pass,
            n_finite_raw=int(frame_table["raw_rms"].notna().sum()),
        ),
        "audit_status": status,
        "repeatability_missing_reason": repeatability_reason,
    }
    # The margin uses the unchanged frozen threshold, not a new learned boundary.
    card["in_band_ratio_p10_margin"] = card["in_band_ratio_p10"] - config.qc.min_in_band_energy_ratio

    block_rows = []
    for block in range(config.n_blocks):
        subset = frame_table[frame_table["block_idx"] == block]
        block_summary = _component_summary(subset)
        block_row = {
            **identity,
            "block_idx": block,
            "stored_frame_start": block * config.frames_per_block,
            "stored_frame_stop_exclusive": (block + 1) * config.frames_per_block,
            "n_raw": len(subset),
            "n_pass": int(subset["qc_pass"].sum()),
            "pass_fraction": float(subset["qc_pass"].mean()),
            "n_nan_inf": int(subset["qc_nan_inf"].sum()),
            "n_flatline": int(subset["qc_flatline"].sum()),
            "n_low_in_band": int(subset["qc_low_in_band"].sum()),
            "n_rms_flagged": int(subset["qc_rms_flag"].sum()),
            **block_summary,
            "component_missing_reasons": _component_missing_reasons(
                block_summary,
                n_pass=int(subset["qc_pass"].sum()),
                n_finite_raw=int(subset["raw_rms"].notna().sum()),
            ),
        }
        block_row["in_band_ratio_p10_margin"] = (
            block_row["in_band_ratio_p10"] - config.qc.min_in_band_energy_ratio
        )
        block_rows.append(block_row)
    return card, block_rows


def build_diagnostic_scattering(config: RadarAuditConfig) -> tuple:
    """Instantiate the three frozen numpy WST banks once for reuse across sessions."""
    n_input = 534 - 2 * config.preprocess.edge_trim
    banks = []
    for tiling in config.wst.tilings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scattering = build_scattering(
                tiling,
                config.wst,
                n_in=n_input,
                fs_hz=config.preprocess.fs_hz,
            )
            # Measuring output time is one zero-signal transform. Cache it on the
            # bank so missing-session rows do not repeat that work hundreds of times.
            scattering._dehyd_quality_shape = scattering_shape(scattering)
            banks.append(scattering)
    return tuple(banks)


def _pooled_path_blocks(scattering_values: np.ndarray, meta) -> np.ndarray:
    """Raw WST [N,C,P,T] -> pooled blocks [N,P,C*K] in canonical path order."""
    pooled = pool_stats_batch(scattering_values, meta)
    n_frames, n_channels, n_paths, _ = scattering_values.shape
    if pooled.shape[1] % (n_channels * n_paths) != 0:
        raise SessionAuditError("pooled dimension is not divisible into canonical path blocks")
    n_statistics = pooled.shape[1] // (n_channels * n_paths)
    return pooled.reshape(n_frames, n_channels, n_paths, n_statistics).transpose(0, 2, 1, 3).reshape(
        n_frames, n_paths, n_channels * n_statistics
    )


def _dimensionless_views(path_blocks: np.ndarray, epsilon_factor: float) -> tuple[dict, dict]:
    """Construct the approved within-path and across-path dimensionless views."""
    path_norms = np.linalg.norm(path_blocks, axis=-1)
    positive = path_norms[path_norms > 0.0]
    if positive.size == 0:
        n_paths = path_blocks.shape[1]
        return {}, {
            "missing_reason": "no_positive_path_norm",
            "epsilon_path": 0.0,
            "n_near_zero_path_blocks": int(path_norms.size),
            "n_inactive_paths": n_paths,
            "active_paths": np.zeros(n_paths, dtype=bool),
        }
    epsilon_path = epsilon_factor * float(np.median(positive))
    near_zero = path_norms <= epsilon_path
    normalized_blocks = np.zeros_like(path_blocks, dtype=np.float64)
    np.divide(
        path_blocks,
        path_norms[:, :, None],
        out=normalized_blocks,
        where=~near_zero[:, :, None],
    )
    n_paths = path_blocks.shape[1]
    shape_vectors = normalized_blocks.reshape(path_blocks.shape[0], -1) / math.sqrt(n_paths)

    floored_norms = path_norms.copy()
    floored_norms[near_zero] = 0.0
    composition_norm = np.linalg.norm(floored_norms, axis=1)
    composition = np.zeros_like(floored_norms)
    np.divide(
        floored_norms,
        composition_norm[:, None],
        out=composition,
        where=composition_norm[:, None] > 0.0,
    )
    active_paths = np.median(path_norms, axis=0) > epsilon_path
    common = {
        "epsilon_path": epsilon_path,
        "n_near_zero_path_blocks": int(near_zero.sum()),
        "n_inactive_paths": int((~active_paths).sum()),
        "active_paths": active_paths,
    }
    return {
        "within_path_shape": (shape_vectors, normalized_blocks),
        "path_energy_composition": (composition, None),
    }, common


def _positive_median(values) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0.0]
    return float(np.median(positive)) if positive.size else None


def _safe_ratio(numerator: float, denominator: float) -> tuple[float, bool, str]:
    if numerator == 0.0 and denominator == 0.0:
        return 0.0, True, ""
    if denominator == 0.0:
        return float("nan"), False, "zero_within_block_wobble_nonzero_between"
    return numerator / denominator, False, ""


def _json(values) -> str:
    def clean(value):
        value = float(value)
        return None if not np.isfinite(value) else value

    return json.dumps([clean(value) for value in values], separators=(",", ":"))


def summarize_repeatability_vectors(
    vectors: np.ndarray,
    frame_indices: np.ndarray,
    config: RadarAuditConfig,
    *,
    normalized_path_blocks: np.ndarray | None,
    active_paths: np.ndarray,
    n_near_zero_path_blocks: int,
    n_inactive_paths: int,
    epsilon_path: float,
) -> tuple[dict, ViewGeometry]:
    """Compute five-block typical and worst-block geometry for one fixed cell/view."""
    vectors = np.asarray(vectors, dtype=np.float64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if vectors.ndim != 2 or vectors.shape[0] != frame_indices.size:
        raise SessionAuditError("repeatability vectors must be [N,D] and row-aligned to frame indices")
    block_ids = assign_stored_index_blocks(
        frame_indices,
        expected_frames=config.expected_frames_per_session,
        frames_per_block=config.frames_per_block,
        n_blocks=config.n_blocks,
    )
    session_centroid = np.median(vectors, axis=0)
    frame_session_distances = np.linalg.norm(vectors - session_centroid, axis=1)
    positive_distance = _positive_median(frame_session_distances)
    epsilon_distance = 0.0 if positive_distance is None else config.epsilon_distance_factor * positive_distance
    session_wobble = float(np.median(frame_session_distances))

    block_centroids = []
    block_wobbles = []
    for block in range(config.n_blocks):
        block_vectors = vectors[block_ids == block]
        if block_vectors.size == 0:
            raise SessionAuditError("repeatability received an empty stored-index block")
        centroid = np.median(block_vectors, axis=0)
        block_centroids.append(centroid)
        block_wobbles.append(float(np.median(np.linalg.norm(block_vectors - centroid, axis=1))))
    block_centroids_array = np.stack(block_centroids)
    block_wobbles_array = np.asarray(block_wobbles)

    cosine_values, pair_numerators, pair_denominators, separation_values = [], [], [], []
    identical_flags = []
    missing_reasons: list[str] = []
    for first, second in itertools.combinations(range(config.n_blocks), 2):
        a, b = block_centroids_array[first], block_centroids_array[second]
        norm_product = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm_product == 0.0:
            cosine_values.append(float("nan"))
            missing_reasons.append("zero_centroid_norm")
        else:
            cosine_values.append(float(np.dot(a, b) / norm_product))
        numerator = float(np.linalg.norm(a - b))
        denominator = math.sqrt(
            block_wobbles_array[first] ** 2
            + block_wobbles_array[second] ** 2
            + epsilon_distance**2
        )
        ratio, identical, reason = _safe_ratio(numerator, denominator)
        pair_numerators.append(numerator)
        pair_denominators.append(denominator)
        separation_values.append(ratio)
        identical_flags.append(identical)
        if reason:
            missing_reasons.append(reason)

    block_session_distances = np.linalg.norm(block_centroids_array - session_centroid, axis=1)
    leave_numerators, leave_denominators, leave_values, leave_identical = [], [], [], []
    leave_denominator = math.sqrt(session_wobble**2 + epsilon_distance**2)
    for block in range(config.n_blocks):
        leave_centroid = np.median(vectors[block_ids != block], axis=0)
        numerator = float(np.linalg.norm(leave_centroid - session_centroid))
        ratio, identical, reason = _safe_ratio(numerator, leave_denominator)
        leave_numerators.append(numerator)
        leave_denominators.append(leave_denominator)
        leave_values.append(ratio)
        leave_identical.append(identical)
        if reason:
            missing_reasons.append(reason)

    finite_cos = np.asarray(cosine_values)[np.isfinite(cosine_values)]
    finite_sep = np.asarray(separation_values)[np.isfinite(separation_values)]
    finite_leave = np.asarray(leave_values)[np.isfinite(leave_values)]
    row = {
        "vector_dimension": int(vectors.shape[1]),
        "n_paths": int(active_paths.size),
        "n_near_zero_path_blocks": n_near_zero_path_blocks,
        "n_inactive_paths": n_inactive_paths,
        "epsilon_path": epsilon_path,
        "epsilon_distance": epsilon_distance,
        "session_wobble": session_wobble,
        "usable_block_count": config.n_blocks,
        "cosine_similarity_median": float(np.median(finite_cos)) if finite_cos.size else np.nan,
        "cosine_similarity_minimum": float(np.min(finite_cos)) if finite_cos.size else np.nan,
        "separation_to_wobble_median": float(np.median(finite_sep)) if finite_sep.size else np.nan,
        "separation_to_wobble_maximum": float(np.max(finite_sep)) if finite_sep.size else np.nan,
        "block_to_session_distance_maximum": float(np.max(block_session_distances)),
        "max_leave_one_block_influence": float(np.max(finite_leave)) if finite_leave.size else np.nan,
        "pair_cosine_values": _json(cosine_values),
        "pair_distance_numerators": _json(pair_numerators),
        "pair_separation_denominators": _json(pair_denominators),
        "pair_separation_values": _json(separation_values),
        "pair_exactly_identical": json.dumps(identical_flags, separators=(",", ":")),
        "block_to_session_distances": _json(block_session_distances),
        "leave_one_block_numerators": _json(leave_numerators),
        "leave_one_block_denominators": _json(leave_denominators),
        "leave_one_block_values": _json(leave_values),
        "leave_one_block_exactly_identical": json.dumps(leave_identical, separators=(",", ":")),
        "missing_reason": ";".join(sorted(set(missing_reasons))),
    }
    geometry = ViewGeometry(
        frame_indices=frame_indices,
        block_ids=block_ids,
        session_centroid=session_centroid,
        block_centroids=block_centroids_array,
        block_wobbles=block_wobbles_array,
        epsilon_distance=epsilon_distance,
        normalized_path_blocks=normalized_path_blocks,
        active_paths=active_paths,
        n_near_zero_path_blocks=n_near_zero_path_blocks,
        n_inactive_paths=n_inactive_paths,
        epsilon_path=epsilon_path,
    )
    return row, geometry


def _repeatability_identity(card: dict, channel: str, tiling_idx: int, order: int, view: str, config) -> dict:
    tiling = config.wst.tilings[tiling_idx]
    return {
        "subject": card["subject"],
        "session_idx": card["session_idx"],
        "session_name": card["session_name"],
        "rel_path": card["rel_path"],
        "diagnostic_channel": channel,
        "diagnostic_role": "primary" if channel == "mag" else "sensitivity",
        "tiling_idx": tiling_idx,
        "q1": tiling.q[0],
        "q2": tiling.q[1],
        "invariance_ms": tiling.invariance_ms,
        "scattering_order": order,
        "view": view,
    }


def empty_repeatability_rows(
    card: dict, config: RadarAuditConfig, scattering_banks: tuple | None = None
) -> list[dict]:
    rows = []
    reason = card["repeatability_missing_reason"] or "not_computed"
    shapes = None
    if scattering_banks is not None:
        shapes = [bank._dehyd_quality_shape for bank in scattering_banks]
    for channel, tiling_idx, order, view in itertools.product(
        CHANNELS, range(len(config.wst.tilings)), (0, 1, 2), VIEWS
    ):
        row = _repeatability_identity(card, channel, tiling_idx, order, view, config)
        n_paths = np.nan
        vector_dimension = np.nan
        if shapes is not None:
            shape = shapes[tiling_idx]
            order_array = np.asarray(shape["order"])
            n_paths = int(np.sum(order_array == order))
            n_time = int(shape["n_time"])
            first_length = n_time // 2
            second_length = n_time - first_length
            # Three means plus each segment whose length supports a standard deviation.
            n_statistics = 3 + int(n_time >= 2) + int(first_length >= 2) + int(second_length >= 2)
            if view == "within_path_shape":
                vector_dimension = n_paths * n_statistics * (1 if channel == "mag" else 2)
            else:
                vector_dimension = n_paths
        row.update(
            {
                "vector_dimension": vector_dimension,
                "n_paths": n_paths,
                "n_near_zero_path_blocks": np.nan,
                "n_inactive_paths": np.nan,
                "epsilon_path": np.nan,
                "epsilon_distance": np.nan,
                "session_wobble": np.nan,
                "usable_block_count": 0,
                "cosine_similarity_median": np.nan,
                "cosine_similarity_minimum": np.nan,
                "separation_to_wobble_median": np.nan,
                "separation_to_wobble_maximum": np.nan,
                "block_to_session_distance_maximum": np.nan,
                "max_leave_one_block_influence": np.nan,
                "pair_cosine_values": "[]",
                "pair_distance_numerators": "[]",
                "pair_separation_denominators": "[]",
                "pair_separation_values": "[]",
                "pair_exactly_identical": "[]",
                "block_to_session_distances": "[]",
                "leave_one_block_numerators": "[]",
                "leave_one_block_denominators": "[]",
                "leave_one_block_values": "[]",
                "leave_one_block_exactly_identical": "[]",
                "missing_reason": (
                    "not_applicable_single_path" if view == "path_energy_composition" and order == 0 else reason
                ),
            }
        )
        rows.append(row)
    return rows


def compute_session_wst_repeatability(
    cube: np.ndarray,
    passing_indices,
    card: dict,
    config: RadarAuditConfig,
    scattering_banks: tuple,
) -> tuple[list[dict], dict[tuple, ViewGeometry]]:
    """Compute all fixed magnitude-primary and I/Q-sensitivity WST cells."""
    if card["audit_status"] != "REPEATABILITY_ANALYSABLE":
        return empty_repeatability_rows(card, config, scattering_banks), {}
    passing_indices = np.asarray(sorted(int(index) for index in passing_indices), dtype=np.int64)
    qc_cube = cube[:, :, passing_indices]
    rows: list[dict] = []
    geometries: dict[tuple, ViewGeometry] = {}
    empty_templates = {
        (row["diagnostic_channel"], row["tiling_idx"], row["scattering_order"], row["view"]): row
        for row in empty_repeatability_rows(card, config, scattering_banks)
    }

    for channel in CHANNELS:
        preprocessed = preprocess_cube(qc_cube, config.preprocess, reduction="a", channel=channel)
        for tiling_idx, scattering in enumerate(scattering_banks):
            raw_scattering = scatter_frames(preprocessed, scattering)
            meta = scattering.meta()
            order_array = np.asarray(meta["order"])
            pooled_blocks = _pooled_path_blocks(raw_scattering, meta)
            for order in (0, 1, 2):
                order_blocks = pooled_blocks[:, order_array == order, :]
                views, common = _dimensionless_views(order_blocks, config.epsilon_path_factor)
                for view in VIEWS:
                    identity = _repeatability_identity(card, channel, tiling_idx, order, view, config)
                    if view == "path_energy_composition" and order_blocks.shape[1] < 2:
                        row = dict(empty_templates[(channel, tiling_idx, order, view)])
                        row.update(
                            {
                                "n_near_zero_path_blocks": common["n_near_zero_path_blocks"],
                                "n_inactive_paths": common["n_inactive_paths"],
                                "epsilon_path": common["epsilon_path"],
                            }
                        )
                        rows.append(row)
                        continue
                    if not views:
                        row = dict(empty_templates[(channel, tiling_idx, order, view)])
                        row["missing_reason"] = common["missing_reason"]
                        rows.append(row)
                        continue
                    vectors, normalized_blocks = views[view]
                    metrics, geometry = summarize_repeatability_vectors(
                        vectors,
                        passing_indices,
                        config,
                        normalized_path_blocks=normalized_blocks,
                        active_paths=common["active_paths"],
                        n_near_zero_path_blocks=common["n_near_zero_path_blocks"],
                        n_inactive_paths=common["n_inactive_paths"],
                        epsilon_path=common["epsilon_path"],
                    )
                    rows.append({**identity, **metrics})
                    geometries[(channel, tiling_idx, order, view)] = geometry
    if len(rows) != len(CHANNELS) * len(config.wst.tilings) * 3 * len(VIEWS):
        raise SessionAuditError(f"repeatability schema produced {len(rows)} rows, expected 36")
    return rows, geometries


RANK_DIRECTIONS = {
    "pass_fraction": "higher_is_steadier",
    "fail_fraction": "lower_is_steadier",
    "in_band_ratio_p10_margin": "higher_is_steadier",
    "peak_bin_mode_share": "higher_is_steadier",
    "cosine_similarity_median": "higher_is_steadier",
    "cosine_similarity_minimum": "higher_is_steadier",
    "rms_flag_fraction_defined": "lower_is_steadier",
    "peak_bin_iqr": "lower_is_steadier",
    "separation_to_wobble_median": "lower_is_steadier",
    "separation_to_wobble_maximum": "lower_is_steadier",
    "block_to_session_distance_maximum": "lower_is_steadier",
    "max_leave_one_block_influence": "lower_is_steadier",
}


def build_subject_relative_table(cards: pd.DataFrame, repeatability: pd.DataFrame) -> pd.DataFrame:
    """Long-form raw component values plus deterministic dense within-subject ranks."""
    identity = ["subject", "session_idx", "session_name"]
    excluded = set(identity + ["rel_path", "audit_status", "repeatability_missing_reason"])
    rows = []
    for card in cards.to_dict("records"):
        for metric, value in card.items():
            if metric in excluded or isinstance(value, (str, bool)):
                continue
            rows.append(
                {
                    **{key: card[key] for key in identity},
                    "diagnostic_channel": "",
                    "tiling_idx": np.nan,
                    "scattering_order": np.nan,
                    "view": "",
                    "metric": metric,
                    "value": value,
                    "rank_direction": RANK_DIRECTIONS.get(metric, "not_ranked"),
                    "missing_reason": "" if pd.notna(value) else "undefined_component",
                }
            )
    repeat_metrics = [
        "cosine_similarity_median",
        "cosine_similarity_minimum",
        "separation_to_wobble_median",
        "separation_to_wobble_maximum",
        "block_to_session_distance_maximum",
        "max_leave_one_block_influence",
        "n_near_zero_path_blocks",
        "n_inactive_paths",
    ]
    for record in repeatability.to_dict("records"):
        for metric in repeat_metrics:
            rows.append(
                {
                    **{key: record[key] for key in identity},
                    "diagnostic_channel": record["diagnostic_channel"],
                    "tiling_idx": record["tiling_idx"],
                    "scattering_order": record["scattering_order"],
                    "view": record["view"],
                    "metric": metric,
                    "value": record[metric],
                    "rank_direction": RANK_DIRECTIONS.get(metric, "not_ranked"),
                    "missing_reason": record["missing_reason"] if pd.isna(record[metric]) else "",
                }
            )
    table = pd.DataFrame(rows)
    group_keys = ["subject", "diagnostic_channel", "tiling_idx", "scattering_order", "view", "metric"]
    table["within_subject_dense_rank"] = np.nan
    for _, indices in table.groupby(group_keys, dropna=False).groups.items():
        subset = table.loc[indices]
        direction = subset["rank_direction"].iloc[0]
        if direction == "not_ranked":
            continue
        ascending = direction == "lower_is_steadier"
        table.loc[indices, "within_subject_dense_rank"] = subset["value"].rank(
            method="dense", ascending=ascending, na_option="keep"
        )
    return table.sort_values(group_keys + ["session_idx"], na_position="first").reset_index(drop=True)


def recorded_equal_mass_pairs(mass_sessions: pd.DataFrame) -> list[dict]:
    """Adjacent within-subject pairs whose recorded workbook masses are exactly equal."""
    pairs = []
    for subject, group in mass_sessions.sort_values(["subject", "session_idx"]).groupby("subject"):
        records = list(group.to_dict("records"))
        for first, second in zip(records, records[1:], strict=False):
            if second["session_idx"] != first["session_idx"] + 1:
                raise SessionAuditError(f"subject {subject} mass rows are not contiguous")
            if float(first["mass_kg"]) == float(second["mass_kg"]):
                pairs.append(
                    {
                        "subject": int(subject),
                        "session_a_idx": int(first["session_idx"]),
                        "session_a_name": first["session_name"],
                        "session_b_idx": int(second["session_idx"]),
                        "session_b_name": second["session_name"],
                        "recorded_mass_kg": float(first["mass_kg"]),
                    }
                )
    return pairs


def _geometry_from_common_shape(
    geometry: ViewGeometry, common_paths: np.ndarray, config: RadarAuditConfig
) -> ViewGeometry:
    blocks = geometry.normalized_path_blocks
    if blocks is None:
        raise SessionAuditError("common-path recomputation requires within-path blocks")
    selected = blocks[:, common_paths, :]
    vectors = selected.reshape(selected.shape[0], -1) / math.sqrt(selected.shape[1])
    _, recomputed = summarize_repeatability_vectors(
        vectors,
        geometry.frame_indices,
        config,
        normalized_path_blocks=selected,
        active_paths=np.ones(selected.shape[1], dtype=bool),
        n_near_zero_path_blocks=int(np.count_nonzero(np.linalg.norm(selected, axis=-1) == 0.0)),
        n_inactive_paths=0,
        epsilon_path=geometry.epsilon_path,
    )
    return recomputed


EQUAL_MASS_COLUMNS = [
    "subject",
    "session_a_idx",
    "session_a_name",
    "session_b_idx",
    "session_b_name",
    "recorded_mass_kg",
    "diagnostic_channel",
    "diagnostic_role",
    "tiling_idx",
    "scattering_order",
    "view",
    "between_session_distance",
    "between_distance_numerator",
    "within_a",
    "within_b",
    "epsilon_distance_a",
    "epsilon_distance_b",
    "between_to_within_denominator",
    "between_to_within_ratio",
    "exactly_identical",
    "common_path_count",
    "common_vector_dimension",
    "n_near_zero_path_blocks_a",
    "n_near_zero_path_blocks_b",
    "n_inactive_paths_a",
    "n_inactive_paths_b",
    "missing_reason",
]


def compare_adjacent_session_geometry(
    first_card: dict,
    second_card: dict,
    first_geometry: dict[tuple, ViewGeometry],
    second_geometry: dict[tuple, ViewGeometry],
    config: RadarAuditConfig,
) -> list[dict]:
    """Compare one adjacent radar pair without accepting mass or another outcome."""
    if first_card["subject"] != second_card["subject"]:
        raise SessionAuditError("adjacent radar comparison cannot cross subjects")
    if second_card["session_idx"] != first_card["session_idx"] + 1:
        raise SessionAuditError("adjacent radar comparison requires consecutive sessions")
    pair = {
        "subject": int(first_card["subject"]),
        "session_a_idx": int(first_card["session_idx"]),
        "session_a_name": str(first_card["session_name"]),
        "session_b_idx": int(second_card["session_idx"]),
        "session_b_name": str(second_card["session_name"]),
    }
    rows = []
    for channel, tiling_idx, order, view in itertools.product(
        CHANNELS, range(len(config.wst.tilings)), (0, 1, 2), VIEWS
    ):
        cell = (channel, tiling_idx, order, view)
        identity = {
            **pair,
            "diagnostic_channel": channel,
            "diagnostic_role": "primary" if channel == "mag" else "sensitivity",
            "tiling_idx": tiling_idx,
            "scattering_order": order,
            "view": view,
        }
        if view == "path_energy_composition" and order == 0:
            rows.append({**identity, "missing_reason": "not_applicable_single_path"})
            continue
        if cell not in first_geometry or cell not in second_geometry:
            rows.append({**identity, "missing_reason": "session_repeatability_unavailable"})
            continue
        first = first_geometry[cell]
        second = second_geometry[cell]
        original_first = first
        original_second = second
        if view == "within_path_shape":
            common = first.active_paths & second.active_paths
            if not common.any():
                rows.append({**identity, "missing_reason": "no_common_active_paths"})
                continue
            first = _geometry_from_common_shape(first, common, config)
            second = _geometry_from_common_shape(second, common, config)
            common_paths = int(common.sum())
        else:
            common_paths = int(first.active_paths.size)
        numerator = float(np.linalg.norm(first.session_centroid - second.session_centroid))
        within_first = float(
            np.median(
                [
                    np.linalg.norm(first.block_centroids[a] - first.block_centroids[b])
                    for a, b in itertools.combinations(range(config.n_blocks), 2)
                ]
            )
        )
        within_second = float(
            np.median(
                [
                    np.linalg.norm(second.block_centroids[a] - second.block_centroids[b])
                    for a, b in itertools.combinations(range(config.n_blocks), 2)
                ]
            )
        )
        denominator = math.sqrt(
            within_first**2
            + within_second**2
            + first.epsilon_distance**2
            + second.epsilon_distance**2
        )
        ratio, identical, reason = _safe_ratio(numerator, denominator)
        rows.append(
            {
                **identity,
                "between_session_distance": numerator,
                "between_distance_numerator": numerator,
                "within_a": within_first,
                "within_b": within_second,
                "epsilon_distance_a": first.epsilon_distance,
                "epsilon_distance_b": second.epsilon_distance,
                "between_to_within_denominator": denominator,
                "between_to_within_ratio": ratio,
                "exactly_identical": identical,
                "common_path_count": common_paths,
                "common_vector_dimension": int(first.session_centroid.size),
                "n_near_zero_path_blocks_a": first.n_near_zero_path_blocks,
                "n_near_zero_path_blocks_b": second.n_near_zero_path_blocks,
                "n_inactive_paths_a": original_first.n_inactive_paths,
                "n_inactive_paths_b": original_second.n_inactive_paths,
                "missing_reason": reason,
            }
        )
    return rows


def select_recorded_equal_mass_rows(
    mass_sessions: pd.DataFrame, adjacent_geometry: pd.DataFrame
) -> pd.DataFrame:
    """Apply the outcome-bearing exact-mass filter to finalized radar pair geometry."""
    selected = []
    for pair in recorded_equal_mass_pairs(mass_sessions):
        mask = (
            (adjacent_geometry["subject"] == pair["subject"])
            & (adjacent_geometry["session_a_idx"] == pair["session_a_idx"])
            & (adjacent_geometry["session_b_idx"] == pair["session_b_idx"])
        )
        pair_rows = adjacent_geometry.loc[mask].copy()
        if len(pair_rows) != 36:
            raise SessionAuditError("recorded-equal-mass pair lacks its 36 radar geometry cells")
        pair_rows["recorded_mass_kg"] = pair["recorded_mass_kg"]
        selected.append(pair_rows)
    if not selected:
        return pd.DataFrame(columns=EQUAL_MASS_COLUMNS)
    return pd.concat(selected, ignore_index=True).reindex(columns=EQUAL_MASS_COLUMNS)


def build_recorded_equal_mass_table(
    mass_sessions: pd.DataFrame,
    all_geometry: dict[tuple, dict[tuple, ViewGeometry]],
    config: RadarAuditConfig,
) -> pd.DataFrame:
    """Pure small-fixture convenience wrapper around the streaming production path."""
    adjacent_rows = []
    for subject in sorted(mass_sessions["subject"].unique()):
        subject_rows = mass_sessions[mass_sessions["subject"] == subject].sort_values("session_idx")
        records = subject_rows.to_dict("records")
        for first, second in zip(records, records[1:], strict=False):
            first_card = {
                "subject": subject,
                "session_idx": first["session_idx"],
                "session_name": first["session_name"],
            }
            second_card = {
                "subject": subject,
                "session_idx": second["session_idx"],
                "session_name": second["session_name"],
            }
            adjacent_rows.extend(
                compare_adjacent_session_geometry(
                    first_card,
                    second_card,
                    all_geometry.get((subject, first["session_idx"]), {}),
                    all_geometry.get((subject, second["session_idx"]), {}),
                    config,
                )
            )
    adjacent = pd.DataFrame(adjacent_rows)
    return select_recorded_equal_mass_rows(mass_sessions, adjacent)


def ensure_write_within(path: str | Path, approved_roots) -> Path:
    """Fail closed unless a prospective runtime write is below an approved root."""
    resolved = Path(path).resolve()
    roots = [Path(root).resolve() for root in approved_roots]
    if not any(resolved.is_relative_to(root) for root in roots):
        raise SessionAuditError(f"audit write is outside approved roots: {resolved}")
    return resolved


def require_fresh_output_roots(config: RadarAuditConfig) -> None:
    """Never overwrite a previous audit; archive it before intentionally rerunning."""
    for root in (config.output_results_dir, config.output_figures_dir):
        if root.exists():
            raise SessionAuditError(f"audit output root is not fresh (already exists): {root}")


def write_csv(frame: pd.DataFrame, path: str | Path, config: RadarAuditConfig) -> Path:
    """Write a same-machine deterministic CSV and verify its row count on reread."""
    path = ensure_write_within(path, (config.output_results_dir, config.output_figures_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    )
    reread = pd.read_csv(path)
    if len(reread) != len(frame):
        raise SessionAuditError(f"CSV row loss at {path}: {len(frame)} -> {len(reread)}")
    return path


def write_json(payload: dict, path: str | Path, config: RadarAuditConfig) -> Path:
    path = ensure_write_within(path, (config.output_results_dir, config.output_figures_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def snapshot_existing_results(config: RadarAuditConfig) -> dict:
    """Hash every pre-audit local result byte; never dereference remote pointer text."""
    root = config.existing_results_dir.resolve()
    excluded = config.output_results_dir.resolve()
    files = []
    if root.exists():
        for path in sorted((candidate for candidate in root.rglob("*") if candidate.is_file())):
            resolved = path.resolve()
            if resolved.is_relative_to(excluded):
                continue
            stat = path.stat()
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": "frozen_local_results_v1",
        "root": str(root),
        "files": files,
        "absent_remote_artifacts": [
            {"name": name, "status": "not_locally_present"}
            for name in config.absent_remote_artifacts
        ],
    }


def verify_frozen_results(snapshot: dict, config: RadarAuditConfig) -> None:
    """Require the complete pre-audit result-file set and every byte to be unchanged."""
    after = snapshot_existing_results(config)
    expected_by_path = {entry["path"]: entry for entry in snapshot["files"]}
    actual_by_path = {entry["path"]: entry for entry in after["files"]}
    problems = []
    for rel_path in sorted(set(expected_by_path) - set(actual_by_path)):
        problems.append(f"missing: {rel_path}")
    for rel_path in sorted(set(actual_by_path) - set(expected_by_path)):
        problems.append(f"unexpected new file: {rel_path}")
    for rel_path in sorted(set(expected_by_path) & set(actual_by_path)):
        expected = expected_by_path[rel_path]
        actual = actual_by_path[rel_path]
        for key in ("size_bytes", "mtime_ns", "sha256"):
            if actual[key] != expected[key]:
                problems.append(f"{rel_path}: {key} changed")
    if problems:
        raise SessionAuditError("frozen result neutrality failed:\n  " + "\n  ".join(problems))


def build_radar_provenance(
    config: RadarAuditConfig,
    *,
    raw_sessions,
    source_artifacts,
    radar_artifacts,
    census: dict,
    git_record: dict,
) -> dict:
    """Target-free provenance: intentionally never accepts or records a workbook."""
    radar_files = [
        {"rel_path": session.rel_path, "sha256": sha256_file(session.info.path)}
        for session in raw_sessions
    ]
    return {
        "schema_version": "quality_10ghz_radar_provenance_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "descriptive_radar_only_quality_audit",
        "config": radar_config_record(config),
        "inputs": {"radar_files": radar_files, "source_artifacts": source_artifacts},
        "outputs": radar_artifacts,
        "census": census,
        "git": dict(git_record),
        "packages": _package_versions(),
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "interpretation": (
            "High repeatability means stable radar structure within an acquisition; "
            "it does not prove hydration validity and does not change sample eligibility."
        ),
    }


def _save_figure(fig, path: Path, config: RadarAuditConfig) -> Path:
    path = ensure_write_within(path, (config.output_results_dir, config.output_figures_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", metadata={"Software": "dehyd quality audit"})
    return path


def plot_session_component_heatmap(cards: pd.DataFrame, path: Path, config: RadarAuditConfig) -> Path:
    """Six separate raw-unit panels; components are never collapsed into one score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("pass_fraction", "QC pass fraction"),
        ("rms_flag_fraction_defined", "RMS-flag fraction"),
        ("in_band_ratio_p10_margin", "10th-pct in-band margin"),
        ("raw_rms_median", "Raw RMS (hardware/geometry sensitive)"),
        ("peak_bin_mode_share", "Peak-bin mode share"),
        ("peak_bin_iqr", "Peak-bin IQR"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for axis, (column, title) in zip(axes.ravel(), panels, strict=True):
        matrix = cards.pivot(index="subject", columns="session_idx", values=column).sort_index().to_numpy()
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xticks(range(5), ["8am", "10am", "12pm", "2pm", "4pm"])
        axis.set_yticks(range(16), range(1, 17))
        axis.set_ylabel("Subject")
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle("10 GHz session components — separate raw scales, no composite score")
    out = _save_figure(fig, path, config)
    plt.close(fig)
    return out


def plot_subject_10_card(cards: pd.DataFrame, path: Path, config: RadarAuditConfig) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subject = cards[cards["subject"] == 10].sort_values("session_idx")
    if len(subject) != 5:
        raise SessionAuditError("subject 10 quality card requires all five sessions")
    panels = [
        ("pass_fraction", "QC pass fraction"),
        ("rms_flag_fraction_defined", "RMS-flag fraction"),
        ("in_band_ratio_p10_margin", "In-band p10 margin"),
        ("raw_rms_median", "Raw RMS"),
        ("peak_bin_mode_share", "Peak-bin mode share"),
        ("peak_bin_iqr", "Peak-bin IQR"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    labels = subject["session_name"].tolist()
    for axis, (column, title) in zip(axes.ravel(), panels, strict=True):
        axis.plot(labels, subject[column], marker="o")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.suptitle("Subject 10 — component card (no overall quality score)")
    out = _save_figure(fig, path, config)
    plt.close(fig)
    return out


def plot_recorded_equal_mass(table: pd.DataFrame, path: Path, config: RadarAuditConfig) -> Path:
    """Magnitude-primary cells in separate panels; no cross-cell numeric ranking."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required_columns = {"diagnostic_channel", "tiling_idx", "scattering_order", "view"}
    if required_columns.issubset(table.columns):
        primary = table[table["diagnostic_channel"] == "mag"].copy()
    else:
        # An empty table is valid when the workbook contains no adjacent pair with
        # exactly equal recorded mass. Still make the declared, clearly empty figure.
        primary = pd.DataFrame(columns=sorted(required_columns))
    cells = [
        (tiling, order, view)
        for tiling in range(3)
        for order in range(3)
        for view in VIEWS
        if not (view == "path_energy_composition" and order == 0)
    ]
    fig, axes = plt.subplots(5, 3, figsize=(14, 16), constrained_layout=True)
    pair_labels = sorted(
        primary.apply(
            lambda row: f"S{int(row.subject)} {row.session_a_name}-{row.session_b_name}", axis=1
        ).unique()
    ) if not primary.empty else []
    for axis, (tiling, order, view) in zip(axes.ravel(), cells, strict=True):
        cell = primary[
            (primary["tiling_idx"] == tiling)
            & (primary["scattering_order"] == order)
            & (primary["view"] == view)
        ]
        values = {
            f"S{int(row.subject)} {row.session_a_name}-{row.session_b_name}": row.between_to_within_ratio
            for row in cell.itertuples()
            if hasattr(row, "between_to_within_ratio") and pd.notna(row.between_to_within_ratio)
        }
        axis.scatter(range(len(pair_labels)), [values.get(label, np.nan) for label in pair_labels], s=18)
        axis.set_title(f"T{tiling + 1}, order {order}, {view}", fontsize=9)
        axis.set_xticks(range(len(pair_labels)), pair_labels, rotation=90, fontsize=6)
        axis.set_ylabel("between / within")
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Recorded-equal-mass pairs — review aid only; not QC, exclusion, or hydration proof",
        fontsize=12,
    )
    out = _save_figure(fig, path, config)
    plt.close(fig)
    return out
