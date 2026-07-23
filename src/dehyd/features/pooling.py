"""Pooling of scattering coefficients to per-frame vectors, and session aggregation.

Two per-frame feature families (implementation_plan.md "WST parameterization",
"Feature families") and the one-vector-per-session analysis unit (§"Analysis unit —
session-level primary"). Pure functions, no fitted state.

The pooled family's element order is a CONTRACT — a fixed enumeration derived from the
kymatio `meta()` path order — because a silent reordering would scramble features across
the CV loop. `feature_layout` / `session_feature_layout` expose that enumeration as
machine-readable metadata so M6 transforms and Exp E interpretability consume the actual
(tiling-dependent) layout instead of reconstructing it.

The **degenerate-segment rule** (a deliberate ROADMAP §3.3 departure, A-M4-6): a segment
contributes its mean always, its std (ddof=0) only when the segment has >= 2 samples.
With the measured output lengths the short tilings have n_time = 3, whose 1-sample first
half would otherwise ship a structurally-zero std column for every frame, path, and
subject. The rule depends on `n_time` alone — no data-dependence, no leakage vector.
"""

from __future__ import annotations

import numpy as np

SEGMENTS = ("global", "first", "second")
STATISTICS = ("mean", "std")
AGGREGATES = ("frame_mean", "frame_median")


class PoolingError(ValueError):
    """Raised when a pooling input violates a contract."""


def _segment_slices(n_time: int) -> dict:
    """The three segments over the time axis; split at n_time // 2 (frozen)."""
    half = n_time // 2
    return {"global": slice(0, n_time), "first": slice(0, half), "second": slice(half, n_time)}


def _segment_has_std(n_time: int, segment: str) -> bool:
    """A segment gets a std only with >= 2 samples (the degenerate-segment rule)."""
    sl = _segment_slices(n_time)[segment]
    return (sl.stop - sl.start) >= 2


def pool_stats(S: np.ndarray, meta) -> np.ndarray:
    """Per-frame pooled statistics -> 1-D vector.

    S: [C x n_paths x n_time] (post-log). Per channel, per path: mean over each of the
    global / first-half / second-half segments, and std (ddof=0) over each segment that
    has >= 2 samples. Element order: channel -> path (meta order) -> segment (global,
    first, second) -> statistic (mean, then std where defined). Raises on n_time < 2 or
    an S-vs-meta path-count mismatch.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 3:
        raise PoolingError(f"pool_stats expects [C, n_paths, n_time], got {S.shape}")
    order = np.asarray(meta["order"] if isinstance(meta, dict) else meta.meta()["order"])
    n_channels, n_paths, n_time = S.shape
    if n_paths != order.shape[0]:
        raise PoolingError(
            f"path-count mismatch: S has {n_paths} paths, meta has {order.shape[0]}"
        )
    if n_time < 2:
        raise PoolingError(f"n_time must be >= 2 so each half is nonempty, got {n_time}")

    # Vectorized over channels and paths (a per-path Python loop is the cohort hotspot):
    # each column is a [C, n_paths] statistic; columns are assembled in the exact
    # per-path order (segment -> stat, std only where the segment has >= 2 samples),
    # then transposed to channel -> path -> column and flattened. Identical values and
    # order to the reference loop; pinned by the hand-computation tests.
    slices = _segment_slices(n_time)
    columns = []
    for segment in SEGMENTS:
        seg = S[:, :, slices[segment]]  # [C, n_paths, seg_len]
        columns.append(seg.mean(axis=-1))
        if _segment_has_std(n_time, segment):
            columns.append(seg.std(axis=-1, ddof=0))
    stacked = np.stack(columns, axis=-1)  # [C, n_paths, n_stats]
    return stacked.reshape(-1)


def feature_layout(meta, n_time: int, n_channels: int) -> tuple:
    """Per-element metadata of the pooled per-frame vector.

    One (channel, path, segment, statistic) tuple per element, in exactly the pooled
    order (`pool_stats`), including the degenerate-segment rule. The building block of
    `session_feature_layout`. `len(feature_layout(...)) == len(pool_stats(S, meta))`.
    """
    order = np.asarray(meta["order"] if isinstance(meta, dict) else meta.meta()["order"])
    n_paths = order.shape[0]
    layout = []
    for c in range(n_channels):
        for p in range(n_paths):
            for segment in SEGMENTS:
                layout.append((c, p, segment, "mean"))
                if _segment_has_std(n_time, segment):
                    layout.append((c, p, segment, "std"))
    return tuple(layout)


def flatten_series(S: np.ndarray) -> np.ndarray:
    """Raw-flattened scattering series, order channel -> path -> time.

    Diagnostic / DL family only (session-aggregated before any classical metric); never
    a classical session-level feature on its own. Length C * n_paths * n_time.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 3:
        raise PoolingError(f"flatten_series expects [C, n_paths, n_time], got {S.shape}")
    return S.reshape(-1)


def flat_layout(meta, n_time: int, n_channels: int) -> tuple:
    """Per-element metadata of `flatten_series`: (channel, path, time_index)."""
    order = np.asarray(meta["order"] if isinstance(meta, dict) else meta.meta()["order"])
    n_paths = order.shape[0]
    return tuple(
        (c, p, t) for c in range(n_channels) for p in range(n_paths) for t in range(n_time)
    )


def aggregate_session(frame_vectors: np.ndarray) -> np.ndarray:
    """One session's per-frame vectors -> one session vector.

    frame_vectors: [n_frames x D] (pooled OR raw-flattened). Returns [2D] =
    concat(mean over frames, median over frames) — the frozen session-level analysis
    unit, mean-block THEN median-block. NOT a fitted transform. Raises on 0 frames,
    non-2-D, or non-finite input. n_frames = 1 IS allowed by the primitive (mean =
    median = that row); cohort eligibility, not this function, forbids tiny sessions.
    """
    frame_vectors = np.asarray(frame_vectors, dtype=np.float64)
    if frame_vectors.ndim != 2:
        raise PoolingError(f"aggregate_session expects [n_frames, D], got {frame_vectors.shape}")
    if frame_vectors.shape[0] == 0:
        raise PoolingError("aggregate_session got 0 frames")
    if not np.all(np.isfinite(frame_vectors)):
        raise PoolingError("aggregate_session input contains non-finite values")
    mean = frame_vectors.mean(axis=0)
    median = np.median(frame_vectors, axis=0)
    return np.concatenate([mean, median])


def session_feature_layout(meta, n_time: int, n_channels: int, *, family: str) -> tuple:
    """Per-element metadata of the `aggregate_session` OUTPUT (twice the per-frame length).

    An `aggregate` field (frame_mean | frame_median) is prepended to each per-frame
    tuple, mean block first then median block — exactly the frozen concat order. This is
    what M6 and interpretability consume; a per-frame-only layout would leave half the
    model columns unmapped. `family` "pooled" -> (aggregate, channel, path, segment,
    statistic); "flat" -> (aggregate, channel, path, time_index).
    """
    if family == "pooled":
        base = feature_layout(meta, n_time, n_channels)
    elif family == "flat":
        base = flat_layout(meta, n_time, n_channels)
    else:
        raise PoolingError(f"family must be 'pooled' or 'flat', got {family!r}")
    return tuple((agg, *element) for agg in AGGREGATES for element in base)
