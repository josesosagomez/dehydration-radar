"""Reusable manifest->features extraction wiring (library code in `src/`).

This lives in `src/` — not inside a CLI script — so the M7 harness and `run_wst.py` are
both thin consumers of it and the repository's dependency direction (library <- scripts,
never the reverse) holds. It is a linear composition of the public features/ functions:
preprocess -> scatter -> order-log -> pool/flatten -> session-aggregate.

Two forms:
  * `extract_session_features` — the single-variant REFERENCE, correct by construction.
  * `extract_session_variants` — the cohort-loop form: preprocess ONCE per (reduction,
    channel), scatter ONCE per tiling, then derive every (log x family) session vector
    and the pre-log scale diagnostics from each shared raw tensor. Deriving those
    separately would recompute the unfitted raw tensor four times (turning the ~14-min
    batched cohort projection into ~56 min). This is in-run reuse of a deterministic
    intermediate, NOT the deferred persistent feature cache.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass

import numpy as np

from ..config import Config, PreprocessConfig, WSTConfig, WSTTiling
from ..preprocess.pipeline import preprocess_cube
from .pooling import aggregate_session, flatten_series, pool_stats
from .wst import apply_order_log, build_scattering, preprocessed_length, scatter_frames

FAMILIES = ("pooled", "flat")

CANONICAL_PREPROCESS = PreprocessConfig()
CANONICAL_WST = WSTConfig()  # backend "numpy" is part of the canonical artifact spec


class CanonicalSpecError(ValueError):
    """Raised when a non-canonical config would write the primary WST artifact."""


@dataclass(frozen=True)
class SessionVariantResult:
    """Everything one session's extraction produces, from the scatter-once pass."""

    vectors: dict  # {(tiling_index, log_on, family): session_vector [2D]}
    prelog_scale: dict  # {tiling_index: (v0, v1, v2)} — pre-log order-0/1/2 scale
    shapes: dict  # {tiling_index: (n_paths, n_time)}
    all_finite: bool
    # Milestone 7 (keep_raw=True only): the RAW pre-log scattering tensor + meta order per
    # tiling, so the fold-local tuned-ε branch can be reconstructed from a persistent store
    # without re-scattering. None by default (M4 behaviour unchanged; vectors/prelog_scale
    # are computed identically whether or not raw is kept).
    raw: dict | None = None  # {tiling_index: {"S": [N,C,P,t] pre-log, "order": [P]}}


def _build(tiling: WSTTiling, wst_cfg: WSTConfig, n_in: int, fs_hz: float):
    # Callers build many banks; the border warning is asserted at its source (T-W3).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_scattering(tiling, wst_cfg, n_in=n_in, fs_hz=fs_hz)


def _per_frame(logged: np.ndarray, meta, family: str) -> np.ndarray:
    """logged [N, C, P, t] -> per-frame vectors [N, D] for the given family."""
    if family == "pooled":
        return np.stack([pool_stats(logged[i], meta) for i in range(logged.shape[0])])
    if family == "flat":
        return np.stack([flatten_series(logged[i]) for i in range(logged.shape[0])])
    raise CanonicalSpecError(f"family must be one of {FAMILIES}, got {family!r}")


def _apply_log(S: np.ndarray, meta, wst_cfg: WSTConfig, log_on: bool) -> np.ndarray:
    return np.stack([apply_order_log(S[i], meta, wst_cfg, log_on=log_on) for i in range(S.shape[0])])


def _prelog_scale(S: np.ndarray, meta) -> tuple:
    """Pre-log order-0/1/2 coefficient scale (frozen reduction order).

    S is the RAW scattering tensor [N, C, P, t] BEFORE any log — orders 1/2 are
    non-negative there, so their mean IS a magnitude; measuring logged values would be
    circular. Per frame, per order o: mean over time -> [N, C, P_o]; mean over the paths
    of order o -> [N, C]; mean over channels -> [N]. Session value = median over frames.
    Order 0 is the single order-0 path's signed time-mean (mean over its one path is
    itself), mean over channels, median over frames.
    """
    order = np.asarray(meta["order"])
    time_mean = S.mean(axis=-1)  # [N, C, P]
    out = []
    for o in (0, 1, 2):
        per_channel_frame = time_mean[:, :, order == o].mean(axis=-1)  # [N, C]
        per_frame = per_channel_frame.mean(axis=-1)  # [N]
        out.append(float(np.median(per_frame)))
    return tuple(out)


def extract_session_features(
    cube: np.ndarray,
    pre: PreprocessConfig,
    wst_cfg: WSTConfig,
    *,
    reduction: str,
    channel: str,
    tiling: WSTTiling,
    log_on: bool,
    family: str,
) -> np.ndarray:
    """One session's QC-passed frames -> one session vector (the single-variant reference)."""
    frames = preprocess_cube(cube, pre, reduction=reduction, channel=channel)
    scattering = _build(tiling, wst_cfg, frames.shape[-1], pre.fs_hz)
    meta = scattering.meta()
    S = scatter_frames(frames, scattering)
    logged = _apply_log(S, meta, wst_cfg, log_on)
    return aggregate_session(_per_frame(logged, meta, family))


def extract_session_variants(
    cube: np.ndarray,
    pre: PreprocessConfig,
    wst_cfg: WSTConfig,
    *,
    reduction: str,
    channel: str,
    keep_raw: bool = False,
) -> SessionVariantResult:
    """Cohort-loop form: preprocess once, scatter once per tiling, derive all variants.

    `keep_raw=True` additionally returns the RAW pre-log scattering tensor + meta order per
    tiling (the milestone-7 feature-store input for the tuned-ε branch). It does NOT change
    `vectors`/`prelog_scale`/`shapes`/`all_finite` — those are computed from the same `S`."""
    frames = preprocess_cube(cube, pre, reduction=reduction, channel=channel)
    n_in = frames.shape[-1]

    vectors: dict = {}
    prelog_scale: dict = {}
    shapes: dict = {}
    raw: dict = {}
    all_finite = True

    for ti, tiling in enumerate(wst_cfg.tilings):
        scattering = _build(tiling, wst_cfg, n_in, pre.fs_hz)
        meta = scattering.meta()
        S = scatter_frames(frames, scattering)  # scattered ONCE per tiling
        shapes[ti] = (S.shape[-2], S.shape[-1])
        prelog_scale[ti] = _prelog_scale(S, meta)
        if keep_raw:
            raw[ti] = {"S": S.copy(), "order": np.asarray(meta["order"]).copy()}
        for log_on in (False, True):
            logged = _apply_log(S, meta, wst_cfg, log_on)
            for family in FAMILIES:
                vec = aggregate_session(_per_frame(logged, meta, family))
                vectors[(ti, log_on, family)] = vec
                all_finite = all_finite and bool(np.all(np.isfinite(vec)))

    return SessionVariantResult(
        vectors=vectors, prelog_scale=prelog_scale, shapes=shapes, all_finite=all_finite,
        raw=(raw if keep_raw else None),
    )


def canonical_spec_guard(config: Config) -> None:
    """Refuse to write the primary WST artifact unless config is exactly canonical.

    The curated CSV is the PRIMARY artifact; every non-primary result must be explicitly
    labelled, so a non-canonical run must not overwrite it. `config.preprocess` must equal
    the whole canonical `PreprocessConfig()` (the M3 guard) AND `config.wst` must equal the
    canonical `WSTConfig()` — INCLUDING `backend == "numpy"`, since numpy is the canonical
    artifact backend. Names whatever deviates.
    """
    deviations = []
    for section, canonical in (("preprocess", CANONICAL_PREPROCESS), ("wst", CANONICAL_WST)):
        actual = getattr(config, section)
        for f in dataclasses.fields(canonical):
            got = getattr(actual, f.name)
            want = getattr(canonical, f.name)
            if got != want:
                deviations.append(f"{section}.{f.name}={got!r} (canonical: {want!r})")
    if deviations:
        raise CanonicalSpecError(
            "run_wst.py writes the PRIMARY curated artifact and refuses a non-canonical "
            "config; deviating fields: " + "; ".join(deviations)
        )
