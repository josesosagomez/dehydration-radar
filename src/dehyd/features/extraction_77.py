"""The 77 GHz WST extraction chain — steps 6-10 of the Exp G primary chain (band 2, M5).

A linear, followable composition of the fs/shape-agnostic features/ functions, per session:

    load        [N, 256 fast, 256 chirp, 16 rx]  float64
    steps 1-5   [N, 27 gate, 256 chirp, 16 rx]   complex   (preprocess_cube_77)
    step 6      per frame: 16 rx x 27 gate = 432 COMPLEX slow-time series (len 256), each
                split into 2 robust-standardized real channels -> batch [432, 2, 256]
    scatter     scatter_frames([432,2,256]) -> S [432, 2, P, t]     (P, t MEASURED)
    step 7      reshape [16, 27, 2, P, t] -> mean over gate bins -> per-Rx [16, 2, P, t]
    step 8      Rx fusion: mean (primary) / median (secondary) over rx -> [2, P, t]
    log         order-aware log on the FUSED tensor (A-M5-4); branch off / on+frozen-ε
                (call arg); on+tuned-ε applies a train-fitted per-order ε (supplied by M7)
    step 9      pool_stats -> per-frame vector [D]   (or flatten_series for "flat")
    step 10     aggregate_session([N, D]) -> [2D]   (frame-mean concat frame-median)

No fitted state, no selection. The 27-gate and 16-Rx loops are folded into the scatter batch
dim, never Python loops (the M4 pool_stats hotspot lesson). M5 realizes the two
data-independent log branches (off, on+frozen-ε) end to end and provides the tuned-ε
APPLICATION path (epsilon_by_order), but NEVER computes a data-dependent ε — that is
fold-local and lives in the M7 harness (computing it here would be leakage).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields

import numpy as np

from ..config import (
    Config,
    Preprocess77Config,
    QC77Config,
    WST77Config,
    WSTTiling,
)
from ..preprocess.pipeline_77 import preprocess_cube_77
from ..preprocess.standardize import EPS, MAD_TO_SIGMA, StandardizeError
from .pooling import aggregate_session, flatten_series, pool_stats
from .wst import (
    WSTError,
    build_scattering,
    octaves_j,
    scatter_frames,
    scattering_shape,
    t_samples,
)

FUSIONS = ("mean", "median")  # mean is primary/frozen; median a labeled secondary variant
FAMILIES = ("pooled", "flat")  # pooled is the classical family; flat is diagnostic/DL-only
LOG_BRANCHES = ("off", "on_frozen_eps", "on_tuned_eps")
# The two data-independent branches M5 materializes end to end (on_tuned_eps is M7-only).
DATA_INDEPENDENT_LOG_BRANCHES = ("off", "on_frozen_eps")


class CanonicalSpecError77(ValueError):
    """Raised when a non-canonical 77 GHz config would write the primary WST artifact."""


def prf_hz(pre77: Preprocess77Config) -> float:
    """Slow-time sample rate = pulse-repetition frequency = 1 / chirp_time (1953.125 Hz)."""
    return 1.0 / pre77.chirp_time_s


def _build77(tiling: WSTTiling, wst77: WST77Config, n_in: int, fs_hz: float):
    # Callers build many banks; the border warning is asserted at its source (T-W3).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_scattering(tiling, wst77, n_in=n_in, fs_hz=fs_hz)


def slow_time_signal_batch(gated_frame: np.ndarray, standardize: str) -> np.ndarray:
    """One gated frame [n_gate, n_chirp, n_rx] -> [n_rx*n_gate, 2, n_chirp] standardized channels.

    The n_rx*n_gate complex slow-time series (one per (Rx, gate bin)) are laid out
    **rx-major, bin-minor** (a FROZEN fold order), split into 2 real channels {real, imag},
    then VECTORIZED-standardized along the time axis for all channels at once — NOT a
    per-series to_channels loop. The batch form is bit-equivalent to stacked
    to_channels(series, "iq", standardize) (same median/MAD/1.4826/eps), pinned by T-W77.

    RAISES if any standardized channel is all-zero: a constant slow-time series has MAD 0
    and maps to zeros — the plan's nonzero-energy assertion.
    """
    gated_frame = np.asarray(gated_frame)
    if gated_frame.ndim != 3:
        raise WSTError(
            f"slow_time_signal_batch expects [n_gate, n_chirp, n_rx], got {gated_frame.shape}"
        )
    n_gate, n_chirp, n_rx = gated_frame.shape
    # rx-major, bin-minor: [n_rx, n_gate, n_chirp] -> [n_rx*n_gate, n_chirp].
    series = np.transpose(gated_frame, (2, 0, 1)).reshape(n_rx * n_gate, n_chirp)
    channels = np.stack([series.real, series.imag], axis=1)  # [432, 2, n_chirp]

    if standardize == "robust":
        med = np.median(channels, axis=-1, keepdims=True)
        mad = np.median(np.abs(channels - med), axis=-1, keepdims=True)
        out = (channels - med) / (MAD_TO_SIGMA * mad + EPS)
    elif standardize == "meanstd":
        mean = channels.mean(axis=-1, keepdims=True)
        std = channels.std(axis=-1, ddof=0, keepdims=True)
        out = (channels - mean) / (std + EPS)
    else:
        raise StandardizeError(f"unknown standardize method {standardize!r}")

    if np.any(np.all(out == 0.0, axis=-1)):
        raise WSTError(
            "a standardized slow-time channel is all-zero (constant series -> MAD 0); "
            "the nonzero-energy assertion — this gate bin carries no slow-time variation"
        )
    return np.ascontiguousarray(out, dtype=np.float64)


def _frame_per_rx_tensor(gated_frame: np.ndarray, scattering, standardize: str) -> np.ndarray:
    """One gated frame -> per-Rx scattering tensor [n_rx, 2, P, t] (steps 6-7, pre-fusion)."""
    n_gate, n_chirp, n_rx = gated_frame.shape
    batch = slow_time_signal_batch(gated_frame, standardize)  # [n_rx*n_gate, 2, n_chirp]
    S = scatter_frames(batch, scattering)  # [n_rx*n_gate, 2, P, t]
    n_paths, n_time = S.shape[-2], S.shape[-1]
    S = S.reshape(n_rx, n_gate, 2, n_paths, n_time)  # undo the rx-major/bin-minor fold
    return S.mean(axis=1)  # mean over the 27 gate bins -> [n_rx, 2, P, t]


def _fuse_rx(per_rx: np.ndarray, fusion: str) -> np.ndarray:
    """Rx fusion over the 16 receivers -> [2, P, t]. mean is primary; median secondary."""
    if fusion == "mean":
        return per_rx.mean(axis=0)
    if fusion == "median":
        return np.median(per_rx, axis=0)
    raise WSTError(f"fusion must be one of {FUSIONS}, got {fusion!r}")


def apply_order_log_77(S: np.ndarray, meta, wst77: WST77Config, *, log_branch: str,
                       epsilon_by_order: dict | None = None) -> np.ndarray:
    """Order-aware log on a fused per-frame tensor [C, P, t] (A-M5-4).

    off -> unchanged. on_frozen_eps -> orders 1,2 become log(S + wst77.log_epsilon).
    on_tuned_eps -> orders 1,2 become log(S + epsilon_by_order[order]) with the per-order ε
    the M7 harness fitted train-only (M5 never computes it). **Order 0 stays linear** in all
    branches (it is a signed low-pass of standardized input and can be negative). The tuned-ε
    ε is applied identically regardless of role — no leakage in the mechanism.
    """
    S = np.asarray(S, dtype=np.float64)
    order = np.asarray(meta["order"] if isinstance(meta, dict) else meta.meta()["order"])
    if S.shape[-2] != order.shape[0]:
        raise WSTError(
            f"path-count mismatch: S has {S.shape[-2]} paths, meta has {order.shape[0]}"
        )
    if log_branch == "off":
        return S
    if log_branch not in ("on_frozen_eps", "on_tuned_eps"):
        raise WSTError(f"log_branch must be one of {LOG_BRANCHES}, got {log_branch!r}")

    out = S.copy()
    if log_branch == "on_frozen_eps":
        logged = order >= 1  # orders 1 and 2 are modulus-based; order 0 stays linear
        out[..., logged, :] = np.log(S[..., logged, :] + wst77.log_epsilon)
        return out

    if epsilon_by_order is None:
        raise WSTError("log_branch 'on_tuned_eps' requires epsilon_by_order (order -> ε)")
    for o in (1, 2):
        mask = order == o
        if mask.any():
            out[..., mask, :] = np.log(S[..., mask, :] + float(epsilon_by_order[o]))
    return out


def _pool_family(logged: np.ndarray, meta, family: str) -> np.ndarray:
    """Fused, logged [C, P, t] -> per-frame vector [D]."""
    if family == "pooled":
        return pool_stats(logged, meta)
    if family == "flat":
        return flatten_series(logged)
    raise CanonicalSpecError77(f"family must be one of {FAMILIES}, got {family!r}")


def _prelog_scale_77(fused_frames: np.ndarray, meta) -> tuple:
    """Pre-log order-0/1/2 coefficient scale on the fused tensor (frozen reduction order).

    fused_frames [N, C, P, t] is RAW (pre-log) — orders 1/2 are non-negative there, so their
    mean IS a magnitude. Per frame, per order o: mean over time -> [N, C, P_o]; mean over the
    order-o paths -> [N, C]; mean over channels -> [N]. Session value = median over frames.
    Order 0 is the single order-0 path's signed time-mean, mean over channels, median over
    frames. This is what the M7 tuned-ε branch consumes to compute a train-only ε; M5 records
    it, never applies a data-dependent ε.
    """
    order = np.asarray(meta["order"] if isinstance(meta, dict) else meta.meta()["order"])
    time_mean = fused_frames.mean(axis=-1)  # [N, C, P]
    out = []
    for o in (0, 1, 2):
        per_channel_frame = time_mean[:, :, order == o].mean(axis=-1)  # [N, C]
        per_frame = per_channel_frame.mean(axis=-1)  # [N]
        out.append(float(np.median(per_frame)))
    return tuple(out)


def extract_frame_features_77(gated_frame: np.ndarray, scattering, meta, wst77: WST77Config,
                              *, standardize: str, log_branch: str, fusion: str, family: str,
                              epsilon_by_order: dict | None = None) -> np.ndarray:
    """One gated frame -> one per-frame feature vector [D] (steps 6-9)."""
    per_rx = _frame_per_rx_tensor(gated_frame, scattering, standardize)
    fused = _fuse_rx(per_rx, fusion)
    logged = apply_order_log_77(
        fused, meta, wst77, log_branch=log_branch, epsilon_by_order=epsilon_by_order
    )
    return _pool_family(logged, meta, family)


def extract_session_features_77(cube: np.ndarray, pre77: Preprocess77Config, wst77: WST77Config,
                                *, tiling: WSTTiling, log_branch: str, fusion: str, family: str,
                                epsilon_by_order: dict | None = None) -> np.ndarray:
    """One session's frames -> one session vector [2D] (the single-variant reference).

    Correct by construction: preprocess -> per-frame scatter/fuse/log/pool -> aggregate.
    """
    gated = preprocess_cube_77(cube, pre77)  # [N, n_gate, n_chirp, n_rx]
    scattering = _build77(tiling, wst77, gated.shape[2], prf_hz(pre77))  # n_in = n_chirp
    meta = scattering.meta()
    per_frame = np.stack([
        extract_frame_features_77(
            gated[i], scattering, meta, wst77, standardize=pre77.standardize,
            log_branch=log_branch, fusion=fusion, family=family,
            epsilon_by_order=epsilon_by_order,
        )
        for i in range(gated.shape[0])
    ])
    return aggregate_session(per_frame)


@dataclass(frozen=True)
class SessionVariant77Result:
    """Everything one session's extraction produces, from the scatter-once-per-tiling pass."""

    vectors: dict  # {(tiling_idx, log_branch, fusion, family): session_vector [2D]}
    prelog_scale: dict  # {(tiling_idx, fusion): (v0, v1, v2)} — KEYED BY fusion (C5-10)
    shapes: dict  # {tiling_idx: (n_paths, n_time)}
    all_finite: bool


def extract_session_variants_77(cube: np.ndarray, pre77: Preprocess77Config,
                                wst77: WST77Config) -> SessionVariant77Result:
    """Cohort-loop form: preprocess once, scatter once per tiling, derive all variants.

    Per (tiling, fusion) it reuses ONE shared raw fused tensor for every
    (log_branch{off,on_frozen_eps} x family{pooled,flat}) vector plus the pre-log scale.
    IN-RUN reuse of a deterministic intermediate, NOT a persistent cache. Computing both
    fusions/families is for diagnostics + the DL path; only (mean, pooled) is the primary
    classical modeling path.
    """
    gated = preprocess_cube_77(cube, pre77)
    n_in, n_frames = gated.shape[2], gated.shape[0]

    vectors: dict = {}
    prelog_scale: dict = {}
    shapes: dict = {}
    all_finite = True

    for ti, tiling in enumerate(wst77.tilings):
        scattering = _build77(tiling, wst77, n_in, prf_hz(pre77))
        meta = scattering.meta()
        # Per frame -> per-Rx tensor once; keep the RAW (pre-log) fused tensor per fusion.
        per_rx_frames = [
            _frame_per_rx_tensor(gated[i], scattering, pre77.standardize)
            for i in range(n_frames)
        ]
        for fusion in FUSIONS:
            fused_frames = np.stack([_fuse_rx(pr, fusion) for pr in per_rx_frames])  # [N,2,P,t]
            shapes.setdefault(ti, (fused_frames.shape[-2], fused_frames.shape[-1]))
            prelog_scale[(ti, fusion)] = _prelog_scale_77(fused_frames, meta)
            for log_branch in DATA_INDEPENDENT_LOG_BRANCHES:
                logged = np.stack([
                    apply_order_log_77(fused_frames[i], meta, wst77, log_branch=log_branch)
                    for i in range(n_frames)
                ])
                for family in FAMILIES:
                    per_frame = np.stack([_pool_family(logged[i], meta, family) for i in range(n_frames)])
                    vec = aggregate_session(per_frame)
                    vectors[(ti, log_branch, fusion, family)] = vec
                    all_finite = all_finite and bool(np.all(np.isfinite(vec)))

    return SessionVariant77Result(
        vectors=vectors, prelog_scale=prelog_scale, shapes=shapes, all_finite=all_finite
    )


def canonical_spec_guard_77(config: Config) -> None:
    """Refuse to write the primary 77 GHz WST artifact unless config is exactly canonical.

    config.preprocess77 / config.qc77 / config.wst77 must each equal their frozen default
    (INCLUDING wst77.backend == "numpy", the canonical artifact backend). Names whatever
    deviates. The flatline correction (qc77.flatline_skip_leading_bins) is covered by the
    qc77 equality, so a stale rule cannot write a curated artifact.
    """
    deviations = []
    for section, canonical in (
        ("preprocess77", Preprocess77Config()),
        ("qc77", QC77Config()),
        ("wst77", WST77Config()),
    ):
        actual = getattr(config, section)
        for f in fields(canonical):
            got = getattr(actual, f.name)
            want = getattr(canonical, f.name)
            if got != want:
                deviations.append(f"{section}.{f.name}={got!r} (canonical: {want!r})")
    if deviations:
        raise CanonicalSpecError77(
            "run_wst77.py writes the PRIMARY curated artifact and refuses a non-canonical "
            "config; deviating fields: " + "; ".join(deviations)
        )


def wst77_spec(wst77: WST77Config, pre77: Preprocess77Config) -> dict:
    """The full recorded 77 GHz WST design, per tiling, measured at n_in = n_chirp, fs = PRF."""
    from ..data.loader_77ghz import N_CHIRPS

    fs_hz = prf_hz(pre77)
    tilings = []
    for tiling in wst77.tilings:
        t = t_samples(tiling.invariance_ms, fs_hz)
        realized_ms = t / fs_hz * 1e3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scattering = build_scattering(tiling, wst77, n_in=N_CHIRPS, fs_hz=fs_hz)
        shape = scattering_shape(scattering)
        tilings.append({
            "q": tuple(tiling.q),
            "requested_ms": tiling.invariance_ms,
            "t_samples": t,
            "realized_ms": realized_ms,
            "realized_error_frac": abs(realized_ms - tiling.invariance_ms) / tiling.invariance_ms,
            "J": octaves_j(t),
            "n_paths": shape["n_paths"],
            "n_time": shape["n_time"],
            "pad_left": shape["pad_left"],
            "pad_right": shape["pad_right"],
            "padded_len": shape["padded_len"],
        })
    return {
        "backend": wst77.backend,
        "max_order": wst77.max_order,
        "log_epsilon": wst77.log_epsilon,
        "n_in": N_CHIRPS,
        "fs_hz": fs_hz,
        "tilings": tilings,
    }
