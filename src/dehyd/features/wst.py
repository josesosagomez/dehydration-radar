"""Wavelet scattering transform (WST) features via kymatio.

The kymatio parameterization of implementation_plan.md "WST parameterization": the
milliseconds->samples->(J, T) mapping, filter-bank instantiation with **measured**
padding/shape, the batched forward transform, the order-aware log, and the frozen
cross-backend agreement criterion.

Everything here is shape/fs-agnostic (M9 reuses it for the 77 GHz tilings at
fs = 500 kHz): the three 10 GHz tilings and their (Q, invariance_ms) come from
`WSTConfig`, never re-hardcoded, and J / T / the output shape are DERIVED and MEASURED
from the instantiated filter bank, never assumed (no 512, no `padded_len / 2^J`).

Two facts about the pinned stack, measured at build (see HISTORY 2026-07-23):
  * `Scattering1D(shape=(470,))` warns "signal support is too small to avoid border
    effects" for ALL THREE tilings (J = 8 as well as J = 7). The decision is frozen:
    kymatio's native padding is accepted; the warning is asserted present (T-W3) so a
    kymatio change to padding behaviour fails loudly instead of silently altering
    features. It is never silenced by changing a frozen tiling.
  * kymatio's **torch** frontend is float32-only (float64 input raises). So the numpy
    frontend runs float64, the torch frontend runs float32, and the cross-backend check
    compares numpy-float64 against torch-float32 up-cast to float64 -- which clears the
    strict "float64" tolerance with margin (owner decision 2026-07-23; no fallback).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from ..config import PreprocessConfig, WSTConfig, WSTTiling
from ..data.loader_10ghz import N_FAST_TIME


class WSTError(ValueError):
    """Raised when a WST input or configuration violates a contract."""


# --------------------------------------------------------------- ms -> samples -> J


def t_samples(invariance_ms: float, fs_hz: float) -> int:
    """MATLAB InvarianceScale (ms) -> kymatio averaging support T (samples).

    `round(ms * 1e-3 * fs)`. At fs = 520834 Hz: 0.20 ms -> 104, 0.30 ms -> 156,
    0.40 ms -> 208. Rounding to the nearest sample realizes the requested invariance
    within <0.2% (recorded per tiling by `wst_spec`).
    """
    return int(round(invariance_ms * 1e-3 * fs_hz))


def octaves_j(t_samples_value: int) -> int:
    """Octave count J so the largest wavelet scale covers the averaging support T.

    `ceil(log2(T))`: 104 -> 7, 156 -> 8, 208 -> 8.
    """
    if t_samples_value < 1:
        raise WSTError(f"T must be >= 1 sample, got {t_samples_value}")
    return int(math.ceil(math.log2(t_samples_value)))


def preprocessed_length(pre: PreprocessConfig) -> int:
    """The WST input length: the trimmed chirp, 534 - 2*edge_trim (= 470 by default)."""
    return N_FAST_TIME - 2 * pre.edge_trim


# ------------------------------------------------------------ filter-bank build/shape


def build_scattering(tiling: WSTTiling, wst_cfg: WSTConfig, *, n_in: int, fs_hz: float):
    """Instantiate the pinned kymatio `Scattering1D` for one tiling.

    The frontend comes from `wst_cfg.backend` ALONE (there is no separate backend
    argument), so a call site can never disagree with the config that provenance
    records. Nothing about padding or output length is passed in or assumed -- kymatio
    computes it from the filter bank. The border-effect UserWarning is left to
    propagate (T-W3 asserts it); callers that build many banks may suppress it locally.
    """
    j = octaves_j(t_samples(tiling.invariance_ms, fs_hz))
    t = t_samples(tiling.invariance_ms, fs_hz)
    if wst_cfg.backend == "torch":
        from kymatio.torch import Scattering1D
    else:
        from kymatio.numpy import Scattering1D
    return Scattering1D(
        J=j,
        shape=(n_in,),
        Q=tiling.q,
        T=t,
        max_order=wst_cfg.max_order,
        out_type="array",
    )


def _measure_n_time(scattering) -> int:
    """Measure the output time length by scattering one zero signal (frontend-aware)."""
    n_in = scattering.shape[0]
    if scattering.frontend_name == "torch":
        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = scattering(torch.zeros(1, n_in, dtype=torch.float32))
        return int(out.shape[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = scattering(np.zeros((1, n_in), dtype=np.float64))
    return int(out.shape[-1])


def scattering_shape(scattering) -> dict:
    """The MEASURED geometry read back from the instantiated object and `meta()`.

    Returns pad_left / pad_right / padded_len / n_paths / n_time and the per-path
    metadata arrays (order, xi, sigma, j) in kymatio's canonical path order -- the
    values the tests assert and `wst_spec` records, never a formula. `meta['key']` is
    an inhomogeneous list of tuples and is deliberately left out of the numeric record.
    """
    meta = scattering.meta()
    order = np.asarray(meta["order"])
    n_time = _measure_n_time(scattering)
    return {
        "pad_left": int(scattering.pad_left),
        "pad_right": int(scattering.pad_right),
        "padded_len": int(scattering.shape[0] + scattering.pad_left + scattering.pad_right),
        "n_paths": int(order.shape[0]),
        "n_time": n_time,
        "order": order,
        "xi": np.asarray(meta["xi"]),
        "sigma": np.asarray(meta["sigma"]),
        "j": np.asarray(meta["j"]),
    }


def wst_spec(wst_cfg: WSTConfig, pre_cfg: PreprocessConfig, *, fs_hz: float | None = None) -> dict:
    """The full recorded WST design, per tiling, for provenance / HISTORY.

    Per tiling: requested ms, realized T samples, realized ms and its error fraction,
    J, Q, and the measured `scattering_shape`. Plus backend and max_order. `fs_hz`
    defaults to the preprocess sampling rate.
    """
    if fs_hz is None:
        fs_hz = pre_cfg.fs_hz
    n_in = preprocessed_length(pre_cfg)
    tilings = []
    for tiling in wst_cfg.tilings:
        t = t_samples(tiling.invariance_ms, fs_hz)
        realized_ms = t / fs_hz * 1e3
        # The border warning is asserted at its source (T-W3); silence the duplicate here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scattering = build_scattering(tiling, wst_cfg, n_in=n_in, fs_hz=fs_hz)
        shape = scattering_shape(scattering)
        tilings.append(
            {
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
            }
        )
    return {
        "backend": wst_cfg.backend,
        "max_order": wst_cfg.max_order,
        "log_epsilon": wst_cfg.log_epsilon,
        "n_in": n_in,
        "fs_hz": fs_hz,
        "tilings": tilings,
    }


# ---------------------------------------------------------------- forward transform


def scatter_frames(frames: np.ndarray, scattering) -> np.ndarray:
    """Batch of preprocessed frames -> scattering tensor.

    frames: float64 [N x C x n_in] (C = 1 mag, 2 iq). Returns float64
    [N, C, n_paths, n_time]. THE batched contract: the N*C signals are folded into
    kymatio's leading batch dimension and scattered in ONE call (per-frame calls are
    ~9-10x slower on the pinned stack). Each channel is scattered independently -- the
    batch dimension changes throughput, never semantics (T-W16 asserts batched output
    == stacked single-frame output bit-identically). numpy runs float64; the torch
    frontend runs float32 (its filter bank is float32) and the output is returned as
    float64 numpy.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise WSTError(f"scatter_frames expects [N, C, n_in], got shape {frames.shape}")
    n_in = scattering.shape[0]
    if frames.shape[-1] != n_in:
        raise WSTError(
            f"frame length {frames.shape[-1]} != scattering input length {n_in} "
            "(the WST was built for a different signal length)"
        )
    if not np.all(np.isfinite(frames)):
        raise WSTError("scatter_frames input contains non-finite samples")

    n, c, _ = frames.shape
    flat = frames.reshape(n * c, n_in)
    if scattering.frontend_name == "torch":
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(flat, dtype=np.float32))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = scattering(tensor).cpu().numpy().astype(np.float64)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.asarray(scattering(flat.astype(np.float64)), dtype=np.float64)
    n_paths, n_time = out.shape[-2], out.shape[-1]
    return out.reshape(n, c, n_paths, n_time)


def scatter_channels(channels: np.ndarray, scattering) -> np.ndarray:
    """Single-frame convenience: [C x n_in] -> [C, n_paths, n_time].

    Defined as exactly `scatter_frames(channels[None])[0]`, so the single-frame and
    batched paths cannot diverge.
    """
    channels = np.asarray(channels)
    if channels.ndim != 2:
        raise WSTError(f"scatter_channels expects [C, n_in], got shape {channels.shape}")
    return scatter_frames(channels[None], scattering)[0]


# ------------------------------------------------------------------- order-aware log


def apply_order_log(
    S: np.ndarray, meta, wst_cfg: WSTConfig, *, log_on: bool, epsilon_by_order: dict | None = None
) -> np.ndarray:
    """Order-aware log (implementation_plan.md, "Averaging / log").

    log_on=False -> S unchanged. log_on=True -> orders 1 and 2 become
    `log(S + eps)`; **order 0 is left linear** (S0 = x * phi is a signed low-pass of the
    median/MAD-standardized input and can be negative -- logging it is the exact bug this
    rule prevents). `meta` (or a dict carrying "order") supplies the per-path order; a
    mismatch between S's path count and `meta` raises. `log_on` is a call argument (an
    inner-CV axis), not config.

    `epsilon_by_order` (the milestone-7 tuned-ε path, mirroring `apply_order_log_77`):
    None -> the frozen `wst_cfg.log_epsilon` for both orders (bit-identical to the M4
    behaviour); a {order -> eps} dict -> that fold-local, train-only ε per order. Passing
    {1: 1e-6, 2: 1e-6} reproduces the frozen path byte-for-byte.
    """
    S = np.asarray(S, dtype=np.float64)
    order = np.asarray(meta["order"] if isinstance(meta, dict) else meta.meta()["order"])
    if S.shape[-2] != order.shape[0]:
        raise WSTError(
            f"path-count mismatch: S has {S.shape[-2]} paths, meta has {order.shape[0]} "
            "(a metadata reorder or mismatched tiling)"
        )
    if not log_on:
        return S
    out = S.copy()
    if epsilon_by_order is None:
        logged = order >= 1  # orders 1 and 2 are modulus-based (>= 0); order 0 stays linear
        out[..., logged, :] = np.log(S[..., logged, :] + wst_cfg.log_epsilon)
        return out
    for o in (1, 2):
        mask = order == o
        if mask.any():
            out[..., mask, :] = np.log(S[..., mask, :] + float(epsilon_by_order[o]))
    return out


# ------------------------------------------------------------- cross-backend agreement

# Exactly two frozen policies, no free tolerances at call sites: "float64" is the
# default and the only policy usable without a fresh owner decision; "float32-fallback"
# is pre-declared but unused (the pinned stack clears the strict bar -- HISTORY).
_AGREEMENT_POLICIES = {
    "float64": (1e-4, 1e-8),
    "float32-fallback": (1e-3, 1e-5),
}


@dataclass(frozen=True)
class AgreementResult:
    passed: bool
    max_elementwise_ratio: float  # max over elements of |a-b| / (atol + rtol*max(|a|,|b|))
    rel_l2: float  # ||a-b||_2 / max(||a||_2, ||b||_2, 1e-12)
    policy: str


def backend_agreement(a: np.ndarray, b: np.ndarray, *, policy: str = "float64") -> AgreementResult:
    """THE frozen cross-backend criterion (imported by T-W9 AND the realdata check).

    Passes iff BOTH hold:
      * elementwise  |a - b| <= atol + rtol * max(|a|, |b|)   (the atol floor handles
        near-zero coefficients where a pure relative test is undefined), AND
      * aggregate    rel_l2 <= rtol.
    Tolerances come from the two-entry policy table -- no caller-supplied tolerances, so
    the gate cannot be loosened at a call site. Raises on unequal shapes, empty arrays,
    non-finite values, or (for the "float64" policy) inputs not already in float64 --
    the comparison is done in float64 so the tolerance reflects algorithm, not dtype.
    Returns the measured components alongside the verdict (a bare bool could not feed
    the HISTORY / SECOND_CHAPTER record).
    """
    if policy not in _AGREEMENT_POLICIES:
        raise WSTError(f"unknown agreement policy {policy!r}; choose from {tuple(_AGREEMENT_POLICIES)}")
    rtol, atol = _AGREEMENT_POLICIES[policy]
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise WSTError(f"agreement inputs have different shapes: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise WSTError("agreement inputs are empty")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise WSTError("agreement inputs contain non-finite values")
    if policy == "float64" and not (a.dtype == np.float64 and b.dtype == np.float64):
        raise WSTError(
            f"the 'float64' policy requires float64 inputs, got {a.dtype} and {b.dtype} "
            "(up-cast the torch-float32 output before comparing)"
        )
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    denom = atol + rtol * np.maximum(np.abs(a64), np.abs(b64))
    max_ratio = float(np.max(np.abs(a64 - b64) / denom))
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    rel_l2 = float(np.linalg.norm(a64 - b64) / max(na, nb, 1e-12))
    passed = bool(max_ratio <= 1.0 and rel_l2 <= rtol)
    return AgreementResult(passed=passed, max_elementwise_ratio=max_ratio, rel_l2=rel_l2, policy=policy)
