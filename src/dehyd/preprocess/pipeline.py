"""The preprocessing sequence, end to end.

One linear function, in the order implementation_plan.md states it, so the whole
pipeline can be read at once:

    3. band gate    zero-phase Butterworth along fast time, per chirp (or the FFT gate)
    4. reduce       Option A (chirp mean) or Option B (isolate the dominant beat bin)
    5. EdgeTrim     32 samples per end, AFTER reduction -> 470 samples
    6. channel      |s|, or {real, imag}
    7. standardize  robust median/MAD z, per signal

`reduction` and `channel` are call arguments rather than config because they are inner-
CV search axes at M6: one config must be able to produce every variant, and nothing
here may pick between them.

The input contract is a **QC-passed frame**: exact shape, and finite. Both are checked,
because a silently-wrong shape would otherwise reduce along the wrong axis and produce
plausible nonsense.
"""

from __future__ import annotations

import numpy as np

from ..config import PreprocessConfig
from ..data.loader_10ghz import N_CHIRPS, N_FAST_TIME
from .filters import apply_band_gate
from .reduce import edge_trim, reduce_option_a, reduce_option_b
from .standardize import to_channels

REDUCTIONS = ("a", "b")
CHANNELS = ("mag", "iq")


class PipelineError(ValueError):
    """Raised when a preprocessing input violates the frame contract."""


def _validate(array: np.ndarray, expected_ndim: int) -> np.ndarray:
    array = np.asarray(array)
    expected = (N_FAST_TIME, N_CHIRPS) if expected_ndim == 2 else (N_FAST_TIME, N_CHIRPS, "n")
    if array.ndim != expected_ndim or array.shape[:2] != (N_FAST_TIME, N_CHIRPS):
        raise PipelineError(f"expected shape {expected}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise PipelineError(
            "input contains non-finite samples — preprocessing consumes QC-passed "
            "frames only (the NaN/Inf screen runs before this stage)"
        )
    return array


def preprocess_frame(
    frame: np.ndarray,
    pre: PreprocessConfig,
    *,
    reduction: str,
    channel: str,
) -> np.ndarray:
    """One QC-passed frame -> float64 [n_channels x 470].

    complex128 [534 fast-time x 20 chirps] in; 1 channel for `mag`, 2 for `iq`.
    """
    if reduction not in REDUCTIONS:
        raise PipelineError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    if channel not in CHANNELS:
        raise PipelineError(f"channel must be one of {CHANNELS}, got {channel!r}")
    frame = _validate(frame, 2)

    gated = apply_band_gate(frame, pre, axis=0)
    reduced = reduce_option_a(gated) if reduction == "a" else reduce_option_b(gated, pre)
    # AFTER reduction, never before: Option B's FFT needs the full 534-sample chirp.
    trimmed = edge_trim(reduced, pre.edge_trim)
    return to_channels(trimmed, channel, pre.standardize)


def preprocess_cube(
    cube: np.ndarray,
    pre: PreprocessConfig,
    *,
    reduction: str,
    channel: str,
) -> np.ndarray:
    """A session cube -> float64 [n_frames x n_channels x 470].

    A plain loop over `preprocess_frame`, for the same reason `run_qc_cube` is one: it
    makes it structurally impossible for a statistic to be shared between frames.
    (The band gate would vectorise across frames, and does so within a frame already;
    the per-frame loop costs little and buys the guarantee.)
    """
    cube = _validate(cube, 3)
    return np.stack(
        [
            preprocess_frame(cube[:, :, i], pre, reduction=reduction, channel=channel)
            for i in range(cube.shape[2])
        ]
    )
