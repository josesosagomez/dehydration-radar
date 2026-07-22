"""Chirp reduction and edge trimming -- steps 4 and 5 of the sequence.

Two reduction branches, both collapsing a frame's 20 filtered chirps into one complex
534-sample signal. They are alternatives selected inside inner CV at M6, never chosen
here:

  * **Option A** -- the mean across the 20 chirps. Coherent averaging: a stationary
    target's return adds in phase while noise averages down.
  * **Option B** -- detect the frame's dominant beat bin inside the range ROI, keep a
    narrow tapered neighbourhood of it, and average the reconstructions. Isolates the
    target's range bin at the cost of discarding everything else in the band.

**EdgeTrim comes AFTER reduction**, matching `wst_extract.m`: the reduction (in
particular Option B's 534-point FFT) needs the full chirp; only the reduced 1-D signal
is trimmed. Keeping `edge_trim` a separate function is what lets the pipeline test
assert that ordering structurally instead of trusting a comment.

Nothing here is fitted. Option B's peak comes from the frame being reduced -- its own
20 chirps -- so the reduction of one frame can never depend on any other frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import hann

from ..config import PreprocessConfig, beat_band_hz

MIN_EFFECTIVE_LENGTH = 32  # the reference's own floor for a WST-able signal


class ReduceError(ValueError):
    """Raised when a reduction input or a derived ROI/mask is structurally unusable."""


def reduce_option_a(frame: np.ndarray) -> np.ndarray:
    """Option A: mean across the chirps. complex [n_fast x n_chirps] -> [n_fast]."""
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ReduceError(f"frame must be 2-D [n_fast x n_chirps], got shape {frame.shape}")
    return frame.mean(axis=1)


def option_b_roi_bins(pre: PreprocessConfig, n_fast: int) -> np.ndarray:
    """Detection ROI: half-spectrum bins whose frequency lies in the model gate.

    Bins 0 .. n_fast//2 - 1 (DC in, Nyquist out -- the QC convention), and **no
    margin**: the QC screen's +/-1000 Hz margin is a QC constant that exists to tolerate
    Hann leakage at a rejection boundary, which has nothing to do with where a peak may
    be detected. The reference's ROI carries no margin either.

    At the default config (df = 520834/534 ~ 975.34 Hz, gate 3257.5-6514.9 Hz) this is
    bins 4, 5, 6. The 0.9-3.0 m candidate gate gives bins 4..10 -- bin 3 (2926.0 Hz)
    misses the 2931.7 Hz lower edge by 5.7 Hz.
    """
    f_lo, f_hi = beat_band_hz(pre.model_gate_m, pre.bandwidth_hz, pre.chirp_time_s)
    freqs = np.arange(n_fast // 2) * (pre.fs_hz / n_fast)
    bins = np.flatnonzero((freqs >= f_lo) & (freqs <= f_hi))
    if bins.size == 0:
        raise ReduceError(
            f"no FFT bin falls in the model gate {tuple(pre.model_gate_m)} m "
            f"({f_lo:.1f}-{f_hi:.1f} Hz) at df={pre.fs_hz / n_fast:.1f} Hz"
        )
    return bins


@dataclass(frozen=True)
class OptionBDetection:
    """One frame's detection result -- the single source of everything derived from it.

    `reduce_option_b` centres its mask on `peak_bin`; the cohort diagnostics compute
    their concentration measures from this same `power` and `roi_bins`. Nothing
    recomputes the spectrum with a subtly different convention.
    """

    peak_bin: int
    power: np.ndarray  # periodic-Hann, chirp-averaged half-spectrum power
    roi_bins: np.ndarray


def detect_option_b_peak(frame: np.ndarray, pre: PreprocessConfig) -> OptionBDetection:
    """Locate the frame's dominant beat bin inside the ROI.

    Periodic Hann (`hann(N,'periodic')` in the reference) because this is a spectral
    measurement, where leakage from a strong neighbour would otherwise bias the peak.
    The window is for DETECTION ONLY -- the mask is applied to the unwindowed FFT.

    If the ROI carries no power the argmax returns its FIRST bin: deterministic and
    in-ROI. Note this does **not** imply a zero output. Detection is windowed while the
    mask is not, and the frequency-domain Hann kernel [-1/4, 1/2, -1/4] can annihilate
    the windowed ROI bins while the unwindowed bins under the mask stay nonzero. The
    frozen behaviour is exactly "mask the first ROI bin, whatever that yields"; the
    diagnostics report `peak_share = NaN` for such a frame, which is how the situation
    is flagged.
    """
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ReduceError(f"frame must be 2-D [n_fast x n_chirps], got shape {frame.shape}")

    n_fast = frame.shape[0]
    window = hann(n_fast, sym=False)
    spectra = np.fft.fft(frame * window[:, None], axis=0)
    power = (np.abs(spectra[: n_fast // 2, :]) ** 2).mean(axis=1)

    roi_bins = option_b_roi_bins(pre, n_fast)
    peak_bin = int(roi_bins[int(np.argmax(power[roi_bins]))])
    return OptionBDetection(peak_bin=peak_bin, power=power, roi_bins=roi_bins)


def option_b_mask(peak_bin: int, n_fast: int, pre: PreprocessConfig) -> np.ndarray:
    """The two-sided tapered keep-mask around `peak_bin`, over all n_fast bins.

    Weights are the interior of `hann(2*nb + 3)`: w(k) = 0.5*(1 + cos(pi*k/(nb+1))) for
    offset k in [-nb, +nb]. At nb = 1 that is [0.5, 1.0, 0.5] -- **full weight on the
    detected peak**. Every kept positive bin is mirrored onto its conjugate
    (n_fast - k) % n_fast with the same weight, so a real-valued interpretation of the
    spectrum is preserved; a bin that is its own mirror (DC, Nyquist) takes the max of
    the two contributions rather than their sum.

    **This is a deliberate correction of the reference** (`wst_extract.m`), which keeps
    only `peakBin + (0:nb)` -- one-sided, contradicting its own "+/-bins" docstring --
    and then applies MATLAB's endpoint-zero `hann(numel(idx))` across the concatenated
    positive+mirror block, which at nb = 1 gives [0, .75, .75, 0] and so **zeroes the
    detected peak itself**, leaving only bin peak+1 at 75% weight. We implement the form
    the docstring and the main plan describe.

    Raises if the mask ends up covering every bin: Option B would then be a pass-through
    of the already-bandpassed signal, i.e. a silently disabled reduction.
    """
    nb = pre.peak_neighbors
    offsets = np.arange(-nb, nb + 1)
    if pre.mask_taper:
        weights = 0.5 * (1.0 + np.cos(np.pi * offsets / (nb + 1)))
    else:
        weights = np.ones_like(offsets, dtype=np.float64)

    mask = np.zeros(n_fast, dtype=np.float64)
    n_half = n_fast // 2
    for offset, weight in zip(offsets, weights):
        bin_index = peak_bin + int(offset)
        if not (0 <= bin_index <= n_half):
            continue  # clamped away at the spectrum edge
        mirror = (n_fast - bin_index) % n_fast
        # max, never sum: a self-mirroring bin (DC/Nyquist) must not gain weight twice.
        mask[bin_index] = max(mask[bin_index], weight)
        mask[mirror] = max(mask[mirror], weight)

    if np.all(mask > 0):
        raise ReduceError(
            f"option-B mask with peak_neighbors={nb} keeps all {n_fast} bins — "
            "the reduction would pass the signal through unchanged"
        )
    return mask


def reduce_option_b(frame: np.ndarray, pre: PreprocessConfig) -> np.ndarray:
    """Option B: isolate the dominant beat bin, then average across chirps.

    Detection is windowed; the mask is applied to the UNWINDOWED chirp spectra, so the
    reconstruction is not tapered twice.
    """
    frame = np.asarray(frame)
    detection = detect_option_b_peak(frame, pre)
    mask = option_b_mask(detection.peak_bin, frame.shape[0], pre)

    spectra = np.fft.fft(frame, axis=0)
    reconstructed = np.fft.ifft(spectra * mask[:, None], axis=0)
    return reconstructed.mean(axis=1)


def edge_trim(signal: np.ndarray, n_trim: int) -> np.ndarray:
    """Drop `n_trim` samples from each end: 534 - 2*32 = 470 at the default config.

    Raises rather than clamping. The reference silently does `min(EdgeTrim, N/4)`, which
    would turn a mis-set config into a quietly different signal length; here it is a
    loud error.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ReduceError(f"edge_trim expects a 1-D signal, got shape {signal.shape}")
    if n_trim < 0:
        raise ReduceError(f"edge_trim n_trim must be >= 0, got {n_trim}")

    effective = signal.size - 2 * n_trim
    if effective < MIN_EFFECTIVE_LENGTH:
        raise ReduceError(
            f"trimming {n_trim} samples from each end of a {signal.size}-sample signal "
            f"leaves {effective}, below the {MIN_EFFECTIVE_LENGTH}-sample floor"
        )
    return signal[n_trim : signal.size - n_trim]
