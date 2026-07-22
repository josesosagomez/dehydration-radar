"""The four frozen 10 GHz QC screens, run on the RAW cube before any filtering.

Ported from `matlab/10ghz_code/wst_integrity_check_dataset.m`. Thresholds are frozen
constants of the design (implementation_plan.md "QC screens & thresholds"), justified
before evaluation rather than tuned, so QC never enters cross-validation.

**The invariant that governs this module: QC is a fixed, per-frame measurable
function.** Every screen sees exactly one frame plus frozen config constants. Nothing
is computed across frames, sessions or subjects, so QC cannot leak and cannot vary
with a model choice. The screens do normalise *within* a frame -- histogram edges from
each chirp's own magnitude range, RMS median/MAD across that frame's own 20 chirps,
the in-band ratio against that frame's own total power -- but none of that is a
population statistic. `run_qc_cube` is deliberately a plain loop over `run_qc_frame`
for the same reason: there is no batch step in which a cross-frame statistic could
appear (tests/test_qc.py::T-QC7 makes that executable).

Two deliberate departures from the reference, both settled in the main plan:
  * the reference screened *filtered* cubes; we screen the raw pre-filter cube, where
    the in-band ratio is actually informative;
  * the reference only *logged* a low in-band ratio; here it is a rejection criterion,
    alongside NaN/Inf and flatline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import hann

from ..config import PreprocessConfig, QCConfig, beat_band_hz
from ..data.loader_10ghz import N_CHIRPS, N_FAST_TIME

# MATLAB's `eps`: guards the MAD denominator and the in-band ratio denominator so a
# degenerate frame yields a finite verdict instead of a division error.
EPS = float(np.finfo(np.float64).eps)


class QCError(ValueError):
    """Raised when a QC input or a QC-derived band is structurally unusable."""


@dataclass(frozen=True)
class FrameQC:
    """One frame's verdict.

    Float diagnostics are **NaN when unavailable** -- see the non-finite contract in
    `run_qc_frame`: a frame carrying NaN/Inf short-circuits, because numpy histogram
    range inference raises on non-finite input and the FFT/RMS diagnostics would be
    non-finite anyway.
    """

    nan_inf: bool
    flatline: bool
    low_in_band: bool
    rms_flag: bool  # DIAGNOSTIC ONLY -- never a reject criterion
    in_band_ratio: float
    n_flatline_chirps: int
    n_rms_outlier_chirps: int
    max_rms_z: float

    @property
    def passed(self) -> bool:
        """Rejection rule = NaN/Inf or flatline or low in-band energy.

        A property rather than a stored field so the rule cannot be violated by
        construction -- in particular `rms_flag` can never leak into it.
        """
        return not (self.nan_inf or self.flatline or self.low_in_band)


def in_band_mask(
    n_fast: int,
    fs_hz: float,
    bandwidth_hz: float,
    chirp_time_s: float,
    gate_m,
    margin_hz: float,
) -> np.ndarray:
    """Boolean mask over the non-negative half-spectrum bins of an n_fast-point FFT.

    Bins 0 .. n_fast//2 - 1: DC included, Nyquist excluded (MATLAB's
    `half = 1:floor(N/2)`). The band is the range gate mapped to beat frequency and
    widened by `margin_hz` per side, clamped to [0, Nyquist].

    Raises if the mask would select no bins (nothing to measure) or *every* bin (the
    ratio would be identically 1 and the screen could never fire). `config._check_qc_band`
    catches the coarse versions of both at load time; these are the exact bin-level
    guards, which need the fast-time length.
    """
    f_lo, f_hi = beat_band_hz(gate_m, bandwidth_hz, chirp_time_s)
    lo = max(0.0, f_lo - margin_hz)
    hi = min(fs_hz / 2.0, f_hi + margin_hz)

    n_half = n_fast // 2
    freqs = np.arange(n_half) * (fs_hz / n_fast)
    mask = (freqs >= lo) & (freqs <= hi)

    n_selected = int(mask.sum())
    if n_selected == 0:
        raise QCError(
            f"in-band mask selects 0 of {n_half} bins for gate {tuple(gate_m)} m "
            f"(band {lo:.1f}-{hi:.1f} Hz at df={fs_hz / n_fast:.1f} Hz) — nothing to measure"
        )
    if n_selected == n_half:
        raise QCError(
            f"in-band mask selects all {n_half} bins for gate {tuple(gate_m)} m "
            f"(band {lo:.1f}-{hi:.1f} Hz) — the ratio would be identically 1 and the "
            "screen could never fire"
        )
    return mask


def _flatline_chirps(frame: np.ndarray, qc: QCConfig) -> int:
    """Number of chirps whose magnitude histogram is over-concentrated.

    A saturated or dead chirp piles its samples into very few magnitude values, so a
    histogram over the chirp's own [min, max] range shows one bin holding a large
    fraction of the samples. Bin edges come from the chirp itself, never from the
    dataset -- that is what keeps the screen per-frame.
    """
    n_fast = frame.shape[0]
    max_count = qc.flatline_max_bin_fraction * n_fast
    n_flagged = 0
    for chirp in frame.T:
        magnitude = np.abs(chirp)
        edges = np.linspace(magnitude.min(), magnitude.max(), qc.histogram_bins + 1)
        if np.any(edges[:-1] >= edges[1:]):
            # The magnitude spread is too small to divide into this many distinct
            # float64 edges -- i.e. every sample would land in one bin. That IS the
            # flatline case, so decide it here: np.histogram would otherwise raise
            # ("Too many bins for data range"), and it raises not only for a exactly
            # constant chirp but for any near-constant one (a noiseless CW tone spans
            # ~1e-16). MATLAB's histcounts reaches the same verdict by choosing its
            # own bin width; we reach it explicitly.
            n_flagged += 1
            continue
        counts, _ = np.histogram(magnitude, bins=edges)
        if counts.max() >= max_count:
            n_flagged += 1
    return n_flagged


def _in_band_ratio(frame: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of half-spectrum power inside the QC band, averaged over the chirps.

    Periodic Hann (`hann(N,'periodic')` in the reference) because these spectra are
    only ever used for band-power ratios, where the periodic form is the consistent
    choice. Windowing matters here -- unlike the primary filtering path, which takes
    no window -- because leakage from a strong out-of-band tone would otherwise
    inflate the in-band sum.
    """
    n_fast, _ = frame.shape
    window = hann(n_fast, sym=False)
    spectra = np.fft.fft(frame * window[:, None], axis=0)
    power = np.abs(spectra[: n_fast // 2, :]) ** 2
    mean_power = power.mean(axis=1)
    return float(mean_power[mask].sum() / max(mean_power.sum(), EPS))


def _rms_outliers(frame: np.ndarray, qc: QCConfig) -> tuple[int, float]:
    """Chirps whose RMS is a robust-z outlier *within this frame's own 20 chirps*.

    Diagnostic only. Computing the z across frames instead would make QC depend on the
    population and therefore on the split -- exactly what must not happen.
    """
    rms = np.sqrt(np.mean(np.abs(frame) ** 2, axis=0))
    median = np.median(rms)
    mad = np.median(np.abs(rms - median))
    z = np.abs(rms - median) / (1.4826 * (mad + EPS))
    return int(np.count_nonzero(z > qc.rms_robust_z_threshold)), float(z.max())


def run_qc_frame(frame: np.ndarray, qc: QCConfig, pre: PreprocessConfig) -> FrameQC:
    """Screen one raw frame: complex [534 fast-time x 20 chirps]."""
    frame = np.asarray(frame)
    if frame.shape != (N_FAST_TIME, N_CHIRPS):
        raise QCError(
            f"frame has shape {frame.shape}, expected ({N_FAST_TIME}, {N_CHIRPS})"
        )

    # --- non-finite contract -------------------------------------------------------
    # A non-finite sample short-circuits the rest: np.histogram raises on non-finite
    # input, and the FFT/RMS diagnostics would be non-finite. Skipped booleans/counts
    # report False/0; skipped float diagnostics report NaN ("unavailable").
    if not np.all(np.isfinite(frame)):
        return FrameQC(
            nan_inf=True,
            flatline=False,
            low_in_band=False,
            rms_flag=False,
            in_band_ratio=float("nan"),
            n_flatline_chirps=0,
            n_rms_outlier_chirps=0,
            max_rms_z=float("nan"),
        )

    n_flatline = _flatline_chirps(frame, qc)

    # ONE fixed QC gate -- qc.qc_gate_m, never preprocess.model_gate_m. The QC-passing
    # population must be identical for every model gate later searched in inner CV;
    # if QC used the model gate, changing a hyperparameter would change which frames
    # exist and the "no test-set tuning" chronology would silently break. The wider
    # band is used so a frame is not rejected for energy a wider candidate model gate
    # would legitimately use.
    mask = in_band_mask(
        frame.shape[0],
        pre.fs_hz,
        pre.bandwidth_hz,
        pre.chirp_time_s,
        qc.qc_gate_m,
        qc.in_band_margin_hz,
    )
    ratio = _in_band_ratio(frame, mask)
    n_rms_outliers, max_z = _rms_outliers(frame, qc)

    return FrameQC(
        nan_inf=False,
        flatline=n_flatline > 0,
        low_in_band=ratio < qc.min_in_band_energy_ratio,
        rms_flag=n_rms_outliers > 0,
        in_band_ratio=ratio,
        n_flatline_chirps=n_flatline,
        n_rms_outlier_chirps=n_rms_outliers,
        max_rms_z=max_z,
    )


def run_qc_cube(cube: np.ndarray, qc: QCConfig, pre: PreprocessConfig) -> list[FrameQC]:
    """Screen a whole session cube: complex [534 x 20 x n_frames].

    A plain loop, deliberately: no statistic may be shared between frames.
    """
    cube = np.asarray(cube)
    if cube.ndim != 3 or cube.shape[:2] != (N_FAST_TIME, N_CHIRPS):
        raise QCError(
            f"cube has shape {cube.shape}, expected ({N_FAST_TIME}, {N_CHIRPS}, n_frames)"
        )
    return [run_qc_frame(cube[:, :, i], qc, pre) for i in range(cube.shape[2])]
