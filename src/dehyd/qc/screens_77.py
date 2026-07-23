"""The frozen 77 GHz per-frame QC screens (band 2, milestone 5).

Three screens, mirroring the frozen Exp G QC: NaN/Inf, flatline, in-band energy (no RMS
diagnostic — that is 10 GHz-only). The invariant is the same as the 10 GHz screens: **QC is
a fixed, per-frame measurable function** — every screen sees exactly one frame plus frozen
constants, nothing is computed across frames/sessions/subjects, so QC cannot leak and cannot
vary with a model choice. `run_qc_cube_77` is a plain loop for exactly that reason.

Flatline is per (Rx, chirp) trace and a frame fails if ANY trace flags — the structural
analog of the 10 GHz any-chirp rule at ~205x the multiplicity (16 Rx x 256 chirps = 4096
traces vs 20). The in-band screen reuses `screens.in_band_mask` as-is.

`qc_smoke_frame` and `qc_in_band_mask_77` are promoted from experiments/audit_77ghz.py (which
imports them back); `run_qc_frame_77`/`run_qc_cube_77`/`FrameQC77` are the production API that
shares the same flatline core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import hann

from ..config import Preprocess77Config, QC77Config
from .screens import EPS, in_band_mask


@dataclass(frozen=True)
class FrameQC77:
    """One 77 GHz frame's verdict.

    Float diagnostics are NaN when unavailable (the non-finite short-circuit contract).
    `per_rx_flatline` is the per-Rx count of flagged traces; `n_flatline_traces` their sum.
    """

    nan_inf: bool
    flatline: bool
    low_in_band: bool
    in_band_ratio: float
    n_flatline_traces: int
    per_rx_flatline: tuple[int, ...]

    @property
    def passed(self) -> bool:
        """Rejection rule = NaN/Inf or flatline or low in-band. A property so the rule
        cannot be violated by construction."""
        return not (self.nan_inf or self.flatline or self.low_in_band)


def qc_in_band_mask_77(n_fast: int, fs_hz: float, bandwidth_hz: float,
                       chirp_time_s: float, gate_m, margin_hz: float) -> np.ndarray:
    """Half-spectrum mask (bins 0..n_fast//2-1, DC in / Nyquist out) — 10 GHz convention.

    Promoted from the audit; its non-raising form is kept for the audit's characterisation
    use. The production screen uses screens.in_band_mask (which raises on a degenerate mask).
    """
    from ..config import SPEED_OF_LIGHT_M_S

    hz_per_m = 2.0 * (bandwidth_hz / chirp_time_s) / SPEED_OF_LIGHT_M_S
    lo = max(0.0, hz_per_m * gate_m[0] - margin_hz)
    hi = min(fs_hz / 2.0, hz_per_m * gate_m[1] + margin_hz)
    freqs = np.arange(n_fast // 2) * (fs_hz / n_fast)
    return (freqs >= lo) & (freqs <= hi)


def _flatline_per_rx(frame: np.ndarray, *, bins: int, max_bin_fraction: float,
                     skip_leading: int = 0) -> list[int]:
    """Per-Rx count of flatlined (Rx, chirp) traces.

    A trace flatlines when its fast-time magnitude piles into few histogram bins: bin edges
    from the trace's own [min, max] (never the dataset — that keeps the screen per-frame),
    a degenerate spread counts as flatline (np.histogram would otherwise raise on a
    near-constant trace), else flag if the max bin count reaches `max_bin_fraction * n_screened`.

    `skip_leading` drops the first `skip_leading` fast-time samples before screening — the M5
    step-6 correction: fast[0] is an embedded frame counter, not echo, and its extreme value
    stretched the [min,max] range into a false flatline. Default 0 keeps the audit's M2
    behaviour; the production screen passes qc77.flatline_skip_leading_bins (= 1).
    """
    n_fast, n_chirps, n_rx = frame.shape
    n_screened = n_fast - skip_leading
    max_count = max_bin_fraction * n_screened
    per_rx = []
    for rx in range(n_rx):
        flagged = 0
        for chirp in range(n_chirps):
            magnitude = np.abs(frame[skip_leading:, chirp, rx])
            edges = np.linspace(magnitude.min(), magnitude.max(), bins + 1)
            if np.any(edges[:-1] >= edges[1:]):
                flagged += 1  # degenerate spread == flatline
                continue
            if np.histogram(magnitude, bins=edges)[0].max() >= max_count:
                flagged += 1
        per_rx.append(flagged)
    return per_rx


def _in_band_ratio_77(frame: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of half-spectrum power inside the band, averaged over ALL chirps and Rx.

    Periodic Hann on the fast-time axis (the QC-reference convention), so out-of-band
    leakage does not inflate the in-band sum.
    """
    n_fast = frame.shape[0]
    window = hann(n_fast, sym=False)[:, None, None]
    spectra = np.fft.fft(frame * window, axis=0)
    power = (np.abs(spectra[: n_fast // 2]) ** 2).mean(axis=(1, 2))
    return float(power[mask].sum() / max(power.sum(), EPS))


def qc_smoke_frame(frame: np.ndarray, mask: np.ndarray, *, bins: int,
                   max_bin_fraction: float, min_ratio: float) -> dict:
    """One 77 GHz frame (fast x chirp x rx) against the frozen rules — dict form.

    Promoted from the audit (its JSON report and tests use the dict). Shares the flatline
    core with run_qc_frame_77. A non-finite frame short-circuits, mirroring the 10 GHz
    per-frame contract.
    """
    if not np.all(np.isfinite(frame)):
        return {"nan_inf": True, "flatline": False, "low_in_band": False,
                "in_band_ratio": float("nan"), "n_flatline_traces": 0, "passed": False,
                "per_rx_flatline": [0] * frame.shape[2]}

    per_rx = _flatline_per_rx(frame, bins=bins, max_bin_fraction=max_bin_fraction)
    n_flatline = int(sum(per_rx))
    ratio = _in_band_ratio_77(frame, mask)
    return {
        "nan_inf": False,
        "flatline": n_flatline > 0,
        "low_in_band": ratio < min_ratio,
        "in_band_ratio": ratio,
        "n_flatline_traces": n_flatline,
        "per_rx_flatline": per_rx,
        "passed": n_flatline == 0 and ratio >= min_ratio,
    }


def run_qc_frame_77(frame: np.ndarray, qc77: QC77Config, pre77: Preprocess77Config) -> FrameQC77:
    """Screen one raw 77 GHz frame: real [n_fast, n_chirp, n_rx].

    Shape-generic (reads its dims from the array) so it runs at any fixture size; only the
    loader hard-asserts the real dimensions. Non-finite short-circuits.
    """
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"frame has shape {frame.shape}, expected (n_fast, n_chirp, n_rx)")

    if not np.all(np.isfinite(frame)):
        return FrameQC77(
            nan_inf=True, flatline=False, low_in_band=False,
            in_band_ratio=float("nan"), n_flatline_traces=0,
            per_rx_flatline=(0,) * frame.shape[2],
        )

    per_rx = _flatline_per_rx(
        frame, bins=qc77.histogram_bins, max_bin_fraction=qc77.flatline_max_bin_fraction,
        skip_leading=qc77.flatline_skip_leading_bins,
    )
    n_flatline = int(sum(per_rx))

    mask = in_band_mask(
        frame.shape[0], pre77.fs_hz, pre77.bandwidth_hz, pre77.chirp_time_s,
        pre77.gate_m, qc77.in_band_margin_hz,
    )
    ratio = _in_band_ratio_77(frame, mask)

    return FrameQC77(
        nan_inf=False,
        flatline=n_flatline > 0,
        low_in_band=ratio < qc77.min_in_band_energy_ratio,
        in_band_ratio=ratio,
        n_flatline_traces=n_flatline,
        per_rx_flatline=tuple(per_rx),
    )


def run_qc_cube_77(cube: np.ndarray, qc77: QC77Config, pre77: Preprocess77Config) -> list[FrameQC77]:
    """Screen a whole session cube: real [n_frames, n_fast, n_chirp, n_rx].

    A plain loop over the leading frame axis, deliberately: no statistic may be shared
    between frames (structurally leak-proof).
    """
    cube = np.asarray(cube)
    if cube.ndim != 4:
        raise ValueError(
            f"cube has shape {cube.shape}, expected (n_frames, n_fast, n_chirp, n_rx)"
        )
    return [run_qc_frame_77(cube[i], qc77, pre77) for i in range(cube.shape[0])]
