"""The 77 GHz preprocessing chain — steps 1-5 of the Exp G primary chain (band 2, M5).

The executable per-frame front of the slow-time I/Q chain: MTI -> fast-time Butterworth
bandpass -> Hann(fast) -> range FFT -> crop to the 2-4 m gate. Reuses the fs-agnostic
filters.py. No EdgeTrim and no reduction — the Doppler slow-time chain has neither
(wst_extract77.m's EdgeTrim=8 belongs to the deferred fast-time secondary branch, A-M5-7).

The invariant: this is a deterministic per-frame function of (one frame, frozen constants).
MTI subtracts a WITHIN-frame mean over that frame's own chirps — no cross-frame statistic.
The complex range-FFT output (step 4) is the FIRST point I/Q exists (the M2 real-float
finding: nothing before step 4 may assume complex input). filtfilt is zero-phase (no
range-peak shift).
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import hann

from ..config import Preprocess77Config, beat_band_hz
from ..qc.axis_check_77 import range_gate_bins
from .filters import bandpass_filtfilt, design_bandpass_sos

# The chain window is the SYMMETRIC Hann (MATLAB hann(N) default, what
# chirpavg_and_fuse_batch.m calls) — deliberately NOT the periodic window the QC screens
# use. Its zero endpoints also neutralise the range-bin-0 frame counter before the FFT.
CHIRP_AXIS = 1  # frame is [n_fast, n_chirp, n_rx]; MTI averages over chirps
FAST_AXIS = 0


def _bandpass_sos(pre77: Preprocess77Config) -> np.ndarray:
    f_lo, f_hi = beat_band_hz(pre77.gate_m, pre77.bandwidth_hz, pre77.chirp_time_s)
    return design_bandpass_sos(f_lo, f_hi, pre77.fs_hz, pre77.butter_order)


def preprocess_frame_77(frame: np.ndarray, pre77: Preprocess77Config) -> np.ndarray:
    """Chain steps 1-5 on one real frame [n_fast, n_chirp, n_rx] -> complex [n_gate, n_chirp, n_rx].

    Shape-generic (reads its dims from the array); only the loader hard-asserts (256,256,16).
    """
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"frame has shape {frame.shape}, expected (n_fast, n_chirp, n_rx)")
    n_fast = frame.shape[FAST_AXIS]

    # 1. MTI: subtract this frame's per-fast-bin mean over chirps (static clutter removal).
    mti = frame - frame.mean(axis=CHIRP_AXIS, keepdims=True)

    # 2. fast-time zero-phase Butterworth over the gate's beat band (real in -> real out).
    filtered = bandpass_filtfilt(mti, _bandpass_sos(pre77), axis=FAST_AXIS)

    # 3. Hann(fast): the chain window (its zero endpoints kill the bin-0 counter).
    windowed = filtered * hann(n_fast, sym=True).reshape(n_fast, 1, 1)

    # 4. range FFT (256-pt) along fast time — the first point I/Q exists.
    range_fft = np.fft.fft(windowed, axis=FAST_AXIS)

    # 5. crop to the 2-4 m gate (bins 27..53 at n_fast=256).
    gate_bins = range_gate_bins(n_fast, pre77.bandwidth_hz, pre77.gate_m)
    return range_fft[gate_bins]


def preprocess_cube_77(cube: np.ndarray, pre77: Preprocess77Config) -> np.ndarray:
    """Chain steps 1-5 over a whole session cube [N, n_fast, n_chirp, n_rx] -> [N, n_gate, n_chirp, n_rx].

    A plain per-frame loop, deliberately: no statistic is shared between frames.
    """
    cube = np.asarray(cube)
    if cube.ndim != 4:
        raise ValueError(
            f"cube has shape {cube.shape}, expected (N, n_fast, n_chirp, n_rx)"
        )
    return np.stack([preprocess_frame_77(cube[i], pre77) for i in range(cube.shape[0])], axis=0)


def _stage_energy(x: np.ndarray, transform_length: int = 1) -> float:
    """Total energy under the Parseval convention (the audit's normalized_energy).

    numpy's FFT is unnormalised (a length-N transform scales total spectral energy by N),
    so spectral-stage energies are divided by the cumulative transform length; without it,
    stage-to-stage ratios are not comparable.
    """
    return float(np.sum(np.abs(x) ** 2) / transform_length)


def chain_stages_77(frame: np.ndarray, pre77: Preprocess77Config) -> list[dict]:
    """Per-stage energies of the chain on one frame — the run_preprocess77 diagnostic.

    Mirrors the audit's chain_stages convention (Parseval normalisation by the cumulative
    transform length), so a stage's energy is comparable across the chain.
    """
    frame = np.asarray(frame)
    n_fast = frame.shape[FAST_AXIS]
    stages = [{"stage": "raw_frame", "energy": _stage_energy(frame)}]

    mti = frame - frame.mean(axis=CHIRP_AXIS, keepdims=True)
    stages.append({"stage": "mti", "energy": _stage_energy(mti)})

    filtered = bandpass_filtfilt(mti, _bandpass_sos(pre77), axis=FAST_AXIS)
    stages.append({"stage": "bandpass", "energy": _stage_energy(filtered)})

    windowed = filtered * hann(n_fast, sym=True).reshape(n_fast, 1, 1)
    range_fft = np.fft.fft(windowed, axis=FAST_AXIS)
    stages.append({"stage": "range_fft", "energy": _stage_energy(range_fft, n_fast)})

    gate_bins = range_gate_bins(n_fast, pre77.bandwidth_hz, pre77.gate_m)
    stages.append({"stage": "range_gate_crop", "energy": _stage_energy(range_fft[gate_bins], n_fast)})
    return stages
