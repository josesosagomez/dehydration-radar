"""The 77 GHz preprocessing chain (T-P77, no private data).

Shape-generic, so most behaviour is tested at small chirp/rx sizes; the gate crop uses the
real n_fast=256. Fixtures are real cosine range/Doppler tones (the loaded 77 GHz cube is real
float64), constructed so MTI, the bandpass, and the range crop each have a checkable effect.
"""

import numpy as np
import pytest

from dehyd.config import Preprocess77Config
from dehyd.preprocess.filters import bandpass_filtfilt, design_bandpass_sos
from dehyd.preprocess.pipeline_77 import (
    _bandpass_sos,
    chain_stages_77,
    preprocess_cube_77,
    preprocess_frame_77,
)
from dehyd.qc.axis_check_77 import range_gate_bins

PRE = Preprocess77Config()
N_FAST = 256
GATE_BIN = 40  # ~3 m: beat 78.1 kHz, inside the 2-4 m band; FFT bin 40 is in the gate 27..53


def static_frame(n_chirp=64, n_rx=2, range_bin=GATE_BIN):
    """A range tone constant across chirps — a STATIC target (MTI should kill it)."""
    f, c, r = np.indices((N_FAST, n_chirp, n_rx))
    return np.cos(2 * np.pi * range_bin * f / N_FAST)


def moving_frame(n_chirp=64, n_rx=2, range_bin=GATE_BIN, dopp_bin=5):
    """A range tone modulated by a Doppler phase across chirps — survives MTI."""
    f, c, r = np.indices((N_FAST, n_chirp, n_rx))
    return np.cos(2 * np.pi * range_bin * f / N_FAST + 2 * np.pi * dopp_bin * c / n_chirp)


# -------------------------------------------------------------------------------- MTI


def test_mti_kills_a_static_target():
    stages = chain_stages_77(static_frame(), PRE)
    raw, mti = stages[0]["energy"], stages[1]["energy"]
    assert mti / raw < 1e-20  # a chirp-constant target is removed to numerical zero


def test_mti_preserves_a_moving_target():
    stages = chain_stages_77(moving_frame(), PRE)
    raw, mti = stages[0]["energy"], stages[1]["energy"]
    assert mti / raw == pytest.approx(1.0, abs=1e-6)  # a Doppler tone is untouched by MTI


# ---------------------------------------------------------------------------- geometry


def test_gate_crop_bins_pinned():
    gate = range_gate_bins(N_FAST, PRE.bandwidth_hz, PRE.gate_m)
    assert gate.tolist() == list(range(27, 54))  # bins 27..53, 27 bins


def test_output_shape_is_27_gate_bins():
    out = preprocess_frame_77(moving_frame(n_chirp=32, n_rx=4), PRE)
    assert out.shape == (27, 32, 4)
    assert out.dtype == np.complex128  # the range FFT is where I/Q first exists


# ------------------------------------------------------------------- axis correctness


def test_range_peak_lands_in_the_gate_at_the_expected_bin():
    """A moving target at gate range peaks at its range bin after the chain (no shift)."""
    out = preprocess_frame_77(moving_frame(), PRE)  # [27, n_chirp, n_rx]
    power = (np.abs(out) ** 2).sum(axis=(1, 2))  # over chirp, rx -> per gate bin
    peak = int(power.argmax())
    assert peak == GATE_BIN - 27  # relative index within the crop = absolute bin 40


def test_fast_chirp_swap_does_not_produce_gate_energy():
    """The chain FFTs along fast; a signal whose range structure sits on the CHIRP axis
    instead yields far less gate energy — the executable form of the axis assumption."""
    correct = moving_frame(n_chirp=N_FAST, n_rx=2)          # range along fast
    swapped = correct.transpose(1, 0, 2)                    # interchange fast <-> chirp
    e_correct = float((np.abs(preprocess_frame_77(correct, PRE)) ** 2).sum())
    e_swapped = float((np.abs(preprocess_frame_77(swapped, PRE)) ** 2).sum())
    assert e_correct > 50 * e_swapped


# ------------------------------------------------------------------------ zero-phase


def test_bandpass_is_zero_phase_no_peak_shift():
    """filtfilt is forward-backward, so a symmetric fast-time pulse keeps its centre."""
    x = np.zeros(N_FAST)
    x[128] = 1.0  # a centred impulse
    y = bandpass_filtfilt(x, _bandpass_sos(PRE), axis=0)
    assert int(np.argmax(np.abs(y))) == 128  # no group-delay shift


# ------------------------------------------------------------- energy accounting (Parseval)


def test_range_fft_stage_uses_the_parseval_convention():
    """range_fft stage energy (÷ n_fast) equals the pre-FFT windowed energy."""
    from scipy.signal.windows import hann

    frame = moving_frame()
    stages = chain_stages_77(frame, PRE)
    # Reconstruct the pre-FFT signal to check the normalisation convention.
    mti = frame - frame.mean(axis=1, keepdims=True)
    filtered = bandpass_filtfilt(mti, _bandpass_sos(PRE), axis=0)
    windowed = filtered * hann(N_FAST, sym=True).reshape(N_FAST, 1, 1)
    windowed_energy = float(np.sum(np.abs(windowed) ** 2))
    range_fft_energy = next(s["energy"] for s in stages if s["stage"] == "range_fft")
    assert range_fft_energy == pytest.approx(windowed_energy, rel=1e-9)


def test_crop_removes_energy():
    stages = chain_stages_77(moving_frame(), PRE)
    fft_e = next(s["energy"] for s in stages if s["stage"] == "range_fft")
    crop_e = next(s["energy"] for s in stages if s["stage"] == "range_gate_crop")
    assert 0 < crop_e < fft_e  # the gate keeps a strict, non-empty subset of range bins


# ------------------------------------------------------------------------ determinism


def test_cube_is_per_frame_and_deterministic():
    cube = np.stack([moving_frame(n_chirp=16, n_rx=2, dopp_bin=d) for d in (2, 5, 7)], axis=0)
    first = preprocess_cube_77(cube, PRE)
    second = preprocess_cube_77(cube, PRE)
    np.testing.assert_array_equal(first, second)
    # Each cube slice equals the per-frame call.
    for i in range(cube.shape[0]):
        np.testing.assert_array_equal(first[i], preprocess_frame_77(cube[i], PRE))


def test_bandpass_sos_matches_the_gate_band():
    """The chain's bandpass is designed on the 2-4 m beat band at fs=500 kHz, order 4."""
    from dehyd.config import beat_band_hz

    f_lo, f_hi = beat_band_hz(PRE.gate_m, PRE.bandwidth_hz, PRE.chirp_time_s)
    expected = design_bandpass_sos(f_lo, f_hi, PRE.fs_hz, PRE.butter_order)
    np.testing.assert_array_equal(_bandpass_sos(PRE), expected)
    assert expected.shape[0] == 4  # order 4 -> 4 second-order sections
