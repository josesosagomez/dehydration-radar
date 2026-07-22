"""Milestone 3 — self-consistency tests for the preprocessing sequence.

Correctness is established by Python-native checks (filter response, zero-phase,
finite-record energy, Option-B ROI/mask arithmetic, determinism), never by numeric
comparison against the MATLAB reference.

Every constant is read from the config object. A test that re-hardcoded 4 / 32 /
[1, 2] / 500 would pass vacuously the moment code and config drifted apart.

Fixtures always carry a little seeded noise: a noiseless tone is degenerate (its MAD is
~0 and, at M2, its magnitude histogram was too narrow to bin) and is not a valid stand-
in for a clean frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import sosfiltfilt, sosfreqz
from scipy.signal.windows import hann

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dehyd.config import PreprocessConfig, beat_band_hz  # noqa: E402
from dehyd.data.loader_10ghz import N_CHIRPS, N_FAST_TIME  # noqa: E402
from dehyd.preprocess.filters import (  # noqa: E402
    FilterError,
    apply_band_gate,
    bandpass_filtfilt,
    default_padlen,
    design_bandpass_sos,
    fft_gate,
    filter_spec,
)
from dehyd.preprocess.reduce import (  # noqa: E402
    ReduceError,
    detect_option_b_peak,
    edge_trim,
    option_b_mask,
    option_b_roi_bins,
    reduce_option_a,
    reduce_option_b,
)
from dehyd.preprocess.pipeline import (  # noqa: E402
    PipelineError,
    preprocess_cube,
    preprocess_frame,
)
from dehyd.preprocess.standardize import (  # noqa: E402
    StandardizeError,
    meanstd_standardize,
    robust_standardize,
    standardize,
    to_channels,
)

SEED = 20260723


@pytest.fixture
def pre() -> PreprocessConfig:
    """The canonical primary spec (frozen dataclass defaults)."""
    return PreprocessConfig()


@pytest.fixture
def band(pre) -> tuple[float, float]:
    return beat_band_hz(pre.model_gate_m, pre.bandwidth_hz, pre.chirp_time_s)


@pytest.fixture
def f_mid(band) -> float:
    """Mid-band = the ARITHMETIC centre, 4886.2 Hz at the default gate.

    One definition, used by every probe here — it is the frequency the T-PP6
    regression values were measured at.
    """
    return 0.5 * (band[0] + band[1])


@pytest.fixture
def sos(pre, band):
    return design_bandpass_sos(band[0], band[1], pre.fs_hz, pre.butter_order)


def tone(freq_hz: float, n: int, fs_hz: float, *, phase: float = 0.0) -> np.ndarray:
    """A unit-amplitude complex exponential — the natural probe for a complex signal."""
    t = np.arange(n) / fs_hz
    return np.exp(2j * np.pi * freq_hz * t + 1j * phase)


def energy(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x) ** 2))


# =============================================================== filters (T-PP1..T-PP8)


def test_pp1_design_from_model_gate_structure_and_padlen(pre, band, sos):
    """T-PP1: right band, right structure, stable poles, and the frozen padlen."""
    # The band is the MODEL gate, which at defaults genuinely differs from the QC gate.
    from dehyd.config import QCConfig

    qc_band = beat_band_hz(QCConfig().qc_gate_m, pre.bandwidth_hz, pre.chirp_time_s)
    assert band != qc_band
    assert band[0] == pytest.approx(3257.5, abs=1.0)
    assert band[1] == pytest.approx(6514.9, abs=1.0)

    # order 4 bandpass -> 8 poles -> 4 second-order sections
    assert sos.shape == (4, 6)

    for section in sos:
        poles = np.roots(section[3:])
        assert np.all(np.abs(poles) < 1.0), "SOS section is not stable"

    # padlen is frozen by explicit passing; 27 for this design under scipy 1.16.3.
    assert default_padlen(sos) == 27

    # ...and passing it explicitly must reproduce the library default bit-for-bit.
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(N_FAST_TIME)
    assert np.array_equal(
        sosfiltfilt(sos, x, padtype="odd", padlen=27), sosfiltfilt(sos, x)
    )


def test_pp2_steady_state_response_passes_in_band_and_stops_outside(pre, band, f_mid, sos):
    """T-PP2: the DESIGN response |H|^2 (filtfilt applies H twice).

    This is a statement about the filter, not about a 534-sample record — the
    finite-record behaviour is a separate claim, pinned in T-PP6.
    """
    f_lo, f_hi = band
    probes = np.array([f_lo, f_mid, f_hi, 0.5 * f_lo, 2.0 * f_hi])
    _, h = sosfreqz(sos, worN=2 * np.pi * probes / pre.fs_hz)
    gain_sq = np.abs(h) ** 2  # forward-backward

    assert gain_sq[1] >= 0.99, "mid-band should pass"
    # -3 dB design corners become -6 dB (0.5) under the squaring.
    assert gain_sq[0] == pytest.approx(0.5, abs=0.02)
    assert gain_sq[2] == pytest.approx(0.5, abs=0.02)
    assert gain_sq[3] <= 1e-4 and gain_sq[4] <= 1e-4, "should stop well outside the band"


def test_pp3_forward_backward_is_zero_phase(pre, f_mid, sos):
    """T-PP3: zero group delay on a test tone.

    A Gaussian-enveloped mid-band burst (sigma = 80 samples: narrow enough that its
    spectrum fits inside the ~3.3 kHz passband, wide enough that the envelope peak is
    well defined) must come out with its peak where it went in.

    The contrast with a single causal pass is what gives the test teeth — the same
    filter applied once shifts this burst by ~131 samples.
    """
    n = N_FAST_TIME
    centre = n // 2
    t = np.arange(n)
    envelope = np.exp(-0.5 * ((t - centre) / 80.0) ** 2)
    x = envelope * tone(f_mid, n, pre.fs_hz)

    y = bandpass_filtfilt(x, sos)
    assert int(np.argmax(np.abs(y))) == centre

    lags = np.arange(-(n - 1), n)

    def peak_lag(out: np.ndarray) -> int:
        a = np.abs(x) - np.abs(x).mean()
        b = np.abs(out) - np.abs(out).mean()
        return int(lags[int(np.argmax(np.correlate(b, a, mode="full")))])

    assert peak_lag(y) == 0

    from scipy.signal import sosfilt

    causal = sosfilt(sos, x.real) + 1j * sosfilt(sos, x.imag)
    assert peak_lag(causal) > 100, "a causal pass must fail this test, or it proves nothing"


def test_pp4_complex_is_filtered_as_real_and_imag(pre, f_mid, sos):
    """T-PP4: complex handling. Gain/energy claims live in T-PP6, not here."""
    rng = np.random.default_rng(SEED)
    x = tone(f_mid, N_FAST_TIME, pre.fs_hz) + 0.01 * (
        rng.standard_normal(N_FAST_TIME) + 1j * rng.standard_normal(N_FAST_TIME)
    )

    y = bandpass_filtfilt(x, sos)

    assert y.dtype == np.complex128
    expected = bandpass_filtfilt(x.real, sos) + 1j * bandpass_filtfilt(x.imag, sos)
    assert np.allclose(y, expected, rtol=0, atol=1e-12)
    # A real filter commutes with conjugation.
    assert np.allclose(bandpass_filtfilt(np.conj(x), sos), np.conj(y), rtol=0, atol=1e-12)

    # The in-band tone keeps its frequency (its spectral peak bin is unchanged).
    peak_in = int(np.argmax(np.abs(np.fft.fft(x))))
    assert int(np.argmax(np.abs(np.fft.fft(y)))) == peak_in


def test_pp5_batched_filtering_equals_per_chirp(pre, sos):
    """T-PP5: the cube-path reshape cannot change semantics."""
    rng = np.random.default_rng(SEED)
    cube = rng.standard_normal((N_FAST_TIME, N_CHIRPS, 3)) + 1j * rng.standard_normal(
        (N_FAST_TIME, N_CHIRPS, 3)
    )

    batched = bandpass_filtfilt(cube.reshape(N_FAST_TIME, -1), sos).reshape(cube.shape)

    for chirp in range(N_CHIRPS):
        for frame in range(3):
            one = bandpass_filtfilt(cube[:, chirp, frame], sos)
            assert np.array_equal(batched[:, chirp, frame], one)


def test_pp6_finite_record_energy_regression(pre, f_mid, sos):
    """T-PP6: what a 534-sample sosfiltfilt ACTUALLY does — measured, not designed.

    These are regression values (scipy 1.16.3, order-4 SOS on the 1-2 m band,
    padtype='odd', padlen=27), documented in MILESTONE_3_PLAN.md §2.2. The band is
    narrow relative to fs, so the filter's transient occupies a large share of the
    record: mid-band retention is ~0.76, not ~1, and the stopband reaches ~-17 dB, not
    the steady-state figure. Trimming 32 samples per end improves both, which is the
    empirical case for EdgeTrim.

    No filter parameter may be changed to make these approach the ideal — that would be
    a design decision, and it would have to be logged as one.
    """
    n, trim = N_FAST_TIME, pre.edge_trim
    assert f_mid == pytest.approx(4886.2, abs=1.0)

    x_in = tone(f_mid, n, pre.fs_hz)
    y_in = bandpass_filtfilt(x_in, sos)
    assert energy(y_in) / energy(x_in) == pytest.approx(0.7595, abs=0.005)
    assert energy(y_in[trim:-trim]) / energy(x_in[trim:-trim]) == pytest.approx(0.8313, abs=0.005)

    x_out = tone(50_000.0, n, pre.fs_hz)
    y_out = bandpass_filtfilt(x_out, sos)
    assert 10 * np.log10(energy(y_out) / energy(x_out)) <= -17.0
    assert 10 * np.log10(
        energy(y_out[trim:-trim]) / energy(x_out[trim:-trim])
    ) <= -20.0

    # Parseval, with the orthonormal convention.
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert energy(np.fft.fft(z, norm="ortho")) == pytest.approx(energy(z), rel=1e-12)


def test_pp7_fft_gate_mask_shape_and_effect(pre, band, f_mid):
    """T-PP7: the ablation gate matches filter_gpt_fft.m semantics."""
    n, fs = N_FAST_TIME, pre.fs_hz
    f_lo, f_hi = band
    tw = pre.fft_gate_transition_hz

    # Recover the mask by gating unit impulses in each bin (mask = FFT-domain gain).
    spectrum = np.fft.fft(fft_gate(np.fft.ifft(np.ones(n)), f_lo, f_hi, fs, tw))
    mask = np.abs(spectrum)
    freqs = np.abs(np.fft.fftfreq(n, d=1.0 / fs))

    assert np.allclose(mask[(freqs >= f_lo) & (freqs <= f_hi)], 1.0, atol=1e-12)
    assert np.allclose(mask[freqs > f_hi + tw], 0.0, atol=1e-12)
    assert np.allclose(mask[(freqs < f_lo - tw) & (freqs > 0)], 0.0, atol=1e-12)
    skirt = mask[(freqs > f_hi) & (freqs < f_hi + tw)]
    if skirt.size:
        assert np.all((skirt > 0.0) & (skirt < 1.0)), "skirt should be strictly tapered"

    # Symmetric in |f|: bin k and bin n-k carry the same weight.
    assert np.allclose(mask[1 : n // 2], mask[n - 1 : n // 2 : -1], atol=1e-12)

    # In-band survives, out-of-band does not; complex in -> complex out.
    x_in = tone(f_mid, n, fs)
    x_out = tone(50_000.0, n, fs)
    assert energy(fft_gate(x_in, f_lo, f_hi, fs, tw)) / energy(x_in) > 0.9
    assert energy(fft_gate(x_out, f_lo, f_hi, fs, tw)) / energy(x_out) < 1e-3
    assert fft_gate(x_in, f_lo, f_hi, fs, tw).dtype == np.complex128


def test_pp8_gate_parameters_come_from_config(pre):
    """T-PP8: changing the gate moves the passband; gate_method switches implementation."""
    from dataclasses import replace

    n = N_FAST_TIME
    wide = replace(pre, model_gate_m=(0.9, 3.0))
    # ~2.6 m sits inside the wide gate but outside the default 1-2 m gate.
    x = tone(2.6 * (2 * (pre.bandwidth_hz / pre.chirp_time_s) / 299_792_458.0), n, pre.fs_hz)

    kept = energy(apply_band_gate(x, wide)) / energy(x)
    rejected = energy(apply_band_gate(x, pre)) / energy(x)
    # ~0.81 vs ~0.035. The rejected figure is not smaller because a 534-sample record
    # leaks (T-PP6): the honest claim is the ratio, not an idealised stopband floor.
    assert kept > 0.5
    assert rejected < 0.1
    assert kept > 10 * rejected

    # The two methods are genuinely different implementations of the same band.
    fft_cfg = replace(pre, gate_method="fft")
    assert not np.allclose(apply_band_gate(x, pre), apply_band_gate(x, fft_cfg))

    spec = filter_spec(pre)
    assert spec["padlen"] == 27 and spec["n_sections"] == 4
    assert spec["gate_method"] == "butterworth"
    assert filter_spec(fft_cfg)["fft_gate_transition_hz"] == pre.fft_gate_transition_hz


def test_design_rejects_band_outside_nyquist(pre):
    """The filter-level guard mirroring the config-level one."""
    with pytest.raises(FilterError, match="not strictly inside"):
        design_bandpass_sos(3000.0, pre.fs_hz, pre.fs_hz, pre.butter_order)


# ============================================================== reduce (T-PP9..T-PP14)


def frame_from_spectrum(spectrum: np.ndarray, n_chirps: int = N_CHIRPS) -> np.ndarray:
    """A frame whose every chirp has exactly the given unwindowed FFT."""
    chirp = np.fft.ifft(spectrum)
    return np.repeat(chirp[:, None], n_chirps, axis=1)


def bin_freq(k: int, pre: PreprocessConfig, n: int = N_FAST_TIME) -> float:
    return k * pre.fs_hz / n


def test_pp9_option_a_is_the_chirp_mean(pre):
    """T-PP9: Option A is exactly the mean across chirps."""
    rng = np.random.default_rng(SEED)
    frame = rng.standard_normal((N_FAST_TIME, N_CHIRPS)) + 1j * rng.standard_normal(
        (N_FAST_TIME, N_CHIRPS)
    )
    reduced = reduce_option_a(frame)
    assert reduced.shape == (N_FAST_TIME,)
    assert np.array_equal(reduced, frame.sum(axis=1) / N_CHIRPS)


def test_pp10_roi_bins_exact_arithmetic(pre):
    """T-PP10: the ROI is the MODEL gate with NO margin, computed independently here."""
    from dataclasses import replace

    df = pre.fs_hz / N_FAST_TIME
    assert df == pytest.approx(975.34, abs=0.01)

    f_lo, f_hi = beat_band_hz(pre.model_gate_m, pre.bandwidth_hz, pre.chirp_time_s)
    expected = [k for k in range(N_FAST_TIME // 2) if f_lo <= k * df <= f_hi]
    assert expected == [4, 5, 6]
    assert list(option_b_roi_bins(pre, N_FAST_TIME)) == expected

    # The 0.9-3.0 m candidate gate: bin 3 (2926.0 Hz) misses the 2931.7 Hz edge.
    wide = replace(pre, model_gate_m=(0.9, 3.0))
    assert list(option_b_roi_bins(wide, N_FAST_TIME)) == [4, 5, 6, 7, 8, 9, 10]
    assert 3 * df < beat_band_hz(wide.model_gate_m, wide.bandwidth_hz, wide.chirp_time_s)[0]

    # A gate too narrow to contain any bin is an error, not an empty selection.
    with pytest.raises(ReduceError, match="no FFT bin"):
        option_b_roi_bins(replace(pre, model_gate_m=(1.0, 1.001)), N_FAST_TIME)


def test_pp11_detection_is_roi_restricted_and_single_sourced(pre):
    """T-PP11: the ROI restriction is real, and ONE result feeds mask + diagnostics."""
    n = N_FAST_TIME
    rng = np.random.default_rng(SEED)
    # An in-ROI tone at bin 5, and a TEN TIMES stronger tone at bin 20 outside it.
    x = tone(bin_freq(5, pre), n, pre.fs_hz) + 10.0 * tone(bin_freq(20, pre), n, pre.fs_hz)
    frame = np.repeat(x[:, None], N_CHIRPS, axis=1) + 0.01 * rng.standard_normal(
        (n, N_CHIRPS)
    )

    detection = detect_option_b_peak(frame, pre)
    assert detection.peak_bin == 5, "argmax must be restricted to the ROI"
    assert list(detection.roi_bins) == [4, 5, 6]
    assert detection.power.shape == (n // 2,)

    # The detection spectrum really is the periodic-Hann, chirp-averaged power.
    window = hann(n, sym=False)
    expected = (np.abs(np.fft.fft(frame * window[:, None], axis=0)[: n // 2, :]) ** 2).mean(axis=1)
    assert np.allclose(detection.power, expected, rtol=0, atol=1e-12)

    # Single-sourcing: the mask reduce_option_b applies is centred on THIS peak_bin.
    reduced = reduce_option_b(frame, pre)
    spectrum = np.fft.fft(reduced)
    kept = np.flatnonzero(np.abs(spectrum) > 1e-9 * np.abs(spectrum).max())
    assert set(kept) <= {4, 5, 6, n - 4, n - 5, n - 6}
    assert int(np.argmax(np.abs(spectrum[: n // 2]))) == detection.peak_bin


@pytest.mark.parametrize(
    "nb, expected_weights",
    [
        (0, [1.0]),
        (1, [0.5, 1.0, 0.5]),
        (2, [0.25, 0.75, 1.0, 0.75, 0.25]),
    ],
)
def test_pp12_mask_weights_across_the_nb_domain(pre, nb, expected_weights):
    """T-PP12 (weights): the peak keeps FULL weight — the reference zeroes it."""
    from dataclasses import replace

    cfg = replace(pre, peak_neighbors=nb)
    peak, n = 50, N_FAST_TIME
    mask = option_b_mask(peak, n, cfg)

    offsets = np.arange(-nb, nb + 1)
    assert np.allclose(mask[peak + offsets], expected_weights)
    assert mask[peak] == 1.0
    # Mirrored with the same weights, and nothing else kept.
    assert np.allclose(mask[(n - (peak + offsets)) % n], expected_weights)
    assert np.count_nonzero(mask) == 2 * len(expected_weights)


def test_pp12_mask_audit_taper_rectangular_and_amplitude(pre):
    """T-PP12 (effect): energy only at peak+/-1 and mirrors; an on-bin tone survives."""
    from dataclasses import replace

    n = N_FAST_TIME
    peak = 5
    frame = frame_from_spectrum(np.eye(n, dtype=complex)[peak] * n)  # unit tone at bin 5

    reduced = reduce_option_b(frame, pre)
    spectrum = np.fft.fft(reduced)
    assert np.abs(spectrum[peak]) / n == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(np.abs(reduced), 1.0, atol=1e-9), "on-bin tone must survive intact"

    # Rectangular masking keeps the same bins with weight 1.
    rect = option_b_mask(peak, n, replace(pre, mask_taper=False))
    assert set(np.unique(rect)) == {0.0, 1.0}
    assert np.count_nonzero(rect) == 6  # 3 positive + 3 mirrors


def test_pp12_mask_clamps_at_the_spectrum_edge(pre):
    """T-PP12 (edge): a peak at the top of the half-spectrum clamps, staying valid."""
    n = N_FAST_TIME
    mask = option_b_mask(n // 2, n, pre)  # peak at the Nyquist-adjacent bin
    assert np.all(np.isfinite(mask))
    assert 0 < np.count_nonzero(mask) < n


def test_pp12_mask_covering_every_bin_raises(pre):
    """T-PP12 (anti-vacuity): a pass-through Option B is an error, not a setting."""
    from dataclasses import replace

    with pytest.raises(ReduceError, match="pass the signal through"):
        option_b_mask(4, N_FAST_TIME, replace(pre, peak_neighbors=N_FAST_TIME))


def test_pp13_zero_roi_power_is_deterministic_not_zero_output(pre):
    """T-PP13: the documented fallback, pinned by a fixture that could expose a
    false "output is then ~0" assumption.

    (a) An exactly zero frame: every ROI power is 0, so the argmax tie-break must
        return the FIRST ROI bin.
    (b) The adversarial windowed-null: unwindowed bins 3..7 = [1, 0, -1, -2, -3].
        Convolving with the periodic-Hann kernel [-1/4, 1/2, -1/4] annihilates
        windowed bins 4, 5, 6 — the whole ROI — while the UNWINDOWED bins the mask
        keeps (3 and 5) are nonzero. Detection therefore has nothing to find, yet the
        reduced output is NOT zero.
    """
    n = N_FAST_TIME
    roi = option_b_roi_bins(pre, n)

    # (a) exact tie -> first ROI bin
    zero_frame = np.zeros((n, N_CHIRPS), dtype=complex)
    zero_detection = detect_option_b_peak(zero_frame, pre)
    assert np.all(zero_detection.power[roi] == 0.0)
    assert zero_detection.peak_bin == int(roi[0])
    assert np.all(reduce_option_b(zero_frame, pre) == 0.0)

    # (b) adversarial windowed null
    spectrum = np.zeros(n, dtype=complex)
    spectrum[3:8] = [1.0, 0.0, -1.0, -2.0, -3.0]
    frame = frame_from_spectrum(spectrum)

    detection = detect_option_b_peak(frame, pre)
    assert np.all(detection.power[roi] <= 1e-20 * detection.power.max())
    assert detection.peak_bin in set(roi.tolist())

    reduced = reduce_option_b(frame, pre)
    assert np.all(np.isfinite(reduced))
    assert np.sum(np.abs(reduced) ** 2) > 0.0, "no zero-output claim is made here"
    # ...and it equals the hand-computed masked, unwindowed reconstruction.
    mask = option_b_mask(detection.peak_bin, n, pre)
    expected = np.fft.ifft(np.fft.fft(frame, axis=0) * mask[:, None], axis=0).mean(axis=1)
    assert np.allclose(reduced, expected, rtol=0, atol=1e-15)
    assert np.array_equal(reduced, reduce_option_b(frame, pre))  # deterministic


def test_pp14_edge_trim_slices_and_refuses_to_clamp(pre):
    """T-PP14: 534 - 2*32 = 470, and an over-large trim raises rather than clamping."""
    rng = np.random.default_rng(SEED)
    signal = rng.standard_normal(N_FAST_TIME) + 1j * rng.standard_normal(N_FAST_TIME)

    trimmed = edge_trim(signal, pre.edge_trim)
    assert trimmed.size == 470
    assert np.array_equal(trimmed, signal[32:502])

    with pytest.raises(ReduceError, match="below the 32-sample floor"):
        edge_trim(signal, N_FAST_TIME // 2)
    with pytest.raises(ReduceError, match="below the 32-sample floor"):
        edge_trim(signal, 252)  # leaves 30 — just under the floor, so it must raise
    assert edge_trim(signal, 251).size == 32  # exactly the floor is allowed


# ========================================================= standardize (T-PP16..T-PP18)


def test_pp16_standardize_formulas_are_exact(pre):
    """T-PP16: both forms match hand computation; ddof=0 is pinned for the plain z."""
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(470) * 3.0 + 7.0

    median = np.median(x)
    mad = np.median(np.abs(x - median))
    expected = (x - median) / (1.4826 * mad + np.finfo(np.float64).eps)
    assert np.array_equal(robust_standardize(x), expected)

    # Centred and unit-scaled in its own (robust) terms.
    y = robust_standardize(x)
    assert np.median(y) == pytest.approx(0.0, abs=1e-12)
    assert 1.4826 * np.median(np.abs(y - np.median(y))) == pytest.approx(1.0, rel=1e-9)

    # A constant signal is finite, not NaN.
    constant = robust_standardize(np.full(470, 2.5))
    assert np.all(constant == 0.0)

    # ddof=0, not MATLAB's ddof=1 — the two differ by sqrt(N/(N-1)) and the exact
    # comparison below fails for the wrong convention.
    expected_plain = (x - x.mean()) / (x.std(ddof=0) + np.finfo(np.float64).eps)
    assert np.array_equal(meanstd_standardize(x), expected_plain)
    assert not np.allclose(meanstd_standardize(x), (x - x.mean()) / x.std(ddof=1))

    assert np.array_equal(standardize(x, "robust"), robust_standardize(x))
    assert np.array_equal(standardize(x, "meanstd"), meanstd_standardize(x))
    with pytest.raises(StandardizeError, match="unknown standardize method"):
        standardize(x, "zscore")


def test_pp17_robust_scale_resists_an_outlier_that_wrecks_meanstd():
    """T-PP17: the reason the primary path is robust, made executable."""
    rng = np.random.default_rng(SEED)
    clean = rng.standard_normal(470)
    spiked = clean.copy()
    spiked[100] = 500.0  # one absurd sample

    robust_shift = np.max(np.abs(robust_standardize(spiked)[:99] - robust_standardize(clean)[:99]))
    plain_shift = np.max(np.abs(meanstd_standardize(spiked)[:99] - meanstd_standardize(clean)[:99]))

    assert robust_shift < 0.05, "the bulk of the signal should barely move"
    assert plain_shift > 0.5, "mean/std should be visibly dragged by the outlier"
    assert plain_shift > 10 * robust_shift


def test_pp18_channels_are_standardized_separately(pre):
    """T-PP18: mag -> [1, N]; iq -> [2, N] with real and imag scaled independently."""
    rng = np.random.default_rng(SEED)
    signal = rng.standard_normal(470) + 1j * (3.0 * rng.standard_normal(470) + 10.0)

    mag = to_channels(signal, "mag", pre.standardize)
    assert mag.shape == (1, 470) and mag.dtype == np.float64
    assert np.array_equal(mag[0], robust_standardize(np.abs(signal)))

    iq = to_channels(signal, "iq", pre.standardize)
    assert iq.shape == (2, 470)
    # Each part from its OWN statistics — a shared scale would fail these.
    assert np.array_equal(iq[0], robust_standardize(signal.real))
    assert np.array_equal(iq[1], robust_standardize(signal.imag))

    # The method is honoured, not hard-coded.
    assert np.array_equal(
        to_channels(signal, "mag", "meanstd")[0], meanstd_standardize(np.abs(signal))
    )
    with pytest.raises(StandardizeError, match="unknown channel"):
        to_channels(signal, "phase", pre.standardize)


# ================================================ pipeline (T-PP15, T-PP19..T-PP22)


@pytest.fixture
def clean_frame(pre):
    """A QC-passable frame: an in-ROI target plus small seeded noise per chirp.

    Noise matters — a noiseless tone is degenerate (zero MAD; at M2 it also broke the
    QC histogram) and would not exercise standardization honestly.
    """
    rng = np.random.default_rng(SEED)
    target = tone(bin_freq(5, pre), N_FAST_TIME, pre.fs_hz)
    noise = 0.05 * (
        rng.standard_normal((N_FAST_TIME, N_CHIRPS))
        + 1j * rng.standard_normal((N_FAST_TIME, N_CHIRPS))
    )
    return target[:, None] + noise


def test_pp15_trim_happens_after_reduction(pre, clean_frame):
    """T-PP15: the ordering is structural, not a comment.

    Option B's detection FFT runs on the full 534-sample chirp. Trimming first would
    change the bin grid (470-point FFT: df = 1108 Hz, ROI bins 3..5 instead of 4..6),
    so the two orders genuinely disagree — which is what makes this test meaningful.
    """
    gated = apply_band_gate(clean_frame, pre, axis=0)

    correct = edge_trim(reduce_option_b(gated, pre), pre.edge_trim)
    trimmed_first = reduce_option_b(gated[pre.edge_trim : -pre.edge_trim, :], pre)

    assert correct.size == trimmed_first.size == 470
    assert not np.allclose(correct, trimmed_first), "the two orders must differ"

    produced = preprocess_frame(clean_frame, pre, reduction="b", channel="mag")
    assert np.array_equal(produced[0], robust_standardize(np.abs(correct)))


@pytest.mark.parametrize("reduction", ["a", "b"])
@pytest.mark.parametrize("channel", ["mag", "iq"])
@pytest.mark.parametrize("gate_method", ["butterworth", "fft"])
def test_pp19_pipeline_equals_manual_composition(pre, clean_frame, reduction, channel, gate_method):
    """T-PP19: the pipeline IS filter -> reduce -> trim -> channels, all 4 variants."""
    from dataclasses import replace

    cfg = replace(pre, gate_method=gate_method)

    gated = apply_band_gate(clean_frame, cfg, axis=0)
    reduced = reduce_option_a(gated) if reduction == "a" else reduce_option_b(gated, cfg)
    expected = to_channels(edge_trim(reduced, cfg.edge_trim), channel, cfg.standardize)

    produced = preprocess_frame(clean_frame, cfg, reduction=reduction, channel=channel)

    assert np.array_equal(produced, expected)
    assert produced.shape == (1 if channel == "mag" else 2, 470)
    assert produced.dtype == np.float64
    assert np.all(np.isfinite(produced))


def test_pp20_frame_in_cube_equals_frame_alone(pre, clean_frame):
    """T-PP20: per-frame independence — the M2 T-QC7 pattern, and the §0 invariant.

    A frame's output must not depend on its companions, or preprocessing would carry a
    cross-frame statistic into the CV loop.
    """
    rng = np.random.default_rng(SEED + 1)
    companion = rng.standard_normal((N_FAST_TIME, N_CHIRPS)) + 1j * rng.standard_normal(
        (N_FAST_TIME, N_CHIRPS)
    )
    loud = 1000.0 * companion  # an extreme neighbour, in case any scale were shared

    cube = np.stack([clean_frame, companion, loud], axis=2)
    out = preprocess_cube(cube, pre, reduction="b", channel="iq")

    assert out.shape == (3, 2, 470)
    for i, frame in enumerate([clean_frame, companion, loud]):
        alone = preprocess_frame(frame, pre, reduction="b", channel="iq")
        assert np.array_equal(out[i], alone)


def test_pp21_input_contract_is_enforced(pre, clean_frame):
    """T-PP21: exact shape and finiteness — a wrong axis must not silently 'work'."""
    with pytest.raises(PipelineError, match="expected shape"):
        preprocess_frame(clean_frame.T, pre, reduction="a", channel="mag")
    with pytest.raises(PipelineError, match="expected shape"):
        preprocess_frame(clean_frame[:100], pre, reduction="a", channel="mag")
    with pytest.raises(PipelineError, match="expected shape"):
        preprocess_cube(clean_frame, pre, reduction="a", channel="mag")

    for bad_value in (np.nan, np.inf):
        bad = clean_frame.copy()
        bad[7, 3] = bad_value
        with pytest.raises(PipelineError, match="non-finite"):
            preprocess_frame(bad, pre, reduction="a", channel="mag")

    with pytest.raises(PipelineError, match="reduction must be"):
        preprocess_frame(clean_frame, pre, reduction="c", channel="mag")
    with pytest.raises(PipelineError, match="channel must be"):
        preprocess_frame(clean_frame, pre, reduction="a", channel="power")


def test_pp22_pipeline_is_deterministic(pre, clean_frame):
    """T-PP22: bit-identical repeats — there is no RNG anywhere in the sequence."""
    for reduction in ("a", "b"):
        for channel in ("mag", "iq"):
            first = preprocess_frame(clean_frame, pre, reduction=reduction, channel=channel)
            second = preprocess_frame(clean_frame, pre, reduction=reduction, channel=channel)
            assert np.array_equal(first, second)


# ============================================= cohort diagnostics (T-PP23, T-PP24)
# run_preprocess.py keeps its logic in pure helpers (the M2 audit pattern), so these
# run without touching the cohort.

sys.path.insert(0, str(REPO_ROOT / "experiments"))

from run_preprocess import (  # noqa: E402
    NotThePrimarySpec,
    check_primary_spec,
    concentration,
    energy_retention,
    lowest_mode,
    median_skipping_missing,
    session_diagnostics,
)


def test_pp23_energy_retention_matches_the_summed_mixture(pre, sos, f_mid):
    """T-PP23 (retention): the expectation is computed on the ACTUAL mixture.

    Energy is quadratic, so component energies do NOT add: the filtered in-band and
    out-of-band parts are non-orthogonal on a finite record (cross-term ~0.007 here).
    Summing component energies would be a subtly different number — 0.38919 against the
    true 0.39608 — so the test filters the mixture itself.
    """
    n = N_FAST_TIME
    x1 = tone(f_mid, n, pre.fs_hz)
    x2 = tone(50_000.0, n, pre.fs_hz)
    mixture = x1 + x2

    frame = np.repeat(mixture[:, None], N_CHIRPS, axis=1)
    gated = apply_band_gate(frame, pre, axis=0)
    ratio = energy_retention(frame, gated)

    expected = energy(bandpass_filtfilt(mixture, sos)) / energy(mixture)
    assert ratio == pytest.approx(expected, rel=1e-12)
    assert ratio == pytest.approx(0.39608, abs=0.005)

    # The component-energy sum is a DIFFERENT number — it must not be used as the
    # expectation, and this pins the distinction so nobody "simplifies" it back.
    component_sum = (energy(bandpass_filtfilt(x1, sos)) + energy(bandpass_filtfilt(x2, sos))) / (
        energy(x1) + energy(x2)
    )
    assert component_sum == pytest.approx(0.38919, abs=0.005)
    assert abs(ratio - component_sum) > 1e-3

    # Signal-level linearity holds separately (and is not a statement about energy).
    assert np.allclose(
        bandpass_filtfilt(mixture, sos),
        bandpass_filtfilt(x1, sos) + bandpass_filtfilt(x2, sos),
        rtol=0,
        atol=1e-9,
    )

    # For THIS fixture, adding out-of-band energy lowers retention. A fixture property,
    # not a law: the cross-term can reverse it at other amplitudes or phases.
    in_band_only = np.repeat(x1[:, None], N_CHIRPS, axis=1)
    assert ratio < energy_retention(in_band_only, apply_band_gate(in_band_only, pre, axis=0))

    with pytest.raises(ValueError, match="zero energy"):
        energy_retention(np.zeros((n, N_CHIRPS), dtype=complex), gated)


def test_pp23_concentration_matches_hand_computation(pre):
    """T-PP23 (concentration): both measures come from the ONE detection result."""
    n = N_FAST_TIME
    rng = np.random.default_rng(SEED)
    x = tone(bin_freq(5, pre), n, pre.fs_hz) + 0.5 * tone(bin_freq(30, pre), n, pre.fs_hz)
    frame = np.repeat(x[:, None], N_CHIRPS, axis=1) + 0.01 * rng.standard_normal((n, N_CHIRPS))

    detection = detect_option_b_peak(frame, pre)
    roi_to_total, peak_share = concentration(detection)

    power, roi = detection.power, detection.roi_bins
    assert roi_to_total == pytest.approx(power[roi].sum() / power.sum(), rel=1e-12)
    assert peak_share == pytest.approx(power[detection.peak_bin] / power[roi].sum(), rel=1e-12)
    assert 0.0 < roi_to_total <= 1.0 and 0.0 < peak_share <= 1.0

    # Zero ROI power -> peak_share is NaN, never a fabricated 0.
    zero_detection = detect_option_b_peak(np.zeros((n, N_CHIRPS), dtype=complex), pre)
    zero_roi_to_total, zero_peak_share = concentration(zero_detection)
    assert zero_roi_to_total == 0.0
    assert np.isnan(zero_peak_share)


def test_pp23_mode_tie_breaks_toward_the_lowest_bin():
    """T-PP23 (tie rule): a crafted tie must resolve the same way on any library."""
    assert lowest_mode([5, 5, 4, 4, 7]) == 4  # 4 and 5 tie at 2 -> lowest wins
    assert lowest_mode([9, 9, 2, 2, 6, 6]) == 2  # three-way tie
    assert lowest_mode([6, 6, 6, 4]) == 6  # a clear mode is unaffected
    assert lowest_mode([3]) == 3


def test_pp23_aggregation_skips_missing_values(pre, clean_frame):
    """T-PP23 (aggregation): defined values only; all-missing -> NaN (an empty cell).

    The aggregation is tested on its own helper rather than through a crafted session,
    because a session in which `peak_share` is missing is not constructible from
    eligible frames: it is undefined only at exactly-zero ROI power, and a frame with
    any energy leaves float-positive power there — while a zero-energy frame is
    rejected by `energy_retention`'s contract. Keeping the two apart tests the rule
    without pretending an impossible input is possible.
    """
    assert median_skipping_missing([0.5, np.nan, 0.7]) == pytest.approx(0.6)
    assert median_skipping_missing([0.4]) == pytest.approx(0.4)
    assert np.isnan(median_skipping_missing([np.nan, np.nan]))
    assert np.isnan(median_skipping_missing([]))

    # And a real session reports every frame as defined, with finite variants.
    diagnostics = session_diagnostics(np.stack([clean_frame, clean_frame], axis=2), [0, 1], pre)
    assert diagnostics["n_peak_share_missing"] == 0
    assert diagnostics["all_variants_finite"] is True
    assert diagnostics["n_eligible_frames"] == 2
    assert 0.0 < diagnostics["energy_retention_median"] <= 1.0


@pytest.mark.parametrize(
    "override, expected_field",
    [
        ({"gate_method": "fft"}, "gate_method"),
        ({"standardize": "meanstd"}, "standardize"),
        ({"model_gate_m": (0.9, 3.0)}, "model_gate_m"),  # an inner-CV candidate, not primary
        ({"edge_trim": 16}, "edge_trim"),
        ({"peak_neighbors": 2}, "peak_neighbors"),
        ({"mask_taper": False}, "mask_taper"),
    ],
)
def test_pp24_primary_only_guard(pre, override, expected_field):
    """T-PP24: only the canonical spec may write the primary curated artifact."""
    from dataclasses import replace

    check_primary_spec(pre)  # the canonical spec is accepted

    with pytest.raises(NotThePrimarySpec, match=expected_field):
        check_primary_spec(replace(pre, **override))


# -------------------------------------------------------------------------- realdata


@pytest.mark.realdata
def test_pipeline_on_one_real_session(real_data_paths, capsys):
    """All four variants over the QC-passing frames of one real file.

    Structural assertions only. The peak-bin and concentration DISTRIBUTIONS are
    printed, never asserted: they are unknown until this runs, and turning them into
    expectations would be threshold-tuning through the back door (the M2 doctrine).
    """
    from dehyd.config import QCConfig
    from dehyd.data.loader_10ghz import load_10ghz_file
    from dehyd.qc.screens import run_qc_cube

    pre = PreprocessConfig()
    qc = QCConfig()

    cube = load_10ghz_file(real_data_paths["data_10ghz_dir"] / "subject_1_8am.mat")
    verdicts = run_qc_cube(cube, qc, pre)
    passing = [i for i, v in enumerate(verdicts) if v.passed]
    assert passing, "subject 1 8am should have QC-passing frames"

    roi = set(option_b_roi_bins(pre, N_FAST_TIME).tolist())
    peaks, shares = [], []

    for index in passing:
        frame = cube[:, :, index]
        gated = apply_band_gate(frame, pre, axis=0)
        detection = detect_option_b_peak(gated, pre)
        assert detection.peak_bin in roi
        peaks.append(detection.peak_bin)

        roi_power = detection.power[detection.roi_bins].sum()
        if roi_power > 0:
            shares.append(detection.power[detection.peak_bin] / roi_power)

        for reduction in ("a", "b"):
            for channel in ("mag", "iq"):
                out = preprocess_frame(frame, pre, reduction=reduction, channel=channel)
                assert out.shape == (1 if channel == "mag" else 2, 470)
                assert np.all(np.isfinite(out))

    # Determinism on real data (one frame is enough; the cost is real I/O).
    first_frame = cube[:, :, passing[0]]
    assert np.array_equal(
        preprocess_frame(first_frame, pre, reduction="b", channel="iq"),
        preprocess_frame(first_frame, pre, reduction="b", channel="iq"),
    )

    df = pre.fs_hz / N_FAST_TIME
    counts = {bin_index: peaks.count(bin_index) for bin_index in sorted(set(peaks))}
    with capsys.disabled():
        print(f"\n  subject 1 8am: {len(passing)}/{cube.shape[2]} frames passed QC")
        print(f"  option-B peak bins {counts} (df = {df:.1f} Hz)")
        print(f"  peak Hz median {np.median([p * df for p in peaks]):.1f}")
        if shares:
            print(f"  peak_share median {np.median(shares):.3f}")
