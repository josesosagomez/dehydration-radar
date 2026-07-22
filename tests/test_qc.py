"""The frozen 10 GHz QC screens.

Every threshold comes from the config object, never a literal in the test — a test
that re-hardcodes 0.25/0.30/4.5/200 would pass vacuously once code and config drift
apart. Frames are synthetic and built at the real (534, 20) shape, which is small
enough to be cheap.

Reference band arithmetic (verified against the implementation, not assumed):
HzPerM ≈ 3257.5 Hz/m, so the 0.9–3.0 m QC gate is 2931.7–9772.4 Hz; widened by the
±1000 Hz margin and binned at df ≈ 975.3 Hz it is mask bins 2..11.
"""

import dataclasses
import math

import numpy as np
import pytest

from dehyd.config import REPO_ROOT, PreprocessConfig, QCConfig, beat_band_hz
from dehyd.data.loader_10ghz import N_CHIRPS, N_FAST_TIME
from dehyd.qc.screens import QCError, FrameQC, in_band_mask, run_qc_cube, run_qc_frame

QC = QCConfig()
PRE = PreprocessConfig()

# Tones chosen against the band arithmetic above (see module docstring).
F_IN_GATE = 3257.5  # ~1.0 m — a seated subject, inside both QC and model gates
F_2_5_M = 8143.7  # ~2.5 m — inside the QC gate, OUTSIDE the 1–2 m model gate
F_IN_MARGIN = 10300.0  # above the 9772 Hz gate top, below the 10772 Hz margin top
F_BEYOND_MARGIN = 12000.0  # one bin past the mask edge
F_FAR_OUT = 50000.0


def pure_tone_frame(freq_hz: float) -> np.ndarray:
    """A noiseless complex beat tone, identical across chirps.

    Note this frame is legitimately FLATLINE: |exp(i.)| is constant, so its magnitude
    histogram is maximally concentrated. Real acquisitions always carry noise, so
    `tone_frame` below is the realistic fixture and this one is used only where the
    degenerate-magnitude behaviour is the thing under test.
    """
    t = np.arange(N_FAST_TIME) / PRE.fs_hz
    chirp = np.exp(2j * np.pi * freq_hz * t)
    return np.repeat(chirp[:, None], N_CHIRPS, axis=1)


def tone_frame(freq_hz: float, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """A realistic beat tone: dominant in-band line plus a little receiver noise.

    Seeded, so every test using it is reproducible.
    """
    rng = np.random.default_rng(seed)
    frame = pure_tone_frame(freq_hz)
    return frame + noise * (
        rng.standard_normal(frame.shape) + 1j * rng.standard_normal(frame.shape)
    )


def flatline_chirp(n_identical: int) -> np.ndarray:
    """A chirp with exactly `n_identical` samples at one magnitude.

    The remainder spreads over a wide range so no *other* histogram bin can also
    accumulate enough samples to flag — this isolates the boundary being tested.
    """
    values = np.zeros(N_FAST_TIME)
    values[n_identical:] = np.linspace(1.0, 2.0, N_FAST_TIME - n_identical)
    return values.astype(np.complex128)


# ----------------------------------------------------------------- T-QC1 clean frame


def test_qc1_clean_frame_passes_every_screen():
    result = run_qc_frame(tone_frame(F_IN_GATE), QC, PRE)
    assert result.passed
    assert not (result.nan_inf or result.flatline or result.low_in_band or result.rms_flag)
    assert result.in_band_ratio > QC.min_in_band_energy_ratio


# ------------------------------------------------- T-QC2 NaN/Inf + non-finite contract


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, complex(0, np.nan)])
def test_qc2_isolated_non_finite_sample_rejects(bad):
    frame = tone_frame(F_IN_GATE)
    frame[17, 3] = bad
    result = run_qc_frame(frame, QC, PRE)
    assert result.nan_inf and not result.passed


def test_qc2_wholly_non_finite_frame_rejects():
    frame = np.full((N_FAST_TIME, N_CHIRPS), np.nan, dtype=np.complex128)
    assert run_qc_frame(frame, QC, PRE).nan_inf


def test_qc2_non_finite_frame_short_circuits_later_screens():
    """The frozen contract: skipped booleans/counts report False/0, floats report NaN.

    np.histogram raises on non-finite input, so the later screens genuinely cannot run
    — the contract makes that explicit instead of letting it crash.
    """
    frame = tone_frame(F_IN_GATE)
    frame[0, 0] = np.nan
    result = run_qc_frame(frame, QC, PRE)
    assert result.nan_inf is True
    assert result.flatline is False
    assert result.low_in_band is False
    assert result.rms_flag is False
    assert result.n_flatline_chirps == 0
    assert result.n_rms_outlier_chirps == 0
    assert np.isnan(result.in_band_ratio)
    assert np.isnan(result.max_rms_z)
    assert not result.passed


# -------------------------------------------------------------------- T-QC3 flatline


def test_qc3_constant_chirp_flags_flatline():
    """Any single flatlined chirp rejects the frame (the reference's any-chirp rule)."""
    frame = tone_frame(F_IN_GATE)
    frame[:, 7] = 1.0 + 0j
    result = run_qc_frame(frame, QC, PRE)
    assert result.flatline and result.n_flatline_chirps == 1
    assert not result.passed


def test_qc3_degenerate_magnitude_spread_is_flatline_not_a_crash():
    """A noiseless CW tone has constant |x| — a spread of ~1e-16, not exactly zero.

    np.histogram cannot build 200 distinct float edges across that and raises
    ("Too many bins for data range"), so the screen decides the degenerate case
    itself. The verdict is *flatline*, which is correct: constant magnitude is what
    the screen exists to catch, and MATLAB's histcounts reaches the same answer by
    choosing its own bin width.
    """
    result = run_qc_frame(pure_tone_frame(F_IN_GATE), QC, PRE)
    assert result.flatline and result.n_flatline_chirps == N_CHIRPS
    assert not result.passed


def test_qc3_flatline_boundary_is_greater_or_equal():
    """0.25 x 534 = 133.5, so a bin of 134 fires and 133 does not."""
    threshold = QC.flatline_max_bin_fraction * N_FAST_TIME
    assert threshold == 133.5  # guards the arithmetic this boundary test relies on

    fires = tone_frame(F_IN_GATE)
    fires[:, 0] = flatline_chirp(134)
    assert run_qc_frame(fires, QC, PRE).n_flatline_chirps == 1

    survives = tone_frame(F_IN_GATE)
    survives[:, 0] = flatline_chirp(133)
    assert run_qc_frame(survives, QC, PRE).n_flatline_chirps == 0


# ------------------------------------------------------------ T-QC4 in-band energy


def test_qc4_far_out_of_band_tone_rejected_in_gate_tone_kept():
    out = run_qc_frame(tone_frame(F_FAR_OUT), QC, PRE)
    assert out.low_in_band and not out.passed
    assert 0.0 <= out.in_band_ratio <= 1.0

    inside = run_qc_frame(tone_frame(F_IN_GATE), QC, PRE)
    assert not inside.low_in_band
    assert 0.0 <= inside.in_band_ratio <= 1.0


# --------------------------------------------------------- T-QC5 RMS is diagnostic


def test_qc5_rms_outlier_flags_but_never_rejects():
    """The reference logs RMS outliers; they are not a rejection criterion here."""
    frame = tone_frame(F_IN_GATE)
    frame[:, 11] *= 100.0
    result = run_qc_frame(frame, QC, PRE)
    assert result.rms_flag and result.n_rms_outlier_chirps == 1
    assert result.max_rms_z > QC.rms_robust_z_threshold
    assert not (result.nan_inf or result.flatline or result.low_in_band)
    assert result.passed  # the whole point


# --------------------------------------------------- T-QC6 thresholds come from config


def test_qc6_flatline_threshold_is_read_from_config():
    frame = tone_frame(F_IN_GATE)
    frame[:, 0] = flatline_chirp(133)
    assert run_qc_frame(frame, QC, PRE).n_flatline_chirps == 0
    looser = dataclasses.replace(QC, flatline_max_bin_fraction=133 / N_FAST_TIME)
    assert run_qc_frame(frame, looser, PRE).n_flatline_chirps == 1


def test_qc6_in_band_threshold_is_read_from_config():
    frame = tone_frame(F_IN_MARGIN)  # ratio ~0.97
    assert not run_qc_frame(frame, QC, PRE).low_in_band
    stricter = dataclasses.replace(QC, min_in_band_energy_ratio=0.99)
    assert run_qc_frame(frame, stricter, PRE).low_in_band


def test_qc6_rms_threshold_is_read_from_config():
    """No perturbation is needed: chirps differ only by noise, so MAD is tiny.

    That makes the robust z very sensitive — an unperturbed frame already sits at
    z ~2.6, below the frozen 4.5 but well above a lowered threshold.
    """
    frame = tone_frame(F_IN_GATE)
    assert not run_qc_frame(frame, QC, PRE).rms_flag
    stricter = dataclasses.replace(QC, rms_robust_z_threshold=0.5)
    assert run_qc_frame(frame, stricter, PRE).rms_flag


# ------------------------------------------------------- T-QC7 data independence


def test_qc7_frame_verdict_does_not_depend_on_companion_frames():
    """THE guard: no cross-frame statistic may enter a per-frame verdict.

    If any screen ever normalised against the cube (a dataset-wide histogram range, a
    session RMS median), this fails — and QC would have become data-dependent, hence
    split-dependent.
    """
    rng = np.random.default_rng(7)
    target = tone_frame(F_IN_GATE, seed=int(rng.integers(1 << 30)))
    alone = run_qc_frame(target, QC, PRE)

    companions = [
        tone_frame(F_FAR_OUT) * 1e6,  # wildly different scale and band
        np.zeros((N_FAST_TIME, N_CHIRPS), dtype=np.complex128),
        tone_frame(F_2_5_M, noise=5.0, seed=3),
    ]
    for position in (0, 2, 3):
        frames = list(companions)
        frames.insert(position, target)
        cube = np.stack(frames, axis=2)
        assert run_qc_cube(cube, QC, PRE)[position] == alone


# ---------------------------------------------------------------- T-QC8 determinism


def test_qc8_repeated_runs_are_identical():
    frame = tone_frame(F_IN_GATE, seed=8)
    assert run_qc_frame(frame, QC, PRE) == run_qc_frame(frame, QC, PRE)


# ------------------------------------------------------- T-QC9 the fixed wider gate


def test_qc9_qc_uses_its_own_wider_gate_not_the_model_gate():
    """A 2.5 m target is inside the QC gate but outside the 1–2 m model gate.

    It must pass QC: the QC-passing population has to be identical for every model
    gate later searched in inner CV, or a hyperparameter choice would change which
    frames exist.
    """
    model_band = beat_band_hz(PRE.model_gate_m, PRE.bandwidth_hz, PRE.chirp_time_s)
    qc_band = beat_band_hz(QC.qc_gate_m, PRE.bandwidth_hz, PRE.chirp_time_s)
    assert model_band[1] < F_2_5_M < qc_band[1]  # the tone really is between the gates

    assert not run_qc_frame(tone_frame(F_2_5_M), QC, PRE).low_in_band

    # ...and the mask genuinely comes from qc_gate_m: building it from the model gate
    # would exclude this tone's bins.
    qc_mask = in_band_mask(
        N_FAST_TIME, PRE.fs_hz, PRE.bandwidth_hz, PRE.chirp_time_s,
        QC.qc_gate_m, QC.in_band_margin_hz,
    )
    model_mask = in_band_mask(
        N_FAST_TIME, PRE.fs_hz, PRE.bandwidth_hz, PRE.chirp_time_s,
        PRE.model_gate_m, QC.in_band_margin_hz,
    )
    tone_bin = round(F_2_5_M / (PRE.fs_hz / N_FAST_TIME))
    assert qc_mask[tone_bin] and not model_mask[tone_bin]


# ------------------------------------------------------------------- T-QC10 margin


def test_qc10_margin_is_used_and_lands_on_the_right_side():
    """An in-gate vs 50 kHz pair cannot show the margin exists; these tones can.

    F_IN_MARGIN sits above the gate top but inside the margin, F_BEYOND_MARGIN one
    bin past the mask edge.
    """
    gate_top = beat_band_hz(QC.qc_gate_m, PRE.bandwidth_hz, PRE.chirp_time_s)[1]
    assert gate_top < F_IN_MARGIN < gate_top + QC.in_band_margin_hz

    assert not run_qc_frame(tone_frame(F_IN_MARGIN), QC, PRE).low_in_band
    assert run_qc_frame(tone_frame(F_BEYOND_MARGIN), QC, PRE).low_in_band

    # Removing the margin drops this tone's captured energy from ~0.97 to ~0.45 (the
    # mask loses its top 3 bins) — proof the margin bins are genuinely counted. It
    # does NOT flip the verdict, because Hann leakage still leaves enough inside the
    # bare gate; asserting a flip here would be asserting something untrue.
    zero_margin = dataclasses.replace(QC, in_band_margin_hz=0.0)
    with_margin = run_qc_frame(tone_frame(F_IN_MARGIN), QC, PRE).in_band_ratio
    without = run_qc_frame(tone_frame(F_IN_MARGIN), zero_margin, PRE).in_band_ratio
    assert without < 0.6 * with_margin

    kwargs = (N_FAST_TIME, PRE.fs_hz, PRE.bandwidth_hz, PRE.chirp_time_s, QC.qc_gate_m)
    assert in_band_mask(*kwargs, QC.in_band_margin_hz).sum() == 10
    assert in_band_mask(*kwargs, 0.0).sum() == 7


# ---------------------------------------------------------------- T-QC11 mask bins


def test_qc11_mask_bin_membership_is_exact():
    mask = in_band_mask(
        N_FAST_TIME, PRE.fs_hz, PRE.bandwidth_hz, PRE.chirp_time_s,
        QC.qc_gate_m, QC.in_band_margin_hz,
    )
    # Non-negative half-spectrum only: DC in, Nyquist bin (267) out.
    assert mask.shape == (N_FAST_TIME // 2,)
    assert mask.shape[0] == 267

    df = PRE.fs_hz / N_FAST_TIME
    f_lo, f_hi = beat_band_hz(QC.qc_gate_m, PRE.bandwidth_hz, PRE.chirp_time_s)
    lo, hi = f_lo - QC.in_band_margin_hz, f_hi + QC.in_band_margin_hz
    expected = [k for k in range(N_FAST_TIME // 2) if lo <= k * df <= hi]
    assert np.flatnonzero(mask).tolist() == expected
    assert expected[0] == 2 and expected[-1] == 11  # pins the arithmetic itself


# ------------------------------------------- T-QC12 window, zero power, reason overlap


def test_qc12_window_is_periodic_hann():
    """`hann(N,'periodic')` in the reference — sym=False, not scipy's default."""
    from scipy.signal.windows import hann

    from dehyd.qc.screens import _in_band_ratio

    mask = in_band_mask(
        N_FAST_TIME, PRE.fs_hz, PRE.bandwidth_hz, PRE.chirp_time_s,
        QC.qc_gate_m, QC.in_band_margin_hz,
    )
    frame = tone_frame(F_IN_GATE)
    window = hann(N_FAST_TIME, sym=False)
    spectra = np.fft.fft(frame * window[:, None], axis=0)
    power = (np.abs(spectra[: N_FAST_TIME // 2, :]) ** 2).mean(axis=1)
    expected = power[mask].sum() / power.sum()
    assert _in_band_ratio(frame, mask) == pytest.approx(expected)

    symmetric = hann(N_FAST_TIME, sym=True)
    assert not np.allclose(window, symmetric)  # the choice is observable


def test_qc12_all_zero_frame_is_finite_and_fires_two_screens():
    """Guarded denominator: ratio 0, never a division error. Reasons legitimately overlap."""
    frame = np.zeros((N_FAST_TIME, N_CHIRPS), dtype=np.complex128)
    result = run_qc_frame(frame, QC, PRE)
    assert result.in_band_ratio == 0.0
    assert result.low_in_band and result.flatline
    assert not result.nan_inf  # zeros are finite
    assert not result.passed


# ----------------------------------------------------------- T-QC13 shape validation


@pytest.mark.parametrize(
    "shape",
    [(N_FAST_TIME, N_CHIRPS - 1), (N_FAST_TIME + 1, N_CHIRPS), (N_CHIRPS, N_FAST_TIME),
     (N_FAST_TIME,), (N_FAST_TIME, N_CHIRPS, 2)],
)
def test_qc13_run_qc_frame_rejects_wrong_shapes(shape):
    with pytest.raises(QCError, match="expected"):
        run_qc_frame(np.zeros(shape, dtype=np.complex128), QC, PRE)


@pytest.mark.parametrize(
    "shape",
    [(N_FAST_TIME, N_CHIRPS), (N_CHIRPS, N_FAST_TIME, 3), (N_FAST_TIME, N_CHIRPS + 1, 3)],
)
def test_qc13_run_qc_cube_rejects_wrong_shapes(shape):
    with pytest.raises(QCError, match="expected"):
        run_qc_cube(np.zeros(shape, dtype=np.complex128), QC, PRE)


# ------------------------------------------------------------- T-QC14 rejection rule


def test_qc14_rejection_rule_holds_over_a_seeded_battery():
    """passed == not (nan_inf or flatline or low_in_band) — rms_flag never enters it.

    Seeded so the battery is reproducible.
    """
    rng = np.random.default_rng(20260721)
    frames = []
    for _ in range(6):
        frames.append(tone_frame(F_IN_GATE, seed=int(rng.integers(1 << 30))))
        frames.append(tone_frame(F_FAR_OUT, seed=int(rng.integers(1 << 30))))
        spiked = tone_frame(F_IN_GATE, seed=int(rng.integers(1 << 30)))
        spiked[:, rng.integers(N_CHIRPS)] *= 200.0
        frames.append(spiked)
        flat = tone_frame(F_IN_GATE, seed=int(rng.integers(1 << 30)))
        flat[:, rng.integers(N_CHIRPS)] = 1.0 + 0j
        frames.append(flat)
        nan_frame = tone_frame(F_IN_GATE, seed=int(rng.integers(1 << 30)))
        nan_frame[rng.integers(N_FAST_TIME), rng.integers(N_CHIRPS)] = np.nan
        frames.append(nan_frame)
        frames.append(np.zeros((N_FAST_TIME, N_CHIRPS), dtype=np.complex128))

    results = [run_qc_frame(f, QC, PRE) for f in frames]
    for r in results:
        assert r.passed == (not (r.nan_inf or r.flatline or r.low_in_band))

    # The battery must actually exercise the interesting case, or the identity above
    # is vacuous: frames flagged for RMS that still pass.
    assert any(r.rms_flag and r.passed for r in results)
    assert any(not r.passed for r in results)


def test_qc14_rms_flag_alone_never_rejects():
    for flag in (True, False):
        result = FrameQC(
            nan_inf=False, flatline=False, low_in_band=False, rms_flag=flag,
            in_band_ratio=0.9, n_flatline_chirps=0, n_rms_outlier_chirps=int(flag),
            max_rms_z=9.9 if flag else 0.1,
        )
        assert result.passed


# ------------------------------------------------------------ T-QC15 mask bin guards


def test_qc15_mask_with_no_bin_support_raises():
    """A band above Nyquist selects nothing — nothing to measure."""
    with pytest.raises(QCError, match="selects 0"):
        in_band_mask(
            N_FAST_TIME, PRE.fs_hz, PRE.bandwidth_hz, PRE.chirp_time_s,
            (100.0, 200.0), QC.in_band_margin_hz,
        )


def test_qc15_mask_covering_every_bin_raises():
    """A screen that can never fire is an error, not a configuration."""
    with pytest.raises(QCError, match="selects all"):
        in_band_mask(
            N_FAST_TIME, PRE.fs_hz, PRE.bandwidth_hz, PRE.chirp_time_s,
            QC.qc_gate_m, 300_000.0,
        )


# ---------------------------------------------------------------------- realdata


@pytest.mark.realdata
def test_full_cohort_qc_survival(real_data_paths, capsys):
    """Run the frozen screens over all 80 real files and report survival.

    Asserts STRUCTURAL properties only. There is deliberately no expected-survival-rate
    assertion: the rates are unknown until this runs, and pinning them would be
    threshold-tuning by the back door.
    """
    from dehyd.config import load_config
    from dehyd.data.ground_truth import load_ground_truth
    from dehyd.data.manifest import apply_qc, build_manifest, session_qc_report

    config = load_config(REPO_ROOT / "configs" / "exp_a_regression.yaml")
    gt = load_ground_truth(config.paths.weight_xlsx)
    manifest_qc = apply_qc(build_manifest(config.paths, gt), config.paths, config)
    report = session_qc_report(manifest_qc)

    assert len(report) == 80  # every subject x session cell present
    assert len(manifest_qc) == 8000

    finite = manifest_qc["qc_in_band_ratio"].dropna()
    assert ((finite >= 0.0) & (finite <= 1.0)).all()

    reconciles = report["n_pass"] + report["n_fail_any"] == report["n_frames"]
    assert reconciles.all()

    expected_min = [math.ceil(config.qc.min_frame_fraction * n) for n in report["n_frames"]]
    assert report["min_pass"].tolist() == expected_min

    with capsys.disabled():
        print(f"\n  frames passing QC : {int(manifest_qc['qc_pass'].sum())} / 8000")
        print(f"  sessions eligible : {int(report['eligible'].sum())} / 80")
        print(report.to_string(index=False))


@pytest.mark.realdata
def test_qc_is_deterministic_on_real_data(real_data_paths):
    """Same file twice -> identical verdicts. Bounded cost: one file, not eighty."""
    from dehyd.data.loader_10ghz import load_10ghz_file

    cube = load_10ghz_file(real_data_paths["mat_files"][0])
    assert run_qc_cube(cube, QC, PRE) == run_qc_cube(cube, QC, PRE)

