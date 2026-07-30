"""T-M9-cnn: the frozen Exp D architectures and per-frame input constructions.

Every expected value here is derived from the SPECIFICATION's arithmetic (plan §2.7,
`implementation_plan.md` §D (i)/(ii) + A-M6-2 + O-M9-6), never from running the module:

  * the parameter counts are summed by hand from the layer definitions;
  * the spectrogram values are `log(|STFT| + tiny)` on a signal whose windowed DFT is
    known analytically (a constant signal through a periodic Hann window), so the two
    wrong implementations the plan names — log-*power*, and a `1e-30` floor — fail by a
    factor 2 and by ~639 nats respectively;
  * the `SpectrogramNorm` statistics are means/stds of small integer fixtures computed on
    paper, on an asymmetric 2-channel fixture that a cross-channel-shared or
    wrong-axis reduction cannot reproduce.
"""

import math

import numpy as np
import pytest
import torch

from dehyd.config import BaselineConfig
from dehyd.models import cnn
from dehyd.models.cnn import (
    Cnn1d,
    Cnn2d,
    CnnInputError,
    SpectrogramNorm,
    build_network,
    matched_input_10,
    matched_input_77,
    raw_beat_input_10,
    raw_input_77,
    spec_input_10_matched,
    spec_input_10_raw,
    spec_input_77_matched,
    spec_input_77_raw,
    spectrogram,
)
from dehyd.preprocess.standardize import robust_standardize

# Hand-derived constants (never read back from the module under test).
LN2 = math.log(2.0)
LOG_TINY = -1022.0 * LN2          # float64 tiny is exactly 2**-1022 -> log = -708.3964185322641
LOG_32 = 5.0 * LN2                # = 3.4657359027997265
LOG_16 = 4.0 * LN2                # = 2.772588722239781

N_BEAT_10 = 534                   # the stored raw chirp-mean beat length
N_MATCHED_10 = 470                # 534 - 2*32 edge trim
N_SLOWTIME_77 = 256

# STFT geometry from Hann 64 / hop 16 / nfft 128: F = nfft//2 + 1, T = (N - 64)//16 + 1.
N_FREQ = 65
T_534, T_470, T_256 = 30, 26, 13


# --------------------------------------------------------------- frozen-constant sourcing


def test_module_constants_equal_the_frozen_baseline_config():
    """The §2.7 API takes no config (`Cnn1d(in_channels)`, `spectrogram(x_1d)`), so the
    constants live here as literals — written INDEPENDENTLY of `BaselineConfig` and pinned
    against it, the same two-separately-written-values doctrine `exp_c.FRANK_HALL_MAX_ITER`
    uses. A drift in either direction fails."""
    b = BaselineConfig()
    assert cnn.CNN1D_CHANNELS == b.cnn_channels == (16, 32, 64)
    assert cnn.CNN1D_KERNEL == b.cnn_kernel == 7
    assert cnn.CNN1D_POOL == b.cnn_pool == 4
    assert cnn.CNN2D_CHANNELS == b.cnn2d_channels == (16, 32)
    assert cnn.CNN2D_KERNEL == b.cnn2d_kernel == 3
    assert cnn.CNN2D_POOL == b.cnn2d_pool == 2
    assert cnn.SPECTROGRAM_HANN == b.spectrogram_hann == 64
    assert cnn.SPECTROGRAM_HOP == b.spectrogram_hop == 16
    assert cnn.SPECTROGRAM_NFFT == b.spectrogram_nfft == 128
    cnn.assert_frozen_constants(b)      # the fail-closed run-time form


def test_assert_frozen_constants_refuses_a_drifted_config():
    import dataclasses

    drifted = dataclasses.replace(BaselineConfig(), cnn_kernel=5)
    with pytest.raises(cnn.CnnError, match="cnn_kernel"):
        cnn.assert_frozen_constants(drifted)


# ------------------------------------------------------------------- architecture pins


@pytest.mark.parametrize(
    "in_channels, expected",
    [
        # conv1 (C*16*7 + 16) + bn1 (32) + conv2 (16*32*7 + 32 = 3616) + bn2 (64)
        #   + conv3 (32*64*7 + 64 = 14400) + bn3 (128) + fc (64 + 1 = 65)
        (1, 128 + 32 + 3616 + 64 + 14400 + 128 + 65),      # = 18433
        (2, 240 + 32 + 3616 + 64 + 14400 + 128 + 65),      # = 18545
    ],
)
def test_cnn1d_parameter_count_is_the_hand_summed_architecture(in_channels, expected):
    model = Cnn1d(in_channels)
    assert sum(p.numel() for p in model.parameters()) == expected


@pytest.mark.parametrize(
    "in_channels, expected",
    [
        # conv1 (C*16*9 + 16) + bn1 (32) + conv2 (16*32*9 + 32 = 4640) + bn2 (64) + fc (33)
        (1, 160 + 32 + 4640 + 64 + 33),                    # = 4929
        (2, 304 + 32 + 4640 + 64 + 33),                    # = 5073
    ],
)
def test_cnn2d_parameter_count_is_the_hand_summed_architecture(in_channels, expected):
    model = Cnn2d(in_channels)
    assert sum(p.numel() for p in model.parameters()) == expected


@pytest.mark.parametrize(
    "in_channels, n_samples",
    [(2, N_BEAT_10), (2, N_MATCHED_10), (1, N_SLOWTIME_77), (2, N_SLOWTIME_77)],
)
def test_cnn1d_maps_every_band_family_input_to_one_scalar_per_frame(in_channels, n_samples):
    model = Cnn1d(in_channels).eval()
    out = model(torch.zeros(3, in_channels, n_samples))
    assert out.shape == (3,)


@pytest.mark.parametrize("in_channels, n_time", [(2, T_534), (2, T_470), (1, T_256), (2, T_256)])
def test_cnn2d_maps_every_spectrogram_variant_to_one_scalar_per_frame(in_channels, n_time):
    model = Cnn2d(in_channels).eval()
    out = model(torch.zeros(3, in_channels, N_FREQ, n_time))
    assert out.shape == (3,)


def test_build_network_dispatches_1d_vs_2d_by_family():
    assert isinstance(build_network("cnn1d_raw", 2), Cnn1d)
    assert isinstance(build_network("cnn1d_matched", 2), Cnn1d)
    assert isinstance(build_network("spec2d_raw", 2), Cnn2d)
    assert isinstance(build_network("spec2d_matched", 1), Cnn2d)
    with pytest.raises(cnn.CnnError):
        build_network("physics", 1)


def test_architectures_are_bit_deterministic_under_a_fixed_seed():
    """Framework-default init is a pure function of the global seed set immediately before
    construction (the `torch_fit.py` convention) — the whole per-fit seed derivation rests
    on it."""
    x = torch.randn(4, 2, N_SLOWTIME_77, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(11)
    a = Cnn1d(2).eval()
    torch.manual_seed(11)
    b = Cnn1d(2).eval()
    with torch.no_grad():
        assert a(x).numpy().tobytes() == b(x).numpy().tobytes()
    torch.manual_seed(12)
    c = Cnn1d(2).eval()
    with torch.no_grad():
        assert a(x).numpy().tobytes() != c(x).numpy().tobytes()


# ------------------------------------------------------- per-signal robust standardization


def _complex_beat(scale, shift, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=N_BEAT_10) * scale + shift) + 1j * (
        rng.normal(size=N_BEAT_10) * scale - shift
    )


def test_raw_beat_input_10_is_two_robust_channels_from_the_signals_own_statistics():
    sig = _complex_beat(3.0, 5.0, seed=1)
    out = raw_beat_input_10(sig)

    assert out.shape == (2, N_BEAT_10)
    # `to_channels(sig, "iq", "robust")` is the one definition of this step; asserting the
    # value (not just the shape) pins that the CNN input is the SAME robust z the
    # preprocessing chain uses, not a re-implementation.
    assert out[0].tobytes() == robust_standardize(sig.real).tobytes()
    assert out[1].tobytes() == robust_standardize(sig.imag).tobytes()
    assert np.median(out[0]) == pytest.approx(0.0, abs=1e-12)


def test_robust_standardization_shares_nothing_between_frames():
    """Two frames of wildly different scale: frame 0's output must not move when frame 1
    is replaced. A pooled (per-batch) statistic — the obvious wrong implementation — would
    change it."""
    frame_a = _complex_beat(1.0, 0.0, seed=2)
    frame_b = _complex_beat(1000.0, 500.0, seed=3)
    frame_b2 = _complex_beat(0.001, -7.0, seed=4)

    alone = raw_beat_input_10(frame_a)
    with_b = np.stack([raw_beat_input_10(f) for f in (frame_a, frame_b)])
    with_b2 = np.stack([raw_beat_input_10(f) for f in (frame_a, frame_b2)])

    assert with_b[0].tobytes() == alone.tobytes()
    assert with_b2[0].tobytes() == alone.tobytes()


def test_raw_input_77_is_one_robust_channel():
    rng = np.random.default_rng(5)
    sig = rng.normal(size=N_SLOWTIME_77) * 4 - 9
    out = raw_input_77(sig)
    assert out.shape == (1, N_SLOWTIME_77)
    assert out[0].tobytes() == robust_standardize(sig).tobytes()


def test_matched_input_10_returns_the_stored_array_untouched():
    """The store's `sig__matched_iq` IS `preprocess_cube(..., channel="iq")`, whose last
    step is already the robust per-channel z (`preprocess/pipeline.py:73`). Applying it a
    second time would not be the frozen matched signal, so this builder is the identity."""
    rng = np.random.default_rng(6)
    stored = rng.normal(size=(2, N_MATCHED_10)) * 3.0 + 5.0     # deliberately NOT centred
    out = matched_input_10(stored)

    assert out.tobytes() == stored.tobytes()
    # the wrong implementation (standardize again) is a different array on this fixture
    restandardized = np.stack([robust_standardize(stored[0]), robust_standardize(stored[1])])
    assert out.tobytes() != restandardized.tobytes()


def test_matched_input_77_applies_the_robust_step_at_load():
    """The 77 GHz store deliberately keeps this tensor pre-standardization (§2.9) while
    `implementation_plan.md:877-892` defines the matched 77 GHz input as robust-standardized
    per channel — so the step happens HERE, unlike the 10 GHz matched branch."""
    rng = np.random.default_rng(7)
    stored = rng.normal(size=(2, N_SLOWTIME_77)) * 2.0 + 11.0
    out = matched_input_77(stored)

    assert out.shape == (2, N_SLOWTIME_77)
    assert out.tobytes() != stored.tobytes()
    assert out[0].tobytes() == robust_standardize(stored[0]).tobytes()
    assert out[1].tobytes() == robust_standardize(stored[1]).tobytes()
    # a shared-across-channel scale would not reproduce two independently centred channels
    assert np.median(out[0]) == pytest.approx(0.0, abs=1e-12)
    assert np.median(out[1]) == pytest.approx(0.0, abs=1e-12)


def test_matched_input_77_also_accepts_the_complex_form_of_the_same_signal():
    rng = np.random.default_rng(8)
    real, imag = rng.normal(size=N_SLOWTIME_77), rng.normal(size=N_SLOWTIME_77)
    assert matched_input_77(real + 1j * imag).tobytes() == matched_input_77(
        np.stack([real, imag])
    ).tobytes()


def test_input_builders_refuse_a_wrong_shape():
    with pytest.raises(CnnInputError):
        raw_beat_input_10(np.zeros((2, N_BEAT_10)))          # not a 1-D complex signal
    with pytest.raises(CnnInputError):
        matched_input_10(np.zeros(N_MATCHED_10))             # not [2, N]
    with pytest.raises(CnnInputError):
        raw_input_77(np.zeros((1, N_SLOWTIME_77)))           # not a 1-D real signal


# ------------------------------------------------------------------- the STFT itself


def test_spectrogram_is_the_literal_log_magnitude_on_a_hand_computed_fixture():
    """Two channels: an all-zero signal and a constant-1 signal, both length 64 (exactly one
    STFT frame), so every value is analytic.

      * zero channel: |STFT| = 0 everywhere -> log(0 + tiny) = -1022*ln2 = -708.396...
        A `1e-30` floor gives -69.078 instead, ~639 nats away.
      * constant channel through the PERIODIC Hann w[n] = 0.5(1 - cos(2*pi*n/64)):
          bin 0 = sum(w) = 32                      -> log 32 = 5*ln2 = 3.4657...
          bin 2 (the 64-periodic tone) = -16       -> log 16 = 4*ln2 = 2.7726...
        A log-POWER implementation returns 2x these (log 1024, log 256).
    """
    x = np.stack([np.zeros(cnn.SPECTROGRAM_HANN), np.ones(cnn.SPECTROGRAM_HANN)])
    spec = spectrogram(x)

    assert spec.shape == (2, N_FREQ, 1)
    assert np.allclose(spec[0], LOG_TINY, atol=1e-9)
    assert spec[1, 0, 0] == pytest.approx(LOG_32, abs=1e-9)
    assert spec[1, 2, 0] == pytest.approx(LOG_16, abs=1e-9)
    # the two wrong forms, stated so the discrimination is explicit
    assert abs(spec[1, 0, 0] - 2 * LOG_32) > 1.0          # not log-power
    assert abs(spec[0, 0, 0] - math.log(1e-30)) > 100.0   # not a 1e-30 floor


def test_spectrogram_output_is_finite_and_asserted():
    with pytest.raises(CnnInputError):
        spectrogram(np.array([np.nan] * cnn.SPECTROGRAM_HANN))


@pytest.mark.parametrize(
    "signal, expected",
    [
        (np.zeros(N_BEAT_10, dtype=complex), (2, N_FREQ, T_534)),       # 10 GHz raw (O-M9-6)
        (np.zeros((2, N_MATCHED_10)), (2, N_FREQ, T_470)),              # 10 GHz matched
        (np.zeros(N_SLOWTIME_77), (1, N_FREQ, T_256)),                  # 77 GHz raw
        (np.zeros((2, N_SLOWTIME_77)), (2, N_FREQ, T_256)),             # 77 GHz matched
    ],
)
def test_spectrogram_shape_and_channel_conventions(signal, expected):
    """O-M9-6: a complex input is real/imag STFT'd separately and stacked (2 channels), the
    77 GHz raw real signal stays 1-channel. Orientation is [channel, frequency, time]."""
    assert spectrogram(signal).shape == expected


def test_spectrogram_refuses_a_signal_shorter_than_the_window():
    with pytest.raises(CnnInputError):
        spectrogram(np.zeros(cnn.SPECTROGRAM_HANN - 1))


# ------------------------------------- which signal each spectrogram variant consumes (§2.7)


def test_raw_spectrogram_branches_bypass_the_robust_step():
    """Scaling the stored array by 2 multiplies |STFT| by 2, i.e. ADDS log 2 to every bin
    above the floor. A branch that robust-standardized first would be scale-invariant
    (difference ~ 0), which is exactly the wrong implementation §2.7 rules out."""
    rng = np.random.default_rng(9)
    beat = rng.normal(size=N_BEAT_10) + 1j * rng.normal(size=N_BEAT_10)
    slow = rng.normal(size=N_SLOWTIME_77)

    for sig, builder in ((beat, spec_input_10_raw), (slow, spec_input_77_raw)):
        delta = builder(2.0 * sig) - builder(sig)
        assert np.allclose(delta, LN2, atol=1e-9)


def test_matched_spectrogram_branches_consume_the_matched_builders_output():
    """10 GHz: the store's already-standardized array, with NO second standardization.
    77 GHz: `matched_input_77`'s robust-standardized output. Both bytewise."""
    rng = np.random.default_rng(10)
    stored_10 = rng.normal(size=(2, N_MATCHED_10)) * 3.0 + 5.0
    stored_77 = rng.normal(size=(2, N_SLOWTIME_77)) * 2.0 + 11.0

    assert spec_input_10_matched(stored_10).tobytes() == spectrogram(stored_10).tobytes()
    assert spec_input_10_matched(stored_10).tobytes() == spectrogram(
        matched_input_10(stored_10)
    ).tobytes()
    assert spec_input_77_matched(stored_77).tobytes() == spectrogram(
        matched_input_77(stored_77)
    ).tobytes()
    # ... and NOT the unstandardized stored 77 GHz array
    assert spec_input_77_matched(stored_77).tobytes() != spectrogram(stored_77).tobytes()


def test_matched_77_spectrogram_is_invariant_to_scaling_the_stored_signal():
    """The robust z is scale-equivariant, so the 77 GHz matched branch — unlike the raw
    branches above — does not move when the stored array is scaled. (Not bytewise: the
    frozen `1.4826*MAD + eps` guard sits outside the scale factor, an O(1e-16) effect.)"""
    rng = np.random.default_rng(11)
    stored = rng.normal(size=(2, N_SLOWTIME_77)) * 2.0 + 11.0
    assert np.allclose(spec_input_77_matched(3.0 * stored), spec_input_77_matched(stored),
                       rtol=1e-10, atol=1e-10)


def test_frame_input_table_names_one_store_key_and_one_builder_per_band_family():
    assert set(cnn.FRAME_INPUT) == {
        (band, family) for band in ("10ghz", "77ghz") for family in cnn.CNN_FAMILIES
    }
    assert cnn.FRAME_INPUT[("10ghz", "cnn1d_raw")] == ("sig__raw_beat", raw_beat_input_10)
    assert cnn.FRAME_INPUT[("10ghz", "spec2d_raw")] == ("sig__raw_beat", spec_input_10_raw)
    assert cnn.FRAME_INPUT[("10ghz", "cnn1d_matched")] == ("sig__matched_iq", matched_input_10)
    assert cnn.FRAME_INPUT[("10ghz", "spec2d_matched")] == ("sig__matched_iq", spec_input_10_matched)
    assert cnn.FRAME_INPUT[("77ghz", "cnn1d_raw")] == ("sig__raw_slowtime", raw_input_77)
    assert cnn.FRAME_INPUT[("77ghz", "spec2d_raw")] == ("sig__raw_slowtime", spec_input_77_raw)
    assert cnn.FRAME_INPUT[("77ghz", "cnn1d_matched")] == ("sig__matched_iq", matched_input_77)
    assert cnn.FRAME_INPUT[("77ghz", "spec2d_matched")] == ("sig__matched_iq", spec_input_77_matched)


# ----------------------------------------------------------------- SpectrogramNorm


def _asymmetric_norm_fixture():
    """[2 frames, 2 channels, 3 frequencies, 2 times], built so every per-(channel,
    frequency) statistic is a different hand-computable number and the two channels have
    deliberately different per-frequency scales.

        channel 0: f0 -> {1,3,5,7}      f1 -> {10,10,10,10}   f2 -> {0,0,0,0}
        channel 1: f0 -> {100,...}      f1 -> {2,4,6,8}       f2 -> {1,1,1,3}
    """
    frame0 = np.array([[[1.0, 3.0], [10.0, 10.0], [0.0, 0.0]],
                       [[100.0, 100.0], [2.0, 4.0], [1.0, 1.0]]])
    frame1 = np.array([[[5.0, 7.0], [10.0, 10.0], [0.0, 0.0]],
                       [[100.0, 100.0], [6.0, 8.0], [1.0, 3.0]]])
    return np.stack([frame0, frame1])


def test_spectrogram_norm_statistics_are_per_channel_frequency_over_frames_and_time():
    """Hand-computed. Population std (ddof=0):
         ch0 f0 {1,3,5,7}: mean 4,   var (9+1+1+9)/4 = 5      -> sd sqrt(5)
         ch0 f1 {10x4}:    mean 10,  sd 0 -> scale 1.0
         ch0 f2 {0x4}:     mean 0,   sd 0 -> scale 1.0
         ch1 f0 {100x4}:   mean 100, sd 0 -> scale 1.0
         ch1 f1 {2,4,6,8}: mean 5,   var 5                    -> sd sqrt(5)
         ch1 f2 {1,1,1,3}: mean 1.5, var (0.25*3 + 2.25)/4 = 0.75 -> sd sqrt(0.75)

    Pooling the two channels would give ch-shared f0 mean 51.5; reducing over frames only
    would give a [C, F, T] parameter. Both fail this fixture.
    """
    norm = SpectrogramNorm.fit(_asymmetric_norm_fixture())

    assert norm.mean.shape == (2, 3)
    assert norm.scale.shape == (2, 3)
    assert norm.mean == pytest.approx(np.array([[4.0, 10.0, 0.0], [100.0, 5.0, 1.5]]))
    assert norm.scale == pytest.approx(
        np.array([[math.sqrt(5.0), 1.0, 1.0], [1.0, math.sqrt(5.0), math.sqrt(0.75)]])
    )
    assert norm.n_zero_variance_cells == 3


def test_spectrogram_norm_zero_variance_cell_uses_scale_one_not_std_plus_tiny():
    """A constant training bin under `std + tiny` would amplify a differing validation value
    by ~1e308. With `scale = 1.0` the same value stays a centred raw number, O(1)."""
    norm = SpectrogramNorm.fit(_asymmetric_norm_fixture())
    # ch0 f1 has zero training variance; a validation frame reads 13 there.
    validation = np.zeros((1, 2, 3, 2))
    validation[0, 0, 1, :] = 13.0
    out = norm.transform(validation)

    assert out[0, 0, 1, 0] == pytest.approx(3.0)          # (13 - 10) / 1.0
    assert abs(out[0, 0, 1, 0]) < 1e3                     # `std + tiny` gives ~1.3e308 here


def test_spectrogram_norm_transform_is_the_broadcast_centre_and_scale():
    norm = SpectrogramNorm.fit(_asymmetric_norm_fixture())
    x = _asymmetric_norm_fixture()
    out = norm.transform(x)

    assert out.shape == x.shape
    assert out[0, 0, 0, 0] == pytest.approx((1.0 - 4.0) / math.sqrt(5.0))
    assert out[1, 1, 2, 1] == pytest.approx((3.0 - 1.5) / math.sqrt(0.75))
    assert np.all(out[:, 0, 2, :] == 0.0)                 # constant-zero bin stays zero


def test_spectrogram_norm_params_carry_mean_scale_and_the_substituted_cell_count():
    """§2.7: the substituted `scale` and a count of substituted cells go into the fit
    record, so a fold where the fallback fired is visible in the audit."""
    params = SpectrogramNorm.fit(_asymmetric_norm_fixture()).params()
    assert set(params) == {"mean", "scale", "n_zero_variance_cells"}
    assert params["mean"].shape == (2, 3)
    assert int(params["n_zero_variance_cells"]) == 3
    for value in params.values():
        value.tobytes()                                   # every param is a numpy array


def test_spectrogram_norm_is_a_function_of_the_rows_it_was_fit_on_only():
    """The direct form of trap 12: rows outside `x_train` cannot move the statistics. (The
    fold-level version — the norm fit inside the CV loop — is T-M9-cnnpath's mutation test.)"""
    train = _asymmetric_norm_fixture()
    held_out = np.full((1, 2, 3, 2), 1e6)

    base = SpectrogramNorm.fit(train)
    with_extra_row_present_but_not_fit = SpectrogramNorm.fit(
        np.concatenate([train, held_out])[: train.shape[0]]
    )
    assert base.mean.tobytes() == with_extra_row_present_but_not_fit.mean.tobytes()
    assert base.scale.tobytes() == with_extra_row_present_but_not_fit.scale.tobytes()

    # ... and the power companion: fitting ON the extra row does move it.
    pooled = SpectrogramNorm.fit(np.concatenate([train, held_out]))
    assert pooled.mean.tobytes() != base.mean.tobytes()


def test_spectrogram_norm_refuses_a_non_four_dimensional_tensor():
    with pytest.raises(CnnInputError):
        SpectrogramNorm.fit(np.zeros((4, 2, 65)))
