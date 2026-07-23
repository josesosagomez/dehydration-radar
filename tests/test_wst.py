"""WST feature extraction — path structure, order-aware log, cross-backend, batching.

Synthetic and private-data-free (the one real-file end-to-end check is `realdata`).
Constants come from the config object, with ONE deliberate exception (T-W17): the frozen
defaults are pinned as literals there, because if implementation and expectation both
read `WSTConfig` an accidental edit to a default passes every other test. Every other
test derives its expectations from config or from explicit inputs.

Fixtures use seeded RNG noise; a noiseless tone is not a valid fixture (M2/M3 lesson —
here also a WST with near-zero higher-order paths).
"""

import warnings

import numpy as np
import pytest

from dehyd.config import PreprocessConfig, WSTConfig, WSTTiling
from dehyd.features import wst as W
from dehyd.features.wst import (
    apply_order_log,
    backend_agreement,
    build_scattering,
    octaves_j,
    scatter_channels,
    scatter_frames,
    scattering_shape,
    t_samples,
    wst_spec,
)

FS = 520834.0
N_IN = 470  # 534 - 2*32
# Measured on the pinned stack at build (HISTORY 2026-07-23); pinned as regression values.
EXPECTED_SHAPE = {  # tiling index -> (n_paths, n_time)
    0: (742, 7),
    1: (466, 3),
    2: (349, 3),
}


@pytest.fixture(scope="module")
def wst_cfg():
    return WSTConfig()


@pytest.fixture(scope="module")
def pre_cfg():
    return PreprocessConfig()


def _numpy_scattering(tiling, wst_cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_scattering(tiling, wst_cfg, n_in=N_IN, fs_hz=FS)


def _seeded_frame(seed, n_channels=1, n_in=N_IN):
    """A standardized-scale [C, n_in] fixture with seeded noise (never a bare tone)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_in))


# ------------------------------------------------------- T-W1: ms -> samples -> J


def test_ms_to_samples_and_j_from_explicit_inputs():
    # Explicit literal inputs, NOT the config, so this is independent of T-W17.
    assert t_samples(0.20, FS) == 104
    assert t_samples(0.30, FS) == 156
    assert t_samples(0.40, FS) == 208
    assert octaves_j(104) == 7
    assert octaves_j(156) == 8
    assert octaves_j(208) == 8
    for ms, t in [(0.20, 104), (0.30, 156), (0.40, 208)]:
        realized_ms = t / FS * 1e3
        assert abs(realized_ms - ms) / ms < 0.002  # <0.2% realized-invariance error


# --------------------------------------------------- T-W2: measured geometry / meta


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_measured_shape_and_meta(idx, wst_cfg):
    tiling = wst_cfg.tilings[idx]
    scattering = _numpy_scattering(tiling, wst_cfg)
    shape = scattering_shape(scattering)
    assert (shape["n_paths"], shape["n_time"]) == EXPECTED_SHAPE[idx]
    # meta carries orders 0/1/2, and order-1 xi is strictly decreasing (kymatio convention)
    order = shape["order"]
    assert set(np.unique(order).tolist()) == {0, 1, 2}
    xi1 = shape["xi"][order == 1, 0]
    assert np.all(np.diff(xi1) < 0)
    # padded length is the measured pad math, not a next-power-of-two assumption
    assert shape["padded_len"] == N_IN + shape["pad_left"] + shape["pad_right"]


def test_wst_spec_records_triple_and_shape(wst_cfg, pre_cfg):
    spec = wst_spec(wst_cfg, pre_cfg)
    assert spec["backend"] == "numpy"
    assert spec["max_order"] == 2
    assert spec["n_in"] == N_IN
    assert len(spec["tilings"]) == 3
    for idx, t in enumerate(spec["tilings"]):
        assert (t["n_paths"], t["n_time"]) == EXPECTED_SHAPE[idx]
        assert t["realized_error_frac"] < 0.002
    js = [t["J"] for t in spec["tilings"]]
    assert js == [7, 8, 8]


# ----------------------------------------- T-W3: border warning present, all tilings


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_border_warning_emitted_for_every_tiling(idx, wst_cfg):
    tiling = wst_cfg.tilings[idx]
    with pytest.warns(UserWarning, match="border effects"):
        build_scattering(tiling, wst_cfg, n_in=N_IN, fs_hz=FS)


# --------------------------------------------------- T-W4: channel contract, I/Q


def test_scatter_channels_iq_scattered_separately(wst_cfg):
    scattering = _numpy_scattering(wst_cfg.tilings[0], wst_cfg)
    frame = _seeded_frame(7, n_channels=2)
    out = scatter_channels(frame, scattering)
    assert out.shape[0] == 2 and out.dtype == np.float64
    # channel 0 is scatter(real), channel 1 is scatter(imag) — independent passes
    assert np.array_equal(out[0], scatter_channels(frame[:1], scattering)[0])
    assert np.array_equal(out[1], scatter_channels(frame[1:2], scattering)[0])


def test_mag_channel_shape(wst_cfg):
    scattering = _numpy_scattering(wst_cfg.tilings[0], wst_cfg)
    out = scatter_channels(_seeded_frame(3, n_channels=1), scattering)
    assert out.shape == (1, *EXPECTED_SHAPE[0])


# -------------------------------------------------- T-W16: batched == single-frame


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_batched_equals_stacked_single_frames(idx, wst_cfg):
    scattering = _numpy_scattering(wst_cfg.tilings[idx], wst_cfg)
    rng = np.random.default_rng(100 + idx)
    frames = rng.standard_normal((4, 2, N_IN))
    batched = scatter_frames(frames, scattering)
    stacked = np.stack([scatter_channels(frames[i], scattering) for i in range(4)])
    assert np.array_equal(batched, stacked)


# ----------------------------------------------------- T-W5: finiteness battery


@pytest.mark.parametrize("idx", [0, 1, 2])
@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("log_on", [True, False])
def test_finiteness_every_branch(idx, n_channels, log_on, wst_cfg):
    scattering = _numpy_scattering(wst_cfg.tilings[idx], wst_cfg)
    frame = _seeded_frame(50 + idx, n_channels=n_channels)
    S = scatter_channels(frame, scattering)
    logged = apply_order_log(S, scattering.meta(), wst_cfg, log_on=log_on)
    assert np.all(np.isfinite(logged))
    # all three orders are represented in the output
    assert set(np.unique(scattering.meta()["order"]).tolist()) == {0, 1, 2}


# ---------------------------------------------------- T-W6: order-aware log rule


def test_order_log_leaves_order0_linear(wst_cfg):
    scattering = _numpy_scattering(wst_cfg.tilings[0], wst_cfg)
    meta = scattering.meta()
    order = np.asarray(meta["order"])
    S = scatter_channels(_seeded_frame(9, n_channels=1), scattering)
    logged = apply_order_log(S, meta, wst_cfg, log_on=True)
    o0 = order == 0
    o12 = order >= 1
    # order 0 untouched (linear); orders 1/2 are log(S + eps)
    assert np.array_equal(logged[:, o0, :], S[:, o0, :])
    assert np.allclose(logged[:, o12, :], np.log(S[:, o12, :] + wst_cfg.log_epsilon))


def test_order_log_off_is_identity(wst_cfg):
    scattering = _numpy_scattering(wst_cfg.tilings[1], wst_cfg)
    S = scatter_channels(_seeded_frame(11, n_channels=2), scattering)
    assert np.array_equal(apply_order_log(S, scattering.meta(), wst_cfg, log_on=False), S)


def test_order_log_negative_order0_survives(wst_cfg):
    """A crafted negative order-0 coefficient must NOT be logged (it would be NaN)."""
    scattering = _numpy_scattering(wst_cfg.tilings[0], wst_cfg)
    meta = scattering.meta()
    order = np.asarray(meta["order"])
    n_paths = order.shape[0]
    S = np.ones((1, n_paths, 3), dtype=np.float64)
    S[:, order == 0, :] = -0.5  # order 0 is a signed low-pass; make it negative
    logged = apply_order_log(S, meta, wst_cfg, log_on=True)
    assert np.all(np.isfinite(logged))
    assert np.all(logged[:, order == 0, :] == -0.5)  # untouched


def test_order_log_path_count_mismatch_raises(wst_cfg):
    scattering = _numpy_scattering(wst_cfg.tilings[0], wst_cfg)
    meta = scattering.meta()
    S = np.ones((1, 5, 3), dtype=np.float64)  # 5 != n_paths
    with pytest.raises(W.WSTError, match="path-count mismatch"):
        apply_order_log(S, meta, wst_cfg, log_on=True)


# ---------------------------------------------------------- T-W8: determinism


def test_determinism_same_backend(wst_cfg):
    scattering = _numpy_scattering(wst_cfg.tilings[0], wst_cfg)
    frame = _seeded_frame(42, n_channels=2)
    a = scatter_channels(frame, scattering)
    b = scatter_channels(frame, scattering)
    assert np.array_equal(a, b)


# ---------------------------------------------- T-W7: shift stability (frozen fixture)

from dehyd.preprocess.standardize import robust_standardize  # noqa: E402

SHIFT_SAMPLES = 8
# Descriptive drift pins (seed 0, deterministic), NOT the acceptance gate.
TW7_D_PINS = {0: 0.05366, 1: 0.13732, 2: 0.15600}
BORDER_B_PINS = {0: 0.62926, 1: 0.64850, 2: 0.67862}


def _tw7_signal():
    """The frozen T-W7 fixture: an in-band, integer-cycle tone + seeded noise, robust-z."""
    n = np.arange(N_IN)
    f = 4 * FS / N_IN  # 4 whole cycles in 470 samples -> circular shift = exact translation
    base = np.sin(2 * np.pi * f * n / FS + 0.7) + 0.1 * np.random.default_rng(0).standard_normal(N_IN)
    return robust_standardize(base)


def _rel_dist(v, w):
    v = np.asarray(v, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    return float(np.linalg.norm(v - w) / max(np.linalg.norm(v), 1e-12))


def _path_time_means(signal, scattering):
    """m(.) — the per-path global time-mean vector for a single-channel signal."""
    return scatter_channels(signal[None, :], scattering)[0].mean(axis=-1)


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_shift_stability_relative_gate(idx, wst_cfg):
    x = _tw7_signal()
    x_shift = np.roll(x, SHIFT_SAMPLES)  # circular: an exact translation of the tone
    d_input = _rel_dist(x_shift, x)
    # Fixture-sanity anchor, justified analytically (2*sin(pi*f*s/fs) ~ 0.42) before any WST.
    assert d_input > 0.2
    scattering = _numpy_scattering(wst_cfg.tilings[idx], wst_cfg)
    d_scatter = _rel_dist(_path_time_means(x_shift, scattering), _path_time_means(x, scattering))
    # Gate: the averaging must at least HALVE the shift effect (border-effect-robust).
    assert d_scatter <= 0.5 * d_input
    # Descriptive drift pin (never the gate).
    assert abs(d_scatter - TW7_D_PINS[idx]) < 1e-3


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_border_metric_is_descriptive(idx, wst_cfg):
    """b characterizes how much edge content reaches the averaged coefficients. No gate."""
    x = _tw7_signal()
    x_edged = x.copy()
    x_edged[:32] = 0.0
    x_edged[-32:] = 0.0  # 32 = the EdgeTrim width, a fixed reference edge
    scattering = _numpy_scattering(wst_cfg.tilings[idx], wst_cfg)
    b = _rel_dist(_path_time_means(x_edged, scattering), _path_time_means(x, scattering))
    assert abs(b - BORDER_B_PINS[idx]) < 1e-3


# ------------------------------------------------------ T-W9: cross-backend gate


@pytest.mark.parametrize("idx", [0, 1, 2])
@pytest.mark.parametrize("log_on", [True, False])
def test_cross_backend_agreement(idx, log_on, wst_cfg):
    tiling = wst_cfg.tilings[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sn = build_scattering(tiling, wst_cfg, n_in=N_IN, fs_hz=FS)
        st = build_scattering(tiling, WSTConfig(backend="torch"), n_in=N_IN, fs_hz=FS)
    frame = _seeded_frame(200 + idx, n_channels=2)
    a_raw = scatter_channels(frame, sn)
    b_raw = scatter_channels(frame, st)
    a = apply_order_log(a_raw, sn.meta(), wst_cfg, log_on=log_on)
    b = apply_order_log(b_raw, st.meta(), wst_cfg, log_on=log_on)
    res = backend_agreement(a, b, policy="float64")
    assert res.passed and res.policy == "float64"
    # both frontends are float64 at the comparison (torch up-cast inside scatter_frames)
    assert a.dtype == np.float64 and b.dtype == np.float64


def test_backend_agreement_rejects_bad_inputs():
    a = np.ones((2, 3), dtype=np.float64)
    with pytest.raises(W.WSTError, match="different shapes"):
        backend_agreement(a, np.ones((2, 4)))
    with pytest.raises(W.WSTError, match="empty"):
        backend_agreement(np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64))
    with pytest.raises(W.WSTError, match="non-finite"):
        backend_agreement(a, np.array([[1.0, np.nan, 3.0], [1, 2, 3]]))
    with pytest.raises(W.WSTError, match="float64"):
        backend_agreement(a.astype(np.float32), a.astype(np.float32))
    with pytest.raises(W.WSTError, match="unknown agreement policy"):
        backend_agreement(a, a, policy="loose")


def test_backend_agreement_reports_components():
    a = np.array([[1.0, 2.0, 3.0]])
    b = a + 1e-9
    res = backend_agreement(a, b)
    assert res.passed
    assert res.max_elementwise_ratio < 1.0 and res.rel_l2 < 1e-4


# ----------------------------------------------------- T-W13: input contracts


def test_scatter_frames_input_contract(wst_cfg):
    scattering = _numpy_scattering(wst_cfg.tilings[0], wst_cfg)
    with pytest.raises(W.WSTError, match="N, C, n_in"):
        scatter_frames(np.zeros((2, N_IN)), scattering)  # 2-D, not 3-D
    with pytest.raises(W.WSTError, match="!= scattering input length"):
        scatter_frames(np.zeros((1, 1, N_IN + 5)), scattering)
    bad = np.zeros((1, 1, N_IN))
    bad[0, 0, 0] = np.inf
    with pytest.raises(W.WSTError, match="non-finite"):
        scatter_frames(bad, scattering)


def test_scatter_channels_rejects_wrong_ndim(wst_cfg):
    scattering = _numpy_scattering(wst_cfg.tilings[0], wst_cfg)
    with pytest.raises(W.WSTError, match=r"\[C, n_in\]"):
        scatter_channels(np.zeros(N_IN), scattering)


# ----------------------------------------------- T-W17: frozen-defaults contract


def test_frozen_defaults_are_the_expected_literals():
    """The one deliberate literal test — an accidental edit to a default fails here."""
    cfg = WSTConfig()
    assert cfg.tilings == (
        WSTTiling(q=(10, 4), invariance_ms=0.20),
        WSTTiling(q=(8, 2), invariance_ms=0.30),
        WSTTiling(q=(6, 2), invariance_ms=0.40),
    )
    assert cfg.max_order == 2
    assert cfg.log_epsilon == 1e-6
    assert cfg.backend == "numpy"
    assert PreprocessConfig().fs_hz == 520834.0


# ============================================================ pooling (features/pooling)

from dehyd.features.pooling import (  # noqa: E402
    PoolingError,
    aggregate_session,
    feature_layout,
    flatten_series,
    pool_stats,
    session_feature_layout,
)

# A tiny hand-computable fixture: 1 channel, 2 paths (order 0 and 1).
FAKE_META = {"order": np.array([0, 1])}


def _hand_pool_one_path(series, n_time):
    """global mean[, std]; first mean[, std if >=2]; second mean[, std]."""
    half = n_time // 2
    g, f, s = series, series[:half], series[half:]
    vals = [g.mean(), g.std(ddof=0), f.mean()]
    if len(f) >= 2:
        vals.append(f.std(ddof=0))
    vals += [s.mean(), s.std(ddof=0)]
    return vals


def test_pool_stats_ntime3_drops_one_sample_half_std():
    # n_time = 3: first half is [0:1] (1 sample) -> no std -> 5 stats/path.
    S = np.array([[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]])  # [C=1, paths=2, n_time=3]
    out = pool_stats(S, FAKE_META)
    expected = _hand_pool_one_path(S[0, 0], 3) + _hand_pool_one_path(S[0, 1], 3)
    assert out.shape == (10,)  # 2 paths * 5 stats
    assert np.allclose(out, expected)
    # no structural-zero column: the dropped first-half std is simply absent
    layout = feature_layout(FAKE_META, n_time=3, n_channels=1)
    assert len(layout) == len(out)
    assert ("first", "std") not in [(seg, stat) for (_, _, seg, stat) in layout]


def test_pool_stats_ntime7_keeps_all_six():
    rng = np.random.default_rng(1)
    S = rng.standard_normal((1, 2, 7))
    out = pool_stats(S, FAKE_META)
    assert out.shape == (12,)  # 2 paths * 6 stats
    expected = _hand_pool_one_path(S[0, 0], 7) + _hand_pool_one_path(S[0, 1], 7)
    assert np.allclose(out, expected)
    assert len(feature_layout(FAKE_META, n_time=7, n_channels=1)) == 12


def test_pool_stats_ddof_zero_pinned():
    S = np.array([[[1.0, 3.0, 5.0, 7.0]]])  # one path, n_time=4
    meta = {"order": np.array([1])}
    out = pool_stats(S, meta)
    # global std ddof=0 of [1,3,5,7] is 2.2360679..., ddof=1 would be 2.5819...
    layout = feature_layout(meta, n_time=4, n_channels=1)
    gstd_idx = layout.index((0, 0, "global", "std"))
    assert np.isclose(out[gstd_idx], np.std(S[0, 0], ddof=0))
    assert not np.isclose(out[gstd_idx], np.std(S[0, 0], ddof=1))


def test_pool_stats_permuted_order_fails():
    """A reference that pools statistic-before-segment must NOT match the contract."""
    S = np.random.default_rng(2).standard_normal((1, 2, 7))
    out = pool_stats(S, FAKE_META)
    # wrong order: all means then all stds within a path
    wrong = []
    for p in range(2):
        series = S[0, p]
        half = 7 // 2
        segs = [series, series[:half], series[half:]]
        wrong += [seg.mean() for seg in segs] + [seg.std(ddof=0) for seg in segs]
    assert not np.allclose(out, wrong)


def test_pool_stats_raises_on_bad_input():
    with pytest.raises(PoolingError, match="n_time must be >= 2"):
        pool_stats(np.ones((1, 2, 1)), FAKE_META)
    with pytest.raises(PoolingError, match="path-count mismatch"):
        pool_stats(np.ones((1, 3, 4)), FAKE_META)
    with pytest.raises(PoolingError, match=r"\[C, n_paths, n_time\]"):
        pool_stats(np.ones((2, 4)), FAKE_META)


def test_flatten_series_layout():
    S = np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3)
    flat = flatten_series(S)
    assert flat.shape == (12,)
    assert np.array_equal(flat, S.reshape(-1))  # channel -> path -> time


def test_flatten_series_through_aggregate_session():
    """The raw family is wired end to end, not just laid out."""
    S = np.random.default_rng(4).standard_normal((1, 2, 5))
    per_frame = np.stack([flatten_series(S), flatten_series(S * 2)])
    session = aggregate_session(per_frame)
    assert session.shape == (2 * per_frame.shape[1],)
    assert np.all(np.isfinite(session))


def test_aggregate_session_concat_mean_median():
    fv = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    out = aggregate_session(fv)
    assert np.allclose(out, [3.0, 30.0, 3.0, 30.0])  # mean block, then median block


def test_aggregate_session_single_frame_allowed():
    v = np.array([[2.0, 4.0, 6.0]])
    assert np.allclose(aggregate_session(v), [2.0, 4.0, 6.0, 2.0, 4.0, 6.0])


def test_aggregate_session_rejects_bad_input():
    with pytest.raises(PoolingError, match="0 frames"):
        aggregate_session(np.empty((0, 3)))
    with pytest.raises(PoolingError, match=r"\[n_frames, D\]"):
        aggregate_session(np.ones(5))
    with pytest.raises(PoolingError, match="non-finite"):
        aggregate_session(np.array([[1.0, np.nan]]))


@pytest.mark.parametrize("n_time", [3, 7])
def test_session_feature_layout_matches_session_vector(n_time):
    S = np.random.default_rng(n_time).standard_normal((1, 2, n_time))
    per_frame = np.stack([pool_stats(S, FAKE_META), pool_stats(S * 1.5, FAKE_META)])
    session = aggregate_session(per_frame)
    layout = session_feature_layout(FAKE_META, n_time=n_time, n_channels=1, family="pooled")
    assert len(layout) == session.shape[0]
    # first block is frame_mean, second is frame_median, each = the per-frame layout
    d = per_frame.shape[1]
    assert all(el[0] == "frame_mean" for el in layout[:d])
    assert all(el[0] == "frame_median" for el in layout[d:])
    assert layout[0][1:] == feature_layout(FAKE_META, n_time, 1)[0]


def test_session_feature_layout_flat_family():
    layout = session_feature_layout(FAKE_META, n_time=3, n_channels=1, family="flat")
    # 2 aggregates * (1 channel * 2 paths * 3 time) = 12
    assert len(layout) == 12
    assert layout[0] == ("frame_mean", 0, 0, 0)


# ============================================================ extraction (T-W14, T-W18)

import types  # noqa: E402
from dataclasses import replace  # noqa: E402

from dehyd.data.loader_10ghz import N_CHIRPS, N_FAST_TIME  # noqa: E402
from dehyd.features import extraction as EX  # noqa: E402
from dehyd.features.extraction import (  # noqa: E402
    CanonicalSpecError,
    canonical_spec_guard,
    extract_session_features,
    extract_session_variants,
)

VARIANT_KEYS = [
    (ti, log_on, family)
    for ti in range(3)
    for log_on in (False, True)
    for family in ("pooled", "flat")
]


def _synthetic_cube(seed=0, n_frames=4):
    """A small complex128 [534, 20, n_frames] cube (real signal band + seeded noise)."""
    rng = np.random.default_rng(seed)
    n = np.arange(N_FAST_TIME)
    f = 5 * FS / N_FAST_TIME
    tone = np.sin(2 * np.pi * f * n / FS)[:, None, None]
    cube = tone + 0.3 * rng.standard_normal((N_FAST_TIME, N_CHIRPS, n_frames))
    cube = cube + 1j * 0.3 * rng.standard_normal((N_FAST_TIME, N_CHIRPS, n_frames))
    return cube.astype(np.complex128)


def test_variants_equal_single_variant_and_reuse(monkeypatch, pre_cfg, wst_cfg):
    cube = _synthetic_cube(seed=1)
    result = extract_session_variants(cube, pre_cfg, wst_cfg, reduction="a", channel="iq")

    # 1) every variant vector bit-identical to the single-variant reference
    for ti, log_on, family in VARIANT_KEYS:
        ref = extract_session_features(
            cube, pre_cfg, wst_cfg,
            reduction="a", channel="iq",
            tiling=wst_cfg.tilings[ti], log_on=log_on, family=family,
        )
        assert np.array_equal(result.vectors[(ti, log_on, family)], ref)

    # 2) shapes and finiteness recorded
    assert result.shapes == {0: (742, 7), 1: (466, 3), 2: (349, 3)}
    assert result.all_finite
    assert set(result.prelog_scale) == {0, 1, 2}


def test_variants_scatter_once_per_tiling(monkeypatch, pre_cfg, wst_cfg):
    calls = {"preprocess": 0, "scatter": 0}
    real_pre = EX.preprocess_cube
    real_scatter = EX.scatter_frames
    monkeypatch.setattr(EX, "preprocess_cube", lambda *a, **k: calls.__setitem__("preprocess", calls["preprocess"] + 1) or real_pre(*a, **k))
    monkeypatch.setattr(EX, "scatter_frames", lambda *a, **k: calls.__setitem__("scatter", calls["scatter"] + 1) or real_scatter(*a, **k))
    extract_session_variants(_synthetic_cube(seed=2), pre_cfg, wst_cfg, reduction="a", channel="mag")
    assert calls["preprocess"] == 1  # one preprocessing pass per (reduction, channel)
    assert calls["scatter"] == 3  # one scattering per tiling, not per log/family combo


def test_prelog_scale_matches_manual(pre_cfg, wst_cfg):
    cube = _synthetic_cube(seed=3)
    frames = EX.preprocess_cube(cube, pre_cfg, reduction="a", channel="mag")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = build_scattering(wst_cfg.tilings[0], wst_cfg, n_in=N_IN, fs_hz=FS)
    meta = sc.meta()
    order = np.asarray(meta["order"])
    S = scatter_frames(frames, sc)  # raw, pre-log
    tmean = S.mean(axis=-1)
    manual = tuple(
        float(np.median(tmean[:, :, order == o].mean(axis=-1).mean(axis=-1))) for o in (0, 1, 2)
    )
    result = extract_session_variants(cube, pre_cfg, wst_cfg, reduction="a", channel="mag")
    assert np.allclose(result.prelog_scale[0], manual)


import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import run_wst  # noqa: E402


def test_run_wst_dimension_summary(wst_cfg, pre_cfg):
    """T-W15: nominal / effective / raw dimensions equal their §2.6 definitions."""
    spec = wst_spec(wst_cfg, pre_cfg)
    dims = {(d["tiling"], d["channel"]): d for d in run_wst.dimension_summary(wst_cfg, spec)}
    # T1 (n_time=7): all 6 stats kept -> effective == nominal
    t1 = dims[(0, "mag")]
    assert t1["pooled_nominal"] == 742 * 6
    assert t1["pooled_effective"] == 742 * 6  # no 1-sample half
    assert t1["raw_frame_dim"] == 742 * 7
    # T2 (n_time=3): 1-sample first half drops one std -> 5 stats/path
    t2 = dims[(1, "iq")]
    assert t2["pooled_nominal"] == 2 * 466 * 6
    assert t2["pooled_effective"] == 2 * 466 * 5
    assert t2["pooled_session_dim"] == 2 * t2["pooled_effective"]
    assert t2["raw_frame_dim"] == 2 * 466 * 3


def _fake_config(preprocess, wst):
    return types.SimpleNamespace(preprocess=preprocess, wst=wst)


def test_canonical_spec_guard_accepts_canonical():
    canonical_spec_guard(_fake_config(PreprocessConfig(), WSTConfig()))  # no raise


def test_canonical_spec_guard_rejects_deviations():
    # a preprocess ablation
    with pytest.raises(CanonicalSpecError, match="preprocess.standardize"):
        canonical_spec_guard(_fake_config(replace(PreprocessConfig(), standardize="meanstd"), WSTConfig()))
    # the 0.9-3.0 m inner-CV candidate gate
    with pytest.raises(CanonicalSpecError, match="preprocess.model_gate_m"):
        canonical_spec_guard(_fake_config(replace(PreprocessConfig(), model_gate_m=(0.9, 3.0)), WSTConfig()))
    # non-default WST constants
    with pytest.raises(CanonicalSpecError, match="wst.max_order"):
        canonical_spec_guard(_fake_config(PreprocessConfig(), replace(WSTConfig(), max_order=1)))
    with pytest.raises(CanonicalSpecError, match="wst.log_epsilon"):
        canonical_spec_guard(_fake_config(PreprocessConfig(), replace(WSTConfig(), log_epsilon=1e-3)))
    # torch backend — numpy is the canonical artifact backend
    with pytest.raises(CanonicalSpecError, match="wst.backend"):
        canonical_spec_guard(_fake_config(PreprocessConfig(), replace(WSTConfig(), backend="torch")))


# -------------------------------------------------------------------------- realdata


@pytest.mark.realdata
def test_wst_on_one_real_session(real_data_paths, capsys, wst_cfg, pre_cfg):
    """Preprocess -> WST -> both families -> session aggregate on one real file.

    Structural only. Distributions are unknown until this runs; asserting them would be
    back-door tuning (M2/M3 doctrine).
    """
    from dehyd.data.loader_10ghz import load_10ghz_file
    from dehyd.qc.screens import run_qc_cube
    from dehyd.config import QCConfig
    from dehyd.preprocess.pipeline import preprocess_cube

    cube = load_10ghz_file(real_data_paths["data_10ghz_dir"] / "subject_1_8am.mat")
    verdicts = run_qc_cube(cube, QCConfig(), pre_cfg)
    passing = [i for i, v in enumerate(verdicts) if v.passed]
    assert passing, "subject 1 8am should have QC-passing frames"
    qc_cube = cube[:, :, passing]

    frames = preprocess_cube(qc_cube, pre_cfg, reduction="a", channel="iq")  # [N, 2, 470]
    spec = wst_spec(wst_cfg, pre_cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx, tiling in enumerate(wst_cfg.tilings):
            sn = build_scattering(tiling, wst_cfg, n_in=N_IN, fs_hz=FS)
            st = build_scattering(tiling, WSTConfig(backend="torch"), n_in=N_IN, fs_hz=FS)
            S = scatter_frames(frames, sn)
            assert S.shape[-2:] == (spec["tilings"][idx]["n_paths"], spec["tilings"][idx]["n_time"])

            for log_on in (True, False):
                logged = np.stack(
                    [apply_order_log(S[i], sn.meta(), wst_cfg, log_on=log_on) for i in range(S.shape[0])]
                )
                pooled = np.stack([pool_stats(logged[i], sn.meta()) for i in range(S.shape[0])])
                flat = np.stack([flatten_series(logged[i]) for i in range(S.shape[0])])
                pooled_session = aggregate_session(pooled)
                flat_session = aggregate_session(flat)
                assert np.all(np.isfinite(pooled_session))
                assert np.all(np.isfinite(flat_session))
                assert pooled_session.shape[0] == len(
                    session_feature_layout(sn.meta(), spec["tilings"][idx]["n_time"], 2, family="pooled")
                )

            # cross-backend on the real frames, identical helper as T-W9
            Sn = scatter_frames(frames[:4], sn)
            St = scatter_frames(frames[:4], st)
            res = backend_agreement(Sn, St, policy="float64")
            with capsys.disabled():
                print(
                    f"\n  T{idx+1}: shape {tuple(S.shape[-2:])}, cross-backend "
                    f"max_ratio {res.max_elementwise_ratio:.4f} relL2 {res.rel_l2:.2e} "
                    f"{'PASS' if res.passed else 'FAIL'}"
                )
            assert res.passed

    # determinism on real data
    sn = build_scattering(wst_cfg.tilings[0], wst_cfg, n_in=N_IN, fs_hz=FS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.array_equal(scatter_frames(frames[:3], sn), scatter_frames(frames[:3], sn))
