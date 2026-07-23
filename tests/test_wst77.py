"""The 77 GHz WST extraction chain (T-W77 + T-R77, no private data except the marked smoke).

Correctness-critical parts: the measured geometry (regression-pinned), the frozen batch
shape and rx-major/bin-minor fold order, the batch-standardize bit-equivalence to
to_channels, range-bin averaging, Rx fusion, log placement, the pre-log scale, the tuned-ε
handoff, the variants==single-variant equivalence, the canonical guard, and the numpy-vs-torch
cross-backend agreement (the precondition for backend: torch).
"""

import warnings

import numpy as np
import pytest

from dehyd.config import (
    Config,
    Preprocess77Config,
    QC77Config,
    WST77Config,
    load_config,
)
from dehyd.features.extraction_77 import (
    CanonicalSpecError77,
    _build77,
    _frame_per_rx_tensor,
    _fuse_rx,
    _prelog_scale_77,
    apply_order_log_77,
    canonical_spec_guard_77,
    extract_session_features_77,
    extract_session_variants_77,
    prf_hz,
    slow_time_signal_batch,
    wst77_spec,
)
from dehyd.features.wst import WSTError, backend_agreement, build_scattering, scatter_frames
from dehyd.features.pooling import session_feature_layout
from dehyd.preprocess.standardize import to_channels

WST = WST77Config()
PRE = Preprocess77Config()
N_GATE = 27
N_CHIRP = 256
N_RX = 16


def gated_frame(seed=0, n_gate=N_GATE, n_chirp=N_CHIRP, n_rx=N_RX):
    """A non-degenerate complex gated frame [n_gate, n_chirp, n_rx] (varying slow-time)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_gate, n_chirp, n_rx)) + 1j * rng.standard_normal((n_gate, n_chirp, n_rx))


def raw_cube(n_frames=2, seed=1):
    """A small real raw cube [N, 256, 256, 16] for the session-level paths."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_frames, N_CHIRP, N_CHIRP, N_RX))


# ---------------------------------------------------------------------- geometry


def test_geometry_regression_pinned():
    spec = wst77_spec(WST, PRE)
    assert spec["n_in"] == 256
    assert prf_hz(PRE) == pytest.approx(1953.125)
    got = [(t["t_samples"], t["J"], t["padded_len"]) for t in spec["tilings"]]
    assert got == [(39, 6, 512), (78, 7, 512), (117, 7, 512)]
    for t in spec["tilings"]:
        assert t["n_paths"] > 0 and t["n_time"] > 0
        assert t["realized_error_frac"] < 0.002  # <0.2% invariance error


def test_border_warning_is_present_not_silenced():
    """kymatio warns 'signal support is too small' at n_in=256 — asserted, never silenced."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_scattering(WST.tilings[0], WST, n_in=256, fs_hz=prf_hz(PRE))
    assert any("too small" in str(w.message).lower() or "border" in str(w.message).lower()
               for w in caught)


# ------------------------------------------------------------ batch shape + fold order


def test_batch_shape_and_per_rx_reshape():
    sc = _build77(WST.tilings[0], WST, N_CHIRP, prf_hz(PRE))
    frame = gated_frame()
    batch = slow_time_signal_batch(frame, "robust")
    assert batch.shape == (N_RX * N_GATE, 2, N_CHIRP) == (432, 2, 256)
    per_rx = _frame_per_rx_tensor(frame, sc, "robust")
    assert per_rx.shape[0] == N_RX and per_rx.shape[1] == 2  # [16, 2, P, t]


def test_fold_order_rx_major_bin_minor_and_bit_equivalent_to_to_channels():
    """batch[rx*27+bin] must equal to_channels(gated[bin,:,rx], 'iq', 'robust') bit-for-bit."""
    frame = gated_frame(seed=3)
    batch = slow_time_signal_batch(frame, "robust")
    for rx, bin_ in [(0, 0), (0, 5), (3, 26), (15, 13)]:
        expected = to_channels(frame[bin_, :, rx], "iq", "robust")  # [2, 256]
        np.testing.assert_array_equal(batch[rx * N_GATE + bin_], expected)


def test_range_bin_averaging_matches_independent_scatter():
    """per_rx[rx] is the mean over the 27 gate bins of that rx's per-bin scatters."""
    sc = _build77(WST.tilings[2], WST, N_CHIRP, prf_hz(PRE))  # smallest n_paths tiling
    frame = gated_frame(seed=4)
    per_rx = _frame_per_rx_tensor(frame, sc, "robust")
    # Independently scatter one rx's 27 bins and average.
    rx = 7
    chans = np.stack([to_channels(frame[b, :, rx], "iq", "robust") for b in range(N_GATE)])  # [27,2,256]
    S = scatter_frames(chans, sc)  # [27, 2, P, t]
    np.testing.assert_allclose(per_rx[rx], S.mean(axis=0), rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------- fusion + log


def test_fusion_mean_and_median_differ():
    sc = _build77(WST.tilings[0], WST, N_CHIRP, prf_hz(PRE))
    per_rx = _frame_per_rx_tensor(gated_frame(seed=5), sc, "robust")
    assert not np.allclose(_fuse_rx(per_rx, "mean"), _fuse_rx(per_rx, "median"))


def test_fused_then_log_differs_from_log_then_fused():
    """The frozen order is fuse THEN log (A-M5-4); log-then-fuse would change the semantics."""
    sc = _build77(WST.tilings[0], WST, N_CHIRP, prf_hz(PRE))
    meta = sc.meta()
    per_rx = _frame_per_rx_tensor(gated_frame(seed=6), sc, "robust")
    fused_then_log = apply_order_log_77(_fuse_rx(per_rx, "mean"), meta, WST, log_branch="on_frozen_eps")
    log_then_fused = np.mean(
        [apply_order_log_77(per_rx[r], meta, WST, log_branch="on_frozen_eps") for r in range(N_RX)],
        axis=0,
    )
    assert not np.allclose(fused_then_log, log_then_fused)


def test_order_zero_stays_linear_under_log():
    sc = _build77(WST.tilings[0], WST, N_CHIRP, prf_hz(PRE))
    meta = sc.meta()
    order = np.asarray(meta["order"])
    fused = _fuse_rx(_frame_per_rx_tensor(gated_frame(seed=7), sc, "robust"), "mean")
    logged = apply_order_log_77(fused, meta, WST, log_branch="on_frozen_eps")
    np.testing.assert_array_equal(logged[:, order == 0, :], fused[:, order == 0, :])
    assert not np.allclose(logged[:, order >= 1, :], fused[:, order >= 1, :])


# ------------------------------------------------------------------ zero-energy guard


def test_zero_energy_assertion_fires():
    frame = gated_frame(seed=8)
    frame[0, :, 0] = 3.0 + 0j  # a constant slow-time series -> MAD 0 -> all-zero channel
    with pytest.raises(WSTError, match="all-zero"):
        slow_time_signal_batch(frame, "robust")


def test_per_channel_standardization_is_independent():
    """real and imag are standardized from their OWN statistics (distinct scales survive)."""
    frame = np.zeros((1, N_CHIRP, 1), dtype=np.complex128)
    rng = np.random.default_rng(9)
    frame[0, :, 0] = 1.0 * rng.standard_normal(N_CHIRP) + 1j * 1000.0 * rng.standard_normal(N_CHIRP)
    batch = slow_time_signal_batch(frame, "robust")  # [1, 2, 256]
    # After per-channel robust-z both channels have unit-ish MAD despite the 1000x input scale.
    assert abs(np.median(np.abs(batch[0, 0])) - np.median(np.abs(batch[0, 1]))) < 0.5


# --------------------------------------------------------------------- pre-log scale


def test_prelog_scale_keyed_by_fusion_and_hand_checked():
    sc = _build77(WST.tilings[2], WST, N_CHIRP, prf_hz(PRE))
    meta = sc.meta()
    order = np.asarray(meta["order"])
    frames = [_frame_per_rx_tensor(gated_frame(seed=s), sc, "robust") for s in (10, 11)]
    fused_mean = np.stack([_fuse_rx(pr, "mean") for pr in frames])
    fused_median = np.stack([_fuse_rx(pr, "median") for pr in frames])
    scale_mean = _prelog_scale_77(fused_mean, meta)
    scale_median = _prelog_scale_77(fused_median, meta)
    assert scale_mean != scale_median  # keyed by fusion (C5-10)
    # Hand-check order-1 for the mean fusion.
    tm = fused_mean.mean(axis=-1)  # [N, C, P]
    hand_o1 = float(np.median(tm[:, :, order == 1].mean(axis=-1).mean(axis=-1)))
    assert scale_mean[1] == pytest.approx(hand_o1)


# ------------------------------------------------------------------- tuned-ε handoff


def test_tuned_eps_applied_before_pooling_and_role_independent():
    sc = _build77(WST.tilings[0], WST, N_CHIRP, prf_hz(PRE))
    meta = sc.meta()
    fused = _fuse_rx(_frame_per_rx_tensor(gated_frame(seed=12), sc, "robust"), "mean")
    eps = {1: 3e-4, 2: 7e-5}
    a = apply_order_log_77(fused, meta, WST, log_branch="on_tuned_eps", epsilon_by_order=eps)
    b = apply_order_log_77(fused, meta, WST, log_branch="on_tuned_eps", epsilon_by_order=eps)
    np.testing.assert_array_equal(a, b)  # identical regardless of caller (no role dependence)
    # It differs from the frozen-ε branch and hand-matches log(S+eps_o) on order 1.
    order = np.asarray(meta["order"])
    frozen = apply_order_log_77(fused, meta, WST, log_branch="on_frozen_eps")
    assert not np.allclose(a[:, order == 1, :], frozen[:, order == 1, :])
    np.testing.assert_allclose(a[:, order == 1, :], np.log(fused[:, order == 1, :] + eps[1]))


def test_tuned_eps_requires_epsilon_mapping():
    sc = _build77(WST.tilings[0], WST, N_CHIRP, prf_hz(PRE))
    fused = _fuse_rx(_frame_per_rx_tensor(gated_frame(seed=13), sc, "robust"), "mean")
    with pytest.raises(WSTError, match="epsilon_by_order"):
        apply_order_log_77(fused, sc.meta(), WST, log_branch="on_tuned_eps")


# ----------------------------------------------------- variants == single-variant reference


def test_variants_equal_single_variant_reference():
    cube = raw_cube(n_frames=2, seed=20)
    variants = extract_session_variants_77(cube, PRE, WST)
    for ti in range(len(WST.tilings)):
        for log_branch in ("off", "on_frozen_eps"):
            for fusion in ("mean", "median"):
                for family in ("pooled", "flat"):
                    ref = extract_session_features_77(
                        cube, PRE, WST, tiling=WST.tilings[ti], log_branch=log_branch,
                        fusion=fusion, family=family,
                    )
                    got = variants.vectors[(ti, log_branch, fusion, family)]
                    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)
    assert variants.all_finite
    assert set(variants.prelog_scale) == {(ti, f) for ti in range(3) for f in ("mean", "median")}


def _n_time(scattering):
    from dehyd.features.wst import scattering_shape
    return scattering_shape(scattering)["n_time"]


def test_pooled_session_dims_match_layout():
    cube = raw_cube(n_frames=2, seed=21)
    vec = extract_session_features_77(cube, PRE, WST, tiling=WST.tilings[0],
                                      log_branch="off", fusion="mean", family="pooled")
    sc = _build77(WST.tilings[0], WST, N_CHIRP, prf_hz(PRE))
    # 2 channels (real/imag); session layout is twice the per-frame length (mean+median).
    layout = session_feature_layout(sc.meta(), _n_time(sc), 2, family="pooled")
    assert vec.shape[0] == len(layout)


# ---------------------------------------------------------------- canonical spec guard


def _canonical_config(tmp_path):
    data = tmp_path / "d10"; data.mkdir()
    d77 = tmp_path / "d77"; d77.mkdir()
    xlsx = tmp_path / "w.xlsx"; xlsx.write_bytes(b"")
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({
        "paths": {"data_10ghz_dir": str(data), "weight_xlsx": str(xlsx),
                  "results_dir": str(tmp_path / "r"), "data_77ghz_dir": str(d77)},
        "run": {"seed": 1, "seed_set": [1, 2, 3, 4, 5], "device": "cpu"},
    }), encoding="utf-8")
    return p


def test_canonical_spec_guard_accepts_default(tmp_path):
    canonical_spec_guard_77(load_config(_canonical_config(tmp_path)))  # must not raise


def test_canonical_spec_guard_rejects_non_default(tmp_path):
    import yaml
    base = _canonical_config(tmp_path)
    override = tmp_path / "o.yaml"
    override.write_text(yaml.safe_dump({"wst77": {"backend": "torch"}}), encoding="utf-8")
    with pytest.raises(CanonicalSpecError77, match="wst77.backend"):
        canonical_spec_guard_77(load_config(base, override))


def test_canonical_spec_guard_rejects_stale_flatline_rule(tmp_path):
    """The step-6 flatline field is inside the qc77 canonical check — a stale value is caught."""
    import yaml
    base = _canonical_config(tmp_path)
    override = tmp_path / "o.yaml"
    override.write_text(yaml.safe_dump({"qc77": {"flatline_skip_leading_bins": 0}}), encoding="utf-8")
    with pytest.raises(CanonicalSpecError77, match="flatline_skip_leading_bins"):
        canonical_spec_guard_77(load_config(base, override))


# ------------------------------------------------------- cross-backend agreement (torch)


def test_numpy_vs_torch_cross_backend_agreement():
    """The precondition for backend: torch — numpy-f64 vs torch-f32 on raw + logged tensors."""
    torch = pytest.importorskip("torch")
    frame = gated_frame(seed=30)
    batch = slow_time_signal_batch(frame, "robust")
    sc_np = _build77(WST.tilings[0], WST, N_CHIRP, prf_hz(PRE))
    wst_torch = WST77Config(backend="torch")
    sc_t = _build77(WST.tilings[0], wst_torch, N_CHIRP, prf_hz(PRE))
    s_np = scatter_frames(batch, sc_np)
    s_t = scatter_frames(batch, sc_t)
    assert backend_agreement(s_np, s_t.astype(np.float64), policy="float64").passed
    # ...and after the order-aware log (both log states via the fused tensor).
    meta = sc_np.meta()
    for log_branch in ("off", "on_frozen_eps"):
        f_np = apply_order_log_77(_fuse_rx(s_np.reshape(N_RX, N_GATE, 2, *s_np.shape[-2:]).mean(1), "mean"),
                                  meta, WST, log_branch=log_branch)
        f_t = apply_order_log_77(_fuse_rx(s_t.astype(np.float64).reshape(N_RX, N_GATE, 2, *s_t.shape[-2:]).mean(1), "mean"),
                                 meta, WST, log_branch=log_branch)
        assert backend_agreement(f_np, f_t, policy="float64").passed


# ---------------------------------------------------------------------- realdata (T-R77)


@pytest.mark.realdata
def test_extract_real_file_frames(real_data_77_paths):
    """Non-curated smoke on one real file: load -> preprocess -> one-tiling extract -> finite."""
    from dehyd.data.loader_77ghz import load_77ghz_file

    cube = load_77ghz_file(real_data_77_paths["data_77ghz_dir"] / "subject_1_8am.mat")
    small = cube[:3]  # first 3 frames
    vec = extract_session_features_77(small, PRE, WST, tiling=WST.tilings[0],
                                      log_branch="on_frozen_eps", fusion="mean", family="pooled")
    assert np.all(np.isfinite(vec))
    assert vec.shape[0] > 0
