"""T-M7-store: the per-session feature store round-trip, the store-vs-direct tuned-ε
reconstruction equivalence, and fail-closed fingerprint validation (staleness, same-count
frame-membership change (C4), store/analysis commit mismatch (C16), missing session)."""

import numpy as np
import pytest

from dehyd.config import PreprocessConfig, WSTConfig, load_config
from dehyd.data.loader_10ghz import N_CHIRPS, N_FAST_TIME
from dehyd.features.extraction import extract_session_variants
from dehyd.features.pooling import aggregate_session, pool_stats_batch
from dehyd.features.store import (
    StoreError,
    compute_fingerprint,
    read_session_store,
    validate_store,
    write_session_store,
)
from dehyd.features.wst import apply_order_log

PRE = PreprocessConfig()
WST = WSTConfig()
FS = PRE.fs_hz


def _synthetic_cube(seed=0, n_frames=4):
    rng = np.random.default_rng(seed)
    n = np.arange(N_FAST_TIME)
    tone = np.sin(2 * np.pi * (5 * FS / N_FAST_TIME) * n / FS)[:, None, None]
    cube = tone + 0.3 * rng.standard_normal((N_FAST_TIME, N_CHIRPS, n_frames))
    cube = cube + 1j * 0.3 * rng.standard_normal((N_FAST_TIME, N_CHIRPS, n_frames))
    return cube.astype(np.complex128)


@pytest.fixture(scope="module")
def config():
    return load_config("configs/exp_a_regression.yaml")


@pytest.fixture(scope="module")
def config77():
    return load_config("configs/exp_a_regression_77ghz.yaml")


# ------------------------------------------------------ keep_raw + reconstruction


def test_keep_raw_leaves_vectors_and_prelog_bit_identical():
    cube = _synthetic_cube(seed=1)
    plain = extract_session_variants(cube, PRE, WST, reduction="a", channel="iq")
    with_raw = extract_session_variants(cube, PRE, WST, reduction="a", channel="iq", keep_raw=True)
    assert plain.raw is None and with_raw.raw is not None
    assert set(plain.vectors) == set(with_raw.vectors)
    for k in plain.vectors:
        assert plain.vectors[k].tobytes() == with_raw.vectors[k].tobytes()
    assert plain.prelog_scale == with_raw.prelog_scale


def test_store_vs_direct_frozen_epsilon_reconstruction_equivalence():
    """The frozen-ε session vector reconstructed from the stored RAW tensor equals the
    vector the extraction computed directly — byte-for-byte."""
    res = extract_session_variants(_synthetic_cube(seed=2), PRE, WST, reduction="a", channel="mag", keep_raw=True)
    for ti in range(len(WST.tilings)):
        S = res.raw[ti]["S"]
        meta = {"order": res.raw[ti]["order"]}
        logged = apply_order_log(
            S, meta, WST, log_on=True, epsilon_by_order={1: WST.log_epsilon, 2: WST.log_epsilon}
        )
        vec = aggregate_session(pool_stats_batch(logged, meta))
        assert vec.tobytes() == res.vectors[(ti, True, "pooled")].tobytes()


# ------------------------------------------------------------------ round-trip


def test_npz_round_trip_and_lazy_read(tmp_path):
    npz = {
        "vec__g0__A__mag__t0__off": np.arange(6, dtype=float),
        "raw__g0__A__mag__t0": np.ones((3, 1, 5, 8)),
        "order__t0": np.array([0, 1, 2]),
    }
    fp = {"spec_hash": "x", "n_frames": 3, "store_version": 1}
    write_session_store("10ghz", 1, "10am", npz, fp, tmp_path)
    store = read_session_store("10ghz", 1, "10am", tmp_path)
    assert "vec__g0__A__mag__t0__off" in store
    assert store["vec__g0__A__mag__t0__off"].tobytes() == npz["vec__g0__A__mag__t0__off"].tobytes()
    assert store["raw__g0__A__mag__t0"].shape == (3, 1, 5, 8)
    assert store.fingerprint["n_frames"] == 3


def test_missing_session_fails_closed(tmp_path):
    with pytest.raises(StoreError, match="missing store file"):
        read_session_store("10ghz", 9, "nope", tmp_path)


# ------------------------------------------------------------- fingerprint validation


@pytest.fixture
def stored_session(tmp_path, config):
    """Write one session with a correctly computed fingerprint; return (dir, expected)."""
    raw_file = tmp_path / "subject_1_10am.mat"
    raw_file.write_bytes(b"pretend-radar-bytes")
    frame_ids = [0, 1, 2, 3]
    fp = compute_fingerprint(config, "10ghz", frame_ids=frame_ids, raw_path=raw_file, session_eligible=True)
    write_session_store("10ghz", 1, "10am", {"order__t0": np.array([0, 1])}, fp, tmp_path)
    expected = {(1, "10am"): fp}
    return tmp_path, expected, config, raw_file, frame_ids


def test_validate_passes_on_matching_store(stored_session):
    store_dir, expected, _, _, _ = stored_session
    commit = expected[(1, "10am")]["git"]["commit"]
    validate_store("10ghz", store_dir, expected, analysis_commit=commit)  # no raise


def test_validate_rejects_spec_hash_drift(stored_session):
    store_dir, expected, _, _, _ = stored_session
    drifted = dict(expected[(1, "10am")], spec_hash="TAMPERED")
    with pytest.raises(StoreError, match="spec_hash"):
        validate_store("10ghz", store_dir, {(1, "10am"): drifted}, analysis_commit=None)


def test_validate_rejects_qc_config_drift(stored_session):
    store_dir, expected, _, _, _ = stored_session
    drifted = dict(expected[(1, "10am")], qc_config_hash="TAMPERED")
    with pytest.raises(StoreError, match="qc_config_hash"):
        validate_store("10ghz", store_dir, {(1, "10am"): drifted}, analysis_commit=None)


def test_same_count_frame_membership_change_fails_closed(stored_session):
    """C4: swap one selected frame for another (same count, same raw file) -> rejected."""
    store_dir, expected, config, raw_file, _ = stored_session
    changed = compute_fingerprint(
        config, "10ghz", frame_ids=[0, 1, 2, 4], raw_path=raw_file, session_eligible=True
    )
    assert changed["n_frames"] == expected[(1, "10am")]["n_frames"]         # same count
    assert changed["raw_sha256"] == expected[(1, "10am")]["raw_sha256"]     # same file
    assert changed["frame_ids_sha256"] != expected[(1, "10am")]["frame_ids_sha256"]
    with pytest.raises(StoreError, match="frame_ids_sha256"):
        validate_store("10ghz", store_dir, {(1, "10am"): changed}, analysis_commit=None)


def test_store_analysis_commit_mismatch_fails_closed(stored_session):
    """C16: a store built at one commit cannot back an analysis at another."""
    store_dir, expected, _, _, _ = stored_session
    with pytest.raises(StoreError, match="commit mismatch"):
        validate_store("10ghz", store_dir, expected, analysis_commit="a-different-commit-sha")


# ------------------------------------------------------------- build + clean-tree


def test_build_session_npz_10ghz_has_all_keys_and_reconstructs(config):
    from dehyd.features.store import build_session_npz_10ghz, raw_key, vec_key

    cube = _synthetic_cube(seed=5, n_frames=4)
    npz = build_session_npz_10ghz(cube, [0, 1, 2, 3], config)
    n_gates = len(config.search_10ghz.range_gate_m)
    n_tilings = len(config.wst.tilings)
    # every gate x reduction x channel x tiling has off/frozen vectors + a raw tensor.
    for gi in range(n_gates):
        for r in config.search_10ghz.reduction:
            for c in config.search_10ghz.channel:
                for ti in range(n_tilings):
                    assert vec_key(gi, r, c, ti, "off") in npz
                    assert vec_key(gi, r, c, ti, "frozen") in npz
                    assert raw_key(gi, r, c, ti) in npz
    # a stored raw tensor reconstructs its stored frozen vector.
    S = npz[raw_key(0, "A", "mag", 0)]
    order = npz["order__t0"]
    meta = {"order": order}
    logged = apply_order_log(S, meta, WST, log_on=True, epsilon_by_order={1: WST.log_epsilon, 2: WST.log_epsilon})
    vec = aggregate_session(pool_stats_batch(logged, meta))
    assert vec.tobytes() == npz[vec_key(0, "A", "mag", 0, "frozen")].tobytes()


# ==================================================== T-M9-store: schema v2 signal arrays
#
# The Exp D per-frame signal arrays (plan §2.9). Everything here is checked against the
# FROZEN INPUT DEFINITION composed by hand in the test (`implementation_plan.md:891-896`,
# `:902-904`, `:921-935`, A-M6-2 (i)/(i-ablation)), never against what the builder returned.


def test_store_version_is_two_and_travels_in_the_fingerprint(config, tmp_path):
    """The bump is what makes every v1 store fail closed — the intended behaviour."""
    from dehyd.features.store import STORE_VERSION

    assert STORE_VERSION == 2
    raw_file = tmp_path / "s.mat"
    raw_file.write_bytes(b"bytes")
    fp = compute_fingerprint(config, "10ghz", frame_ids=[0], raw_path=raw_file, session_eligible=True)
    assert fp["store_version"] == 2


def test_v1_sidecar_fails_closed_on_store_version(stored_session):
    """A store built under schema v1 (no Exp D signal arrays) must be REFUSED, not read:
    `build_frames_d` would otherwise fail deep inside a run instead of at validation."""
    store_dir, expected, _, _, _ = stored_session
    v1_sidecar = dict(expected[(1, "10am")], store_version=1)
    write_session_store("10ghz", 2, "10am", {"order__t0": np.array([0, 1])}, v1_sidecar, store_dir)
    with pytest.raises(StoreError, match="store_version"):
        validate_store("10ghz", store_dir, {(2, "10am"): expected[(1, "10am")]}, analysis_commit=None)


def test_signal_key_names_agree_with_their_two_consumers():
    """One name, three independently written tables (store / cnn.FRAME_INPUT /
    exp_d.PHYSICS_SIGNAL_KEY). A rename in one place must fail here, not silently make a
    family read a key that does not exist."""
    from dehyd.eval.exp_d import PHYSICS_SIGNAL_KEY
    from dehyd.features.store import (
        SIG_MATCHED_IQ,
        SIG_RAW_BEAT_10GHZ,
        SIG_RAW_SLOWTIME_77GHZ,
    )
    from dehyd.models.cnn import FRAME_INPUT

    assert (SIG_RAW_BEAT_10GHZ, SIG_RAW_SLOWTIME_77GHZ, SIG_MATCHED_IQ) == (
        "sig__raw_beat", "sig__raw_slowtime", "sig__matched_iq",
    )
    assert FRAME_INPUT[("10ghz", "cnn1d_raw")][0] == SIG_RAW_BEAT_10GHZ
    assert FRAME_INPUT[("10ghz", "cnn1d_matched")][0] == SIG_MATCHED_IQ
    assert FRAME_INPUT[("77ghz", "cnn1d_raw")][0] == SIG_RAW_SLOWTIME_77GHZ
    assert FRAME_INPUT[("77ghz", "cnn1d_matched")][0] == SIG_MATCHED_IQ
    assert PHYSICS_SIGNAL_KEY == {"10ghz": SIG_RAW_BEAT_10GHZ, "77ghz": SIG_RAW_SLOWTIME_77GHZ}


def test_10ghz_signal_shapes_and_dtypes(config):
    from dehyd.features.store import session_signals_10ghz

    sub = _synthetic_cube(seed=11, n_frames=3)
    sig = session_signals_10ghz(sub, config)
    assert sig["sig__raw_beat"].shape == (3, N_FAST_TIME) == (3, 534)
    assert sig["sig__raw_beat"].dtype == np.complex128
    assert sig["sig__matched_iq"].shape == (3, 2, N_FAST_TIME - 2 * PRE.edge_trim) == (3, 2, 470)
    assert sig["sig__matched_iq"].dtype == np.float64


def test_10ghz_raw_beat_is_the_ungated_untrimmed_unstandardized_chirp_mean(config):
    """The frozen "raw beat": chirp mean of the RAW frame. Three negative companions make
    the assertion falsifiable — a builder that bandpassed, trimmed, or standardized would
    still produce a plausible [N, 534]-ish array, and each of those is ruled out here."""
    from dehyd.preprocess.filters import apply_band_gate
    from dehyd.features.store import session_signals_10ghz

    sub = _synthetic_cube(seed=12, n_frames=2)
    raw_beat = session_signals_10ghz(sub, config)["sig__raw_beat"]

    for i in range(sub.shape[2]):
        expected = np.mean(sub[:, :, i], axis=1)          # the chirp mean, by hand
        assert raw_beat[i].tobytes() == expected.tobytes()
        gated = np.mean(apply_band_gate(sub[:, :, i], config.preprocess, axis=0), axis=1)
        assert raw_beat[i].tobytes() != gated.tobytes()   # NOT bandpassed
    assert raw_beat.shape[1] == 534                        # NOT edge-trimmed
    # NOT standardized: a robust z is median-centred, so its own median is ~0.
    assert abs(float(np.median(raw_beat.real))) > 1e-9


def test_10ghz_matched_iq_is_bytewise_the_wst_chains_own_preprocessed_frames(config):
    """One definition, two consumers (§2.9): the stored matched array IS what the WST chain
    preprocesses at the default model gate — so the CNN ablation and the WST features are
    provably reading the same signal, not two similar ones."""
    import dataclasses

    from dehyd.preprocess.pipeline import preprocess_cube
    from dehyd.features.store import session_signals_10ghz

    sub = _synthetic_cube(seed=13, n_frames=2)
    stored = session_signals_10ghz(sub, config)["sig__matched_iq"]

    # the gate the WST chain uses at gate index 0 == the default preprocess gate
    gate = config.search_10ghz.range_gate_m[0]
    assert tuple(gate) == tuple(config.preprocess.model_gate_m)
    pre = dataclasses.replace(config.preprocess, model_gate_m=tuple(gate))
    chain = preprocess_cube(sub, pre, reduction="a", channel="iq")
    assert stored.tobytes() == np.asarray(chain, dtype=np.float64).tobytes()

    # ...and it is genuinely NOT the raw beat's real/imag pair (rules out a copy of the
    # wrong array under this key).
    raw = session_signals_10ghz(sub, config)["sig__raw_beat"]
    assert stored.shape[-1] != raw.shape[-1]


def test_10ghz_signals_follow_the_selected_frame_order(config):
    """Row k is frame_ids[k], not frame k — the alignment `build_frames_d` and the physics
    spine both assume when they zip the store against the QC spine."""
    from dehyd.features.store import build_session_npz_10ghz

    cube = _synthetic_cube(seed=14, n_frames=5)
    frame_ids = [3, 1]
    npz = build_session_npz_10ghz(cube, frame_ids, config)
    assert npz["sig__raw_beat"].shape == (2, 534)
    for k, frame_id in enumerate(frame_ids):
        assert npz["sig__raw_beat"][k].tobytes() == np.mean(cube[:, :, frame_id], axis=1).tobytes()
    assert npz["sig__matched_iq"].shape == (2, 2, 470)


def _cube_77(seed=0, n_frames=2, n_fast=256, n_chirp=256, n_rx=16):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_frames, n_fast, n_chirp, n_rx))


def test_77ghz_signal_shapes_dtypes_and_raw_reduction(config77):
    """A-M6-2 (i): mean over fast time and Rx, keeping the chirp (slow-time) axis. The
    expected value is accumulated in a DIFFERENT order (per chirp) than the implementation's
    single `mean(axis=(1, 3))`, so this is a value check, not a restatement — hence
    `allclose` rather than a byte comparison."""
    from dehyd.features.store import session_signals_77ghz

    cube = _cube_77(seed=3, n_frames=2)
    sig = session_signals_77ghz(cube, config77)
    raw = sig["sig__raw_slowtime"]
    assert raw.shape == (2, 256) and raw.dtype == np.float64
    assert sig["sig__matched_iq"].shape == (2, 2, 256)
    assert sig["sig__matched_iq"].dtype == np.float64

    by_chirp = np.array([[cube[i, :, c, :].mean() for c in range(256)] for i in range(2)])
    assert np.allclose(raw, by_chirp, rtol=0, atol=1e-12)
    # the chirp axis really is retained: a builder that averaged it away could not vary here
    assert float(np.std(raw[0])) > 0.0


def test_77ghz_matched_equals_the_hand_composed_chain_at_rx_zero(config77):
    """Chain steps 1-5 -> Rx 0 -> mean over the 27 gate bins -> {real, imag}. The two wrong
    readings this rules out: the WRONG Rx, and a mean ACROSS Rx (the fusion A-M6-2 forbids
    pre-WST)."""
    from dehyd.preprocess.pipeline_77 import preprocess_frame_77
    from dehyd.features.store import session_signals_77ghz

    cube = _cube_77(seed=4, n_frames=2)
    stored = session_signals_77ghz(cube, config77)["sig__matched_iq"]

    for i in range(cube.shape[0]):
        chain = preprocess_frame_77(cube[i], config77.preprocess77)
        assert chain.shape[0] == 27                      # the frozen 2-4 m gate crop
        expected = chain[:, :, 0].mean(axis=0)
        assert stored[i].tobytes() == np.stack([expected.real, expected.imag]).tobytes()

        wrong_rx = chain[:, :, 1].mean(axis=0)
        assert stored[i][0].tobytes() != wrong_rx.real.tobytes()
        fused = chain.mean(axis=(0, 2))
        assert stored[i][0].tobytes() != fused.real.tobytes()


def test_77ghz_matched_is_stored_pre_standardization(config77):
    """cnn.matched_input_77 applies the robust per-channel z AT LOAD, so the store must not
    have applied it already (a doubly standardized signal is not the frozen matched input),
    and the physics path must be able to read absolute magnitudes."""
    from dehyd.models.cnn import matched_input_77
    from dehyd.features.store import session_signals_77ghz

    cube = _cube_77(seed=5, n_frames=1)
    stored = session_signals_77ghz(cube, config77)["sig__matched_iq"][0]
    assert stored.tobytes() != matched_input_77(stored).tobytes()
    # a robust z is median-centred; the stored array is not.
    assert max(abs(float(np.median(stored[0]))), abs(float(np.median(stored[1])))) > 1e-12


def test_assert_clean_tree_refuses_dirty_or_uncommitted(monkeypatch):
    import dehyd.features.store as store

    monkeypatch.setattr(store, "_git_info", lambda: {"commit": "abc", "dirty": True, "branch": "x"})
    with pytest.raises(StoreError, match="DIRTY"):
        store.assert_clean_tree()

    monkeypatch.setattr(store, "_git_info", lambda: {"commit": None, "dirty": False, "branch": "x"})
    with pytest.raises(StoreError, match="no git commit"):
        store.assert_clean_tree()

    monkeypatch.setattr(store, "_git_info", lambda: {"commit": "abc", "dirty": False, "branch": "x"})
    store.assert_clean_tree()  # clean + committed -> no raise
