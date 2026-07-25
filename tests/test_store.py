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
