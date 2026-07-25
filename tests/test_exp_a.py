"""T-M7 Exp A driver: the end-to-end synthetic-store outer-mutation property (C1/D5).

Drives the REAL two-stage `run_exp_a` (StoreBackedFeatures + Stage-1/Stage-2 selection +
fold-local tuned-ε reconstruction) over a small fabricated store — no private data — and
proves that mutating the outer-test subject's stored vec/raw/prelog/target leaves every
pre-scoring quantity (winners, tuned ε, fitted state, training predictions) bit-identical;
only the held-out prediction may move.
"""

import numpy as np
import pytest

from dehyd.config import load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval.exp_a import StoreBackedFeatures, run_exp_a
from dehyd.features.pooling import aggregate_session, pool_stats_batch
from dehyd.features.store import (
    order_key,
    prelog_key,
    raw_key,
    read_session_store,
    vec_key,
    write_session_store,
)
from dehyd.features.wst import apply_order_log

P, T, CN, NFR = 6, 4, 1, 3  # tiny path/time/channel/frame dims for a fast synthetic store
ORDER = np.array([0, 1, 1, 2, 2, 2])  # length P, all orders present


@pytest.fixture(scope="module")
def config():
    return load_config("configs/exp_a_regression.yaml")


def _make_sessions(n_subjects=4, sessions=3):
    out = []
    for s in range(1, n_subjects + 1):
        for i in range(sessions):
            out.append({
                "subject": s,
                "session_idx": i,
                "session_name": SESSION_NAMES[i],
                "delta_m_pct": float(-0.3 * i - 0.05 * s),  # a clock-correlated target
            })
    return out


def _write_store(store_dir, sessions, config, seed=0):
    rng = np.random.default_rng(seed)
    wst = config.wst
    eps = {1: wst.log_epsilon, 2: wst.log_epsilon}
    meta = {"order": ORDER}
    for s in sessions:
        npz = {}
        for ti in range(len(wst.tilings)):
            npz[order_key(ti)] = ORDER
            for gi in range(len(config.search_10ghz.range_gate_m)):
                for r in config.search_10ghz.reduction:
                    for c in config.search_10ghz.channel:
                        raw = np.abs(rng.normal(size=(NFR, CN, P, T))) + 0.01
                        off = aggregate_session(pool_stats_batch(apply_order_log(raw, meta, wst, log_on=False), meta))
                        fr = aggregate_session(pool_stats_batch(apply_order_log(raw, meta, wst, log_on=True, epsilon_by_order=eps), meta))
                        npz[vec_key(gi, r, c, ti, "off")] = off
                        npz[vec_key(gi, r, c, ti, "frozen")] = fr
                        npz[raw_key(gi, r, c, ti)] = raw
                        npz[prelog_key(gi, r, c, ti)] = np.array([raw.mean(), raw.mean(), raw.mean()])
        write_session_store("10ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR}, store_dir)


def _mutate_test_subject(store_dir, sessions, subject, seed=99):
    """Eligibility-preserving: overwrite the held-out subject's stored arrays + targets."""
    rng = np.random.default_rng(seed)
    for s in sessions:
        if s["subject"] != subject:
            continue
        store = read_session_store("10ghz", s["subject"], s["session_name"], store_dir)
        npz = {k: store[k].copy() for k in store.keys()}
        store.close()
        for k in list(npz):
            if k.startswith("raw__"):
                # raw orders 1/2 are modulus-based (non-negative) — keep them so, or the
                # tuned-ε log(raw+ε) produces NaN. Eligibility-preserving value change.
                npz[k] = np.abs(rng.normal(size=npz[k].shape) * 5) + 0.01
            elif k.startswith(("vec__", "prelog__")):
                npz[k] = (rng.normal(size=npz[k].shape) * 5 + 5).astype(npz[k].dtype)
        write_session_store("10ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR}, store_dir)
        s["delta_m_pct"] = float(rng.normal() * 5 + 5)  # mutate the label too


def _run(store_dir, sessions, config):
    provider = StoreBackedFeatures("10ghz", sessions, store_dir, config)
    session_index = np.array([s["session_idx"] for s in sessions])
    return run_exp_a(config, "10ghz", provider, seeds=(0,), session_index=session_index)


def _fold(results, subject):
    return next(r for r in results if r.test_subject == subject)


def test_exp_a_runs_the_staged_search_end_to_end(tmp_path, config):
    sessions = _make_sessions()
    _write_store(tmp_path, sessions, config)
    results = _run(tmp_path, sessions, config)
    assert len(results) == 4  # one per selectable outer fold (4 subjects)
    for r in results:
        assert r.selected_family in ("ridge", "svr", "rf", "gbm", "knn")
        assert len(r.selected_feature_key) == 5  # 10 GHz feature key
        assert r.baseline_predictions.shape == r.test_predictions.shape


def test_headline_path_outer_mutation_property(tmp_path, config):
    """C1/D5: mutating the outer-test subject's store leaves everything pre-scoring identical."""
    held_out = 2
    sessions_a = _make_sessions()
    _write_store(tmp_path / "base", sessions_a, config)
    base = _fold(_run(tmp_path / "base", sessions_a, config), held_out)

    sessions_b = _make_sessions()
    _write_store(tmp_path / "mut", sessions_b, config)
    _mutate_test_subject(tmp_path / "mut", sessions_b, held_out)
    mut = _fold(_run(tmp_path / "mut", sessions_b, config), held_out)

    # selection + fitted state determined entirely by training subjects -> bit-identical.
    assert base.selected_feature_key == mut.selected_feature_key
    assert base.selected_family == mut.selected_family
    assert base.selected_params == mut.selected_params
    for fb, fm in zip(base.final_fits, mut.final_fits, strict=True):
        assert fb.quantity == fm.quantity
        for k in fb.params:
            assert fb.params[k].tobytes() == fm.params[k].tobytes()
    # only the held-out subject's prediction may move (its features were mutated).
    assert base.test_predictions.tobytes() != mut.test_predictions.tobytes()
