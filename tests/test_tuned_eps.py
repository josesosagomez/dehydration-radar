"""Step-7 tuned-ε math: the apply_order_log ε extension, batched pooling, and the
fold-local train-only ε computation. All pure functions, no data, no fitting."""

import numpy as np
import pytest

from dehyd.config import WSTConfig
from dehyd.eval.harness import tuned_epsilons
from dehyd.features.pooling import pool_stats, pool_stats_batch
from dehyd.features.wst import apply_order_log


def _meta(n_paths):
    # A plausible order vector: one order-0 path, then some order-1 and order-2 paths.
    order = np.array([0] + [1] * (n_paths // 2) + [2] * (n_paths - 1 - n_paths // 2))
    return {"order": order}


def test_apply_order_log_epsilon_by_order_matches_frozen_at_1e6():
    """epsilon_by_order={1:1e-6, 2:1e-6} reproduces the frozen log_epsilon path byte-for-byte."""
    rng = np.random.default_rng(0)
    n_paths, n_time = 7, 8
    meta = _meta(n_paths)
    S = np.abs(rng.normal(size=(2, n_paths, n_time)))  # non-negative modulus-like
    wst = WSTConfig()  # log_epsilon = 1e-6

    frozen = apply_order_log(S, meta, wst, log_on=True)
    tuned = apply_order_log(S, meta, wst, log_on=True, epsilon_by_order={1: 1e-6, 2: 1e-6})
    assert frozen.tobytes() == tuned.tobytes()


def test_apply_order_log_none_is_unchanged_default():
    rng = np.random.default_rng(1)
    meta = _meta(5)
    S = np.abs(rng.normal(size=(2, 5, 4)))
    wst = WSTConfig()
    a = apply_order_log(S, meta, wst, log_on=True)
    b = apply_order_log(S, meta, wst, log_on=True, epsilon_by_order=None)
    assert a.tobytes() == b.tobytes()
    # log_on=False returns S unchanged regardless of epsilon_by_order.
    assert apply_order_log(S, meta, wst, log_on=False, epsilon_by_order={1: 0.1, 2: 0.2}) is not None
    assert np.array_equal(apply_order_log(S, meta, wst, log_on=False), S)


def test_apply_order_log_different_eps_changes_orders_1_and_2_only():
    rng = np.random.default_rng(2)
    meta = _meta(7)
    order = meta["order"]
    S = np.abs(rng.normal(size=(1, 7, 6))) + 0.01
    wst = WSTConfig()
    out = apply_order_log(S, meta, wst, log_on=True, epsilon_by_order={1: 0.5, 2: 0.9})
    # order 0 path stays linear (unchanged); orders 1/2 are logged.
    assert np.array_equal(out[:, order == 0, :], S[:, order == 0, :])
    assert np.allclose(out[:, order == 1, :], np.log(S[:, order == 1, :] + 0.5))
    assert np.allclose(out[:, order == 2, :], np.log(S[:, order == 2, :] + 0.9))


def test_pool_stats_batch_equals_looped_pool_stats():
    rng = np.random.default_rng(3)
    n, c, p, t = 5, 2, 7, 8
    meta = _meta(p)
    S = rng.normal(size=(n, c, p, t))
    batched = pool_stats_batch(S, meta)
    looped = np.stack([pool_stats(S[i], meta) for i in range(n)])
    assert batched.shape == looped.shape
    assert batched.tobytes() == looped.tobytes()


# ------------------------------------------------------------------ tuned_epsilons


def test_tuned_epsilons_is_train_only_and_subject_balanced():
    # subject 1: two sessions; subject 2: one session; subject 3 (not in train) huge values.
    prelog = {
        1: [(-0.1, 0.02, 0.001), (-0.2, 0.04, 0.003)],   # means: o1=0.03, o2=0.002
        2: [(-0.3, 0.10, 0.005)],                          # o1=0.10, o2=0.005
        3: [(-0.9, 99.0, 99.0)],                           # must be ignored (not a train subject)
    }
    eps = tuned_epsilons(prelog, {1, 2}, k=0.1)
    # order 1: median over subjects of [0.03, 0.10] = 0.065 -> eps = 0.0065
    assert eps[1] == pytest.approx(0.1 * np.median([0.03, 0.10]))
    # order 2: median over [0.002, 0.005] = 0.0035 -> eps = 0.00035
    assert eps[2] == pytest.approx(0.1 * np.median([0.002, 0.005]))
    # subject 3's huge values did not leak in.
    assert eps[1] < 1.0 and eps[2] < 1.0


def test_tuned_epsilons_falls_back_on_nonpositive_or_missing():
    # order-1 scales are negative/zero -> non-positive ε -> fallback; missing subject data too.
    prelog = {1: [(0.0, -0.5, 0.0)], 2: []}
    eps = tuned_epsilons(prelog, {1, 2}, k=0.1, fallback=1e-6)
    assert eps[1] == 1e-6  # negative scale -> non-positive candidate -> fallback
    assert eps[2] == 1e-6  # no finite values at all -> fallback
