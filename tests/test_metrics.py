"""T-M7-metrics: Exp A scoring + the subject-cluster BCa bootstrap.

The bootstrap tests use a small B for speed; determinism (same rng_seed → identical CI)
is asserted separately, and the statistical shape (interval brackets the point, fallback
fires on degenerate input, skips are counted) is checked structurally rather than against
hand-computed BCa endpoints.
"""

import math

import numpy as np
import pytest

from dehyd.eval.metrics import (
    BootstrapCI,
    mean_difference_ci,
    per_subject_pearson_r,
    pooled_pearson_r,
    session_rmse,
    subject_balanced_mae,
    subject_cluster_bootstrap,
    subject_cluster_bootstrap_pooled,
    wilcoxon_signed_rank,
)


# ---------------------------------------------------------------- point metrics


def test_subject_balanced_mae_pinned_to_5_5():
    """The frozen T17 fixture: subject-balanced (5.5), never pooled (25/7)."""
    subjects = np.array([1, 1, 1, 1, 1, 2, 2])
    y_true = np.zeros(7)
    y_pred = np.array([1, 1, 1, 1, 1, 10, 10], dtype=float)
    assert subject_balanced_mae(subjects, y_true, y_pred) == 5.5
    assert subject_balanced_mae(subjects, y_true, y_pred) != 25 / 7


def test_subject_balanced_mae_equals_pooled_on_equal_counts():
    subjects = np.array([1, 1, 2, 2])
    y_true = np.zeros(4)
    y_pred = np.array([2.0, 4.0, 6.0, 8.0])
    # subject 1 mean |err| = 3, subject 2 = 7 -> 5.0; pooled mean = 5.0 too here.
    assert subject_balanced_mae(subjects, y_true, y_pred) == 5.0


def test_session_rmse_hand_value():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([3.0, 0.0, 4.0])
    assert session_rmse(y_true, y_pred) == pytest.approx(math.sqrt((9 + 0 + 16) / 3))


def test_pooled_pearson_r_perfect_and_degenerate():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert pooled_pearson_r(x, 2 * x + 1) == pytest.approx(1.0)
    assert math.isnan(pooled_pearson_r(x, np.ones_like(x)))  # zero variance in y_pred
    assert math.isnan(pooled_pearson_r(np.ones_like(x), x))  # zero variance in y_true


def test_per_subject_pearson_r_min_sessions_rule():
    subjects = np.array([1, 1, 1, 2, 2])  # subj 1 has 3, subj 2 has 2
    y_true = np.array([1.0, 2.0, 3.0, 1.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0, 6.0])
    r = per_subject_pearson_r(subjects, y_true, y_pred, min_sessions=3)
    assert set(r) == {1}  # subject 2 excluded (only 2 sessions)
    assert r[1] == pytest.approx(1.0)


# ---------------------------------------------------------- additive bootstrap


def test_bootstrap_is_deterministic_under_seed():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a = subject_cluster_bootstrap(vals, b=500, rng_seed=20260721)
    c = subject_cluster_bootstrap(vals, b=500, rng_seed=20260721)
    assert a == c
    d = subject_cluster_bootstrap(vals, b=500, rng_seed=1)
    assert (a.low, a.high) != (d.low, d.high)  # different seed -> different draws


def test_bootstrap_point_is_the_mean_and_interval_brackets_it():
    vals = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    ci = subject_cluster_bootstrap(vals, b=2000, rng_seed=7)
    assert ci.point == pytest.approx(np.mean(vals))
    assert ci.low <= ci.point <= ci.high
    assert ci.n_eval == len(vals)
    assert ci.method == "bca"
    assert ci.n_skipped == 0 and ci.unreliable is False


def test_bca_falls_back_to_percentile_on_degenerate_input():
    """All-equal subject values → every replicate equals the point → z0 undefined → fallback."""
    ci = subject_cluster_bootstrap([5.0] * 6, b=300, rng_seed=3)
    assert ci.method == "percentile"
    assert ci.low == pytest.approx(5.0) and ci.high == pytest.approx(5.0)


def test_mean_difference_ci_delegates_to_additive_bootstrap():
    diffs = [-1.0, -2.0, 0.5, -3.0, -0.5]
    ci = mean_difference_ci(diffs, b=500, rng_seed=11)
    assert isinstance(ci, BootstrapCI)
    assert ci.point == pytest.approx(np.mean(diffs))


# ------------------------------------------------------------ pooled bootstrap


def _pooled_data():
    # 4 subjects, 2 sessions each; two seeds with different predictions.
    subjects = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y_pred_by_seed = np.stack([y_true + 0.1, y_true - 0.2])  # (2 seeds, 8 sessions)
    return subjects, y_true, y_pred_by_seed


def test_pooled_seed_collapse_point_is_seed_averaged_metric():
    subjects, y_true, y_pred_by_seed = _pooled_data()
    ci = subject_cluster_bootstrap_pooled(
        subjects, y_true, y_pred_by_seed, session_rmse, b=300, rng_seed=5
    )
    expected = np.mean([session_rmse(y_true, y_pred_by_seed[k]) for k in range(2)])
    assert ci.point == pytest.approx(expected)
    assert ci.low <= ci.point <= ci.high
    assert ci.n_eval == 4


def test_pooled_bootstrap_skips_and_flags_unreliable_on_constant_truth():
    """Constant y_true → pooled r undefined for every resample → all skipped, unreliable."""
    subjects = np.array([1, 1, 2, 2, 3, 3])
    y_true = np.zeros(6)
    y_pred_by_seed = np.stack([np.arange(6.0), np.arange(6.0) + 1])
    ci = subject_cluster_bootstrap_pooled(
        subjects, y_true, y_pred_by_seed, pooled_pearson_r, b=200, rng_seed=9
    )
    assert ci.n_skipped == 200
    assert ci.unreliable is True


def test_pooled_bootstrap_no_skips_when_metric_always_defined():
    subjects, y_true, y_pred_by_seed = _pooled_data()
    ci = subject_cluster_bootstrap_pooled(
        subjects, y_true, y_pred_by_seed, session_rmse, b=200, rng_seed=2
    )
    assert ci.n_skipped == 0 and ci.unreliable is False


# ------------------------------------------------------------------- Wilcoxon


def test_wilcoxon_matches_scipy_and_handles_all_zero():
    from scipy import stats

    diffs = np.array([-1.0, -2.0, 0.5, -3.0, -0.5, -1.5])
    stat, p = wilcoxon_signed_rank(diffs)
    ref = stats.wilcoxon(diffs)
    assert stat == pytest.approx(ref.statistic)
    assert p == pytest.approx(ref.pvalue)

    s0, p0 = wilcoxon_signed_rank(np.zeros(5))
    assert math.isnan(s0) and math.isnan(p0)
