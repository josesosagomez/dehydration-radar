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
    adjacent_accuracy,
    class_unit_mae,
    confusion_counts,
    equal_session_residual_mae,
    holm_adjusted,
    mean_difference_ci,
    per_session_residual_mae,
    per_subject_pearson_r,
    pooled_pearson_r,
    quadratic_weighted_kappa,
    session_rmse,
    session_weighted_bootstrap,
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


# ------------------------------------------------------- Exp B objective (T-M8-objective)


def _unequal_eligibility_fixture():
    """4 subjects with UNEQUAL eligible-session counts (subject 1: sessions 1-4; subject 2:
    sessions 1,2,4; subjects 3/4: sessions 1,4 only) -- the exact condition under which
    implementation_plan.md:1208-1217 says the session- and subject-weighted estimands diverge.
    y_true == 0 everywhere; y_pred set so each session has a constant, distinct residual
    (session s -> residual s), making every estimand hand-computable."""
    rows = [
        (1, 1, 1.0), (2, 1, 1.0), (3, 1, 1.0), (4, 1, 1.0),   # session 1: 4 subjects, err 1
        (1, 2, 2.0), (2, 2, 2.0),                              # session 2: 2 subjects, err 2
        (1, 3, 3.0),                                           # session 3: 1 subject,  err 3
        (1, 4, 4.0), (2, 4, 4.0), (3, 4, 4.0), (4, 4, 4.0),   # session 4: 4 subjects, err 4
    ]
    subjects = np.array([r[0] for r in rows])
    session_idx = np.array([r[1] for r in rows])
    y_pred = np.array([r[2] for r in rows])
    y_true = np.zeros(len(rows))
    return subjects, session_idx, y_true, y_pred


def test_per_session_residual_mae_hand_value_and_presence_only():
    subjects, session_idx, y_true, y_pred = _unequal_eligibility_fixture()
    per_session = per_session_residual_mae(session_idx, y_true, y_pred)
    assert per_session == {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    # averages only over sessions present in the call -- omit session 3 entirely.
    mask = session_idx != 3
    partial = per_session_residual_mae(session_idx[mask], y_true[mask], y_pred[mask])
    assert set(partial) == {1, 2, 4}


def test_equal_session_residual_mae_diverges_from_subject_and_pooled():
    subjects, session_idx, y_true, y_pred = _unequal_eligibility_fixture()
    session_weighted = equal_session_residual_mae(subjects, y_true, y_pred, session_idx)
    assert session_weighted == pytest.approx(2.5)               # mean(1, 2, 3, 4)
    assert session_weighted != pytest.approx(subject_balanced_mae(subjects, y_true, y_pred))
    subj_bal = subject_balanced_mae(subjects, y_true, y_pred)
    assert subj_bal == pytest.approx((2.5 + 7 / 3 + 2.5 + 2.5) / 4)
    naive_pooled = float(np.abs(y_true - y_pred).mean())
    assert naive_pooled == pytest.approx(27 / 11)
    assert session_weighted != pytest.approx(naive_pooled)


def test_equal_session_residual_mae_nan_on_empty():
    assert math.isnan(equal_session_residual_mae([], [], [], []))


# ------------------------------------------------------------------- Holm (T-M8-holm)


def test_holm_adjusted_hand_computed_family_of_4():
    p = [0.01, 0.04, 0.03, 0.02]
    adjusted = holm_adjusted(p, family_size=4)
    assert adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_adjusted_step_down_monotonicity_and_input_order():
    p = [0.03, 0.01, 0.04, 0.02]  # not pre-sorted
    adjusted = holm_adjusted(p, family_size=4)
    assert len(adjusted) == 4
    # walking the p-values in ascending order, adjusted values are non-decreasing.
    order = sorted(range(4), key=lambda i: p[i])
    seq = [adjusted[i] for i in order]
    assert seq == sorted(seq)


def test_holm_adjusted_clips_at_one():
    p = [0.5, 0.6, 0.9, 0.99]
    adjusted = holm_adjusted(p, family_size=4)
    assert all(v == pytest.approx(1.0) for v in adjusted)


def test_holm_adjusted_nan_occupies_a_slot_but_not_ranked():
    p = [0.01, float("nan"), 0.03]
    adjusted = holm_adjusted(p, family_size=4)
    assert math.isnan(adjusted[1])
    assert adjusted[0] == pytest.approx(0.04)   # 4 * 0.01
    assert adjusted[2] == pytest.approx(0.09)   # max(0.04, 3 * 0.03)


def test_holm_adjusted_family_size_4_strictly_stronger_than_len_3():
    p = [0.02, 0.03, 0.04]
    default_family = holm_adjusted(p)                    # family_size = len(p) = 3
    pinned_family = holm_adjusted(p, family_size=4)
    assert all(pf >= df for pf, df in zip(pinned_family, default_family, strict=True))
    assert any(pf > df for pf, df in zip(pinned_family, default_family, strict=True))


# --------------------------------------------------- session-weighted bootstrap (T-M8-bootstrap)


def test_session_weighted_bootstrap_deterministic_under_seed():
    subjects, session_idx, y_true, y_pred = _unequal_eligibility_fixture()
    a = session_weighted_bootstrap(subjects, session_idx, y_true, y_pred[None, :], b=300, rng_seed=2026)
    b = session_weighted_bootstrap(subjects, session_idx, y_true, y_pred[None, :], b=300, rng_seed=2026)
    assert a == b


def test_session_weighted_bootstrap_point_provably_differs_from_subject_weighted():
    subjects, session_idx, y_true, y_pred = _unequal_eligibility_fixture()
    ci = session_weighted_bootstrap(subjects, session_idx, y_true, y_pred[None, :], b=300, rng_seed=11)
    assert ci.point == pytest.approx(2.5)
    assert ci.point != pytest.approx(subject_balanced_mae(subjects, y_true, y_pred))


def test_session_weighted_bootstrap_reference_gives_difference_form():
    subjects, session_idx, y_true, y_pred = _unequal_eligibility_fixture()
    baseline = np.zeros_like(y_pred)  # session-mean baseline == 0 on the residual scale
    ci = session_weighted_bootstrap(
        subjects, session_idx, y_true, y_pred[None, :], y_pred_reference=baseline, b=300, rng_seed=5,
    )
    radar_agg = equal_session_residual_mae(subjects, y_true, y_pred, session_idx)
    baseline_agg = equal_session_residual_mae(subjects, y_true, baseline, session_idx)
    assert ci.point == pytest.approx(radar_agg - baseline_agg)


def test_session_weighted_bootstrap_empty_session_replicate_skipped_and_can_trip_unreliable():
    """A-M8-2: subject 3 is the ONLY subject holding session 3; with 3 total subjects, a
    bootstrap resample that never draws subject 3 loses session 3 entirely -- that replicate
    must be skipped-and-counted, not silently averaged over the surviving 3 sessions."""
    subjects = np.array([1, 1, 1, 2, 2, 2, 3])
    session_idx = np.array([1, 2, 4, 1, 2, 4, 3])
    y_true = np.zeros(7)
    y_pred = np.array([1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 3.0])
    ci = session_weighted_bootstrap(subjects, session_idx, y_true, y_pred[None, :], b=300, rng_seed=7)
    assert ci.n_skipped > 0
    assert ci.unreliable is True


# ---------------------------------------------------------- Exp C ordinal metrics (T-M9-metrics)


def test_class_unit_mae_hand_value():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 2, 2, 1, 4])
    # |diffs| = [0, 1, 0, 2, 0] -> mean 0.6
    assert class_unit_mae(y_true, y_pred) == pytest.approx(0.6)


def test_class_unit_mae_nan_on_empty():
    assert math.isnan(class_unit_mae([], []))


def test_class_unit_mae_is_pooled_not_subject_balanced():
    """The frozen T17 fixture in class-space: pooled (25/7), NOT subject-balanced (5.5) --
    guards against accidentally routing Exp C's inner objective through the subject-balanced
    reduction, which is Exp A's convention, not Exp C's (`:766-769` says plain pooled mean)."""
    subjects = np.array([1, 1, 1, 1, 1, 2, 2])
    y_true = np.zeros(7)
    y_pred = np.array([1, 1, 1, 1, 1, 10, 10], dtype=float)
    pooled = class_unit_mae(y_true, y_pred)
    assert pooled == pytest.approx(25 / 7)
    assert pooled != pytest.approx(subject_balanced_mae(subjects, y_true, y_pred))


def test_adjacent_accuracy_hand_value():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 2, 2, 1, 4])
    # |diffs| = [0, 1, 0, 2, 0]; <=1 -> [T, T, T, F, T] -> 4/5
    assert adjacent_accuracy(y_true, y_pred) == pytest.approx(0.8)


def test_adjacent_accuracy_nan_on_empty():
    assert math.isnan(adjacent_accuracy([], []))


def test_confusion_counts_orientation_and_sum():
    """Asymmetric fixture: counts[true, pred] must not equal its transpose here, so a
    row/column swap in the implementation is caught."""
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0])
    counts = confusion_counts(y_true, y_pred, n_classes=5)
    assert counts.shape == (5, 5)
    assert counts.sum() == 4
    assert counts[0, 1] == 2   # true=0, pred=1 (twice)
    assert counts[0, 0] == 1   # true=0, pred=0
    assert counts[1, 0] == 1   # true=1, pred=0
    assert counts[0, 1] != counts[1, 0]  # orientation: rows are true, not pred


def test_qwk_hand_computed_against_standard_formula():
    """n_classes=3 fixture, worked by hand from the definition (quadratic weights
    w_ij=(i-j)^2/(K-1)^2, E_ij = row_marginal_i * col_marginal_j / n):

    confusion (rows=true): [[2,0,0],[1,0,1],[0,1,1]], n=6, r=[2,2,2], c=[3,1,2].
    expected_disagreement = sum(w*E) = 27/12 = 2.25
    observed_disagreement = sum(w*O) = 3/4 = 0.75
    kappa = 1 - 0.75/2.25 = 2/3.
    Independently cross-checked against
    sklearn.metrics.cohen_kappa_score(weights='quadratic', labels=[0,1,2]) == 0.6666...
    """
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 2])
    kappa = quadratic_weighted_kappa(y_true, y_pred, n_classes=3)
    assert kappa == pytest.approx(2 / 3)


def test_qwk_defined_for_single_class_truth_vs_varying_predictor():
    """O-M9-8 (8a): single-class truth against a varying predictor is DEFINED (kappa=0),
    not NaN -- the frozen text's motivating parenthetical ('QWK undefined on a single-class
    validation set') is not the mathematically correct trigger on the fixed 5-class grid.
    Cross-checked against cohen_kappa_score(weights='quadratic', labels=[0,1,2,3,4])."""
    from sklearn.metrics import cohen_kappa_score

    y_true = np.array([0, 0])
    y_pred = np.array([0, 1])
    kappa = quadratic_weighted_kappa(y_true, y_pred, n_classes=5)
    ref = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2, 3, 4])
    assert not math.isnan(kappa)
    assert kappa == pytest.approx(ref)
    assert kappa == pytest.approx(0.0)


def test_qwk_defined_for_constant_predictor_vs_multiclass_truth():
    """O-M9-8 (8a): a constant predictor against multi-class truth is likewise DEFINED
    (kappa=0), matching cohen_kappa_score, not the single-class pre-check reading (8b)."""
    from sklearn.metrics import cohen_kappa_score

    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 0, 0, 0, 0])
    kappa = quadratic_weighted_kappa(y_true, y_pred, n_classes=5)
    ref = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2, 3, 4])
    assert not math.isnan(kappa)
    assert kappa == pytest.approx(ref)
    assert kappa == pytest.approx(0.0)


def test_qwk_nan_only_on_empty_and_both_sides_constant_and_equal():
    """The zero-expected-disagreement case: both true and predicted concentrate on the same
    single class -> NaN. This is the ONLY non-empty case that is undefined under (8a)."""
    assert math.isnan(quadratic_weighted_kappa([], []))
    assert math.isnan(quadratic_weighted_kappa(np.array([0, 0]), np.array([0, 0]), n_classes=5))
    # different-but-both-constant is NOT this case -- defined, per the test above.
