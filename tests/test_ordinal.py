"""T-M9-ordinal: the two frozen Exp C ordinal families (`models/ordinal.py`).

Every expected value here is derived from MILESTONE_9_PLAN §2.2's arithmetic by hand, not
read back from the implementation. The fixtures are chosen so that the *plausible wrong*
implementation fails:

  * cutpoints from the training TARGETS instead of the regressor's own in-sample
    PREDICTIONS (§5 trap 4) — the knn(k=n) fixture makes the two sets provably different;
  * `np.searchsorted(..., side="left")` instead of `"right"` (§5 trap 5) — the knn(k=1)
    fixture puts every prediction exactly ON a cutpoint;
  * nudging a tied cutpoint to `itself + min_separation` instead of to
    `previous + min_separation` — the all-tied fixture stays tied under the former;
  * the REJECTED O-M9-2 decision rule (`class = Σ_k 1[P(>k) > 0.5]`) instead of
    floor-then-argmax — the non-monotone cumulative fixture separates them;
  * inverse-frequency weights without the `K_present` normalization (O-M9-7) — the mean
    weight is exactly 1 only with it.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from dehyd.config import ExpCConfig
from dehyd.models.ordinal import (
    FrankHallOrdinal,
    OrdinalViabilityError,
    ThresholdedOrdinalRegressor,
    _class_probabilities_from_cumulative,
    fitted_state_params_ordinal,
    inverse_frequency_class_weights,
)
from dehyd.models.regressors import RIDGE_SOLVER

EXP_C = ExpCConfig()
QUANTILES = EXP_C.cutpoint_quantiles           # (0.2, 0.4, 0.6, 0.8)
MIN_SEP = EXP_C.cutpoint_min_separation        # 1e-9


def thresholded(base_family, base_params, *, weighted, seed=0):
    return ThresholdedOrdinalRegressor(
        base_family, base_params,
        quantiles=QUANTILES, min_separation=MIN_SEP, weighted=weighted, seed=seed,
    )


def state_bytes(family, model) -> dict:
    return {k: v.tobytes() for k, v in fitted_state_params_ordinal(family, model).items()}


def two_column_y(l_values, class_values) -> np.ndarray:
    """Exp C's target convention: column 0 = L = -Δm%, column 1 = class (S0-S4)."""
    return np.column_stack([np.asarray(l_values, dtype=float), np.asarray(class_values, dtype=float)])


# ------------------------------------------------------- O-M9-7 inverse-frequency weights


def test_inverse_frequency_weights_hand_computed():
    # n = 5, K_present = 3, n_0 = 3, n_1 = n_2 = 1
    #   w(0) = 5 / (3 * 3) = 5/9,  w(1) = w(2) = 5 / (3 * 1) = 5/3
    weights = inverse_frequency_class_weights([0, 0, 0, 1, 2])
    assert weights == pytest.approx([5 / 9, 5 / 9, 5 / 9, 5 / 3, 5 / 3])


def test_inverse_frequency_weights_have_mean_one():
    """The O-M9-7 normalization's whole point: mean weight 1, so the frozen grids'
    regularization strengths keep their Exp A meaning. The un-normalized alternative
    `w(c) = n / n_c` has mean K_present (= 3 here), so it fails this."""
    weights = inverse_frequency_class_weights([0, 0, 0, 1, 2])
    assert float(np.mean(weights)) == pytest.approx(1.0)


def test_inverse_frequency_weights_are_all_one_when_balanced():
    # n = 10, K_present = 5, n_c = 2 for every c  ->  w = 10 / (5 * 2) = 1
    weights = inverse_frequency_class_weights([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    assert weights == pytest.approx(np.ones(10))


def test_inverse_frequency_weights_count_only_present_classes():
    # Class 3 absent: n = 4, K_present = 2 (classes 0 and 4), n_0 = 3, n_4 = 1
    #   w(0) = 4 / (2 * 3) = 2/3,  w(4) = 4 / (2 * 1) = 2
    weights = inverse_frequency_class_weights([0, 0, 0, 4])
    assert weights == pytest.approx([2 / 3, 2 / 3, 2 / 3, 2.0])


# --------------------------------------------- family (a): cutpoint source and orientation


def test_cutpoints_come_from_in_sample_predictions_not_targets():
    """knn with k = n predicts the training mean for every row, so the prediction
    quantiles (all 3.2) and the target quantiles ([0.8, 1.6, 2.4, 4.4]) are provably
    different sets. An implementation quantiling the TARGETS fails here (§5 trap 4)."""
    x = np.arange(5, dtype=float).reshape(5, 1)
    l_values = [0.0, 1.0, 2.0, 3.0, 10.0]          # mean = 16/5 = 3.2 exactly
    y2 = two_column_y(l_values, [0, 1, 2, 3, 4])

    model = thresholded("knn", {"n_neighbors": 5}, weighted=False).fit(x, y2)

    # All four prediction quantiles are 3.2, so the strict-increase nudge lifts each one
    # to its PREDECESSOR + 1e-9 (chained, exactly as the spec states).
    c0 = 3.2
    c1 = c0 + MIN_SEP
    c2 = c1 + MIN_SEP
    c3 = c2 + MIN_SEP
    assert list(model.cutpoints_) == [c0, c1, c2, c3]
    assert np.all(np.diff(model.cutpoints_) > 0.0)     # searchsorted-safe

    # The target quantiles are a different set entirely (linear interpolation on
    # [0, 1, 2, 3, 10] at h = 4q gives 0.8, 1.6, 2.4, 4.4).
    assert np.quantile(l_values, QUANTILES) == pytest.approx([0.8, 1.6, 2.4, 4.4])

    # Every prediction equals cutpoint 0 exactly; side="right" sends it to class 1.
    assert list(model.predict(x)) == [1.0] * 5


def test_predictions_exactly_on_a_cutpoint_go_to_the_higher_class():
    """knn with k = 1 reproduces the training targets exactly in-sample, so cutpoints are
    the target quantiles [1, 2, 3, 4] and four of the six predictions sit exactly on one.
    side="left" would give [0, 0, 1, 2, 3, 4] and fails here (§5 trap 5)."""
    x = np.arange(6, dtype=float).reshape(6, 1)
    y2 = two_column_y([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0, 0, 1, 2, 3, 4])

    model = thresholded("knn", {"n_neighbors": 1}, weighted=False).fit(x, y2)

    assert list(model.cutpoints_) == [1.0, 2.0, 3.0, 4.0]   # already strictly increasing
    assert list(model.predict(x)) == [0.0, 1.0, 2.0, 3.0, 4.0, 4.0]


def test_predicted_classes_are_floats_in_the_five_class_range():
    x = np.arange(12, dtype=float).reshape(6, 2)
    y2 = two_column_y([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0, 1, 2, 3, 4, 4])
    preds = thresholded("ridge", {"alpha": 1.0}, weighted=True).fit(x, y2).predict(x)
    assert preds.dtype == np.dtype(float)
    assert set(np.unique(preds)).issubset({0.0, 1.0, 2.0, 3.0, 4.0})


# ------------------------------------------------ family (a): weights reach the base fit


def test_class_weights_are_the_sample_weights_the_base_regressor_is_fit_with():
    """Hand-computed weights, then the SAME ridge fit reproduced independently. An
    implementation that computed weights but never passed them to `.fit()` fails on the
    second assertion (the weighted and unweighted coefficients differ here)."""
    rng = np.random.default_rng(20260730)
    x = rng.normal(size=(6, 2))
    l_values = np.array([0.0, 0.4, 0.9, 1.6, 2.5, 3.6])
    classes = [0, 0, 0, 0, 1, 2]
    # n = 6, K_present = 3, n_0 = 4, n_1 = n_2 = 1
    #   w(0) = 6 / (3 * 4) = 0.5,  w(1) = w(2) = 6 / (3 * 1) = 2.0
    expected_weights = np.array([0.5, 0.5, 0.5, 0.5, 2.0, 2.0])

    model = thresholded("ridge", {"alpha": 1.0}, weighted=True).fit(x, two_column_y(l_values, classes))

    assert model.class_weights_ == pytest.approx(expected_weights)
    reference = Ridge(alpha=1.0, solver=RIDGE_SOLVER).fit(x, l_values, sample_weight=expected_weights)
    np.testing.assert_array_equal(model.base_model_.coef_, reference.coef_)

    unweighted = Ridge(alpha=1.0, solver=RIDGE_SOLVER).fit(x, l_values)
    assert not np.array_equal(model.base_model_.coef_, unweighted.coef_)


def test_knn_is_fit_unweighted():
    """The frozen `class_weight_unsupported_families = ("knn",)`: no weights computed, and
    the audited state records the absence as an empty array."""
    x = np.arange(10, dtype=float).reshape(5, 2)
    y2 = two_column_y([0.0, 1.0, 2.0, 3.0, 4.0], [0, 0, 0, 1, 2])

    model = thresholded("knn", {"n_neighbors": 3}, weighted=False).fit(x, y2)

    assert model.class_weights_ is None
    assert fitted_state_params_ordinal("ord_a_knn", model)["class_weights_"].size == 0


# ------------------------------------------------------ family (a): train-only fitted state


def test_cutpoints_and_weights_are_a_function_of_the_training_rows_alone():
    """The mutation property at estimator level: the fitted state must depend on the rows
    handed to `.fit()` and on nothing else. Fails against any implementation that reached
    for the full array the provider built (all 8 rows) instead of the training slice."""
    rng = np.random.default_rng(4242)
    x_all = rng.normal(size=(8, 3))
    l_all = rng.normal(size=8)
    classes_all = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=float)
    train = slice(0, 5)

    def fit_on_train(l_values, classes):
        y2 = two_column_y(l_values, classes)
        return thresholded("ridge", {"alpha": 1.0}, weighted=True).fit(x_all[train], y2[train])

    before = state_bytes("ord_a_ridge", fit_on_train(l_all, classes_all))

    l_mutated = l_all.copy()
    l_mutated[5:] += 100.0                 # held-out L moved far away
    classes_mutated = classes_all.copy()
    classes_mutated[5:] = 4.0              # held-out classes collapsed onto one class
    after = state_bytes("ord_a_ridge", fit_on_train(l_mutated, classes_mutated))

    assert before == after


def test_mutating_a_training_row_does_move_the_fitted_state():
    """The power companion to the test above — without it, a `fit` that ignored its inputs
    entirely would also pass."""
    rng = np.random.default_rng(4242)
    x_all = rng.normal(size=(8, 3))
    l_all = rng.normal(size=8)
    classes_all = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=float)
    train = slice(0, 5)

    def fit_on_train(l_values):
        y2 = two_column_y(l_values, classes_all)
        return thresholded("ridge", {"alpha": 1.0}, weighted=True).fit(x_all[train], y2[train])

    before = state_bytes("ord_a_ridge", fit_on_train(l_all))
    l_mutated = l_all.copy()
    l_mutated[0] += 100.0
    assert state_bytes("ord_a_ridge", fit_on_train(l_mutated)) != before


@pytest.mark.parametrize("base_family,params", [("rf", {"n_estimators": 20, "max_depth": 3}),
                                                ("gbm", {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 2})])
def test_seeded_fits_are_bit_deterministic(base_family, params):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 3))
    y2 = two_column_y(rng.normal(size=20), np.arange(20) % 5)
    a = thresholded(base_family, params, weighted=True, seed=7).fit(x, y2)
    b = thresholded(base_family, params, weighted=True, seed=7).fit(x, y2)
    assert state_bytes(f"ord_a_{base_family}", a) == state_bytes(f"ord_a_{base_family}", b)


def test_two_column_target_is_required():
    x = np.arange(10, dtype=float).reshape(5, 2)
    with pytest.raises(ValueError, match="2-column"):
        thresholded("ridge", {"alpha": 1.0}, weighted=True).fit(x, np.arange(5, dtype=float))


# ------------------------------------------------------- family (b): Frank-Hall recovery


def test_probability_recovery_is_successive_differences_of_the_cumulatives():
    """Hand-computed on exactly-representable cumulatives P(>k) = [0.875, 0.75, 0.5, 0.25]:
        p(0) = 1 - 0.875 = 0.125
        p(1) = 0.875 - 0.75 = 0.125
        p(2) = 0.75 - 0.5  = 0.25
        p(3) = 0.5 - 0.25  = 0.25
        p(4) = 0.25 - 0    = 0.25
    """
    recovered = _class_probabilities_from_cumulative(np.array([[0.875, 0.75, 0.5, 0.25]]))
    assert list(recovered[0]) == [0.125, 0.125, 0.25, 0.25, 0.25]
    assert float(recovered.sum()) == 1.0


def test_negative_successive_differences_are_floored_then_argmaxed():
    """The four binaries are unlinked, so P(>k) need not decrease in k. On
    P(>k) = [0.25, 0.5, 0.75, 0.5] the raw differences are

        p = [0.75, -0.25, -0.25, 0.25, 0.5]   ->  floored  [0.75, 0, 0, 0.25, 0.5]

    so O-M9-2's floor-then-argmax gives class 0. The REJECTED alternative
    `class = Σ_k 1[P(>k) > 0.5]` gives class 1 on this row, and an implementation taking
    |difference| would give 0.75 vs 0.5 -> still 0 but with a wrong probability vector,
    which the floored values below catch.
    """
    cumulative = np.array([[0.25, 0.5, 0.75, 0.5]])
    recovered = _class_probabilities_from_cumulative(cumulative)
    assert list(recovered[0]) == [0.75, 0.0, 0.0, 0.25, 0.5]
    assert int(np.argmax(recovered[0])) == 0


def test_argmax_ties_break_toward_the_lower_class():
    """[0.125, 0.125, 0.25, 0.25, 0.25] has a three-way maximum; O-M9-2 takes the lowest."""
    recovered = _class_probabilities_from_cumulative(np.array([[0.875, 0.75, 0.5, 0.25]]))
    assert int(np.argmax(recovered[0])) == 2


def _separable_frank_hall_fixture():
    """Five well-separated groups, two rows each, one per class — every binary threshold
    1[class > k] is linearly separable, so the recovered class should be exact."""
    x = np.array([[c * 10.0 + d] for c in range(5) for d in (0.0, 1.0)])
    classes = np.repeat(np.arange(5), 2)
    return x, two_column_y(np.zeros(10), classes), classes


def test_frank_hall_recovers_the_true_class_on_separable_data():
    x, y2, classes = _separable_frank_hall_fixture()
    model = FrankHallOrdinal(10.0).fit(x, y2)
    assert len(model.classifiers_) == 4                      # K - 1 thresholds
    assert list(model.predict(x)) == [float(c) for c in classes]


def test_frank_hall_predict_proba_is_a_normalized_five_column_matrix():
    x, y2, _ = _separable_frank_hall_fixture()
    proba = FrankHallOrdinal(10.0).fit(x, y2).predict_proba(x)
    assert proba.shape == (10, 5)
    assert np.all(proba >= 0.0)
    assert proba.sum(axis=1) == pytest.approx(np.ones(10))


def test_frank_hall_predict_agrees_with_its_own_probability_matrix():
    x, y2, _ = _separable_frank_hall_fixture()
    model = FrankHallOrdinal(10.0).fit(x, y2)
    assert list(model.predict(x)) == [float(j) for j in np.argmax(model.predict_proba(x), axis=1)]


def test_frank_hall_raises_on_a_single_class_binary_target():
    """classes = [0, 0, 1, 1]: threshold k = 0 is fine, k = 1 has an all-zero target. The
    harness viability check (step 4) makes this unreachable in a real run — this is the
    defense-in-depth path, and it must be typed and loud, never a silent all-zeros fit."""
    x = np.arange(4, dtype=float).reshape(4, 1)
    y2 = two_column_y([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1])
    with pytest.raises(OrdinalViabilityError, match="k=1"):
        FrankHallOrdinal(1.0).fit(x, y2)


def test_frank_hall_fits_are_bit_deterministic():
    x, y2, _ = _separable_frank_hall_fixture()
    a = FrankHallOrdinal(1.0).fit(x, y2)
    b = FrankHallOrdinal(1.0).fit(x, y2)
    assert state_bytes("ord_b_frank_hall", a) == state_bytes("ord_b_frank_hall", b)


def test_frank_hall_state_is_a_function_of_the_training_rows_alone():
    rng = np.random.default_rng(99)
    x_all = np.concatenate([np.array([[c * 10.0 + d] for c in range(5) for d in (0.0, 1.0)]),
                            rng.normal(size=(4, 1))])
    classes_all = np.concatenate([np.repeat(np.arange(5), 2), np.array([0, 1, 2, 3])])
    train = slice(0, 10)

    def fit_on_train(classes):
        y2 = two_column_y(np.zeros(len(classes)), classes)
        return FrankHallOrdinal(1.0).fit(x_all[train], y2[train])

    before = state_bytes("ord_b_frank_hall", fit_on_train(classes_all))
    mutated = classes_all.copy()
    mutated[10:] = 4                      # held-out rows collapsed onto one class
    assert state_bytes("ord_b_frank_hall", fit_on_train(mutated)) == before


# ------------------------------------------------------------- auditable fitted state


def test_ord_a_state_is_the_base_state_plus_cutpoints_and_weights():
    x = np.arange(20, dtype=float).reshape(10, 2)
    y2 = two_column_y(np.arange(10, dtype=float), np.arange(10) % 5)
    model = thresholded("ridge", {"alpha": 1.0}, weighted=True).fit(x, y2)

    state = fitted_state_params_ordinal("ord_a_ridge", model)
    assert set(state) == {"coef_", "intercept_", "cutpoints_", "class_weights_"}
    assert all(isinstance(v, np.ndarray) for v in state.values())
    assert state["cutpoints_"].shape == (4,)
    assert state["class_weights_"].shape == (10,)


def test_ord_b_state_stacks_every_threshold_coefficient():
    x, y2, _ = _separable_frank_hall_fixture()
    model = FrankHallOrdinal(1.0).fit(x, y2)

    state = fitted_state_params_ordinal("ord_b_frank_hall", model)
    assert set(state) == {"coef_", "intercept_", "class_weights_"}
    assert all(isinstance(v, np.ndarray) for v in state.values())
    assert state["coef_"].shape == (4, 1)          # 4 thresholds x 1 feature
    assert state["intercept_"].shape == (4,)
    assert state["class_weights_"].shape == (10,)


def test_unknown_ordinal_family_in_state_extractor_raises():
    x, y2, _ = _separable_frank_hall_fixture()
    model = FrankHallOrdinal(1.0).fit(x, y2)
    with pytest.raises(ValueError, match="unknown ordinal family"):
        fitted_state_params_ordinal("ord_c_nonsense", model)
