"""T-M10-mult: the multiplicity foundation (MILESTONE_10_PLAN.md §2.4, §4.1, §5.5).

Milestone 10's robustness bootstrap resamples SUBJECTS with replacement, so a training
subject can appear m_s > 1 times. The frozen requirement is not "weight things somehow" but
something much stronger and much easier to test: the result must equal what an **explicitly
duplicated cohort** would produce. So almost every test here is a direct-equivalence test —
fit twice, once with multiplicities on the unique rows and once on a physically duplicated
copy of the same data, and require the fitted state to agree.

The second requirement is byte-neutrality: with no multiplicity supplied, every path must run
the statements milestones 7-9 ran, because Experiments A-D are already reported and must not
move. Those tests compare against the pre-existing functions directly, not against a
recomputation.
"""

from __future__ import annotations

import numpy as np
import pytest

from dehyd.models.baselines import (
    fit_session_index_baseline,
    fit_session_mean_baseline,
    session_means,
)
from dehyd.models.ordinal import inverse_frequency_class_weights
from dehyd.models.regressors import (
    SAMPLE_WEIGHT_SUPPORTED,
    build_estimator,
    expand_by_multiplicity,
    fit_pipeline,
    fitted_state_params,
)

FAMILY_PARAMS = {
    "ridge": {"alpha": 1.0},
    "svr": {"C": 1.0, "epsilon": 0.1},
    "knn": {"n_neighbors": 3},
    "rf": {"n_estimators": 10, "max_depth": 3},
    "gbm": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
}


def _cohort(n_rows=12, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features))
    y = X[:, 0] * 0.7 - X[:, 1] * 0.3 + 0.1 * rng.standard_normal(n_rows)
    return X, y


def _duplicate(m, *arrays):
    """The reference implementation of a bootstrap cohort: literally repeat the rows."""
    return tuple(np.repeat(a, m, axis=0) for a in arrays)


def _state_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes() for k in a)


# ------------------------------------------------------------------- expansion primitive


def test_expansion_repeats_contiguously_in_original_order():
    """Plan §2.4 fixes the order: original canonical order, each row repeated contiguously.
    Anything else (a sort, a shuffle, an interleave) would still 'contain' the right rows
    while making knn's neighbour sets and Exp C's cutpoint quantiles order-dependent."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([10.0, 11.0, 12.0])
    Xe, ye = expand_by_multiplicity(np.array([2, 1, 3]), X, y)
    assert ye.tolist() == [10.0, 10.0, 11.0, 12.0, 12.0, 12.0]
    assert Xe.ravel().tolist() == [0.0, 0.0, 1.0, 2.0, 2.0, 2.0]


def test_expansion_rejects_malformed_multiplicity():
    X, y = _cohort(4)
    for bad, match in [
        (np.array([[1, 1], [1, 1]]), "1-D"),
        (np.array([1.0, 1.0, 1.0, 1.0]), "integer"),
        (np.array([1, -1, 1, 1]), "non-negative"),
        (np.array([1, 1, 1]), "entries"),
    ]:
        with pytest.raises(Exception, match=match):
            expand_by_multiplicity(bad, X, y)


# ------------------------------------------------ A-M10-8: why sample_weight was rejected
#
# The plan (§2.4/§4.1) prescribes `sample_weight` for the four families that accept one, with
# expansion reserved for knn. These two tests are the evidence that it cannot work: they pin
# the exact mechanism by which weighting diverges from duplication, and they are written so
# that if a future sklearn ever made weighting equivalent, they would FAIL and the amendment
# could be revisited rather than silently outliving its reason.


def test_svr_sample_weight_diverges_from_duplication_because_gamma_is_data_dependent():
    """`gamma='scale'` is `1 / (n_features * X.var())`, computed from the rows passed to
    `fit` and ignoring `sample_weight` — so a weighted fit uses the UNIQUE rows' variance and
    a duplicated fit the drawn cohort's. Pinning `gamma` makes them agree EXACTLY, which is
    what identifies the cause rather than merely observing a difference."""
    from sklearn.svm import SVR

    X, y = _cohort(n_rows=10, seed=3)
    m = np.array([1, 3, 1, 2, 1, 1, 4, 1, 2, 1])
    X_dup, y_dup = _duplicate(m, X, y)
    probe = _cohort(n_rows=5, seed=99)[0]

    assert not np.isclose(X.var(), X_dup.var())        # the mechanism itself
    scaled = SVR(C=1.0, epsilon=0.1).fit(X, y, sample_weight=m.astype(float))
    duplicated = SVR(C=1.0, epsilon=0.1).fit(X_dup, y_dup)
    assert np.max(np.abs(scaled.predict(probe) - duplicated.predict(probe))) > 1e-3

    pinned_w = SVR(C=1.0, epsilon=0.1, gamma=0.25).fit(X, y, sample_weight=m.astype(float))
    pinned_d = SVR(C=1.0, epsilon=0.1, gamma=0.25).fit(X_dup, y_dup)
    assert np.array_equal(pinned_w.predict(probe), pinned_d.predict(probe))


def test_rf_sample_weight_diverges_from_duplication_under_the_frozen_configuration():
    """A forest bootstraps `n_samples` rows uniformly, and `n_samples` is 10 for the weighted
    fit and 17 for the duplicated one — the resampling itself differs before any weight is
    consulted. `bootstrap=True` is sklearn's default and therefore the frozen grid's setting,
    so this is the configuration the milestone would actually have run.

    Only this is asserted. With `bootstrap=False` the two agree on some fixtures and not
    others (the per-split feature permutation draws a different number of RNG values at
    n=10 than at n=17), and a data-dependent claim is not evidence for an amendment.
    """
    from sklearn.ensemble import RandomForestRegressor

    X, y = _cohort(n_rows=10, seed=3)
    m = np.array([1, 3, 1, 2, 1, 1, 4, 1, 2, 1])
    X_dup, y_dup = _duplicate(m, X, y)
    probe = _cohort(n_rows=5, seed=99)[0]

    kwargs = dict(n_estimators=10, max_depth=3, random_state=0)   # bootstrap=True by default
    weighted = RandomForestRegressor(**kwargs).fit(X, y, sample_weight=m.astype(float))
    duplicated = RandomForestRegressor(**kwargs).fit(X_dup, y_dup)
    assert np.max(np.abs(weighted.predict(probe) - duplicated.predict(probe))) > 1e-3


def test_the_sample_weight_capability_table_is_recorded_but_not_used_for_dispatch():
    """It stays as documentation of the road not taken — accepting a weight is not the same
    as being equivalent to duplicating a row, which is the whole content of A-M10-8."""
    assert SAMPLE_WEIGHT_SUPPORTED == {"ridge", "svr", "rf", "gbm"}


# --------------------------------------------------------------------- byte-neutrality


@pytest.mark.parametrize("family", list(FAMILY_PARAMS))
def test_no_multiplicity_runs_the_unchanged_fit(family):
    """`row_multiplicity=None` must execute `pipe.fit(X, y)` — the milestone-7 statement —
    not a weights-of-one emulation of it. Experiments A-D are already reported."""
    X, y = _cohort(seed=1)
    direct = build_estimator(family, FAMILY_PARAMS[family], seed=0).fit(X, y)
    dispatched = fit_pipeline(
        build_estimator(family, FAMILY_PARAMS[family], seed=0), X, y, row_multiplicity=None,
    )
    assert _state_equal(
        fitted_state_params(family, direct.named_steps["model"]),
        fitted_state_params(family, dispatched.named_steps["model"]),
    )
    assert direct.named_steps["scaler"].mean_.tobytes() == \
        dispatched.named_steps["scaler"].mean_.tobytes()


@pytest.mark.parametrize("family", list(FAMILY_PARAMS))
def test_all_ones_multiplicity_takes_the_expansion_branch_and_lands_identically(family):
    """m == 1 everywhere still goes down the expansion branch (a no-op repeat), so this
    checks the branch itself is faithful rather than that it was skipped."""
    X, y = _cohort(seed=2)
    ones = np.ones(len(y), dtype=int)
    plain = fit_pipeline(build_estimator(family, FAMILY_PARAMS[family], seed=0), X, y,
                         row_multiplicity=None)
    expanded = fit_pipeline(build_estimator(family, FAMILY_PARAMS[family], seed=0), X, y,
                            row_multiplicity=ones)
    assert _state_equal(
        fitted_state_params(family, plain.named_steps["model"]),
        fitted_state_params(family, expanded.named_steps["model"]),
    )


# ----------------------------------------------- direct equivalence with a duplicated cohort


@pytest.mark.parametrize("family", list(FAMILY_PARAMS))
def test_weighted_fit_equals_an_explicitly_duplicated_cohort(family):
    """The frozen contract, for every family and for BOTH routes.

    `rf`/`gbm` are included deliberately even though they are stochastic: with a fixed
    `random_state` and identical effective data they must still agree, and their bootstrap
    resampling is over an internally weighted sample.
    """
    X, y = _cohort(n_rows=10, seed=3)
    m = np.array([1, 3, 1, 2, 1, 1, 4, 1, 2, 1])
    X_dup, y_dup = _duplicate(m, X, y)

    weighted = fit_pipeline(build_estimator(family, FAMILY_PARAMS[family], seed=0), X, y,
                            row_multiplicity=m)
    duplicated = build_estimator(family, FAMILY_PARAMS[family], seed=0).fit(X_dup, y_dup)

    probe = _cohort(n_rows=5, seed=99)[0]
    assert np.allclose(weighted.predict(probe), duplicated.predict(probe), rtol=0, atol=1e-9)


@pytest.mark.parametrize("family", list(FAMILY_PARAMS))
def test_the_scaler_is_fit_on_the_duplicated_population_too(family):
    """A scaler fit on the UNIQUE rows would standardize against a population that was never
    drawn. The mean/scale must be the duplicated cohort's."""
    X, y = _cohort(n_rows=10, seed=4)
    m = np.array([1, 3, 1, 2, 1, 1, 4, 1, 2, 1])
    X_dup, _ = _duplicate(m, X, y)

    weighted = fit_pipeline(build_estimator(family, FAMILY_PARAMS[family], seed=0), X, y,
                            row_multiplicity=m)
    scaler = weighted.named_steps["scaler"]
    assert np.allclose(scaler.mean_, X_dup.mean(axis=0), rtol=0, atol=1e-12)
    assert np.allclose(scaler.scale_, X_dup.std(axis=0), rtol=0, atol=1e-12)


def test_knn_is_genuinely_expanded_not_weighted():
    """A drawn-twice subject must occupy two neighbour slots — a vote has no weight to carry.
    So the fitted neighbour table itself has to grow."""
    X, y = _cohort(n_rows=8, seed=5)
    m = np.array([1, 2, 1, 3, 1, 1, 1, 1])
    fitted = fit_pipeline(build_estimator("knn", {"n_neighbors": 3}, seed=0), X, y,
                          row_multiplicity=m)
    assert fitted.named_steps["model"]._fit_X.shape[0] == int(m.sum()) == 11


# ------------------------------------------------------------ Exp C effective class weights


def test_exp_c_class_weights_become_the_effective_count_rule_under_expansion():
    """Plan §2.4 asks for `w_row = m_s * n_eff / (K_present * n_c_eff)`. Under A-M10-8 no such
    formula is implemented: the estimator is handed the EXPANDED labels, and its existing
    `n / (K_present * n_c)` is already that rule. This checks the two agree — per unique row,
    the duplicated cohort's copies must carry the total weight the formula would assign."""
    classes = np.array([0, 1, 1, 2, 3, 4])
    m = np.array([2, 1, 3, 1, 1, 2])

    duplicated_weights = inverse_frequency_class_weights(np.repeat(classes, m))
    start, per_unique_row = 0, []
    for count in m:
        per_unique_row.append(duplicated_weights[start:start + count].sum())
        start += count

    present, inverse, counts = np.unique(classes, return_inverse=True, return_counts=True)
    n_effective = m.sum()
    n_c_effective = np.array([m[inverse == i].sum() for i in range(present.size)])
    formula = m * (n_effective / (present.size * n_c_effective))[inverse]

    assert np.allclose(per_unique_row, formula, rtol=0, atol=1e-12)
    # ...and the frozen O-M9-7 property survives: mean weight 1 over the drawn cohort.
    assert np.isclose(duplicated_weights.sum(), n_effective)


# ------------------------------------------------------------------ Exp C ordinal estimators


def _ordinal_cohort(seed=7):
    rng = np.random.default_rng(seed)
    classes = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    X = rng.standard_normal((classes.size, 3)) + classes[:, None] * 0.5
    y2 = np.column_stack([-0.4 * classes + 0.05 * rng.standard_normal(classes.size), classes])
    return X, y2, classes


@pytest.mark.parametrize("family", ["ord_a_ridge", "ord_a_svr", "ord_a_knn", "ord_b_frank_hall"])
def test_ordinal_multiplicity_equals_the_duplicated_cohort(family):
    """Covers arm (a) weighted, arm (a) knn, and arm (b). Trivially true by construction under
    A-M10-8 — which is the point: the property the milestone needs holds because the code IS
    duplication, not because a weighting scheme was argued to be equivalent to it."""
    X, y2, _ = _ordinal_cohort()
    m = np.array([1, 2, 1, 1, 3, 1, 1, 2, 1, 1])
    params = {"C": 1.0} if family == "ord_b_frank_hall" else FAMILY_PARAMS[
        family.removeprefix("ord_a_")
    ]

    weighted = fit_pipeline(build_estimator(family, params, seed=0), X, y2,
                            row_multiplicity=m)
    X_dup, y2_dup = _duplicate(m, X, y2)
    duplicated = build_estimator(family, params, seed=0).fit(X_dup, y2_dup)

    probe = _ordinal_cohort(seed=42)[0]
    assert np.allclose(weighted.predict(probe), duplicated.predict(probe), rtol=0, atol=1e-9)


def test_arm_a_cutpoints_come_from_multiplicity_repeated_predictions():
    """Plan §2.4 requires arm (a)'s cutpoints to be quantiles of the in-sample predictions
    "repeated contiguously by m_s". Under A-M10-8 that happens because the fit already sees
    the repeated rows — checked against the duplicated cohort's cutpoints directly, since a
    quantile is a property of the drawn sample that a weight could not have expressed."""
    X, y2, _ = _ordinal_cohort(seed=8)
    m = np.array([1, 3, 1, 1, 2, 1, 4, 1, 1, 1])

    weighted = fit_pipeline(build_estimator("ord_a_ridge", {"alpha": 1.0}, seed=0), X, y2,
                            row_multiplicity=m)
    X_dup, y2_dup = _duplicate(m, X, y2)
    duplicated = build_estimator("ord_a_ridge", {"alpha": 1.0}, seed=0).fit(X_dup, y2_dup)

    assert np.allclose(weighted.named_steps["model"].cutpoints_,
                       duplicated.named_steps["model"].cutpoints_, rtol=0, atol=1e-12)


def test_ordinal_estimators_are_byte_neutral_without_multiplicity():
    X, y2, classes = _ordinal_cohort(seed=9)
    for family, params in [("ord_a_ridge", {"alpha": 1.0}), ("ord_b_frank_hall", {"C": 1.0})]:
        direct = build_estimator(family, params, seed=0).fit(X, y2)
        dispatched = fit_pipeline(build_estimator(family, params, seed=0), X, y2,
                                  row_multiplicity=None)
        assert _state_equal(
            fitted_state_params(family, direct.named_steps["model"]),
            fitted_state_params(family, dispatched.named_steps["model"]),
        )
        # and the weights really are the frozen O-M9-7 vector
        assert np.allclose(direct.named_steps["model"].class_weights_,
                           inverse_frequency_class_weights(classes))


# ------------------------------------------------------------------------- the baselines


def _baseline_cohort():
    subjects = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    session_idx = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    targets = np.array([-0.1, -0.5, -0.2, -0.6, -0.3, -0.7, -0.4, -0.8])
    return subjects, session_idx, targets


def test_session_index_baseline_is_byte_neutral_without_multiplicity():
    subjects, session_idx, targets = _baseline_cohort()
    outcome = fit_session_index_baseline(subjects, session_idx, targets, {1, 2, 3})
    assert outcome.model["means"] == [
        float(np.mean([-0.1, -0.2, -0.3])), float(np.mean([-0.5, -0.6, -0.7])),
    ]
    # The audit record must keep exactly the milestone-7 keys when nothing was resampled.
    assert set(outcome.fit_record.params) == {"indices", "means", "global"}


def test_session_index_baseline_weights_equal_a_duplicated_cohort():
    """Both means AND both effective denominators, which the plan requires auditing."""
    subjects, session_idx, targets = _baseline_cohort()
    train = {1, 2, 3}
    multiplicity = {1: 2, 2: 1, 3: 3}

    weighted = fit_session_index_baseline(
        subjects, session_idx, targets, train, subject_multiplicity=multiplicity
    )
    # the duplicated cohort, built by hand
    rows = [(s, i, t) for s, i, t in zip(subjects, session_idx, targets) if s in train]
    dup = [r for s, i, t in rows for r in [(s, i, t)] * multiplicity[s]]
    dup_subjects = np.array([r[0] for r in dup])
    dup_idx = np.array([r[1] for r in dup])
    dup_targets = np.array([r[2] for r in dup])
    reference = fit_session_index_baseline(dup_subjects, dup_idx, dup_targets, train)

    assert np.allclose(weighted.model["means"], reference.model["means"], rtol=0, atol=1e-12)
    assert np.isclose(weighted.model["global"], reference.model["global"], atol=1e-12)
    effective = weighted.fit_record.params["effective_rows"]
    assert effective[0] == float(len(dup))                       # global denominator
    assert effective[1:].tolist() == [3.0 + 3.0, 3.0 + 3.0]      # per session: 2+1+3 each


def test_session_means_weight_by_subject_copies_but_keep_the_distinct_subject_rule():
    """Multiplicity must not make a degenerate session viable: drawing one subject three
    times gives a session no more independent information than drawing it once."""
    subjects = np.array([1, 1, 2, 2, 3])
    session_idx = np.array([0, 1, 0, 1, 1])
    targets = np.array([-0.1, -0.5, -0.3, -0.7, -0.9])

    # Session 0 has only subject 1 among the training subjects -> dropped, however many
    # copies of subject 1 were drawn.
    means, dropped = session_means(subjects, session_idx, targets, {1, 3},
                                   subject_multiplicity={1: 5, 3: 1})
    assert dropped == (0,)
    assert np.isclose(means[1], np.average([-0.5, -0.9], weights=[5.0, 1.0]))

    # ...and without multiplicity the milestone-8 statement is unchanged.
    plain, plain_dropped = session_means(subjects, session_idx, targets, {1, 3})
    assert plain_dropped == (0,)
    assert np.isclose(plain[1], np.mean([-0.5, -0.9]))


def test_session_mean_baseline_threads_multiplicity_through():
    subjects, session_idx, targets = _baseline_cohort()
    outcome = fit_session_mean_baseline(
        subjects, session_idx, targets, {1, 2, 3}, subject_multiplicity={1: 3, 2: 1, 3: 1}
    )
    assert np.isclose(
        outcome.model["means"][0], np.average([-0.1, -0.2, -0.3], weights=[3.0, 1.0, 1.0])
    )
