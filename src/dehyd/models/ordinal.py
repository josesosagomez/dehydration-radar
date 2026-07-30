"""Experiment C's two frozen ordinal families, as sklearn-compatible estimators.

The point of wrapping them as estimators is that the milestone-7 harness engine then runs
them completely unchanged: `_fit_once` still builds `Pipeline([("scaler", StandardScaler()),
("model", ...)])` and still calls `.fit(X, y)` / `.predict(X)`, so Exp C inherits the same
fold construction, the same guard-before-every-fit contract and the same fit-audit records
as Exp A and Exp B.

**The 2-column target convention.** Exp C's `y` is `(n_rows, 2)`:

    y[:, 0] = L = -Δm%   (the continuous fluid-loss predictor's target)
    y[:, 1] = class      (the ordered S0-S4 label, i.e. the session index)

`StandardScaler` ignores `y` entirely, so only the two estimators below ever read it. The
harness's `_score` grew a fail-fast assert (milestone 9 step 4) so a 2-column `y` can never
silently reach Exp A's 1-D metric.

**Family (a) — `ThresholdedOrdinalRegressor`.** A continuous regressor for `L` whose own
IN-SAMPLE predictions on its training rows are cut at the frozen quantiles (0.2, 0.4, 0.6,
0.8) into the five ordered classes. The cutpoint source is the frozen
`ExpCConfig.cutpoint_source = family_a_regressor_in_sample_predictions_inner_train`: cutting
the training *targets* instead would look nearly identical and leak nothing, but it is a
different rule and behaves differently under a biased regressor.

**Family (b) — `FrankHallOrdinal`.** Amendment A-M6-5: `statsmodels.OrderedModel` (the
literal cumulative-link model) was verified to have no `sample_weight` support and therefore
cannot carry the mandatory inverse-frequency class weights, so the ordinal decomposition is
implemented as K-1 independent binary logistic thresholds `1[class > k]`. Because those four
fits are unlinked, the recovered successive differences can be negative — O-M9-2 floors them
at 0 and takes the argmax, ties to the lower class.

Both families consume the same train-only inverse-frequency class weights (O-M9-7), except
knn, which sklearn gives no `sample_weight` and which the freeze lists in
`class_weight_unsupported_families`.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

# The frozen S0-S4 grid (implementation_plan.md §C). K-1 = 4 Frank-Hall thresholds.
N_CLASSES = 5

# The family-id namespace Exp C's candidates use. `regressors.py` builds the authorized
# family tuple from these, so the two modules cannot disagree on what an ordinal id is.
ORDINAL_A_PREFIX = "ord_a_"            # ord_a_<base family> -> ThresholdedOrdinalRegressor
ORDINAL_B_FAMILY = "ord_b_frank_hall"  # the sole member of arm (b)

__all__ = [
    "N_CLASSES",
    "ORDINAL_A_PREFIX",
    "ORDINAL_B_FAMILY",
    "OrdinalViabilityError",
    "FrankHallOrdinal",
    "ThresholdedOrdinalRegressor",
    "fitted_state_params_ordinal",
    "inverse_frequency_class_weights",
]


class OrdinalViabilityError(ValueError):
    """Training rows that cannot support an ordinal fit (a single-class binary target).

    Typed so a caller can tell "this fold is structurally non-evaluable" apart from a bug.
    The harness's class-coverage viability check (milestone 9 step 4) makes this unreachable
    in a real run; it is defense in depth, and it must be loud rather than silently fitting
    a degenerate all-one-class logistic.
    """


def inverse_frequency_class_weights(class_labels) -> np.ndarray:
    """Per-row training weights, O-M9-7: `w(c) = n / (K_present · n_c)`.

    sklearn's own "balanced" convention. The `K_present` divisor is what makes the mean
    weight exactly 1 (the weights sum to n), so the frozen ridge/SVR/logistic grids keep the
    regularization *meaning* they had in Exp A — without it every fit would behave as though
    it had K_present times as much data.
    """
    labels = np.asarray(class_labels)
    if labels.size == 0:
        return np.zeros(0, dtype=float)
    present, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    per_class = labels.size / (present.size * counts.astype(float))
    return per_class[inverse]


def _split_target(y2) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the 2-column Exp C target into (continuous L, integer class labels)."""
    y2 = np.asarray(y2, dtype=float)
    if y2.ndim != 2 or y2.shape[1] != 2:
        raise ValueError(
            f"the ordinal estimators need a 2-column y = [L, class], got shape {y2.shape}"
        )
    # The class rides in a float column, so round rather than truncate: a 3 that arrived as
    # 2.9999999 must not silently become class 2.
    return y2[:, 0].copy(), np.rint(y2[:, 1]).astype(int)


def _strictly_increasing(cutpoints, min_separation: float) -> np.ndarray:
    """Lift tied/inverted cutpoints until the array is strictly increasing.

    Each offending cutpoint goes to its PREDECESSOR + `min_separation`, not to itself +
    `min_separation`: on a constant-prediction fold all four quantiles are equal, and the
    latter would leave them equal to each other after the nudge. `np.searchsorted` needs a
    strictly sorted array for the class boundaries to be well defined.

    `min_separation` is the frozen 1e-9, which is a real separation at the magnitude of
    L = -Δm% (a few percent); it is an absolute, not relative, nudge.
    """
    cuts = np.asarray(cutpoints, dtype=float).copy()
    for i in range(1, cuts.size):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + min_separation
    return cuts


class ThresholdedOrdinalRegressor(BaseEstimator):
    """Exp C family (a): a continuous L-regressor thresholded into the five ordered classes.

    Inherits `BaseEstimator` only — deliberately neither `RegressorMixin` nor
    `ClassifierMixin`, because `predict` returns ordered CLASS INDICES as floats and neither
    mixin's `score` semantics (R², accuracy) is the Exp C objective. `BaseEstimator` is
    still needed: sklearn's `Pipeline.predict` calls `check_is_fitted` on the final step,
    which needs `__sklearn_tags__`.
    """

    def __init__(self, base_family, base_params, *, quantiles, min_separation, weighted, seed):
        self.base_family = base_family
        self.base_params = base_params
        self.quantiles = quantiles
        self.min_separation = min_separation
        self.weighted = weighted
        self.seed = seed

    def fit(self, X, y2):
        # Deferred import: `regressors` imports this module for its ordinal dispatch, so a
        # module-level import here would close the cycle.
        from .regressors import _bare_model

        L, classes = _split_target(y2)
        X = np.asarray(X, dtype=float)

        model = _bare_model(self.base_family, dict(self.base_params), seed=self.seed)
        if self.weighted:
            self.class_weights_ = inverse_frequency_class_weights(classes)
            model.fit(X, L, sample_weight=self.class_weights_)
        else:
            self.class_weights_ = None      # knn: frozen as unweighted, no sample_weight API
            model.fit(X, L)
        self.base_model_ = model

        # The frozen cutpoint source: quantiles of this regressor's OWN in-sample
        # predictions on the very rows it was just fit on — never of the targets, and never
        # of anything predicted on held-out rows.
        in_sample = model.predict(X)
        self.cutpoints_ = _strictly_increasing(
            np.quantile(in_sample, np.asarray(self.quantiles, dtype=float)),
            self.min_separation,
        )
        return self

    def predict(self, X):
        l_hat = self.base_model_.predict(np.asarray(X, dtype=float))
        # side="right": a prediction landing exactly ON a cutpoint belongs to the HIGHER
        # class. Returned as float so both ordinal families emit the same dtype as the
        # continuous families the harness was built for.
        return np.searchsorted(self.cutpoints_, l_hat, side="right").astype(float)


def _class_probabilities_from_cumulative(cumulative) -> np.ndarray:
    """Recover P(class = j) from the K-1 cumulative probabilities P(class > k) (A-M6-5).

    Successive differences of the padded row `[1, P(>0), ..., P(>K-2), 0]`. The four binary
    classifiers are fit independently, so P(>k) need not decrease in k and a difference can
    come out negative; O-M9-2 floors those at 0. The raw differences telescope to exactly 1
    per row, so flooring can only raise a row's sum — `predict_proba`'s renormalization can
    never divide by zero.
    """
    cum = np.asarray(cumulative, dtype=float)
    n_rows = cum.shape[0]
    padded = np.column_stack([np.ones(n_rows), cum, np.zeros(n_rows)])
    return np.clip(-np.diff(padded, axis=1), 0.0, None)


class FrankHallOrdinal(BaseEstimator):
    """Exp C family (b): the Frank-Hall ordinal decomposition (A-M6-5).

    K-1 = 4 independent binary logistic fits on the targets `1[class > k]`, each carrying
    the same train-only inverse-frequency class weights. `max_iter = 1000` is a solver
    convergence bound, not a tuned quantity. A `ConvergenceWarning` from lbfgs is promoted
    to an exception: a non-converged threshold fit must stop the run, never contribute
    coefficients to a reported result.
    """

    # The A-M6-5 implementation tag that `ExpCConfig.proportional_odds_impl` names. Written
    # out here rather than read from the config so Exp C's fit guard compares two
    # independently-stated values instead of a value with itself.
    impl = "frank_hall_ordinal_decomposition_sklearn_logisticregression"

    def __init__(self, C, *, max_iter=1000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y2):
        _, classes = _split_target(y2)
        X = np.asarray(X, dtype=float)
        self.class_weights_ = inverse_frequency_class_weights(classes)

        self.classifiers_ = []
        for k in range(N_CLASSES - 1):
            binary = (classes > k).astype(int)
            if np.unique(binary).size < 2:
                raise OrdinalViabilityError(
                    f"Frank-Hall threshold k={k}: the binary target 1[class > {k}] is "
                    f"single-class on the training rows (classes present: "
                    f"{sorted(set(classes.tolist()))}) — this fold does not cover S0-S4"
                )
            clf = LogisticRegression(C=self.C, solver="lbfgs", max_iter=self.max_iter)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                clf.fit(X, binary, sample_weight=self.class_weights_)
            self.classifiers_.append(clf)
        return self

    def _cumulative(self, X) -> np.ndarray:
        """P(class > k) for k = 0..K-2, shape (n_rows, K-1).

        Column 1 of `predict_proba` is P(y = 1) because each binary target contains both 0
        and 1 (guaranteed by the single-class check in `fit`), so `classes_ == [0, 1]`.
        """
        X = np.asarray(X, dtype=float)
        return np.column_stack([clf.predict_proba(X)[:, 1] for clf in self.classifiers_])

    def predict(self, X):
        probabilities = _class_probabilities_from_cumulative(self._cumulative(X))
        # argmax over the FLOORED (unnormalized) probabilities — renormalizing is a positive
        # per-row scaling and cannot move the argmax. np.argmax returns the FIRST maximum,
        # which is O-M9-2's "ties broken toward the lower class".
        return np.argmax(probabilities, axis=1).astype(float)

    def predict_proba(self, X) -> np.ndarray:
        """The recovered (floored, renormalized) class-probability matrix, for the reports."""
        probabilities = _class_probabilities_from_cumulative(self._cumulative(X))
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def fitted_state_params_ordinal(family: str, model) -> dict[str, np.ndarray]:
    """The complete prediction-determining state of a FITTED ordinal estimator.

    Every value is an np.ndarray so the held-out-mutation tests can compare it bit-for-bit
    (C15), exactly as `regressors.fitted_state_params` does for the five base families. An
    unweighted family (knn) records `class_weights_` as an EMPTY array: per-row weights are
    never empty for a non-empty training set, so empty unambiguously means "fit unweighted".
    """
    # Deferred import: see `ThresholdedOrdinalRegressor.fit`.
    from .regressors import RegressorError, fitted_state_params

    weights = getattr(model, "class_weights_", None)
    class_weights = (
        np.zeros(0, dtype=float) if weights is None else np.asarray(weights, dtype=float).copy()
    )

    if family.startswith(ORDINAL_A_PREFIX):
        state = dict(fitted_state_params(family[len(ORDINAL_A_PREFIX):], model.base_model_))
        state["cutpoints_"] = np.asarray(model.cutpoints_, dtype=float).copy()
        state["class_weights_"] = class_weights
        return state
    if family == ORDINAL_B_FAMILY:
        return {
            "coef_": np.vstack([np.atleast_2d(clf.coef_) for clf in model.classifiers_]),
            "intercept_": np.concatenate(
                [np.atleast_1d(clf.intercept_) for clf in model.classifiers_]
            ),
            "class_weights_": class_weights,
        }
    raise RegressorError(
        f"unknown ordinal family {family!r} (expected {ORDINAL_A_PREFIX}<base family> "
        f"or {ORDINAL_B_FAMILY!r})"
    )
