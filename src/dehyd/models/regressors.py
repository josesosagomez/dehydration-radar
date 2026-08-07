"""The five classical regression families + grid enumeration + auditable fitted state.

One definition each of a family's estimator and its hyperparameter grid, so the harness
never inlines a model. Every estimator is a `Pipeline([("scaler", StandardScaler()),
("model", ...)])` fit inside the CV fold — the scaler is the fit-on-train transform the
audit records as `quantity="scaler"`; the model step is recorded as `quantity=<family>`.

Milestone 9 adds Experiment C's six ordinal family ids on top of the same machinery:
`ord_a_<base family>` wraps one of the five base regressors in the frozen thresholding
rule, and `ord_b_frank_hall` is the Frank-Hall decomposition. Both live in `ordinal.py`;
this module only dispatches to them, so there is still exactly one place that turns a
family id into an estimator. `_bare_model` is the shared factory the five base families and
the `ord_a_*` wrapper both build from — the wrapper needs the *unpipelined* regressor,
since the scaler already sits outside it.

Determinism (bit-identity is claimed per-machine): ridge/svr/knn are deterministic; rf/gbm
take `random_state=seed`. No `n_jobs` anywhere — the harness runs the numeric work under
`threadpool_limits(1)`.

Auditable fitted state (MILESTONE_7_PLAN §2.3, C11/C15/C20): `fitted_state_params` returns
the COMPLETE prediction-determining state of a fitted model as `{str: np.ndarray}` so the
mutation property tests can compare it bit-for-bit. For rf/gbm the trees are too large to
store verbatim, so the state is a uint8 sha256 digest over a canonical serialization of
every tree's node arrays PLUS the fitted initializer (`gbm.init_`) and the combining
hyperparameters (`learning_rate`/`n_estimators`/`max_depth`) — all three determine the
output, so perturbing any of them changes the digest.
"""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ..config import ExpCConfig, ModelGridConfig
from ..eval.selection import SIMPLICITY_RANK  # re-export: one source for the family ranking
from .ordinal import (
    ORDINAL_A_PREFIX,
    ORDINAL_B_FAMILY,
    FrankHallOrdinal,
    ThresholdedOrdinalRegressor,
    fitted_state_params_ordinal,
)

RIDGE_SOLVER = "cholesky"  # deterministic; matches the frozen reference procedure
MODEL_FAMILIES = ("ridge", "svr", "rf", "gbm", "knn")

# Families whose `.fit` accepts `sample_weight` at all. Recorded because it is the obvious
# first thing to reach for when implementing the milestone-10 bootstrap -- and, for svr and
# rf, the wrong thing: accepting a weight is not the same as being equivalent to duplicating
# a row. See `fit_pipeline` (amendment A-M10-8) for the two mechanisms and the tests that
# pin them. knn accepts no weight at all, which is also why the frozen
# `ExpCConfig.class_weight_unsupported_families` lists it.
SAMPLE_WEIGHT_SUPPORTED = frozenset({"ridge", "svr", "rf", "gbm"})
# The six authorized Exp C family ids (arm (a) = one per base family, arm (b) = one).
ORDINAL_FAMILIES = tuple(ORDINAL_A_PREFIX + f for f in MODEL_FAMILIES) + (ORDINAL_B_FAMILY,)
SEED_SENSITIVE = frozenset({"rf", "gbm", "ord_a_rf", "ord_a_gbm"})  # the rest ignore the seed

# The frozen Exp C wrapper constants. `load_config` rejects any run YAML that changes an
# M6 section, so these defaults ARE the only values a run can carry; Exp C's fit guard
# re-checks each fitted wrapper against the run's own ExpCConfig anyway.
EXP_C = ExpCConfig()

__all__ = [
    "SIMPLICITY_RANK",
    "MODEL_FAMILIES",
    "ORDINAL_FAMILIES",
    "SEED_SENSITIVE",
    "SAMPLE_WEIGHT_SUPPORTED",
    "RIDGE_SOLVER",
    "build_estimator",
    "enumerate_grid",
    "expand_by_multiplicity",
    "fit_pipeline",
    "fitted_state_params",
]


class RegressorError(ValueError):
    """Unknown family or malformed hyperparameter dict."""


def build_estimator(family: str, params: dict, *, seed: int) -> Pipeline:
    """One family's estimator, wrapped in a train-fit StandardScaler pipeline."""
    model = (
        _ordinal_model(family, params, seed=seed)
        if family.startswith("ord_")
        else _bare_model(family, params, seed=seed)
    )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def _bare_model(family: str, params: dict, *, seed: int):
    """The UNPIPELINED estimator for one of the five base families.

    Factored out of `build_estimator` at milestone 9 so `ThresholdedOrdinalRegressor` can
    build the same base regressor without a nested scaler pipeline. The five branches are
    unchanged from milestone 7 (pinned by `tests/test_m9_pin.py`).
    """
    if family == "ridge":
        model = Ridge(alpha=params["alpha"], solver=RIDGE_SOLVER)
    elif family == "svr":
        model = SVR(C=params["C"], epsilon=params["epsilon"])  # rbf kernel (default)
    elif family == "knn":
        model = KNeighborsRegressor(n_neighbors=params["n_neighbors"])
    elif family == "rf":
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=seed,
        )
    elif family == "gbm":
        model = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=seed,
        )
    else:
        raise RegressorError(f"unknown family {family!r} (expected one of {MODEL_FAMILIES})")
    return model


def _ordinal_model(family: str, params: dict, *, seed: int):
    """The unpipelined Exp C estimator for one of the six ordinal family ids.

    The wrapper constants come from the frozen `ExpCConfig`, so a candidate carries only
    the BASE family's own grid parameters (`{"alpha": ...}`, `{"C": ..., "epsilon": ...}`,
    ...) for arm (a) and `{"C": ...}` for arm (b) — the thresholding rule, the quantiles and
    the class-weighting are protocol, not hyperparameters, and are never searched over.
    """
    if family.startswith(ORDINAL_A_PREFIX):
        base_family = family[len(ORDINAL_A_PREFIX):]
        if base_family in MODEL_FAMILIES:
            return ThresholdedOrdinalRegressor(
                base_family,
                params,
                quantiles=EXP_C.cutpoint_quantiles,
                min_separation=EXP_C.cutpoint_min_separation,
                # knn is the frozen `class_weight_unsupported_families` entry — sklearn's
                # KNeighborsRegressor.fit takes no sample_weight at all.
                weighted=base_family not in EXP_C.class_weight_unsupported_families,
                seed=seed,
            )
    elif family == ORDINAL_B_FAMILY:
        return FrankHallOrdinal(params["C"])
    raise RegressorError(f"unknown family {family!r} (expected one of {ORDINAL_FAMILIES})")


# ------------------------------------------------- milestone 10: multiplicity-aware fitting
#
# The robustness bootstrap (plan §2.4) resamples SUBJECTS with replacement, so a training
# subject can appear m_s > 1 times. The frozen requirement is that this equals an explicitly
# duplicated cohort -- "multiplicity must reach the complete procedure" -- and that the
# default path stays byte-identical to Experiments A-D. Both are enforced here, in ONE
# dispatch, rather than at each of the harness's call sites.


def expand_by_multiplicity(row_multiplicity, *arrays):
    """Repeat each row `m` times, contiguously, in the caller's original row order.

    `np.repeat` on axis 0 gives exactly the frozen order (plan §2.4: "row order is original
    canonical order with each row repeated contiguously"), so the expanded cohort is a
    deterministic function of the inputs -- no sort, no shuffle, no RNG.
    """
    m = np.asarray(row_multiplicity)
    if m.ndim != 1:
        raise RegressorError(f"row_multiplicity must be 1-D, got shape {m.shape}")
    if not np.issubdtype(m.dtype, np.integer):
        raise RegressorError(f"row_multiplicity must be integer counts, got dtype {m.dtype}")
    if np.any(m < 0):
        raise RegressorError("row_multiplicity must be non-negative")
    for array in arrays:
        if len(array) != m.size:
            raise RegressorError(
                f"row_multiplicity has {m.size} entries but an array has {len(array)} rows"
            )
    return tuple(np.repeat(np.asarray(a), m, axis=0) for a in arrays)


def fit_pipeline(pipe, X, y, *, row_multiplicity=None):
    """Fit a scaler+model pipeline, optionally under integer row multiplicities.

    `row_multiplicity=None` executes `pipe.fit(X, y)` -- the exact statement milestones 7-9
    ran, so every Experiment A-D fit stays byte-identical. That is the whole reason this is a
    dispatch rather than a rewrite of the fitting code.

    Otherwise the rows are **physically duplicated**, for every family. `sample_weight` is
    deliberately not used, which is amendment **A-M10-8** and a departure from
    `plans/MILESTONE_10_PLAN.md` §2.4/§4.1 ("weighted families pass row multiplicity to both
    `StandardScaler.fit(sample_weight=...)` and estimator `fit(sample_weight=...)`", with
    expansion reserved for knn). Weighting cannot satisfy the plan's OWN acceptance criterion
    -- §5.5's "direct-equivalence fixtures compare the multiplicity implementation with an
    explicitly duplicated cohort" -- for two of the five families, and the reasons are
    mechanical rather than a matter of tolerance (both are pinned by test):

      * **svr** -- the frozen grid leaves `gamma` at sklearn's default `"scale"`, i.e.
        `1 / (n_features * X.var())`. `X.var()` is computed from the rows actually passed to
        `fit` and ignores `sample_weight`, so a weighted fit uses the UNIQUE rows' variance
        and a duplicated fit the drawn cohort's. Different kernel width, different model.
        With `gamma` pinned to a constant the two agree to 0.0 exactly, which is what
        identifies the cause.
      * **rf** -- weights enter as weighted node counts, not as rows. The fits differ under
        `bootstrap=True` (the bootstrap draws n_samples uniformly, and n_samples differs) and
        still differ under `bootstrap=False`. RF weighting is not row duplication in any
        configuration.

    Ridge, gbm and the logistic thresholds ARE weight-equivalent, so expansion changes
    nothing for them; it simply makes one rule serve all families instead of a per-family
    argument about which is which. The cost is nil: a bootstrap replicate draws N subjects
    with replacement, so the expanded cohort has about the same number of rows as the
    original -- expansion is size-neutral here, not a blow-up.

    Expanding before the pipeline also means the scaler sees the drawn population (a scaler
    fit on the unique rows would standardize against a cohort that was never drawn), and Exp
    C's estimators compute their train-only inverse-frequency class weights on the expanded
    labels, which is exactly the effective-count formula §2.4 specifies.
    """
    if row_multiplicity is None:
        return pipe.fit(X, y)                      # the unchanged milestone-7 statement
    X_expanded, y_expanded = expand_by_multiplicity(row_multiplicity, X, y)
    return pipe.fit(X_expanded, y_expanded)


def enumerate_grid(family: str, grid: ModelGridConfig) -> list[dict]:
    """The family's full hyperparameter grid as a list of param dicts (each <= budget_k)."""
    if family == "ridge":
        return [{"alpha": a} for a in grid.ridge_alphas]
    if family == "svr":
        return [{"C": c, "epsilon": e} for c in grid.svr_c for e in grid.svr_epsilon]
    if family == "knn":
        return [{"n_neighbors": k} for k in grid.knn_n_neighbors]
    if family == "rf":
        return [
            {"n_estimators": n, "max_depth": d}
            for n in grid.rf_n_estimators
            for d in grid.rf_max_depth
        ]
    if family == "gbm":
        return [
            {"n_estimators": n, "learning_rate": lr, "max_depth": d}
            for n in grid.gbm_n_estimators
            for lr in grid.gbm_learning_rate
            for d in grid.gbm_max_depth
        ]
    raise RegressorError(f"unknown family {family!r} (expected one of {MODEL_FAMILIES})")


def _tree_bytes(tree) -> bytes:
    """A tree's full decision geometry, canonically ordered (node arrays are index-aligned)."""
    t = tree.tree_
    parts = (t.children_left, t.children_right, t.feature, t.threshold, t.value)
    return b"".join(np.ascontiguousarray(p).tobytes() for p in parts)


def _ensemble_digest(family: str, model) -> np.ndarray:
    """uint8 sha256 over the COMPLETE prediction-determining state of an rf/gbm ensemble.

    Binds: the combining hyperparameters (so `learning_rate`/`n_estimators`/`max_depth`
    changes are caught), the fitted initializer `gbm.init_.constant_` (contributes to every
    output), and every tree's node arrays. Returned as an np.ndarray so it lives in a
    FitRecord.params dict and is `.tobytes()`-comparable.
    """
    h = hashlib.sha256()
    h.update(
        repr(
            (
                family,
                int(model.n_estimators),
                model.max_depth,
                float(getattr(model, "learning_rate", 0.0)),
            )
        ).encode()
    )
    if family == "gbm":
        const = getattr(model.init_, "constant_", None)
        if const is not None:
            h.update(np.ascontiguousarray(np.asarray(const, dtype=float)).tobytes())
        for stage in model.estimators_:            # shape (n_estimators, 1) for regression
            for tree in np.atleast_1d(stage):
                h.update(_tree_bytes(tree))
    else:  # rf
        for tree in model.estimators_:
            h.update(_tree_bytes(tree))
    return np.frombuffer(h.digest(), dtype=np.uint8)


def fitted_state_params(family: str, model) -> dict[str, np.ndarray]:
    """The complete prediction-determining state of a FITTED model step, as {str: ndarray}.

    Bit-comparable (every value is an np.ndarray, per C15), so held-out-subject mutation
    tests can assert each family's state is invariant.
    """
    if family.startswith("ord_"):
        return fitted_state_params_ordinal(family, model)
    if family == "ridge":
        return {
            "coef_": np.atleast_1d(model.coef_).copy(),
            "intercept_": np.atleast_1d(model.intercept_).copy(),
        }
    if family == "svr":
        return {
            "support_vectors_": np.asarray(model.support_vectors_).copy(),
            "dual_coef_": np.asarray(model.dual_coef_).copy(),
            "intercept_": np.atleast_1d(model.intercept_).copy(),
            "_gamma": np.atleast_1d(float(getattr(model, "_gamma", np.nan))),
        }
    if family == "knn":
        return {
            "_fit_X": np.asarray(model._fit_X).copy(),
            "_y": np.asarray(model._y).copy(),
        }
    if family in ("rf", "gbm"):
        return {"ensemble_digest": _ensemble_digest(family, model)}
    raise RegressorError(f"unknown family {family!r} (expected one of {MODEL_FAMILIES})")
