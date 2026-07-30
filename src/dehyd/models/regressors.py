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
    "RIDGE_SOLVER",
    "build_estimator",
    "enumerate_grid",
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
