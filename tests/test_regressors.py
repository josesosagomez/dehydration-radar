"""T-M7-regressors: the five families, grid sizes vs budget_k, determinism, and the
per-family auditable fitted-state capture (C11/C15/C20)."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from dehyd.config import ModelGridConfig
from dehyd.models.regressors import (
    MODEL_FAMILIES,
    SEED_SENSITIVE,
    build_estimator,
    enumerate_grid,
    fitted_state_params,
)

GRID = ModelGridConfig()
BUDGET_K = 12


def _xy(seed=0, n=40, d=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = x[:, 0] * 2.0 - x[:, 1] + rng.normal(scale=0.1, size=n)
    return x, y


def _first_params(family):
    return enumerate_grid(family, GRID)[0]


# ------------------------------------------------------------------- grid sizes


@pytest.mark.parametrize(
    "family,expected",
    [("ridge", 8), ("svr", 12), ("knn", 7), ("rf", 6), ("gbm", 8)],
)
def test_grid_sizes_match_documented_counts_and_fit_budget(family, expected):
    grid = enumerate_grid(family, GRID)
    assert len(grid) == expected
    assert len(grid) <= BUDGET_K


def test_all_families_enumerated():
    assert set(MODEL_FAMILIES) == {"ridge", "svr", "rf", "gbm", "knn"}


# ---------------------------------------------------------------- build + fit


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_each_family_builds_and_fits(family):
    x, y = _xy()
    with threadpool_limits(1):
        pipe = build_estimator(family, _first_params(family), seed=0).fit(x, y)
        preds = pipe.predict(x)
    assert preds.shape == (x.shape[0],)
    assert np.all(np.isfinite(preds))


def test_ridge_uses_cholesky_solver():
    pipe = build_estimator("ridge", {"alpha": 1.0}, seed=0)
    assert pipe.named_steps["model"].solver == "cholesky"


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_same_seed_is_bit_deterministic(family):
    x, y = _xy()
    params = _first_params(family)
    with threadpool_limits(1):
        a = build_estimator(family, params, seed=7).fit(x, y).predict(x)
        b = build_estimator(family, params, seed=7).fit(x, y).predict(x)
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("family", sorted(SEED_SENSITIVE))
def test_seed_sensitive_families_differ_across_seeds(family):
    x, y = _xy()
    params = _first_params(family)
    with threadpool_limits(1):
        a = build_estimator(family, params, seed=1).fit(x, y).predict(x)
        b = build_estimator(family, params, seed=2).fit(x, y).predict(x)
    assert a.tobytes() != b.tobytes()


# --------------------------------------------------- auditable fitted state


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_fitted_state_params_are_all_ndarrays(family):
    x, y = _xy()
    with threadpool_limits(1):
        pipe = build_estimator(family, _first_params(family), seed=0).fit(x, y)
    state = fitted_state_params(family, pipe.named_steps["model"])
    assert state and all(isinstance(v, np.ndarray) for v in state.values())
    # bit-comparable: same fit -> identical bytes
    with threadpool_limits(1):
        pipe2 = build_estimator(family, _first_params(family), seed=0).fit(x, y)
    state2 = fitted_state_params(family, pipe2.named_steps["model"])
    assert set(state) == set(state2)
    for k in state:
        assert state[k].tobytes() == state2[k].tobytes()


def test_svr_state_includes_support_vectors():
    x, y = _xy()
    with threadpool_limits(1):
        model = build_estimator("svr", _first_params("svr"), seed=0).fit(x, y).named_steps["model"]
    assert "support_vectors_" in fitted_state_params("svr", model)


def test_gbm_digest_changes_when_init_or_learning_rate_or_a_tree_changes():
    """C20: the ensemble digest binds init_, learning_rate, AND the trees."""
    x, y = _xy()
    with threadpool_limits(1):
        base = build_estimator("gbm", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 2}, seed=0).fit(x, y)
    base_digest = fitted_state_params("gbm", base.named_steps["model"])["ensemble_digest"].tobytes()

    # Different learning_rate -> different combination -> different digest.
    with threadpool_limits(1):
        lr = build_estimator("gbm", {"n_estimators": 100, "learning_rate": 0.01, "max_depth": 2}, seed=0).fit(x, y)
    assert fitted_state_params("gbm", lr.named_steps["model"])["ensemble_digest"].tobytes() != base_digest

    # A shifted target changes both init_ (mean) and the fitted trees -> different digest.
    with threadpool_limits(1):
        shifted = build_estimator("gbm", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 2}, seed=0).fit(x, y + 10.0)
    assert fitted_state_params("gbm", shifted.named_steps["model"])["ensemble_digest"].tobytes() != base_digest

    # Directly perturbing init_ alone changes the digest (init_ contributes to every output).
    model = base.named_steps["model"]
    model.init_.constant_ = model.init_.constant_ + 1.0
    assert fitted_state_params("gbm", model)["ensemble_digest"].tobytes() != base_digest
