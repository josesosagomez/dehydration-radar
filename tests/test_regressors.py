"""T-M7-regressors: the five families, grid sizes vs budget_k, determinism, and the
per-family auditable fitted-state capture (C11/C15/C20).

Milestone 9 appends the Exp C ordinal dispatch (T-M9-ordinal's `regressors.py` half): the
six `ord_a_*`/`ord_b_frank_hall` ids, the frozen wrapper constants `build_estimator` injects
from `ExpCConfig`, and the `_bare_model` factoring the wrapper shares with the five base
families. The estimators' own behaviour is tested in `tests/test_ordinal.py`.
"""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from dehyd.config import ExpCConfig, ModelGridConfig
from dehyd.models.ordinal import FrankHallOrdinal, ThresholdedOrdinalRegressor
from dehyd.models.regressors import (
    MODEL_FAMILIES,
    ORDINAL_FAMILIES,
    SEED_SENSITIVE,
    RegressorError,
    _bare_model,
    build_estimator,
    enumerate_grid,
    fitted_state_params,
)

GRID = ModelGridConfig()
EXP_C = ExpCConfig()
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


@pytest.mark.parametrize("family", sorted(SEED_SENSITIVE & set(MODEL_FAMILIES)))
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


# ------------------------------------------------- Exp C ordinal dispatch (milestone 9)


def _ordinal_xy(seed=0, n=40, d=5):
    """Exp C's 2-column target: column 0 = L, column 1 = class. All five classes present."""
    x, l_values = _xy(seed=seed, n=n, d=d)
    classes = np.arange(n) % 5
    return x, np.column_stack([l_values, classes.astype(float)])


def _ordinal_params(family):
    """The BASE family's own grid entry — the ordinal wrapper adds no hyperparameters."""
    if family == "ord_b_frank_hall":
        return {"C": EXP_C.proportional_odds_c_grid[0]}
    return _first_params(family.removeprefix("ord_a_"))


def test_ordinal_families_are_the_six_authorized_ids():
    assert set(ORDINAL_FAMILIES) == {
        "ord_a_ridge", "ord_a_svr", "ord_a_rf", "ord_a_gbm", "ord_a_knn", "ord_b_frank_hall"
    }


def test_seed_sensitive_gains_only_the_ordinal_ensembles():
    """knn/ridge/svr ignore the seed whether or not they are wrapped, so wrapping must not
    silently promote them to 5-seed candidates (that would multiply the fit budget)."""
    assert SEED_SENSITIVE == {"rf", "gbm", "ord_a_rf", "ord_a_gbm"}


@pytest.mark.parametrize("family", ORDINAL_FAMILIES)
def test_each_ordinal_family_builds_and_fits_through_the_pipeline(family):
    """The whole point of the wrapper: the M7 Pipeline path runs 2-column y unchanged."""
    x, y2 = _ordinal_xy()
    with threadpool_limits(1):
        pipe = build_estimator(family, _ordinal_params(family), seed=0).fit(x, y2)
        preds = pipe.predict(x)
    assert preds.shape == (x.shape[0],)
    assert set(np.unique(preds)).issubset({0.0, 1.0, 2.0, 3.0, 4.0})


@pytest.mark.parametrize("base_family", MODEL_FAMILIES)
def test_ord_a_wrapper_carries_the_frozen_expc_constants(base_family):
    model = build_estimator(f"ord_a_{base_family}", _first_params(base_family), seed=3).named_steps["model"]
    assert isinstance(model, ThresholdedOrdinalRegressor)
    assert model.base_family == base_family
    assert model.base_params == _first_params(base_family)
    assert model.quantiles == EXP_C.cutpoint_quantiles          # (0.2, 0.4, 0.6, 0.8)
    assert model.min_separation == EXP_C.cutpoint_min_separation  # 1e-9
    assert model.seed == 3
    # knn is the sole frozen `class_weight_unsupported_families` entry.
    assert model.weighted is (base_family != "knn")


def test_ord_b_wrapper_carries_the_grid_c_and_the_recorded_max_iter():
    model = build_estimator("ord_b_frank_hall", {"C": 10.0}, seed=0).named_steps["model"]
    assert isinstance(model, FrankHallOrdinal)
    assert model.C == 10.0
    assert model.max_iter == 1000
    assert model.impl == EXP_C.proportional_odds_impl


@pytest.mark.parametrize("family", ["ord_a_bogus", "ord_b_frankhall", "ord_x"])
def test_unknown_ordinal_family_raises(family):
    with pytest.raises(RegressorError, match="unknown family"):
        build_estimator(family, {"alpha": 1.0}, seed=0)


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_bare_model_is_the_same_estimator_the_pipeline_wraps(family):
    """The `_bare_model` factoring is behaviour-preserving: the pipeline's model step is
    configured exactly as the standalone factory builds it (byte-neutrality at unit level;
    `tests/test_m9_pin.py` pins the same claim through the whole engine)."""
    params = _first_params(family)
    pipeline_model = build_estimator(family, params, seed=5).named_steps["model"]
    standalone = _bare_model(family, params, seed=5)
    assert type(standalone) is type(pipeline_model)
    assert standalone.get_params() == pipeline_model.get_params()


@pytest.mark.parametrize("family", ["ord_a_rf", "ord_a_gbm"])
def test_ordinal_ensembles_differ_across_seeds(family):
    """Asserted on the FITTED STATE, not on the class predictions: `SEED_SENSITIVE` means
    "this fit depends on `random_state`", which is why the harness runs it once per seed.
    Thresholding into 5 classes can quantize a small seed-driven change in L away entirely
    (gbm, whose seed only breaks split ties), so a prediction-level assertion would be
    testing the coarseness of the class grid rather than seed sensitivity."""
    x, y2 = _ordinal_xy()
    with threadpool_limits(1):
        a = build_estimator(family, _ordinal_params(family), seed=1).fit(x, y2).named_steps["model"]
        b = build_estimator(family, _ordinal_params(family), seed=2).fit(x, y2).named_steps["model"]
    state_a = fitted_state_params(family, a)
    state_b = fitted_state_params(family, b)
    assert state_a["ensemble_digest"].tobytes() != state_b["ensemble_digest"].tobytes()


@pytest.mark.parametrize("family", ORDINAL_FAMILIES)
def test_fitted_state_params_dispatches_to_the_ordinal_extractor(family):
    x, y2 = _ordinal_xy()
    with threadpool_limits(1):
        model = build_estimator(family, _ordinal_params(family), seed=0).fit(x, y2).named_steps["model"]
    state = fitted_state_params(family, model)
    assert state and all(isinstance(v, np.ndarray) for v in state.values())
    assert "class_weights_" in state
    if family.startswith("ord_a_"):
        assert state["cutpoints_"].shape == (len(EXP_C.cutpoint_quantiles),)
