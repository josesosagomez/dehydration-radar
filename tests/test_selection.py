"""The frozen tie-break (`eval/selection.py`) — the only executable code milestone 6 adds
for model selection (T-C6-stage).

Every test builds `CandidateScore`s by hand: no features, no fitting, no cohort data. The
behavioural claim that the M7 harness fits on inner-training and scores on inner-validation
across real GroupKFold folds is verified at M7 against the real harness, not here.
"""

import math

import pytest

from dehyd.eval.selection import (
    SIMPLICITY_RANK,
    CandidateScore,
    OrdinalCandidateScore,
    SelectionError,
    select_candidate,
    select_candidate_ordinal,
)


def score(cid, mae, rank=0, dim=10, var=0.0):
    return CandidateScore(cid, mae, rank, dim, var)


def test_lowest_mae_wins_outright():
    winner = select_candidate([score("a", 1.0), score("b", 0.5), score("c", 0.8)])
    assert winner.candidate_id == "b"


def test_mae_tie_broken_by_simplicity_rank():
    # equal MAE; ridge (rank 0) beats gbm (rank 4)
    winner = select_candidate([score("gbm", 0.5, rank=4), score("ridge", 0.5, rank=0)])
    assert winner.candidate_id == "ridge"


def test_mae_and_rank_tie_broken_by_feature_dimension():
    winner = select_candidate([score("big", 0.5, rank=1, dim=99), score("small", 0.5, rank=1, dim=10)])
    assert winner.candidate_id == "small"


def test_full_tie_broken_by_inner_fold_variance():
    winner = select_candidate(
        [score("hi", 0.5, rank=1, dim=10, var=0.9), score("lo", 0.5, rank=1, dim=10, var=0.1)]
    )
    assert winner.candidate_id == "lo"


def test_complete_tie_is_deterministic_first_in_order():
    a, b = score("a", 0.5), score("b", 0.5)
    assert select_candidate([a, b]).candidate_id == "a"
    assert select_candidate([b, a]).candidate_id == "b"


def test_empty_list_raises():
    with pytest.raises(SelectionError, match="empty"):
        select_candidate([])


@pytest.mark.parametrize("bad_mae", [math.nan, math.inf, -math.inf])
def test_non_finite_mae_candidate_is_excluded_not_compared(bad_mae):
    # The non-finite candidate has a "better" (smaller) rank/dim/var, yet must not win.
    winner = select_candidate([score("bad", bad_mae, rank=0, dim=1, var=0.0), score("ok", 2.0, rank=4)])
    assert winner.candidate_id == "ok"


@pytest.mark.parametrize("bad_var", [math.nan, math.inf, -1.0])
def test_non_finite_or_negative_variance_candidate_is_excluded(bad_var):
    winner = select_candidate([score("bad", 0.1, var=bad_var), score("ok", 2.0)])
    assert winner.candidate_id == "ok"


def test_all_non_comparable_raises():
    with pytest.raises(SelectionError, match="no comparable candidate"):
        select_candidate([score("a", math.nan), score("b", 1.0, var=math.inf)])


def test_simplicity_rank_ordering_is_the_frozen_one():
    # ridge < knn < svr < rf < gbm (MILESTONE_6_PLAN.md §2.1a) — knn deliberately ahead of svr.
    # M9 adds Exp C's six ordinal ids; the five base ranks and their order are unchanged, and
    # each `ord_a_*` mirrors its base family (the thresholding wrapper adds no capacity).
    assert SIMPLICITY_RANK == {
        "ridge": 0, "knn": 1, "svr": 2, "rf": 3, "gbm": 4,
        "ord_a_ridge": 0, "ord_a_knn": 1, "ord_a_svr": 2, "ord_a_rf": 3, "ord_a_gbm": 4,
        "ord_b_frank_hall": 0,
    }
    assert SIMPLICITY_RANK["ridge"] < SIMPLICITY_RANK["knn"] < SIMPLICITY_RANK["svr"]
    assert SIMPLICITY_RANK["svr"] < SIMPLICITY_RANK["rf"] < SIMPLICITY_RANK["gbm"]


# ------------------------------------------------- Exp C's ordinal tie-break (T-M9-selection)


def ordinal(cid, mae, qwk=0.5, rank=0, dim=10, var=0.0, evaluable=5):
    return OrdinalCandidateScore(cid, mae, qwk, rank, dim, var, evaluable)


def test_ordinal_lowest_class_mae_wins_outright():
    # The worse-MAE candidate has a far better QWK, yet MAE is the frozen PRIMARY metric.
    winner = select_candidate_ordinal([ordinal("a", 1.0, qwk=0.9), ordinal("b", 0.5, qwk=0.1)])
    assert winner.candidate_id == "b"


def test_ordinal_mae_tie_broken_by_higher_qwk():
    # Direction check: QWK is MAXIMIZED. A copy-paste of the MAE rung (lower wins) fails here.
    winner = select_candidate_ordinal([ordinal("low", 0.5, qwk=0.1), ordinal("high", 0.5, qwk=0.8)])
    assert winner.candidate_id == "high"


def test_ordinal_undefined_qwk_loses_to_any_finite_qwk_at_equal_mae():
    # Even a NEGATIVE (worse-than-chance) QWK outranks an undefined one.
    winner = select_candidate_ordinal(
        [ordinal("nan", 0.5, qwk=math.nan), ordinal("finite", 0.5, qwk=-0.9)]
    )
    assert winner.candidate_id == "finite"


def test_undefined_qwk_on_both_sides_falls_through_to_the_next_rung():
    """The rung below QWK must still decide when neither candidate has a defined QWK.

    This is the test that fails against the obvious `key=(mae, -qwk, rank, ...)`: with
    `-nan` at position 2 the tuple comparison is False in both directions, so `min` returns
    the FIRST input ("gbm" here) and the simplicity rung never runs.
    """
    scores = [ordinal("gbm", 0.5, qwk=math.nan, rank=4), ordinal("ridge", 0.5, qwk=math.nan, rank=0)]
    assert select_candidate_ordinal(scores).candidate_id == "ridge"
    assert select_candidate_ordinal(list(reversed(scores))).candidate_id == "ridge"


def test_ordinal_mae_and_qwk_tie_broken_by_simplicity_rank():
    winner = select_candidate_ordinal(
        [ordinal("gbm", 0.5, qwk=0.4, rank=4), ordinal("ridge", 0.5, qwk=0.4, rank=0)]
    )
    assert winner.candidate_id == "ridge"


def test_ordinal_tie_through_rank_broken_by_feature_dimension():
    winner = select_candidate_ordinal(
        [ordinal("big", 0.5, qwk=0.4, rank=1, dim=99), ordinal("small", 0.5, qwk=0.4, rank=1, dim=10)]
    )
    assert winner.candidate_id == "small"


def test_ordinal_full_tie_broken_by_inner_fold_variance():
    winner = select_candidate_ordinal(
        [
            ordinal("hi", 0.5, qwk=0.4, rank=1, dim=10, var=0.9),
            ordinal("lo", 0.5, qwk=0.4, rank=1, dim=10, var=0.1),
        ]
    )
    assert winner.candidate_id == "lo"


def test_ordinal_complete_tie_is_deterministic_first_in_order():
    a, b = ordinal("a", 0.5), ordinal("b", 0.5)
    assert select_candidate_ordinal([a, b]).candidate_id == "a"
    assert select_candidate_ordinal([b, a]).candidate_id == "b"


def test_ordinal_more_evaluable_folds_is_not_itself_a_tie_break():
    """`n_evaluable_inner_folds` gates comparability (>= 1) and is recorded — it is NOT a
    rung of the frozen order, so a fully-tied pair still goes to input order."""
    winner = select_candidate_ordinal(
        [ordinal("few", 0.5, evaluable=1), ordinal("many", 0.5, evaluable=5)]
    )
    assert winner.candidate_id == "few"


def test_zero_evaluable_inner_folds_is_incomparable_even_with_finite_scores():
    # Finite MAE, finite variance, best QWK, best rank — but no evaluable inner fold behind
    # any of it, so it must not win.
    winner = select_candidate_ordinal(
        [ordinal("empty", 0.1, qwk=0.99, rank=0, dim=1, var=0.0, evaluable=0), ordinal("ok", 2.0, rank=4)]
    )
    assert winner.candidate_id == "ok"


@pytest.mark.parametrize("bad_mae", [math.nan, math.inf, -math.inf])
def test_ordinal_non_finite_class_mae_is_excluded(bad_mae):
    winner = select_candidate_ordinal([ordinal("bad", bad_mae, qwk=0.99), ordinal("ok", 2.0)])
    assert winner.candidate_id == "ok"


@pytest.mark.parametrize("bad_var", [math.nan, math.inf, -1.0])
def test_ordinal_non_finite_or_negative_variance_is_excluded(bad_var):
    winner = select_candidate_ordinal([ordinal("bad", 0.1, var=bad_var), ordinal("ok", 2.0)])
    assert winner.candidate_id == "ok"


def test_ordinal_empty_list_raises():
    with pytest.raises(SelectionError, match="empty"):
        select_candidate_ordinal([])


def test_all_non_comparable_ordinal_raises_and_counts_the_non_evaluable_cells():
    scores = [ordinal("a", math.nan, evaluable=0), ordinal("b", 1.0, evaluable=0)]
    with pytest.raises(SelectionError, match="2 had zero evaluable inner folds"):
        select_candidate_ordinal(scores)
