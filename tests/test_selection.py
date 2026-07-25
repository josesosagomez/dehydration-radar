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
    SelectionError,
    select_candidate,
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
    assert SIMPLICITY_RANK == {"ridge": 0, "knn": 1, "svr": 2, "rf": 3, "gbm": 4}
    assert SIMPLICITY_RANK["ridge"] < SIMPLICITY_RANK["knn"] < SIMPLICITY_RANK["svr"]
