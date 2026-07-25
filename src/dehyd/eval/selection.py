"""The frozen tie-break for nested-CV model selection.

This is the ONLY executable code milestone 6 adds beyond config and the protocol-freeze
guard. It is a pure, stateless comparison over ALREADY-COMPUTED candidate scores: it
never fits a model, never sees a frame or a subject, and never reads cohort data, so it
stays inside M6's "no predictive computation" invariant while still being real, tested
code rather than a prose promise.

`implementation_plan.md` §"LOSO harness" states the tie-break as: lower session-level MAE,
then "simpler model (fewer effective parameters / smaller feature dim)", then lower
inner-fold variance. "Simpler model" and "smaller feature dim" are made executable as two
ordered components (a literal effective-parameter count is not comparable across
ridge/knn/svr/rf/gbm, so `simplicity_rank` is a frozen ordinal family ranking instead):

    lower inner_val_mae
      -> lower simplicity_rank        (ridge=0 < knn=1 < svr=2 < rf=3 < gbm=4)
      -> lower feature_dimension      (the pooled-WST vector length for this candidate)
      -> lower inner_fold_variance

The M7 harness computes each candidate's scores (by fitting on inner-training subjects and
scoring on inner-validation subjects across the real GroupKFold folds) and then calls
`select_candidate` here, so the tie-break has exactly ONE definition the harness and the
plan both point at. That behavioural claim — the fit/score half — is verified at M7 against
the real harness, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The frozen simplicity ordering (lower = simpler = preferred), from MILESTONE_6_PLAN.md
# §2.1a: ridge < knn < svr < rf < gbm. An ordinal ranking, NOT a literal parameter count
# (those are not comparable across these families — KNN has no parametric complexity in the
# same sense) and deliberately NOT the config.MODEL_FAMILIES enumeration order (knn sorts
# ahead of svr here on simplicity grounds).
SIMPLICITY_RANK = {"ridge": 0, "knn": 1, "svr": 2, "rf": 3, "gbm": 4}


class SelectionError(ValueError):
    """Raised when no candidate is selectable (empty input, or all non-finite)."""


@dataclass(frozen=True)
class CandidateScore:
    """One candidate configuration's already-computed selection scores.

    Every field is supplied BY THE CALLER (the M7 harness) — this module computes none of
    them. `simplicity_rank` and `feature_dimension` are deterministic functions of the
    candidate alone (the frozen family ranking and the recorded pooled-WST geometry), not
    fold-dependent quantities, so the tie-break is reproducible.
    """

    candidate_id: str
    inner_val_mae: float
    simplicity_rank: int
    feature_dimension: int
    inner_fold_variance: float


def _is_comparable(score: CandidateScore) -> bool:
    """A candidate enters the tie-break only if its MAE and variance are usable.

    A non-finite MAE (a non-evaluable fold produced no score) or a non-finite/negative
    variance makes the candidate incomparable — Python's NaN ordering would otherwise make
    the winner depend silently on input order. Mirrors the Exp C fold-viability doctrine:
    non-evaluable configs are skipped in selection, not ranked.
    """
    return (
        math.isfinite(score.inner_val_mae)
        and math.isfinite(score.inner_fold_variance)
        and score.inner_fold_variance >= 0.0
    )


def select_candidate(scores: list[CandidateScore]) -> CandidateScore:
    """Return the winning candidate under the frozen tie-break.

    Non-comparable candidates (non-finite MAE, or non-finite/negative variance) are
    filtered out first; if none remain, raises `SelectionError` rather than returning an
    arbitrary result. On a genuine full tie the first such candidate in input order wins,
    so the result is deterministic. No randomness, no I/O, no model object in scope.
    """
    if not scores:
        raise SelectionError("select_candidate got an empty candidate list")

    comparable = [s for s in scores if _is_comparable(s)]
    if not comparable:
        raise SelectionError(
            f"no comparable candidate among {len(scores)} (all had non-finite MAE or "
            "non-finite/negative inner-fold variance) — the fold contributes no score"
        )

    # min() is a stable pick: on a full key tie it returns the first in input order.
    return min(
        comparable,
        key=lambda s: (
            s.inner_val_mae,
            s.simplicity_rank,
            s.feature_dimension,
            s.inner_fold_variance,
        ),
    )
