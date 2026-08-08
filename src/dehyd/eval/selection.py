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

Milestone 9 adds Experiment C's ordinal tie-break (`select_candidate_ordinal`, O-M9-1) to
this same module rather than to `exp_c.py`, so the single-tie-break-source doctrine holds
for every experiment:

    lower inner_val_class_mae            (the frozen primary, class-unit MAE)
      -> HIGHER inner_val_qwk            (the frozen secondary; undefined ranks last)
      -> lower simplicity_rank
      -> lower feature_dimension
      -> lower inner_fold_variance       (the frozen Exp A tail, unchanged)

The aggregation that produces those numbers — evaluable-inner-folds-only `nanmean`/`nanstd`
— lives in `exp_c.py`, deliberately NOT in the harness, so Exp A and Exp B keep their plain
`np.mean`/`np.std` over all folds and stay byte-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The frozen simplicity ordering (lower = simpler = preferred), from MILESTONE_6_PLAN.md
# §2.1a: ridge < knn < svr < rf < gbm. An ordinal ranking, NOT a literal parameter count
# (those are not comparable across these families — KNN has no parametric complexity in the
# same sense) and deliberately NOT the config.MODEL_FAMILIES enumeration order (knn sorts
# ahead of svr here on simplicity grounds).
#
# Milestone 9's six Exp C ids reuse their base family's rank: the thresholding wrapper adds
# no capacity of its own, so `ord_a_svr` is exactly as "simple" as `svr`. `ord_b_frank_hall`
# is the SOLE family in its arm, so its value can never decide a comparison across families;
# 0 records that it is linear (four logistic thresholds) and nothing more.
SIMPLICITY_RANK = {
    "ridge": 0, "knn": 1, "svr": 2, "rf": 3, "gbm": 4,
    "ord_a_ridge": 0, "ord_a_knn": 1, "ord_a_svr": 2, "ord_a_rf": 3, "ord_a_gbm": 4,
    "ord_b_frank_hall": 0,
}


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


# ------------------------------------------------- Experiment G's fusion-weight tie-break


def select_alpha(alpha_grid, objective_values, *, tie_break: str = "closest_to_one") -> float:
    """Return the fusion weight minimizing the objective, under Exp G's frozen tie-break.

    Experiment G combines the two bands as `alpha * pred_10 + (1 - alpha) * pred_77` and picks
    `alpha` off a 21-point grid by the smallest subject-balanced out-of-fold MAE. That is an
    argmin over a grid of floats, and grid argmins tie often — several alphas near the optimum
    can give bit-identical objectives when one band's predictions barely move the combination.
    `ExpGConfig.alpha_tie_break` freezes the resolution as `closest_to_one`, i.e. **ties keep
    the most weight on the primary 10 GHz band**, so a tie can never quietly manufacture a
    fusion effect out of numerical noise.

    It lives here, with `select_candidate`, for the reason this module exists at all: a
    tie-break inlined at its call site is a tie-break nobody tests. `exp_g.py` never compares
    two alphas itself.

    Args:
        alpha_grid: the candidate weights, in the frozen `ExpGConfig.alpha_grid` order.
        objective_values: one already-computed objective per alpha, same order and length.
            Lower is better. Non-finite entries are not selectable.
        tie_break: must be `"closest_to_one"` — the only frozen rule. Named rather than
            implied so a config carrying anything else fails loudly instead of silently
            getting this behaviour.

    Raises:
        SelectionError: empty/length-mismatched input, an unfrozen tie-break, or no finite
            objective anywhere on the grid.
    """
    alphas = [float(a) for a in alpha_grid]
    objectives = [float(v) for v in objective_values]
    if not alphas:
        raise SelectionError("select_alpha got an empty alpha grid")
    if len(alphas) != len(objectives):
        raise SelectionError(
            f"select_alpha got {len(alphas)} alphas and {len(objectives)} objective values"
        )
    if tie_break != "closest_to_one":
        raise SelectionError(
            f"unknown alpha tie-break {tie_break!r}; the frozen rule is 'closest_to_one'"
        )

    comparable = [(a, v) for a, v in zip(alphas, objectives) if math.isfinite(v)]
    if not comparable:
        raise SelectionError(
            f"no finite objective among {len(alphas)} alpha values — the fold selects no weight"
        )

    # `-alpha` is the last key only to make the result total: on [0, 1] no two distinct alphas
    # are equidistant from 1.0, so it never actually decides anything; it is there so a future
    # grid that did contain such a pair would still resolve toward the 10 GHz band rather than
    # toward whichever value came first in the tuple.
    return min(comparable, key=lambda pair: (pair[1], abs(pair[0] - 1.0), -pair[0]))[0]


# ----------------------------------------------------- Experiment C's ordinal tie-break


@dataclass(frozen=True)
class OrdinalCandidateScore:
    """One Exp C candidate's already-computed ordinal selection scores.

    As with `CandidateScore`, every field is supplied by the caller — here `exp_c.py`, which
    aggregates `StageOutcome.inner_scores` over the candidate's EVALUABLE inner folds only
    (`np.nanmean` / `np.nanstd`) and passes the count as `n_evaluable_inner_folds`. That
    count is part of the record because the class-coverage viability predicate is
    candidate-independent: one inner fold missing a class knocks the same cell out for every
    candidate, and `implementation_plan.md:793-800` makes the outer fold contribute no score
    only when ALL configs are non-evaluable, not when one inner fold is.

    `inner_val_qwk` comes from the stored FIRST-SEED validation predictions (O-M9-1), never
    from a re-fit: recomputing would double-fit and can drift for rf/gbm.
    """

    candidate_id: str
    inner_val_class_mae: float
    inner_val_qwk: float
    simplicity_rank: int
    feature_dimension: int
    inner_fold_variance: float
    n_evaluable_inner_folds: int


def _is_comparable_ordinal(score: OrdinalCandidateScore) -> bool:
    """`_is_comparable`'s conditions plus at least one evaluable inner fold.

    A candidate scored on zero evaluable inner folds has no evidence behind it at all; its
    `nanmean` would be NaN anyway, but the count is checked explicitly so the reason a
    candidate was skipped is a recorded field rather than an inference from a NaN.
    """
    return (
        math.isfinite(score.inner_val_class_mae)
        and math.isfinite(score.inner_fold_variance)
        and score.inner_fold_variance >= 0.0
        and score.n_evaluable_inner_folds >= 1
    )


def _ordinal_key(score: OrdinalCandidateScore) -> tuple:
    """The O-M9-1 order as one sortable tuple.

    QWK is MAXIMIZED, hence the negation, and an undefined QWK must rank below every finite
    one. It is encoded as a flag plus a SUBSTITUTED 0.0 rather than as a bare `-nan`: two
    NaN-QWK candidates would otherwise compare False in both directions at that position, and
    Python's tuple comparison would stop right there and hand the win to whichever came first
    in input order — silently skipping the simplicity, dimension and variance rungs beneath.
    With the substitution all undefined-QWK candidates tie exactly and fall through.
    """
    qwk_undefined = not math.isfinite(score.inner_val_qwk)
    return (
        score.inner_val_class_mae,
        int(qwk_undefined),
        0.0 if qwk_undefined else -score.inner_val_qwk,
        score.simplicity_rank,
        score.feature_dimension,
        score.inner_fold_variance,
    )


def select_candidate_ordinal(scores: list[OrdinalCandidateScore]) -> OrdinalCandidateScore:
    """Return the winning Exp C candidate under the O-M9-1 ordinal tie-break.

    Same shape as `select_candidate`: non-comparable candidates are filtered out first, a
    genuine full tie goes to the first in input order, and an empty comparable set raises
    `SelectionError` rather than returning something arbitrary. `exp_c.py` catches that
    error and re-raises it naming the outer fold's test subject and the missing classes.
    """
    if not scores:
        raise SelectionError("select_candidate_ordinal got an empty candidate list")

    comparable = [s for s in scores if _is_comparable_ordinal(s)]
    if not comparable:
        n_no_evaluable = sum(1 for s in scores if s.n_evaluable_inner_folds < 1)
        raise SelectionError(
            f"no comparable candidate among {len(scores)} (non-finite class-unit MAE or "
            f"non-finite/negative inner-fold variance; {n_no_evaluable} had zero evaluable "
            "inner folds) — the fold contributes no ordinal score"
        )

    return min(comparable, key=_ordinal_key)
