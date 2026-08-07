"""Experiment C — the ordinal 5-class (S0-S4) secondary task, on the same harness engine.

The ordinal sibling of `exp_a.py`/`exp_b.py`. It reuses Exp A's search-space enumeration and
store-backed X path UNCHANGED (A-M9-1: family (a)'s space *is* Exp A's frozen space, one
enumeration, never a second copy) and adds only what is genuinely new:

  * the 2-column target `y = [L, class]` with the frozen sign convention `L = -Δm%`, so the
    class order increases monotonically with fluid loss (`implementation_plan.md:762-765`);
  * the ordinal objective (pooled class-unit MAE) as the harness `score_fn`;
  * the two frozen arms sharing ONE Stage-1 feature key — arm (a) = the five base families
    wrapped in the frozen thresholding rule (the primary ordinal model), arm (b) = Frank-Hall
    over the frozen C grid (the "comparison");
  * the §2.3 inner-fold aggregation (evaluable folds only) that feeds
    `selection.select_candidate_ordinal`, and the per-fit authorization guard below.

**Why the aggregation lives here and not in the harness.** The class-coverage viability
predicate is candidate-INDEPENDENT: one inner-training set missing a class knocks the same
cell out for every candidate, so the harness's plain `np.mean` over all inner folds would go
NaN for *every* candidate and silently promote "one inner fold lost a class" into "this outer
fold produced no ordinal result" — stricter than the frozen rule, which does that only when
*all* configs are non-evaluable (`:793-800`). Exp C therefore reads `StageOutcome.inner_scores`
and reduces it itself with `nanmean`/`nanstd` over the evaluable folds; Exp A's and Exp B's
aggregation is untouched and stays byte-identical.

**Why there is no baseline comparison here** (plan §5 trap 16): the session-index baseline
predicts the Exp C class *perfectly* — the class IS the session index — so any radar-vs-baseline
framing is degenerate, and the freeze registers none. Exp C reports its ordinal metrics
absolutely.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..features import store as store_mod
from ..features.protocol_freeze import protocol_freeze_guard
from ..models import regressors
from ..models.ordinal import ORDINAL_A_PREFIX, ORDINAL_B_FAMILY
from . import exp_a, fold_parallel, harness
from . import metrics as M
from .harness import Candidate, FeatureBundle, require_complete_active
from .selection import (
    SIMPLICITY_RANK,
    OrdinalCandidateScore,
    SelectionError,
    select_candidate_ordinal,
)

ARMS = ("a", "b")

# The `active` protocol keys arm (b) must carry: the band key set MINUS `model_family`.
# `ord_b_frank_hall` has NO legal `model_family` value — the frozen whitelist is the five
# regressor families, forcing a fake value would corrupt the protocol record, and extending
# the whitelist would reopen M6 (plan §5 trap 2). Arm (b)'s record therefore omits the key,
# and `harness.require_complete_active` (which demands exactly the band set) cannot validate
# it, so exp_c states its own completeness contract here.
REQUIRED_ACTIVE_KEYS_C = {
    band: frozenset(keys - {"model_family"}) for band, keys in harness.REQUIRED_ACTIVE_KEYS.items()
}

# The Frank-Hall solver's convergence bound, stated here INDEPENDENTLY of
# `FrankHallOrdinal`'s own default so the authorization guard below compares two separately
# written values rather than a value with itself (the same doctrine as `FrankHallOrdinal.impl`
# vs `ExpCConfig.proportional_odds_impl`). Not a tuned quantity.
FRANK_HALL_MAX_ITER = 1000

# Exp C RNG offsets off config.run.seed -- FIXED and NAMED, never a running counter (the Exp B
# trap-10 doctrine): Exp A occupies +0..3, Exp B +100..134, Exp C +200..212.
RNG_OFFSET_EXPC_BASE = 200
_ARM_RNG_OFFSET = {"a": 0, "b": 10}
_METRIC_RNG_OFFSET = {"class_unit_mae": 0, "adjacent_accuracy": 1, "quadratic_weighted_kappa": 2}


def _rng_offset(arm: str, metric: str) -> int:
    return RNG_OFFSET_EXPC_BASE + _ARM_RNG_OFFSET[arm] + _METRIC_RNG_OFFSET[metric]


def _all_rng_offsets() -> list[int]:
    """Every resolved RNG offset this module uses -- tested directly for pairwise
    distinctness, and against Exp B's and Exp A's."""
    return [_rng_offset(arm, metric) for arm in ARMS for metric in _METRIC_RNG_OFFSET]


class ExpCError(ValueError):
    """A malformed Exp C spine, or an outer fold in which no candidate is ordinally
    evaluable (the frozen "the fold contributes no ordinal score" case, named)."""


class ExpCProtocolError(ExpCError):
    """A fit that is about to run is not authorized by Exp C's frozen protocol."""


# ---------------------------------------------------------------- data spine + eligibility


def build_sessions_c(config, band) -> list[dict]:
    """Exp A's `build_sessions()` plus the two ordinal columns. ALL S0-S4 sessions are kept:
    unlike Exp B (which excludes S0 because its Δm% is identically 0 and would be a free
    "prediction"), the five classes ARE Exp C's task and S0 is the lowest of them.

    `loss_l = -Δm%` is the frozen sign convention (`:762-765`) — positive loss magnitude, so
    class order increases monotonically with L. `class_idx` is the session index: the class
    IS the session stage.
    """
    return [
        {**s, "loss_l": -float(s["delta_m_pct"]), "class_idx": int(s["session_idx"])}
        for s in exp_a.build_sessions(config, band)
    ]


def evaluable_subjects_c(sessions) -> list[int]:
    """Exp A's rule (>= 1 eligible session, `:605-610`) — not Exp B's S1-S4 rule, since Exp C
    keeps S0. `sessions` already contains only eligible sessions, so this is the distinct
    subject set present."""
    return sorted({int(s["subject"]) for s in sessions})


# --------------------------------------------------------------------- the ordinal provider


class OrdinalFeatures:
    """Wraps (does not subclass) `exp_a.StoreBackedFeatures` -- the X path, including the
    tuned-ε cache keyed by (feature_key, frozenset(train_subjects)), is reused byte for byte,
    so the one genuinely fitted feature quantity is computed and audited exactly as in Exp A.

    The only difference is the target: `y` is the (n_rows, 2) matrix `[L, class]`. The
    pipeline's `StandardScaler` ignores y entirely, so only the ordinal estimators read it,
    and `harness._score` fails fast if a 2-column y ever reaches the 1-D Exp A metric.
    """

    def __init__(self, band, sessions, store_dir, config, *, subject_multiplicity=None):
        self.band = band
        self.config = config
        self.subject_multiplicity = subject_multiplicity
        self.base = exp_a.StoreBackedFeatures(
            band, sessions, store_dir, config, subject_multiplicity=subject_multiplicity
        )
        self.subjects = self.base.subjects
        self.session_idx = np.array([int(s["session_idx"]) for s in sessions])
        self.classes = np.array([int(s["class_idx"]) for s in sessions])
        self.loss_l = np.array([float(s["loss_l"]) for s in sessions], dtype=float)
        self.y = np.column_stack([self.loss_l, self.classes.astype(float)])

    def data_for(self, candidate, train_subjects) -> FeatureBundle:
        base_bundle = self.base.data_for(candidate, train_subjects)
        return FeatureBundle(
            subjects=base_bundle.subjects,
            X=base_bundle.X,
            y=self.y,
            extra_fits=base_bundle.extra_fits,
            session_idx=self.session_idx,
        )


def ordinal_class_mae_score(subjects, y_true2, y_pred, session_idx) -> float:
    """Exp C's frozen inner objective, in the shape the harness `score_fn` hook expects:
    pooled `class_unit_mae` between the truth column `y[:, 1]` and the predicted classes.
    Module-level (never a lambda or closure) so it survives multiprocessing pickling."""
    return M.class_unit_mae(np.asarray(y_true2)[:, 1], y_pred)


# ------------------------------------------------------------------ candidate enumeration


def stage1_candidates_c(config, band, anchor_alpha) -> list[Candidate]:
    """Exp A's Stage-1 feature-axis enumeration (72 / 9 candidates) with the family swapped to
    the family-(a) ridge anchor. A-M9-1 mandates ONE enumeration of the frozen space, so this
    delegates rather than re-listing the axes.

    The `active` record is Exp A's unchanged — it names `model_family: ridge`, the BASE family,
    which is what the fitted regressor genuinely is (the thresholding wrapper is protocol, not
    a model family) and the only value the frozen whitelist accepts.
    """
    return [
        dataclasses.replace(c, family=ORDINAL_A_PREFIX + "ridge")
        for c in exp_a.stage1_candidates(config, band, anchor_alpha)
    ]


def stage2_candidates_a(config, band, feature_key, winner_active) -> list[Candidate]:
    """Arm (a): the five base families × their frozen grids (41 candidates), each wrapped in
    the frozen thresholding rule, at the Stage-1 winning feature key."""
    return [
        dataclasses.replace(
            c, family=ORDINAL_A_PREFIX + c.family, candidate_id=ORDINAL_A_PREFIX + c.candidate_id
        )
        for c in exp_a.stage2_candidates(config, band, feature_key, winner_active)
    ]


def stage2_candidates_b(config, band, feature_key, winner_active) -> list[Candidate]:
    """Arm (b): Frank-Hall over the frozen C grid at the SAME Stage-1 feature key.

    The `active` record drops `model_family` (see `REQUIRED_ACTIVE_KEYS_C`); the feature axes
    are carried verbatim from the Stage-1 winner, so the two arms are compared on identical
    features.
    """
    active = tuple((k, v) for k, v in dict(winner_active).items() if k != "model_family")
    return [
        Candidate(
            candidate_id=f"{ORDINAL_B_FAMILY}_{i}",
            family=ORDINAL_B_FAMILY,
            model_params=(("C", c),),
            feature_key=feature_key,
            active=active,
        )
        for i, c in enumerate(config.exp_c.proportional_odds_c_grid)
    ]


# ------------------------------------------------------------------- the fit authorization


def require_complete_active_c(active: dict) -> None:
    """Arm (b)'s fail-closed completeness contract: exactly the band's feature axes, no
    `model_family`. The mirror of `harness.require_complete_active`, which cannot be used here
    because it demands the full band key set (plan §5 trap 2)."""
    band = active.get("band")
    required = REQUIRED_ACTIVE_KEYS_C.get(band)
    if required is None:
        raise ExpCProtocolError(f"active.band must be '10ghz' or '77ghz', got {band!r}")
    present = set(active)
    missing, unexpected = required - present, present - required
    if missing or unexpected:
        raise ExpCProtocolError(
            f"arm (b) active record for band {band!r} has wrong keys: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)} (required exactly {sorted(required)})"
        )


def _stage1_anchor(config, candidate) -> float:
    band = dict(candidate.active or ())["band"]
    search = config.search_10ghz if band == "10ghz" else config.search_77ghz
    return search.stage1_anchor_ridge_alpha


def assert_exp_c_fit_authorized(candidate: Candidate, config, *, arm: str) -> None:
    """Bind the computation that is ABOUT TO RUN to Exp C's frozen protocol.

    Neither existing guard does this. `protocol_freeze_guard._check_active` validates only the
    keys *present* in `active` (`protocol_freeze.py:116-136`), and the completeness contracts
    only check which feature-axis keys are there — so without this an unauthorized family, an
    off-grid hyperparameter, or a wrapper built with the wrong cutpoint quantiles would reach
    `.fit()` with every other guard passing.

    It reads the estimator the run will actually build (`regressors._ordinal_model`, the same
    factory `build_estimator` delegates to) rather than re-deriving the wrapper's properties
    from the candidate id, so a factory that returned the wrong base regressor is caught
    instead of confirmed. `arm` is the arm being run — `"stage1"`, `"a"` or `"b"` — because
    "the family is authorized" and "the family belongs to THIS arm" are different claims.
    """
    exp_c_cfg = config.exp_c
    family = candidate.family
    where = f"Exp C fit authorization failed for candidate {candidate.candidate_id!r}"

    if family not in regressors.ORDINAL_FAMILIES:
        raise ExpCProtocolError(
            f"{where}: model family {family!r} is not one of the six authorized Exp C "
            f"families {regressors.ORDINAL_FAMILIES}"
        )
    if arm == "stage1":
        if family != ORDINAL_A_PREFIX + "ridge":
            raise ExpCProtocolError(
                f"{where}: Stage 1 runs the frozen stage1_anchor_model (ridge) only, "
                f"got model family {family!r}"
            )
    elif ("b" if family == ORDINAL_B_FAMILY else "a") != arm:
        raise ExpCProtocolError(
            f"{where}: model family {family!r} does not belong to the arm being run ({arm!r})"
        )

    model = regressors._ordinal_model(family, candidate.params(), seed=0)

    if family.startswith(ORDINAL_A_PREFIX):
        base_family = family[len(ORDINAL_A_PREFIX):]
        if model.base_family != base_family:
            raise ExpCProtocolError(
                f"{where}: the wrapper's base_family {model.base_family!r} disagrees with the "
                f"candidate's family {family!r}"
            )
        active_model_family = dict(candidate.active or ()).get("model_family")
        if active_model_family != base_family:
            raise ExpCProtocolError(
                f"{where}: active.model_family={active_model_family!r} does not match the "
                f"wrapper's base_family {base_family!r}"
            )
        base_params = dict(model.base_params)
        if arm == "stage1":
            anchor = _stage1_anchor(config, candidate)
            if base_params != {"alpha": anchor}:
                raise ExpCProtocolError(
                    f"{where}: Stage 1 is frozen at stage1_anchor_ridge_alpha={anchor!r}, "
                    f"got base_params={base_params!r}"
                )
        elif base_params not in regressors.enumerate_grid(base_family, config.model_grid):
            raise ExpCProtocolError(
                f"{where}: base_params={base_params!r} is not a member of the frozen "
                f"{base_family} grid"
            )
        if tuple(model.quantiles) != tuple(exp_c_cfg.cutpoint_quantiles):
            raise ExpCProtocolError(
                f"{where}: cutpoint_quantiles={tuple(model.quantiles)!r} != the frozen "
                f"{tuple(exp_c_cfg.cutpoint_quantiles)!r}"
            )
        if model.min_separation != exp_c_cfg.cutpoint_min_separation:
            raise ExpCProtocolError(
                f"{where}: cutpoint_min_separation={model.min_separation!r} != the frozen "
                f"{exp_c_cfg.cutpoint_min_separation!r}"
            )
        expected_weighted = base_family not in exp_c_cfg.class_weight_unsupported_families
        if bool(model.weighted) != expected_weighted:
            raise ExpCProtocolError(
                f"{where}: weighted={model.weighted!r} contradicts the frozen "
                f"class_weight_unsupported_families={exp_c_cfg.class_weight_unsupported_families!r}"
            )
        return

    if model.C not in exp_c_cfg.proportional_odds_c_grid:
        raise ExpCProtocolError(
            f"{where}: C={model.C!r} is not in the frozen proportional_odds_c_grid "
            f"{exp_c_cfg.proportional_odds_c_grid!r}"
        )
    if model.impl != exp_c_cfg.proportional_odds_impl:
        raise ExpCProtocolError(
            f"{where}: implementation tag {model.impl!r} != the frozen "
            f"proportional_odds_impl {exp_c_cfg.proportional_odds_impl!r}"
        )
    if model.max_iter != FRANK_HALL_MAX_ITER:
        raise ExpCProtocolError(
            f"{where}: max_iter={model.max_iter!r} != the recorded solver bound "
            f"{FRANK_HALL_MAX_ITER}"
        )


def _before_fit_c(config, arm: str):
    """The `before_fit` hook: THREE checks before every single fit, in this order --
    (1) the milestone-6 protocol freeze on the `active` record, (2) Exp C's own authorization
    of the estimator about to be built, (3) the arm's `active`-completeness contract."""

    def before_fit(candidate):
        active = dict(candidate.active)
        protocol_freeze_guard(config, active=active)
        assert_exp_c_fit_authorized(candidate, config, arm=arm)
        if arm == "b":
            require_complete_active_c(active)
        else:
            require_complete_active(active)

    return before_fit


# ------------------------------------------------- the §2.3 ordinal inner-fold aggregation


def _classes_by_subject(sessions) -> dict[int, np.ndarray]:
    """{subject: the classes of that subject's rows, in canonical spine order} -- the same
    row order the provider and hence `InnerResult.val_predictions` use."""
    out: dict[int, list] = {}
    for s in sessions:
        out.setdefault(int(s["subject"]), []).append(int(s["class_idx"]))
    return {k: np.asarray(v, dtype=int) for k, v in out.items()}


def _as_classes(values) -> np.ndarray:
    """Predicted classes as integers. Both ordinal families already emit exact whole floats
    (searchsorted / argmax), so this rounds rather than truncates purely so a 3 that ever
    arrived as 2.9999999 could not read as class 2 (`ordinal._split_target`'s convention)."""
    return np.rint(np.asarray(values, dtype=float)).astype(int)


def _cell_qwk(cell, classes_by_subject) -> float:
    """One inner cell's QWK, from its STORED first-seed validation predictions.

    Never a re-predict: recomputing after selection would double-fit and can drift for rf/gbm,
    and O-M9-1 fixes the tie-break to the same `InnerResult`s the primary MAE came from.
    """
    y_true, y_pred = [], []
    for subject in sorted(cell.val_predictions):
        predictions = np.asarray(cell.val_predictions[subject], dtype=float)
        truth = classes_by_subject[int(subject)]
        if predictions.shape[0] != truth.shape[0]:
            raise ExpCError(
                f"validation predictions for subject {subject} have {predictions.shape[0]} rows "
                f"but the spine has {truth.shape[0]} — the provider and the spine disagree"
            )
        y_true.append(truth)
        y_pred.append(predictions)
    if not y_true:
        return float("nan")
    return M.quadratic_weighted_kappa(np.concatenate(y_true), _as_classes(np.concatenate(y_pred)))


@dataclass(frozen=True)
class QWKExposure:
    """Undefined-QWK accounting for one search stage.

    `nan_inner_folds` counts distinct validation folds that produced at least one undefined
    candidate QWK. The cell counts retain the candidate-level exposure needed to interpret
    that fold count: one fold can legitimately produce several undefined QWK values because
    undefinedness depends on both its truth and each candidate's predictions.
    """

    nan_inner_folds: frozenset[tuple[int, ...]]
    n_nan_evaluation_cells: int
    n_evaluation_cells: int


def _ordinal_candidate_scores(stage, sessions) -> tuple[list[OrdinalCandidateScore], QWKExposure]:
    """`StageOutcome` -> the ordinal selection scores, aggregated over EVALUABLE inner folds
    only, plus explicit fold- and candidate-cell-level QWK exposure accounting (O-M9-8).

    `stage.candidate_scores` (the harness's plain-mean aggregation) is deliberately ignored
    for the MAE and the variance; only `feature_dimension` is read from it, since that is a
    measured property of the bundle rather than a fold aggregation.
    """
    classes_by_subject = _classes_by_subject(sessions)
    n_candidates = len(stage.candidates)
    n_inner = stage.inner_scores.shape[1]

    scores = []
    nan_inner_folds: set[tuple[int, ...]] = set()
    n_nan_evaluation_cells = 0
    n_evaluation_cells = 0
    for ci, candidate in enumerate(stage.candidates):
        per_fold = stage.inner_scores[ci, :]
        evaluable = np.isfinite(per_fold)
        n_evaluable = int(np.count_nonzero(evaluable))
        if n_evaluable:
            values = per_fold[evaluable]
            class_mae = float(np.mean(values))
            variance = float(np.std(values, ddof=0))   # population std, as in Exp A
        else:
            class_mae = variance = float("nan")

        qwks = []
        for fj in range(n_inner):
            # inner_results is flat, fold-major / candidate-minor (the frozen zip order).
            cell = stage.inner_results[fj * n_candidates + ci]
            if cell.reason is not None:
                continue
            n_evaluation_cells += 1
            kappa = _cell_qwk(cell, classes_by_subject)
            if math.isfinite(kappa):
                qwks.append(kappa)
            else:
                n_nan_evaluation_cells += 1
                nan_inner_folds.add(tuple(sorted(int(s) for s in cell.inner_val)))

        scores.append(
            OrdinalCandidateScore(
                candidate_id=candidate.candidate_id,
                inner_val_class_mae=class_mae,
                inner_val_qwk=float(np.mean(qwks)) if qwks else float("nan"),
                simplicity_rank=SIMPLICITY_RANK[candidate.family],
                feature_dimension=stage.candidate_scores[ci].feature_dimension,
                inner_fold_variance=variance,
                n_evaluable_inner_folds=n_evaluable,
            )
        )
    return scores, QWKExposure(
        nan_inner_folds=frozenset(nan_inner_folds),
        n_nan_evaluation_cells=n_nan_evaluation_cells,
        n_evaluation_cells=n_evaluation_cells,
    )


def _missing_classes_by_inner_fold(fold, classes_by_subject) -> tuple:
    """((val_subjects, missing classes), ...) per inner fold -- computed from the spine and
    the fold alone (a pure function of the inner-TRAINING rows, exactly like the harness's
    viability predicate), so naming the cause never consults a held-out label."""
    out = []
    for inner in fold.inner_folds:
        present: set[int] = set()
        for subject in sorted(inner.train_subjects):
            present.update(classes_by_subject.get(int(subject), np.empty(0, dtype=int)).tolist())
        missing = tuple(c for c in harness.ORDINAL_CLASSES if c not in present)
        out.append((tuple(sorted(inner.val_subjects)), missing))
    return tuple(out)


def _n_single_class_truth_inner_val(fold, classes_by_subject) -> int:
    """How many of this outer fold's inner-validation sets have a single distinct true class
    (O-M9-8's exposure counter). Candidate-independent by construction: it reads the spine and
    the fold, never a prediction, so it cannot depend on which model was fit."""
    n = 0
    for inner in fold.inner_folds:
        truth = [c for s in sorted(inner.val_subjects) for c in classes_by_subject[int(s)].tolist()]
        if truth and len(set(truth)) == 1:
            n += 1
    return n


def _reason_counts(stage) -> dict:
    return dict(Counter(ir.reason for ir in stage.inner_results if ir.reason is not None))


def _select_ordinal(stage, sessions, fold, stage_label):
    """Score -> `select_candidate_ordinal` -> the winning Candidate. Every Exp C selection
    routes through `selection.py`; nothing here re-implements a tie-break."""
    scores, qwk_exposure = _ordinal_candidate_scores(stage, sessions)
    try:
        winner_score = select_candidate_ordinal(scores)
    except SelectionError as err:
        # The frozen "if ALL configs are non-evaluable the fold contributes no ordinal score"
        # case (`:793-797`). Name the fold and the missing classes rather than letting the
        # generic "no comparable candidate" message stand alone (the Exp B trap-5 lesson).
        per_inner = _missing_classes_by_inner_fold(fold, _classes_by_subject(sessions))
        union = sorted({c for _, missing in per_inner for c in missing})
        raise ExpCError(
            f"Exp C {stage_label}, test_subject={fold.test_subject}: no candidate was "
            f"comparable (no evaluable inner fold, or a non-finite class-unit MAE) — this fold "
            f"contributes no ordinal score. missing_classes={set(union)}; "
            f"per_inner_fold(val_subjects, missing)={per_inner}"
        ) from err
    by_id = {c.candidate_id: c for c in stage.candidates}
    return by_id[winner_score.candidate_id], winner_score, qwk_exposure


# ----------------------------------------------------------------------------- staged run


@dataclass
class ExpCArmResult:
    """One arm's outcome on one outer fold."""

    arm: str
    selected_feature_key: tuple
    selected_family: str
    selected_params: dict
    test_predictions: np.ndarray          # first-seed predicted classes (float, in {0..4})
    seed_outcomes: list
    final_fits: list
    n_evaluable_inner_folds: int          # of the SELECTED candidate (§2.3, published per fold)
    viability_reason_counts: dict


@dataclass
class ExpCFoldResult:
    test_subject: int
    stage1_feature_key: tuple
    stage1_selected_params: dict          # always the frozen ridge anchor; recorded, not assumed
    stage1_n_evaluable_inner_folds: int
    stage1_viability_reason_counts: dict
    arm_a: ExpCArmResult
    arm_b: ExpCArmResult
    test_classes: np.ndarray              # the held-out subject's true S0-S4 classes
    test_targets: np.ndarray              # ... and its L = -Δm%
    test_session_idx: np.ndarray
    n_single_class_truth_inner_val: int
    n_qwk_nan_inner: int                  # distinct inner folds with >=1 undefined QWK
    n_qwk_nan_inner_evaluation_cells: int = 0
    n_qwk_inner_evaluation_cells: int = 0
    reason: str | None = None             # non-None: this fold contributes no out-of-fold rows

    def arm_result(self, arm: str) -> ExpCArmResult:
        return self.arm_a if arm == "a" else self.arm_b


@dataclass
class _ExpCFoldTrace:
    """The full inner-search trace retained only long enough for regression tests to audit it."""

    result: ExpCFoldResult
    stage1: harness.StageOutcome
    stage2_by_arm: dict[str, harness.StageOutcome]


def _run_single_fold_c_trace(config, band, sessions, store_dir, fold, seeds) -> _ExpCFoldTrace:
    """Run one fold and retain both search stages for the load-bearing mutation tests.

    Production calls `_run_single_fold_c`, which returns only `trace.result`; no additional
    search is run and no trace is serialized into normal result artifacts.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        provider = OrdinalFeatures(band, sessions, store_dir, config)
        anchor = (
            config.search_10ghz if band == "10ghz" else config.search_77ghz
        ).stage1_anchor_ridge_alpha
        classes_by_subject = _classes_by_subject(sessions)

        stage1 = harness._score_candidates_on_fold(
            stage1_candidates_c(config, band, anchor), fold, seeds,
            _before_fit_c(config, "stage1"), provider.data_for, score_fn=ordinal_class_mae_score,
        )
        w1, w1_score, stage1_qwk = _select_ordinal(stage1, sessions, fold, "Stage 1")

        arms = {}
        stage2_by_arm = {}
        nan_inner_folds = set(stage1_qwk.nan_inner_folds)
        n_nan_evaluation_cells = stage1_qwk.n_nan_evaluation_cells
        n_evaluation_cells = stage1_qwk.n_evaluation_cells
        for arm, build_candidates in (("a", stage2_candidates_a), ("b", stage2_candidates_b)):
            before_fit = _before_fit_c(config, arm)
            stage2 = harness._score_candidates_on_fold(
                build_candidates(config, band, w1.feature_key, dict(w1.active)), fold, seeds,
                before_fit, provider.data_for, score_fn=ordinal_class_mae_score,
            )
            stage2_by_arm[arm] = stage2
            winner, winner_score, qwk_exposure = _select_ordinal(
                stage2, sessions, fold, f"Stage 2 arm ({arm})"
            )
            nan_inner_folds.update(qwk_exposure.nan_inner_folds)
            n_nan_evaluation_cells += qwk_exposure.n_nan_evaluation_cells
            n_evaluation_cells += qwk_exposure.n_evaluation_cells
            final_fits, _, test_pred, _, seed_outcomes = harness._final_refit(
                winner, fold, seeds, before_fit, provider.data_for,
                score_fn=ordinal_class_mae_score,
            )
            arms[arm] = ExpCArmResult(
                arm=arm,
                selected_feature_key=winner.feature_key,
                selected_family=winner.family,
                selected_params=winner.params(),
                test_predictions=test_pred,
                seed_outcomes=seed_outcomes,
                final_fits=final_fits,
                n_evaluable_inner_folds=winner_score.n_evaluable_inner_folds,
                viability_reason_counts=_reason_counts(stage2),
            )

        test_rows = np.isin(provider.subjects, [fold.test_subject])
        return _ExpCFoldTrace(
            result=ExpCFoldResult(
                test_subject=fold.test_subject,
                stage1_feature_key=w1.feature_key,
                stage1_selected_params=w1.params(),
                stage1_n_evaluable_inner_folds=w1_score.n_evaluable_inner_folds,
                stage1_viability_reason_counts=_reason_counts(stage1),
                arm_a=arms["a"],
                arm_b=arms["b"],
                test_classes=provider.classes[test_rows],
                test_targets=provider.loss_l[test_rows],
                test_session_idx=provider.session_idx[test_rows],
                n_single_class_truth_inner_val=_n_single_class_truth_inner_val(
                    fold, classes_by_subject
                ),
                n_qwk_nan_inner=len(nan_inner_folds),
                n_qwk_nan_inner_evaluation_cells=n_nan_evaluation_cells,
                n_qwk_inner_evaluation_cells=n_evaluation_cells,
                reason=None,
            ),
            stage1=stage1,
            stage2_by_arm=stage2_by_arm,
        )


def _run_single_fold_c(config, band, sessions, store_dir, fold, seeds) -> ExpCFoldResult:
    """Run ONE outer fold end to end.

    Top-level + picklable so it can run in a worker process; builds its own provider and pins
    single-threaded math, mirroring Exp A/B. The internal trace is discarded after returning
    the ordinary result.
    """
    return _run_single_fold_c_trace(config, band, sessions, store_dir, fold, seeds).result


def run_exp_c(config, band, sessions, store_dir, *, seeds, n_workers=1) -> list[ExpCFoldResult]:
    """The selectable outer folds, run through the shared `fold_parallel` pool and reassembled
    in canonical test-subject order (bit-identical to the serial run). Folds come only from
    `splits.py`."""
    folds = [f for f in harness.nested_loso_splits(evaluable_subjects_c(sessions)) if f.selectable]
    tasks = [(config, band, sessions, store_dir, fold, seeds) for fold in folds]
    results = fold_parallel.run_folds_parallel(_run_single_fold_c, tasks, n_workers, "exp_c")
    results.sort(key=lambda r: r.test_subject)
    return results


# ------------------------------------------------------------------------------- reporting


def _oof_matrix_c(results, arm):
    """(subjects, session_idx, true classes, per-seed predicted classes) across folds, for one
    arm. Deterministic winners carry one SeedOutcome; the 5-seed protocol replicates it so the
    seed axis is consistent (Exp A's convention)."""
    usable = [r for r in results if r.reason is None]
    n_seeds = max((len(r.arm_result(arm).seed_outcomes) for r in usable), default=0)
    subjects, session_idx, y_true = [], [], []
    per_seed = [[] for _ in range(n_seeds)]
    for r in usable:
        arm_result = r.arm_result(arm)
        subjects += [r.test_subject] * len(r.test_classes)
        session_idx += r.test_session_idx.tolist()
        y_true += r.test_classes.tolist()
        for k in range(n_seeds):
            outcome = arm_result.seed_outcomes[min(k, len(arm_result.seed_outcomes) - 1)]
            per_seed[k] += _as_classes(outcome.test_predictions).tolist()
    return (
        np.array(subjects, dtype=int),
        np.array(session_idx, dtype=int),
        np.array(y_true, dtype=float),
        np.array(per_seed, dtype=float).reshape(n_seeds, len(y_true)),
    )


def _ci_dict(c: M.BootstrapCI) -> dict:
    return {"point": c.point, "low": c.low, "high": c.high, "method": c.method,
            "n_eval": c.n_eval, "n_skipped": c.n_skipped, "unreliable": c.unreliable}


def _nan_ci_dict() -> dict:
    return {"point": float("nan"), "low": float("nan"), "high": float("nan"), "method": "none",
            "n_eval": 0, "n_skipped": 0, "unreliable": True}


def _per_subject_class_mae(subjects, y_true, pred_by_seed) -> dict:
    """Seed-averaged per-subject class-unit MAE -- DESCRIPTIVE (the frozen "per-subject
    distribution is always shown"), never the headline: the headline class-unit MAE is pooled,
    with the pooled seed-collapse."""
    out = {}
    for s in sorted(set(subjects.tolist())):
        rows = subjects == s
        per_seed = [
            float(np.mean(np.abs(y_true[rows] - pred_by_seed[k, rows])))
            for k in range(pred_by_seed.shape[0])
        ]
        out[int(s)] = float(np.mean(per_seed))
    return out


def summarize_exp_c(results, config) -> dict:
    """Both arms' ordinal headline metrics with subject-cluster CIs, the LOSO confusion
    matrix, the selection-stability table, and the O-M9-8 exposure counters.

    All three metrics are pooled/nonlinear (`:1199-1204`), so every CI uses the pooled
    seed-collapse: within each subject resample the metric is recomputed per seed and averaged
    across seeds. QWK can be undefined inside a resample; those replicates are skipped and
    counted by the shared bootstrap machinery, and the CI is flagged unreliable past the frozen
    5% threshold. There is deliberately NO baseline comparison (plan §5 trap 16).
    """
    usable = [r for r in results if r.reason is None]
    stats = config.stats
    seed = config.run.seed

    arms: dict = {}
    outer_qwk_nan_folds: set[int] = set()
    outer_qwk_nan_evaluation_cells = 0
    outer_qwk_evaluation_cells = 0
    cohort = {}                # the out-of-fold shape (identical across arms), for the header
    for arm in ARMS:
        subjects, _session_idx, y_true, pred_by_seed = _oof_matrix_c(usable, arm)
        n_seeds = pred_by_seed.shape[0]
        # n_eval_subjects and n_rows genuinely are arm-invariant, so they belong in the
        # header. n_seeds is NOT: arm a realizes 5 seeds only on the folds where it selects
        # a seed-sensitive family (rf/gbm), so its realized count differs from arm b's.
        # Writing it here made the header field silently arm-b-only — it now lives per arm.
        cohort = {"n_eval_subjects": len(set(subjects.tolist())), "n_rows": int(len(y_true))}

        if len(y_true) and n_seeds:
            cis = {
                name: _ci_dict(M.subject_cluster_bootstrap_pooled(
                    subjects, y_true, pred_by_seed, metric_fn,
                    b=stats.bootstrap_b, rng_seed=seed + _rng_offset(arm, name),
                    skip_threshold_pct=stats.undefined_metric_skip_threshold_pct,
                ))
                for name, metric_fn in (
                    ("class_unit_mae", M.class_unit_mae),
                    ("adjacent_accuracy", M.adjacent_accuracy),
                    ("quadratic_weighted_kappa", M.quadratic_weighted_kappa),
                )
            }
            confusion = np.mean(
                [M.confusion_counts(y_true, pred_by_seed[k]) for k in range(n_seeds)], axis=0
            )
            per_subject = _per_subject_class_mae(subjects, y_true, pred_by_seed)
        else:
            cis = {name: _nan_ci_dict() for name in
                   ("class_unit_mae", "adjacent_accuracy", "quadratic_weighted_kappa")}
            confusion = np.zeros((5, 5))
            per_subject = {}

        for r in usable:
            for outcome in r.arm_result(arm).seed_outcomes:
                outer_qwk_evaluation_cells += 1
                kappa = M.quadratic_weighted_kappa(
                    r.test_classes, _as_classes(outcome.test_predictions)
                )
                if not math.isfinite(kappa):
                    outer_qwk_nan_evaluation_cells += 1
                    outer_qwk_nan_folds.add(int(r.test_subject))

        arm_results = [r.arm_result(arm) for r in usable]
        arms[arm] = {
            "n_seeds": int(n_seeds),   # REALIZED seed count for this arm — see the note above
            **cis,
            "per_subject_class_mae": per_subject,
            "confusion_matrix_mean_over_seeds": confusion.tolist(),
            "selection_frequency": exp_a._selection_frequency(arm_results),
            "per_fold": [
                {
                    "test_subject": r.test_subject,
                    "selected_feature_key": str(r.arm_result(arm).selected_feature_key),
                    "selected_family": r.arm_result(arm).selected_family,
                    "n_evaluable_inner_folds": r.arm_result(arm).n_evaluable_inner_folds,
                    "viability_reason_counts": r.arm_result(arm).viability_reason_counts,
                }
                for r in usable
            ],
        }

    return {
        "conditional_exploratory": True,
        "task": "ordinal_5class_s0_s4",
        **cohort,
        "arms": arms,
        "qwk_undefinedness": {
            # O-M9-8: how much the (8a) trigger actually changed, reported rather than
            # assumed negligible. "val folds" at the outer level = the held-out test rows.
            "inner": {
                "n_single_class_truth_val_folds": sum(r.n_single_class_truth_inner_val for r in usable),
                "n_qwk_nan": sum(r.n_qwk_nan_inner for r in usable),
                "n_qwk_nan_evaluation_cells": sum(
                    r.n_qwk_nan_inner_evaluation_cells for r in usable
                ),
                "n_qwk_evaluation_cells": sum(r.n_qwk_inner_evaluation_cells for r in usable),
                "evaluation_cell": "stage_x_candidate_x_inner_fold",
            },
            "outer": {
                "n_single_class_truth_val_folds": sum(
                    1 for r in usable
                    if r.test_classes.size and len(set(r.test_classes.tolist())) == 1
                ),
                "n_qwk_nan": len(outer_qwk_nan_folds),
                "n_qwk_nan_evaluation_cells": outer_qwk_nan_evaluation_cells,
                "n_qwk_evaluation_cells": outer_qwk_evaluation_cells,
                "evaluation_cell": "arm_x_realized_seed_x_outer_fold",
            },
        },
        "stage1": {
            "per_fold": [
                {
                    "test_subject": r.test_subject,
                    "selected_feature_key": str(r.stage1_feature_key),
                    "n_evaluable_inner_folds": r.stage1_n_evaluable_inner_folds,
                    "viability_reason_counts": r.stage1_viability_reason_counts,
                }
                for r in usable
            ],
        },
    }


def _write_predictions_csv(results, out_path) -> None:
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "arm", "seed", "session_idx", "y_class_true", "y_class_pred"])
        for r in results:
            if r.reason is not None:
                continue
            for arm in ARMS:
                for outcome in r.arm_result(arm).seed_outcomes:
                    for si, yt, yp in zip(
                        r.test_session_idx, r.test_classes, outcome.test_predictions, strict=True
                    ):
                        w.writerow([r.test_subject, arm, outcome.seed, int(si), int(yt), int(yp)])


def _write_selection_table_csv(results, out_path) -> None:
    """One row per (fold, stage/arm), carrying the §2.3 evaluability record so a fold selected
    on fewer than the full inner folds is visible in the artifact, never silent."""
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test_subject", "arm", "feature_key", "family", "params",
                    "n_evaluable_inner_folds", "viability_reason_counts", "reason"])
        for r in results:
            w.writerow([r.test_subject, "stage1", r.stage1_feature_key, "ord_a_ridge",
                        r.stage1_selected_params, r.stage1_n_evaluable_inner_folds,
                        r.stage1_viability_reason_counts, r.reason])
            for arm in ARMS:
                a = r.arm_result(arm)
                w.writerow([r.test_subject, arm, a.selected_feature_key, a.selected_family,
                            a.selected_params, a.n_evaluable_inner_folds,
                            a.viability_reason_counts, r.reason])


def _write_confusion_csv(summary, out_path) -> None:
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "true_class"] + [f"pred_{j}" for j in range(5)])
        for arm in ARMS:
            matrix = summary["arms"][arm]["confusion_matrix_mean_over_seeds"]
            for i, row in enumerate(matrix):
                w.writerow([arm, i] + list(row))


def write_exp_c_reports(results, summary, out_dir, band) -> dict:
    """metrics_exp_c_{band}.json, predictions_{band}.csv, selection_table_{band}.csv,
    confusion_{band}.csv + .png -- only ever called in `--full-cohort` mode."""
    import matplotlib
    matplotlib.use("Agg")  # headless: no display
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    metrics_path = out_dir / f"metrics_exp_c_{band}.json"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics"] = metrics_path

    pred_path = out_dir / f"predictions_{band}.csv"
    _write_predictions_csv(results, pred_path)
    paths["predictions"] = pred_path

    sel_path = out_dir / f"selection_table_{band}.csv"
    _write_selection_table_csv(results, sel_path)
    paths["selection_table"] = sel_path

    confusion_path = out_dir / f"confusion_{band}.csv"
    _write_confusion_csv(summary, confusion_path)
    paths["confusion"] = confusion_path

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, arm in zip(axes, ARMS, strict=True):
        matrix = np.array(summary["arms"][arm]["confusion_matrix_mean_over_seeds"], dtype=float)
        ax.imshow(matrix, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(5), [f"S{j}" for j in range(5)])
        ax.set_yticks(range(5), [f"S{i}" for i in range(5)])
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(f"Exp C arm ({arm}) — {band}")
    figure_path = out_dir / f"confusion_{band}.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=120)
    plt.close(fig)
    paths["confusion_figure"] = figure_path
    return paths


def _assert_mechanism_ok_c(results, sessions) -> None:
    """Structural checks that reveal no performance: fold-role disjointness, fit-audit
    coverage, S0 PRESENT (unlike Exp B — all five classes are the task here), and predicted
    classes on the frozen {0..4} grid."""
    assert any(int(s["session_idx"]) == 0 for s in sessions), (
        "Exp C's spine must contain S0 rows — S0 is the lowest of the five ordered classes"
    )
    folds = harness.nested_loso_splits(evaluable_subjects_c(sessions))
    for fold in folds:
        if not fold.selectable:
            continue
        assert fold.test_subject not in fold.train_subjects
        for inner in fold.inner_folds:
            assert inner.train_subjects.isdisjoint(inner.val_subjects)
            assert fold.test_subject not in inner.train_subjects
            assert fold.test_subject not in inner.val_subjects
    for r in results:
        if r.reason is not None:
            continue
        for arm in ARMS:
            arm_result = r.arm_result(arm)
            for record in arm_result.final_fits:
                assert r.test_subject not in record.subjects
            for outcome in arm_result.seed_outcomes:
                assert set(np.unique(outcome.test_predictions)).issubset(set(range(5)))


def run_and_report_c(config, band, sessions, store_dir, run_dir, *, mode, analysis_commit,
                     n_workers=1) -> dict:
    """validate_store (fail-closed, commit-match) -> run_exp_c -> mechanism assertions ->
    smoke (a structural run-log ONLY: no class metric, no confusion matrix, no selected
    configuration) or the full reporting set."""
    store_mod.validate_store(
        band, store_dir, exp_a.expected_fingerprints(config, band, sessions),
        analysis_commit=analysis_commit,
    )
    results = run_exp_c(config, band, sessions, store_dir, seeds=config.run.seed_set,
                        n_workers=n_workers)
    _assert_mechanism_ok_c(results, sessions)   # structural, not performance

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        log = run_dir / f"run_log_{band}.json"
        log.write_text(json.dumps({
            "stage": "exp-c-smoke", "band": band, "mode": "mechanism-only",
            "n_folds": len(results), "n_sessions": len(sessions),
            "note": "performance values suppressed -- mechanism-only smoke",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"run_log": log}

    summary = summarize_exp_c(results, config)
    return write_exp_c_reports(results, summary, run_dir, band)
