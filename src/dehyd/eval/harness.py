"""The fit-on-train-only nested-LOSO engine (sklearn path).

ONE generic candidate engine serves (a) Experiment A's staged search over store-backed
features, (b) the frozen leakage suite's reference shim (ridge over an alpha grid on a
fixed Dataset), and (c) Experiment B's session-residualized search, via one optional
keyword-only `score_fn` hook (`None` -> the original, unchanged `subject_balanced_mae`
scoring; a supplied `score_fn(subjects, y_true, y_pred, session_idx)` overrides it — see
`_score`). Milestone 9 adds (d) Experiment C's ordinal search on the same engine: its
estimators read a 2-column `y = [L, class]` and its non-evaluable cells are marked through
the existing `_viability_reason` mechanism (see there). It:

  * consumes folds ONLY from `eval/splits.py` (subject-level leakage is structural);
  * routes every tie-break through `eval/selection.py::select_candidate` (never inline);
  * calls a `before_fit(candidate)` hook immediately before every estimator `.fit()`
    (Exp A passes `protocol_freeze_guard(config, active=...)`);
  * emits per-fold `FitRecord`s (every fitted quantity + the subject set it came from),
    consumed by `fit_audit`.

Determinism: all numeric work runs under `threadpool_limits(1)`; ridge/svr/knn are
deterministic and rf/gbm take `random_state=seed`, so two runs are bit-identical
per-machine.

Assembly vs execution order. Candidates are *executed* candidate-major so a store-backed
provider can hold one candidate's raw tensors at a time (memory), but `inner_results` and
the `(n_candidates, n_inner_folds)` `inner_scores` matrix are *assembled* fold-major /
candidate-minor — the flat ordering the frozen `zip(strict=True)` comparison requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from .metrics import subject_balanced_mae
from .selection import CandidateScore, select_candidate
from .splits import nested_loso_splits
from ..models.ordinal import N_CLASSES
from ..models.regressors import (
    SEED_SENSITIVE,
    SIMPLICITY_RANK,
    build_estimator,
    fit_pipeline,
    fitted_state_params,
)

# The exact set of `active` protocol keys the guard requires per band (C5). The guard
# validates only present keys, so the harness enforces completeness fail-closed before use.
REQUIRED_ACTIVE_KEYS = {
    "10ghz": frozenset({"band", "reduction", "channel", "tiling", "log_branch", "range_gate_m", "model_family"}),
    "77ghz": frozenset({"band", "reduction", "channel", "gate_m", "tiling", "log_branch", "model_family"}),
}

# The classes an Exp C inner-training set must cover, as a CONSTANT (S0-S4). See
# `_viability_reason` for why this may never be replaced by a data-derived class set.
ORDINAL_CLASSES = tuple(range(N_CLASSES))   # (0, 1, 2, 3, 4)


class HarnessError(ValueError):
    """A malformed candidate, an incomplete `active` record, or an unusable fold."""


# --------------------------------------------------------------------- data types


@dataclass
class Dataset:
    """Session-level data: one row per (subject, session). (The shim re-exports this.)"""

    subjects: np.ndarray   # (n_rows,) subject id per row
    features: np.ndarray   # (n_rows, n_features)
    targets: np.ndarray    # (n_rows,)

    def rows_for(self, subject_set) -> np.ndarray:
        return np.isin(self.subjects, sorted(subject_set))

    def subject_ids(self) -> list[int]:
        return sorted(set(self.subjects.tolist()))


@dataclass(frozen=True)
class Candidate:
    """One configuration to fit. `model_params`/`active` are tuples of pairs (hashable)."""

    candidate_id: str
    family: str
    model_params: tuple                       # ((name, value), ...) sorted
    feature_key: tuple | None = None          # None => the provider's fixed features
    active: tuple | None = None               # ((key, value), ...) protocol record, or None

    def params(self) -> dict:
        return dict(self.model_params)


@dataclass
class FeatureBundle:
    """What a data provider returns for one (candidate, train_subjects): the feature
    matrix (computed with any train-only fitted transform, e.g. tuned-ε) plus the extra
    fitted quantities to record. `extra_fits` are (quantity, params) pairs; the harness
    adds the role and subject set."""

    subjects: np.ndarray
    X: np.ndarray
    y: np.ndarray
    extra_fits: tuple = ()   # ((quantity, {str: np.ndarray}), ...)
    session_idx: np.ndarray | None = None
        # Per-row session index, aligned to subjects/X/y. Only objectives that group BY
        # SESSION (Exp B's equal-session residual MAE) need it. Appended AFTER extra_fits
        # with a None default so every existing positional construction (exp_a.py,
        # fixed_feature_provider below, test_harness.py) stays valid unchanged.


@dataclass
class FitRecord:
    quantity: str
    role: str                 # "inner_train" | "outer_train"
    subjects: frozenset
    params: dict


@dataclass
class InnerResult:
    inner_train: frozenset
    inner_val: frozenset
    candidate_id: str
    score: float
    val_predictions: dict
    fits: list
    reason: str | None = None   # non-None => non-evaluable for this fold (viability, C6/C21)


@dataclass
class SeedOutcome:
    seed: int
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    test_score: float


@dataclass
class FoldResult:
    test_subject: int
    train_subjects: frozenset
    selected: Candidate | None
    inner_scores: np.ndarray             # (n_candidates, n_inner_folds)
    inner_results: list                  # flat, fold-major / candidate-minor
    final_fits: list
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    test_score: float
    seed_outcomes: list = field(default_factory=list)


@dataclass
class StageOutcome:
    candidates: list
    inner_scores: np.ndarray
    inner_results: list
    candidate_scores: list               # list[CandidateScore], one per candidate


# --------------------------------------------------------------- provider helpers


def fixed_feature_provider(dataset: Dataset):
    """Provider for the shim / fixed-feature case: same features regardless of the fit's
    training subjects, no extra fitted quantities."""

    def provider(candidate: Candidate, train_subjects) -> FeatureBundle:
        return FeatureBundle(dataset.subjects, dataset.features, dataset.targets, extra_fits=())

    return provider


def require_complete_active(active: dict | None) -> None:
    """Fail closed if a per-fit protocol record omits a required band key (C5).

    `protocol_freeze_guard` validates only keys that are present, so an under-populated
    `active` (e.g. {"band": "10ghz"}) would pass its whitelist check silently. The harness
    demands the exact band key set before any fit that carries an `active` record.
    """
    if active is None:
        return
    band = active.get("band")
    required = REQUIRED_ACTIVE_KEYS.get(band)
    if required is None:
        raise HarnessError(f"active.band must be '10ghz' or '77ghz', got {band!r}")
    present = set(active)
    missing = required - present
    extra = present - required
    if missing or extra:
        raise HarnessError(
            f"active record for band {band!r} has wrong keys: missing={sorted(missing)}, "
            f"unexpected={sorted(extra)} (required exactly {sorted(required)})"
        )


def tuned_epsilons(prelog_by_subject, train_subjects, *, k: float = 0.1, fallback: float = 1e-6) -> dict:
    """The fold-local tuned-ε per order (the one genuinely fitted WST quantity, train-only).

    For orders o in {1, 2}: ε_o = k · scale_o, where scale_o = median over TRAINING subjects
    of (mean over that subject's eligible training sessions of the stored per-session pre-log
    scale for order o). Subject-balanced and computed from `train_subjects` only, so it is
    train-only at every CV level. A non-finite / non-positive aggregate falls back to
    `fallback` (1e-6). Order 0 stays linear (never logged), so no ε is produced for it.

    `prelog_by_subject`: {subject_id: [(v0, v1, v2), ...]} — one pre-log tuple per that
    subject's eligible session, already selected for the active tiling (and fusion, 77 GHz).
    """
    train = sorted(train_subjects)
    eps: dict = {}
    for o in (1, 2):
        per_subject_means = []
        for s in train:
            vals = [tpl[o] for tpl in prelog_by_subject.get(s, []) if np.isfinite(tpl[o])]
            if vals:
                per_subject_means.append(float(np.mean(vals)))
        scale = float(np.median(per_subject_means)) if per_subject_means else float("nan")
        candidate_eps = k * scale
        eps[o] = candidate_eps if (np.isfinite(candidate_eps) and candidate_eps > 0.0) else fallback
    return eps


def subject_row_multiplicity(subjects, subject_multiplicity) -> np.ndarray | None:
    """Expand a {subject: m_s} map to one integer copy-count per ROW, aligned to `subjects`.

    The milestone-10 robustness bootstrap resamples subjects, so multiplicity is naturally
    per-subject; everything downstream of here works per-row. `None` in, `None` out, so a
    caller can pass it straight through without branching.
    """
    if subject_multiplicity is None:
        return None
    return np.array(
        [int(subject_multiplicity.get(int(s), 1)) for s in np.asarray(subjects)], dtype=int
    )


def _effective_rows(train_rows, row_multiplicity) -> int:
    """How many rows a fit actually sees: the plain count, or the duplicated count."""
    if row_multiplicity is None:
        return int(np.count_nonzero(train_rows))
    return int(np.asarray(row_multiplicity)[train_rows].sum())


def _weighted_subject_balanced_mae(subjects, y_true, y_pred, subject_weights) -> float:
    """`subject_balanced_mae` with each DISTINCT subject weighted by its copy count.

    Plan §2.4: "inner-validation objectives and outer replicate summaries weight each subject
    by m_s". Repeating rows would not achieve this — the metric is subject-BALANCED, so
    duplicating a subject's rows leaves its own mean unchanged and the outer average over
    distinct subjects untouched. The weight has to enter at the across-subject average, which
    is what a duplicated cohort's subject-balanced mean actually is.
    """
    subjects = np.asarray(subjects)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ordered = sorted(set(subjects.tolist()))
    per_subject, weights = [], []
    for s in ordered:
        mask = subjects == s
        per_subject.append(np.abs(y_true[mask] - y_pred[mask]).mean())
        # Multiplicity is per SUBJECT, so every row of a subject carries the same count.
        weights.append(float(np.asarray(subject_weights)[mask][0]))
    return float(np.average(per_subject, weights=weights))


def _viability_reason(candidate: Candidate, bundle: FeatureBundle, train_rows,
                      row_multiplicity=None) -> str | None:
    """Explicit, enumerated PRE-FIT viability predicates (C6/C21). Returns a reason code
    if the candidate cannot be fit on these training rows, else None. NOT a catch-all: any
    unexpected fit/predict exception is left to propagate loudly.

    (a) KNN's k against the training row count, keyed on the PARAMETER name rather than the
        family name, so `knn` and Experiment C's `ord_a_knn` — which carries the identical
        `n_neighbors` grid inside the thresholding wrapper — are one rule with one reason
        string. The comparison stays strictly `>` (k == n_train_rows is viable).

    (b) Experiment C's frozen fold-viability rule (`implementation_plan.md:793-801`): with a
        2-column ordinal y (`y[:, 1]` = the S0-S4 class), the training rows must cover all
        five classes. The required set is the CONSTANT `ORDINAL_CLASSES`, never
        `set(bundle.y[:, 1])`, for two independent reasons:

          (i) the frozen rule is "its inner-training set lacks any of the 5 classes"; a
              bundle-relative predicate would silently stop requiring a class that QC had
              removed cohort-wide, which is a weaker rule than the frozen one;
          (ii) `OrdinalFeatures` mirrors `StoreBackedFeatures`, whose bundles carry ALL
              session rows (the row mask is applied afterwards, in
              `_score_candidates_on_fold`), so `set(bundle.y[:, 1])` would include
              inner-validation and outer-test labels — making which cells are fit at all a
              function of held-out labels.

        Against the constant, the predicate is a pure function of the training rows. It is
        therefore candidate-independent by construction, but is still evaluated and recorded
        per cell, matching "such configs are skipped in ordinal selection (recorded)".
    """
    # Under a bootstrap the knn rule must count the rows the fit ACTUALLY sees. A candidate
    # with k = 15 is non-viable on 10 unique training rows but viable once a drawn subject's
    # rows are duplicated to 17 — and it is the duplicated cohort that gets fit, so using the
    # unique count here would reject candidates the replicate can legitimately evaluate.
    n_train_rows = _effective_rows(train_rows, row_multiplicity)
    params = candidate.params()
    if "n_neighbors" in params:
        k = params["n_neighbors"]
        if k > n_train_rows:
            return f"knn_n_neighbors_{k}_gt_train_rows_{n_train_rows}"
    if bundle.y.ndim == 2:
        # Round, don't truncate: the class rides in a float column (`ordinal._split_target`
        # uses the same rint), so a 3 that arrived as 2.9999999 must not read as class 2.
        present = set(np.rint(bundle.y[train_rows, 1]).astype(int).tolist())
        missing = [c for c in ORDINAL_CLASSES if c not in present]
        if missing:
            return "ordinal_missing_class_" + "_".join(str(c) for c in missing) + "_in_inner_train"
    return None


# --------------------------------------------------------------------- fitting


def _bundle_fits(bundle: FeatureBundle, role: str, subjects) -> list:
    return [FitRecord(q, role, frozenset(subjects), p) for q, p in bundle.extra_fits]


def _score(score_fn, bundle: FeatureBundle, rows, y_pred, row_multiplicity=None) -> float:
    """The one scoring choke point. `score_fn=None` -> the CURRENT, UNCHANGED call to
    `subject_balanced_mae(bundle.subjects[rows], bundle.y[rows], y_pred)`. A supplied
    `score_fn` additionally receives `bundle.session_idx[rows]` (or None if the bundle
    carries none): `score_fn(subjects, y_true, y_pred, session_idx) -> float`.

    `row_multiplicity` (milestone 10) enters differently for the two cases, because the two
    metric shapes need different things (plan §2.4):

      * the built-in subject-balanced MAE is weighted at the across-subject average, since
        repeating a subject's rows cannot change a subject-balanced mean;
      * a supplied `score_fn` (Exp B's equal-session residual MAE, Exp C's ordinal metrics)
        is pooled or ordinal, so its evaluation rows are deterministically REPEATED by m_s
        before it is called — "evaluation rows are deterministically repeated by m_s before
        metric calculation". Repetition is the general rule; weighting is the special case
        that a subject-balanced statistic requires.
    """
    if score_fn is None:
        if bundle.y.ndim != 1:
            # Fail-fast (M9 step 4, plan §5 trap 1). `subject_balanced_mae` is defined on a
            # 1-D target; fed Exp C's 2-column [L, class] y it does not reliably crash —
            # `y_true[rows] - y_pred[rows]` broadcasts for any subject contributing exactly
            # 2 rows and returns a plausible-looking, meaningless float.
            raise HarnessError(
                "score_fn=None scores with subject_balanced_mae, which is defined on a 1-D "
                f"target, but this bundle's y has shape {bundle.y.shape}. A 2-column ordinal "
                "y must be scored by an explicit score_fn (Exp C always supplies one)."
            )
        if row_multiplicity is None:
            return subject_balanced_mae(bundle.subjects[rows], bundle.y[rows], y_pred)
        return _weighted_subject_balanced_mae(
            bundle.subjects[rows], bundle.y[rows], y_pred, np.asarray(row_multiplicity)[rows]
        )
    session_idx = bundle.session_idx[rows] if bundle.session_idx is not None else None
    if row_multiplicity is None:
        return score_fn(bundle.subjects[rows], bundle.y[rows], y_pred, session_idx)
    copies = np.asarray(row_multiplicity)[rows]
    return score_fn(
        np.repeat(bundle.subjects[rows], copies, axis=0),
        np.repeat(bundle.y[rows], copies, axis=0),
        np.repeat(np.asarray(y_pred), copies, axis=0),
        None if session_idx is None else np.repeat(session_idx, copies, axis=0),
    )


def _fit_once(candidate, X, y, train_rows, seed, before_fit, row_multiplicity=None):
    if before_fit is not None:
        before_fit(candidate)
    pipe = build_estimator(candidate.family, candidate.params(), seed=seed)
    train_multiplicity = None if row_multiplicity is None else np.asarray(row_multiplicity)[train_rows]
    # `fit_pipeline(row_multiplicity=None)` executes the literal `pipe.fit(...)` this line
    # used to be, so Experiments A-D are byte-identical through this path.
    fit_pipeline(pipe, X[train_rows], y[train_rows], row_multiplicity=train_multiplicity)
    return pipe


def _multiplicity_audit(train_rows, row_multiplicity, subjects) -> dict:
    """The extra audit fields a bootstrap fit must carry (plan §2.4).

    Returned EMPTY when nothing was resampled, so an ordinary fit record keeps exactly the
    keys milestones 7-9 wrote and "byte-neutral by default" holds of the audit too.
    """
    if row_multiplicity is None:
        return {}
    copies = np.asarray(row_multiplicity)[train_rows]
    fitted_subjects = np.asarray(subjects)[train_rows]
    distinct = sorted(set(fitted_subjects.tolist()))
    return {
        "multiplicity_subjects": np.array(distinct, dtype=np.int64),
        "multiplicity_counts": np.array(
            [int(copies[fitted_subjects == s][0]) for s in distinct], dtype=np.int64
        ),
        "effective_weighted_row_count": np.array([float(copies.sum())], dtype=float),
        # "row_duplication" is the only mode A-M10-8 leaves; recorded explicitly so an
        # artifact says how it was weighted instead of leaving it to be inferred.
        "weighting_mode": np.frombuffer(b"row_duplication", dtype=np.uint8),
    }


def _model_and_scaler_fits(pipe, candidate, role, subjects, audit=None) -> tuple[FitRecord, FitRecord]:
    audit = audit or {}
    scaler = pipe.named_steps["scaler"]
    scaler_fit = FitRecord(
        "scaler", role, frozenset(subjects),
        {"mean_": scaler.mean_.copy(), "scale_": scaler.scale_.copy(), **audit},
    )
    model = pipe.named_steps["model"]
    model_fit = FitRecord(
        candidate.family, role, frozenset(subjects),
        {**fitted_state_params(candidate.family, model), **audit},
    )
    return scaler_fit, model_fit


def _seed_list(candidate, seeds):
    return tuple(seeds) if candidate.family in SEED_SENSITIVE else (seeds[0],)


def _fit_score_inner(candidate, bundle, inner, seeds, before_fit, *, score_fn=None,
                     row_multiplicity=None) -> tuple:
    """Fit on inner-train, score inner-val (`score_fn`, mean over seeds; `None` -> the
    original subject-balanced MAE, unchanged)."""
    subjects, X, y = bundle.subjects, bundle.X, bundle.y
    train_rows = np.isin(subjects, sorted(inner.train_subjects))
    val_rows = np.isin(subjects, sorted(inner.val_subjects))
    audit = _multiplicity_audit(train_rows, row_multiplicity, subjects)

    per_seed_scores = []
    val_preds_first = None
    scaler_fit = None
    model_fits = []
    for seed in _seed_list(candidate, seeds):
        pipe = _fit_once(candidate, X, y, train_rows, seed, before_fit, row_multiplicity)
        preds = pipe.predict(X[val_rows])
        per_seed_scores.append(_score(score_fn, bundle, val_rows, preds, row_multiplicity))
        if val_preds_first is None:
            val_preds_first = preds
        sfit, mfit = _model_and_scaler_fits(
            pipe, candidate, "inner_train", inner.train_subjects, audit
        )
        if scaler_fit is None:
            scaler_fit = sfit
        model_fits.append(mfit)

    score = float(np.mean(per_seed_scores))
    val_predictions = {
        int(s): val_preds_first[subjects[val_rows] == s] for s in sorted(inner.val_subjects)
    }
    fits = _bundle_fits(bundle, "inner_train", inner.train_subjects) + [scaler_fit] + model_fits
    return score, val_predictions, fits


def _score_candidates_on_fold(candidates, fold, seeds, before_fit, data_for, *, score_fn=None,
                              row_multiplicity=None) -> StageOutcome:
    n_c, n_f = len(candidates), len(fold.inner_folds)
    inner_scores = np.full((n_c, n_f), np.nan)
    cells: dict = {}
    feature_dims = [0] * n_c

    for ci, candidate in enumerate(candidates):          # candidate-major execution
        for fj, inner in enumerate(fold.inner_folds):
            bundle = data_for(candidate, inner.train_subjects)
            feature_dims[ci] = int(bundle.X.shape[1])
            train_rows = np.isin(bundle.subjects, sorted(inner.train_subjects))
            reason = _viability_reason(candidate, bundle, train_rows, row_multiplicity)
            if reason is not None:
                cells[(ci, fj)] = InnerResult(
                    inner.train_subjects, inner.val_subjects, candidate.candidate_id,
                    float("nan"), {}, [], reason=reason,
                )
                continue
            score, val_predictions, fits = _fit_score_inner(
                candidate, bundle, inner, seeds, before_fit, score_fn=score_fn,
                row_multiplicity=row_multiplicity,
            )
            inner_scores[ci, fj] = score
            cells[(ci, fj)] = InnerResult(
                inner.train_subjects, inner.val_subjects, candidate.candidate_id,
                score, val_predictions, fits,
            )

    # Assemble flat inner_results fold-major / candidate-minor (the frozen zip order).
    inner_results = [cells[(ci, fj)] for fj in range(n_f) for ci in range(n_c)]

    candidate_scores = []
    for ci, candidate in enumerate(candidates):
        per_fold = inner_scores[ci, :]
        candidate_scores.append(
            CandidateScore(
                candidate_id=candidate.candidate_id,
                inner_val_mae=float(np.mean(per_fold)),          # NaN if non-evaluable anywhere
                simplicity_rank=SIMPLICITY_RANK[candidate.family],
                feature_dimension=feature_dims[ci],
                inner_fold_variance=float(np.std(per_fold, ddof=0)),   # O1: population std
            )
        )
    return StageOutcome(list(candidates), inner_scores, inner_results, candidate_scores)


def select_stage_winner(stage: StageOutcome) -> Candidate:
    """Route the tie-break through `select_candidate` (never inline) and map the winning
    id back to its Candidate."""
    winner = select_candidate(stage.candidate_scores)
    by_id = {c.candidate_id: c for c in stage.candidates}
    return by_id[winner.candidate_id]


def _final_refit(candidate, fold, seeds, before_fit, data_for, *, score_fn=None,
                 row_multiplicity=None) -> tuple:
    """Refit the winner on all outer-training subjects (per seed), predict train + test.

    The held-out subject is never resampled: it is one subject with one role, and its score
    is the replicate's estimate for it. So `_score` on the test rows sees the multiplicities
    of the test subject alone (all 1 under the plan's LOSO-over-distinct-drawn-subjects
    construction), and only the TRAINING side is duplicated.
    """
    bundle = data_for(candidate, fold.train_subjects)
    subjects, X, y = bundle.subjects, bundle.X, bundle.y
    train_rows = np.isin(subjects, sorted(fold.train_subjects))
    test_rows = np.isin(subjects, [fold.test_subject])
    audit = _multiplicity_audit(train_rows, row_multiplicity, subjects)

    seed_outcomes = []
    scaler_fit = None
    model_fits = []
    for seed in _seed_list(candidate, seeds):
        pipe = _fit_once(candidate, X, y, train_rows, seed, before_fit, row_multiplicity)
        train_preds = pipe.predict(X[train_rows])
        test_preds = pipe.predict(X[test_rows])
        test_score = _score(score_fn, bundle, test_rows, test_preds)
        seed_outcomes.append(SeedOutcome(seed, train_preds, test_preds, test_score))
        sfit, mfit = _model_and_scaler_fits(
            pipe, candidate, "outer_train", fold.train_subjects, audit
        )
        if scaler_fit is None:
            scaler_fit = sfit
        model_fits.append(mfit)

    final_fits = _bundle_fits(bundle, "outer_train", fold.train_subjects) + [scaler_fit] + model_fits
    head = seed_outcomes[0]  # single-value frozen fields use the first seed (all equal if deterministic)
    return final_fits, head.train_predictions, head.test_predictions, head.test_score, seed_outcomes


def run_nested_candidates(
    dataset: Dataset,
    candidates,
    *,
    seeds=(0,),
    before_fit=None,
    data_for=None,
    score_fn=None,
    subject_multiplicity=None,
    **split_kwargs,
) -> list[FoldResult]:
    """Single-stage nested LOSO over `candidates`. Folds come only from `splits.py`;
    non-selectable outer folds contribute nothing. This is the shim's entry point.
    `score_fn=None` scores with the original `subject_balanced_mae`, unchanged; a supplied
    `score_fn(subjects, y_true, y_pred, session_idx)` overrides scoring at every fit (see
    `_score`) without changing fold construction, the tie-break, or anything else."""
    if data_for is None:
        data_for = fixed_feature_provider(dataset)
    folds = nested_loso_splits(dataset.subject_ids(), **split_kwargs)
    # Folds come from the DISTINCT subject ids either way: plan §2.4's "LOSO roles are
    # constructed over distinct drawn subjects; every copy of one original subject always
    # has one role". Multiplicity changes how much each training subject weighs, never who
    # is held out.
    row_multiplicity = subject_row_multiplicity(dataset.subjects, subject_multiplicity)

    results = []
    with threadpool_limits(1):
        for fold in folds:
            if not fold.selectable:
                continue
            stage = _score_candidates_on_fold(
                candidates, fold, seeds, before_fit, data_for, score_fn=score_fn,
                row_multiplicity=row_multiplicity,
            )
            winner = select_stage_winner(stage)
            final_fits, train_pred, test_pred, test_score, seed_outcomes = _final_refit(
                winner, fold, seeds, before_fit, data_for, score_fn=score_fn,
                row_multiplicity=row_multiplicity,
            )
            results.append(
                FoldResult(
                    test_subject=fold.test_subject,
                    train_subjects=fold.train_subjects,
                    selected=winner,
                    inner_scores=stage.inner_scores,
                    inner_results=stage.inner_results,
                    final_fits=final_fits,
                    train_predictions=train_pred,
                    test_predictions=test_pred,
                    test_score=test_score,
                    seed_outcomes=seed_outcomes,
                )
            )
    return results


def fit_audit(results) -> list[dict]:
    """Every fitted quantity -> the subject set it was estimated from (roles included).

    Duck-typed: works on engine `FoldResult`s, the shim's view objects, and torch results,
    since it only reads `.inner_results`/`.final_fits` and each record's `.quantity/.role/
    .subjects` plus the inner `.inner_val`.
    """
    audit = []
    for result in results:
        for inner in result.inner_results:
            for record in inner.fits:
                audit.append(
                    {
                        "test_subject": result.test_subject,
                        "quantity": record.quantity,
                        "role": record.role,
                        "fitted_on": record.subjects,
                        "inner_val": inner.inner_val,
                    }
                )
        for record in result.final_fits:
            audit.append(
                {
                    "test_subject": result.test_subject,
                    "quantity": record.quantity,
                    "role": record.role,
                    "fitted_on": record.subjects,
                    "inner_val": None,
                }
            )
    return audit
