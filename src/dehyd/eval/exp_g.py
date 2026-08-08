"""Experiment G — matched-session decision-level 10 GHz + 77 GHz fusion (MILESTONE_10_PLAN.md §2.3).

The question is narrow: on the sessions where BOTH radars produced an eligible recording, does
combining the two bands' predictions beat the 10 GHz band alone? The combiner is the frozen
constrained convex one,

    pred_fused = alpha * pred_10 + (1 - alpha) * pred_77,        alpha in {0.00, 0.05, ..., 1.00}

and the whole difficulty is producing an honest `alpha`. Four things shape this module:

  * **Matched cells, not matched frames.** The two front ends differ in geometry, frame rate and
    frame count, so no frame-to-frame alignment is attempted. The unit is the session cell
    `(subject, session_idx)`, and every condition — 10-only, 77-only, equal-weight, learned —
    trains and scores on the *same* matched cell population, so the four numbers are comparable.

  * **Selection-honest meta-training (A-M10-3).** `alpha` is fit on out-of-fold predictions of the
    outer-training subjects, and those predictions must come from a procedure that never saw the
    subject being predicted — not just never *fit* on it, never *selected* on it either. So for
    outer fold `s`, each attached inner fold's validation group `V` becomes a meta-validation
    group, and the COMPLETE Exp-A staged selection is re-run over `selection_folds(T_s \\ V)`
    before refitting on `T_s \\ V` and predicting `V`. The existing `InnerResult.val_predictions`
    could not be used for this: it keeps only the first seed, carries no session keys, and its
    candidate was chosen using the very validation outcomes it stores (Varma & Simon 2006;
    Cawley & Talbot 2010, plan §8.1).

  * **Five seed labels, not five observations.** A deterministic winning family is fit once and
    its prediction copied to the five configured seed labels with `deterministic_source_seed`
    recorded. Seeds are labels for a stochastic model's realizations, never extra observations.

  * **Fail-closed, whole-fold.** If `selection_folds` cannot build folds (fewer than two
    selection-training subjects) or no candidate survives a required further fold, the ENTIRE
    outer fold is non-evaluable and contributes nothing to any table. Partial meta-validation
    coverage is never used, and an outer fold that cannot produce a learned prediction does not
    get to contribute its 10-only number either — that would put the four conditions on different
    cells, which is the one thing the matched design exists to prevent.

Nothing here constructs a fold (all levels come from `splits.py`), enumerates a candidate
(`exp_a.stage1_candidates`/`stage2_candidates`), breaks a tie (`selection.select_candidate` and
`selection.select_alpha`), or re-implements a fit (`harness._score_candidates_on_fold` /
`_final_refit`). Under A-M10-4 only this decision-level combiner is implemented; the feature-level
variant is deferred and is not a Milestone-10 completion criterion.

**Fit-audit granularity.** `fit_audit_g.csv` records the fit chain BEHIND EVERY REPORTED
PREDICTION — the staged selection over each level's pool, that level's tuned-ε / scaler / model
refit, and the fusion alpha — which is exactly the enumeration plan §5.1 asks the audit to cover
("selection, scaler, model, and alpha subject sets") and what §5.4 needs for "every OOF and
outer-final prediction resolves to one complete base-selection record and fit-audit chain". The
inner-CV fits *inside* a staged selection are not rows: at ~113 candidates x 5 further folds x 6
levels x 2 bands x 16 folds they are order 10^5-10^6 records shipped through the spawn-pool pickle
for provenance nothing consumes, and they back no reported prediction. This is a granularity
reading of §3, not an amendment: G's per-candidate SELECTION table
(`fusion_base_selection.csv`) is kept in full, because §5.4/§8.2 need the losing candidates'
scores to prove outer outcomes are never read.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..features import store as store_mod
from ..features.protocol_freeze import protocol_freeze_guard
from . import fold_parallel, harness
from . import metrics as M
from .exp_a import (
    StoreBackedFeatures,
    build_sessions,
    expected_fingerprints,
    stage1_candidates,
    stage2_candidates,
)
from .harness import require_complete_active
from .reference_gate import json_sha256
from .selection import SelectionError, select_alpha
from .splits import (
    DEFAULT_MIN_TRAIN_SUBJECTS,
    DEFAULT_N_INNER_MAX,
    OuterFold,
    SplitError,
    nested_loso_splits,
    selection_folds,
)

EXP_G_SCHEMA_VERSION = 1

# `BANDS[0]` is the primary band throughout: alpha weights it, and the headline contrast is
# `fused - 10-only`. The order is therefore part of the contract, not presentation.
BANDS = ("10ghz", "77ghz")

# The level label used for the outer-final selection, i.e. the one run on all of T_s whose winner
# predicts the held-out subject. Meta levels are labelled "meta_{i}" for inner fold i.
OUTER_FINAL = "outer_final"

# The canonical non-evaluability reasons, in the order they can arise. A fold reports the first
# that applies and contributes nothing further. Everything ELSE — a key mismatch between the two
# bands, a target disagreement, a duplicated prediction key — is a hard error, not a reason: it
# means the tables are not describing what they claim to, which no exclusion row can excuse.
EXCLUSION_REASONS = (
    "outer_fold_not_selectable",
    "insufficient_selection_training_subjects",
    "no_surviving_candidate",
)


class ExpGError(ValueError):
    """A malformed matched population, a lineage disagreement between the two band configs,
    or a fusion table that failed its own key/consistency contract."""


# ------------------------------------------------------------------ the two-band lineage gate


def assert_shared_protocol(config_10, config_77) -> None:
    """Refuse two band configs that do not agree on what the analysis IS.

    `run_fusion.py` loads the two bands separately — they carry different front ends, different
    WST sections and different search spaces, and merging them through the config loader would
    produce a config describing neither band. What they must nonetheless share is everything the
    fusion estimand depends on: the run seeds (so seed label `k` means the same realization in
    both bands — `ExpGConfig.seed_pairing`), the split constants (so both bands' folds are the
    same folds), the fusion rule itself, the statistical protocol, and the weight workbook the
    target is read from.

    The target DEFINITION is checked with teeth further down, in `build_matched_population`:
    agreeing on the workbook path is a lineage check, but agreeing on `delta_m_pct` cell by cell
    is the real one, and a mismatch there is fatal.
    """
    checks = {
        "run.seed": (config_10.run.seed, config_77.run.seed),
        "run.seed_set": (tuple(config_10.run.seed_set), tuple(config_77.run.seed_set)),
        "split.n_inner_max": (config_10.split.n_inner_max, config_77.split.n_inner_max),
        "split.min_train_subjects": (
            config_10.split.min_train_subjects, config_77.split.min_train_subjects,
        ),
        "paths.weight_xlsx": (config_10.paths.weight_xlsx, config_77.paths.weight_xlsx),
        "paths.results_dir": (config_10.paths.results_dir, config_77.paths.results_dir),
        "exp_g": (config_10.exp_g, config_77.exp_g),
        "stats": (config_10.stats, config_77.stats),
    }
    mismatched = [f"{k}: 10 GHz {a!r} vs 77 GHz {b!r}" for k, (a, b) in checks.items() if a != b]
    if mismatched:
        raise ExpGError(
            "the 10 GHz and 77 GHz configs disagree on shared analysis constants, so a fused "
            "estimand would not be well defined: " + "; ".join(mismatched)
        )

    # Exp G builds folds at three levels and must build the SAME folds Experiment A would. Exp A
    # calls `nested_loso_splits(subjects)` at its defaults, so a config that carried different
    # split constants would describe a protocol this module does not run.
    if (config_10.split.n_inner_max, config_10.split.min_train_subjects) != (
        DEFAULT_N_INNER_MAX, DEFAULT_MIN_TRAIN_SUBJECTS
    ):
        raise ExpGError(
            f"split constants are (n_inner_max={config_10.split.n_inner_max}, "
            f"min_train_subjects={config_10.split.min_train_subjects}) but Experiment A runs at "
            f"({DEFAULT_N_INNER_MAX}, {DEFAULT_MIN_TRAIN_SUBJECTS}) — G's levels must be A's levels"
        )


# ------------------------------------------------------------------- the matched population


def _spine_by_key(sessions, band) -> dict:
    """{(subject, session_idx): record}, refusing a duplicated key.

    `exp_a.build_sessions` groups by exactly this key so a duplicate should be impossible; the
    check is here because §2.3 names it and because "impossible" is a claim about today's
    builder, not about the join this module performs on its output.
    """
    out: dict = {}
    for record in sessions:
        key = (int(record["subject"]), int(record["session_idx"]))
        if key in out:
            raise ExpGError(f"duplicate {band} session key {key} in the band spine")
        out[key] = record
    return out


def build_matched_population(config_10, config_77):
    """Build both band spines independently, then inner-join them on `(subject, session_idx)`.

    Returns `(matched, sessions_10, sessions_77, unmatched)` where `matched` is one dict per
    matched cell in canonical `(subject, session_idx)` order, `sessions_10`/`sessions_77` are the
    two bands' own session records restricted to those cells **in that same order** (so row `i`
    of either band's feature bundle is the same cell), and `unmatched` is one dict per cell that
    exists in one band only.

    Fails on a duplicated key, a target that disagrees between bands, a session name that
    disagrees, or a non-finite target. A disagreeing target is the serious one: the two bands
    read the same workbook, so a difference means the join is not aligning what it thinks it is.
    """
    spine_10 = _spine_by_key(build_sessions(config_10, "10ghz"), "10ghz")
    spine_77 = _spine_by_key(build_sessions(config_77, "77ghz"), "77ghz")

    matched, sessions_10, sessions_77, unmatched = [], [], [], []
    for key in sorted(set(spine_10) | set(spine_77)):
        subject, session_idx = key
        in_10, in_77 = key in spine_10, key in spine_77
        if not (in_10 and in_77):
            unmatched.append({
                "subject": subject,
                "session_idx": session_idx,
                "missing_band": "77ghz" if in_10 else "10ghz",
                "reason": "session_not_eligible_in_that_band",
            })
            continue

        record_10, record_77 = spine_10[key], spine_77[key]
        target_10 = float(record_10["delta_m_pct"])
        target_77 = float(record_77["delta_m_pct"])
        if not np.isfinite(target_10):
            raise ExpGError(f"non-finite delta_m_pct for {key}: {target_10}")
        if target_10 != target_77:
            raise ExpGError(
                f"the two bands disagree on delta_m_pct for {key}: 10 GHz {target_10!r} vs "
                f"77 GHz {target_77!r} — both read the same weight workbook, so the join is "
                "not aligning the cells it thinks it is"
            )
        if record_10["session_name"] != record_77["session_name"]:
            raise ExpGError(
                f"inconsistent session name for {key}: 10 GHz {record_10['session_name']!r} vs "
                f"77 GHz {record_77['session_name']!r}"
            )

        matched.append({
            "subject": subject,
            "session_idx": session_idx,
            "session_name": record_10["session_name"],
            "delta_m_pct": target_10,
            "n_frames_10": len(record_10["frame_ids"]),
            "n_frames_77": len(record_77["frame_ids"]),
        })
        sessions_10.append(record_10)
        sessions_77.append(record_77)

    if not matched:
        raise ExpGError("the two band spines have no session cell in common")
    return matched, sessions_10, sessions_77, unmatched


def population_summary(matched, unmatched) -> dict:
    """The §2.3 population record: what each band had, what survived the join, and per subject.

    A cell present in one band only is recorded with the band it is MISSING from, so the
    pre-match inventory of band X is the matched cells plus the cells missing from the other
    band — the two views are the same table read two ways, which is why no separate count is
    carried around and allowed to drift.
    """
    subjects = sorted({c["subject"] for c in matched})
    per_subject = {s: sum(1 for c in matched if c["subject"] == s) for s in subjects}
    only_in = {
        band: [u for u in unmatched if u["missing_band"] == other]
        for band, other in (("10ghz", "77ghz"), ("77ghz", "10ghz"))
    }
    return {
        "n_subjects_10ghz_before_matching": len(
            set(subjects) | {u["subject"] for u in only_in["10ghz"]}),
        "n_subjects_77ghz_before_matching": len(
            set(subjects) | {u["subject"] for u in only_in["77ghz"]}),
        "n_cells_10ghz_before_matching": len(matched) + len(only_in["10ghz"]),
        "n_cells_77ghz_before_matching": len(matched) + len(only_in["77ghz"]),
        "n_subjects_g": len(subjects),
        "n_matched_cells": len(matched),
        "n_unmatched_cells": len(unmatched),
        "sessions_per_matched_subject": {int(s): int(n) for s, n in per_subject.items()},
    }


# --------------------------------------------------------------- one selection level, one band


def _selection_record_id(test_subject, level, band) -> str:
    """The stable id every OOF row, selection row and fit-audit row of one level joins on."""
    return f"g|s{int(test_subject)}|{level}|{band}"


def _before_fit(config):
    """Exp A's own per-fit protocol gate, unchanged: complete `active` record, then the guard."""

    def before_fit(candidate):
        active = dict(candidate.active)
        require_complete_active(active)
        protocol_freeze_guard(config, active=active)

    return before_fit


def _stage_rows(record_id, test_subject, level, band, stage_name, stage, winner,
                train_subjects, validation_subjects, seeds) -> list[dict]:
    """One row per ENUMERATED candidate of one stage — the losers included, deliberately.

    §5.4 requires a fixture in which the outer-test outcome *would* have chosen a different
    candidate, proving those outcomes are never read; that test needs every candidate's inner
    score, not just the winner's. This is the distinction A-M10-10 draws: H records winners
    because its selection-honesty is structural (it reuses A/B/C orchestration unchanged),
    G records the full enumeration because its selection-honesty is what is under test.
    """
    train = sorted(int(s) for s in train_subjects)
    validation = sorted(int(s) for s in validation_subjects) if validation_subjects else []
    seeds_json = json.dumps([int(s) for s in seeds])
    n_folds = int(stage.inner_scores.shape[1])

    rows = []
    for candidate, score in zip(stage.candidates, stage.candidate_scores, strict=True):
        axes = dict(candidate.active)
        axes.pop("model_family", None)          # it has its own column
        rows.append({
            "outer_test_subject": int(test_subject),
            "meta_fold_or_outer_final": level,
            "band": band,
            "stage": stage_name,
            "candidate": candidate.candidate_id,
            "selection_record_id": record_id,
            "train_subjects_json": json.dumps(train),
            "train_subjects_sha256": json_sha256(train),
            # Blank for the outer-final level (§3): there is no meta-validation group there —
            # its winner is applied to the held-out subject, which is never a validation group.
            "validation_subjects_json": json.dumps(validation) if validation else "",
            "validation_subjects_sha256": json_sha256(validation) if validation else "",
            "feature_key": str(tuple(candidate.feature_key)),
            "active_axes_json": json.dumps(axes, sort_keys=True, default=str),
            "family": candidate.family,
            "params_json": json.dumps(candidate.params(), sort_keys=True),
            "candidate_score": score.inner_val_mae,
            "candidate_score_variance": score.inner_fold_variance,
            "n_selection_folds": n_folds,
            "model_seeds_json": seeds_json,
            "selected": candidate.candidate_id == winner.candidate_id,
        })
    return rows


def _staged_selection(config, band, provider, level_fold, seeds, before_fit):
    """Exp A's two-stage search, verbatim, over whatever folds `level_fold` carries.

    Stage 1 searches the feature axes at the fixed ridge anchor; Stage 2 searches family x grid
    at the Stage-1 winner's feature key. Scores aggregate exactly as in ordinary Exp A — the
    subject-balanced validation MAE is averaged over seeds within a fold by
    `_fit_score_inner`, then over folds by `_score_candidates_on_fold`, and `select_candidate`
    decides. Returns `(stage1, winner1, stage2, winner2)`.
    """
    anchor = (config.search_10ghz if band == "10ghz" else config.search_77ghz).stage1_anchor_ridge_alpha
    stage1 = harness._score_candidates_on_fold(
        stage1_candidates(config, band, anchor), level_fold, seeds, before_fit, provider.data_for,
    )
    winner1 = harness.select_stage_winner(stage1)
    stage2 = harness._score_candidates_on_fold(
        stage2_candidates(config, band, winner1.feature_key, dict(winner1.active)),
        level_fold, seeds, before_fit, provider.data_for,
    )
    return stage1, winner1, stage2, harness.select_stage_winner(stage2)


def _labeled_seed_predictions(seed_outcomes, seeds):
    """`[(seed_label, deterministic_source_seed, predictions), ...]` over the CONFIGURED labels.

    `harness._seed_list` fits a deterministic family once (families outside `SEED_SENSITIVE`
    ignore the seed) and a stochastic one per seed. §2.3 asks for five seed LABELS either way:
    a deterministic winner's single result is copied to all five with the seed it actually came
    from recorded, so a reader can tell five realizations from one realization labelled five
    times. It is never counted as five independent observations.
    """
    labels = [int(s) for s in seeds]
    if len(seed_outcomes) == len(labels):
        return [(int(o.seed), None, o.test_predictions) for o in seed_outcomes]
    if len(seed_outcomes) != 1:
        raise ExpGError(
            f"expected either 1 or {len(labels)} seed outcomes, got {len(seed_outcomes)}"
        )
    source = int(seed_outcomes[0].seed)
    return [(label, source, seed_outcomes[0].test_predictions) for label in labels]


def _refit_and_predict(winner, train_subjects, predict_subjects, seeds, before_fit, data_for):
    """Refit `winner` on `train_subjects` and predict each subject of `predict_subjects`.

    `harness._final_refit` is the frozen refit path and predicts exactly one held-out subject,
    so a meta-validation group of several subjects is walked one subject at a time. The refit is
    a deterministic function of (winner, training rows, seed), so every pass fits the identical
    model — the loop buys one subject's predictions per pass, not a different model per subject,
    and its handful of extra fits is nothing against a staged search of thousands. The fit
    records are therefore taken from the first pass only.

    Returns `(final_fits, {subject: seed_outcomes})`.
    """
    final_fits = None
    per_subject: dict[int, list] = {}
    for subject in sorted(int(s) for s in predict_subjects):
        level = OuterFold(
            test_subject=subject,
            train_subjects=frozenset(int(s) for s in train_subjects),
            selectable=True,
            inner_folds=(),
        )
        fits, _, _, _, seed_outcomes = harness._final_refit(
            winner, level, seeds, before_fit, data_for
        )
        if final_fits is None:
            final_fits = fits
        per_subject[subject] = seed_outcomes
    return final_fits or [], per_subject


# ------------------------------------------------------------------------ the band-fold worker


@dataclass
class BandFoldResult:
    """Everything one (outer fold, band) produced: its OOF rows, its outer-test predictions,
    and the provenance of both. `reason` non-None means this band made the whole outer fold
    non-evaluable for learned fusion."""

    test_subject: int
    band: str
    meta_oof_rows: list = field(default_factory=list)
    outer_rows: list = field(default_factory=list)
    selection_rows: list = field(default_factory=list)
    fit_audit_rows: list = field(default_factory=list)
    exclusions: list = field(default_factory=list)
    reason: str | None = None


def _prediction_rows(sessions, subject, labeled, record_id, band, test_subject,
                     meta_fold) -> list[dict]:
    """The (subject, session_idx, seed) rows one predicted subject contributes.

    `harness._final_refit` predicts `np.isin(provider.subjects, [subject])` in provider row
    order, and the provider's row order IS `sessions`' order, so zipping the prediction vector
    against that subject's session records in order is the alignment — not an assumption about
    which sessions a subject has.
    """
    cells = [s for s in sessions if int(s["subject"]) == int(subject)]
    rows = []
    for seed, source, predictions in labeled:
        if len(predictions) != len(cells):
            raise ExpGError(
                f"subject {subject} has {len(cells)} matched cells but the refit returned "
                f"{len(predictions)} predictions"
            )
        for cell, y_pred in zip(cells, predictions, strict=True):
            rows.append({
                "outer_test_subject": int(test_subject),
                "meta_fold": meta_fold,
                "band": band,
                "subject": int(subject),
                "session_idx": int(cell["session_idx"]),
                "seed": int(seed),
                "deterministic_source_seed": "" if source is None else int(source),
                "selection_record_id": record_id,
                "y_true": float(cell["delta_m_pct"]),
                "y_pred": float(y_pred),
            })
    return rows


def _fit_audit_row(record_id, test_subject, level, band, quantity, role, fitted_subjects,
                   predicted_subjects) -> dict:
    """One audit row, with the invariant checked while the subject sets are still in hand.

    "Fitted on a subject it then predicted" is the failure this whole table exists to make
    impossible, so it is a hard error raised where the sets are readable — not a row written
    for someone to notice later.
    """
    fitted = sorted(int(s) for s in fitted_subjects)
    predicted = sorted(int(s) for s in predicted_subjects)
    overlap = sorted(set(fitted) & set(predicted))
    if overlap:
        raise ExpGError(
            f"{quantity!r} at {record_id} was fit on subject(s) it then predicted: {overlap}"
        )
    return {
        "outer_test_subject": int(test_subject),
        "meta_fold_or_outer_final": level,
        "band": band,
        "quantity": quantity,
        "role": role,
        "fitted_subjects_json": json.dumps(fitted),
        "fitted_subjects_sha256": json_sha256(fitted),
        "predicted_subjects_json": json.dumps(predicted),
        "selection_record_id": record_id,
    }


def _level_fit_audit_rows(record_id, test_subject, level, band, final_fits, train_subjects,
                          predicted_subjects) -> list[dict]:
    """One level's audit chain: the staged selection, then every fitted quantity of its refit.

    `final_fits` is what `harness._final_refit` returns — the tuned-ε (when the winner is on the
    tuned branch), the scaler, and one model record per realized seed — each already carrying
    the subject set it was estimated from.
    """
    rows = [_fit_audit_row(record_id, test_subject, level, band, "staged_selection",
                           "selection_train", train_subjects, predicted_subjects)]
    rows += [
        _fit_audit_row(record_id, test_subject, level, band, record.quantity, record.role,
                       record.subjects, predicted_subjects)
        for record in final_fits
    ]
    return rows


def _run_band_fold(config, band, sessions, store_dir, fold, seeds) -> BandFoldResult:
    """One (outer fold, band): every meta level's OOF predictions plus the outer-final ones.

    Top-level and picklable so it is the unit of a spawn-context Pool. Builds its OWN
    store-backed provider (open npz handles are not shareable across processes) and pins
    single-threaded math, so the result is bit-identical serial or parallel — the same contract
    `exp_a._run_single_fold` carries.

    `(outer fold, band)` rather than `(outer fold)` is the parallel unit because the two bands'
    work is completely independent given the fold: alpha needs both bands' prediction TABLES,
    which are pure data, so pairing them is arithmetic done later by the caller. That doubles
    the parallel width to 2 x n_folds and roughly halves wall-clock on a node with cores to
    spare.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        result = BandFoldResult(test_subject=int(fold.test_subject), band=band)
        provider = StoreBackedFeatures(band, sessions, store_dir, config)
        before_fit = _before_fit(config)
        train_subjects = sorted(int(s) for s in fold.train_subjects)

        # --- the meta levels: one per attached inner fold, each honest about its own group V ---
        for meta_index, inner in enumerate(fold.inner_folds):
            level = f"meta_{meta_index}"
            record_id = _selection_record_id(fold.test_subject, level, band)
            validation = sorted(int(s) for s in inner.val_subjects)
            pool = sorted(s for s in train_subjects if s not in set(validation))
            try:
                further = selection_folds(pool, n_inner_max=DEFAULT_N_INNER_MAX)
            except SplitError as err:
                result.reason = "insufficient_selection_training_subjects"
                result.exclusions.append({
                    "outer_test_subject": int(fold.test_subject), "meta_fold": level,
                    "band": band, "reason": result.reason, "detail": str(err),
                })
                return result

            level_fold = OuterFold(
                test_subject=int(fold.test_subject),      # carried for provenance only
                train_subjects=frozenset(pool),
                selectable=True,
                inner_folds=further,
            )
            try:
                stage1, winner1, stage2, winner2 = _staged_selection(
                    config, band, provider, level_fold, seeds, before_fit
                )
            except SelectionError as err:
                result.reason = "no_surviving_candidate"
                result.exclusions.append({
                    "outer_test_subject": int(fold.test_subject), "meta_fold": level,
                    "band": band, "reason": result.reason, "detail": f"{type(err).__name__}: {err}",
                })
                return result

            result.selection_rows += _stage_rows(
                record_id, fold.test_subject, level, band, "stage1", stage1, winner1,
                pool, validation, seeds,
            )
            result.selection_rows += _stage_rows(
                record_id, fold.test_subject, level, band, "stage2", stage2, winner2,
                pool, validation, seeds,
            )

            final_fits, per_subject = _refit_and_predict(
                winner2, pool, validation, seeds, before_fit, provider.data_for
            )
            for subject, seed_outcomes in per_subject.items():
                result.meta_oof_rows += _prediction_rows(
                    sessions, subject, _labeled_seed_predictions(seed_outcomes, seeds),
                    record_id, band, fold.test_subject, meta_fold=level,
                )
            result.fit_audit_rows += _level_fit_audit_rows(
                record_id, fold.test_subject, level, band, final_fits, pool, validation
            )

        # --- the outer-final level: the ordinary Exp-A staged selection on all of T_s ---
        record_id = _selection_record_id(fold.test_subject, OUTER_FINAL, band)
        try:
            stage1, winner1, stage2, winner2 = _staged_selection(
                config, band, provider, fold, seeds, before_fit
            )
        except SelectionError as err:
            result.reason = "no_surviving_candidate"
            result.exclusions.append({
                "outer_test_subject": int(fold.test_subject), "meta_fold": OUTER_FINAL,
                "band": band, "reason": result.reason, "detail": f"{type(err).__name__}: {err}",
            })
            return result

        result.selection_rows += _stage_rows(
            record_id, fold.test_subject, OUTER_FINAL, band, "stage1", stage1, winner1,
            train_subjects, (), seeds,
        )
        result.selection_rows += _stage_rows(
            record_id, fold.test_subject, OUTER_FINAL, band, "stage2", stage2, winner2,
            train_subjects, (), seeds,
        )

        final_fits, per_subject = _refit_and_predict(
            winner2, train_subjects, [int(fold.test_subject)], seeds, before_fit, provider.data_for
        )
        result.outer_rows = _prediction_rows(
            sessions, int(fold.test_subject),
            _labeled_seed_predictions(per_subject[int(fold.test_subject)], seeds),
            record_id, band, fold.test_subject, meta_fold=OUTER_FINAL,
        )
        result.fit_audit_rows += _level_fit_audit_rows(
            record_id, fold.test_subject, OUTER_FINAL, band, final_fits, train_subjects,
            [int(fold.test_subject)],
        )
        return result


# ------------------------------------------------------------------------ alpha and the fusion


def _keyed(rows) -> dict:
    """{(subject, session_idx, seed): (y_true, y_pred)} — refusing a duplicated prediction key."""
    out: dict = {}
    for row in rows:
        key = (row["subject"], row["session_idx"], row["seed"])
        if key in out:
            raise ExpGError(f"duplicate prediction key {key}")
        out[key] = (row["y_true"], row["y_pred"])
    return out


def _require_matching_keys(table_10, table_77, what) -> list:
    """Both bands must predict exactly the same cells, with the same targets. Sorted keys out."""
    if set(table_10) != set(table_77):
        only_10 = sorted(set(table_10) - set(table_77))
        only_77 = sorted(set(table_77) - set(table_10))
        raise ExpGError(
            f"{what}: the two bands' prediction keys differ (10 GHz only: {only_10[:5]}, "
            f"77 GHz only: {only_77[:5]})"
        )
    keys = sorted(table_10)
    for key in keys:
        if table_10[key][0] != table_77[key][0]:
            raise ExpGError(
                f"{what}: the two bands disagree on y_true at {key}: "
                f"{table_10[key][0]!r} vs {table_77[key][0]!r}"
            )
    return keys


def _alpha_grid_rows(config, test_subject, table_10, table_77, seeds):
    """The full objective grid and the selected alpha for one outer fold.

    For each alpha and each seed LABEL, the objective is the subject-balanced MAE of the fused
    OOF predictions across `T_s` — subject-balanced because a subject with more matched sessions
    must not weigh more in a per-subject estimand. Alpha is chosen from the MEAN over the five
    paired seed labels, then `select_alpha` applies the frozen closest-to-1.0 tie-break.
    """
    keys = _require_matching_keys(table_10, table_77, f"meta OOF for outer subject {test_subject}")
    labels = [int(s) for s in seeds]
    by_seed = {
        label: [k for k in keys if k[2] == label] for label in labels
    }
    missing = [label for label, ks in by_seed.items() if not ks]
    if missing:
        raise ExpGError(
            f"outer subject {test_subject}: no meta OOF rows for seed label(s) {missing} — the "
            "five-seed contract requires every configured label to be covered"
        )
    # ...and nothing beyond them. A row carrying an unconfigured label would otherwise be
    # dropped silently here, quietly shrinking the population alpha is fit on.
    unexpected = sorted({k[2] for k in keys} - set(labels))
    if unexpected:
        raise ExpGError(
            f"outer subject {test_subject}: meta OOF rows carry unconfigured seed label(s) "
            f"{unexpected}; the configured labels are {labels}"
        )

    grid = [float(a) for a in config.exp_g.alpha_grid]
    per_alpha_seed: dict[tuple[float, int], float] = {}
    mean_over_seeds: list[float] = []
    for alpha in grid:
        per_seed = []
        for label in labels:
            ks = by_seed[label]
            subjects = np.array([k[0] for k in ks])
            y_true = np.array([table_10[k][0] for k in ks], dtype=float)
            fused = np.array([
                alpha * table_10[k][1] + (1.0 - alpha) * table_77[k][1] for k in ks
            ], dtype=float)
            value = M.subject_balanced_mae(subjects, y_true, fused)
            per_alpha_seed[(alpha, label)] = value
            per_seed.append(value)
        mean_over_seeds.append(float(np.mean(per_seed)))

    alpha = select_alpha(grid, mean_over_seeds, tie_break=config.exp_g.alpha_tie_break)
    rows = [
        {
            "outer_test_subject": int(test_subject),
            "alpha": a,
            "seed": label,
            "subject_balanced_oof_mae": per_alpha_seed[(a, label)],
            "mean_over_seeds": mean_over_seeds[i],
            "selected": a == alpha,
        }
        for i, a in enumerate(grid)
        for label in labels
    ]
    return alpha, rows


@dataclass
class ExpGFoldResult:
    """One outer fold's fused outcome, or the reason it is non-evaluable for learned fusion."""

    test_subject: int
    alpha: float | None = None
    alpha_grid_rows: list = field(default_factory=list)
    prediction_rows: list = field(default_factory=list)
    meta_oof_rows: list = field(default_factory=list)
    selection_rows: list = field(default_factory=list)
    fit_audit_rows: list = field(default_factory=list)
    exclusions: list = field(default_factory=list)
    reason: str | None = None


def _fuse_fold(config, fold, band_results, seeds) -> ExpGFoldResult:
    """Pair the two bands, fit alpha on the meta OOF rows, and apply it to the outer test rows.

    Every number here is arithmetic over the two bands' saved prediction tables — no model is
    fit and no store is read, which is exactly why the band work can be parallelised past the
    fold and joined afterwards.
    """
    result = ExpGFoldResult(test_subject=int(fold.test_subject))
    by_band = {r.band: r for r in band_results}
    for band in BANDS:
        band_result = by_band[band]
        result.selection_rows += band_result.selection_rows
        result.fit_audit_rows += band_result.fit_audit_rows
        result.meta_oof_rows += band_result.meta_oof_rows
        result.exclusions += band_result.exclusions
    for band in BANDS:
        if by_band[band].reason is not None:
            result.reason = by_band[band].reason
            return result

    # "No partial meta-validation coverage is ever used" (§2.3). The whole-fold fail-closed rule
    # above should already guarantee it, so this is the check that the guarantee held: alpha must
    # be fit over ALL of T_s, never over whichever meta groups happened to succeed.
    for band in BANDS:
        covered = {row["subject"] for row in by_band[band].meta_oof_rows}
        if covered != set(fold.train_subjects):
            raise ExpGError(
                f"outer subject {fold.test_subject}, {band}: meta OOF rows cover "
                f"{sorted(covered)} but the outer-training set is {sorted(fold.train_subjects)}"
            )

    alpha, result.alpha_grid_rows = _alpha_grid_rows(
        config, fold.test_subject,
        _keyed(by_band["10ghz"].meta_oof_rows), _keyed(by_band["77ghz"].meta_oof_rows), seeds,
    )
    result.alpha = alpha

    outer_10 = _keyed(by_band["10ghz"].outer_rows)
    outer_77 = _keyed(by_band["77ghz"].outer_rows)
    keys = _require_matching_keys(
        outer_10, outer_77, f"outer-test predictions for subject {fold.test_subject}"
    )
    for subject, session_idx, seed in keys:
        key = (subject, session_idx, seed)
        y_true, pred_10 = outer_10[key]
        _, pred_77 = outer_77[key]
        result.prediction_rows.append({
            "outer_test_subject": int(fold.test_subject),
            "subject": subject,
            "session_idx": session_idx,
            "seed": seed,
            "y_true": y_true,
            "pred_10": pred_10,
            "pred_77": pred_77,
            # Predictions are combined per seed label and scored per seed label; they are never
            # averaged across seeds before scoring (§2.3).
            "pred_equal_weight": 0.5 * pred_10 + 0.5 * pred_77,
            "pred_fused": alpha * pred_10 + (1.0 - alpha) * pred_77,
            "alpha": alpha,
        })

    # Alpha is a fitted quantity and audits like any other: estimated from the outer-training
    # subjects' OOF rows only, applied to the held-out subject. Its band is "fused" because it
    # is the one quantity in Experiment G that belongs to neither band alone.
    result.fit_audit_rows.append(_fit_audit_row(
        _selection_record_id(fold.test_subject, OUTER_FINAL, "fused"), fold.test_subject,
        OUTER_FINAL, "fused", "fusion_alpha", "outer_train", sorted(fold.train_subjects),
        [int(fold.test_subject)],
    ))
    return result


# ------------------------------------------------------------------------------ orchestration


def run_exp_g(config_10, config_77, sessions_10, sessions_77, store_dir, *, seeds, n_workers=1):
    """Every outer fold's fused result, in canonical test-subject order.

    The parallel unit is `(outer fold, band)`; results come back in completion order and are
    regrouped by test subject before the (cheap, deterministic) fusion arithmetic runs.
    """
    subjects = sorted({int(s["subject"]) for s in sessions_10})
    folds = nested_loso_splits(subjects)

    tasks = []
    skipped = []
    for fold in folds:
        if not fold.selectable:
            skipped.append(fold)
            continue
        tasks.append((config_10, "10ghz", sessions_10, store_dir, fold, seeds))
        tasks.append((config_77, "77ghz", sessions_77, store_dir, fold, seeds))

    band_results = fold_parallel.run_folds_parallel(
        _run_band_fold, tasks, n_workers, "exp-g", unit="fold-bands",
    )
    grouped: dict[int, list] = {}
    for band_result in band_results:
        grouped.setdefault(band_result.test_subject, []).append(band_result)

    results = [
        ExpGFoldResult(
            test_subject=int(fold.test_subject),
            reason="outer_fold_not_selectable",
            exclusions=[{
                "outer_test_subject": int(fold.test_subject), "meta_fold": "", "band": "",
                "reason": "outer_fold_not_selectable",
                "detail": f"only {len(fold.train_subjects)} training subjects",
            }],
        )
        for fold in skipped
    ]
    for fold in folds:
        if fold.selectable:
            results.append(_fuse_fold(config_10, fold, grouped[int(fold.test_subject)], seeds))
    results.sort(key=lambda r: r.test_subject)
    return results


# ---------------------------------------------------------------------------------- reporting


# The four conditions, named so that `pred_{condition}` is the prediction column and
# `mae_{condition}` the per-subject column. All four are scored on the identical matched cells.
CONDITIONS = ("10", "77", "equal_weight", "fused")


def per_subject_rows(results) -> list[dict]:
    """Per-subject MAE for the four conditions, after the frozen additive seed collapse.

    `StatsConfig.seed_collapse_additive` is "average_per_subject_before_resample": each seed
    label's MAE is computed for the subject and the labels are then averaged. The primary
    contrast is the subject-ADDITIVE `fused - 10`, so it is formed here, per subject, and never
    as a session-weighted pooled difference — on a cohort with unequal session counts the two
    are different numbers.
    """
    rows = []
    for result in results:
        if result.reason is not None:
            continue
        by_seed: dict[int, list] = {}
        for row in result.prediction_rows:
            by_seed.setdefault(row["seed"], []).append(row)
        per_condition = {}
        for condition in CONDITIONS:
            column = f"pred_{condition}"
            per_seed = [
                float(np.mean([abs(r["y_true"] - r[column]) for r in seed_rows]))
                for seed_rows in by_seed.values()
            ]
            per_condition[condition] = float(np.mean(per_seed))
        n_sessions = len(next(iter(by_seed.values())))
        rows.append({
            "subject": int(result.test_subject),
            "n_sessions": int(n_sessions),
            "mae_10": per_condition["10"],
            "mae_77": per_condition["77"],
            "mae_equal_weight": per_condition["equal_weight"],
            "mae_fused": per_condition["fused"],
            "difference_fused_minus_10": per_condition["fused"] - per_condition["10"],
        })
    rows.sort(key=lambda r: r["subject"])
    return rows


def _ci_dict(ci) -> dict:
    return {"point": ci.point, "low": ci.low, "high": ci.high, "method": ci.method,
            "n_eval": ci.n_eval, "n_skipped": ci.n_skipped, "unreliable": ci.unreliable}


def summarize_exp_g(results, subject_rows, population, config) -> dict:
    """The primary fused-minus-10 estimand with its subject-cluster CI, plus the descriptive
    secondaries. Fusion is not required to beat 10 GHz; the sign is reported as observed."""
    stats = config.stats
    seed = config.run.seed
    mae = {c: np.array([r[f"mae_{c}"] for r in subject_rows], dtype=float) for c in CONDITIONS}
    primary = mae["fused"] - mae["10"]

    def difference_ci(values, offset):
        return _ci_dict(M.mean_difference_ci(values, b=stats.bootstrap_b, rng_seed=seed + offset))

    evaluable = [r for r in results if r.reason is None]
    return {
        "conditional_exploratory": True,
        "schema_version": EXP_G_SCHEMA_VERSION,
        "analysis": "matched_session_decision_fusion",
        "combiner": "alpha * pred_10 + (1 - alpha) * pred_77",
        "feature_level_variant": "deferred (A-M10-4); not a milestone-10 completion criterion",
        "population": population,
        "n_subjects_g": len(subject_rows),
        "n_evaluable_outer_folds": len(evaluable),
        "n_seeds": len(config.run.seed_set),
        "alpha_grid": [float(a) for a in config.exp_g.alpha_grid],
        "alpha_tie_break": config.exp_g.alpha_tie_break,
        "alpha_objective": config.exp_g.objective,
        "alpha_by_outer_fold": {int(r.test_subject): float(r.alpha) for r in evaluable},
        "primary": {
            "estimand": "mean_over_subject(mean_over_seed(MAE_fused - MAE_10ghz))",
            "direction": "negative favours fusion",
            "mean_difference_fused_minus_10": difference_ci(primary, 0),
            "sign": "negative" if float(np.mean(primary)) < 0 else (
                "positive" if float(np.mean(primary)) > 0 else "zero"),
            "n_subjects": len(subject_rows),
        },
        "secondary_descriptive": {
            "note": "descriptive only — no additional p-value family is created (§2.3)",
            "mean_difference_77ghz_minus_10": difference_ci(mae["77"] - mae["10"], 1),
            "mean_difference_equal_weight_minus_10": difference_ci(
                mae["equal_weight"] - mae["10"], 2),
            "subject_balanced_mae_10ghz": _ci_dict(M.subject_cluster_bootstrap(
                mae["10"], b=stats.bootstrap_b, rng_seed=seed + 3)),
            "subject_balanced_mae_77ghz": _ci_dict(M.subject_cluster_bootstrap(
                mae["77"], b=stats.bootstrap_b, rng_seed=seed + 4)),
            "subject_balanced_mae_equal_weight": _ci_dict(M.subject_cluster_bootstrap(
                mae["equal_weight"], b=stats.bootstrap_b, rng_seed=seed + 5)),
            "subject_balanced_mae_fused": _ci_dict(M.subject_cluster_bootstrap(
                mae["fused"], b=stats.bootstrap_b, rng_seed=seed + 6)),
        },
        "per_subject_difference_fused_minus_10": {
            int(r["subject"]): float(r["difference_fused_minus_10"]) for r in subject_rows
        },
        "limitation": (
            "Fusion applies only to subjects/sessions with both eligible bands in the original "
            "16-subject cohort. The result is conditional and exploratory, generalizes to no "
            "other cohort, and cannot rescue the single-band validation outcome."
        ),
    }


# ---------------------------------------------------------------------------------- artifacts

MATCHED_POPULATION_COLUMNS = (
    "subject", "session_idx", "session_name", "delta_m_pct", "n_frames_10", "n_frames_77",
)
UNMATCHED_POPULATION_COLUMNS = ("subject", "session_idx", "missing_band", "reason")
META_OOF_COLUMNS = (
    "outer_test_subject", "meta_fold", "band", "subject", "session_idx", "seed",
    "deterministic_source_seed", "selection_record_id", "y_true", "y_pred",
)
BASE_SELECTION_COLUMNS = (
    "outer_test_subject", "meta_fold_or_outer_final", "band", "stage", "candidate",
    "selection_record_id", "train_subjects_json", "train_subjects_sha256",
    "validation_subjects_json", "validation_subjects_sha256", "feature_key", "active_axes_json",
    "family", "params_json", "candidate_score", "candidate_score_variance", "n_selection_folds",
    "model_seeds_json", "selected",
)
FIT_AUDIT_COLUMNS = (
    "outer_test_subject", "meta_fold_or_outer_final", "band", "quantity", "role",
    "fitted_subjects_json", "fitted_subjects_sha256", "predicted_subjects_json",
    "selection_record_id",
)
ALPHA_GRID_COLUMNS = (
    "outer_test_subject", "alpha", "seed", "subject_balanced_oof_mae", "mean_over_seeds",
    "selected",
)
PREDICTIONS_COLUMNS = (
    "outer_test_subject", "subject", "session_idx", "seed", "y_true", "pred_10", "pred_77",
    "pred_equal_weight", "pred_fused", "alpha",
)
PER_SUBJECT_COLUMNS = (
    "subject", "n_sessions", "mae_10", "mae_77", "mae_equal_weight", "mae_fused",
    "difference_fused_minus_10",
)
EXCLUSIONS_COLUMNS = ("outer_test_subject", "meta_fold", "band", "reason", "detail")


def _write_csv(path, columns, rows) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fusion_figure(subject_rows, path) -> Path:
    """The comparison figure, drawn ONLY from the per-subject table that was just written.

    Two panels because the two things a reader wants are different questions: the left panel is
    each condition's per-subject MAE (level), the right is the paired per-subject fused - 10
    difference (the actual estimand), with zero marked so the sign is read off the plot rather
    than inferred.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: no display
    import matplotlib.pyplot as plt

    subjects = [r["subject"] for r in subject_rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(subjects), dtype=float)
    width = 0.2
    for i, (condition, label) in enumerate([
        ("10", "10 GHz"), ("77", "77 GHz"), ("equal_weight", "equal weight"), ("fused", "fused"),
    ]):
        left.bar(x + (i - 1.5) * width, [r[f"mae_{condition}"] for r in subject_rows],
                 width=width, label=label)
    left.set_xticks(x)
    left.set_xticklabels([str(s) for s in subjects], fontsize=8)
    left.set_xlabel("subject")
    left.set_ylabel("MAE (Δm% points)")
    left.set_title("per-subject MAE by condition")
    left.legend(fontsize=8)

    differences = [r["difference_fused_minus_10"] for r in subject_rows]
    right.bar(x, differences, width=0.6, color="#4C72B0")
    right.axhline(0.0, color="k", lw=1)
    right.set_xticks(x)
    right.set_xticklabels([str(s) for s in subjects], fontsize=8)
    right.set_xlabel("subject")
    right.set_ylabel("fused − 10 GHz (Δm% points)")
    right.set_title("paired difference (negative favours fusion)")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def write_exp_g_reports(results, matched, unmatched, subject_rows, summary, out_dir) -> dict:
    """The nine §3 artifacts plus the figure. Returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    paths["matched_population"] = _write_csv(
        out_dir / "matched_population.csv", MATCHED_POPULATION_COLUMNS, matched)
    paths["unmatched_population"] = _write_csv(
        out_dir / "unmatched_population.csv", UNMATCHED_POPULATION_COLUMNS, unmatched)
    paths["meta_oof"] = _write_csv(
        out_dir / "fusion_meta_oof.csv", META_OOF_COLUMNS,
        [row for r in results for row in r.meta_oof_rows])
    paths["base_selection"] = _write_csv(
        out_dir / "fusion_base_selection.csv", BASE_SELECTION_COLUMNS,
        [row for r in results for row in r.selection_rows])
    paths["fit_audit"] = _write_csv(
        out_dir / "fit_audit_g.csv", FIT_AUDIT_COLUMNS,
        [row for r in results for row in r.fit_audit_rows])
    paths["alpha_grid"] = _write_csv(
        out_dir / "fusion_alpha_grid.csv", ALPHA_GRID_COLUMNS,
        [row for r in results for row in r.alpha_grid_rows])
    paths["predictions"] = _write_csv(
        out_dir / "predictions_g.csv", PREDICTIONS_COLUMNS,
        [row for r in results for row in r.prediction_rows])
    paths["per_subject"] = _write_csv(
        out_dir / "per_subject_g.csv", PER_SUBJECT_COLUMNS, subject_rows)
    paths["exclusions"] = _write_csv(
        out_dir / "exclusions_g.csv", EXCLUSIONS_COLUMNS,
        [row for r in results for row in r.exclusions])

    # The metrics JSON names the provenance tables by content hash, so a summary can never be
    # read next to a base-selection or fit-audit table it was not computed against.
    payload = dict(summary)
    payload["artifact_sha256"] = {
        "fusion_base_selection.csv": _file_sha256(paths["base_selection"]),
        "fit_audit_g.csv": _file_sha256(paths["fit_audit"]),
    }
    metrics_path = out_dir / "metrics_exp_g.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics"] = metrics_path

    paths["figure"] = _fusion_figure(subject_rows, out_dir / "fusion_comparison.png")
    return paths


def run_and_report(config_10, config_77, matched, sessions_10, sessions_77, unmatched, store_dir,
                   run_dir, *, mode, analysis_commit, n_workers=1) -> dict:
    """Validate both stores, run Exp G, and cross the reporting boundary.

    `mode="full"` writes every artifact; `mode="smoke"` is MECHANISM-ONLY — the identical
    selection/fusion path runs but no performance value is surfaced, only the structural
    run-log, matching the A/B/C/H doctrine.
    """
    for band, config, sessions in (
        ("10ghz", config_10, sessions_10), ("77ghz", config_77, sessions_77),
    ):
        store_mod.validate_store(
            band, store_dir, expected_fingerprints(config, band, sessions),
            analysis_commit=analysis_commit,
        )

    seeds = config_10.run.seed_set
    results = run_exp_g(config_10, config_77, sessions_10, sessions_77, store_dir,
                        seeds=seeds, n_workers=n_workers)
    _assert_mechanism_ok(results, sessions_10)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        log = run_dir / "run_log_exp_g.json"
        log.write_text(json.dumps({
            "stage": "exp-g-smoke", "mode": "mechanism-only",
            "n_matched_cells": len(matched),
            "n_subjects_g": len({c["subject"] for c in matched}),
            "n_outer_folds": len(results),
            "n_evaluable_outer_folds": sum(1 for r in results if r.reason is None),
            "exclusion_reasons": sorted({r.reason for r in results if r.reason is not None}),
            "note": "performance values suppressed -- mechanism-only smoke",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"run_log": log}

    subject_rows = per_subject_rows(results)
    if not subject_rows:
        raise ExpGError(
            "no outer fold produced a fused prediction — Experiment G has no estimand to report"
        )
    population = population_summary(matched, unmatched)
    summary = summarize_exp_g(results, subject_rows, population, config_10)
    return write_exp_g_reports(results, matched, unmatched, subject_rows, summary, run_dir)


def _assert_mechanism_ok(results, sessions_10) -> None:
    """Structural checks that reveal no performance value.

    The load-bearing ones: every reported outer fold holds out exactly one subject, that subject
    never appears in a fitted subject set at any of the three levels, and every alpha came from
    a grid the config froze rather than from anything the outer test rows could reach.
    """
    subjects = sorted({int(s["subject"]) for s in sessions_10})
    folds = {f.test_subject: f for f in nested_loso_splits(subjects)}
    for result in results:
        assert result.test_subject in folds
        fold = folds[result.test_subject]
        assert result.test_subject not in fold.train_subjects
        for row in result.fit_audit_rows:
            fitted = json.loads(row["fitted_subjects_json"])
            predicted = json.loads(row["predicted_subjects_json"])
            assert result.test_subject not in fitted, row
            assert not set(fitted) & set(predicted), row
        if result.reason is not None:
            assert result.reason in EXCLUSION_REASONS
            assert not result.prediction_rows
            continue
        assert result.alpha is not None
        assert result.prediction_rows
        for row in result.prediction_rows:
            assert row["subject"] == result.test_subject
