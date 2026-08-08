"""Experiment H — the selection-variance robustness bootstrap (MILESTONE_10_PLAN.md §2.4).

The question this answers is narrow and worth stating plainly: Experiments A, B and C each
*select* a model inside every outer fold, so their headline numbers are the output of a
procedure, not of a fixed estimator. How much would that number move if the 16-subject cohort
had been a different draw of 16 subjects? `R = 200` times we draw N subject IDs with
replacement, re-run the **complete** procedure on that drawn cohort — selection included — and
record the estimand. The spread of those 200 numbers is the selection-variance range.

Three things make this different from an ordinary bootstrap CI and are why the module reads the
way it does:

  * **It refits.** Nothing here reuses a stored prediction. Every replicate re-enumerates
    candidates, re-scores inner folds, re-selects and refits, through `exp_a.run_exp_a`,
    `exp_b.run_exp_b` and `exp_c.run_exp_c` unchanged. This module owns the resampling and the
    bookkeeping; it owns no candidate enumeration, no tie-break and no fold construction
    (plan §4.2 step 3: "robustness never copies their candidate enumeration or selection logic").
  * **It is not a BCa interval** (A-M10-5). The output is labelled
    `selection_variance_empirical_95pct_range` and is the plain empirical 2.5th/97.5th
    percentile pair of the successful replicates. Applying BCa to an arbitrary vector of
    already-bootstrapped estimates is invalid, so the label is load-bearing, not cosmetic.
  * **A replicate is all-or-nothing.** The distinct-subject and all-five-classes checks are
    coarse prechecks; if any nested selection has no surviving candidate, any outer prediction
    is absent, or Exp B's four-session aggregate is unavailable, the whole result replicate is
    skipped with the first canonical reason and counted. It is never summarized over the folds
    that happened to work — that would report the easy folds' variance as the procedure's.

Multiplicity reaches the whole procedure through the step-2 foundation: the `{subject: m_s}`
MAPPING is passed down, and each provider/harness call expands it against **its own** bundle's
rows (Exp B drops degenerate sessions, so a row-aligned array built once here would attach
counts to the wrong rows silently — see `tests/test_multiplicity.py::
test_multiplicity_stays_aligned_when_a_provider_drops_rows`).

LOSO roles are built over the DISTINCT drawn subjects, so every copy of one original subject
always has one role — a drawn subject is never simultaneously trained on and held out.

`R = 200` full refits is a large job (~1,200 core-hours per experiment-band, ~2,000+ for Exp C),
so the second half of this module is the shard/merge layer: `run_replicate_range` computes a
contiguous replicate range, `write_shard`/`read_shards` move it between SLURM array tasks, and
the merge summarizes. That split is science-neutral for one specific reason — each replicate's
cohort is a pure function of its own seed tuple, so a replicate range is a complete unit of work
and no fit can depend on which process ran it. What sharding *can* get wrong is bookkeeping, and
`read_shards` refuses all of it by name: gaps, overlaps, and any lineage difference.

**A-M10-10 (raised here, 2026-08-08).** `robustness_selection.csv` records the SELECTED
candidate of each stage rather than one row per enumerated candidate, and
`fit_audit_robustness.csv` records the outer-level fits the reused orchestration returns rather
than also the inner-CV fits. See `_selection_rows` and `_fit_audit_rows` for the full reason.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..features import store as store_mod
from . import exp_a, exp_b, exp_c, fold_parallel
from . import metrics as M
from .exp_b import ExpBError
from .exp_c import ExpCError, ExpCProtocolError
from .reference_gate import json_sha256, winner_active
from .selection import SelectionError

ROBUSTNESS_SCHEMA_VERSION = 1

# The versioned enum plan §2.4 freezes for the RNG tuple. Values, not positions, are the
# contract: renumbering these would silently re-draw every replicate in the milestone.
EXPERIMENT_CODE = {"a": 1, "b": 2, "c": 3}
BAND_CODE = {"10ghz": 10, "77ghz": 77}

EXPERIMENTS = ("a", "b", "c")
BANDS = ("10ghz", "77ghz")

# The estimands §2.4 registers, per experiment, in report order. One draw is shared by every
# entry of one experiment-band replicate, so these are always computed together from a single
# run of the procedure — never from separate re-draws.
ESTIMANDS = {
    "a": ("selected_radar_subject_balanced_mae", "radar_minus_session_index_mae"),
    "b": ("radar_minus_baseline_equal_session_aggregate",),
    "c": ("arm_a_class_unit_mae", "arm_b_class_unit_mae"),
}

# The canonical skip reasons, IN PRECEDENCE ORDER. A replicate reports the FIRST that applies
# (plan §2.4), so this tuple is the definition of "first" and the order is part of the
# contract, not an implementation detail.
SKIP_REASONS = (
    "insufficient_distinct_subjects",
    "ordinal_missing_classes",
    "no_surviving_candidate",
    "missing_outer_prediction",
    "expb_primary_aggregate_unavailable",
    "non_finite_estimate",
)

# Exp B's primary aggregate is defined over exactly these four sessions (S0 is excluded at the
# spine). A replicate whose out-of-fold rows miss any of them has no four-session aggregate.
EXPB_PRIMARY_SESSIONS = (1, 2, 3, 4)

ORDINAL_CLASSES = (0, 1, 2, 3, 4)

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
CONCLUSIVE = "conclusive"
INCONCLUSIVE = "inconclusive"

# The label A-M10-5 requires. Never "bca": these endpoints are percentiles of an already-
# bootstrapped vector, and BCa needs the observed statistic plus an original-unit jackknife.
RANGE_LABEL = "selection_variance_empirical_95pct_range"

# How a fit's rows were weighted, for the audit table. "row_duplication" is what the estimator
# dispatch reports under A-M10-8; "multiplicity_weighted" covers the fitted quantities that
# consume m_s without duplicating rows (the tuned-ε median repeats per-subject SCALES, the
# session means use m_s as subject-copy weights); "none" is an unresampled fit.
WEIGHTING_NONE = "none"
WEIGHTING_MULTIPLICITY = "multiplicity_weighted"


class RobustnessError(ValueError):
    """A malformed robustness request (unknown experiment/band, or an unusable cohort)."""


# ----------------------------------------------------------------------- the RNG freeze


def robustness_seed(config) -> int:
    """The bootstrap's root seed.

    **Owner decision, 2026-08-08:** the root seed is `config.run.seed` (20260721), the run-level
    seed every other stochastic quantity in this project already derives from. `StatsConfig`
    carries the four `robustness_*` thresholds but deliberately gains no `robustness_seed`
    field: the M6 sections are frozen records, and adding one would be a config change made to
    express a value that already exists. Pinned by `tests/test_robustness.py` so it can never
    become an accident of whichever attribute an implementation reached for first.
    """
    return int(config.run.seed)


def model_seeds(config):
    """The configured model seed set.

    Named rather than inlined because §2.4 makes a point of it: candidate model seeds remain
    the configured seeds and are NOT derived from the resampling seed. A replicate changes
    which subjects are drawn, never which seeds are fit.
    """
    return config.run.seed_set


def replicate_seed_tuple(config, experiment: str, band: str, replicate: int) -> tuple:
    """`[robustness_seed, experiment_code, band_code, replicate]` — plan §2.4's frozen tuple.

    The experiment and band codes sit in the tuple (rather than the replicate index alone)
    precisely so the six jobs of the launch matrix draw *different* cohorts: a shared draw
    across experiments would correlate their ranges for no scientific reason.
    """
    if experiment not in EXPERIMENT_CODE:
        raise RobustnessError(f"unknown experiment {experiment!r} (expected one of {EXPERIMENTS})")
    if band not in BAND_CODE:
        raise RobustnessError(f"unknown band {band!r} (expected one of {BANDS})")
    return (robustness_seed(config), EXPERIMENT_CODE[experiment], BAND_CODE[band], int(replicate))


def draw_subject_multiplicity(subjects, seed_tuple) -> tuple[dict, str]:
    """Draw N subject IDs with replacement and return ({subject: m_s}, generated seed state).

    N is the number of ORIGINAL subjects, so the drawn cohort is the same size as the cohort it
    stands in for. The 128-bit state generated from the `SeedSequence` is returned (and saved)
    alongside the tuple because §2.4 asks for both: the tuple says what was requested, the state
    says what NumPy actually produced from it, which is what a future NumPy would have to
    reproduce for the replicates to be identical.

    The draw itself is `rng.integers(0, n, size=n)` over the sorted subject list — the same
    idiom `metrics._cluster_bootstrap_over_rows` already uses for its subject-cluster
    resampling, so the project has one way of drawing subjects with replacement.
    """
    ordered = sorted(int(s) for s in subjects)
    if not ordered:
        raise RobustnessError("cannot draw a bootstrap replicate from an empty subject pool")
    sequence = np.random.SeedSequence([int(v) for v in seed_tuple])
    state = sequence.generate_state(4, dtype=np.uint32)      # 4 x 32 bits = the 128-bit state
    rng = np.random.default_rng(sequence)

    n = len(ordered)
    counts: dict[int, int] = {}
    for j in rng.integers(0, n, size=n):
        subject = ordered[int(j)]
        counts[subject] = counts.get(subject, 0) + 1
    return dict(sorted(counts.items())), "".join(f"{int(v):08x}" for v in state)


def canonical_multiplicity(multiplicity) -> list:
    """`[[subject, m_s], ...]`, sorted, plain ints — the one serialization of a draw.

    A dict has no canonical serialization of its own, and the fit-audit table joins fits to
    draws by a hash of this form, so the canonicalization lives in exactly one place.
    """
    return [[int(s), int(m)] for s, m in sorted(dict(multiplicity or {}).items())]


def multiplicity_sha256(multiplicity) -> str:
    return json_sha256(canonical_multiplicity(multiplicity))


def _restricted(multiplicity, subjects) -> dict:
    """The part of a draw one fit actually saw: m_s over that fit's own training subjects."""
    if multiplicity is None:
        return {}
    return {int(s): int(multiplicity.get(int(s), 1)) for s in sorted(int(x) for x in subjects)}


# ------------------------------------------------------------------ the experiment spine


def build_spine(config, experiment: str, band: str) -> list[dict]:
    """The experiment's own session records, under its own frozen eligibility rule.

    Exp B excludes S0 at the source; Exp C keeps all five sessions and adds the ordinal
    columns. Reusing the three builders (rather than filtering one spine here) is what keeps
    the resampled cohort identical in shape to the ordinary run of that experiment.
    """
    if experiment == "a":
        return exp_a.build_sessions(config, band)
    if experiment == "b":
        return exp_b.build_sessions_b(config, band)
    if experiment == "c":
        return exp_c.build_sessions_c(config, band)
    raise RobustnessError(f"unknown experiment {experiment!r} (expected one of {EXPERIMENTS})")


def spine_subjects(experiment: str, sessions) -> list[int]:
    """The subject pool the draw samples from — again each experiment's own rule."""
    if experiment == "b":
        return exp_b.evaluable_subjects_b(sessions)
    if experiment == "c":
        return exp_c.evaluable_subjects_c(sessions)
    return sorted({int(s["subject"]) for s in sessions})


def _sessions_for(sessions, drawn_subjects) -> list[dict]:
    keep = {int(s) for s in drawn_subjects}
    return [s for s in sessions if int(s["subject"]) in keep]


# ------------------------------------------------------------------------ one replicate


@dataclass
class ReplicateOutcome:
    """One (experiment, band, replicate): its draw, its status, and everything it produced."""

    experiment: str
    band: str
    replicate: int
    seed_tuple: tuple
    generated_seed_state: str
    multiplicity: dict
    n_distinct_subjects: int
    status: str
    skip_reason: str | None = None
    skip_detail: str | None = None
    estimates: dict = field(default_factory=dict)          # arm_or_contrast -> float
    selection_rows: list = field(default_factory=list)
    fit_audit_rows: list = field(default_factory=list)
    # {"subject_sets": {sha: [ids]}, "multiplicity_maps": {sha: [[s, m], ...]}} — the companion
    # JSON's content for this replicate, so a hash in the CSV always resolves to a real map.
    audit_maps: dict = field(default_factory=dict)


def _run_procedure(config, experiment, band, sessions, store_dir, seeds, multiplicity,
                   n_workers=1):
    """Run one complete A/B/C pass on `sessions` under `multiplicity` (or `None`).

    A REPLICATE always passes `n_workers=1`: robustness parallelises at the replicate level,
    which is the coarser and more efficient axis (200 independent replicates vs ~10 folds), and
    nesting a spawn pool inside a spawn-pool worker is not something to rely on. The full-cohort
    POINT estimate is the one caller that may raise it — it is a single ordinary run with no
    replicate to parallelise against, and A/B/C guarantee fold parallelism is bit-identical to
    the serial run.
    """
    if experiment == "a":
        session_index = np.array([int(s["session_idx"]) for s in sessions])
        return exp_a.run_exp_a(
            config, band, sessions, store_dir, seeds=seeds, session_index=session_index,
            n_workers=n_workers, subject_multiplicity=multiplicity,
        )
    if experiment == "b":
        return exp_b.run_exp_b(
            config, band, sessions, store_dir, seeds=seeds, n_workers=n_workers,
            subject_multiplicity=multiplicity,
        )
    return exp_c.run_exp_c(
        config, band, sessions, store_dir, seeds=seeds, n_workers=n_workers,
        subject_multiplicity=multiplicity,
    )


def _weights_for(subject_ids, multiplicity):
    """Per-DISTINCT-subject weights m_s, or `None` when nothing was resampled.

    `None` matters: it routes the caller to `np.mean` rather than `np.average(weights=1)`, so
    the full-cohort point estimate is byte-identical to the one `summarize_exp_a` reports.
    """
    if multiplicity is None:
        return None
    return np.array([float(multiplicity.get(int(s), 1)) for s in subject_ids], dtype=float)


def _mean(values, weights):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if weights is None else float(np.average(values, weights=weights))


def _row_copies(subjects, multiplicity):
    if multiplicity is None:
        return None
    return np.array([int(multiplicity.get(int(s), 1)) for s in np.asarray(subjects)], dtype=int)


def _repeat(array, copies):
    return np.asarray(array) if copies is None else np.repeat(np.asarray(array), copies, axis=0)


# -- the estimands ---------------------------------------------------------------------
#
# Each returns (estimates, skip_reason). The estimand definitions are restated in the one place
# a replicate computes them, so a reader can check them against §2.4 without tracing back
# through three summarizers.
#
# `required_subjects=None` turns OFF the coverage requirement, and only the full-cohort point
# estimate passes it. The requirement is §2.4's all-or-nothing rule for a REPLICATE ("every
# required distinct outer subject must produce its complete OOF result ... never computed over
# the remaining easier folds"), and it is deliberately not imposed on the original run: Exp B
# legitimately records a fold with `reason="no_surviving_test_rows"` in its exclusion ledger and
# `summarize_exp_b` still reports the primary aggregate over the folds that contributed. Applying
# the replicate rule there would make the point estimate unobtainable in exactly the case where
# the ordinary Exp B run reports one — and the point has to BE the number that run reports.
# Exp B's four-session viability check is not part of this and applies in both cases, because
# `summarize_exp_b` applies it too.


def _exp_a_estimates(results, required_subjects, multiplicity):
    """Selected radar subject-balanced MAE, and radar − session-index MAE.

    Both weight each DISTINCT subject by m_s (§2.4: "outer replicate summaries weight each
    subject by m_s"). Repeating rows would not do it — the metric is subject-BALANCED, so a
    subject's own mean is unchanged by duplicating its sessions; the weight has to enter at
    the across-subject average, which is what a duplicated cohort's value actually is.
    """
    if required_subjects is not None:
        covered = {int(r.test_subject) for r in results if len(r.test_targets)}
        if covered != set(required_subjects):
            return None, "missing_outer_prediction"

    subjects, y_true, pred_by_seed, base_pred = exp_a._per_seed_matrix(results)
    n_seeds = pred_by_seed.shape[0]
    subject_ids = sorted(set(subjects.tolist()))

    radar, baseline = [], []
    for s in subject_ids:
        rows = subjects == s
        radar.append(float(np.mean([
            np.abs(y_true[rows] - pred_by_seed[k, rows]).mean() for k in range(n_seeds)
        ])))
        baseline.append(float(np.abs(y_true[rows] - base_pred[rows]).mean()))

    weights = _weights_for(subject_ids, multiplicity)
    radar = np.array(radar, dtype=float)
    baseline = np.array(baseline, dtype=float)
    return {
        "selected_radar_subject_balanced_mae": _mean(radar, weights),
        "radar_minus_session_index_mae": _mean(radar - baseline, weights),
    }, None


def _exp_b_estimates(results, required_subjects, multiplicity):
    """The primary equal-session aggregate, radar − baseline.

    A POOLED/nonlinear estimand (an average of per-session averages), so §2.4's general rule
    applies: repeat the evaluation rows by m_s, then compute the metric. The reference
    prediction is evaluated on the SAME repeated rows, so the difference stays genuinely paired.
    """
    if required_subjects is not None:
        if any(r.reason is not None for r in results):
            return None, "missing_outer_prediction"
        covered = {int(r.test_subject) for r in results if len(r.test_targets)}
        if covered != set(required_subjects):
            return None, "missing_outer_prediction"

    subjects, session_idx, y_true, pred_by_seed, base = exp_b._oof_matrix(results)
    # The run-level viability rule `summarize_exp_b` applies, now per replicate: a missing
    # session would make `equal_session_residual_mae` silently report a three-session mean
    # labelled as the four-session primary.
    if not set(EXPB_PRIMARY_SESSIONS).issubset(set(session_idx.tolist())):
        return None, "expb_primary_aggregate_unavailable"

    copies = _row_copies(subjects, multiplicity)
    sessions_rep = _repeat(session_idx, copies)
    y_rep = _repeat(y_true, copies)
    reference = M.equal_session_residual_mae(None, y_rep, _repeat(base, copies), sessions_rep)

    per_seed = [
        M.equal_session_residual_mae(None, y_rep, _repeat(pred_by_seed[k], copies), sessions_rep)
        - reference
        for k in range(pred_by_seed.shape[0])
    ]
    return {"radar_minus_baseline_equal_session_aggregate": float(np.mean(per_seed))}, None


def _exp_c_estimates(results, required_subjects, multiplicity):
    """Class-unit MAE for arm (a) and arm (b).

    Pooled again, so the evaluation rows are repeated by m_s before the metric. Adjacent
    accuracy and QWK deliberately get no refit-robustness range (§2.4) — they keep their
    existing conditional CIs.
    """
    if required_subjects is not None:
        if any(r.reason is not None for r in results):
            return None, "missing_outer_prediction"
        covered = {int(r.test_subject) for r in results if len(r.test_classes)}
        if covered != set(required_subjects):
            return None, "missing_outer_prediction"

    estimates = {}
    for arm in exp_c.ARMS:
        subjects, _session_idx, y_true, pred_by_seed = exp_c._oof_matrix_c(results, arm)
        copies = _row_copies(subjects, multiplicity)
        y_rep = _repeat(y_true, copies)
        per_seed = [
            M.class_unit_mae(y_rep, _repeat(pred_by_seed[k], copies))
            for k in range(pred_by_seed.shape[0])
        ]
        estimates[f"arm_{arm}_class_unit_mae"] = float(np.mean(per_seed))
    return estimates, None


ESTIMATORS = {"a": _exp_a_estimates, "b": _exp_b_estimates, "c": _exp_c_estimates}


# -- provenance rows -------------------------------------------------------------------


def _feature_axes(config, band, feature_key) -> dict:
    """The winning feature key's protocol axes, from Exp A's own `active` builders.

    `model_family` is stripped: it has its own column here, and Exp C arm (b)'s real `active`
    record legitimately carries no `model_family` at all (`REQUIRED_ACTIVE_KEYS_C`), so
    leaving Exp A's ridge default in the JSON would misdescribe half of Exp C's rows.
    """
    axes = winner_active(config, band, tuple(feature_key))
    axes.pop("model_family", None)
    return axes


def _selection_rows(config, experiment, band, replicate, results, multiplicity):
    """The selected candidate of each search stage, per outer fold, per estimand.

    **A-M10-10.** These are WINNER rows, not one row per enumerated candidate. §4.2 step 3
    requires this module to reuse A/B/C orchestration unchanged, and `run_exp_a`/`run_exp_b`/
    `run_exp_c` return the selected candidate and the outer-train fit records only — each fold
    worker discards its per-candidate `StageOutcome` after selecting. Recording every candidate
    would mean changing all three fold-result shapes and shipping ~113 candidate records x ~10
    folds x 200 replicates through the spawn-pool pickle and onto disk (order 10^6 rows across
    the six launch-matrix jobs). §5.5's own acceptance criterion asks that "every successful
    real robustness estimate resolves to complete winner/feature and fit-audit rows", which the
    winner-level table satisfies. The consequence is visible in the data: `inner_score` and
    `inner_score_variance` are blank for every experiment and `n_inner_folds` is populated only
    for Exp C, which alone returns its winner's evaluable-inner-fold count.

    Rows are emitted once per ESTIMAND so that each estimand's provenance is complete on its own
    rows and joins to `robustness_replicates.csv` on the full
    (experiment, band, arm_or_contrast, replicate) key. Exp A's two estimands and Exp C's shared
    Stage 1 therefore repeat identical content under different `arm_or_contrast` values.
    """
    sha = multiplicity_sha256(multiplicity)
    seeds_json = json.dumps([int(s) for s in model_seeds(config)])

    def row(estimand, test_subject, stage, feature_key, family, params, n_inner_folds):
        return {
            "experiment": experiment,
            "band": band,
            "arm_or_contrast": estimand,
            "replicate": replicate,
            "outer_test_subject": int(test_subject),
            "stage": stage,
            "candidate": f"{family}@{tuple(feature_key)}",
            "feature_key": str(tuple(feature_key)),
            "active_axes_json": json.dumps(_feature_axes(config, band, feature_key), sort_keys=True),
            "family": family,
            "params_json": json.dumps({str(k): v for k, v in dict(params).items()}, sort_keys=True),
            "inner_score": "",
            "inner_score_variance": "",
            "n_inner_folds": "" if n_inner_folds is None else int(n_inner_folds),
            "model_seeds_json": seeds_json,
            "selected": True,
            "multiplicity_sha256": sha,
        }

    rows = []
    if experiment in ("a", "b"):
        for r in results:
            if getattr(r, "reason", None) is not None:
                continue
            for estimand in ESTIMANDS[experiment]:
                rows.append(row(estimand, r.test_subject, "stage2_final", r.selected_feature_key,
                                r.selected_family, r.selected_params, None))
        return rows

    for r in results:
        if r.reason is not None:
            continue
        for arm in exp_c.ARMS:
            estimand = f"arm_{arm}_class_unit_mae"
            arm_result = r.arm_result(arm)
            rows.append(row(estimand, r.test_subject, "stage1", r.stage1_feature_key,
                            "ord_a_ridge", r.stage1_selected_params,
                            r.stage1_n_evaluable_inner_folds))
            rows.append(row(estimand, r.test_subject, f"stage2_arm_{arm}",
                            arm_result.selected_feature_key, arm_result.selected_family,
                            arm_result.selected_params, arm_result.n_evaluable_inner_folds))
    return rows


def _fit_audit_rows(experiment, band, replicate, results, multiplicity):
    """One row per real outer-level fit, plus the canonical maps its hashes key into.

    **A-M10-10** again: these are the `final_fits` the reused orchestration returns — the
    outer-train scaler, model (one per realized seed), tuned-ε, Exp-B session-mean and Exp-A
    session-index-baseline records. The inner-CV `FitRecord`s live on `InnerResult`s that each
    fold worker discards, so they are not reachable from here without changing all three
    fold-result shapes.

    `weighting_mode` and `effective_weighted_row_count` come from the record's own params, which
    step 2 populates for the scaler/model fits that went through the expansion dispatch. A fit
    that consumes m_s WITHOUT duplicating rows (the tuned-ε median repeats per-subject scales;
    the session means use m_s as subject-copy weights) reports `multiplicity_weighted` and an
    empty effective count rather than an invented one — its multiplicity hash is still the draw
    restricted to its own training subjects, so it joins to the companion JSON like any other.
    """
    rows: list[dict] = []
    subject_sets: dict[str, list] = {}
    multiplicity_maps: dict[str, list] = {}

    def emit(estimand, test_subject, records):
        for record in records:
            fitted = sorted(int(s) for s in record.subjects)
            fitted_sha = json_sha256(fitted)
            subject_sets[fitted_sha] = fitted
            # The invariant this whole table exists to police, checked where the subject set is
            # still in hand rather than after it has been reduced to a hash.
            if int(test_subject) in fitted:
                raise RobustnessError(
                    f"held-out subject {test_subject} appears in the fitted subject set of "
                    f"{record.quantity!r} ({record.role}) — replicate {replicate}"
                )

            counts = record.params.get("multiplicity_counts")
            if counts is not None:
                fit_map = dict(zip(record.params["multiplicity_subjects"].tolist(), counts.tolist()))
                mode = record.params["weighting_mode"].tobytes().decode("utf-8")
                effective = float(record.params["effective_weighted_row_count"][0])
            else:
                fit_map = _restricted(multiplicity, fitted)
                mode = WEIGHTING_NONE if multiplicity is None else WEIGHTING_MULTIPLICITY
                effective = ""
            fit_sha = multiplicity_sha256(fit_map)
            multiplicity_maps[fit_sha] = canonical_multiplicity(fit_map)

            rows.append({
                "experiment": experiment,
                "band": band,
                "arm_or_contrast": estimand,
                "replicate": replicate,
                "outer_test_subject": int(test_subject),
                "stage": "outer_final",
                "quantity": record.quantity,
                "role": record.role,
                "fitted_subjects_sha256": fitted_sha,
                "multiplicity_sha256": fit_sha,
                "weighting_mode": mode,
                "effective_weighted_row_count": effective,
            })

    if experiment in ("a", "b"):
        for r in results:
            if getattr(r, "reason", None) is not None:
                continue
            for estimand in ESTIMANDS[experiment]:
                emit(estimand, r.test_subject, r.final_fits)
    else:
        for r in results:
            if r.reason is not None:
                continue
            for arm in exp_c.ARMS:
                emit(f"arm_{arm}_class_unit_mae", r.test_subject, r.arm_result(arm).final_fits)

    return rows, {"subject_sets": subject_sets, "multiplicity_maps": multiplicity_maps}


# -- the replicate worker --------------------------------------------------------------


def run_replicate(config, experiment, band, sessions, store_dir, seeds, subjects, replicate):
    """Draw, precheck, run the full procedure, and reduce to this replicate's estimands.

    Top-level and picklable so it can be the unit of a spawn-context Pool. Only the enumerated
    "no surviving candidate" errors are caught: `SelectionError` (a tie-break with nothing
    comparable) and the named wrappers Exp B and Exp C raise around it. Everything else —
    `ExpCProtocolError` above all, which means an unauthorized fit was about to run — propagates
    loudly, per the standing rule that only the pre-defined non-evaluability doctrine may
    degrade gracefully.
    """
    seed_tuple = replicate_seed_tuple(config, experiment, band, replicate)
    multiplicity, seed_state = draw_subject_multiplicity(subjects, seed_tuple)
    drawn = sorted(multiplicity)

    def skipped(reason, detail=None):
        return ReplicateOutcome(
            experiment=experiment, band=band, replicate=replicate, seed_tuple=seed_tuple,
            generated_seed_state=seed_state, multiplicity=multiplicity,
            n_distinct_subjects=len(drawn), status=STATUS_SKIPPED, skip_reason=reason,
            skip_detail=detail,
        )

    # Precheck 1 — the coarse cohort-size rule. At 4 distinct subjects every outer fold still
    # has 3 training subjects, which is `splits.DEFAULT_MIN_TRAIN_SUBJECTS`, so the whole drawn
    # cohort stays selectable; below it, folds would start dropping out silently.
    if len(drawn) < int(config.stats.robustness_min_distinct_subjects):
        return skipped("insufficient_distinct_subjects")

    replicate_sessions = _sessions_for(sessions, drawn)

    # Precheck 2 — Exp C additionally requires all five classes across the resampled cohort.
    if experiment == "c":
        present = {int(s["class_idx"]) for s in replicate_sessions}
        missing = sorted(set(ORDINAL_CLASSES) - present)
        if missing or len(present) < int(config.stats.robustness_ordinal_min_classes):
            return skipped("ordinal_missing_classes", f"missing={missing}")

    try:
        results = _run_procedure(config, experiment, band, replicate_sessions, store_dir,
                                 seeds, multiplicity)
    except ExpCProtocolError:
        raise                                   # a protocol violation is never a skip
    except (SelectionError, ExpBError, ExpCError) as err:
        return skipped("no_surviving_candidate", f"{type(err).__name__}: {err}")

    estimates, reason = ESTIMATORS[experiment](results, drawn, multiplicity)
    if reason is not None:
        return skipped(reason)
    if any(not math.isfinite(v) for v in estimates.values()):
        return skipped("non_finite_estimate")

    fit_rows, maps = _fit_audit_rows(experiment, band, replicate, results, multiplicity)
    return ReplicateOutcome(
        experiment=experiment, band=band, replicate=replicate, seed_tuple=seed_tuple,
        generated_seed_state=seed_state, multiplicity=multiplicity,
        n_distinct_subjects=len(drawn), status=STATUS_OK, estimates=estimates,
        selection_rows=_selection_rows(config, experiment, band, replicate, results, multiplicity),
        fit_audit_rows=fit_rows, audit_maps=maps,
    )


def original_point_estimates(config, experiment, band, sessions, store_dir, seeds,
                             n_workers=1) -> dict:
    """The full-cohort point estimate, at multiplicity one, by the SAME estimand definitions.

    Two deliberate differences from a replicate, both so this is the number the ORDINARY A/B/C
    run reports rather than a near-miss of it:

      * `subject_multiplicity=None`, not an all-ones map — the all-ones path would take the
        expansion branch and swap `np.mean` for `np.average(weights=1)`, which can differ in the
        last ulp;
      * `required_subjects=None`, so §2.4's all-or-nothing REPLICATE rule is not imposed here.
        Exp B may legitimately record a fold with no surviving test rows and still report its
        primary aggregate; demanding complete coverage would make the point unobtainable in
        exactly that case. Exp B's four-session viability check still applies, because
        `summarize_exp_b` applies it.

    A non-finite point is still fatal: the replicate spread has to be a range around something.
    """
    results = _run_procedure(config, experiment, band, sessions, store_dir, seeds, None,
                             n_workers=n_workers)
    estimates, reason = ESTIMATORS[experiment](results, None, None)
    if reason is None and any(not math.isfinite(v) for v in estimates.values()):
        reason = "non_finite_estimate"
    if reason is not None:
        raise RobustnessError(
            f"the full-cohort Exp {experiment.upper()} {band} point estimate is unavailable "
            f"({reason}) — the robustness range would have nothing to be a range around"
        )
    return estimates


# ------------------------------------------------------------------------- summarizing


def empirical_range(estimates) -> tuple[float, float]:
    """The frozen endpoints: `np.quantile(x, [0.025, 0.975], method="linear")`.

    §2.4 pins the interpolation method as well as the probabilities, because NumPy offers nine
    rules and they disagree materially at n = 200 (roughly five observations per tail). The
    caller sorts by replicate id first, so the input vector is a deterministic function of the
    run rather than of completion order in the worker pool.
    """
    values = np.asarray(estimates, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975], method="linear")
    return float(low), float(high)


def summarize(config, experiment, band, outcomes, original_point, *, replicates_requested):
    """Reduce the replicate outcomes to one summary record per estimand.

    The `min_successful` threshold is read from `StatsConfig` and NEVER scaled to the number of
    replicates actually requested: the `R = 8` smoke is *supposed* to come back inconclusive,
    because that is the proof the rule is enforced rather than adapted (plan §4.2 step 3).
    """
    ordered = sorted(outcomes, key=lambda o: o.replicate)     # by replicate ID, then quantile
    successful = [o for o in ordered if o.status == STATUS_OK]
    min_successful = int(config.stats.robustness_min_successful_replicates)

    skip_counts: dict[str, int] = {}
    for o in ordered:
        if o.skip_reason is not None:
            skip_counts[o.skip_reason] = skip_counts.get(o.skip_reason, 0) + 1

    rows = []
    for estimand in ESTIMANDS[experiment]:
        values = np.array([o.estimates[estimand] for o in successful], dtype=float)
        low, high = empirical_range(values)
        rows.append({
            "experiment": experiment,
            "band": band,
            "arm_or_contrast": estimand,
            "original_point": float(original_point[estimand]),
            "r_requested": int(replicates_requested),
            "n_attempted": len(ordered),
            "n_successful": len(successful),
            "n_skipped": len(ordered) - len(successful),
            "min_successful_replicates": min_successful,
            "status": CONCLUSIVE if len(successful) >= min_successful else INCONCLUSIVE,
            "replicate_mean": float(np.mean(values)) if values.size else float("nan"),
            "replicate_median": float(np.median(values)) if values.size else float("nan"),
            "replicate_sd": float(np.std(values, ddof=0)) if values.size else float("nan"),
            "range_low": low,
            "range_high": high,
            "range_label": RANGE_LABEL,
            "ci_method": RANGE_LABEL,             # explicitly NOT "bca" (A-M10-5)
        })
    return rows, skip_counts


# ---------------------------------------------------------------------------- artifacts

REPLICATE_COLUMNS = (
    "experiment", "band", "arm_or_contrast", "replicate", "robustness_seed_tuple_json",
    "generated_seed_state", "multiplicity_json", "n_distinct_subjects", "status",
    "skip_reason", "estimate",
)
SELECTION_COLUMNS = (
    "experiment", "band", "arm_or_contrast", "replicate", "outer_test_subject", "stage",
    "candidate", "feature_key", "active_axes_json", "family", "params_json", "inner_score",
    "inner_score_variance", "n_inner_folds", "model_seeds_json", "selected",
    "multiplicity_sha256",
)
FIT_AUDIT_COLUMNS = (
    "experiment", "band", "arm_or_contrast", "replicate", "outer_test_subject", "stage",
    "quantity", "role", "fitted_subjects_sha256", "multiplicity_sha256", "weighting_mode",
    "effective_weighted_row_count",
)
SUMMARY_COLUMNS = (
    "experiment", "band", "arm_or_contrast", "original_point", "r_requested", "n_attempted",
    "n_successful", "n_skipped", "min_successful_replicates", "status", "replicate_mean",
    "replicate_median", "replicate_sd", "range_low", "range_high", "range_label", "ci_method",
)


def _write_csv(path, columns, rows) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def replicate_rows(experiment, outcomes) -> list[dict]:
    """One row per (estimand, replicate) — skipped replicates included, with a blank estimate.

    Skipped replicates are rows, not omissions: the skip count and its reason are part of the
    result, and a table that silently contained only the successes would read as if the
    procedure had always worked.
    """
    rows = []
    for outcome in sorted(outcomes, key=lambda o: o.replicate):
        for estimand in ESTIMANDS[experiment]:
            rows.append({
                "experiment": outcome.experiment,
                "band": outcome.band,
                "arm_or_contrast": estimand,
                "replicate": outcome.replicate,
                "robustness_seed_tuple_json": json.dumps(list(outcome.seed_tuple)),
                "generated_seed_state": outcome.generated_seed_state,
                "multiplicity_json": json.dumps(canonical_multiplicity(outcome.multiplicity)),
                "n_distinct_subjects": outcome.n_distinct_subjects,
                "status": outcome.status,
                "skip_reason": outcome.skip_reason or "",
                "estimate": outcome.estimates.get(estimand, ""),
            })
    return rows


def metrics_payload(config, experiment, band, summary_rows, skip_counts, original_point,
                    *, replicates_requested) -> dict:
    return {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "conditional_exploratory": True,
        "analysis": "selection_variance_robustness",
        "experiment": experiment,
        "band": band,
        "robustness_seed": robustness_seed(config),
        "seed_tuple_form": "[robustness_seed, experiment_code, band_code, replicate]",
        "experiment_code": EXPERIMENT_CODE[experiment],
        "band_code": BAND_CODE[band],
        "model_seeds": [int(s) for s in model_seeds(config)],
        "replicates_requested": int(replicates_requested),
        "replicates_configured": int(config.stats.robustness_replicates_r),
        "min_distinct_subjects": int(config.stats.robustness_min_distinct_subjects),
        "min_successful_replicates": int(config.stats.robustness_min_successful_replicates),
        "ordinal_min_classes": int(config.stats.robustness_ordinal_min_classes),
        "quantile_method": "linear",
        "quantile_probabilities": [0.025, 0.975],
        "range_label": RANGE_LABEL,
        "original_point_estimate": {k: float(v) for k, v in original_point.items()},
        "skip_reason_counts": {r: int(skip_counts.get(r, 0)) for r in SKIP_REASONS},
        "estimands": summary_rows,
        "note": (
            "Empirical percentiles of a refit bootstrap distribution, NOT a BCa interval "
            "(A-M10-5). The original point estimate is recomputed here at multiplicity one "
            "with the identical estimand definition, so point and spread are comparable. "
            "Selection and fit-audit tables record winners and outer-level fits (A-M10-10)."
        ),
    }


def write_robustness_reports(config, experiment, band, outcomes, summary_rows, skip_counts,
                             original_point, out_dir, *, replicates_requested) -> dict:
    """The five §3 artifacts plus the fit-audit companion JSON. No figures: H's plots read
    saved tables, and this stage produces no figure of its own."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(outcomes, key=lambda o: o.replicate)
    paths = {}

    paths["replicates"] = _write_csv(
        out_dir / "robustness_replicates.csv", REPLICATE_COLUMNS, replicate_rows(experiment, ordered)
    )
    paths["selection"] = _write_csv(
        out_dir / "robustness_selection.csv", SELECTION_COLUMNS,
        [row for o in ordered for row in o.selection_rows],
    )
    paths["fit_audit"] = _write_csv(
        out_dir / "fit_audit_robustness.csv", FIT_AUDIT_COLUMNS,
        [row for o in ordered for row in o.fit_audit_rows],
    )

    subject_sets: dict[str, list] = {}
    multiplicity_maps: dict[str, list] = {}
    for o in ordered:
        subject_sets.update(o.audit_maps.get("subject_sets", {}))
        multiplicity_maps.update(o.audit_maps.get("multiplicity_maps", {}))
    companion = out_dir / "fit_audit_robustness_maps.json"
    companion.write_text(json.dumps({
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "subject_sets": subject_sets,
        "multiplicity_maps": multiplicity_maps,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["fit_audit_maps"] = companion

    paths["summary"] = _write_csv(out_dir / "robustness_summary.csv", SUMMARY_COLUMNS, summary_rows)

    metrics_path = out_dir / "metrics_robustness.json"
    metrics_path.write_text(json.dumps(
        metrics_payload(config, experiment, band, summary_rows, skip_counts, original_point,
                        replicates_requested=replicates_requested),
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics"] = metrics_path
    return paths


# ---------------------------------------------------------------------- shards and merge
#
# `R = 200` full-procedure refits is a large job: measured against the Exp B anchor (a full
# 16-fold run took 01:04:20 on 16 cores with all folds in one wave, i.e. ~1 core-hour per fold),
# one replicate is ~6 core-hours and one (experiment, band) is ~1,200 core-hours — ~2,000+ for
# Exp C's two arms. A single sbatch cannot always buy that inside one wall-time.
#
# Sharding is safe here for one specific reason and it is worth being explicit about it: each
# replicate's cohort is a pure function of its own seed tuple `[robustness_seed,
# experiment_code, band_code, replicate]`, so a contiguous replicate range is a complete,
# self-contained unit of work. Splitting 1..R across array tasks cannot change a single drawn
# cohort, a single fit, or a single estimate — only which process computed it. What sharding CAN
# get wrong is bookkeeping: a missing shard, an overlapping range, or a shard produced at a
# different commit. All three are refused, by name, in `read_shards`.
#
# Shards deliberately do NOT call `record_run`: that hashes every raw file (tens of GB at
# 77 GHz), and doing it once per array task would be twenty times the I/O for one run's worth of
# provenance. The MERGE writes the authoritative run directory; each shard instead self-attests
# a small lineage block that the merge validates before it reads a single estimate.

SHARD_SCHEMA_VERSION = 1


def shard_filename(experiment, band, start, stop) -> str:
    """Zero-padded so a plain lexicographic listing is also replicate order."""
    return f"robustness_shard_{experiment}_{band}_{int(start):05d}_{int(stop):05d}.json"


def shard_lineage(config, experiment, band, subjects, replicates, analysis_commit) -> dict:
    """What every shard of one job must agree on, before any of its numbers are used.

    `config_hash` is `exp_b.config_fingerprint` — the SAME named helper the Exp B variant and
    the Exp D run groups use, never a second hashing recipe. `subjects_sha256` is here because
    two shards run against different cohorts would still agree on commit and config while
    drawing from different pools, which would silently mix two experiments' replicates.
    """
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "experiment": experiment,
        "band": band,
        "replicates_requested": int(replicates),
        "robustness_seed": robustness_seed(config),
        "model_seeds": [int(s) for s in model_seeds(config)],
        "analysis_commit": analysis_commit,
        "config_hash": exp_b.config_fingerprint(config),
        "n_subjects": len(subjects),
        "subjects_sha256": json_sha256(sorted(int(s) for s in subjects)),
    }


def _outcome_to_json(outcome: ReplicateOutcome) -> dict:
    return {
        "experiment": outcome.experiment,
        "band": outcome.band,
        "replicate": int(outcome.replicate),
        "seed_tuple": [int(v) for v in outcome.seed_tuple],
        "generated_seed_state": outcome.generated_seed_state,
        "multiplicity": canonical_multiplicity(outcome.multiplicity),
        "n_distinct_subjects": int(outcome.n_distinct_subjects),
        "status": outcome.status,
        "skip_reason": outcome.skip_reason,
        "skip_detail": outcome.skip_detail,
        "estimates": {k: float(v) for k, v in outcome.estimates.items()},
        "selection_rows": outcome.selection_rows,
        "fit_audit_rows": outcome.fit_audit_rows,
        "audit_maps": outcome.audit_maps,
    }


def _outcome_from_json(payload: dict) -> ReplicateOutcome:
    return ReplicateOutcome(
        experiment=payload["experiment"],
        band=payload["band"],
        replicate=int(payload["replicate"]),
        seed_tuple=tuple(int(v) for v in payload["seed_tuple"]),
        generated_seed_state=payload["generated_seed_state"],
        # JSON turns integer keys into strings; the mapping is int-keyed everywhere else and
        # `multiplicity[s]` is looked up by int subject id, so it has to come back as ints.
        multiplicity={int(s): int(m) for s, m in payload["multiplicity"]},
        n_distinct_subjects=int(payload["n_distinct_subjects"]),
        status=payload["status"],
        skip_reason=payload["skip_reason"],
        skip_detail=payload["skip_detail"],
        estimates={k: float(v) for k, v in payload["estimates"].items()},
        selection_rows=payload["selection_rows"],
        fit_audit_rows=payload["fit_audit_rows"],
        audit_maps=payload["audit_maps"],
    )


def write_shard(shard_dir, lineage, start, stop, outcomes) -> Path:
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / shard_filename(lineage["experiment"], lineage["band"], start, stop)
    path.write_text(json.dumps({
        "lineage": lineage,
        "replicate_start": int(start),
        "replicate_stop": int(stop),
        "outcomes": [_outcome_to_json(o) for o in sorted(outcomes, key=lambda o: o.replicate)],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_shards(shard_dir, lineage) -> list:
    """Every shard of one job, validated FAIL-CLOSED, as one replicate-ordered outcome list.

    Enumerating a directory is not the "glob discovery" §1.3 forbids: that rule is about
    silently picking a *latest* run out of several candidates. Here the contiguity check below
    means the set either covers 1..R exactly once or the merge refuses — there is nothing to
    pick, and a missing or duplicated shard is an error rather than a quieter answer.

    Refused, each by name: a lineage field that differs from this run's (commit, config hash,
    seed, cohort, R, band, experiment), a replicate id outside its own shard's declared range,
    a gap in 1..R, and an overlap.
    """
    shard_dir = Path(shard_dir)
    experiment, band = lineage["experiment"], lineage["band"]
    paths = sorted(shard_dir.glob(f"robustness_shard_{experiment}_{band}_*.json"))
    if not paths:
        raise RobustnessError(
            f"no robustness shards for {experiment}/{band} under {shard_dir} — the array stage "
            f"either did not run or wrote somewhere else"
        )

    by_replicate: dict[int, ReplicateOutcome] = {}
    owner: dict[int, str] = {}
    for path in paths:
        shard = json.loads(path.read_text(encoding="utf-8"))
        found = shard.get("lineage", {})
        for field_name, expected in lineage.items():
            if found.get(field_name) != expected:
                raise RobustnessError(
                    f"shard {path.name}: lineage field {field_name!r} mismatch "
                    f"(expected {expected!r}, found {found.get(field_name)!r})"
                )
        start, stop = int(shard["replicate_start"]), int(shard["replicate_stop"])
        for payload in shard["outcomes"]:
            outcome = _outcome_from_json(payload)
            if not start <= outcome.replicate <= stop:
                raise RobustnessError(
                    f"shard {path.name}: replicate {outcome.replicate} is outside its own "
                    f"declared range {start}..{stop}"
                )
            if outcome.replicate in by_replicate:
                raise RobustnessError(
                    f"replicate {outcome.replicate} appears in two shards "
                    f"({owner[outcome.replicate]} and {path.name}) — overlapping ranges"
                )
            by_replicate[outcome.replicate] = outcome
            owner[outcome.replicate] = path.name

    expected_ids = set(range(1, int(lineage["replicates_requested"]) + 1))
    missing = sorted(expected_ids - set(by_replicate))
    if missing:
        raise RobustnessError(
            f"the shards under {shard_dir} do not cover replicates {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''} "
            f"({len(missing)} of {len(expected_ids)} missing) — a partial set is never summarized"
        )
    extra = sorted(set(by_replicate) - expected_ids)
    if extra:
        raise RobustnessError(f"the shards contain replicates outside 1..R: {extra[:10]}")

    return [by_replicate[r] for r in sorted(by_replicate)]


# ------------------------------------------------------------------------- orchestration


def run_replicate_range(config, experiment, band, sessions, store_dir, *, seeds, start, stop,
                        n_workers=1) -> list:
    """Replicates `start..stop` inclusive, in replicate order. The unit of work everywhere.

    A whole run is the range `1..R`; an array shard is a contiguous sub-range. The two produce
    the same outcomes because each replicate's draw comes from its own seed tuple and nothing
    else.
    """
    if experiment not in EXPERIMENTS:
        raise RobustnessError(f"unknown experiment {experiment!r} (expected one of {EXPERIMENTS})")
    start, stop = int(start), int(stop)
    if start < 1 or stop < start:
        raise RobustnessError(f"replicate range must satisfy 1 <= start <= stop, got {start}..{stop}")

    subjects = spine_subjects(experiment, sessions)
    if len(subjects) < int(config.stats.robustness_min_distinct_subjects):
        raise RobustnessError(
            f"Exp {experiment.upper()} {band} has {len(subjects)} evaluable subjects, below the "
            f"frozen robustness_min_distinct_subjects="
            f"{config.stats.robustness_min_distinct_subjects} — every replicate would skip"
        )

    tasks = [
        (config, experiment, band, sessions, store_dir, seeds, subjects, r)
        for r in range(start, stop + 1)
    ]
    outcomes = fold_parallel.run_folds_parallel(
        run_replicate, tasks, n_workers, f"robustness-{experiment}-{band}", unit="replicates",
    )
    outcomes.sort(key=lambda o: o.replicate)
    return outcomes


def run_robustness(config, experiment, band, sessions, store_dir, *, seeds, replicates,
                   n_workers=1):
    """The single-job path: every replicate plus the full-cohort point, reduced to summary rows.

    Returns `(outcomes, summary_rows, skip_counts)`. Replicates are the parallel unit: they are
    independent by construction and each is frozen by its own seed tuple, so a pool changes
    wall-clock only.
    """
    outcomes = run_replicate_range(config, experiment, band, sessions, store_dir, seeds=seeds,
                                   start=1, stop=int(replicates), n_workers=n_workers)
    original_point = original_point_estimates(config, experiment, band, sessions, store_dir, seeds,
                                              n_workers=n_workers)
    summary_rows, skip_counts = summarize(
        config, experiment, band, outcomes, original_point, replicates_requested=replicates,
    )
    return outcomes, summary_rows, skip_counts


def run_and_report_shard(config, experiment, band, sessions, store_dir, shard_dir, *,
                         analysis_commit, replicates, start, stop, n_workers=1) -> dict:
    """One array task: replicates `start..stop`, written as a self-attesting shard file.

    A shard writes NO summary and NO percentile range — it cannot, and the refusal is the point:
    a range over a sub-range of replicates is not the estimand, and `min_successful` is defined
    against the whole `R`. Only the merge summarizes.
    """
    start, stop = int(start), int(stop)
    if stop > int(replicates):
        raise RobustnessError(
            f"replicate range {start}..{stop} runs past R={replicates}; the shard set must cover "
            f"exactly 1..R"
        )
    store_mod.validate_store(
        band, store_dir, exp_a.expected_fingerprints(config, band, sessions),
        analysis_commit=analysis_commit,
    )
    outcomes = run_replicate_range(config, experiment, band, sessions, store_dir,
                                   seeds=model_seeds(config), start=start, stop=stop,
                                   n_workers=n_workers)
    assert_mechanism_ok(outcomes, sessions, experiment)      # structural, not performance

    lineage = shard_lineage(config, experiment, band, spine_subjects(experiment, sessions),
                            replicates, analysis_commit)
    return {"shard": write_shard(shard_dir, lineage, start, stop, outcomes)}


def run_and_report_merge(config, experiment, band, sessions, store_dir, shard_dir, run_dir, *,
                         mode, analysis_commit, replicates, n_workers=1) -> dict:
    """Merge every shard of one job, compute the full-cohort point ONCE, and report.

    The point estimate lives here rather than in a designated shard: it is one ordinary
    full-cohort run at multiplicity `None`, so making one array task asymmetric to carry it
    would buy nothing and would make that shard's failure mode different from its siblings'.
    """
    store_mod.validate_store(
        band, store_dir, exp_a.expected_fingerprints(config, band, sessions),
        analysis_commit=analysis_commit,
    )
    lineage = shard_lineage(config, experiment, band, spine_subjects(experiment, sessions),
                            replicates, analysis_commit)
    outcomes = read_shards(shard_dir, lineage)               # fail-closed before any number is used
    assert_mechanism_ok(outcomes, sessions, experiment)

    original_point = original_point_estimates(config, experiment, band, sessions, store_dir,
                                              model_seeds(config), n_workers=n_workers)
    summary_rows, skip_counts = summarize(
        config, experiment, band, outcomes, original_point, replicates_requested=replicates,
    )
    return _report(config, experiment, band, outcomes, summary_rows, skip_counts, run_dir,
                   mode=mode, replicates=replicates)


def run_and_report_robustness(config, experiment, band, sessions, store_dir, run_dir, *, mode,
                              analysis_commit, replicates, n_workers=1) -> dict:
    """validate_store -> run every replicate -> smoke run-log, or the five full artifacts.

    `mode="smoke"` follows the A/B/C doctrine: MECHANISM-ONLY. It runs the identical resampling
    and refit path but surfaces no estimate — only the counts, the skip reasons and the
    conclusive/inconclusive status, which are mechanism rather than performance. That status is
    exactly what the `R = 8` smoke exists to demonstrate is enforced.
    """
    store_mod.validate_store(
        band, store_dir, exp_a.expected_fingerprints(config, band, sessions),
        analysis_commit=analysis_commit,
    )
    outcomes, summary_rows, skip_counts = run_robustness(
        config, experiment, band, sessions, store_dir, seeds=model_seeds(config),
        replicates=replicates, n_workers=n_workers,
    )
    assert_mechanism_ok(outcomes, sessions, experiment)      # structural, not performance
    return _report(config, experiment, band, outcomes, summary_rows, skip_counts, run_dir,
                   mode=mode, replicates=replicates)


def _report(config, experiment, band, outcomes, summary_rows, skip_counts, run_dir, *, mode,
            replicates) -> dict:
    """The single reporting boundary, shared by the single-job and merge paths."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        log = run_dir / f"run_log_robustness_{experiment}_{band}.json"
        log.write_text(json.dumps({
            "stage": "robustness-smoke", "experiment": experiment, "band": band,
            "mode": "mechanism-only",
            "n_replicates_attempted": len(outcomes),
            "n_successful": sum(1 for o in outcomes if o.status == STATUS_OK),
            "min_successful_replicates": int(config.stats.robustness_min_successful_replicates),
            "status": summary_rows[0]["status"] if summary_rows else INCONCLUSIVE,
            "skip_reason_counts": {r: int(skip_counts.get(r, 0)) for r in SKIP_REASONS},
            "note": "estimate values suppressed -- mechanism-only smoke",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"run_log": log}

    original_point = {row["arm_or_contrast"]: row["original_point"] for row in summary_rows}
    return write_robustness_reports(
        config, experiment, band, outcomes, summary_rows, skip_counts, original_point, run_dir,
        replicates_requested=replicates,
    )


def assert_mechanism_ok(outcomes, sessions, experiment) -> None:
    """Structural checks that reveal no performance value.

    The load-bearing one is the last: on every outer fold of every replicate, the held-out
    subject must be absent from every fitted subject set. Multiplicity changes how much a
    training subject weighs; it must never give one subject two roles.
    """
    pool = set(spine_subjects(experiment, sessions))
    for outcome in outcomes:
        assert set(outcome.multiplicity).issubset(pool), "drew a subject outside the spine"
        assert all(m >= 1 for m in outcome.multiplicity.values())
        # A draw is N-with-replacement over the pool, so the copies always sum back to N.
        assert sum(outcome.multiplicity.values()) == len(pool)
        if outcome.status != STATUS_OK:
            assert outcome.skip_reason in SKIP_REASONS
            continue
        assert outcome.fit_audit_rows, "a successful replicate must record its fits"
        subject_sets = outcome.audit_maps["subject_sets"]
        for row in outcome.fit_audit_rows:
            fitted = subject_sets[row["fitted_subjects_sha256"]]
            assert row["outer_test_subject"] not in fitted
            assert set(fitted).issubset(set(outcome.multiplicity))
        for row in outcome.selection_rows:
            assert row["outer_test_subject"] in outcome.multiplicity
