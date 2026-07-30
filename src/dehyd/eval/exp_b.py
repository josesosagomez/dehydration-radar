"""Experiment B — clock-decoupling: the session-mean-residualized fluid-loss regression.

The residualized-target sibling of `exp_a.py`, composed on the SAME generic harness engine
via the `score_fn`/`FeatureBundle.session_idx` hook (`harness.py`). Reuses Exp A's exact
search-space enumeration and store-backed X path (`exp_a.stage1_candidates`,
`exp_a.stage2_candidates`, `exp_a.StoreBackedFeatures`) UNCHANGED — never a second copy of
either (A-M6-3 requires one enumeration of the frozen search space) — and adds only what is
genuinely new: S0-excluded session spines, the train-only session-mean residualizing
provider, the equal-session-weighted objective, and Exp B's own reporting/statistics.

Why residualize: within a FIXED session every subject was measured at the same clock time
but lost different amounts of fluid, so predicting `Delta-m%(subj, session) - mu_s` (mu_s =
the train-only session mean) tests whether radar tracks between-subject fluid-loss variation
rather than decoding the clock — see MILESTONE_8_PLAN.md §0 for the full rationale.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..features import store as store_mod
from ..features.protocol_freeze import protocol_freeze_guard
from ..models import baselines
from . import exp_a
from . import fold_parallel
from . import harness
from . import metrics as M
from .harness import FeatureBundle, require_complete_active
from .selection import SelectionError


class ExpBError(ValueError):
    """A malformed Exp B session spine, a fold-viability failure with no usable candidate,
    or a session-specific-variant shard/provenance-lineage mismatch."""


# Exp B RNG offsets off config.run.seed -- FIXED, NAMED, never a running counter (plan §5
# trap 10): a running counter would silently re-map every downstream CI's seed the moment
# one offset is added, reordered, or made conditional, changing already-reported intervals
# with no visible diff at the call site. Exp A occupies config.run.seed + 0..3.
RNG_OFFSET_PRIMARY_RADAR = 100
RNG_OFFSET_PRIMARY_BASELINE = 101
RNG_OFFSET_PRIMARY_DIFFERENCE = 102
RNG_OFFSET_PAIRED_DIFFERENCE = 103
RNG_OFFSET_SESSION_RADAR_MAE_BASE = 110       # + session index (1..4) -> 111..114
RNG_OFFSET_SESSION_BASELINE_MAE_BASE = 120    # + session index -> 121..124
RNG_OFFSET_SESSION_DIFFERENCE_BASE = 130      # + session index -> 131..134


def _all_rng_offsets() -> list[int]:
    """Every resolved RNG offset this module uses, including per-session expansions --
    tested directly for pairwise distinctness (trap 10)."""
    fixed = [RNG_OFFSET_PRIMARY_RADAR, RNG_OFFSET_PRIMARY_BASELINE,
             RNG_OFFSET_PRIMARY_DIFFERENCE, RNG_OFFSET_PAIRED_DIFFERENCE]
    per_session = [
        base + s
        for base in (RNG_OFFSET_SESSION_RADAR_MAE_BASE, RNG_OFFSET_SESSION_BASELINE_MAE_BASE,
                      RNG_OFFSET_SESSION_DIFFERENCE_BASE)
        for s in (1, 2, 3, 4)
    ]
    return fixed + per_session


# ---------------------------------------------------------------- data spine + eligibility


def build_sessions_b(config, band) -> list[dict]:
    """Exp A's `build_sessions()`, filtered to session_idx in {1,2,3,4} -- S0 EXCLUDED AT
    THE SOURCE (its Delta-m% is identically 0, which would give every fold a free,
    perfectly-"predicted" session that deflates every MAE). Same record shape as Exp A."""
    return [s for s in exp_a.build_sessions(config, band) if s["session_idx"] != 0]


def evaluable_subjects_b(sessions) -> list[int]:
    """Subjects with >= 1 eligible S1-S4 session (implementation_plan.md:611-613) -- NOT Exp
    A's ">= 1 eligible session" rule, which would admit an S0-only subject into
    `nested_loso_splits` with zero rows once S0 is filtered, crashing downstream. `sessions`
    is assumed already S0-filtered (i.e. `build_sessions_b`'s output), so this is simply the
    distinct subject set present."""
    return sorted({int(s["subject"]) for s in sessions})


# --------------------------------------------------------------- residualizing feature path


class SessionResidualFeatures:
    """Wraps (does not subclass) `exp_a.StoreBackedFeatures` -- the X path, including its
    tuned-eps cache keyed by (feature_key, frozenset(train_subjects)), is reused byte-for-byte.

    `data_for` additionally computes the train-only session means (mu_s) via
    `baselines.session_means` (the single train-only mu_s computation, shared with the fit
    audit and, independently, with `fit_session_mean_baseline`), drops degenerate sessions'
    rows from subjects/X/y/session_idx in lockstep, and residualizes y = raw_y - mu_s on the
    kept rows. mu_s is emitted via `extra_fits`, so it is audited exactly like any other
    fitted quantity (at both CV levels, for free, via harness.py's existing `_bundle_fits`).
    """

    def __init__(self, band, sessions, store_dir, config):
        if any(int(s["session_idx"]) == 0 for s in sessions):
            raise ExpBError(
                "SessionResidualFeatures got an S0 row -- S0 must be excluded upstream, at "
                "the session-spine level (build_sessions_b), never here (trap 3: Delta-m%(S0) "
                "is identically 0, which would give every fold a free, perfectly-'predicted' "
                "session)."
            )
        self.band = band
        self.config = config
        self.base = exp_a.StoreBackedFeatures(band, sessions, store_dir, config)
        self.subjects = self.base.subjects
        self.session_idx = np.array([s["session_idx"] for s in sessions])
        self.y_raw = self.base.y
        self._drop_cache: dict = {}   # frozenset(train_subjects) -> ({session: mu_s}, dropped)

    def drop_for(self, train_subjects) -> tuple[dict, tuple]:
        """Train-only mu_s + the drop set for this train_subjects -- cached so it is computed
        exactly ONCE per (fold, train_subjects) regardless of how many candidates/stages call
        `data_for` with the same train set (candidate-independence, T-M8-provider)."""
        key = frozenset(train_subjects)
        if key not in self._drop_cache:
            self._drop_cache[key] = baselines.session_means(
                self.subjects, self.session_idx, self.y_raw, train_subjects, min_train_subjects=2
            )
        return self._drop_cache[key]

    def data_for(self, candidate, train_subjects) -> FeatureBundle:
        base_bundle = self.base.data_for(candidate, train_subjects)
        means, dropped = self.drop_for(train_subjects)
        kept = ~np.isin(self.session_idx, dropped)
        mu_row = np.array([means[int(s)] for s in self.session_idx[kept]], dtype=float)
        extra = base_bundle.extra_fits + (("session_means", {
            "indices": np.array(sorted(means), dtype=np.int64),
            "means": np.array([means[i] for i in sorted(means)], dtype=float),
            "dropped": np.array(dropped, dtype=np.int64),
        }),)
        return FeatureBundle(
            subjects=base_bundle.subjects[kept],
            X=base_bundle.X[kept],
            y=base_bundle.y[kept] - mu_row,
            extra_fits=extra,
            session_idx=self.session_idx[kept],
        )


def equal_session_objective(subjects, y_true, y_pred, session_idx) -> float:
    """Module-level (picklable under spawn) wrapper over `metrics.equal_session_residual_mae`.
    This exact function object is the `score_fn` passed into harness calls -- never a lambda
    or closure, so it survives multiprocessing pickling."""
    return M.equal_session_residual_mae(subjects, y_true, y_pred, session_idx)


# ----------------------------------------------------------------------------- staged run


@dataclass
class ExpBFoldResult:
    test_subject: int
    selected_feature_key: tuple
    selected_family: str
    selected_params: dict
    test_predictions: np.ndarray
    test_targets: np.ndarray            # residual scale
    test_session_idx: np.ndarray
    seed_outcomes: list
    baseline_predictions: np.ndarray    # == np.zeros(...) on the residual scale, by construction
    final_fits: list
    dropped_sessions_outer: tuple
    dropped_sessions_inner: tuple        # ((sorted(train_subjects), dropped), ...)
    reason: str | None = None            # non-None: this fold contributes no out-of-fold rows


def _session_train_subject_counts(provider: SessionResidualFeatures, train_subjects) -> dict:
    """{session: n distinct training subjects with a row in that session} -- diagnostic-only,
    used to name the cause when every candidate in a fold goes non-finite (trap 5)."""
    train_rows = np.isin(provider.subjects, sorted(train_subjects))
    counts: dict = {}
    for s in sorted(set(provider.session_idx[train_rows].tolist())):
        mask = train_rows & (provider.session_idx == s)
        counts[int(s)] = len(set(provider.subjects[mask].tolist()))
    return counts


def _run_single_fold_b(config, band, sessions, store_dir, fold, seeds) -> ExpBFoldResult:
    """Run ONE outer fold end to end. Top-level + picklable so it can run in a worker
    process. Builds its OWN `SessionResidualFeatures` (open npz handles are not shareable
    across processes) and pins single-threaded math, mirroring `exp_a._run_single_fold`."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        provider = SessionResidualFeatures(band, sessions, store_dir, config)
        anchor = (config.search_10ghz if band == "10ghz" else config.search_77ghz).stage1_anchor_ridge_alpha

        def before_fit(candidate):
            active = dict(candidate.active)
            require_complete_active(active)          # fail-closed completeness (C5)
            protocol_freeze_guard(config, active=active)

        # trap 1 mitigation: check surviving test rows BEFORE any fit, using the SAME
        # session_means() call (same train_subjects) the provider itself will use, so the
        # baseline and the residualization never compute two different mu_s.
        means, dropped_outer = provider.drop_for(fold.train_subjects)
        kept = ~np.isin(provider.session_idx, dropped_outer)
        test_rows_mask = (provider.subjects == fold.test_subject) & kept
        if not test_rows_mask.any():
            return ExpBFoldResult(
                test_subject=fold.test_subject,
                selected_feature_key=(),
                selected_family="",
                selected_params={},
                test_predictions=np.array([], dtype=float),
                test_targets=np.array([], dtype=float),
                test_session_idx=np.array([], dtype=int),
                seed_outcomes=[],
                baseline_predictions=np.array([], dtype=float),
                final_fits=[],
                dropped_sessions_outer=dropped_outer,
                dropped_sessions_inner=(),
                reason="no_surviving_test_rows",
            )

        try:
            s1 = harness._score_candidates_on_fold(
                exp_a.stage1_candidates(config, band, anchor), fold, seeds, before_fit,
                provider.data_for, score_fn=equal_session_objective,
            )
            w1 = harness.select_stage_winner(s1)
            s2 = harness._score_candidates_on_fold(
                exp_a.stage2_candidates(config, band, w1.feature_key, dict(w1.active)),
                fold, seeds, before_fit, provider.data_for, score_fn=equal_session_objective,
            )
            w2 = harness.select_stage_winner(s2)
        except SelectionError as err:
            # trap 5: every candidate going NaN (e.g. no session had any inner-val rows this
            # fold) is a candidate-independent cause -- name it, don't let SelectionError's
            # generic "no comparable candidate" message stand alone.
            raise ExpBError(
                f"Exp B fold test_subject={fold.test_subject}: no candidate produced a finite "
                f"equal_session_residual_mae -- likely candidate-independent (e.g. no session "
                f"had any inner-val rows this fold). dropped_sessions_outer={dropped_outer}, "
                f"per_session_outer_train_subject_counts="
                f"{_session_train_subject_counts(provider, fold.train_subjects)}"
            ) from err
        final_fits, _, test_pred, _, seed_outcomes = harness._final_refit(
            w2, fold, seeds, before_fit, provider.data_for, score_fn=equal_session_objective,
        )

        test_targets = provider.y_raw[test_rows_mask] - np.array(
            [means[int(s)] for s in provider.session_idx[test_rows_mask]], dtype=float
        )
        test_session_idx = provider.session_idx[test_rows_mask]
        baseline_pred = np.zeros_like(test_targets)   # session-mean baseline == 0 residual

        outer_key = frozenset(fold.train_subjects)
        dropped_inner = tuple(sorted(
            (tuple(sorted(ts)), d)
            for ts, (m, d) in provider._drop_cache.items()
            if ts != outer_key
        ))

        return ExpBFoldResult(
            test_subject=fold.test_subject,
            selected_feature_key=w2.feature_key,
            selected_family=w2.family,
            selected_params=w2.params(),
            test_predictions=test_pred,
            test_targets=test_targets,
            test_session_idx=test_session_idx,
            seed_outcomes=seed_outcomes,
            baseline_predictions=baseline_pred,
            final_fits=final_fits,
            dropped_sessions_outer=dropped_outer,
            dropped_sessions_inner=dropped_inner,
            reason=None,
        )


def _run_folds_parallel(config, band, sessions, store_dir, subjects, seeds, n_workers) -> list[ExpBFoldResult]:
    """Shared fold-parallel execution: build folds over `subjects`, run each independently
    (serially, or via a spawn-context Pool), reassemble in canonical test-subject order. Used
    by BOTH the pooled model (`run_exp_b`) and each session-specific search
    (`run_exp_b_one_session`) -- one execution strategy, not two.

    The pool and its heartbeat live in `fold_parallel.run_folds_parallel` since M9 step 5 (Exp C
    and Exp D run the same machinery); what stays here is what is genuinely Exp B's: which folds
    exist, what a task carries, and the canonical test-subject ordering of the reassembled
    results."""
    folds = [f for f in harness.nested_loso_splits(subjects) if f.selectable]
    tasks = [(config, band, sessions, store_dir, fold, seeds) for fold in folds]
    results = fold_parallel.run_folds_parallel(_run_single_fold_b, tasks, n_workers, "exp_b")
    results.sort(key=lambda r: r.test_subject)
    return results


def run_exp_b(config, band, sessions, store_dir, *, seeds, n_workers=1) -> list[ExpBFoldResult]:
    """Mirrors `exp_a.run_exp_a`'s fold-parallel structure and spawn-context Pool exactly
    (results sorted by test_subject for deterministic reassembly)."""
    subjects = evaluable_subjects_b(sessions)
    return _run_folds_parallel(config, band, sessions, store_dir, subjects, seeds, n_workers)


# ----------------------------------------------------------------------------- reporting


def _oof_matrix(results):
    """Concatenate (subjects, session_idx, residual y_true, per-seed y_pred, baseline zeros)
    across folds in canonical order. Folds with `reason is not None` contribute nothing.
    Asserts (subject, session) pairs are UNIQUE across the whole matrix."""
    usable = [r for r in results if r.reason is None]
    n_seeds = max((len(r.seed_outcomes) for r in usable), default=0)
    subjects, session_idx, y_true, base = [], [], [], []
    per_seed = [[] for _ in range(n_seeds)]
    for r in usable:
        n = len(r.test_targets)
        subjects += [r.test_subject] * n
        session_idx += r.test_session_idx.tolist()
        y_true += r.test_targets.tolist()
        base += r.baseline_predictions.tolist()
        for k in range(n_seeds):
            so = r.seed_outcomes[k] if k < len(r.seed_outcomes) else r.seed_outcomes[0]
            per_seed[k] += so.test_predictions.tolist()

    subjects_arr = np.array(subjects, dtype=int)
    session_idx_arr = np.array(session_idx, dtype=int)
    y_true_arr = np.array(y_true, dtype=float)
    pred_by_seed = np.array(per_seed, dtype=float).reshape(n_seeds, len(y_true))
    base_arr = np.array(base, dtype=float)

    pairs = list(zip(subjects_arr.tolist(), session_idx_arr.tolist()))
    assert len(pairs) == len(set(pairs)), "duplicate (subject, session) pair in the Exp B OOF matrix"

    return subjects_arr, session_idx_arr, y_true_arr, pred_by_seed, base_arr


def _ci_dict(c: M.BootstrapCI) -> dict:
    return {"point": c.point, "low": c.low, "high": c.high, "method": c.method,
            "n_eval": c.n_eval, "n_skipped": c.n_skipped, "unreliable": c.unreliable}


def _nan_ci_dict() -> dict:
    """A CI-shaped stand-in for an undefined estimand (e.g. an empty complete-case set) --
    NaNs, not an exception (plan §5 trap 13)."""
    return {"point": float("nan"), "low": float("nan"), "high": float("nan"), "method": "none",
            "n_eval": 0, "n_skipped": 0, "unreliable": True}


def _complete_case_subjects(subjects, session_idx) -> list[int]:
    """Subjects with ALL FOUR S1-S4 sessions present in the out-of-fold data -- the
    subject-weighted, complete-case estimand's population (the paired Wilcoxon companion)."""
    by_subject: dict[int, set] = {}
    for s, si in zip(subjects.tolist(), session_idx.tolist()):
        by_subject.setdefault(int(s), set()).add(int(si))
    return sorted(s for s, sess in by_subject.items() if {1, 2, 3, 4}.issubset(sess))


def summarize_exp_b(results, config) -> dict:
    """The Exp B headline: A-M8-1's primary session-weighted aggregate CI, the subject-
    weighted complete-case Wilcoxon companion, and the per-session Holm-4 exploratory
    breakdown. PRIMARY pooled model only -- the session-specific variant's summary is
    `summarize_variant_session` (a separate function, not a mode of this one)."""
    subjects, session_idx, y_true, pred_by_seed, base = _oof_matrix(results)
    n_seeds = pred_by_seed.shape[0]
    stats = config.stats
    seed = config.run.seed

    n_eval_by_session = {s: int(np.sum(session_idx == s)) for s in (1, 2, 3, 4)}
    n_eval_subjects_aggregate = len(set(subjects.tolist()))

    # (C4) run-level viability: distinct from A-M8-2's per-replicate skip-and-count rule --
    # this catches a session missing from the WHOLE run's out-of-fold data, which
    # equal_session_residual_mae alone would silently average away.
    primary_viable = all(n_eval_by_session[s] > 0 for s in (1, 2, 3, 4))
    primary_unavailable_reason = None
    if not primary_viable:
        missing = [s for s in (1, 2, 3, 4) if n_eval_by_session[s] == 0]
        primary_unavailable_reason = f"session(s) {missing} have zero out-of-fold rows across the full cohort"

    primary_aggregate = None
    if primary_viable:
        radar_ci = M.session_weighted_bootstrap(
            subjects, session_idx, y_true, pred_by_seed,
            b=stats.bootstrap_b, rng_seed=seed + RNG_OFFSET_PRIMARY_RADAR,
        )
        baseline_ci = M.session_weighted_bootstrap(
            subjects, session_idx, y_true, base[None, :],
            b=stats.bootstrap_b, rng_seed=seed + RNG_OFFSET_PRIMARY_BASELINE,
        )
        difference_ci = M.session_weighted_bootstrap(
            subjects, session_idx, y_true, pred_by_seed, y_pred_reference=base,
            b=stats.bootstrap_b, rng_seed=seed + RNG_OFFSET_PRIMARY_DIFFERENCE,
        )
        primary_aggregate = {
            "radar": _ci_dict(radar_ci), "baseline": _ci_dict(baseline_ci),
            "difference_radar_minus_baseline": _ci_dict(difference_ci),
        }

    # seed-averaged |residual| per row (radar) and the (seed-independent) baseline's |residual|
    # -- the shared per-row building blocks for both the paired companion and the per-session
    # exploratory breakdown below.
    seed_avg_abs_err = np.abs(y_true[None, :] - pred_by_seed).mean(axis=0) if n_seeds else np.zeros_like(y_true)
    baseline_abs_err = np.abs(y_true - base)

    complete_case = _complete_case_subjects(subjects, session_idx)
    if complete_case:
        radar_per_subj = np.array([seed_avg_abs_err[subjects == s].mean() for s in complete_case])
        base_per_subj = np.array([baseline_abs_err[subjects == s].mean() for s in complete_case])
        diffs = radar_per_subj - base_per_subj
        wstat, wp = M.wilcoxon_signed_rank(diffs)
        diff_ci_dict = _ci_dict(
            M.mean_difference_ci(diffs, b=stats.bootstrap_b, rng_seed=seed + RNG_OFFSET_PAIRED_DIFFERENCE)
        )
    else:
        wstat, wp = float("nan"), float("nan")
        diff_ci_dict = _nan_ci_dict()

    paired = {
        "n_complete_case": len(complete_case),
        "wilcoxon_statistic": wstat, "wilcoxon_p": wp,
        "mean_difference_radar_minus_baseline": diff_ci_dict,
    }

    per_session_partial: dict = {}
    per_session_p: dict = {}
    for s in (1, 2, 3, 4):
        mask = session_idx == s
        n_eval = int(mask.sum())
        if n_eval == 0:
            per_session_partial[s] = {
                "n_eval": 0, "radar_mae": _nan_ci_dict(), "baseline_mae": _nan_ci_dict(),
                "mean_difference": _nan_ci_dict(), "wilcoxon_p": float("nan"),
            }
            per_session_p[s] = float("nan")
            continue
        radar_vals, base_vals = seed_avg_abs_err[mask], baseline_abs_err[mask]
        diff_vals = radar_vals - base_vals
        wstat_s, wp_s = M.wilcoxon_signed_rank(diff_vals)
        per_session_p[s] = wp_s
        per_session_partial[s] = {
            "n_eval": n_eval,
            "radar_mae": _ci_dict(M.subject_cluster_bootstrap(
                radar_vals, b=stats.bootstrap_b, rng_seed=seed + RNG_OFFSET_SESSION_RADAR_MAE_BASE + s)),
            "baseline_mae": _ci_dict(M.subject_cluster_bootstrap(
                base_vals, b=stats.bootstrap_b, rng_seed=seed + RNG_OFFSET_SESSION_BASELINE_MAE_BASE + s)),
            "mean_difference": _ci_dict(M.subject_cluster_bootstrap(
                diff_vals, b=stats.bootstrap_b, rng_seed=seed + RNG_OFFSET_SESSION_DIFFERENCE_BASE + s)),
            "wilcoxon_p": wp_s,
        }

    holm_p = M.holm_adjusted([per_session_p[s] for s in (1, 2, 3, 4)],
                              family_size=stats.holm_family_expb_per_session)
    per_session_exploratory: dict = {"holm_family_size": stats.holm_family_expb_per_session}
    for i, s in enumerate((1, 2, 3, 4)):
        per_session_partial[s]["holm_p"] = holm_p[i]
        per_session_exploratory[str(s)] = per_session_partial[s]

    dropped_sessions = {
        "outer_by_fold": {str(r.test_subject): list(r.dropped_sessions_outer) for r in results},
        "inner": [
            {"test_subject": r.test_subject, "train_subjects": list(ts), "dropped": list(d)}
            for r in results for ts, d in r.dropped_sessions_inner
        ],
    }

    return {
        "conditional_exploratory": True,
        "estimand_primary": "session_weighted_equal_weight_per_session",
        "estimand_paired": "subject_weighted_complete_case_s1_s4",
        "n_eval_subjects_aggregate": n_eval_subjects_aggregate,
        "n_eval_by_session": {str(s): n_eval_by_session[s] for s in (1, 2, 3, 4)},
        "n_rows": int(len(y_true)),
        "n_seeds": n_seeds,
        "dropped_sessions": dropped_sessions,
        "primary_viable": primary_viable,
        "primary_unavailable_reason": primary_unavailable_reason,
        "primary_aggregate": primary_aggregate,
        "paired_subject_weighted_complete_case": paired,
        "per_session_exploratory": per_session_exploratory,
        "selection_frequency": exp_a._selection_frequency([r for r in results if r.reason is None]),
        "session_specific_variant": None,
    }


def _write_predictions_csv(results, out_path) -> None:
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "session_idx", "seed", "y_true_residual", "y_pred_residual", "baseline_pred_residual"])
        for r in results:
            if r.reason is not None:
                continue
            for so in r.seed_outcomes:
                for si, yt, yp, bp in zip(
                    r.test_session_idx, r.test_targets, so.test_predictions, r.baseline_predictions, strict=True
                ):
                    w.writerow([r.test_subject, int(si), so.seed, yt, yp, bp])


def _write_selection_table_csv(results, out_path) -> None:
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test_subject", "feature_key", "family", "params", "reason"])
        for r in results:
            w.writerow([r.test_subject, r.selected_feature_key, r.selected_family, r.selected_params, r.reason])


def _write_dropped_folds_csv(results, out_path) -> None:
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test_subject", "dropped_sessions_outer", "dropped_sessions_inner", "reason"])
        for r in results:
            w.writerow([r.test_subject, list(r.dropped_sessions_outer), list(r.dropped_sessions_inner), r.reason])


def write_exp_b_reports(results, summary, out_dir, band) -> dict:
    """metrics_exp_b_{band}.json, predictions_b_{band}.csv, selection_table_b_{band}.csv,
    dropped_sessions_{band}.csv, scatter_b_{band}.png (residual scale) -- same shapes/naming
    convention as `write_exp_a_reports`."""
    import matplotlib
    matplotlib.use("Agg")  # headless: no display
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    metrics_path = out_dir / f"metrics_exp_b_{band}.json"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics"] = metrics_path

    pred_path = out_dir / f"predictions_b_{band}.csv"
    _write_predictions_csv(results, pred_path)
    paths["predictions"] = pred_path

    sel_path = out_dir / f"selection_table_b_{band}.csv"
    _write_selection_table_csv(results, sel_path)
    paths["selection_table"] = sel_path

    dropped_path = out_dir / f"dropped_sessions_{band}.csv"
    _write_dropped_folds_csv(results, dropped_path)
    paths["dropped_sessions"] = dropped_path

    usable = [r for r in results if r.reason is None]
    if usable:
        y_true = np.concatenate([r.test_targets for r in usable])
        y_pred = np.concatenate([r.test_predictions for r in usable])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, s=18, alpha=0.7)
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("actual residual Δm% (raw − μ_s)")
        ax.set_ylabel("predicted residual Δm%")
        ax.set_title(f"Exp B predicted vs actual residual ({band})")
        scatter_path = out_dir / f"scatter_b_{band}.png"
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=120)
        plt.close(fig)
        paths["scatter"] = scatter_path
    return paths


def _assert_mechanism_ok_b(results, sessions) -> None:
    """Fold-role disjointness (as `exp_a._assert_mechanism_ok`) PLUS: no session_idx == 0 row
    anywhere in any result; every emitted fit's subject set excludes the held-out subject at
    the outer level. Reused UNCHANGED by the session-specific variant."""
    subjects_pool = evaluable_subjects_b(sessions)
    folds = harness.nested_loso_splits(subjects_pool)
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
        assert not np.any(r.test_session_idx == 0)
        for rec in r.final_fits:
            assert r.test_subject not in rec.subjects


def run_and_report_b(config, band, sessions, store_dir, run_dir, *, mode, analysis_commit, n_workers=1) -> dict:
    """validate_store -> run_exp_b -> _assert_mechanism_ok_b -> smoke (structural run-log
    only, NO performance value, matching exp_a's C9/C14 doctrine) or full reporting for the
    PRIMARY pooled model ONLY. There is no variant flag here at all: the session-specific
    variant is invoked through an entirely separate call tree, so "smoke never touches the
    variant" is structural, not a default that could be overridden."""
    store_mod.validate_store(
        band, store_dir, exp_a.expected_fingerprints(config, band, sessions), analysis_commit=analysis_commit
    )
    results = run_exp_b(config, band, sessions, store_dir, seeds=config.run.seed_set, n_workers=n_workers)
    _assert_mechanism_ok_b(results, sessions)   # structural, not performance

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        log = run_dir / f"run_log_{band}.json"
        log.write_text(json.dumps({
            "stage": "exp-b-smoke", "band": band, "mode": "mechanism-only",
            "n_folds": len(results), "n_sessions": len(sessions),
            "note": "performance values suppressed -- mechanism-only smoke",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"run_log": log}

    summary = summarize_exp_b(results, config)
    paths = write_exp_b_reports(results, summary, run_dir, band)
    return paths


# --- session-specific secondary variant (step 7/10.5; implementation_plan.md:722-724) ---


def eligible_subjects_for_session(sessions, session) -> list[int]:
    """The subjects eligible for session `session` -- sorted subject IDs. Used BOTH inside
    `run_exp_b_one_session` (to build that session's nested_loso folds) AND by
    `--init-run-group` (to populate the group provenance's authoritative
    `expected_subjects_by_session`) -- ONE definition, so fold construction and validation can
    never silently diverge."""
    return sorted({int(s["subject"]) for s in sessions if int(s["session_idx"]) == session})


def config_fingerprint(config) -> str:
    """sha256(json.dumps(config_to_dict(config), sort_keys=True)) -- `config_to_dict` is the
    SAME function `provenance.record_run` imports from `..config` and uses to populate
    `payload["config"]`, so this hash is guaranteed byte-identical-content-equivalent to what
    `provenance.json`'s own "config" field holds. ONE named helper, called identically by
    `--init-run-group` and every `--session` task -- never two independently-invented hashing
    recipes that could silently diverge."""
    from ..config import config_to_dict

    return hashlib.sha256(json.dumps(config_to_dict(config), sort_keys=True).encode("utf-8")).hexdigest()


def run_exp_b_one_session(config, band, sessions, store_dir, session, *, seeds, n_workers=1) -> list[ExpBFoldResult]:
    """THE REAL UNIT OF WORK (C11): one session s's fully independent nested-LOSO search, over
    a provider that keeps ONLY session s's rows (residualized by that session's own train-only
    mu_s, via the same `baselines.session_means` single-source-of-truth) and the SAME
    `equal_session_objective`, which degenerates to plain single-session residual MAE when only
    one session is present -- no new objective needed. Only
    `eligible_subjects_for_session(sessions, session)` enters its own nested-LOSO folds -- a
    variable count <= 15, per the frozen spec -- so this session's outer folds are distinct
    from the pooled model's and from every other session's. This is what a single SLURM array
    task runs directly; cross-session concurrency comes from the array, not from anything
    in-process. ANY UNEXPECTED EXCEPTION PROPAGATES (C12): this function never catches a
    generic exception to produce a placeholder/partial result; only the harness's own
    pre-defined non-evaluability doctrine may degrade gracefully, unchanged from the primary
    path."""
    session_rows = [s for s in sessions if int(s["session_idx"]) == session]
    subjects = eligible_subjects_for_session(sessions, session)
    return _run_folds_parallel(config, band, session_rows, store_dir, subjects, seeds, n_workers)


def run_exp_b_session_specific(config, band, sessions, store_dir, *, seeds, n_workers=1) -> dict[int, list]:
    """Sequential convenience wrapper: {s: run_exp_b_one_session(..., s, ...) for s in
    (1,2,3,4)}. Used ONLY by the synthetic-store test (T-M8-variant), where running four tiny
    searches sequentially is fine -- NOT the real-IBEX path (C11). The real path is four
    separate array tasks each calling `run_exp_b_one_session` directly."""
    return {
        s: run_exp_b_one_session(config, band, sessions, store_dir, s, seeds=seeds, n_workers=n_workers)
        for s in (1, 2, 3, 4)
    }


def summarize_variant_session(results_s, session, config) -> dict:
    """The per-session summary for ONE session-specific model: {n_eval, radar_mae, baseline_mae,
    mean_difference, selection_frequency}. `selection_frequency` reuses
    `exp_a._selection_frequency` UNCHANGED -- the mandatory selection-stability table applies to
    every experiment, this variant included. Still DESCRIPTIVE ONLY on the inferential side --
    deliberately NO p-value (C16): the frozen protocol defines Holm-4 for the PRIMARY model's own
    per-session breakdown only and authorizes no multiplicity rule for these four
    independently-fitted secondary models; inventing one here would be a third undisclosed
    post-Exp-A protocol completion."""
    subjects, session_idx, y_true, pred_by_seed, base = _oof_matrix(results_s)
    stats = config.stats
    seed = config.run.seed
    n_eval = int(len(y_true))
    selection_frequency = exp_a._selection_frequency([r for r in results_s if r.reason is None])

    if n_eval == 0:
        return {
            "conditional_exploratory": True, "n_eval": 0,
            "radar_mae": _nan_ci_dict(), "baseline_mae": _nan_ci_dict(), "mean_difference": _nan_ci_dict(),
            "selection_frequency": selection_frequency,
        }

    n_seeds = pred_by_seed.shape[0]
    seed_avg_abs_err = np.abs(y_true[None, :] - pred_by_seed).mean(axis=0) if n_seeds else np.zeros_like(y_true)
    baseline_abs_err = np.abs(y_true - base)
    diff = seed_avg_abs_err - baseline_abs_err

    # Reuses the SAME named per-session offsets the primary path's per-session breakdown uses
    # for this session -- this variant IS that session's search, run in isolation, and the two
    # summarize calls never combine numbers, so no cross-run RNG collision is possible.
    radar_ci = M.subject_cluster_bootstrap(
        seed_avg_abs_err, b=stats.bootstrap_b, rng_seed=seed + RNG_OFFSET_SESSION_RADAR_MAE_BASE + session)
    baseline_ci = M.subject_cluster_bootstrap(
        baseline_abs_err, b=stats.bootstrap_b, rng_seed=seed + RNG_OFFSET_SESSION_BASELINE_MAE_BASE + session)
    diff_ci = M.subject_cluster_bootstrap(
        diff, b=stats.bootstrap_b, rng_seed=seed + RNG_OFFSET_SESSION_DIFFERENCE_BASE + session)

    return {
        "conditional_exploratory": True,
        "n_eval": n_eval,
        "radar_mae": _ci_dict(radar_ci),
        "baseline_mae": _ci_dict(baseline_ci),
        "mean_difference": _ci_dict(diff_ci),
        "selection_frequency": selection_frequency,
    }


def _validate_shard(shard, *, band, session, run_group_id, analysis_commit, config_hash, expected_subjects) -> None:
    def check(field, expected, found):
        if found != expected:
            raise ExpBError(
                f"session-specific shard for session {session}: {field} mismatch "
                f"(expected {expected!r}, found {found!r})"
            )

    check("band", band, shard.get("band"))
    check("session", session, shard.get("session"))
    check("run_group_id", run_group_id, shard.get("run_group_id"))
    check("analysis_commit", analysis_commit, shard.get("analysis_commit"))
    check("config_hash", config_hash, shard.get("config_hash"))
    check("n_eval_subjects", sorted(expected_subjects), sorted(shard.get("n_eval_subjects", [])))


def merge_session_specific_reports(band, run_dir) -> dict:
    """Reads `run_dir/provenance.json` (written ONCE by `--init-run-group`) for the run-group's
    authoritative lineage: `analysis_commit` <- `provenance["git"]["commit"]` (the schema's own
    native field); `config_hash`/`expected_subjects_by_session` <-
    `provenance["extra"]["config_hash"]`/`provenance["extra"]["expected_subjects_by_session"]`
    (both `extra`-nested -- `record_run`'s body does `if extra: payload["extra"] = extra`, so
    `extra` content NESTS, it is never flattened to the top level). For each session s in 1..4
    whose `session_specific_{band}_s{s}.json` is PRESENT under `run_dir`: parses it and
    FAIL-CLOSED validates that its embedded band/session/run_group_id/analysis_commit/
    config_hash/n_eval_subjects all match the group's own authoritative values EXACTLY
    (mirroring `store.validate_store`'s `_check_match` precedent, not a new pattern). A missing
    file (task crashed, still running, never submitted) is simply absent from
    `completed_sessions` -- NOT an error. A PRESENT but malformed or mismatched shard IS an
    error: raises `ExpBError` naming the session, the field, and both the expected and found
    values, rather than silently excluding it or counting it. Store validity itself is NOT
    re-checked here (already enforced, per-task, before that task was allowed to fit anything)
    -- this function validates the SHARDS' lineage, not the store directly."""
    run_dir = Path(run_dir)
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    analysis_commit = provenance["git"]["commit"]
    config_hash = provenance["extra"]["config_hash"]
    expected_subjects_by_session = provenance["extra"]["expected_subjects_by_session"]

    out: dict = {
        "conditional_exploratory": True,
        "note": (
            "Secondary robustness variant (implementation_plan.md:722-724): four INDEPENDENTLY "
            "fitted single-session models, run as separate array tasks, not the pooled model's "
            "per-session breakdown. Never elevated to primary. DESCRIPTIVE ONLY -- no p-values, "
            "by design: the frozen protocol's Holm-4 applies to the primary model's per-session "
            "breakdown only and does not define a family size for these four secondary models, "
            "so this variant reports effect sizes + conditional/exploratory CIs and no "
            "significance claim, rather than deciding an undisclosed multiplicity rule after "
            "Exp A."
        ),
    }
    completed = []
    for s in (1, 2, 3, 4):
        shard_path = run_dir / f"session_specific_{band}_s{s}.json"
        if not shard_path.exists():
            continue
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        _validate_shard(
            shard, band=band, session=s, run_group_id=run_dir.name,
            analysis_commit=analysis_commit, config_hash=config_hash,
            expected_subjects=expected_subjects_by_session[str(s)],
        )
        out[str(s)] = shard["summary"]
        completed.append(s)
    out["completed_sessions"] = completed
    return out
