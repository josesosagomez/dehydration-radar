"""Experiment A — the staged fluid-loss regression composed on the generic harness engine.

This is the composition layer: it enumerates the frozen search space into `harness.Candidate`s,
provides store-backed features (`StoreBackedFeatures`), runs the two-stage search per outer
fold (Stage 1 feature axes at the fixed ridge anchor → Stage 2 family × grid on the winner),
refits, and fits the session-index-only baseline alongside — all through the same engine the
frozen leakage suite exercises. Nothing here constructs a fold or re-implements a tie-break.

Feature keys:
  10 GHz: (gate_idx, reduction, channel, tiling_idx, branch) with branch in {off, frozen, tuned}
  77 GHz: (tiling_idx, branch) — reduction/channel/gate are fixed (Exp G)
"tuned" is the on_tuned_eps branch, reconstructed fold-locally from the stored RAW tensor with
a train-only ε; off/frozen read the stored data-independent session vectors directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import store as store_mod
from ..features.pooling import aggregate_session, pool_stats_batch
from ..features.protocol_freeze import protocol_freeze_guard
from ..features.wst import apply_order_log
from ..features.extraction_77 import apply_order_log_77
from . import harness
from .harness import Candidate, FeatureBundle, require_complete_active, tuned_epsilons
from . import metrics as M

# log branch <-> stored vector name / reconstruction flag
_BRANCH_TO_LOG = {"off": "off", "frozen": "on_frozen_eps", "tuned": "on_tuned_eps"}
TILING_LABELS_10 = ("T1", "T2", "T3")
TILING_LABELS_77 = ("T1_77", "T2_77", "T3_77")


# ------------------------------------------------------------ candidate enumeration


def _active_10(gi, r, c, ti, branch, family, gate):
    return (
        ("band", "10ghz"), ("reduction", r), ("channel", c), ("tiling", TILING_LABELS_10[ti]),
        ("log_branch", _BRANCH_TO_LOG[branch]), ("range_gate_m", tuple(gate)), ("model_family", family),
    )


def _active_77(ti, branch, family, config):
    s = config.search_77ghz
    return (
        ("band", "77ghz"), ("reduction", s.reduction), ("channel", s.channel),
        ("gate_m", tuple(s.gate_m)), ("tiling", TILING_LABELS_77[ti]),
        ("log_branch", _BRANCH_TO_LOG[branch]), ("model_family", family),
    )


def stage1_candidates(config, band, anchor_alpha):
    """The feature-axis search at the fixed ridge anchor (10 GHz: 72 combos; 77 GHz: 9)."""
    cands = []
    if band == "10ghz":
        gates = config.search_10ghz.range_gate_m
        for gi, gate in enumerate(gates):
            for r in config.search_10ghz.reduction:
                for c in config.search_10ghz.channel:
                    for ti in range(len(config.wst.tilings)):
                        for branch in ("off", "frozen", "tuned"):
                            cands.append(Candidate(
                                candidate_id=f"g{gi}_{r}_{c}_t{ti}_{branch}",
                                family="ridge",
                                model_params=(("alpha", anchor_alpha),),
                                feature_key=(gi, r, c, ti, branch),
                                active=_active_10(gi, r, c, ti, branch, "ridge", gate),
                            ))
    else:
        for ti in range(len(config.wst77.tilings)):
            for branch in ("off", "frozen", "tuned"):
                cands.append(Candidate(
                    candidate_id=f"t{ti}_{branch}",
                    family="ridge",
                    model_params=(("alpha", anchor_alpha),),
                    feature_key=(ti, branch),
                    active=_active_77(ti, branch, "ridge", config),
                ))
    return cands


def stage2_candidates(config, band, feature_key, winner_active):
    """Model family × grid at the Stage-1 winning feature configuration (each grid <= budget_k).

    Reuses the winner's feature axes verbatim; only `model_family` and the model params vary."""
    from ..models.regressors import MODEL_FAMILIES, enumerate_grid

    cands = []
    for family in MODEL_FAMILIES:
        active = dict(winner_active)          # preserves the band-specific key set
        active["model_family"] = family
        active_tuple = tuple(active.items())
        for i, params in enumerate(enumerate_grid(family, config.model_grid)):
            cands.append(Candidate(
                candidate_id=f"{family}_{i}",
                family=family,
                model_params=tuple(sorted(params.items())),
                feature_key=feature_key,
                active=active_tuple,
            ))
    return cands


# --------------------------------------------------------------- store-backed features


class StoreBackedFeatures:
    """Provides `data_for(candidate, train_subjects) -> FeatureBundle` from a per-session store.

    Canonical row order = the `sessions` order the caller passes (sorted by subject, session).
    Data-independent branches (off/frozen) read stored session vectors (cached per feature key).
    The tuned branch computes a train-only ε from stored pre-log scales and reconstructs the
    session vectors from the stored RAW tensors, streaming one candidate's raw at a time.
    """

    def __init__(self, band, sessions, store_dir, config):
        self.band = band
        self.config = config
        self.subjects = np.array([s["subject"] for s in sessions])
        self.y = np.array([s["delta_m_pct"] for s in sessions], dtype=float)
        self._sessions = sessions
        self._stores = [
            store_mod.read_session_store(band, s["subject"], s["session_name"], store_dir)
            for s in sessions
        ]
        self._vec_cache: dict = {}
        self._prelog_cache: dict = {}
        self._tuned_cache: dict = {}
        self._raw_cache_fk = None       # single-entry raw cache (see _raw_frames)
        self._raw_cache = None
        self._eps_k = (config.search_10ghz if band == "10ghz" else config.search_77ghz).tuned_eps_k

    # ---- key helpers per band ----
    def _vec_key(self, fk, name):
        if self.band == "10ghz":
            gi, r, c, ti, _ = fk
            return store_mod.vec_key(gi, r, c, ti, name)
        ti, _ = fk
        return store_mod.vec77_key(ti, name)

    def _raw_key(self, fk):
        if self.band == "10ghz":
            gi, r, c, ti, _ = fk
            return store_mod.raw_key(gi, r, c, ti)
        return store_mod.raw77_key(fk[0])

    def _prelog_key(self, fk):
        if self.band == "10ghz":
            gi, r, c, ti, _ = fk
            return store_mod.prelog_key(gi, r, c, ti)
        return store_mod.prelog77_key(fk[0])

    def _order_key(self, fk):
        ti = fk[3] if self.band == "10ghz" else fk[0]
        return store_mod.order_key(ti)

    def _log_eps_cfg(self):
        return self.config.wst.log_epsilon if self.band == "10ghz" else self.config.wst77.log_epsilon

    # ---- feature assembly ----
    def _vec_matrix(self, fk, name):
        key = (self.band, fk[:-1], name)
        if key not in self._vec_cache:
            self._vec_cache[key] = np.stack([st[self._vec_key(fk, name)] for st in self._stores])
        return self._vec_cache[key]

    def _prelog_by_subject(self, fk):
        pk = self._prelog_key(fk)
        if pk not in self._prelog_cache:
            out: dict = {}
            for s, st in zip(self._sessions, self._stores):
                out.setdefault(s["subject"], []).append(tuple(st[pk].tolist()))
            self._prelog_cache[pk] = out
        return self._prelog_cache[pk]

    def _raw_frames(self, fk):
        """Load the raw pre-log tensors + meta order for one feature_key, cached single-entry.
        Candidate-major execution keeps one feature_key active at a time, so this eliminates
        the repeated (large) npz reads across a candidate's inner folds while capping memory to
        one feature_key's worth of raw tensors."""
        if self._raw_cache_fk != fk:
            rk, ok = self._raw_key(fk), self._order_key(fk)
            self._raw_cache = [(st[rk], np.asarray(st[ok])) for st in self._stores]
            self._raw_cache_fk = fk
        return self._raw_cache

    def _tuned_matrix(self, fk, eps):
        rows = []
        for S, order in self._raw_frames(fk):
            meta = {"order": order}
            if self.band == "10ghz":
                logged = apply_order_log(S, meta, self.config.wst, log_on=True, epsilon_by_order=eps)
            else:
                logged = np.stack([
                    apply_order_log_77(S[i], meta, self.config.wst77,
                                       log_branch="on_tuned_eps", epsilon_by_order=eps)
                    for i in range(S.shape[0])
                ])
            rows.append(aggregate_session(pool_stats_batch(logged, meta)))
        return np.stack(rows)

    def data_for(self, candidate: Candidate, train_subjects) -> FeatureBundle:
        fk = candidate.feature_key
        branch = fk[-1]
        if branch in ("off", "frozen"):
            X = self._vec_matrix(fk, branch)
            return FeatureBundle(self.subjects, X, self.y, extra_fits=())
        # tuned: train-only ε from stored pre-log scales, then reconstruct from raw. The
        # (feature_key, train set) fully determine ε and the reconstructed matrix, so cache
        # by that key — Stage 2 shares one feature_key across all families/grids/seeds, and
        # would otherwise recompute the identical reconstruction dozens of times per fold.
        cache_key = (fk, frozenset(train_subjects))
        if cache_key not in self._tuned_cache:
            eps = tuned_epsilons(self._prelog_by_subject(fk), train_subjects, k=self._eps_k,
                                 fallback=self._log_eps_cfg())
            self._tuned_cache[cache_key] = (self._tuned_matrix(fk, eps), eps)
        X, eps = self._tuned_cache[cache_key]
        extra = (("tuned_epsilon", {"epsilon": np.array([eps[1], eps[2]], dtype=float)}),)
        return FeatureBundle(self.subjects, X, self.y, extra_fits=extra)


# ----------------------------------------------------------------- staged run


@dataclass
class ExpAFoldResult:
    test_subject: int
    selected_feature_key: tuple
    selected_family: str
    selected_params: dict
    test_predictions: np.ndarray          # first-seed (headline single-value fields)
    test_targets: np.ndarray
    seed_outcomes: list
    baseline_predictions: np.ndarray
    final_fits: list


def _run_single_fold(config, band, sessions, store_dir, session_index, fold, seeds) -> ExpAFoldResult:
    """Run ONE outer fold end to end and return its ExpAFoldResult. Top-level + picklable so it
    can run in a worker process. Builds its OWN store-backed provider (open npz handles are not
    shareable across processes) and pins single-threaded math, so a fold's result is bit-identical
    whether run serially or in parallel — the folds are independent and deterministic."""
    from threadpoolctl import threadpool_limits

    from ..models.baselines import fit_session_index_baseline, predict_session_index

    with threadpool_limits(1):
        provider = StoreBackedFeatures(band, sessions, store_dir, config)
        anchor = (config.search_10ghz if band == "10ghz" else config.search_77ghz).stage1_anchor_ridge_alpha

        def before_fit(candidate):
            active = dict(candidate.active)
            require_complete_active(active)          # fail-closed completeness (C5)
            protocol_freeze_guard(config, active=active)

        s1 = harness._score_candidates_on_fold(
            stage1_candidates(config, band, anchor), fold, seeds, before_fit, provider.data_for
        )
        w1 = harness.select_stage_winner(s1)
        s2 = harness._score_candidates_on_fold(
            stage2_candidates(config, band, w1.feature_key, dict(w1.active)),
            fold, seeds, before_fit, provider.data_for,
        )
        w2 = harness.select_stage_winner(s2)
        final_fits, _, test_pred, _, seed_outcomes = harness._final_refit(
            w2, fold, seeds, before_fit, provider.data_for
        )

        # session-index-only baseline (K=1), fit on the same outer-training subjects.
        base = fit_session_index_baseline(
            provider.subjects, session_index, provider.y, fold.train_subjects
        )
        test_rows = np.isin(provider.subjects, [fold.test_subject])
        base_pred = predict_session_index(base.model, session_index[test_rows])

        return ExpAFoldResult(
            test_subject=fold.test_subject,
            selected_feature_key=w2.feature_key,
            selected_family=w2.family,
            selected_params=w2.params(),
            test_predictions=test_pred,
            test_targets=provider.y[test_rows],
            seed_outcomes=seed_outcomes,
            baseline_predictions=base_pred,
            final_fits=final_fits + [base.fit_record],
        )


def run_exp_a(config, band, sessions, store_dir, *, seeds, session_index, n_workers=1) -> list:
    """Per outer fold: Stage 1 (feature axes at ridge anchor) → Stage 2 (family × grid) → refit,
    plus the session-index-only baseline. The outer folds are independent, so with `n_workers>1`
    they run in parallel worker processes (each single-threaded) and the results are reassembled
    in canonical test-subject order — **bit-identical to the serial run**, just faster. Guard runs
    before every fit inside each fold. Folds come only from `splits.py`."""
    subjects = sorted({int(s["subject"]) for s in sessions})
    folds = [f for f in harness.nested_loso_splits(subjects) if f.selectable]
    tasks = [(config, band, sessions, store_dir, session_index, fold, seeds) for fold in folds]

    if n_workers <= 1 or len(tasks) <= 1:
        results = [_run_single_fold(*t) for t in tasks]
    else:
        import multiprocessing as mp

        # spawn (not fork): a clean worker with no inherited open npz handles or BLAS state, so
        # behaviour is identical on Linux (IBEX) and elsewhere.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(n_workers, len(tasks))) as pool:
            results = pool.starmap(_run_single_fold, tasks)

    results.sort(key=lambda r: r.test_subject)   # deterministic order regardless of completion
    return results


# ----------------------------------------------------------------- reporting

def _per_seed_matrix(results):
    """Assemble [n_seeds, n_sessions] test predictions across folds (canonical fold order).

    Deterministic winners carry one SeedOutcome; the 5-seed protocol replicates it across
    seeds (identical), so pooled metrics have a consistent seed axis. y_true / subjects /
    baseline are aligned the same way."""
    n_seeds = max(len(r.seed_outcomes) for r in results)
    subjects, y_true, base = [], [], []
    per_seed = [[] for _ in range(n_seeds)]
    for r in results:
        n = len(r.test_targets)
        subjects += [r.test_subject] * n
        y_true += r.test_targets.tolist()
        base += r.baseline_predictions.tolist()
        for k in range(n_seeds):
            so = r.seed_outcomes[k] if k < len(r.seed_outcomes) else r.seed_outcomes[0]
            per_seed[k] += so.test_predictions.tolist()
    return (np.array(subjects), np.array(y_true, float),
            np.array(per_seed, float), np.array(base, float))


def summarize_exp_a(results, config) -> dict:
    """The Exp A headline metrics + subject-cluster CIs + the session-index baseline comparison.

    All CIs are conditional/exploratory (they resample fixed selected models). Seed collapse is
    metric-type-aware: per-subject MAE averages seeds within subject then bootstraps subjects;
    pooled RMSE/r recompute per seed within each resample."""
    subjects, y_true, pred_by_seed, base_pred = _per_seed_matrix(results)
    n_seeds = pred_by_seed.shape[0]
    stats = config.stats
    subj_ids = sorted(set(subjects.tolist()))

    # per-subject MAE, seed-averaged (additive metric).
    per_subject_mae = []
    for s in subj_ids:
        m = subjects == s
        per_seed_mae = [np.abs(y_true[m] - pred_by_seed[k, m]).mean() for k in range(n_seeds)]
        per_subject_mae.append(float(np.mean(per_seed_mae)))
    per_subject_mae = np.array(per_subject_mae)

    mae_ci = M.subject_cluster_bootstrap(per_subject_mae, b=stats.bootstrap_b, rng_seed=config.run.seed)
    rmse_ci = M.subject_cluster_bootstrap_pooled(
        subjects, y_true, pred_by_seed, M.session_rmse, b=stats.bootstrap_b, rng_seed=config.run.seed + 1
    )
    r_ci = M.subject_cluster_bootstrap_pooled(
        subjects, y_true, pred_by_seed, M.pooled_pearson_r, b=stats.bootstrap_b, rng_seed=config.run.seed + 2,
        skip_threshold_pct=stats.undefined_metric_skip_threshold_pct,
    )

    # baseline comparison: per-subject radar MAE vs baseline MAE, paired over subjects.
    base_per_subject = []
    for s in subj_ids:
        m = subjects == s
        base_per_subject.append(float(np.abs(y_true[m] - base_pred[m]).mean()))
    diffs = per_subject_mae - np.array(base_per_subject)   # radar - baseline (negative = radar better)
    wstat, wp = M.wilcoxon_signed_rank(diffs)
    diff_ci = M.mean_difference_ci(diffs, b=stats.bootstrap_b, rng_seed=config.run.seed + 3)

    # pooled r additionally on S1..S4 only (exclude the S0=0 anchor); S0 <=> target ~ 0.
    s1_s4 = np.abs(y_true) > 1e-9
    pooled_r_s1s4 = M.pooled_pearson_r(y_true[s1_s4], pred_by_seed[0][s1_s4]) if s1_s4.sum() >= 2 else float("nan")

    def ci_dict(c):
        return {"point": c.point, "low": c.low, "high": c.high, "method": c.method,
                "n_eval": c.n_eval, "n_skipped": c.n_skipped, "unreliable": c.unreliable}

    return {
        "conditional_exploratory": True,
        "n_eval_subjects": len(subj_ids),
        "n_sessions": int(len(y_true)),
        "n_seeds": n_seeds,
        "subject_balanced_mae": ci_dict(mae_ci),
        "session_rmse": ci_dict(rmse_ci),
        "pooled_pearson_r": ci_dict(r_ci),
        "pooled_pearson_r_s1_s4_seed0": pooled_r_s1s4,
        "per_subject_mae": {int(s): float(v) for s, v in zip(subj_ids, per_subject_mae)},
        "per_subject_pearson_r": {int(k): float(v) for k, v in
                                  M.per_subject_pearson_r(subjects, y_true, pred_by_seed[0],
                                                          min_sessions=stats.per_subject_pearson_r_min_sessions).items()},
        "baseline_session_index_only": {
            "per_subject_mae": {int(s): float(v) for s, v in zip(subj_ids, base_per_subject)},
            "wilcoxon_statistic": wstat, "wilcoxon_p": wp,
            "mean_difference_radar_minus_baseline": ci_dict(diff_ci),
        },
        "selection_frequency": _selection_frequency(results),
    }


def _selection_frequency(results) -> dict:
    from collections import Counter
    families = Counter(r.selected_family for r in results)
    branches = Counter(r.selected_feature_key[-1] for r in results)
    tilings = Counter(r.selected_feature_key[3] if len(r.selected_feature_key) == 5
                      else r.selected_feature_key[0] for r in results)
    return {"family": dict(families), "log_branch": dict(branches), "tiling_idx": dict(tilings)}


def write_exp_a_reports(results, summary, out_dir, band) -> dict:
    """Write the FULL-run artifacts (metrics JSON, predictions CSV, selection table, scatter).
    Returns {name: path}. Only called in --full-cohort mode (never the mechanism-only smoke)."""
    import csv
    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")  # headless: no display
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    metrics_path = out_dir / f"metrics_exp_a_{band}.json"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics"] = metrics_path

    pred_path = out_dir / f"predictions_{band}.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "seed", "y_true", "y_pred"])
        for r in results:
            for so in r.seed_outcomes:
                for yt, yp in zip(r.test_targets, so.test_predictions):
                    w.writerow([r.test_subject, so.seed, yt, yp])
    paths["predictions"] = pred_path

    sel_path = out_dir / f"selection_table_{band}.csv"
    with sel_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test_subject", "feature_key", "family", "params"])
        for r in results:
            w.writerow([r.test_subject, r.selected_feature_key, r.selected_family, r.selected_params])
    paths["selection_table"] = sel_path

    y_true = np.concatenate([r.test_targets for r in results])
    y_pred = np.concatenate([r.test_predictions for r in results])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=18, alpha=0.7)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("actual Δm%")
    ax.set_ylabel("predicted Δm%")
    ax.set_title(f"Exp A predicted vs actual ({band})")
    scatter_path = out_dir / f"scatter_{band}.png"
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)
    paths["scatter"] = scatter_path
    return paths


# ---------------------------------------------------------- data spine + orchestration

def build_sessions(config, band):
    """One record per eligible session in canonical (subject, session_idx) order, carrying the
    Δm% target, the QC-selected frame ids, and the raw rel_path (for fingerprint validation)."""
    from ..data.ground_truth import load_ground_truth
    from ..data.manifest import apply_qc, build_manifest, eligible_frames
    from ..data.manifest_77 import apply_qc_77, build_manifest_77

    gt = load_ground_truth(config.paths.weight_xlsx)
    if band == "10ghz":
        manifest_qc = apply_qc(build_manifest(config.paths, gt), config.paths, config)
    else:
        from ..config import require_77ghz_dir
        require_77ghz_dir(config)
        manifest_qc = apply_qc_77(build_manifest_77(config.paths, gt), config.paths, config)
    pop = eligible_frames(manifest_qc)

    dm = {(int(r.subject), int(r.session_idx)): float(r.delta_m_pct) for r in gt.sessions.itertuples()}
    records = []
    for (subject, session_idx), g in pop.groupby(["subject", "session_idx"]):
        records.append({
            "subject": int(subject),
            "session_idx": int(session_idx),
            "session_name": g["session_name"].iloc[0],
            "rel_path": g["rel_path"].iloc[0],
            "frame_ids": g["frame_idx"].tolist(),
            "delta_m_pct": dm[(int(subject), int(session_idx))],
        })
    records.sort(key=lambda s: (s["subject"], s["session_idx"]))
    return records


def select_subset_subjects(subjects, k=6):
    """The k lowest evaluable subject ids (deterministic) — the smoke subset."""
    return sorted(set(int(s) for s in subjects))[:k]


def expected_fingerprints(config, band, sessions):
    from ..data.manifest import resolve_path
    from ..data.manifest_77 import resolve_path_77
    out = {}
    for s in sessions:
        raw = resolve_path(config.paths, s["rel_path"]) if band == "10ghz" else resolve_path_77(config.paths, s["rel_path"])
        out[(s["subject"], s["session_name"])] = store_mod.compute_fingerprint(
            config, band, frame_ids=s["frame_ids"], raw_path=raw, session_eligible=True
        )
    return out


def run_and_report(config, band, sessions, store_dir, run_dir, *, mode, analysis_commit, n_workers=1) -> dict:
    """Validate the store, run Exp A, and cross the reporting boundary.

    `mode="full"` writes every performance artifact; `mode="smoke"` is MECHANISM-ONLY — it
    runs the identical search/scoring path but surfaces NO performance value (no metrics,
    predictions, scatter, or selection table), writing only a structural run-log. In both
    modes it asserts train/val/test disjointness and full fit-audit coverage. `n_workers>1`
    parallelises the independent outer folds (bit-identical result). Returns {name: path} for
    full, or {"run_log": path} for smoke."""
    import json
    from pathlib import Path

    store_mod.validate_store(band, store_dir, expected_fingerprints(config, band, sessions),
                             analysis_commit=analysis_commit)
    session_index = np.array([s["session_idx"] for s in sessions])
    results = run_exp_a(config, band, sessions, store_dir, seeds=config.run.seed_set,
                        session_index=session_index, n_workers=n_workers)

    _assert_mechanism_ok(results, sessions)  # structural, not performance

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        log = run_dir / f"run_log_{band}.json"
        log.write_text(json.dumps({
            "stage": "exp-a-smoke", "band": band, "mode": "mechanism-only",
            "n_folds": len(results), "n_sessions": len(sessions),
            "note": "performance values suppressed until the owner clears the freeze",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"run_log": log}

    summary = summarize_exp_a(results, config)
    paths = write_exp_a_reports(results, summary, run_dir, band)
    return paths


def _assert_mechanism_ok(results, sessions) -> None:
    """Structural checks that don't reveal performance: fold-role disjointness + audit coverage."""
    folds = harness.nested_loso_splits(sorted({int(s["subject"]) for s in sessions}))
    for fold in folds:
        if not fold.selectable:
            continue
        assert fold.test_subject not in fold.train_subjects
        for inner in fold.inner_folds:
            assert inner.train_subjects.isdisjoint(inner.val_subjects)
            assert fold.test_subject not in inner.train_subjects
            assert fold.test_subject not in inner.val_subjects
    # every recorded fit names a training subject set that excludes the held-out subject.
    for r in results:
        for rec in r.final_fits:
            assert r.test_subject not in rec.subjects
