"""T-M8 Exp B driver: `SessionResidualFeatures` + `run_exp_b`, end to end on a synthetic
store (run half) -- no private data -- plus `summarize_exp_b`/reporting on hand-built
`ExpBFoldResult`s (report half, mirroring `test_run_regression.py`'s stubbing pattern for
Exp A). Covers T-M8-provider, T-M8-degenerate, T-M8-residual-leak, T-M8-outer-mutation (C2),
and T-M8-report. The session-specific variant (T-M8-variant) lives once step 7 exists.
"""

import json

import numpy as np
import pytest

from dehyd.config import load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_a, harness
from dehyd.eval.exp_b import (
    ExpBError,
    ExpBFoldResult,
    SessionResidualFeatures,
    build_sessions_b,
    config_fingerprint,
    eligible_subjects_for_session,
    equal_session_objective,
    evaluable_subjects_b,
    merge_session_specific_reports,
    run_and_report_b,
    run_exp_b,
    run_exp_b_one_session,
    run_exp_b_session_specific,
    summarize_exp_b,
    summarize_variant_session,
    write_exp_b_reports,
)
from dehyd.eval.harness import SeedOutcome, require_complete_active
from dehyd.features.pooling import aggregate_session, pool_stats_batch
from dehyd.features.protocol_freeze import protocol_freeze_guard
from dehyd.features.store import (
    order_key,
    prelog_key,
    raw_key,
    read_session_store,
    vec_key,
    write_session_store,
)
from dehyd.features.wst import apply_order_log
from dehyd.models.baselines import fit_session_mean_baseline

P, T, CN, NFR = 6, 4, 1, 3  # tiny path/time/channel/frame dims for a fast synthetic store
ORDER = np.array([0, 1, 1, 2, 2, 2])  # length P, all orders present


@pytest.fixture(scope="module")
def config():
    return load_config("configs/exp_a_regression.yaml", "configs/exp_b.yaml")


# ------------------------------------------------------------------------------- fixtures


def _make_sessions_b(n_subjects=6, session_indices=(1, 2, 3, 4)):
    out = []
    for s in range(1, n_subjects + 1):
        for i in session_indices:
            out.append({
                "subject": s,
                "session_idx": i,
                "session_name": SESSION_NAMES[i],
                "delta_m_pct": float(-0.3 * i - 0.05 * s),   # a clock-correlated target
            })
    return out


def _make_sessions_degenerate(n_subjects=6):
    """Subject 1 holds ALL 4 sessions; subjects 2..n skip session 3 -- session 3 has < 2
    eligible training subjects in EVERY outer fold (T-M8-degenerate)."""
    out = []
    for s in range(1, n_subjects + 1):
        idxs = (1, 2, 3, 4) if s == 1 else (1, 2, 4)
        for i in idxs:
            out.append({
                "subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
                "delta_m_pct": float(-0.3 * i - 0.05 * s),
            })
    return out


def _write_store(store_dir, sessions, config, seed=0):
    rng = np.random.default_rng(seed)
    wst = config.wst
    eps = {1: wst.log_epsilon, 2: wst.log_epsilon}
    meta = {"order": ORDER}
    for s in sessions:
        npz = {}
        for ti in range(len(wst.tilings)):
            npz[order_key(ti)] = ORDER
            for gi in range(len(config.search_10ghz.range_gate_m)):
                for r in config.search_10ghz.reduction:
                    for c in config.search_10ghz.channel:
                        raw = np.abs(rng.normal(size=(NFR, CN, P, T))) + 0.01
                        off = aggregate_session(pool_stats_batch(apply_order_log(raw, meta, wst, log_on=False), meta))
                        fr = aggregate_session(pool_stats_batch(apply_order_log(raw, meta, wst, log_on=True, epsilon_by_order=eps), meta))
                        npz[vec_key(gi, r, c, ti, "off")] = off
                        npz[vec_key(gi, r, c, ti, "frozen")] = fr
                        npz[raw_key(gi, r, c, ti)] = raw
                        npz[prelog_key(gi, r, c, ti)] = np.array([raw.mean(), raw.mean(), raw.mean()])
        write_session_store("10ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR}, store_dir)


def _mutate_features_on_disk(store_dir, sessions, subject, seed=99):
    """Eligibility-preserving: overwrite one subject's stored arrays (not labels)."""
    rng = np.random.default_rng(seed)
    for s in sessions:
        if s["subject"] != subject:
            continue
        store = read_session_store("10ghz", s["subject"], s["session_name"], store_dir)
        npz = {k: store[k].copy() for k in store.keys()}
        store.close()
        for k in list(npz):
            if k.startswith("raw__"):
                npz[k] = np.abs(rng.normal(size=npz[k].shape) * 5) + 0.01
            elif k.startswith(("vec__", "prelog__")):
                npz[k] = (rng.normal(size=npz[k].shape) * 5 + 5).astype(npz[k].dtype)
        write_session_store("10ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR}, store_dir)


def _mutate_label(sessions, subject, seed=99):
    rng = np.random.default_rng(seed)
    for s in sessions:
        if s["subject"] == subject:
            s["delta_m_pct"] = float(rng.normal() * 5 + 5)


def _run(store_dir, sessions, config, n_workers=1):
    return run_exp_b(config, "10ghz", sessions, store_dir, seeds=(0,), n_workers=n_workers)


def _fold(results, subject):
    return next(r for r in results if r.test_subject == subject)


def _before_fit(config):
    def before_fit(candidate):
        active = dict(candidate.active)
        require_complete_active(active)
        protocol_freeze_guard(config, active=active)
    return before_fit


def _stage1_outcome(sessions, store_dir, config, band, fold, seeds=(0,)):
    provider = SessionResidualFeatures(band, sessions, store_dir, config)
    anchor = (config.search_10ghz if band == "10ghz" else config.search_77ghz).stage1_anchor_ridge_alpha
    return harness._score_candidates_on_fold(
        exp_a.stage1_candidates(config, band, anchor), fold, seeds, _before_fit(config),
        provider.data_for, score_fn=equal_session_objective,
    )


def _run_fold_with_stages(sessions, store_dir, config, band, fold, seeds=(0,)):
    """Mirrors `exp_b._run_single_fold_b`'s internals but returns the raw intermediate
    StageOutcomes too, so a test can inspect both stages' inner scores/winners directly
    (ExpBFoldResult itself only exposes the FINAL winner)."""
    provider = SessionResidualFeatures(band, sessions, store_dir, config)
    anchor = (config.search_10ghz if band == "10ghz" else config.search_77ghz).stage1_anchor_ridge_alpha
    before_fit = _before_fit(config)

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
    final_fits, train_pred, test_pred, test_score, seed_outcomes = harness._final_refit(
        w2, fold, seeds, before_fit, provider.data_for, score_fn=equal_session_objective,
    )
    _, dropped_outer = provider.drop_for(fold.train_subjects)
    return {
        "s1": s1, "w1": w1, "s2": s2, "w2": w2, "final_fits": final_fits,
        "train_pred": train_pred, "test_pred": test_pred, "test_score": test_score,
        "dropped_outer": dropped_outer,
    }


def _fold_for(sessions, test_subject):
    return next(
        f for f in harness.nested_loso_splits(evaluable_subjects_b(sessions))
        if f.test_subject == test_subject
    )


# ------------------------------------------------------------- data spine + eligibility


def test_build_sessions_b_excludes_s0(monkeypatch, config):
    fake_sessions = [
        {"subject": 1, "session_idx": i, "session_name": SESSION_NAMES[i], "delta_m_pct": 0.0}
        for i in range(5)
    ]
    monkeypatch.setattr(exp_a, "build_sessions", lambda cfg, band: fake_sessions)
    out = build_sessions_b(config, "10ghz")
    assert {s["session_idx"] for s in out} == {1, 2, 3, 4}
    assert len(out) == 4


def test_evaluable_subjects_b_is_distinct_subjects_of_filtered_sessions():
    sessions = _make_sessions_b(n_subjects=3, session_indices=(1, 2))
    assert evaluable_subjects_b(sessions) == [1, 2, 3]


def test_session_residual_features_rejects_s0_rows(tmp_path, config):
    sessions = [
        {"subject": 1, "session_idx": 0, "session_name": SESSION_NAMES[0], "delta_m_pct": 0.0},
        {"subject": 1, "session_idx": 1, "session_name": SESSION_NAMES[1], "delta_m_pct": 1.0},
        {"subject": 2, "session_idx": 1, "session_name": SESSION_NAMES[1], "delta_m_pct": 2.0},
    ]
    with pytest.raises(ExpBError, match="S0"):
        SessionResidualFeatures("10ghz", sessions, tmp_path, config)


# -------------------------------------------------------------------- run_exp_b end to end


def test_run_exp_b_runs_the_staged_search_end_to_end(tmp_path, config):
    sessions = _make_sessions_b()
    _write_store(tmp_path, sessions, config)
    results = _run(tmp_path, sessions, config)
    assert len(results) == 6  # one per selectable outer fold (6 subjects, 5 >= min_train_subjects)
    for r in results:
        assert r.reason is None
        assert r.selected_family in ("ridge", "svr", "rf", "gbm", "knn")
        assert len(r.selected_feature_key) == 5
        assert r.baseline_predictions.shape == r.test_predictions.shape
        assert np.all(r.baseline_predictions == 0.0)          # baseline == 0 residual, by construction
        assert not np.any(r.test_session_idx == 0)             # trap 3 sanity


def test_parallel_folds_are_bit_identical_to_serial(tmp_path, config):
    sessions = _make_sessions_b()
    _write_store(tmp_path, sessions, config)
    serial = _run(tmp_path, sessions, config, n_workers=1)
    parallel = _run(tmp_path, sessions, config, n_workers=2)

    assert [r.test_subject for r in serial] == [r.test_subject for r in parallel]
    for rs, rp in zip(serial, parallel, strict=True):
        assert rs.selected_feature_key == rp.selected_feature_key
        assert rs.selected_family == rp.selected_family
        assert rs.selected_params == rp.selected_params
        assert rs.test_predictions.tobytes() == rp.test_predictions.tobytes()
        assert rs.dropped_sessions_outer == rp.dropped_sessions_outer
        for fs, fp in zip(rs.final_fits, rp.final_fits, strict=True):
            assert fs.quantity == fp.quantity
            for k in fs.params:
                assert fs.params[k].tobytes() == fp.params[k].tobytes()


# ------------------------------------------------------------------------ T-M8-provider


def test_provider_x_path_and_residual_and_alignment_match_on_kept_rows(tmp_path, config):
    sessions = _make_sessions_b()
    _write_store(tmp_path, sessions, config)
    provider = SessionResidualFeatures("10ghz", sessions, tmp_path, config)
    base = exp_a.StoreBackedFeatures("10ghz", sessions, tmp_path, config)

    anchor = config.search_10ghz.stage1_anchor_ridge_alpha
    candidate = exp_a.stage1_candidates(config, "10ghz", anchor)[0]
    train_subjects = {1, 2, 3, 4, 5}

    bundle = provider.data_for(candidate, train_subjects)
    base_bundle = base.data_for(candidate, train_subjects)
    means, dropped = provider.drop_for(train_subjects)
    kept = ~np.isin(provider.session_idx, dropped)

    assert dropped == ()   # every subject has every session in this fixture
    assert bundle.X.tobytes() == base_bundle.X[kept].tobytes()
    assert bundle.subjects.tolist() == base_bundle.subjects[kept].tolist()
    assert bundle.session_idx.tolist() == provider.session_idx[kept].tolist()

    mu_row = np.array([means[int(s)] for s in provider.session_idx[kept]])
    assert np.allclose(bundle.y, base_bundle.y[kept] - mu_row)

    quantities = {q for q, _ in bundle.extra_fits}
    assert "session_means" in quantities


def test_provider_drop_set_is_candidate_independent(tmp_path, config):
    sessions = _make_sessions_b()
    _write_store(tmp_path, sessions, config)
    provider = SessionResidualFeatures("10ghz", sessions, tmp_path, config)
    anchor = config.search_10ghz.stage1_anchor_ridge_alpha
    cands = exp_a.stage1_candidates(config, "10ghz", anchor)[:5]
    train_subjects = {1, 2, 3, 4, 5}

    drops = [provider.drop_for(train_subjects) for _ in cands]
    assert all(d == drops[0] for d in drops)
    assert len(provider._drop_cache) == 1   # one cache entry regardless of candidate count


def test_outer_bundle_session_means_matches_fit_session_mean_baseline_bytewise(tmp_path, config):
    sessions = _make_sessions_b()
    _write_store(tmp_path, sessions, config)
    provider = SessionResidualFeatures("10ghz", sessions, tmp_path, config)
    anchor = config.search_10ghz.stage1_anchor_ridge_alpha
    candidate = exp_a.stage1_candidates(config, "10ghz", anchor)[0]
    train_subjects = {1, 2, 3, 4, 5}

    bundle = provider.data_for(candidate, train_subjects)
    session_means_entry = dict(bundle.extra_fits)["session_means"]

    out = fit_session_mean_baseline(provider.subjects, provider.session_idx, provider.y_raw, train_subjects)

    assert session_means_entry["indices"].tobytes() == out.fit_record.params["indices"].tobytes()
    assert session_means_entry["means"].tobytes() == out.fit_record.params["means"].tobytes()
    assert session_means_entry["dropped"].tobytes() == out.fit_record.params["dropped"].tobytes()


# ----------------------------------------------------------------------- T-M8-degenerate


def test_degenerate_session_drop_absent_from_bundle_and_oof_present_in_dropped(tmp_path, config):
    sessions = _make_sessions_degenerate()
    _write_store(tmp_path, sessions, config)
    results = _run(tmp_path, sessions, config)

    r1 = _fold(results, 1)   # subject 1 is the only one with a session-3 row
    assert r1.dropped_sessions_outer == (3,)
    assert set(r1.test_session_idx.tolist()) == {1, 2, 4}

    for r in results:
        assert 3 in r.dropped_sessions_outer
        for _ts, dropped in r.dropped_sessions_inner:
            assert 3 in dropped


def test_degenerate_drop_does_not_perturb_surviving_mu(tmp_path, config):
    degenerate = _make_sessions_degenerate()
    _write_store(tmp_path / "degenerate", degenerate, config)
    provider_d = SessionResidualFeatures("10ghz", degenerate, tmp_path / "degenerate", config)
    train_subjects = {2, 3, 4, 5, 6}
    means_d, dropped_d = provider_d.drop_for(train_subjects)
    assert dropped_d == (3,)

    full = _make_sessions_b(n_subjects=6)
    _write_store(tmp_path / "full", full, config)
    provider_f = SessionResidualFeatures("10ghz", full, tmp_path / "full", config)
    means_f, dropped_f = provider_f.drop_for(train_subjects)
    assert dropped_f == ()

    for s in (1, 2, 4):
        assert np.array(means_d[s]).tobytes() == np.array(means_f[s]).tobytes()


# ----------------------------------------------------------------------- T-M8-residual-leak


def test_residual_leak_inner_validation_label_mutation_leaves_fits_untouched(tmp_path, config):
    held_out, mutated_subject = 1, 5
    sessions_a = _make_sessions_b()
    _write_store(tmp_path / "base", sessions_a, config)
    base = _stage1_outcome(sessions_a, tmp_path / "base", config, "10ghz", _fold_for(sessions_a, held_out))

    sessions_b = _make_sessions_b()
    _write_store(tmp_path / "mut", sessions_b, config)
    _mutate_label(sessions_b, mutated_subject)
    mut = _stage1_outcome(sessions_b, tmp_path / "mut", config, "10ghz", _fold_for(sessions_b, held_out))

    checked = 0
    for lhs, rhs in zip(base.inner_results, mut.inner_results, strict=True):
        assert lhs.inner_val == rhs.inner_val
        if mutated_subject in lhs.inner_val:
            for fb, fm in zip(lhs.fits, rhs.fits, strict=True):
                assert fb.quantity == fm.quantity
                for k in fb.params:
                    assert fb.params[k].tobytes() == fm.params[k].tobytes()
            checked += 1
    assert checked > 0, "fixture never put the mutated subject in a validation role"


def test_residual_leak_inner_train_label_mutation_moves_mu_power(tmp_path, config):
    held_out, mutated_subject = 1, 5
    sessions_a = _make_sessions_b()
    _write_store(tmp_path / "base", sessions_a, config)
    base = _stage1_outcome(sessions_a, tmp_path / "base", config, "10ghz", _fold_for(sessions_a, held_out))

    sessions_b = _make_sessions_b()
    _write_store(tmp_path / "mut", sessions_b, config)
    _mutate_label(sessions_b, mutated_subject)
    mut = _stage1_outcome(sessions_b, tmp_path / "mut", config, "10ghz", _fold_for(sessions_b, held_out))

    moved = False
    for lhs, rhs in zip(base.inner_results, mut.inner_results, strict=True):
        if mutated_subject in lhs.inner_val:
            continue   # only inner-TRAIN folds are the power case here
        for fb, fm in zip(lhs.fits, rhs.fits, strict=True):
            if fb.quantity == "session_means" and fb.params["means"].tobytes() != fm.params["means"].tobytes():
                moved = True
    assert moved


def test_residual_leak_held_out_label_mutation_leaves_outer_train_fits_identical(tmp_path, config):
    held_out = 1
    sessions_a = _make_sessions_b()
    _write_store(tmp_path / "base", sessions_a, config)
    base = _run_fold_with_stages(sessions_a, tmp_path / "base", config, "10ghz", _fold_for(sessions_a, held_out))

    sessions_b = _make_sessions_b()
    _write_store(tmp_path / "mut", sessions_b, config)
    _mutate_label(sessions_b, held_out)
    mut = _run_fold_with_stages(sessions_b, tmp_path / "mut", config, "10ghz", _fold_for(sessions_b, held_out))

    for fb, fm in zip(base["final_fits"], mut["final_fits"], strict=True):
        assert fb.quantity == fm.quantity
        for k in fb.params:
            assert fb.params[k].tobytes() == fm.params[k].tobytes()


# ----------------------------------------------------------------------- T-M8-outer-mutation (C2)


def test_outer_mutation_property_end_to_end(tmp_path, config):
    """(C2) The REAL Exp B composition, not just fit-record mutation: mutate the held-out
    subject's stored vec/raw/prelog arrays AND its target; re-run SessionResidualFeatures +
    drop-row + equal_session_objective + Stage-1/Stage-2 search; assert the drop set, both
    stages' inner scores and winners, every fitted parameter, and outer-training predictions
    are unchanged -- only the held-out subject's own prediction/score may move."""
    held_out = 1
    sessions_a = _make_sessions_b()
    _write_store(tmp_path / "base", sessions_a, config)
    base = _run_fold_with_stages(sessions_a, tmp_path / "base", config, "10ghz", _fold_for(sessions_a, held_out))

    sessions_b = _make_sessions_b()
    _write_store(tmp_path / "mut", sessions_b, config)
    _mutate_features_on_disk(tmp_path / "mut", sessions_b, held_out)
    _mutate_label(sessions_b, held_out)
    mut = _run_fold_with_stages(sessions_b, tmp_path / "mut", config, "10ghz", _fold_for(sessions_b, held_out))

    assert base["dropped_outer"] == mut["dropped_outer"]
    assert base["s1"].inner_scores.tobytes() == mut["s1"].inner_scores.tobytes()
    assert base["w1"].feature_key == mut["w1"].feature_key
    assert base["s2"].inner_scores.tobytes() == mut["s2"].inner_scores.tobytes()
    assert base["w2"].feature_key == mut["w2"].feature_key
    assert base["w2"].family == mut["w2"].family
    assert base["w2"].params() == mut["w2"].params()
    for fb, fm in zip(base["final_fits"], mut["final_fits"], strict=True):
        assert fb.quantity == fm.quantity
        for k in fb.params:
            assert fb.params[k].tobytes() == fm.params[k].tobytes()
    assert base["train_pred"].tobytes() == mut["train_pred"].tobytes()
    # power: the held-out subject's own prediction DOES move (its features were mutated).
    assert base["test_pred"].tobytes() != mut["test_pred"].tobytes()


# --------------------------------------------------------------------------- T-M8-report
#
# summarize_exp_b/write_exp_b_reports/run_and_report_b operate purely on a list of
# ExpBFoldResult + config -- tested directly on hand-built results (mirroring
# test_run_regression.py's ExpAFoldResult-stubbing pattern), not through the expensive
# store+search path.


def _fake_result_b(test_subject, rows, *, dropped_outer=(), reason=None):
    """`rows`: [(session_idx, y_true_residual, y_pred_residual), ...] for this subject's
    out-of-fold contribution. `baseline_predictions` == zeros, by construction."""
    session_idx = np.array([r[0] for r in rows], dtype=int)
    y_true = np.array([r[1] for r in rows], dtype=float)
    y_pred = np.array([r[2] for r in rows], dtype=float)
    return ExpBFoldResult(
        test_subject=test_subject,
        selected_feature_key=(0, "A", "mag", 0, "off"),
        selected_family="ridge",
        selected_params={"alpha": 1.0},
        test_predictions=y_pred,
        test_targets=y_true,
        test_session_idx=session_idx,
        seed_outcomes=[SeedOutcome(0, y_pred, y_pred, 0.1)],
        baseline_predictions=np.zeros_like(y_true),
        final_fits=[],
        dropped_sessions_outer=dropped_outer,
        dropped_sessions_inner=(),
        reason=reason,
    )


def _unequal_eligibility_results_b():
    """5 subjects, unequal S1-S4 coverage (subjects 1/2 complete-case; 3/4/5 not). Every
    row's radar residual error == its own session index (1/2/3/4), y_true == 0.5 constant,
    so radar's per-session MAE is hand-computable and the session- vs subject-weighted
    aggregates provably diverge (implementation_plan.md:1208-1217)."""
    coverage = {1: (1, 2, 3, 4), 2: (1, 2, 3, 4), 3: (1, 2, 4), 4: (1, 4), 5: (1, 4)}
    y_true = 0.5
    return [
        _fake_result_b(subj, [(s, y_true, y_true - s) for s in sessions])
        for subj, sessions in coverage.items()
    ]


def test_summarize_per_session_n_eval_and_n_complete_case(config):
    summary = summarize_exp_b(_unequal_eligibility_results_b(), config)
    assert summary["n_eval_by_session"] == {"1": 5, "2": 3, "3": 2, "4": 5}
    assert summary["paired_subject_weighted_complete_case"]["n_complete_case"] == 2  # subjects 1, 2
    assert summary["conditional_exploratory"] is True


def test_summarize_primary_aggregate_provably_differs_from_naive_subject_mean(config):
    results = _unequal_eligibility_results_b()
    summary = summarize_exp_b(results, config)
    assert summary["primary_viable"] is True
    session_weighted = summary["primary_aggregate"]["radar"]["point"]
    assert session_weighted == pytest.approx(2.5)   # mean(1, 2, 3, 4)

    # the naive subject-balanced comparator, computed directly from the same fixture.
    coverage = {1: (1, 2, 3, 4), 2: (1, 2, 3, 4), 3: (1, 2, 4), 4: (1, 4), 5: (1, 4)}
    subject_balanced = np.mean([np.mean(sessions) for sessions in coverage.values()])
    assert session_weighted != pytest.approx(subject_balanced)


def test_summarize_holm_family_size_is_4(config):
    summary = summarize_exp_b(_unequal_eligibility_results_b(), config)
    assert summary["per_session_exploratory"]["holm_family_size"] == 4
    p_values = [summary["per_session_exploratory"][str(s)]["wilcoxon_p"] for s in (1, 2, 3, 4)]
    from dehyd.eval.metrics import holm_adjusted
    expected = holm_adjusted(p_values, family_size=4)
    for s, exp_p in zip((1, 2, 3, 4), expected, strict=True):
        got = summary["per_session_exploratory"][str(s)]["holm_p"]
        assert (got == pytest.approx(exp_p)) or (np.isnan(got) and np.isnan(exp_p))


def test_summarize_per_session_baseline_mae_is_a_bootstrap_ci_not_a_float(config):
    """(C5) baseline_mae must carry the same subject-cluster CI shape as radar_mae."""
    summary = summarize_exp_b(_unequal_eligibility_results_b(), config)
    baseline_mae = summary["per_session_exploratory"]["1"]["baseline_mae"]
    assert isinstance(baseline_mae, dict)
    assert set(baseline_mae) == {"point", "low", "high", "method", "n_eval", "n_skipped", "unreliable"}
    assert not isinstance(baseline_mae, float)


def test_summarize_empty_complete_case_yields_nans_not_an_exception(config):
    """No subject covers all 4 S1-S4 sessions, but every session is still covered by SOME
    subject (primary stays viable) -- isolates the paired/companion NaN behaviour from C4's
    run-level viability check."""
    results = [
        _fake_result_b(1, [(1, 0.5, 0.4), (2, 0.5, 0.4), (3, 0.5, 0.4)]),
        _fake_result_b(2, [(1, 0.5, 0.4), (2, 0.5, 0.4), (4, 0.5, 0.4)]),
        _fake_result_b(3, [(1, 0.5, 0.4), (3, 0.5, 0.4), (4, 0.5, 0.4)]),
        _fake_result_b(4, [(2, 0.5, 0.4), (3, 0.5, 0.4), (4, 0.5, 0.4)]),
    ]
    summary = summarize_exp_b(results, config)   # must not raise
    paired = summary["paired_subject_weighted_complete_case"]
    assert paired["n_complete_case"] == 0
    assert np.isnan(paired["wilcoxon_p"])
    assert np.isnan(paired["mean_difference_radar_minus_baseline"]["point"])
    assert summary["primary_viable"] is True   # every session still covered by someone


def test_summarize_session_globally_missing_sets_primary_not_viable(config):
    """(C4) A session with ZERO out-of-fold rows across the WHOLE run sets
    primary_viable=false / primary_aggregate=null with a named reason -- never a silently-
    degraded three-session mean reported as the four-session primary."""
    results = [
        _fake_result_b(subj, [(s, 0.5, 0.4) for s in (1, 2, 4)])   # session 3 NEVER appears
        for subj in (1, 2, 3, 4)
    ]
    summary = summarize_exp_b(results, config)
    assert summary["n_eval_by_session"]["3"] == 0
    assert summary["primary_viable"] is False
    assert summary["primary_aggregate"] is None
    assert summary["primary_unavailable_reason"] is not None and "3" in summary["primary_unavailable_reason"]


# --------------------------------------------------------- reporting boundary (mirrors C9/C14)


def test_run_and_report_b_smoke_writes_no_performance_value(tmp_path, config, monkeypatch):
    sessions = _make_sessions_b(n_subjects=4)
    _write_store(tmp_path, sessions, config)
    fake_results = _unequal_eligibility_results_b()
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda *a, **k: {})
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)
    import dehyd.eval.exp_b as exp_b_mod
    monkeypatch.setattr(exp_b_mod, "run_exp_b", lambda *a, **k: fake_results)
    monkeypatch.setattr(exp_b_mod, "_assert_mechanism_ok_b", lambda *a, **k: None)

    run_dir = tmp_path / "run"
    outputs = run_and_report_b(config, "10ghz", sessions, tmp_path, run_dir, mode="smoke", analysis_commit="x")

    assert set(outputs) == {"run_log"}
    files = {p.name for p in run_dir.iterdir()}
    assert not any(n.startswith(("metrics_", "predictions_b_", "scatter_b_", "selection_table_b_")) for n in files)


def test_run_and_report_b_full_writes_metrics_and_scatter(tmp_path, config, monkeypatch):
    sessions = _make_sessions_b(n_subjects=4)
    _write_store(tmp_path, sessions, config)
    fake_results = _unequal_eligibility_results_b()
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda *a, **k: {})
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)
    import dehyd.eval.exp_b as exp_b_mod
    monkeypatch.setattr(exp_b_mod, "run_exp_b", lambda *a, **k: fake_results)
    monkeypatch.setattr(exp_b_mod, "_assert_mechanism_ok_b", lambda *a, **k: None)

    run_dir = tmp_path / "run"
    outputs = run_and_report_b(config, "10ghz", sessions, tmp_path, run_dir, mode="full", analysis_commit="x")

    assert "metrics" in outputs and "scatter" in outputs
    assert outputs["metrics"].exists() and outputs["scatter"].exists()
    summary = json.loads(outputs["metrics"].read_text())
    assert summary["conditional_exploratory"] is True
    assert "primary_aggregate" in summary and "per_session_exploratory" in summary


# ---------------------------------------------------------------------- T-M8-variant
#
# `run_exp_b_one_session`/`run_exp_b_session_specific`/`summarize_variant_session`/
# `merge_session_specific_reports` tested directly against `eval/exp_b.py`; the CLI wiring
# (`--session-specific`/`--init-run-group`/`--session`/`--merge-sessions`) and the sbatch
# orchestration live in `test_run_clock_decoupling.py` once step 8 exists.


def _make_sessions_unequal_session_eligibility(n_subjects=6):
    """subjects 1-4 have all 4 sessions; subject 5 skips session 3; subject 6 skips session 2
    -- eligible_subjects_for_session differs across sessions, so each session-specific
    search's own outer folds differ from each other and from the pooled model's."""
    out = []
    for s in range(1, n_subjects + 1):
        if s == 5:
            idxs = (1, 2, 4)
        elif s == 6:
            idxs = (1, 3, 4)
        else:
            idxs = (1, 2, 3, 4)
        for i in idxs:
            out.append({
                "subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
                "delta_m_pct": float(-0.3 * i - 0.05 * s),
            })
    return out


def test_eligible_subjects_for_session_hand_value():
    sessions = _make_sessions_unequal_session_eligibility()
    assert eligible_subjects_for_session(sessions, 1) == [1, 2, 3, 4, 5, 6]
    assert eligible_subjects_for_session(sessions, 2) == [1, 2, 3, 4, 5]
    assert eligible_subjects_for_session(sessions, 3) == [1, 2, 3, 4, 6]
    assert eligible_subjects_for_session(sessions, 4) == [1, 2, 3, 4, 5, 6]


def test_config_fingerprint_deterministic_and_sha256_shaped(config):
    h1 = config_fingerprint(config)
    h2 = config_fingerprint(config)
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64
    int(h1, 16)   # valid hex


def test_session_specific_folds_are_distinct_from_each_other_and_from_pooled():
    sessions = _make_sessions_unequal_session_eligibility()
    pooled_folds = harness.nested_loso_splits(evaluable_subjects_b(sessions))
    pooled_pairs = {(f.test_subject, f.train_subjects) for f in pooled_folds}

    folds_by_session = {
        s: harness.nested_loso_splits(eligible_subjects_for_session(sessions, s)) for s in (1, 2, 3, 4)
    }
    pairs_by_session = {
        s: {(f.test_subject, f.train_subjects) for f in folds} for s, folds in folds_by_session.items()
    }

    # sessions 2 and 3 have genuinely different eligible-subject universes (subject 5/6
    # excluded respectively), so their fold sets must differ from each other...
    assert pairs_by_session[2] != pairs_by_session[3]
    # ...and from the pooled model's (which includes all 6 subjects).
    assert pairs_by_session[2] != pooled_pairs
    assert pairs_by_session[3] != pooled_pairs


def test_run_exp_b_session_specific_sequential_wrapper(tmp_path, config):
    sessions = _make_sessions_b(n_subjects=4)
    _write_store(tmp_path, sessions, config)
    out = run_exp_b_session_specific(config, "10ghz", sessions, tmp_path, seeds=(0,))
    assert set(out) == {1, 2, 3, 4}
    for s, results in out.items():
        assert all(r.reason is None for r in results)
        assert all(int(si) == s for r in results for si in r.test_session_idx)


def test_run_exp_b_one_session_unexpected_exception_propagates_uncaught(tmp_path, config, monkeypatch):
    """(C12) A real bug (here, injected into the protocol-freeze guard path) must crash the
    call, never be downgraded to a placeholder result."""
    sessions = _make_sessions_b(n_subjects=4)
    _write_store(tmp_path, sessions, config)

    import dehyd.eval.exp_b as exp_b_mod

    def boom(*a, **k):
        raise RuntimeError("boom in protocol_freeze_guard, simulating an unexpected failure")

    monkeypatch.setattr(exp_b_mod, "protocol_freeze_guard", boom)

    with pytest.raises(RuntimeError, match="boom"):
        run_exp_b_one_session(config, "10ghz", sessions, tmp_path, 1, seeds=(0,))


def test_summarize_variant_session_has_no_p_value_fields(config):
    results_s = [_fake_result_b(subj, [(1, 0.5, 0.4)]) for subj in (1, 2, 3, 4)]
    summary = summarize_variant_session(results_s, 1, config)
    assert "wilcoxon_p" not in summary
    assert "holm_p" not in summary
    assert set(summary) == {
        "conditional_exploratory", "n_eval", "radar_mae", "baseline_mae",
        "mean_difference", "selection_frequency",
    }


# -------------------------------------------------------------- merge_session_specific_reports


def _write_provenance_json(run_dir, *, commit="abc123", config_hash="hash1", expected_subjects_by_session):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "git": {"commit": commit},
        "extra": {"config_hash": config_hash, "expected_subjects_by_session": expected_subjects_by_session},
    }
    (run_dir / "provenance.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_shard(run_dir, band, session, *, run_group_id=None, analysis_commit="abc123",
                  config_hash="hash1", n_eval_subjects=(1, 2, 3)):
    run_group_id = run_dir.name if run_group_id is None else run_group_id
    shard = {
        "run_group_id": run_group_id, "band": band, "session": session,
        "analysis_commit": analysis_commit, "config_hash": config_hash,
        "seed_set": [1, 2, 3, 4, 5], "n_eval_subjects": list(n_eval_subjects),
        "summary": {"n_eval": len(n_eval_subjects), "radar_mae": {}, "baseline_mae": {}, "mean_difference": {}},
    }
    (run_dir / f"session_specific_{band}_s{session}.json").write_text(json.dumps(shard), encoding="utf-8")


def test_merge_partial_completion_computed_from_present_and_valid_shards(tmp_path):
    run_dir = tmp_path / "20260728T000000000000Z_abcd1234"
    expected = {str(s): [1, 2, 3] for s in (1, 2, 3, 4)}
    _write_provenance_json(run_dir, expected_subjects_by_session=expected)
    _write_shard(run_dir, "10ghz", 1, n_eval_subjects=(1, 2, 3))
    _write_shard(run_dir, "10ghz", 2, n_eval_subjects=(1, 2, 3))
    # sessions 3, 4: no shard file present -- simply absent, not an error.

    merged = merge_session_specific_reports("10ghz", run_dir)
    assert merged["completed_sessions"] == [1, 2]
    assert "1" in merged and "2" in merged
    assert "3" not in merged and "4" not in merged
    assert merged["conditional_exploratory"] is True


def test_merge_raises_on_analysis_commit_mismatch(tmp_path):
    run_dir = tmp_path / "20260728T000000000000Z_abcd1234"
    _write_provenance_json(run_dir, commit="the_real_commit", expected_subjects_by_session={"1": [1, 2, 3]})
    _write_shard(run_dir, "10ghz", 1, analysis_commit="a_stale_commit", n_eval_subjects=(1, 2, 3))
    with pytest.raises(ExpBError, match="session 1"):
        merge_session_specific_reports("10ghz", run_dir)


def test_merge_raises_on_config_hash_mismatch(tmp_path):
    run_dir = tmp_path / "20260728T000000000000Z_abcd1234"
    _write_provenance_json(run_dir, config_hash="the_real_hash", expected_subjects_by_session={"1": [1, 2, 3]})
    _write_shard(run_dir, "10ghz", 1, config_hash="a_stale_hash", n_eval_subjects=(1, 2, 3))
    with pytest.raises(ExpBError, match="session 1"):
        merge_session_specific_reports("10ghz", run_dir)


def test_merge_raises_on_session_field_disagreeing_with_filename(tmp_path):
    run_dir = tmp_path / "20260728T000000000000Z_abcd1234"
    _write_provenance_json(run_dir, expected_subjects_by_session={"1": [1, 2, 3]})
    _write_shard(run_dir, "10ghz", 1, n_eval_subjects=(1, 2, 3))
    shard_path = run_dir / "session_specific_10ghz_s1.json"
    shard = json.loads(shard_path.read_text())
    shard["session"] = 2   # tamper: disagrees with the s1 filename
    shard_path.write_text(json.dumps(shard), encoding="utf-8")
    with pytest.raises(ExpBError):
        merge_session_specific_reports("10ghz", run_dir)


def test_merge_raises_on_run_group_id_mismatch(tmp_path):
    run_dir = tmp_path / "20260728T000000000000Z_abcd1234"
    _write_provenance_json(run_dir, expected_subjects_by_session={"1": [1, 2, 3]})
    _write_shard(run_dir, "10ghz", 1, run_group_id="some_other_run_dir_name", n_eval_subjects=(1, 2, 3))
    with pytest.raises(ExpBError):
        merge_session_specific_reports("10ghz", run_dir)


def test_merge_raises_on_n_eval_subjects_mismatch_against_group_provenance(tmp_path):
    run_dir = tmp_path / "20260728T000000000000Z_abcd1234"
    _write_provenance_json(run_dir, expected_subjects_by_session={"1": [1, 2, 3]})
    _write_shard(run_dir, "10ghz", 1, n_eval_subjects=(1, 2, 9))   # disagrees with [1, 2, 3]
    with pytest.raises(ExpBError):
        merge_session_specific_reports("10ghz", run_dir)
