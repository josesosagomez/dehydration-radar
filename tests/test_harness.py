"""T-M7-harness: the generic nested-LOSO engine, on synthetic data (no private data).

Covers: flat inner_results order + score-matrix shape, selection routed through
select_candidate over real GroupKFold folds, two-run bit-identity, non-selectable-fold
exclusion, before_fit-once-per-fit + active-completeness fail-closed, KNN fold-viability,
per-seed outer outcomes, tuned-ε-style train-only extra fits, and the per-family held-out
mutation property. The frozen-suite rebind (step 5) and the store-backed end-to-end
mutation (step 10) live in their own files.
"""

import numpy as np
import pytest

from dehyd.eval import harness
from dehyd.eval.harness import (
    Candidate,
    Dataset,
    FeatureBundle,
    HarnessError,
    fit_audit,
    require_complete_active,
    run_nested_candidates,
)
from dehyd.eval.metrics import subject_balanced_mae
from dehyd.eval.selection import SelectionError


def make_dataset(n_subjects=6, sessions=4, n_features=5, seed=20260721):
    rng = np.random.default_rng(seed)
    subjects, feats, targs = [], [], []
    for s in range(1, n_subjects + 1):
        for _ in range(sessions):
            x = rng.normal(size=n_features)
            subjects.append(s)
            feats.append(x)
            targs.append(float(x[0] * 2.0 - x[1] + rng.normal(scale=0.1)))
    return Dataset(np.array(subjects), np.array(feats, dtype=float), np.array(targs, dtype=float))


def ridge_candidates(alphas=(0.1, 1.0, 10.0)):
    return [Candidate(f"ridge_a{a}", "ridge", (("alpha", a),)) for a in alphas]


def fold_for(results, subject):
    return next(r for r in results if r.test_subject == subject)


# ---------------------------------------------------- structure & determinism


def test_score_matrix_shape_and_flat_inner_results_order():
    data = make_dataset()
    cands = ridge_candidates()
    results = run_nested_candidates(data, cands)
    r = results[0]
    n_c = len(cands)
    n_f = r.inner_scores.shape[1]
    assert r.inner_scores.shape[0] == n_c
    assert len(r.inner_results) == n_c * n_f
    # Flat order is fold-major / candidate-minor: the first n_c share one inner_val.
    first_block = r.inner_results[:n_c]
    assert len({ir.inner_val for ir in first_block}) == 1
    assert [ir.candidate_id for ir in first_block] == [c.candidate_id for c in cands]
    # ...and the block's inner_val differs from the next block's.
    assert r.inner_results[0].inner_val != r.inner_results[n_c].inner_val


def test_two_runs_are_bit_identical():
    data = make_dataset()
    a = run_nested_candidates(data, ridge_candidates())
    b = run_nested_candidates(data, ridge_candidates())
    for ra, rb in zip(a, b, strict=True):
        assert ra.selected.candidate_id == rb.selected.candidate_id
        assert ra.inner_scores.tobytes() == rb.inner_scores.tobytes()
        assert ra.test_predictions.tobytes() == rb.test_predictions.tobytes()
        assert ra.train_predictions.tobytes() == rb.train_predictions.tobytes()


def test_selection_is_routed_through_select_candidate(monkeypatch):
    """The winner must come from eval.selection.select_candidate, not an inline rule."""
    import dehyd.eval.harness as h

    calls = {"n": 0}
    real = h.select_candidate

    def spy(scores):
        calls["n"] += 1
        return real(scores)

    monkeypatch.setattr(h, "select_candidate", spy)
    results = run_nested_candidates(make_dataset(), ridge_candidates())
    assert calls["n"] == len(results)  # once per selectable fold
    # And the selected candidate is genuinely one of the inputs.
    assert all(r.selected.family == "ridge" for r in results)


def test_non_selectable_folds_excluded():
    """3 subjects -> every LOSO fold trains on 2 (< min_train_subjects) -> no results."""
    data = make_dataset(n_subjects=3)
    assert run_nested_candidates(data, ridge_candidates()) == []


def test_before_fit_called_once_per_estimator_fit():
    data = make_dataset(n_subjects=6, sessions=4)
    cands = ridge_candidates(alphas=(0.1, 1.0))  # 2 candidates
    counter = {"n": 0}
    results = run_nested_candidates(data, cands, before_fit=lambda c: counter.__setitem__("n", counter["n"] + 1))
    # per selectable fold: n_candidates * n_inner inner fits + 1 final fit (ridge, 1 seed).
    n_f = results[0].inner_scores.shape[1]
    expected = sum(len(cands) * n_f + 1 for _ in results)
    assert counter["n"] == expected


# ------------------------------------------------------- active completeness (C5)


def test_active_completeness_accepts_full_record():
    require_complete_active(
        {"band": "10ghz", "reduction": "A", "channel": "mag", "tiling": "T1",
         "log_branch": "off", "range_gate_m": (1.0, 2.0), "model_family": "ridge"}
    )  # no raise


@pytest.mark.parametrize(
    "omit",
    ["reduction", "channel", "tiling", "log_branch", "range_gate_m", "model_family"],
)
def test_active_completeness_rejects_each_missing_key(omit):
    full = {"band": "10ghz", "reduction": "A", "channel": "mag", "tiling": "T1",
            "log_branch": "off", "range_gate_m": (1.0, 2.0), "model_family": "ridge"}
    full.pop(omit)
    with pytest.raises(HarnessError, match="missing"):
        require_complete_active(full)


def test_active_completeness_rejects_unknown_band_and_extra_key():
    with pytest.raises(HarnessError, match="band"):
        require_complete_active({"band": "5ghz"})
    full = {"band": "77ghz", "reduction": "slow_time_iq_primary", "channel": "iq",
            "gate_m": (2.0, 4.0), "tiling": "T1_77", "log_branch": "off",
            "model_family": "ridge", "bonus": 1}
    with pytest.raises(HarnessError, match="unexpected"):
        require_complete_active(full)


# ------------------------------------------------------------ fold viability (C6/C21)


def test_knn_candidate_non_evaluable_but_selection_proceeds():
    """A KNN k larger than the inner-train row count is non-evaluable; a viable ridge wins."""
    data = make_dataset(n_subjects=6, sessions=2)  # inner-train ~ 4 subj * 2 = 8 rows
    cands = [
        Candidate("ridge_a1", "ridge", (("alpha", 1.0),)),
        Candidate("knn_huge", "knn", (("n_neighbors", 999),)),
    ]
    results = run_nested_candidates(data, cands)
    assert all(r.selected.family == "ridge" for r in results)
    # the KNN row is non-evaluable (NaN) in the score matrix; its inner results carry a reason.
    r = results[0]
    knn_rows = [ir for ir in r.inner_results if ir.candidate_id == "knn_huge"]
    assert knn_rows and all(ir.reason and "knn_n_neighbors" in ir.reason for ir in knn_rows)
    assert np.isnan(r.inner_scores[1]).all()


def test_all_non_evaluable_raises_selection_error():
    data = make_dataset(n_subjects=6, sessions=2)
    cands = [Candidate("knn_huge", "knn", (("n_neighbors", 999),))]
    with pytest.raises(SelectionError):
        run_nested_candidates(data, cands)


def test_unexpected_fit_error_propagates_not_swallowed():
    """A bug in the estimator must fail loudly, never be recast as non-evaluable."""
    data = make_dataset()
    bad = Candidate("bad", "ridge", (("alpha", 1.0),))

    def before_fit(c):
        raise RuntimeError("boom in a hook, simulating an unexpected failure")

    with pytest.raises(RuntimeError, match="boom"):
        run_nested_candidates(data, [bad], before_fit=before_fit)


# ----------------------------------------------------------- seeds & extra fits


def test_per_seed_outer_outcomes_are_separate():
    data = make_dataset()
    cand = Candidate("rf", "rf", (("n_estimators", 100), ("max_depth", 3)))
    results = run_nested_candidates(data, [cand], seeds=(1, 2, 3))
    r = results[0]
    assert len(r.seed_outcomes) == 3
    preds = [so.test_predictions.tobytes() for so in r.seed_outcomes]
    assert len(set(preds)) > 1  # different seeds -> different predictions, never ensembled


def _eps_provider(dataset, record):
    """A tuned-ε-style provider: an extra fitted quantity computed from train subjects only.
    `record` collects (train_subjects, epsilon) so a test can inspect train-only-ness."""

    def provider(candidate, train_subjects):
        eps = float(dataset.targets[dataset.rows_for(train_subjects)].mean())
        record.append((frozenset(train_subjects), eps))
        return FeatureBundle(
            dataset.subjects, dataset.features, dataset.targets,
            extra_fits=(("tuned_epsilon", {"epsilon": np.array([eps, eps])}),),
        )

    return provider


def test_audit_covers_extra_fits_and_epsilon_is_train_only():
    data = make_dataset()
    record = []
    results = run_nested_candidates(
        data, ridge_candidates(alphas=(1.0,)), data_for=_eps_provider(data, record)
    )
    audit = fit_audit(results)
    eps_entries = [e for e in audit if e["quantity"] == "tuned_epsilon"]
    assert eps_entries, "tuned_epsilon must appear in the fit audit"
    # Every ε was fitted on a subject set that EXCLUDES the held-out test subject.
    for e in eps_entries:
        assert e["test_subject"] not in e["fitted_on"]
    # For inner-train ε records, the fitted-on set is disjoint from that fold's inner_val.
    for e in eps_entries:
        if e["role"] == "inner_train":
            assert e["fitted_on"].isdisjoint(e["inner_val"])


# --------------------------------------------------- per-family mutation property (C11)


def _mutate_held_out(dataset, subject, *, seed=99):
    rng = np.random.default_rng(seed)
    ds = Dataset(dataset.subjects.copy(), dataset.features.copy(), dataset.targets.copy())
    rows = ds.subjects == subject
    ds.features[rows] = rng.normal(size=ds.features[rows].shape)
    ds.targets[rows] = rng.normal(size=int(rows.sum()))
    return ds


FAMILY_CANDIDATES = {
    "ridge": [Candidate("ridge_a1", "ridge", (("alpha", 1.0),)), Candidate("ridge_a10", "ridge", (("alpha", 10.0),))],
    "svr": [Candidate("svr", "svr", (("C", 1.0), ("epsilon", 0.1)))],
    "knn": [Candidate("knn5", "knn", (("n_neighbors", 5),))],
    "rf": [Candidate("rf", "rf", (("n_estimators", 100), ("max_depth", 3)))],
    "gbm": [Candidate("gbm", "gbm", (("n_estimators", 100), ("learning_rate", 0.1), ("max_depth", 2)))],
}


# ------------------------------------------------------ shim purity & by-reference view


def test_reference_procedure_imports_no_sklearn():
    """The rebound shim must be a thin adapter with zero fitting code (no sklearn import)."""
    import ast
    import inspect

    import reference_procedure

    tree = ast.parse(inspect.getsource(reference_procedure))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(n.name for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(m.split(".")[0] == "sklearn" for m in imported), sorted(imported)


def test_shim_view_passes_engine_arrays_by_reference():
    """Every view field IS the engine object, so frozen .tobytes() checks test the harness."""
    import reference_procedure as rp

    data = make_dataset()
    cands = [harness.Candidate(f"ridge_a{a}", "ridge", (("alpha", a),)) for a in rp.ALPHA_GRID]
    results = harness.run_nested_candidates(data, cands, seeds=(0,))
    view = rp._view(results[0])
    assert view.inner_scores is results[0].inner_scores
    assert view.inner_results is results[0].inner_results
    assert view.final_fits is results[0].final_fits
    assert view.test_predictions is results[0].test_predictions
    assert view.selected_alpha in rp.ALPHA_GRID


# ------------------------------------------------------------- score_fn hook (T-M8-harness-hook)


def _negated_subject_balanced_mae(subjects, y_true, y_pred, session_idx):
    """A deliberately-disagreeing scorer: exactly reverses the ranking `subject_balanced_mae`
    would produce (best <-> worst), so a fixture using it proves the hook has real power over
    selection, not just that it is wired through."""
    return -subject_balanced_mae(subjects, y_true, y_pred)


def test_score_fn_none_matches_omitting_it():
    data = make_dataset()
    cands = ridge_candidates()
    a = run_nested_candidates(data, cands, score_fn=None)
    b = run_nested_candidates(data, cands)
    for ra, rb in zip(a, b, strict=True):
        assert ra.selected.candidate_id == rb.selected.candidate_id
        assert ra.inner_scores.tobytes() == rb.inner_scores.tobytes()
        assert ra.test_score == rb.test_score


def test_custom_score_fn_changes_selected_candidate_on_disagreement_fixture():
    data = make_dataset()
    cands = ridge_candidates(alphas=(0.1, 1.0, 10.0))
    default_results = run_nested_candidates(data, cands)
    inverted_results = run_nested_candidates(data, cands, score_fn=_negated_subject_balanced_mae)
    default_ids = [r.selected.candidate_id for r in default_results]
    inverted_ids = [r.selected.candidate_id for r in inverted_results]
    assert default_ids != inverted_ids  # the hook demonstrably changed selection, not just wiring


def test_score_fn_receives_none_session_idx_when_bundle_carries_none():
    """`fixed_feature_provider`'s bundles carry `session_idx=None`; a scorer that tolerates
    None must receive exactly that, not a crash or a synthesized default."""
    data = make_dataset()
    seen = []

    def spy_score_fn(subjects, y_true, y_pred, session_idx):
        seen.append(session_idx)
        return subject_balanced_mae(subjects, y_true, y_pred)

    run_nested_candidates(data, ridge_candidates(alphas=(1.0,)), score_fn=spy_score_fn)
    assert seen
    assert all(v is None for v in seen)


def test_row_subset_provider_masks_recompute_and_empty_val_predictions_not_crash():
    """A provider returning a ROW SUBSET (Exp B's degenerate-session drop pattern): masks
    must recompute from `bundle.subjects` (not the full dataset), and a val subject left with
    zero surviving rows yields an EMPTY (not crashing) `val_predictions` entry, so long as the
    fold's other val subjects still have rows.

    The drop is applied only for INNER calls (`train_subjects` smaller than the full outer
    training set), never for the outer `_final_refit` call -- a provider that ALSO emptied the
    outer test subject's own rows would hit the documented, expected, harness-level zero-row
    `predict` crash (plan §5 trap 1), which is a caller's responsibility elsewhere, not what
    this row-subset-safety property is about."""
    data = make_dataset(n_subjects=6, sessions=4)
    dropped_subject = 1
    full_outer_train_size = len(set(data.subjects.tolist())) - 1

    def provider(candidate, train_subjects):
        if len(train_subjects) == full_outer_train_size:      # the outer _final_refit call
            return FeatureBundle(data.subjects, data.features, data.targets, extra_fits=())
        keep = data.subjects != dropped_subject                # an inner call: apply the drop
        return FeatureBundle(data.subjects[keep], data.features[keep], data.targets[keep], extra_fits=())

    # n_inner_max=2 over 5 outer-training subjects forces every inner val group to have >= 2
    # subjects, so dropping one still leaves the fold's other val subject(s) with real rows.
    results = run_nested_candidates(
        data, ridge_candidates(alphas=(1.0,)), data_for=provider, n_inner_max=2
    )
    assert results

    found_dropped_in_val = False
    for r in results:
        for inner in r.inner_results:
            if dropped_subject in inner.inner_val:
                found_dropped_in_val = True
                assert dropped_subject in inner.val_predictions
                assert len(inner.val_predictions[dropped_subject]) == 0
    assert found_dropped_in_val


@pytest.mark.parametrize("family", ["ridge", "svr", "knn", "rf", "gbm"])
def test_held_out_mutation_leaves_everything_pre_scoring_identical(family):
    data = make_dataset(n_subjects=6, sessions=4)
    held_out = 3
    base = run_nested_candidates(data, FAMILY_CANDIDATES[family])
    mutated = run_nested_candidates(_mutate_held_out(data, held_out), FAMILY_CANDIDATES[family])

    rb = fold_for(base, held_out)
    rm = fold_for(mutated, held_out)
    assert rb.selected.candidate_id == rm.selected.candidate_id
    assert rb.inner_scores.tobytes() == rm.inner_scores.tobytes()
    assert rb.train_predictions.tobytes() == rm.train_predictions.tobytes()
    # every fitted model/scaler parameter identical
    for fb, fm in zip(rb.final_fits, rm.final_fits, strict=True):
        assert fb.quantity == fm.quantity
        assert set(fb.params) == set(fm.params)
        for k in fb.params:
            assert fb.params[k].tobytes() == fm.params[k].tobytes()
    # only the held-out subject's prediction may move (features were mutated).
    assert rb.test_predictions.tobytes() != rm.test_predictions.tobytes()
