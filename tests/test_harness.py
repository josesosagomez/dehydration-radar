"""T-M7-harness: the generic nested-LOSO engine, on synthetic data (no private data).

Covers: flat inner_results order + score-matrix shape, selection routed through
select_candidate over real GroupKFold folds, two-run bit-identity, non-selectable-fold
exclusion, before_fit-once-per-fit + active-completeness fail-closed, KNN fold-viability,
per-seed outer outcomes, tuned-ε-style train-only extra fits, and the per-family held-out
mutation property. The frozen-suite rebind (step 5) and the store-backed end-to-end
mutation (step 10) live in their own files.

T-M9-harness (milestone 9 step 4) appends the ordinal half of `_viability_reason`: the
2-column-y class-coverage rule (`implementation_plan.md:793-801`), its C1 independence
property, the knn check's re-keying on the PARAMETER name (so `ord_a_knn` is covered), and
the `_score` fail-fast on a 2-column y with the default scorer.
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
from dehyd.eval.metrics import class_unit_mae, subject_balanced_mae
from dehyd.eval.selection import SelectionError
from dehyd.eval.splits import nested_loso_splits


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


# ------------------------------------------- ordinal fold viability (T-M9-harness, M9 step 4)
#
# The frozen Exp C rule (`implementation_plan.md:793-801`): a configuration whose
# INNER-TRAINING set lacks any of the five S0-S4 classes is non-evaluable and is recorded,
# not fit. Two wrong implementations these fixtures are built to fail:
#
#   (W1) requiring `set(bundle.y[:, 1])` (a data-derived set) instead of the constant
#        {0, 1, 2, 3, 4). `OrdinalFeatures` mirrors `StoreBackedFeatures`, whose bundles
#        carry ALL session rows — inner-validation and outer-test included — so (W1) makes
#        which cells are fit at all a function of held-out labels, and silently stops
#        requiring a class that QC removed cohort-wide.
#   (W2) checking coverage over all bundle rows rather than the training rows only, which
#        is the same leak with the opposite sign.


ORDINAL_CANDIDATES = [
    Candidate("ord_a_ridge_a1", "ord_a_ridge", (("alpha", 1.0),)),
    Candidate("ord_b_fh_C1", "ord_b_frank_hall", (("C", 1.0),)),
]


def _ordinal_class_mae(subjects, y_true, y_pred, session_idx):
    """Exp C's frozen inner objective in the shape the harness hook expects: the class-unit
    MAE between the truth column `y[:, 1]` and the estimators' class predictions."""
    return class_unit_mae(y_true[:, 1], y_pred)


def _ordinal_dataset(classes_by_subject, *, n_features=3, seed=20260731):
    """A 2-column-y Dataset: `y[:, 0] = L` (continuous), `y[:, 1] = class` (session index).

    `classes_by_subject` maps subject id -> the class labels that subject contributes, one
    row each, so a fixture can make a class rare (present in a single subject) or absent
    from the whole cohort. `L` rises with the class so the thresholded regressor is not
    degenerate.
    """
    rng = np.random.default_rng(seed)
    subjects, feats, targets = [], [], []
    for subject in sorted(classes_by_subject):
        for klass in classes_by_subject[subject]:
            x = rng.normal(size=n_features)
            subjects.append(subject)
            feats.append(x)
            targets.append((float(klass) + 0.1 * float(x[0]), float(klass)))
    return Dataset(
        np.array(subjects), np.array(feats, dtype=float), np.array(targets, dtype=float)
    )


ALL_CLASSES = {s: (0, 1, 2, 3, 4) for s in range(1, 7)}          # 6 subjects x 5 rows
CLASS_4_ONLY_IN_SUBJECT_6 = {**{s: (0, 1, 2, 3) for s in range(1, 6)}, 6: (0, 1, 2, 3, 4)}
CLASS_3_ABSENT_EVERYWHERE = {s: (0, 1, 2, 4) for s in range(1, 7)}
CLASSES_3_AND_4_ABSENT = {s: (0, 1, 2) for s in range(1, 7)}


def _one_fold_cells(dataset, candidates, *, test_subject=1, data_for=None):
    """`(StageOutcome, reason map)` for ONE outer fold, via `_score_candidates_on_fold`.

    Deliberately not `run_nested_candidates`: the missing-class fixtures below produce folds
    where NO candidate is evaluable, and `select_candidate` raises there by design (the
    frozen "the fold contributes no score" path). What these tests are about is the viability
    decision that happens strictly before selection.
    """
    fold = next(
        f for f in nested_loso_splits(dataset.subject_ids()) if f.test_subject == test_subject
    )
    provider = data_for if data_for is not None else harness.fixed_feature_provider(dataset)
    stage = harness._score_candidates_on_fold(
        candidates, fold, (0,), None, provider, score_fn=_ordinal_class_mae
    )
    reasons = tuple(
        (ir.candidate_id, tuple(sorted(ir.inner_val)), ir.reason) for ir in stage.inner_results
    )
    return stage, reasons


def test_two_column_y_runs_through_the_whole_engine_when_every_class_is_covered():
    """The baseline: with all five classes in every inner-training set nothing is marked
    non-evaluable, and a 2-column y flows through fit -> predict -> score -> select -> refit."""
    results = run_nested_candidates(
        _ordinal_dataset(ALL_CLASSES), ORDINAL_CANDIDATES, score_fn=_ordinal_class_mae
    )
    assert len(results) == 6
    for r in results:
        assert all(ir.reason is None for ir in r.inner_results)
        assert np.isfinite(r.inner_scores).all()
        assert r.selected.family in ("ord_a_ridge", "ord_b_frank_hall")
        # predictions are class indices on the frozen grid
        assert set(np.unique(r.test_predictions)).issubset(set(range(5)))


def test_missing_class_in_one_inner_train_marks_exactly_those_cells():
    """Class 4 lives only in subject 6, so on the test-subject-1 fold exactly the inner fold
    that holds subject 6 OUT has an inner-training set missing a class.

    Hand-derived: outer-training subjects are {2,3,4,5,6}; GroupKFold(min(5, 5)) holds out
    one subject per inner fold, so 1 of the 5 inner folds is affected and 4 are not. This
    also fails against (W2): a coverage check over all bundle rows sees subject 6's class-4
    rows sitting in the inner-VALIDATION block and returns None.
    """
    stage, reasons = _one_fold_cells(
        _ordinal_dataset(CLASS_4_ONLY_IN_SUBJECT_6), ORDINAL_CANDIDATES
    )
    blocked = {(cid, val) for cid, val, reason in reasons if reason is not None}
    assert {val for _, val in blocked} == {(6,)}
    # candidate-independent by construction: the rule reads rows, never the candidate
    assert {cid for cid, _ in blocked} == {"ord_a_ridge_a1", "ord_b_fh_C1"}
    assert all(
        reason == "ordinal_missing_class_4_in_inner_train"
        for _, val, reason in reasons
        if val == (6,)
    )
    for inner in stage.inner_results:
        if inner.reason is None:
            assert inner.fits and np.isfinite(inner.score)
        else:
            # recorded, never fit: no FitRecords, no predictions, NaN in the score matrix
            assert inner.fits == [] and inner.val_predictions == {} and np.isnan(inner.score)
    assert np.isnan(stage.inner_scores).sum() == len(ORDINAL_CANDIDATES)
    # Without the check, `ord_b_frank_hall` would raise OrdinalViabilityError on this cell
    # (its 1[class > 3] binary target is single-class) instead of being recorded as skipped.


def test_globally_absent_class_still_blocks_every_cell():
    """C1(ii): class 3 appears NOWHERE in the bundle, and every cell is still blocked.

    This is the direct (W1) discriminator: a `set(bundle.y[:, 1])`-relative predicate finds
    the required set to be {0, 1, 2, 4}, sees every inner-training set cover it, and returns
    None for all ten cells — i.e. it silently stops requiring a class the cohort lost.
    """
    stage, reasons = _one_fold_cells(
        _ordinal_dataset(CLASS_3_ABSENT_EVERYWHERE), ORDINAL_CANDIDATES
    )
    assert all(reason == "ordinal_missing_class_3_in_inner_train" for _, _, reason in reasons)
    assert np.isnan(stage.inner_scores).all()


def test_several_missing_classes_are_all_named_lowest_first():
    """Both absent classes appear in the reason, ascending — the `{c}` slot is the whole
    missing set, so the single-class case stays exactly `ordinal_missing_class_3_...`."""
    _, reasons = _one_fold_cells(_ordinal_dataset(CLASSES_3_AND_4_ABSENT), ORDINAL_CANDIDATES)
    assert all(reason == "ordinal_missing_class_3_4_in_inner_train" for _, _, reason in reasons)


def _collapse_to_class_0(classes):
    """Deletes classes 1-4 from the rows it touches (the (W1) discriminator)."""
    return np.zeros_like(classes)


def _shift_classes_by_two(classes):
    """A pure permutation of the label set: no class is created or destroyed."""
    return (classes + 2.0) % 5.0


def _non_training_class_mutator(dataset, rewrite):
    """A provider that rewrites the CLASS COLUMN of every row outside the fit's OWN training
    subjects — exactly the inner-validation and outer-test rows, recomputed per fit.

    Same subjects, same X, same `L` column: only `y[:, 1]` moves, and only on rows the fit is
    forbidden to learn from.
    """

    def provider(candidate, train_subjects):
        y = dataset.targets.copy()
        held_out = ~np.isin(dataset.subjects, sorted(train_subjects))
        y[held_out, 1] = rewrite(y[held_out, 1])
        return FeatureBundle(dataset.subjects, dataset.features, y, extra_fits=())

    return provider


@pytest.mark.parametrize("rewrite", [_collapse_to_class_0, _shift_classes_by_two])
def test_viability_decisions_ignore_non_training_class_labels(rewrite):
    """C1(i): the coverage predicate is a pure function of the inner-TRAINING rows.

    Rewriting every inner-validation and outer-test class label — including a rewrite that
    deletes class 4 from the bundle entirely — leaves the reason strings and the whole
    viability decision map bytewise identical. Under (W1) the collapse case flips the
    subject-6 cells from `ordinal_missing_class_4_in_inner_train` to None, because the
    data-derived required set shrinks to {0, 1, 2, 3}.
    """
    dataset = _ordinal_dataset(CLASS_4_ONLY_IN_SUBJECT_6)
    base_stage, base_reasons = _one_fold_cells(dataset, ORDINAL_CANDIDATES)
    mutated_stage, mutated_reasons = _one_fold_cells(
        dataset, ORDINAL_CANDIDATES, data_for=_non_training_class_mutator(dataset, rewrite)
    )

    assert mutated_reasons == base_reasons
    assert (
        np.isnan(mutated_stage.inner_scores).tobytes()
        == np.isnan(base_stage.inner_scores).tobytes()
    )
    # ...and the fixture is live: held-out labels moved, so the SCORES computed against them
    # genuinely changed. Without this the invariance above could hold vacuously.
    assert not np.array_equal(
        mutated_stage.inner_scores, base_stage.inner_scores, equal_nan=True
    )


def test_knn_row_count_check_is_keyed_on_the_parameter_not_the_family():
    """`ord_a_knn` carries the identical `n_neighbors` grid inside the thresholding wrapper,
    so one rule must cover both families and produce the same reason string.

    Hand-derived row count: 6 subjects x 5 classes, outer-training 5 subjects, GroupKFold(5)
    holds one out -> every inner-training set is 4 subjects x 5 rows = 20. The k = 20 / k = 21
    pair pins the comparison as strict `>`, exactly as `test_m9_pin.py` pins it for `knn`.
    """
    dataset = _ordinal_dataset(ALL_CLASSES)
    candidates = [
        Candidate("ord_a_knn_k20", "ord_a_knn", (("n_neighbors", 20),)),
        Candidate("ord_a_knn_k21", "ord_a_knn", (("n_neighbors", 21),)),
        Candidate("ord_a_knn_huge", "ord_a_knn", (("n_neighbors", 999),)),
    ]
    _, reasons = _one_fold_cells(dataset, candidates)
    expected = {
        "ord_a_knn_k20": None,
        "ord_a_knn_k21": "knn_n_neighbors_21_gt_train_rows_20",
        "ord_a_knn_huge": "knn_n_neighbors_999_gt_train_rows_20",
    }
    for cid, _, reason in reasons:
        assert reason == expected[cid]


# --------------------------------------------- 2-column y meets the default scorer (trap 1)


def test_default_score_fn_fails_fast_on_two_column_y():
    """`subject_balanced_mae` is defined on a 1-D target. Fed a 2-column y it does not
    reliably crash: `y_true[rows] - y_pred[rows]` broadcasts whenever a subject happens to
    contribute exactly 2 rows, returning a meaningless (2, 2) error block and a plausible
    float. The fail-fast check turns that silent wrong number into a named error."""
    dataset = _ordinal_dataset({1: (0, 1), 2: (2, 3)})
    bundle = FeatureBundle(dataset.subjects, dataset.features, dataset.targets, extra_fits=())
    rows = np.ones(dataset.subjects.shape, dtype=bool)
    with pytest.raises(HarnessError, match="1-D"):
        harness._score(None, bundle, rows, np.zeros(int(rows.sum())))


def test_two_column_y_without_score_fn_raises_through_the_engine():
    """The same fail-fast, reached the way a caller would actually reach it."""
    with pytest.raises(HarnessError, match="1-D"):
        run_nested_candidates(_ordinal_dataset(ALL_CLASSES), ORDINAL_CANDIDATES)
