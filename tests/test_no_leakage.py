"""THE leakage test. Green from milestone 1 and kept green forever.

Makes subject-level leakage structurally detectable rather than a matter of intent.
Four parts:

  A (T1-T6)   split structure
  B (T7-T9)   frame mapping, plus R1 on the real data
  C (T10-T18) the strong mutation property test, at BOTH CV levels
  D (T19)     fit-audit role structure

The mandatory suite needs no private data (data/ is gitignored, so a clean checkout
must still be green). Real-data checks are realdata-marked: skipped under plain
`pytest`, hard-failing under `pytest --realdata`.

Because R1 is realdata-marked and also skipped by default, the acceptance check that
"only T18 is skipped" must exclude it:

    uv run pytest tests/test_no_leakage.py -m "not realdata"

Within that selection every test must PASS except T18, which is skipped until the
torch fit path exists in harness.py (M6). A green run with everything skipped would be
exactly the silent failure this file exists to prevent.

At milestone 1 Part C runs against tests/reference_procedure.py, which defines the
contract harness.py must satisfy; at M6 it rebinds to the real harness.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
from threadpoolctl import threadpool_info, threadpool_limits

from dehyd.data.ground_truth import GroundTruth
from dehyd.data.manifest import build_manifest
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval.splits import nested_loso_splits

from reference_procedure import (  # tests/ is on sys.path under pytest's default import mode
    ALPHA_GRID,
    Dataset,
    fit_audit,
    run_nested_loso,
    subject_balanced_mae,
)

N_FAST_TIME = 534
N_CHIRPS = 20
FULL_COHORT = list(range(1, 17))


# =========================================================== Part A — split structure


def test_t1_pairwise_disjoint_roles():
    """T1: {inner_train, inner_val, {test}} are pairwise disjoint in every fold."""
    for fold in nested_loso_splits(FULL_COHORT):
        for inner in fold.inner_folds:
            test = {fold.test_subject}
            assert not (inner.train_subjects & inner.val_subjects)
            assert not (inner.train_subjects & test)
            assert not (inner.val_subjects & test)


def test_t2_inner_sets_within_outer_train():
    """T2: no inner set may reach outside the outer-training subjects."""
    for fold in nested_loso_splits(FULL_COHORT):
        for inner in fold.inner_folds:
            assert inner.train_subjects <= fold.train_subjects
            assert inner.val_subjects <= fold.train_subjects


def test_t3_inner_val_partitions_outer_train():
    """T3: each outer-training subject validates exactly once."""
    for fold in nested_loso_splits(FULL_COHORT):
        validated = [s for inner in fold.inner_folds for s in inner.val_subjects]
        assert sorted(validated) == sorted(fold.train_subjects)


def test_t4_each_subject_held_out_once_and_holds_one_role():
    """T4: no subject occupies two roles within a fold; each is test exactly once."""
    folds = nested_loso_splits(FULL_COHORT)
    assert sorted(f.test_subject for f in folds) == FULL_COHORT

    for fold in folds:
        for inner in fold.inner_folds:
            roles = list(inner.train_subjects) + list(inner.val_subjects) + [fold.test_subject]
            assert len(roles) == len(set(roles))


@pytest.mark.parametrize(
    "n_subjects,expected_inner,selectable",
    [(16, 5, True), (6, 5, True), (4, 3, True), (3, 0, False)],
)
def test_t5_adaptive_and_non_selectable(n_subjects, expected_inner, selectable):
    """T5: inner count adapts; below the floor the fold is non-selectable, not degenerate."""
    for fold in nested_loso_splits(list(range(1, n_subjects + 1))):
        assert fold.selectable is selectable
        assert len(fold.inner_folds) == expected_inner


def test_t6_determinism():
    """T6: identical input -> identical folds, no RNG anywhere."""
    assert nested_loso_splits(FULL_COHORT) == nested_loso_splits(FULL_COHORT)


# ============================================================= Part B — frame mapping


@dataclass(frozen=True)
class FakePaths:
    data_10ghz_dir: Path


@pytest.fixture(scope="module")
def synthetic_manifest(tmp_path_factory):
    """A manifest built through the real build_manifest code path, no private data.

    GroundTruth is constructed in memory (it is just two DataFrames) rather than via a
    synthetic workbook — openpyxl cannot write a formula together with a cached value.
    """
    data_dir = tmp_path_factory.mktemp("10ghz")
    subjects = list(range(1, 9))
    for subject in subjects:
        for session_idx in range(5):
            cube = np.zeros((N_FAST_TIME, N_CHIRPS, 4), dtype=np.complex128)
            cube.real[:] = subject
            sio.savemat(
                str(data_dir / f"subject_{subject}_{SESSION_NAMES[session_idx]}.mat"),
                {"framesRadar": cube},
            )

    rows, subject_rows = [], []
    for subject in subjects:
        baseline = 80.0 + subject
        for idx in range(5):
            mass = baseline - 0.3 * idx
            rows.append(
                {
                    "subject": subject,
                    "session_idx": idx,
                    "session_name": SESSION_NAMES[idx],
                    "mass_kg": mass,
                    "delta_m_kg": mass - baseline,
                    "delta_m_pct": (mass - baseline) / baseline * 100.0,
                }
            )
        subject_rows.append(
            {
                "subject": subject,
                "age": 30,
                "height_cm": 175.0,
                "baseline_mass_kg": baseline,
                "bmi": baseline / 1.75**2,
            }
        )
    gt = GroundTruth(sessions=pd.DataFrame(rows), subjects=pd.DataFrame(subject_rows))
    return build_manifest(FakePaths(data_dir), gt)


def _assert_frame_mapping(manifest):
    """T7-T9 as reusable assertions, so R1 can apply them to the real manifest."""
    # T7: (subject, session, frame) is unique.
    keys = manifest[["subject", "session_idx", "frame_idx"]]
    assert not keys.duplicated().any()

    # T8: rel_path -> subject is a function.
    assert (manifest.groupby("rel_path")["subject"].nunique() == 1).all()

    # T9: the executable form of the LOSO invariant.
    subjects = sorted(manifest.subject.unique().tolist())
    for fold in nested_loso_splits(subjects):
        train_rows = manifest[manifest.subject.isin(sorted(fold.train_subjects))]
        assert (train_rows.subject != fold.test_subject).all()
        assert len(train_rows) == len(manifest[manifest.subject != fold.test_subject])

        for inner in fold.inner_folds:
            inner_rows = manifest[manifest.subject.isin(sorted(inner.train_subjects))]
            assert fold.test_subject not in set(inner_rows.subject)
            assert not (set(inner_rows.subject) & inner.val_subjects)


def test_t7_t8_t9_frame_mapping(synthetic_manifest):
    """T7-T9: unique frames, file->subject a function, no test frames in training."""
    _assert_frame_mapping(synthetic_manifest)


def test_t9_no_held_out_frames_in_training_any_session(synthetic_manifest):
    """T9, stated the way the invariant reads: NO frame of ANY session leaks."""
    manifest = synthetic_manifest
    for fold in nested_loso_splits(sorted(manifest.subject.unique().tolist())):
        train_frames = manifest[manifest.subject.isin(sorted(fold.train_subjects))]
        held_out = manifest[manifest.subject == fold.test_subject]

        assert len(held_out) > 0
        for session_idx in held_out.session_idx.unique():
            leaked = train_frames[
                (train_frames.subject == fold.test_subject)
                & (train_frames.session_idx == session_idx)
            ]
            assert leaked.empty


@pytest.mark.realdata
def test_r1_real_manifest_frame_mapping(real_data_paths):
    """R1: the same assertions on the real 80-file cohort.

    Skipped only under the default `pytest`; under `--realdata` absent or incomplete
    data is a hard failure (see tests/conftest.py).
    """
    from dehyd.config import load_config
    from dehyd.data.ground_truth import load_ground_truth

    cfg = load_config("configs/exp_a_regression.yaml")
    manifest = build_manifest(cfg.paths, load_ground_truth(cfg.paths.weight_xlsx))

    assert len(manifest) == 8000
    assert sorted(manifest.subject.unique().tolist()) == FULL_COHORT
    _assert_frame_mapping(manifest)


# ================================================== Part C — mutation property tests


def make_dataset(seed=20260721, n_subjects=8, sessions_per_subject=None, n_features=6):
    """Deterministic synthetic session-level data.

    8 subjects -> 7 outer-training subjects -> a genuine 5-fold inner GroupKFold.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    subjects, features, targets = [], [], []

    for subject in range(1, n_subjects + 1):
        n_sessions = (
            sessions_per_subject[subject] if sessions_per_subject else 5
        )
        offset = rng.normal(0, 0.3)
        for _ in range(n_sessions):
            x = rng.normal(size=n_features)
            subjects.append(subject)
            features.append(x)
            targets.append(float(x[:3].sum() * 0.5 + offset + rng.normal(0, 0.05)))

    return Dataset(
        subjects=np.array(subjects),
        features=np.array(features),
        targets=np.array(targets),
    )


def mutate_subject(data: Dataset, subject: int, *, features=True, labels=True, seed=99):
    """Eligibility-preserving mutation: same rows and membership, different values."""
    rng = np.random.default_rng(seed)
    mutated = Dataset(
        subjects=data.subjects.copy(),
        features=data.features.copy(),
        targets=data.targets.copy(),
    )
    rows = mutated.subjects == subject
    if features:
        mutated.features[rows] = rng.normal(size=mutated.features[rows].shape) * 10 + 5
    if labels:
        mutated.targets[rows] = rng.normal(size=mutated.targets[rows].shape) * 10 + 5

    # The mutation must not change WHICH rows exist — only their values.
    assert np.array_equal(mutated.subjects, data.subjects)
    assert mutated.features.shape == data.features.shape
    return mutated


def fold_by_test_subject(results, subject):
    return next(r for r in results if r.test_subject == subject)


def assert_fits_identical(a, b):
    assert len(a) == len(b)
    for lhs, rhs in zip(a, b, strict=True):
        assert lhs.quantity == rhs.quantity
        assert lhs.subjects == rhs.subjects
        for key, value in lhs.params.items():
            assert value.tobytes() == rhs.params[key].tobytes(), lhs.quantity


def test_t10_determinism_precondition():
    """T10: two unmutated runs are bit-identical.

    Without this, every bit-for-bit comparison below would be vacuous.
    """
    data = make_dataset()
    first = run_nested_loso(data)
    second = run_nested_loso(data)

    for a, b in zip(first, second, strict=True):
        assert a.selected_alpha == b.selected_alpha
        assert a.inner_scores.tobytes() == b.inner_scores.tobytes()
        assert a.train_predictions.tobytes() == b.train_predictions.tobytes()
        assert a.test_predictions.tobytes() == b.test_predictions.tobytes()
        assert_fits_identical(a.final_fits, b.final_fits)


def test_determinism_fixture_is_single_threaded():
    """The thread limit is verified, not merely documented."""
    with threadpool_limits(1):
        assert all(info["num_threads"] == 1 for info in threadpool_info())


@pytest.mark.parametrize(
    "mutate_features,mutate_labels",
    [(True, False), (False, True), (True, True)],
    ids=["features", "labels", "both"],
)
def test_t11_t14_outer_test_mutation_changes_nothing_pre_scoring(
    mutate_features, mutate_labels
):
    """T11-T14: mutating the held-out subject leaves everything decided before scoring.

    Selected config, inner scores, every fitted parameter, and the training-set
    predictions must be bit-identical; only the held-out prediction/score may move.
    """
    data = make_dataset()
    held_out = 3
    mutated = mutate_subject(
        data, held_out, features=mutate_features, labels=mutate_labels
    )

    base = fold_by_test_subject(run_nested_loso(data), held_out)
    after = fold_by_test_subject(run_nested_loso(mutated), held_out)

    assert base.selected_alpha == after.selected_alpha                      # T11
    assert base.inner_scores.tobytes() == after.inner_scores.tobytes()      # T12
    assert_fits_identical(base.final_fits, after.final_fits)                # T13
    for lhs, rhs in zip(base.inner_results, after.inner_results, strict=True):
        assert_fits_identical(lhs.fits, rhs.fits)                           # T13 (inner)
        assert lhs.score == rhs.score
    assert base.train_predictions.tobytes() == after.train_predictions.tobytes()  # T14


def test_t15_power_feature_mutation_moves_the_held_out_prediction():
    """T15: a mutation test that cannot fail proves nothing."""
    data = make_dataset()
    held_out = 3
    mutated = mutate_subject(data, held_out, features=True, labels=False)

    base = fold_by_test_subject(run_nested_loso(data), held_out)
    after = fold_by_test_subject(run_nested_loso(mutated), held_out)

    assert base.test_predictions.tobytes() != after.test_predictions.tobytes()


def test_t15_power_label_mutation_moves_the_score_not_the_prediction():
    """T15: labels do not enter prediction, but do enter the score."""
    data = make_dataset()
    held_out = 3
    mutated = mutate_subject(data, held_out, features=False, labels=True)

    base = fold_by_test_subject(run_nested_loso(data), held_out)
    after = fold_by_test_subject(run_nested_loso(mutated), held_out)

    assert base.test_predictions.tobytes() == after.test_predictions.tobytes()
    assert base.test_score != after.test_score


def test_t16_inner_validation_mutation_leaves_that_folds_fits_untouched():
    """T16: catches fitting on inner_train + inner_val.

    The outer-test mutation cannot detect this, because inner-validation subjects ARE
    outer-training subjects. Scope is deliberate: only the fits of folds where the
    mutated subject is VALIDATION are invariant. Folds where it is inner-train
    legitimately change, and so may the selected config (selection consumes val scores).
    """
    data = make_dataset()
    held_out, mutated_subject = 1, 5
    mutated = mutate_subject(data, mutated_subject, features=True, labels=True)

    base = fold_by_test_subject(run_nested_loso(data), held_out)
    after = fold_by_test_subject(run_nested_loso(mutated), held_out)

    checked = 0
    for lhs, rhs in zip(base.inner_results, after.inner_results, strict=True):
        assert lhs.inner_val == rhs.inner_val
        if mutated_subject in lhs.inner_val:
            assert_fits_identical(lhs.fits, rhs.fits)
            checked += 1
    assert checked > 0, "fixture never put the mutated subject in a validation role"


def test_t16_power_inner_validation_predictions_do_change():
    """T16 power: the mutated subject's own validation predictions must move."""
    data = make_dataset()
    held_out, mutated_subject = 1, 5
    mutated = mutate_subject(data, mutated_subject, features=True, labels=False)

    base = fold_by_test_subject(run_nested_loso(data), held_out)
    after = fold_by_test_subject(run_nested_loso(mutated), held_out)

    moved = False
    for lhs, rhs in zip(base.inner_results, after.inner_results, strict=True):
        if mutated_subject in lhs.inner_val:
            before = lhs.val_predictions[mutated_subject]
            now = rhs.val_predictions[mutated_subject]
            moved = moved or before.tobytes() != now.tobytes()
    assert moved


def test_t17_selection_objective_is_subject_balanced():
    """T17: a pooled-session objective must not pass as subject-balanced.

    With equal session counts the two are numerically identical, so this uses a
    deliberately unequal fixture and checks against a hand-calculated value.
    """
    subjects = np.array([1, 1, 1, 1, 1, 2, 2])
    y_true = np.zeros(7)
    y_pred = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0])

    # Subject-balanced: mean(mean(|1|)*5, mean(|10|)*2) = mean(1, 10) = 5.5
    assert subject_balanced_mae(subjects, y_true, y_pred) == pytest.approx(5.5)
    # Pooled would be (5*1 + 2*10) / 7 = 3.571..., which must NOT be what we compute.
    assert subject_balanced_mae(subjects, y_true, y_pred) != pytest.approx(25 / 7)


def test_t17_unequal_session_counts_run_end_to_end():
    """T17: the procedure works when subjects contribute different session counts."""
    counts = {1: 5, 2: 5, 3: 4, 4: 2, 5: 5, 6: 3, 7: 5, 8: 4}
    data = make_dataset(sessions_per_subject=counts)
    results = run_nested_loso(data)

    assert len(results) == 8
    assert all(r.selected_alpha in ALPHA_GRID for r in results)
    for result in results:
        assert np.isfinite(result.inner_scores).all()


@pytest.mark.skip(reason="torch fit path lands with harness.py at M6")
def test_t18_torch_mutation_property():
    """T18: the torch leg of the mutation property test.

    SKIP SCOPE IS CRITICAL: both guards are inside this function. A module-level
    pytest.importorskip("torch") would skip T1-T17 and T19 as well, letting this file
    report green while none of its core assertions ran.

    torch enters the environment at M4 (WST cross-backend validation), but there is no
    torch TRAINING procedure to test until harness.py at M6 — hence the static skip
    above, which is removed then. Must be green before any torch result is reported.

    Asserts, under the same mutation protocol: bit-identical epoch budget (median of
    inner-fold selections), input-normalization statistics, class/sampler weights,
    early-stopping selection, every state_dict tensor, and training-set predictions —
    only the held-out prediction/score may change.
    """
    pytest.importorskip("torch")
    raise AssertionError("unreachable until M6")


# ================================================================ Part D — fit audit


def test_t19_fit_audit_roles_are_correct():
    """T19: every fitted quantity names the subject set it was estimated from.

    Inner-selection fits come from exactly that fold's inner_train; the final refit
    from exactly the full outer_train; no audited set ever contains the test subject,
    and no inner fit's set contains its own validation subjects.
    """
    data = make_dataset()
    results = run_nested_loso(data)
    audit = fit_audit(results)
    assert audit

    by_test = {r.test_subject: r for r in results}
    for entry in audit:
        fold = by_test[entry["test_subject"]]

        assert entry["test_subject"] not in entry["fitted_on"]
        assert entry["fitted_on"] <= fold.train_subjects

        if entry["role"] == "inner_train":
            assert not (entry["fitted_on"] & entry["inner_val"])
        else:
            assert entry["role"] == "outer_train"
            assert entry["fitted_on"] == fold.train_subjects


def test_t19_audit_covers_every_fitted_quantity():
    """An audit that silently omits a fitted quantity would be worthless."""
    data = make_dataset()
    results = run_nested_loso(data)
    audit = fit_audit(results)

    quantities = {entry["quantity"] for entry in audit}
    assert quantities == {"scaler", "ridge"}

    expected = sum(len(r.inner_results) * 2 + 2 for r in results)
    assert len(audit) == expected
