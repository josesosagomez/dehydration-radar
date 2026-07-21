"""Invariants S1-S7 of the nested-LOSO splitter.

These are re-asserted end to end in test_no_leakage.py; here they are checked at the
unit level, including the small-n boundary behaviour.
"""

import pytest

from dehyd.eval.splits import (
    InnerFold,
    OuterFold,
    SplitError,
    iter_triples,
    nested_loso_splits,
)

FULL_COHORT = list(range(1, 17))


# ------------------------------------------------------------------ basic structure


def test_one_outer_fold_per_subject():
    folds = nested_loso_splits(FULL_COHORT)
    assert len(folds) == 16
    assert [f.test_subject for f in folds] == FULL_COHORT  # sorted order


def test_s4_every_subject_held_out_exactly_once():
    folds = nested_loso_splits(FULL_COHORT)
    held_out = [f.test_subject for f in folds]
    assert sorted(held_out) == FULL_COHORT
    assert len(held_out) == len(set(held_out))


def test_s1_test_subject_never_in_train():
    for fold in nested_loso_splits(FULL_COHORT):
        assert fold.test_subject not in fold.train_subjects
        assert fold.train_subjects == frozenset(FULL_COHORT) - {fold.test_subject}


def test_unsorted_and_noncontiguous_ids_are_normalized():
    ids = [7, 3, 11, 2, 40, 5]
    folds = nested_loso_splits(ids)
    assert [f.test_subject for f in folds] == sorted(ids)


# ------------------------------------------------------------------- inner folds


def test_s2_inner_sets_disjoint_and_within_outer_train():
    for fold in nested_loso_splits(FULL_COHORT):
        for inner in fold.inner_folds:
            assert not (inner.train_subjects & inner.val_subjects)
            assert inner.train_subjects <= fold.train_subjects
            assert inner.val_subjects <= fold.train_subjects
            assert fold.test_subject not in inner.train_subjects
            assert fold.test_subject not in inner.val_subjects


def test_s3_inner_val_sets_partition_outer_train():
    """Each training subject validates exactly once — asserted, not assumed."""
    for fold in nested_loso_splits(FULL_COHORT):
        seen: list[int] = []
        for inner in fold.inner_folds:
            seen.extend(inner.val_subjects)
        assert sorted(seen) == sorted(fold.train_subjects)


def test_inner_train_and_val_reconstruct_outer_train():
    for fold in nested_loso_splits(FULL_COHORT):
        for inner in fold.inner_folds:
            assert inner.train_subjects | inner.val_subjects == fold.train_subjects


def test_s5_sets_non_empty_when_selectable():
    for fold in nested_loso_splits(FULL_COHORT):
        assert fold.selectable
        assert fold.train_subjects
        assert fold.inner_folds
        for inner in fold.inner_folds:
            assert inner.train_subjects
            assert inner.val_subjects


# ---------------------------------------------------------- S6 adaptive fold count


@pytest.mark.parametrize(
    "n_subjects,expected_inner",
    [
        (16, 5),  # full cohort: 15 training subjects -> capped at 5
        (6, 5),   # smoke subset: 5 training subjects -> 5
        (4, 3),   # 3 training subjects -> 3
    ],
)
def test_s6_inner_fold_count_adapts(n_subjects, expected_inner):
    folds = nested_loso_splits(list(range(1, n_subjects + 1)))
    for fold in folds:
        assert fold.selectable
        assert len(fold.inner_folds) == expected_inner


def test_s6_three_subjects_is_non_selectable():
    """n_train = 2 < min_train_subjects: reported, not run with a degenerate split."""
    folds = nested_loso_splits([1, 2, 3])
    assert len(folds) == 3
    for fold in folds:
        assert not fold.selectable
        assert fold.inner_folds == ()
        assert fold.train_subjects  # the fold still exists and is reported


def test_boundary_n_train_three_is_selectable_with_two_subject_inner_fits():
    """Documented consequence of the pool-level rule (owner decision 4)."""
    folds = nested_loso_splits([1, 2, 3, 4])
    fold = folds[0]
    assert fold.selectable
    assert len(fold.inner_folds) == 3
    for inner in fold.inner_folds:
        assert len(inner.train_subjects) == 2
        assert len(inner.val_subjects) == 1


def test_min_train_subjects_is_configurable():
    strict = nested_loso_splits([1, 2, 3, 4], min_train_subjects=4)
    assert all(not f.selectable for f in strict)


def test_n_inner_max_is_respected():
    folds = nested_loso_splits(FULL_COHORT, n_inner_max=3)
    assert all(len(f.inner_folds) == 3 for f in folds)


# --------------------------------------------------------------------- determinism


def test_s7_two_calls_are_identical():
    assert nested_loso_splits(FULL_COHORT) == nested_loso_splits(FULL_COHORT)


def test_input_order_does_not_matter():
    forward = nested_loso_splits([1, 2, 3, 4, 5, 6])
    backward = nested_loso_splits([6, 5, 4, 3, 2, 1])
    assert forward == backward


def test_folds_are_frozen_dataclasses():
    """Folds must not be mutable in place by a consumer."""
    fold = nested_loso_splits(FULL_COHORT)[0]
    assert isinstance(fold, OuterFold)
    assert isinstance(fold.inner_folds[0], InnerFold)
    with pytest.raises(Exception):
        fold.test_subject = 99


# ------------------------------------------------------------------ input validation


def test_duplicate_subjects_rejected():
    with pytest.raises(SplitError, match="duplicates"):
        nested_loso_splits([1, 2, 2, 3])


def test_too_few_subjects_rejected():
    with pytest.raises(SplitError, match="at least 2 subjects"):
        nested_loso_splits([1])


def test_invalid_parameters_rejected():
    with pytest.raises(SplitError, match="n_inner_max"):
        nested_loso_splits(FULL_COHORT, n_inner_max=1)
    with pytest.raises(SplitError, match="min_train_subjects"):
        nested_loso_splits(FULL_COHORT, min_train_subjects=1)


# ------------------------------------------------------------------------ flat view


def test_iter_triples_yields_every_inner_fold():
    folds = nested_loso_splits(FULL_COHORT)
    triples = list(iter_triples(folds))
    assert len(triples) == 16 * 5

    for inner_train, inner_val, test_subject in triples:
        assert test_subject not in inner_train
        assert test_subject not in inner_val
        assert not (inner_train & inner_val)


def test_iter_triples_skips_non_selectable_folds():
    assert list(iter_triples(nested_loso_splits([1, 2, 3]))) == []
