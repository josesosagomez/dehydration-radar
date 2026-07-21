"""THE single source of cross-validation folds.

No other module in this codebase may construct train/val/test indices. Everything —
the sklearn path, the torch path, every experiment, every baseline — consumes folds
from `nested_loso_splits` so that subject-level leakage is structurally impossible
rather than a matter of each caller remembering to be careful.

Structure (implementation_plan.md, "LOSO harness, nested-CV protocol"):

  outer   leave-one-SUBJECT-out over the evaluable subjects; the held-out subject is
          touched only for final scoring.
  inner   GroupKFold(min(n_inner_max, n_train)) over the outer-training subjects only,
          grouped by subject, for model/config selection.

Evaluability (QC, session eligibility, N_eval) is the CALLER's concern from milestone 2
onward — this module never computes it. It receives the subjects that are already
deemed evaluable and produces folds over exactly those.

Determinism: no RNG anywhere. Subjects are sorted, and GroupKFold's assignment is
deterministic, so the same input always yields byte-identical folds.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold

DEFAULT_N_INNER_MAX = 5
DEFAULT_MIN_TRAIN_SUBJECTS = 3


class SplitError(ValueError):
    """Raised when the requested split is impossible or the input is malformed."""


@dataclass(frozen=True)
class InnerFold:
    """One inner (selection) fold, built only from outer-training subjects."""

    train_subjects: frozenset[int]
    val_subjects: frozenset[int]


@dataclass(frozen=True)
class OuterFold:
    """One outer (scoring) fold: exactly one held-out subject."""

    test_subject: int
    train_subjects: frozenset[int]
    selectable: bool
    inner_folds: tuple[InnerFold, ...]


def nested_loso_splits(
    subject_ids: Sequence[int],
    *,
    n_inner_max: int = DEFAULT_N_INNER_MAX,
    min_train_subjects: int = DEFAULT_MIN_TRAIN_SUBJECTS,
) -> list[OuterFold]:
    """Nested leave-one-subject-out folds.

    Args:
        subject_ids: the evaluable subjects. Duplicates are an error — a subject
            cannot appear twice.
        n_inner_max: cap on inner folds; the actual count adapts to the number of
            training subjects.
        min_train_subjects: below this many outer-training subjects the fold is marked
            non-selectable (no inner folds) rather than run with a degenerate split.
            This constrains the outer-training POOL, not each inner fit: at the
            boundary (n_train == 3) GroupKFold(3) trains each inner model on 2
            subjects, which is accepted — the real cohort runs at n_train == 15, and
            the alternative at the boundary is discarding the fold entirely.

    Returns:
        One OuterFold per subject, ordered by subject id.
    """
    ids = list(subject_ids)
    if len(ids) != len(set(ids)):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        raise SplitError(f"subject_ids contains duplicates: {duplicated}")
    if n_inner_max < 2:
        raise SplitError(f"n_inner_max must be >= 2, got {n_inner_max}")
    if min_train_subjects < 2:
        raise SplitError(f"min_train_subjects must be >= 2, got {min_train_subjects}")

    subjects = sorted(ids)
    if len(subjects) < 2:
        raise SplitError(
            f"need at least 2 subjects for leave-one-subject-out, got {len(subjects)}"
        )

    folds: list[OuterFold] = []
    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        selectable = len(train_subjects) >= min_train_subjects

        inner_folds: tuple[InnerFold, ...] = ()
        if selectable:
            inner_folds = _inner_folds(train_subjects, n_inner_max)

        folds.append(
            OuterFold(
                test_subject=test_subject,
                train_subjects=frozenset(train_subjects),
                selectable=selectable,
                inner_folds=inner_folds,
            )
        )
    return folds


def _inner_folds(train_subjects: list[int], n_inner_max: int) -> tuple[InnerFold, ...]:
    """Subject-grouped inner folds over the outer-training subjects only."""
    n_splits = min(n_inner_max, len(train_subjects))
    groups = np.asarray(train_subjects)
    # One row per subject: we are splitting subjects, not frames. Frame-level selection
    # is applied downstream by filtering on these subject sets.
    x = np.zeros((len(train_subjects), 1))

    folds = []
    for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(x, groups=groups):
        folds.append(
            InnerFold(
                train_subjects=frozenset(groups[train_idx].tolist()),
                val_subjects=frozenset(groups[val_idx].tolist()),
            )
        )
    return tuple(folds)


def iter_triples(
    folds: Sequence[OuterFold],
) -> Iterator[tuple[frozenset[int], frozenset[int], int]]:
    """Flat view: (inner_train, inner_val, test_subject) per inner fold.

    Reconciles the plan's "(train, val, test)" phrasing with the fact that one outer
    fold has several inner folds. Non-selectable outer folds contribute nothing.
    """
    for fold in folds:
        for inner in fold.inner_folds:
            yield inner.train_subjects, inner.val_subjects, fold.test_subject
