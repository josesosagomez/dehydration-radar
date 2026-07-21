"""A deterministic nested select-and-refit procedure — the contract for harness.py.

This is TEST CODE, deliberately not in src/. It exists so tests/test_no_leakage.py can
assert the fit-on-train-only property at milestone 1, before harness.py exists (M6).
It defines exactly what the real harness must satisfy: at M6 the leakage tests rebind
to harness.py and this module is deleted.

It is written to be *auditable*, not fast: every fitted quantity is recorded together
with the subject set it was estimated from, so the tests can verify roles rather than
trust the implementation.

Determinism: Ridge has no n_jobs, and BLAS thread counts set from inside a test arrive
too late to matter, so the numeric work runs inside threadpool_limits(1) and pins an
explicitly deterministic solver rather than leaving solver="auto" free to switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dehyd.eval.splits import nested_loso_splits

# A small enumerated grid — the point is the protocol, not the model.
ALPHA_GRID = (0.1, 1.0, 10.0)

RIDGE_SOLVER = "cholesky"  # deterministic; "auto" may switch algorithm


@dataclass
class FitRecord:
    """One fitted quantity and the subject set it was estimated from."""

    quantity: str
    role: str  # "inner_train" or "outer_train"
    subjects: frozenset[int]
    params: dict[str, np.ndarray]


@dataclass
class InnerResult:
    inner_train: frozenset[int]
    inner_val: frozenset[int]
    alpha: float
    score: float
    val_predictions: dict[int, np.ndarray]  # per validation subject
    fits: list[FitRecord] = field(default_factory=list)


@dataclass
class FoldResult:
    test_subject: int
    train_subjects: frozenset[int]      # the outer-training set the audit is checked against
    selected_alpha: float | None
    inner_scores: np.ndarray            # (n_alphas, n_inner_folds)
    inner_results: list[InnerResult]
    final_fits: list[FitRecord]
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    test_score: float


@dataclass
class Dataset:
    """Session-level data: one row per (subject, session)."""

    subjects: np.ndarray   # (n_rows,) subject id per row
    features: np.ndarray   # (n_rows, n_features)
    targets: np.ndarray    # (n_rows,)

    def rows_for(self, subject_set) -> np.ndarray:
        return np.isin(self.subjects, sorted(subject_set))

    def subject_ids(self) -> list[int]:
        return sorted(set(self.subjects.tolist()))


def subject_balanced_mae(
    subjects: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Mean over subjects of each subject's mean |error| across its sessions.

    Subject-balanced, NOT pooled: a subject with more eligible sessions must not
    dominate the objective. With equal session counts the two coincide, which is why
    a deliberately unequal fixture is needed to tell them apart.
    """
    per_subject = [
        np.abs(y_true[subjects == s] - y_pred[subjects == s]).mean()
        for s in sorted(set(subjects.tolist()))
    ]
    return float(np.mean(per_subject))


def _fit_pipeline(x: np.ndarray, y: np.ndarray, alpha: float):
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha, solver=RIDGE_SOLVER).fit(scaler.transform(x), y)
    return scaler, model


def _predict(scaler, model, x: np.ndarray) -> np.ndarray:
    return model.predict(scaler.transform(x))


def _fit_records(scaler, model, role: str, subjects: frozenset[int]) -> list[FitRecord]:
    return [
        FitRecord(
            quantity="scaler",
            role=role,
            subjects=subjects,
            params={"mean_": scaler.mean_.copy(), "scale_": scaler.scale_.copy()},
        ),
        FitRecord(
            quantity="ridge",
            role=role,
            subjects=subjects,
            params={
                "coef_": np.atleast_1d(model.coef_).copy(),
                "intercept_": np.atleast_1d(model.intercept_).copy(),
            },
        ),
    ]


def run_fold(data: Dataset, fold) -> FoldResult:
    """Select alpha on inner folds, refit on all outer-training subjects, then score."""
    with threadpool_limits(1):
        inner_results: list[InnerResult] = []
        scores = np.full((len(ALPHA_GRID), len(fold.inner_folds)), np.nan)

        for j, inner in enumerate(fold.inner_folds):
            train_rows = data.rows_for(inner.train_subjects)
            val_rows = data.rows_for(inner.val_subjects)

            for i, alpha in enumerate(ALPHA_GRID):
                scaler, model = _fit_pipeline(
                    data.features[train_rows], data.targets[train_rows], alpha
                )
                predictions = _predict(scaler, model, data.features[val_rows])
                score = subject_balanced_mae(
                    data.subjects[val_rows], data.targets[val_rows], predictions
                )
                scores[i, j] = score

                inner_results.append(
                    InnerResult(
                        inner_train=inner.train_subjects,
                        inner_val=inner.val_subjects,
                        alpha=alpha,
                        score=score,
                        val_predictions={
                            s: predictions[data.subjects[val_rows] == s]
                            for s in sorted(inner.val_subjects)
                        },
                        fits=_fit_records(scaler, model, "inner_train", inner.train_subjects),
                    )
                )

        # Selection: lowest mean inner score; ties broken toward the larger alpha
        # (the simpler, more regularized model).
        mean_scores = scores.mean(axis=1)
        best = int(np.argmin(mean_scores))
        ties = np.flatnonzero(mean_scores == mean_scores[best])
        selected_alpha = max(ALPHA_GRID[i] for i in ties)

        # Final refit on ALL outer-training subjects, then score the held-out subject.
        train_rows = data.rows_for(fold.train_subjects)
        test_rows = data.rows_for({fold.test_subject})
        scaler, model = _fit_pipeline(
            data.features[train_rows], data.targets[train_rows], selected_alpha
        )
        train_predictions = _predict(scaler, model, data.features[train_rows])
        test_predictions = _predict(scaler, model, data.features[test_rows])
        test_score = subject_balanced_mae(
            data.subjects[test_rows], data.targets[test_rows], test_predictions
        )

    return FoldResult(
        test_subject=fold.test_subject,
        train_subjects=fold.train_subjects,
        selected_alpha=selected_alpha,
        inner_scores=scores,
        inner_results=inner_results,
        final_fits=_fit_records(scaler, model, "outer_train", fold.train_subjects),
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        test_score=test_score,
    )


def run_nested_loso(data: Dataset, **split_kwargs) -> list[FoldResult]:
    """Folds come only from eval/splits.py — never constructed here."""
    folds = nested_loso_splits(data.subject_ids(), **split_kwargs)
    return [run_fold(data, fold) for fold in folds if fold.selectable]


def fit_audit(results: list[FoldResult]) -> list[dict]:
    """Every fitted quantity -> the subject set it was estimated from."""
    audit = []
    for result in results:
        for inner in result.inner_results:
            for record in inner.fits:
                audit.append(
                    {
                        "test_subject": result.test_subject,
                        "quantity": record.quantity,
                        "role": record.role,
                        "fitted_on": record.subjects,
                        "inner_val": inner.inner_val,
                    }
                )
        for record in result.final_fits:
            audit.append(
                {
                    "test_subject": result.test_subject,
                    "quantity": record.quantity,
                    "role": record.role,
                    "fitted_on": record.subjects,
                    "inner_val": None,
                }
            )
    return audit
