"""The Exp A pre-registered primary comparison: the session-index-only baseline.

Predict Δm% from time of day (session index) alone — the number the radar must beat given
the fasting-clock / hydration confound. K = 1 (no hyperparameters, no inner CV); fit on the
outer-training subjects only, inside the same outer folds, and audited like any fit (it emits
a `FitRecord`).

Owner decisions (Step 0b, 2026-07-25):
  * O2 — behaviour for a time-of-day index ABSENT from an outer-training fold: fall back to
    the GLOBAL training-fold mean Δm%, so every test session stays scored and radar and
    baseline are compared on the identical session set (the paired Wilcoxon needs this).
    (Cannot occur in the full 15-train-subject cohort; matters only for the smoke / sparse.)
  * O3 — the guard path: the baseline uses none of the WST search axes, so it is guarded at
    the CONFIG level (`protocol_freeze_guard(config, active=None)`) before the fit, never with
    a per-fit WST `active` record. That call lives in the entrypoint, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eval.harness import FitRecord


@dataclass
class BaselineFitOutcome:
    model: dict          # {"indices": [...], "means": [...], "global": float}
    fit_record: FitRecord


def fit_session_index_baseline(
    subjects, session_idx, targets, train_subjects, *, role: str = "outer_train"
) -> BaselineFitOutcome:
    """Per-time-of-day mean Δm% over the training rows, plus the global-train-mean fallback (O2).

    Fit uses ONLY rows whose subject is in `train_subjects` (train-only)."""
    subjects = np.asarray(subjects)
    session_idx = np.asarray(session_idx)
    targets = np.asarray(targets, dtype=float)
    train_rows = np.isin(subjects, sorted(train_subjects))

    idx_tr = session_idx[train_rows]
    y_tr = targets[train_rows]
    global_mean = float(y_tr.mean())
    means = {int(i): float(y_tr[idx_tr == i].mean()) for i in sorted(set(idx_tr.tolist()))}

    model = {
        "indices": sorted(means),
        "means": [means[i] for i in sorted(means)],
        "global": global_mean,
    }
    fit_record = FitRecord(
        quantity="session_index_means",
        role=role,
        subjects=frozenset(train_subjects),
        params={
            "indices": np.array(model["indices"], dtype=np.int64),
            "means": np.array(model["means"], dtype=float),
            "global": np.array([global_mean], dtype=float),
        },
    )
    return BaselineFitOutcome(model=model, fit_record=fit_record)


def predict_session_index(model: dict, session_idx) -> np.ndarray:
    """Predict per row: the training mean for that time index, or the global mean (O2) for an
    index absent from training."""
    lookup = dict(zip(model["indices"], model["means"]))
    return np.array([lookup.get(int(i), model["global"]) for i in np.asarray(session_idx)], dtype=float)
