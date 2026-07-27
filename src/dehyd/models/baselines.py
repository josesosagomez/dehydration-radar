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


# --------------------------------------------------------------- Exp B: session means (μ_s)
#
# The single train-only computation of per-session means μ_s, shared by three consumers: the
# residualizing feature provider (subtracts μ_s from the target), the Exp B baseline (predicts
# μ_s, i.e. residual 0), and the fit audit (μ_s must appear as an audited fitted quantity at
# both CV levels). One function, three callers — never three computations.
#
# Deliberately DOES NOT copy Exp A's O2 global-mean fallback: a session with too few eligible
# training subjects is DROPPED (excluded from residualization, the objective, and reporting for
# that fold), never filled from validation/test labels, other subjects, or a global fallback.


def session_means(
    subjects, session_idx, targets, train_subjects, *, min_train_subjects: int = 2
) -> tuple[dict[int, float], tuple[int, ...]]:
    """Train-only per-session mean μ_s = mean of `targets` over rows whose subject is in
    `train_subjects` and whose session is s. A session with fewer than `min_train_subjects`
    DISTINCT eligible training subjects has an undefined/unstable mean and is DROPPED for this
    fold. Returns ({s: mu_s} over kept sessions, sorted dropped session indices).

    Explicitly accumulates over SORTED sessions and SORTED subject membership (rather than
    relying on the caller's row order) so the float sum is bit-identical whether this runs
    serially or inside a parallel worker.
    """
    subjects = np.asarray(subjects)
    session_idx = np.asarray(session_idx)
    targets = np.asarray(targets, dtype=float)
    train_rows = np.isin(subjects, sorted(train_subjects))

    means: dict[int, float] = {}
    dropped: list[int] = []
    # Iterate every session present ANYWHERE (not just among train rows): a session with
    # ZERO eligible training subjects must be classified as dropped too, not silently
    # omitted from both `means` and `dropped` because it never appears in `train_rows`.
    for s in sorted(set(session_idx.tolist())):
        session_mask = train_rows & (session_idx == s)
        subj_here = sorted(set(subjects[session_mask].tolist()))
        if len(subj_here) < min_train_subjects:
            dropped.append(int(s))
            continue
        vals = [float(targets[(subjects == subj) & session_mask][0]) for subj in subj_here]
        means[int(s)] = float(np.mean(vals))
    return means, tuple(sorted(dropped))


def fit_session_mean_baseline(
    subjects, session_idx, targets, train_subjects, *, role: str = "outer_train", min_train_subjects: int = 2
) -> BaselineFitOutcome:
    """Exp B's pre-registered baseline: predict each session's train-only mean Δm% — i.e.
    residual 0. Mirrors `fit_session_index_baseline`'s shape but deliberately differs: NO
    global-mean fallback (Exp A's O2 does not apply here) — a degenerate session is DROPPED,
    matching `session_means`. Emits quantity="session_means" with all-ndarray `FitRecord` params."""
    means, dropped = session_means(
        subjects, session_idx, targets, train_subjects, min_train_subjects=min_train_subjects
    )
    model = {
        "indices": sorted(means),
        "means": [means[i] for i in sorted(means)],
        "dropped": list(dropped),
    }
    fit_record = FitRecord(
        quantity="session_means",
        role=role,
        subjects=frozenset(train_subjects),
        params={
            "indices": np.array(model["indices"], dtype=np.int64),
            "means": np.array(model["means"], dtype=float),
            "dropped": np.array(model["dropped"], dtype=np.int64),
        },
    )
    return BaselineFitOutcome(model=model, fit_record=fit_record)


def predict_session_mean(model: dict, session_idx) -> np.ndarray:
    """Per-row μ_s. RAISES (KeyError) on a session index absent from the model (a dropped or
    unseen session) — by construction a dropped session's rows never reach here; silently
    imputing would reintroduce exactly the leak the drop rule forbids. This is the one place
    Exp B deliberately does NOT copy Exp A's O2 fallback."""
    lookup = dict(zip(model["indices"], model["means"]))
    return np.array([lookup[int(i)] for i in np.asarray(session_idx)], dtype=float)
