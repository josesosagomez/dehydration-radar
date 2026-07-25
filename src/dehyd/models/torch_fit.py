"""A deterministic, single-threaded CPU torch training path — the T18 target.

Structurally distinct from the sklearn engine: inner folds select an EPOCH BUDGET (not a
candidate), and the outer refit trains on all outer-training subjects for exactly that
budget with no early stopping. It shares the harness's contracts (folds from
`eval/splits.py`, `Dataset`, `subject_balanced_mae`, `FitRecord`, `fit_audit`).

Training design (MILESTONE_7_PLAN §2.5, C19): **true patience/min-delta early stopping**
monitored on inner-val `subject_balanced_mae` — NOT a run-to-fixed-max substitute, so this
is the same algorithm Exp D's DL baselines will run (only the constants differ), and T18
protects the real path. The optimizer never sees validation data: each epoch's weights are
a pure function of the train data and the seed, so under an inner-validation mutation the
per-epoch train trajectory is identical over the common prefix of executed epochs (only the
val-driven stop time / checkpoint may differ). Determinism: `torch.manual_seed`,
single-threaded, float64, full-batch (no shuffle).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..eval.metrics import subject_balanced_mae
from ..eval.splits import nested_loso_splits


@dataclass(frozen=True)
class TorchFitSpec:
    max_epochs: int = 25
    patience: int = 5
    min_delta: float = 1e-4
    lr: float = 1e-2
    weight_decay: float = 0.0
    hidden: int = 8


class TinyMLP(nn.Module):
    """A small fixed MLP (n_features -> hidden -> 1). Deterministic init when the caller
    seeds `torch.manual_seed(seed)` immediately before construction."""

    def __init__(self, n_features: int, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


@dataclass
class FitRecord:
    quantity: str
    role: str
    subjects: frozenset
    params: dict


@dataclass
class TorchInnerResult:
    inner_val: frozenset
    n_epochs_selected: int              # gradient steps to reach the best-val checkpoint
    val_history: list
    train_pred_per_epoch: list          # per-epoch train predictions (common-prefix test)
    fits: list


@dataclass
class TorchFoldResult:
    test_subject: int
    train_subjects: frozenset
    epoch_budget: int
    inner_results: list
    final_fits: list
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    test_score: float


# --------------------------------------------------------------- train-only stats


def _normalize_stats(x_tr: np.ndarray):
    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def _sampler_weights(subjects: np.ndarray) -> np.ndarray:
    """Inverse session-count per row, normalized to mean 1 — so each subject contributes
    equally (aligning the loss with the subject-balanced objective). Train-only."""
    subjects = np.asarray(subjects)
    counts = {s: int((subjects == s).sum()) for s in set(subjects.tolist())}
    w = np.array([1.0 / counts[s] for s in subjects], dtype=float)
    return w * (len(w) / w.sum())


def _state_to_numpy(state) -> dict:
    return {k: v.detach().numpy().copy() for k, v in state.items()}


def _train_steps(x_tr_n, y_tr, w_tr, spec, *, seed, max_epochs, val=None):
    """Full-batch weighted-MSE training, deterministic. If `val` is given, early-stop on
    inner-val subject-balanced MAE (patience/min_delta) and keep the best-val checkpoint.

    Returns: (best_state, n_epochs_selected, val_history, train_pred_per_epoch). With no
    `val`, runs exactly `max_epochs` steps and returns the final state (n_epochs_selected =
    max_epochs).
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = TinyMLP(x_tr_n.shape[1], hidden=spec.hidden).double()
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)

    xt = torch.tensor(x_tr_n, dtype=torch.float64)
    yt = torch.tensor(np.asarray(y_tr, dtype=float), dtype=torch.float64)
    wt = torch.tensor(w_tr, dtype=torch.float64)

    val_history: list = []
    train_pred_per_epoch: list = []
    best_metric = float("inf")
    best_state = None
    best_epoch = -1
    since_improve = 0

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        pred = model(xt)
        loss = (wt * (pred - yt) ** 2).mean()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            train_pred_per_epoch.append(model(xt).numpy().copy())

        if val is None:
            continue

        x_val_n, y_val, subj_val = val
        with torch.no_grad():
            vpred = model(torch.tensor(x_val_n, dtype=torch.float64)).numpy()
        metric = subject_balanced_mae(subj_val, y_val, vpred)
        val_history.append(metric)
        if metric < best_metric - spec.min_delta:
            best_metric, best_epoch = metric, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= spec.patience:
                break

    if val is None:
        return copy.deepcopy(model.state_dict()), max_epochs, val_history, train_pred_per_epoch
    if best_state is None:  # never improved past epoch 0's baseline; keep epoch 0
        best_state, best_epoch = copy.deepcopy(model.state_dict()), 0
    return best_state, best_epoch + 1, val_history, train_pred_per_epoch


def _predict(state, spec, x_n) -> np.ndarray:
    model = TinyMLP(x_n.shape[1], hidden=spec.hidden).double()
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x_n, dtype=torch.float64)).numpy()


def _rows(subjects, subject_set) -> np.ndarray:
    return np.isin(subjects, sorted(subject_set))


def run_torch_nested(data, spec: TorchFitSpec, *, seed: int, **split_kwargs) -> list[TorchFoldResult]:
    """Nested LOSO with the torch fit path. Folds come only from `splits.py`.

    Per outer fold: inner folds early-stop → epoch_budget = median of the selected epoch
    counts; outer refit on all outer-training subjects for exactly that many epochs (no
    early stop), then score the held-out subject.
    """
    subjects, features, targets = data.subjects, data.features, data.targets
    folds = nested_loso_splits(data.subject_ids(), **split_kwargs)
    results = []

    for fold in folds:
        if not fold.selectable:
            continue

        inner_results = []
        selected_counts = []
        for inner in fold.inner_folds:
            tr = _rows(subjects, inner.train_subjects)
            va = _rows(subjects, inner.val_subjects)
            mean, std = _normalize_stats(features[tr])
            w = _sampler_weights(subjects[tr])
            x_tr_n = (features[tr] - mean) / std
            x_val_n = (features[va] - mean) / std
            state, n_sel, val_hist, tph = _train_steps(
                x_tr_n, targets[tr], w, spec, seed=seed, max_epochs=spec.max_epochs,
                val=(x_val_n, targets[va], subjects[va]),
            )
            selected_counts.append(n_sel)
            inner_results.append(
                TorchInnerResult(
                    inner_val=inner.val_subjects,
                    n_epochs_selected=n_sel,
                    val_history=val_hist,
                    train_pred_per_epoch=tph,
                    fits=[
                        FitRecord("input_norm", "inner_train", inner.train_subjects,
                                  {"mean": mean.copy(), "std": std.copy()}),
                        FitRecord("sampler_weights", "inner_train", inner.train_subjects,
                                  {"weights": w.copy()}),
                        FitRecord("mlp_state", "inner_train", inner.train_subjects,
                                  _state_to_numpy(state)),
                    ],
                )
            )

        epoch_budget = int(np.median(selected_counts))

        # Final refit on ALL outer-training subjects for exactly epoch_budget steps.
        tr = _rows(subjects, fold.train_subjects)
        te = _rows(subjects, [fold.test_subject])
        mean, std = _normalize_stats(features[tr])
        w = _sampler_weights(subjects[tr])
        x_tr_n = (features[tr] - mean) / std
        state, _, _, _ = _train_steps(
            x_tr_n, targets[tr], w, spec, seed=seed, max_epochs=epoch_budget, val=None,
        )
        train_pred = _predict(state, spec, x_tr_n)
        test_pred = _predict(state, spec, (features[te] - mean) / std)
        test_score = subject_balanced_mae(subjects[te], targets[te], test_pred)

        results.append(
            TorchFoldResult(
                test_subject=fold.test_subject,
                train_subjects=fold.train_subjects,
                epoch_budget=epoch_budget,
                inner_results=inner_results,
                final_fits=[
                    FitRecord("input_norm", "outer_train", fold.train_subjects,
                              {"mean": mean.copy(), "std": std.copy()}),
                    FitRecord("sampler_weights", "outer_train", fold.train_subjects,
                              {"weights": w.copy()}),
                    FitRecord("mlp_state", "outer_train", fold.train_subjects,
                              _state_to_numpy(state)),
                ],
                train_predictions=train_pred,
                test_predictions=test_pred,
                test_score=test_score,
            )
        )
    return results
