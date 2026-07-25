"""T-M7-torch: the deterministic torch fit path + the inner-validation mutation contract.

The outer-test mutation contract is asserted by the frozen T18 (test_no_leakage.py); here
we cover two-run bit-identity, seed sensitivity, true early stopping, the median epoch
budget, train-only normalization/weights, runtime, and — separately (C13/C18) — the
inner-validation mutation contract over the COMMON PREFIX of executed epochs.
"""

import time

import numpy as np
import pytest

from dehyd.eval.harness import Dataset
from dehyd.models.torch_fit import (
    TorchFitSpec,
    _normalize_stats,
    _sampler_weights,
    _train_steps,
    run_torch_nested,
)

SPEC = TorchFitSpec(max_epochs=25, patience=4, min_delta=1e-4, lr=1e-2)


def make_dataset(n_subjects=8, sessions=5, n_features=6, seed=20260721):
    rng = np.random.default_rng(seed)
    subjects, feats, targs = [], [], []
    for s in range(1, n_subjects + 1):
        offset = rng.normal(0, 0.3)
        for _ in range(sessions):
            x = rng.normal(size=n_features)
            subjects.append(s)
            feats.append(x)
            targs.append(float(x[:3].sum() * 0.5 + offset + rng.normal(0, 0.05)))
    return Dataset(np.array(subjects), np.array(feats, dtype=float), np.array(targs, dtype=float))


def _state_bytes(fits):
    rec = next(f for f in fits if f.quantity == "mlp_state")
    return b"".join(v.tobytes() for _, v in sorted(rec.params.items()))


def fold_for(results, subject):
    return next(r for r in results if r.test_subject == subject)


# ------------------------------------------------------------ determinism / seeds


def test_two_runs_are_bit_identical():
    data = make_dataset()
    a = run_torch_nested(data, SPEC, seed=0)
    b = run_torch_nested(data, SPEC, seed=0)
    for ra, rb in zip(a, b, strict=True):
        assert ra.epoch_budget == rb.epoch_budget
        assert _state_bytes(ra.final_fits) == _state_bytes(rb.final_fits)
        assert ra.train_predictions.tobytes() == rb.train_predictions.tobytes()
        assert ra.test_predictions.tobytes() == rb.test_predictions.tobytes()


def test_different_seeds_give_different_weights():
    data = make_dataset()
    a = run_torch_nested(data, SPEC, seed=1)
    b = run_torch_nested(data, SPEC, seed=2)
    assert _state_bytes(a[0].final_fits) != _state_bytes(b[0].final_fits)


# ---------------------------------------------------- early stopping / budget


def test_early_stopping_can_halt_before_max_epochs():
    """With a plateau-prone target and modest patience, at least one inner fold halts early."""
    data = make_dataset()
    spec = TorchFitSpec(max_epochs=60, patience=3, min_delta=1e-3, lr=2e-2)
    results = run_torch_nested(data, spec, seed=0)
    selected = [ir.n_epochs_selected for r in results for ir in r.inner_results]
    assert max(selected) <= spec.max_epochs
    assert min(selected) < spec.max_epochs  # something stopped early


def test_epoch_budget_is_median_of_inner_selections():
    data = make_dataset()
    results = run_torch_nested(data, SPEC, seed=0)
    for r in results:
        expected = int(np.median([ir.n_epochs_selected for ir in r.inner_results]))
        assert r.epoch_budget == expected


def test_runtime_is_well_under_budget():
    data = make_dataset()
    t0 = time.perf_counter()
    run_torch_nested(data, SPEC, seed=0)
    assert time.perf_counter() - t0 < 90.0


# -------------------------------------------------- train-only under held-out mutation


def test_norm_and_weights_and_state_are_train_only_under_held_out_mutation():
    data = make_dataset()
    held_out = 3
    rng = np.random.default_rng(99)
    mutated = Dataset(data.subjects.copy(), data.features.copy(), data.targets.copy())
    rows = mutated.subjects == held_out
    mutated.features[rows] = rng.normal(size=mutated.features[rows].shape) * 10 + 5
    mutated.targets[rows] = rng.normal(size=int(rows.sum())) * 10 + 5

    base = fold_for(run_torch_nested(data, SPEC, seed=0), held_out)
    mut = fold_for(run_torch_nested(mutated, SPEC, seed=0), held_out)

    assert base.epoch_budget == mut.epoch_budget
    for fb, fm in zip(base.final_fits, mut.final_fits, strict=True):
        assert fb.quantity == fm.quantity
        for k in fb.params:
            assert fb.params[k].tobytes() == fm.params[k].tobytes()
    assert base.train_predictions.tobytes() == mut.train_predictions.tobytes()
    assert base.test_predictions.tobytes() != mut.test_predictions.tobytes()  # held-out moved


# -------------------------------------------- inner-validation mutation (C13/C18)


def test_inner_val_mutation_leaves_train_trajectory_identical_on_common_prefix():
    """Mutating a validation subject cannot change training (it only drives stop/selection).

    Same train data, two different val sets → the per-epoch train trajectory is bit-identical
    over the common prefix of executed epochs; only the stop time / selected checkpoint moves.
    """
    data = make_dataset()
    tr_subjects = {1, 2, 3, 4, 5}
    tr = np.isin(data.subjects, sorted(tr_subjects))
    mean, std = _normalize_stats(data.features[tr])
    w = _sampler_weights(data.subjects[tr])
    x_tr_n = (data.features[tr] - mean) / std

    def run_with_val(val_subject, mutate):
        va = np.isin(data.subjects, [val_subject])
        y_val = data.targets[va].copy()
        if mutate:
            y_val = y_val * 10 + 5  # eligibility-preserving value change
        x_val_n = (data.features[va] - mean) / std
        return _train_steps(
            x_tr_n, data.targets[tr], w, SPEC, seed=0, max_epochs=SPEC.max_epochs,
            val=(x_val_n, y_val, data.subjects[va]),
        )

    _, n_a, _, tph_a = run_with_val(6, mutate=False)
    _, n_b, _, tph_b = run_with_val(6, mutate=True)

    common = min(len(tph_a), len(tph_b))
    assert common >= 1
    for e in range(common):
        assert tph_a[e].tobytes() == tph_b[e].tobytes()  # training ignored val — identical
    # The val-driven selection may differ; the train trajectory over the common prefix may not.
    assert n_a >= 1 and n_b >= 1
