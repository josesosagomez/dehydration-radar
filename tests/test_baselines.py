"""T-M7-baselines: the session-index-only baseline — per-index train means, the O2
global-mean fallback for a training-absent index, train-only fitting, and the FitRecord."""

import numpy as np
import pytest

from dehyd.models.baselines import fit_session_index_baseline, predict_session_index


def _data():
    # 3 subjects x 5 time indices (0..4). Δm% grows with the index (a clock signal).
    subjects, idx, y = [], [], []
    for s in (1, 2, 3):
        for i in range(5):
            subjects.append(s)
            idx.append(i)
            y.append(float(i) + 0.1 * s)
    return np.array(subjects), np.array(idx), np.array(y)


def test_per_index_means_are_training_means():
    subjects, idx, y = _data()
    out = fit_session_index_baseline(subjects, idx, y, {1, 2})  # subject 3 held out
    # index i mean over subjects 1,2 = i + 0.1*mean(1,2) = i + 0.15
    for i in range(5):
        pred = predict_session_index(out.model, [i])[0]
        assert pred == pytest.approx(i + 0.15)


def test_absent_index_falls_back_to_global_train_mean():
    """O2: a time index absent from training predicts the global training mean."""
    subjects = np.array([1, 1, 2, 2])
    idx = np.array([0, 1, 0, 1])          # only indices 0 and 1 seen in training
    y = np.array([2.0, 4.0, 6.0, 8.0])
    out = fit_session_index_baseline(subjects, idx, y, {1, 2})
    assert out.model["global"] == 5.0
    assert predict_session_index(out.model, [4])[0] == 5.0    # unseen index -> global mean
    assert predict_session_index(out.model, [0])[0] == 4.0    # seen index -> its mean


def test_fit_is_train_only():
    """Mutating a NON-training (held-out) subject's rows changes nothing fitted."""
    subjects, idx, y = _data()
    base = fit_session_index_baseline(subjects, idx, y, {1, 2})
    y2 = y.copy()
    y2[subjects == 3] = 999.0  # mutate the held-out subject
    mutated = fit_session_index_baseline(subjects, idx, y2, {1, 2})
    assert base.model == mutated.model
    assert base.fit_record.params["means"].tobytes() == mutated.fit_record.params["means"].tobytes()


def test_fit_record_shape_and_role():
    subjects, idx, y = _data()
    out = fit_session_index_baseline(subjects, idx, y, {1, 2})
    fr = out.fit_record
    assert fr.quantity == "session_index_means"
    assert fr.role == "outer_train"
    assert fr.subjects == frozenset({1, 2})
    assert set(fr.params) == {"indices", "means", "global"}
    assert all(isinstance(v, np.ndarray) for v in fr.params.values())
