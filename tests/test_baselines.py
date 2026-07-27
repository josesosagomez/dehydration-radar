"""T-M7-baselines: the session-index-only baseline — per-index train means, the O2
global-mean fallback for a training-absent index, train-only fitting, and the FitRecord.

T-M8-mu (below): the session-mean baseline (μ_s) — the deliberate CONTRAST with O2 (a
degenerate session is dropped, never imputed)."""

import numpy as np
import pytest

from dehyd.models.baselines import (
    fit_session_index_baseline,
    fit_session_mean_baseline,
    predict_session_index,
    predict_session_mean,
    session_means,
)


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


# ------------------------------------------------------------------------ T-M8-mu


def _mu_fixture(session3_dropped=True):
    """5 subjects x 4 sessions. train_subjects = {1,2,3,4}, subject 5 held out. Session 3 has
    only ONE eligible training subject (1) when session3_dropped=True (subjects 2/3/4 skip it)
    -- below min_train_subjects=2, so it must be dropped. targets = 10*session + subject, so
    every session's mean is hand-computable and every value is unique (catches indexing bugs)."""
    subjects, session_idx, targets = [], [], []
    for subj in (1, 2, 3, 4, 5):
        sessions_for_subj = (1, 2, 3, 4) if (subj == 1 or not session3_dropped) else (1, 2, 4)
        for sess in sessions_for_subj:
            subjects.append(subj)
            session_idx.append(sess)
            targets.append(float(10 * sess + subj))
    return np.array(subjects), np.array(session_idx), np.array(targets)


def test_session_means_counts_distinct_training_subjects_not_rows():
    subjects, session_idx, targets = _mu_fixture(session3_dropped=True)
    means, dropped = session_means(subjects, session_idx, targets, {1, 2, 3, 4})
    assert dropped == (3,)               # session 3: only subject 1 eligible -> dropped
    assert set(means) == {1, 2, 4}
    assert means[1] == pytest.approx(np.mean([11, 12, 13, 14]))
    assert means[2] == pytest.approx(np.mean([21, 22, 23, 24]))
    assert means[4] == pytest.approx(np.mean([41, 42, 43, 44]))


def test_session_means_zero_eligible_subjects_is_also_dropped():
    """A session with ZERO rows among train_subjects (not merely < min_train_subjects) must
    still be classified as dropped -- never silently omitted from both `means` and `dropped`
    because it never appears among train_rows at all."""
    subjects, session_idx, targets = _mu_fixture(session3_dropped=True)
    # exclude subject 1 entirely -- the only subject with a session-3 row -- so session 3 has
    # ZERO eligible training subjects among {2, 3, 4}, not just one.
    means, dropped = session_means(subjects, session_idx, targets, {2, 3, 4})
    assert dropped == (3,)
    assert 3 not in means


def test_session_means_dropped_session_never_imputed():
    subjects, session_idx, targets = _mu_fixture(session3_dropped=True)
    means, dropped = session_means(subjects, session_idx, targets, {1, 2, 3, 4})
    assert 3 not in means
    assert dropped == (3,)


def test_session_means_is_train_only():
    """Mutating the held-out subject's (5, not in train_subjects) target leaves mu_s bytewise
    identical."""
    subjects, session_idx, targets = _mu_fixture(session3_dropped=True)
    base, _ = session_means(subjects, session_idx, targets, {1, 2, 3, 4})

    targets2 = targets.copy()
    targets2[subjects == 5] = 999.0
    mutated, _ = session_means(subjects, session_idx, targets2, {1, 2, 3, 4})

    for s in base:
        assert np.array(base[s]).tobytes() == np.array(mutated[s]).tobytes()


def test_session_means_drop_is_independent_across_sessions():
    """Dropping session 3 (too few eligible subjects) leaves mu_1/mu_2/mu_4 bytewise
    unchanged compared to a fixture where session 3 is NOT dropped."""
    subjects_d, session_idx_d, targets_d = _mu_fixture(session3_dropped=True)
    means_dropped, dropped = session_means(subjects_d, session_idx_d, targets_d, {1, 2, 3, 4})
    assert dropped == (3,)

    subjects_k, session_idx_k, targets_k = _mu_fixture(session3_dropped=False)
    means_kept, dropped_none = session_means(subjects_k, session_idx_k, targets_k, {1, 2, 3, 4})
    assert dropped_none == ()

    for s in (1, 2, 4):
        assert np.array(means_dropped[s]).tobytes() == np.array(means_kept[s]).tobytes()


def test_fit_session_mean_baseline_emits_all_ndarray_fit_record():
    subjects, session_idx, targets = _mu_fixture(session3_dropped=True)
    out = fit_session_mean_baseline(subjects, session_idx, targets, {1, 2, 3, 4})
    fr = out.fit_record
    assert fr.quantity == "session_means"
    assert fr.role == "outer_train"
    assert fr.subjects == frozenset({1, 2, 3, 4})
    assert set(fr.params) == {"indices", "means", "dropped"}
    assert all(isinstance(v, np.ndarray) for v in fr.params.values())
    assert fr.params["dropped"].tolist() == [3]


def test_predict_session_mean_matches_fitted_means():
    subjects, session_idx, targets = _mu_fixture(session3_dropped=True)
    out = fit_session_mean_baseline(subjects, session_idx, targets, {1, 2, 3, 4})
    preds = predict_session_mean(out.model, [1, 2, 4])
    means, _ = session_means(subjects, session_idx, targets, {1, 2, 3, 4})
    assert preds.tolist() == pytest.approx([means[1], means[2], means[4]])


def test_predict_session_mean_raises_on_dropped_or_unknown_index():
    subjects, session_idx, targets = _mu_fixture(session3_dropped=True)
    out = fit_session_mean_baseline(subjects, session_idx, targets, {1, 2, 3, 4})
    with pytest.raises(KeyError):
        predict_session_mean(out.model, [3])     # dropped session -- must NOT fall back
    with pytest.raises(KeyError):
        predict_session_mean(out.model, [99])    # unseen session
