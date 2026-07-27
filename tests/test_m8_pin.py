"""M8 step 1 — behaviour pin, captured from the pre-M8 code (plan §1 step 1).

Steps 2 (`metrics.py`'s `_cluster_bootstrap_over_rows` extraction) and 4 (`harness.py`'s
`score_fn` thread) both claim "no behaviour change to Exp A's path" — a claim that is
unverifiable without a byte-for-byte pin captured BEFORE either edit. The literals below
were captured by running the exact fixtures in this file against the pre-M8 tree (see
HISTORY.md's M8 step-1 entry for the capture method); they are not recomputed at test time,
so any future drift in `run_nested_candidates` or `subject_cluster_bootstrap_pooled` shows
up as a failing assertion here, not a silently-passing tautology.

Re-run after step 2 (metrics.py) and after step 4 (harness.py) — both must stay green with
ZERO changes to this file, per the plan's own step-1/step-2/step-4 sequencing.
"""

import hashlib
import math

import numpy as np

from dehyd.eval.harness import Candidate, Dataset, run_nested_candidates
from dehyd.eval.metrics import pooled_pearson_r, session_rmse, subject_cluster_bootstrap_pooled


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ fixtures (frozen)


def _make_dataset(n_subjects=8, sessions=4, n_features=5, seed=20260728) -> Dataset:
    rng = np.random.default_rng(seed)
    subjects, feats, targs = [], [], []
    for s in range(1, n_subjects + 1):
        for _ in range(sessions):
            x = rng.normal(size=n_features)
            subjects.append(s)
            feats.append(x)
            targs.append(float(x[0] * 2.0 - x[1] + 0.5 * x[2] + rng.normal(scale=0.1)))
    return Dataset(np.array(subjects), np.array(feats, dtype=float), np.array(targs, dtype=float))


def _multi_family_candidates():
    return [
        Candidate("ridge_a0.1", "ridge", (("alpha", 0.1),)),
        Candidate("ridge_a1.0", "ridge", (("alpha", 1.0),)),
        Candidate("ridge_a10.0", "ridge", (("alpha", 10.0),)),
        Candidate("svr", "svr", (("C", 1.0), ("epsilon", 0.1))),
        Candidate("knn5", "knn", (("n_neighbors", 5),)),
        Candidate("rf", "rf", (("n_estimators", 50), ("max_depth", 3))),
        Candidate("gbm", "gbm", (("n_estimators", 50), ("learning_rate", 0.1), ("max_depth", 2))),
    ]


def _pooled_bootstrap_fixture():
    subjects = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    rng = np.random.default_rng(20260728)
    y_true = rng.normal(size=12) * 2.0 + 1.0
    y_pred_by_seed = np.stack([y_true + rng.normal(scale=0.3, size=12) for _ in range(3)])
    return subjects, y_true, y_pred_by_seed


# ------------------------------------------------------------------ captured pin values


HARNESS_PIN = [
    {"test_subject": 1, "selected_candidate_id": "ridge_a0.1",
     "inner_scores_sha256": "6d0fa18265b9ba5836cf6c1d40d2079aa1696bb43cd752d9c2de694fb2491bf3",
     "inner_scores_shape": (7, 5), "test_score_hex": "0x1.a6699a838fea8p-5",
     "test_predictions_sha256": "d91a7d36c0cd4d1b6a351555c05e556c6d502d36003c0d6349d0c9825bd0abc8"},
    {"test_subject": 2, "selected_candidate_id": "ridge_a0.1",
     "inner_scores_sha256": "d389c1ba9e1c2b135b76165fc0cee02779c1aa62e9d39349027c7e8de0663329",
     "inner_scores_shape": (7, 5), "test_score_hex": "0x1.06a925bdc7330p-3",
     "test_predictions_sha256": "a82719d826df4491eb7aac711965bee721fc2bb022f4e80ffc344b68f8cf10f7"},
    {"test_subject": 3, "selected_candidate_id": "ridge_a0.1",
     "inner_scores_sha256": "3c35a0465867531d606790722aede34937003c7da9548351dcb9324e9928fb77",
     "inner_scores_shape": (7, 5), "test_score_hex": "0x1.4d2e1039eeb78p-3",
     "test_predictions_sha256": "6035668077c2c6d86b7abeac99dc48f0fe13a58c5eb26caca5b4f34f50c352b5"},
    {"test_subject": 4, "selected_candidate_id": "ridge_a0.1",
     "inner_scores_sha256": "f01a540e444e34bcc02bcd3b29a0b4fb82365d65b9f55799923d8a150077ad5c",
     "inner_scores_shape": (7, 5), "test_score_hex": "0x1.bf9062a78066cp-4",
     "test_predictions_sha256": "cfa880cd7afe57c6bfedad4fcb9a60d7b12a058cb4793900fac8c6bee33a923a"},
    {"test_subject": 5, "selected_candidate_id": "ridge_a0.1",
     "inner_scores_sha256": "4c6b41412ddbb454540f1b1a2ce41aa7ec9689e4a9313e4a836b2bdd1949be42",
     "inner_scores_shape": (7, 5), "test_score_hex": "0x1.8580c322fc428p-5",
     "test_predictions_sha256": "824d02de760241966f217362655c4f90bb0901e9f2611424ed20a6e42644c3be"},
    {"test_subject": 6, "selected_candidate_id": "ridge_a0.1",
     "inner_scores_sha256": "44d6ff8c5862b0f02c4b434894c1456cd35aca45ef8c806b9d8d3666c9fc6c89",
     "inner_scores_shape": (7, 5), "test_score_hex": "0x1.c935b320236a9p-4",
     "test_predictions_sha256": "2dae10301af816c8c9ccb6cf8c52b8c2a8be27859079bbf40464e6579f491908"},
    {"test_subject": 7, "selected_candidate_id": "ridge_a0.1",
     "inner_scores_sha256": "4ad4b942bbdf8f3b332f24564856c7f2105483d8b361d1951ebeb3a38d0f4096",
     "inner_scores_shape": (7, 5), "test_score_hex": "0x1.9a4022f7833f2p-4",
     "test_predictions_sha256": "a5691edfc105745a20e0bb2ab1919fa8d937aa3798a23ccce223bc339e1ba345"},
    {"test_subject": 8, "selected_candidate_id": "ridge_a0.1",
     "inner_scores_sha256": "c9fe666c0bc6658ea3f9f67a96c1cebca7c0e62052686ba831637331387b141a",
     "inner_scores_shape": (7, 5), "test_score_hex": "0x1.ac70c23b6e431p-4",
     "test_predictions_sha256": "52ad89a1f4a08f550f1b7f48b82331a99bae0d9f66111c202e4370a016e3852a"},
]

BOOTSTRAP_PIN_RMSE = {
    "point_hex": "0x1.37a6ac8216c3bp-2", "low_hex": "0x1.017fe93fcd703p-2",
    "high_hex": "0x1.77001002879b2p-2", "method": "bca",
    "n_eval": 6, "n_skipped": 0, "unreliable": False,
}

# constant y_true -> pooled r undefined on every resample -> all skipped, percentile fallback,
# point/low/high are NaN (exercises A-M8-2's skip-and-count machinery pre-extraction).
BOOTSTRAP_PIN_R_SKIP = {
    "point_is_nan": True, "low_is_nan": True, "high_is_nan": True, "method": "percentile",
    "n_eval": 6, "n_skipped": 500, "unreliable": True,
}


# ------------------------------------------------------------------------------- tests


def test_run_nested_candidates_pin():
    results = run_nested_candidates(_make_dataset(), _multi_family_candidates(), seeds=(0, 1, 2))
    assert len(results) == len(HARNESS_PIN)
    for r, expected in zip(results, HARNESS_PIN, strict=True):
        assert r.test_subject == expected["test_subject"]
        assert r.selected.candidate_id == expected["selected_candidate_id"]
        assert r.inner_scores.shape == expected["inner_scores_shape"]
        assert _sha(r.inner_scores.tobytes()) == expected["inner_scores_sha256"]
        assert r.test_score == float.fromhex(expected["test_score_hex"])
        assert _sha(r.test_predictions.tobytes()) == expected["test_predictions_sha256"]


def test_subject_cluster_bootstrap_pooled_pin_rmse():
    subjects, y_true, y_pred_by_seed = _pooled_bootstrap_fixture()
    ci = subject_cluster_bootstrap_pooled(
        subjects, y_true, y_pred_by_seed, session_rmse, b=500, rng_seed=424242
    )
    expected = BOOTSTRAP_PIN_RMSE
    assert ci.point == float.fromhex(expected["point_hex"])
    assert ci.low == float.fromhex(expected["low_hex"])
    assert ci.high == float.fromhex(expected["high_hex"])
    assert ci.method == expected["method"]
    assert ci.n_eval == expected["n_eval"]
    assert ci.n_skipped == expected["n_skipped"]
    assert ci.unreliable == expected["unreliable"]


def test_subject_cluster_bootstrap_pooled_pin_skip_and_count():
    """A-M8-2 relevant: the pre-existing skip-and-count path, pinned before the extraction."""
    subjects, _, y_pred_by_seed = _pooled_bootstrap_fixture()
    y_true_const = np.zeros(12)
    ci = subject_cluster_bootstrap_pooled(
        subjects, y_true_const, y_pred_by_seed, pooled_pearson_r, b=500, rng_seed=424243
    )
    expected = BOOTSTRAP_PIN_R_SKIP
    assert math.isnan(ci.point) == expected["point_is_nan"]
    assert math.isnan(ci.low) == expected["low_is_nan"]
    assert math.isnan(ci.high) == expected["high_is_nan"]
    assert ci.method == expected["method"]
    assert ci.n_eval == expected["n_eval"]
    assert ci.n_skipped == expected["n_skipped"]
    assert ci.unreliable == expected["unreliable"]
