"""M9 step 1 — behaviour pin, captured from the pre-M9 code (plan §1 step 1).

Two later M9 steps claim byte-neutrality for every existing (Exp A / Exp B / frozen-suite)
path, and neither claim is verifiable without a trace captured BEFORE the edits:

  * step 3 factors `_bare_model` out of `regressors.build_estimator` and adds the ordinal
    dispatch — the five existing families' estimators and fitted state must not move;
  * step 4 generalizes `harness._viability_reason` (knn check keyed by param name instead
    of family name, plus a 2-D-y class-coverage branch) and updates its one call site in
    `_score_candidates_on_fold` — every 1-D-y path must be bytewise unchanged.

`tests/test_m8_pin.py` already pins the *outer* summary of `run_nested_candidates`
(selected id, `inner_scores` sha, `test_score`, `test_predictions` sha). This file pins the
**full** result object: every `InnerResult` (including its `reason`, per-seed-averaged
score, first-seed `val_predictions`, and every `FitRecord`'s quantity/role/subject set/
fitted-parameter bytes), every `final_fits` record, every `SeedOutcome`, and the selected
`Candidate`'s identity — collapsed into one sha256 per outer fold over a canonical byte
encoding. A drift anywhere inside a fitted model's state (the step-3 risk) or in which
cells are fit at all (the step-4 risk) fails here even though the M8 pin's coarser summary
could survive it.

Two deliberate choices, both forced by the plan:

  * The literals below were captured by running these exact fixtures against the pre-M9
    tree (see HISTORY.md's M9 step-1 entry for the capture method). They are not recomputed
    at test time, so this file is a pin, not a tautology.
  * `_viability_reason`'s outputs are pinned **through the engine** (`InnerResult.reason`),
    never by calling the private function directly. Step 4 changes its signature from
    `(candidate, n_train_rows)` to `(candidate, bundle, train_rows)`; a direct-call pin
    would have to be edited at exactly the moment it is supposed to be evidence, and D2
    requires the step-1 pins to be bytewise intact *after* the harness edit. The reason
    strings are the observable that matters, and they are signature-independent.

Re-run after step 3 and after step 4 — both must stay green with ZERO changes to this file.
"""

import hashlib

import numpy as np

from dehyd.eval.harness import Candidate, Dataset, run_nested_candidates


# ------------------------------------------------------------------ canonical byte trace


def _canonical_bytes(obj) -> bytes:
    """Deterministic byte encoding of everything a `FoldResult` holds.

    Type tags are part of the encoding so that (say) an int 3 and the string "3" can never
    collide, and floats go in as `float.hex()` so NaN and every last mantissa bit survive.
    """
    if obj is None:
        return b"none|"
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return f"ndarray|{arr.dtype.str}|{arr.shape}|".encode() + arr.tobytes()
    if isinstance(obj, (bool, np.bool_)):        # before int: Python bool IS an int
        return f"bool|{bool(obj)}|".encode()
    if isinstance(obj, (int, np.integer)):
        return f"int|{int(obj)}|".encode()
    if isinstance(obj, (float, np.floating)):
        return f"float|{float(obj).hex()}|".encode()
    if isinstance(obj, str):
        return f"str|{obj}|".encode()
    if isinstance(obj, (frozenset, set)):
        return b"set|" + b"".join(_canonical_bytes(v) for v in sorted(obj))
    if isinstance(obj, dict):
        return b"dict|" + b"".join(
            _canonical_bytes(k) + _canonical_bytes(v) for k, v in sorted(obj.items())
        )
    if isinstance(obj, (list, tuple)):
        return b"seq|" + b"".join(_canonical_bytes(v) for v in obj)
    raise TypeError(f"no canonical encoding for {type(obj)!r}")


def _candidate_trace(candidate) -> list:
    return [
        candidate.candidate_id,
        candidate.family,
        candidate.model_params,
        candidate.feature_key,
        candidate.active,
    ]


def _fit_trace(record) -> list:
    return [record.quantity, record.role, record.subjects, record.params]


def _inner_trace(inner) -> list:
    return [
        inner.inner_train,
        inner.inner_val,
        inner.candidate_id,
        inner.score,
        inner.val_predictions,
        [_fit_trace(f) for f in inner.fits],
        inner.reason,
    ]


def _fold_trace_sha(fold) -> str:
    """One sha256 over the COMPLETE fold result, in the engine's own assembly order."""
    trace = [
        fold.test_subject,
        fold.train_subjects,
        None if fold.selected is None else _candidate_trace(fold.selected),
        fold.inner_scores,
        [_inner_trace(ir) for ir in fold.inner_results],
        [_fit_trace(f) for f in fold.final_fits],
        fold.train_predictions,
        fold.test_predictions,
        fold.test_score,
        [[s.seed, s.train_predictions, s.test_predictions, s.test_score] for s in fold.seed_outcomes],
    ]
    return hashlib.sha256(_canonical_bytes(trace)).hexdigest()


# ------------------------------------------------------------------ fixtures (frozen)


def _make_dataset(n_subjects=8, sessions=4, n_features=5, seed=20260728) -> Dataset:
    """The M8 pin's dataset, verbatim (same generator, same seed).

    Deliberately the same inputs as `test_m8_pin.py`: if this file's trace fails while the
    M8 pin still passes, the drift is localized to what the trace covers and the M8 summary
    does not (fitted state, reasons, per-seed outcomes) — which is exactly the step-3/step-4
    diagnostic this pin exists to give.
    """
    rng = np.random.default_rng(seed)
    subjects, feats, targs = [], [], []
    for s in range(1, n_subjects + 1):
        for _ in range(sessions):
            x = rng.normal(size=n_features)
            subjects.append(s)
            feats.append(x)
            targs.append(float(x[0] * 2.0 - x[1] + 0.5 * x[2] + rng.normal(scale=0.1)))
    return Dataset(np.array(subjects), np.array(feats, dtype=float), np.array(targs, dtype=float))


def _trace_candidates():
    """All five existing families (step-3's byte-neutrality surface) plus one non-evaluable
    knn, so the viability path's reason strings are inside the trace too."""
    return [
        Candidate("ridge_a1.0", "ridge", (("alpha", 1.0),)),
        Candidate("svr", "svr", (("C", 1.0), ("epsilon", 0.1))),
        Candidate("knn5", "knn", (("n_neighbors", 5),)),
        Candidate("knn_huge", "knn", (("n_neighbors", 999),)),
        Candidate("rf", "rf", (("n_estimators", 50), ("max_depth", 3))),
        Candidate("gbm", "gbm", (("n_estimators", 50), ("learning_rate", 0.1), ("max_depth", 2))),
    ]


def _viability_dataset() -> Dataset:
    """6 subjects x 2 sessions, so every inner-training row count is hand-derivable.

    Outer fold: 5 training subjects -> GroupKFold(min(5, 5)) -> 5 inner folds, each holding
    out exactly one subject, so every inner-training set is 4 subjects x 2 sessions = 8 rows.
    """
    rng = np.random.default_rng(20260730)
    subjects, feats, targs = [], [], []
    for s in range(1, 7):
        for _ in range(2):
            x = rng.normal(size=3)
            subjects.append(s)
            feats.append(x)
            targs.append(float(x[0] - 0.5 * x[1]))
    return Dataset(np.array(subjects), np.array(feats, dtype=float), np.array(targs, dtype=float))


# ------------------------------------------------------------------ captured pin values


# One sha256 per outer fold over `_fold_trace_sha`'s canonical encoding, captured on the
# pre-M9 tree. Fold order is the engine's (ascending test subject).
BYTE_TRACE_PIN = [
    (1, "d91558affc81ee5d39f135783f526fb73e9379e5a7d48f973c24f9990a13efef"),
    (2, "41d2310848d3ee4d791edbe5cead2a5c816312121e2c09437d18e9e8eaed2da3"),
    (3, "712c723bbd5fee616c0917b71064398a9b40938a2a40fe460e72b2604260bc2f"),
    (4, "e226d7a73bd488ee91ea892834913a9ee35cca4d1e5dd87ff98cf3f9cbf70de7"),
    (5, "691dd9ddbdd5a714b9b9556d0ff94fb6f3e741c3a04a65b2964568549d5f38fc"),
    (6, "a3d0e03a32ff479788d83be582362e09431c68dfdd176e4bcd6675bd50bb4d27"),
    (7, "f99a516ecffbb5d2d49d37daa3f86ea4ed9a280b9d1ef1f95b48d53051d08e15"),
    (8, "13efd51fc01c6612abe5f45508b4ea456b903d12d9d7cc44dd8fe11b0da28e60"),
]


# Hand-derived, NOT captured: inner-training rows = 4 subjects x 2 sessions = 8 for every
# inner fold of `_viability_dataset()`, and the current predicate is `k > n_train_rows`.
VIABILITY_N_TRAIN_ROWS = 8
VIABILITY_REASON_PIN = {
    "ridge_a1.0": None,                                    # non-knn family -> never checked
    "knn_k8": None,                                        # k == n_train_rows: strictly viable
    "knn_k9": "knn_n_neighbors_9_gt_train_rows_8",         # k = n_train_rows + 1: the boundary
    "knn_huge": "knn_n_neighbors_999_gt_train_rows_8",
}


# ------------------------------------------------------------------------------- tests


def test_run_nested_candidates_full_byte_trace_pin():
    """The full result object, not just its summary — the step-3/step-4 byte-neutrality pin."""
    results = run_nested_candidates(_make_dataset(), _trace_candidates(), seeds=(0, 1, 2))
    assert len(results) == len(BYTE_TRACE_PIN)
    for fold, (expected_subject, expected_sha) in zip(results, BYTE_TRACE_PIN, strict=True):
        assert fold.test_subject == expected_subject
        assert _fold_trace_sha(fold) == expected_sha


def test_viability_reason_pin_through_the_engine():
    """`_viability_reason`'s current outputs, observed where the engine consumes them.

    The three expected strings are derived from the arithmetic (8 inner-training rows) and
    the current predicate, not read back from the code. The `knn_k8` / `knn_k9` pair pins
    the comparison as strict `>`: an implementation using `>=` would mark `knn_k8`
    non-evaluable and fail here.
    """
    candidates = [
        Candidate("ridge_a1.0", "ridge", (("alpha", 1.0),)),
        Candidate("knn_k8", "knn", (("n_neighbors", VIABILITY_N_TRAIN_ROWS),)),
        Candidate("knn_k9", "knn", (("n_neighbors", VIABILITY_N_TRAIN_ROWS + 1),)),
        Candidate("knn_huge", "knn", (("n_neighbors", 999),)),
    ]
    results = run_nested_candidates(_viability_dataset(), candidates)

    assert len(results) == 6                       # every subject's fold is selectable
    for fold in results:
        assert fold.inner_scores.shape == (4, 5)   # 4 candidates x 5 inner folds
        for ci, candidate in enumerate(candidates):
            expected_reason = VIABILITY_REASON_PIN[candidate.candidate_id]
            cells = [ir for ir in fold.inner_results if ir.candidate_id == candidate.candidate_id]
            assert len(cells) == 5
            assert all(ir.reason == expected_reason for ir in cells)
            # A non-evaluable cell is scored NaN and carries no fits; a viable one is finite.
            if expected_reason is None:
                assert np.isfinite(fold.inner_scores[ci]).all()
                assert all(ir.fits for ir in cells)
            else:
                assert np.isnan(fold.inner_scores[ci]).all()
                assert all(ir.fits == [] and ir.val_predictions == {} for ir in cells)
