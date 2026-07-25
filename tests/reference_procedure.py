"""The nested select-and-refit contract — a THIN ADAPTER over the real harness (M7).

At milestones 1–6 this file WAS the procedure under test (a self-contained sklearn
reference). At milestone 7 the leakage suite rebinds to the real
`src/dehyd/eval/harness.py` WITHOUT editing the byte-for-byte-frozen
`tests/test_no_leakage.py`: this module is rewritten to delegate to `harness.py`, so the
frozen tests now exercise the real engine. (The stale "M6" wording that survives inside
the frozen test file means "M7" — the pre-A-M5-2 renumber could not be fixed there
without breaking the freeze.)

It contains ZERO fitting code of its own — no sklearn import, no model. `Dataset` and
`subject_balanced_mae` are re-exported from the engine/metrics; `run_nested_loso` builds
the ridge-over-alpha candidates and returns a thin 9-field VIEW of each engine
`FoldResult` whose fields ARE the engine's own arrays/lists (passed by reference), so
every `.tobytes()` bit-identity assertion in the frozen suite checks engine output.

The reference's old max-alpha tie-break is gone: ties now route through the single frozen
`eval/selection.py::select_candidate` like every other search. The frozen tests pin only
`ALPHA_GRID` membership and cross-run bit-identity, not which alpha wins, so this is sound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dehyd.eval import harness
from dehyd.eval.harness import Dataset  # re-export: the frozen keyword-ctor shape
from dehyd.eval.metrics import subject_balanced_mae  # re-export: the 5.5-pinned metric

# A small enumerated grid — the point is the protocol, not the model.
ALPHA_GRID = (0.1, 1.0, 10.0)

__all__ = ["ALPHA_GRID", "Dataset", "subject_balanced_mae", "run_nested_loso", "fit_audit"]


@dataclass
class FoldResult:
    """The frozen VIEW: exactly the 9 attributes the leakage suite reads. Every field is
    the engine object itself (arrays/lists by reference), so bit-identity assertions test
    the real harness."""

    test_subject: int
    train_subjects: frozenset
    selected_alpha: float | None
    inner_scores: np.ndarray
    inner_results: list
    final_fits: list
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    test_score: float


def _view(result) -> FoldResult:
    selected_alpha = None if result.selected is None else result.selected.params()["alpha"]
    return FoldResult(
        test_subject=result.test_subject,
        train_subjects=result.train_subjects,
        selected_alpha=selected_alpha,
        inner_scores=result.inner_scores,
        inner_results=result.inner_results,
        final_fits=result.final_fits,
        train_predictions=result.train_predictions,
        test_predictions=result.test_predictions,
        test_score=result.test_score,
    )


def run_nested_loso(data: Dataset, **split_kwargs) -> list[FoldResult]:
    """Nested LOSO ridge-over-ALPHA_GRID, via the real harness. Folds come only from
    `eval/splits.py` (through the harness) — never constructed here."""
    candidates = [
        harness.Candidate(f"ridge_a{a}", "ridge", (("alpha", a),), feature_key=None, active=None)
        for a in ALPHA_GRID
    ]
    results = harness.run_nested_candidates(data, candidates, seeds=(0,), **split_kwargs)
    return [_view(r) for r in results]


def fit_audit(results) -> list[dict]:
    """Every fitted quantity -> its subject set. Duck-typed over the views (they expose
    `.inner_results`/`.final_fits`/`.test_subject`)."""
    return harness.fit_audit(results)
