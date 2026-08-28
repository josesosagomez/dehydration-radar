"""T-M9-frame-split: the owner's sanctioned EXPLORATORY frame-level random split.

Every number this module produces is forbidden from every report, so what has to be tested
is not accuracy but ISOLATION and HONESTY:

  * isolation — no `eval.splits` import, an output-path ALLOWLIST, and a complete CLI
    invocation that leaves `results/runs/` untouched (C6, D11);
  * honesty — the refit really is the LOSO run's own modal configuration, read from the
    ARTIFACT (C5) whose identity and lineage are recorded (C24), and the one fitted quantity
    that could smuggle in a SECOND leak — the tuned-ε pre-log scale — is provably a function
    of training frames only, computed through the frozen subject-balanced hierarchy (C10).

Hand-computed values below are derived from the plan's own arithmetic (§2.10) before any
implementation was run: the CNN modal-pair + floor-median-budget reduction, the ε hierarchy
on a deliberately unbalanced fixture, and `per_frame_prelog`'s per-frame intermediate.
"""

import ast
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dehyd.config import config_to_dict, load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_b, exp_d, frame_split
from dehyd.eval.frame_split import (
    EXPLORATORY_TAG,
    FRAME_SPLIT_MATRIX,
    KFOLD_SEED_OFFSET,
    TUNED_EPS_AGGREGATION,
    WST_REGRESSION_METRICS,
    WST_REGRESSION_UNIT,
    FrameSplitError,
    _require_exploratory_path,
    exploratory_dir,
    load_source_run,
    modal_classical_config,
    modal_cnn_config,
    per_frame_prelog,
    run_frame_split,
    training_frame_epsilons,
    write_exploratory_provenance,
    write_frame_split_reports,
)
from dehyd.features.store import raw_key, order_key, write_session_store

MODULE = Path(frame_split.__file__)

N_SUBJECTS = 4
N_FRAMES = 5
N_PATHS = 3          # meta order [0, 1, 2] — one path per order
N_TIME = 4
FEATURE_KEY = (0, "A", "mag", 0, "tuned")


# ------------------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def base_config():
    return load_config("configs/exp_a_regression.yaml", "configs/exp_c.yaml")


def _tmp_config(base_config, tmp_path, **paths):
    data_10 = tmp_path / "data10"
    data_10.mkdir(exist_ok=True)
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    return dataclasses.replace(
        base_config,
        paths=dataclasses.replace(base_config.paths, data_10ghz_dir=data_10,
                                  results_dir=results, **paths),
    )


def _sessions(n_subjects=N_SUBJECTS, n_frames=N_FRAMES):
    out = []
    for subject in range(1, n_subjects + 1):
        for session_idx in range(5):
            delta = 0.0 if session_idx == 0 else -(0.4 * session_idx + 0.1 * subject)
            out.append({
                "subject": subject, "session_idx": session_idx,
                "session_name": SESSION_NAMES[session_idx],
                "rel_path": f"subject_{subject}_{SESSION_NAMES[session_idx]}.mat",
                "frame_ids": list(range(n_frames)),
                "delta_m_pct": delta,
                "loss_l": -delta, "class_idx": session_idx,
            })
    return out


def _raw_tensor(rng, n_frames, session_idx, subject):
    """[N, C, P, t] with orders 1 and 2 strictly positive (the raw scattering tensor is
    non-negative there, which is what makes the pre-log scale a magnitude) and a signal that
    depends on the session, so the ordinal fit has something to key on."""
    S = rng.random((n_frames, 1, N_PATHS, N_TIME)) + 0.5
    S[:, :, 1:, :] += 0.7 * session_idx + 0.05 * subject
    S[:, :, 0, :] -= 1.0                                   # order 0 may be signed
    return S


def _write_store(store_dir, sessions, band="10ghz", seed=0):
    rng = np.random.default_rng(seed)
    for s in sessions:
        n = len(s["frame_ids"])
        write_session_store(
            band, s["subject"], s["session_name"],
            {
                raw_key(0, "A", "mag", 0): _raw_tensor(rng, n, s["session_idx"], s["subject"]),
                order_key(0): np.array([0, 1, 2]),
                "sig__raw_beat": (rng.standard_normal((n, 534))
                                  + 1j * rng.standard_normal((n, 534))),
            },
            {"n_frames": n}, store_dir,
        )


def _selection_table(path, rows):
    """An Exp C `selection_table_{band}.csv` in `exp_c._write_selection_table_csv`'s schema."""
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["test_subject", "arm", "feature_key", "family", "params",
                         "n_evaluable_inner_folds", "viability_reason_counts", "reason"])
        for row in rows:
            writer.writerow([row["test_subject"], row["arm"], row["feature_key"], row["family"],
                             row["params"], 5, {}, row.get("reason", "")])


def _exp_c_source(run_dir, band, *, commit="c0ffee", config_hash="cfg0",
                  feature_key=FEATURE_KEY, family="ord_a_ridge", params=None):
    """A minimal but REAL Exp C run dir: the selection-table artifact plus the provenance the
    lineage check reads."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    params = {"alpha": 10.0} if params is None else params
    rows = []
    for arm in ("a", "b"):
        fam = family if arm == "a" else "ord_b_frank_hall"
        par = params if arm == "a" else {"C": 1.0}
        for subject in range(1, N_SUBJECTS + 1):
            rows.append({"test_subject": subject, "arm": arm, "feature_key": repr(feature_key),
                         "family": fam, "params": repr(par)})
    _selection_table(run_dir / f"selection_table_{band}.csv", rows)
    (run_dir / "provenance.json").write_text(
        json.dumps({"git": {"commit": commit}, "extra": {"config_hash": config_hash}}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _exp_a_source(run_dir, band, *, commit="c0ffee", config_hash="cfg0",
                  provenance_config=None, feature_key=FEATURE_KEY, family="ridge",
                  params=None):
    """A minimal real Exp A selection artifact plus its lineage-bearing provenance."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    params = {"alpha": 1.0} if params is None else params
    with (run_dir / f"selection_table_{band}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_subject", "feature_key", "family", "params"])
        for subject in range(1, N_SUBJECTS + 1):
            writer.writerow([subject, repr(feature_key), family, repr(params)])
    provenance = {"git": {"commit": commit}}
    if provenance_config is None:
        provenance["extra"] = {"config_hash": config_hash}
    else:
        provenance["config"] = provenance_config
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance) + "\n", encoding="utf-8"
    )
    return run_dir


# ------------------------------------------------------- structural isolation (constraint 2)


def _module_ast():
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def test_frame_split_never_imports_the_fold_source():
    """`eval/splits.py` is the ONLY source of LOSO folds in this project; the exploratory path
    makes its own leaky split and must not be able to reach for the real one."""
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Import):
            assert all("splits" not in alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert "splits" not in (node.module or "")
            assert all(alias.name != "splits" for alias in node.names)


def test_frame_split_never_reads_a_prelog_store_key():
    """The stored `prelog__*` tuples are per-session medians over ALL of a session's frames.
    Under a pooled frame split, consuming one fits ε on the very rows being scored — a
    SECOND, unsanctioned leak (§5 trap 19). The module must reach for raw arrays instead."""
    tree = _module_ast()
    # docstrings may (and do) DISCUSS the stored tuples; what is forbidden is naming one as a
    # key or calling its key builder, so the prose is excluded and the code is not.
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("prelog_key", "prelog77_key")
        if isinstance(node, ast.Name):
            assert node.id not in ("prelog_key", "prelog77_key")
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            assert "prelog__" not in node.value


def test_output_path_allowlist_admits_only_tagged_exploratory_paths(tmp_path):
    results = tmp_path / "results"
    root = results / "exploratory_frame_split" / "10ghz"

    good = root / f"ordinal_arm_a_{EXPLORATORY_TAG}.json"
    assert _require_exploratory_path(good, results_dir=results) == good.resolve()

    # a run directory — the exact thing D11 forbids this path from creating
    with pytest.raises(FrameSplitError, match="exploratory_frame_split"):
        _require_exploratory_path(results / "runs" / "x" / f"a_{EXPLORATORY_TAG}.json",
                                  results_dir=results)
    # correctly tagged but outside the exploratory root
    with pytest.raises(FrameSplitError, match="exploratory_frame_split"):
        _require_exploratory_path(results / f"a_{EXPLORATORY_TAG}.json", results_dir=results)
    # inside the root but untagged: a stray file that could be read as a LOSO artifact
    with pytest.raises(FrameSplitError, match=EXPLORATORY_TAG):
        _require_exploratory_path(root / "metrics_exp_c_10ghz.json", results_dir=results)
    # ...and an escape attempt via traversal resolves before the check
    with pytest.raises(FrameSplitError):
        _require_exploratory_path(root / ".." / ".." / f"a_{EXPLORATORY_TAG}.json",
                                  results_dir=results)


# ------------------------------------------------------------- the sanctioned 18-run matrix


def test_matrix_is_exactly_the_owner_authorized_eighteen_runs():
    assert len(FRAME_SPLIT_MATRIX) == 18
    assert len(set(FRAME_SPLIT_MATRIX)) == 18
    ordinal = {(t, u, b) for t, u, b in FRAME_SPLIT_MATRIX if t == "ordinal"}
    assert ordinal == {("ordinal", u, b) for u in ("arm_a", "arm_b") for b in ("10ghz", "77ghz")}
    regression = {u for t, u, _ in FRAME_SPLIT_MATRIX if t == "regression"}
    assert regression == {*exp_d.EXPD_FAMILIES, WST_REGRESSION_UNIT}
    assert len(regression) == 7
    assert {
        ("regression", WST_REGRESSION_UNIT, band) for band in ("10ghz", "77ghz")
    } <= set(FRAME_SPLIT_MATRIX)


def test_run_frame_split_refuses_a_unit_outside_the_matrix(base_config, tmp_path):
    config = _tmp_config(base_config, tmp_path)
    with pytest.raises(FrameSplitError, match="sanctioned exploratory runs"):
        run_frame_split(config, "10ghz", "regression", "exp_a", source_run_dir=tmp_path)


# --------------------------------------------------- the modal reductions (hand-computed)


def test_modal_cnn_reduction_is_the_modal_pair_and_the_floored_median_budget():
    """HAND-COMPUTED from §2.10. Eight folds, two (lr, wd) pairs selected four times each:

      (1e-3, 0.0)  on folds 0, 2, 4, 6 with budgets 10, 13, 20, 5
      (3e-4, 1e-4) on folds 1, 3, 5, 7 with budgets  7,  4,  9, 30

    The count ties, so the tie-break picks the pair chosen by the LOWEST fold id -> fold 0's
    (1e-3, 0.0). Its four budgets sort to [5, 10, 13, 20], whose median is (10 + 13)/2 = 11.5
    -> floor -> 11. A `np.median` without the floor would return the non-integral 11.5, and
    taking the median over ALL eight folds would give (9 + 10)/2 = 9.5 -> 9.
    """
    rows = []
    for fold, (lr, wd, budget) in enumerate([
        (1e-3, 0.0, 10), (3e-4, 1e-4, 7), (1e-3, 0.0, 13), (3e-4, 1e-4, 4),
        (1e-3, 0.0, 20), (3e-4, 1e-4, 9), (1e-3, 0.0, 5), (3e-4, 1e-4, 30),
    ]):
        rows.append({"fold_id": str(fold), "learning_rate": str(lr),
                     "weight_decay": str(wd), "epoch_budget": str(budget)})

    resolved = modal_cnn_config(rows)
    assert resolved["lr"] == 1e-3 and resolved["weight_decay"] == 0.0
    assert resolved["epoch_budget"] == 11 and isinstance(resolved["epoch_budget"], int)
    assert resolved["modal_fold_id"] == 0 and resolved["n_selected"] == 4
    assert sorted(resolved["budgets"]) == [5, 10, 13, 20]


def test_modal_cnn_reduction_prefers_a_strict_majority_over_fold_order():
    """The tie-break must only fire on a tie: a pair selected more often wins even when a
    rival was selected on fold 0."""
    rows = [
        {"fold_id": "0", "learning_rate": "3e-4", "weight_decay": "0.0", "epoch_budget": "3"},
        {"fold_id": "1", "learning_rate": "1e-3", "weight_decay": "0.0", "epoch_budget": "8"},
        {"fold_id": "2", "learning_rate": "1e-3", "weight_decay": "0.0", "epoch_budget": "12"},
    ]
    resolved = modal_cnn_config(rows)
    assert (resolved["lr"], resolved["weight_decay"]) == (1e-3, 0.0)
    assert resolved["epoch_budget"] == 10       # median([8, 12]) = 10.0 -> floor -> 10


def test_modal_classical_reduction_ties_toward_the_lowest_fold_and_drops_dead_folds():
    """HAND-COMPUTED: two configurations selected twice each, so the tie resolves to the one
    chosen by the lowest fold id (Exp C's fold identity is its held-out subject). The
    non-selectable fold (a `reason`) must NOT be counted — here it would otherwise give the
    svr configuration a 3-2 majority and flip the winner."""
    ridge = {"feature_key": repr((0, "A", "mag", 0, "off")), "family": "ord_a_ridge",
             "params": repr({"alpha": 100.0})}
    svr = {"feature_key": repr((0, "B", "iq", 1, "frozen")), "family": "ord_a_svr",
           "params": repr({"C": 1.0, "epsilon": 0.1})}
    rows = [
        {"test_subject": 0, "arm": "a", **svr, "reason": "no evaluable inner folds"},
        {"test_subject": 1, "arm": "a", **ridge, "reason": ""},
        {"test_subject": 2, "arm": "a", **ridge, "reason": ""},
        {"test_subject": 3, "arm": "a", **svr, "reason": ""},
        {"test_subject": 4, "arm": "a", **svr, "reason": ""},
    ]
    resolved = modal_classical_config(rows)
    assert resolved["feature_key"] == (0, "A", "mag", 0, "off")
    assert resolved["family"] == "ord_a_ridge" and resolved["params"] == {"alpha": 100.0}
    assert resolved["modal_fold_id"] == 1 and resolved["n_selected"] == 2
    assert resolved["n_folds"] == 4


def test_modal_classical_config_comes_from_the_artifact_not_a_recomputation(base_config, tmp_path):
    """(C5) The artifact is authoritative. Its recorded configuration here is `alpha = 100.0`
    on the `off` branch — a configuration nothing about this fixture's data would produce —
    and rewriting the artifact changes what the run refits, which is only possible if the
    file is what is being read."""
    config = _tmp_config(base_config, tmp_path)
    run_dir = _exp_c_source(tmp_path / "loso", "10ghz", feature_key=(0, "A", "mag", 0, "off"),
                            params={"alpha": 100.0})
    first = load_source_run(config, "10ghz", "ordinal", "arm_a", run_dir)
    assert first.resolved_config["params"] == {"alpha": 100.0}
    assert first.resolved_config["feature_key"] == (0, "A", "mag", 0, "off")

    _exp_c_source(run_dir, "10ghz", feature_key=(0, "A", "mag", 0, "tuned"),
                  params={"alpha": 0.1})
    second = load_source_run(config, "10ghz", "ordinal", "arm_a", run_dir)
    assert second.resolved_config["params"] == {"alpha": 0.1}
    assert second.artifact_sha256 != first.artifact_sha256


def test_source_run_lineage_is_validated_by_name(base_config, tmp_path):
    """(C24) An artifact from another revision or another config describes a selection this
    run cannot claim to be reproducing."""
    config = _tmp_config(base_config, tmp_path)
    run_dir = _exp_c_source(tmp_path / "loso", "10ghz", commit="c0ffee", config_hash="cfg0")

    load_source_run(config, "10ghz", "ordinal", "arm_a", run_dir,
                    analysis_commit="c0ffee", config_hash="cfg0")   # no raise
    with pytest.raises(FrameSplitError, match="analysis_commit"):
        load_source_run(config, "10ghz", "ordinal", "arm_a", run_dir,
                        analysis_commit="different", config_hash="cfg0")
    with pytest.raises(FrameSplitError, match="config_hash"):
        load_source_run(config, "10ghz", "ordinal", "arm_a", run_dir,
                        analysis_commit="c0ffee", config_hash="other")
    with pytest.raises(FrameSplitError, match="does not exist"):
        load_source_run(config, "10ghz", "ordinal", "arm_a", tmp_path / "nope")


def test_exp_a_source_recovers_hash_from_real_provenance_config(base_config, tmp_path):
    """Exp A stores the full canonical config, not ``extra.config_hash``.

    The accepted source must therefore match ``config_fingerprint(config)`` exactly, and a
    one-field mutation in that persisted config must still fail closed.
    """
    config = _tmp_config(base_config, tmp_path)
    expected_hash = exp_b.config_fingerprint(config)
    run_dir = _exp_a_source(
        tmp_path / "exp_a_loso",
        "10ghz",
        provenance_config=config_to_dict(config),
    )

    source = load_source_run(
        config,
        "10ghz",
        "regression",
        WST_REGRESSION_UNIT,
        run_dir,
        analysis_commit="c0ffee",
        config_hash=expected_hash,
    )
    assert source.config_hash == expected_hash

    changed = config_to_dict(config)
    changed["run"]["seed"] += 1
    _exp_a_source(run_dir, "10ghz", provenance_config=changed)
    with pytest.raises(FrameSplitError, match="config_hash"):
        load_source_run(
            config,
            "10ghz",
            "regression",
            WST_REGRESSION_UNIT,
            run_dir,
            analysis_commit="c0ffee",
            config_hash=expected_hash,
        )


# ------------------------------------------------ the tuned-ε recomputation (C10, trap 19)


def test_per_frame_prelog_is_the_prelog_scales_own_per_frame_intermediate():
    """HAND-COMPUTED. S = [2 frames, 1 channel, 3 paths, 2 samples], order [0, 1, 2]:

        frame 0 paths [1,3], [5,7], [9,11]   -> time means 2, 6, 10
        frame 1 paths [2,4], [6,8], [10,12]  -> time means 3, 7, 11

    One path per order and one channel, so the two later means are identity: the per-frame
    values are order 0 -> [2, 3], order 1 -> [6, 7], order 2 -> [10, 11]. Their medians
    (2.5, 6.5, 10.5) are exactly what `_prelog_scale` returns for the whole session — which
    is the point: narrowing the frame population is the ONLY change.
    """
    from dehyd.features.extraction import _prelog_scale

    S = np.array([
        [[[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]],
        [[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]],
    ])
    meta = {"order": np.array([0, 1, 2])}

    per_frame = per_frame_prelog(S, meta)
    assert per_frame[0].tolist() == [2.0, 3.0]
    assert per_frame[1].tolist() == [6.0, 7.0]
    assert per_frame[2].tolist() == [10.0, 11.0]

    whole_session = tuple(float(np.median(per_frame[o])) for o in (0, 1, 2))
    assert whole_session == (2.5, 6.5, 10.5)
    assert whole_session == _prelog_scale(S, meta)      # bytewise-equal floats, one recipe


def test_training_frame_epsilons_keep_the_frozen_subject_balanced_hierarchy():
    """HAND-COMPUTED from `:477-500` with the innermost population narrowed to training
    frames. Order-1 per-frame values, with each session's training mask:

        subject 1, session A: [1, 1, 1, 1]  all training   -> session median 1
        subject 1, session B: [3, 5]        all training   -> session median 4
        subject 1, session C: [100, 100]    NO training    -> dropped
        subject 2, session A: [10]          training       -> session median 10
        subject 3, session A: [1000]        NO training    -> subject dropped

        subject means: 1 -> (1 + 4)/2 = 2.5 ; 2 -> 10        (subject 3 absent)
        scale = median([2.5, 10]) = 6.25 ; eps = 0.1 * 6.25 = 0.625

    A POOLED median over the seven training frames [1,1,1,1,3,5,10] would be 1, giving
    eps = 0.1 — so a pooled implementation fails this test, which is why it is written.
    """
    def block(values):
        return {0: np.zeros(len(values)), 1: np.asarray(values, dtype=float),
                2: np.asarray(values, dtype=float)}

    per_session = [block([1, 1, 1, 1]), block([3, 5]), block([100, 100]), block([10]),
                   block([1000])]
    subjects = [1, 1, 1, 2, 3]
    masks = [np.ones(4, bool), np.ones(2, bool), np.zeros(2, bool), np.ones(1, bool),
             np.zeros(1, bool)]

    eps = training_frame_epsilons(per_session, subjects, masks, k=0.1, fallback=1e-6)
    assert eps[1] == pytest.approx(0.625, rel=0, abs=1e-15)
    assert eps[2] == pytest.approx(0.625, rel=0, abs=1e-15)
    assert eps[1] != pytest.approx(0.1, abs=1e-9)      # the rejected pooled-frame reading


def test_training_frame_epsilons_fall_back_when_no_frame_is_training():
    """The frozen 1e-6 fallback for a non-finite / non-positive aggregate, unchanged."""
    eps = training_frame_epsilons([{0: np.zeros(2), 1: np.ones(2), 2: np.ones(2)}], [1],
                                  [np.zeros(2, bool)], k=0.1, fallback=1e-6)
    assert eps == {1: 1e-6, 2: 1e-6}


# --------------------------------------------------------------- the run, end to end (10 GHz)


@pytest.fixture(scope="module")
def ordinal_run(base_config, tmp_path_factory):
    """One real exploratory ordinal run on a synthetic store, at the TUNED branch — the only
    branch with a fitted feature quantity, and therefore the only one whose leakage can be
    tested at all."""
    tmp_path = tmp_path_factory.mktemp("frame_split")
    config = _tmp_config(base_config, tmp_path)
    sessions = _sessions()
    _write_store(config.paths.results_dir, sessions)
    run_dir = _exp_c_source(tmp_path / "loso", "10ghz", feature_key=FEATURE_KEY,
                            params={"alpha": 1.0})
    result = run_frame_split(config, "10ghz", "ordinal", "arm_a", source_run_dir=run_dir,
                             sessions=sessions, store_dir=config.paths.results_dir)
    return config, sessions, run_dir, result


def _run_again(config, sessions, run_dir):
    return run_frame_split(config, "10ghz", "ordinal", "arm_a", source_run_dir=run_dir,
                           sessions=sessions, store_dir=config.paths.results_dir)


def test_ordinal_run_reports_five_leaky_folds_with_ordinal_metrics_and_accuracy(ordinal_run):
    """Plain accuracy is included DELIBERATELY: the paper-comparable number this path exists
    to produce is a frame-level accuracy, which no LOSO report is allowed to carry."""
    _, sessions, _, result = ordinal_run
    assert result.k == 5 and len(result.per_fold) == 5
    assert result.n_frames == len(sessions) * N_FRAMES
    assert sum(fold["n_test_frames"] for fold in result.per_fold) == result.n_frames
    for fold in result.per_fold:
        assert 0.0 <= fold["accuracy"] <= 1.0
        assert fold["class_unit_mae"] >= 0.0
        assert set(("class_unit_mae", "adjacent_accuracy", "quadratic_weighted_kappa")) <= set(fold)
    assert set(result.summary) == set(frame_split.ORDINAL_METRICS)
    assert result.tuned_eps_aggregation == TUNED_EPS_AGGREGATION


def test_kfold_is_seeded_and_the_run_is_deterministic(ordinal_run):
    config, sessions, run_dir, result = ordinal_run
    assert result.kfold_random_state == int(config.run.seed) + KFOLD_SEED_OFFSET
    again = _run_again(config, sessions, run_dir)
    assert again.per_fold == result.per_fold
    assert again.frame_order_sha256 == result.frame_order_sha256
    assert again.fold_assignment_sha256 == result.fold_assignment_sha256


def _fold_rows(n_frames, seed, k=5):
    """The fold assignment, re-derived in the test from the plan's own recipe rather than
    read out of the result — so the mutation test targets a frame the IMPLEMENTATION is
    obliged to hold out, not one it merely says it does."""
    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    return list(splitter.split(np.arange(n_frames)))


def _scale_store_frames(config, sessions, positions, *, scale=7.0):
    """Multiply the named pooled-frame positions' raw scattering coefficients, leaving
    membership (subject / session / frame id) untouched — the T18 eligibility-preserving
    mutation. Returns the set of session rows touched."""
    from dehyd.features.store import read_session_store

    per_session = np.array([len(s["frame_ids"]) for s in sessions])
    starts = np.concatenate([[0], np.cumsum(per_session)])
    by_session: dict = {}
    for position in positions:
        row = int(np.searchsorted(starts, position, side="right") - 1)
        by_session.setdefault(row, []).append(int(position - starts[row]))

    for row, offsets in by_session.items():
        session = sessions[row]
        store = read_session_store("10ghz", session["subject"], session["session_name"],
                                   config.paths.results_dir)
        try:
            arrays = {key: np.array(store[key]) for key in store.keys()}
        finally:
            store.close()
        arrays[raw_key(0, "A", "mag", 0)][offsets] *= scale
        write_session_store("10ghz", session["subject"], session["session_name"], arrays,
                            {"n_frames": len(session["frame_ids"])}, config.paths.results_dir)
    return set(by_session)


@pytest.mark.parametrize("mutate_held_out", [True, False])
def test_tuned_epsilon_and_every_fitted_state_depend_on_training_frames_only(
    base_config, tmp_path, mutate_held_out
):
    """(C10) The parametrization is the point: the SAME machinery must be invariant under a
    held-out mutation and must MOVE under a training mutation. Without the second half the
    test would pass against an implementation that fit nothing at all.

    The held-out frame is chosen from a session that also contributes TRAINING frames, so the
    invariance cannot come from the session being absent — it has to come from ε being a
    function of training rows only.
    """
    config = _tmp_config(base_config, tmp_path)
    sessions = _sessions()
    _write_store(config.paths.results_dir, sessions)
    run_dir = _exp_c_source(tmp_path / "loso", "10ghz", feature_key=FEATURE_KEY,
                            params={"alpha": 1.0})
    kwargs = dict(source_run_dir=run_dir, sessions=sessions,
                  store_dir=config.paths.results_dir)
    before = run_frame_split(config, "10ghz", "ordinal", "arm_a", **kwargs)

    n_frames = len(sessions) * N_FRAMES
    train_idx, test_idx = _fold_rows(n_frames, int(config.run.seed) + KFOLD_SEED_OFFSET)[0]
    session_of = np.repeat(np.arange(len(sessions)), N_FRAMES)
    touched = _scale_store_frames(config, sessions, test_idx if mutate_held_out else train_idx,
                                  scale=7.0)
    if mutate_held_out:
        # the invariance must not come from a whole session being held out
        assert touched & set(session_of[train_idx].tolist())

    after = run_frame_split(config, "10ghz", "ordinal", "arm_a", **kwargs)
    fold0_before, fold0_after = before.per_fold[0], after.per_fold[0]
    if mutate_held_out:
        assert fold0_after["tuned_epsilon"] == fold0_before["tuned_epsilon"]
        assert fold0_after["fitted_state_sha256"] == fold0_before["fitted_state_sha256"]
    else:
        # every hierarchy step (session median -> subject mean -> subject median) is
        # positively homogeneous, so scaling every TRAINING frame by 7 must scale eps by
        # exactly 7 — a far stronger companion than "it differs"
        assert fold0_after["tuned_epsilon"] == pytest.approx(
            [7.0 * e for e in fold0_before["tuned_epsilon"]], rel=1e-9
        )
        assert fold0_after["fitted_state_sha256"] != fold0_before["fitted_state_sha256"]


def test_frame_order_and_fold_assignment_are_hashed_into_the_record(base_config, tmp_path):
    """(C24) `k_folds` + `random_state` describe the RECIPE, not what ran: the payload pins
    neither the frame order the KFold indexes into nor the artifact that defines the
    configuration. A permuted-but-identical cohort must not be provenance-identical."""
    config = _tmp_config(base_config, tmp_path)
    sessions = _sessions()
    _write_store(config.paths.results_dir, sessions)
    run_dir = _exp_c_source(tmp_path / "loso", "10ghz", feature_key=(0, "A", "mag", 0, "off"),
                            params={"alpha": 1.0})
    kwargs = dict(source_run_dir=run_dir, store_dir=config.paths.results_dir)

    straight = run_frame_split(config, "10ghz", "ordinal", "arm_a", sessions=sessions, **kwargs)
    permuted = run_frame_split(config, "10ghz", "ordinal", "arm_a",
                               sessions=list(reversed(sessions)), **kwargs)
    assert straight.frame_order_sha256 != permuted.frame_order_sha256
    assert straight.fold_assignment_sha256 != permuted.fold_assignment_sha256
    assert straight.fold_assignment_sha256 != straight.frame_order_sha256


# ------------------------------------------------------------------ the regression families


def test_full_wst_regression_refits_exp_a_modal_config_and_reports_both_scales(
    base_config, tmp_path
):
    config = _tmp_config(base_config, tmp_path)
    sessions = _sessions()
    _write_store(config.paths.results_dir, sessions)
    source = _exp_a_source(
        tmp_path / "loso_a", "10ghz", feature_key=FEATURE_KEY,
        family="ridge", params={"alpha": 1.0},
    )

    result = run_frame_split(
        config, "10ghz", "regression", WST_REGRESSION_UNIT,
        source_run_dir=source, sessions=sessions, store_dir=config.paths.results_dir,
    )

    assert result.resolved_config["feature_key"] == FEATURE_KEY
    assert result.resolved_config["family"] == "ridge"
    assert result.resolved_config["params"] == {"alpha": 1.0}
    assert result.metric_names == WST_REGRESSION_METRICS
    assert result.tuned_eps_aggregation == TUNED_EPS_AGGREGATION
    assert result.notes["scientific_status"] == "leaky_protocol_demonstration_only"
    assert result.notes["not_directly_comparable_to_loso"] is True
    assert len(result.per_fold) == 5
    for fold in result.per_fold:
        assert fold["n_test_sessions"] > 0
        assert fold["frame_mae"] >= 0.0
        assert fold["session_mae"] >= 0.0
        assert fold["subject_balanced_session_mae"] >= 0.0
        assert len(fold["tuned_epsilon"]) == 2
        assert len(fold["fitted_state_sha256"]) == 64
    assert set(result.summary) == set(WST_REGRESSION_METRICS)


def test_full_wst_regression_fitted_state_ignores_held_out_frames(base_config, tmp_path):
    config = _tmp_config(base_config, tmp_path)
    sessions = _sessions()
    _write_store(config.paths.results_dir, sessions)
    source = _exp_a_source(tmp_path / "loso_a", "10ghz", feature_key=FEATURE_KEY)
    kwargs = dict(
        source_run_dir=source, sessions=sessions, store_dir=config.paths.results_dir,
    )
    before = run_frame_split(
        config, "10ghz", "regression", WST_REGRESSION_UNIT, **kwargs
    )

    _, test_idx = _fold_rows(
        len(sessions) * N_FRAMES, int(config.run.seed) + KFOLD_SEED_OFFSET
    )[0]
    _scale_store_frames(config, sessions, test_idx, scale=11.0)
    after = run_frame_split(
        config, "10ghz", "regression", WST_REGRESSION_UNIT, **kwargs
    )

    assert after.per_fold[0]["tuned_epsilon"] == before.per_fold[0]["tuned_epsilon"]
    assert after.per_fold[0]["fitted_state_sha256"] == before.per_fold[0]["fitted_state_sha256"]


def test_77ghz_full_wst_regression_uses_77ghz_keys(base_config, tmp_path):
    from dehyd.features.store import raw77_key

    config = _tmp_config(base_config, tmp_path)
    sessions = _sessions()
    rng = np.random.default_rng(19)
    for session in sessions:
        n = len(session["frame_ids"])
        write_session_store(
            "77ghz", session["subject"], session["session_name"],
            {
                raw77_key(0): _raw_tensor(
                    rng, n, session["session_idx"], session["subject"]
                ),
                order_key(0): np.array([0, 1, 2]),
            },
            {"n_frames": n}, config.paths.results_dir,
        )
    source = _exp_a_source(
        tmp_path / "loso_a77", "77ghz", feature_key=(0, "frozen")
    )

    result = run_frame_split(
        config, "77ghz", "regression", WST_REGRESSION_UNIT,
        source_run_dir=source, sessions=sessions, store_dir=config.paths.results_dir,
    )

    assert result.resolved_config["feature_key"] == (0, "frozen")
    assert result.tuned_eps_aggregation is None
    assert len(result.per_fold) == 5
    assert all(fold["tuned_epsilon"] is None for fold in result.per_fold)
    assert all(np.isfinite(fold["session_mae"]) for fold in result.per_fold)


def test_physics_and_session_index_frame_splits_run_and_are_labelled(base_config, tmp_path):
    config = _tmp_config(base_config, tmp_path)
    sessions = _sessions()
    _write_store(config.paths.results_dir, sessions)
    source = tmp_path / "loso_d"
    _fake_family_run(config, source, "physics", "10ghz")
    _fake_family_run(config, source, "session_index", "10ghz")

    physics = run_frame_split(config, "10ghz", "regression", "physics",
                              source_run_dir=source, sessions=sessions,
                              store_dir=config.paths.results_dir)
    assert len(physics.per_fold) == 5
    assert all(np.isfinite(fold["frame_mae"]) for fold in physics.per_fold)
    assert physics.notes == {}

    session_index = run_frame_split(config, "10ghz", "regression", "session_index",
                                    source_run_dir=source, sessions=sessions,
                                    store_dir=config.paths.results_dir)
    # near-degenerate by construction: every test frame's session is trained on
    assert session_index.notes == {"degenerate_by_construction": True}


def test_77ghz_ordinal_frame_split_uses_the_band_s_own_keys_and_log_branch(base_config,
                                                                            tmp_path):
    """The 77 GHz arms are 4 of the 16 sanctioned runs and their feature key is `(tiling,
    branch)`, not the 10 GHz 5-tuple — a different store key builder and a different
    order-log function (`apply_order_log_77`, whose branch names are off / on_frozen_eps /
    on_tuned_eps)."""
    from dehyd.features.store import raw77_key

    config = _tmp_config(base_config, tmp_path)
    sessions = _sessions()
    rng = np.random.default_rng(3)
    for s in sessions:
        n = len(s["frame_ids"])
        write_session_store(
            "77ghz", s["subject"], s["session_name"],
            {raw77_key(0): _raw_tensor(rng, n, s["session_idx"], s["subject"]),
             order_key(0): np.array([0, 1, 2])},
            {"n_frames": n}, config.paths.results_dir,
        )
    run_dir = _exp_c_source(tmp_path / "loso77", "77ghz", feature_key=(0, "frozen"),
                            params={"alpha": 1.0})

    result = run_frame_split(config, "77ghz", "ordinal", "arm_a", source_run_dir=run_dir,
                             sessions=sessions, store_dir=config.paths.results_dir)
    assert result.resolved_config["feature_key"] == (0, "frozen")
    assert len(result.per_fold) == 5
    assert result.tuned_eps_aggregation is None          # the frozen branch fits no epsilon
    assert all(fold["tuned_epsilon"] is None for fold in result.per_fold)


def test_cnn_frame_split_refits_the_reduced_configuration_from_the_merged_artifact(
    base_config, tmp_path
):
    """The CNN branch end to end on a synthetic store: the modal (lr, weight_decay) and the
    floored-median epoch budget come out of the merged LOSO selection table, and the refit
    runs with no inner search and no early stopping."""
    config = _tmp_config(base_config, tmp_path)
    sessions = _sessions()
    _write_store(config.paths.results_dir, sessions)
    source = _fake_family_run(config, tmp_path / "loso_cnn", "cnn1d_raw", "10ghz",
                              cnn=(1e-3, 0.0, 2))

    result = run_frame_split(config, "10ghz", "regression", "cnn1d_raw", source_run_dir=source,
                             sessions=sessions, store_dir=config.paths.results_dir)

    assert result.resolved_config["lr"] == 1e-3
    assert result.resolved_config["weight_decay"] == 0.0
    assert result.resolved_config["epoch_budget"] == 2
    assert len(result.per_fold) == 5
    assert all(np.isfinite(fold["frame_mae"]) for fold in result.per_fold)
    assert result.tuned_eps_aggregation is None          # no WST feature quantity here


def _fake_family_run(config, run_dir, family, band, *, cnn=None):
    """A self-consistent Exp D family artifact set, written by the REAL writer so the
    exploratory reader is exercised against the same acceptance rules the comparison stage
    uses."""
    run_dir = Path(run_dir)
    prediction_rows, selection_rows = [], []
    for fold_id, subject in enumerate(range(1, N_SUBJECTS + 1)):
        for session_idx in range(5):
            prediction_rows.append({
                "fold_id": fold_id, "subject": subject, "session_idx": session_idx, "seed": 1,
                "y_true_delta_m_pct": -0.4 * session_idx, "y_pred": -0.3 * session_idx,
                "n_frames_aggregated": N_FRAMES,
            })
        lr, weight_decay, budget = cnn if cnn else ("", "", "")
        selection_rows.append({
            "fold_id": fold_id, "test_subject": subject,
            "selected_config": "cfg2" if cnn else "n/a",
            "learning_rate": lr, "weight_decay": weight_decay, "epoch_budget": budget,
            # the writer's own acceptance rule: the budget must be the median of the counts
            # the row itself lists, so a CNN fixture has to be internally consistent too
            "selected_epoch_counts": [budget] * 3 if cnn else [],
            "per_config_inner_scores": [0.5], "inner_score": 0.5,
            "n_inner_folds": 3, "fitted_coefficients": {},
        })
    exp_d.write_family_artifacts(
        band, family, run_dir, prediction_rows=prediction_rows, selection_rows=selection_rows,
        deterministic=cnn is None, bootstrap_b=20, rng_seed=config.run.seed,
        skip_threshold_pct=20.0,
        lineage={"analysis_commit": "c0ffee", "config_hash": "cfg0", "run_group_id": "g"},
    )
    return run_dir


# ---------------------------------------------------------------------- report + provenance


def test_reports_are_written_only_through_the_allowlist_and_are_tagged(ordinal_run):
    config, _, _, result = ordinal_run
    paths = write_frame_split_reports(result, config)
    for path in paths.values():
        assert EXPLORATORY_TAG in path.name
        assert exploratory_dir(config, "10ghz").resolve() == path.parent.resolve()
    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert payload["leaky_protocol"] is True and payload["never_report"] is True
    assert payload["source_run"]["artifact_sha256"] == result.source_run["artifact_sha256"]
    header = paths["per_fold"].read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("fold,n_train_frames,n_test_frames,")


def test_exploratory_provenance_refuses_an_empty_manifest(ordinal_run):
    """(C21) `_hash_inputs` reads the manifest to know which files to hash; an empty one
    would produce a record that attests nothing about the data it claims to describe."""
    config, _, _, result = ordinal_run
    with pytest.raises(FrameSplitError, match="empty QC manifest"):
        write_exploratory_provenance(config, "10ghz", "ordinal", "arm_a",
                                     pd.DataFrame(columns=["subject", "session_idx", "rel_path"]),
                                     exploratory_dir(config, "10ghz"), data_dir=None,
                                     result=result)


def _minimal_manifest_qc(data_dir, sessions):
    """A minimal but REAL manifest: one dummy raw file per session, hashed for real by
    `_hash_inputs`. Bypasses QC/ground truth, but exercises the REAL payload construction."""
    rows = []
    for s in sessions:
        (Path(data_dir) / s["rel_path"]).write_bytes(f"radar bytes {s['rel_path']}".encode())
        rows.append({"subject": s["subject"], "session_idx": s["session_idx"],
                     "rel_path": s["rel_path"]})
    return pd.DataFrame(rows)


def test_exploratory_provenance_hashes_against_the_band_correct_data_root(base_config, tmp_path,
                                                                          ordinal_run):
    """(C22) `_hash_inputs` silently DEFAULTS to the 10 GHz root, so a 77 GHz run that let it
    default would hash 10 GHz bytes under a 77 GHz label. Two distinct roots holding the same
    logical files must therefore give different hashes."""
    _, sessions, _, result = ordinal_run
    data_77 = tmp_path / "data77"
    data_77.mkdir()
    config = _tmp_config(base_config, tmp_path,
                         weight_xlsx=tmp_path / "weights.xlsx", data_77ghz_dir=data_77)
    (tmp_path / "weights.xlsx").write_bytes(b"pretend workbook")
    manifest = _minimal_manifest_qc(config.paths.data_10ghz_dir, sessions)
    for s in sessions:                                  # same names, different bytes
        (data_77 / s["rel_path"]).write_bytes(f"SEVENTY-SEVEN {s['rel_path']}".encode())

    out_10 = write_exploratory_provenance(config, "10ghz", "ordinal", "arm_a", manifest,
                                          exploratory_dir(config, "10ghz"), data_dir=None,
                                          result=result)
    result_77 = dataclasses.replace(result, band="77ghz")
    out_77 = write_exploratory_provenance(config, "77ghz", "ordinal", "arm_a", manifest,
                                          exploratory_dir(config, "77ghz"), data_dir=data_77,
                                          result=result_77)

    payload_10 = json.loads(out_10.read_text(encoding="utf-8"))
    payload_77 = json.loads(out_77.read_text(encoding="utf-8"))
    hashes_10 = {e["rel_path"]: e["sha256"] for e in payload_10["inputs"]["radar_files"]}
    hashes_77 = {e["rel_path"]: e["sha256"] for e in payload_77["inputs"]["radar_files"]}
    assert set(hashes_10) == set(hashes_77) and hashes_10 != hashes_77

    # the tags sit at the TOP level as well as in `extra`, so a reader who opens the file and
    # looks no further still sees that it is unreportable
    assert payload_10["leaky_protocol"] is True and payload_10["never_report"] is True
    extra = payload_10["extra"]
    assert extra["leaky_protocol"] is True and extra["never_report"] is True
    assert extra["frame_order_sha256"] == result.frame_order_sha256
    assert extra["fold_assignment_sha256"] == result.fold_assignment_sha256
    assert extra["source_run"]["artifact_sha256"] == result.source_run["artifact_sha256"]
    assert extra["resolved_config"]["params"] == {"alpha": 1.0}
    assert payload_10["folds"] == []                  # there are no LOSO folds in this path


# --------------------------------------------------------------- the CLI, end to end (C6/D11)


def _cli_setup(base_config, tmp_path, sessions):
    """A real config file whose paths point at tmp, so the CLI's own `load_config` is used."""
    results = tmp_path / "results"
    data_10 = tmp_path / "data10"
    for directory in (results, data_10):
        directory.mkdir(exist_ok=True)
    (tmp_path / "weights.xlsx").write_bytes(b"pretend workbook")
    overlay = tmp_path / "paths_overlay.yaml"
    overlay.write_text(
        "paths:\n"
        f"  data_10ghz_dir: {data_10.as_posix()}\n"
        f"  weight_xlsx: {(tmp_path / 'weights.xlsx').as_posix()}\n"
        f"  results_dir: {results.as_posix()}\n",
        encoding="utf-8",
    )
    _write_store(results, sessions)
    return overlay, results, data_10


def test_cli_run_creates_nothing_under_results_runs(base_config, tmp_path, monkeypatch):
    """(C6, D11) The isolation claim, checked the only way that cannot be fooled: snapshot the
    whole `results/` tree before and after a COMPLETE invocation and diff it. `record_run`
    would have created `results/runs/<stamp>_<rev>/provenance.json` no matter how the metrics
    files were named, which is exactly why this path must not call it."""
    import experiments.run_frame_split_exploratory as cli

    sessions = _sessions()
    overlay, results, data_10 = _cli_setup(base_config, tmp_path, sessions)
    manifest = _minimal_manifest_qc(data_10, sessions)
    source = _exp_c_source(tmp_path / "loso", "10ghz", feature_key=FEATURE_KEY,
                           params={"alpha": 1.0})

    monkeypatch.setattr(cli, "_build_sessions", lambda *a, **k: sessions)
    monkeypatch.setattr(cli, "_build_manifest_qc", lambda *a, **k: manifest)
    monkeypatch.setattr(cli.exp_b, "config_fingerprint", lambda config: "cfg0")
    monkeypatch.setattr(cli, "_git_info", lambda: {"commit": "c0ffee", "dirty": False,
                                                   "branch": "b"})

    (results / "runs").mkdir()
    before = sorted(p.relative_to(results).as_posix() for p in (results / "runs").rglob("*"))

    assert cli.main([
        "--config", "configs/exp_a_regression.yaml", "--config", "configs/exp_c.yaml",
        "--config", str(overlay), "--band", "10ghz", "--task", "ordinal", "--unit", "arm_a",
        "--source-run-dir", str(source),
    ]) == 0

    after = sorted(p.relative_to(results).as_posix() for p in (results / "runs").rglob("*"))
    assert after == before == []

    written = sorted(p.name for p in (results / "exploratory_frame_split" / "10ghz").iterdir())
    assert written == [
        f"ordinal_arm_a_{EXPLORATORY_TAG}.csv",
        f"ordinal_arm_a_{EXPLORATORY_TAG}.json",
        f"ordinal_arm_a_provenance_{EXPLORATORY_TAG}.json",
    ]
    # nothing this path writes may be mistaken for a reportable artifact
    assert not list(results.rglob("metrics_exp_*"))
    provenance = json.loads(
        (results / "exploratory_frame_split" / "10ghz"
         / f"ordinal_arm_a_provenance_{EXPLORATORY_TAG}.json").read_text(encoding="utf-8")
    )
    assert provenance["extra"]["never_report"] is True
    assert provenance["inputs"]["radar_files"]           # the manifest really was hashed


def test_cli_rejects_unknown_aliases_and_units_outside_their_task(base_config, tmp_path):
    """The authorized name is radar_wst; ambiguous aliases and cross-task units fail."""
    import experiments.run_frame_split_exploratory as cli

    base = ["--config", "configs/exp_a_regression.yaml", "--band", "10ghz",
            "--source-run-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        cli.main(base + ["--task", "regression", "--unit", "exp_a"])
    with pytest.raises(SystemExit):
        cli.main(base + ["--task", "ordinal", "--unit", "physics"])
    with pytest.raises(SystemExit):
        cli.main(base + ["--task", "regression", "--unit", "arm_a"])


def test_cli_accepts_exactly_the_eighteen_matrix_pairs():
    """The CLI's accepted (task, unit) set IS the module's matrix — one table, not two."""
    import experiments.run_frame_split_exploratory as cli

    pairs = {(task, unit) for task in cli.TASKS for unit in frame_split.TASK_UNITS[task]}
    assert pairs == {(task, unit) for task, unit, _ in FRAME_SPLIT_MATRIX}
    assert len(pairs) * len(frame_split.BANDS) == 18


def test_full_wst_frame_split_launcher_is_cpu_only_and_quarantined():
    source = (
        Path("scripts/ibex/run_wst_regression_frame_split.sbatch")
        .read_text(encoding="utf-8")
    )
    assert "#SBATCH --cpus-per-task=1" in source
    assert "#SBATCH --mem=64G" in source
    assert "--task regression" in source
    assert "--unit radar_wst" in source
    assert "SOURCE_RUN_DIR" in source
    assert "configs/ibex_sosagojm.yaml" in source
    assert "run_regression.py" not in source
    assert "record_run" not in source
    assert "--gpus" not in source and "--gres" not in source


def test_sosagojm_ibex_overlay_changes_paths_only_and_launchers_accept_it():
    overlay = yaml.safe_load(Path("configs/ibex_sosagojm.yaml").read_text(encoding="utf-8"))
    assert set(overlay) == {"paths"}
    assert overlay["paths"] == {
        "data_10ghz_dir": "/ibex/user/sosagojm/dehydration_loso_diagnostic/data/10ghz",
        "data_77ghz_dir": "/ibex/user/sosagojm/dehydration_loso_diagnostic/data/77ghz",
        "weight_xlsx": (
            "/ibex/user/sosagojm/dehydration_loso_diagnostic/data/weight/"
            "metadata_subjects_info.xlsx"
        ),
        "results_dir": "/ibex/user/sosagojm/dehydration_radar_2/results",
    }
    for script_name in ("extract10.sbatch", "extract77.sbatch", "run_exp_a.sbatch"):
        source = Path("scripts/ibex", script_name).read_text(encoding="utf-8")
        assert "IBEX_CONFIG=${IBEX_CONFIG:-configs/ibex.yaml}" in source
        assert '--config "$IBEX_CONFIG"' in source
