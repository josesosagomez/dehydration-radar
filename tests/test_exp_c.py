"""T-M9 Exp C driver (`eval/exp_c.py`, milestone 9 step 6): the ordinal two-arm composition.

Same split as `test_exp_b.py`: the RUN half runs the real staged search end to end on a
synthetic store (no private data), the REPORT half runs on hand-built `ExpCFoldResult`s.
Groups: T-M9-expc-provider, T-M9-expc-leak, T-M9-expc-mutation, T-M9-expc-viability,
T-M9-expc-report, plus exp_c's half of T-M9-parallel.

Every expected value below is derived from the SPECIFICATION's arithmetic (plan §2.3/§2.6,
`implementation_plan.md` §C), never from running the implementation — the aggregation and
QWK fixtures exist specifically to catch a formula that runs fine but is wrong.
"""

import dataclasses
import json
import math

import numpy as np
import pytest

from dehyd.config import ExpCConfig, load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_a, exp_b, exp_c, fold_parallel, harness
from dehyd.eval.exp_c import (
    ARMS,
    ExpCArmResult,
    ExpCError,
    ExpCFoldResult,
    ExpCProtocolError,
    OrdinalFeatures,
    assert_exp_c_fit_authorized,
    build_sessions_c,
    evaluable_subjects_c,
    ordinal_class_mae_score,
    run_and_report_c,
    run_exp_c,
    summarize_exp_c,
    write_exp_c_reports,
)
from dehyd.eval.harness import Candidate, InnerResult, SeedOutcome, StageOutcome
from dehyd.eval.selection import CandidateScore
from dehyd.features.pooling import aggregate_session, pool_stats_batch
from dehyd.features.store import (
    order_key,
    prelog_key,
    raw_key,
    read_session_store,
    vec_key,
    write_session_store,
)
from dehyd.features.wst import apply_order_log
from dehyd.models import regressors

P, T, CN, NFR = 6, 4, 1, 3  # tiny path/time/channel/frame dims for a fast synthetic store
ORDER = np.array([0, 1, 1, 2, 2, 2])  # length P, all orders present

ALL_CLASSES = (0, 1, 2, 3, 4)


@pytest.fixture(scope="module")
def config():
    return load_config("configs/exp_a_regression.yaml", "configs/exp_c.yaml")


# ------------------------------------------------------------------------------- fixtures


def _session_record(subject, session_idx):
    """One Exp C spine record, in `build_sessions_c`'s own shape (Δm%(S0) is identically 0)."""
    delta_m_pct = 0.0 if session_idx == 0 else -(0.3 * session_idx + 0.05 * subject)
    return {
        "subject": subject,
        "session_idx": session_idx,
        "session_name": SESSION_NAMES[session_idx],
        "delta_m_pct": delta_m_pct,
        "loss_l": -delta_m_pct,
        "class_idx": session_idx,
    }


def _make_sessions_c(n_subjects=6, classes_by_subject=None):
    out = []
    for s in range(1, n_subjects + 1):
        idxs = ALL_CLASSES if classes_by_subject is None else classes_by_subject[s]
        for i in idxs:
            out.append(_session_record(s, i))
    return out


def _make_sessions_class3_only_in_subject_6(n_subjects=6):
    """Class 3 lives ONLY in subject 6, so on the test-subject-1 outer fold exactly ONE of
    the five inner folds (the one holding subject 6 out) has an inner-training set missing a
    class — and on the test-subject-6 fold EVERY inner-training set does."""
    return _make_sessions_c(
        n_subjects,
        {s: (ALL_CLASSES if s == 6 else (0, 1, 2, 4)) for s in range(1, n_subjects + 1)},
    )


def _write_store(store_dir, sessions, config, seed=0):
    rng = np.random.default_rng(seed)
    wst = config.wst
    eps = {1: wst.log_epsilon, 2: wst.log_epsilon}
    meta = {"order": ORDER}
    for s in sessions:
        npz = {}
        for ti in range(len(wst.tilings)):
            npz[order_key(ti)] = ORDER
            for gi in range(len(config.search_10ghz.range_gate_m)):
                for r in config.search_10ghz.reduction:
                    for c in config.search_10ghz.channel:
                        raw = np.abs(rng.normal(size=(NFR, CN, P, T))) + 0.01
                        off = aggregate_session(
                            pool_stats_batch(apply_order_log(raw, meta, wst, log_on=False), meta))
                        fr = aggregate_session(pool_stats_batch(
                            apply_order_log(raw, meta, wst, log_on=True, epsilon_by_order=eps), meta))
                        npz[vec_key(gi, r, c, ti, "off")] = off
                        npz[vec_key(gi, r, c, ti, "frozen")] = fr
                        npz[raw_key(gi, r, c, ti)] = raw
                        npz[prelog_key(gi, r, c, ti)] = np.array([raw.mean(), raw.mean(), raw.mean()])
        write_session_store("10ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR}, store_dir)


def _mutate_features_on_disk(store_dir, sessions, subject, seed=99):
    """Eligibility-preserving: overwrite one subject's stored arrays (not labels)."""
    rng = np.random.default_rng(seed)
    for s in sessions:
        if s["subject"] != subject:
            continue
        store = read_session_store("10ghz", s["subject"], s["session_name"], store_dir)
        npz = {k: store[k].copy() for k in store.keys()}
        store.close()
        for k in list(npz):
            if k.startswith("raw__"):
                npz[k] = np.abs(rng.normal(size=npz[k].shape) * 5) + 0.01
            elif k.startswith(("vec__", "prelog__")):
                npz[k] = (rng.normal(size=npz[k].shape) * 5 + 5).astype(npz[k].dtype)
        write_session_store("10ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR}, store_dir)


def _mutate_targets_and_classes(sessions, subject, seed=99):
    """Move a subject's L AND its ordinal class (the class column is the 2-column y's own
    held-out label — mutating only L would leave half the leak surface untested)."""
    rng = np.random.default_rng(seed)
    for s in sessions:
        if s["subject"] != subject:
            continue
        s["loss_l"] = float(rng.normal() * 5 + 5)
        s["class_idx"] = int(4 - s["session_idx"])   # a permutation of the frozen grid


def _fold_for(sessions, test_subject, n_subjects=None):
    subjects = evaluable_subjects_c(sessions) if n_subjects is None else list(range(1, n_subjects + 1))
    return next(f for f in harness.nested_loso_splits(subjects) if f.test_subject == test_subject)


def _classes_by_subject(sessions):
    out = {}
    for s in sessions:
        out.setdefault(int(s["subject"]), []).append(int(s["class_idx"]))
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------- T-M9-expc-provider: spine + features


def test_build_sessions_c_keeps_s0_and_adds_loss_and_class(monkeypatch, config):
    fake = [
        {"subject": 1, "session_idx": i, "session_name": SESSION_NAMES[i],
         "delta_m_pct": 0.0 if i == 0 else -0.5 * i}
        for i in range(5)
    ]
    monkeypatch.setattr(exp_a, "build_sessions", lambda cfg, band: fake)
    out = build_sessions_c(config, "10ghz")

    assert [s["session_idx"] for s in out] == [0, 1, 2, 3, 4]      # S0 STAYS (unlike Exp B)
    assert [s["class_idx"] for s in out] == [0, 1, 2, 3, 4]        # the class IS the session
    # the frozen sign convention L = -Δm%: loss grows with the session, never shrinks.
    assert [s["loss_l"] for s in out] == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_evaluable_subjects_c_is_exp_as_rule():
    sessions = _make_sessions_c(n_subjects=3, classes_by_subject={1: (0,), 2: (0, 3), 3: ALL_CLASSES})
    assert evaluable_subjects_c(sessions) == [1, 2, 3]   # >= 1 eligible session, Exp A's rule


@pytest.mark.parametrize("branch_index", [0, 1, 2])
def test_ordinal_features_x_is_bytewise_exp_as_on_every_branch(tmp_path, config, branch_index):
    """The X path must be Exp A's, byte for byte, on off/frozen/tuned — a re-implemented
    (rather than wrapped) provider would drift on the tuned branch first."""
    sessions = _make_sessions_c(n_subjects=4)
    _write_store(tmp_path, sessions, config)
    provider = OrdinalFeatures("10ghz", sessions, tmp_path, config)
    base = exp_a.StoreBackedFeatures("10ghz", sessions, tmp_path, config)

    anchor = config.search_10ghz.stage1_anchor_ridge_alpha
    branch = ("off", "frozen", "tuned")[branch_index]
    candidate = next(c for c in exp_a.stage1_candidates(config, "10ghz", anchor)
                     if c.feature_key[-1] == branch)
    train_subjects = {1, 2, 3}

    bundle = provider.data_for(candidate, train_subjects)
    base_bundle = base.data_for(candidate, train_subjects)

    assert bundle.X.tobytes() == base_bundle.X[:].tobytes()
    assert bundle.subjects.tolist() == base_bundle.subjects.tolist()
    # the tuned-ε FitRecord travels unchanged, so it is audited exactly as in Exp A
    assert [q for q, _ in bundle.extra_fits] == [q for q, _ in base_bundle.extra_fits]
    for (_, lhs), (_, rhs) in zip(bundle.extra_fits, base_bundle.extra_fits, strict=True):
        for k in lhs:
            assert lhs[k].tobytes() == rhs[k].tobytes()


def test_ordinal_features_y_is_the_two_column_loss_class_matrix(tmp_path, config):
    sessions = _make_sessions_c(n_subjects=4)
    _write_store(tmp_path, sessions, config)
    provider = OrdinalFeatures("10ghz", sessions, tmp_path, config)
    base = exp_a.StoreBackedFeatures("10ghz", sessions, tmp_path, config)
    anchor = config.search_10ghz.stage1_anchor_ridge_alpha
    candidate = exp_a.stage1_candidates(config, "10ghz", anchor)[0]

    bundle = provider.data_for(candidate, {1, 2, 3})

    assert bundle.y.shape == (len(sessions), 2)
    # column 0 is L = -Δm% (the sign flip is the frozen convention, not a copy of Exp A's y)
    assert bundle.y[:, 0].tolist() == (-base.y).tolist()
    assert bundle.y[:, 1].tolist() == [float(s["class_idx"]) for s in sessions]
    assert bundle.session_idx.tolist() == [s["session_idx"] for s in sessions]


def test_ordinal_class_mae_score_reads_the_class_column_not_the_loss_column():
    """Hand-computed: |0-0| + |1-2| + |2-2| = 1 over 3 rows -> 1/3. Scoring the L column
    (y[:, 0]) instead would give a completely different number on this fixture."""
    y2 = np.array([[9.0, 0.0], [8.0, 1.0], [7.0, 2.0]])
    y_pred = np.array([0.0, 2.0, 2.0])
    got = ordinal_class_mae_score(np.array([1, 1, 2]), y2, y_pred, np.array([0, 1, 2]))
    assert got == pytest.approx(1.0 / 3.0)


# ------------------------------------------------ T-M9-expc-provider: the fit-authorization guard


def _stage1_candidate(config, band="10ghz"):
    anchor = (config.search_10ghz if band == "10ghz" else config.search_77ghz).stage1_anchor_ridge_alpha
    return exp_c.stage1_candidates_c(config, band, anchor)[0]


def test_arm_active_records_carry_the_base_family_and_arm_b_carries_none(config):
    """(trap 2) Arm (a)'s protocol record names the BASE family — the fitted regressor
    genuinely is that family, and the frozen whitelist has no `ord_a_*` value. Arm (b) has no
    legal `model_family` at all, so its record omits the key rather than inventing one."""
    winner = _stage1_candidate(config)
    a_cands = exp_c.stage2_candidates_a(config, "10ghz", winner.feature_key, dict(winner.active))
    b_cands = exp_c.stage2_candidates_b(config, "10ghz", winner.feature_key, dict(winner.active))

    assert dict(winner.active)["model_family"] == "ridge"     # stage 1 anchor, base family
    assert winner.family == "ord_a_ridge"

    families = {c.family: dict(c.active).get("model_family") for c in a_cands}
    assert families == {f"ord_a_{f}": f for f in regressors.MODEL_FAMILIES}
    assert len(a_cands) == 41            # 8 + 12 + 6 + 8 + 7, the frozen per-family grids
    assert len({c.candidate_id for c in a_cands}) == 41

    assert len(b_cands) == 3             # the frozen proportional_odds_c_grid
    for c in b_cands:
        assert c.family == "ord_b_frank_hall"
        assert "model_family" not in dict(c.active)
        assert set(dict(c.active)) == exp_c.REQUIRED_ACTIVE_KEYS_C["10ghz"]


def test_fit_guard_binds_active_model_family_to_the_actual_wrapper(config, monkeypatch):
    """A legal active value is not enough: it must name the regressor actually being fit."""
    candidate = _authorized_arm_a_candidate(config)
    bad_active = tuple(
        (key, "ridge" if key == "model_family" else value)
        for key, value in candidate.active
    )
    mismatched = dataclasses.replace(candidate, active=bad_active)

    built = []

    def forbidden_build(*args, **kwargs):
        built.append((args, kwargs))
        raise AssertionError("build_estimator must not be reached")

    monkeypatch.setattr(harness, "build_estimator", forbidden_build)
    with pytest.raises(ExpCProtocolError, match=r"active\.model_family"):
        harness._fit_once(
            mismatched,
            np.zeros((1, 1)),
            np.zeros((1, 2)),
            np.array([True]),
            0,
            exp_c._before_fit_c(config, "a"),
        )
    assert built == []


def test_stage1_reuses_exp_as_enumeration_with_the_family_swapped(config):
    """A-M9-1 = ONE enumeration of the frozen space: Stage 1 must be Exp A's candidate list
    with the family swapped, never a second hand-written enumeration."""
    anchor = config.search_10ghz.stage1_anchor_ridge_alpha
    exp_a_cands = exp_a.stage1_candidates(config, "10ghz", anchor)
    exp_c_cands = exp_c.stage1_candidates_c(config, "10ghz", anchor)

    assert len(exp_c_cands) == len(exp_a_cands) == 72
    for lhs, rhs in zip(exp_a_cands, exp_c_cands, strict=True):
        assert rhs.family == "ord_a_ridge"
        assert rhs.feature_key == lhs.feature_key
        assert rhs.active == lhs.active                   # the base family stays "ridge"
        assert rhs.model_params == (("alpha", anchor),)


def test_assert_exp_c_fit_authorized_accepts_the_real_candidates(config):
    winner = _stage1_candidate(config)
    assert_exp_c_fit_authorized(winner, config, arm="stage1")
    for c in exp_c.stage2_candidates_a(config, "10ghz", winner.feature_key, dict(winner.active)):
        assert_exp_c_fit_authorized(c, config, arm="a")
    for c in exp_c.stage2_candidates_b(config, "10ghz", winner.feature_key, dict(winner.active)):
        assert_exp_c_fit_authorized(c, config, arm="b")


def _authorized_arm_a_candidate(config):
    winner = _stage1_candidate(config)
    return next(c for c in exp_c.stage2_candidates_a(config, "10ghz", winner.feature_key,
                                                      dict(winner.active))
                if c.family == "ord_a_svr")


def test_unauthorized_family_id_is_refused_and_no_estimator_is_built(config, monkeypatch):
    """The whole point of this guard: `protocol_freeze_guard` validates only the keys present
    in `active`, so a candidate carrying a perfectly legal `active` record with an ILLEGAL
    family would reach `.fit()` with every other guard passing. `_fit_once` calls
    `before_fit(candidate)` and only then `build_estimator`, so a spy on the latter proves no
    fit was reached."""
    built = []
    monkeypatch.setattr(harness, "build_estimator", lambda *a, **k: built.append(a))

    winner = _stage1_candidate(config)
    rogue = dataclasses.replace(winner, family="ridge")   # a real Exp A family, not an Exp C one
    with pytest.raises(ExpCProtocolError, match="family"):
        exp_c._before_fit_c(config, "stage1")(rogue)
    assert built == []


def test_off_grid_arm_a_hyperparameters_are_refused(config):
    candidate = _authorized_arm_a_candidate(config)
    off_grid = dataclasses.replace(candidate, model_params=(("C", 7.0), ("epsilon", 0.1)))
    with pytest.raises(ExpCProtocolError, match="base_params"):
        assert_exp_c_fit_authorized(off_grid, config, arm="a")


def test_stage1_refuses_a_non_anchor_alpha(config):
    """Stage 1 is frozen at the ridge anchor α=1.0; any other member of the ridge grid is a
    legal Stage-2 value but an unauthorized Stage-1 one."""
    winner = _stage1_candidate(config)
    off_anchor = dataclasses.replace(winner, model_params=(("alpha", 10.0),))
    with pytest.raises(ExpCProtocolError, match="stage1_anchor_ridge_alpha"):
        assert_exp_c_fit_authorized(off_anchor, config, arm="stage1")


def test_base_family_disagreeing_with_the_candidate_id_is_refused(config, monkeypatch):
    """The guard binds the estimator that will actually be built, not the id: a factory that
    returned an SVR wrapper for an `ord_a_ridge` id must be caught. Re-deriving `base_family`
    from the candidate id instead of reading the built wrapper passes this vacuously."""
    winner = _stage1_candidate(config)
    real = regressors._ordinal_model

    def wrong_factory(family, params, *, seed):
        model = real(family, params, seed=seed)
        model.base_family = "svr"
        return model

    monkeypatch.setattr(regressors, "_ordinal_model", wrong_factory)
    with pytest.raises(ExpCProtocolError, match="base_family"):
        assert_exp_c_fit_authorized(winner, config, arm="stage1")


def test_mismatched_cutpoint_quantiles_are_refused(config, monkeypatch):
    """The wrapper constants are protocol, never hyperparameters: if the factory's frozen
    ExpCConfig ever drifted from the run's, the fit must be refused, not silently run."""
    monkeypatch.setattr(
        regressors, "EXP_C", dataclasses.replace(ExpCConfig(), cutpoint_quantiles=(0.1, 0.4, 0.6, 0.8))
    )
    with pytest.raises(ExpCProtocolError, match="cutpoint_quantiles"):
        assert_exp_c_fit_authorized(_stage1_candidate(config), config, arm="stage1")


def test_off_grid_frank_hall_c_is_refused(config):
    winner = _stage1_candidate(config)
    candidate = exp_c.stage2_candidates_b(config, "10ghz", winner.feature_key, dict(winner.active))[0]
    off_grid = dataclasses.replace(candidate, model_params=(("C", 5.0),))
    with pytest.raises(ExpCProtocolError, match="proportional_odds_c_grid"):
        assert_exp_c_fit_authorized(off_grid, config, arm="b")


def test_a_candidate_from_the_other_arm_is_refused(config):
    winner = _stage1_candidate(config)
    b_candidate = exp_c.stage2_candidates_b(config, "10ghz", winner.feature_key, dict(winner.active))[0]
    with pytest.raises(ExpCProtocolError, match="arm"):
        assert_exp_c_fit_authorized(b_candidate, config, arm="a")


def test_frank_hall_max_iter_bound_matches_the_estimator_default(config):
    """The guard compares two INDEPENDENTLY stated values (exp_c's recorded bound vs the
    estimator's own default), so this pins that they still agree."""
    from dehyd.models.ordinal import FrankHallOrdinal

    assert FrankHallOrdinal(1.0).max_iter == exp_c.FRANK_HALL_MAX_ITER == 1000


# ------------------------------------- T-M9-expc-viability: the §2.3 inner-fold aggregation
#
# Hand-built StageOutcomes, so the aggregation arithmetic is checked against the
# specification rather than against the search's own output.


def _cell(candidate_id, inner_train, inner_val, score, val_predictions, *, reason=None):
    return InnerResult(
        frozenset(inner_train), frozenset(inner_val), candidate_id, score,
        val_predictions, [], reason=reason,
    )


def _hand_stage_outcome():
    """ONE candidate, FIVE inner folds, one of them non-evaluable.

    inner_scores        = [0.5, nan, 1.5, 2.5, 0.5]
    evaluable folds     = 4
    nanmean             = (0.5 + 1.5 + 2.5 + 0.5) / 4                     = 1.25
    nanstd(ddof=0)      = sqrt(((0.75² + 0.25² + 1.25² + 0.75²)) / 4)     = sqrt(0.6875)

    Each subject in the spine carries classes [0, 1], so each evaluable cell's QWK is
    hand-computable on the fixed 5-class grid (weights (i-j)²/16):
      fold 0  truth [0,1] pred [0,1] -> observed 0,      expected 1/16   -> κ =  1.0
      fold 2  truth [0,1] pred [1,0] -> observed 2/16,   expected 1/16   -> κ = -1.0
      fold 3  truth [0,1] pred [0,1] ->                                     κ =  1.0
      fold 4  truth [0,1] pred [0,0] -> observed 1/16,   expected 1/16   -> κ =  0.0
    mean over the folds where QWK is defined = (1 - 1 + 1 + 0) / 4 = 0.25
    """
    sessions = [_session_record(s, i) for s in range(1, 6) for i in (0, 1)]
    candidates = [Candidate("cand", "ord_a_ridge", (("alpha", 1.0),), feature_key=(0, "A", "mag", 0, "off"))]
    inner_scores = np.array([[0.5, np.nan, 1.5, 2.5, 0.5]])
    preds = {
        0: {1: np.array([0.0, 1.0])},
        2: {2: np.array([1.0, 0.0])},
        3: {3: np.array([0.0, 1.0])},
        4: {4: np.array([0.0, 0.0])},
    }
    cells = []
    for fj in range(5):
        if fj == 1:
            cells.append(_cell("cand", {2, 3}, {1}, float("nan"), {},
                               reason="ordinal_missing_class_3_in_inner_train"))
        else:
            cells.append(_cell("cand", {2, 3}, set(preds[fj]), float(inner_scores[0, fj]), preds[fj]))
    # The harness's own CandidateScore carries the plain-mean MAE (NaN here, which is exactly
    # what §2.3 rejects) and the measured feature dimension; only the latter may be reused.
    harness_scores = [CandidateScore("cand", float("nan"), 0, 12, float("nan"))]
    return sessions, StageOutcome(candidates, inner_scores, cells, harness_scores)


def test_ordinal_candidate_scores_aggregate_over_evaluable_folds_only():
    """The plain-mean aggregation the harness uses for Exp A/B returns NaN here and would
    make the whole outer fold non-selectable — stricter than the frozen rule (§2.3)."""
    sessions, stage = _hand_stage_outcome()
    scores, _ = exp_c._ordinal_candidate_scores(stage, sessions)

    assert len(scores) == 1
    score = scores[0]
    assert score.inner_val_class_mae == pytest.approx(1.25)
    assert score.inner_fold_variance == pytest.approx(math.sqrt(0.6875))
    assert score.n_evaluable_inner_folds == 4
    assert score.inner_val_qwk == pytest.approx(0.25)
    assert score.simplicity_rank == 0            # ord_a_ridge mirrors ridge
    assert score.feature_dimension == 12         # carried over from the harness's measurement


def test_ordinal_candidate_scores_read_the_stored_first_seed_predictions():
    """(trap 6) The QWK tie-break must come from the stored `InnerResult.val_predictions` the
    MAE came from — a re-predict would double-fit and can drift for rf/gbm. Rewriting the
    stored predictions must therefore move the QWK; an implementation that re-predicts (or
    that reads the class column of y) is unaffected by this mutation."""
    sessions, stage = _hand_stage_outcome()
    before = exp_c._ordinal_candidate_scores(stage, sessions)[0][0].inner_val_qwk

    # fold 0 goes from a perfect (κ = 1) to an inverted (κ = -1) validation prediction:
    # the mean becomes (-1 - 1 + 1 + 0) / 4 = -0.25.
    stage.inner_results[0].val_predictions[1] = np.array([1.0, 0.0])
    after = exp_c._ordinal_candidate_scores(stage, sessions)[0][0].inner_val_qwk

    assert before == pytest.approx(0.25)
    assert after == pytest.approx(-0.25)


def test_qwk_undefined_cells_are_skipped_and_counted_not_propagated():
    """O-M9-8's operative trigger: QWK is undefined only when the expected disagreement is
    exactly 0 (both marginals on the SAME single class). Such a fold falls back to MAE-only
    ranking for that candidate — the frozen `:798-800` behaviour — and is counted."""
    sessions = [_session_record(s, i) for s in range(1, 6) for i in (0, 1)]
    sessions.append(_session_record(6, 0))            # subject 6: a single S0 session
    candidates = [Candidate("cand", "ord_a_ridge", (("alpha", 1.0),), feature_key=(0, "A", "mag", 0, "off"))]
    inner_scores = np.array([[1.0, 2.0]])
    cells = [
        _cell("cand", {2, 3}, {1}, 1.0, {1: np.array([0.0, 1.0])}),    # κ = 1.0
        _cell("cand", {2, 3}, {6}, 2.0, {6: np.array([0.0])}),         # 1x1 grid mass -> NaN
    ]
    stage = StageOutcome(candidates, inner_scores, cells,
                         [CandidateScore("cand", float("nan"), 0, 12, float("nan"))])

    scores, exposure = exp_c._ordinal_candidate_scores(stage, sessions)
    assert scores[0].inner_val_class_mae == pytest.approx(1.5)   # BOTH folds count for the MAE
    assert scores[0].n_evaluable_inner_folds == 2
    assert scores[0].inner_val_qwk == pytest.approx(1.0)         # only the defined fold
    assert exposure.nan_inner_folds == frozenset({(6,)})
    assert exposure.n_nan_evaluation_cells == 1
    assert exposure.n_evaluation_cells == 2


def test_qwk_exposure_deduplicates_folds_but_retains_candidate_cell_counts():
    """Two candidates can produce two undefined values on one validation fold; the report's
    `n_qwk_nan` unit is still the fold, with candidate-cell exposure recorded separately."""
    sessions = [_session_record(6, 0)]
    candidates = [
        Candidate(f"cand{i}", "ord_a_ridge", (("alpha", 1.0),),
                  feature_key=(0, "A", "mag", 0, "off"))
        for i in (1, 2)
    ]
    cells = [
        _cell(candidate.candidate_id, {2, 3}, {6}, 0.0, {6: np.array([0.0])})
        for candidate in candidates
    ]
    stage = StageOutcome(
        candidates,
        np.zeros((2, 1)),
        cells,
        [CandidateScore(c.candidate_id, 0.0, 0, 12, 0.0) for c in candidates],
    )

    _, exposure = exp_c._ordinal_candidate_scores(stage, sessions)
    assert exposure.nan_inner_folds == frozenset({(6,)})
    assert exposure.n_nan_evaluation_cells == 2
    assert exposure.n_evaluation_cells == 2


def test_a_candidate_with_no_evaluable_inner_fold_is_incomparable():
    sessions = [_session_record(s, i) for s in range(1, 6) for i in (0, 1)]
    candidates = [Candidate("dead", "ord_a_knn", (("n_neighbors", 15),), feature_key=(0, "A", "mag", 0, "off"))]
    cells = [_cell("dead", {2, 3}, {1}, float("nan"), {}, reason="knn_n_neighbors_15_gt_train_rows_4")
             for _ in range(2)]
    stage = StageOutcome(candidates, np.full((1, 2), np.nan), cells,
                         [CandidateScore("dead", float("nan"), 1, 12, float("nan"))])

    scores, _ = exp_c._ordinal_candidate_scores(stage, sessions)
    assert scores[0].n_evaluable_inner_folds == 0
    assert not np.isfinite(scores[0].inner_val_class_mae)


def test_single_class_truth_inner_val_counter_is_candidate_independent():
    """Counted from the fold + the spine only (never from predictions), so the O-M9-8
    exposure number cannot depend on which model was fit."""
    sessions = _make_sessions_c(n_subjects=5)
    sessions.append(_session_record(6, 0))            # subject 6 contributes ONE S0 row
    fold = _fold_for(sessions, 1)
    n = exp_c._n_single_class_truth_inner_val(fold, _classes_by_subject(sessions))
    # outer-train {2,3,4,5,6}, GroupKFold(5) -> one subject per inner-val; only subject 6's
    # validation set is single-class.
    assert n == 1


# ------------------------------------------ T-M9-expc-viability: the real composition


def _run_fold(sessions, store_dir, config, test_subject, band="10ghz", seeds=(0,)):
    return exp_c._run_single_fold_c(
        config, band, sessions, store_dir, _fold_for(sessions, test_subject), seeds
    )


def _run_fold_trace(sessions, store_dir, config, test_subject, band="10ghz", seeds=(0,)):
    return exp_c._run_single_fold_c_trace(
        config, band, sessions, store_dir, _fold_for(sessions, test_subject), seeds
    )


def test_one_inner_fold_missing_a_class_still_selects_a_winner(tmp_path, config):
    """(§2.3, trap 3) One non-evaluable inner fold must NOT sink the outer fold: the
    remaining four still produce a winner, and `n_evaluable_inner_folds` records that the
    selection stood on four cells. The plain-mean aggregation NaNs every candidate here."""
    sessions = _make_sessions_class3_only_in_subject_6()
    _write_store(tmp_path, sessions, config)

    result = _run_fold(sessions, tmp_path, config, test_subject=1)

    assert result.stage1_n_evaluable_inner_folds == 4
    assert result.stage1_viability_reason_counts["ordinal_missing_class_3_in_inner_train"] == 72
    for arm in ARMS:
        arm_result = result.arm_result(arm)
        assert arm_result.selected_family.startswith("ord_")
        assert arm_result.n_evaluable_inner_folds == 4
        assert set(np.unique(arm_result.test_predictions)).issubset(set(range(5)))


def test_all_inner_folds_missing_a_class_names_the_subject_and_the_classes(tmp_path, config):
    """The all-non-evaluable case IS the frozen "the fold contributes no ordinal score" path;
    it must surface as a named Exp C error, not a bare "no comparable candidate"."""
    sessions = _make_sessions_class3_only_in_subject_6()
    _write_store(tmp_path, sessions, config)

    with pytest.raises(ExpCError, match=r"test_subject=6"):
        _run_fold(sessions, tmp_path, config, test_subject=6)
    with pytest.raises(ExpCError, match=r"missing_classes.*3"):
        _run_fold(sessions, tmp_path, config, test_subject=6)


# --------------------------------------------------------------------- T-M9-expc-leak


def _stage1_outcome(sessions, store_dir, config, fold, band="10ghz", seeds=(0,)):
    provider = OrdinalFeatures(band, sessions, store_dir, config)
    anchor = config.search_10ghz.stage1_anchor_ridge_alpha
    return harness._score_candidates_on_fold(
        exp_c.stage1_candidates_c(config, band, anchor)[:6], fold, seeds,
        exp_c._before_fit_c(config, "stage1"), provider.data_for,
        score_fn=ordinal_class_mae_score,
    )


def test_inner_val_class_mutation_leaves_every_inner_train_fit_bytewise_identical(tmp_path, config):
    """T16 pattern over the REAL Exp C composition: an inner-VALIDATION subject's L and class
    may not touch the cutpoints, the class weights, the scaler, the base model state or the
    tuned-ε that were fit on the inner-TRAINING rows."""
    held_out, mutated = 1, 5
    sessions_a = _make_sessions_c()
    _write_store(tmp_path / "base", sessions_a, config)
    base = _stage1_outcome(sessions_a, tmp_path / "base", config, _fold_for(sessions_a, held_out))

    sessions_b = _make_sessions_c()
    _write_store(tmp_path / "mut", sessions_b, config)
    _mutate_targets_and_classes(sessions_b, mutated)
    mut = _stage1_outcome(sessions_b, tmp_path / "mut", config, _fold_for(sessions_b, held_out))

    checked = 0
    for lhs, rhs in zip(base.inner_results, mut.inner_results, strict=True):
        assert lhs.inner_val == rhs.inner_val
        assert lhs.reason == rhs.reason
        if mutated not in lhs.inner_val:
            continue
        for fb, fm in zip(lhs.fits, rhs.fits, strict=True):
            assert fb.quantity == fm.quantity
            for k in fb.params:
                assert fb.params[k].tobytes() == fm.params[k].tobytes()
        checked += 1
    assert checked > 0, "fixture never put the mutated subject in a validation role"


def test_inner_train_class_mutation_moves_the_cutpoints(tmp_path, config):
    """The power companion: the same mutation applied to an inner-TRAINING subject DOES move
    the fitted state — otherwise the test above would pass against an implementation that
    never fits anything on the labels at all."""
    held_out, mutated = 1, 5
    sessions_a = _make_sessions_c()
    _write_store(tmp_path / "base", sessions_a, config)
    base = _stage1_outcome(sessions_a, tmp_path / "base", config, _fold_for(sessions_a, held_out))

    sessions_b = _make_sessions_c()
    _write_store(tmp_path / "mut", sessions_b, config)
    _mutate_targets_and_classes(sessions_b, mutated)
    mut = _stage1_outcome(sessions_b, tmp_path / "mut", config, _fold_for(sessions_b, held_out))

    moved = False
    for lhs, rhs in zip(base.inner_results, mut.inner_results, strict=True):
        if mutated in lhs.inner_val:
            continue
        for fb, fm in zip(lhs.fits, rhs.fits, strict=True):
            if fb.quantity == "ord_a_ridge" and fb.params["cutpoints_"].tobytes() != fm.params["cutpoints_"].tobytes():
                moved = True
    assert moved


# ------------------------------------------------------------------ T-M9-expc-mutation


def test_outer_mutation_property_end_to_end(tmp_path, config):
    """(C2 pattern, over the real two-arm Exp C composition) Mutate the held-out subject's
    stored tensors AND its L AND its ordinal class; every inner score, both arms' winners,
    every fitted parameter (cutpoints, class weights, scaler, base state, tuned-ε) and the
    outer-training predictions must be bytewise unchanged. Only the held-out subject's own
    predictions may move."""
    held_out = 1
    sessions_a = _make_sessions_c(n_subjects=4)
    _write_store(tmp_path / "base", sessions_a, config)
    base_trace = _run_fold_trace(sessions_a, tmp_path / "base", config, held_out)
    base = base_trace.result

    sessions_b = _make_sessions_c(n_subjects=4)
    _write_store(tmp_path / "mut", sessions_b, config)
    _mutate_features_on_disk(tmp_path / "mut", sessions_b, held_out)
    _mutate_targets_and_classes(sessions_b, held_out)
    mut_trace = _run_fold_trace(sessions_b, tmp_path / "mut", config, held_out)
    mut = mut_trace.result

    def assert_fit_records_identical(lhs, rhs):
        assert len(lhs) == len(rhs)
        for fb, fm in zip(lhs, rhs, strict=True):
            assert fb.quantity == fm.quantity
            assert fb.role == fm.role
            assert fb.subjects == fm.subjects
            assert set(fb.params) == set(fm.params)
            for key in fb.params:
                assert fb.params[key].tobytes() == fm.params[key].tobytes()

    def assert_stage_identical(lhs, rhs):
        assert lhs.inner_scores.tobytes() == rhs.inner_scores.tobytes()
        assert [c.candidate_id for c in lhs.candidates] == [
            c.candidate_id for c in rhs.candidates
        ]
        assert len(lhs.inner_results) == len(rhs.inner_results)
        for lb, lm in zip(lhs.inner_results, rhs.inner_results, strict=True):
            assert lb.inner_train == lm.inner_train
            assert lb.inner_val == lm.inner_val
            assert lb.candidate_id == lm.candidate_id
            assert lb.reason == lm.reason
            assert set(lb.val_predictions) == set(lm.val_predictions)
            for subject in lb.val_predictions:
                assert (
                    lb.val_predictions[subject].tobytes()
                    == lm.val_predictions[subject].tobytes()
                )
            assert_fit_records_identical(lb.fits, lm.fits)

    # The full inner search, not only its selected winner, is held-out-subject invariant.
    assert_stage_identical(base_trace.stage1, mut_trace.stage1)
    for arm in ARMS:
        assert_stage_identical(
            base_trace.stage2_by_arm[arm], mut_trace.stage2_by_arm[arm]
        )

    assert base.stage1_feature_key == mut.stage1_feature_key
    assert base.stage1_selected_params == mut.stage1_selected_params
    assert base.stage1_n_evaluable_inner_folds == mut.stage1_n_evaluable_inner_folds
    assert base.stage1_viability_reason_counts == mut.stage1_viability_reason_counts
    assert base.n_single_class_truth_inner_val == mut.n_single_class_truth_inner_val
    assert base.n_qwk_nan_inner == mut.n_qwk_nan_inner
    assert (
        base.n_qwk_nan_inner_evaluation_cells
        == mut.n_qwk_nan_inner_evaluation_cells
    )
    assert base.n_qwk_inner_evaluation_cells == mut.n_qwk_inner_evaluation_cells
    held_out_output_moved = False
    for arm in ARMS:
        lhs, rhs = base.arm_result(arm), mut.arm_result(arm)
        assert lhs.selected_feature_key == rhs.selected_feature_key
        assert lhs.selected_family == rhs.selected_family
        assert lhs.selected_params == rhs.selected_params
        assert lhs.n_evaluable_inner_folds == rhs.n_evaluable_inner_folds
        assert lhs.viability_reason_counts == rhs.viability_reason_counts
        assert_fit_records_identical(lhs.final_fits, rhs.final_fits)
        for sb, sm in zip(lhs.seed_outcomes, rhs.seed_outcomes, strict=True):
            assert sb.train_predictions.tobytes() == sm.train_predictions.tobytes()
            held_out_output_moved |= (
                sb.test_predictions.tobytes() != sm.test_predictions.tobytes()
                or sb.test_score != sm.test_score
            )
    # Power: the fixture genuinely moves a held-out prediction or score. Comparing only the
    # deliberately mutated truth would make the no-leakage assertions above vacuous.
    assert base.test_classes.tolist() != mut.test_classes.tolist()
    assert held_out_output_moved


# ----------------------------------------------------------------------- T-M9-parallel


def test_exp_c_routes_through_fold_parallel(monkeypatch, tmp_path, config):
    """Rules out exp_c growing its own copy of the pool (the drift the step-5 extraction
    exists to prevent) and pins the label every Exp C IBEX log line carries."""
    calls = []

    def recorder(worker, tasks, n_workers, label):
        calls.append({"worker": worker, "tasks": list(tasks), "n_workers": n_workers, "label": label})
        return []

    monkeypatch.setattr(fold_parallel, "run_folds_parallel", recorder)
    sessions = _make_sessions_c()

    assert run_exp_c(config, "10ghz", sessions, tmp_path, seeds=(0,), n_workers=3) == []
    assert len(calls) == 1
    assert calls[0]["worker"] is exp_c._run_single_fold_c
    assert calls[0]["label"] == "exp_c"
    assert calls[0]["n_workers"] == 3
    assert len(calls[0]["tasks"]) == 6
    assert [t[4].test_subject for t in calls[0]["tasks"]] == [1, 2, 3, 4, 5, 6]


def test_parallel_folds_are_bit_identical_to_serial(tmp_path, config):
    sessions = _make_sessions_c(n_subjects=4)
    _write_store(tmp_path, sessions, config)
    serial = run_exp_c(config, "10ghz", sessions, tmp_path, seeds=(0,), n_workers=1)
    parallel = run_exp_c(config, "10ghz", sessions, tmp_path, seeds=(0,), n_workers=2)

    assert [r.test_subject for r in serial] == [r.test_subject for r in parallel] == [1, 2, 3, 4]
    for rs, rp in zip(serial, parallel, strict=True):
        for arm in ARMS:
            lhs, rhs = rs.arm_result(arm), rp.arm_result(arm)
            assert lhs.selected_feature_key == rhs.selected_feature_key
            assert lhs.selected_family == rhs.selected_family
            assert lhs.selected_params == rhs.selected_params
            assert lhs.test_predictions.tobytes() == rhs.test_predictions.tobytes()
            for fb, fm in zip(lhs.final_fits, rhs.final_fits, strict=True):
                for k in fb.params:
                    assert fb.params[k].tobytes() == fm.params[k].tobytes()


# ------------------------------------------------------------------- T-M9-expc-report
#
# `summarize_exp_c` / `write_exp_c_reports` / `run_and_report_c` consume a list of
# ExpCFoldResult + config, so they are tested on hand-built results (test_exp_b.py's
# stubbing pattern) rather than through the expensive store+search path.


def _fake_arm(arm, classes_pred_by_seed, *, family="ord_a_ridge", n_evaluable=5):
    seed_outcomes = [
        SeedOutcome(seed, np.array([]), np.array(preds, dtype=float), 0.0)
        for seed, preds in enumerate(classes_pred_by_seed)
    ]
    return ExpCArmResult(
        arm=arm,
        selected_feature_key=(0, "A", "mag", 0, "off"),
        selected_family=family if arm == "a" else "ord_b_frank_hall",
        selected_params={"alpha": 1.0} if arm == "a" else {"C": 1.0},
        test_predictions=seed_outcomes[0].test_predictions,
        seed_outcomes=seed_outcomes,
        final_fits=[],
        n_evaluable_inner_folds=n_evaluable,
        viability_reason_counts={},
    )


def _fake_result_c(test_subject, true_classes, preds_a_by_seed, preds_b_by_seed, *,
                   n_single_class_truth_inner_val=0, n_qwk_nan_inner=0,
                   n_qwk_nan_inner_evaluation_cells=0, n_qwk_inner_evaluation_cells=0):
    true_classes = np.array(true_classes, dtype=int)
    return ExpCFoldResult(
        test_subject=test_subject,
        stage1_feature_key=(0, "A", "mag", 0, "off"),
        stage1_selected_params={"alpha": 1.0},
        stage1_n_evaluable_inner_folds=5,
        stage1_viability_reason_counts={},
        arm_a=_fake_arm("a", preds_a_by_seed),
        arm_b=_fake_arm("b", preds_b_by_seed),
        test_classes=true_classes,
        test_targets=true_classes.astype(float) * 0.5,
        test_session_idx=true_classes.copy(),
        n_single_class_truth_inner_val=n_single_class_truth_inner_val,
        n_qwk_nan_inner=n_qwk_nan_inner,
        n_qwk_nan_inner_evaluation_cells=n_qwk_nan_inner_evaluation_cells,
        n_qwk_inner_evaluation_cells=n_qwk_inner_evaluation_cells,
    )


def _fake_results_c():
    """Four subjects with all five classes; arm (a) predicts perfectly for seed 0 and shifts
    one row for seed 1, arm (b) is off by one class on the last row."""
    out = []
    for subject in (1, 2, 3, 4):
        truth = [0, 1, 2, 3, 4]
        out.append(_fake_result_c(
            subject, truth,
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 3]],
            [[0, 1, 2, 3, 3], [0, 1, 2, 3, 3]],
        ))
    return out


def test_summarize_exp_c_reports_both_arms_with_the_three_ordinal_metrics(config):
    summary = summarize_exp_c(_fake_results_c(), config)

    assert summary["conditional_exploratory"] is True
    assert set(summary["arms"]) == {"a", "b"}
    for arm in ARMS:
        block = summary["arms"][arm]
        for metric in ("class_unit_mae", "adjacent_accuracy", "quadratic_weighted_kappa"):
            ci = block[metric]
            assert set(ci) == {"point", "low", "high", "method", "n_eval", "n_skipped", "unreliable"}
        assert set(block["per_subject_class_mae"]) == {1, 2, 3, 4}
    # arm (a): seed 0 is perfect, seed 1 misses one of five rows per subject
    # -> pooled class-unit MAE per seed = 0.0 and 0.2, seed-averaged = 0.1.
    assert summary["arms"]["a"]["class_unit_mae"]["point"] == pytest.approx(0.1)
    # arm (b): both seeds miss the same single row -> 0.2 for each seed.
    assert summary["arms"]["b"]["class_unit_mae"]["point"] == pytest.approx(0.2)
    # every prediction is within one class of the truth on this fixture.
    assert summary["arms"]["a"]["adjacent_accuracy"]["point"] == pytest.approx(1.0)


def test_summarize_exp_c_confusion_matrix_is_the_per_seed_average(config):
    summary = summarize_exp_c(_fake_results_c(), config)
    confusion = np.array(summary["arms"]["a"]["confusion_matrix_mean_over_seeds"], dtype=float)

    assert confusion.shape == (5, 5)
    assert confusion.sum() == pytest.approx(20.0)     # 4 subjects x 5 rows
    # true class 4: seed 0 predicts 4 (4 rows), seed 1 predicts 3 (4 rows) -> the seed MEAN
    # is 2.0 in each of those two cells. A first-seed-only or a pooled-over-seeds count
    # (8.0 / 4.0) fails here.
    assert confusion[4, 4] == pytest.approx(2.0)
    assert confusion[4, 3] == pytest.approx(2.0)


def test_summarize_exp_c_has_no_baseline_comparison_field(config):
    """(trap 16) The session-index baseline predicts the Exp C class perfectly by
    construction, so any radar-vs-baseline framing here is degenerate and the freeze
    registers none. A p-value in this summary would be an undisclosed protocol invention."""
    summary = summarize_exp_c(_fake_results_c(), config)
    flat = json.dumps(summary)
    assert "baseline" not in flat
    assert "wilcoxon" not in flat and "holm" not in flat


def test_summarize_exp_c_reports_the_o_m9_8_counters_at_both_cv_levels(config):
    """Exactly one single-class-truth validation fold at each level: one inner fold carries
    the flag, and subject 4 contributes a single S0 test row at the outer level."""
    results = _fake_results_c()
    results[0] = dataclasses.replace(
        results[0],
        n_single_class_truth_inner_val=1,
        n_qwk_nan_inner=1,
        n_qwk_nan_inner_evaluation_cells=2,
        n_qwk_inner_evaluation_cells=10,
    )
    results[3] = _fake_result_c(4, [0], [[0], [0]], [[3], [3]])   # single-class outer truth

    summary = summarize_exp_c(results, config)
    counters = summary["qwk_undefinedness"]

    assert counters["inner"]["n_single_class_truth_val_folds"] == 1
    assert counters["inner"]["n_qwk_nan"] == 1
    assert counters["inner"]["n_qwk_nan_evaluation_cells"] == 2
    assert counters["inner"]["n_qwk_evaluation_cells"] == 10
    assert counters["inner"]["evaluation_cell"] == "stage_x_candidate_x_inner_fold"
    assert counters["outer"]["n_single_class_truth_val_folds"] == 1
    # subject 4's outer QWK: arm (a) predicts the same single class (both marginals on one
    # class -> undefined) for two realized seeds, arm (b) predicts a different class
    # (defined, κ = 0). The fold count is one; the arm-seed cell count is two.
    assert counters["outer"]["n_qwk_nan"] == 1
    assert counters["outer"]["n_qwk_nan_evaluation_cells"] == 2
    assert counters["outer"]["n_qwk_evaluation_cells"] == 16
    assert counters["outer"]["evaluation_cell"] == "arm_x_realized_seed_x_outer_fold"


def test_summarize_exp_c_selection_frequency_carries_the_per_fold_evaluability(config):
    summary = summarize_exp_c(_fake_results_c(), config)
    block = summary["arms"]["a"]
    assert block["selection_frequency"]["family"] == {"ord_a_ridge": 4}
    per_fold = {row["test_subject"]: row for row in block["per_fold"]}
    assert set(per_fold) == {1, 2, 3, 4}
    assert per_fold[1]["n_evaluable_inner_folds"] == 5
    assert "viability_reason_counts" in per_fold[1]


def test_rng_offsets_are_pairwise_distinct_including_exp_bs(config):
    offsets = exp_c._all_rng_offsets()
    assert sorted(offsets) == [200, 201, 202, 210, 211, 212]
    assert len(set(offsets)) == len(offsets)
    # Exp A occupies seed+0..3; Exp B's are named in exp_b.
    assert not set(offsets) & set(exp_b._all_rng_offsets())
    assert not set(offsets) & {0, 1, 2, 3}


def test_write_exp_c_reports_writes_every_artifact(tmp_path, config):
    results = _fake_results_c()
    summary = summarize_exp_c(results, config)
    paths = write_exp_c_reports(results, summary, tmp_path, "10ghz")

    assert set(paths) == {"metrics", "predictions", "selection_table", "confusion", "confusion_figure"}
    assert all(p.exists() for p in paths.values())

    written = json.loads(paths["metrics"].read_text())
    assert written["conditional_exploratory"] is True

    lines = paths["predictions"].read_text().strip().splitlines()
    assert lines[0] == "subject,arm,seed,session_idx,y_class_true,y_class_pred"
    assert len(lines) == 1 + 4 * 2 * 2 * 5   # subjects x arms x seeds x rows

    confusion_rows = paths["confusion"].read_text().strip().splitlines()
    assert confusion_rows[0].startswith("arm,true_class,")
    assert len(confusion_rows) == 1 + 2 * 5

    # one row per (fold, stage/arm), each carrying the §2.3 evaluability record so a fold
    # selected on fewer than the full inner folds is visible in the artifact.
    selection_rows = paths["selection_table"].read_text().strip().splitlines()
    assert selection_rows[0].split(",") == [
        "test_subject", "arm", "feature_key", "family", "params",
        "n_evaluable_inner_folds", "viability_reason_counts", "reason",
    ]
    assert len(selection_rows) == 1 + 4 * 3          # subjects x (stage1, arm a, arm b)
    assert sum("stage1" in row for row in selection_rows) == 4


def test_run_and_report_c_smoke_writes_no_performance_value(tmp_path, config, monkeypatch):
    sessions = _make_sessions_c(n_subjects=4)
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda *a, **k: {})
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)
    monkeypatch.setattr(exp_c, "run_exp_c", lambda *a, **k: _fake_results_c())
    monkeypatch.setattr(exp_c, "_assert_mechanism_ok_c", lambda *a, **k: None)

    run_dir = tmp_path / "run"
    outputs = run_and_report_c(config, "10ghz", sessions, tmp_path, run_dir,
                               mode="smoke", analysis_commit="x")

    assert set(outputs) == {"run_log"}
    files = {p.name for p in run_dir.iterdir()}
    assert files == {"run_log_10ghz.json"}
    log = json.loads(outputs["run_log"].read_text())
    # mechanism-only: structural counts, never a class metric or a selected configuration.
    assert set(log) == {"stage", "band", "mode", "n_folds", "n_sessions", "note"}


def test_run_and_report_c_full_writes_the_metrics_and_confusion_artifacts(tmp_path, config, monkeypatch):
    sessions = _make_sessions_c(n_subjects=4)
    monkeypatch.setattr(exp_a, "expected_fingerprints", lambda *a, **k: {})
    monkeypatch.setattr(exp_a.store_mod, "validate_store", lambda *a, **k: None)
    monkeypatch.setattr(exp_c, "run_exp_c", lambda *a, **k: _fake_results_c())
    monkeypatch.setattr(exp_c, "_assert_mechanism_ok_c", lambda *a, **k: None)

    run_dir = tmp_path / "run"
    outputs = run_and_report_c(config, "10ghz", sessions, tmp_path, run_dir,
                               mode="full", analysis_commit="x")

    assert outputs["metrics"].exists() and outputs["confusion_figure"].exists()
    summary = json.loads(outputs["metrics"].read_text())
    assert set(summary["arms"]) == {"a", "b"}


def test_assert_mechanism_ok_c_requires_s0_and_in_range_classes(config):
    """Unlike Exp B (which excludes S0 at the spine), all five classes ARE Exp C's task, so a
    spine that lost S0 is a broken run, not a valid one."""
    sessions = _make_sessions_c(n_subjects=4)
    results = _fake_results_c()
    exp_c._assert_mechanism_ok_c(results, sessions)          # must not raise

    without_s0 = [s for s in sessions if s["session_idx"] != 0]
    with pytest.raises(AssertionError):
        exp_c._assert_mechanism_ok_c(results, without_s0)

    off_grid = _fake_results_c()
    off_grid[0].arm_a.seed_outcomes[0].test_predictions[0] = 7.0
    with pytest.raises(AssertionError):
        exp_c._assert_mechanism_ok_c(off_grid, sessions)
