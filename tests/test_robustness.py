"""T-M10-rob: Experiment H's selection-variance robustness bootstrap (`eval/robustness.py`).

Split the way `test_exp_b.py` / `test_exp_c.py` are: a RUN half that drives the real refit path
on a synthetic store (no private data), and a REDUCE half that runs on hand-built fold results
so the estimand arithmetic, the skip rules and the artifact schemas can be checked against the
SPECIFICATION rather than against whatever the implementation happened to produce.

Groups: T-M10-rob-seed (the RNG freeze), T-M10-rob-skip (all-or-nothing replicates),
T-M10-rob-estimand (the four estimand definitions), T-M10-rob-audit (winner/fit provenance),
T-M10-rob-range (percentiles and the min_successful rule), T-M10-rob-e2e.

Every expected number below is derived from plan §2.4's arithmetic by hand. The load-bearing
ones are: the estimand at multiplicity one must equal what `summarize_exp_a` already reports,
and the estimand under a draw must equal what an explicitly duplicated cohort would give.
"""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from dehyd.config import StatsConfig, load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_a, robustness
from dehyd.eval.exp_a import ExpAFoldResult
from dehyd.eval.exp_b import ExpBError, ExpBFoldResult
from dehyd.eval.exp_c import ExpCArmResult, ExpCFoldResult, ExpCProtocolError
from dehyd.eval.harness import FitRecord, SeedOutcome
from dehyd.eval.robustness import RobustnessError
from dehyd.eval.selection import SelectionError
from dehyd.features.pooling import aggregate_session, pool_stats_batch
from dehyd.features.store import order_key, prelog_key, raw_key, vec_key, write_session_store
from dehyd.features.wst import apply_order_log

P, T, CN, NFR = 6, 4, 1, 3       # tiny path/time/channel/frame dims for a fast synthetic store
ORDER = np.array([0, 1, 1, 2, 2, 2])


@pytest.fixture(scope="module")
def config():
    """The frozen search space, with the smoke overlay's single seed.

    `search_10ghz`/`model_grid`/`stats` are milestone-6 sections and cannot be shrunk — the
    protocol-freeze guard rejects any deviation — so the RUN-half tests pay for the real
    113-candidate staged search and are kept deliberately few and small.
    """
    return load_config("configs/exp_a_regression.yaml", "configs/stats.yaml", "configs/smoke.yaml")


# ------------------------------------------------------------------------------- fixtures


def _spine(n_subjects=5, classes=range(5)):
    out = []
    for s in range(1, n_subjects + 1):
        for i in classes:
            delta = 0.0 if i == 0 else -(0.3 * i + 0.05 * s)
            out.append({
                "subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
                "delta_m_pct": delta, "loss_l": -delta, "class_idx": i,
            })
    return out


def _write_store(store_dir, sessions, config, seed=0):
    """One synthetic session store, in `exp_a.StoreBackedFeatures`'s own key layout."""
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
                        off = aggregate_session(pool_stats_batch(
                            apply_order_log(raw, meta, wst, log_on=False), meta))
                        frozen = aggregate_session(pool_stats_batch(
                            apply_order_log(raw, meta, wst, log_on=True, epsilon_by_order=eps), meta))
                        npz[vec_key(gi, r, c, ti, "off")] = off
                        npz[vec_key(gi, r, c, ti, "frozen")] = frozen
                        npz[raw_key(gi, r, c, ti)] = raw
                        npz[prelog_key(gi, r, c, ti)] = np.array([raw.mean()] * 3)
        write_session_store("10ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR}, store_dir)


def _a_fold(subject, y_true, y_pred, baseline, *, fits=()):
    """One hand-built Exp A fold result: one seed, explicit predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return ExpAFoldResult(
        test_subject=subject, selected_feature_key=(0, "A", "mag", 0, "off"),
        selected_family="ridge", selected_params={"alpha": 1.0},
        test_predictions=y_pred, test_targets=y_true,
        seed_outcomes=[SeedOutcome(1, np.array([]), y_pred, 0.0)],
        baseline_predictions=np.asarray(baseline, dtype=float), final_fits=list(fits),
    )


def _b_fold(subject, session_idx, y_true, y_pred, *, reason=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return ExpBFoldResult(
        test_subject=subject, selected_feature_key=(0, "A", "mag", 0, "off"),
        selected_family="ridge", selected_params={"alpha": 1.0},
        test_predictions=y_pred, test_targets=y_true,
        test_session_idx=np.asarray(session_idx, dtype=int),
        seed_outcomes=[SeedOutcome(1, np.array([]), y_pred, 0.0)],
        baseline_predictions=np.zeros_like(y_true), final_fits=[],
        dropped_sessions_outer=(), dropped_sessions_inner=(), reason=reason,
    )


def _c_arm(arm, predictions):
    return ExpCArmResult(
        arm=arm, selected_feature_key=(0, "A", "mag", 0, "off"),
        selected_family="ord_a_ridge" if arm == "a" else "ord_b_frank_hall",
        selected_params={"alpha": 1.0} if arm == "a" else {"C": 1.0},
        test_predictions=np.asarray(predictions, dtype=float),
        seed_outcomes=[SeedOutcome(1, np.array([]), np.asarray(predictions, dtype=float), 0.0)],
        final_fits=[], n_evaluable_inner_folds=4, viability_reason_counts={},
    )


def _c_fold(subject, classes, pred_a, pred_b):
    return ExpCFoldResult(
        test_subject=subject, stage1_feature_key=(0, "A", "mag", 0, "off"),
        stage1_selected_params={"alpha": 1.0}, stage1_n_evaluable_inner_folds=4,
        stage1_viability_reason_counts={}, arm_a=_c_arm("a", pred_a), arm_b=_c_arm("b", pred_b),
        test_classes=np.asarray(classes, dtype=float),
        test_targets=np.asarray(classes, dtype=float),
        test_session_idx=np.asarray(classes, dtype=int),
        n_single_class_truth_inner_val=0, n_qwk_nan_inner=0,
    )


def _replicate_drawing_at_least(config, experiment, band, subjects, k):
    """The first replicate index whose draw covers >= k distinct subjects.

    The draw is frozen by the seed tuple, so which replicate that is cannot be chosen freely —
    a test that wants to exercise a rule PAST the coarse distinct-subject precheck has to find
    a replicate the precheck lets through rather than assume replicate 1 does.
    """
    for replicate in range(1, 200):
        drawn, _ = robustness.draw_subject_multiplicity(
            subjects, robustness.replicate_seed_tuple(config, experiment, band, replicate)
        )
        if len(drawn) >= k:
            return replicate
    raise AssertionError(f"no replicate drew {k} distinct subjects from {subjects}")


def _outcome(replicate, status, estimates=None, skip_reason=None, multiplicity=None):
    return robustness.ReplicateOutcome(
        experiment="a", band="10ghz", replicate=replicate,
        seed_tuple=(20260721, 1, 10, replicate),
        generated_seed_state="0" * 32, multiplicity=multiplicity or {1: 1, 2: 1, 3: 1, 4: 1},
        n_distinct_subjects=4, status=status, skip_reason=skip_reason,
        estimates=estimates or {},
    )


# ------------------------------------------------------- T-M10-rob-seed: the RNG freeze


def test_robustness_seed_is_config_run_seed_and_is_not_a_new_config_field(config):
    """The owner's 2026-08-08 decision, pinned so it can never become an accident of whichever
    attribute an implementation reached for. `StatsConfig` deliberately gains NO
    `robustness_seed` field: the M6 sections are frozen records and the value already exists."""
    assert robustness.robustness_seed(config) == config.run.seed == 20260721
    assert not hasattr(StatsConfig(), "robustness_seed")
    assert {f.name for f in dataclasses.fields(StatsConfig) if f.name.startswith("robustness")} == {
        "robustness_replicates_r", "robustness_min_distinct_subjects",
        "robustness_min_successful_replicates", "robustness_ordinal_min_classes",
    }


def test_seed_tuple_is_the_frozen_form_with_the_versioned_codes(config):
    """Plan §2.4: `[robustness_seed, experiment_code, band_code, replicate]`, a,b,c -> 1,2,3
    and 10ghz,77ghz -> 10,77. The VALUES are the contract; renumbering them would silently
    re-draw every replicate in the milestone."""
    assert robustness.EXPERIMENT_CODE == {"a": 1, "b": 2, "c": 3}
    assert robustness.BAND_CODE == {"10ghz": 10, "77ghz": 77}
    assert robustness.replicate_seed_tuple(config, "a", "10ghz", 1) == (20260721, 1, 10, 1)
    assert robustness.replicate_seed_tuple(config, "c", "77ghz", 200) == (20260721, 3, 77, 200)
    for bad in (("z", "10ghz"), ("a", "24ghz")):
        with pytest.raises(RobustnessError):
            robustness.replicate_seed_tuple(config, *bad, 1)


def test_the_same_tuple_reproduces_the_same_draw_and_the_same_128_bit_state(config):
    subjects = range(1, 17)
    tuple_ = robustness.replicate_seed_tuple(config, "a", "10ghz", 7)
    first = robustness.draw_subject_multiplicity(subjects, tuple_)
    second = robustness.draw_subject_multiplicity(subjects, tuple_)
    assert first == second
    state = first[1]
    assert len(state) == 32 and int(state, 16) >= 0        # 4 x uint32 = the 128-bit state


def test_a_draw_is_n_subject_ids_with_replacement(config):
    """Every replicate draws N — the cohort size it stands in for — so the copies sum back to
    N and the distinct count is whatever the draw happened to cover."""
    subjects = list(range(1, 17))
    for replicate in range(1, 25):
        drawn, _ = robustness.draw_subject_multiplicity(
            subjects, robustness.replicate_seed_tuple(config, "a", "10ghz", replicate)
        )
        assert sum(drawn.values()) == len(subjects)
        assert set(drawn).issubset(subjects)
        assert all(m >= 1 for m in drawn.values())


def test_different_experiments_bands_and_replicates_draw_different_cohorts(config):
    """The experiment and band sit in the tuple precisely so the six launch-matrix jobs do not
    share one draw — a shared cohort would correlate their ranges for no scientific reason."""
    subjects = list(range(1, 17))

    def draw(experiment, band, replicate):
        return robustness.draw_subject_multiplicity(
            subjects, robustness.replicate_seed_tuple(config, experiment, band, replicate)
        )[0]

    assert draw("a", "10ghz", 1) != draw("b", "10ghz", 1)
    assert draw("a", "10ghz", 1) != draw("a", "77ghz", 1)
    assert draw("a", "10ghz", 1) != draw("a", "10ghz", 2)


def test_one_draw_is_shared_by_every_arm_of_one_experiment_band_replicate(config):
    """§2.4: "the resulting subject multiplicity draw is shared by all arms/contrasts of that
    experiment-band replicate". The draw is a pure function of (experiment, band, replicate),
    which is what makes that structurally true rather than a convention to remember."""
    results = [_c_fold(s, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 2, 4]) for s in (1, 2, 3, 4)]
    multiplicity = {1: 2, 2: 1, 3: 1, 4: 1}
    rows = robustness._selection_rows(config, "c", "10ghz", 5, results, multiplicity)
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm_or_contrast"], set()).add(row["multiplicity_sha256"])
    assert set(by_arm) == {"arm_a_class_unit_mae", "arm_b_class_unit_mae"}
    assert len(set().union(*by_arm.values())) == 1        # one draw, one hash, both arms


def test_model_seeds_are_the_configured_seeds_not_derived_from_the_resampling_seed(config):
    """§2.4 is explicit about this: a replicate changes which subjects are drawn, never which
    seeds are fit."""
    assert tuple(robustness.model_seeds(config)) == tuple(config.run.seed_set)


# --------------------------------------------------- T-M10-rob-skip: all-or-nothing replicates


def test_skip_reasons_are_the_canonical_ordered_set():
    """The tuple IS the definition of "the first canonical reason" (§2.4), so its ORDER is part
    of the contract and is pinned here rather than left to reading the code."""
    assert robustness.SKIP_REASONS == (
        "insufficient_distinct_subjects",
        "ordinal_missing_classes",
        "no_surviving_candidate",
        "missing_outer_prediction",
        "expb_primary_aggregate_unavailable",
        "non_finite_estimate",
    )


def test_a_draw_with_too_few_distinct_subjects_skips_before_touching_the_store(config, tmp_path):
    """The coarse precheck. At 4 distinct subjects every outer fold still has 3 training
    subjects (`splits.DEFAULT_MIN_TRAIN_SUBJECTS`); below it, folds would silently drop out."""
    subjects = [1, 2, 3, 4]
    # Only the replicates whose DRAW is short are run: `tmp_path` holds no store, so a
    # replicate that got past the precheck would fail on a missing npz instead of proving
    # anything. That the precheck fires before any store access is itself the claim here.
    thin = [
        r for r in range(1, 40)
        if len(robustness.draw_subject_multiplicity(
            subjects, robustness.replicate_seed_tuple(config, "a", "10ghz", r))[0]) < 4
    ]
    assert thin, "a 4-subject pool must sometimes draw fewer than 4 distinct subjects"
    skipped = [
        robustness.run_replicate(config, "a", "10ghz", _spine(4), tmp_path,
                                 config.run.seed_set, subjects, r)
        for r in thin
    ]
    for outcome in skipped:
        assert outcome.status == robustness.STATUS_SKIPPED
        assert outcome.skip_reason == "insufficient_distinct_subjects"
        assert outcome.n_distinct_subjects < config.stats.robustness_min_distinct_subjects
        assert outcome.estimates == {}          # no partial estimate exists, at all
        assert outcome.selection_rows == [] and outcome.fit_audit_rows == []


def test_exp_c_additionally_requires_all_five_classes_in_the_resampled_cohort(config, tmp_path):
    """Exp C's own precheck. A cohort with no S4 row cannot produce a 5-class ordinal result,
    and the reason names the missing classes rather than reporting a generic failure."""
    spine = _spine(6, classes=(0, 1, 2, 3))          # class 4 is absent cohort-wide
    subjects = [1, 2, 3, 4, 5, 6]
    replicate = _replicate_drawing_at_least(config, "c", "10ghz", subjects, 4)
    outcome = robustness.run_replicate(
        config, "c", "10ghz", spine, tmp_path, config.run.seed_set, subjects, replicate
    )
    assert outcome.status == robustness.STATUS_SKIPPED
    assert outcome.skip_reason == "ordinal_missing_classes"
    assert "missing=[4]" in outcome.skip_detail
    assert outcome.estimates == {}


def test_a_nested_selection_with_no_surviving_candidate_skips_the_whole_replicate(config, tmp_path,
                                                                                  monkeypatch):
    """§2.4: the whole result replicate is skipped, "never computed over the remaining easier
    folds". Each of the three enumerated no-candidate errors maps to the same canonical reason."""
    subjects = [1, 2, 3, 4, 5, 6]
    replicate = _replicate_drawing_at_least(config, "a", "10ghz", subjects, 4)
    for error in (SelectionError("nothing comparable"),
                  ExpBError("no candidate produced a finite equal_session_residual_mae"),
                  robustness.ExpCError("no candidate was comparable")):
        def explode(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(robustness, "_run_procedure", explode)
        outcome = robustness.run_replicate(
            config, "a", "10ghz", _spine(6), tmp_path, config.run.seed_set, subjects, replicate
        )
        assert outcome.skip_reason == "no_surviving_candidate"
        assert outcome.estimates == {}
        assert type(error).__name__ in outcome.skip_detail


def test_a_protocol_violation_is_never_swallowed_as_a_skip(config, tmp_path, monkeypatch):
    """`ExpCProtocolError` subclasses `ExpCError`, so a naive `except ExpCError` would turn an
    unauthorized fit into a quietly counted skip. It must propagate."""
    def explode(*_args, **_kwargs):
        raise ExpCProtocolError("family 'linear_regression' is not authorized")

    subjects = [1, 2, 3, 4, 5, 6]
    replicate = _replicate_drawing_at_least(config, "c", "10ghz", subjects, 4)
    monkeypatch.setattr(robustness, "_run_procedure", explode)
    with pytest.raises(ExpCProtocolError):
        robustness.run_replicate(config, "c", "10ghz", _spine(6), tmp_path, config.run.seed_set,
                                 subjects, replicate)


def test_a_missing_outer_prediction_skips_rather_than_summarizing_the_folds_that_worked():
    """A replicate that covers 3 of its 4 required subjects has no estimate — not a 3-subject
    one. This is the fixture §5.5 asks for: it passes the coarse distinct-subject check and
    fails a nested fold."""
    results = [_a_fold(s, [-1.0, -2.0], [-1.2, -1.8], [-0.9, -2.1]) for s in (1, 2, 3)]
    estimates, reason = robustness._exp_a_estimates(results, [1, 2, 3, 4], {1: 1, 2: 1, 3: 1, 4: 1})
    assert estimates is None and reason == "missing_outer_prediction"

    b_results = [_b_fold(1, [1, 2, 3, 4], [-1.0] * 4, [-1.1] * 4),
                 _b_fold(2, [], [], [], reason="no_surviving_test_rows")]
    estimates, reason = robustness._exp_b_estimates(b_results, [1, 2], None)
    assert estimates is None and reason == "missing_outer_prediction"


def test_the_all_or_nothing_rule_is_a_replicate_rule_not_a_full_cohort_rule():
    """§2.4's completeness requirement is written about a REPLICATE. Imposing it on the original
    full-cohort point would make that point unobtainable in exactly the case where the ordinary
    Exp B run reports one: a fold with `reason="no_surviving_test_rows"` is recorded in Exp B's
    exclusion ledger and `summarize_exp_b` still reports the primary aggregate over the folds
    that contributed. `required_subjects=None` is what distinguishes the two callers."""
    results = [
        _b_fold(1, [1, 2, 3, 4], [-1.0, -2.0, -3.0, -4.0], [-1.1, -1.9, -3.2, -3.8]),
        _b_fold(2, [1, 2, 3, 4], [-1.0, -2.0, -3.0, -4.0], [-0.9, -2.1, -2.8, -4.2]),
        _b_fold(3, [], [], [], reason="no_surviving_test_rows"),
    ]
    # as a replicate: skipped, all-or-nothing.
    assert robustness._exp_b_estimates(results, [1, 2, 3], None)[1] == "missing_outer_prediction"
    # as the full-cohort point: reported, over the folds that contributed.
    estimates, reason = robustness._exp_b_estimates(results, None, None)
    assert reason is None
    assert math.isfinite(estimates["radar_minus_baseline_equal_session_aggregate"])


def test_exp_b_without_all_four_sessions_reports_the_aggregate_unavailable():
    """`equal_session_residual_mae` silently averages whatever sessions it is handed, so a
    replicate missing S4 would otherwise report a three-session mean labelled as the frozen
    four-session primary aggregate."""
    results = [_b_fold(s, [1, 2, 3], [-1.0, -2.0, -3.0], [-1.1, -1.9, -3.2]) for s in (1, 2, 3, 4)]
    estimates, reason = robustness._exp_b_estimates(results, [1, 2, 3, 4], None)
    assert estimates is None and reason == "expb_primary_aggregate_unavailable"
    # ...and this check is NOT a replicate-only rule: `summarize_exp_b` applies it too, so the
    # full-cohort point refuses on the same grounds rather than reporting a three-session mean.
    assert robustness._exp_b_estimates(results, None, None)[1] == "expb_primary_aggregate_unavailable"


def test_a_non_finite_estimate_is_a_skip_not_a_reported_nan(config, tmp_path, monkeypatch):
    monkeypatch.setattr(
        robustness, "_run_procedure",
        lambda *a, **k: [_a_fold(s, [-1.0], [float("nan")], [-1.0]) for s in (1, 2, 3, 4, 5, 6)],
    )
    outcome = robustness.run_replicate(config, "a", "10ghz", _spine(6), tmp_path,
                                       config.run.seed_set, [1, 2, 3, 4, 5, 6], 1)
    assert outcome.skip_reason in ("non_finite_estimate", "missing_outer_prediction")


# ----------------------------------------------- T-M10-rob-estimand: the estimand definitions


def test_exp_a_estimands_at_multiplicity_one_equal_what_summarize_exp_a_reports(config):
    """The strongest available statement about the full-cohort point: it is not "close to" the
    reported Exp A number, it is the same number, produced by `np.mean` over the same array.

    `bootstrap_b` is shrunk only so the comparison's CI machinery is cheap — the POINT estimate
    a bootstrap reports is the statistic on the original subjects and does not depend on B.
    """
    results = [
        _a_fold(1, [-1.0, -2.0, -3.0], [-1.2, -1.7, -3.4], [-0.8, -2.2, -2.9]),
        _a_fold(2, [-0.5, -1.5], [-0.9, -1.1], [-0.6, -1.4]),
        _a_fold(3, [-2.0, -2.5, -4.0], [-1.5, -2.9, -3.1], [-1.9, -2.6, -3.8]),
        _a_fold(4, [-1.0], [-0.4], [-1.2]),
    ]
    cheap = dataclasses.replace(config, stats=dataclasses.replace(config.stats, bootstrap_b=50))
    reported = exp_a.summarize_exp_a(results, cheap)
    estimates, reason = robustness._exp_a_estimates(results, [1, 2, 3, 4], None)

    assert reason is None
    assert estimates["selected_radar_subject_balanced_mae"] == \
        reported["subject_balanced_mae"]["point"]
    assert estimates["radar_minus_session_index_mae"] == \
        reported["baseline_session_index_only"]["mean_difference_radar_minus_baseline"]["point"]


def test_exp_a_estimands_weight_each_distinct_subject_by_m_s():
    """§2.4: "outer replicate summaries weight each subject by m_s". Repeating rows cannot do
    it — the metric is subject-BALANCED, so a subject's own mean is unchanged by duplicating
    its sessions. Checked against hand arithmetic AND against the value a physically duplicated
    cohort of distinct subjects would give."""
    results = [
        _a_fold(1, [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]),      # radar MAE 1, baseline MAE 2
        _a_fold(2, [0.0], [5.0], [1.0]),                     # radar MAE 5, baseline MAE 1
        _a_fold(3, [0.0], [3.0], [3.0]),                     # radar MAE 3, baseline MAE 3
        _a_fold(4, [0.0], [7.0], [4.0]),                     # radar MAE 7, baseline MAE 4
    ]
    multiplicity = {1: 3, 2: 1, 3: 2, 4: 1}
    estimates, _ = robustness._exp_a_estimates(results, [1, 2, 3, 4], multiplicity)

    hand = (1 * 3 + 5 * 1 + 3 * 2 + 7 * 1) / 7
    assert estimates["selected_radar_subject_balanced_mae"] == pytest.approx(hand)
    # ...and it is exactly the plain mean over the explicitly duplicated subject list.
    duplicated = [1, 1, 1, 5, 3, 3, 7]
    assert estimates["selected_radar_subject_balanced_mae"] == pytest.approx(np.mean(duplicated))

    diff_hand = ((1 - 2) * 3 + (5 - 1) * 1 + (3 - 3) * 2 + (7 - 4) * 1) / 7
    assert estimates["radar_minus_session_index_mae"] == pytest.approx(diff_hand)


def test_exp_b_estimand_repeats_evaluation_rows_before_the_equal_session_aggregate():
    """Exp B's aggregate is POOLED (an average of per-session averages), so §2.4's general rule
    applies: repeat the rows by m_s, then compute the metric. Within a session the mean over
    rows becomes multiplicity-weighted; the equal weight PER SESSION is untouched."""
    results = [
        _b_fold(1, [1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
        _b_fold(2, [1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]),
        _b_fold(3, [1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]),
        _b_fold(4, [1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], [7.0, 7.0, 7.0, 7.0]),
    ]
    multiplicity = {1: 3, 2: 1, 3: 2, 4: 1}
    estimates, reason = robustness._exp_b_estimates(results, [1, 2, 3, 4], multiplicity)
    assert reason is None

    # y_true is 0 everywhere, so the baseline (residual 0) contributes 0 and each session's
    # radar MAE is the multiplicity-weighted mean of {1, 5, 3, 7} with weights {3, 1, 2, 1}.
    per_session = (1 * 3 + 5 * 1 + 3 * 2 + 7 * 1) / 7
    assert estimates["radar_minus_baseline_equal_session_aggregate"] == pytest.approx(per_session)


def test_exp_c_class_unit_mae_repeats_rows_before_the_pooled_metric():
    results = [
        _c_fold(1, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [1, 1, 2, 3, 4]),   # arm a exact, arm b off by 1
        _c_fold(2, [0, 1, 2, 3, 4], [1, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        _c_fold(3, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        _c_fold(4, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
    ]
    multiplicity = {1: 4, 2: 1, 3: 1, 4: 1}
    estimates, reason = robustness._exp_c_estimates(results, [1, 2, 3, 4], multiplicity)
    assert reason is None
    # Pooled over the repeated rows: 35 rows total, arm (a) errs on subject 2's single row.
    assert estimates["arm_a_class_unit_mae"] == pytest.approx(1 / 35)
    # arm (b) errs on subject 1's single row, and subject 1 was drawn four times.
    assert estimates["arm_b_class_unit_mae"] == pytest.approx(4 / 35)


def test_only_class_unit_mae_gets_a_refit_robustness_range():
    """§2.4: adjacent accuracy and QWK keep their existing conditional CIs and are given no
    refit-robustness range. The registered estimand list is where that is enforced."""
    assert robustness.ESTIMANDS["c"] == ("arm_a_class_unit_mae", "arm_b_class_unit_mae")
    assert not any("qwk" in e or "adjacent" in e
                   for es in robustness.ESTIMANDS.values() for e in es)


# ------------------------------------------------------- T-M10-rob-range: percentiles + status


def test_empirical_range_is_numpys_linear_quantile_and_nothing_else():
    """§2.4 pins the interpolation method as well as the probabilities. NumPy offers nine rules
    and they disagree materially at n = 200 (about five observations per tail), so the other
    plausible choices must NOT reproduce these endpoints."""
    values = np.arange(200, dtype=float)
    low, high = robustness.empirical_range(values)
    expected = np.quantile(values, [0.025, 0.975], method="linear")
    assert (low, high) == (float(expected[0]), float(expected[1]))
    for other in ("lower", "higher", "nearest"):
        assert (low, high) != tuple(np.quantile(values, [0.025, 0.975], method=other))


def test_the_range_is_computed_after_sorting_by_replicate_id(config):
    """The vector must be a deterministic function of the run, not of completion order in the
    worker pool — replicates come back from `fold_parallel` in whatever order they finish."""
    estimates = {r: float(r) for r in range(1, 21)}
    ordered = [_outcome(r, robustness.STATUS_OK, {"selected_radar_subject_balanced_mae": v,
                                                  "radar_minus_session_index_mae": -v})
               for r, v in estimates.items()]
    shuffled = [ordered[i] for i in (7, 3, 19, 0, 11, 2, 15, 5, 1, 9, 18, 4, 13, 6, 17, 8, 12, 10, 16, 14)]
    point = {"selected_radar_subject_balanced_mae": 1.0, "radar_minus_session_index_mae": 0.0}

    rows_a, _ = robustness.summarize(config, "a", "10ghz", ordered, point, replicates_requested=20)
    rows_b, _ = robustness.summarize(config, "a", "10ghz", shuffled, point, replicates_requested=20)
    assert rows_a == rows_b


def test_r8_is_inconclusive_because_min_successful_is_never_scaled(config):
    """Plan §4.2 step 3: "A smoke with R=8 must intentionally report inconclusive under the real
    min_successful=100; the threshold is never scaled." All eight replicates SUCCEED here — the
    inconclusive verdict comes from the rule, not from failure."""
    outcomes = [_outcome(r, robustness.STATUS_OK,
                         {"selected_radar_subject_balanced_mae": 1.0 + 0.1 * r,
                          "radar_minus_session_index_mae": -0.1 * r})
                for r in range(1, 9)]
    point = {"selected_radar_subject_balanced_mae": 1.4, "radar_minus_session_index_mae": -0.4}
    rows, skip_counts = robustness.summarize(config, "a", "10ghz", outcomes, point,
                                             replicates_requested=8)
    assert skip_counts == {}
    for row in rows:
        assert row["n_successful"] == 8
        assert row["min_successful_replicates"] == 100 == config.stats.robustness_min_successful_replicates
        assert row["status"] == robustness.INCONCLUSIVE
        assert row["r_requested"] == 8


def test_one_hundred_successes_is_the_boundary(config):
    """The rule is ">= min_successful", so 100 is conclusive and 99 is not — checked at the
    boundary rather than somewhere comfortably inside it."""
    point = {"selected_radar_subject_balanced_mae": 1.0, "radar_minus_session_index_mae": 0.0}
    for n_success, expected in ((99, robustness.INCONCLUSIVE), (100, robustness.CONCLUSIVE)):
        outcomes = [_outcome(r, robustness.STATUS_OK,
                             {"selected_radar_subject_balanced_mae": float(r),
                              "radar_minus_session_index_mae": -float(r)})
                    for r in range(1, n_success + 1)]
        outcomes += [_outcome(r, robustness.STATUS_SKIPPED,
                              skip_reason="insufficient_distinct_subjects")
                     for r in range(n_success + 1, 201)]
        rows, skip_counts = robustness.summarize(config, "a", "10ghz", outcomes, point,
                                                 replicates_requested=200)
        assert rows[0]["status"] == expected
        assert rows[0]["n_skipped"] == 200 - n_success
        assert skip_counts["insufficient_distinct_subjects"] == 200 - n_success


def test_the_range_is_never_labelled_bca(config):
    """A-M10-5. BCa needs the observed statistic and an original-unit jackknife; these endpoints
    are percentiles of an already-bootstrapped vector, so the label is load-bearing."""
    outcomes = [_outcome(r, robustness.STATUS_OK,
                         {"selected_radar_subject_balanced_mae": float(r),
                          "radar_minus_session_index_mae": -float(r)})
                for r in range(1, 11)]
    point = {"selected_radar_subject_balanced_mae": 5.0, "radar_minus_session_index_mae": -5.0}
    rows, _ = robustness.summarize(config, "a", "10ghz", outcomes, point, replicates_requested=10)
    for row in rows:
        assert row["range_label"] == row["ci_method"] == "selection_variance_empirical_95pct_range"
        assert "bca" not in row["ci_method"]
    assert robustness.RANGE_LABEL == "selection_variance_empirical_95pct_range"


# ------------------------------------------- T-M10-rob-audit: winner and fit-audit provenance


def test_a_held_out_subject_inside_a_fitted_set_is_a_hard_error_not_a_recorded_row():
    """The invariant the fit-audit table exists to police, checked while the subject set is
    still in hand rather than after it has been reduced to a hash."""
    leaking = _a_fold(1, [-1.0], [-1.1], [-0.9], fits=[
        FitRecord("scaler", "outer_train", frozenset({1, 2, 3}), {})
    ])
    with pytest.raises(RobustnessError, match="held-out subject 1"):
        robustness._fit_audit_rows("a", "10ghz", 3, [leaking], {1: 1, 2: 1, 3: 1})


def test_every_fit_audit_hash_resolves_in_the_companion_maps():
    """§3: "companion JSON stores canonical subject/multiplicity maps keyed by hash". A hash
    that resolves to nothing would make the table unauditable."""
    fits = [
        FitRecord("scaler", "outer_train", frozenset({2, 3}), {
            "multiplicity_subjects": np.array([2, 3], dtype=np.int64),
            "multiplicity_counts": np.array([3, 1], dtype=np.int64),
            "effective_weighted_row_count": np.array([8.0]),
            "weighting_mode": np.frombuffer(b"row_duplication", dtype=np.uint8),
        }),
        FitRecord("tuned_epsilon", "outer_train", frozenset({2, 3}), {}),
    ]
    rows, maps = robustness._fit_audit_rows(
        "a", "10ghz", 1, [_a_fold(1, [-1.0], [-1.1], [-0.9], fits=fits)], {1: 1, 2: 3, 3: 1}
    )
    assert len(rows) == 2 * len(robustness.ESTIMANDS["a"])
    for row in rows:
        assert row["fitted_subjects_sha256"] in maps["subject_sets"]
        assert row["multiplicity_sha256"] in maps["multiplicity_maps"]

    scaler = next(r for r in rows if r["quantity"] == "scaler")
    assert scaler["weighting_mode"] == "row_duplication"
    assert scaler["effective_weighted_row_count"] == 8.0
    assert maps["multiplicity_maps"][scaler["multiplicity_sha256"]] == [[2, 3], [3, 1]]

    # A quantity that consumes m_s WITHOUT duplicating rows (the tuned-eps median repeats
    # per-subject scales) says so, and still resolves to the draw restricted to its own
    # training subjects rather than to the whole replicate's map.
    eps = next(r for r in rows if r["quantity"] == "tuned_epsilon")
    assert eps["weighting_mode"] == "multiplicity_weighted"
    assert eps["effective_weighted_row_count"] == ""
    assert maps["multiplicity_maps"][eps["multiplicity_sha256"]] == [[2, 3], [3, 1]]


def test_an_unresampled_fit_records_weighting_mode_none():
    rows, _ = robustness._fit_audit_rows(
        "a", "10ghz", 0, [_a_fold(1, [-1.0], [-1.1], [-0.9], fits=[
            FitRecord("ridge", "outer_train", frozenset({2, 3}), {})
        ])], None,
    )
    assert {row["weighting_mode"] for row in rows} == {"none"}


def test_selection_rows_carry_each_estimand_and_reconstruct_the_feature_axes(config):
    """Rows are emitted once per ESTIMAND so each estimand's provenance is complete on its own
    rows and joins to `robustness_replicates.csv` on the full key. The axes are reconstructed
    through Exp A's own `active` builders, and `model_family` is stripped because Exp C arm (b)
    legitimately has none."""
    results = [_a_fold(s, [-1.0], [-1.1], [-0.9]) for s in (1, 2, 3, 4)]
    rows = robustness._selection_rows(config, "a", "10ghz", 5, results, {1: 2, 2: 1, 3: 1, 4: 1})

    assert len(rows) == 4 * len(robustness.ESTIMANDS["a"])
    assert {r["arm_or_contrast"] for r in rows} == set(robustness.ESTIMANDS["a"])
    for row in rows:
        axes = json.loads(row["active_axes_json"])
        assert "model_family" not in axes
        assert axes["band"] == "10ghz" and axes["log_branch"] == "off"
        assert row["selected"] is True
        assert json.loads(row["model_seeds_json"]) == list(config.run.seed_set)
        # A-M10-10: the winner-level table has no per-candidate inner score to record.
        assert row["inner_score"] == "" and row["inner_score_variance"] == ""


def test_exp_c_selection_rows_cover_stage_one_and_both_arms(config):
    results = [_c_fold(s, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) for s in (1, 2, 3, 4)]
    rows = robustness._selection_rows(config, "c", "10ghz", 1, results, {1: 1, 2: 1, 3: 1, 4: 1})
    stages = {(r["arm_or_contrast"], r["stage"]) for r in rows}
    assert stages == {
        ("arm_a_class_unit_mae", "stage1"), ("arm_a_class_unit_mae", "stage2_arm_a"),
        ("arm_b_class_unit_mae", "stage1"), ("arm_b_class_unit_mae", "stage2_arm_b"),
    }
    assert {r["n_inner_folds"] for r in rows} == {4}     # Exp C alone returns this


# -------------------------------------------------------- T-M10-rob-artifacts: the five files


def _fake_run(config, tmp_path, n_success=6, n_skip=2):
    outcomes = []
    for r in range(1, n_success + 1):
        # Each fold's fit names ITS OWN outer-training subjects — the held-out one excluded,
        # which `_fit_audit_rows` enforces and a shared subject set would violate.
        results = [
            _a_fold(s, [-1.0], [-1.0 - 0.01 * r], [-0.9], fits=[
                FitRecord("ridge", "outer_train", frozenset({1, 2, 3, 4}) - {s}, {})
            ])
            for s in (1, 2, 3, 4)
        ]
        multiplicity = {1: 2, 2: 1, 3: 1, 4: 1}
        estimates, _ = robustness._exp_a_estimates(results, [1, 2, 3, 4], multiplicity)
        fit_rows, maps = robustness._fit_audit_rows("a", "10ghz", r, results, multiplicity)
        outcomes.append(robustness.ReplicateOutcome(
            experiment="a", band="10ghz", replicate=r, seed_tuple=(20260721, 1, 10, r),
            generated_seed_state="ab" * 16, multiplicity=multiplicity, n_distinct_subjects=4,
            status=robustness.STATUS_OK, estimates=estimates,
            selection_rows=robustness._selection_rows(config, "a", "10ghz", r, results, multiplicity),
            fit_audit_rows=fit_rows, audit_maps=maps,
        ))
    for r in range(n_success + 1, n_success + n_skip + 1):
        outcomes.append(_outcome(r, robustness.STATUS_SKIPPED,
                                 skip_reason="insufficient_distinct_subjects"))
    point = {"selected_radar_subject_balanced_mae": 0.1, "radar_minus_session_index_mae": 0.0}
    rows, skip_counts = robustness.summarize(config, "a", "10ghz", outcomes, point,
                                             replicates_requested=n_success + n_skip)
    paths = robustness.write_robustness_reports(
        config, "a", "10ghz", outcomes, rows, skip_counts, point, tmp_path,
        replicates_requested=n_success + n_skip,
    )
    return outcomes, rows, paths


def test_write_robustness_reports_writes_the_five_artifacts_with_the_declared_columns(config, tmp_path):
    import csv

    _outcomes, _rows, paths = _fake_run(config, tmp_path)
    assert set(paths) == {"replicates", "selection", "fit_audit", "fit_audit_maps", "summary", "metrics"}
    for name, columns in (("replicates", robustness.REPLICATE_COLUMNS),
                          ("selection", robustness.SELECTION_COLUMNS),
                          ("fit_audit", robustness.FIT_AUDIT_COLUMNS),
                          ("summary", robustness.SUMMARY_COLUMNS)):
        with Path(paths[name]).open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            assert tuple(next(reader)) == columns

    metrics = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))
    assert metrics["range_label"] == robustness.RANGE_LABEL
    assert metrics["quantile_method"] == "linear"
    assert metrics["quantile_probabilities"] == [0.025, 0.975]
    assert metrics["robustness_seed"] == config.run.seed
    assert set(metrics["skip_reason_counts"]) == set(robustness.SKIP_REASONS)
    assert metrics["original_point_estimate"]["selected_radar_subject_balanced_mae"] == 0.1


def test_skipped_replicates_are_rows_not_omissions(config, tmp_path):
    """A table containing only the successes would read as if the procedure had always worked."""
    import csv

    _outcomes, _rows, paths = _fake_run(config, tmp_path, n_success=6, n_skip=2)
    with Path(paths["replicates"]).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8 * len(robustness.ESTIMANDS["a"])
    skipped = [r for r in rows if r["status"] == robustness.STATUS_SKIPPED]
    assert len(skipped) == 2 * len(robustness.ESTIMANDS["a"])
    for row in skipped:
        assert row["skip_reason"] == "insufficient_distinct_subjects"
        assert row["estimate"] == ""                    # no partial estimate is ever written


def test_every_replicate_row_records_its_seed_tuple_and_generated_state(config, tmp_path):
    import csv

    _outcomes, _rows, paths = _fake_run(config, tmp_path)
    with Path(paths["replicates"]).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tuple_ = json.loads(row["robustness_seed_tuple_json"])
            assert tuple_[:3] == [20260721, 1, 10]
            assert tuple_[3] == int(row["replicate"])
            assert len(row["generated_seed_state"]) == 32


# ------------------------------------------------------------------- T-M10-rob-e2e: real refits


def test_end_to_end_exp_a_replicates_refit_audit_clean_and_report_inconclusive(config, tmp_path):
    """The RUN half: two real replicates of the complete staged search on a synthetic store.

    Deliberately small (5 subjects, 1 seed, R=2) and deliberately ONE test rather than several,
    because the milestone-6 search space cannot be shrunk — the protocol-freeze guard rejects
    any deviation — so every replicate here pays for the real 113-candidate staged search on
    each of its folds. What it proves that the REDUCE half cannot: the draw really reaches the
    refits, the outer roles stay disjoint under multiplicity, every estimate resolves to
    complete winner and fit-audit rows, and the recorded effective row counts are what explicit
    duplication would give.
    """
    sessions = _spine(5)
    _write_store(tmp_path, sessions, config)

    outcomes, summary_rows, skip_counts = robustness.run_robustness(
        config, "a", "10ghz", sessions, tmp_path, seeds=config.run.seed_set, replicates=2,
    )
    robustness.assert_mechanism_ok(outcomes, sessions, "a")

    assert [o.replicate for o in outcomes] == [1, 2]
    successful = [o for o in outcomes if o.status == robustness.STATUS_OK]
    assert successful, f"both replicates skipped: {[o.skip_reason for o in outcomes]}"
    for outcome in successful:
        assert set(outcome.estimates) == set(robustness.ESTIMANDS["a"])
        assert all(math.isfinite(v) for v in outcome.estimates.values())
        # every estimate resolves to complete winner + fit-audit rows (plan §5.5)
        folds_with_selection = {r["outer_test_subject"] for r in outcome.selection_rows}
        folds_with_fits = {r["outer_test_subject"] for r in outcome.fit_audit_rows}
        assert folds_with_selection == folds_with_fits == set(outcome.multiplicity)
        assert {r["multiplicity_sha256"] for r in outcome.selection_rows} == \
            {robustness.multiplicity_sha256(outcome.multiplicity)}

        # §5.5: "effective row counts match explicit duplication". Each subject contributes 5
        # sessions here, so an outer fit sees exactly `sum(m_s over its train subjects) * 5`.
        weighted = [r for r in outcome.fit_audit_rows if r["weighting_mode"] == "row_duplication"]
        assert weighted, "a bootstrap fit must report row_duplication (A-M10-8)"
        for row in weighted:
            fitted = outcome.audit_maps["subject_sets"][row["fitted_subjects_sha256"]]
            expected = sum(outcome.multiplicity[s] for s in fitted) * 5
            assert row["effective_weighted_row_count"] == float(expected)

    # R = 2 with the frozen min_successful = 100: inconclusive, on purpose.
    for row in summary_rows:
        assert row["status"] == robustness.INCONCLUSIVE
        assert math.isfinite(row["original_point"])
    assert set(skip_counts).issubset(set(robustness.SKIP_REASONS))

    # --- the claim sharding rests on, on real refits rather than synthetic outcomes ---
    # Replicate 2 computed ALONE must equal replicate 2 computed as part of the range 1..2:
    # its cohort is a pure function of its own seed tuple, so the range it was run in cannot
    # reach it. This is what makes a contiguous replicate range a complete unit of work.
    alone = robustness.run_replicate_range(config, "a", "10ghz", sessions, tmp_path,
                                           seeds=config.run.seed_set, start=2, stop=2)
    assert len(alone) == 1
    reference = next(o for o in outcomes if o.replicate == 2)
    assert alone[0].multiplicity == reference.multiplicity
    assert alone[0].generated_seed_state == reference.generated_seed_state
    assert alone[0].status == reference.status
    assert alone[0].estimates == reference.estimates

    # ...and a real shard survives the JSON round-trip byte-for-byte, payloads included.
    lineage = robustness.shard_lineage(config, "a", "10ghz",
                                       robustness.spine_subjects("a", sessions), 2, "deadbeef")
    shard_dir = tmp_path / "shards"
    robustness.write_shard(shard_dir, lineage, 1, 1, [o for o in outcomes if o.replicate == 1])
    robustness.write_shard(shard_dir, lineage, 2, 2, alone)
    merged = robustness.read_shards(shard_dir, lineage)
    assert [o.replicate for o in merged] == [1, 2]
    for restored, original in zip(merged, outcomes, strict=True):
        assert restored.estimates == original.estimates
        assert restored.multiplicity == original.multiplicity
        assert restored.selection_rows == original.selection_rows
        assert restored.fit_audit_rows == original.fit_audit_rows
    robustness.assert_mechanism_ok(merged, sessions, "a")


def test_a_cohort_below_the_minimum_refuses_rather_than_skipping_every_replicate(config, tmp_path):
    with pytest.raises(RobustnessError, match="robustness_min_distinct_subjects"):
        robustness.run_robustness(config, "a", "10ghz", _spine(3), tmp_path,
                                  seeds=config.run.seed_set, replicates=2)


# ------------------------------------------------ T-M10-rob-shard: array shards and the merge


def _lineage(config, subjects=(1, 2, 3, 4, 5), replicates=8, commit="deadbeef"):
    return robustness.shard_lineage(config, "a", "10ghz", list(subjects), replicates, commit)


def _shard(tmp_path, config, start, stop, lineage=None):
    outcomes = [
        _outcome(r, robustness.STATUS_OK,
                 {"selected_radar_subject_balanced_mae": float(r),
                  "radar_minus_session_index_mae": -float(r)})
        for r in range(start, stop + 1)
    ]
    return robustness.write_shard(tmp_path, lineage or _lineage(config), start, stop, outcomes)


def test_a_shard_set_covering_1_to_r_merges_back_to_the_single_job_outcome_list(config, tmp_path):
    """The whole justification for sharding: a contiguous replicate range is a complete unit of
    work, because each replicate's cohort is a pure function of its own seed tuple. So the
    reassembled list must be exactly what one allocation would have produced."""
    for start, stop in ((1, 3), (4, 6), (7, 8)):
        _shard(tmp_path, config, start, stop)

    merged = robustness.read_shards(tmp_path, _lineage(config))
    assert [o.replicate for o in merged] == list(range(1, 9))
    assert [o.estimates["selected_radar_subject_balanced_mae"] for o in merged] == \
        [float(r) for r in range(1, 9)]
    # the JSON round-trip must not turn the int-keyed multiplicity map into a string-keyed one
    for outcome in merged:
        assert all(isinstance(s, int) for s in outcome.multiplicity)


def test_a_missing_shard_refuses_by_naming_the_uncovered_replicates(config, tmp_path):
    """A partial set is never summarized — the array's `--dependency=afterany` means the merge
    runs even when a task died, so this refusal is what turns a dead task into a visible error
    instead of a range over the replicates that happened to finish."""
    _shard(tmp_path, config, 1, 3)
    _shard(tmp_path, config, 7, 8)
    with pytest.raises(RobustnessError, match=r"do not cover replicates \[4, 5, 6\]"):
        robustness.read_shards(tmp_path, _lineage(config))


def test_overlapping_shard_ranges_refuse(config, tmp_path):
    """Two tasks that both computed replicate 3 would double-count it in the percentile vector."""
    _shard(tmp_path, config, 1, 4)
    _shard(tmp_path, config, 3, 8)
    with pytest.raises(RobustnessError, match="appears in two shards"):
        robustness.read_shards(tmp_path, _lineage(config))


@pytest.mark.parametrize("field,bad", [
    ("analysis_commit", "0123456"),
    ("config_hash", "not-the-same-config"),
    ("robustness_seed", 1),
    ("replicates_requested", 200),
    ("subjects_sha256", "a" * 64),
])
def test_a_shard_from_a_different_commit_config_seed_or_cohort_refuses(config, tmp_path, field, bad):
    """Mirrors `exp_b._validate_shard`'s fail-closed contract, field by field and by name.
    `subjects_sha256` is in the lineage precisely because two shards run against different
    cohorts would still agree on commit and config while drawing from different pools."""
    good = _lineage(config)
    _shard(tmp_path, config, 1, 4, good)
    _shard(tmp_path, config, 5, 8, {**good, field: bad})
    with pytest.raises(RobustnessError, match=field):
        robustness.read_shards(tmp_path, good)


def test_a_replicate_outside_its_own_shards_declared_range_refuses(config, tmp_path):
    """The declared range is what the gap/overlap check reasons about, so a shard whose
    contents disagree with its own header would make that reasoning meaningless."""
    lineage = _lineage(config)
    outcomes = [_outcome(r, robustness.STATUS_OK,
                         {"selected_radar_subject_balanced_mae": 1.0,
                          "radar_minus_session_index_mae": 0.0})
                for r in (1, 2, 7)]
    robustness.write_shard(tmp_path, lineage, 1, 4, outcomes)
    _shard(tmp_path, config, 5, 8, lineage)
    with pytest.raises(RobustnessError, match="outside its own"):
        robustness.read_shards(tmp_path, lineage)


def test_an_empty_shard_directory_refuses_rather_than_reporting_zero_replicates(config, tmp_path):
    with pytest.raises(RobustnessError, match="no robustness shards"):
        robustness.read_shards(tmp_path, _lineage(config))


def test_a_shard_range_past_r_is_refused_before_any_fitting(config, tmp_path):
    with pytest.raises(RobustnessError, match="runs past R"):
        robustness.run_and_report_shard(
            config, "a", "10ghz", _spine(5), tmp_path, tmp_path / "shards",
            analysis_commit="deadbeef", replicates=8, start=7, stop=12,
        )


def test_shard_filenames_sort_into_replicate_order():
    """Zero-padded so a plain directory listing is also replicate order — a merge that read
    shard 100 before shard 20 would still be correct (it sorts by replicate), but the log and
    the directory would read as if it had not."""
    names = [robustness.shard_filename("a", "10ghz", s, s + 9) for s in (1, 11, 91, 101, 191)]
    assert names == sorted(names)
