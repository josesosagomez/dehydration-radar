"""T-M10-F: Experiment F — the not-estimable HR record (A-M10-2) and the covariate sensitivity.

Structured as plan §5.3 asks: the inventory that makes "no heart rate" evidence rather than a
claim, the exact config-to-workbook map and covariate order/units, the per-inner-fold scaler and
subject-balanced scoring, the frozen alpha tie-break, the variant/target differences, the
approved-source gate, and the no-silent-complete-case-drop rule. Plus the two structural
properties the design rests on: models 1/2 never read the store, and models 3/4 share a
byte-identical radar block within a fold.

Fast by construction: a 6-path synthetic store, a stub ground-truth table (no .xlsx), and a
reduced alpha grid wherever the grid itself is not what is under test.
"""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dehyd.config import load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_a, exp_b, exp_c, exp_d, exp_f
from dehyd.eval.exp_f import ExpFError
from dehyd.eval.reference_gate import ReferenceGateError
from dehyd.eval.splits import nested_loso_splits
from dehyd.features.pooling import aggregate_session, pool_stats_batch
from dehyd.features.store import (
    order_key,
    prelog_key,
    raw_key,
    vec_key,
    write_session_store,
)
from dehyd.features.wst import apply_order_log
from dehyd.models.regressors import build_estimator, fit_pipeline

P, T, CN, NFR = 6, 4, 1, 3
ORDER = np.array([0, 1, 1, 2, 2, 2])
N_SUBJECTS = 8
FEATURE_KEY = (0, "A", "mag", 0, "off")


@pytest.fixture(scope="module")
def config():
    return load_config("configs/exp_a_regression.yaml", "configs/exp_f.yaml", "configs/stats.yaml")


# ------------------------------------------------------------------------------- fixtures


def _sessions(n_subjects=N_SUBJECTS):
    """An S0-S4 spine — F keeps S0, unlike Exp B."""
    out = []
    for s in range(1, n_subjects + 1):
        baseline = 70.0 + s
        for i in range(5):
            mass = baseline - 0.25 * i
            out.append({
                "subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
                "rel_path": f"s{s}_{SESSION_NAMES[i]}.mat", "frame_ids": [0, 1, 2],
                "delta_m_pct": (mass - baseline) / baseline * 100.0,
                "delta_m_kg": mass - baseline,
            })
    return out


def _gt(n_subjects=N_SUBJECTS, overrides=None):
    """A stub ground truth: only `.subjects` is ever read, so the tests need no workbook.

    `overrides` is {subject: {column: value}} — an explicit parameter rather than **kwargs,
    because a **kwargs version silently swallowed `overrides={...}` as a column name and left
    the missing-covariate guard untested.
    """
    overrides = overrides or {}
    rows = []
    for s in range(1, n_subjects + 1):
        row = {"subject": s, "age": 20.0 + s, "height_cm": 165.0 + s,
               "baseline_mass_kg": 70.0 + s, "bmi": (70.0 + s) / ((1.65 + 0.01 * s) ** 2)}
        row.update(overrides.get(s, {}))
        rows.append(row)
    return SimpleNamespace(subjects=pd.DataFrame(rows))


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
                        off = aggregate_session(pool_stats_batch(
                            apply_order_log(raw, meta, wst, log_on=False), meta))
                        frozen = aggregate_session(pool_stats_batch(
                            apply_order_log(raw, meta, wst, log_on=True, epsilon_by_order=eps), meta))
                        npz[vec_key(gi, r, c, ti, "off")] = off
                        npz[vec_key(gi, r, c, ti, "frozen")] = frozen
                        npz[raw_key(gi, r, c, ti)] = raw
                        npz[prelog_key(gi, r, c, ti)] = np.array([raw.mean()] * 3)
        write_session_store("10ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR},
                            store_dir)


def _source(n_subjects=N_SUBJECTS, feature_key=FEATURE_KEY):
    return exp_f.ExpASource(
        run_path="/approved/exp_a_run",
        selection_sha256="a" * 64,
        feature_key_by_subject={s: feature_key for s in range(1, n_subjects + 1)},
    )


@pytest.fixture(scope="module")
def e2e(tmp_path_factory, config):
    """ONE complete Exp F run on a synthetic store. Returns (results, sessions, store_dir, gt)."""
    store_dir = tmp_path_factory.mktemp("store_f10")
    sessions = _sessions()
    gt = _gt()
    _write_store(store_dir, sessions, config)
    inputs = exp_f.build_design_inputs(config, gt, sessions)
    results = exp_f.run_exp_f(config, "10ghz", sessions, store_dir, _source(), inputs, n_workers=1)
    return results, sessions, store_dir, gt


# ------------------------------------------------- the unavailable heart-rate question


def test_the_inventory_reports_zero_hr_observations_and_the_exact_status(config):
    rows = exp_f.hr_inventory(config, _gt())
    hr = next(r for r in rows if r["variable"] == "heart_rate")

    assert hr["availability"] == "missing"
    assert hr["n_values"] == 0
    assert hr["observation_unit"] == "not_applicable"
    # the evidence is an ACTUAL inspection: the workbook's own column labels and the data roots
    assert "weight" in hr["source_checked"].lower() or ".xlsx" in hr["source_checked"]
    assert "data_10ghz_dir" in hr["source_checked"]
    assert exp_f.HR_STATUS in hr["reason"]

    summary = exp_f.hr_summary(rows)
    assert summary["status"] == "not_estimable_missing_heart_rate"
    assert summary["n_hr_observations"] == 0
    assert summary["uncontrolled_variables"] == ["glucose", "heart_rate", "temperature"]
    assert "NOT an HR adjustment" in summary["note"]


def test_temperature_and_glucose_are_recorded_as_uncontrolled_not_adjusted(config):
    rows = {r["variable"]: r for r in exp_f.hr_inventory(config, _gt())}
    for variable in ("temperature", "glucose"):
        assert rows[variable]["availability"] == "missing"
        assert rows[variable]["n_values"] == 0
        assert "uncontrolled" in rows[variable]["reason"]


def test_the_static_covariates_are_recorded_as_available_and_not_as_hr(config):
    rows = {r["variable"]: r for r in exp_f.hr_inventory(config, _gt())}
    for name in exp_f.COVARIATE_COLUMN_MAP:
        row = rows[name]
        assert row["availability"] == "available"
        assert row["observation_unit"] == "per_subject"
        assert row["n_values"] == N_SUBJECTS
        assert "NOT a heart-rate adjustment" in row["reason"]


# ------------------------------------------------- covariates: map, order, units, missingness


def test_the_config_to_workbook_map_is_exact():
    """§2.2 writes this map out in full; a wrong column would silently model the wrong thing."""
    assert exp_f.COVARIATE_COLUMN_MAP == {
        "age": "age", "height": "height_cm", "baseline_mass": "baseline_mass_kg", "bmi": "bmi",
    }


def test_the_covariate_block_follows_the_frozen_order_and_units(config):
    gt = _gt()
    subjects = np.array([1, 2, 3])
    block = exp_f.covariate_block(gt, subjects, config.exp_f.covariates_primary)

    assert config.exp_f.covariates_primary == ("age", "height", "baseline_mass", "bmi")
    assert block.shape == (3, 4)
    table = gt.subjects.set_index("subject")
    for row, subject in enumerate(subjects):
        assert block[row, 0] == table.loc[subject, "age"]
        assert block[row, 1] == table.loc[subject, "height_cm"]      # cm, not m
        assert block[row, 2] == table.loc[subject, "baseline_mass_kg"]   # kg
        assert block[row, 3] == pytest.approx(table.loc[subject, "bmi"])


def test_the_sensitivity_block_is_age_and_height_only(config):
    assert config.exp_f.covariates_sensitivity == ("age", "height")
    block = exp_f.covariate_block(_gt(), np.array([1, 2]), config.exp_f.covariates_sensitivity)
    assert block.shape == (2, 2)


def test_a_missing_or_non_finite_covariate_fails_naming_the_subject_and_column(config):
    """§5.3: no silent complete-case drop. Dropping a subject would change the cohort between
    variants and quietly un-pair the contrasts."""
    gt = _gt(overrides={4: {"bmi": float("nan")}})
    with pytest.raises(ExpFError, match=r"subject 4 .*'bmi'"):
        exp_f.covariate_block(gt, np.array([3, 4]), config.exp_f.covariates_primary)

    with pytest.raises(ExpFError, match="subject 99 has no workbook covariate record"):
        exp_f.covariate_block(_gt(), np.array([99]), config.exp_f.covariates_primary)


def test_the_clock_is_a_one_hot_over_the_fixed_s0_to_s4_domain():
    clock = exp_f.clock_one_hot([0, 1, 4, 4])
    assert clock.shape == (4, 5)
    assert clock.sum(axis=1).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert clock[2].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    # a fold missing a session leaves an all-zero column rather than changing the design width
    narrow = exp_f.clock_one_hot([1, 1])
    assert narrow.shape == (2, 5) and narrow[:, 0].sum() == 0.0
    with pytest.raises(ExpFError, match="outside the frozen clock domain"):
        exp_f.clock_one_hot([0, 5])


# ------------------------------------------------------------- fold-local alpha selection


def _fold(subjects, test_subject):
    return next(f for f in nested_loso_splits(subjects) if f.test_subject == test_subject)


def test_every_inner_alpha_fit_has_its_own_inner_train_scaler_and_subject_balanced_score(config):
    """Recomputed by hand: the reported inner score must be the mean over inner folds of a
    subject-balanced MAE from a pipeline fit on that inner fold's TRAINING rows only."""
    rng = np.random.default_rng(0)
    subjects = np.repeat(np.arange(1, 9), 3)
    X = rng.normal(size=(subjects.size, 4))
    y = rng.normal(size=subjects.size)
    fold = _fold(sorted(set(subjects.tolist())), 1)
    train = subjects != 1

    alphas = (0.1, 1.0)
    choice = exp_f.select_alpha(X[train], y[train], subjects[train], fold, alphas, seed=7)

    expected = {}
    for alpha in alphas:
        per_fold = []
        for inner in fold.inner_folds:
            inner_train = np.isin(subjects[train], sorted(inner.train_subjects))
            inner_val = np.isin(subjects[train], sorted(inner.val_subjects))
            pipeline = build_estimator("ridge", {"alpha": alpha}, seed=7)
            fit_pipeline(pipeline, X[train][inner_train], y[train][inner_train])
            from dehyd.eval.metrics import subject_balanced_mae
            per_fold.append(subject_balanced_mae(
                subjects[train][inner_val], y[train][inner_val],
                pipeline.predict(X[train][inner_val])))
            # the scaler saw the inner-training rows and nothing else
            assert pipeline.named_steps["scaler"].mean_ == pytest.approx(
                X[train][inner_train].mean(axis=0))
        expected[alpha] = (float(np.mean(per_fold)), float(np.std(per_fold, ddof=0)))

    best = min(expected, key=lambda a: expected[a][0])
    assert choice.alpha == best
    assert choice.inner_score == pytest.approx(expected[best][0])
    assert choice.inner_score_variance == pytest.approx(expected[best][1])
    assert choice.n_inner_folds == len(fold.inner_folds)


def test_a_full_alpha_tie_resolves_to_the_first_frozen_alpha(config):
    """`select_candidate` is a stable min over INPUT order, and F iterates the frozen
    `ExpFConfig.ridge_alphas` tuple — so a design where alpha cannot matter picks alphas[0]."""
    subjects = np.repeat(np.arange(1, 9), 2)
    # an all-zero design: every alpha gives the identical (intercept-only) model, a full tie
    X = np.zeros((subjects.size, 3))
    y = np.linspace(-1.0, 1.0, subjects.size)
    fold = _fold(sorted(set(subjects.tolist())), 1)
    train = subjects != 1

    choice = exp_f.select_alpha(X[train], y[train], subjects[train], fold,
                                config.exp_f.ridge_alphas, seed=1)
    assert choice.alpha == config.exp_f.ridge_alphas[0] == 0.001


def test_the_alpha_grid_is_the_frozen_ordered_tuple(config):
    assert config.exp_f.ridge_alphas == (0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


# ------------------------------------------------------- the four models and three variants


def test_all_four_models_and_three_variants_are_produced_per_fold(e2e):
    results, sessions, _, _ = e2e
    rows = [row for r in results for row in r.prediction_rows]
    n_folds = sum(1 for r in results if r.reason is None)

    assert {r["model_id"] for r in rows} == set(exp_f.MODEL_IDS)
    assert {r["analysis_variant"] for r in rows} == set(exp_f.VARIANTS)
    # 4 models x 3 variants x 5 sessions per held-out subject
    assert len(rows) == n_folds * 4 * 3 * 5
    assert exp_f.MODEL_NUMBER == {
        "clock": 1, "clock_covariates": 2, "clock_radar": 3, "clock_radar_covariates": 4}


def test_there_is_no_combined_kg_reduced_variant():
    """§2.2 freezes exactly three non-factorial variants; a fourth would be a plan amendment."""
    assert exp_f.VARIANTS == ("pct_full", "pct_reduced", "kg_full")


def test_pct_reduced_reuses_pct_full_s_covariate_free_models_byte_identically(e2e):
    """§2.2: models 1 and 3 contain no covariate block, so the reduced variant reuses them
    rather than refitting an identical model that could differ by floating-point accident."""
    results, _, _, _ = e2e
    rows = [row for r in results for row in r.prediction_rows]
    for model_id in ("clock", "clock_radar"):
        full = [(r["subject"], r["session_idx"], r["y_pred"]) for r in rows
                if r["analysis_variant"] == "pct_full" and r["model_id"] == model_id]
        reduced = [(r["subject"], r["session_idx"], r["y_pred"]) for r in rows
                   if r["analysis_variant"] == "pct_reduced" and r["model_id"] == model_id]
        assert full == reduced and full

    selections = [row for r in results for row in r.selection_rows]
    for model_id in ("clock", "clock_radar"):
        alphas = {v: [s["selected_alpha"] for s in selections
                      if s["analysis_variant"] == v and s["model_id"] == model_id]
                  for v in ("pct_full", "pct_reduced")}
        assert alphas["pct_full"] == alphas["pct_reduced"]


def test_the_covariate_models_differ_between_pct_full_and_pct_reduced(e2e):
    """The power companion to the reuse test: the variants must not be identical everywhere,
    or the reuse assertion above would pass trivially."""
    results, _, _, _ = e2e
    rows = [row for r in results for row in r.prediction_rows]
    full = [r["y_pred"] for r in rows
            if r["analysis_variant"] == "pct_full" and r["model_id"] == "clock_covariates"]
    reduced = [r["y_pred"] for r in rows
               if r["analysis_variant"] == "pct_reduced" and r["model_id"] == "clock_covariates"]
    assert full != reduced


def test_kg_full_uses_the_signed_kg_target_and_keeps_the_percentage_feature_key(e2e):
    """The kg variant changes the TARGET only: its radar block still carries the pre-specified
    percentage-target Exp-A feature key, because radar features are not reselected on kg."""
    results, sessions, _, _ = e2e
    rows = [row for r in results for row in r.prediction_rows]
    kg = [r for r in rows if r["analysis_variant"] == "kg_full"]
    assert {r["target_name"] for r in kg} == {"delta_m_kg"}
    assert {r["target_name"] for r in rows if r["analysis_variant"] != "kg_full"} == {"delta_m_pct"}

    truth = {(s["subject"], s["session_idx"]): s["delta_m_kg"] for s in sessions}
    for row in kg:
        assert row["y_true"] == pytest.approx(truth[(row["subject"], row["session_idx"])])
        assert row["y_true"] <= 0.0        # signed: loss stays negative

    selections = [row for r in results for row in r.selection_rows]
    keys = {s["feature_key"] for s in selections
            if s["analysis_variant"] == "kg_full" and s["model_id"] in exp_f.RADAR_MODELS}
    assert keys == {repr(FEATURE_KEY)}


def test_models_three_and_four_share_a_byte_identical_radar_block_within_a_fold(config, e2e):
    """§2.2. Recomputed here from the provider: the radar block is built once per fold, so the
    two radar models cannot be standing on different matrices."""
    _, sessions, store_dir, gt = e2e
    fold = _fold(exp_f.evaluable_subjects_f(sessions), 1)
    candidate = exp_f.radar_candidate(config, "10ghz", FEATURE_KEY)
    provider = exp_a.StoreBackedFeatures("10ghz", sessions, store_dir, config)
    radar = provider.data_for(candidate, fold.train_subjects).X

    inputs = exp_f.build_design_inputs(config, gt, sessions)
    designs = exp_f._designs_for_variant(
        "pct_full", inputs.clock, radar, inputs.covariates_full, inputs.covariates_reduced)
    n_clock = inputs.clock.shape[1]
    block_3 = designs["clock_radar"][:, n_clock:]
    block_4 = designs["clock_radar_covariates"][:, n_clock:n_clock + radar.shape[1]]
    assert np.array_equal(block_3, block_4)
    assert np.array_equal(block_3, radar)


def test_models_one_and_two_never_read_the_store(config, e2e):
    """A structural check, not a promise: the covariate-free-of-radar designs are built from
    clock and covariates alone, so no store access can be hiding in them."""
    _, sessions, _, gt = e2e
    inputs = exp_f.build_design_inputs(config, gt, sessions)
    sentinel = np.full((len(sessions), 3), np.nan)   # a radar block that would poison any use
    designs = exp_f._designs_for_variant(
        "pct_full", inputs.clock, sentinel, inputs.covariates_full, inputs.covariates_reduced)

    assert np.all(np.isfinite(designs["clock"]))
    assert np.all(np.isfinite(designs["clock_covariates"]))
    assert designs["clock"].shape[1] == inputs.clock.shape[1]
    assert designs["clock_covariates"].shape[1] == inputs.clock.shape[1] + 4
    # ... and the radar models DO depend on it, so the sentinel is a real probe
    assert not np.all(np.isfinite(designs["clock_radar"]))


# ------------------------------------------------------------------------ leakage / folds


def test_every_reported_fold_is_an_outer_loso_fold_predicting_only_its_held_out_subject(e2e):
    results, sessions, _, _ = e2e
    folds = {f.test_subject: f for f in nested_loso_splits(exp_f.evaluable_subjects_f(sessions))}
    for result in results:
        assert result.test_subject in folds
        assert result.test_subject not in folds[result.test_subject].train_subjects
        assert {int(r["subject"]) for r in result.prediction_rows} == {result.test_subject}
    exp_f._assert_mechanism_ok_f(results, sessions)


def test_mutating_the_held_out_subject_leaves_every_train_derived_fit_unchanged(
        tmp_path, config):
    """§5.1's mutation property in F's shape: the held-out subject's target may move its own
    score, but must not move any selected alpha or any other fold's fitted model."""
    sessions = _sessions(n_subjects=6)
    gt = _gt(6)
    _write_store(tmp_path, sessions, config)
    source = _source(6)

    before = exp_f.run_exp_f(config, "10ghz", sessions, tmp_path, source,
                            exp_f.build_design_inputs(config, gt, sessions), n_workers=1)
    victim = before[0].test_subject
    moved = [dict(s, delta_m_pct=s["delta_m_pct"] - 9.0) if s["subject"] == victim else dict(s)
             for s in sessions]
    after = exp_f.run_exp_f(config, "10ghz", moved, tmp_path, source,
                            exp_f.build_design_inputs(config, gt, moved), n_workers=1)

    victim_before = next(r for r in before if r.test_subject == victim)
    victim_after = next(r for r in after if r.test_subject == victim)
    # its own fold trains on the OTHER subjects -> identical alphas and identical predictions
    assert victim_after.selection_rows == victim_before.selection_rows
    pct = [(r["model_id"], r["y_pred"]) for r in victim_before.prediction_rows
           if r["analysis_variant"] == "pct_full"]
    pct_after = [(r["model_id"], r["y_pred"]) for r in victim_after.prediction_rows
                 if r["analysis_variant"] == "pct_full"]
    assert pct == pct_after
    # ... and the y_true it is scored against DID move, so the fixture perturbed something
    assert [r["y_true"] for r in victim_after.prediction_rows] != [
        r["y_true"] for r in victim_before.prediction_rows]


# --------------------------------------------------------------- the approved Exp-A source


def _sources_payload(tmp_path, band="10ghz", status="approved", table=None):
    table = table or (tmp_path / "selection_table_10ghz.csv")
    if not table.exists():
        table.write_text(
            "test_subject,feature_key,family,params\n"
            + "".join(f"{s},\"(0, 'A', 'mag', 0, 'off')\",ridge,\"{{'alpha': 1.0}}\"\n"
                      for s in range(1, N_SUBJECTS + 1)),
            encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    payload = {
        "schema_version": "exp_a_sources_v1",
        "status": status,
        "bands": {band: {
            "band": band,
            "status": status,
            "mismatched_evidence_classes": [] if status == "approved" else ["selection"],
            "final_run": {"path": str(tmp_path)},
            "final_selection_table": {"path": str(table), "sha256": digest},
        }},
    }
    path = tmp_path / "exp_a_sources.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, table


def test_an_approved_source_yields_the_per_fold_feature_keys(tmp_path):
    path, table = _sources_payload(tmp_path)
    source = exp_f.load_exp_a_source(path, "10ghz")
    assert source.feature_key_by_subject[3] == FEATURE_KEY
    assert len(source.feature_key_by_subject) == N_SUBJECTS
    assert source.selection_sha256 == source.selection_sha256.lower()
    assert len(source.selection_sha256) == 64


def test_a_not_approved_band_is_refused_before_any_fit(tmp_path):
    path, _ = _sources_payload(tmp_path, status="not_approved")
    with pytest.raises(ReferenceGateError, match="refuses to consume"):
        exp_f.load_exp_a_source(path, "10ghz")


def test_a_selection_table_that_changed_after_approval_is_refused(tmp_path):
    """The approval is evidence about specific BYTES. Re-hashing on read is what ties the file
    F actually parsed to the bytes the gate approved."""
    path, table = _sources_payload(tmp_path)
    table.write_text(table.read_text(encoding="utf-8") + "9,\"(0, 'A', 'mag', 0, 'off')\",ridge,\"{'alpha': 1.0}\"\n",
                     encoding="utf-8")
    with pytest.raises(ExpFError, match="changed after approval"):
        exp_f.load_exp_a_source(path, "10ghz")


def test_a_missing_selection_table_is_refused(tmp_path):
    path, table = _sources_payload(tmp_path)
    table.unlink()
    with pytest.raises(ExpFError, match="missing on disk"):
        exp_f.load_exp_a_source(path, "10ghz")


def test_a_fold_with_no_approved_feature_key_stops_the_run(tmp_path, config):
    """F cannot invent a feature key for a fold the approved table does not cover."""
    sessions = _sessions(n_subjects=6)
    _write_store(tmp_path, sessions, config)
    partial = exp_f.ExpASource(run_path="/approved", selection_sha256="b" * 64,
                               feature_key_by_subject={1: FEATURE_KEY})
    with pytest.raises(ExpFError, match="cannot invent a feature key"):
        exp_f.run_exp_f(config, "10ghz", sessions, tmp_path, partial,
                        exp_f.build_design_inputs(config, _gt(6), sessions), n_workers=1)


def test_the_radar_candidate_carries_the_key_but_not_the_exp_a_estimator(config):
    """§2.2: models 3/4 reuse the selected FEATURE KEY, never an Exp-A fitted estimator."""
    candidate = exp_f.radar_candidate(config, "10ghz", FEATURE_KEY)
    assert candidate.feature_key == FEATURE_KEY
    assert candidate.model_params == ()          # no Exp-A alpha rides along
    assert dict(candidate.active)["model_family"] == "ridge"


# ---------------------------------------------------------------------------- contrasts


def test_the_four_contrasts_are_the_frozen_nested_differences():
    assert exp_f.CONTRASTS == (
        ("radar_given_clock", "clock_radar", "clock"),
        ("radar_given_clock_covariates", "clock_radar_covariates", "clock_covariates"),
        ("covariates_given_clock", "clock_covariates", "clock"),
        ("covariates_given_clock_radar", "clock_radar_covariates", "clock_radar"),
    )
    assert exp_f.PRIMARY_CONTRASTS == ("radar_given_clock", "radar_given_clock_covariates")


def test_contrast_direction_is_with_minus_without(e2e):
    results, _, _, _ = e2e
    rows = [row for r in results for row in r.prediction_rows]
    contrasts = exp_f.contrast_rows(rows)
    maes = exp_f.per_subject_mae(rows)
    for row in contrasts:
        contrast = next(c for c in exp_f.CONTRASTS if c[0] == row["contrast_id"])
        _, with_model, without_model = contrast
        assert row["mae_with"] == pytest.approx(maes[(row["analysis_variant"], with_model, row["subject"])][0])
        assert row["mae_without"] == pytest.approx(maes[(row["analysis_variant"], without_model, row["subject"])][0])
        assert row["difference_with_minus_without"] == pytest.approx(
            row["mae_with"] - row["mae_without"])


def test_only_pct_full_radar_increments_are_primary_and_they_carry_holm_2(e2e, config):
    results, sessions, _, _ = e2e
    summary = exp_f.summarize_exp_f(
        results, exp_f.hr_inventory(config, _gt()), _source(), config, "10ghz", sessions)
    by_key = {(c["analysis_variant"], c["contrast_id"]): c for c in summary["contrasts"]}

    for contrast_id in exp_f.PRIMARY_CONTRASTS:
        record = by_key[("pct_full", contrast_id)]
        assert record["multiplicity_family"] == exp_f.FAMILY_PRIMARY
        assert np.isfinite(record["p_value_holm"])
        assert record["p_value_holm"] >= record["p_value_unadjusted"]

    for contrast_id in exp_f.EXPLORATORY_CONTRASTS:
        record = by_key[("pct_full", contrast_id)]
        assert record["multiplicity_family"] == config.stats.expf_exploratory_correction
        assert np.isnan(record["p_value_holm"])          # reported individually, uncorrected

    for variant in ("pct_reduced", "kg_full"):
        for contrast_id, _, _ in exp_f.CONTRASTS:
            record = by_key[(variant, contrast_id)]
            assert record["multiplicity_family"] == exp_f.FAMILY_SENSITIVITY
            assert np.isnan(record["p_value_holm"])      # never a second Holm family

    assert summary["available_analysis"]["primary_family_size"] == 2


def test_every_contrast_reports_ci_wilcoxon_and_the_pairing_counts(e2e, config):
    results, sessions, _, _ = e2e
    summary = exp_f.summarize_exp_f(
        results, exp_f.hr_inventory(config, _gt()), _source(), config, "10ghz", sessions)
    n_folds = sum(1 for r in results if r.reason is None)
    for record in summary["contrasts"]:
        assert record["n_paired_subjects"] == n_folds
        assert record["n_ties"] == record["n_paired_subjects"] - record["n_nonzero_pairs"]
        for field in ("mean_difference", "median_difference", "ci_low", "ci_high",
                      "wilcoxon_statistic", "p_value_unadjusted"):
            assert field in record
        assert record["ci_method"] in ("bca", "percentile")


def test_the_rng_offsets_are_distinct_within_f_and_against_the_other_experiments():
    offsets = exp_f._all_rng_offsets()
    assert len(offsets) == len(set(offsets)) == len(exp_f.VARIANTS) * len(exp_f.CONTRASTS)
    others = set(exp_b._all_rng_offsets()) | set(exp_c._all_rng_offsets()) | {0, 1, 2, 3}
    assert not set(offsets) & others
    assert min(offsets) >= exp_f.RNG_OFFSET_EXPF_BASE
    assert exp_f.RNG_OFFSET_EXPF_BASE > exp_d.RNG_OFFSET_EXPD_BASE


# --------------------------------------------------------------------------- artifacts


@pytest.fixture(scope="module")
def written(e2e, config, tmp_path_factory):
    results, sessions, _, _ = e2e
    availability = exp_f.hr_inventory(config, _gt())
    summary = exp_f.summarize_exp_f(results, availability, _source(), config, "10ghz", sessions)
    out = tmp_path_factory.mktemp("exp_f_reports")
    return exp_f.write_exp_f_reports(results, availability, summary, out, "10ghz"), out, summary


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_five_artifact_rows_are_written_with_the_exact_column_lists(written):
    paths, out, _ = written
    for name in ("confound_availability.csv", "predictions_f_10ghz.csv", "selection_f_10ghz.csv",
                 "contrasts_f_10ghz.csv", "exclusions_f_10ghz.csv", "metrics_exp_f_10ghz.json"):
        assert (out / name).exists(), name
    assert list(_read(paths["availability"])[0]) == list(exp_f.AVAILABILITY_COLUMNS)
    assert list(_read(paths["predictions"])[0]) == list(exp_f.PREDICTIONS_COLUMNS)
    assert list(_read(paths["selection"])[0]) == list(exp_f.SELECTION_COLUMNS)
    assert list(_read(paths["contrasts"])[0]) == list(exp_f.CONTRASTS_COLUMNS)


def test_the_selection_table_records_the_approved_source_on_every_row(written):
    paths, _, _ = written
    rows = _read(paths["selection"])
    assert {r["source_exp_a_final_run"] for r in rows} == {"/approved/exp_a_run"}
    assert {r["source_selection_sha256"] for r in rows} == {"a" * 64}
    # only the radar models carry a feature key; the clock/covariate models have none to carry
    for row in rows:
        if row["model_id"] in exp_f.RADAR_MODELS:
            assert row["feature_key"] == repr(FEATURE_KEY)
        else:
            assert row["feature_key"] == ""


def test_the_deterministic_ridge_uses_the_first_configured_seed_label_once(written, config):
    paths, _, _ = written
    seeds = {r["seed"] for r in _read(paths["predictions"])}
    assert seeds == {str(config.run.seed_set[0])}


def test_the_metrics_json_states_the_unanswerable_question_before_the_answerable_one(written):
    _, _, summary = written
    assert summary["heart_rate_question"]["status"] == exp_f.HR_STATUS
    assert summary["heart_rate_question"]["n_hr_observations"] == 0
    assert summary["available_analysis"]["status"] == "clock_and_static_covariate_sensitivity_only"
    assert "does not address heart rate" in summary["available_analysis"]["limitation"]
    assert "uncontrolled" in summary["available_analysis"]["limitation"]
    # the key order in the emitted JSON puts the HR record ahead of the analysis
    keys = list(summary)
    assert keys.index("heart_rate_question") < keys.index("available_analysis")
    assert keys.index("heart_rate_question") < keys.index("contrasts")


def test_the_smoke_writes_the_hr_inventory_but_no_contrast_value(tmp_path, config, monkeypatch):
    """The inventory is the registered ANSWER, not a performance value, so it survives the
    mechanism-only boundary that suppresses everything else."""
    sessions = _sessions(n_subjects=6)
    _write_store(tmp_path, sessions, config)
    sources_path, _ = _sources_payload(tmp_path)
    monkeypatch.setattr(exp_f.store_mod, "validate_store", lambda *a, **k: None)
    monkeypatch.setattr(exp_f.exp_a, "expected_fingerprints", lambda *a, **k: {})
    monkeypatch.setattr(exp_f, "load_ground_truth", _gt, raising=False)
    monkeypatch.setattr("dehyd.data.ground_truth.load_ground_truth", lambda *a, **k: _gt(6))

    run_dir = tmp_path / "run"
    outputs = exp_f.run_and_report_f(
        config, "10ghz", sessions, tmp_path, run_dir, mode="smoke", analysis_commit="deadbeef",
        exp_a_sources=sources_path, n_workers=1)

    assert set(outputs) == {"availability", "run_log"}
    payload = json.loads(outputs["run_log"].read_text(encoding="utf-8"))
    assert payload["mode"] == "mechanism-only"
    assert payload["heart_rate_question"]["status"] == exp_f.HR_STATUS
    assert not list(run_dir.glob("contrasts_f_*.csv"))
    assert not list(run_dir.glob("metrics_exp_f_*.json"))
    assert "difference" not in json.dumps(payload)
