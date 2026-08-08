"""T-M10-assembly: the explicit run map and the per-experiment adapters.

§5.5's last bullet in full: "Assembly round-trips each actual A-D schema plus synthetic E-G
schemas; missing/mismatched source artifacts fail closed. Run directories are supplied
explicitly, never discovered by glob."

**How "actual schema" is established here matters.** Rather than transcribing each experiment's
metrics keys into a fixture — which would test my transcription, not the schema — every A-D
fixture is produced by calling the experiment's OWN `summarize_*` function on small synthetic
fold results. If Exp B ever renames `primary_aggregate`, this file fails. The one exception is
Exp A, which additionally round-trips the real committed M7 artifact on disk, because that one
exists and is the strongest possible evidence.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from dehyd.config import load_config
from dehyd.eval import assembly
from dehyd.eval.assembly import AssemblyError, SourceRun

REPO_ROOT = Path(__file__).resolve().parents[1]
# The two committed M7 Exp A runs — the only real artifacts version-controlled in this repo.
REAL_EXP_A_10 = REPO_ROOT / "results" / "runs" / "20260727T111437230187Z_f36c4fb2"
REAL_EXP_A_77 = REPO_ROOT / "results" / "runs" / "20260727T115046533408Z_f36c4fb2"


@pytest.fixture(scope="module")
def config():
    """A fast statistical config: the real b=10000 bootstrap would dominate this file."""
    base = load_config("configs/exp_a_regression.yaml", "configs/exp_b.yaml",
                       "configs/exp_c.yaml", "configs/stats.yaml")
    return dataclasses.replace(base, stats=dataclasses.replace(base.stats, bootstrap_b=64))


# ------------------------------------------------------------------------------- fixtures


def _write_run(tmp_path, name, artifacts, *, commit="abc123def456", config_payload=None):
    """A synthetic run directory: provenance.json plus whatever artifacts are supplied."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps({
        "git": {"commit": commit, "branch": "v1_milestone_10", "dirty": False},
        "config": config_payload if config_payload is not None else {"run": {"seed": 20260721}},
    }), encoding="utf-8")
    for name_, content in artifacts.items():
        path = run_dir / name_
        if isinstance(content, (dict, list)):
            path.write_text(json.dumps(content), encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")
    return run_dir


def _csv(columns, rows):
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in columns))
    return "\n".join(lines) + "\n"


def _stub(names):
    """Every required artifact present but empty-ish — enough to register a run."""
    return {name: ("{}" if name.endswith(".json") else "col\n") for name in names}


def _run_for(tmp_path, experiment, band, *, metrics=None, extra=None, name=None):
    """A registrable run directory for one experiment/band, with a real metrics payload."""
    names = assembly.required_artifacts(experiment, band)
    artifacts = _stub(names)
    if metrics is not None:
        metrics_name = next(n for n in names if n.startswith("metrics_") and n.endswith(".json"))
        artifacts[metrics_name] = metrics
    artifacts.update(extra or {})
    return _write_run(tmp_path, name or f"{experiment}_{band or 'xband'}", artifacts)


# ------------------------------------------------------------- the explicit run map


def test_the_manifest_records_lineage_for_every_required_artifact(tmp_path):
    run_dir = _run_for(tmp_path, "d", "10ghz")
    manifest = assembly.build_run_manifest([SourceRun("d", "10ghz", run_dir)])

    entry = manifest["runs"]["d_10ghz"]
    assert manifest["discovery"] == "explicit_only_no_glob"
    assert entry["source_commit"] == "abc123def456"
    assert len(entry["resolved_config_sha256"]) == 64
    assert set(entry["artifacts"]) == set(assembly.required_artifacts("d", "10ghz"))
    for record in entry["artifacts"].values():
        assert len(record["sha256"]) == 64
        assert Path(record["path"]).is_file()


def test_a_missing_required_artifact_fails_closed_naming_it(tmp_path):
    names = assembly.required_artifacts("c", "10ghz")
    artifacts = _stub(names)
    del artifacts[f"confusion_10ghz.csv"]
    run_dir = _write_run(tmp_path, "c_partial", artifacts)
    with pytest.raises(AssemblyError, match="confusion_10ghz.csv.*is missing"):
        assembly.build_run_manifest([SourceRun("c", "10ghz", run_dir)])


def test_a_run_directory_that_does_not_exist_fails_closed(tmp_path):
    with pytest.raises(AssemblyError, match="run directory does not exist"):
        assembly.build_run_manifest([SourceRun("a", "10ghz", tmp_path / "nope")])


def test_a_run_without_a_commit_is_refused(tmp_path):
    run_dir = _run_for(tmp_path, "d", "10ghz")
    (run_dir / "provenance.json").write_text(json.dumps({"git": {}, "config": {}}),
                                             encoding="utf-8")
    with pytest.raises(AssemblyError, match="records no git commit"):
        assembly.build_run_manifest([SourceRun("d", "10ghz", run_dir)])


def test_one_experiment_band_maps_to_exactly_one_run(tmp_path):
    a = _run_for(tmp_path, "d", "10ghz", name="d_one")
    b = _run_for(tmp_path, "d", "10ghz", name="d_two")
    with pytest.raises(AssemblyError, match="supplied twice"):
        assembly.build_run_manifest([SourceRun("d", "10ghz", a), SourceRun("d", "10ghz", b)])


def test_g_is_cross_band_and_every_other_experiment_is_per_band(tmp_path):
    run_dir = _run_for(tmp_path, "g", None)
    assembly.build_run_manifest([SourceRun("g", None, run_dir)])
    with pytest.raises(AssemblyError, match="cross-band and takes band=None"):
        assembly.build_run_manifest([SourceRun("g", "10ghz", run_dir)])
    with pytest.raises(AssemblyError, match="per-band and requires a band"):
        assembly.build_run_manifest([SourceRun("a", None, run_dir)])


def test_an_unknown_experiment_is_refused(tmp_path):
    with pytest.raises(AssemblyError, match="unknown experiment"):
        assembly.required_artifacts("zzz", "10ghz")


def test_assembly_never_discovers_a_run(tmp_path):
    with pytest.raises(AssemblyError, match="never discovers runs"):
        assembly.build_run_manifest([])


def test_an_artifact_that_changed_after_registration_stops_the_milestone(tmp_path):
    run_dir = _run_for(tmp_path, "d", "10ghz")
    manifest = assembly.build_run_manifest([SourceRun("d", "10ghz", run_dir)])
    assembly.validate_manifest(manifest)          # clean to start with

    (run_dir / "composite_10ghz.csv").write_text("col\ntampered\n", encoding="utf-8")
    with pytest.raises(AssemblyError, match="changed after registration"):
        assembly.validate_manifest(manifest)


def test_a_registered_artifact_that_vanished_stops_the_milestone(tmp_path):
    run_dir = _run_for(tmp_path, "d", "10ghz")
    manifest = assembly.build_run_manifest([SourceRun("d", "10ghz", run_dir)])
    (run_dir / "composite_10ghz.csv").unlink()
    with pytest.raises(AssemblyError, match="vanished"):
        assembly.validate_manifest(manifest)


def test_the_required_artifact_names_are_per_experiment_not_a_pattern():
    """The names genuinely differ; a pattern that worked for four experiments would silently
    accept a wrong file for the fifth."""
    assert "predictions_10ghz.csv" in assembly.required_artifacts("a", "10ghz")
    assert "predictions_b_10ghz.csv" in assembly.required_artifacts("b", "10ghz")
    assert not any("predictions" in n for n in assembly.required_artifacts("d", "10ghz"))
    assert all(not n.endswith("_10ghz.csv") for n in assembly.required_artifacts("g", None))


# --------------------------------------------------- round-tripping the ACTUAL schemas


def test_the_real_committed_exp_a_artifact_round_trips(tmp_path):
    """The strongest evidence available: a real, committed, full-cohort Exp A run."""
    if not REAL_EXP_A_10.is_dir():
        pytest.skip("the committed M7 Exp A run is not present")
    manifest = assembly.build_run_manifest([SourceRun("a", "10ghz", REAL_EXP_A_10)])
    tables = assembly.assemble(manifest)

    headline = {r["metric"]: r for r in tables["headline"]}
    assert set(headline) == {"subject_balanced_mae", "session_rmse", "pooled_pearson_r"}
    assert headline["subject_balanced_mae"]["primary_or_secondary"] == "primary"
    assert headline["subject_balanced_mae"]["n_subjects"] == 16
    assert headline["subject_balanced_mae"]["ci_method"] == "bca"
    assert headline["subject_balanced_mae"]["ci_low"] < headline["subject_balanced_mae"]["estimate"]
    assert headline["pooled_pearson_r"]["primary_or_secondary"] == "secondary"
    # both the radar and the baseline per-subject series, 16 subjects each
    assert len(tables["per_subject"]) == 32
    assert {r["model_or_contrast"] for r in tables["per_subject"]} == {
        "selected_radar", "session_index_baseline"}
    assert len(tables["paired"]) == 1
    paired = tables["paired"][0]
    assert paired["comparison"] == "radar_minus_session_index_baseline"
    assert paired["direction"] in ("negative_favours_first_term", "positive_favours_second_term")
    assert paired["n_pairs"] == 16


def _exp_a_metrics(config):
    """A real Exp-A summary, produced by `summarize_exp_a` itself."""
    from dehyd.eval.exp_a import ExpAFoldResult, summarize_exp_a
    from dehyd.eval.harness import SeedOutcome

    # `_per_seed_matrix` reads predictions off `seed_outcomes`, not off `test_predictions` —
    # a fixture with an empty seed list produces a zero-width matrix, which is how this was
    # caught. Two seeds, so the seed axis is real rather than degenerate.
    rng = np.random.default_rng(0)
    results = []
    for subject in range(1, 7):
        n = 4
        y = rng.normal(size=n)
        outcomes = [
            SeedOutcome(seed=seed, train_predictions=np.zeros(0),
                        test_predictions=y + rng.normal(scale=0.2, size=n), test_score=0.3)
            for seed in (1, 2)
        ]
        results.append(ExpAFoldResult(
            test_subject=subject, selected_feature_key=(0, "A", "mag", 0, "off"),
            selected_family="ridge", selected_params={"alpha": 1.0},
            test_predictions=outcomes[0].test_predictions,
            test_targets=y, seed_outcomes=outcomes,
            baseline_predictions=np.zeros(n), final_fits=[],
        ))
    return summarize_exp_a(results, config)


def test_the_exp_a_schema_round_trips_from_its_own_summarizer(tmp_path, config):
    metrics = _exp_a_metrics(config)
    run_dir = _run_for(tmp_path, "a", "10ghz", metrics=metrics)
    tables = assembly.assemble(assembly.build_run_manifest([SourceRun("a", "10ghz", run_dir)]))
    assert len(tables["headline"]) == 3
    assert {r["experiment"] for r in tables["headline"]} == {"a"}
    assert all(r["source_run"] == str(run_dir) for r in tables["headline"])
    assert all(r["source_artifact"] == "metrics_exp_a_10ghz.json" for r in tables["headline"])


def test_the_exp_b_schema_round_trips_including_an_unavailable_primary(tmp_path, config):
    """Exp B's primary aggregate can legitimately be UNAVAILABLE. Assembly must report that as
    a status with a reason in the exclusion ledger, never omit the row."""
    metrics = {
        "conditional_exploratory": True,
        "n_eval_subjects_aggregate": 12, "n_rows": 44,
        "primary_viable": False, "primary_aggregate": None,
        "primary_unavailable_reason": "session(s) [4] have zero out-of-fold rows",
        "paired_subject_weighted_complete_case": {
            "n_complete_case": 10, "wilcoxon_p": 0.25,
            "mean_difference_radar_minus_baseline": {
                "point": 0.11, "low": -0.02, "high": 0.24, "method": "bca"},
        },
        "per_session_exploratory": {"holm_family_size": 4, "1": {
            "n_eval": 12, "wilcoxon_p": 0.3, "holm_p": 0.9,
            "mean_difference": {"point": -0.05, "low": -0.2, "high": 0.1, "method": "bca"}}},
        "dropped_sessions": {"outer_by_fold": {"3": [4]}},
    }
    run_dir = _run_for(tmp_path, "b", "10ghz", metrics=metrics)
    tables = assembly.assemble(assembly.build_run_manifest([SourceRun("b", "10ghz", run_dir)]))

    headline = tables["headline"]
    assert len(headline) == 1 and headline[0]["status"] == "primary_unavailable"
    assert headline[0]["estimate"] == "" and headline[0]["ci_low"] == ""
    reasons = [r["reason"] for r in tables["exclusions"]]
    assert any("zero out-of-fold rows" in r for r in reasons)
    assert any("degenerate_session_means_dropped" in r for r in reasons)
    families = {r["multiplicity_family"] for r in tables["paired"]}
    assert "holm_4_expb_per_session" in families


def test_the_exp_b_schema_round_trips_a_viable_primary(tmp_path):
    ci = {"point": 0.4, "low": 0.3, "high": 0.5, "method": "bca"}
    metrics = {
        "n_eval_subjects_aggregate": 16, "n_rows": 60, "primary_viable": True,
        "primary_aggregate": {"radar": ci, "baseline": dict(ci, point=0.6),
                              "difference_radar_minus_baseline": dict(ci, point=-0.2)},
        "paired_subject_weighted_complete_case": {},
        "per_session_exploratory": {}, "dropped_sessions": {"outer_by_fold": {}},
    }
    run_dir = _run_for(tmp_path, "b", "10ghz", metrics=metrics)
    tables = assembly.assemble(assembly.build_run_manifest([SourceRun("b", "10ghz", run_dir)]))
    headline = {r["model_or_contrast"]: r for r in tables["headline"]}
    assert headline["radar"]["primary_or_secondary"] == "primary"
    assert headline["baseline"]["primary_or_secondary"] == "secondary"
    assert tables["paired"][0]["direction"] == "negative_favours_first_term"
    assert tables["exclusions"] == []


def test_the_exp_c_schema_round_trips_both_arms(tmp_path):
    """Exp C's secondary ordinal metrics carry NO interval (§2.4), so their CI columns must be
    blank rather than filled with the point estimate twice."""
    def arm(mae):
        return {
            "class_unit_mae": {"point": mae, "low": mae - 0.1, "high": mae + 0.1, "method": "bca"},
            "adjacent_accuracy": {"point": 0.8},
            "quadratic_weighted_kappa": {"point": 0.3},
            "per_subject_class_mae": {"1": 0.5, "2": 1.5},
        }
    metrics = {"n_eval_subjects": 16, "n_rows": 73, "arms": {"a": arm(1.1), "b": arm(1.3)}}
    run_dir = _run_for(tmp_path, "c", "10ghz", metrics=metrics)
    tables = assembly.assemble(assembly.build_run_manifest([SourceRun("c", "10ghz", run_dir)]))

    by_key = {(r["model_or_contrast"], r["metric"]): r for r in tables["headline"]}
    assert by_key[("arm_a", "class_unit_mae")]["primary_or_secondary"] == "primary"
    assert by_key[("arm_a", "class_unit_mae")]["ci_method"] == "bca"
    qwk = by_key[("arm_b", "quadratic_weighted_kappa")]
    assert qwk["primary_or_secondary"] == "secondary"
    assert qwk["ci_low"] == "" and qwk["ci_high"] == ""
    assert qwk["status"] == "conditional_exploratory_no_interval"
    assert len(tables["per_subject"]) == 4          # 2 arms x 2 subjects


def test_the_real_exp_d_schema_round_trips_per_family_and_both_comparisons(tmp_path):
    """Exp D has per-FAMILY baselines and two frozen comparisons, and no prediction table."""
    ci = {"point": 0.47, "low": 0.41, "high": 0.57, "method": "bca"}
    metrics = {
        "band": "10ghz", "n_eval": 16,
        "radar": {"subject_balanced_mae": ci, "per_subject_mae": {"1": 0.4, "2": 0.5}},
        "per_family_metrics": {
            "physics": {"subject_balanced_mae": dict(ci, point=0.55)},
            "session_index": {"subject_balanced_mae": dict(ci, point=0.27)},
        },
        "primary_vs_session_index": {
            "n_eval": 16, "wilcoxon_p": 3.05e-05,
            "mean_difference_radar_minus_baseline": dict(ci, point=0.2)},
        "composite": {
            "n_eval": 16, "wilcoxon_p": 0.0021, "correction": "single_uncorrected",
            "mean_difference_radar_minus_composite": dict(ci, point=-0.099)},
    }
    run_dir = _run_for(tmp_path, "d", "10ghz", metrics=metrics)
    tables = assembly.assemble(assembly.build_run_manifest([SourceRun("d", "10ghz", run_dir)]))

    families = {r["model_or_contrast"] for r in tables["headline"]}
    assert families == {"selected_radar", "physics", "session_index"}
    comparisons = {r["comparison"]: r for r in tables["paired"]}
    assert comparisons["radar_minus_session_index"]["primary_or_secondary"] == "primary"
    assert comparisons["radar_minus_composite"]["primary_or_secondary"] == "secondary"
    assert comparisons["radar_minus_composite"]["direction"] == "negative_favours_first_term"
    assert len(tables["per_subject"]) == 2


def test_exp_e_contributes_no_headline_row_because_it_states_no_estimate(tmp_path):
    """The module's clearest case of not inventing a shape: E is descriptive, so it is
    registered, validated and pointed at — but never given a fabricated headline estimate."""
    metrics = {"status": "descriptive", "n_paths": 742, "n_evaluable_outer_folds": 16}
    extra = {"exclusions_e_10ghz.csv": _csv(
        ("band", "outer_fold", "test_subject", "reason", "detail"),
        [{"band": "10ghz", "outer_fold": 7, "test_subject": 7,
          "reason": "no_surviving_test_rows", "detail": ""}])}
    run_dir = _run_for(tmp_path, "e", "10ghz", metrics=metrics, extra=extra)
    tables = assembly.assemble(assembly.build_run_manifest([SourceRun("e", "10ghz", run_dir)]))

    assert tables["headline"] == []
    assert tables["per_subject"] == []
    assert tables["paired"] == []
    assert [r["reason"] for r in tables["exclusions"]] == ["no_surviving_test_rows"]
    assert tables["descriptive"]["e_10ghz"]["n_paths"] == 742
    assert "importance_summary_10ghz.csv" in tables["descriptive"]["e_10ghz"]["tables"]


def test_the_exp_f_schema_round_trips_every_multiplicity_family(tmp_path):
    metrics = {
        "n_subjects_f": 16,
        "heart_rate_question": {"status": "not_estimable_missing_heart_rate",
                                "n_hr_observations": 0,
                                "uncontrolled_variables": ["glucose", "heart_rate", "temperature"]},
        "contrasts": [
            {"contrast_id": "radar_given_clock", "analysis_variant": "pct_full",
             "multiplicity_family": "holm_2_pct_full_radar_increments", "mean_difference": -0.1,
             "ci_low": -0.3, "ci_high": 0.1, "ci_method": "bca", "p_value_unadjusted": 0.02,
             "p_value_holm": 0.04, "n_paired_subjects": 16, "n_nonzero_pairs": 15, "n_ties": 1},
            {"contrast_id": "covariates_given_clock", "analysis_variant": "pct_full",
             "multiplicity_family": "none_reported_individually", "mean_difference": 0.02,
             "ci_low": -0.1, "ci_high": 0.14, "ci_method": "bca", "p_value_unadjusted": 0.6,
             "p_value_holm": float("nan"), "n_paired_subjects": 16, "n_nonzero_pairs": 16,
             "n_ties": 0},
            {"contrast_id": "radar_given_clock", "analysis_variant": "kg_full",
             "multiplicity_family": "none_sensitivity", "mean_difference": -0.05,
             "ci_low": -0.2, "ci_high": 0.1, "ci_method": "bca", "p_value_unadjusted": 0.3,
             "p_value_holm": float("nan"), "n_paired_subjects": 16, "n_nonzero_pairs": 14,
             "n_ties": 2},
        ],
    }
    extra = {"contrasts_f_10ghz.csv": _csv(
        ("subject", "contrast_id", "analysis_variant", "target_name", "n_sessions", "mae_with",
         "mae_without", "difference_with_minus_without"),
        [{"subject": 1, "contrast_id": "radar_given_clock", "analysis_variant": "pct_full",
          "target_name": "delta_m_pct", "n_sessions": 5, "mae_with": 0.4, "mae_without": 0.5,
          "difference_with_minus_without": -0.1}])}
    run_dir = _run_for(tmp_path, "f", "10ghz", metrics=metrics, extra=extra)
    tables = assembly.assemble(assembly.build_run_manifest([SourceRun("f", "10ghz", run_dir)]))

    by_comparison = {r["comparison"]: r for r in tables["paired"]}
    primary = by_comparison["radar_given_clock::pct_full"]
    assert primary["primary_or_secondary"] == "primary"
    assert primary["p_value_adjusted"] == 0.04
    assert primary["n_ties"] == 1 and primary["n_nonzero_pairs"] == 15
    assert primary["estimate"] == -0.1 and primary["ci_method"] == "bca"
    assert by_comparison["covariates_given_clock::pct_full"]["primary_or_secondary"] == "secondary"
    assert by_comparison["radar_given_clock::kg_full"]["multiplicity_family"] == "none_sensitivity"
    # the not-estimable HR record travels into the exclusion ledger
    assert {r["identifier"] for r in tables["exclusions"]} == {
        "glucose", "heart_rate", "temperature"}
    assert any("not_estimable_missing_heart_rate" in r["reason"] for r in tables["exclusions"])
    assert len(tables["per_subject"]) == 1


def test_the_exp_g_schema_round_trips_with_a_blank_band(tmp_path):
    """G is cross-band, and its `primary` is a WRAPPER around the CI dict — reading it as a CI
    dict directly would silently blank every interval column."""
    metrics = {
        "n_subjects_g": 14,
        "primary": {
            "estimand": "mean_over_subject(...)", "direction": "negative favours fusion",
            "sign": "positive", "n_subjects": 14,
            "mean_difference_fused_minus_10": {"point": 0.03, "low": -0.01, "high": 0.08,
                                               "method": "bca"}},
    }
    extra = {"per_subject_g.csv": _csv(
        ("subject", "n_sessions", "mae_10", "mae_77", "mae_equal_weight", "mae_fused",
         "difference_fused_minus_10"),
        [{"subject": 1, "n_sessions": 4, "mae_10": 0.4, "mae_77": 0.6, "mae_equal_weight": 0.45,
          "mae_fused": 0.43, "difference_fused_minus_10": 0.03}])}
    run_dir = _run_for(tmp_path, "g", None, metrics=metrics, extra=extra)
    tables = assembly.assemble(assembly.build_run_manifest([SourceRun("g", None, run_dir)]))

    paired = tables["paired"][0]
    assert paired["band"] == ""
    assert paired["comparison"] == "fused_minus_10ghz"
    assert paired["estimate"] == 0.03 and paired["ci_low"] == -0.01 and paired["ci_method"] == "bca"
    assert paired["test"] == "subject_cluster_bootstrap_ci"
    assert paired["n_pairs"] == 14
    assert {r["model_or_contrast"] for r in tables["per_subject"]} == {
        "10", "77", "equal_weight", "fused"}
    assert all(r["band"] == "" for r in tables["per_subject"])


def test_the_robustness_range_is_never_relabelled_as_a_confidence_interval(tmp_path):
    """§7: "its empirical range is not mislabeled BCa". The label is carried through from the
    artifact verbatim rather than reconstructed."""
    extra = {"robustness_summary.csv": _csv(
        ("experiment", "band", "arm_or_contrast", "original_point", "range_low", "range_high",
         "range_label", "status", "ci_method"),
        [{"experiment": "a", "band": "10ghz", "arm_or_contrast": "subject_balanced_mae",
          "original_point": 0.47, "range_low": 0.39, "range_high": 0.58,
          "range_label": "selection_variance_empirical_95pct_range", "status": "conclusive",
          "ci_method": ""}])}
    run_dir = _run_for(tmp_path, "robustness", "10ghz", extra=extra)
    tables = assembly.assemble(
        assembly.build_run_manifest([SourceRun("robustness", "10ghz", run_dir)]))
    row = tables["headline"][0]
    assert row["ci_method"] == "selection_variance_empirical_95pct_range"
    assert "bca" not in row["ci_method"].lower()
    assert row["status"] == "conclusive"
    assert row["primary_or_secondary"] == "secondary"


# ------------------------------------------------------------------- the final tables


def test_the_final_tables_are_written_with_the_exact_column_lists(tmp_path, config):
    sources = [
        SourceRun("a", "10ghz", _run_for(tmp_path, "a", "10ghz", metrics=_exp_a_metrics(config))),
        SourceRun("e", "10ghz", _run_for(
            tmp_path, "e", "10ghz", metrics={"status": "descriptive", "n_paths": 6},
            extra={"exclusions_e_10ghz.csv": _csv(
                ("band", "outer_fold", "test_subject", "reason", "detail"), [])})),
    ]
    out = tmp_path / "assembled"
    paths = assembly.assemble_and_report(sources, out)

    import csv as _csv_mod
    for key, columns in (("headline", assembly.HEADLINE_COLUMNS),
                         ("per_subject", assembly.PER_SUBJECT_COLUMNS),
                         ("paired", assembly.PAIRED_COLUMNS),
                         ("exclusions", assembly.EXCLUSIONS_COLUMNS)):
        with open(paths[key], newline="", encoding="utf-8") as fh:
            assert _csv_mod.DictReader(fh).fieldnames == list(columns), key

    summary = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert summary["discovery"] == "explicit_only_no_glob"
    assert set(summary["sources"]) == {"a_10ghz", "e_10ghz"}
    assert summary["sources"]["a_10ghz"]["source_commit"] == "abc123def456"
    assert summary["counts"]["headline"] == 3
    assert any("no headline row" in note for note in summary["notes"])
    assert any("NOT a BCa" in note for note in summary["notes"])
    # the manifest is written next to the tables and is itself an artifact
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == assembly.MANIFEST_SCHEMA_VERSION


def test_per_subject_results_never_contain_a_frame_row(tmp_path, config):
    """§3: "no frame rows". Every per-subject row is one subject's aggregate, so the count can
    never exceed the number of subjects times the conditions reported for them."""
    run_dir = _run_for(tmp_path, "a", "10ghz", metrics=_exp_a_metrics(config))
    tables = assembly.assemble(assembly.build_run_manifest([SourceRun("a", "10ghz", run_dir)]))
    subjects = {r["subject"] for r in tables["per_subject"]}
    conditions = {r["model_or_contrast"] for r in tables["per_subject"]}
    assert len(tables["per_subject"]) == len(subjects) * len(conditions)
    assert all(isinstance(r["subject"], int) for r in tables["per_subject"])


def test_assemble_validates_before_reading_anything(tmp_path):
    run_dir = _run_for(tmp_path, "d", "10ghz")
    manifest = assembly.build_run_manifest([SourceRun("d", "10ghz", run_dir)])
    (run_dir / "composite_10ghz.csv").write_text("col\nmoved\n", encoding="utf-8")
    with pytest.raises(AssemblyError, match="changed after registration"):
        assembly.assemble(manifest)


def test_an_incomplete_metrics_file_fails_closed_naming_the_experiment_and_key(tmp_path):
    """Assembly is the last step before publication, so a truncated or half-written metrics
    file has to be diagnosable from the message alone — never a bare KeyError from inside an
    adapter, which says nothing about WHICH source was malformed."""
    run_dir = _run_for(tmp_path, "d", "10ghz", metrics={"band": "10ghz"})   # no n_eval
    manifest = assembly.build_run_manifest([SourceRun("d", "10ghz", run_dir)])
    with pytest.raises(AssemblyError, match=r"d/10ghz: metrics_exp_d_10ghz\.json has no 'n_eval'"):
        assembly.assemble(manifest)


def test_an_incomplete_exp_a_metrics_file_names_the_missing_ci_block(tmp_path):
    run_dir = _run_for(tmp_path, "a", "10ghz",
                       metrics={"n_eval_subjects": 16, "n_sessions": 73,
                                "subject_balanced_mae": {"point": 0.4}})
    manifest = assembly.build_run_manifest([SourceRun("a", "10ghz", run_dir)])
    with pytest.raises(AssemblyError, match="has no 'session_rmse'"):
        assembly.assemble(manifest)


def test_a_manifest_with_the_wrong_schema_version_is_refused():
    with pytest.raises(AssemblyError, match="run manifest schema"):
        assembly.validate_manifest({"schema_version": "something_else", "runs": {}})
