"""T-M10-entrypoints (Exp H half): `experiments/run_robustness.py` + `run_robustness.sbatch`.

Flag validation, the `R`/seed wiring the milestone's launch matrix depends on, the mechanism-
only reporting boundary, and static checks on the IBEX wrapper (its argv mapping and that
`bash -n` parses it). The bootstrap itself is covered by `test_robustness.py`; here
`run_and_report_robustness` is monkeypatched so the CLI's OWN wiring is tested fast, against a
REAL `record_run` call.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import robustness

import experiments.run_robustness as rr

REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH = REPO_ROOT / "scripts" / "ibex" / "run_robustness.sbatch"
SHARDED_SBATCH = REPO_ROOT / "scripts" / "ibex" / "run_robustness_sharded.sbatch"
SUBMIT_SH = REPO_ROOT / "scripts" / "ibex" / "submit_robustness_sharded.sh"

CONFIG_ARGS = ["--config", "configs/exp_a_regression.yaml", "--config", "configs/stats.yaml"]


def _sessions(n_subjects=8):
    return [
        {"subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
         "rel_path": f"subject_{s}_{SESSION_NAMES[i]}.mat", "frame_ids": [0],
         "delta_m_pct": -0.3 * i, "loss_l": 0.3 * i, "class_idx": i}
        for s in range(1, n_subjects + 1)
        for i in range(5)
    ]


def _manifest(data_dir, sessions):
    rows = []
    for s in sessions:
        (data_dir / s["rel_path"]).write_bytes(f"radar {s['rel_path']}".encode())
        rows.append({"subject": s["subject"], "session_idx": s["session_idx"],
                     "rel_path": s["rel_path"]})
    return pd.DataFrame(rows)


def _overlay(tmp_path):
    data_10 = tmp_path / "data10"
    results = tmp_path / "results"
    for directory in (data_10, results):
        directory.mkdir(exist_ok=True)
    (tmp_path / "weights.xlsx").write_bytes(b"pretend workbook")
    path = tmp_path / "paths_overlay.yaml"
    path.write_text("\n".join([
        "paths:",
        f"  data_10ghz_dir: {data_10.as_posix()}",
        f"  weight_xlsx: {(tmp_path / 'weights.xlsx').as_posix()}",
        f"  results_dir: {results.as_posix()}",
    ]) + "\n", encoding="utf-8")
    return path, data_10, results


def _patch(monkeypatch, sessions, data_dir, captured):
    monkeypatch.setattr(rr.robustness, "build_spine", lambda *a, **k: sessions)
    monkeypatch.setattr(rr, "_build_manifest_qc", lambda config, band: _manifest(data_dir, sessions))

    def fake_run(config, experiment, band, sess, store_dir, run_dir, *, mode, analysis_commit,
                 replicates, n_workers=1):
        captured.update({"stage": "single", "experiment": experiment, "band": band, "mode": mode,
                         "replicates": replicates, "n_sessions": len(sess), "run_dir": run_dir})
        return {"run_log": run_dir / "run_log.json"}

    def fake_shard(config, experiment, band, sess, store_dir, shard_dir, *, analysis_commit,
                   replicates, start, stop, n_workers=1):
        captured.update({"stage": "shard", "experiment": experiment, "band": band,
                         "replicates": replicates, "start": start, "stop": stop,
                         "shard_dir": shard_dir, "n_sessions": len(sess)})
        return {"shard": Path(shard_dir) / "shard.json"}

    def fake_merge(config, experiment, band, sess, store_dir, shard_dir, run_dir, *, mode,
                   analysis_commit, replicates, n_workers=1):
        captured.update({"stage": "merge", "experiment": experiment, "band": band, "mode": mode,
                         "replicates": replicates, "shard_dir": shard_dir, "run_dir": run_dir})
        return {"summary": run_dir / "robustness_summary.csv"}

    monkeypatch.setattr(rr.robustness, "run_and_report_robustness", fake_run)
    monkeypatch.setattr(rr.robustness, "run_and_report_shard", fake_shard)
    monkeypatch.setattr(rr.robustness, "run_and_report_merge", fake_merge)


# ------------------------------------------------------------------- flag validation


def test_exactly_one_of_subset_or_full_cohort_is_required():
    with pytest.raises(SystemExit):
        rr.main(CONFIG_ARGS + ["--experiment", "a"])
    with pytest.raises(SystemExit):
        rr.main(CONFIG_ARGS + ["--experiment", "a", "--subset", "6subjects", "--full-cohort"])


def test_experiment_is_required_and_restricted_to_a_b_c():
    with pytest.raises(SystemExit):
        rr.main(CONFIG_ARGS + ["--full-cohort"])
    with pytest.raises(SystemExit):
        rr.main(CONFIG_ARGS + ["--experiment", "d", "--full-cohort"])


def test_unknown_band_is_rejected():
    with pytest.raises(SystemExit):
        rr.main(CONFIG_ARGS + ["--experiment", "a", "--band", "24ghz", "--full-cohort"])


def test_a_non_positive_replicate_count_is_rejected():
    with pytest.raises(SystemExit):
        rr.main(CONFIG_ARGS + ["--experiment", "a", "--replicates", "0", "--full-cohort"])


# --------------------------------------------------------------- R, seeds, provenance


def test_replicates_defaults_to_the_frozen_r_and_the_flag_overrides_it(tmp_path, monkeypatch):
    """`R = 200` is pre-registered in `StatsConfig`, so the default must come from there rather
    than from a literal in the CLI. The flag exists for the smoke and for a re-run at a
    different R, which the launch matrix passes explicitly."""
    overlay, data_10, _ = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _sessions(), data_10, captured)

    assert rr.main(CONFIG_ARGS + ["--config", str(overlay), "--experiment", "a",
                                  "--full-cohort"]) == 0
    assert captured["replicates"] == 200

    assert rr.main(CONFIG_ARGS + ["--config", str(overlay), "--experiment", "b",
                                  "--replicates", "8", "--subset", "6subjects"]) == 0
    assert captured["replicates"] == 8 and captured["experiment"] == "b"


def test_the_run_records_its_experiment_replicates_and_robustness_seed(tmp_path, monkeypatch):
    """The provenance has to say which draw produced these numbers. Recording the root seed
    (and R) is what lets a later assembly step re-derive every replicate's seed tuple."""
    overlay, data_10, results = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _sessions(), data_10, captured)

    assert rr.main(CONFIG_ARGS + ["--config", str(overlay), "--experiment", "c",
                                  "--replicates", "200", "--full-cohort"]) == 0

    run_dirs = sorted((results / "runs").iterdir())
    payload = json.loads((run_dirs[0] / "provenance.json").read_text(encoding="utf-8"))
    assert payload["extra"]["stage"] == "robustness-c-full"
    assert payload["extra"]["experiment"] == "c"
    assert payload["extra"]["replicates"] == 200
    assert payload["extra"]["robustness_seed"] == 20260721
    assert payload["extra"]["n_eval"] == 8
    assert len(payload["folds"]) == 8            # a real fold-role manifest, not an empty slot
    assert captured["run_dir"] == run_dirs[0]


def test_smoke_keeps_the_six_lowest_subjects_and_writes_no_estimate(tmp_path, monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _sessions(), data_10, captured)

    assert rr.main(CONFIG_ARGS + ["--config", str(overlay), "--experiment", "a",
                                  "--replicates", "8", "--subset", "6subjects"]) == 0

    assert captured["mode"] == "smoke"
    assert captured["n_sessions"] == 30          # the 6 lowest subjects x 5 sessions
    payload = json.loads(
        (sorted((results / "runs").iterdir())[0] / "provenance.json").read_text(encoding="utf-8")
    )
    assert payload["extra"]["stage"] == "robustness-a-smoke"
    # the CLI itself writes no numerical artifact; the boundary lives in the library call
    assert not list(results.rglob("robustness_*.csv"))
    assert not list(results.rglob("metrics_robustness.json"))


def test_full_cohort_refuses_the_single_smoke_seed(tmp_path, monkeypatch):
    """Model seeds are the configured seeds and are never derived from the resampling seed
    (§2.4), so this entrypoint is the only place they can silently shrink. Same guard
    `run_ordinal.py` carries."""
    overlay, data_10, _ = _overlay(tmp_path)
    seeds = tmp_path / "smoke_seeds.yaml"
    seeds.write_text("run:\n  seed_set: [1]\n", encoding="utf-8")
    _patch(monkeypatch, _sessions(), data_10, {})

    with pytest.raises(SystemExit, match="seed_set"):
        rr.main(CONFIG_ARGS + ["--config", str(overlay), "--config", str(seeds),
                               "--experiment", "a", "--full-cohort"])


def test_the_smoke_accepts_the_single_seed_overlay(tmp_path, monkeypatch):
    overlay, data_10, _ = _overlay(tmp_path)
    seeds = tmp_path / "smoke_seeds.yaml"
    seeds.write_text("run:\n  seed_set: [1]\n", encoding="utf-8")
    captured = {}
    _patch(monkeypatch, _sessions(), data_10, captured)

    assert rr.main(CONFIG_ARGS + ["--config", str(overlay), "--config", str(seeds),
                                  "--experiment", "a", "--replicates", "8",
                                  "--subset", "6subjects"]) == 0
    assert captured["mode"] == "smoke"


# ------------------------------------------------------------------ shard and merge stages


def test_shard_mode_needs_all_three_of_its_flags():
    """A shard that does not declare its own range cannot be checked for gaps or overlaps at
    merge time, so a half-specified shard is refused at the CLI rather than written."""
    base = CONFIG_ARGS + ["--experiment", "a", "--full-cohort"]
    for partial in (["--replicate-start", "1"],
                    ["--replicate-stop", "10"],
                    ["--shard-out", "shards"],
                    ["--replicate-start", "1", "--replicate-stop", "10"]):
        with pytest.raises(SystemExit):
            rr.main(base + partial)


def test_shard_and_merge_are_separate_stages():
    with pytest.raises(SystemExit):
        rr.main(CONFIG_ARGS + ["--experiment", "a", "--full-cohort",
                               "--replicate-start", "1", "--replicate-stop", "10",
                               "--shard-out", "shards", "--merge-shards", "shards"])


def test_a_shard_task_writes_no_run_directory(tmp_path, monkeypatch):
    """Shards deliberately skip `record_run`: it hashes every raw file (tens of GB at 77 GHz),
    and twenty array tasks doing it would be twenty times the I/O for one run's worth of
    provenance. The MERGE writes the authoritative run directory; the shard self-attests a
    lineage block instead."""
    overlay, data_10, results = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _sessions(), data_10, captured)

    assert rr.main(CONFIG_ARGS + ["--config", str(overlay), "--experiment", "a",
                                  "--replicates", "200", "--full-cohort",
                                  "--replicate-start", "11", "--replicate-stop", "20",
                                  "--shard-out", str(tmp_path / "shards")]) == 0

    assert captured["stage"] == "shard"
    assert (captured["start"], captured["stop"]) == (11, 20)
    assert captured["replicates"] == 200          # the shard still knows the whole R
    assert not (results / "runs").exists()        # no run directory, no raw-file hashing


def test_the_merge_stage_writes_the_run_directory_and_records_its_shard_source(tmp_path,
                                                                               monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _sessions(), data_10, captured)
    shard_dir = str(tmp_path / "shards")

    assert rr.main(CONFIG_ARGS + ["--config", str(overlay), "--experiment", "b",
                                  "--replicates", "200", "--full-cohort",
                                  "--merge-shards", shard_dir]) == 0

    assert captured["stage"] == "merge" and captured["shard_dir"] == shard_dir
    run_dirs = sorted((results / "runs").iterdir())
    payload = json.loads((run_dirs[0] / "provenance.json").read_text(encoding="utf-8"))
    assert payload["extra"]["shard_source"] == shard_dir
    assert payload["extra"]["replicates"] == 200
    assert captured["run_dir"] == run_dirs[0]


# --------------------------------------------------------------- the mechanism-only smoke


def test_the_smoke_run_log_reports_inconclusive_and_no_estimate(tmp_path, monkeypatch):
    """Plan §4.2 step 3, at the reporting boundary: a short run comes back INCONCLUSIVE under
    the frozen `min_successful = 100`, and the log carries counts and skip reasons but no
    estimate value.

    The eight replicates here all SUCCEED — the verdict comes from the rule, not from failure.
    """
    from dehyd.config import load_config

    config = load_config("configs/exp_a_regression.yaml", "configs/stats.yaml")
    sessions = _sessions(5)
    # A 5-subject pool drawn 5 times, covering 4 distinct subjects — the shape
    # `assert_mechanism_ok` checks (copies sum back to N; the held-out subject is absent from
    # every fitted set).
    multiplicity = {1: 2, 2: 1, 3: 1, 4: 1}
    fit_rows, audit_maps = [], {"subject_sets": {}, "multiplicity_maps": {}}
    for test_subject in multiplicity:
        fitted = sorted(set(multiplicity) - {test_subject})
        sha = robustness.json_sha256(fitted)
        audit_maps["subject_sets"][sha] = fitted
        fit_rows.append({"outer_test_subject": test_subject, "fitted_subjects_sha256": sha})

    outcomes = [
        robustness.ReplicateOutcome(
            experiment="a", band="10ghz", replicate=r, seed_tuple=(20260721, 1, 10, r),
            generated_seed_state="ab" * 16, multiplicity=multiplicity,
            n_distinct_subjects=4, status=robustness.STATUS_OK,
            estimates={"selected_radar_subject_balanced_mae": 1.0 + 0.1 * r,
                       "radar_minus_session_index_mae": -0.1 * r},
            fit_audit_rows=fit_rows, audit_maps=audit_maps,
        )
        for r in range(1, 9)
    ]
    point = {"selected_radar_subject_balanced_mae": 1.4, "radar_minus_session_index_mae": -0.4}
    summary_rows, skip_counts = robustness.summarize(
        config, "a", "10ghz", outcomes, point, replicates_requested=8
    )

    monkeypatch.setattr(robustness, "store_mod", type("_S", (), {
        "validate_store": staticmethod(lambda *a, **k: None)})())
    monkeypatch.setattr(robustness, "exp_a", type("_A", (), {
        "expected_fingerprints": staticmethod(lambda *a, **k: {})})())
    monkeypatch.setattr(robustness, "run_robustness",
                        lambda *a, **k: (outcomes, summary_rows, skip_counts))

    paths = robustness.run_and_report_robustness(
        config, "a", "10ghz", sessions, tmp_path / "store", tmp_path, mode="smoke",
        analysis_commit="deadbeef", replicates=8,
    )
    log = json.loads(Path(paths["run_log"]).read_text(encoding="utf-8"))
    assert set(paths) == {"run_log"}                       # no CSV, no metrics JSON
    assert log["status"] == robustness.INCONCLUSIVE
    assert log["n_replicates_attempted"] == 8 and log["n_successful"] == 8
    assert log["min_successful_replicates"] == 100
    assert set(log["skip_reason_counts"]) == set(robustness.SKIP_REASONS)
    # No estimand is named and no estimate VALUE appears anywhere in the log — the counts, the
    # reasons and the status are mechanism; the numbers are not.
    serialized = json.dumps(log)
    for estimand in robustness.ESTIMANDS["a"]:
        assert estimand not in serialized
    for outcome in outcomes:
        for value in outcome.estimates.values():
            assert repr(value) not in serialized


# ------------------------------------------------------------------ the IBEX wrapper


def test_sbatch_header_carries_its_own_fixed_resources():
    """A single, non-dispatching job, so (like run_exp_a/b/c.sbatch) it legitimately carries
    its own #SBATCH resource directives rather than taking them from a submit script."""
    header = [line for line in SBATCH.read_text(encoding="utf-8").splitlines()
              if line.startswith("#SBATCH")]
    joined = "\n".join(header)
    for directive in ("--job-name", "--cpus-per-task", "--mem", "--time", "--output", "--error"):
        assert directive in joined
    assert "--gres" not in joined and "gpu" not in joined.lower()   # E/F/G/H are entirely CPU


def test_sbatch_maps_exactly_the_launch_matrix_environment_variables():
    """§6's matrix passes EXPERIMENT, BAND, MODE and REPLICATES and nothing else, so the
    wrapper must map exactly those onto the payload's flags — a wrapper that quietly defaulted
    REPLICATES at the Python level would make the pre-registered R invisible in the job log."""
    text = SBATCH.read_text(encoding="utf-8")
    for var in ("EXPERIMENT", "BAND", "MODE", "REPLICATES"):
        assert f"{var}=${{{var}:-" in text, f"{var} must be an overridable default"
    assert "experiments/run_robustness.py" in text
    assert '--experiment "$EXPERIMENT"' in text
    assert '--band "$BAND"' in text
    assert '--replicates "$REPLICATES"' in text
    assert 'SEL=(--full-cohort)' in text and 'SEL=(--subset 6subjects)' in text
    # the 77 GHz band must select the 77 GHz config, not silently reuse the 10 GHz one
    assert 'CFG=configs/exp_a_regression_77ghz.yaml' in text
    assert "set -euo pipefail" in text      # a failing payload must fail the job


@pytest.mark.parametrize("path", [SBATCH, SHARDED_SBATCH, SUBMIT_SH], ids=lambda p: p.name)
def test_every_shell_artifact_parses_with_bash_dash_n(path):
    """A cheap syntax gate for the shell artifacts this step adds. Bytes with LF forced: on this
    repo `text=True` would rewrite \\n to \\r\\n, and a CRLF after a line-continuation backslash
    breaks bash's parser — which is exactly why the two older IBEX-script test files fail on
    this machine while this one does not."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on PATH")
    script = path.read_bytes().replace(b"\r\n", b"\n")
    result = subprocess.run(["bash", "-n", "-"], input=script, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


# ------------------------------------------------------------- the sharded IBEX wrapper


def test_the_sharded_sbatch_header_carries_no_resource_directives():
    """The array tasks and the merge want very different allocations, and #SBATCH lines are
    parsed from the script TEXT before STAGE is ever evaluated — so one fixed set of numbers
    here would silently apply to both. Same contract `run_exp_b_variant.sbatch` follows."""
    header = [line for line in SHARDED_SBATCH.read_text(encoding="utf-8").splitlines()
              if line.startswith("#SBATCH")]
    assert any(line.startswith("#SBATCH --job-name") for line in header)
    for line in header:
        for directive in ("--cpus-per-task", "--mem", "--time", "--array", "--output", "--error"):
            assert directive not in line, f"unexpected resource directive in shared header: {line!r}"


def test_the_array_stage_derives_a_contiguous_replicate_range_from_the_task_id():
    """`START = TASK*SHARD_SIZE + 1`, `STOP` clamped to R — so `--array=0-19` at SHARD_SIZE=10
    covers 1..200 exactly, with the last shard taking the remainder rather than dropping it."""
    text = SHARDED_SBATCH.read_text(encoding="utf-8")
    assert "START=$(( TASK * SHARD_SIZE + 1 ))" in text
    assert "STOP=$(( START + SHARD_SIZE - 1 ))" in text
    assert '[ "$STOP" -gt "$REPLICATES" ] && STOP=$REPLICATES' in text
    # a task id past the end must fail loudly, not write an empty shard that fails the merge
    assert 'if [ "$START" -gt "$REPLICATES" ]; then' in text
    assert '--replicate-start "$START"' in text and '--replicate-stop "$STOP"' in text
    assert '--shard-out "$SHARD_DIR"' in text
    assert '--merge-shards "$SHARD_DIR"' in text


@pytest.mark.parametrize("replicates,shard_size", [(200, 10), (200, 7), (205, 10), (8, 3), (1, 10)])
def test_the_shard_arithmetic_covers_1_to_r_exactly_when_run_by_bash(replicates, shard_size):
    """The two formulas — the submit script's ceiling division and the sbatch's per-task range —
    have to agree, or the array leaves a gap in 1..R. The merge would refuse the set, but hours
    later, after the whole array had run. So this EXERCISES both idioms against real bash on the
    remainder cases rather than only grepping for them (the pattern
    `test_exp_b_ibex_scripts.py::test_percent_percent_semicolon_star_...` established).

    Verified: exact cover means every replicate appears, and appears once — a gap and an overlap
    are different bugs and this catches both.
    """
    if shutil.which("bash") is None:
        pytest.skip("bash not available on PATH")
    script = f"""
R={replicates}; SHARD_SIZE={shard_size}
n_shards=$(( (R + SHARD_SIZE - 1) / SHARD_SIZE ))
array_max=$(( n_shards - 1 ))
for TASK in $(seq 0 $array_max); do
  START=$(( TASK * SHARD_SIZE + 1 ))
  STOP=$(( START + SHARD_SIZE - 1 ))
  [ "$STOP" -gt "$R" ] && STOP=$R
  if [ "$START" -gt "$R" ]; then echo "OVERSHOOT $TASK"; exit 1; fi
  seq $START $STOP
done
"""
    result = subprocess.run(["bash", "-s"], input=script.encode(), capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    emitted = [int(line) for line in result.stdout.decode().split()]
    assert emitted == list(range(1, replicates + 1)), "shards must cover 1..R exactly once"


def test_the_submit_script_derives_the_array_bound_from_r_and_shard_size():
    """Passing the array bound separately would let it drift from R and leave a gap in 1..R —
    the merge would refuse the set, but only hours later, after the whole array had run."""
    text = SUBMIT_SH.read_text(encoding="utf-8")
    assert "n_shards=$(( (REPLICATES + SHARD_SIZE - 1) / SHARD_SIZE ))" in text   # ceiling division
    assert "array_max=$(( n_shards - 1 ))" in text
    assert '--array=0-"${array_max}"' in text


def test_the_merge_is_chained_afterany_so_a_dead_task_surfaces_as_a_refused_shard_set():
    """`afterok` would leave the merge unsubmitted after a failed task, which reads as "nothing
    happened". `afterany` makes the merge run and refuse the incomplete set by naming the
    missing replicates — partial coverage is never summarized."""
    text = SUBMIT_SH.read_text(encoding="utf-8")
    assert '--dependency=afterany:"$array_job_id"' in text
    # ...and nowhere in the EXECUTED lines does afterok appear (the header comment explains why
    # it would be wrong, so a whole-file search would match its own explanation).
    code = [line for line in text.splitlines() if not line.strip().startswith("#")]
    assert not any("afterok" in line for line in code)
    assert 'array_job_id="${array_raw%%;*}"' in text     # strip a ";<cluster>" suffix (C25)
    assert 'merge_job_id="${merge_raw%%;*}"' in text
    assert "ARRAY_TIME:?" in text                        # required, no unsafe default
    assert "set -euo pipefail" in text


def test_the_submit_script_never_calls_git_and_requires_a_revision_file():
    """Copied-tree workflow, same as `submit_exp_b_variant.sh`: there is no .git on IBEX, so
    provenance comes from a stamped REVISION file and the DEHYD_GIT_* vars are unset."""
    text = SUBMIT_SH.read_text(encoding="utf-8")
    assert "[ -f REVISION ]" in text
    assert "unset DEHYD_GIT_COMMIT DEHYD_GIT_BRANCH DEHYD_GIT_DIRTY" in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "rev-parse" in stripped:
            continue                                     # the comment telling you to stamp it
        assert not stripped.startswith("git "), f"submit script must not call git: {line!r}"
