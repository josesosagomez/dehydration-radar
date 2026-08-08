"""T-M10-entrypoints (assembly half): `run_stats_assembly.py` + `run_stats_assembly.sbatch`,
and the `--run-dir-out` pointer mechanism the explicit run map is built from.

The two calls of the plan's launch matrix are the spine of this file: `--validate-only` must
check everything and publish nothing, and the second call must read the SAME manifest and write
every table. Everything else here defends the one rule that makes assembly traceable — nothing
is ever discovered.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dehyd.eval import assembly
from dehyd.provenance import ProvenanceError, read_run_dir_pointer, write_run_dir_pointer

import experiments.run_stats_assembly as rsa

REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH = REPO_ROOT / "scripts" / "ibex" / "run_stats_assembly.sbatch"

DRIVERS = ("run_regression.py", "run_clock_decoupling.py", "run_interpretability.py",
           "run_confound.py", "run_fusion.py", "run_robustness.py")
WRAPPERS = ("run_exp_a.sbatch", "run_exp_b.sbatch", "run_exp_e.sbatch", "run_exp_f.sbatch",
            "run_exp_g.sbatch", "run_robustness.sbatch")


# A minimal but COMPLETE metrics payload per experiment. Assembly fails closed on an
# incomplete one (that is `_require`'s whole job), so a run directory used to exercise the CLI
# has to carry a real payload rather than an empty object.
_MINIMAL_METRICS = {
    "d": {"band": "10ghz", "n_eval": 16, "radar": {}, "per_family_metrics": {}},
    "g": {"n_subjects_g": 14, "primary": {}},
    "e": {"status": "descriptive", "n_paths": 6},
    "robustness": {},
}


def _run_dir(tmp_path, experiment, band, name=None):
    """A registrable run directory: provenance plus every required artifact."""
    run_dir = tmp_path / (name or f"{experiment}_{band or 'xband'}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps({
        "git": {"commit": "0123456789ab"}, "config": {"run": {"seed": 20260721}},
    }), encoding="utf-8")
    metrics = json.dumps(_MINIMAL_METRICS.get(experiment, {}))
    for name_ in assembly.required_artifacts(experiment, band):
        if name_.startswith("metrics_") and name_.endswith(".json"):
            (run_dir / name_).write_text(metrics, encoding="utf-8")
        elif name_.endswith(".json"):
            (run_dir / name_).write_text("{}", encoding="utf-8")
        elif name_ == "per_subject_g.csv":
            (run_dir / name_).write_text(
                "subject,n_sessions,mae_10,mae_77,mae_equal_weight,mae_fused,"
                "difference_fused_minus_10\n1,4,0.4,0.6,0.45,0.43,0.03\n", encoding="utf-8")
        elif name_.startswith("exclusions_"):
            (run_dir / name_).write_text("band,outer_fold,test_subject,reason,detail\n",
                                         encoding="utf-8")
        elif name_ == "robustness_summary.csv":
            (run_dir / name_).write_text(
                "experiment,band,arm_or_contrast,original_point,range_low,range_high,"
                "range_label,status,ci_method\n", encoding="utf-8")
        else:
            (run_dir / name_).write_text("col\n", encoding="utf-8")
    return run_dir


# --------------------------------------------------------------- the pointer mechanism


def test_the_pointer_round_trips_an_absolute_run_directory(tmp_path):
    run_dir = tmp_path / "some_run"
    run_dir.mkdir()
    pointer = write_run_dir_pointer(tmp_path / "sources" / "exp_a_10.txt", run_dir)
    assert pointer.is_file()
    assert read_run_dir_pointer(pointer) == run_dir.resolve()
    assert Path(pointer.read_text(encoding="utf-8").strip()).is_absolute()


def test_the_pointer_write_leaves_no_temporary_behind(tmp_path):
    run_dir = tmp_path / "some_run"
    run_dir.mkdir()
    pointer = write_run_dir_pointer(tmp_path / "exp.txt", run_dir)
    assert not list(pointer.parent.glob("*.tmp"))


def test_an_empty_or_missing_pointer_fails_closed(tmp_path):
    with pytest.raises(ProvenanceError, match="no run-directory pointer"):
        read_run_dir_pointer(tmp_path / "absent.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="did not finish"):
        read_run_dir_pointer(empty)


def test_a_pointer_at_a_nonexistent_directory_fails_closed(tmp_path):
    pointer = tmp_path / "stale.txt"
    pointer.write_text(str(tmp_path / "gone") + "\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="not a directory"):
        read_run_dir_pointer(pointer)


def test_every_experiment_driver_can_hand_its_run_directory_forward():
    """The explicit run map is only buildable if each job can name its own directory. A driver
    without `--run-dir-out` would force manifest construction back to eyeballing a log."""
    for name in DRIVERS:
        text = (REPO_ROOT / "experiments" / name).read_text(encoding="utf-8")
        assert "--run-dir-out" in text, name
        assert "write_run_dir_pointer" in text, name


def test_every_experiment_wrapper_passes_run_dir_out_through():
    for name in WRAPPERS:
        text = (REPO_ROOT / "scripts" / "ibex" / name).read_text(encoding="utf-8")
        assert "RUN_DIR_OUT=${RUN_DIR_OUT:-}" in text, name
        assert '--run-dir-out "$RUN_DIR_OUT"' in text, name
        assert '"${OUT[@]}"' in text, name


def test_the_pointer_is_written_only_after_a_successful_run():
    """A crashed job must leave no pointer, so manifest construction fails closed rather than
    registering a half-written directory. Structural: the call sits after the reporting call,
    on the success path only."""
    for name in DRIVERS:
        text = (REPO_ROOT / "experiments" / name).read_text(encoding="utf-8")
        write_at = text.index("write_run_dir_pointer(args.run_dir_out")
        assert text.index("run_and_report", 0) < write_at or "robustness" in name, name
        # and it is guarded, so an omitted flag writes nothing
        assert "if args.run_dir_out:" in text, name


# ------------------------------------------------------------------- flag validation


def test_the_run_manifest_path_is_required():
    with pytest.raises(SystemExit):
        rsa.main([])


def test_out_is_required_unless_validate_only(tmp_path):
    with pytest.raises(SystemExit):
        rsa.main(["--run-manifest", str(tmp_path / "m.json")])


def test_a_malformed_or_unknown_source_spec_is_refused(tmp_path):
    manifest = str(tmp_path / "m.json")
    with pytest.raises(SystemExit, match="EXPERIMENT"):
        rsa.main(["--run-manifest", manifest, "--validate-only", "--run", "a:10ghz"])
    with pytest.raises(SystemExit, match="unknown experiment"):
        rsa.main(["--run-manifest", manifest, "--validate-only", "--run", "zzz:10ghz=/tmp"])


def test_with_no_sources_and_no_manifest_there_is_nothing_to_assemble(tmp_path):
    with pytest.raises(SystemExit, match="never discovers runs"):
        rsa.main(["--run-manifest", str(tmp_path / "absent.json"), "--validate-only"])


def test_the_entrypoint_never_globs_for_a_run():
    text = Path(rsa.__file__).read_text(encoding="utf-8")
    body = text.split('"""', 2)[2]
    for forbidden in ("glob(", "rglob(", "iterdir(", "latest"):
        assert forbidden not in body, forbidden


# --------------------------------------------------------- the two launch-matrix calls


def test_validate_only_checks_everything_and_publishes_nothing(tmp_path, capsys):
    run_dir = _run_dir(tmp_path, "d", "10ghz")
    manifest = tmp_path / "run_manifest.json"
    out = tmp_path / "out"

    assert rsa.main(["--run-manifest", str(manifest), "--run", f"d:10ghz={run_dir}",
                     "--validate-only"]) == 0

    assert manifest.is_file()                       # the map itself IS written
    assert not out.exists()                         # ... but no numerical summary is
    printed = capsys.readouterr().out
    assert "validate-only OK" in printed
    assert "no numerical summary written" in printed


def test_the_second_call_reads_the_same_manifest_and_writes_every_table(tmp_path):
    run_dir = _run_dir(tmp_path, "d", "10ghz")
    manifest = tmp_path / "run_manifest.json"
    out = tmp_path / "out"

    assert rsa.main(["--run-manifest", str(manifest), "--run", f"d:10ghz={run_dir}",
                     "--validate-only"]) == 0
    before = manifest.read_bytes()
    # the second call names NO sources — it reads the map the first call wrote
    assert rsa.main(["--run-manifest", str(manifest), "--out", str(out)]) == 0

    assert manifest.read_bytes() == before          # re-reading must not rewrite it
    for name in ("run_manifest.json", "headline_metrics.csv", "per_subject_results.csv",
                 "paired_comparisons.csv", "analysis_exclusions.csv", "metrics_milestone10.json"):
        assert (out / name).is_file(), name


def test_a_source_that_changed_after_registration_stops_the_second_call(tmp_path):
    run_dir = _run_dir(tmp_path, "d", "10ghz")
    manifest = tmp_path / "run_manifest.json"
    assert rsa.main(["--run-manifest", str(manifest), "--run", f"d:10ghz={run_dir}",
                     "--validate-only"]) == 0

    (run_dir / "composite_10ghz.csv").write_text("col\ntampered\n", encoding="utf-8")
    with pytest.raises(assembly.AssemblyError, match="changed after registration"):
        rsa.main(["--run-manifest", str(manifest), "--out", str(tmp_path / "out")])


def test_sources_can_be_supplied_as_pointer_files_written_by_a_completed_job(tmp_path):
    """The launch-matrix path: each wrapper saves its run directory, and assembly reads those
    files. Still explicit — the mapping from experiment to pointer is named on the command
    line; only the directory itself travels in the file."""
    run_dir = _run_dir(tmp_path, "g", None)
    pointer = write_run_dir_pointer(tmp_path / "sources" / "exp_g.txt", run_dir)
    manifest = tmp_path / "run_manifest.json"

    assert rsa.main(["--run-manifest", str(manifest), "--run-from-file", f"g={pointer}",
                     "--validate-only"]) == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert set(payload["runs"]) == {"g"}
    assert payload["runs"]["g"]["run_dir"] == str(run_dir.resolve())


def test_a_stale_pointer_stops_assembly_rather_than_registering_it(tmp_path):
    pointer = tmp_path / "exp_e.txt"
    pointer.write_text(str(tmp_path / "never_ran") + "\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="not a directory"):
        rsa.main(["--run-manifest", str(tmp_path / "m.json"), "--validate-only",
                  "--run-from-file", f"e:10ghz={pointer}"])


def test_rebuilding_with_new_sources_replaces_the_map(tmp_path):
    first = _run_dir(tmp_path, "d", "10ghz")
    second = _run_dir(tmp_path, "d", "77ghz")
    manifest = tmp_path / "run_manifest.json"

    rsa.main(["--run-manifest", str(manifest), "--run", f"d:10ghz={first}", "--validate-only"])
    assert set(json.loads(manifest.read_text(encoding="utf-8"))["runs"]) == {"d_10ghz"}
    rsa.main(["--run-manifest", str(manifest), "--run", f"d:10ghz={first}",
              "--run", f"d:77ghz={second}", "--validate-only"])
    assert set(json.loads(manifest.read_text(encoding="utf-8"))["runs"]) == {"d_10ghz", "d_77ghz"}


# ------------------------------------------------------------------ the IBEX wrapper


def test_the_sbatch_maps_to_the_two_documented_calls():
    text = SBATCH.read_text(encoding="utf-8")
    assert "experiments/run_stats_assembly.py" in text
    assert '--run-manifest "$RUN_MANIFEST"' in text
    assert "MODE=(--validate-only)" in text
    assert 'MODE=(--out "$OUT_DIR")' in text
    assert "set -euo pipefail" in text


def test_the_sbatch_refuses_to_run_without_an_explicit_manifest():
    text = SBATCH.read_text(encoding="utf-8")
    assert "RUN_MANIFEST=${RUN_MANIFEST:?" in text
    assert "never discovers runs" in text


def test_the_sbatch_header_is_the_smallest_in_the_repo_because_it_fits_nothing():
    """Assembly reads saved artifacts and writes CSV/JSON. Anything larger would be cargo."""
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=2" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "#SBATCH --mem=8G" in text
    assert "FITS NOTHING" in text


@pytest.mark.parametrize("path", [SBATCH])
def test_every_shell_artifact_parses_with_bash_dash_n(path):
    """Bytes with LF forced: `text=True` would rewrite \\n to \\r\\n on this repo, and this file
    carries a ${VAR:?message} guard, which is the quoting class of bug that broke here before."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on PATH")
    script = path.read_bytes().replace(b"\r\n", b"\n")
    result = subprocess.run(["bash", "-n", "-"], input=script, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


@pytest.mark.parametrize("name", WRAPPERS)
def test_every_patched_wrapper_still_parses_with_bash_dash_n(name):
    if shutil.which("bash") is None:
        pytest.skip("bash not available on PATH")
    script = (REPO_ROOT / "scripts" / "ibex" / name).read_bytes().replace(b"\r\n", b"\n")
    result = subprocess.run(["bash", "-n", "-"], input=script, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
