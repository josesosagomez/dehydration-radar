"""T-M10-entrypoints (Exp G half): `experiments/run_fusion.py` + `run_exp_g.sbatch`.

Flag validation, the two-config loading rule the whole experiment rests on, the mechanism-only
reporting boundary, and static checks on the IBEX wrapper (its argv mapping, its sizing note,
and that `bash -n` parses it). The fusion itself is covered by `test_exp_g.py`; here
`exp_g.run_and_report` is monkeypatched so the CLI's OWN wiring is tested fast, against a REAL
`record_run` call.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from dehyd.data.sessions import SESSION_NAMES

import experiments.run_fusion as rf

REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH = REPO_ROOT / "scripts" / "ibex" / "run_exp_g.sbatch"

BAND_ARGS = [
    "--config-10", "configs/exp_a_regression.yaml",
    "--config-77", "configs/exp_a_regression_77ghz.yaml",
    "--shared-config", "configs/exp_g_fusion.yaml",
    "--shared-config", "configs/stats.yaml",
]


def _matched(n_subjects=8, n_sessions=5):
    return [
        {"subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
         "delta_m_pct": -0.3 * i, "n_frames_10": 4, "n_frames_77": 6}
        for s in range(1, n_subjects + 1)
        for i in range(n_sessions)
    ]


def _band_sessions(matched, band):
    return [
        {"subject": c["subject"], "session_idx": c["session_idx"],
         "session_name": c["session_name"],
         "rel_path": f"{band}_s{c['subject']}_{c['session_name']}.mat",
         "frame_ids": list(range(c[f"n_frames_{band[:2]}"])), "delta_m_pct": c["delta_m_pct"]}
        for c in matched
    ]


def _manifest(data_dir, sessions):
    rows = []
    for s in sessions:
        (data_dir / s["rel_path"]).write_bytes(f"radar {s['rel_path']}".encode())
        rows.append({"subject": s["subject"], "session_idx": s["session_idx"],
                     "rel_path": s["rel_path"]})
    return pd.DataFrame(rows)


def _overlay(tmp_path):
    """A shared paths overlay — applied to BOTH bands, which is also what the real run does."""
    data_10, data_77, results = (tmp_path / "data10", tmp_path / "data77", tmp_path / "results")
    for directory in (data_10, data_77, results):
        directory.mkdir(exist_ok=True)
    (tmp_path / "weights.xlsx").write_bytes(b"pretend workbook")
    path = tmp_path / "paths_overlay.yaml"
    path.write_text("\n".join([
        "paths:",
        f"  data_10ghz_dir: {data_10.as_posix()}",
        f"  data_77ghz_dir: {data_77.as_posix()}",
        f"  weight_xlsx: {(tmp_path / 'weights.xlsx').as_posix()}",
        f"  results_dir: {results.as_posix()}",
    ]) + "\n", encoding="utf-8")
    return path, data_10, results


def _patch(monkeypatch, matched, data_dir, captured):
    sessions_10 = _band_sessions(matched, "10ghz")
    sessions_77 = _band_sessions(matched, "77ghz")
    monkeypatch.setattr(
        rf.exp_g, "build_matched_population",
        lambda c10, c77: (list(matched), list(sessions_10), list(sessions_77), []),
    )
    monkeypatch.setattr(rf, "_build_manifest_qc",
                        lambda config: _manifest(data_dir, sessions_10))

    def fake_run(config_10, config_77, matched_cells, s10, s77, unmatched, store_dir, run_dir,
                 *, mode, analysis_commit, n_workers=1):
        captured.update({"mode": mode, "n_matched": len(matched_cells), "n_10": len(s10),
                         "n_77": len(s77), "run_dir": run_dir, "store_dir": store_dir,
                         "seed_set_10": tuple(config_10.run.seed_set),
                         "seed_set_77": tuple(config_77.run.seed_set)})
        return {"run_log": run_dir / "run_log_exp_g.json"}

    monkeypatch.setattr(rf.exp_g, "run_and_report", fake_run)


# ------------------------------------------------------------------- flag validation


def test_exactly_one_of_subset_or_full_cohort_is_required():
    with pytest.raises(SystemExit):
        rf.main(BAND_ARGS)
    with pytest.raises(SystemExit):
        rf.main(BAND_ARGS + ["--subset", "6subjects", "--full-cohort"])


def test_both_band_configs_are_required():
    with pytest.raises(SystemExit):
        rf.main(["--config-10", "configs/exp_a_regression.yaml", "--full-cohort"])
    with pytest.raises(SystemExit):
        rf.main(["--config-77", "configs/exp_a_regression_77ghz.yaml", "--full-cohort"])


# ------------------------------------------------------- the two-config loading rule


def test_the_two_band_configs_are_loaded_separately_and_never_merged(tmp_path, monkeypatch):
    """The heart of the entrypoint. Each band keeps its own front end, WST section and search
    space; only the shared overlays are applied to both. A merged config would describe
    neither band — and would silently give the 77 GHz run a 10 GHz search space."""
    overlay, data_10, _ = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _matched(), data_10, captured)

    loaded = []
    real_load = rf.load_config
    monkeypatch.setattr(rf, "load_config",
                        lambda *paths: loaded.append(paths) or real_load(*paths))

    assert rf.main(BAND_ARGS + ["--shared-config", str(overlay), "--full-cohort"]) == 0
    assert len(loaded) == 2
    assert loaded[0][0] == "configs/exp_a_regression.yaml"
    assert loaded[1][0] == "configs/exp_a_regression_77ghz.yaml"
    # the shared overlays trail BOTH, in the order given, so paths/stats/exp_g win in both
    assert loaded[0][1:] == loaded[1][1:]
    assert loaded[0][1:] == ("configs/exp_g_fusion.yaml", "configs/stats.yaml", str(overlay))


def test_band_configs_that_disagree_on_a_shared_constant_are_refused(tmp_path, monkeypatch):
    overlay, data_10, _ = _overlay(tmp_path)
    _patch(monkeypatch, _matched(), data_10, {})
    drift = tmp_path / "drifted_seed.yaml"
    drift.write_text("run:\n  seed: 1\n  seed_set: [1, 2, 3, 4, 5]\n  device: cpu\n",
                     encoding="utf-8")

    with pytest.raises(rf.exp_g.ExpGError, match="disagree on shared analysis constants"):
        rf.main([
            "--config-10", "configs/exp_a_regression.yaml",
            "--config-77", "configs/exp_a_regression_77ghz.yaml", "--config-77", str(drift),
            "--shared-config", "configs/exp_g_fusion.yaml",
            "--shared-config", "configs/stats.yaml", "--shared-config", str(overlay),
            "--full-cohort",
        ])


def test_a_missing_77ghz_data_root_fails_before_any_work(tmp_path, monkeypatch):
    """G cannot run at all without the 77 GHz root, and the failure must come from the named
    guard rather than from a None reaching the loader."""
    from dehyd.config import ConfigError

    data_10, results = tmp_path / "data10", tmp_path / "results"
    for directory in (data_10, results):
        directory.mkdir()
    (tmp_path / "weights.xlsx").write_bytes(b"pretend workbook")
    overlay = tmp_path / "no77.yaml"
    overlay.write_text("\n".join([
        "paths:",
        f"  data_10ghz_dir: {data_10.as_posix()}",
        "  data_77ghz_dir: null",
        f"  weight_xlsx: {(tmp_path / 'weights.xlsx').as_posix()}",
        f"  results_dir: {results.as_posix()}",
    ]) + "\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="data_77ghz_dir"):
        rf.main(BAND_ARGS + ["--shared-config", str(overlay), "--full-cohort"])


# --------------------------------------------------------- mode, subset, provenance


def test_smoke_keeps_the_six_lowest_matched_subjects_in_both_bands(tmp_path, monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _matched(n_subjects=8, n_sessions=5), data_10, captured)

    assert rf.main(BAND_ARGS + ["--shared-config", str(overlay), "--subset", "6subjects"]) == 0
    assert captured["mode"] == "smoke"
    # the subset is applied to the MATCHED population and to both band row lists identically
    assert captured["n_matched"] == 30 and captured["n_10"] == 30 and captured["n_77"] == 30
    payload = json.loads(
        (sorted((results / "runs").iterdir())[0] / "provenance.json").read_text(encoding="utf-8")
    )
    assert payload["extra"]["stage"] == "exp-g-smoke"
    # the CLI itself writes no numerical artifact; the boundary lives in the library call
    assert not list(results.rglob("predictions_g.csv"))
    assert not list(results.rglob("metrics_exp_g.json"))


def test_the_run_records_both_bands_and_the_matched_population(tmp_path, monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _matched(), data_10, captured)

    assert rf.main(BAND_ARGS + ["--shared-config", str(overlay), "--full-cohort"]) == 0

    run_dirs = sorted((results / "runs").iterdir())
    payload = json.loads((run_dirs[0] / "provenance.json").read_text(encoding="utf-8"))
    assert payload["extra"]["stage"] == "exp-g-full"
    assert payload["extra"]["band"] == "10ghz+77ghz"
    assert len(payload["extra"]["config_77_sha256"]) == 64      # the 77 GHz config, pinned
    assert payload["extra"]["n_eval"] == 8
    assert payload["extra"]["n_sessions"] == 40
    assert payload["extra"]["population"]["n_matched_cells"] == 40
    assert len(payload["folds"]) == 8            # a real fold-role manifest, not an empty slot
    assert captured["run_dir"] == run_dirs[0]
    assert captured["store_dir"] == results


def test_full_cohort_refuses_the_single_smoke_seed(tmp_path, monkeypatch):
    """`ExpGConfig.seed_pairing` pairs seed label k across the two bands, so a reduced seed set
    would silently change what "the five paired seed labels" means."""
    overlay, data_10, _ = _overlay(tmp_path)
    _patch(monkeypatch, _matched(), data_10, {})

    with pytest.raises(SystemExit, match="frozen at"):
        rf.main(BAND_ARGS + ["--shared-config", str(overlay),
                             "--shared-config", "configs/smoke.yaml", "--full-cohort"])


def test_the_smoke_may_use_the_reduced_seed_set(tmp_path, monkeypatch):
    overlay, data_10, _ = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _matched(), data_10, captured)

    assert rf.main(BAND_ARGS + ["--shared-config", str(overlay),
                                "--shared-config", "configs/smoke.yaml",
                                "--subset", "6subjects"]) == 0
    assert captured["seed_set_10"] == (1,) and captured["seed_set_77"] == (1,)


# ------------------------------------------------------------------ the IBEX wrapper


def test_the_sbatch_maps_to_the_exact_documented_payload():
    text = SBATCH.read_text(encoding="utf-8")
    assert "experiments/run_fusion.py" in text
    for flag in (
        "--config-10 configs/exp_a_regression.yaml",
        "--config-77 configs/exp_a_regression_77ghz.yaml",
        "--shared-config configs/exp_g_fusion.yaml",
        "--shared-config configs/stats.yaml",
        "--shared-config configs/ibex.yaml",
    ):
        assert flag in text
    assert 'SEL=(--full-cohort)' in text and 'SEL=(--subset 6subjects)' in text
    assert "set -euo pipefail" in text          # a nonzero exit must fail the job


def test_the_sbatch_header_is_sized_from_the_measured_anchor():
    """Exp G is the milestone's most expensive classical job (~192 core-hours at 16 subjects,
    ~6 h wall on 32 cores). The header must allocate with margin, and must say why."""
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=32" in text
    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --mem=128G" in text
    assert "192 staged selections" in text      # the sizing arithmetic is written down
    assert "HISTORY 2026-07-28" in text         # against the measured anchor, not a guess


@pytest.mark.parametrize("path", [SBATCH])
def test_every_shell_artifact_parses_with_bash_dash_n(path):
    """A cheap syntax gate. Bytes with LF forced: on this repo `text=True` would rewrite \\n to
    \\r\\n, and a CRLF after a line-continuation backslash breaks bash's parser."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on PATH")
    script = path.read_bytes().replace(b"\r\n", b"\n")
    result = subprocess.run(["bash", "-n", "-"], input=script, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
