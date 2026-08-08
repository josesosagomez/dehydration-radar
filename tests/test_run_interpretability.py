"""T-M10-entrypoints (Exp E half): `experiments/run_interpretability.py` + `run_exp_e.sbatch`.

Flag validation, the one-config-list-per-band rule, the mechanism-only reporting boundary, and
static checks on the IBEX wrapper — its argv mapping, its MEASURED sizing note, and that
`bash -n` parses it. The attribution itself is covered by `test_exp_e.py`; here
`exp_e.run_and_report_e` is monkeypatched so the CLI's own wiring is tested fast, against a
REAL `record_run` call.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from dehyd.data.sessions import SESSION_NAMES

import experiments.run_interpretability as ri

REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH = REPO_ROOT / "scripts" / "ibex" / "run_exp_e.sbatch"

BASE_ARGS = [
    "--config", "configs/exp_a_regression.yaml",
    "--config", "configs/exp_e.yaml",
    "--config", "configs/stats.yaml",
]


def _sessions(band="10ghz", n_subjects=8, session_indices=(1, 2, 3, 4)):
    """An Exp-B spine shape: S0 already excluded, plus the manifest fields `record_run` needs."""
    return [
        {"subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
         "rel_path": f"{band}_s{s}_{SESSION_NAMES[i]}.mat",
         "frame_ids": [0, 1, 2], "delta_m_pct": float(-0.3 * i - 0.05 * s)}
        for s in range(1, n_subjects + 1) for i in session_indices
    ]


def _manifest(data_dir, sessions):
    rows = []
    for s in sessions:
        (data_dir / s["rel_path"]).write_bytes(f"radar {s['rel_path']}".encode())
        rows.append({"subject": s["subject"], "session_idx": s["session_idx"],
                     "rel_path": s["rel_path"]})
    return pd.DataFrame(rows)


def _overlay(tmp_path, *, with_77=True):
    data_10, data_77, results = tmp_path / "data10", tmp_path / "data77", tmp_path / "results"
    for directory in (data_10, data_77, results):
        directory.mkdir(exist_ok=True)
    (tmp_path / "weights.xlsx").write_bytes(b"pretend workbook")
    path = tmp_path / "paths_overlay.yaml"
    path.write_text("\n".join([
        "paths:",
        f"  data_10ghz_dir: {data_10.as_posix()}",
        f"  data_77ghz_dir: {data_77.as_posix() if with_77 else 'null'}",
        f"  weight_xlsx: {(tmp_path / 'weights.xlsx').as_posix()}",
        f"  results_dir: {results.as_posix()}",
    ]) + "\n", encoding="utf-8")
    return path, data_10, results


def _patch(monkeypatch, sessions, data_dir, captured):
    monkeypatch.setattr(ri.exp_b, "build_sessions_b", lambda config, band: list(sessions))
    monkeypatch.setattr(ri, "_build_manifest_qc",
                        lambda config, band: _manifest(data_dir, sessions))

    def fake_run(config, band, band_sessions, store_dir, run_dir, *, mode, analysis_commit,
                 n_workers=1):
        captured.update({"mode": mode, "band": band, "n_sessions": len(band_sessions),
                         "subjects": sorted({s["subject"] for s in band_sessions}),
                         "run_dir": run_dir, "store_dir": store_dir})
        return {"run_log": run_dir / f"run_log_exp_e_{band}.json"}

    monkeypatch.setattr(ri.exp_e, "run_and_report_e", fake_run)


# ------------------------------------------------------------------- flag validation


def test_exactly_one_of_subset_or_full_cohort_is_required():
    with pytest.raises(SystemExit):
        ri.main(BASE_ARGS)
    with pytest.raises(SystemExit):
        ri.main(BASE_ARGS + ["--subset", "6subjects", "--full-cohort"])


def test_an_unknown_band_is_refused():
    with pytest.raises(SystemExit):
        ri.main(BASE_ARGS + ["--band", "24ghz", "--full-cohort"])


# ---------------------------------------------------- one config list, one band, per run


def test_the_config_list_is_loaded_in_order(tmp_path, monkeypatch):
    """Unlike Exp G, E takes ONE `--config` list: the two bands are independent analyses of
    two different filter banks and share nothing at run time."""
    overlay, data_10, _ = _overlay(tmp_path)
    _patch(monkeypatch, _sessions(), data_10, {})

    loaded = []
    real_load = ri.load_config
    monkeypatch.setattr(ri, "load_config",
                        lambda *paths: loaded.append(paths) or real_load(*paths))

    assert ri.main(BASE_ARGS + ["--config", str(overlay), "--band", "10ghz", "--full-cohort"]) == 0
    assert loaded == [("configs/exp_a_regression.yaml", "configs/exp_e.yaml",
                       "configs/stats.yaml", str(overlay))]


def test_the_77ghz_run_requires_the_77ghz_data_root(tmp_path, monkeypatch):
    from dehyd.config import ConfigError

    overlay, data_10, _ = _overlay(tmp_path, with_77=False)
    _patch(monkeypatch, _sessions(band="77ghz"), data_10, {})
    with pytest.raises(ConfigError, match="data_77ghz_dir"):
        ri.main([
            "--config", "configs/exp_a_regression_77ghz.yaml", "--config", "configs/exp_e.yaml",
            "--config", "configs/stats.yaml", "--config", str(overlay),
            "--band", "77ghz", "--full-cohort",
        ])


# --------------------------------------------------------- mode, subset, provenance


def test_smoke_keeps_the_six_lowest_evaluable_subjects(tmp_path, monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _sessions(n_subjects=8), data_10, captured)

    assert ri.main(BASE_ARGS + ["--config", str(overlay), "--band", "10ghz",
                                "--subset", "6subjects"]) == 0
    assert captured["mode"] == "smoke"
    assert captured["subjects"] == [1, 2, 3, 4, 5, 6]
    assert captured["n_sessions"] == 24
    # the CLI itself writes no numerical artifact; the boundary lives in the library call
    assert not list(results.rglob("importance_folds_10ghz.csv"))
    assert not list(results.rglob("metrics_exp_e_10ghz.json"))


def test_the_run_records_the_band_and_a_real_fold_manifest(tmp_path, monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _sessions(n_subjects=8), data_10, captured)

    assert ri.main(BASE_ARGS + ["--config", str(overlay), "--band", "10ghz",
                                "--full-cohort"]) == 0

    run_dirs = sorted((results / "runs").iterdir())
    payload = json.loads((run_dirs[0] / "provenance.json").read_text(encoding="utf-8"))
    assert payload["extra"]["stage"] == "exp-e-full"
    assert payload["extra"]["band"] == "10ghz"
    assert payload["extra"]["n_eval"] == 8
    assert payload["extra"]["n_sessions"] == 32
    assert len(payload["folds"]) == 8            # a real fold-role manifest, not an empty slot
    assert captured["run_dir"] == run_dirs[0]
    assert captured["store_dir"] == results


def test_the_fixed_anchor_is_reported_before_the_run(tmp_path, monkeypatch, capsys):
    """The printed header names the FIXED model, so a log can be checked against the frozen
    anchor without opening an artifact."""
    overlay, data_10, _ = _overlay(tmp_path)
    _patch(monkeypatch, _sessions(), data_10, {})

    assert ri.main(BASE_ARGS + ["--config", str(overlay), "--band", "10ghz", "--full-cohort"]) == 0
    out = capsys.readouterr().out
    assert "FIXED" in out and "'tiling': 'T1'" in out and "alpha=1.0" in out
    assert "never the best A/B model" in out


# ------------------------------------------------------------------ the IBEX wrapper


def test_the_sbatch_maps_to_the_exact_documented_payload():
    text = SBATCH.read_text(encoding="utf-8")
    assert "experiments/run_interpretability.py" in text
    for flag in ("--config \"$CFG\"", "--config configs/exp_e.yaml",
                 "--config configs/stats.yaml", "--config configs/ibex.yaml",
                 "--band \"$BAND\""):
        assert flag in text, flag
    assert 'CFG=configs/exp_a_regression_77ghz.yaml' in text     # the 77 GHz half is reachable
    assert 'SEL=(--full-cohort)' in text and 'SEL=(--subset 6subjects)' in text
    assert "set -euo pipefail" in text          # a nonzero exit must fail the job


def test_the_sbatch_header_is_sized_from_measurement_not_cloned_from_exp_g():
    """E is the CHEAP milestone-10 job: `1 + n_paths` deterministic ridge fits per fold and no
    search at all. Copying Exp G's 32-core / 24 h header would reserve ~700 core-hours for
    minutes of work, so the header must be small AND must show the arithmetic."""
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --time=01:00:00" in text
    assert "#SBATCH --mem=32G" in text
    # not Exp G's header, which is the specific mistake this file exists to avoid
    assert "--cpus-per-task=32" not in text and "--time=24:00:00" not in text

    assert "742 paths" in text and "424 paths" in text          # the measured path counts
    assert "16.7 ms per refit" in text                          # the measured unit cost
    assert "12.4 s per fold" in text and "7.1 s per fold" in text
    assert "validate_store" in text                             # what the hour is actually for


@pytest.mark.parametrize("path", [SBATCH])
def test_every_shell_artifact_parses_with_bash_dash_n(path):
    """A cheap syntax gate. Bytes with LF forced: on this repo `text=True` would rewrite \\n to
    \\r\\n, and a CRLF after a line-continuation backslash breaks bash's parser."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on PATH")
    script = path.read_bytes().replace(b"\r\n", b"\n")
    result = subprocess.run(["bash", "-n", "-"], input=script, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
