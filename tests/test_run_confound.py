"""T-M10-entrypoints (Exp F half): `experiments/run_confound.py` + `run_exp_f.sbatch`.

Flag validation — including that `--exp-a-sources` is required and never discovered — the
mechanism-only reporting boundary, and static checks on the IBEX wrapper. The analysis itself is
covered by `test_exp_f.py`; here `exp_f.run_and_report_f` is monkeypatched so the CLI's own
wiring is tested fast, against a REAL `record_run` call.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_f

import experiments.run_confound as rc

REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH = REPO_ROOT / "scripts" / "ibex" / "run_exp_f.sbatch"

BASE_ARGS = [
    "--config", "configs/exp_a_regression.yaml",
    "--config", "configs/exp_f.yaml",
    "--config", "configs/stats.yaml",
]


def _sessions(band="10ghz", n_subjects=8):
    out = []
    for s in range(1, n_subjects + 1):
        baseline = 70.0 + s
        for i in range(5):
            mass = baseline - 0.25 * i
            out.append({
                "subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
                "rel_path": f"{band}_s{s}_{SESSION_NAMES[i]}.mat", "frame_ids": [0, 1, 2],
                "delta_m_pct": (mass - baseline) / baseline * 100.0,
                "delta_m_kg": mass - baseline,
            })
    return out


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


def _sources_file(tmp_path):
    path = tmp_path / "exp_a_sources.json"
    path.write_text(json.dumps({"schema_version": "exp_a_sources_v1"}), encoding="utf-8")
    return path


def _patch(monkeypatch, sessions, data_dir, captured):
    monkeypatch.setattr(rc.exp_f, "build_sessions_f", lambda config, band: list(sessions))
    monkeypatch.setattr(rc, "_build_manifest_qc",
                        lambda config, band: _manifest(data_dir, sessions))

    def fake_run(config, band, band_sessions, store_dir, run_dir, *, mode, analysis_commit,
                 exp_a_sources, n_workers=1):
        captured.update({"mode": mode, "band": band, "n_sessions": len(band_sessions),
                         "subjects": sorted({s["subject"] for s in band_sessions}),
                         "run_dir": run_dir, "store_dir": store_dir,
                         "exp_a_sources": str(exp_a_sources)})
        return {"run_log": run_dir / f"run_log_exp_f_{band}.json"}

    monkeypatch.setattr(rc.exp_f, "run_and_report_f", fake_run)


# ------------------------------------------------------------------- flag validation


def test_exactly_one_of_subset_or_full_cohort_is_required(tmp_path):
    sources = _sources_file(tmp_path)
    with pytest.raises(SystemExit):
        rc.main(BASE_ARGS + ["--exp-a-sources", str(sources)])
    with pytest.raises(SystemExit):
        rc.main(BASE_ARGS + ["--exp-a-sources", str(sources),
                             "--subset", "6subjects", "--full-cohort"])


def test_the_exp_a_sources_pointer_is_required(tmp_path):
    """§1.3: F consumes an EXPLICIT approved pointer. There is no default and no discovery, so
    omitting it must fail at the CLI rather than fall back to a glob."""
    with pytest.raises(SystemExit):
        rc.main(BASE_ARGS + ["--full-cohort"])


def test_the_entrypoint_never_globs_for_a_sources_file():
    """A structural check on the source: any `glob`/`latest` discovery of the Exp-A run would
    defeat the gate, so the entrypoint must contain none."""
    text = Path(rc.__file__).read_text(encoding="utf-8")
    body = text.split('"""', 2)[2]
    for forbidden in ("glob(", "rglob(", "iterdir(", "latest"):
        assert forbidden not in body, forbidden


# --------------------------------------------------------- config loading and provenance


def test_the_config_list_is_loaded_in_order(tmp_path, monkeypatch):
    overlay, data_10, _ = _overlay(tmp_path)
    _patch(monkeypatch, _sessions(), data_10, {})
    loaded = []
    real_load = rc.load_config
    monkeypatch.setattr(rc, "load_config",
                        lambda *paths: loaded.append(paths) or real_load(*paths))

    assert rc.main(BASE_ARGS + ["--config", str(overlay),
                                "--exp-a-sources", str(_sources_file(tmp_path)),
                                "--band", "10ghz", "--full-cohort"]) == 0
    assert loaded == [("configs/exp_a_regression.yaml", "configs/exp_f.yaml",
                       "configs/stats.yaml", str(overlay))]


def test_smoke_keeps_the_six_lowest_evaluable_subjects(tmp_path, monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _sessions(n_subjects=8), data_10, captured)

    assert rc.main(BASE_ARGS + ["--config", str(overlay),
                                "--exp-a-sources", str(_sources_file(tmp_path)),
                                "--band", "10ghz", "--subset", "6subjects"]) == 0
    assert captured["mode"] == "smoke"
    assert captured["subjects"] == [1, 2, 3, 4, 5, 6]
    assert captured["n_sessions"] == 30            # 6 subjects x S0-S4
    assert not list(results.rglob("contrasts_f_10ghz.csv"))
    assert not list(results.rglob("metrics_exp_f_10ghz.json"))


def test_the_run_records_the_band_the_sources_pointer_and_the_hr_status(tmp_path, monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    captured = {}
    _patch(monkeypatch, _sessions(n_subjects=8), data_10, captured)
    sources = _sources_file(tmp_path)

    assert rc.main(BASE_ARGS + ["--config", str(overlay), "--exp-a-sources", str(sources),
                                "--band", "10ghz", "--full-cohort"]) == 0

    run_dirs = sorted((results / "runs").iterdir())
    payload = json.loads((run_dirs[0] / "provenance.json").read_text(encoding="utf-8"))
    assert payload["extra"]["stage"] == "exp-f-full"
    assert payload["extra"]["band"] == "10ghz"
    assert payload["extra"]["n_eval"] == 8
    assert payload["extra"]["n_sessions"] == 40
    assert payload["extra"]["exp_a_sources"] == str(sources)
    # the not-estimable answer is in the provenance, not only in the artifacts
    assert payload["extra"]["heart_rate_status"] == exp_f.HR_STATUS
    assert len(payload["folds"]) == 8
    assert captured["exp_a_sources"] == str(sources)


def test_the_77ghz_run_requires_the_77ghz_data_root(tmp_path, monkeypatch):
    from dehyd.config import ConfigError

    overlay, data_10, _ = _overlay(tmp_path, with_77=False)
    _patch(monkeypatch, _sessions(band="77ghz"), data_10, {})
    with pytest.raises(ConfigError, match="data_77ghz_dir"):
        rc.main([
            "--config", "configs/exp_a_regression_77ghz.yaml", "--config", "configs/exp_f.yaml",
            "--config", "configs/stats.yaml", "--config", str(overlay),
            "--exp-a-sources", str(_sources_file(tmp_path)), "--band", "77ghz", "--full-cohort",
        ])


def test_the_printed_header_says_heart_rate_is_not_estimable(tmp_path, monkeypatch, capsys):
    overlay, data_10, _ = _overlay(tmp_path)
    _patch(monkeypatch, _sessions(), data_10, {})
    assert rc.main(BASE_ARGS + ["--config", str(overlay),
                                "--exp-a-sources", str(_sources_file(tmp_path)),
                                "--band", "10ghz", "--full-cohort"]) == 0
    out = capsys.readouterr().out
    assert exp_f.HR_STATUS in out
    assert "NOT an HR adjustment" in out


# ------------------------------------------------------------------ the IBEX wrapper


def test_the_sbatch_maps_to_the_exact_documented_payload():
    text = SBATCH.read_text(encoding="utf-8")
    assert "experiments/run_confound.py" in text
    for flag in ("--config \"$CFG\"", "--config configs/exp_f.yaml",
                 "--config configs/stats.yaml", "--config configs/ibex.yaml",
                 "--exp-a-sources \"$EXP_A_SOURCES\"", "--band \"$BAND\""):
        assert flag in text, flag
    assert "CFG=configs/exp_a_regression_77ghz.yaml" in text
    assert "SEL=(--full-cohort)" in text and "SEL=(--subset 6subjects)" in text
    assert "set -euo pipefail" in text


def test_the_sbatch_refuses_to_launch_without_an_explicit_sources_pointer():
    """`${EXP_A_SOURCES:?...}` fails the job at submit time rather than letting the run start
    and discover the pointer is missing after validate_store has already burned the time."""
    text = SBATCH.read_text(encoding="utf-8")
    assert "EXP_A_SOURCES=${EXP_A_SOURCES:?" in text
    assert "never discovers it" in text


def test_the_sbatch_header_is_sized_from_measurement_not_cloned_from_exp_g():
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --time=01:00:00" in text
    assert "#SBATCH --mem=32G" in text
    assert "--cpus-per-task=32" not in text and "--time=24:00:00" not in text
    assert "19.7 ms" in text                    # the measured per-fit cost
    assert "41 fits" in text or "41 fits" in text.replace("=", " ")
    assert "validate_store" in text             # what the hour is actually for


@pytest.mark.parametrize("path", [SBATCH])
def test_every_shell_artifact_parses_with_bash_dash_n(path):
    """A cheap syntax gate. Bytes with LF forced: on this repo `text=True` would rewrite \\n to
    \\r\\n, and a CRLF after a line-continuation backslash breaks bash's parser. It also catches
    the quoting class of bug this repo's history had — an apostrophe inside a ${VAR:?message}
    breaks the parser even within double quotes, and this file carries such a message."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on PATH")
    script = path.read_bytes().replace(b"\r\n", b"\n")
    result = subprocess.run(["bash", "-n", "-"], input=script, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
