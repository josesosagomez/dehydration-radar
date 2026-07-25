"""Provenance recording: hashes, canonical serialization, collision policy.

All tests write into tmp_path (via a Config whose results_dir points there), never
into the repo — so a test run cannot itself change the git-dirty state that a later
assertion reads.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from dehyd.config import (
    Config,
    PathsConfig,
    PreprocessConfig,
    QCConfig,
    RunConfig,
    SplitConfig,
    WSTConfig,
)
from dehyd.data.ground_truth import GroundTruth
from dehyd.data.manifest import build_manifest
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval.splits import nested_loso_splits
from dehyd.provenance import ProvenanceError, record_run, sha256_file

N_FAST_TIME = 534
N_CHIRPS = 20


@dataclass(frozen=True)
class FakePaths:
    data_10ghz_dir: Path


@pytest.fixture
def setup(tmp_path):
    """A tiny but complete run: 3 subjects x 5 sessions x 2 frames."""
    data_dir = tmp_path / "10ghz"
    data_dir.mkdir()
    subjects = (1, 2, 3)
    for subject in subjects:
        for session_idx in range(5):
            cube = np.full((N_FAST_TIME, N_CHIRPS, 2), subject + 1j, dtype=np.complex128)
            sio.savemat(
                str(data_dir / f"subject_{subject}_{SESSION_NAMES[session_idx]}.mat"),
                {"framesRadar": cube},
            )

    xlsx = tmp_path / "weights.xlsx"
    xlsx.write_bytes(b"pretend workbook")

    rows, subject_rows = [], []
    for subject in subjects:
        baseline = 80.0 + subject
        for idx in range(5):
            mass = baseline - 0.3 * idx
            rows.append(
                {
                    "subject": subject,
                    "session_idx": idx,
                    "session_name": SESSION_NAMES[idx],
                    "mass_kg": mass,
                    "delta_m_kg": mass - baseline,
                    "delta_m_pct": (mass - baseline) / baseline * 100.0,
                }
            )
        subject_rows.append(
            {
                "subject": subject,
                "age": 30,
                "height_cm": 175.0,
                "baseline_mass_kg": baseline,
                "bmi": baseline / 1.75**2,
            }
        )
    gt = GroundTruth(sessions=pd.DataFrame(rows), subjects=pd.DataFrame(subject_rows))

    config = Config(
        paths=PathsConfig(
            data_10ghz_dir=data_dir,
            weight_xlsx=xlsx,
            results_dir=tmp_path / "results",
        ),
        run=RunConfig(seed=7, seed_set=(1, 2, 3, 4, 5), device="cpu"),
        split=SplitConfig(),
        qc=QCConfig(),
        preprocess=PreprocessConfig(),
        wst=WSTConfig(),
    )
    manifest = build_manifest(FakePaths(data_dir), gt)
    folds = nested_loso_splits(sorted(manifest.subject.unique().tolist()))
    return config, manifest, folds


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ contents


def test_writes_provenance_json(setup):
    config, manifest, folds = setup
    out = record_run(config, manifest, folds)

    assert out.exists()
    assert out.name == "provenance.json"
    assert out.parent.parent.name == "runs"


def test_hashes_every_radar_file_and_the_workbook(setup):
    config, manifest, folds = setup
    payload = load(record_run(config, manifest, folds))

    radar = payload["inputs"]["radar_files"]
    assert len(radar) == 15  # 3 subjects x 5 sessions
    assert all(len(entry["sha256"]) == 64 for entry in radar)

    # The label source must be hashed too, or labels could change undetected.
    assert payload["inputs"]["ground_truth"]["sha256"] == sha256_file(config.paths.weight_xlsx)


def test_records_logical_paths_not_physical(setup):
    config, manifest, folds = setup
    payload = load(record_run(config, manifest, folds))

    for entry in payload["inputs"]["radar_files"]:
        assert not Path(entry["rel_path"]).is_absolute()
        assert ".." not in entry["rel_path"]
        assert entry["rel_path"].endswith(".mat")


def test_records_config_folds_and_environment(setup):
    config, manifest, folds = setup
    payload = load(record_run(config, manifest, folds))

    assert payload["config"]["preprocess"]["butter_order"] == 4
    assert payload["seed"] == 7
    assert payload["seed_set"] == [1, 2, 3, 4, 5]
    assert payload["device"] == "cpu"
    assert payload["manifest"]["n_frames"] == len(manifest)
    assert payload["manifest"]["n_subjects"] == 3
    assert len(payload["folds"]) == 3
    assert payload["packages"]["numpy"] is not None
    assert payload["packages"]["torch"] is not None  # entered the env at milestone 4
    assert "commit" in payload["git"]
    assert payload["platform"]["python"].startswith("3.11")


def test_fold_roles_are_recorded(setup):
    config, manifest, folds = setup
    payload = load(record_run(config, manifest, folds))

    for recorded, fold in zip(payload["folds"], folds, strict=True):
        assert recorded["test_subject"] == fold.test_subject
        assert recorded["train_subjects"] == sorted(fold.train_subjects)
        assert recorded["test_subject"] not in recorded["train_subjects"]


# ------------------------------------------------------------ canonical serialization


def test_serialization_is_canonically_sorted(setup):
    config, manifest, folds = setup
    payload = load(record_run(config, manifest, folds))

    rel_paths = [entry["rel_path"] for entry in payload["inputs"]["radar_files"]]
    assert rel_paths == sorted(rel_paths)

    test_subjects = [f["test_subject"] for f in payload["folds"]]
    assert test_subjects == sorted(test_subjects)

    for fold in payload["folds"]:
        assert fold["train_subjects"] == sorted(fold["train_subjects"])


def test_repeated_runs_differ_only_by_timestamp(setup):
    """Byte-identical inputs -> byte-identical provenance, modulo time."""
    config, manifest, folds = setup
    first = load(record_run(config, manifest, folds))
    second = load(record_run(config, manifest, folds))

    for payload in (first, second):
        payload.pop("timestamp_utc")
    assert first == second


def test_hash_changes_when_data_changes(setup):
    """Sanity: the hash actually tracks file contents.

    NB: radar_files is sorted by rel_path, which is NOT the manifest's
    (subject, session_idx) order — "subject_1_10am.mat" sorts before
    "subject_1_8am.mat" — so entries are looked up by path, never by position.
    """
    config, manifest, folds = setup
    before = load(record_run(config, manifest, folds))

    rel_path = manifest.rel_path.iloc[0]
    cube = np.full((N_FAST_TIME, N_CHIRPS, 2), 99 + 1j, dtype=np.complex128)
    sio.savemat(str(config.paths.data_10ghz_dir / rel_path), {"framesRadar": cube})
    after = load(record_run(config, manifest, folds))

    def hash_of(payload, wanted):
        return next(e["sha256"] for e in payload["inputs"]["radar_files"] if e["rel_path"] == wanted)

    assert hash_of(before, rel_path) != hash_of(after, rel_path)
    # Every other file's hash is unchanged.
    untouched = [p for p in manifest.rel_path.unique() if p != rel_path]
    assert all(hash_of(before, p) == hash_of(after, p) for p in untouched)


# --------------------------------------------------------------- output authority


def test_results_dir_is_created_on_demand(setup):
    config, manifest, folds = setup
    assert not config.paths.results_dir.exists()
    record_run(config, manifest, folds)
    assert config.paths.results_dir.exists()


def test_each_run_gets_its_own_directory(setup):
    config, manifest, folds = setup
    first = record_run(config, manifest, folds)
    second = record_run(config, manifest, folds)
    assert first.parent != second.parent


def test_run_directory_name_is_filesystem_safe(setup):
    """Windows forbids ':' in paths, so an ISO timestamp cannot be a directory name."""
    config, manifest, folds = setup
    stamp = record_run(config, manifest, folds).parent.name

    assert ":" not in stamp
    assert stamp.endswith(tuple("0123456789abcdef")) and "Z_" in stamp
    # ...while the JSON keeps a real ISO-8601 timestamp.
    assert "T" in load(record_run(config, manifest, folds))["timestamp_utc"]


def test_refuses_to_overwrite_existing_provenance(setup, monkeypatch):
    config, manifest, folds = setup
    out = record_run(config, manifest, folds)

    # Force the same run directory (as if two runs shared a timestamp and revision).
    import dehyd.provenance as prov

    monkeypatch.setattr(prov, "RUN_STAMP_FORMAT", "'fixed'")
    record_run(config, manifest, folds)
    with pytest.raises(ProvenanceError, match="refusing to overwrite"):
        record_run(config, manifest, folds)
    assert out.exists()


def test_extra_payload_is_recorded(setup):
    config, manifest, folds = setup
    payload = load(record_run(config, manifest, folds, extra={"note": "smoke"}))
    assert payload["extra"] == {"note": "smoke"}


def test_no_folds_is_allowed(setup):
    config, manifest, _ = setup
    payload = load(record_run(config, manifest, None))
    assert payload["folds"] == []


# --------------------------------------------------- M5: 77 GHz data_dir parameter


def test_data_dir_selects_the_hashed_root(setup, tmp_path):
    """A 77 GHz run hashes rel_paths against the passed data_dir, not the 10 GHz root."""
    config, manifest, _ = setup
    default = load(record_run(config, manifest))
    alt = tmp_path / "77ghz"
    alt.mkdir()
    for rel in manifest["rel_path"].unique():
        (alt / rel).write_bytes(b"different-bytes-" + rel.encode())
    alt_payload = load(record_run(config, manifest, data_dir=alt))

    default_hashes = {e["rel_path"]: e["sha256"] for e in default["inputs"]["radar_files"]}
    alt_hashes = {e["rel_path"]: e["sha256"] for e in alt_payload["inputs"]["radar_files"]}
    assert default_hashes != alt_hashes  # data_dir changed which bytes were hashed
    assert set(default_hashes) == set(alt_hashes)  # ...but the same logical files


def test_sliced_manifest_hashes_only_its_files(setup):
    """An array task passes its single-session manifest slice, so only that file is hashed."""
    config, manifest, _ = setup
    one_rel = manifest["rel_path"].iloc[0]
    payload = load(record_run(config, manifest[manifest["rel_path"] == one_rel]))
    assert len(payload["inputs"]["radar_files"]) == 1
    assert payload["inputs"]["radar_files"][0]["rel_path"] == one_rel


# ------------------------------------- M7: git-commit env fallback (compute nodes)
# On IBEX compute nodes the in-process `git` call returns None (safe.directory did not
# take). The sbatch submit wrapper captures the revision at submit time into
# DEHYD_GIT_COMMIT/_BRANCH/_DIRTY, and _git_info falls back to them, so a cluster run
# still self-attests its revision. These tests kill the live git call to force that path.


@pytest.fixture
def no_live_git(monkeypatch):
    """Make every in-process `git ...` subprocess fail, as on a compute node."""
    import dehyd.provenance as prov

    def boom(*args, **kwargs):
        raise OSError("git unavailable (simulated compute node)")

    monkeypatch.setattr(prov.subprocess, "run", boom)
    # Ensure no stray env leaks between tests.
    for var in ("DEHYD_GIT_COMMIT", "DEHYD_GIT_BRANCH", "DEHYD_GIT_DIRTY"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_git_commit_env_fallback_when_git_fails(setup, no_live_git):
    config, manifest, folds = setup
    no_live_git.setenv("DEHYD_GIT_COMMIT", "deadbeef" * 5)  # 40-char sha
    no_live_git.setenv("DEHYD_GIT_BRANCH", "v1_milestone_7")
    no_live_git.setenv("DEHYD_GIT_DIRTY", "false")

    out = record_run(config, manifest, folds)
    payload = load(out)
    assert payload["git"]["commit"] == "deadbeef" * 5
    assert payload["git"]["branch"] == "v1_milestone_7"
    assert payload["git"]["dirty"] is False
    # The run directory's short-rev comes from the env commit, not "nogit".
    assert out.parent.name.endswith("deadbeef")


def test_git_degrades_to_none_without_env(setup, no_live_git):
    """With neither a live git nor the env vars, git fields are None (dir uses 'nogit')."""
    config, manifest, folds = setup
    out = record_run(config, manifest, folds)
    payload = load(out)
    assert payload["git"]["commit"] is None
    assert payload["git"]["branch"] is None
    assert payload["git"]["dirty"] is None
    assert out.parent.name.endswith("nogit")


def test_live_git_is_not_overridden_by_env(setup, monkeypatch):
    """A working local checkout reports its own true commit, never a stale env value."""
    config, manifest, folds = setup
    monkeypatch.setenv("DEHYD_GIT_COMMIT", "0" * 40)
    payload = load(record_run(config, manifest, folds))
    # This repo is a real git checkout, so the live call answers and the env is ignored.
    assert payload["git"]["commit"] != "0" * 40


def test_env_dirty_parsing(monkeypatch):
    from dehyd.provenance import _env_dirty

    for truthy in ("1", "true", "TRUE", "yes", " Yes "):
        monkeypatch.setenv("DEHYD_GIT_DIRTY", truthy)
        assert _env_dirty() is True
    for falsy in ("0", "false", "no", ""):
        monkeypatch.setenv("DEHYD_GIT_DIRTY", falsy)
        assert _env_dirty() is False
    monkeypatch.delenv("DEHYD_GIT_DIRTY", raising=False)
    assert _env_dirty() is None
