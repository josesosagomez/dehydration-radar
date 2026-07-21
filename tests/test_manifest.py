"""Manifest construction and the C1-C6 structural gate.

Mandatory tests use synthetic inventories: small savemat files in a tmp dir plus a
GroundTruth built directly in memory (it is just two DataFrames). Constructing the
GroundTruth directly rather than round-tripping a synthetic workbook sidesteps the
openpyxl formula-cache limitation and keeps these tests about the manifest.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from dehyd.data.ground_truth import GroundTruth
from dehyd.data.manifest import COLUMN_DTYPES, SORT_KEYS, ManifestError, build_manifest
from dehyd.data.sessions import SESSION_NAMES

N_FAST_TIME = 534
N_CHIRPS = 20


@dataclass(frozen=True)
class FakePaths:
    data_10ghz_dir: Path


def make_ground_truth(subjects=(1, 2, 3), sessions=range(5)) -> GroundTruth:
    rows = []
    for subject in subjects:
        baseline = 80.0 + subject
        for idx in sessions:
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
    subject_rows = [
        {
            "subject": s,
            "age": 30,
            "height_cm": 175.0,
            "baseline_mass_kg": 80.0 + s,
            "bmi": (80.0 + s) / 1.75**2,
        }
        for s in subjects
    ]
    return GroundTruth(sessions=pd.DataFrame(rows), subjects=pd.DataFrame(subject_rows))


def write_file(data_dir: Path, subject: int, session_idx: int, n_frames=3, shape=None):
    shape = shape or (N_FAST_TIME, N_CHIRPS, n_frames)
    cube = np.zeros(shape, dtype=np.complex128)
    cube.real[:] = 1.0  # non-degenerate
    path = data_dir / f"subject_{subject}_{SESSION_NAMES[session_idx]}.mat"
    sio.savemat(str(path), {"framesRadar": cube})
    return path


@pytest.fixture
def inventory(tmp_path):
    """A complete, valid 3-subject x 5-session inventory."""
    data_dir = tmp_path / "10ghz"
    data_dir.mkdir()
    for subject in (1, 2, 3):
        for session_idx in range(5):
            write_file(data_dir, subject, session_idx, n_frames=3)
    return FakePaths(data_10ghz_dir=data_dir)


# ------------------------------------------------------------------- happy path


def test_builds_expected_shape(inventory):
    manifest = build_manifest(inventory, make_ground_truth())

    assert len(manifest) == 3 * 5 * 3  # subjects x sessions x frames
    assert set(manifest.columns) == set(COLUMN_DTYPES)
    assert sorted(manifest.subject.unique()) == [1, 2, 3]
    assert sorted(manifest.session_idx.unique()) == [0, 1, 2, 3, 4]


def test_dtypes_are_fixed(inventory):
    manifest = build_manifest(inventory, make_ground_truth())
    for column, dtype in COLUMN_DTYPES.items():
        assert manifest[column].dtype == dtype, column


def test_rel_path_is_logical_not_physical(inventory):
    """Identity must be portable: no absolute paths, no '..' segments."""
    manifest = build_manifest(inventory, make_ground_truth())

    assert set(manifest.rel_path.unique()) == {
        f"subject_{s}_{SESSION_NAMES[i]}.mat" for s in (1, 2, 3) for i in range(5)
    }
    assert not manifest.rel_path.str.contains(r"\.\.").any()
    assert not manifest.rel_path.str.startswith("/").any()
    assert not manifest.rel_path.str.contains(":").any()  # no drive letters


def test_targets_joined_correctly(inventory):
    gt = make_ground_truth()
    manifest = build_manifest(inventory, gt)

    expected = {
        (int(r.subject), int(r.session_idx)): float(r.delta_m_pct)
        for r in gt.sessions.itertuples()
    }
    for row in manifest.itertuples():
        assert row.delta_m_pct == pytest.approx(expected[(row.subject, row.session_idx)])

    # S0 is the baseline, so its target is identically zero.
    assert (manifest[manifest.session_idx == 0].delta_m_pct == 0).all()


def test_class_label_is_session_index(inventory):
    manifest = build_manifest(inventory, make_ground_truth())
    assert (manifest.class_label == manifest.session_idx).all()


def test_actual_frame_count_recorded_not_assumed(tmp_path):
    """Frame counts vary; eligibility at M2 depends on the real per-file count."""
    data_dir = tmp_path / "10ghz"
    data_dir.mkdir()
    counts = {0: 3, 1: 7, 2: 2, 3: 5, 4: 4}
    for session_idx, n in counts.items():
        write_file(data_dir, 1, session_idx, n_frames=n)

    manifest = build_manifest(FakePaths(data_dir), make_ground_truth(subjects=(1,)))

    assert len(manifest) == sum(counts.values())
    for session_idx, n in counts.items():
        rows = manifest[manifest.session_idx == session_idx]
        assert rows.n_frames_in_file.unique().tolist() == [n]
        assert rows.frame_idx.tolist() == list(range(n))


# ------------------------------------------------------------------ determinism


def test_two_builds_are_frame_for_frame_identical(inventory, monkeypatch):
    """Filesystem enumeration order must not affect the result."""
    first = build_manifest(inventory, make_ground_truth())

    real_glob = Path.glob

    def shuffled_glob(self, pattern):
        return reversed(sorted(real_glob(self, pattern)))

    monkeypatch.setattr(Path, "glob", shuffled_glob)
    second = build_manifest(inventory, make_ground_truth())

    pd.testing.assert_frame_equal(first, second)


def test_manifest_is_sorted(inventory):
    manifest = build_manifest(inventory, make_ground_truth())
    pd.testing.assert_frame_equal(
        manifest, manifest.sort_values(SORT_KEYS).reset_index(drop=True)
    )


# --------------------------------------------------- C1-C6 structural failures


def test_c1_missing_file_fails(inventory):
    """A missing subject x session cell must be named, not silently dropped."""
    (inventory.data_10ghz_dir / "subject_2_12pm.mat").unlink()
    with pytest.raises(ManifestError, match="missing radar file"):
        build_manifest(inventory, make_ground_truth())


def test_c1_error_names_every_missing_cell(inventory):
    (inventory.data_10ghz_dir / "subject_2_12pm.mat").unlink()
    (inventory.data_10ghz_dir / "subject_3_8am.mat").unlink()
    with pytest.raises(ManifestError) as excinfo:
        build_manifest(inventory, make_ground_truth())
    message = str(excinfo.value)
    assert "subject 2 12pm" in message and "subject 3 8am" in message


def test_c3_unparseable_file_fails(inventory):
    (inventory.data_10ghz_dir / "subject_9_9am.mat").write_bytes(b"")
    with pytest.raises(ManifestError, match="unmatched file"):
        build_manifest(inventory, make_ground_truth())


def test_c4_file_without_ground_truth_row_fails(inventory):
    """A 4th subject's files with no workbook row is an unmatched record."""
    for session_idx in range(5):
        write_file(inventory.data_10ghz_dir, 4, session_idx)
    with pytest.raises(ManifestError, match="no ground-truth row"):
        build_manifest(inventory, make_ground_truth())


def test_c4_ground_truth_row_without_file_fails(inventory):
    """The bijection is checked in both directions."""
    with pytest.raises(ManifestError, match="missing radar file"):
        build_manifest(inventory, make_ground_truth(subjects=(1, 2, 3, 4)))


def test_c5_malformed_file_fails(inventory):
    """A structurally wrong cube is caught even though its name is valid."""
    write_file(inventory.data_10ghz_dir, 2, 2, shape=(256, 20, 3))
    with pytest.raises(ManifestError, match="malformed file"):
        build_manifest(inventory, make_ground_truth())


def test_c5_wrong_matlab_class_fails(inventory):
    path = inventory.data_10ghz_dir / "subject_2_2pm.mat"
    sio.savemat(str(path), {"framesRadar": np.ones((N_FAST_TIME, N_CHIRPS, 3), np.int16)})
    with pytest.raises(ManifestError, match="malformed file"):
        build_manifest(inventory, make_ground_truth())


def test_missing_data_dir_fails(tmp_path):
    with pytest.raises(ManifestError, match="not a directory"):
        build_manifest(FakePaths(tmp_path / "nope"), make_ground_truth())


# ------------------------------------------------------------------- realdata


@pytest.mark.realdata
def test_real_manifest_builds_and_validates(real_data_paths):
    from dehyd.config import load_config
    from dehyd.data.ground_truth import load_ground_truth

    cfg = load_config("configs/exp_a_regression.yaml")
    gt = load_ground_truth(cfg.paths.weight_xlsx)
    manifest = build_manifest(cfg.paths, gt)

    assert len(manifest) == 16 * 5 * 100  # 8000 frames
    assert sorted(manifest.subject.unique()) == list(range(1, 17))
    assert manifest.n_frames_in_file.unique().tolist() == [100]
    assert manifest.groupby(["subject", "session_idx"]).size().eq(100).all()
    for column, dtype in COLUMN_DTYPES.items():
        assert manifest[column].dtype == dtype, column
