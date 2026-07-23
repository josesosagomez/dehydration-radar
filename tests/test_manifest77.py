"""77 GHz manifest construction, the C1-C6 gate, and the QC bookkeeping (T-M77, no private data).

build_manifest_77 uses inspect_77ghz_file, which hard-asserts the real (16,256,256,n) shape,
so the inventory fixtures are full-shape but n_frames=1 and all-zeros — inspect reads metadata
only, so they stay a few KB on disk and cost nothing to scan. The QC eligibility arithmetic is
unit-tested on a synthetic joined frame (via _finalize_qc_77) so it needs no 1 GB load; one
small full-shape apply_qc_77 integration test exercises the load->screens->join->eligibility glue.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from dehyd.config import Preprocess77Config, QC77Config
from dehyd.data.ground_truth import GroundTruth
from dehyd.data.loader_77ghz import N_CHIRPS, N_FAST, N_RX, RADAR_VAR, reverse_axes
from dehyd.data.manifest import COLUMN_DTYPES, ManifestError, _join_qc
from dehyd.data.manifest_77 import (
    QC77_COLUMN_DTYPES,
    _finalize_qc_77,
    apply_qc_77,
    build_manifest_77,
    eligible_frames,
    evaluable_subjects,
    resolve_path_77,
    session_qc_report_77,
)
from dehyd.data.sessions import SESSION_NAMES


@dataclass(frozen=True)
class FakePaths77:
    data_77ghz_dir: Path


@dataclass(frozen=True)
class FakeConfig77:
    qc77: QC77Config
    preprocess77: Preprocess77Config


CONFIG = FakeConfig77(qc77=QC77Config(), preprocess77=Preprocess77Config())


def make_ground_truth(subjects=(1, 2), sessions=range(5)) -> GroundTruth:
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
        {"subject": s, "age": 30, "height_cm": 175.0, "baseline_mass_kg": 80.0 + s,
         "bmi": (80.0 + s) / 1.75**2}
        for s in subjects
    ]
    return GroundTruth(sessions=pd.DataFrame(rows), subjects=pd.DataFrame(subject_rows))


def write_on_disk(path: Path, loaded_cube: np.ndarray, *, name: str = RADAR_VAR) -> Path:
    """Write a [n_frames, fast, chirp, rx] loaded cube back to on-disk (rx, chirp, fast, frame)."""
    storage = np.transpose(loaded_cube, (3, 2, 1, 0))  # reverse_axes is its own inverse
    with h5py.File(path, "w") as handle:
        handle.create_dataset(name, data=storage, compression="gzip")
    return path


def write_meta_file(data_dir: Path, subject: int, session_idx: int, n_frames: int = 1) -> Path:
    """A full-shape all-zeros file — enough for inspect_77ghz_file (metadata only)."""
    cube = np.zeros((n_frames, N_FAST, N_CHIRPS, N_RX), dtype=np.float64)
    path = data_dir / f"subject_{subject}_{SESSION_NAMES[session_idx]}.mat"
    return write_on_disk(path, cube)


@pytest.fixture
def inventory(tmp_path):
    """A complete, valid 2-subject x 5-session inventory (metadata-only fixtures)."""
    data_dir = tmp_path / "77ghz"
    data_dir.mkdir()
    for subject in (1, 2):
        for session_idx in range(5):
            write_meta_file(data_dir, subject, session_idx)
    return FakePaths77(data_77ghz_dir=data_dir)


# ------------------------------------------------------------------- happy path


def test_builds_expected_shape(inventory):
    manifest = build_manifest_77(inventory, make_ground_truth())
    assert len(manifest) == 2 * 5 * 1  # subjects x sessions x frames
    assert set(manifest.columns) == set(COLUMN_DTYPES)
    assert sorted(manifest.subject.unique()) == [1, 2]
    for column, dtype in COLUMN_DTYPES.items():
        assert manifest[column].dtype == dtype, column


def test_rel_path_is_logical_not_physical(inventory):
    manifest = build_manifest_77(inventory, make_ground_truth())
    assert set(manifest.rel_path.unique()) == {
        f"subject_{s}_{SESSION_NAMES[i]}.mat" for s in (1, 2) for i in range(5)
    }
    assert not manifest.rel_path.str.contains(r"\.\.").any()
    assert not manifest.rel_path.str.startswith("/").any()


def test_actual_frame_count_recorded_not_assumed(tmp_path):
    """Frame counts vary; eligibility depends on the real per-file count, never 125."""
    data_dir = tmp_path / "77ghz"
    data_dir.mkdir()
    counts = {0: 1, 1: 3, 2: 2, 3: 1, 4: 2}
    for session_idx, n in counts.items():
        write_meta_file(data_dir, 1, session_idx, n_frames=n)

    manifest = build_manifest_77(FakePaths77(data_dir), make_ground_truth(subjects=(1,)))
    assert len(manifest) == sum(counts.values())
    for session_idx, n in counts.items():
        rows = manifest[manifest.session_idx == session_idx]
        assert rows.n_frames_in_file.unique().tolist() == [n]
        assert rows.frame_idx.tolist() == list(range(n))


def test_two_builds_are_identical(inventory, monkeypatch):
    first = build_manifest_77(inventory, make_ground_truth())
    real_glob = Path.glob

    def shuffled_glob(self, pattern):
        return reversed(sorted(real_glob(self, pattern)))

    monkeypatch.setattr(Path, "glob", shuffled_glob)
    second = build_manifest_77(inventory, make_ground_truth())
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------- C1-C6 structural failures


def test_c1_missing_file_fails(inventory):
    (inventory.data_77ghz_dir / "subject_2_12pm.mat").unlink()
    with pytest.raises(ManifestError, match="missing radar file"):
        build_manifest_77(inventory, make_ground_truth())


def test_c3_unparseable_file_fails(inventory):
    (inventory.data_77ghz_dir / "subject_9_9am.mat").write_bytes(b"")
    with pytest.raises(ManifestError, match="unmatched file"):
        build_manifest_77(inventory, make_ground_truth())


def test_c2_duplicate_key_fails(inventory):
    """subject_1_8am and subject_01_8am both parse to (1, 0)."""
    write_meta_file(inventory.data_77ghz_dir, 1, 0)  # ensure valid
    dup = inventory.data_77ghz_dir / "subject_01_8am.mat"
    write_on_disk(dup, np.zeros((1, N_FAST, N_CHIRPS, N_RX)))
    with pytest.raises(ManifestError, match="duplicate"):
        build_manifest_77(inventory, make_ground_truth())


def test_c4_file_without_ground_truth_row_fails(inventory):
    for session_idx in range(5):
        write_meta_file(inventory.data_77ghz_dir, 3, session_idx)
    with pytest.raises(ManifestError, match="no ground-truth row"):
        build_manifest_77(inventory, make_ground_truth())


def test_c4_ground_truth_row_without_file_fails(inventory):
    with pytest.raises(ManifestError, match="missing radar file"):
        build_manifest_77(inventory, make_ground_truth(subjects=(1, 2, 3)))


def test_c5_malformed_file_fails(inventory):
    """A structurally wrong cube is caught even though its name is valid."""
    bad = inventory.data_77ghz_dir / "subject_2_2pm.mat"
    write_on_disk(bad, np.zeros((2, 32, 32, 4)))  # wrong shape
    with pytest.raises(ManifestError, match="malformed file"):
        build_manifest_77(inventory, make_ground_truth())


def test_missing_data_dir_fails(tmp_path):
    with pytest.raises(ManifestError, match="not a directory"):
        build_manifest_77(FakePaths77(tmp_path / "nope"), make_ground_truth())


def test_none_data_dir_fails():
    with pytest.raises(ManifestError, match="not set"):
        build_manifest_77(FakePaths77(None), make_ground_truth())


def test_resolve_path_77_needs_the_dir():
    with pytest.raises(ManifestError, match="not set"):
        resolve_path_77(FakePaths77(None), "subject_1_8am.mat")


# ============================================================ QC bookkeeping
# The eligibility arithmetic and reporting are tested on a synthetic joined frame; the
# screens themselves are tested in test_qc77.py.


def _synthetic_joined(n_frames: int, n_pass: int, subject=1, session_idx=0) -> pd.DataFrame:
    """A hand-built join result: the first n_pass frames pass, the rest fail nan_inf."""
    rows = []
    for i in range(n_frames):
        passed = i < n_pass
        rows.append(
            {
                "subject": subject, "session_idx": session_idx,
                "session_name": SESSION_NAMES[session_idx],
                "rel_path": f"subject_{subject}_{SESSION_NAMES[session_idx]}.mat",
                "n_frames_in_file": n_frames, "frame_idx": i,
                "delta_m_pct": 0.0, "class_label": session_idx,
                "qc_nan_inf": not passed, "qc_flatline": False, "qc_low_in_band": False,
                "qc_pass": passed, "qc_in_band_ratio": 0.9 if passed else float("nan"),
                "qc_n_flatline_traces": 0, "qc_rx_max_flatline": 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "n_frames, n_pass, expect_min_pass, expect_eligible",
    [
        (3, 2, 2, True),   # ceil(0.5*3) = 2
        (3, 1, 2, False),
        (4, 2, 2, True),   # exactly half is enough
        (4, 1, 2, False),
        (5, 3, 3, True),
        (5, 2, 3, False),
    ],
)
def test_eligibility_uses_ceil_of_actual_frame_count(
    n_frames, n_pass, expect_min_pass, expect_eligible
):
    merged = _finalize_qc_77(_synthetic_joined(n_frames, n_pass), QC77Config().min_frame_fraction)
    assert (merged["session_min_pass"] == expect_min_pass).all()
    assert (merged["session_eligible"] == expect_eligible).all()
    # The reconciliation identity holds frame-for-frame.
    assert (merged["session_n_pass"] == n_pass).all()
    assert int((~merged["qc_fail_any"]).sum()) == n_pass


def test_report_reconciles_through_fail_any():
    merged = _finalize_qc_77(_synthetic_joined(4, 3), QC77Config().min_frame_fraction)
    report = session_qc_report_77(merged)
    row = report.iloc[0]
    assert row["n_pass"] + row["n_fail_any"] == row["n_frames"]
    assert row["n_nan_inf"] == 1 and row["n_pass"] == 3


def test_eligible_frames_and_evaluable_subjects_on_synthetic():
    # subject 1: session 0 eligible (2/2 pass); subject 2: session 0 ineligible (0/3).
    s1 = _finalize_qc_77(_synthetic_joined(2, 2, subject=1), QC77Config().min_frame_fraction)
    s2 = _finalize_qc_77(_synthetic_joined(3, 0, subject=2), QC77Config().min_frame_fraction)
    merged = pd.concat([s1, s2], ignore_index=True)
    frames = eligible_frames(merged)
    assert set(frames["subject"]) == {1}
    assert frames["qc_pass"].all() and frames["session_eligible"].all()
    assert evaluable_subjects(merged) == (1,)


# ------------------------------------------------------------ fail-closed join (imported)


def test_imported_join_still_fails_closed(inventory):
    manifest = build_manifest_77(inventory, make_ground_truth())
    rows = pd.DataFrame(
        {"rel_path": manifest["rel_path"], "frame_idx": manifest["frame_idx"], "qc_pass": True}
    )
    doubled = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ManifestError, match="duplicate"):
        _join_qc(manifest, doubled)
    with pytest.raises(ManifestError, match="one-to-one"):
        _join_qc(manifest, rows.iloc[1:])


# ------------------------------------------------- apply_qc_77 integration (small, real screens)


def clean_frame_77(seed: int) -> np.ndarray:
    """A real in-gate beat tone plus noise: passes every 77 GHz screen. [fast, chirp, rx]."""
    rng = np.random.default_rng(seed)
    pre = Preprocess77Config()
    f_in = 78_000.0  # ~3 m beat frequency, inside the 2-4 m gate (mask bins 26..54)
    t = np.arange(N_FAST) / pre.fs_hz
    tone = np.cos(2 * np.pi * f_in * t)
    frame = np.broadcast_to(tone[:, None, None], (N_FAST, N_CHIRPS, N_RX)).copy()
    frame += 0.02 * rng.standard_normal(frame.shape)  # break the flatline
    return frame


def failing_frame_77(kind: str) -> np.ndarray:
    if kind == "nan":
        frame = clean_frame_77(0)
        frame[0, 0, 0] = np.nan
        return frame
    if kind == "zero":  # all-zero: flatline AND low in-band
        return np.zeros((N_FAST, N_CHIRPS, N_RX), dtype=np.float64)
    raise ValueError(kind)


def write_qc_file_77(data_dir, subject, session_idx, clean, failing_kinds):
    frames = [clean_frame_77(s) for s in range(clean)]
    frames += [failing_frame_77(k) for k in failing_kinds]
    cube = np.stack(frames, axis=0)  # [n_frames, fast, chirp, rx]
    path = data_dir / f"subject_{subject}_{SESSION_NAMES[session_idx]}.mat"
    return write_on_disk(path, cube)


def test_apply_qc_77_end_to_end_small(tmp_path):
    data_dir = tmp_path / "77ghz"
    data_dir.mkdir()
    # 1 subject, 2 sessions (GT range(2)): session 0 = 2 clean; session 1 = 1 clean + 1 NaN.
    write_qc_file_77(data_dir, 1, 0, clean=2, failing_kinds=[])
    write_qc_file_77(data_dir, 1, 1, clean=1, failing_kinds=["nan"])

    manifest = build_manifest_77(FakePaths77(data_dir), make_ground_truth(subjects=(1,), sessions=range(2)))
    manifest_qc = apply_qc_77(manifest, FakePaths77(data_dir), CONFIG)

    assert set(manifest_qc.columns) == set(COLUMN_DTYPES) | set(QC77_COLUMN_DTYPES)
    for column, dtype in {**COLUMN_DTYPES, **QC77_COLUMN_DTYPES}.items():
        assert manifest_qc[column].dtype == dtype, column

    report = session_qc_report_77(manifest_qc).set_index("session_idx")
    assert report.loc[0, "n_pass"] == 2 and report.loc[0, "n_fail_any"] == 0
    assert report.loc[1, "n_pass"] == 1 and report.loc[1, "n_nan_inf"] == 1
    assert bool(report.loc[0, "eligible"]) and bool(report.loc[1, "eligible"])
    for _, row in report.iterrows():
        assert row["n_pass"] + row["n_fail_any"] == row["n_frames"]


# ---------------------------------------------------------------------- realdata


@pytest.mark.realdata
def test_real_77ghz_manifest_builds_and_validates(real_data_77_paths):
    from dehyd.config import load_config
    from dehyd.data.ground_truth import load_ground_truth

    cfg = load_config("configs/exp_77ghz.yaml")
    gt = load_ground_truth(cfg.paths.weight_xlsx)
    manifest = build_manifest_77(cfg.paths, gt)

    assert sorted(manifest.subject.unique()) == list(range(1, 17))
    assert manifest.groupby(["subject", "session_idx"]).ngroups == 16 * 5
    for column, dtype in COLUMN_DTYPES.items():
        assert manifest[column].dtype == dtype, column
