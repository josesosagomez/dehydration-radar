"""Filename parsing and 10 GHz file inspection/loading.

Mandatory tests use synthetic .mat files written with scipy.io.savemat; the real
80-file checks are realdata-marked.
"""

import numpy as np
import pytest
import scipy.io as sio

from dehyd.data.loader_10ghz import (
    N_CHIRPS,
    N_FAST_TIME,
    FileInfo,
    LoaderError,
    inspect_10ghz_file,
    load_10ghz_file,
    parse_10ghz_filename,
)
from dehyd.data.sessions import SESSION_NAMES


def write_mat(path, cube=None, var="framesRadar", n_frames=3, extra=None):
    """Write a synthetic radar file. Defaults produce a structurally valid one."""
    if cube is None:
        rng = np.random.default_rng(0)
        cube = (
            rng.standard_normal((N_FAST_TIME, N_CHIRPS, n_frames))
            + 1j * rng.standard_normal((N_FAST_TIME, N_CHIRPS, n_frames))
        )
    payload = {var: cube}
    if extra:
        payload.update(extra)
    sio.savemat(str(path), payload)
    return path


# ------------------------------------------------------------------ filename parsing


@pytest.mark.parametrize(
    "name,expected",
    [
        ("subject_1_8am.mat", (1, 0)),
        ("subject_7_12pm.mat", (7, 2)),
        ("subject_16_4pm.mat", (16, 4)),
    ],
)
def test_parses_valid_names(name, expected):
    assert parse_10ghz_filename(name) == expected


def test_all_session_names_parse():
    for idx, session in enumerate(SESSION_NAMES):
        assert parse_10ghz_filename(f"subject_3_{session}.mat") == (3, idx)


@pytest.mark.parametrize(
    "name",
    [
        "subject_1_9am.mat",      # not a real session
        "subject_x_8am.mat",      # non-numeric subject
        "subject_1_8am.MAT",      # wrong extension case
        "subject_1.mat",          # missing session
        "sub_1_8am.mat",          # wrong prefix
        "subject_1_8am.mat.bak",  # trailing junk
        "notes.txt",
    ],
)
def test_rejects_invalid_names(name):
    with pytest.raises(LoaderError, match="does not match"):
        parse_10ghz_filename(name)


# ---------------------------------------------------------------------- inspection


def test_inspect_valid_file(tmp_path):
    path = write_mat(tmp_path / "subject_4_2pm.mat", n_frames=7)
    info = inspect_10ghz_file(path)

    assert isinstance(info, FileInfo)
    assert (info.subject, info.session_idx) == (4, 3)
    assert info.n_frames == 7
    assert info.shape == (N_FAST_TIME, N_CHIRPS, 7)


def test_inspect_reads_actual_frame_count_not_100(tmp_path):
    """Frame count must come from the file: eligibility is computed from the real N."""
    path = write_mat(tmp_path / "subject_2_8am.mat", n_frames=42)
    assert inspect_10ghz_file(path).n_frames == 42


def test_missing_frames_radar_variable(tmp_path):
    path = write_mat(tmp_path / "subject_1_8am.mat", var="somethingElse")
    with pytest.raises(LoaderError, match="no 'framesRadar'"):
        inspect_10ghz_file(path)


def test_wrong_matlab_class(tmp_path):
    """An int16 array of the right shape must not pass as the double cube."""
    cube = np.ones((N_FAST_TIME, N_CHIRPS, 3), dtype=np.int16)
    path = write_mat(tmp_path / "subject_1_8am.mat", cube=cube)
    with pytest.raises(LoaderError, match="MATLAB class"):
        inspect_10ghz_file(path)


def test_wrong_first_two_axes(tmp_path):
    cube = np.zeros((256, 20, 3), dtype=np.complex128)
    path = write_mat(tmp_path / "subject_1_8am.mat", cube=cube)
    with pytest.raises(LoaderError, match="expected"):
        inspect_10ghz_file(path)


def test_wrong_dimensionality(tmp_path):
    cube = np.zeros((N_FAST_TIME, N_CHIRPS), dtype=np.complex128)
    path = write_mat(tmp_path / "subject_1_8am.mat", cube=cube)
    with pytest.raises(LoaderError, match="3 dimensions"):
        inspect_10ghz_file(path)


def test_zero_frames(tmp_path):
    cube = np.zeros((N_FAST_TIME, N_CHIRPS, 0), dtype=np.complex128)
    path = write_mat(tmp_path / "subject_1_8am.mat", cube=cube)
    # savemat/whosmat may report the empty axis as a shape mismatch or 0 frames;
    # either way it must be rejected.
    with pytest.raises(LoaderError):
        inspect_10ghz_file(path)


def test_unreadable_file(tmp_path):
    path = tmp_path / "subject_1_8am.mat"
    path.write_bytes(b"this is not a MAT file")
    with pytest.raises(LoaderError, match="cannot read MAT header"):
        inspect_10ghz_file(path)


# -------------------------------------------------------------------------- loading


def test_load_returns_complex128_cube(tmp_path):
    path = write_mat(tmp_path / "subject_9_10am.mat", n_frames=5)
    cube = load_10ghz_file(path)

    assert cube.dtype == np.complex128
    assert cube.shape == (N_FAST_TIME, N_CHIRPS, 5)


def test_load_rejects_non_complex(tmp_path):
    """A real-valued double array is structurally valid but not the expected signal."""
    cube = np.ones((N_FAST_TIME, N_CHIRPS, 3), dtype=np.float64)
    path = write_mat(tmp_path / "subject_1_8am.mat", cube=cube)
    with pytest.raises(LoaderError, match="expected complex128"):
        load_10ghz_file(path)


def test_load_ignores_frames_radar_iq(tmp_path):
    """framesRadarIQ must never be needed — only framesRadar is requested."""
    rng = np.random.default_rng(1)
    iq = rng.standard_normal((20834, 2, 3))
    path = write_mat(tmp_path / "subject_1_8am.mat", n_frames=3, extra={"framesRadarIQ": iq})
    assert load_10ghz_file(path).shape == (N_FAST_TIME, N_CHIRPS, 3)


# ------------------------------------------------------------------------- realdata


@pytest.mark.realdata
def test_inspect_all_real_files(real_data_paths):
    infos = [inspect_10ghz_file(p) for p in real_data_paths["mat_files"]]

    assert len(infos) == 80
    assert {i.subject for i in infos} == set(range(1, 17))
    assert all(i.shape == (N_FAST_TIME, N_CHIRPS, 100) for i in infos)


@pytest.mark.realdata
def test_load_one_real_file(real_data_paths):
    cube = load_10ghz_file(real_data_paths["mat_files"][0])

    assert cube.dtype == np.complex128
    assert cube.shape[:2] == (N_FAST_TIME, N_CHIRPS)
    assert np.isfinite(cube).all()
