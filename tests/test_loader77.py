"""77 GHz loader — the on-disk contract, on synthetic HDF5 fixtures (T-L77, no private data).

Two fixture sizes (C5-23). The accept path is proven by one FULL-SHAPE (16,256,256,n)
fixture, because inspect_77ghz_file hard-asserts the real dimensions; every rejection path
and the pure helpers use tiny fixtures. Full-shape fixtures are mostly zeros with a few
distinct markers, so they gzip to a few KB on disk while still catching an axis-transpose bug.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from dehyd.data.loader_77ghz import (
    N_CHIRPS,
    N_FAST,
    N_RX,
    RADAR_VAR,
    FileInfo77,
    LoaderError77,
    inspect_77ghz_file,
    load_77ghz_file,
    parse_77ghz_filename,
    reverse_axes,
    to_numeric,
)


def write_on_disk(path: Path, storage: np.ndarray, *, name: str = RADAR_VAR,
                  compound: bool = False) -> Path:
    """Write an array in the ON-DISK (rx, chirp, fast, frame) layout, gzip-chunked."""
    with h5py.File(path, "w") as handle:
        if compound:
            dtype = np.dtype([("real", "<f8"), ("imag", "<f8")])
            data = np.empty(storage.shape, dtype=dtype)
            data["real"] = storage.real
            data["imag"] = storage.imag
            handle.create_dataset(name, data=data, compression="gzip")
        else:
            handle.create_dataset(name, data=storage, compression="gzip")
    return path


def full_shape_fixture(path: Path, n_frames: int = 2) -> tuple[Path, np.ndarray]:
    """A real-float64 (16,256,256,n) on-disk cube: zeros plus a few distinct markers.

    Markers at distinct (rx, chirp, fast, frame) coordinates with distinct values, so a
    fast<->chirp transpose or a wrong frame axis cannot pass. Returns the on-disk array so
    the test can compare against reverse_axes(storage) bit-for-bit.
    """
    storage = np.zeros((N_RX, N_CHIRPS, N_FAST, n_frames), dtype=np.float64)
    # (rx, chirp, fast, frame) = value
    storage[3, 100, 50, 0] = 7.0
    storage[10, 5, 200, 1] = -3.5
    storage[0, 255, 0, 0] = 1.25
    storage[15, 0, 255, 1] = 9.0
    write_on_disk(path, storage)
    return path, storage


# --------------------------------------------------------------------- filename parse


def test_parse_filename():
    assert parse_77ghz_filename("subject_7_12pm.mat") == (7, 2)
    assert parse_77ghz_filename("subject_16_4pm.mat") == (16, 4)
    assert parse_77ghz_filename(Path("/data/77ghz/subject_1_8am.mat")) == (1, 0)


@pytest.mark.parametrize(
    "name",
    ["subject_7.mat", "subject_7_9am.mat", "subj_7_8am.mat", "subject_7_8am.h5",
     "subject_x_8am.mat"],
)
def test_parse_filename_rejects_bad_names(name):
    with pytest.raises(LoaderError77, match="does not match"):
        parse_77ghz_filename(name)


# --------------------------------------------------------- reverse_axes / to_numeric


def test_reverse_axes_round_trip_small_dims():
    """Distinct value per coordinate, so a transposed axis pair cannot pass by symmetry."""
    rng = np.random.default_rng(3)
    n_rx, n_chirp, n_fast, n_frame = 2, 3, 4, 5
    storage = rng.standard_normal((n_rx, n_chirp, n_fast, n_frame))
    cube = reverse_axes(storage)
    assert cube.shape == (n_frame, n_fast, n_chirp, n_rx)
    for rx in range(n_rx):
        for chirp in range(n_chirp):
            for fast in range(n_fast):
                for frame in range(n_frame):
                    assert cube[frame, fast, chirp, rx] == storage[rx, chirp, fast, frame]


def test_to_numeric_real_passthrough_and_compound():
    real = np.ones((2, 2), dtype=np.float32)
    assert to_numeric(real).dtype == np.float64
    dt = np.dtype([("real", "<f8"), ("imag", "<f8")])
    comp = np.zeros((2,), dtype=dt)
    comp["real"], comp["imag"] = [1.0, 2.0], [3.0, 4.0]
    got = to_numeric(comp)
    assert got.dtype == np.complex128
    np.testing.assert_array_equal(got, np.array([1 + 3j, 2 + 4j]))


# ------------------------------------------------------------------ accept path (full)


def test_full_shape_fixture_inspected(tmp_path):
    path, storage = full_shape_fixture(tmp_path / "subject_7_12pm.mat", n_frames=2)
    info = inspect_77ghz_file(path)
    assert isinstance(info, FileInfo77)
    assert (info.subject, info.session_idx, info.n_frames) == (7, 2, 2)
    assert info.shape == (N_RX, N_CHIRPS, N_FAST, 2)


def test_full_shape_fixture_loaded_bit_for_bit(tmp_path):
    path, storage = full_shape_fixture(tmp_path / "subject_1_8am.mat", n_frames=2)
    cube = load_77ghz_file(path)
    assert cube.shape == (2, N_FAST, N_CHIRPS, N_RX)
    assert cube.dtype == np.float64
    np.testing.assert_array_equal(cube, reverse_axes(storage))
    # The markers land at the reversed coordinates (rx,chirp,fast,frame)->(frame,fast,chirp,rx).
    assert cube[0, 50, 100, 3] == 7.0
    assert cube[1, 200, 5, 10] == -3.5
    assert cube[0, 0, 255, 0] == 1.25
    assert cube[1, 255, 0, 15] == 9.0


# ----------------------------------------------------------------- rejection paths


def test_missing_var_rejected(tmp_path):
    path = tmp_path / "subject_2_10am.mat"
    write_on_disk(path, np.zeros((2, 2, 2, 2)), name="somethingElse")
    with pytest.raises(LoaderError77, match="no 'framesRadar'"):
        inspect_77ghz_file(path)


def test_compound_dtype_rejected(tmp_path):
    """The M2 finding: 77 GHz files are REAL float64; a complex file is never coerced."""
    path = tmp_path / "subject_3_12pm.mat"
    write_on_disk(path, np.zeros((2, 2, 2, 2)) + 0j, compound=True)
    with pytest.raises(LoaderError77, match="compound"):
        inspect_77ghz_file(path)


def test_float32_rejected(tmp_path):
    path = tmp_path / "subject_4_2pm.mat"
    write_on_disk(path, np.zeros((2, 2, 2, 2), dtype=np.float32))
    with pytest.raises(LoaderError77, match="float32|64-bit"):
        inspect_77ghz_file(path)


def test_big_endian_rejected(tmp_path):
    """Little-endian float64 is the confirmed contract; big-endian is format drift (C5-18)."""
    path = tmp_path / "subject_5_4pm.mat"
    write_on_disk(path, np.zeros((2, 2, 2, 2), dtype=">f8"))
    with pytest.raises(LoaderError77, match="byte order"):
        inspect_77ghz_file(path)


def test_wrong_shape_rejected(tmp_path):
    path = tmp_path / "subject_6_8am.mat"
    write_on_disk(path, np.zeros((2, 32, 32, 4), dtype=np.float64))
    with pytest.raises(LoaderError77, match="on-disk shape"):
        inspect_77ghz_file(path)


def test_non_mat_file_rejected(tmp_path):
    path = tmp_path / "subject_8_10am.mat"
    path.write_bytes(b"not an HDF5 file at all")
    with pytest.raises(LoaderError77, match="cannot open"):
        inspect_77ghz_file(path)


# ---------------------------------------------------------------------- realdata


@pytest.mark.realdata
def test_inspect_real_file(real_data_77_paths):
    """One real file inspects to the confirmed (16, 256, 256, n) real-float64 contract."""
    path = real_data_77_paths["data_77ghz_dir"] / "subject_1_8am.mat"
    info = inspect_77ghz_file(path)
    assert info.shape[:3] == (N_RX, N_CHIRPS, N_FAST)
    assert info.n_frames > 0
