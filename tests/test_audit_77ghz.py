"""The 77 GHz audit's logic, on synthetic HDF5 fixtures — no private data.

The audit only ever runs once, on one real file, so its correctness-critical parts
(storage handling, axis reversal, the semantic metrics and their three-way verdict,
the bounded-read contract, energy normalisation, the failure-path JSON) cannot be
validated by that run. They are validated here instead, which is why the script keeps
its numerics in pure, parameterised helpers.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import audit_77ghz as audit  # noqa: E402

N_FAST = 32
N_CHIRP = 32
N_RX = 2
GATE_BINS = np.array([8, 9, 10])
DC_HALFWIDTH = 3


# ------------------------------------------------------------------------ fixtures


def tone_along(axis: str, n_frames: int = 4) -> np.ndarray:
    """A cube in the reversed (frame, fast, chirp, rx) layout.

    'fast'  — a complex tone at gate bin 9 along fast-time, constant across chirps:
              the pattern a static target at the gate range produces.
    'chirp' — the same structure with the two candidate axes interchanged.
    """
    frame, fast, chirp, rx = np.indices((n_frames, N_FAST, N_CHIRP, N_RX))
    index = fast if axis == "fast" else chirp
    return np.exp(2j * np.pi * 9 * index / N_FAST).astype(np.complex128)


def noise_cube(n_frames: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (n_frames, N_FAST, N_CHIRP, N_RX)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def write_fixture(path: Path, cube: np.ndarray, compound: bool = False) -> Path:
    """Write a (frame, fast, chirp, rx) cube in the ON-DISK (rx, chirp, fast, frame) layout.

    The transpose is the same full reversal the audit applies on read — it is its own
    inverse — so a fixture written here must come back out unchanged.
    """
    storage = np.transpose(cube, (3, 2, 1, 0))
    with h5py.File(path, "w") as handle:
        if compound:
            dtype = np.dtype([("real", "<f8"), ("imag", "<f8")])
            data = np.empty(storage.shape, dtype=dtype)
            data["real"] = storage.real
            data["imag"] = storage.imag
            handle.create_dataset(audit.RADAR_VAR, data=data)
        else:
            handle.create_dataset(audit.RADAR_VAR, data=storage.real)
    return path


@pytest.fixture
def config_yaml(tmp_path):
    """A loadable config whose results_dir is inside tmp_path."""
    data_dir = tmp_path / "radar"
    data_dir.mkdir()
    xlsx = tmp_path / "w.xlsx"
    xlsx.write_bytes(b"")
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_10ghz_dir": str(data_dir),
                    "weight_xlsx": str(xlsx),
                    "results_dir": str(tmp_path / "results"),
                },
                "run": {"seed": 1, "seed_set": [1, 2, 3, 4, 5], "device": "cpu"},
            }
        ),
        encoding="utf-8",
    )
    return path


def run_audit(config_yaml, mat_path, extra=()):
    code = audit.main(
        ["--config", str(config_yaml), "--file", str(mat_path), *extra]
    )
    out = Path(yaml.safe_load(config_yaml.read_text())["paths"]["results_dir"])
    return code, json.loads((out / "qc" / "audit_77ghz.json").read_text())


# ------------------------------------------------- storage handling + axis mapping


def test_compound_complex_round_trips_through_read_and_reversal(tmp_path):
    """Every on-disk (rx, chirp, fast, frame) element must land at (frame, fast, chirp, rx).

    Written with distinct values per coordinate, so a transposed pair of axes cannot
    pass by symmetry.
    """
    rng = np.random.default_rng(3)
    expected = (
        rng.standard_normal((5, N_FAST, N_CHIRP, N_RX))
        + 1j * rng.standard_normal((5, N_FAST, N_CHIRP, N_RX))
    )
    path = write_fixture(tmp_path / "c.mat", expected, compound=True)

    with h5py.File(path, "r") as handle:
        dset = handle[audit.RADAR_VAR]
        assert dset.shape == (N_RX, N_CHIRP, N_FAST, 5)  # the on-disk layout
        assert audit.describe_storage(dset)["representation"] == "compound_complex"
        cube = audit.reverse_axes(audit.to_numeric(audit.read_frames(dset, 5)))

    assert cube.shape == (5, N_FAST, N_CHIRP, N_RX)
    np.testing.assert_array_equal(cube, expected)


def test_real_float_storage_is_accepted_and_passed_through(tmp_path):
    """The real 77 GHz files are plain float64 — complex arrives only at the range FFT."""
    path = write_fixture(tmp_path / "r.mat", tone_along("fast", n_frames=3))
    with h5py.File(path, "r") as handle:
        dset = handle[audit.RADAR_VAR]
        storage = audit.describe_storage(dset)
        assert storage["representation"] == "real_float"
        assert storage["verdict"] == "ACCEPTED"
        assert audit.to_numeric(audit.read_frames(dset, 3)).dtype == np.float64


def test_unsupported_storage_is_rejected(tmp_path):
    """Integer fields, odd field names, native complex: recorded, never coerced."""
    path = tmp_path / "bad.mat"
    with h5py.File(path, "w") as handle:
        dtype = np.dtype([("re", "<i4"), ("im", "<i4")])
        handle.create_dataset(audit.RADAR_VAR, shape=(2, 2, 2, 2), dtype=dtype)
    with h5py.File(path, "r") as handle:
        storage = audit.describe_storage(handle[audit.RADAR_VAR])
    assert storage["verdict"] == "REJECTED"
    assert storage["representation"] == "unsupported"
    assert storage["field_names"] == ["re", "im"]  # observed, so the finding is actionable


# ----------------------------------------------------------------- verdict paths


def test_proposed_axis_signal_is_accepted():
    metrics = audit.axis_metrics(tone_along("fast"), 1, 2, GATE_BINS, DC_HALFWIDTH)
    assert metrics["G_fast"] > metrics["G_chirp"]
    assert audit.axis_verdict(metrics) == "ACCEPTED"


def test_swapped_axis_signal_is_rejected():
    """REJECTED is reserved for positive evidence FOR the swap, not mere failure."""
    metrics = audit.axis_metrics(tone_along("chirp"), 1, 2, GATE_BINS, DC_HALFWIDTH)
    assert audit.axis_verdict(metrics) == "REJECTED"


def test_low_information_signal_is_inconclusive():
    """Noise supports neither assignment — that is not evidence the axes are swapped."""
    metrics = audit.axis_metrics(noise_cube(seed=11), 1, 2, GATE_BINS, DC_HALFWIDTH)
    assert audit.axis_verdict(metrics) == "INCONCLUSIVE"


def test_axis_metrics_are_computed_symmetrically():
    """Interchanging the axes must interchange the metrics — no hidden asymmetry."""
    forward = audit.axis_metrics(tone_along("fast"), 1, 2, GATE_BINS, DC_HALFWIDTH)
    swapped = audit.axis_metrics(tone_along("chirp"), 1, 2, GATE_BINS, DC_HALFWIDTH)
    assert forward["G_fast"] == pytest.approx(swapped["G_chirp"])
    assert forward["D_chirp"] == pytest.approx(swapped["D_fast"])


# ---------------------------------------------------------- bounded-read contract


class SpyDataset:
    """Records every read and enforces the bounded-slab contract.

    Full slices on the NON-frame axes (rx, chirp, fast) are required — each slab must
    read them completely. What is forbidden is an ellipsis/whole-dataset read or any
    request spanning the entire frame axis.
    """

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.requests = []

    def __getitem__(self, key):
        if key is Ellipsis or (isinstance(key, tuple) and any(k is Ellipsis for k in key)):
            raise AssertionError("ellipsis / whole-dataset read")
        if not isinstance(key, tuple) or len(key) != self._array.ndim:
            raise AssertionError(f"expected an explicit per-axis index tuple, got {key!r}")
        for axis, part in enumerate(key[:-1]):
            if part != slice(None):
                raise AssertionError(f"non-frame axis {axis} must be read in full")
        frame_part = key[-1]
        if not isinstance(frame_part, slice):
            raise AssertionError("the frame axis must be sliced")
        start, stop, _ = frame_part.indices(self.shape[-1])
        if stop - start >= self.shape[-1]:
            raise AssertionError("request spans the entire frame axis")
        self.requests.append((start, stop))
        return self._array[key]


def test_read_frames_never_reads_the_whole_dataset():
    array = np.arange(N_RX * N_CHIRP * N_FAST * 20).reshape(N_RX, N_CHIRP, N_FAST, 20)
    spy = SpyDataset(array)
    out = audit.read_frames(spy, 6)
    np.testing.assert_array_equal(out, array[:, :, :, :6])
    assert spy.requests == [(0, 6)]


def test_read_frames_splits_into_bounded_blocks():
    array = np.arange(N_RX * N_CHIRP * N_FAST * 20).reshape(N_RX, N_CHIRP, N_FAST, 20)
    spy = SpyDataset(array)
    out = audit.read_frames(spy, 6, block_size=2)
    np.testing.assert_array_equal(out, array[:, :, :, :6])
    # Each requested frame is fetched exactly once, in order.
    assert spy.requests == [(0, 2), (2, 4), (4, 6)]
    assert [i for start, stop in spy.requests for i in range(start, stop)] == list(range(6))


def test_read_frames_validates_the_request():
    spy = SpyDataset(np.zeros((N_RX, N_CHIRP, N_FAST, 20)))
    for bad in (0, -1, 21):
        with pytest.raises(audit.AuditError, match="n-frames"):
            audit.read_frames(spy, bad)


# ------------------------------------------------------ Parseval + crop accounting


def test_parseval_holds_on_the_full_spectrum_of_one_axis():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    spectrum = np.fft.fft(x)
    assert audit.normalized_energy(spectrum, 64) == pytest.approx(audit.normalized_energy(x))


def test_parseval_normalisation_is_cumulative_across_two_axes():
    """An implementation dividing only by the LAST transform length fails this."""
    rng = np.random.default_rng(6)
    x = rng.standard_normal((16, 8)) + 1j * rng.standard_normal((16, 8))
    two_axis = np.fft.fft(np.fft.fft(x, axis=0), axis=1)

    assert audit.normalized_energy(two_axis, 16 * 8) == pytest.approx(
        audit.normalized_energy(x)
    )
    assert audit.normalized_energy(two_axis, 8) != pytest.approx(
        audit.normalized_energy(x)
    )


def test_cropping_removes_energy_and_is_accounted_bin_by_bin():
    """Cropping is NOT Parseval-preserving: a cropped spectrum is a sum over its bins."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    spectrum = np.fft.fft(x)
    cropped = spectrum[GATE_BINS]

    assert audit.normalized_energy(cropped, 64) == pytest.approx(
        float(np.sum(np.abs(spectrum[GATE_BINS]) ** 2) / 64)
    )
    assert audit.normalized_energy(cropped, 64) < audit.normalized_energy(spectrum, 64)


def test_crop_preserves_energy_only_when_support_lies_inside_it():
    """Equality to time-domain energy needs the strict in-crop-support precondition."""
    spectrum = np.zeros(64, dtype=np.complex128)
    spectrum[GATE_BINS] = [1.0, 2.0, 3.0]  # all support inside the retained bins
    x = np.fft.ifft(spectrum)
    assert audit.normalized_energy(spectrum[GATE_BINS], 64) == pytest.approx(
        audit.normalized_energy(x)
    )


# ------------------------------------------------------------------- QC smoke rules


def test_qc_smoke_short_circuits_a_non_finite_frame():
    """Mirrors the frozen 10 GHz per-frame contract: skip, don't crash or fabricate."""
    frame = noise_cube(n_frames=1)[0].real.copy()
    frame[0, 0, 0] = np.nan
    mask = np.zeros(N_FAST // 2, dtype=bool)
    mask[GATE_BINS] = True

    result = audit.qc_smoke_frame(frame, mask, bins=16, max_bin_fraction=0.25, min_ratio=0.3)
    assert result["nan_inf"] is True
    assert result["flatline"] is False and result["low_in_band"] is False
    assert result["n_flatline_traces"] == 0
    assert np.isnan(result["in_band_ratio"])
    assert result["passed"] is False


def test_unavailable_floats_serialise_as_null_not_nan():
    """`NaN` is not valid JSON; allow_nan=False makes anything unhandled fail loudly."""
    payload = audit.json_safe(
        {"ratio": float("nan"), "inf": float("inf"), "ok": 0.25, "n": np.int64(3)}
    )
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text) == {"ratio": None, "inf": None, "ok": 0.25, "n": 3}
    assert "NaN" not in text


def test_constant_trace_counts_as_flatline():
    frame = np.ones((N_FAST, N_CHIRP, N_RX))
    mask = np.zeros(N_FAST // 2, dtype=bool)
    mask[GATE_BINS] = True
    result = audit.qc_smoke_frame(frame, mask, bins=16, max_bin_fraction=0.25, min_ratio=0.3)
    assert result["n_flatline_traces"] == N_CHIRP * N_RX
    assert result["per_rx_flatline"] == [N_CHIRP] * N_RX


# --------------------------------------------------------- frozen constants derive


def test_frozen_constants_match_independent_computation():
    dr = audit.SPEED_OF_LIGHT_M_S / (2 * audit.BANDWIDTH_HZ)
    assert dr == pytest.approx(0.0749, abs=1e-4)

    gate = audit.range_gate_bins(256, audit.BANDWIDTH_HZ, audit.RANGE_GATE_M)
    assert (gate[0], gate[-1]) == (27, 53)

    assert audit.PRF_HZ == pytest.approx(1953.125)
    assert audit.QC_IN_BAND_MARGIN_HZ == pytest.approx(audit.FS_HZ / 256)

    mask = np.flatnonzero(
        audit.qc_in_band_mask_77(
            256, audit.FS_HZ, audit.BANDWIDTH_HZ, audit.CHIRP_TIME_S,
            audit.RANGE_GATE_M, audit.QC_IN_BAND_MARGIN_HZ,
        )
    )
    assert (mask[0], mask[-1]) == (26, 54)


# ----------------------------------------------------------------- failure paths


def test_shape_mismatch_writes_provenance_json_and_exits_nonzero(tmp_path, config_yaml):
    path = write_fixture(tmp_path / "wrong.mat", tone_along("fast", n_frames=4))
    code, findings = run_audit(config_yaml, path)

    assert code != 0
    assert findings["verdicts"]["H1_shape"] == "REJECTED"
    assert findings["verdicts"]["H1_axes"] == "NOT_RUN"
    assert findings["verdicts"]["qc_smoke"] == "NOT_RUN"
    assert findings["verdicts"]["chain"] == "NOT_RUN"
    assert findings["storage"]["shape"] == [N_RX, N_CHIRP, N_FAST, 4]  # observed, recorded
    # ...and it is still provenance-bearing, so the failure is attributable.
    for key in ("input", "git", "versions", "constants", "conventions", "timestamp_utc"):
        assert key in findings
    assert findings["input"]["sha256"]


def test_unsupported_storage_writes_json_and_exits_nonzero(tmp_path, config_yaml):
    path = tmp_path / "int.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(audit.RADAR_VAR, shape=audit.EXPECTED_SHAPE, dtype="<i4")
    code, findings = run_audit(config_yaml, path)

    assert code != 0
    assert findings["verdicts"]["H1_storage"] == "REJECTED"
    assert findings["verdicts"]["H1_axes"] == "NOT_RUN"
    assert findings["storage"]["dtype_name"] == "int32"


def test_out_must_be_a_bare_filename(tmp_path, config_yaml):
    """The resolved config is the single output-path authority."""
    path = write_fixture(tmp_path / "x.mat", tone_along("fast", n_frames=2))
    with pytest.raises(SystemExit):
        audit.main(
            ["--config", str(config_yaml), "--file", str(path),
             "--out", str(tmp_path / "elsewhere.json")]
        )


# ------------------------------------------------- non-finite slab rule (end-to-end)


SMALL_SHAPE = (N_RX, N_CHIRP, N_FAST, 8)  # (rx, chirp, fast, frame), on-disk order


def _end_to_end_fixture(path: Path, n_frames: int, nan_frames=()) -> Path:
    """A small but structurally real cube, so the whole script can run in a test.

    32 fast-time bins still give a non-empty range gate (bins 27..31) and a non-empty
    QC mask (bins 4..6), which is what the end-to-end path needs.
    """
    rng = np.random.default_rng(1)
    data = rng.standard_normal((*SMALL_SHAPE[:3], n_frames))
    for frame in nan_frames:
        data[0, 0, 0, frame] = np.nan
    with h5py.File(path, "w") as handle:
        handle.create_dataset(audit.RADAR_VAR, data=data)
    return path


def test_non_finite_frames_are_excluded_and_recorded(tmp_path, config_yaml, monkeypatch):
    """A non-default --n-frames is used so a hard-coded 5 or 10 cannot hide."""
    monkeypatch.setattr(audit, "EXPECTED_SHAPE", (*SMALL_SHAPE[:3], 8))
    path = _end_to_end_fixture(tmp_path / "nf.mat", 8, nan_frames=(1, 4))
    _, findings = run_audit(config_yaml, path, extra=["--n-frames", "6"])

    slab = findings["slab"]
    assert slab["requested_frame_count"] == 6
    assert slab["min_finite_frames"] == 3  # ceil(0.5 * 6), NOT a hard-coded 5
    assert slab["excluded_frame_indices"] == [1, 4]
    assert slab["effective_frame_indices"] == [0, 2, 3, 5]
    assert findings["verdicts"]["H1_axes"] != "NOT_RUN"
    # The QC smoke deliberately still sees ALL frames — reporting them is its job.
    assert findings["qc_smoke"]["n_frames"] == 6
    assert findings["qc_smoke"]["n_nan_inf"] == 2


def test_too_few_finite_frames_stops_and_reports(tmp_path, config_yaml, monkeypatch):
    monkeypatch.setattr(audit, "EXPECTED_SHAPE", (*SMALL_SHAPE[:3], 8))
    path = _end_to_end_fixture(tmp_path / "few.mat", 8, nan_frames=(0, 1, 2))
    code, findings = run_audit(config_yaml, path, extra=["--n-frames", "4"])

    assert code != 0
    assert findings["slab"]["min_finite_frames"] == 2
    assert findings["verdicts"]["H1_axes"] == "NOT_RUN"
    assert findings["verdicts"]["chain"] == "NOT_RUN"
    assert findings["verdicts"]["qc_smoke"] in ("NON_DEGENERATE", "DEGENERATE")
