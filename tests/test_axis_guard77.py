"""The per-entrypoint axis-certification guard (T-A77, no private data).

require_accepted_axis is the hard guard every 77 GHz extraction/preprocess entrypoint (and
the smoke) calls per file: a matching ACCEPTED record in the survival CSV (keyed to raw
sha256 + axis_spec_hash) OR an inline semantic check, both requiring ACCEPTED. It uses a
tiny on-disk file for hashing and an injected loader so it needs no full-shape fixture.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dehyd.config import Preprocess77Config
from dehyd.qc.axis_check_77 import AxisCertError, axis_spec_hash, require_accepted_axis
from dehyd.provenance import sha256_file

CONFIG = SimpleNamespace(preprocess77=Preprocess77Config())


def real_tone_cube(axis, bin_idx=40, n_fast=64, n_chirp=64, n_rx=2, n_frames=2):
    _, fast, chirp, _ = np.indices((n_frames, n_fast, n_chirp, n_rx))
    index = fast if axis == "fast" else chirp
    return np.cos(2 * np.pi * bin_idx * index / n_fast).astype(np.float64)


@pytest.fixture
def raw_file(tmp_path):
    """A tiny file that just needs to exist for sha256 (the loader is injected)."""
    p = tmp_path / "subject_1_8am.mat"
    p.write_bytes(b"placeholder-bytes-for-hashing")
    return p


def _survival_csv(tmp_path, path, verdict, *, spec_hash=None):
    csv = tmp_path / "qc_survival_77ghz.csv"
    pd.DataFrame([{
        "rel_path": path.name,
        "sha256": sha256_file(path),
        "axis_spec_hash": spec_hash if spec_hash is not None else axis_spec_hash(CONFIG),
        "axis_verdict": verdict,
        "eligible": True,
    }]).to_csv(csv, index=False)
    return csv


def _accepting_loader(path):
    return real_tone_cube("fast")


def _rejecting_loader(path):
    return real_tone_cube("chirp")


# --------------------------------------------------------------------------- record path


def test_guard_passes_with_matching_accepted_record(raw_file, tmp_path):
    csv = _survival_csv(tmp_path, raw_file, "ACCEPTED")
    # The record matches, so the loader must NOT be called (raise if it is).
    def poison(_):
        raise AssertionError("loader was called despite a valid ACCEPTED record")
    out = require_accepted_axis(raw_file, CONFIG, survival_csv=csv, load=poison)
    assert out["source"] == "record" and out["verdict"] == "ACCEPTED"


def test_guard_rejects_non_accepted_record(raw_file, tmp_path):
    csv = _survival_csv(tmp_path, raw_file, "INCONCLUSIVE")
    with pytest.raises(AxisCertError, match="not ACCEPTED"):
        require_accepted_axis(raw_file, CONFIG, survival_csv=csv, load=_accepting_loader)


def test_guard_record_sha_mismatch_falls_to_inline(raw_file, tmp_path):
    """A record whose sha256 does not match this file is ignored; the inline check runs."""
    csv = tmp_path / "qc_survival_77ghz.csv"
    pd.DataFrame([{"rel_path": raw_file.name, "sha256": "deadbeef",
                   "axis_spec_hash": axis_spec_hash(CONFIG), "axis_verdict": "ACCEPTED"}]).to_csv(csv, index=False)
    out = require_accepted_axis(raw_file, CONFIG, survival_csv=csv, load=_accepting_loader)
    assert out["source"] == "inline" and out["verdict"] == "ACCEPTED"


def test_guard_spec_hash_mismatch_falls_to_inline(raw_file, tmp_path):
    """A changed axis-relevant constant invalidates the certificate -> inline re-check."""
    csv = _survival_csv(tmp_path, raw_file, "ACCEPTED", spec_hash="not-the-current-hash")
    out = require_accepted_axis(raw_file, CONFIG, survival_csv=csv, load=_accepting_loader)
    assert out["source"] == "inline"


# --------------------------------------------------------------------------- inline path


def test_guard_inline_accepts_proposed_mapping(raw_file):
    out = require_accepted_axis(raw_file, CONFIG, survival_csv=None, load=_accepting_loader)
    assert out["source"] == "inline" and out["verdict"] == "ACCEPTED"


def test_guard_inline_rejects_swapped_mapping(raw_file):
    with pytest.raises(AxisCertError, match="not ACCEPTED|REJECTED"):
        require_accepted_axis(raw_file, CONFIG, survival_csv=None, load=_rejecting_loader)


def test_guard_inline_rejects_inconclusive(raw_file):
    rng = np.random.default_rng(0)
    with pytest.raises(AxisCertError):
        require_accepted_axis(raw_file, CONFIG, survival_csv=None,
                              load=lambda _: rng.standard_normal((2, 64, 64, 2)))
