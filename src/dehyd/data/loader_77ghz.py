"""77 GHz Inras radar file loading — milestone-5 scope (band 2).

Filename parsing, HDF5 header inspection, and a full load with hard assertions. QC
screens, the semantic axis check, and preprocessing live in their own modules.

Verified facts about the real files (M2 audit, results/qc/audit_77ghz.json — not assumed
from the paper):
  data/77ghz/subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat, MAT v7.3 / HDF5, gzip-chunked.
  `framesRadar` is a REAL float64 array stored on disk as (Nrx, Nchirps, Nfast, Nframes)
  = (16, 256, 256, n_frames). It is REAL — I/Q only arises after the range FFT, so MTI
  (on real ADC data) and every pre-FFT step assume real input. The MAT v7.3 chunk layout
  spans all frames, so any frame-subset read decompresses the whole file anyway; hence
  load_77ghz_file reads the WHOLE dataset in one call (a per-frame loop would decompress
  the file n_frames times).

reverse_axes / to_numeric are the audit's promoted helpers; experiments/audit_77ghz.py
imports them back so there is one copy, and tests/test_audit_77ghz.py keeps guarding them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .sessions import SESSION_INDEX, SESSION_NAMES

N_RX = 16
N_CHIRPS = 256
N_FAST = 256
RADAR_VAR = "framesRadar"

# On-disk axis order as h5py presents it: (Nrx, Nchirps, Nfast, Nframes). The frame
# count is the only free dimension.
ON_DISK_LEADING_SHAPE = (N_RX, N_CHIRPS, N_FAST)

_FILENAME_RE = re.compile(
    r"^subject_(?P<subject>\d+)_(?P<session>" + "|".join(SESSION_NAMES) + r")\.mat$"
)


class LoaderError77(ValueError):
    """Raised when a 77 GHz file's name or contents violate the expected structure."""


@dataclass(frozen=True)
class FileInfo77:
    path: Path
    subject: int
    session_idx: int
    n_frames: int
    shape: tuple[int, ...]  # the ON-DISK shape (16, 256, 256, n_frames)


def parse_77ghz_filename(path: str | Path) -> tuple[int, int]:
    """`subject_7_12pm.mat` -> (7, 2). Raises on anything that does not match.

    Strictness is deliberate (the loader_10ghz rule): an unparseable name is an
    *unmatched file*, which the manifest must fail on rather than quietly skip.
    """
    name = Path(path).name
    match = _FILENAME_RE.match(name)
    if match is None:
        raise LoaderError77(
            f"filename does not match subject_<id>_<session>.mat: {name!r} "
            f"(sessions: {', '.join(SESSION_NAMES)})"
        )
    return int(match.group("subject")), SESSION_INDEX[match.group("session")]


def to_numeric(raw: np.ndarray) -> np.ndarray:
    """Compound (real, imag) -> complex128; a real float array is passed through.

    Promoted from the audit. The 77 GHz loader only ever hands this a real float64 array
    (inspect_77ghz_file rejects compound before any read), but the compound branch stays
    because the audit's round-trip test exercises it.
    """
    if raw.dtype.names is None:
        return np.asarray(raw, dtype=np.float64)
    return raw["real"].astype(np.float64) + 1j * raw["imag"].astype(np.float64)


def reverse_axes(cube: np.ndarray) -> np.ndarray:
    """(Nrx, Nchirps, Nfast, Nframes) -> (Nframes, Nfast, Nchirps, Nrx).

    Promoted from the audit. MAT v7.3 stores dimensions in reverse of the MATLAB-logical
    order, so the full axis reversal recovers the layout chirpavg_and_fuse_batch.m expects.
    The two size-256 axes (fast-time vs chirps) are indistinguishable by shape; this
    function only records the assumed mapping — the semantic axis check (axis_check_77)
    certifies it per file at QC time.
    """
    return np.transpose(cube, (3, 2, 1, 0))


def inspect_77ghz_file(path: str | Path) -> FileInfo77:
    """Read the HDF5 metadata only (no chunk decompress) and validate the on-disk contract.

    Enforces the M2 finding hard, because every downstream assumption rests on it:
      * COMPOUND (real, imag) is REJECTED — MTI and the pre-FFT chain assume real ADC
        data; a complex file is never silently coerced.
      * dtype must be a REAL float of EXACTLY 8 bytes (float64). float32 / other widths /
        big-endian are REJECTED — format drift is a stop-and-report (C5-18), not a
        silent reinterpretation.
      * on-disk shape must be exactly (16, 256, 256, n_frames), n_frames > 0. The frame
        count is read from the file, never assumed to be 125.
    """
    path = Path(path)
    subject, session_idx = parse_77ghz_filename(path)

    try:
        handle = h5py.File(path, "r")
    except Exception as exc:  # unreadable / not an HDF5 file
        raise LoaderError77(f"cannot open HDF5 file {path}: {exc}") from exc

    with handle:
        if RADAR_VAR not in handle:
            found = ", ".join(handle.keys()) or "<nothing>"
            raise LoaderError77(f"{path}: no '{RADAR_VAR}' dataset (found: {found})")
        dset = handle[RADAR_VAR]
        dtype = dset.dtype
        shape = tuple(int(d) for d in dset.shape)

    # dtype first, so a tiny wrong-dtype fixture is rejected on dtype (not shape).
    if dtype.names is not None:
        raise LoaderError77(
            f"{path}: '{RADAR_VAR}' is a compound dtype {dtype.names} — the M2 audit "
            "confirmed the 77 GHz files are REAL float64 (I/Q arises only after the range "
            "FFT); a complex/compound file is never coerced. Stop and re-inspect."
        )
    if dtype.kind != "f":
        raise LoaderError77(
            f"{path}: '{RADAR_VAR}' dtype {dtype} is not a real float (kind {dtype.kind!r})"
        )
    if dtype.itemsize != 8:
        raise LoaderError77(
            f"{path}: '{RADAR_VAR}' dtype {dtype} is {dtype.itemsize*8}-bit float, expected "
            "64-bit (float64) — format drift; float32 is not silently accepted (C5-18)"
        )
    # The confirmed contract is little-endian float64. '<' and '=' are little/native on the
    # LE dev + IBEX machines; explicit big-endian ('>') is drift and is rejected.
    if dtype.byteorder not in ("<", "=", "|"):
        raise LoaderError77(
            f"{path}: '{RADAR_VAR}' dtype {dtype} has unexpected byte order "
            f"{dtype.byteorder!r}, expected little-endian float64 (C5-18)"
        )

    if len(shape) != 4 or shape[:3] != ON_DISK_LEADING_SHAPE:
        raise LoaderError77(
            f"{path}: '{RADAR_VAR}' on-disk shape {shape}, expected "
            f"({N_RX}, {N_CHIRPS}, {N_FAST}, n_frames)"
        )
    n_frames = shape[3]
    if n_frames <= 0:
        raise LoaderError77(f"{path}: '{RADAR_VAR}' has {n_frames} frames")

    return FileInfo77(
        path=path,
        subject=subject,
        session_idx=session_idx,
        n_frames=n_frames,
        shape=shape,
    )


def load_77ghz_file(path: str | Path) -> np.ndarray:
    """Load one session's cube: float64 [n_frames, 256 fast, 256 chirp, 16 rx].

    Reads the WHOLE dataset in one call, then reverse_axes. The chunk layout spans all
    frames, so any frame-subset read decompresses the entire file regardless — a whole-
    file read is both simplest and cheapest. ~1.05 GB in memory for a 125-frame file.
    """
    path = Path(path)
    info = inspect_77ghz_file(path)

    with h5py.File(path, "r") as handle:
        raw = handle[RADAR_VAR][()]  # whole dataset, one decompress

    cube = reverse_axes(to_numeric(raw))

    expected = (info.n_frames, N_FAST, N_CHIRPS, N_RX)
    if cube.shape != expected:
        raise LoaderError77(
            f"{path}: loaded shape {cube.shape} disagrees with the reversed on-disk "
            f"shape {expected}"
        )
    if cube.dtype != np.float64:
        raise LoaderError77(
            f"{path}: '{RADAR_VAR}' loaded as {cube.dtype}, expected float64"
        )
    return cube
