"""10 GHz CN0566 radar file loading — milestone-1 scope.

Filename parsing, header inspection, and a full load with hard assertions. QC screens
and frame filtering are milestone 2 and deliberately absent here.

Verified facts about the real files (not assumed from the paper):
  data/10ghz/subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat, MAT v5, zlib-compressed.
  `framesRadar` is a MATLAB *double* array of shape [534 fast-time x 20 chirps x
  N frames], complex. On disk the elements use the compact miINT16 type (a MAT-file
  space optimization) but the array CLASS is double, so loadmat returns complex128.
  `framesRadarIQ` [20834 x 2 x N] is raw pre-arrangement IQ, unused by the reference
  pipeline and unused here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from .sessions import SESSION_INDEX, SESSION_NAMES

N_FAST_TIME = 534
N_CHIRPS = 20

RADAR_VAR = "framesRadar"

_FILENAME_RE = re.compile(
    r"^subject_(?P<subject>\d+)_(?P<session>" + "|".join(SESSION_NAMES) + r")\.mat$"
)


class LoaderError(ValueError):
    """Raised when a radar file's name or contents violate the expected structure."""


@dataclass(frozen=True)
class FileInfo:
    path: Path
    subject: int
    session_idx: int
    n_frames: int
    shape: tuple[int, ...]


def parse_10ghz_filename(path: str | Path) -> tuple[int, int]:
    """`subject_7_12pm.mat` -> (7, 2). Raises on anything that does not match.

    Strictness is deliberate: an unparseable name is an *unmatched file*, which the
    manifest must fail on rather than quietly skip.
    """
    name = Path(path).name
    match = _FILENAME_RE.match(name)
    if match is None:
        raise LoaderError(
            f"filename does not match subject_<id>_<session>.mat: {name!r} "
            f"(sessions: {', '.join(SESSION_NAMES)})"
        )
    return int(match.group("subject")), SESSION_INDEX[match.group("session")]


def inspect_10ghz_file(path: str | Path) -> FileInfo:
    """Read the MAT header only (no array data) and validate structure.

    Uses scipy.io.whosmat so building an 80-file manifest costs ~1.4 s rather than
    decompressing ~1.4 GB. The frame count is read from the file, never assumed to
    be 100 — files may vary and session eligibility is computed from the actual count.
    """
    path = Path(path)
    subject, session_idx = parse_10ghz_filename(path)

    try:
        contents = sio.whosmat(str(path))
    except Exception as exc:  # unreadable / not a MAT file
        raise LoaderError(f"cannot read MAT header of {path}: {exc}") from exc

    entry = next((item for item in contents if item[0] == RADAR_VAR), None)
    if entry is None:
        found = ", ".join(name for name, _, _ in contents) or "<nothing>"
        raise LoaderError(f"{path}: no '{RADAR_VAR}' variable (found: {found})")

    _, shape, mat_class = entry

    # The array class must be double: complex128 on load depends on it, and a
    # differently-typed variable of the right shape would otherwise slip through.
    if mat_class != "double":
        raise LoaderError(
            f"{path}: '{RADAR_VAR}' has MATLAB class {mat_class!r}, expected 'double'"
        )
    if len(shape) != 3:
        raise LoaderError(f"{path}: '{RADAR_VAR}' has shape {shape}, expected 3 dimensions")
    if shape[:2] != (N_FAST_TIME, N_CHIRPS):
        raise LoaderError(
            f"{path}: '{RADAR_VAR}' has shape {shape}, expected "
            f"({N_FAST_TIME}, {N_CHIRPS}, n_frames)"
        )

    n_frames = int(shape[2])
    if n_frames <= 0:
        raise LoaderError(f"{path}: '{RADAR_VAR}' has {n_frames} frames")

    return FileInfo(
        path=path,
        subject=subject,
        session_idx=session_idx,
        n_frames=n_frames,
        shape=tuple(int(d) for d in shape),
    )


def load_10ghz_file(path: str | Path) -> np.ndarray:
    """Load one session's cube: complex128 [534 fast-time x 20 chirps x n_frames].

    Only `framesRadar` is requested, so the large unused `framesRadarIQ` array is never
    decompressed.
    """
    path = Path(path)
    info = inspect_10ghz_file(path)

    mat = sio.loadmat(str(path), variable_names=[RADAR_VAR])
    if RADAR_VAR not in mat:
        raise LoaderError(f"{path}: '{RADAR_VAR}' missing after load")

    cube = mat[RADAR_VAR]
    if cube.dtype != np.complex128:
        raise LoaderError(
            f"{path}: '{RADAR_VAR}' loaded as {cube.dtype}, expected complex128"
        )
    if cube.shape != info.shape:
        raise LoaderError(
            f"{path}: loaded shape {cube.shape} disagrees with header {info.shape}"
        )
    return cube
