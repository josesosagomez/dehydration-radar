"""The frame index table, and the structural gate over the raw file inventory.

One row per frame, joining radar files to their ground-truth target. This is where a
mislabeled, missing, duplicated or stray file is caught: the owner-confirmed
radar `subject_N` <-> workbook "Subject N" identity is *verified* here rather than
trusted, so a silently absent or renamed file cannot reach modeling.

File identity is LOGICAL, not physical: `rel_path` is relative to the configured
data_10ghz_dir (e.g. "subject_1_8am.mat") and is resolved against that root for I/O and
hashing. A repository-relative path would not be portable — on IBEX the data root lives
outside the repo, so the same file would acquire machine-specific '..' segments and a
different identity on each machine.

QC columns (per-frame reason codes, session eligibility) are added at milestone 2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ground_truth import GroundTruth
from .loader_10ghz import LoaderError, inspect_10ghz_file, parse_10ghz_filename
from .sessions import SESSION_NAMES

COLUMN_DTYPES = {
    "subject": "int64",
    "session_idx": "int64",
    "session_name": "string",
    "rel_path": "string",
    "n_frames_in_file": "int64",
    "frame_idx": "int64",
    "delta_m_pct": "float64",
    "class_label": "int64",
}

SORT_KEYS = ["subject", "session_idx", "frame_idx"]


class ManifestError(ValueError):
    """Raised when the file inventory does not match the ground truth exactly."""


def _describe(cells) -> str:
    return ", ".join(f"subject {s} {SESSION_NAMES[i]}" for s, i in sorted(cells))


def build_manifest(paths, gt: GroundTruth) -> pd.DataFrame:
    """Build the per-frame manifest, failing on any structural mismatch.

    `paths` is a PathsConfig (or anything with .data_10ghz_dir).
    """
    data_dir = Path(paths.data_10ghz_dir)
    if not data_dir.is_dir():
        raise ManifestError(f"data_10ghz_dir is not a directory: {data_dir}")

    problems: list[str] = []

    # --- C3: every .mat file parses to a valid (subject, session) ------------------
    infos: dict[tuple[int, int], object] = {}
    duplicates: list[str] = []
    for path in sorted(data_dir.glob("*.mat")):
        try:
            subject, session_idx = parse_10ghz_filename(path)
        except LoaderError as exc:
            problems.append(f"unmatched file: {exc}")
            continue

        key = (subject, session_idx)
        # --- C2: no (subject, session) claimed twice -------------------------------
        if key in infos:
            duplicates.append(f"subject {subject} {SESSION_NAMES[session_idx]}")
            continue

        # --- C5: per-file structure ------------------------------------------------
        try:
            infos[key] = inspect_10ghz_file(path)
        except LoaderError as exc:
            problems.append(f"malformed file: {exc}")

    if duplicates:
        problems.append(f"duplicate (subject, session) files: {', '.join(sorted(duplicates))}")

    # --- C4/C1: bijection with the ground truth -----------------------------------
    expected = {(int(r.subject), int(r.session_idx)) for r in gt.sessions.itertuples()}
    found = set(infos)

    missing = expected - found
    if missing:
        problems.append(f"missing radar file for: {_describe(missing)}")

    unmatched = found - expected
    if unmatched:
        problems.append(f"radar file with no ground-truth row: {_describe(unmatched)}")

    if problems:
        raise ManifestError("manifest validation failed:\n  " + "\n  ".join(problems))

    # --- build rows ---------------------------------------------------------------
    targets = {
        (int(r.subject), int(r.session_idx)): float(r.delta_m_pct)
        for r in gt.sessions.itertuples()
    }

    rows = []
    for (subject, session_idx), info in infos.items():
        rel_path = info.path.relative_to(data_dir).as_posix()
        delta = targets[(subject, session_idx)]
        # --- C6: the ACTUAL frame count, never an assumed 100 ----------------------
        for frame_idx in range(info.n_frames):
            rows.append(
                {
                    "subject": subject,
                    "session_idx": session_idx,
                    "session_name": SESSION_NAMES[session_idx],
                    "rel_path": rel_path,
                    "n_frames_in_file": info.n_frames,
                    "frame_idx": frame_idx,
                    "delta_m_pct": delta,
                    "class_label": session_idx,
                }
            )

    manifest = pd.DataFrame(rows, columns=list(COLUMN_DTYPES))
    # Deterministic order: filesystem enumeration order must never reach training
    # order, hashes, or saved artifacts.
    manifest = manifest.sort_values(SORT_KEYS).reset_index(drop=True)
    return manifest.astype(COLUMN_DTYPES)


def resolve_path(paths, rel_path: str) -> Path:
    """Logical manifest identity -> the physical file on this machine."""
    return Path(paths.data_10ghz_dir) / rel_path
