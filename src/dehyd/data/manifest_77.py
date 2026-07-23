"""The 77 GHz frame index table and its structural gate (band 2, milestone 5).

A mirror of manifest.py for the 77 GHz files: one row per frame, joining radar files to
their ground-truth target (the SAME body-mass Δm — 77 GHz is the same 16-subject cohort).
It **reuses the genuinely subtle pieces by import** rather than reimplementing them —
`_join_qc` (the fail-closed one-to-one QC join, where a silent bug would drop/duplicate a
frame), `eligible_frames`, `evaluable_subjects`, and the base `COLUMN_DTYPES`/`SORT_KEYS`/
`_describe`/`ManifestError` — and mirrors only the shallow bookkeeping whose columns differ.

Mirroring rather than parameterizing manifest.py is deliberate (CLAUDE.md: no factory-style
indirection): injecting a loader/QC/path triple into build_manifest would touch the frozen,
artifact-pinned 10 GHz path for zero 10 GHz benefit. The divergent-copy risk is contained by
sharing the subtle join/eligibility logic through import.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..qc.screens_77 import run_qc_cube_77
from .ground_truth import GroundTruth
from .loader_77ghz import LoaderError77, inspect_77ghz_file, load_77ghz_file, parse_77ghz_filename
from .manifest import (  # reused as-is — column-generic and subtle
    COLUMN_DTYPES,
    JOIN_KEYS,
    SORT_KEYS,
    ManifestError,
    _describe,
    _join_qc,
    eligible_frames,  # noqa: F401 — re-exported for the 77 GHz CLIs
    evaluable_subjects,  # noqa: F401 — re-exported for the 77 GHz CLIs
)
from .sessions import SESSION_NAMES

# Added by apply_qc_77. Mirrors QC_COLUMN_DTYPES but for the three 77 GHz screens (no RMS
# diagnostic) plus the per-Rx flatline summary. The three reason flags are INDEPENDENT,
# NON-EXCLUSIVE incidence markers; `qc_fail_any` is the one that reconciles.
QC77_COLUMN_DTYPES = {
    "qc_nan_inf": "bool",
    "qc_flatline": "bool",
    "qc_low_in_band": "bool",
    "qc_pass": "bool",
    "qc_fail_any": "bool",
    "qc_in_band_ratio": "float64",  # NaN when unavailable (non-finite frame)
    "qc_n_flatline_traces": "int64",
    "qc_rx_max_flatline": "int64",  # max over the 16 per-Rx flatline counts
    "session_n_pass": "int64",
    "session_min_pass": "int64",
    "session_eligible": "bool",
}

QC77_REASONS = ("qc_nan_inf", "qc_flatline", "qc_low_in_band")


def resolve_path_77(paths, rel_path: str) -> Path:
    """Logical manifest identity -> the physical 77 GHz file on this machine."""
    if paths.data_77ghz_dir is None:
        raise ManifestError("paths.data_77ghz_dir is not set (a 77 GHz run needs it)")
    return Path(paths.data_77ghz_dir) / rel_path


def build_manifest_77(paths, gt: GroundTruth) -> pd.DataFrame:
    """Build the per-frame 77 GHz manifest, failing on any structural mismatch.

    Mirrors build_manifest (C1-C6) against paths.data_77ghz_dir via inspect_77ghz_file. The
    per-file frame count is read from the file, never assumed 125.
    """
    if paths.data_77ghz_dir is None:
        raise ManifestError("paths.data_77ghz_dir is not set (a 77 GHz run needs it)")
    data_dir = Path(paths.data_77ghz_dir)
    if not data_dir.is_dir():
        raise ManifestError(f"data_77ghz_dir is not a directory: {data_dir}")

    problems: list[str] = []

    # --- C3: every .mat file parses to a valid (subject, session) ------------------
    infos: dict[tuple[int, int], object] = {}
    duplicates: list[str] = []
    for path in sorted(data_dir.glob("*.mat")):
        try:
            subject, session_idx = parse_77ghz_filename(path)
        except LoaderError77 as exc:
            problems.append(f"unmatched file: {exc}")
            continue

        key = (subject, session_idx)
        # --- C2: no (subject, session) claimed twice -------------------------------
        if key in infos:
            duplicates.append(f"subject {subject} {SESSION_NAMES[session_idx]}")
            continue

        # --- C5: per-file structure ------------------------------------------------
        try:
            infos[key] = inspect_77ghz_file(path)
        except LoaderError77 as exc:
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
        # --- C6: the ACTUAL frame count, never an assumed 125 ----------------------
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
    # Deterministic order: filesystem enumeration order must never reach training order.
    manifest = manifest.sort_values(SORT_KEYS).reset_index(drop=True)
    return manifest.astype(COLUMN_DTYPES)


# ------------------------------------------------------------------------------ QC


def _qc_rows_77(manifest: pd.DataFrame, paths, config) -> pd.DataFrame:
    """Run the frozen 77 GHz screens over every file, one load each."""
    rows = []
    for rel_path in manifest["rel_path"].drop_duplicates():
        cube = load_77ghz_file(resolve_path_77(paths, rel_path))
        for frame_idx, result in enumerate(
            run_qc_cube_77(cube, config.qc77, config.preprocess77)
        ):
            rows.append(
                {
                    "rel_path": rel_path,
                    "frame_idx": frame_idx,
                    "qc_nan_inf": result.nan_inf,
                    "qc_flatline": result.flatline,
                    "qc_low_in_band": result.low_in_band,
                    "qc_pass": result.passed,
                    "qc_in_band_ratio": result.in_band_ratio,
                    "qc_n_flatline_traces": result.n_flatline_traces,
                    "qc_rx_max_flatline": (
                        max(result.per_rx_flatline) if result.per_rx_flatline else 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _finalize_qc_77(merged: pd.DataFrame, min_frame_fraction: float) -> pd.DataFrame:
    """Add qc_fail_any + per-session eligibility to an already QC-joined frame.

    Factored out of apply_qc_77 so the eligibility arithmetic is unit-testable without
    loading 1 GB cubes and running 4096-trace-per-frame screens. Eligibility uses the
    file's ACTUAL frame count: a session is retained iff at least
    ceil(min_frame_fraction * n_frames_in_file) frames survive.
    """
    merged = merged.copy()
    merged["qc_fail_any"] = ~merged["qc_pass"]
    by_session = merged.groupby(["subject", "session_idx"])["qc_pass"].transform("sum")
    merged["session_n_pass"] = by_session
    merged["session_min_pass"] = [
        math.ceil(min_frame_fraction * n) for n in merged["n_frames_in_file"]
    ]
    merged["session_eligible"] = merged["session_n_pass"] >= merged["session_min_pass"]
    return merged.sort_values(SORT_KEYS).reset_index(drop=True)


def apply_qc_77(manifest: pd.DataFrame, paths, config) -> pd.DataFrame:
    """Add per-frame QC verdicts and per-session eligibility to the 77 GHz manifest.

    Dropped sessions are simply absent from eligible_frames -- never imputed -- but stay
    visible in the manifest and session_qc_report_77.
    """
    merged = _join_qc(manifest, _qc_rows_77(manifest, paths, config))
    merged = _finalize_qc_77(merged, config.qc77.min_frame_fraction)
    return merged.astype({**COLUMN_DTYPES, **QC77_COLUMN_DTYPES})


def session_qc_report_77(manifest_qc: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, session): what survived, what was removed and why (no RMS).

    Per-reason columns are non-additive incidence counts; the identity that holds is
    `n_pass + n_fail_any == n_frames`.
    """
    grouped = manifest_qc.groupby(["subject", "session_idx"], as_index=False).agg(
        session_name=("session_name", "first"),
        rel_path=("rel_path", "first"),
        n_frames=("frame_idx", "count"),
        n_pass=("qc_pass", "sum"),
        n_fail_any=("qc_fail_any", "sum"),
        n_nan_inf=("qc_nan_inf", "sum"),
        n_flatline=("qc_flatline", "sum"),
        n_low_in_band=("qc_low_in_band", "sum"),
        min_pass=("session_min_pass", "first"),
        eligible=("session_eligible", "first"),
    )
    return grouped.sort_values(["subject", "session_idx"]).reset_index(drop=True)
