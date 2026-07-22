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

`apply_qc` and friends (milestone 2) add the per-frame reason codes and session
eligibility on top. They are bookkeeping only: the screens themselves live in
`dehyd.qc.screens`, and this module never computes one.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..qc.screens import run_qc_cube
from .ground_truth import GroundTruth
from .loader_10ghz import LoaderError, inspect_10ghz_file, load_10ghz_file, parse_10ghz_filename
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

# Added by apply_qc. The three reason flags are INDEPENDENT, NON-EXCLUSIVE incidence
# markers -- one frame can fail several screens at once -- so they never sum to the
# number of rejected frames. `qc_fail_any` is the one that reconciles.
QC_COLUMN_DTYPES = {
    "qc_nan_inf": "bool",
    "qc_flatline": "bool",
    "qc_low_in_band": "bool",
    "qc_rms_flag": "bool",
    "qc_pass": "bool",
    "qc_fail_any": "bool",
    "qc_in_band_ratio": "float64",  # NaN when unavailable (non-finite frame)
    "qc_max_rms_z": "float64",  # NaN when unavailable
    "qc_n_flatline_chirps": "int64",
    "qc_n_rms_outlier_chirps": "int64",
    "session_n_pass": "int64",
    "session_min_pass": "int64",
    "session_eligible": "bool",
}

QC_REASONS = ("qc_nan_inf", "qc_flatline", "qc_low_in_band")

JOIN_KEYS = ["rel_path", "frame_idx"]

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


# ------------------------------------------------------------------------------ QC


def _qc_rows(manifest: pd.DataFrame, paths, config) -> pd.DataFrame:
    """Run the frozen screens over every file, one load each."""
    rows = []
    for rel_path in manifest["rel_path"].drop_duplicates():
        cube = load_10ghz_file(resolve_path(paths, rel_path))
        for frame_idx, result in enumerate(run_qc_cube(cube, config.qc, config.preprocess)):
            rows.append(
                {
                    "rel_path": rel_path,
                    "frame_idx": frame_idx,
                    "qc_nan_inf": result.nan_inf,
                    "qc_flatline": result.flatline,
                    "qc_low_in_band": result.low_in_band,
                    "qc_rms_flag": result.rms_flag,
                    "qc_pass": result.passed,
                    "qc_in_band_ratio": result.in_band_ratio,
                    "qc_max_rms_z": result.max_rms_z,
                    "qc_n_flatline_chirps": result.n_flatline_chirps,
                    "qc_n_rms_outlier_chirps": result.n_rms_outlier_chirps,
                }
            )
    return pd.DataFrame(rows)


def _join_qc(manifest: pd.DataFrame, qc_rows: pd.DataFrame) -> pd.DataFrame:
    """Attach QC results by (rel_path, frame_idx), failing closed.

    The right key is necessary but not sufficient: this join must be unable to
    silently duplicate, drop, or misattach a frame. Never join by row index --
    `rel_path` string order is not session order (`subject_1_10am` sorts before
    `subject_1_8am`), a trap already paid for in milestone 1.
    """
    for frame, label in ((manifest, "manifest"), (qc_rows, "QC results")):
        duplicated = frame.duplicated(subset=JOIN_KEYS)
        if duplicated.any():
            offenders = frame.loc[duplicated, JOIN_KEYS].head(5).to_dict("records")
            raise ManifestError(f"duplicate (rel_path, frame_idx) in {label}: {offenders}")

    merged = manifest.merge(
        qc_rows, on=JOIN_KEYS, how="outer", validate="one_to_one", indicator=True
    )
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        offenders = unmatched[JOIN_KEYS + ["_merge"]].head(5).to_dict("records")
        raise ManifestError(
            f"QC results do not match the manifest one-to-one: {offenders} "
            f"({len(unmatched)} unmatched of {len(merged)})"
        )
    if len(merged) != len(manifest):
        raise ManifestError(
            f"QC join changed the row count: {len(manifest)} -> {len(merged)}"
        )
    return merged.drop(columns="_merge")


def apply_qc(manifest: pd.DataFrame, paths, config) -> pd.DataFrame:
    """Add per-frame QC verdicts and per-session eligibility to the manifest.

    Eligibility uses the file's ACTUAL frame count, never an assumed 100: a session is
    retained iff at least `ceil(min_frame_fraction * n_frames_in_file)` frames survive.
    Dropped sessions are simply absent from `eligible_frames` -- never imputed -- but
    stay visible in the manifest and in `session_qc_report`, because QC failure may
    itself correlate with hydration or acquisition quality.
    """
    merged = _join_qc(manifest, _qc_rows(manifest, paths, config))
    merged["qc_fail_any"] = ~merged["qc_pass"]

    by_session = merged.groupby(["subject", "session_idx"])["qc_pass"].transform("sum")
    merged["session_n_pass"] = by_session
    merged["session_min_pass"] = [
        math.ceil(config.qc.min_frame_fraction * n) for n in merged["n_frames_in_file"]
    ]
    merged["session_eligible"] = merged["session_n_pass"] >= merged["session_min_pass"]

    merged = merged.sort_values(SORT_KEYS).reset_index(drop=True)
    return merged.astype({**COLUMN_DTYPES, **QC_COLUMN_DTYPES})


def session_qc_report(manifest_qc: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, session): what survived, what was removed and why.

    The per-reason columns are non-additive incidence counts (a frame failing two
    screens appears in both). The identity that holds is
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
        n_rms_flagged=("qc_rms_flag", "sum"),
        min_pass=("session_min_pass", "first"),
        eligible=("session_eligible", "first"),
    )
    return grouped.sort_values(["subject", "session_idx"]).reset_index(drop=True)


def eligible_frames(manifest_qc: pd.DataFrame) -> pd.DataFrame:
    """The analysis population: passing frames of eligible sessions.

    This is the ONLY view modeling may consume. Ineligible sessions are absent here
    and never imputed; they remain in `session_qc_report` so missingness stays visible.
    """
    keep = manifest_qc["qc_pass"] & manifest_qc["session_eligible"]
    return manifest_qc[keep].reset_index(drop=True)


def evaluable_subjects(manifest_qc: pd.DataFrame) -> tuple[int, ...]:
    """Subjects with >= 1 eligible session -- the Exp A rule, which drives N_eval."""
    eligible = manifest_qc[manifest_qc["session_eligible"]]
    return tuple(sorted(int(s) for s in eligible["subject"].unique()))
