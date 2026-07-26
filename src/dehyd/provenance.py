"""Per-run provenance: what data, what code, what config produced a result.

Every run writes one provenance.json. `config.paths.results_dir` is the single output
authority — there is no out_dir parameter, so the destination cannot be specified two
ways and disagree. Tests point results_dir at a tmp directory.

Radar files are recorded as {rel_path, sha256}: the logical manifest identity (§
manifest.py) plus a hash of the resolved physical file, so a run on Windows and a run
on IBEX over the same data produce identical entries while the hash still proves which
bytes were read. The ground-truth workbook is hashed too — otherwise labels could
change without provenance noticing.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .config import config_to_dict

# Filesystem-safe UTC stamp: no colons (invalid in Windows paths) and microsecond
# precision so two runs started in the same second cannot collide.
RUN_STAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "openpyxl",
    "PyYAML",
    "scikit-learn",
    "kymatio",
    "pytest",
    "torch",  # absent until milestone 4; recorded as None until then
    "h5py",   # absent until milestone 2
    "matplotlib",  # added milestone 7 (Exp A scatter)
)


class ProvenanceError(RuntimeError):
    """Raised when a run's provenance cannot be written safely."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _env_dirty():
    """Parse DEHYD_GIT_DIRTY into a bool, or None if unset/blank.

    The submit wrapper captures the tree state at submit time; "1"/"true"/"yes" mean a
    dirty tree, "0"/"false"/"no" a clean one. Anything else (or unset) is None.
    """
    raw = os.environ.get("DEHYD_GIT_DIRTY")
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no", ""):
        return False
    return None


def _revision_file_commit():
    """A commit hash from a `REVISION` file at the repo root, or None.

    For environments that are a COPY of the tree rather than a git checkout — e.g. IBEX,
    where a private repo can't be pulled so the folders are copied over. Create it on the
    machine that DOES have git before copying:  `git rev-parse HEAD > REVISION`. Its first
    line is the commit hash. Ignored by git (so it never dirties a real checkout), and only
    consulted when both live git and the DEHYD_GIT_* env vars gave nothing.
    """
    path = Path(__file__).resolve().parents[2] / "REVISION"
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text.splitlines()[0].strip()
    return None


def _git_info() -> dict:
    """Git revision for provenance, with fallbacks for environments git can't answer.

    Precedence per field: live `git` -> DEHYD_GIT_* env vars -> (commit only) a `REVISION`
    file at the repo root. On IBEX compute nodes the in-process `git` call returns None
    (safe.directory does not take there); and when the tree was COPIED rather than cloned
    (no .git at all), the env vars or the REVISION file supply the commit. A run therefore
    self-attests its revision even where git itself cannot answer.
    """
    def run(args):
        try:
            out = subprocess.run(
                args,
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    # Per-field fallback: use a fallback only where the live git call gave nothing, so a
    # working local checkout always reports its own true state and never a stale value.
    if commit is None:
        commit = os.environ.get("DEHYD_GIT_COMMIT") or _revision_file_commit() or None
    if branch is None:
        branch = os.environ.get("DEHYD_GIT_BRANCH") or None
    dirty = None if status is None else bool(status)
    if dirty is None:
        dirty = _env_dirty()

    return {"commit": commit, "dirty": dirty, "branch": branch}


def _package_versions() -> dict:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _fold_manifest(folds) -> list[dict]:
    """Each subject's role per fold, in a canonical order."""
    if folds is None:
        return []
    return [
        {
            "test_subject": fold.test_subject,
            "train_subjects": sorted(fold.train_subjects),
            "selectable": fold.selectable,
            "inner_folds": [
                {
                    "train_subjects": sorted(inner.train_subjects),
                    "val_subjects": sorted(inner.val_subjects),
                }
                for inner in fold.inner_folds
            ],
        }
        for fold in sorted(folds, key=lambda f: f.test_subject)
    ]


def _hash_inputs(config, manifest, data_dir=None) -> dict:
    # data_dir defaults to the 10 GHz root (all existing call sites unchanged); 77 GHz
    # entrypoints pass require_77ghz_dir(config). Array tasks pass a single-session manifest
    # slice, so only that one file is hashed (hashing 22 GB in each of 80 tasks is pure waste).
    data_dir = Path(config.paths.data_10ghz_dir if data_dir is None else data_dir)

    radar = [
        {"rel_path": rel_path, "sha256": sha256_file(data_dir / rel_path)}
        for rel_path in sorted(manifest["rel_path"].unique().tolist())
    ]
    workbook = Path(config.paths.weight_xlsx)
    return {
        "radar_files": radar,
        "ground_truth": {
            "rel_path": workbook.name,
            "sha256": sha256_file(workbook),
        },
    }


def record_run(config, manifest, folds=None, extra: dict | None = None, data_dir=None) -> Path:
    """Write provenance.json for this run and return its path.

    Output goes to results_dir/runs/<stamp>_<git-shortrev>/provenance.json. If that
    file already exists the call raises rather than overwriting: repeated runs must
    never silently clobber the record of an earlier one.

    `data_dir` selects which data root the manifest's rel_paths hash against; it defaults
    to the 10 GHz root, and 77 GHz entrypoints pass require_77ghz_dir(config).
    """
    now = datetime.now(timezone.utc)
    git = _git_info()
    short_rev = (git["commit"] or "nogit")[:8]

    run_dir = Path(config.paths.results_dir) / "runs" / f"{now.strftime(RUN_STAMP_FORMAT)}_{short_rev}"
    out_path = run_dir / "provenance.json"
    if out_path.exists():
        raise ProvenanceError(f"provenance already exists, refusing to overwrite: {out_path}")
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": now.isoformat(),
        "config": config_to_dict(config),
        "inputs": _hash_inputs(config, manifest, data_dir=data_dir),
        "manifest": {
            "n_frames": int(len(manifest)),
            "n_subjects": int(manifest["subject"].nunique()),
            "n_sessions": int(manifest.groupby(["subject", "session_idx"]).ngroups),
        },
        "folds": _fold_manifest(folds),
        "git": git,
        "packages": _package_versions(),
        "device": config.run.device,
        "seed": config.run.seed,
        "seed_set": list(config.run.seed_set),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    if extra:
        payload["extra"] = extra

    # sort_keys so byte-identical inputs give byte-identical JSON and diffs are useful.
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_path
