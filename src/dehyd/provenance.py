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
)


class ProvenanceError(RuntimeError):
    """Raised when a run's provenance cannot be written safely."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info() -> dict:
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
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


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
