"""Correct the quality-training metrics without rerunning a fitted model.

The first report unevenly weighted folds because deterministic winners emitted seed 1
once while stochastic winners emitted seeds 1--5.  Experiment A/C reporting replicates
deterministic predictions across the realized seed axis.  This one-time entry point
authenticates the completed run, archives only its invalid reports, and recomputes those
reports from the unchanged prediction CSV files.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.eval.quality_training_sensitivity import (  # noqa: E402
    PROTOCOLS,
    QualityTrainingError,
    assert_identical_test_keys,
    summarize_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (REPO_ROOT / "results" / "quality_training_sensitivity_10ghz").resolve()
ARCHIVE_ROOT = (
    REPO_ROOT
    / "archive"
    / "results"
    / "quality_training_sensitivity_seed_collapse_invalid_20260831"
).resolve()
TRAINING_COMMIT = "4aa814d1338ec25671d34bf0f1a73dac91762fe9"
ORIGINAL_PROVENANCE_SHA256 = "8d64d45a297d8523efa52c4bb7e496f1073f3dffa5df4b3bb7ed6bd47d94b306"
ORIGINAL_METRICS_SHA256 = {
    "loso/metrics.json": "d0fb08d4f7f0db079bf70e6026db35ddb3e4d11d3e94622ec7dd0ca47bde1852",
    "subject_overlap_session_cv/metrics.json": "44b12acf873697be118238e46aacae5aece4d4f2a0e01e22d0786608b1615a92",
}
PREDICTION_FIELDS = (
    "protocol", "task", "split_id", "arm", "treatment", "seed", "row_index",
    "subject", "session_idx", "y_true", "y_pred",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git_command(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def capture_clean_correction_commit() -> dict:
    """Capture Git before mutation, allowing only this run's untracked result files.

    A completed IBEX result root is normally untracked.  Any tracked edit or any other
    untracked path still fails closed, so the correction source itself must be committed.
    """
    try:
        commit = _git_command("rev-parse", "HEAD")
        branch = _git_command("rev-parse", "--abbrev-ref", "HEAD")
        status_lines = [
            line for line in _git_command("status", "--porcelain", "--untracked-files=all").splitlines()
            if line
        ]
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QualityTrainingError("could not capture correction Git state") from exc

    allowed_untracked = {
        f"?? {path.relative_to(REPO_ROOT).as_posix()}" for path in _expected_result_files()
    }
    unexpected = sorted(set(status_lines) - allowed_untracked)
    if unexpected:
        raise QualityTrainingError(
            "correction requires committed source and no unrelated worktree changes; "
            f"unexpected status entries: {unexpected}"
        )
    if any(line not in allowed_untracked for line in status_lines):
        raise QualityTrainingError("correction worktree state could not be authenticated")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise QualityTrainingError("correction requires a full clean Git commit")
    if commit == TRAINING_COMMIT:
        raise QualityTrainingError("correction source is not yet committed separately from training")
    return {"commit": commit, "branch": branch, "dirty_except_fixed_result_root": False}


def _expected_result_files() -> list[Path]:
    names = ("predictions.csv", "fit_audit.jsonl", "selections.json", "split_manifest.json", "metrics.json")
    files = [RESULT_ROOT / protocol / name for protocol in PROTOCOLS for name in names]
    return [*files, RESULT_ROOT / "provenance.json"]


def _verify_original_run() -> dict:
    expected_files = set(_expected_result_files())
    if not RESULT_ROOT.is_dir():
        raise QualityTrainingError(f"missing fixed result root: {RESULT_ROOT}")
    actual_files = {path.resolve() for path in RESULT_ROOT.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(str(path) for path in expected_files - actual_files)
        extra = sorted(str(path) for path in actual_files - expected_files)
        raise QualityTrainingError(f"result file inventory changed; missing={missing}, extra={extra}")

    provenance_path = RESULT_ROOT / "provenance.json"
    if _sha256_file(provenance_path) != ORIGINAL_PROVENANCE_SHA256:
        raise QualityTrainingError("original provenance hash does not match the completed IBEX run")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != "quality_training_sensitivity_provenance_v1":
        raise QualityTrainingError("original provenance schema changed")
    if (provenance.get("git") or {}).get("commit") != TRAINING_COMMIT:
        raise QualityTrainingError("original training commit is not the authenticated 4aa814d run")

    recorded_hashes = provenance.get("output_sha256")
    expected_relative = {
        path.relative_to(RESULT_ROOT).as_posix()
        for path in expected_files
        if path.name != "provenance.json"
    }
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != expected_relative:
        raise QualityTrainingError("original provenance output inventory changed")
    for relative_path, recorded_hash in recorded_hashes.items():
        if _sha256_file(RESULT_ROOT / relative_path) != recorded_hash:
            raise QualityTrainingError(f"original output hash mismatch: {relative_path}")
    for relative_path, expected_hash in ORIGINAL_METRICS_SHA256.items():
        if recorded_hashes.get(relative_path) != expected_hash:
            raise QualityTrainingError(f"invalid-report hash changed: {relative_path}")
    return provenance


def _load_valid_predictions(protocol: str) -> list[dict]:
    path = RESULT_ROOT / protocol / "predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_FIELDS:
            raise QualityTrainingError(f"{protocol}: prediction schema changed")
        rows = []
        try:
            for raw in reader:
                row = dict(raw)
                for name in ("seed", "row_index", "subject", "session_idx"):
                    row[name] = int(row[name])
                for name in ("y_true", "y_pred"):
                    row[name] = float(row[name])
                rows.append(row)
        except (TypeError, ValueError) as exc:
            raise QualityTrainingError(f"{protocol}: invalid prediction value") from exc
    if not rows or {row["protocol"] for row in rows} != {protocol}:
        raise QualityTrainingError(f"{protocol}: prediction rows have the wrong protocol")
    assert_identical_test_keys(rows)
    # The summary call performs the stricter seed-axis, key-order, truth, and finite checks.
    summarize_predictions(rows)
    return rows


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def correct_existing_reports() -> dict:
    """Archive the three invalid reports and publish corrected reporting-only artifacts."""
    correction_git = capture_clean_correction_commit()
    original_provenance = _verify_original_run()
    if ARCHIVE_ROOT.exists():
        raise QualityTrainingError(f"refusing to overwrite correction archive: {ARCHIVE_ROOT}")

    predictions_by_protocol = {
        protocol: _load_valid_predictions(protocol) for protocol in PROTOCOLS
    }
    metric_bytes = {
        f"{protocol}/metrics.json": _json_bytes(summarize_predictions(rows))
        for protocol, rows in predictions_by_protocol.items()
    }

    output_hashes = dict(original_provenance["output_sha256"])
    for relative_path, content in metric_bytes.items():
        output_hashes[relative_path] = _sha256_bytes(content)

    corrected_provenance = copy.deepcopy(original_provenance)
    corrected_provenance["schema_version"] = "quality_training_sensitivity_provenance_v2"
    corrected_provenance["original_training_git"] = copy.deepcopy(original_provenance["git"])
    corrected_provenance["output_sha256"] = output_hashes
    corrected_provenance["reporting_correction"] = {
        "schema_version": "quality_training_seed_collapse_correction_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": correction_git,
        "reason": (
            "The v1 reporter scored raw seed groups, overweighting stochastic folds. "
            "Metrics now replicate deterministic seed-1 folds across each group's realized "
            "seed axis, matching canonical Experiment A/C reporting."
        ),
        "models_or_features_rerun": False,
        "predictions_changed": False,
        "original_provenance_sha256": ORIGINAL_PROVENANCE_SHA256,
        "archived_invalid_reports": [
            "provenance.json", "loso/metrics.json",
            "subject_overlap_session_cv/metrics.json",
        ],
        "source_sha256": {
            "experiments/correct_quality_training_seed_collapse.py": _sha256_file(Path(__file__)),
            "src/dehyd/eval/quality_training_sensitivity.py": _sha256_file(
                REPO_ROOT / "src" / "dehyd" / "eval" / "quality_training_sensitivity.py"
            ),
        },
    }
    provenance_bytes = _json_bytes(corrected_provenance)

    staged_root = Path(tempfile.mkdtemp(prefix=".quality-seed-correction-", dir=RESULT_ROOT.parent))
    replacements = {
        RESULT_ROOT / relative_path: content for relative_path, content in metric_bytes.items()
    }
    replacements[RESULT_ROOT / "provenance.json"] = provenance_bytes
    archive_pairs = {
        current: ARCHIVE_ROOT / current.relative_to(RESULT_ROOT) for current in replacements
    }
    moved_to_archive: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        staged_paths = {}
        for current, content in replacements.items():
            staged = staged_root / current.relative_to(RESULT_ROOT)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(content)
            staged_paths[current] = staged

        for current, archived in archive_pairs.items():
            archived.parent.mkdir(parents=True, exist_ok=True)
            current.replace(archived)
            moved_to_archive.append((current, archived))
        for current, staged in staged_paths.items():
            staged.replace(current)
            installed.append(current)

        unchanged_paths = [
            relative for relative in output_hashes
            if relative not in metric_bytes
        ]
        for relative_path in unchanged_paths:
            if _sha256_file(RESULT_ROOT / relative_path) != output_hashes[relative_path]:
                raise QualityTrainingError(
                    f"unchanged artifact changed during correction: {relative_path}"
                )
        for relative_path, expected_hash in output_hashes.items():
            if _sha256_file(RESULT_ROOT / relative_path) != expected_hash:
                raise QualityTrainingError(f"corrected output hash mismatch: {relative_path}")
    except BaseException:
        for current in installed:
            current.unlink(missing_ok=True)
        for current, archived in reversed(moved_to_archive):
            if archived.exists():
                archived.replace(current)
        if ARCHIVE_ROOT.exists():
            shutil.rmtree(ARCHIVE_ROOT)
        raise
    finally:
        if staged_root.exists():
            shutil.rmtree(staged_root)

    return {
        "results_root": str(RESULT_ROOT),
        "archive_root": str(ARCHIVE_ROOT),
        "training_commit": TRAINING_COMMIT,
        "correction_commit": correction_git["commit"],
        "models_or_features_rerun": False,
    }


def main() -> int:
    print(json.dumps(correct_existing_reports(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
