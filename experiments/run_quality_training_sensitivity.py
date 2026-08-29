"""Run the exploratory 10 GHz quality-aware training sensitivity.

Use ``--synthetic-smoke`` locally.  ``--full-cohort`` is the production CPU job for
IBEX; it writes LOSO and the optimistic subject-overlap diagnostic separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd import provenance  # noqa: E402
from dehyd.config import config_to_dict, load_config  # noqa: E402
from dehyd.eval import exp_a, exp_b  # noqa: E402
from dehyd.eval.quality_training_sensitivity import (  # noqa: E402
    ExactRowFeatureSource,
    PROTOCOLS,
    TASKS,
    QualityTrainingError,
    align_sessions,
    authenticate_reference,
    canonical_keys_from_reference,
    load_quality_margin,
    load_quality_training_config,
    publish_atomically,
    require_full_model_seeds,
    run_protocol,
    synthetic_mechanism_smoke,
    verify_loso_regression_baseline_replay,
    write_protocol_outputs,
)
from dehyd.features import store as store_mod  # noqa: E402
from dehyd.features.protocol_freeze import protocol_freeze_guard  # noqa: E402


def _require_clean_commit() -> dict:
    git = provenance._git_info()
    commit = str(git.get("commit", ""))
    if (
        git.get("dirty") is not False
        or len(commit) != 40
        or commit != commit.lower()
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise QualityTrainingError(
            "the production run requires a clean 40-character Git commit; commit the "
            "implementation and rebuild the 10 GHz store at that commit first"
        )
    return git


def _run_full(config_path: Path, base_config_overlays: tuple[str, ...] = ()) -> dict:
    git = _require_clean_commit()
    sensitivity = load_quality_training_config(config_path)
    base_config_paths = tuple(str(path) for path in sensitivity.base_configs) + base_config_overlays
    base = load_config(*base_config_paths)
    require_full_model_seeds(base.run.seed_set)
    protocol_freeze_guard(base)
    reference_band = authenticate_reference(sensitivity)
    canonical_keys = canonical_keys_from_reference(reference_band)
    sessions = align_sessions(exp_a.build_sessions(base, "10ghz"), reference_band)
    quality = load_quality_margin(sensitivity, canonical_keys)

    # A store built at an older commit is not silently blessed.  This usually means the
    # first IBEX step after merging this implementation is a commit-matched store rebuild.
    expected_fingerprints = exp_a.expected_fingerprints(base, "10ghz", sessions)
    store_mod.validate_store(
        "10ghz",
        base.paths.results_dir,
        expected_fingerprints,
        analysis_commit=git["commit"],
    )
    store_lineage = _store_lineage(base.paths.results_dir, sessions)
    source = ExactRowFeatureSource(
        sessions, base.paths.results_dir, base, quality["in_band_ratio_p10_margin"].to_numpy(float)
    )
    by_protocol = {protocol: ([], [], [], []) for protocol in PROTOCOLS}
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    replay_gate = None
    for protocol in PROTOCOLS:
        for task in TASKS:
            outputs = run_protocol(
                protocol=protocol,
                task=task,
                config=base,
                sessions=sessions,
                source=source,
                quality_margin=source.quality_margin,
                reference_band=reference_band,
                split_seed=sensitivity.split_seed,
                seeds=tuple(base.run.seed_set),
                n_workers=n_workers,
            )
            for target, rows in zip(by_protocol[protocol], outputs, strict=True):
                target.extend(rows)
            if protocol == "loso" and task == "regression":
                replay_gate = verify_loso_regression_baseline_replay(
                    by_protocol[protocol][0], by_protocol[protocol][1],
                    by_protocol[protocol][2], reference_band,
                )
    if replay_gate is None:
        raise QualityTrainingError("LOSO regression baseline replay gate did not run")

    config_paths = [config_path.resolve(), *(Path(path).resolve() for path in base_config_paths)]
    config_hashes = [
        {"path": str(path), "sha256": _file_hash(path)} for path in config_paths
    ]

    def write_complete_run(staging_root: Path) -> dict:
        artifact_paths = []
        for protocol in PROTOCOLS:
            written = write_protocol_outputs(staging_root, protocol, *by_protocol[protocol])
            artifact_paths.extend(written.values())
        output_hashes = {
            str(path.relative_to(staging_root)).replace("\\", "/"): _file_hash(path)
            for path in sorted(artifact_paths)
        }
        provenance_payload = _build_sensitivity_provenance(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            git=git,
            config_hashes=config_hashes,
            resolved_config_fingerprint=exp_b.config_fingerprint(base),
            resolved_config=config_to_dict(base),
            model_seed_set=tuple(base.run.seed_set),
            session_split_seed=sensitivity.split_seed,
            replay_gate=replay_gate,
            source_hashes={
                "reference_manifest": _file_hash(sensitivity.reference_manifest),
                "exp_a_sources": _file_hash(sensitivity.exp_a_sources),
                "quality_provenance": _file_hash(sensitivity.quality_provenance),
                "session_quality": _file_hash(sensitivity.session_quality),
            },
            store_lineage=store_lineage,
            packages=provenance._package_versions(),
            platform_info={
                "system": platform.system(), "release": platform.release(),
                "machine": platform.machine(), "cpu_model": provenance._cpu_model(),
            },
            output_hashes=output_hashes,
        )
        path = staging_root / "provenance.json"
        path.write_text(
            json.dumps(provenance_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {"provenance_name": path.name}

    publish_atomically(sensitivity.results_dir, write_complete_run)
    return {
        "results_root": str(sensitivity.results_dir),
        "provenance": str(sensitivity.results_dir / "provenance.json"),
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _store_lineage(store_dir: Path, sessions: list[dict]) -> dict:
    fingerprints = []
    for session in sessions:
        fingerprint = store_mod.read_fingerprint(
            store_dir, "10ghz", session["subject"], session["session_name"]
        )
        fingerprints.append({
            "subject": int(session["subject"]),
            "session_name": str(session["session_name"]),
            "git_commit": (fingerprint.get("git") or {}).get("commit"),
            "spec_hash": fingerprint.get("spec_hash"),
            "qc_config_hash": fingerprint.get("qc_config_hash"),
            "frame_ids_sha256": fingerprint.get("frame_ids_sha256"),
            "store_version": fingerprint.get("store_version"),
        })
    payload = json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "band_dir": str(store_mod.band_dir(store_dir, "10ghz").resolve()),
        "n_fingerprints": len(fingerprints),
        "fingerprints_sha256": hashlib.sha256(payload).hexdigest(),
        "build_commits": sorted({row["git_commit"] for row in fingerprints}),
        "spec_hashes": sorted({row["spec_hash"] for row in fingerprints}),
        "qc_config_hashes": sorted({row["qc_config_hash"] for row in fingerprints}),
        "store_versions": sorted({row["store_version"] for row in fingerprints}),
    }


def _build_sensitivity_provenance(
    *, timestamp_utc: str, git: dict, config_hashes: list[dict],
    resolved_config_fingerprint: str, resolved_config: dict,
    model_seed_set: tuple[int, ...], session_split_seed: int, replay_gate: dict,
    source_hashes: dict, store_lineage: dict, packages: dict, platform_info: dict,
    output_hashes: dict,
) -> dict:
    """Construct the complete JSON-serializable production provenance record."""
    payload = {
        "schema_version": "quality_training_sensitivity_provenance_v1",
        "timestamp_utc": timestamp_utc,
        "git": git,
        "configs": config_hashes,
        "resolved_config_fingerprint": resolved_config_fingerprint,
        "resolved_config": resolved_config,
        "seeds": {
            "model_seed_set": list(model_seed_set),
            "session_split_seed": session_split_seed,
            "inner_split_seed_rule": "session_split_seed + 1000 + zero_based_outer_fold",
        },
        "quality_signal": "radar-only in_band_ratio_p10_margin",
        "quality_threshold": 0.0,
        "negative_training_rows": [[4, 2], [8, 0], [8, 2], [12, 0], [16, 3]],
        "n_sessions": 73,
        "n_subjects": 16,
        "protocol_labels": {
            "loso": "primary new-subject generalization",
            "subject_overlap_session_cv": "optimistic post-hoc diagnostic; not generalization",
        },
        "baseline_replay_gate": replay_gate,
        "source_hashes": source_hashes,
        "store_lineage": store_lineage,
        "packages": packages,
        "platform": platform_info,
        "output_sha256": output_hashes,
    }
    # Fail here, still inside staging, if a future field introduces a non-serializable value.
    json.dumps(payload, sort_keys=True, allow_nan=False)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config", default="configs/quality_training_sensitivity_10ghz.yaml", metavar="PATH"
    )
    parser.add_argument(
        "--base-config", action="append", default=[], metavar="PATH",
        help="later paths-only overlay, e.g. configs/ibex_sosagojm.yaml",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-smoke", action="store_true")
    group.add_argument("--full-cohort", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic_smoke:
        print(json.dumps(synthetic_mechanism_smoke(), indent=2, sort_keys=True))
        return 0
    outputs = _run_full(Path(args.config), tuple(args.base_config))
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
