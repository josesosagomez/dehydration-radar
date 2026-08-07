"""The Experiment-A reference gate (MILESTONE_10_PLAN.md §1.3, §3, §4.2 step 1).

Milestone 10 rebuilds both feature stores. `store._check_match` compares the building
git commit against the analysis commit by strict equality, so the moment the stores move,
the *evidence* behind the M9 Exp-A results stops being recomputable: the stored session
vectors, the raw pre-log tensors the tuned-ε branch reconstructs from, and the fold-local
ε those reconstructions used all belong to a store that no longer exists.

This module freezes that evidence into one immutable JSON **before** the rebuild
(`snapshot`), and afterwards proves the rebuilt store still gives Exp A the same answers
(`compare`) before Experiment F is allowed to trust Exp A's per-fold feature selection.

What is recorded, per band (the plan's evidence classes):

  population   the canonical (subject, session_idx) spine, the QC-selected frame
               membership per session, and the Δm% targets;
  folds        the outer LOSO fold manifest, built by the SAME call Exp A makes;
  candidates   the Stage-1 enumeration, plus per fold the Stage-2 enumeration at that
               fold's selected feature key;
  tables       SHA-256 of the run's own selection/prediction/metrics artifacts;
  features     branch-aware, because the three log branches are not the same kind of
               evidence at all —
                 off/frozen  read a stored, data-independent session vector: hashing the
                             stored array IS the complete input evidence;
                 tuned       is reconstructed fold-locally from the stored RAW tensor with
                             a train-only ε, so its evidence is the raw/prelog/order inputs
                             PLUS the fold's training subjects, the ε those subjects
                             produce, and the hash of the matrix rebuilt from them.
  scores       the per-fold, per-seed held-out predictions and the subject MAE they give.

Nothing here re-implements Exp A. Folds come from `splits.py`, candidates from
`exp_a.stage1_candidates`/`stage2_candidates`, and every feature matrix comes from Exp A's
own `StoreBackedFeatures.data_for` — so a drift in any of those shows up as a changed hash
rather than being masked by a second, subtly different implementation living here.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..features import store as store_mod
from ..provenance import _git_info, fold_manifest, sha256_file
from . import exp_a
from .harness import Candidate
from .splits import DEFAULT_MIN_TRAIN_SUBJECTS, DEFAULT_N_INNER_MAX, nested_loso_splits

MANIFEST_SCHEMA_VERSION = "reference_exp_a_manifest_v1"
SOURCES_SCHEMA_VERSION = "exp_a_sources_v1"

BANDS = ("10ghz", "77ghz")

# A snapshot is only usable as the milestone's reference if it was taken from a store whose
# build commit equals the commit of the Exp-A run it is evidence for. A snapshot taken from
# any other store is still written — it is a useful mechanism check — but it is stamped
# DEGRADED and `compare` refuses it, so a convenience run can never silently become the
# scientific gate.
GRADE_AUTHORITATIVE = "authoritative"
GRADE_DEGRADED_STORE_COMMIT = "degraded_store_commit_divergence"

# The evidence classes `compare` checks one by one. Every one of them must be `match`
# before Exp F may consume the final Exp-A selection table.
EVIDENCE_CLASSES = (
    "population",
    "folds",
    "stage1_candidates",
    "selected_feature_keys",
    "stage2_candidates",
    "feature_inputs",
    "tuned_epsilon",
    "feature_matrices",
    "selection_table",
    "predictions",
    "scores",
)


class ReferenceGateError(RuntimeError):
    """A missing/ambiguous reference artifact, or a refused comparison."""


# ------------------------------------------------------------------ canonical hashing


def _canonical(obj):
    """Plain JSON-able data with a single spelling per value.

    Tuples become lists and numpy scalars become Python scalars, so a hash cannot change
    just because a value arrived as `np.int64(3)` on one run and `3` on the next.
    """
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def json_sha256(obj) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def array_sha256(arr) -> str:
    """Hash an array's dtype, shape and exact bytes.

    `tobytes()` rather than a value comparison, for the reason `scripts/compare_stores.py`
    documents: the question is "did this bit pattern change?", so NaN must equal NaN and
    +0.0 must not equal −0.0.
    """
    return arrays_sha256([("", arr)])


def arrays_sha256(labelled_arrays) -> str:
    """One hash over an ordered sequence of (label, array) pairs."""
    digest = hashlib.sha256()
    for label, arr in labelled_arrays:
        a = np.ascontiguousarray(arr)
        digest.update(label.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(a.dtype.str.encode("utf-8"))
        digest.update(repr(a.shape).encode("utf-8"))
        digest.update(a.tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------- reading a run directory


@dataclass(frozen=True)
class SelectedFold:
    """One row of an Exp-A `selection_table_{band}.csv`."""

    test_subject: int
    feature_key: tuple
    family: str
    params: dict


def _require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise ReferenceGateError(f"missing {what}: {path}")
    return path


def run_artifacts(run_dir, band) -> dict:
    """The three Exp-A full-run artifacts this gate reads, by name."""
    run_dir = Path(run_dir)
    return {
        "provenance": run_dir / "provenance.json",
        "metrics": run_dir / f"metrics_exp_a_{band}.json",
        "predictions": run_dir / f"predictions_{band}.csv",
        "selection_table": run_dir / f"selection_table_{band}.csv",
    }


def read_run_provenance(run_dir, band) -> dict:
    """The run's provenance, checked to actually be a full-cohort Exp A run for `band`."""
    path = _require_file(Path(run_dir) / "provenance.json", "provenance.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    extra = payload.get("extra") or {}
    if extra.get("stage") != "exp-a-full":
        raise ReferenceGateError(
            f"{run_dir} is not a full-cohort Exp A run: extra.stage is {extra.get('stage')!r}, "
            "expected 'exp-a-full' — a smoke or another experiment cannot be a reference"
        )
    if extra.get("band") != band:
        raise ReferenceGateError(
            f"{run_dir} is a {extra.get('band')!r} run but was supplied as the {band!r} reference"
        )
    commit = (payload.get("git") or {}).get("commit")
    if not commit:
        raise ReferenceGateError(f"{run_dir}/provenance.json records no git commit")
    return payload


def read_selection_table(run_dir, band) -> list[SelectedFold]:
    """Parse `selection_table_{band}.csv` (written with `repr()` of a tuple and a dict)."""
    path = _require_file(Path(run_dir) / f"selection_table_{band}.csv", f"{band} selection table")
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                SelectedFold(
                    test_subject=int(row["test_subject"]),
                    feature_key=tuple(ast.literal_eval(row["feature_key"])),
                    family=row["family"],
                    params=dict(ast.literal_eval(row["params"])),
                )
            )
    if not rows:
        raise ReferenceGateError(f"{path} has no rows")
    subjects = [r.test_subject for r in rows]
    if len(set(subjects)) != len(subjects):
        raise ReferenceGateError(f"{path} has duplicate test_subject rows")
    return sorted(rows, key=lambda r: r.test_subject)


def read_predictions(run_dir, band) -> dict:
    """`predictions_{band}.csv` grouped as {subject: {seed: [(y_true, y_pred), ...]}}.

    File order is preserved inside each (subject, seed) group — it is the canonical session
    order of that fold, and the gate compares it as an ordered sequence.
    """
    path = _require_file(Path(run_dir) / f"predictions_{band}.csv", f"{band} predictions")
    by_subject: dict = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            subject, seed = int(row["subject"]), int(row["seed"])
            by_subject.setdefault(subject, {}).setdefault(seed, []).append(
                (float(row["y_true"]), float(row["y_pred"]))
            )
    if not by_subject:
        raise ReferenceGateError(f"{path} has no rows")
    return by_subject


# ------------------------------------------------------------------- evidence builders


def population_evidence(sessions) -> dict:
    """The canonical session spine, the QC-selected frame membership, and the targets."""
    rows = []
    for s in sessions:
        rows.append(
            {
                "subject": int(s["subject"]),
                "session_idx": int(s["session_idx"]),
                "session_name": s["session_name"],
                "n_frames": int(len(s["frame_ids"])),
                "frame_ids_sha256": store_mod.frame_ids_sha256(s["frame_ids"]),
                "delta_m_pct": float(s["delta_m_pct"]),
            }
        )
    subjects = sorted({r["subject"] for r in rows})
    return {
        "n_subjects": len(subjects),
        "n_sessions": len(rows),
        "subjects": subjects,
        "sessions": rows,
        "session_keys_sha256": json_sha256(
            [[r["subject"], r["session_idx"], r["session_name"]] for r in rows]
        ),
        "frame_population_sha256": json_sha256(
            [[r["subject"], r["session_idx"], r["n_frames"], r["frame_ids_sha256"]] for r in rows]
        ),
        "targets_sha256": json_sha256(
            [[r["subject"], r["session_idx"], r["delta_m_pct"]] for r in rows]
        ),
    }


def fold_evidence(subjects) -> tuple[dict, list]:
    """The outer folds, built exactly the way `exp_a.run_exp_a` builds them.

    Exp A calls `nested_loso_splits(subjects)` with no split kwargs, so it takes the module
    defaults rather than `config.split`. The gate must reproduce the folds Exp A actually
    used, so it calls it the same way and records which constants that resolved to.
    """
    folds = nested_loso_splits(subjects)
    manifest = fold_manifest(folds)
    evidence = {
        "n_folds": len(folds),
        "n_selectable": sum(1 for f in folds if f.selectable),
        "constructed_with": {
            "n_inner_max": DEFAULT_N_INNER_MAX,
            "min_train_subjects": DEFAULT_MIN_TRAIN_SUBJECTS,
            "source": "eval.splits.nested_loso_splits (module defaults, as exp_a.run_exp_a calls it)",
        },
        "manifest": manifest,
        "sha256": json_sha256(manifest),
    }
    return evidence, folds


def _candidate_record(candidate: Candidate) -> list:
    return [
        candidate.candidate_id,
        candidate.family,
        _canonical(candidate.model_params),
        _canonical(candidate.feature_key),
        _canonical(candidate.active),
    ]


def anchor_alpha(config, band) -> float:
    space = config.search_10ghz if band == "10ghz" else config.search_77ghz
    return float(space.stage1_anchor_ridge_alpha)


def winner_active(config, band, feature_key) -> dict:
    """The protocol `active` record for a feature key, from Exp A's own builders.

    Reuses the private `_active_10`/`_active_77` deliberately: the band key sets are what
    `require_complete_active` enforces, and a second copy of them here is exactly the kind
    of near-duplicate that drifts.
    """
    if band == "10ghz":
        gate_index, reduction, channel, tiling_index, branch = feature_key
        gate = config.search_10ghz.range_gate_m[gate_index]
        return dict(
            exp_a._active_10(gate_index, reduction, channel, tiling_index, branch, "ridge", gate)
        )
    tiling_index, branch = feature_key
    return dict(exp_a._active_77(tiling_index, branch, "ridge", config))


def stage1_evidence(config, band) -> dict:
    candidates = exp_a.stage1_candidates(config, band, anchor_alpha(config, band))
    records = [_candidate_record(c) for c in candidates]
    return {
        "n_candidates": len(candidates),
        "anchor_ridge_alpha": anchor_alpha(config, band),
        "candidate_ids": [c.candidate_id for c in candidates],
        "sha256": json_sha256(records),
    }


def stage2_evidence(config, band, feature_key) -> dict:
    candidates = exp_a.stage2_candidates(
        config, band, feature_key, winner_active(config, band, feature_key)
    )
    records = [_candidate_record(c) for c in candidates]
    return {
        "n_candidates": len(candidates),
        "candidate_ids": [c.candidate_id for c in candidates],
        "sha256": json_sha256(records),
    }


def _feature_store_keys(band, feature_key) -> dict:
    """The store array names one feature key consumes, by role."""
    branch = feature_key[-1]
    if band == "10ghz":
        gate_index, reduction, channel, tiling_index, _ = feature_key
        order = store_mod.order_key(tiling_index)
        if branch in ("off", "frozen"):
            return {"vector": store_mod.vec_key(gate_index, reduction, channel, tiling_index, branch)}
        return {
            "raw": store_mod.raw_key(gate_index, reduction, channel, tiling_index),
            "prelog": store_mod.prelog_key(gate_index, reduction, channel, tiling_index),
            "order": order,
        }
    tiling_index, _ = feature_key
    if branch in ("off", "frozen"):
        return {"vector": store_mod.vec77_key(tiling_index, branch)}
    return {
        "raw": store_mod.raw77_key(tiling_index),
        "prelog": store_mod.prelog77_key(tiling_index),
        "order": store_mod.order_key(tiling_index),
    }


def feature_input_evidence(band, sessions, store_dir, feature_key) -> dict:
    """Branch-aware hashes of the STORED arrays one feature key reads.

    For `off`/`frozen` that is the stored session vector — the complete input, since those
    branches are data-independent. For `tuned` it is the raw pre-log tensor, the pre-log
    scales, and the kymatio path order: the three inputs the fold-local reconstruction
    consumes. Both a per-session hash and one combined hash are kept, so a later mismatch
    names the session it happened in instead of only failing.
    """
    keys = _feature_store_keys(band, feature_key)
    per_session, combined = [], {role: [] for role in keys}
    for s in sessions:
        store = store_mod.read_session_store(band, s["subject"], s["session_name"], store_dir)
        try:
            row = {"subject": int(s["subject"]), "session_name": s["session_name"]}
            for role, key in keys.items():
                if key not in store:
                    raise ReferenceGateError(
                        f"{band} store for subject {s['subject']} / {s['session_name']} has no "
                        f"array {key!r} (needed for feature key {feature_key!r})"
                    )
                arr = store[key]
                row[f"{role}_sha256"] = array_sha256(arr)
                row[f"{role}_shape"] = list(arr.shape)
                combined[role].append((f"{s['subject']}/{s['session_name']}", arr))
            per_session.append(row)
        finally:
            store.close()

    evidence = {
        "branch": feature_key[-1],
        "store_keys": keys,
        "n_sessions": len(per_session),
        "per_session": per_session,
    }
    for role, arrays in combined.items():
        evidence[f"{role}_sha256"] = arrays_sha256(arrays)
    return evidence


def fold_feature_evidence(provider, config, band, feature_key, train_subjects) -> dict:
    """The fold-local feature matrix Exp A would fit on, via Exp A's own provider.

    For `tuned` this is the whole point of the gate: ε is a fitted quantity (a train-only
    median of stored pre-log scales), so the reconstructed matrix depends on the fold, and
    both the ε and the matrix it produced are recorded.

    The matrix itself is far too large to store, so it is kept as a hash plus four order
    statistics. The hash is what the comparison tests; the statistics exist so a mismatch
    can be *quantified* by whoever has to decide whether the milestone stops, instead of
    only being reported as "different".
    """
    candidate = Candidate(
        candidate_id="reference_gate",
        family="ridge",
        model_params=(("alpha", anchor_alpha(config, band)),),
        feature_key=feature_key,
        active=tuple(winner_active(config, band, feature_key).items()),
    )
    bundle = provider.data_for(candidate, train_subjects)
    x = np.asarray(bundle.X, dtype=float)
    evidence = {
        "feature_matrix_sha256": array_sha256(bundle.X),
        "feature_matrix_shape": list(bundle.X.shape),
        "feature_matrix_dtype": str(bundle.X.dtype),
        "feature_matrix_summary": {
            "sum": float(np.sum(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean_abs": float(np.mean(np.abs(x))),
        },
    }
    extra = dict(bundle.extra_fits)
    if "tuned_epsilon" in extra:
        eps = np.asarray(extra["tuned_epsilon"]["epsilon"], dtype=float)
        evidence["tuned_epsilon"] = {"order_1": float(eps[0]), "order_2": float(eps[1])}
    return evidence


def prediction_evidence(by_subject, test_subject) -> dict:
    """The held-out predictions and the subject MAE they give, per seed.

    Kept as values, not only as a hash: a mismatch should be readable by a human deciding
    whether the milestone stops, and 5 seeds × a handful of sessions is small.
    """
    if test_subject not in by_subject:
        raise ReferenceGateError(f"predictions table has no rows for test subject {test_subject}")
    per_seed = by_subject[test_subject]
    seeds = sorted(per_seed)
    y_true_ref = [yt for yt, _ in per_seed[seeds[0]]]
    mae_by_seed, y_pred_by_seed = {}, {}
    for seed in seeds:
        rows = per_seed[seed]
        y_true = [yt for yt, _ in rows]
        if y_true != y_true_ref:
            raise ReferenceGateError(
                f"subject {test_subject}: seed {seed} carries different y_true values than seed "
                f"{seeds[0]} — the prediction table is not seed-aligned"
            )
        y_pred = [yp for _, yp in rows]
        y_pred_by_seed[str(seed)] = y_pred
        mae_by_seed[str(seed)] = float(
            np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        )
    return {
        "n_sessions": len(y_true_ref),
        "seeds": seeds,
        "y_true": y_true_ref,
        "y_pred_by_seed": y_pred_by_seed,
        "subject_mae_by_seed": mae_by_seed,
        "subject_mae_seed_mean": float(np.mean(list(mae_by_seed.values()))),
        "sha256": json_sha256([y_true_ref, y_pred_by_seed]),
    }


# ----------------------------------------------------------------------- store evidence


def superseded_run_evidence(run_dir, band, reference_evidence: dict) -> dict:
    """Artifact-only evidence for an Exp-A run whose backing store no longer exists.

    Milestone 10 named the `*_f0a46aa6` Exp-A runs as its reference. By the time
    implementation started, the IBEX stores had already been rebuilt at `3f465ab` (the
    commit move HISTORY 2026-08-04 records), so those runs' *feature* evidence — stored
    vectors, raw tensors, the fold-local ε — is not recomputable on any machine. Their
    **artifacts** still exist, and that is enough to answer the only question the
    substitution raises: does moving the reference to the `3f465abc` pair change any of
    Exp A's answers?

    So this records the superseded run's tables and compares them to the new reference by
    the O-M9-5 criterion — selection-table byte-identity, then bounded |Δy_pred|. It never
    contributes to `reference_grade`: it is provenance for why the reference moved, not a
    second gate. A `differs` status is not fatal here, but it *is* a finding, because it
    would mean the two runs selected different features for at least one fold.
    """
    run_dir = Path(run_dir).resolve()
    provenance = read_run_provenance(run_dir, band)
    selected = read_selection_table(run_dir, band)
    predictions = read_predictions(run_dir, band)

    artifacts = {
        name: {"path": str(path), "sha256": sha256_file(_require_file(path, name))}
        for name, path in run_artifacts(run_dir, band).items()
    }
    folds = [
        {
            "test_subject": fold.test_subject,
            "feature_key_repr": repr(fold.feature_key),
            "predictions": prediction_evidence(predictions, fold.test_subject),
        }
        for fold in selected
    ]

    tolerance = pred_tolerance()
    max_delta, fault = _max_pred_delta({"selected_folds": folds}, reference_evidence)
    selection_identical = (
        artifacts["selection_table"]["sha256"] == reference_evidence["selection_table_sha256"]
    )
    keys_identical = (
        {f["test_subject"]: f["feature_key_repr"] for f in folds}
        == _fold_map(reference_evidence, "feature_key_repr")
    )
    equivalent = selection_identical and keys_identical and fault is None and max_delta <= tolerance

    return {
        "path": str(run_dir),
        "name": run_dir.name,
        "commit": provenance["git"]["commit"],
        "store_recomputable": False,
        "artifacts": artifacts,
        "selected_feature_keys": {f["test_subject"]: f["feature_key_repr"] for f in folds},
        "vs_reference": {
            "criterion": f"selection-table byte-identity AND max|dy_pred| <= {tolerance:.1e} (O-M9-5)",
            "selection_table_byte_identical": selection_identical,
            "selected_feature_keys_identical": keys_identical,
            "max_abs_pred_delta": None if fault else max_delta,
            "fault": fault,
            "status": "equivalent" if equivalent else "differs",
        },
    }


def store_evidence(config, band, sessions, store_dir, *, hash_npz=True) -> dict:
    """Paths, sidecar fingerprints and file hashes of the store backing this band."""
    band_directory = store_mod.band_dir(store_dir, band)
    sidecars, commits = [], set()
    for s in sessions:
        stem = store_mod.session_stem(s["subject"], s["session_name"])
        sidecar_path = _require_file(
            band_directory / f"{stem}.fingerprint.json", f"{band} fingerprint for {stem}"
        )
        npz_path = _require_file(band_directory / f"{stem}.npz", f"{band} store file for {stem}")
        fingerprint = json.loads(sidecar_path.read_text(encoding="utf-8"))
        commits.add((fingerprint.get("git") or {}).get("commit"))
        sidecars.append(
            {
                "stem": stem,
                "subject": int(s["subject"]),
                "session_name": s["session_name"],
                "fingerprint": fingerprint,
                "fingerprint_sha256": sha256_file(sidecar_path),
                "npz_bytes": npz_path.stat().st_size,
                "npz_sha256": sha256_file(npz_path) if hash_npz else None,
            }
        )
    return {
        "band_dir": str(band_directory),
        "n_sessions": len(sidecars),
        "build_commits": sorted(c for c in commits if c),
        "spec_hash": store_mod.spec_hash(config, band),
        "qc_config_hash": store_mod.qc_config_hash(config, band),
        "store_version": store_mod.STORE_VERSION,
        "npz_hashed": bool(hash_npz),
        "sessions": sidecars,
        "sha256": json_sha256([[s["stem"], s["fingerprint"]] for s in sidecars]),
    }


# ------------------------------------------------------------------------- band evidence


def build_band_evidence(config, band, run_dir, *, hash_npz=True,
                        allow_store_commit_divergence=False, superseded_run_dirs=()) -> dict:
    """Every evidence class for one band, from one Exp-A full-cohort run directory.

    The store is validated fail-closed against the RUN's commit (not the current HEAD):
    the question this gate answers is "is this store the one that backed this run?", and
    tying it to whatever is checked out would make the answer depend on the working tree.
    """
    run_dir = Path(run_dir).resolve()
    provenance = read_run_provenance(run_dir, band)
    run_commit = provenance["git"]["commit"]

    sessions = exp_a.build_sessions(config, band)
    store_dir = config.paths.results_dir
    store = store_evidence(config, band, sessions, store_dir, hash_npz=hash_npz)

    divergent = [c for c in store["build_commits"] if c != run_commit]
    grade = GRADE_AUTHORITATIVE
    if divergent:
        if not allow_store_commit_divergence:
            raise ReferenceGateError(
                f"{band}: the store under {store['band_dir']} was built at "
                f"{store['build_commits']} but the reference run {run_dir.name} was executed at "
                f"{run_commit} — this store did not back that run, so evidence taken from it "
                "would not be evidence about the reference (store._check_match enforces the same "
                "equality). Point --store-dir at the store that produced the run, or pass "
                "--allow-store-commit-divergence to write a DEGRADED, non-authoritative snapshot."
            )
        grade = GRADE_DEGRADED_STORE_COMMIT
    else:
        # Only meaningful when the commits agree; with them agreeing this is the same
        # fail-closed check every Exp A run performs before fitting anything.
        store_mod.validate_store(
            band, store_dir, exp_a.expected_fingerprints(config, band, sessions),
            analysis_commit=run_commit,
        )

    population = population_evidence(sessions)
    folds_evidence, folds = fold_evidence(population["subjects"])
    selected = read_selection_table(run_dir, band)
    predictions = read_predictions(run_dir, band)

    selectable = {f.test_subject: f for f in folds if f.selectable}
    missing = sorted(set(selectable) - {s.test_subject for s in selected})
    unexpected = sorted({s.test_subject for s in selected} - set(selectable))
    if missing or unexpected:
        raise ReferenceGateError(
            f"{band}: the selection table does not cover the selectable outer folds "
            f"(missing={missing}, unexpected={unexpected})"
        )

    provider = exp_a.StoreBackedFeatures(band, sessions, store_dir, config)

    # Feature-key-major so the provider's single-entry raw cache stays warm: the tuned
    # reconstruction re-reads every session's raw tensor, and re-reading it once per fold
    # instead of once per feature key would dominate the runtime.
    by_key: dict = {}
    for fold in selected:
        by_key.setdefault(fold.feature_key, []).append(fold)

    feature_inputs, fold_rows = {}, []
    for feature_key in sorted(by_key, key=repr):
        feature_inputs[repr(feature_key)] = feature_input_evidence(
            band, sessions, store_dir, feature_key
        )
        for fold in by_key[feature_key]:
            train_subjects = selectable[fold.test_subject].train_subjects
            row = {
                "test_subject": fold.test_subject,
                "feature_key": _canonical(fold.feature_key),
                "feature_key_repr": repr(fold.feature_key),
                "branch": fold.feature_key[-1],
                "family": fold.family,
                "params": _canonical(fold.params),
                "train_subjects": sorted(train_subjects),
                "train_subjects_sha256": json_sha256(sorted(train_subjects)),
                "stage2_candidates": stage2_evidence(config, band, fold.feature_key),
                "predictions": prediction_evidence(predictions, fold.test_subject),
            }
            row.update(
                fold_feature_evidence(provider, config, band, fold.feature_key, train_subjects)
            )
            fold_rows.append(row)
    fold_rows.sort(key=lambda r: r["test_subject"])

    artifacts = {
        name: {"path": str(path), "sha256": sha256_file(_require_file(path, name))}
        for name, path in run_artifacts(run_dir, band).items()
    }

    evidence = {
        "band": band,
        "reference_grade": grade,
        "run": {
            "path": str(run_dir),
            "name": run_dir.name,
            "commit": run_commit,
            "branch": (provenance.get("git") or {}).get("branch"),
            "timestamp_utc": provenance.get("timestamp_utc"),
            "seed": provenance.get("seed"),
            "seed_set": provenance.get("seed_set"),
            "artifacts": artifacts,
        },
        "config_sha256": json_sha256(_config_payload(config)),
        "store": store,
        "population": population,
        "folds": folds_evidence,
        "stage1_candidates": stage1_evidence(config, band),
        "feature_inputs": feature_inputs,
        "selected_folds": fold_rows,
        "selection_table_sha256": artifacts["selection_table"]["sha256"],
        "predictions_sha256": artifacts["predictions"]["sha256"],
    }
    evidence["superseded_runs"] = [
        superseded_run_evidence(path, band, evidence) for path in superseded_run_dirs
    ]
    return evidence


def _config_payload(config) -> dict:
    """The resolved config with `paths` removed.

    Paths are machine-specific — the same protocol run on IBEX and locally must hash the
    same — and the store/run identity is already pinned by explicit path fields elsewhere
    in the manifest.
    """
    from ..config import config_to_dict

    payload = config_to_dict(config)
    payload.pop("paths", None)
    return payload


# ------------------------------------------------------------------------------ snapshot


def snapshot(configs, run_dirs, *, hash_npz=True, allow_store_commit_divergence=False,
             superseded_run_dirs=None) -> dict:
    """The immutable pre-rebuild reference manifest for both bands.

    `configs` / `run_dirs` are {band: value}. A band may be omitted, which is recorded as
    such — a partial snapshot is honest and still useful, and `compare` refuses to approve
    a band the snapshot never covered.

    `superseded_run_dirs` is {band: [run_dir, ...]}: earlier Exp-A runs whose stores are
    already gone, recorded artifact-only alongside the live reference (see
    `superseded_run_evidence`).
    """
    superseded_run_dirs = superseded_run_dirs or {}
    bands = {}
    for band in BANDS:
        if band not in run_dirs:
            continue
        bands[band] = build_band_evidence(
            configs[band], band, run_dirs[band],
            hash_npz=hash_npz, allow_store_commit_divergence=allow_store_commit_divergence,
            superseded_run_dirs=superseded_run_dirs.get(band, ()),
        )

    if not bands:
        raise ReferenceGateError("no reference run directory was supplied for any band")

    grade = (
        GRADE_AUTHORITATIVE
        if all(b["reference_grade"] == GRADE_AUTHORITATIVE for b in bands.values())
        else GRADE_DEGRADED_STORE_COMMIT
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_grade": grade,
        "bands_covered": sorted(bands),
        "tool_git": _git_info(),
        "bands": bands,
    }


# ------------------------------------------------------------------------------- compare


def _fold_map(band_evidence, field) -> dict:
    return {
        row["test_subject"]: row[field]
        for row in band_evidence["selected_folds"]
        if field in row
    }


def _hashes_only(feature_inputs) -> dict:
    return {
        key: {role: value for role, value in evidence.items() if role.endswith("_sha256")}
        for key, evidence in feature_inputs.items()
    }


def pred_tolerance() -> float:
    """The O-M9-5 prediction tolerance, read from its one definition in `exp_d.py`.

    Imported lazily because `exp_d` imports torch at module scope, and this gate is a
    torch-free CPU tool that runs before any model is fitted. Re-declaring the number here
    instead would give the project two tolerances that could drift apart, which is exactly
    what the single-definition rule exists to prevent.
    """
    from .exp_d import O_M9_5_PRED_TOLERANCE

    return O_M9_5_PRED_TOLERANCE


def _max_pred_delta(reference: dict, final: dict) -> tuple[float, str | None]:
    """Largest |Δy_pred| across every fold/seed/session, or a fault describing why not.

    Alignment is strict and `y_true` must match exactly, for the same reason
    `exp_d._max_abs_pred_delta` requires it: the tolerance exists for float noise in a
    fitted model's output and nothing else. Differing ground truth, seed labels or session
    counts mean different data or different splits — a protocol fault, not noise.
    """
    reference_preds = _fold_map(reference, "predictions")
    final_preds = _fold_map(final, "predictions")
    if sorted(reference_preds) != sorted(final_preds):
        return float("nan"), "the two runs cover different test subjects"

    worst = 0.0
    for subject in sorted(reference_preds):
        a, b = reference_preds[subject], final_preds[subject]
        if a["y_true"] != b["y_true"]:
            return float("nan"), f"subject {subject}: y_true differs (data or split fault)"
        if a["seeds"] != b["seeds"]:
            return float("nan"), f"subject {subject}: seed labels differ ({a['seeds']} vs {b['seeds']})"
        for seed in a["y_pred_by_seed"]:
            delta = np.abs(
                np.array(a["y_pred_by_seed"][seed]) - np.array(b["y_pred_by_seed"][seed])
            )
            worst = max(worst, float(delta.max()) if delta.size else 0.0)
    return worst, None


def _class_comparisons(reference: dict, final: dict) -> list[dict]:
    """One status row per evidence class, comparing a snapshot band to a fresh one.

    The criterion is the two-part shape O-M9-5 already established for this project, for
    the reason recorded there: every *discrete* outcome is compared with no tolerance at
    all, and only the fitted models' float output is allowed last-ulp noise, bounded by the
    same `O_M9_5_PRED_TOLERANCE` constant rather than a second one invented here.

    So: the population, the folds, both candidate enumerations, which feature key each fold
    selected, the stored store arrays those keys read, the fold-local ε, the reconstructed
    feature matrices and the selection table are all exact. Only `predictions` (and the
    subject MAE derived from it) carry a tolerance — and that tolerance can never rescue a
    genuine drift, because a genuine drift changes a model selection, which lands in the
    tolerance-free `selection_table` class first.
    """
    tolerance = pred_tolerance()
    population = all(
        reference["population"][field] == final["population"][field]
        for field in ("session_keys_sha256", "frame_population_sha256", "targets_sha256")
    )
    max_delta, fault = _max_pred_delta(reference, final)
    predictions_ok = fault is None and max_delta <= tolerance

    exact = [
        ("population", population, None),
        ("folds", reference["folds"]["sha256"] == final["folds"]["sha256"], None),
        ("stage1_candidates",
         reference["stage1_candidates"]["sha256"] == final["stage1_candidates"]["sha256"], None),
        ("selected_feature_keys",
         _fold_map(reference, "feature_key_repr") == _fold_map(final, "feature_key_repr"), None),
        ("stage2_candidates",
         {k: v["sha256"] for k, v in _fold_map(reference, "stage2_candidates").items()}
         == {k: v["sha256"] for k, v in _fold_map(final, "stage2_candidates").items()}, None),
        ("feature_inputs",
         _hashes_only(reference["feature_inputs"]) == _hashes_only(final["feature_inputs"]), None),
        ("tuned_epsilon",
         _fold_map(reference, "tuned_epsilon") == _fold_map(final, "tuned_epsilon"), None),
        ("feature_matrices",
         _fold_map(reference, "feature_matrix_sha256") == _fold_map(final, "feature_matrix_sha256"),
         None),
        ("selection_table",
         reference["selection_table_sha256"] == final["selection_table_sha256"], None),
    ]

    rows = [
        {"evidence_class": name, "criterion": "exact", "status": "match" if ok else "mismatch",
         "detail": detail}
        for name, ok, detail in exact
    ]
    rows.append({
        "evidence_class": "predictions",
        "criterion": f"selection-table byte-identity (above) AND max|dy_pred| <= {tolerance:.1e} (O-M9-5)",
        "status": "match" if predictions_ok else "mismatch",
        "detail": fault if fault else f"max|dy_pred| = {max_delta:.3e}",
        "max_abs_pred_delta": None if fault else max_delta,
        "byte_identical": reference["predictions_sha256"] == final["predictions_sha256"],
    })
    reference_mae = {k: v["subject_mae_seed_mean"] for k, v in _fold_map(reference, "predictions").items()}
    final_mae = {k: v["subject_mae_seed_mean"] for k, v in _fold_map(final, "predictions").items()}
    worst_mae = (
        max((abs(reference_mae[s] - final_mae[s]) for s in reference_mae if s in final_mae),
            default=float("nan"))
        if sorted(reference_mae) == sorted(final_mae) else float("nan")
    )
    rows.append({
        "evidence_class": "scores",
        "criterion": f"max|d subject MAE| <= {tolerance:.1e}",
        "status": "match" if worst_mae <= tolerance else "mismatch",
        "detail": f"max|d subject MAE| = {worst_mae:.3e}",
        "max_abs_subject_mae_delta": None if np.isnan(worst_mae) else float(worst_mae),
    })
    assert {row["evidence_class"] for row in rows} == set(EVIDENCE_CLASSES)
    return rows


def compare(manifest: dict, configs, final_run_dirs, *, hash_npz=True) -> dict:
    """Prove the rebuilt stores still give Exp A the same answers, and approve the sources.

    Reads only the immutable snapshot plus explicit final-run directories — never a glob,
    never a "latest" directory. Every evidence class must match; anything else leaves the
    band `not_approved`, which `run_confound.py` refuses.
    """
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ReferenceGateError(
            f"reference manifest schema is {manifest.get('schema_version')!r}, "
            f"expected {MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.get("reference_grade") != GRADE_AUTHORITATIVE:
        raise ReferenceGateError(
            f"reference manifest is {manifest.get('reference_grade')!r}, not "
            f"{GRADE_AUTHORITATIVE!r} — a degraded snapshot is a mechanism check, never the "
            "scientific gate. Re-take the snapshot from the store that backed the reference runs."
        )

    bands = {}
    for band in sorted(final_run_dirs):
        if band not in manifest["bands"]:
            raise ReferenceGateError(
                f"the reference manifest does not cover band {band!r} "
                f"(covered: {manifest['bands_covered']}) — it cannot approve a band it never saw"
            )
        final = build_band_evidence(configs[band], band, final_run_dirs[band], hash_npz=hash_npz)
        comparisons = _class_comparisons(manifest["bands"][band], final)
        mismatched = [c["evidence_class"] for c in comparisons if c["status"] != "match"]
        bands[band] = {
            "band": band,
            "status": "approved" if not mismatched else "not_approved",
            "mismatched_evidence_classes": mismatched,
            "comparisons": comparisons,
            "reference_run": {
                "path": manifest["bands"][band]["run"]["path"],
                "commit": manifest["bands"][band]["run"]["commit"],
                "artifacts": manifest["bands"][band]["run"]["artifacts"],
            },
            "final_run": {
                "path": final["run"]["path"],
                "commit": final["run"]["commit"],
                "artifacts": final["run"]["artifacts"],
            },
            "final_store": {
                "band_dir": final["store"]["band_dir"],
                "build_commits": final["store"]["build_commits"],
                "spec_hash": final["store"]["spec_hash"],
                "qc_config_hash": final["store"]["qc_config_hash"],
                "sha256": final["store"]["sha256"],
            },
            "final_config_sha256": final["config_sha256"],
            "reference_config_sha256": manifest["bands"][band]["config_sha256"],
            "final_selection_table": final["run"]["artifacts"]["selection_table"],
        }

    return {
        "schema_version": SOURCES_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_manifest_grade": manifest["reference_grade"],
        "status": "approved" if all(b["status"] == "approved" for b in bands.values()) else "not_approved",
        "bands_approved": sorted(b for b, v in bands.items() if v["status"] == "approved"),
        "tool_git": _git_info(),
        "bands": bands,
    }


def load_approved_sources(path, band) -> dict:
    """Read `exp_a_sources.json` and return the approved final-run record for one band.

    The single entry point Experiment F uses. It refuses anything that is not an approved
    band of an approved comparison, so F can never be handed a run directory the gate did
    not pass.
    """
    path = Path(path)
    payload = json.loads(_require_file(path, "exp_a_sources.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != SOURCES_SCHEMA_VERSION:
        raise ReferenceGateError(
            f"{path}: schema is {payload.get('schema_version')!r}, expected {SOURCES_SCHEMA_VERSION!r}"
        )
    band_record = (payload.get("bands") or {}).get(band)
    if band_record is None:
        raise ReferenceGateError(f"{path}: no record for band {band!r}")
    if band_record.get("status") != "approved":
        raise ReferenceGateError(
            f"{path}: band {band!r} is {band_record.get('status')!r} "
            f"(mismatched: {band_record.get('mismatched_evidence_classes')}) — Exp F refuses to "
            "consume an Exp-A selection table the reference gate did not approve"
        )
    return band_record


def write_json(path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
