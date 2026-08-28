"""The owner's sanctioned EXPLORATORY frame-level random split — deliberately leaky.

**Nothing this module produces may appear in any reported result.** It exists because the
owner asked (2026-07-30, `implementation_plan.md:925-941`) for a paper-comparable private
number alongside the LOSO protocol: the original study split frames at random, so subjects
appear on both sides of the split and every number here is optimistically biased by subject
identity. The three hard constraints of that decision are implemented structurally, not by
convention:

  1. **In addition to LOSO, never instead of it.** Every unit here is a REFIT of the
     configuration its LOSO run already selected — there is no search, so nothing about the
     protocol can be chosen on these numbers (Step 0 item 2).
  2. **Structurally isolated.** This module never imports `eval.splits` (the fold source);
     it makes its own `KFold` over pooled frames, and it writes only through
     `_require_exploratory_path`, an allowlist that admits exactly
     `results/exploratory_frame_split/**/…frameSplit_leaked_exploratory…`. It does not call
     `provenance.record_run`, which would create `results/runs/<stamp>_<rev>/` (D11 forbids
     that for this path); it calls the same public payload builder and writes elsewhere.
  3. **Tagged as unreportable.** Every filename carries `frameSplit_leaked_exploratory` and
     every JSON carries `leaky_protocol` / `never_report` at the top level.

**The one leak that is sanctioned is subject overlap — and nothing else.** Fit-on-train-only
still governs every fitted quantity, which is why the tuned-ε branch is recomputed here from
the RAW scattering tensors restricted to each fold's TRAINING frames instead of reading the
stored `prelog__*` tuples: those tuples are per-session medians over *all* of a session's
frames, so under a pooled frame split consuming one would fit ε on the very rows being
scored — a second, unsanctioned leak (plan §5 trap 19). The frozen subject-balanced
hierarchy (`:477-500`) is kept intact; only its innermost population narrows from "the
session" to "that session's training frames".
"""

from __future__ import annotations

import csv
import hashlib
import json
from ast import literal_eval
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from ..features import store as store_mod
from ..features.pooling import pool_stats_batch
from ..features.wst import apply_order_log
from ..features.extraction_77 import apply_order_log_77
from ..models import cnn
from ..models.baselines import fit_session_index_baseline, predict_session_index
from ..models.regressors import build_estimator, fitted_state_params
from ..provenance import build_provenance_payload, sha256_file
from . import exp_d
from . import metrics as M
from .harness import tuned_epsilons

BANDS = ("10ghz", "77ghz")

# The KFold's `random_state`, frozen by the plan as `config.run.seed + 900`; fits take their
# own offset so a fold's model seed can never coincide with the split's.
KFOLD_SEED_OFFSET = 900
FIT_SEED_OFFSET = 950

EXPLORATORY_DIRNAME = "exploratory_frame_split"
EXPLORATORY_TAG = "frameSplit_leaked_exploratory"
TUNED_EPS_AGGREGATION = "frozen_hierarchy_training_frames_only"

# THE MATRIX (plan §2.10, D11), amended by explicit owner authorization on 2026-08-28.
# The original 16 runs remain unchanged; the full Exp-A WST regressor is now added once per
# band as `radar_wst`. It uses the same quarantine, modal-config refit and train-frame-only
# fitted-transform rules. `session_index` is near-degenerate under a frame split (every test
# frame's session is trained on) and runs only because the original sanction said "every Exp
# D baseline".
ORDINAL_UNITS = ("arm_a", "arm_b")
WST_REGRESSION_UNIT = "radar_wst"
REGRESSION_UNITS = (*exp_d.EXPD_FAMILIES, WST_REGRESSION_UNIT)
TASK_UNITS = {"ordinal": ORDINAL_UNITS, "regression": REGRESSION_UNITS}
FRAME_SPLIT_MATRIX = tuple(
    (task, unit, band)
    for task in ("ordinal", "regression")
    for unit in TASK_UNITS[task]
    for band in BANDS
)
DEGENERATE_UNITS = ("session_index",)

ORDINAL_METRICS = ("class_unit_mae", "adjacent_accuracy", "quadratic_weighted_kappa", "accuracy")
REGRESSION_METRICS = ("frame_mae",)
WST_REGRESSION_METRICS = (
    "frame_mae",
    "frame_rmse",
    "frame_pearson_r",
    "session_mae",
    "subject_balanced_session_mae",
    "session_rmse",
    "session_pearson_r",
)


class FrameSplitError(ValueError):
    """A unit outside the sanctioned matrix, a bad source artifact, or a forbidden path."""


# ------------------------------------------------------------------- output isolation


def exploratory_root(config) -> Path:
    return Path(config.paths.results_dir) / EXPLORATORY_DIRNAME


def exploratory_dir(config, band) -> Path:
    return exploratory_root(config) / band


def _require_exploratory_path(path, *, results_dir) -> Path:
    """An ALLOWLIST, not a `runs/`-substring refusal: the resolved path must live under
    `results/exploratory_frame_split/` and its filename must carry the tag. A refusal keyed
    on what is forbidden would pass anything spelled differently; this passes only what is
    explicitly permitted."""
    resolved = Path(path).resolve()
    root = (Path(results_dir) / EXPLORATORY_DIRNAME).resolve()
    if not resolved.is_relative_to(root):
        raise FrameSplitError(
            f"refusing to write {resolved} — the exploratory frame split may only write "
            f"under {root} (its outputs must never be mistaken for a reportable run)"
        )
    if EXPLORATORY_TAG not in resolved.name:
        raise FrameSplitError(
            f"refusing to write {resolved.name} — every exploratory filename must carry "
            f"{EXPLORATORY_TAG!r} so a stray file can never be read as a LOSO artifact"
        )
    return resolved


def _write_tagged(path, text, *, results_dir) -> Path:
    out = _require_exploratory_path(path, results_dir=results_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


# ------------------------------------------------------- the source LOSO configuration


@dataclass(frozen=True)
class SourceRun:
    """The LOSO artifact this exploratory run mirrors, and what it resolved to.

    Recorded in full in the exploratory provenance: without it two runs that read different
    selection tables would carry indistinguishable provenance (C24), since the payload's
    manifest summary pins neither the artifact nor the configuration it defines.
    """

    run_dir: Path
    artifact_path: Path
    artifact_rel_path: str
    artifact_sha256: str
    analysis_commit: str | None
    config_hash: str | None
    resolved_config: dict

    def as_dict(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "analysis_commit": self.analysis_commit,
            "config_hash": self.config_hash,
            "artifact_rel_path": self.artifact_rel_path,
            "artifact_sha256": self.artifact_sha256,
        }


def _read_csv_rows(path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _lowest_fold_winner(rows, key_of, fold_of):
    """The most-selected key, ties broken toward the key chosen by the LOWEST fold id.

    Both reductions this module needs (the classical (feature_key, family, params) triple
    and the CNN (lr, weight_decay) pair) are this same rule, so it is written once.
    """
    counts = Counter(key_of(row) for row in rows)
    if not counts:
        raise FrameSplitError("no selected configuration to reduce — the artifact has no rows")
    best = max(counts.values())
    tied = {key for key, count in counts.items() if count == best}
    for row in sorted(rows, key=fold_of):
        if key_of(row) in tied:
            return key_of(row), best, int(fold_of(row))
    raise FrameSplitError("unreachable: a tied key must appear in the rows it was counted from")


def modal_classical_config(rows) -> dict:
    """The most-selected classical (feature_key, family, params) triple.

    `rows` are one Exp C arm or the Exp A LOSO `selection_table_{band}.csv` — the ARTIFACT,
    never a recomputation (plan §5 trap 15): a recomputation could silently diverge from
    what the LOSO run actually selected. The fold identity is its held-out subject, so that
    is the "fold id" the tie-break orders on.
    """
    usable = [r for r in rows if not (r.get("reason") or "").strip()]
    if not usable:
        raise FrameSplitError(
            "every row of this arm's selection table records a non-selectable fold — there "
            "is no modal configuration to refit"
        )

    def key_of(row):
        return (row["feature_key"], row["family"], row["params"])

    (feature_key, family, params), n_selected, fold_id = _lowest_fold_winner(
        usable, key_of, lambda r: int(r["test_subject"])
    )
    return {
        "feature_key": tuple(literal_eval(feature_key)),
        "family": family,
        "params": dict(literal_eval(params)),
        "modal_fold_id": fold_id,
        "n_selected": int(n_selected),
        "n_folds": len(usable),
    }


def modal_cnn_config(rows) -> dict:
    """One CNN family's cross-fold reduction: the modal `(lr, weight_decay)` pair (ties
    toward the lowest fold id), and `int(floor(median(budgets)))` over ONLY the folds that
    selected that pair.

    Floor rather than `np.median` because a plain median over an even count returns the mean
    of the two middle values, which is not an integral epoch count (plan §5 trap 15).
    """
    if not rows:
        raise FrameSplitError("the merged CNN selection table has no rows to reduce")

    def key_of(row):
        return (float(row["learning_rate"]), float(row["weight_decay"]))

    (lr, weight_decay), n_selected, fold_id = _lowest_fold_winner(
        rows, key_of, lambda r: int(r["fold_id"])
    )
    budgets = [int(r["epoch_budget"]) for r in rows if key_of(r) == (lr, weight_decay)]
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "epoch_budget": int(np.floor(np.median(np.asarray(budgets, dtype=float)))),
        "budgets": budgets,
        "modal_fold_id": fold_id,
        "n_selected": int(n_selected),
        "n_folds": len(rows),
    }


def _check_lineage(where, found, expected, field_name) -> None:
    if expected is not None and found != expected:
        raise FrameSplitError(
            f"{where}: {field_name} is {found!r} but this run's is {expected!r} — the "
            "exploratory refit must mirror a LOSO run of the SAME code and config, or the "
            "configuration it claims to reuse is not the one that was selected"
        )


def _provenance_config_hash(provenance, provenance_path) -> str:
    """Recover Exp A's config fingerprint from its canonical provenance payload.

    Exp A predates the later run-group schema and therefore records the complete
    ``config`` mapping instead of duplicating its SHA-256 under ``extra.config_hash``.
    This is deliberately the same JSON recipe as ``exp_b.config_fingerprint``; hashing
    the persisted mapping lets the frame-split runner verify the source without weakening
    the same-config requirement or asking users to edit a historical artifact.
    """
    config_payload = provenance.get("config")
    if not isinstance(config_payload, dict):
        raise FrameSplitError(
            f"Exp A provenance {provenance_path} lacks its canonical config mapping"
        )
    return hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_source_run(config, band, task, unit, run_dir, *, analysis_commit=None,
                    config_hash=None) -> SourceRun:
    """Read (and validate) the LOSO artifact whose selection this run refits.

    Fails closed BEFORE any fitting on a missing artifact or a lineage mismatch: an
    exploratory number that mirrors a differently-configured or differently-coded LOSO run
    would be untraceable to the selection it claims to reproduce (C24).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FrameSplitError(f"the source LOSO run dir {run_dir} does not exist")

    if task == "ordinal" or unit == WST_REGRESSION_UNIT:
        artifact = run_dir / f"selection_table_{band}.csv"
        if not artifact.is_file():
            experiment = "Exp C" if task == "ordinal" else "Exp A"
            raise FrameSplitError(f"missing {experiment} selection table {artifact}")
        rows = _read_csv_rows(artifact)
        if task == "ordinal":
            arm = "a" if unit == "arm_a" else "b"
            rows = [row for row in rows if row["arm"] == arm]
        resolved = modal_classical_config(rows)
        provenance_path = run_dir / "provenance.json"
        if not provenance_path.is_file():
            raise FrameSplitError(f"the source LOSO run dir {run_dir} has no provenance.json")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        found_commit = (provenance.get("git") or {}).get("commit")
        found_hash = (provenance.get("extra") or {}).get("config_hash")
        if unit == WST_REGRESSION_UNIT and found_hash is None:
            found_hash = _provenance_config_hash(provenance, provenance_path)
    else:
        artifact = run_dir / f"selection_{unit}_{band}.csv"
        metrics_path = run_dir / f"metrics_{unit}_{band}.json"
        # the four-artifact acceptance rules live in exp_d, so the comparison stage and this
        # path cannot disagree about what "a readable family" means
        exp_d.load_family_artifacts(run_dir, band, unit)
        lineage = json.loads(metrics_path.read_text(encoding="utf-8")).get("lineage") or {}
        found_commit = lineage.get("analysis_commit")
        found_hash = lineage.get("config_hash")
        resolved = (
            modal_cnn_config(_read_csv_rows(artifact))
            if unit in cnn.CNN_FAMILIES
            else {"note": "no hyperparameters: the frozen estimator is refit per leaky fold"}
        )

    where = f"source LOSO artifact {artifact}"
    _check_lineage(where, found_commit, analysis_commit, "analysis_commit")
    _check_lineage(where, found_hash, config_hash, "config_hash")
    return SourceRun(
        run_dir=run_dir,
        artifact_path=artifact,
        artifact_rel_path=artifact.name,
        artifact_sha256=sha256_file(artifact),
        analysis_commit=found_commit,
        config_hash=found_hash,
        resolved_config=resolved,
    )


# ---------------------------------------------- the train-only pre-log scale, per FRAME


def per_frame_prelog(S, meta) -> dict:
    """`extraction._prelog_scale`'s per-frame intermediate, stopped one step early.

    That function's session value is `median over frames` of exactly these numbers (time
    mean -> mean over the order's paths -> mean over channels), so taking the median over a
    SUBSET of frames narrows the innermost population and changes nothing else. Returns
    `{order: [n_frames]}` for orders 0, 1, 2.
    """
    order = np.asarray(meta["order"] if isinstance(meta, dict) else meta.meta()["order"])
    time_mean = np.asarray(S, dtype=np.float64).mean(axis=-1)          # [N, C, P]
    return {
        o: time_mean[:, :, order == o].mean(axis=-1).mean(axis=-1)     # [N, C] -> [N]
        for o in (0, 1, 2)
    }


def training_frame_epsilons(per_session_prelog, session_subjects, train_masks, *, k,
                            fallback) -> dict:
    """The frozen tuned-ε hierarchy with its innermost population narrowed to training frames.

    session scale = median over that session's TRAINING frames -> per-subject value = mean
    over that subject's sessions with >= 1 training frame -> scale_o = median over subjects
    with >= 1 such session -> ε_o = k · scale_o. The last three steps are
    `harness.tuned_epsilons` itself, called with per-session tuples built from training
    frames only — one hierarchy, not a second implementation of it.

    Pooling every training frame into one median is explicitly REJECTED: it would overweight
    subjects and sessions that contribute more frames, discarding the subject balancing that
    `:485-489` gives as the reason for the two-stage form.
    """
    prelog_by_subject: dict = {}
    for values, subject, mask in zip(per_session_prelog, session_subjects, train_masks, strict=True):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue        # a session with no training frames drops out of its subject's mean
        prelog_by_subject.setdefault(int(subject), []).append(
            tuple(float(np.median(np.asarray(values[o])[mask])) for o in (0, 1, 2))
        )
    # a subject with no such session is absent here, so it drops out of the median too
    return tuned_epsilons(prelog_by_subject, sorted(prelog_by_subject), k=k, fallback=fallback)


# ------------------------------------------------------------------ the pooled frame table


@dataclass
class FrameTable:
    """One row per QC-passed frame of every eligible session, in canonical (subject,
    session_idx, stored frame order) order — the order the KFold indexes into."""

    band: str
    subjects: np.ndarray
    session_row: np.ndarray
    session_idx: np.ndarray
    frame_ids: np.ndarray
    y: np.ndarray                        # Δm% per frame (regression target)
    y_class: np.ndarray | None = None    # S0-S4 per frame (ordinal truth)
    y_loss: np.ndarray | None = None     # L = -Δm% per frame
    x: np.ndarray | None = None          # fold-independent features, when there are any
    raw: list | None = None              # [(S, order)] per session, for the tuned branch
    branch: str | None = None

    def identity_lines(self) -> list[str]:
        return [
            f"{int(s)}|{int(i)}|{int(f)}"
            for s, i, f in zip(self.subjects, self.session_idx, self.frame_ids, strict=True)
        ]


def _session_arrays(sessions):
    subjects, session_row, session_idx, frame_ids, y = [], [], [], [], []
    for row, session in enumerate(sessions):
        for frame_id in session["frame_ids"]:
            subjects.append(int(session["subject"]))
            session_row.append(row)
            session_idx.append(int(session["session_idx"]))
            frame_ids.append(int(frame_id))
            y.append(float(session["delta_m_pct"]))
    return (np.array(subjects, dtype=int), np.array(session_row, dtype=int),
            np.array(session_idx, dtype=int), np.array(frame_ids, dtype=int),
            np.array(y, dtype=float))


def _raw_keys(band, feature_key):
    if band == "10ghz":
        gi, r, c, ti, _ = feature_key
        return store_mod.raw_key(gi, r, c, ti), store_mod.order_key(ti)
    ti, _ = feature_key
    return store_mod.raw77_key(ti), store_mod.order_key(ti)


def _read_raw_wst_sessions(band, sessions, store_dir, feature_key) -> list:
    """Read the per-frame raw WST tensors at one authenticated feature key."""
    raw_key, order_key = _raw_keys(band, feature_key)
    raw = []
    for session in sessions:
        store = store_mod.read_session_store(band, session["subject"], session["session_name"],
                                             store_dir)
        try:
            if raw_key not in store:
                raise FrameSplitError(
                    f"store for {band} {session['subject']}/{session['session_name']} has no "
                    f"{raw_key!r} — the exploratory refit reconstructs per-frame vectors from "
                    "the raw tensors"
                )
            S = np.asarray(store[raw_key])
            order = np.asarray(store[order_key])
        finally:
            store.close()
        expected = len(session["frame_ids"])
        if S.shape[0] != expected:
            raise FrameSplitError(
                f"store for {band} {session['subject']}/{session['session_name']} holds "
                f"{S.shape[0]} raw frames but the QC spine selected {expected}"
            )
        raw.append((S, order))
    return raw


def build_ordinal_frame_table(config, band, sessions, store_dir, feature_key) -> FrameTable:
    """Per-frame pooled WST vectors for the ordered S0--S4 task."""
    subjects, session_row, session_idx, frame_ids, y = _session_arrays(sessions)
    raw = _read_raw_wst_sessions(band, sessions, store_dir, feature_key)

    branch = feature_key[-1]
    table = FrameTable(
        band=band, subjects=subjects, session_row=session_row, session_idx=session_idx,
        frame_ids=frame_ids, y=y,
        y_class=np.array([int(s["class_idx"]) for s in sessions], dtype=int)[session_row],
        y_loss=np.array([float(s["loss_l"]) for s in sessions], dtype=float)[session_row],
        raw=raw, branch=branch,
    )
    if branch != "tuned":
        table.x = _ordinal_features(config, band, raw, branch, epsilons=None)
    return table


def build_wst_regression_frame_table(config, band, sessions, store_dir,
                                     feature_key) -> FrameTable:
    """Per-frame WST design for the newly authorized full Exp-A regressor.

    Session aggregates in the normal feature store cannot be used here because a random
    frame split places frames from the same session on both sides. The raw tensors are
    therefore reconstructed into one pooled vector per frame, exactly as for Exp C.
    """
    subjects, session_row, session_idx, frame_ids, y = _session_arrays(sessions)
    raw = _read_raw_wst_sessions(band, sessions, store_dir, feature_key)
    branch = feature_key[-1]
    table = FrameTable(
        band=band,
        subjects=subjects,
        session_row=session_row,
        session_idx=session_idx,
        frame_ids=frame_ids,
        y=y,
        raw=raw,
        branch=branch,
    )
    if branch != "tuned":
        table.x = _ordinal_features(config, band, raw, branch, epsilons=None)
    return table


def _ordinal_features(config, band, raw, branch, *, epsilons) -> np.ndarray:
    """The per-frame pooled vectors for every session, stacked in canonical frame order."""
    rows = []
    for S, order in raw:
        meta = {"order": order}
        if band == "10ghz":
            eps = ({1: config.wst.log_epsilon, 2: config.wst.log_epsilon}
                   if branch == "frozen" else epsilons)
            logged = apply_order_log(S, meta, config.wst, log_on=branch != "off",
                                     epsilon_by_order=eps)
        else:
            log_branch = {"off": "off", "frozen": "on_frozen_eps", "tuned": "on_tuned_eps"}[branch]
            logged = np.stack([
                apply_order_log_77(S[i], meta, config.wst77, log_branch=log_branch,
                                   epsilon_by_order=epsilons)
                for i in range(S.shape[0])
            ])
        rows.append(pool_stats_batch(logged, meta))
    return np.ascontiguousarray(np.concatenate(rows, axis=0))


def build_physics_frame_table(config, band, sessions, store_dir) -> FrameTable:
    """The per-frame power-ratio scalar as a one-column design matrix. Reads the same
    unstandardized stored signal the LOSO physics baseline reads."""
    subjects, session_row, session_idx, frame_ids, y = _session_arrays(sessions)
    key = exp_d.PHYSICS_SIGNAL_KEY[band]
    scalars = []
    for session in sessions:
        store = store_mod.read_session_store(band, session["subject"], session["session_name"],
                                             store_dir)
        try:
            signals = np.asarray(store[key])
        finally:
            store.close()
        scalars += [exp_d.physics_frame_scalar(config, band, signals[i])
                    for i in range(len(session["frame_ids"]))]
    return FrameTable(
        band=band, subjects=subjects, session_row=session_row, session_idx=session_idx,
        frame_ids=frame_ids, y=y, x=np.asarray(scalars, dtype=float).reshape(-1, 1),
    )


def build_session_index_frame_table(config, band, sessions) -> FrameTable:
    """No radar input at all: the session label is the whole model. Near-degenerate under a
    frame split, and run only because the sanction says "every Exp D baseline"."""
    subjects, session_row, session_idx, frame_ids, y = _session_arrays(sessions)
    return FrameTable(band=band, subjects=subjects, session_row=session_row,
                      session_idx=session_idx, frame_ids=frame_ids, y=y)


# ------------------------------------------------------------------------- the leaky folds


def _state_sha256(params) -> str:
    """A byte handle on a fold's fitted state, so the mutation property is checkable from
    the run's own output rather than from an instrumented internal."""
    digest = hashlib.sha256()
    for name in sorted(params):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(params[name]).tobytes())
    return digest.hexdigest()


def _ordinal_fold_metrics(y_true, y_pred) -> dict:
    """The three frozen ordinal metrics plus PLAIN ACCURACY — deliberately, because the
    paper-comparable number this exploratory path exists to produce is an accuracy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "class_unit_mae": M.class_unit_mae(y_true, y_pred),
        "adjacent_accuracy": M.adjacent_accuracy(y_true, y_pred),
        "quadratic_weighted_kappa": M.quadratic_weighted_kappa(y_true, y_pred),
        "accuracy": float(np.mean(np.rint(y_pred) == np.rint(y_true))) if len(y_true) else float("nan"),
    }


@dataclass
class FrameSplitResult:
    band: str
    task: str
    unit: str
    k: int
    kfold_random_state: int
    n_frames: int
    n_subjects: int
    n_sessions: int
    metric_names: tuple
    per_fold: list
    summary: dict
    resolved_config: dict
    source_run: dict
    frame_order_sha256: str
    fold_assignment_sha256: str
    tuned_eps_aggregation: str | None = None
    notes: dict = field(default_factory=dict)


def _summarize(per_fold, metric_names) -> dict:
    out = {}
    for name in metric_names:
        values = np.array([fold[name] for fold in per_fold], dtype=float)
        finite = values[np.isfinite(values)]
        out[name] = {
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "sd": float(np.std(finite, ddof=0)) if finite.size else float("nan"),
            "n_folds_evaluated": int(finite.size),
        }
    return out


def _wst_fold_design(config, band, table, train_idx):
    """Return the WST design, fitting tuned epsilon on training frames only."""
    epsilons = None
    if table.branch == "tuned":
        train_mask = np.zeros(len(table.subjects), dtype=bool)
        train_mask[train_idx] = True
        per_session, session_subjects, masks = [], [], []
        for row, (S, order) in enumerate(table.raw):
            rows_here = table.session_row == row
            per_session.append(per_frame_prelog(S, {"order": order}))
            session_subjects.append(int(table.subjects[rows_here][0]))
            masks.append(train_mask[rows_here])
        search = config.search_10ghz if band == "10ghz" else config.search_77ghz
        log_cfg = config.wst if band == "10ghz" else config.wst77
        epsilons = training_frame_epsilons(
            per_session, session_subjects, masks, k=search.tuned_eps_k,
            fallback=log_cfg.log_epsilon,
        )
        x = _ordinal_features(config, band, table.raw, table.branch, epsilons=epsilons)
    else:
        x = table.x
    return x, epsilons


def _fit_ordinal_fold(config, band, table, resolved, train_idx, test_idx, *, seed):
    """One leaky fold of an Exp C arm using only train-fitted quantities."""
    x, epsilons = _wst_fold_design(config, band, table, train_idx)

    y = np.column_stack([table.y_loss, table.y_class.astype(float)])
    estimator = build_estimator(resolved["family"], resolved["params"], seed=seed)
    estimator.fit(x[train_idx], y[train_idx])
    predictions = np.asarray(estimator.predict(x[test_idx]), dtype=float)

    state = dict(fitted_state_params(resolved["family"], estimator.named_steps["model"]))
    scaler = estimator.named_steps["scaler"]
    state["scaler_mean_"] = np.asarray(scaler.mean_, dtype=float)
    state["scaler_scale_"] = np.asarray(scaler.scale_, dtype=float)
    metrics = _ordinal_fold_metrics(table.y_class[test_idx], predictions)
    metrics["tuned_epsilon"] = ([float(epsilons[1]), float(epsilons[2])]
                                if epsilons is not None else None)
    metrics["fitted_state_sha256"] = _state_sha256(state)
    return metrics


def _finite_pearson_r(truth, prediction) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.size < 2 or truth.std(ddof=0) == 0.0 or prediction.std(ddof=0) == 0.0:
        return float("nan")
    return float(np.corrcoef(truth, prediction)[0, 1])


def _wst_regression_fold_metrics(table, test_idx, predictions) -> dict:
    """Frame metrics plus within-fold session aggregation for comparison with LOSO.

    The session metrics take the median prediction over that fold's held-out frames from
    each represented session. They remain leaky because other frames from those sessions
    are in training; their purpose is only to put the protocol comparison on the same
    session scale as Exp A, never to estimate generalization.
    """
    truth = np.asarray(table.y[test_idx], dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if truth.shape != predictions.shape or truth.size == 0:
        raise FrameSplitError("WST regression fold has empty or misaligned predictions")

    session_truth, session_prediction, session_subject = [], [], []
    test_session_rows = table.session_row[test_idx]
    for session_row in sorted(set(test_session_rows.tolist())):
        mask = test_session_rows == session_row
        values = truth[mask]
        if not np.all(values == values[0]):
            raise FrameSplitError("one session carries different regression targets")
        session_truth.append(float(values[0]))
        session_prediction.append(float(np.median(predictions[mask])))
        session_subject.append(int(table.subjects[test_idx][mask][0]))

    session_truth = np.asarray(session_truth, dtype=float)
    session_prediction = np.asarray(session_prediction, dtype=float)
    session_subject = np.asarray(session_subject, dtype=int)
    session_absolute_error = np.abs(session_truth - session_prediction)
    per_subject_mae = [
        float(session_absolute_error[session_subject == subject].mean())
        for subject in sorted(set(session_subject.tolist()))
    ]
    return {
        "frame_mae": float(np.mean(np.abs(truth - predictions))),
        "frame_rmse": float(np.sqrt(np.mean((truth - predictions) ** 2))),
        "frame_pearson_r": _finite_pearson_r(truth, predictions),
        "session_mae": float(session_absolute_error.mean()),
        "subject_balanced_session_mae": float(np.mean(per_subject_mae)),
        "session_rmse": float(np.sqrt(np.mean((session_truth - session_prediction) ** 2))),
        "session_pearson_r": _finite_pearson_r(session_truth, session_prediction),
        "n_test_sessions": int(session_truth.size),
    }


def _fit_wst_regression_fold(config, band, table, resolved, train_idx, test_idx, *, seed):
    """Refit the modal full-WST Exp-A estimator on one leaky random-frame fold."""
    x, epsilons = _wst_fold_design(config, band, table, train_idx)
    estimator = build_estimator(resolved["family"], resolved["params"], seed=seed)
    estimator.fit(x[train_idx], table.y[train_idx])
    predictions = np.asarray(estimator.predict(x[test_idx]), dtype=float)

    state = dict(fitted_state_params(resolved["family"], estimator.named_steps["model"]))
    scaler = estimator.named_steps["scaler"]
    state["scaler_mean_"] = np.asarray(scaler.mean_, dtype=float)
    state["scaler_scale_"] = np.asarray(scaler.scale_, dtype=float)
    metrics = _wst_regression_fold_metrics(table, test_idx, predictions)
    metrics["tuned_epsilon"] = ([float(epsilons[1]), float(epsilons[2])]
                                if epsilons is not None else None)
    metrics["fitted_state_sha256"] = _state_sha256(state)
    return metrics


def _fit_cnn_fold(config, band, unit, frames, resolved, train_idx, test_idx, *, seed, device):
    """One leaky fold of a CNN family: the modal (lr, weight_decay) at the reduced epoch
    budget, no early stopping and no inner search. The spectrogram normalization is still
    fit on the fold's TRAINING frames — the sanctioned leak is subject overlap, not a
    pooled fitted transform."""
    prepared = exp_d._prepare(
        frames, train_idx, test_idx, role="frame_split_train",
        subjects=sorted(set(frames.subjects[train_idx].tolist())),
    )
    baselines = config.baselines
    outcome = exp_d._train_cnn(
        prepared.x_train, prepared.y_train, prepared.weights, family=unit,
        lr=resolved["lr"], weight_decay=resolved["weight_decay"],
        max_epochs=resolved["epoch_budget"], batch_size=baselines.batch_size,
        betas=baselines.adam_betas, patience=baselines.early_stopping_patience,
        min_delta=baselines.early_stopping_min_delta, seed_value=seed, device=device, val=None,
    )
    predictions = exp_d._predict_with_state(
        unit, int(frames.X.shape[1]), outcome.state, prepared.x_eval, device
    )
    state = cnn.torch_module_state_to_numpy(outcome.state)
    for record in prepared.fits:
        state.update({f"{record.quantity}__{k}": v for k, v in record.params.items()})
    return {
        "frame_mae": float(np.mean(np.abs(frames.y[test_idx] - predictions))),
        "fitted_state_sha256": _state_sha256(state),
    }


def _fit_cheap_fold(unit, table, train_idx, test_idx):
    """The two deterministic baselines, refit on the fold's training FRAMES.

    `fit_session_index_baseline` is reused verbatim; its row selection is by "subject", so
    passing row positions as the subject axis selects training frames with the frozen
    estimator rather than a second copy of it.
    """
    if unit == "physics":
        slope, intercept = exp_d._least_squares_line(
            table.x[train_idx, 0], table.y[train_idx]
        )
        predictions = slope * table.x[test_idx, 0] + intercept
        state = {"slope": np.asarray(slope), "intercept": np.asarray(intercept)}
    else:
        rows = np.arange(len(table.y))
        outcome = fit_session_index_baseline(
            rows, table.session_idx, table.y, rows[train_idx], role="frame_split_train"
        )
        predictions = predict_session_index(outcome.model, table.session_idx[test_idx])
        state = dict(outcome.fit_record.params)
    return {
        "frame_mae": float(np.mean(np.abs(table.y[test_idx] - np.asarray(predictions, dtype=float)))),
        "fitted_state_sha256": _state_sha256(state),
    }


def run_frame_split(config, band, task, unit, k=5, *, source_run_dir, sessions=None,
                    store_dir=None, analysis_commit=None, config_hash=None,
                    device=None) -> FrameSplitResult:
    """One of the sanctioned exploratory runs: refit the unit's modal LOSO configuration
    on 80% of pooled shuffled frames, score the held-out 20%, five times.

    NOT a nested search (Step 0 item 2): the owner asked for paper-comparable numbers for
    private comparison, not a second validated procedure, and a search inside a leaky fold
    would spend its own allocation producing numbers that are forbidden from every report
    regardless of quality.
    """
    if (task, unit, band) not in FRAME_SPLIT_MATRIX:
        raise FrameSplitError(
            f"({task!r}, {unit!r}, {band!r}) is not one of the {len(FRAME_SPLIT_MATRIX)} "
            "sanctioned exploratory runs — the matrix covers Exp C, every Exp D baseline, "
            "and the separately authorized full-WST Exp A refit"
        )
    store_dir = config.paths.results_dir if store_dir is None else store_dir
    device = config.run.device if device is None else device
    if sessions is None:
        from . import exp_a, exp_c   # deferred: building a spine reaches the data layer

        sessions = (exp_c.build_sessions_c(config, band) if task == "ordinal"
                    else exp_a.build_sessions(config, band))

    source = load_source_run(config, band, task, unit, source_run_dir,
                             analysis_commit=analysis_commit, config_hash=config_hash)
    resolved = source.resolved_config

    frames = None
    if task == "ordinal":
        table = build_ordinal_frame_table(config, band, sessions, store_dir,
                                          resolved["feature_key"])
        metric_names = ORDINAL_METRICS
    elif unit == WST_REGRESSION_UNIT:
        table = build_wst_regression_frame_table(
            config, band, sessions, store_dir, resolved["feature_key"]
        )
        metric_names = WST_REGRESSION_METRICS
    elif unit in cnn.CNN_FAMILIES:
        frames = exp_d.build_frames_d(config, band, unit, sessions, store_dir)
        table = FrameTable(band=band, subjects=frames.subjects, session_row=frames.session_row,
                           session_idx=frames.session_idx[frames.session_row],
                           frame_ids=frames.frame_ids, y=frames.y)
        metric_names = REGRESSION_METRICS
    elif unit == "physics":
        table = build_physics_frame_table(config, band, sessions, store_dir)
        metric_names = REGRESSION_METRICS
    else:
        table = build_session_index_frame_table(config, band, sessions)
        metric_names = REGRESSION_METRICS

    random_state = int(config.run.seed) + KFOLD_SEED_OFFSET
    splitter = KFold(n_splits=int(k), shuffle=True, random_state=random_state)
    identities = table.identity_lines()

    per_fold, assignment = [], list(identities)
    for j, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(identities)))):
        for i in test_idx:
            assignment[i] = f"{identities[i]}|fold{j}"
        if task == "ordinal":
            # rf/gbm are the only seed-sensitive ordinal families; a single refit has no seed
            # protocol of its own, so it takes the FIRST seed of the frozen set — the same
            # "first-seed" convention O-M9-1 uses for the ordinal tie-break's QWK.
            metrics = _fit_ordinal_fold(config, band, table, resolved, train_idx, test_idx,
                                        seed=int(config.run.seed_set[0]))
        elif unit == WST_REGRESSION_UNIT:
            metrics = _fit_wst_regression_fold(
                config, band, table, resolved, train_idx, test_idx,
                seed=int(config.run.seed_set[0]),
            )
        elif unit in cnn.CNN_FAMILIES:
            metrics = _fit_cnn_fold(config, band, unit, frames, resolved, train_idx, test_idx,
                                    seed=int(config.run.seed) + FIT_SEED_OFFSET + j,
                                    device=device)
        else:
            metrics = _fit_cheap_fold(unit, table, train_idx, test_idx)
        per_fold.append({"fold": j, "n_train_frames": int(len(train_idx)),
                         "n_test_frames": int(len(test_idx)), **metrics})

    notes = {}
    if unit in DEGENERATE_UNITS:
        notes["degenerate_by_construction"] = True
    if unit == WST_REGRESSION_UNIT:
        notes.update({
            "scientific_status": "leaky_protocol_demonstration_only",
            "session_metrics": (
                "median over each fold's held-out frames per session; other frames from the "
                "same subject/session remain in training"
            ),
            "not_directly_comparable_to_loso": True,
        })
    return FrameSplitResult(
        band=band, task=task, unit=unit, k=int(k), kfold_random_state=random_state,
        n_frames=len(identities), n_subjects=len(set(table.subjects.tolist())),
        n_sessions=len(sessions), metric_names=tuple(metric_names),
        per_fold=per_fold, summary=_summarize(per_fold, metric_names),
        resolved_config=resolved, source_run=source.as_dict(),
        frame_order_sha256=exp_d._sha256_lines(identities),
        fold_assignment_sha256=exp_d._sha256_lines(assignment),
        tuned_eps_aggregation=(
            TUNED_EPS_AGGREGATION
            if (task == "ordinal" or unit == WST_REGRESSION_UNIT)
            and table.branch == "tuned"
            else None
        ),
        notes=notes,
    )


# ----------------------------------------------------------------------------- reporting


def _stem(result) -> str:
    return f"{result.task}_{result.unit}_{EXPLORATORY_TAG}"


def result_payload(result) -> dict:
    return {
        "leaky_protocol": True,
        "never_report": True,
        "band": result.band,
        "task": result.task,
        "unit": result.unit,
        "k_folds": result.k,
        "kfold_random_state": result.kfold_random_state,
        "n_frames": result.n_frames,
        "n_subjects": result.n_subjects,
        "n_sessions": result.n_sessions,
        "per_fold": result.per_fold,
        "summary": result.summary,
        "resolved_config": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in result.resolved_config.items()},
        "source_run": result.source_run,
        "frame_order_sha256": result.frame_order_sha256,
        "fold_assignment_sha256": result.fold_assignment_sha256,
        "tuned_eps_aggregation": result.tuned_eps_aggregation,
        "notes": result.notes,
    }


def write_frame_split_reports(result, config) -> dict:
    """`{task}_{unit}_frameSplit_leaked_exploratory.{json,csv}` under the band's exploratory
    directory — through the allowlist, so an output path is never merely *hoped* to be
    outside `results/runs/`."""
    results_dir = config.paths.results_dir
    out_dir = exploratory_dir(config, result.band)
    stem = _stem(result)

    json_path = _write_tagged(
        out_dir / f"{stem}.json",
        json.dumps(result_payload(result), indent=2, sort_keys=True) + "\n",
        results_dir=results_dir,
    )
    columns = ["fold", "n_train_frames", "n_test_frames", *result.metric_names]
    lines = [",".join(columns)]
    for fold in result.per_fold:
        lines.append(",".join(str(fold[c]) for c in columns))
    csv_path = _write_tagged(out_dir / f"{stem}.csv", "\n".join(lines) + "\n",
                             results_dir=results_dir)
    return {"metrics": json_path, "per_fold": csv_path}


def write_exploratory_provenance(config, band, task, unit, manifest, out_dir, *, data_dir,
                                 result) -> Path:
    """The exploratory run's provenance — the SAME payload builder every LOSO run uses,
    written under the exploratory root instead of `results/runs/`.

    `manifest` and `data_dir` are parameters rather than implied because
    `provenance._hash_inputs` needs the QC manifest to know which files to hash and silently
    defaults to the 10 GHz root: a 77 GHz run that let it default would hash 10 GHz files
    under a 77 GHz label (the M8 C19/C22 failure mode).

    `folds=None` is deliberate — there are no LOSO folds here — but the seeded KFold's
    recipe is not by itself a record of what ran, since nothing in the payload pins the
    frame ORDER it indexes into nor the LOSO artifact whose modal configuration defines the
    computation. Both go in `extra`, hashed.
    """
    if manifest is None or len(manifest) == 0:
        raise FrameSplitError(
            "refusing to write exploratory provenance with an empty QC manifest — "
            "`_hash_inputs` would then hash no radar file at all and the record would "
            "attest nothing about the data (C21)"
        )
    payload = build_provenance_payload(
        config, manifest, folds=None, data_dir=data_dir,
        extra={
            "leaky_protocol": True,
            "never_report": True,
            "task": task,
            "unit": unit,
            "band": band,
            "k_folds": result.k,
            "kfold_random_state": result.kfold_random_state,
            "tuned_eps_aggregation": result.tuned_eps_aggregation,
            "frame_order_sha256": result.frame_order_sha256,
            "fold_assignment_sha256": result.fold_assignment_sha256,
            "source_run": result.source_run,
            "resolved_config": result_payload(result)["resolved_config"],
        },
    )
    # ...and at the TOP level too, not only inside `extra`: every JSON this path writes must
    # announce itself as unreportable to a reader who opens it and looks no further.
    payload["leaky_protocol"] = True
    payload["never_report"] = True
    return _write_tagged(
        Path(out_dir) / f"{task}_{unit}_provenance_{EXPLORATORY_TAG}.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        results_dir=config.paths.results_dir,
    )
