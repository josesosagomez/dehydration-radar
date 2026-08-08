"""Experiment F — the unavailable heart-rate question, plus the available-covariate sensitivity.

**A-M10-2 governs this whole module, and the two halves must never be conflated.**

*The registered question* was whether radar predictions are simply explained by heart rate
recorded before acquisition. The required HR observations do not exist: the repository holds
radar files and the weight workbook, and the workbook has name, age, height, five masses, loss
and notes — no HR field, and no external HR file was delivered. So F reports
`status="not_estimable_missing_heart_rate"` with `n_hr_observations=0` and the inventory
evidence behind it. It does **not** correlate radar against a proxy, does not fabricate values,
and — the failure mode that actually matters — does **not** let the static-covariate analysis be
read as an HR adjustment. Temperature logs are lost and glucose was never measured; both stay
uncontrolled and are recorded as such.

*The available analysis* is a separate, separately named thing: four nested ridge models under
outer LOSO on the S0-S4 eligible rows —

    1 clock  ·  2 clock+covariates  ·  3 clock+radar  ·  4 clock+radar+covariates

— in three non-factorial variants (`pct_full`, `pct_reduced`, `kg_full`). Its conclusion is a
limited clock/static-covariate sensitivity result and nothing more.

Two structural properties worth stating because the tests pin them: models 1 and 2 never read
the feature store at all (no radar block is built for them), and models 3 and 4 share the
byte-identical radar matrix within a fold (it is built once and handed to both). The radar block
comes from the Exp-A-selected FEATURE KEY for that fold — never an Exp-A fitted estimator — and
that key is read only from an Exp-A run the reference gate has marked `approved`
(`reference_gate.load_approved_sources`), so F can never consume a selection table that did not
pass the gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..features import store as store_mod
from ..features.protocol_freeze import protocol_freeze_guard
from ..models.regressors import build_estimator, fit_pipeline
from . import exp_a, fold_parallel, reference_gate
from . import metrics as M
from .harness import Candidate, require_complete_active
from .selection import SIMPLICITY_RANK, CandidateScore, select_candidate
from .splits import nested_loso_splits


class ExpFError(ValueError):
    """A malformed F spine, a missing/non-finite covariate, an Exp-A source the reference gate
    did not approve, or a selected feature key F cannot resolve against the store."""


# ------------------------------------------------------------------ the frozen F vocabulary

# The four nested models, in the plan's order. The ids are the artifact vocabulary; the numbers
# are only for reading §2.2 alongside the code.
MODEL_IDS = ("clock", "clock_covariates", "clock_radar", "clock_radar_covariates")
MODEL_NUMBER = {"clock": 1, "clock_covariates": 2, "clock_radar": 3, "clock_radar_covariates": 4}
RADAR_MODELS = ("clock_radar", "clock_radar_covariates")
COVARIATE_MODELS = ("clock_covariates", "clock_radar_covariates")

# Exactly three non-factorial variants. There is deliberately NO combined
# kg-plus-reduced-covariate variant: §2.2 freezes these three and no more.
VARIANTS = ("pct_full", "pct_reduced", "kg_full")
VARIANT_TARGET = {"pct_full": "delta_m_pct", "pct_reduced": "delta_m_pct", "kg_full": "delta_m_kg"}

# The config-to-workbook map, exactly as §2.2 writes it. Height is cm and mass is kg; the ORDER
# is part of the frozen design, so the covariate block's columns are these keys in this order.
COVARIATE_COLUMN_MAP = {
    "age": "age",
    "height": "height_cm",
    "baseline_mass": "baseline_mass_kg",
    "bmi": "bmi",
}

# Clock is a session-index one-hot over the FULL S0-S4 domain, not over the sessions a given
# fold happens to contain. Keeping the domain fixed makes the design matrix the same width in
# every fold and every variant; a session absent from a fold's training rows simply leaves an
# all-zero column, which the scaler passes through and ridge gives a zero coefficient.
CLOCK_SESSIONS = (0, 1, 2, 3, 4)

# (contrast_id, with_model, without_model). "with component minus without component", so a
# NEGATIVE difference means the added component improved prediction.
CONTRASTS = (
    ("radar_given_clock", "clock_radar", "clock"),                                    # 3 - 1
    ("radar_given_clock_covariates", "clock_radar_covariates", "clock_covariates"),   # 4 - 2
    ("covariates_given_clock", "clock_covariates", "clock"),                          # 2 - 1
    ("covariates_given_clock_radar", "clock_radar_covariates", "clock_radar"),        # 4 - 3
)
# The ONLY primary family: Holm-2 over the two radar increments, in `pct_full` alone.
PRIMARY_CONTRASTS = ("radar_given_clock", "radar_given_clock_covariates")
EXPLORATORY_CONTRASTS = ("covariates_given_clock", "covariates_given_clock_radar")
PRIMARY_VARIANT = "pct_full"
FAMILY_PRIMARY = "holm_2_pct_full_radar_increments"
FAMILY_SENSITIVITY = "none_sensitivity"
# The exploratory label is NOT invented here: `StatsConfig.expf_exploratory_correction` already
# froze it ("none_reported_individually"), and reading it keeps the artifact's vocabulary and
# the frozen protocol record as one string rather than two that can drift.

# Exp F RNG offsets off config.run.seed -- FIXED and NAMED, never a running counter (the Exp B
# trap-10 doctrine): Exp A occupies +0..3, Exp B +100..134, Exp C +200..212, Exp D +300...,
# Exp F +400..423.
RNG_OFFSET_EXPF_BASE = 400
_VARIANT_RNG_OFFSET = {"pct_full": 0, "pct_reduced": 10, "kg_full": 20}
_CONTRAST_RNG_OFFSET = {contrast_id: i for i, (contrast_id, _, _) in enumerate(CONTRASTS)}


def _rng_offset(variant: str, contrast_id: str) -> int:
    return RNG_OFFSET_EXPF_BASE + _VARIANT_RNG_OFFSET[variant] + _CONTRAST_RNG_OFFSET[contrast_id]


def _all_rng_offsets() -> list[int]:
    """Every resolved RNG offset this module uses -- tested for pairwise distinctness, and
    against Exp A's, B's, C's and D's."""
    return [_rng_offset(v, c) for v in VARIANTS for c, _, _ in CONTRASTS]


# ------------------------------------------------------- the unavailable heart-rate question

HR_STATUS = "not_estimable_missing_heart_rate"

AVAILABILITY_COLUMNS = (
    "variable", "availability", "source_checked", "observation_unit", "n_values", "reason",
)


def hr_inventory(config, gt) -> list[dict]:
    """One row per confound source, from an ACTUAL inspection rather than a remembered claim.

    The workbook's own column labels are read back and searched for a heart-rate field, and the
    configured data roots are listed by file extension. That is what makes `n_hr_observations=0`
    evidence instead of an assertion: if an HR file were ever added, this row would change and
    the plan requires implementation to stop and amend before reading it.
    """
    from ..data.ground_truth import EXPECTED_ROW1

    workbook = Path(config.paths.weight_xlsx)
    labels = sorted(str(v) for v in EXPECTED_ROW1.values())
    hr_labels = [label for label in labels if "heart" in label.lower() or label.lower() in {"hr", "bpm"}]

    roots = [("data_10ghz_dir", config.paths.data_10ghz_dir)]
    if config.paths.data_77ghz_dir is not None:
        roots.append(("data_77ghz_dir", config.paths.data_77ghz_dir))
    extensions = {}
    for _, root in roots:
        root = Path(root)
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file():
                    extensions[path.suffix.lower()] = extensions.get(path.suffix.lower(), 0) + 1
    roots_checked = "; ".join(f"{name}={Path(path)}" for name, path in roots)
    extension_note = ", ".join(f"{ext or '<none>'}:{n}" for ext, n in sorted(extensions.items())) or "empty"

    n_subjects = int(len(gt.subjects))
    rows = [{
        "variable": "heart_rate",
        "availability": "missing",
        "source_checked": f"{workbook} (columns: {', '.join(labels)}); {roots_checked}",
        "observation_unit": "not_applicable",
        "n_values": 0,
        "reason": (
            "no heart-rate column in the weight workbook "
            f"(matching labels found: {hr_labels or 'none'}) and no HR file in the data roots "
            f"(files by extension: {extension_note}); the registered HR question is "
            f"{HR_STATUS} and no proxy is substituted"
        ),
    }, {
        "variable": "temperature",
        "availability": "missing",
        "source_checked": f"{workbook}; {roots_checked}",
        "observation_unit": "not_applicable",
        "n_values": 0,
        "reason": "temperature logs are lost; the variable remains uncontrolled and is not adjusted for",
    }, {
        "variable": "glucose",
        "availability": "missing",
        "source_checked": f"{workbook}; {roots_checked}",
        "observation_unit": "not_applicable",
        "n_values": 0,
        "reason": "glucose was never measured; the variable remains uncontrolled and is not adjusted for",
    }]
    for name, column in COVARIATE_COLUMN_MAP.items():
        values = gt.subjects[column]
        rows.append({
            "variable": name,
            "availability": "available",
            "source_checked": f"{workbook} column {column!r}",
            "observation_unit": "per_subject",
            "n_values": int(values.notna().sum()),
            "reason": (
                f"static covariate read from the workbook as {column!r} for {n_subjects} subjects; "
                "a static covariate is NOT a heart-rate adjustment"
            ),
        })
    return rows


def hr_summary(availability_rows) -> dict:
    """The not-estimable record itself, derived from the inventory rows rather than restated."""
    hr = next(r for r in availability_rows if r["variable"] == "heart_rate")
    return {
        "status": HR_STATUS,
        "n_hr_observations": int(hr["n_values"]),
        "hr_availability": hr["availability"],
        "evidence": hr["reason"],
        "uncontrolled_variables": sorted(
            r["variable"] for r in availability_rows if r["availability"] == "missing"
        ),
        "note": (
            "The static-covariate analysis below is NOT an HR adjustment and must never be "
            "reported as one. Temperature and glucose remain uncontrolled."
        ),
    }


# ------------------------------------------------------------------------- the F data spine


def build_sessions_f(config, band) -> list[dict]:
    """Exp A's S0-S4 eligible spine plus the signed kg target the `kg_full` variant needs.

    S0 is KEPT here, unlike Exp B: F's question is about clock and static covariates over the
    whole day, and the session-index one-hot is exactly what makes S0 informative rather than a
    free row. `delta_m_kg = mass_kg - baseline_mass_kg` comes from the workbook, so loss stays
    negative and no sign convention is re-derived here.
    """
    from ..data.ground_truth import load_ground_truth

    gt = load_ground_truth(config.paths.weight_xlsx)
    kg = {(int(r.subject), int(r.session_idx)): float(r.delta_m_kg) for r in gt.sessions.itertuples()}
    sessions = []
    for record in exp_a.build_sessions(config, band):
        key = (int(record["subject"]), int(record["session_idx"]))
        if key not in kg:
            raise ExpFError(f"no workbook mass record for subject {key[0]} session {key[1]}")
        sessions.append(dict(record, delta_m_kg=kg[key]))
    return sessions


def evaluable_subjects_f(sessions) -> list[int]:
    return sorted({int(s["subject"]) for s in sessions})


def covariate_block(gt, subjects_per_row, names) -> np.ndarray:
    """[n_rows x len(names)] static covariates, in the frozen `names` order.

    Fails closed with the offending subject AND column named. §5.3 is explicit that there is no
    silent complete-case drop: a subject missing a covariate is a data problem to be fixed, not
    a row to quietly delete, because dropping it would change the cohort between variants and
    make the contrasts non-paired.
    """
    table = gt.subjects.set_index("subject")
    columns = []
    for name in names:
        column = COVARIATE_COLUMN_MAP[name]
        values = []
        for subject in subjects_per_row:
            if int(subject) not in table.index:
                raise ExpFError(f"subject {int(subject)} has no workbook covariate record")
            value = table.loc[int(subject), column]
            if value is None or not np.isfinite(float(value)):
                raise ExpFError(
                    f"subject {int(subject)} has a missing/non-finite {column!r} ({value!r}) — "
                    "Exp F does not complete-case drop"
                )
            values.append(float(value))
        columns.append(values)
    return np.array(columns, dtype=float).T


def clock_one_hot(session_idx) -> np.ndarray:
    """Session-index one-hot over the fixed S0-S4 domain (`ExpFConfig.clock_encoding`)."""
    session_idx = np.asarray(session_idx, dtype=int)
    unknown = sorted(set(session_idx.tolist()) - set(CLOCK_SESSIONS))
    if unknown:
        raise ExpFError(f"session indices {unknown} are outside the frozen clock domain {CLOCK_SESSIONS}")
    return np.stack([(session_idx == s).astype(float) for s in CLOCK_SESSIONS], axis=1)


# --------------------------------------------------------------- the approved Exp-A source


@dataclass(frozen=True)
class ExpASource:
    """The approved final Exp-A run F reads its per-fold feature keys from."""

    run_path: str
    selection_sha256: str
    feature_key_by_subject: dict


def load_exp_a_source(sources_path, band) -> ExpASource:
    """Read the per-fold selected feature keys from an APPROVED final Exp-A run.

    `load_approved_sources` refuses anything the reference gate left `not_approved`, so the
    refusal lives in one place and F cannot be handed an unapproved table by any call path.
    The selection table is then re-hashed on read and checked against the hash the gate
    recorded: the approval is evidence about specific bytes, and this is what ties the file F
    actually parsed to those bytes.
    """
    record = reference_gate.load_approved_sources(sources_path, band)
    artifact = record["final_selection_table"]
    path = Path(artifact["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if digest is None:
        raise ExpFError(f"approved Exp-A selection table is missing on disk: {path}")
    if digest != artifact["sha256"]:
        raise ExpFError(
            f"{path} hashes to {digest} but the reference gate approved {artifact['sha256']} — "
            "the approved selection table changed after approval"
        )
    folds = reference_gate.read_selection_table(record["final_run"]["path"], band)
    return ExpASource(
        run_path=str(record["final_run"]["path"]),
        selection_sha256=artifact["sha256"],
        feature_key_by_subject={int(f.test_subject): tuple(f.feature_key) for f in folds},
    )


def radar_candidate(config, band, feature_key) -> Candidate:
    """A Candidate carrying one Exp-A-selected feature key, for the store-backed X path only.

    F refits ridge with its own fold-local alpha, so the Exp-A family/params are deliberately
    NOT carried over — §2.2: models 3/4 reuse the selected feature key, never an Exp-A fitted
    estimator. The `active` record is rebuilt through Exp A's own helpers so the protocol guard
    sees exactly the shape it sees everywhere else.
    """
    feature_key = tuple(feature_key)
    if band == "10ghz":
        gi, r, c, ti, branch = feature_key
        gate = config.search_10ghz.range_gate_m[gi]
        active = exp_a._active_10(gi, r, c, ti, branch, "ridge", gate)
    else:
        ti, branch = feature_key
        active = exp_a._active_77(ti, branch, "ridge", config)
    return Candidate(
        candidate_id=f"exp_f_radar_{'_'.join(str(v) for v in feature_key)}",
        family="ridge",
        model_params=(),
        feature_key=feature_key,
        active=active,
    )


# ---------------------------------------------------------------- fold-local alpha selection


@dataclass(frozen=True)
class AlphaChoice:
    alpha: float
    inner_score: float
    inner_score_variance: float
    n_inner_folds: int


def select_alpha(X, y, subjects, fold, alphas, *, seed) -> AlphaChoice:
    """Choose one model's ridge alpha inside the outer-training subjects (§2.2).

    Every inner fold fits a FRESH `StandardScaler`+ridge pipeline on that inner fold's training
    rows only, and scores subject-balanced MAE on its validation rows. Each alpha becomes a
    `CandidateScore` with an identical simplicity rank and this model's fixed feature dimension,
    so `select_candidate`'s frozen key reduces to (mean MAE, then inner-fold variance) — and a
    FULL tie falls through to the first alpha in the frozen ordered tuple, because
    `select_candidate` is a stable `min` over input order. That is why the alphas are iterated
    in `ExpFConfig.ridge_alphas` order rather than sorted here.
    """
    subjects = np.asarray(subjects)
    scores = []
    for alpha in alphas:
        per_fold = []
        for inner in fold.inner_folds:
            train = np.isin(subjects, sorted(inner.train_subjects))
            val = np.isin(subjects, sorted(inner.val_subjects))
            if not train.any() or not val.any():
                continue
            pipeline = build_estimator("ridge", {"alpha": float(alpha)}, seed=seed)
            fit_pipeline(pipeline, X[train], y[train])
            per_fold.append(
                M.subject_balanced_mae(subjects[val], y[val], pipeline.predict(X[val]))
            )
        values = np.array(per_fold, dtype=float)
        scores.append(CandidateScore(
            candidate_id=f"alpha={float(alpha)!r}",
            inner_val_mae=float(np.mean(values)) if values.size else float("nan"),
            simplicity_rank=SIMPLICITY_RANK["ridge"],      # identical for every alpha
            feature_dimension=int(X.shape[1]),             # identical for every alpha
            inner_fold_variance=float(np.std(values, ddof=0)) if values.size else float("nan"),
        ))
    winner = select_candidate(scores)
    index = [s.candidate_id for s in scores].index(winner.candidate_id)
    return AlphaChoice(
        alpha=float(alphas[index]),
        inner_score=winner.inner_val_mae,
        inner_score_variance=winner.inner_fold_variance,
        n_inner_folds=len(fold.inner_folds),
    )


# ------------------------------------------------------------------------- the fold compute


@dataclass
class ExpFFoldResult:
    test_subject: int
    prediction_rows: list = field(default_factory=list)
    selection_rows: list = field(default_factory=list)
    reason: str | None = None


@dataclass(frozen=True)
class DesignInputs:
    """The row-aligned, FOLD-INDEPENDENT half of every design matrix.

    Clock, covariates and both targets are pure per-row lookups: they do not depend on which
    subject is held out, so they are built once at run level and shipped to the workers as plain
    arrays. Only the radar block is fold-dependent (the tuned-ε branch reconstructs its matrix
    from the fold's training subjects), which is why it alone is built inside the worker.

    The practical consequence is that the workbook is read and fully validated ONCE per run
    rather than once per worker process — and that a test can drive the fold computation without
    an .xlsx at all.
    """

    subjects: np.ndarray
    session_idx: np.ndarray
    clock: np.ndarray
    covariates_full: np.ndarray
    covariates_reduced: np.ndarray
    targets: dict


def build_design_inputs(config, gt, sessions) -> DesignInputs:
    """Assemble the fold-independent design blocks, in the canonical session row order."""
    subjects = np.array([int(s["subject"]) for s in sessions])
    session_idx = np.array([int(s["session_idx"]) for s in sessions])
    return DesignInputs(
        subjects=subjects,
        session_idx=session_idx,
        clock=clock_one_hot(session_idx),
        covariates_full=covariate_block(gt, subjects, config.exp_f.covariates_primary),
        covariates_reduced=covariate_block(gt, subjects, config.exp_f.covariates_sensitivity),
        targets={
            "delta_m_pct": np.array([float(s["delta_m_pct"]) for s in sessions]),
            "delta_m_kg": np.array([float(s["delta_m_kg"]) for s in sessions]),
        },
    )


def _designs_for_variant(variant, clock, radar, covariates_full, covariates_reduced) -> dict:
    """The four model design matrices for one variant.

    `pct_reduced` differs from `pct_full` ONLY in the covariate block, which is why §2.2 lets it
    reuse `pct_full`'s models 1 and 3 byte-identically — they contain no covariate block at all.
    That reuse is done by the caller; this function just makes the difference visible.
    """
    covariates = covariates_reduced if variant == "pct_reduced" else covariates_full
    return {
        "clock": clock,
        "clock_covariates": np.hstack([clock, covariates]),
        "clock_radar": np.hstack([clock, radar]),
        "clock_radar_covariates": np.hstack([clock, radar, covariates]),
    }


def _run_single_fold_f(config, band, sessions, store_dir, fold, source, inputs,
                       seed_label) -> ExpFFoldResult:
    """One outer fold: three variants x four models, each with its own fold-local alpha.

    Top-level and picklable so it can run in a worker process; builds its own provider and pins
    single-threaded math, mirroring `exp_b._run_single_fold_b`.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        subjects, session_idx, targets = inputs.subjects, inputs.session_idx, inputs.targets
        clock = inputs.clock
        covariates_full, covariates_reduced = inputs.covariates_full, inputs.covariates_reduced
        train_rows = np.isin(subjects, sorted(fold.train_subjects))
        test_rows = subjects == fold.test_subject
        if not test_rows.any():
            return ExpFFoldResult(test_subject=fold.test_subject, reason="no_test_rows")

        # The radar block is built ONCE and shared by models 3 and 4 (§2.2: they share
        # byte-identical radar columns within a fold). Models 1 and 2 never reach this code, so
        # "models 1/2 never read the store" is structural rather than a promise.
        feature_key = source.feature_key_by_subject.get(int(fold.test_subject))
        if feature_key is None:
            raise ExpFError(
                f"the approved Exp-A selection table has no row for outer fold "
                f"{fold.test_subject} — F cannot invent a feature key"
            )
        candidate = radar_candidate(config, band, feature_key)
        active = dict(candidate.active)
        require_complete_active(active)
        protocol_freeze_guard(config, active=active)
        provider = exp_a.StoreBackedFeatures(band, sessions, store_dir, config)
        radar = provider.data_for(candidate, fold.train_subjects).X

        prediction_rows, selection_rows = [], []
        reusable = {}
        for variant in VARIANTS:
            target_name = VARIANT_TARGET[variant]
            y = targets[target_name]
            designs = _designs_for_variant(variant, clock, radar, covariates_full, covariates_reduced)
            for model_id in MODEL_IDS:
                # pct_reduced reuses pct_full's covariate-free models VERBATIM rather than
                # refitting an identical model: identical rows, identical design, identical
                # alpha search. Recomputing could only differ by floating-point accident.
                if variant == "pct_reduced" and model_id not in COVARIATE_MODELS:
                    for row in reusable[("pct_full", model_id)]:
                        prediction_rows.append(dict(row, analysis_variant=variant))
                    for row in reusable[("pct_full", model_id, "selection")]:
                        selection_rows.append(dict(row, analysis_variant=variant))
                    continue

                X = designs[model_id]
                choice = select_alpha(X[train_rows], y[train_rows], subjects[train_rows], fold,
                                      config.exp_f.ridge_alphas, seed=seed_label)
                final = build_estimator("ridge", {"alpha": choice.alpha}, seed=seed_label)
                fit_pipeline(final, X[train_rows], y[train_rows])
                predictions = final.predict(X[test_rows])

                model_predictions = [{
                    "band": band,
                    "outer_fold": int(fold.test_subject),
                    "subject": int(subject),
                    "session_idx": int(index),
                    "model_id": model_id,
                    "analysis_variant": variant,
                    "target_name": target_name,
                    "seed": int(seed_label),
                    "y_true": float(true),
                    "y_pred": float(pred),
                } for subject, index, true, pred in zip(
                    subjects[test_rows], session_idx[test_rows], y[test_rows], predictions)]
                model_selection = [{
                    "outer_fold": int(fold.test_subject),
                    "test_subject": int(fold.test_subject),
                    "source_exp_a_final_run": source.run_path,
                    "source_selection_sha256": source.selection_sha256,
                    "feature_key": repr(tuple(feature_key)) if model_id in RADAR_MODELS else "",
                    "model_id": model_id,
                    "analysis_variant": variant,
                    "selected_alpha": choice.alpha,
                    "inner_score": choice.inner_score,
                    "inner_score_variance": choice.inner_score_variance,
                    "n_inner_folds": choice.n_inner_folds,
                }]
                prediction_rows.extend(model_predictions)
                selection_rows.extend(model_selection)
                if variant == "pct_full":
                    reusable[("pct_full", model_id)] = model_predictions
                    reusable[("pct_full", model_id, "selection")] = model_selection

        return ExpFFoldResult(
            test_subject=fold.test_subject,
            prediction_rows=prediction_rows,
            selection_rows=selection_rows,
        )


def run_exp_f(config, band, sessions, store_dir, source, inputs, *, n_workers=1):
    """Every selectable outer fold's four-model, three-variant computation."""
    if not sessions:
        raise ExpFError("Exp F got an empty session spine")
    subjects = evaluable_subjects_f(sessions)
    folds = [f for f in nested_loso_splits(subjects) if f.selectable]
    seed_label = int(config.run.seed_set[0])   # ridge is deterministic: one seed label, once
    tasks = [(config, band, sessions, store_dir, fold, source, inputs, seed_label)
             for fold in folds]
    results = fold_parallel.run_folds_parallel(
        _run_single_fold_f, tasks, n_workers, f"exp-f-{band}",
    )
    results.sort(key=lambda r: r.test_subject)
    return results


# ------------------------------------------------------------------------------ contrasts


def per_subject_mae(prediction_rows) -> dict:
    """{(variant, model_id, subject): mean session |error|} plus each cell's session count."""
    grouped: dict = {}
    for row in prediction_rows:
        key = (row["analysis_variant"], row["model_id"], int(row["subject"]))
        grouped.setdefault(key, []).append(abs(float(row["y_true"]) - float(row["y_pred"])))
    return {key: (float(np.mean(values)), len(values)) for key, values in grouped.items()}


def contrast_rows(prediction_rows) -> list[dict]:
    """One row per (subject, contrast, variant): the paired with/without MAEs and difference.

    Negative means the added component improved that subject's prediction. Only subjects with
    BOTH models present contribute, which keeps every contrast genuinely paired.
    """
    maes = per_subject_mae(prediction_rows)
    rows = []
    for variant in VARIANTS:
        for contrast_id, with_model, without_model in CONTRASTS:
            subjects = sorted({
                subject for (v, _, subject) in maes if v == variant
            })
            for subject in subjects:
                with_key = (variant, with_model, subject)
                without_key = (variant, without_model, subject)
                if with_key not in maes or without_key not in maes:
                    continue
                mae_with, n_sessions = maes[with_key]
                mae_without, _ = maes[without_key]
                rows.append({
                    "subject": subject,
                    "contrast_id": contrast_id,
                    "analysis_variant": variant,
                    "target_name": VARIANT_TARGET[variant],
                    "n_sessions": n_sessions,
                    "mae_with": mae_with,
                    "mae_without": mae_without,
                    "difference_with_minus_without": mae_with - mae_without,
                })
    return rows


def _contrast_summary(rows, variant, contrast_id, config) -> dict:
    """Mean difference with subject-cluster CI, median, Wilcoxon, N, nonzero N and ties."""
    values = np.array([
        r["difference_with_minus_without"] for r in rows
        if r["analysis_variant"] == variant and r["contrast_id"] == contrast_id
    ], dtype=float)
    if variant == PRIMARY_VARIANT and contrast_id in PRIMARY_CONTRASTS:
        family = FAMILY_PRIMARY
    elif variant == PRIMARY_VARIANT:
        family = config.stats.expf_exploratory_correction
    else:
        family = FAMILY_SENSITIVITY

    summary = {
        "contrast_id": contrast_id,
        "analysis_variant": variant,
        "target_name": VARIANT_TARGET[variant],
        "multiplicity_family": family,
        "n_paired_subjects": int(values.size),
        "n_nonzero_pairs": int(np.count_nonzero(values)),
        "n_ties": int(values.size - np.count_nonzero(values)),
        "mean_difference": float(np.mean(values)) if values.size else float("nan"),
        "median_difference": float(np.median(values)) if values.size else float("nan"),
    }
    if values.size:
        ci = M.mean_difference_ci(
            values, b=config.stats.bootstrap_b, level=config.stats.confidence_level,
            rng_seed=int(config.run.seed) + _rng_offset(variant, contrast_id),
            method=config.stats.ci_method,
        )
        statistic, p_value = M.wilcoxon_signed_rank(values)
        summary.update({
            "ci_low": ci.low, "ci_high": ci.high, "ci_method": ci.method,
            "wilcoxon_statistic": statistic, "p_value_unadjusted": p_value,
        })
    else:
        summary.update({"ci_low": float("nan"), "ci_high": float("nan"), "ci_method": "",
                        "wilcoxon_statistic": float("nan"), "p_value_unadjusted": float("nan")})
    return summary


def summarize_exp_f(results, availability_rows, source, config, band, sessions) -> dict:
    """The run-level record: the not-estimable HR answer FIRST, then the available analysis.

    The ordering is not cosmetic. A reader who takes the covariate models for an HR adjustment
    has misread the entire experiment, so the artifact states what is missing before it states
    what was measured.
    """
    prediction_rows = [row for r in results for row in r.prediction_rows]
    rows = contrast_rows(prediction_rows)

    summaries = [
        _contrast_summary(rows, variant, contrast_id, config)
        for variant in VARIANTS for contrast_id, _, _ in CONTRASTS
    ]
    # Holm-2 over the two primary radar increments in pct_full ALONE. The family size is pinned
    # to 2 so a contrast missing from a given run cannot silently weaken the correction, and no
    # sensitivity contrast is ever admitted into it.
    primary = [s for s in summaries
               if s["analysis_variant"] == PRIMARY_VARIANT and s["contrast_id"] in PRIMARY_CONTRASTS]
    primary.sort(key=lambda s: PRIMARY_CONTRASTS.index(s["contrast_id"]))
    adjusted = M.holm_adjusted([s["p_value_unadjusted"] for s in primary],
                               family_size=config.stats.holm_family_expf_primary)
    for summary, value in zip(primary, adjusted):
        summary["p_value_holm"] = float(value)
    for summary in summaries:
        summary.setdefault("p_value_holm", float("nan"))

    return {
        "stage": "exp-f",
        "band": band,
        "heart_rate_question": hr_summary(availability_rows),
        "available_analysis": {
            "status": "clock_and_static_covariate_sensitivity_only",
            "models": {model_id: MODEL_NUMBER[model_id] for model_id in MODEL_IDS},
            "variants": list(VARIANTS),
            "covariates_primary": list(config.exp_f.covariates_primary),
            "covariates_sensitivity": list(config.exp_f.covariates_sensitivity),
            "covariate_column_map": dict(COVARIATE_COLUMN_MAP),
            "clock_encoding": config.exp_f.clock_encoding,
            "ridge_alphas": list(config.exp_f.ridge_alphas),
            "primary_family": FAMILY_PRIMARY,
            "primary_family_size": config.stats.holm_family_expf_primary,
            "exploratory_correction": config.stats.expf_exploratory_correction,
            "primary_contrasts": list(PRIMARY_CONTRASTS),
            "exploratory_contrasts": list(EXPLORATORY_CONTRASTS),
            "direction": "difference = with component - without component; negative favours the component",
            "limitation": (
                "A limited clock/static-covariate sensitivity result. It does not address heart "
                "rate, and temperature and glucose remain uncontrolled."
            ),
        },
        "exp_a_source": {
            "final_run": source.run_path,
            "selection_table_sha256": source.selection_sha256,
            "n_folds_with_feature_key": len(source.feature_key_by_subject),
            "gate": "reference_gate.load_approved_sources (approved bands only)",
        },
        "contrasts": summaries,
        "n_subjects_f": len(evaluable_subjects_f(sessions)),
        "n_sessions": len(sessions),
        "n_outer_folds": len(results),
        "n_evaluable_outer_folds": sum(1 for r in results if r.reason is None),
        "exclusion_reasons": sorted({r.reason for r in results if r.reason is not None}),
        "config_sha256": _config_sha256(config),
    }


def _config_sha256(config) -> str:
    from .exp_b import config_fingerprint

    return config_fingerprint(config)


# ------------------------------------------------------------------------------ artifacts


PREDICTIONS_COLUMNS = (
    "band", "outer_fold", "subject", "session_idx", "model_id", "analysis_variant",
    "target_name", "seed", "y_true", "y_pred",
)
SELECTION_COLUMNS = (
    "outer_fold", "test_subject", "source_exp_a_final_run", "source_selection_sha256",
    "feature_key", "model_id", "analysis_variant", "selected_alpha", "inner_score",
    "inner_score_variance", "n_inner_folds",
)
CONTRASTS_COLUMNS = (
    "subject", "contrast_id", "analysis_variant", "target_name", "n_sessions", "mae_with",
    "mae_without", "difference_with_minus_without",
)
EXCLUSIONS_COLUMNS = ("band", "outer_fold", "test_subject", "reason")


def _write_csv(path, columns, rows) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def exclusion_rows(results, band) -> list[dict]:
    return [{"band": band, "outer_fold": int(r.test_subject), "test_subject": int(r.test_subject),
             "reason": r.reason}
            for r in results if r.reason is not None]


def write_exp_f_reports(results, availability_rows, summary, out_dir, band) -> dict:
    """The five §3 F artifact rows. Returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = [row for r in results for row in r.prediction_rows]
    paths = {
        "availability": _write_csv(
            out_dir / "confound_availability.csv", AVAILABILITY_COLUMNS, availability_rows),
        "predictions": _write_csv(
            out_dir / f"predictions_f_{band}.csv", PREDICTIONS_COLUMNS, prediction_rows),
        "selection": _write_csv(
            out_dir / f"selection_f_{band}.csv", SELECTION_COLUMNS,
            [row for r in results for row in r.selection_rows]),
        "contrasts": _write_csv(
            out_dir / f"contrasts_f_{band}.csv", CONTRASTS_COLUMNS, contrast_rows(prediction_rows)),
        "exclusions": _write_csv(
            out_dir / f"exclusions_f_{band}.csv", EXCLUSIONS_COLUMNS, exclusion_rows(results, band)),
    }
    metrics_path = out_dir / f"metrics_exp_f_{band}.json"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics"] = metrics_path
    return paths


def _assert_mechanism_ok_f(results, sessions) -> None:
    """Structural checks that reveal no performance value: every reported fold is an outer LOSO
    fold holding out one subject, and every prediction row belongs to that held-out subject."""
    subjects = evaluable_subjects_f(sessions)
    folds = {f.test_subject: f for f in nested_loso_splits(subjects)}
    for result in results:
        assert result.test_subject in folds, result.test_subject
        assert result.test_subject not in folds[result.test_subject].train_subjects
        for row in result.prediction_rows:
            assert int(row["subject"]) == result.test_subject, row


def run_and_report_f(config, band, sessions, store_dir, run_dir, *, mode, analysis_commit,
                     exp_a_sources, n_workers=1) -> dict:
    """validate_store -> approved-source gate -> run_exp_f -> smoke run-log or full reporting.

    The HR inventory runs in BOTH modes and is written in both: it is not a performance value,
    it is the answer to the registered question, and a smoke that hid it would be hiding the
    one result F can state with certainty.
    """
    from ..data.ground_truth import load_ground_truth

    store_mod.validate_store(
        band, store_dir, exp_a.expected_fingerprints(config, band, sessions),
        analysis_commit=analysis_commit,
    )
    source = load_exp_a_source(exp_a_sources, band)
    gt = load_ground_truth(config.paths.weight_xlsx)      # read and validated ONCE per run
    availability_rows = hr_inventory(config, gt)
    inputs = build_design_inputs(config, gt, sessions)

    results = run_exp_f(config, band, sessions, store_dir, source, inputs, n_workers=n_workers)
    _assert_mechanism_ok_f(results, sessions)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        _write_csv(run_dir / "confound_availability.csv", AVAILABILITY_COLUMNS, availability_rows)
        log = run_dir / f"run_log_exp_f_{band}.json"
        log.write_text(json.dumps({
            "stage": "exp-f-smoke", "band": band, "mode": "mechanism-only",
            "heart_rate_question": hr_summary(availability_rows),
            "n_sessions": len(sessions),
            "n_outer_folds": len(results),
            "n_evaluable_outer_folds": sum(1 for r in results if r.reason is None),
            "n_prediction_rows": sum(len(r.prediction_rows) for r in results),
            "exp_a_source": source.run_path,
            "note": "contrast values suppressed -- mechanism-only smoke",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"availability": run_dir / "confound_availability.csv", "run_log": log}

    if not any(r.reason is None for r in results):
        raise ExpFError("no outer fold produced a prediction — Experiment F has no contrast to report")
    summary = summarize_exp_f(results, availability_rows, source, config, band, sessions)
    return write_exp_f_reports(results, availability_rows, summary, run_dir, band)
