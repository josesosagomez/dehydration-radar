"""Experiment E — leave-one-path-group-out (LOPGO) attribution under outer LOSO.

The question, exactly: for the FIXED, pre-registered Exp-E ridge model on Exp-B residual
targets, how much does held-out-subject residual MAE change when one WST path group is
unavailable and the whole fold-local pipeline is refit?

**A-M10-1 replaced the frozen standalone 4-fold permutation CV with this design.** The M6
record `ExpEConfig.n_folds = 4` / `fold_assignment` therefore describes a protocol that is no
longer run: it is DEAD CONFIG, deliberately left in place because the M6 sections are frozen
records and deleting one would rewrite history. Nothing in this module reads either field. The
folds come from `splits.nested_loso_splits` like every other reported result in the project,
because a 4-fold subject-grouped CV is not LOSO and is undefined for the incomplete validation
trajectories this cohort actually has.

What is fixed, and why that matters more than usual: the model form is the pre-registered
anchor (10 GHz gate (1,2) m / reduction A / magnitude / T1 / log off; 77 GHz T1_77 / I/Q /
mean-Rx fusion / log off; ridge alpha = 1.0), NOT the best model from the Exp A/B outer
results. Attribution is only comparable across folds and paths if every fold interprets the
same model, and a performance-selected model would make each fold's importances describe a
different estimator. This module never selects anything: there is no inner CV, no candidate
enumeration, no tie-break, and no p-value.

The measurement, per outer fold and per canonical Kymatio `path_id`:

    importance_delta_mae_pct_points = ablated_mae - full_mae

in residual Δm% points, where the ablated model has that path group's COMPLETE column block
(both frame aggregates, every channel, all global/half x mean/std columns) deleted BEFORE
scaling, then a fresh StandardScaler and ridge refit on the identical training rows and scored
on the identical held-out rows. Positive means the full model predicted that subject better
with the group.

Two things this quantity is not, stated here because they are easy to slide into (A-M10-6):
it is model reliance / predictive contribution, not a causal claim about tissue dielectrics;
and the sign is a measurement, never an acceptance criterion. Nothing here pre-labels a path
"null" or "physical", and the run does not fail because the table came out one way.
"""

from __future__ import annotations

import csv
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import SPEED_OF_LIGHT_M_S
from ..data.loader_77ghz import N_CHIRPS as N_SLOW_TIME_77
from ..features import store as store_mod
from ..features.extraction_77 import PRIMARY_FUSION_77, prf_hz
from ..features.pooling import session_feature_layout
from ..features.protocol_freeze import protocol_freeze_guard
from ..features.wst import build_scattering, preprocessed_length, scattering_shape
from ..models.regressors import SEED_SENSITIVE, build_estimator, fit_pipeline
from . import exp_a, exp_b, fold_parallel
from . import metrics as M
from .harness import Candidate, require_complete_active
from .splits import nested_loso_splits


class ExpEError(ValueError):
    """A metadata gate failure (reconstructed filter bank vs the stored `order`), a model
    layout that does not describe the stored feature matrix, or a malformed E spine."""


# ------------------------------------------------------------------ the fixed model form


def _tiling_index(config, band) -> int:
    """The Exp-E tiling NAME (`T1` / `T1_77`) -> the store's tiling INDEX.

    The store keys arrays by index (`order__t{ti}`, `vec__...__t{ti}__off`) while the frozen
    Exp-E record names the tiling, so the mapping is made once, here, and fails closed rather
    than defaulting to 0 — a silent fallback would attribute one tiling's importances to
    another tiling's filter bank.
    """
    labels = exp_a.TILING_LABELS_10 if band == "10ghz" else exp_a.TILING_LABELS_77
    name = config.exp_e.tiling_10ghz if band == "10ghz" else config.exp_e.tiling_77ghz
    if name not in labels:
        raise ExpEError(f"Exp E tiling {name!r} is not one of {labels} for band {band}")
    return labels.index(name)


def _gate_index_10(config) -> int:
    """The store's gate index for the frozen Exp-E gate. Fails closed if the search config no
    longer carries that gate at all, rather than silently interpreting a different gate."""
    gate = tuple(float(v) for v in config.exp_e.gate_10ghz_m)
    gates = [tuple(float(v) for v in g) for g in config.search_10ghz.range_gate_m]
    if gate not in gates:
        raise ExpEError(
            f"Exp E's frozen gate {gate} m is not in the 10 GHz search gates {gates} — the "
            "store holds no column block for it"
        )
    return gates.index(gate)


def fixed_candidate(config, band) -> Candidate:
    """THE one candidate Exp E ever fits: the pre-registered feature/model anchor.

    Built as a `Candidate` (rather than a bare feature key) so the existing store-backed
    provider and the existing protocol-freeze guard consume it unchanged — E gets the same
    fail-closed `active` record every Exp A/B fit carries, with no second code path.
    """
    if config.exp_e.model != "ridge":
        raise ExpEError(f"Exp E is frozen at ridge, got model {config.exp_e.model!r}")
    ti = _tiling_index(config, band)
    if band == "10ghz":
        gi = _gate_index_10(config)
        r = config.exp_e.reduction_10ghz
        c = config.exp_e.channel_10ghz
        branch = _log_branch(config.exp_e.log_10ghz)
        return Candidate(
            candidate_id=f"exp_e_g{gi}_{r}_{c}_t{ti}_{branch}",
            family="ridge",
            model_params=(("alpha", float(config.exp_e.ridge_alpha)),),
            feature_key=(gi, r, c, ti, branch),
            active=exp_a._active_10(gi, r, c, ti, branch, "ridge", config.exp_e.gate_10ghz_m),
        )
    branch = _log_branch(config.exp_e.log_77ghz)
    return Candidate(
        candidate_id=f"exp_e_t{ti}_{branch}",
        family="ridge",
        model_params=(("alpha", float(config.exp_e.ridge_alpha)),),
        feature_key=(ti, branch),
        active=exp_a._active_77(ti, branch, "ridge", config),
    )


def _log_branch(log_value: str) -> str:
    """Exp E's `log_*` record ("off") -> the store's branch name. The tuned branch is refused
    outright: it is a train-only FITTED epsilon, so it would make the feature matrix itself
    fold-dependent and the path columns no longer comparable across folds."""
    branch = {v: k for k, v in exp_a._BRANCH_TO_LOG.items()}.get(log_value)
    if branch is None:
        raise ExpEError(f"Exp E log branch {log_value!r} is not one of {tuple(exp_a._BRANCH_TO_LOG.values())}")
    if branch == "tuned":
        raise ExpEError(
            "Exp E is frozen at a data-independent log branch; a tuned-ε matrix is refit per "
            "fold, so its columns would not be comparable across folds"
        )
    return branch


def n_channels(config, band) -> int:
    """Channels in the stored per-frame tensor for the Exp-E anchor: 1 for 10 GHz magnitude,
    2 for an I/Q pair (10 GHz `iq`, and 77 GHz, whose real/imag slow-time channels are
    scattered separately before the mean-Rx fusion)."""
    if band == "10ghz":
        return 1 if config.exp_e.channel_10ghz == "mag" else 2
    if config.search_77ghz.channel != "iq":
        raise ExpEError(f"77 GHz Exp E expects the frozen iq channel pair, got {config.search_77ghz.channel!r}")
    return 2


# ------------------------------------------------ filter-bank reconstruction + the gate


def reconstruct_bank(config, band) -> dict:
    """Rebuild the pinned Kymatio bank for the Exp-E tiling and return its measured geometry.

    Exp E is the ONLY experiment that needs `xi`/`sigma`/`j`: the session stores persist the
    per-path `order` array and nothing else about the bank, because no other stage ever asks
    what a path IS. Rebuilding is safe precisely because the bank is a deterministic function
    of the resolved config (tiling Q + invariance ms, input length, sampling rate) and carries
    no data — but "safe" is not "verified", which is what `assert_bank_matches_store` is for.

    The 10 GHz bank is over the trimmed fast-time chirp at fs; the 77 GHz bank is over the
    slow-time chirp axis at the PRF. Both lengths/rates come from the same helpers the feature
    extraction itself used, never re-derived here.
    """
    if band == "10ghz":
        tiling = config.wst.tilings[_tiling_index(config, band)]
        wst_cfg, n_in, fs_hz = config.wst, preprocessed_length(config.preprocess), config.preprocess.fs_hz
    else:
        tiling = config.wst77.tilings[_tiling_index(config, band)]
        wst_cfg, n_in, fs_hz = config.wst77, N_SLOW_TIME_77, prf_hz(config.preprocess77)
    with warnings.catch_warnings():
        # The border-effect warning is asserted at its source (T-W3); this rebuild is a
        # metadata read, not a second place to re-assert it.
        warnings.simplefilter("ignore")
        scattering = build_scattering(tiling, wst_cfg, n_in=n_in, fs_hz=fs_hz)
    bank = scattering_shape(scattering)
    bank["fs_hz"] = float(fs_hz)
    bank["n_in"] = int(n_in)
    bank["q"] = tuple(int(v) for v in tiling.q)
    bank["invariance_ms"] = float(tiling.invariance_ms)
    return bank


def assert_bank_matches_store(config, band, sessions, store_dir, bank) -> None:
    """FAIL-CLOSED metadata gate (§1.2): the reconstructed `order` must equal the stored
    `order__{tiling}` array for EVERY consumed session, and its path count must match.

    This is not a warning and there is no partial mode. If it fails, the reconstructed bank
    is not the bank the stored features were produced with, so every physics label and every
    path-group column block would be attributed to the wrong path — silently, and in a way no
    downstream number would reveal. A run that cannot prove the correspondence must not
    produce an attribution table at all.
    """
    key = store_mod.order_key(_tiling_index(config, band))
    expected = np.asarray(bank["order"])
    for s in sessions:
        store = store_mod.read_session_store(band, s["subject"], s["session_name"], store_dir)
        try:
            if key not in store:
                raise ExpEError(
                    f"session (subject {s['subject']}, {s['session_name']}) has no {key!r} — "
                    "the store predates the per-tiling order metadata"
                )
            stored = np.asarray(store[key])
        finally:
            store.close()
        if stored.shape != expected.shape or not np.array_equal(stored, expected):
            raise ExpEError(
                f"reconstructed filter bank does not match the store for subject "
                f"{s['subject']} {s['session_name']}: stored {key} has shape {stored.shape}, "
                f"rebuilt bank has shape {expected.shape} — the physics labels and the path "
                "column blocks would describe a different bank than the features do"
            )


def stored_column_count(config, band, session, store_dir) -> int:
    """The width of the stored session vector for the Exp-E anchor, read from one session.

    Read straight off the store rather than by building a provider over every subject: the
    layout gate only needs the column COUNT, and asking a provider for it would mean calling
    `data_for` with a training set that is not any fold's — harmless for this data-independent
    branch, but exactly the shape of thing a reader should not have to reason about.
    """
    fk = fixed_candidate(config, band).feature_key
    key = store_mod.vec_key(*fk) if band == "10ghz" else store_mod.vec77_key(*fk)
    store = store_mod.read_session_store(band, session["subject"], session["session_name"], store_dir)
    try:
        if key not in store:
            raise ExpEError(f"session store has no {key!r} — the Exp-E anchor is not in this store")
        return int(np.asarray(store[key]).shape[0])
    finally:
        store.close()


def column_path_index(config, band, bank, n_model_columns: int) -> np.ndarray:
    """One canonical `path_id` per model column, in the frozen session-vector order.

    A path GROUP is every column sharing one `path_id`: both frame aggregates, every channel,
    and all global/half x mean/std columns — i.e. exactly the columns this array marks with
    the same value. Built from `session_feature_layout`, the same layout M6 and the store
    itself use, so the grouping cannot drift from the pooling that produced the columns.

    Fails closed when the layout does not describe the stored matrix, which is the other half
    of the §1.2 gate: the bank can be right about paths and still be paired with the wrong
    channel count or output length.
    """
    layout = session_feature_layout(bank, bank["n_time"], n_channels(config, band), family="pooled")
    if len(layout) != n_model_columns:
        raise ExpEError(
            f"model layout has {len(layout)} columns but the stored feature matrix has "
            f"{n_model_columns} — the reconstructed bank (n_paths={bank['n_paths']}, "
            f"n_time={bank['n_time']}, n_channels={n_channels(config, band)}) does not "
            "describe these features"
        )
    # layout element = (frame_aggregate, channel, path_id, segment, statistic)
    return np.array([element[2] for element in layout], dtype=np.int64)


# ------------------------------------------------------------------ band-aware physics


def coarse_range_m(beat_hz: float, config) -> float:
    """FMCW beat frequency -> coarse SCENE range. The inverse of `config.beat_band_hz`, which
    is the forward form the QC mask and the config cross-validation share (a round-trip test
    pins the two against each other, so the physics lives in one place).

    r = c * f_b / (2 * slope), slope = bandwidth / chirp duration. Scene distance — NOT tissue
    penetration depth, and not a claim about where in the body a return originated.
    """
    hz_per_m = 2.0 * (config.preprocess.bandwidth_hz / config.preprocess.chirp_time_s) / SPEED_OF_LIGHT_M_S
    return float(beat_hz) / hz_per_m


ORDER0_LIMIT = (
    "no frequency, range or reflected-level claim: per-signal median/MAD standardization "
    "removed absolute level before the transform"
)
LIMIT_10 = "coarse scene range under the FMCW beat mapping; scene distance, not tissue penetration depth"
LIMIT_10_ORDER2 = (
    "xi_1 keeps the coarse beat/range reading; xi_2 is an envelope-modulation centre, not a "
    "second range"
)


def _gate_band_note(xi1_hz: float, config) -> str:
    """Whether a 10 GHz path's beat centre falls inside the analysis gate's beat band.

    Worth saying, and easy to omit: the frozen T1 bank tiles the whole 0-260 kHz fast-time
    axis, so most order-1 paths sit at beat centres of tens of kHz — coarse scene ranges of
    tens of metres, far outside the 1-2 m gate the model was actually fed. The band-pass gate
    suppressed those bands before the transform, so a reader who sees "coarse scene range
    71.6 m" needs to know it describes a filter the signal has little energy in, not a target
    at 72 m. Carried inside `claim_limit` rather than as an extra column, because §3 freezes
    this table's column list.
    """
    from ..config import beat_band_hz

    lo, hi = beat_band_hz(config.exp_e.gate_10ghz_m, config.preprocess.bandwidth_hz,
                          config.preprocess.chirp_time_s)
    gate = tuple(float(v) for v in config.exp_e.gate_10ghz_m)
    inside = "inside" if lo <= xi1_hz <= hi else "OUTSIDE"
    return (f"; xi_1 lies {inside} the {gate[0]}-{gate[1]} m analysis gate band "
            f"({lo / 1e3:.1f}-{hi / 1e3:.1f} kHz), which the range gate band-passed before the "
            "transform")
LIMIT_77 = (
    "slow-time Doppler/modulation-frequency MAGNITUDE, not range; no signed velocity is "
    "reported because the I/Q channels are scattered separately"
)
LIMIT_77_ORDER2 = "xi_2 is an envelope-modulation centre, not range and not a second velocity"


def _finite(value) -> float | str:
    """A finite float, or "" — the §3 rule that non-applicable numeric fields are BLANK rather
    than filled with a zero/NaN a reader could mistake for a measurement. Kymatio pads the
    unused order slots of `xi`/`sigma`/`j` with NaN, so this is where padding becomes blank."""
    value = float(value)
    return value if np.isfinite(value) else ""


def path_metadata_rows(config, band, bank) -> list[dict]:
    """One row per canonical path: the bank's normalized parameters, the band-aware physical
    reading, and the limit that reading carries.

    Kymatio 0.3.0 reports `xi` in cycles/sample, so a finite centre frequency is `xi * fs`;
    `j` is a dyadic subsampling index and is NEVER converted to Hz. Rows are emitted in
    canonical path order (`path_id` = the bank's row index), which is also the column order.
    """
    order = np.asarray(bank["order"])
    xi, sigma, j = np.asarray(bank["xi"]), np.asarray(bank["sigma"]), np.asarray(bank["j"])
    fs_hz = float(bank["fs_hz"])
    input_domain = "fast_time_beat" if band == "10ghz" else "slow_time_doppler"
    rows = []
    for p in range(int(bank["n_paths"])):
        o = int(order[p])
        xi1, xi2 = float(xi[p, 0]), float(xi[p, 1])
        xi1_hz = xi1 * fs_hz if np.isfinite(xi1) else float("nan")
        xi2_hz = xi2 * fs_hz if np.isfinite(xi2) else float("nan")
        range_m = float("nan")
        if o == 0:
            label = "order-0 averaged scaling coefficient (normalized low-frequency structure)"
            limit = ORDER0_LIMIT
        elif band == "10ghz":
            range_m = coarse_range_m(xi1_hz, config)
            if o == 1:
                label = f"fast-time beat centre {xi1_hz:.1f} Hz (coarse scene range {range_m:.2f} m)"
                limit = LIMIT_10 + _gate_band_note(xi1_hz, config)
            else:
                label = (
                    f"beat centre {xi1_hz:.1f} Hz (coarse scene range {range_m:.2f} m), envelope "
                    f"modulated at {xi2_hz:.1f} Hz"
                )
                limit = LIMIT_10_ORDER2 + _gate_band_note(xi1_hz, config)
        else:
            if o == 1:
                label = f"slow-time Doppler/modulation magnitude {xi1_hz:.2f} Hz"
                limit = LIMIT_77
            else:
                label = (
                    f"slow-time Doppler/modulation magnitude {xi1_hz:.2f} Hz, envelope modulated "
                    f"at {xi2_hz:.2f} Hz"
                )
                limit = LIMIT_77_ORDER2
        rows.append({
            "band": band,
            "input_domain": input_domain,
            "path_id": p,
            "scattering_order": o,
            "xi1_normalized": _finite(xi1),
            "xi2_normalized": _finite(xi2),
            "sigma1_normalized": _finite(sigma[p, 0]),
            "sigma2_normalized": _finite(sigma[p, 1]),
            "j1": "" if not np.isfinite(j[p, 0]) else int(j[p, 0]),
            "j2": "" if not np.isfinite(j[p, 1]) else int(j[p, 1]),
            "xi1_hz": _finite(xi1_hz),
            "xi2_hz": _finite(xi2_hz),
            "coarse_range_m": _finite(range_m),
            "physical_label": label,
            "claim_limit": limit,
        })
    return rows


# ------------------------------------------------------------------------- fold compute


@dataclass
class ExpEFoldResult:
    test_subject: int
    n_test_sessions: int
    full_mae: float
    ablated_mae: np.ndarray           # [n_paths], aligned to canonical path_id
    coefficient_rows: list = field(default_factory=list)   # full model only
    dropped_sessions_outer: tuple = ()
    reason: str | None = None         # non-None: contributes no importance value


def _run_single_fold_e(config, band, sessions, store_dir, fold, path_of_column) -> ExpEFoldResult:
    """One outer fold: the full fixed fit, then one refit per path group. Top-level and
    picklable so it can run in a worker process; builds its own provider (open npz handles do
    not survive a spawn) and pins single-threaded math, mirroring `exp_b._run_single_fold_b`.

    `path_of_column` arrives precomputed from the run level, so a worker never rebuilds the
    Kymatio bank: the grouping is a property of the run, identical in every fold, and shipping
    the small integer array is both cheaper and impossible to make fold-dependent by accident.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        provider = exp_b.SessionResidualFeatures(band, sessions, store_dir, config)
        candidate = fixed_candidate(config, band)
        active = dict(candidate.active)
        require_complete_active(active)
        # ONE protocol record governs this whole fold. Every fit below — the full model and
        # all n_paths ablations — runs under exactly this record: an ablation changes which
        # columns of an already-built matrix are visible, never a protocol axis. So the guard
        # runs once per fold rather than n_paths+1 times over an identical dict.
        protocol_freeze_guard(config, active=active)

        # Train-only session means, computed by the SAME call the provider residualizes with,
        # so the drop set and mu_s can never diverge between this check and the features.
        _, dropped_outer = provider.drop_for(fold.train_subjects)
        bundle = provider.data_for(candidate, fold.train_subjects)
        train_rows = np.isin(bundle.subjects, sorted(fold.train_subjects))
        test_rows = bundle.subjects == fold.test_subject
        if not test_rows.any():
            return ExpEFoldResult(
                test_subject=fold.test_subject, n_test_sessions=0, full_mae=float("nan"),
                ablated_mae=np.full(0, np.nan), dropped_sessions_outer=dropped_outer,
                reason="no_surviving_test_rows",
            )
        if path_of_column.size != bundle.X.shape[1]:
            raise ExpEError(
                f"path grouping covers {path_of_column.size} columns but the fold's feature "
                f"matrix has {bundle.X.shape[1]}"
            )

        X_train, y_train = bundle.X[train_rows], bundle.y[train_rows]
        X_test, y_test = bundle.X[test_rows], bundle.y[test_rows]
        subjects_test = bundle.subjects[test_rows]
        session_test = bundle.session_idx[test_rows]

        # Ridge is not seed-sensitive (`regressors.SEED_SENSITIVE`), so the fixed model is one
        # deterministic fit — there is no seed loop and nothing to collapse over. Asserted
        # rather than assumed, so freezing E on a stochastic family later fails here loudly.
        if "ridge" in SEED_SENSITIVE:
            raise ExpEError("ridge became seed-sensitive; Exp E's single-fit design must be revisited")
        params = candidate.params()
        seed = int(config.run.seed)

        full = build_estimator("ridge", params, seed=seed)
        fit_pipeline(full, X_train, y_train)
        full_mae = M.equal_session_residual_mae(
            subjects_test, y_test, full.predict(X_test), session_test
        )

        n_paths = int(path_of_column.max()) + 1
        ablated = np.full(n_paths, np.nan)
        for path_id in range(n_paths):
            keep = path_of_column != path_id
            if not keep.any():
                raise ExpEError(f"path group {path_id} is the entire feature matrix")
            # A FRESH scaler+ridge on the identical rows: the group is removed BEFORE scaling,
            # so the ablated model never sees a mean/scale computed from the deleted columns.
            refit = build_estimator("ridge", params, seed=seed)
            fit_pipeline(refit, X_train[:, keep], y_train)
            ablated[path_id] = M.equal_session_residual_mae(
                subjects_test, y_test, refit.predict(X_test[:, keep]), session_test
            )

        return ExpEFoldResult(
            test_subject=fold.test_subject,
            n_test_sessions=int(len(set(session_test.tolist()))),
            full_mae=float(full_mae),
            ablated_mae=ablated,
            coefficient_rows=_coefficient_rows(full, fold.test_subject, path_of_column),
            dropped_sessions_outer=dropped_outer,
        )


def _coefficient_rows(pipeline, test_subject, path_of_column) -> list[dict]:
    """The FULL fixed model's fitted state, per column — never an ablation refit (§3).

    The coefficients are on the standardized scale, so `coefficient_per_training_sd` is read
    directly off the fitted ridge: it is the predicted change per one training-set SD of that
    column. The scaler's own mean/scale ride along so a reader can return to raw units without
    refitting anything. Descriptive only — a ridge coefficient under thousands of correlated
    columns is not an independent effect estimate.

    The layout tuple (aggregate/channel/segment/statistic) is attached at the run level by
    `coefficient_rows`, not here: it is identical in every fold, so building it inside the
    worker would ship the same thousands of tuples back through the spawn pickle per fold.
    """
    scaler = pipeline.named_steps["scaler"]
    coefficients = np.asarray(pipeline.named_steps["model"].coef_, dtype=float).ravel()
    rows = []
    for i in range(coefficients.size):
        rows.append({
            "model_variant": "full",
            "outer_fold": int(test_subject),
            "test_subject": int(test_subject),
            "feature_index": i,
            "path_id": int(path_of_column[i]),
            "coefficient_per_training_sd": float(coefficients[i]),
            "scaler_mean": float(scaler.mean_[i]),
            "scaler_scale": float(scaler.scale_[i]),
        })
    return rows


def run_exp_e(config, band, sessions, store_dir, *, bank=None, n_workers=1):
    """Every selectable outer fold's LOPGO computation. Returns (results, bank, path_of_column).

    The metadata gate runs ONCE here, before any fit, over every session the run will consume
    — a per-fold check would spend the same evidence 16 times and still let the first fold fit
    before anything was verified.
    """
    if not sessions:
        raise ExpEError("Exp E got an empty session spine — there is nothing to attribute")
    subjects = exp_b.evaluable_subjects_b(sessions)
    # §2.1's wording: "for each SELECTABLE outer fold". E selects nothing, so the flag is not
    # protecting an inner search here — but on the real 16-subject cohort every fold is
    # selectable (15 training subjects against a floor of 3), so the filter is a no-op there
    # and exists to keep E's reported fold set identical to every other experiment's.
    folds = [f for f in nested_loso_splits(subjects) if f.selectable]
    if bank is None:
        bank = reconstruct_bank(config, band)
    assert_bank_matches_store(config, band, sessions, store_dir, bank)

    path_of_column = column_path_index(
        config, band, bank, stored_column_count(config, band, sessions[0], store_dir)
    )

    tasks = [(config, band, sessions, store_dir, fold, path_of_column) for fold in folds]
    results = fold_parallel.run_folds_parallel(
        _run_single_fold_e, tasks, n_workers, f"exp-e-{band}",
    )
    results.sort(key=lambda r: r.test_subject)
    return results, bank, path_of_column


# ---------------------------------------------------------------------------- reporting


IMPORTANCE_FOLDS_COLUMNS = (
    "band", "outer_fold", "test_subject", "path_id", "scattering_order", "feature_group_size",
    "n_test_sessions", "full_mae_pct_points", "ablated_mae_pct_points",
    "importance_delta_mae_pct_points",
)
PATH_METADATA_COLUMNS = (
    "band", "input_domain", "path_id", "scattering_order", "xi1_normalized", "xi2_normalized",
    "sigma1_normalized", "sigma2_normalized", "j1", "j2", "xi1_hz", "xi2_hz", "coarse_range_m",
    "physical_label", "claim_limit",
)
IMPORTANCE_SUMMARY_COLUMNS = PATH_METADATA_COLUMNS + (
    "n_subjects", "mean", "median", "sd", "q25", "q75", "min", "max",
)
RIDGE_COEFFICIENT_COLUMNS = (
    "model_variant", "outer_fold", "test_subject", "feature_index", "frame_aggregate",
    "channel", "path_id", "segment", "statistic", "coefficient_per_training_sd", "scaler_mean",
    "scaler_scale",
)
EXCLUSIONS_COLUMNS = ("band", "outer_fold", "test_subject", "reason", "detail")


def importance_fold_rows(results, band, bank, path_of_column) -> list[dict]:
    """One row per (evaluable fold x path): the pair of MAEs and their difference. A fold that
    produced no held-out residual rows contributes NO importance row — it is in the exclusion
    ledger instead, so a reader can never mistake an absent fold for a zero importance."""
    order = np.asarray(bank["order"])
    group_size = np.bincount(path_of_column, minlength=int(bank["n_paths"]))
    rows = []
    for result in results:
        if result.reason is not None:
            continue
        for path_id in range(int(bank["n_paths"])):
            ablated = float(result.ablated_mae[path_id])
            rows.append({
                "band": band,
                "outer_fold": int(result.test_subject),
                "test_subject": int(result.test_subject),
                "path_id": path_id,
                "scattering_order": int(order[path_id]),
                "feature_group_size": int(group_size[path_id]),
                "n_test_sessions": int(result.n_test_sessions),
                "full_mae_pct_points": float(result.full_mae),
                "ablated_mae_pct_points": ablated,
                "importance_delta_mae_pct_points": ablated - float(result.full_mae),
            })
    return rows


def importance_summary_rows(fold_rows, metadata_rows) -> list[dict]:
    """Per-path descriptive spread over the contributing subjects, joined onto the metadata.

    Deterministic sort `(scattering_order, path_id)` — the §3 rule, so two runs' tables diff
    cleanly and the figure's ordering is not an artifact of dict iteration. `sd` is the
    population standard deviation (ddof=0), the convention owner-decision O1 fixed for every
    dispersion figure in this project. `n_subjects` counts the folds that actually contributed
    a value, which is the honest denominator for the row.

    Each path's values are reduced in TEST-SUBJECT order rather than in the order the rows
    arrived. Floating-point summation is not associative, so without this the last ulp of
    every mean/sd would depend on the order the fold results happened to come back in — and
    `fold_parallel` returns them in COMPLETION order. `run_exp_e` already sorts, so this is
    belt and braces, but it is the difference between "reproducible" and "reproducible as
    long as nobody reorders anything".
    """
    by_path: dict[int, list[tuple[int, float]]] = {}
    for row in fold_rows:
        by_path.setdefault(int(row["path_id"]), []).append(
            (int(row["test_subject"]), float(row["importance_delta_mae_pct_points"]))
        )
    out = []
    for meta in metadata_rows:
        pairs = sorted(by_path.get(int(meta["path_id"]), []))
        values = np.array([value for _, value in pairs], dtype=float)
        summary = dict(meta)
        summary["n_subjects"] = int(values.size)
        if values.size:
            summary.update({
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "sd": float(np.std(values, ddof=0)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
        else:
            for name in ("mean", "median", "sd", "q25", "q75", "min", "max"):
                summary[name] = ""
        out.append(summary)
    out.sort(key=lambda r: (int(r["scattering_order"]), int(r["path_id"])))
    return out


def coefficient_rows(results, config, band, bank) -> list[dict]:
    """The full model's coefficient table, with each column's layout tuple attached.

    The layout is built once here rather than inside the workers: it is a run-level property,
    and rebuilding it per fold would ship 8904 tuples through the spawn pickle 16 times.
    """
    layout = session_feature_layout(bank, bank["n_time"], n_channels(config, band), family="pooled")
    rows = []
    for result in results:
        for row in result.coefficient_rows:
            aggregate, channel, path_id, segment, statistic = layout[int(row["feature_index"])]
            merged = dict(row)
            merged.update({
                "frame_aggregate": aggregate, "channel": int(channel), "path_id": int(path_id),
                "segment": segment, "statistic": statistic, "band": band,
            })
            rows.append({name: merged[name] for name in RIDGE_COEFFICIENT_COLUMNS})
    return rows


def exclusion_rows(results, band) -> list[dict]:
    return [{
        "band": band,
        "outer_fold": int(r.test_subject),
        "test_subject": int(r.test_subject),
        "reason": r.reason,
        "detail": f"dropped_sessions_outer={list(r.dropped_sessions_outer)}",
    } for r in results if r.reason is not None]


def summarize_exp_e(results, config, band, bank, path_of_column, sessions, store_dir) -> dict:
    """The run-level record: the fixed design, the counts, and — FIRST — the interpretation
    context A-M10-6 requires be stated before any path table is read.

    Deliberately carries no ranking, no "top paths" list and no verdict. The summary's job is
    to make the table interpretable, not to interpret it.
    """
    evaluable = [r for r in results if r.reason is None]
    order = np.asarray(bank["order"])
    exp_e = config.exp_e
    design = {
        "model": exp_e.model,
        "ridge_alpha": float(exp_e.ridge_alpha),
        "tiling": exp_e.tiling_10ghz if band == "10ghz" else exp_e.tiling_77ghz,
        "log_branch": exp_e.log_10ghz if band == "10ghz" else exp_e.log_77ghz,
        "q": list(bank["q"]),
        "invariance_ms": bank["invariance_ms"],
        "target": "session-mean-residualized delta_m_pct (Exp B residual scale)",
        "score": "equal_session_residual_mae",
    }
    if band == "10ghz":
        design.update({
            "reduction": exp_e.reduction_10ghz,
            "channel": exp_e.channel_10ghz,
            "range_gate_m": list(exp_e.gate_10ghz_m),
        })
    else:
        design.update({
            "reduction": config.search_77ghz.reduction,
            "channel": config.search_77ghz.channel,
            "gate_m": list(config.search_77ghz.gate_m),
            "rx_fusion": PRIMARY_FUSION_77,
        })
    return {
        "stage": "exp-e",
        "band": band,
        "status": "descriptive",
        "interpretation": [
            "Read the predictive context in the Exp A/B results BEFORE this table: these "
            "importances describe reliance within a fixed model whose out-of-fold predictive "
            "performance is weak, and a difference in MAE under ablation is not evidence that "
            "the removed path carried a hydration signal.",
            "importance_delta_mae_pct_points = ablated_mae - full_mae, in residual delta_m% "
            "points. Positive means the full model predicted that held-out subject better with "
            "the group present.",
            "Attribution is model reliance / predictive contribution, not causality: it cannot "
            "establish or refute a dielectric mechanism, and no path is labelled null or "
            "physical here.",
            "Correlated paths share credit. A group whose information survives in its "
            "neighbours can measure as unimportant while carrying the same signal.",
        ],
        "design": design,
        "fixed_model_note": (
            "The model form is the pre-registered Exp-E anchor, NOT the best model from the "
            "Exp A/B outer results; it is never swapped for one."
        ),
        "dead_config_note": (
            "A-M10-1 replaced the frozen standalone 4-fold permutation CV with leave-one-path-"
            f"group-out refit under outer LOSO. ExpEConfig.n_folds={exp_e.n_folds} and "
            f"fold_assignment={exp_e.fold_assignment!r} are dead config, retained because the "
            "M6 sections are frozen records; nothing in eval/exp_e.py reads them."
        ),
        "n_subjects_e": len(exp_b.evaluable_subjects_b(sessions)),
        "n_sessions": len(sessions),
        "n_outer_folds": len(results),
        "n_evaluable_outer_folds": len(evaluable),
        "n_paths": int(bank["n_paths"]),
        "n_model_columns": int(path_of_column.size),
        "n_columns_per_path_group": int(np.bincount(path_of_column).max()),
        "n_time": int(bank["n_time"]),
        "n_channels": n_channels(config, band),
        "scattering_order_counts": {str(o): int((order == o).sum()) for o in sorted(set(order.tolist()))},
        "bank_fs_hz": float(bank["fs_hz"]),
        "bank_n_in": int(bank["n_in"]),
        "exclusion_reasons": sorted({r.reason for r in results if r.reason is not None}),
        "store_fingerprints_sha256": _store_fingerprints_sha256(band, sessions, store_dir),
        "config_sha256": exp_b.config_fingerprint(config),
    }


def _store_fingerprints_sha256(band, sessions, store_dir) -> str:
    """One hash over the per-session fingerprints the STORE itself records, so a metrics file
    can be tied back to the exact store that produced it.

    Read from the sidecars rather than recomputed via `exp_a.expected_fingerprints`: that
    helper re-hashes every raw session file, which `validate_store` has already done once for
    this run. Repeating it to fill in a provenance field would add minutes of I/O to a job
    whose actual compute is seconds, and would prove nothing new — validation has already
    established that the sidecars equal the expected fingerprints.
    """
    payload = {}
    for s in sessions:
        fingerprint = store_mod.read_fingerprint(store_dir, band, s["subject"], s["session_name"])
        payload[f"{s['subject']}/{s['session_name']}"] = fingerprint
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _write_csv(path, columns, rows) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _interpretability_figure(summary_rows, band, path) -> Path:
    """The map, drawn ONLY from the two saved CSVs' contents (§3).

    Left: every path's mean importance in the table's own canonical order, coloured by
    scattering order, with zero marked so the sign is read rather than inferred. Right: mean
    importance against the path's centre frequency, which is the band-aware axis — a beat
    frequency at 10 GHz, a Doppler/modulation magnitude at 77 GHz. Order-0 has no centre
    frequency and is therefore absent from the right panel by construction, not by filtering.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: no display
    import matplotlib.pyplot as plt

    colours = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    plotted = [r for r in summary_rows if r["mean"] != ""]
    for order in (0, 1, 2):
        rows = [r for r in plotted if int(r["scattering_order"]) == order]
        if not rows:
            continue
        left.scatter([int(r["path_id"]) for r in rows], [float(r["mean"]) for r in rows],
                     s=8, color=colours[order], label=f"order {order}")
        finite = [r for r in rows if r["xi1_hz"] != ""]
        if finite:
            right.scatter([float(r["xi1_hz"]) for r in finite], [float(r["mean"]) for r in finite],
                          s=8, color=colours[order], label=f"order {order}")

    left.axhline(0.0, color="k", lw=1)
    left.set_xlabel("canonical path_id")
    left.set_ylabel("mean importance (Δm% points)")
    left.set_title(f"{band}: mean LOPGO importance per path")
    left.legend(fontsize=8)

    right.axhline(0.0, color="k", lw=1)
    right.set_xscale("log")
    axis = "fast-time beat centre" if band == "10ghz" else "slow-time Doppler/modulation centre"
    right.set_xlabel(f"{axis} xi1 (Hz)")
    right.set_ylabel("mean importance (Δm% points)")
    right.set_title("importance vs path centre frequency")
    right.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def write_exp_e_reports(results, config, band, bank, path_of_column, sessions, store_dir,
                        out_dir) -> dict:
    """The five §3 artifact rows for one band. Returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    metadata = path_metadata_rows(config, band, bank)
    folds = importance_fold_rows(results, band, bank, path_of_column)
    summary_rows = importance_summary_rows(folds, metadata)

    paths["importance_folds"] = _write_csv(
        out_dir / f"importance_folds_{band}.csv", IMPORTANCE_FOLDS_COLUMNS, folds)
    paths["path_metadata"] = _write_csv(
        out_dir / f"path_metadata_{band}.csv", PATH_METADATA_COLUMNS, metadata)
    paths["importance_summary"] = _write_csv(
        out_dir / f"importance_summary_{band}.csv", IMPORTANCE_SUMMARY_COLUMNS, summary_rows)
    paths["ridge_coefficients"] = _write_csv(
        out_dir / f"ridge_coefficients_{band}.csv", RIDGE_COEFFICIENT_COLUMNS,
        coefficient_rows(results, config, band, bank))
    paths["exclusions"] = _write_csv(
        out_dir / f"exclusions_e_{band}.csv", EXCLUSIONS_COLUMNS, exclusion_rows(results, band))

    payload = summarize_exp_e(results, config, band, bank, path_of_column, sessions, store_dir)
    payload["artifact_sha256"] = {
        f"importance_folds_{band}.csv": _file_sha256(paths["importance_folds"]),
        f"path_metadata_{band}.csv": _file_sha256(paths["path_metadata"]),
        f"importance_summary_{band}.csv": _file_sha256(paths["importance_summary"]),
    }
    metrics_path = out_dir / f"metrics_exp_e_{band}.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics"] = metrics_path

    paths["figure"] = _interpretability_figure(
        summary_rows, band, out_dir / f"interpretability_map_{band}.png")
    return paths


def _assert_mechanism_ok_e(results, sessions) -> None:
    """Structural checks that reveal no performance value: every reported fold is an outer
    LOSO fold holding out exactly one subject, and that subject is never in its own training
    set. The full and ablated scores of a fold are computed on the identical held-out rows by
    construction (one `test_rows` mask), which is what makes the difference interpretable."""
    subjects = exp_b.evaluable_subjects_b(sessions)
    folds = {f.test_subject: f for f in nested_loso_splits(subjects)}
    for result in results:
        assert result.test_subject in folds, result.test_subject
        assert result.test_subject not in folds[result.test_subject].train_subjects
        if result.reason is None:
            assert result.n_test_sessions >= 1


def run_and_report_e(config, band, sessions, store_dir, run_dir, *, mode, analysis_commit,
                     bank=None, n_workers=1) -> dict:
    """validate_store -> run_exp_e -> structural assertions -> smoke run-log or full reporting.

    `mode="smoke"` is MECHANISM-ONLY, matching the A/B/C/G doctrine: the identical gate,
    grouping and refit path runs, but no importance value is surfaced — only the structural
    run-log. A smoke exists to prove the mechanism works, never to preview the result.

    `bank` exists so the tests can drive the whole path on a small synthetic store instead of
    a real 742-path Kymatio bank. It is not a way around the metadata gate: whatever bank is
    supplied is still required to equal every consumed session's stored `order`, which is the
    property the gate is actually protecting.
    """
    store_mod.validate_store(
        band, store_dir, exp_a.expected_fingerprints(config, band, sessions),
        analysis_commit=analysis_commit,
    )
    results, bank, path_of_column = run_exp_e(
        config, band, sessions, store_dir, bank=bank, n_workers=n_workers
    )
    _assert_mechanism_ok_e(results, sessions)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        log = run_dir / f"run_log_exp_e_{band}.json"
        log.write_text(json.dumps({
            "stage": "exp-e-smoke", "band": band, "mode": "mechanism-only",
            "n_sessions": len(sessions),
            "n_outer_folds": len(results),
            "n_evaluable_outer_folds": sum(1 for r in results if r.reason is None),
            "n_paths": int(bank["n_paths"]),
            "n_model_columns": int(path_of_column.size),
            "exclusion_reasons": sorted({r.reason for r in results if r.reason is not None}),
            "note": "importance values suppressed -- mechanism-only smoke",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"run_log": log}

    if not any(r.reason is None for r in results):
        raise ExpEError(
            "no outer fold produced a held-out residual row — Experiment E has no attribution "
            "to report"
        )
    return write_exp_e_reports(
        results, config, band, bank, path_of_column, sessions, store_dir, run_dir
    )
