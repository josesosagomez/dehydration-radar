"""Experiment D — the baselines, under Experiment A's identical LOSO harness.

Two halves. The **learned** half (milestone 9 step 7) is the per-frame spine
(`build_frames_d`) and the nested torch path for ONE outer fold (`run_cnn_family`) — the
frozen 6-config grid x early stopping x epoch budget x per-seed final refit. The **cheap**
half (step 8) is the two deterministic baselines under the identical folds — the physics
range/Doppler power ratio and the session-index-only lookup — plus what turns either half
into a reportable result: the four per-family merged artifacts, the GPU fold-array
shard/merge, and the frozen radar-vs-baseline comparison statistics.

**Why this is a separate engine from `harness.py`.** The sklearn engine selects a
*candidate*; a CNN fold selects a *configuration and an epoch budget*, and its unit of data
is a FRAME while its unit of analysis stays a SESSION. So the structure is `torch_fit.py`'s
T18-protected algorithm — the optimizer never sees validation data, and each epoch's
weights are a pure function of the training rows and the seed — extended with the frozen
sampler/batching and the 6-config grid. What is NOT re-implemented: the folds (only
`eval/splits.py`), the tie-break (only `eval/selection.select_candidate`), the session-level
metric (`eval/metrics.subject_balanced_mae`), and the fit-audit shape (`harness.FitRecord`,
so `harness.fit_audit` reads a CNN fold result unchanged).

**The frame -> session step is the median** of the frame-level predictions within a session
(`BaselineConfig.frame_to_session_aggregation`), applied identically to every CNN baseline
and to the physics scalar (O-M9-4). Scoring is always session-level; frames exist only so
the network has rows to train on, and the cheap baselines never see a frame at all beyond
that median.
"""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.signal.windows import hann
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..config import beat_band_hz
from ..features import store as store_mod
from ..models import baselines as baseline_models
from ..models import cnn
from ..provenance import sha256_file
from . import harness
from . import metrics as M
from .harness import FitRecord
from .metrics import subject_balanced_mae
from .selection import CandidateScore, select_candidate

CNN_FAMILIES = cnn.CNN_FAMILIES
BANDS = ("10ghz", "77ghz")

# The two deterministic (K = 1, no grid, no seed dimension) baselines, and the full Exp D
# family set the entrypoint and the comparison report enumerate.
DETERMINISTIC_FAMILIES = ("physics", "session_index")
EXPD_FAMILIES = CNN_FAMILIES + DETERMINISTIC_FAMILIES

# O-M9-3: the composite procedure's candidate set INSIDE each outer fold and, identically,
# the Holm family — the three PRIMARY variants only. The matched-preprocessing ablations are
# reported as ablations and enter no comparison family, so the multiplicity denominator
# stays exactly 3. The tuple order is also the composite's tie-break order.
COMPOSITE_MEMBERS = ("cnn1d_raw", "spec2d_raw", "physics")
ABLATION_FAMILIES = ("cnn1d_matched", "spec2d_matched")

# Exp D RNG offsets off config.run.seed -- FIXED and NAMED blocks of 10, never a running
# counter (the Exp B trap-10 doctrine): Exp A occupies +0..3, Exp B +100..134,
# Exp C +200..212, Exp D +300..373.
RNG_OFFSET_EXPD_BASE = 300
_RNG_BLOCKS = EXPD_FAMILIES + ("composite", "radar")
_RNG_METRIC_OFFSET = {"mae": 0, "rmse": 1, "pooled_pearson_r": 2, "difference_vs_radar": 3}


def _rng_offset(block: str, metric: str) -> int:
    return RNG_OFFSET_EXPD_BASE + 10 * _RNG_BLOCKS.index(block) + _RNG_METRIC_OFFSET[metric]


def _all_rng_offsets() -> list[int]:
    """Every resolved RNG offset this module uses -- tested directly for pairwise
    distinctness, and against Exp A's, Exp B's and Exp C's."""
    return [_rng_offset(block, metric) for block in _RNG_BLOCKS for metric in _RNG_METRIC_OFFSET]

# The frozen baseline hyperparameter grid, written INDEPENDENTLY of `ModelGridConfig` so the
# two sources check each other rather than a value checking itself (the same doctrine as
# `exp_c.FRANK_HALL_MAX_ITER`). 3 x 2 = 6 configs per learned family, well inside the frozen
# budget K = 12 — the budget-parity rule of `implementation_plan.md:917-919`.
FROZEN_BASELINE_LEARNING_RATES = (3e-4, 1e-3, 3e-3)
FROZEN_BASELINE_WEIGHT_DECAYS = (0.0, 1e-4)

# The frozen training protocol this path implements, literally and only.
FROZEN_OPTIMIZER = "adam"
FROZEN_LOSS = "mse"
FROZEN_FRAME_TO_SESSION = "median"
FROZEN_CHECKPOINT_METRIC = "inner_val_session_mae"
FROZEN_CHECKPOINT_DIRECTION = "minimize"

# Per-fit RNG derivation (plan §2.8, §5 trap 8). EVERY fit gets its own generator built from
# this integer — a shared generator would make fit k's data order depend on fits 1..k-1, so a
# held-out mutation would shift TRAINING batches and the mutation property would be false.
# Off `config.run.seed` like every other RNG in this project, at an offset far from the
# metric-bootstrap blocks (Exp A +0..3, Exp B +100..134, Exp C +200..212).
FIT_SEED_BASE = 900_000
_MAX_FOLD_ID = 64          # subject ids / fold positions
_MAX_CONFIGS = 16          # the frozen grid has 6
_MAX_INNER_FOLDS = 8       # the frozen cap is 5; slot _MAX_INNER_FOLDS is the final refit
_MAX_SEEDS = 16            # the frozen seed set is {1..5}

# Prediction chunk size. Fixed (not adaptive) so a prediction is a deterministic function of
# the rows and the state, whatever the set size; BatchNorm runs in eval mode, so chunking is
# numerically irrelevant and exists only to bound GPU memory.
_PREDICT_CHUNK = 256


class ExpDError(ValueError):
    """A malformed Exp D spine, fold, or family."""


class ExpDProtocolError(ExpDError):
    """A computation that is about to run is not authorized by Exp D's frozen protocol."""


# --------------------------------------------------------------- the frozen protocol


def baseline_config_grid(model_grid) -> list[dict]:
    """The frozen 6-config baseline grid, learning-rate major / weight-decay minor.

    Fails closed if `ModelGridConfig` and this module's independently written copy of the
    frozen values disagree — a config edit cannot silently widen the search.
    """
    for name, frozen, configured in (
        ("baseline_learning_rate", FROZEN_BASELINE_LEARNING_RATES,
         tuple(model_grid.baseline_learning_rate)),
        ("baseline_weight_decay", FROZEN_BASELINE_WEIGHT_DECAYS,
         tuple(model_grid.baseline_weight_decay)),
    ):
        if frozen != configured:
            raise ExpDProtocolError(
                f"the frozen baseline grid drifted: model_grid.{name} is {configured!r} but "
                f"the frozen value is {frozen!r} — refusing to search an unfrozen space"
            )
    return [
        {"lr": float(lr), "weight_decay": float(wd)}
        for lr in FROZEN_BASELINE_LEARNING_RATES
        for wd in FROZEN_BASELINE_WEIGHT_DECAYS
    ]


def require_frozen_training_protocol(baselines) -> None:
    """The optimizer, loss, frame->session rule and checkpoint metric are frozen values, not
    switches. This path implements exactly one of each, so a config that names a different
    one must stop rather than be silently ignored."""
    for name, frozen, configured in (
        ("optimizer", FROZEN_OPTIMIZER, baselines.optimizer),
        ("loss", FROZEN_LOSS, baselines.loss),
        ("frame_to_session_aggregation", FROZEN_FRAME_TO_SESSION,
         baselines.frame_to_session_aggregation),
        ("checkpoint_metric", FROZEN_CHECKPOINT_METRIC, baselines.checkpoint_metric),
        ("checkpoint_direction", FROZEN_CHECKPOINT_DIRECTION, baselines.checkpoint_direction),
    ):
        if frozen != configured:
            raise ExpDProtocolError(
                f"BaselineConfig.{name} is {configured!r} but this path implements the frozen "
                f"{frozen!r} only — refusing to run a protocol it does not implement"
            )
    cnn.assert_frozen_constants(baselines)


def fit_seed(run_seed: int, *, fold_id: int, config_index: int, inner_fold: int, seed: int) -> int:
    """The named per-fit RNG derivation: one integer per (fold, config, inner fold, seed).

    A mixed-radix index over the four axes, offset by `FIT_SEED_BASE + run_seed`, so no two
    fits of one run share a generator. `fold_id` is the outer fold's held-out subject (the
    fold's stable identity), and `inner_fold = -1` is the final refit, which has no inner
    fold and takes the reserved slot `_MAX_INNER_FOLDS`.
    """
    slot = _MAX_INNER_FOLDS if inner_fold < 0 else int(inner_fold)
    for name, value, bound in (
        ("fold_id", fold_id, _MAX_FOLD_ID),
        ("config_index", config_index, _MAX_CONFIGS),
        ("inner_fold", slot, _MAX_INNER_FOLDS + 1),
        ("seed", seed, _MAX_SEEDS),
    ):
        if not 0 <= int(value) < bound:
            raise ExpDError(
                f"{name}={value!r} is outside the named seed-derivation radix [0, {bound}) — "
                "widening it would silently collide two fits' generators"
            )
    index = (
        ((int(fold_id) * _MAX_CONFIGS + int(config_index)) * (_MAX_INNER_FOLDS + 1) + slot)
        * _MAX_SEEDS
        + int(seed)
    )
    return int(run_seed) + FIT_SEED_BASE + index


# ------------------------------------------------------------------------ the spine


@dataclass
class FramesD:
    """The per-frame Exp D spine for one (band, family).

    Rows are frames in canonical (subject, session_idx, stored frame order); the
    session-level arrays are indexed by `session_row`, which is the analysis unit.
    """

    band: str
    family: str
    subjects: np.ndarray              # (n_frames,)
    session_row: np.ndarray           # (n_frames,) index into the session-level arrays
    frame_ids: np.ndarray             # (n_frames,) the QC-selected raw frame index
    X: np.ndarray                     # (n_frames, C, N) or (n_frames, C, F, T)
    y: np.ndarray                     # (n_frames,) the session's Δm%, broadcast to its frames
    session_subjects: np.ndarray      # (n_sessions,)
    session_idx: np.ndarray           # (n_sessions,)
    session_delta_m_pct: np.ndarray   # (n_sessions,)


def build_frames_d(config, band, family, sessions, store_dir) -> FramesD:
    """One row per QC-passed frame: subject, session, frame id, the family's stored signal
    put through that family's own per-frame input construction, and the session's Δm%
    broadcast to its frames.

    The broadcast target is a TRAINING device only — the network needs a per-row target —
    and never a scoring unit: every score in this module aggregates frames to sessions with
    the frozen median first.

    `sessions` is `exp_a.build_sessions(config, band)`'s output, so the frame membership is
    the QC-selected one and `frame_ids` pins the store's canonical order.
    """
    require_frozen_training_protocol(config.baselines)
    if band not in BANDS:
        raise ExpDError(f"unknown band {band!r} (expected one of {BANDS})")
    if family not in CNN_FAMILIES:
        raise ExpDError(f"unknown Exp D CNN family {family!r} (expected one of {CNN_FAMILIES})")
    signal_key, build_input = cnn.FRAME_INPUT[(band, family)]

    subjects, session_row, frame_ids, inputs, targets = [], [], [], [], []
    session_subjects, session_idx, session_delta = [], [], []
    for row, session in enumerate(sessions):
        store = store_mod.read_session_store(band, session["subject"], session["session_name"],
                                             store_dir)
        try:
            if signal_key not in store:
                raise ExpDError(
                    f"store for {band} {session['subject']}/{session['session_name']} has no "
                    f"{signal_key!r} — Exp D needs a schema-v2 store"
                )
            signals = np.asarray(store[signal_key])
        finally:
            store.close()

        expected = list(session["frame_ids"])
        if signals.shape[0] != len(expected):
            raise ExpDError(
                f"store for {band} {session['subject']}/{session['session_name']} holds "
                f"{signals.shape[0]} {signal_key!r} frames but the QC spine selected "
                f"{len(expected)} — the frame order is not aligned"
            )
        for i, frame_id in enumerate(expected):
            inputs.append(build_input(signals[i]))
            subjects.append(int(session["subject"]))
            session_row.append(row)
            frame_ids.append(int(frame_id))
            targets.append(float(session["delta_m_pct"]))
        session_subjects.append(int(session["subject"]))
        session_idx.append(int(session["session_idx"]))
        session_delta.append(float(session["delta_m_pct"]))

    return FramesD(
        band=band,
        family=family,
        subjects=np.array(subjects, dtype=int),
        session_row=np.array(session_row, dtype=int),
        frame_ids=np.array(frame_ids, dtype=int),
        # contiguous so a fitted statistic depends on the DATA, not on the memory layout a
        # slice happens to inherit (see `cnn.spectrogram`)
        X=np.ascontiguousarray(np.stack(inputs)),
        y=np.array(targets, dtype=float),
        session_subjects=np.array(session_subjects, dtype=int),
        session_idx=np.array(session_idx, dtype=int),
        session_delta_m_pct=np.array(session_delta, dtype=float),
    )


# ------------------------------------------------------ frame -> session, sampler weights


def median_frame_to_session(values, group, n_groups) -> np.ndarray:
    """The frozen `frame_to_session_aggregation: median` — one value per session, the median
    over that session's selected frames. `group[i]` is row i's session position."""
    values = np.asarray(values, dtype=float)
    group = np.asarray(group)
    return np.array([float(np.median(values[group == g])) for g in range(int(n_groups))])


def session_sampler_weights(session_row) -> np.ndarray:
    """Per-row `1/frames_in_session` — the literal frozen weight, deliberately NOT
    renormalized to mean 1.

    `WeightedRandomSampler` treats the weights as unnormalized probabilities, so the scale
    cannot affect the draw; keeping the literal value means the recorded `FitRecord` is the
    quantity the plan names. Every session then carries the same total mass, which is what
    makes the loss session-balanced — and it is the ONLY balancing applied: the MSE loss
    carries no per-row weight, so nothing is weighted twice.
    """
    session_row = np.asarray(session_row)
    counts = Counter(session_row.tolist())
    return np.array([1.0 / counts[int(k)] for k in session_row], dtype=float)


def epoch_budget_from(selected_epoch_counts) -> int:
    """`:650-655`, with trap 7's population made explicit by the caller: the median over the
    winning config's (inner fold x seed) selected epoch counts.

    `int(np.median(...))` — the same rule `models/torch_fit.py:224` already uses, which
    floors for positive counts, so an even population resolves downward rather than to a
    non-integral epoch count.
    """
    return int(np.median(np.asarray(selected_epoch_counts, dtype=float)))


def _session_groups(frames: FramesD, rows) -> tuple[np.ndarray, np.ndarray]:
    """(group index per selected row, the session positions) in canonical session order."""
    keys = frames.session_row[rows]
    positions = np.array(sorted(set(keys.tolist())), dtype=int)
    lookup = {int(k): i for i, k in enumerate(positions)}
    return np.array([lookup[int(k)] for k in keys], dtype=int), positions


# ---------------------------------------------------------------------- the torch fit


@dataclass
class _FitOutcome:
    state: dict
    n_epochs_selected: int
    val_score: float
    steps_per_epoch: int
    first_epoch_batch_indices: np.ndarray
    val_history: list = field(default_factory=list)
    train_pred_per_epoch: list = field(default_factory=list)


def _predict(model, x, device) -> np.ndarray:
    """`model.eval()` at EVERY predict (§5 trap 9). With `model.train()` still active,
    BatchNorm keeps updating its running statistics, which silently makes a prediction
    depend on the composition of the set being predicted — leakage-adjacent and invisible."""
    model.eval()
    device = torch.device(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(x), _PREDICT_CHUNK):
            chunk = torch.from_numpy(
                np.ascontiguousarray(x[start : start + _PREDICT_CHUNK], dtype=np.float32)
            )
            out.append(model(chunk.to(device)).detach().cpu().numpy().astype(float))
    return np.concatenate(out) if out else np.zeros(0, dtype=float)


def _predict_with_state(family, in_channels, state, x, device) -> np.ndarray:
    model = cnn.build_network(family, in_channels).to(torch.device(device))
    model.load_state_dict(state)
    return _predict(model, x, device)


def _train_cnn(x_train, y_train, weights, *, family, lr, weight_decay, max_epochs, batch_size,
               betas, patience, min_delta, seed_value, device, val=None,
               trace_train=False) -> _FitOutcome:
    """One CNN fit — the T18-protected algorithm with the frozen sampler and batching.

    With `val = (x_val, group, session_subjects, session_truth)` the fit early-stops on the
    frozen `checkpoint_metric`: the val frames' predictions are aggregated to sessions by the
    median, then scored with `subject_balanced_mae` over those session rows. The optimizer
    never sees `val`; only the stop time and the kept checkpoint depend on it. With `val =
    None` the fit runs exactly `max_epochs` epochs (the final refit).

    **The sampler/DataLoader contract, pinned because every option left open changes
    BatchNorm's statistics, the optimizer-step count, the stop time and hence the budget:**
    `WeightedRandomSampler(..., num_samples=len(train), replacement=True, generator=g)` —
    *with* replacement, since drawing `len(train)` rows without replacement would merely
    permute the training set and discard the session balancing entirely; `shuffle=False`
    (mutually exclusive with a sampler, which already fixes the order); `num_workers=0`, so
    no worker RNG or completion order enters the trace; and `drop_last=True`, so every batch
    is exactly `batch_size` rows and one epoch is `floor(len(train)/batch_size)` optimizer
    steps. `drop_last` rather than keeping the short tail because BatchNorm in train mode is
    undefined on a 1-row batch, and because under replacement sampling the dropped remainder
    is a random tail, not a systematically excluded subset.
    """
    device = torch.device(device)
    torch.set_num_threads(1)
    cnn.enable_gpu_determinism(device)

    n_train = int(x_train.shape[0])
    steps_per_epoch = n_train // int(batch_size)
    if steps_per_epoch == 0:
        raise ExpDError(
            f"{n_train} training frames is fewer than one batch_size={batch_size} and "
            "drop_last=True, so this fit would take zero optimizer steps — refusing to "
            "'train' a model that never saw a gradient"
        )

    # framework-default init, made a pure function of the named seed (the torch_fit.py
    # convention: seed the global RNG immediately before construction).
    torch.manual_seed(int(seed_value))
    model = cnn.build_network(family, int(x_train.shape[1])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(lr), betas=tuple(betas), weight_decay=float(weight_decay)
    )
    # MSE with reduction="mean" and NO per-row weighting: the session balancing lives in the
    # sampler and is never applied twice.
    loss_fn = nn.MSELoss(reduction="mean")

    generator = torch.Generator()
    generator.manual_seed(int(seed_value))
    dataset = TensorDataset(
        torch.arange(n_train),
        torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32)),
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(np.asarray(weights, dtype=float), dtype=torch.double),
        num_samples=n_train, replacement=True, generator=generator,
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), sampler=sampler, shuffle=False,
                        drop_last=True, num_workers=0, generator=generator)

    best_metric, best_state, best_epoch, since_improve = float("inf"), None, -1, 0
    val_history: list = []
    train_pred_per_epoch: list = []
    first_epoch_batch_indices: list = []

    for epoch in range(int(max_epochs)):
        model.train()
        for row_index, xb, yb in loader:
            if epoch == 0:
                first_epoch_batch_indices.append(row_index.numpy().copy())
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

        if trace_train:
            # opt-in: 200 full-training-set forward passes are pointless outside the
            # common-prefix contract test, which is the only consumer.
            train_pred_per_epoch.append(_predict(model, x_train, device))
        if val is None:
            continue

        x_val, group, session_subjects, session_truth = val
        session_pred = median_frame_to_session(
            _predict(model, x_val, device), group, len(session_truth)
        )
        metric = subject_balanced_mae(session_subjects, session_truth, session_pred)
        val_history.append(metric)
        if metric < best_metric - float(min_delta):
            best_metric, best_epoch = metric, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= int(patience):
                break

    trace = np.array(first_epoch_batch_indices, dtype=int)
    if val is None:
        return _FitOutcome(copy.deepcopy(model.state_dict()), int(max_epochs), float("nan"),
                           steps_per_epoch, trace, val_history, train_pred_per_epoch)
    if best_state is None:      # never improved on the initial +inf; keep epoch 0
        best_state, best_epoch = copy.deepcopy(model.state_dict()), 0
        best_metric = val_history[0] if val_history else float("nan")
    return _FitOutcome(best_state, best_epoch + 1, float(best_metric), steps_per_epoch, trace,
                       val_history, train_pred_per_epoch)


# --------------------------------------------------------------- one fold, end to end


@dataclass
class CnnInnerCell:
    """One (config, inner fold, seed) cell. `inner_val` + `fits` make this readable by
    `harness.fit_audit` unchanged."""

    config_index: int
    lr: float
    weight_decay: float
    inner_fold: int
    seed: int
    inner_val: frozenset
    n_epochs_selected: int
    val_session_mae: float
    steps_per_epoch: int
    first_epoch_batch_indices: np.ndarray
    fits: list


@dataclass
class CnnSeedOutcome:
    seed: int
    session_predictions: np.ndarray
    session_mae: float


@dataclass
class CnnFoldResult:
    band: str
    family: str
    test_subject: int
    train_subjects: frozenset
    n_inner_folds: int
    config_grid: list                 # the frozen 6, in enumeration order
    per_config_scores: list           # list[CandidateScore], one per config
    selected_config_index: int
    selected_config: dict
    epoch_budget: int
    selected_epoch_counts: list       # the (inner fold x seed) counts the median came from
    inner_results: list               # list[CnnInnerCell]
    final_fits: list
    seed_outcomes: list               # list[CnnSeedOutcome] — scored separately, never ensembled
    test_session_subjects: np.ndarray
    test_session_idx: np.ndarray
    test_session_truth: np.ndarray
    test_n_frames_aggregated: np.ndarray


@dataclass
class _Prepared:
    """One (training rows, evaluation rows) split, with its train-only fitted transform
    already applied to both sides."""

    x_train: np.ndarray
    y_train: np.ndarray
    weights: np.ndarray
    fits: list
    x_eval: np.ndarray
    eval_group: np.ndarray
    eval_session_subjects: np.ndarray
    eval_session_truth: np.ndarray
    eval_session_positions: np.ndarray
    eval_n_frames: np.ndarray

    def val_bundle(self):
        return (self.x_eval, self.eval_group, self.eval_session_subjects,
                self.eval_session_truth)


def _prepare(frames: FramesD, train_rows, eval_rows, *, role, subjects) -> _Prepared:
    """Assemble one fit's tensors and its train-only fitted quantities.

    For the spectrogram families this is where the ONE fitted input transform lives: the
    per-(channel, frequency) mean/std is fit on the TRAINING frames and applied to both
    sides. Computing it on the pooled set would be a leakage vector (§5 trap 12).
    """
    x_train = frames.X[train_rows]
    x_eval = frames.X[eval_rows]
    fits = []
    if x_train.ndim == 4:
        norm = cnn.SpectrogramNorm.fit(x_train)
        x_train = norm.transform(x_train)
        x_eval = norm.transform(x_eval)
        fits.append(FitRecord("spectrogram_norm", role, frozenset(subjects), norm.params()))

    weights = session_sampler_weights(frames.session_row[train_rows])
    fits.append(FitRecord("sampler_weights", role, frozenset(subjects), {"weights": weights.copy()}))

    group, positions = _session_groups(frames, eval_rows)
    return _Prepared(
        x_train=np.ascontiguousarray(x_train, dtype=np.float32),
        y_train=np.ascontiguousarray(frames.y[train_rows], dtype=np.float32),
        weights=weights,
        fits=fits,
        x_eval=np.ascontiguousarray(x_eval, dtype=np.float32),
        eval_group=group,
        eval_session_subjects=frames.session_subjects[positions],
        eval_session_truth=frames.session_delta_m_pct[positions],
        eval_session_positions=positions,
        eval_n_frames=np.array([int(np.count_nonzero(group == g)) for g in range(len(positions))]),
    )


def run_cnn_family(config, band, family, fold, seeds, frames: FramesD, *,
                   device="cpu") -> CnnFoldResult:
    """One outer fold of one CNN family — the unit of work of the GPU fold array.

    Inner: the frozen 6 configs x the fold's inner folds x the seed set, each fit early-
    stopping on the inner-val session MAE. Selection: per-config score = mean over (inner
    fold x seed) of the best-checkpoint score, winner via `select_candidate` (one family, so
    the simplicity and dimension rungs are constant and only the variance can break a
    residual tie). Budget: the median over the winner's (inner fold x seed) selected epoch
    counts — the seed dimension is IN the population, because batching makes every seed's
    trajectory distinct (§5 trap 7). Refit: per seed, all outer-training frames, exactly the
    budget, no early stopping and no validation subject; the held-out subject's frames are
    aggregated to sessions by the median and each seed is scored separately.
    """
    baselines = config.baselines
    require_frozen_training_protocol(baselines)
    if family not in CNN_FAMILIES:
        raise ExpDError(f"unknown Exp D CNN family {family!r} (expected one of {CNN_FAMILIES})")
    if frames.band != band or frames.family != family:
        raise ExpDError(
            f"frames were built for ({frames.band!r}, {frames.family!r}) but this run is "
            f"({band!r}, {family!r})"
        )
    if not fold.selectable:
        raise ExpDError(
            f"outer fold test_subject={fold.test_subject} is not selectable (too few "
            "training subjects for inner CV) — it contributes no Exp D result"
        )

    grid = baseline_config_grid(config.model_grid)
    seeds = tuple(int(s) for s in seeds)
    feature_dimension = cnn.flattened_input_dimension(frames.X)
    in_channels = int(frames.X.shape[1])

    # One preparation per inner fold: the fitted norm and the sampler weights depend on the
    # inner-TRAINING rows alone, never on the config or the seed.
    prepared_inner = [
        (fj, inner, _prepare(
            frames,
            np.isin(frames.subjects, sorted(inner.train_subjects)),
            np.isin(frames.subjects, sorted(inner.val_subjects)),
            role="inner_train", subjects=inner.train_subjects,
        ))
        for fj, inner in enumerate(fold.inner_folds)
    ]

    inner_results: list = []
    scores_by_config: dict = {ci: [] for ci in range(len(grid))}
    epochs_by_config: dict = {ci: [] for ci in range(len(grid))}
    for ci, params in enumerate(grid):
        for fj, inner, prepared in prepared_inner:
            for seed in seeds:
                outcome = _train_cnn(
                    prepared.x_train, prepared.y_train, prepared.weights, family=family,
                    lr=params["lr"], weight_decay=params["weight_decay"],
                    max_epochs=baselines.max_epochs, batch_size=baselines.batch_size,
                    betas=baselines.adam_betas, patience=baselines.early_stopping_patience,
                    min_delta=baselines.early_stopping_min_delta,
                    seed_value=fit_seed(config.run.seed, fold_id=fold.test_subject,
                                        config_index=ci, inner_fold=fj, seed=seed),
                    device=device, val=prepared.val_bundle(),
                )
                scores_by_config[ci].append(outcome.val_score)
                epochs_by_config[ci].append(outcome.n_epochs_selected)
                inner_results.append(CnnInnerCell(
                    config_index=ci, lr=params["lr"], weight_decay=params["weight_decay"],
                    inner_fold=fj, seed=seed, inner_val=inner.val_subjects,
                    n_epochs_selected=outcome.n_epochs_selected,
                    val_session_mae=outcome.val_score,
                    steps_per_epoch=outcome.steps_per_epoch,
                    first_epoch_batch_indices=outcome.first_epoch_batch_indices,
                    fits=prepared.fits + [
                        FitRecord(
                            "cnn_state", "inner_train", frozenset(inner.train_subjects),
                            cnn.torch_module_state_to_numpy(outcome.state),
                        ),
                        # The selected epoch count is a fitted quantity too (§2.8): the
                        # trajectory it indexes into is a pure function of the inner-TRAINING
                        # rows, and only WHERE along it we stop is driven by inner-val. Its
                        # subject set is therefore the training set — which is what the audit
                        # needs to show the held-out subject is absent from.
                        FitRecord(
                            "selected_epochs", "inner_train", frozenset(inner.train_subjects),
                            {"n_epochs_selected": np.asarray(outcome.n_epochs_selected,
                                                             dtype=np.int64)},
                        ),
                    ],
                ))

    per_config_scores = [
        CandidateScore(
            candidate_id=f"cfg{ci}_lr{params['lr']:g}_wd{params['weight_decay']:g}",
            inner_val_mae=float(np.mean(scores_by_config[ci])),
            # one family and one input size: the simplicity and dimension rungs are constant
            # here and can never decide a comparison; the variance rung can.
            simplicity_rank=0,
            feature_dimension=feature_dimension,
            inner_fold_variance=float(np.std(scores_by_config[ci], ddof=0)),
        )
        for ci, params in enumerate(grid)
    ]
    winner = select_candidate(per_config_scores)             # the single tie-break source
    ci_win = next(i for i, s in enumerate(per_config_scores) if s.candidate_id == winner.candidate_id)
    selected_epoch_counts = list(epochs_by_config[ci_win])
    epoch_budget = epoch_budget_from(selected_epoch_counts)

    # Final refit: all outer-training frames, exactly the budget, no early stopping and no
    # validation subject sacrificed (`:650-655`).
    prepared_outer = _prepare(
        frames,
        np.isin(frames.subjects, sorted(fold.train_subjects)),
        np.isin(frames.subjects, [fold.test_subject]),
        role="outer_train", subjects=fold.train_subjects,
    )
    final_fits = list(prepared_outer.fits)
    seed_outcomes = []
    for seed in seeds:
        outcome = _train_cnn(
            prepared_outer.x_train, prepared_outer.y_train, prepared_outer.weights,
            family=family, lr=grid[ci_win]["lr"], weight_decay=grid[ci_win]["weight_decay"],
            max_epochs=epoch_budget, batch_size=baselines.batch_size,
            betas=baselines.adam_betas, patience=baselines.early_stopping_patience,
            min_delta=baselines.early_stopping_min_delta,
            seed_value=fit_seed(config.run.seed, fold_id=fold.test_subject,
                                config_index=ci_win, inner_fold=-1, seed=seed),
            device=device, val=None,
        )
        session_pred = median_frame_to_session(
            _predict_with_state(family, in_channels, outcome.state, prepared_outer.x_eval, device),
            prepared_outer.eval_group, len(prepared_outer.eval_session_truth),
        )
        seed_outcomes.append(CnnSeedOutcome(
            seed=seed,
            session_predictions=session_pred,
            session_mae=subject_balanced_mae(
                prepared_outer.eval_session_subjects, prepared_outer.eval_session_truth,
                session_pred,
            ),
        ))
        final_fits.append(FitRecord(
            "cnn_state", "outer_train", frozenset(fold.train_subjects),
            cnn.torch_module_state_to_numpy(outcome.state),
        ))
    final_fits.append(FitRecord(
        "selected_epochs", "outer_train", frozenset(fold.train_subjects),
        {"n_epochs_selected": np.asarray(epoch_budget, dtype=np.int64)},
    ))

    return CnnFoldResult(
        band=band,
        family=family,
        test_subject=fold.test_subject,
        train_subjects=fold.train_subjects,
        n_inner_folds=len(fold.inner_folds),
        config_grid=grid,
        per_config_scores=per_config_scores,
        selected_config_index=ci_win,
        selected_config=grid[ci_win],
        epoch_budget=epoch_budget,
        selected_epoch_counts=selected_epoch_counts,
        inner_results=inner_results,
        final_fits=final_fits,
        seed_outcomes=seed_outcomes,
        test_session_subjects=prepared_outer.eval_session_subjects,
        test_session_idx=frames.session_idx[prepared_outer.eval_session_positions],
        test_session_truth=prepared_outer.eval_session_truth,
        test_n_frames_aggregated=prepared_outer.eval_n_frames,
    )


# ============================================================ (iii) the physics baseline
#
# A signal-domain energy ratio, correctly labelled: at 10 GHz the beat-frequency axis maps
# to RANGE, so this is target-range vs background-range power (`:920-944`); at 77 GHz the
# primary domain is Doppler, so it is the DC bin against every resolvable motion bin
# (A-M6-2 (iii)). Both read the **unstandardized** stored raw signal — robust
# standardization removes the median and rescales, which destroys the absolute power the
# ratio is made of and, at 77 GHz, would zero the DC bin that IS the target band (§5 trap
# 13). That is why the store keeps these arrays raw.


PHYSICS_EPS_SCALE = 1e-12

# Written out rather than derived from `cnn.FRAME_INPUT` so the two tables check each other:
# the physics scalar and the raw CNN families must consume the same untouched signal.
PHYSICS_SIGNAL_KEY = {"10ghz": "sig__raw_beat", "77ghz": "sig__raw_slowtime"}


def half_spectrum_power(signal) -> np.ndarray:
    """Periodic-Hann-windowed |FFT|^2 over the non-negative half spectrum (bins
    0 .. n//2 - 1, DC included, Nyquist excluded).

    The window is periodic and the half-spectrum convention is `[: n//2]` because that is
    this repo's own convention for every band-power ratio (`qc/screens.py:147-154`) — and
    it is what makes A-M6-2's "bins 0..127" of the 256-point Doppler FFT literally the
    non-negative half spectrum. Windowing matters here because leakage from a strong
    out-of-band tone would otherwise inflate the in-band sum.
    """
    x = np.asarray(signal)
    n = int(x.shape[0])
    spectrum = np.fft.fft(x * hann(n, sym=False))
    return np.abs(spectrum[: n // 2]) ** 2


def band_masks_from_frequencies(freqs, target_band_hz, background_band_hz):
    """Half-open target `[lo, hi)`, closed background `[lo, hi]` — so the shared 1.5 m edge
    bin belongs to the background band and to exactly one band (`:929-931`)."""
    freqs = np.asarray(freqs, dtype=float)
    target = (freqs >= target_band_hz[0]) & (freqs < target_band_hz[1])
    background = (freqs >= background_band_hz[0]) & (freqs <= background_band_hz[1])
    return target, background


def _bin_range_mask(n_bins: int, bin_range) -> np.ndarray:
    mask = np.zeros(int(n_bins), dtype=bool)
    mask[int(bin_range[0]) : int(bin_range[1]) + 1] = True
    return mask


def physics_band_masks(config, band, n):
    """The two band masks over the half spectrum of an `n`-point FFT.

    10 GHz: the frozen range gates mapped through `beat_band_hz`, so the bins follow from
    the radar constants rather than being written down. 77 GHz: the frozen bin partition
    (DC alone vs 1..127), checked to be an exact partition of the half spectrum — the
    frozen `(1, 127)` assumes the 256-point Doppler FFT, and on any other length it would
    silently stop covering it.
    """
    n = int(n)
    n_half = n // 2
    if band == "10ghz":
        pre, cfg = config.preprocess, config.baselines
        target_hz = beat_band_hz(cfg.physics_target_range_m_10ghz, pre.bandwidth_hz, pre.chirp_time_s)
        background_hz = beat_band_hz(cfg.physics_background_range_m_10ghz, pre.bandwidth_hz,
                                     pre.chirp_time_s)
        freqs = np.arange(n_half) * (pre.fs_hz / n)
        target, background = band_masks_from_frequencies(freqs, target_hz, background_hz)
        if not target.any() or not background.any():
            raise ExpDProtocolError(
                f"the frozen 10 GHz physics gates map to {target_hz} / {background_hz} Hz, which "
                f"select {int(target.sum())} / {int(background.sum())} of {n_half} bins at "
                f"df={pre.fs_hz / n:.1f} Hz — an empty band has no power to ratio"
            )
        return target, background
    if band == "77ghz":
        cfg = config.baselines
        prf = 1.0 / config.preprocess77.chirp_time_s
        if abs(prf - cfg.physics_prf_hz_77ghz) > 1e-6:
            raise ExpDProtocolError(
                f"BaselineConfig.physics_prf_hz_77ghz is {cfg.physics_prf_hz_77ghz} but the "
                f"chirp time implies PRF={prf} — one of the two drifted"
            )
        static = _bin_range_mask(n_half, cfg.physics_static_band_bins_77ghz)
        motion = _bin_range_mask(n_half, cfg.physics_motion_band_bins_77ghz)
        if np.any(static & motion) or not np.all(static | motion):
            raise ExpDProtocolError(
                f"the frozen 77 GHz Doppler bands {cfg.physics_static_band_bins_77ghz} / "
                f"{cfg.physics_motion_band_bins_77ghz} are not an exact partition of the "
                f"{n_half} half-spectrum bins of an {n}-point FFT"
            )
        return static, motion
    raise ExpDError(f"unknown band {band!r} (expected one of {BANDS})")


def band_power_ratio_scalar(power, target_mask, background_mask) -> float:
    """The frozen scalar `log10((P_t + eps)/(P_b + eps))` with `eps = 1e-12 (P_t + P_b)`.

    eps is added to BOTH terms so `P_target = 0` gives a finite floor rather than -inf: QC
    guarantees energy somewhere in the combined gate but not necessarily in the target band.
    The one case eps cannot rescue is zero energy in BOTH bands, where eps is itself 0 and
    the ratio is a genuine 0/0 — that stops rather than emitting NaN into a fitted model.
    """
    p_target = float(np.sum(np.asarray(power, dtype=float)[target_mask]))
    p_background = float(np.sum(np.asarray(power, dtype=float)[background_mask]))
    eps = PHYSICS_EPS_SCALE * (p_target + p_background)
    if eps <= 0.0:
        raise ExpDError(
            "physics baseline: no energy in either the target or the background band, so the "
            "frozen eps is 0 and the power ratio is undefined (0/0) — refusing to emit NaN"
        )
    return float(np.log10((p_target + eps) / (p_background + eps)))


def physics_frame_scalar(config, band, signal) -> float:
    """One QC-passed frame -> one scalar, with the frozen finite-output assertion."""
    power = half_spectrum_power(signal)
    target, background = physics_band_masks(config, band, int(np.asarray(signal).shape[0]))
    value = band_power_ratio_scalar(power, target, background)
    if not math.isfinite(value):
        raise ExpDError(f"physics scalar is not finite ({value!r}) — the frozen output assertion")
    return value


# ------------------------------------------- the two cheap baselines under the same folds


@dataclass
class CheapSpine:
    """Session-level rows for one deterministic baseline — one row per eligible session,
    which is the analysis unit (`:567-588`). Frames are already gone by here: the physics
    scalar arrived through the frozen median (O-M9-4) and the session-index baseline reads
    no radar data at all."""

    band: str
    family: str
    subjects: np.ndarray
    session_idx: np.ndarray
    delta_m_pct: np.ndarray
    n_frames: np.ndarray                 # frames the family aggregated; 0 for session_index
    physics_scalar: np.ndarray | None = None


@dataclass
class BaselineFoldResult:
    """One outer fold of one deterministic baseline. `inner_score` is what the composite
    procedure selects on — the same quantity the CNN path's `select_candidate` scores, so
    the two are comparable inside a fold."""

    band: str
    family: str
    fold_id: int
    test_subject: int
    test_session_idx: np.ndarray
    test_truth: np.ndarray
    test_predictions: np.ndarray
    test_n_frames: np.ndarray
    inner_scores: list
    inner_score: float
    n_inner_folds: int
    fitted_coefficients: dict
    fits: list


def selectable_folds(subjects) -> list:
    """The selectable outer folds, in canonical order. Their POSITION in this list is the
    `fold_id` every artifact and every array task uses — not the subject id (§5 trap 14)."""
    return [f for f in harness.nested_loso_splits(sorted(int(s) for s in subjects)) if f.selectable]


def build_physics_spine(config, band, sessions, store_dir) -> CheapSpine:
    """Per frame: the range/Doppler power ratio of the stored unstandardized raw signal.
    Per session: the MEDIAN over that session's QC-passed frames (O-M9-4, the same frozen
    `frame_to_session_aggregation` every CNN baseline uses)."""
    require_frozen_training_protocol(config.baselines)
    if band not in BANDS:
        raise ExpDError(f"unknown band {band!r} (expected one of {BANDS})")
    key = PHYSICS_SIGNAL_KEY[band]

    scalars, n_frames = [], []
    for session in sessions:
        store = store_mod.read_session_store(band, session["subject"], session["session_name"],
                                             store_dir)
        try:
            if key not in store:
                raise ExpDError(
                    f"store for {band} {session['subject']}/{session['session_name']} has no "
                    f"{key!r} — the physics baseline needs a schema-v2 store"
                )
            signals = np.asarray(store[key])
        finally:
            store.close()

        expected = list(session["frame_ids"])
        if signals.shape[0] != len(expected):
            raise ExpDError(
                f"store for {band} {session['subject']}/{session['session_name']} holds "
                f"{signals.shape[0]} {key!r} frames but the QC spine selected {len(expected)} — "
                "the frame order is not aligned"
            )
        per_frame = [physics_frame_scalar(config, band, signals[i]) for i in range(len(expected))]
        scalars.append(float(np.median(per_frame)))
        n_frames.append(len(expected))

    return CheapSpine(
        band=band, family="physics",
        subjects=np.array([int(s["subject"]) for s in sessions], dtype=int),
        session_idx=np.array([int(s["session_idx"]) for s in sessions], dtype=int),
        delta_m_pct=np.array([float(s["delta_m_pct"]) for s in sessions], dtype=float),
        n_frames=np.array(n_frames, dtype=int),
        physics_scalar=np.array(scalars, dtype=float),
    )


def build_session_index_spine(config, band, sessions) -> CheapSpine:
    """The clock/confound reference: the session label is the only input, so this spine
    carries no radar quantity at all and `n_frames` is 0 — the family aggregates none."""
    require_frozen_training_protocol(config.baselines)
    if band not in BANDS:
        raise ExpDError(f"unknown band {band!r} (expected one of {BANDS})")
    return CheapSpine(
        band=band, family="session_index",
        subjects=np.array([int(s["subject"]) for s in sessions], dtype=int),
        session_idx=np.array([int(s["session_idx"]) for s in sessions], dtype=int),
        delta_m_pct=np.array([float(s["delta_m_pct"]) for s in sessions], dtype=float),
        n_frames=np.zeros(len(sessions), dtype=int),
    )


def _least_squares_line(x, y) -> tuple[float, float]:
    """Slope + intercept of the frozen one-dimensional least-squares fit (`:942-944`)."""
    x = np.asarray(x, dtype=float)
    design = np.column_stack([x, np.ones(x.size)])
    solution, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)
    return float(solution[0]), float(solution[1])


def _fit_predict_cheap(spine: CheapSpine, train_subjects, eval_rows, *, role):
    """Fit this baseline on `train_subjects`' sessions only and predict `eval_rows`.

    Returns (predictions, FitRecord, the fitted coefficients as plain data). Both families
    go through here so the fold loop below is written once and the two cannot drift in
    which rows they are allowed to see.
    """
    if spine.family == "physics":
        train_rows = np.isin(spine.subjects, sorted(train_subjects))
        slope, intercept = _least_squares_line(
            spine.physics_scalar[train_rows], spine.delta_m_pct[train_rows]
        )
        record = FitRecord(
            "physics_linear_fit", role, frozenset(train_subjects),
            {"slope": np.asarray(slope, dtype=float),
             "intercept": np.asarray(intercept, dtype=float),
             "n_train_sessions": np.asarray(int(train_rows.sum()), dtype=np.int64)},
        )
        predictions = slope * spine.physics_scalar[eval_rows] + intercept
        return predictions, record, {"slope": slope, "intercept": intercept}

    if spine.family == "session_index":
        # reused VERBATIM, band-agnostic and shared between bands (`:850-854`, `:915-916`)
        outcome = baseline_models.fit_session_index_baseline(
            spine.subjects, spine.session_idx, spine.delta_m_pct, train_subjects, role=role
        )
        predictions = baseline_models.predict_session_index(
            outcome.model, spine.session_idx[eval_rows]
        )
        return predictions, outcome.fit_record, outcome.model

    raise ExpDError(
        f"{spine.family!r} is not a deterministic Exp D baseline (expected one of "
        f"{DETERMINISTIC_FAMILIES})"
    )


def run_cheap_baseline(spine: CheapSpine) -> list[BaselineFoldResult]:
    """One deterministic baseline over every selectable outer fold — the identical folds,
    QC-passed population and analysis unit as Exp A (`:917-919`).

    Each fold is also scored on its INNER folds (fit inner-train, score inner-val
    session-MAE with the same `subject_balanced_mae` the CNN path checkpoints on), so the
    composite procedure has a real inner-CV score to select on. K = 1: there is no grid and
    nothing is selected here.
    """
    results = []
    for fold_id, fold in enumerate(selectable_folds(set(spine.subjects.tolist()))):
        fits, inner_scores = [], []
        for inner in fold.inner_folds:
            val_rows = np.isin(spine.subjects, sorted(inner.val_subjects))
            predictions, record, _ = _fit_predict_cheap(
                spine, inner.train_subjects, val_rows, role="inner_train"
            )
            fits.append(record)
            inner_scores.append(subject_balanced_mae(
                spine.subjects[val_rows], spine.delta_m_pct[val_rows], predictions
            ))

        test_rows = spine.subjects == fold.test_subject
        predictions, record, coefficients = _fit_predict_cheap(
            spine, fold.train_subjects, test_rows, role="outer_train"
        )
        fits.append(record)
        results.append(BaselineFoldResult(
            band=spine.band, family=spine.family, fold_id=fold_id,
            test_subject=int(fold.test_subject),
            test_session_idx=spine.session_idx[test_rows],
            test_truth=spine.delta_m_pct[test_rows],
            test_predictions=np.asarray(predictions, dtype=float),
            test_n_frames=spine.n_frames[test_rows],
            inner_scores=[float(v) for v in inner_scores],
            inner_score=float(np.mean(inner_scores)) if inner_scores else float("nan"),
            n_inner_folds=len(fold.inner_folds),
            fitted_coefficients=coefficients,
            fits=fits,
        ))
    return results


def run_physics(config, band, sessions, store_dir) -> list[BaselineFoldResult]:
    return run_cheap_baseline(build_physics_spine(config, band, sessions, store_dir))


def run_session_index(config, band, sessions) -> list[BaselineFoldResult]:
    return run_cheap_baseline(build_session_index_spine(config, band, sessions))


def assert_mechanism_ok_d(results, subjects) -> None:
    """Structural checks that reveal no performance, matching `exp_a._assert_mechanism_ok`:
    fold-role disjointness, and every recorded fit's subject set excluding the held-out
    subject."""
    for fold in harness.nested_loso_splits(sorted(int(s) for s in subjects)):
        if not fold.selectable:
            continue
        assert fold.test_subject not in fold.train_subjects
        for inner in fold.inner_folds:
            assert inner.train_subjects.isdisjoint(inner.val_subjects)
            assert fold.test_subject not in inner.train_subjects
            assert fold.test_subject not in inner.val_subjects
    for result in results:
        for record in result.fits:
            assert result.test_subject not in record.subjects
        assert len(result.test_predictions) == len(result.test_truth) == len(result.test_session_idx)


# ================================================ the four per-family merged artifacts
#
# ONE schema for every Exp D family, CNN and deterministic alike, so the comparison stage
# and the exploratory frame-split reader need no per-family special-casing and every family
# is independently auditable and regenerable.


PREDICTION_COLUMNS = ["fold_id", "subject", "session_idx", "seed", "y_true_delta_m_pct",
                      "y_pred", "n_frames_aggregated"]
SELECTION_COLUMNS = ["fold_id", "test_subject", "selected_config", "learning_rate",
                     "weight_decay", "epoch_budget", "selected_epoch_counts",
                     "per_config_inner_scores", "inner_score", "n_inner_folds",
                     "fitted_coefficients"]
PER_SUBJECT_COLUMNS = ["subject", "per_seed_session_mae", "seed_averaged_session_mae",
                       "n_sessions"]


def cnn_prediction_rows(result: CnnFoldResult, fold_id) -> list[dict]:
    """One row per (session, seed): the five seeds are scored separately and never
    ensembled (`:644-649`), so they stay five rows, not one averaged one."""
    return [
        {"fold_id": int(fold_id),
         "subject": int(result.test_session_subjects[k]),
         "session_idx": int(result.test_session_idx[k]),
         "seed": int(outcome.seed),
         "y_true_delta_m_pct": float(result.test_session_truth[k]),
         "y_pred": float(outcome.session_predictions[k]),
         "n_frames_aggregated": int(result.test_n_frames_aggregated[k])}
        for outcome in result.seed_outcomes
        for k in range(len(result.test_session_idx))
    ]


def cheap_prediction_rows(result: BaselineFoldResult) -> list[dict]:
    """A deterministic family writes seed 1 ONCE — never five identical copies, which would
    fake five observations for the seed-collapse rules downstream."""
    return [
        {"fold_id": int(result.fold_id), "subject": int(result.test_subject),
         "session_idx": int(result.test_session_idx[k]), "seed": 1,
         "y_true_delta_m_pct": float(result.test_truth[k]),
         "y_pred": float(result.test_predictions[k]),
         "n_frames_aggregated": int(result.test_n_frames[k])}
        for k in range(len(result.test_session_idx))
    ]


def cnn_selection_row(result: CnnFoldResult, fold_id) -> dict:
    return {
        "fold_id": int(fold_id), "test_subject": int(result.test_subject),
        "selected_config": f"cfg{result.selected_config_index}",
        "learning_rate": float(result.selected_config["lr"]),
        "weight_decay": float(result.selected_config["weight_decay"]),
        "epoch_budget": int(result.epoch_budget),
        "selected_epoch_counts": [int(v) for v in result.selected_epoch_counts],
        "per_config_inner_scores": [float(s.inner_val_mae) for s in result.per_config_scores],
        "inner_score": float(result.per_config_scores[result.selected_config_index].inner_val_mae),
        "n_inner_folds": int(result.n_inner_folds),
        "fitted_coefficients": {},
    }


def cheap_selection_row(result: BaselineFoldResult) -> dict:
    """K = 1, so there is no grid selection: the row carries the fitted coefficients
    instead, and the file exists for every family so its reader needs no special-casing."""
    return {
        "fold_id": int(result.fold_id), "test_subject": int(result.test_subject),
        "selected_config": "n/a", "learning_rate": "", "weight_decay": "", "epoch_budget": "",
        "selected_epoch_counts": [],
        "per_config_inner_scores": [float(result.inner_score)],
        "inner_score": float(result.inner_score),
        "n_inner_folds": int(result.n_inner_folds),
        "fitted_coefficients": result.fitted_coefficients,
    }


def _write_csv(path, columns, rows) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[c] for c in columns])


def _prediction_matrix(rows):
    """Canonical session rows + a (n_seeds, n_sessions) prediction matrix.

    Fails closed on a duplicated or missing (session, seed) cell rather than quietly
    reshaping around it — an incomplete matrix would silently change every seed-collapse.
    """
    seeds = sorted({int(r["seed"]) for r in rows})
    sessions = sorted({(int(r["subject"]), int(r["session_idx"])) for r in rows})
    session_at = {key: i for i, key in enumerate(sessions)}
    seed_at = {seed: k for k, seed in enumerate(seeds)}

    truth = np.full(len(sessions), np.nan)
    predictions = np.full((len(seeds), len(sessions)), np.nan)
    for row in rows:
        i = session_at[(int(row["subject"]), int(row["session_idx"]))]
        k = seed_at[int(row["seed"])]
        if not math.isnan(predictions[k, i]):
            raise ExpDError(
                f"duplicate prediction row for subject {row['subject']} session "
                f"{row['session_idx']} seed {row['seed']}"
            )
        predictions[k, i] = float(row["y_pred"])
        truth[i] = float(row["y_true_delta_m_pct"])
    if np.isnan(predictions).any():
        missing = [(sessions[i], seeds[k]) for k, i in zip(*np.nonzero(np.isnan(predictions)))]
        raise ExpDError(
            f"the predictions are not a complete session x seed grid — missing {missing[:5]}"
        )
    return (np.array([s for s, _ in sessions], dtype=int),
            np.array([i for _, i in sessions], dtype=int), truth, predictions, seeds)


def _per_subject_session_mae(subjects, truth, predictions_by_seed) -> dict:
    """The frozen ADDITIVE collapse (`:1193-1199`): per subject, the mean over seeds of that
    seed's mean |error| across the subject's sessions. A deterministic family averages one
    value, which is that value."""
    out = {}
    for subject in sorted(set(subjects.tolist())):
        rows = subjects == subject
        per_seed = [float(np.abs(truth[rows] - predictions_by_seed[k, rows]).mean())
                    for k in range(predictions_by_seed.shape[0])]
        out[int(subject)] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)),
                             "n_sessions": int(rows.sum())}
    return out


def _ci_dict(c: M.BootstrapCI) -> dict:
    return {"point": c.point, "low": c.low, "high": c.high, "method": c.method,
            "n_eval": c.n_eval, "n_skipped": c.n_skipped, "unreliable": c.unreliable}


def write_family_artifacts(band, family, out_dir, *, prediction_rows, selection_rows,
                           deterministic, bootstrap_b, rng_seed, skip_threshold_pct,
                           lineage) -> dict:
    """Write one family's four merged artifacts and return {name: path}.

    The metrics JSON carries the family's MAE, RMSE and pooled Pearson r under the frozen
    METRIC-TYPE-AWARE seed collapse (`:1193-1204`): additive per-subject averaging for MAE,
    recompute-per-seed-then-average inside each resample for RMSE and r.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_folds = sorted({int(r["fold_id"]) for r in prediction_rows})
    selection_folds = sorted(int(r["fold_id"]) for r in selection_rows)
    if prediction_folds != selection_folds:
        raise ExpDError(
            f"{family} {band}: the predictions cover folds {prediction_folds} but the selection "
            f"table covers {selection_folds} — refusing to write an inconsistent family"
        )

    subjects, _session_idx, truth, predictions, seeds = _prediction_matrix(prediction_rows)
    per_subject = _per_subject_session_mae(subjects, truth, predictions)
    subject_ids = sorted(per_subject)

    mae_ci = M.subject_cluster_bootstrap(
        np.array([per_subject[s]["mean"] for s in subject_ids]),
        b=bootstrap_b, rng_seed=rng_seed + _rng_offset(family, "mae"),
    )
    rmse_ci = M.subject_cluster_bootstrap_pooled(
        subjects, truth, predictions, M.session_rmse,
        b=bootstrap_b, rng_seed=rng_seed + _rng_offset(family, "rmse"),
        skip_threshold_pct=skip_threshold_pct,
    )
    r_ci = M.subject_cluster_bootstrap_pooled(
        subjects, truth, predictions, M.pooled_pearson_r,
        b=bootstrap_b, rng_seed=rng_seed + _rng_offset(family, "pooled_pearson_r"),
        skip_threshold_pct=skip_threshold_pct,
    )

    metrics = {
        "conditional_exploratory": True,
        "band": band,
        "family": family,
        "deterministic": bool(deterministic),
        "n_eval": len(subject_ids),
        "n_sessions": int(len(truth)),
        "n_seeds": len(seeds),
        "seed_set": [int(s) for s in seeds],
        "fold_ids": prediction_folds,
        "per_subject_session_mae": {str(s): per_subject[s]["mean"] for s in subject_ids},
        "subject_balanced_mae": _ci_dict(mae_ci),
        "session_rmse": _ci_dict(rmse_ci),
        "pooled_pearson_r": _ci_dict(r_ci),
        "lineage": dict(lineage),
    }

    paths = {
        "predictions": out_dir / f"predictions_{family}_{band}.csv",
        "metrics": out_dir / f"metrics_{family}_{band}.json",
        "selection": out_dir / f"selection_{family}_{band}.csv",
        "per_subject": out_dir / f"per_subject_{family}_{band}.csv",
    }
    _write_csv(paths["predictions"], PREDICTION_COLUMNS, prediction_rows)
    _write_csv(paths["selection"], SELECTION_COLUMNS,
               sorted(selection_rows, key=lambda r: int(r["fold_id"])))
    _write_csv(paths["per_subject"], PER_SUBJECT_COLUMNS, [
        {"subject": s, "per_seed_session_mae": per_subject[s]["per_seed"],
         "seed_averaged_session_mae": per_subject[s]["mean"],
         "n_sessions": per_subject[s]["n_sessions"]}
        for s in subject_ids
    ])
    paths["metrics"].write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    return paths


@dataclass
class FamilyArtifacts:
    """One family's merged artifacts, loaded and cross-checked — the only thing the
    comparison stage is allowed to read."""

    band: str
    family: str
    run_dir: Path
    deterministic: bool
    subjects: np.ndarray
    session_idx: np.ndarray
    y_true: np.ndarray
    predictions_by_seed: np.ndarray
    seeds: list
    per_subject_mae: dict
    n_seeds_by_subject: dict
    fold_ids: list
    test_subject_by_fold: dict
    inner_score_by_fold: dict
    metrics: dict


def load_family_artifacts(run_dir, band, family) -> FamilyArtifacts:
    """Load and VALIDATE one family's four artifacts.

    The comparison stage is not allowed to read a family that is incomplete or internally
    inconsistent, so the checks are here rather than in the caller: all four files present;
    the metrics JSON's per-subject vector recomputes exactly from the predictions CSV; the
    selection table's epoch budget is the median of the counts the row itself lists; the two
    tables cover the same folds.
    """
    run_dir = Path(run_dir)
    names = {
        "predictions": f"predictions_{family}_{band}.csv",
        "metrics": f"metrics_{family}_{band}.json",
        "selection": f"selection_{family}_{band}.csv",
        "per_subject": f"per_subject_{family}_{band}.csv",
    }
    for name in names.values():
        if not (run_dir / name).is_file():
            raise ExpDError(
                f"Exp D family {family} ({band}) is incomplete under {run_dir}: {name} is missing "
                "— refusing to read an incomplete family set"
            )

    prediction_rows = list(csv.DictReader((run_dir / names["predictions"]).open(encoding="utf-8")))
    selection_rows = list(csv.DictReader((run_dir / names["selection"]).open(encoding="utf-8")))
    metrics = json.loads((run_dir / names["metrics"]).read_text(encoding="utf-8"))

    subjects, session_idx, truth, predictions, seeds = _prediction_matrix(prediction_rows)
    per_subject = _per_subject_session_mae(subjects, truth, predictions)

    recorded = {int(k): float(v) for k, v in metrics["per_subject_session_mae"].items()}
    recomputed = {s: per_subject[s]["mean"] for s in per_subject}
    if set(recorded) != set(recomputed) or any(
        not math.isclose(recorded[s], recomputed[s], rel_tol=1e-12, abs_tol=1e-12)
        for s in recorded
    ):
        raise ExpDError(
            f"{names['metrics']} does not recompute from {names['predictions']}: its per-subject "
            f"session-MAE vector is {recorded} but the predictions give {recomputed}"
        )

    prediction_folds = sorted({int(r["fold_id"]) for r in prediction_rows})
    selection_folds = sorted(int(r["fold_id"]) for r in selection_rows)
    if prediction_folds != selection_folds or prediction_folds != sorted(metrics["fold_ids"]):
        raise ExpDError(
            f"Exp D family {family} ({band}) covers folds {prediction_folds} in "
            f"{names['predictions']}, {selection_folds} in {names['selection']} and "
            f"{sorted(metrics['fold_ids'])} in {names['metrics']} — they must agree exactly"
        )

    inner_score_by_fold, test_subject_by_fold = {}, {}
    for row in selection_rows:
        fold_id = int(row["fold_id"])
        inner_score_by_fold[fold_id] = float(row["inner_score"])
        test_subject_by_fold[fold_id] = int(row["test_subject"])
        counts = _parse_literal(row["selected_epoch_counts"])
        if row["epoch_budget"] not in ("", None) and counts:
            if int(row["epoch_budget"]) != epoch_budget_from(counts):
                raise ExpDError(
                    f"{names['selection']} fold {fold_id}: the epoch budget "
                    f"{row['epoch_budget']} is not the median of its own listed epoch counts "
                    f"{counts} (= {epoch_budget_from(counts)})"
                )

    return FamilyArtifacts(
        band=band, family=family, run_dir=run_dir,
        deterministic=bool(metrics["deterministic"]),
        subjects=subjects, session_idx=session_idx, y_true=truth,
        predictions_by_seed=predictions, seeds=seeds,
        per_subject_mae={s: per_subject[s]["mean"] for s in per_subject},
        n_seeds_by_subject={s: len(per_subject[s]["per_seed"]) for s in per_subject},
        fold_ids=prediction_folds, test_subject_by_fold=test_subject_by_fold,
        inner_score_by_fold=inner_score_by_fold, metrics=metrics,
    )


def _parse_literal(text):
    """Read back a list/dict cell written by `csv.writer` (which used `repr`). Only ever
    applied to this module's own artifacts."""
    if text in ("", None):
        return []
    return ast.literal_eval(text) if isinstance(text, str) else text


# ==================================================== the CNN fold-array shard and merge
#
# The M8 session-specific machinery (C19-C22), reused rather than reinvented, plus the one
# field that contract had no need for: an authoritative per-fold ROW CENSUS. Nothing already
# recorded can catch a truncated or stale shard — `record_run`'s manifest holds only cohort
# totals and `fold_manifest` holds fold roles — so a shard that silently dropped test rows
# would have no reference to be rejected by.


def _sha256_lines(lines) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def realized_fold_census(frames: FramesD, fold, seeds) -> dict:
    """This fold's held-out test rows, as counts plus TWO hashes.

    Two, not one, because the artifacts they guard are different files: the frame hash is
    opaque to the session-level predictions CSV, so a CSV that substituted a session while
    preserving `n_session_rows` would pass a count-only check. Seed is deliberately NOT part
    of either identity — it is the CSV's third column and is validated separately, as an
    exact `session identities x seed_set` cross product.
    """
    rows = np.flatnonzero(frames.subjects == fold.test_subject)
    frame_ids, session_ids = [], set()
    for i in rows:
        subject = int(frames.subjects[i])
        session_idx = int(frames.session_idx[frames.session_row[i]])
        frame_ids.append(f"{subject}|{session_idx}|{int(frames.frame_ids[i])}")
        session_ids.add(f"{subject}|{session_idx}")
    return {
        "test_subject": int(fold.test_subject),
        "n_frame_rows": len(frame_ids),
        "n_session_rows": len(session_ids),
        "frame_rows_sha256": _sha256_lines(sorted(frame_ids)),
        "session_rows_sha256": _sha256_lines(sorted(session_ids)),
        "seed_set": [int(s) for s in seeds],
    }


def expected_test_rows_by_fold(frames: FramesD, seeds) -> dict:
    """The authoritative census the merge validates every shard against, keyed by the
    fold's POSITION in the selectable-fold list (§5 trap 14). JSON-safe string keys, since
    this travels inside `record_run`'s `extra`."""
    return {
        str(fold_id): realized_fold_census(frames, fold, seeds)
        for fold_id, fold in enumerate(selectable_folds(set(frames.subjects.tolist())))
    }


def _shard_stem(family, band, fold_id) -> str:
    return f"exp_d_{family}_{band}_fold{int(fold_id)}"


def write_fold_shard(result: CnnFoldResult, frames: FramesD, fold, fold_id, run_dir, *,
                     band, family, seeds, run_group_id, analysis_commit, config_hash):
    """One array task's output: the shard JSON (lineage + its own REALIZED census +
    the fold's selection row) and the fold's predictions CSV, in the merged schema."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = _shard_stem(family, band, fold_id)

    csv_path = run_dir / f"{stem}_predictions.csv"
    _write_csv(csv_path, PREDICTION_COLUMNS, cnn_prediction_rows(result, fold_id))

    shard = {
        "run_group_id": run_group_id,
        "band": band,
        "family": family,
        "fold_id": int(fold_id),
        "test_subject": int(result.test_subject),
        "analysis_commit": analysis_commit,
        "config_hash": config_hash,
        "seed_convention": "one_row_per_seed",
        "census": realized_fold_census(frames, fold, seeds),
        "selection": cnn_selection_row(result, fold_id),
    }
    json_path = run_dir / f"{stem}.json"
    json_path.write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_path, csv_path


def write_noop_marker(run_dir, *, band, family, fold_id, reason) -> Path:
    """A named no-op (§5 trap 14): the 16-task array is fixed-size, so a task whose index
    exceeds the selectable-fold list exits 0 having written this — never an error, and never
    a silent absence the merge cannot tell apart from a crash."""
    path = Path(run_dir) / f"{_shard_stem(family, band, fold_id)}.noop.json"
    path.write_text(json.dumps({
        "band": band, "family": family, "fold_id": int(fold_id), "state": "noop",
        "reason": reason,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_shard(shard, *, band, family, fold_id, run_group_id, analysis_commit,
                    config_hash, census) -> None:
    def check(field_name, expected, found):
        if found != expected:
            raise ExpDError(
                f"Exp D shard fold {fold_id} ({family}, {band}): {field_name} mismatch "
                f"(expected {expected!r}, found {found!r}) — refusing to merge it"
            )

    check("band", band, shard.get("band"))
    check("family", family, shard.get("family"))
    check("fold_id", int(fold_id), shard.get("fold_id"))
    check("run_group_id", run_group_id, shard.get("run_group_id"))
    check("analysis_commit", analysis_commit, shard.get("analysis_commit"))
    check("config_hash", config_hash, shard.get("config_hash"))

    realized = shard.get("census") or {}
    for key in ("test_subject", "n_frame_rows", "n_session_rows", "frame_rows_sha256",
                "session_rows_sha256", "seed_set"):
        check(key, census[key], realized.get(key))


def _validate_fold_predictions(rows, census, *, fold_id, family, band, deterministic) -> None:
    """The predictions CSV must recompute the fold's own session identities and be the exact
    `session identities x seed_set` cross product — no duplicates, no absences."""
    identities = sorted({f"{int(r['subject'])}|{int(r['session_idx'])}" for r in rows})
    if _sha256_lines(identities) != census["session_rows_sha256"]:
        raise ExpDError(
            f"Exp D fold {fold_id} ({family}, {band}) predictions CSV: session_rows_sha256 "
            f"mismatch — its session identities are {identities}, which is not the fold's "
            "expected held-out session set"
        )
    if len(identities) != census["n_session_rows"]:
        raise ExpDError(
            f"Exp D fold {fold_id} ({family}, {band}) predictions CSV: n_session_rows is "
            f"{len(identities)}, expected {census['n_session_rows']}"
        )

    seeds = [1] if deterministic else list(census["seed_set"])
    found = [(f"{int(r['subject'])}|{int(r['session_idx'])}", int(r["seed"])) for r in rows]
    if len(found) != len(set(found)):
        raise ExpDError(
            f"Exp D fold {fold_id} ({family}, {band}) predictions CSV holds duplicate "
            "(session, seed) rows"
        )
    expected = {(identity, seed) for identity in identities for seed in seeds}
    if set(found) != expected:
        raise ExpDError(
            f"Exp D fold {fold_id} ({family}, {band}) predictions CSV is not the exact "
            f"session x seed_set cross product for seeds {seeds}: missing "
            f"{sorted(expected - set(found))[:5]}, unexpected {sorted(set(found) - expected)[:5]}"
        )


def merge_exp_d_folds(band, family, run_dir) -> dict:
    """Fail-closed merge of one (family, band) fold array.

    Every present shard is validated for lineage AND for an exact match of its realized
    census (both hashes, both counts, the seed set) against the run group's authoritative
    `expected_test_rows_by_fold`, and each fold's predictions CSV is re-validated on its own
    recomputed session identities and seed cross product. The family's summary is produced
    ONLY when every selectable fold is present and valid — a partial merge is a named,
    non-reportable state, never a silently smaller cohort.
    """
    run_dir = Path(run_dir)
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    extra = provenance["extra"]
    for field_name, expected in (("band", band), ("family", family)):
        if extra.get(field_name) != expected:
            raise ExpDError(
                f"run group {run_dir.name} is for {field_name}={extra.get(field_name)!r}, not "
                f"{expected!r} — refusing to merge across run groups"
            )
    analysis_commit = provenance["git"]["commit"]
    config_hash = extra["config_hash"]
    expected_by_fold = extra["expected_test_rows_by_fold"]
    deterministic = family in DETERMINISTIC_FAMILIES

    prediction_rows, selection_rows = [], []
    completed, noop, missing = [], [], []
    for fold_id in sorted(int(k) for k in expected_by_fold):
        stem = _shard_stem(family, band, fold_id)
        shard_path = run_dir / f"{stem}.json"
        if not shard_path.is_file():
            (noop if (run_dir / f"{stem}.noop.json").is_file() else missing).append(fold_id)
            continue
        census = expected_by_fold[str(fold_id)]
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        _validate_shard(shard, band=band, family=family, fold_id=fold_id,
                        run_group_id=run_dir.name, analysis_commit=analysis_commit,
                        config_hash=config_hash, census=census)
        rows = list(csv.DictReader((run_dir / f"{stem}_predictions.csv").open(encoding="utf-8")))
        _validate_fold_predictions(rows, census, fold_id=fold_id, family=family, band=band,
                                   deterministic=deterministic)
        prediction_rows += rows
        selection_rows.append(shard["selection"])
        completed.append(fold_id)

    # a marker for an index the selectable-fold list never had (the 16-task array is
    # fixed-size while N_eval can be smaller) is reported separately, not as "missing".
    prefix = f"exp_d_{family}_{band}_fold"
    out_of_range = sorted(
        int(path.name[len(prefix) : -len(".noop.json")])
        for path in run_dir.glob(f"{prefix}*.noop.json")
    )
    out_of_range = [fold_id for fold_id in out_of_range if str(fold_id) not in expected_by_fold]

    merged = {
        "band": band,
        "family": family,
        "run_group_id": run_dir.name,
        "analysis_commit": analysis_commit,
        "config_hash": config_hash,
        "expected_folds": sorted(int(k) for k in expected_by_fold),
        "completed_folds": completed,
        "noop_folds": noop,
        "noop_out_of_range_folds": out_of_range,
        "missing_folds": missing,
    }
    merged["complete"] = completed == merged["expected_folds"]
    merged["state"] = "complete" if merged["complete"] else "partial_non_reportable"
    if not merged["complete"]:
        merged["artifacts"] = None
        merged["note"] = (
            "partial merge: the family summary is deliberately NOT produced. A subset of the "
            "selectable folds is a named non-reportable state, not a smaller cohort."
        )
        return merged

    stats = provenance["config"]["stats"]
    merged["artifacts"] = write_family_artifacts(
        band, family, run_dir,
        prediction_rows=prediction_rows, selection_rows=selection_rows,
        deterministic=deterministic,
        bootstrap_b=int(stats["bootstrap_b"]),
        rng_seed=int(provenance["seed"]),
        skip_threshold_pct=float(stats["undefined_metric_skip_threshold_pct"]),
        lineage={"analysis_commit": analysis_commit, "config_hash": config_hash,
                 "run_group_id": run_dir.name},
    )
    return merged


# ========================================================== the frozen comparisons (§Stats)


@dataclass(frozen=True)
class RadarReference:
    """The radar side of every Exp D comparison, and the PROOF that O-M9-5's precondition
    was met. `summarize_exp_d` accepts nothing else, so a bare run directory cannot slip
    past the bit-identity assert."""

    band: str
    run_dir: Path
    m7_reference: Path
    per_subject_mae: dict
    n_seeds: int
    bit_identity_verified: bool


def _assert_config_sections_match(m9_config, m7_config) -> None:
    """Every config section except `paths` (which legitimately differs between the machine
    that ran M7 and the one running M9) must be identical."""
    for key in sorted(set(m9_config) | set(m7_config)):
        if key == "paths":
            continue
        if m9_config.get(key) != m7_config.get(key):
            raise ExpDProtocolError(
                f"the Exp A re-run's config section {key!r} differs from the M7 run's — the "
                "comparison's radar side must be the same analysis (O-M9-5)"
            )


def _radar_per_subject_mae(predictions_csv):
    """Per-subject session-MAE from Exp A's own predictions CSV, under the frozen additive
    seed collapse: average within a seed, then across seeds."""
    rows = list(csv.DictReader(Path(predictions_csv).open(encoding="utf-8")))
    errors: dict = {}
    for row in rows:
        key = (int(row["subject"]), int(row["seed"]))
        errors.setdefault(key, []).append(abs(float(row["y_true"]) - float(row["y_pred"])))
    subjects = sorted({s for s, _ in errors})
    seeds = sorted({k for _, k in errors})
    per_subject = {
        s: float(np.mean([float(np.mean(errors[(s, k)])) for k in seeds if (s, k) in errors]))
        for s in subjects
    }
    return per_subject, len(seeds)


def load_exp_a_radar(band, exp_a_run_dir, m7_reference_dir, *, analysis_commit) -> RadarReference:
    """O-M9-5: load the M9 Exp A re-run's predictions and REFUSE unless they are
    bit-identical to the M7 artifacts.

    A mismatch stops the milestone (§5 trap 17). Comparing against the fresh predictions
    anyway would convert a detected fault — a drifted store rebuild or drifted code — into a
    silent protocol change.
    """
    run_dir, reference = Path(exp_a_run_dir), Path(m7_reference_dir)
    provenance_path = run_dir / "provenance.json"
    if not provenance_path.is_file():
        raise ExpDProtocolError(f"the Exp A run dir {run_dir} has no provenance.json to validate")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    found_commit = (provenance.get("git") or {}).get("commit")
    if found_commit != analysis_commit:
        raise ExpDProtocolError(
            f"the Exp A re-run at {run_dir} records commit {found_commit!r}, not the analysis "
            f"commit {analysis_commit!r} — the comparison's radar side must be the M9 re-run"
        )
    reference_provenance = reference / "provenance.json"
    if reference_provenance.is_file():
        _assert_config_sections_match(
            provenance.get("config") or {},
            json.loads(reference_provenance.read_text(encoding="utf-8")).get("config") or {},
        )

    predictions = run_dir / f"predictions_{band}.csv"
    reference_predictions = reference / f"predictions_{band}.csv"
    for path in (predictions, reference_predictions):
        if not path.is_file():
            raise ExpDProtocolError(f"missing Exp A predictions artifact {path} (O-M9-5)")
    if sha256_file(predictions) != sha256_file(reference_predictions):
        raise ExpDProtocolError(
            f"{predictions} is NOT bit-identical to the M7 artifact {reference_predictions} — "
            "O-M9-5 makes this a milestone-stopping event: it means the store rebuild or the "
            "code drifted, and no comparison may be computed against either version"
        )

    per_subject, n_seeds = _radar_per_subject_mae(predictions)
    return RadarReference(
        band=band, run_dir=run_dir, m7_reference=reference, per_subject_mae=per_subject,
        n_seeds=n_seeds, bit_identity_verified=True,
    )


def _paired_vs_radar(radar, other, subjects, *, label, block, config, difference_key):
    """One comparison: per-subject metric differences over the N_eval subjects, Wilcoxon
    signed-rank plus a subject-cluster bootstrap CI on the mean difference (`:1400-1403`).
    Negative = radar better."""
    differences = np.array([radar[s] - other[s] for s in subjects], dtype=float)
    statistic, p_value = M.wilcoxon_signed_rank(differences)
    ci = M.mean_difference_ci(
        differences, b=config.stats.bootstrap_b,
        rng_seed=config.run.seed + _rng_offset(block, "difference_vs_radar"),
    )
    return {
        "comparison": f"radar_vs_{label}",
        "n_eval": len(subjects),
        "radar_mae": float(np.mean([radar[s] for s in subjects])),
        "baseline_mae": float(np.mean([other[s] for s in subjects])),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": p_value,
        difference_key: _ci_dict(ci),
    }


def summarize_exp_d(band, config, family_runs, exp_a_run) -> dict:
    """The frozen Exp D comparison report (`:1263-1281`).

    Three things, all pre-registered: the PRIMARY radar vs session-index comparison; the
    COMPOSITE procedure (one comparison, uncorrected) that picks the best of the three
    primary learned families inside each outer fold by inner CV; and the per-family
    exploratory Holm family of exactly 3. The matched-preprocessing ablations are reported
    descriptively and enter no comparison family (O-M9-3).

    The composite splices at the PER-SUBJECT METRIC level, not at the prediction level: a
    prediction-level splice is undefined across families of different seed multiplicity
    (each CNN has 5 per-seed prediction sets, physics has 1), and averaging predictions
    across seeds is forbidden outright (`:644-649`).
    """
    if not isinstance(exp_a_run, RadarReference) or not exp_a_run.bit_identity_verified:
        raise ExpDProtocolError(
            "summarize_exp_d needs a RadarReference produced by load_exp_a_radar — the radar "
            "side may only be read after the O-M9-5 bit-identity assert against the M7 "
            "prediction artifacts has passed"
        )
    if exp_a_run.band != band:
        raise ExpDProtocolError(
            f"the radar reference is for band {exp_a_run.band!r}, not {band!r}"
        )

    absent = [f for f in EXPD_FAMILIES if f not in family_runs]
    if absent:
        raise ExpDError(
            f"Exp D comparisons need all {len(EXPD_FAMILIES)} families; no run dir given for "
            f"{absent} — refusing to report an incomplete family set"
        )
    families = {f: load_family_artifacts(family_runs[f], band, f) for f in EXPD_FAMILIES}

    radar = exp_a_run.per_subject_mae
    subjects = sorted(radar)
    for name, artifacts in families.items():
        if sorted(artifacts.per_subject_mae) != subjects:
            raise ExpDError(
                f"family {name} covers subjects {sorted(artifacts.per_subject_mae)} but the "
                f"radar side covers {subjects} — comparisons must be paired on identical folds"
            )

    primary = _paired_vs_radar(radar, families["session_index"].per_subject_mae, subjects,
                               label="session_index", block="session_index", config=config,
                               difference_key="mean_difference_radar_minus_baseline")

    # the composite procedure: per outer fold, the best MEMBER by its own inner-CV score.
    # Ties break toward the earlier member of the declared COMPOSITE_MEMBERS order.
    fold_ids = sorted(families[COMPOSITE_MEMBERS[0]].fold_ids)
    composite_rows, composite_per_subject = [], {}
    for fold_id in fold_ids:
        ranked = sorted(
            (families[m].inner_score_by_fold[fold_id], rank, m)
            for rank, m in enumerate(COMPOSITE_MEMBERS)
        )
        inner_score, _rank, winner = ranked[0]
        subject = families[winner].test_subject_by_fold[fold_id]
        composite_per_subject[subject] = families[winner].per_subject_mae[subject]
        composite_rows.append({
            "subject": int(subject),
            "selected_family": winner,
            "inner_score": float(inner_score),
            "per_subject_mae": float(composite_per_subject[subject]),
            "n_seeds_averaged": int(families[winner].n_seeds_by_subject[subject]),
        })

    composite = _paired_vs_radar(radar, composite_per_subject, subjects, label="composite",
                                 block="composite", config=config,
                                 difference_key="mean_difference_radar_minus_composite")
    composite.update({
        "members": list(COMPOSITE_MEMBERS),
        "correction": config.stats.composite_baseline_comparison,
        "per_fold": composite_rows,
    })

    per_family = {
        m: _paired_vs_radar(radar, families[m].per_subject_mae, subjects, label=m, block=m,
                            config=config,
                            difference_key="mean_difference_radar_minus_baseline")
        for m in COMPOSITE_MEMBERS
    }
    holm = M.holm_adjusted([per_family[m]["wilcoxon_p"] for m in COMPOSITE_MEMBERS],
                           family_size=config.stats.holm_family_baseline_per_family)
    for m, adjusted in zip(COMPOSITE_MEMBERS, holm, strict=True):
        per_family[m]["holm_p"] = adjusted

    radar_ci = M.subject_cluster_bootstrap(
        np.array([radar[s] for s in subjects]), b=config.stats.bootstrap_b,
        rng_seed=config.run.seed + _rng_offset("radar", "mae"),
    )

    def descriptive(name):
        return {k: families[name].metrics[k]
                for k in ("subject_balanced_mae", "session_rmse", "pooled_pearson_r", "n_eval",
                          "n_seeds", "deterministic")}

    return {
        "conditional_exploratory": True,
        "band": band,
        "n_eval": len(subjects),
        "radar": {
            "per_subject_mae": {str(s): radar[s] for s in subjects},
            "subject_balanced_mae": _ci_dict(radar_ci),
            "n_seeds": exp_a_run.n_seeds,
        },
        "primary_vs_session_index": primary,
        "composite": composite,
        "per_family_exploratory": {
            "holm_family_size": config.stats.holm_family_baseline_per_family,
            "families": per_family,
        },
        # O-M9-3: ablations get their own numbers and NO comparison — no p-value, no Holm slot.
        "ablations_descriptive": {name: descriptive(name) for name in ABLATION_FAMILIES},
        "per_family_metrics": {name: descriptive(name) for name in EXPD_FAMILIES},
        "lineage": {
            "exp_a_run_dir": str(exp_a_run.run_dir),
            "m7_reference": str(exp_a_run.m7_reference),
            "bit_identity_verified": True,
            "family_run_dirs": {f: str(Path(family_runs[f])) for f in EXPD_FAMILIES},
        },
    }


COMPOSITE_COLUMNS = ["subject", "selected_family", "inner_score", "per_subject_mae",
                     "n_seeds_averaged"]


def write_exp_d_comparison_reports(summary, out_dir, band) -> dict:
    """`metrics_exp_d_{band}.json` + `composite_{band}.csv` — the latter so each outer
    fold's composite winner is auditable and the radar pairing is one join on `subject`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": out_dir / f"metrics_exp_d_{band}.json",
        "composite": out_dir / f"composite_{band}.csv",
    }
    paths["metrics"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    _write_csv(paths["composite"], COMPOSITE_COLUMNS, summary["composite"]["per_fold"])
    return paths
