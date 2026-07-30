"""Experiment D — the baselines, under Experiment A's identical LOSO harness.

**Milestone 9 step 7 builds the learned half only:** the per-frame spine
(`build_frames_d`) and the nested torch path for ONE outer fold (`run_cnn_family`) — the
frozen 6-config grid x early stopping x epoch budget x per-seed final refit. The physics
and session-index baselines, the per-family reports, the GPU fold-array shard/merge and the
comparison statistics are step 8 and are deliberately absent here.

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
and, at step 8, to the physics scalar (O-M9-4). Scoring is always session-level; frames
exist only so the network has rows to train on.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..features import store as store_mod
from ..models import cnn
from .harness import FitRecord
from .metrics import subject_balanced_mae
from .selection import CandidateScore, select_candidate

CNN_FAMILIES = cnn.CNN_FAMILIES
BANDS = ("10ghz", "77ghz")

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
