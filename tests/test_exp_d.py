"""T-M9-cnnpath: Exp D's CNN nested torch path on a deterministic CPU fixture.

Milestone 9 step 7 only — the frame spine and `run_cnn_family` (grid x early stopping x
epoch budget x per-seed refit). The physics/session-index baselines, the per-family
reports, the fold-array shard/merge and the comparison statistics are step 8 and are not
touched here.

Everything is asserted against the SPECIFICATION's arithmetic (plan §2.8, `:644-655`,
`:917-919`), never against a recorded run: the sampler weights, the median aggregation, the
epoch-budget median and the per-fit seed derivation are all hand-computed, and the sampler
trace is re-derived from `torch.multinomial` (which is what `WeightedRandomSampler` *is*)
rather than from the implementation's own output.

The fixture uses the 77 GHz signal shapes (256-sample slow time) because they are the
smallest real ones; the 10 GHz shapes are pinned in `test_cnn.py`, which needs no training.
"""

import dataclasses
import math

import numpy as np
import pytest
import torch

from dehyd.config import load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import harness
from dehyd.eval.exp_d import (
    FIT_SEED_BASE,
    CnnFoldResult,
    ExpDError,
    ExpDProtocolError,
    baseline_config_grid,
    build_frames_d,
    fit_seed,
    median_frame_to_session,
    run_cnn_family,
    session_sampler_weights,
)
from dehyd.eval.selection import CandidateScore, select_candidate
from dehyd.features.store import write_session_store
from dehyd.models import cnn

N_SUBJECTS = 4
N_FRAMES_PER_SESSION = 5
N_SLOWTIME = 256
SEEDS = (1, 2)


@pytest.fixture(scope="module")
def config():
    """The frozen config with a SMOKE-sized training budget. Only run-level values move
    (max_epochs, patience) — exactly the CLAUDE.md smoke rule: one code path, different
    config. The grid, batch size, betas, aggregation and checkpoint metric stay frozen."""
    base = load_config("configs/exp_a_regression.yaml")
    return dataclasses.replace(
        base,
        baselines=dataclasses.replace(
            base.baselines, max_epochs=5, early_stopping_patience=2
        ),
    )


# ------------------------------------------------------------------------- fixtures


def _sessions(n_subjects=N_SUBJECTS):
    """One record per session in `exp_a.build_sessions`'s shape."""
    out = []
    for subject in range(1, n_subjects + 1):
        for session_idx in range(5):
            out.append({
                "subject": subject,
                "session_idx": session_idx,
                "session_name": SESSION_NAMES[session_idx],
                "rel_path": f"s{subject}_{session_idx}.mat",
                "frame_ids": list(range(N_FRAMES_PER_SESSION)),
                "delta_m_pct": 0.0 if session_idx == 0 else -(0.4 * session_idx + 0.1 * subject),
            })
    return out


def _write_store_77(store_dir, sessions, seed=0):
    """A synthetic store v2 for the 77 GHz Exp D keys. Written directly (the producers are
    step 9); the KEY NAMES are the contract `build_frames_d` reads."""
    rng = np.random.default_rng(seed)
    for s in sessions:
        n = len(s["frame_ids"])
        # a subject/session-dependent tone + noise, so the target is genuinely learnable
        t = np.arange(N_SLOWTIME) / N_SLOWTIME
        tone = np.sin(2 * np.pi * (3 + s["session_idx"]) * t) * (1.0 + 0.2 * s["subject"])
        raw = tone[None, :] + rng.normal(scale=0.3, size=(n, N_SLOWTIME))
        # a genuinely DIFFERENT array under the same key family, so a swapped store key
        # (or a builder reading the wrong one) is visible rather than coincidentally equal
        matched = 0.5 * tone[None, None, :] + rng.normal(scale=0.5, size=(n, 2, N_SLOWTIME))
        write_session_store(
            "77ghz", s["subject"], s["session_name"],
            {"sig__raw_slowtime": raw, "sig__matched_iq": matched},
            {"n_frames": n}, store_dir,
        )


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("expd_store")
    sessions = _sessions()
    _write_store_77(path, sessions)
    return path, sessions


def _fold(test_subject, n_subjects=N_SUBJECTS):
    return next(
        f for f in harness.nested_loso_splits(list(range(1, n_subjects + 1)))
        if f.test_subject == test_subject
    )


def _mutate_held_out(frames, subject, seed=99):
    """Eligibility-preserving value mutation of the held-out subject's frames AND targets —
    the T18 pattern. Membership (subject/session/frame ids) is untouched."""
    rng = np.random.default_rng(seed)
    mutated = dataclasses.replace(frames, X=frames.X.copy(), y=frames.y.copy())
    rows = mutated.subjects == subject
    mutated.X[rows] = rng.normal(size=mutated.X[rows].shape) * 10 + 5
    mutated.y[rows] = rng.normal(size=int(rows.sum())) * 10 + 5
    return mutated


def _state_bytes(fits, quantity):
    record = next(f for f in fits if f.quantity == quantity)
    return b"".join(k.encode() + v.tobytes() for k, v in sorted(record.params.items()))


# ----------------------------------------------------------- the frozen 6-config grid


def test_baseline_config_grid_is_the_frozen_six_within_the_budget(config):
    """`:917-919` budget parity: 6 configs per learned family, <= K = 12."""
    grid = baseline_config_grid(config.model_grid)
    assert grid == [
        {"lr": 3e-4, "weight_decay": 0.0}, {"lr": 3e-4, "weight_decay": 1e-4},
        {"lr": 1e-3, "weight_decay": 0.0}, {"lr": 1e-3, "weight_decay": 1e-4},
        {"lr": 3e-3, "weight_decay": 0.0}, {"lr": 3e-3, "weight_decay": 1e-4},
    ]
    assert len(grid) == 6 <= config.search_10ghz.budget_k == 12
    assert len(grid) <= config.search_77ghz.budget_k


# ------------------------------------------------------------ the per-fit seed derivation


def test_fit_seed_is_the_named_hand_computed_derivation():
    """index = ((fold_id*16 + config_index)*9 + inner_slot)*16 + seed, offset by
    FIT_SEED_BASE + run_seed. For (run_seed 7, fold 3, config 2, inner 1, seed 5):
        ((3*16 + 2)*9 + 1)*16 + 5 = (450 + 1)*16 + 5 = 7221  ->  7 + 900000 + 7221
    and the final refit (inner_fold = -1) takes the reserved slot 8:
        ((3*16 + 2)*9 + 8)*16 + 5 = 458*16 + 5 = 7333        ->  7 + 900000 + 7333
    """
    assert FIT_SEED_BASE == 900_000
    assert fit_seed(7, fold_id=3, config_index=2, inner_fold=1, seed=5) == 907228
    assert fit_seed(7, fold_id=3, config_index=2, inner_fold=-1, seed=5) == 907340


def test_fit_seed_is_distinct_over_every_realistic_fit_of_a_run():
    """A collision would make two different fits share a sampler draw and an init — the
    exact bleed §5 trap 8 is about."""
    seen = {
        fit_seed(20260730, fold_id=f, config_index=c, inner_fold=i, seed=s)
        for f in range(16) for c in range(6) for i in list(range(5)) + [-1] for s in (1, 2, 3, 4, 5)
    }
    assert len(seen) == 16 * 6 * 6 * 5


def test_fit_seed_refuses_an_index_outside_its_named_radix():
    with pytest.raises(ExpDError):
        fit_seed(1, fold_id=3, config_index=99, inner_fold=0, seed=1)
    with pytest.raises(ExpDError):
        fit_seed(1, fold_id=3, config_index=0, inner_fold=9, seed=1)


# -------------------------------------------------------------- sampler weights (hand)


def test_session_sampler_weights_are_one_over_the_sessions_frame_count():
    """The literal frozen weight `1/frames_in_session` — NOT a mean-1 renormalized variant.
    Session 0 has 3 frames, session 1 has 2, session 2 has 1."""
    session_row = np.array([0, 0, 0, 1, 1, 2])
    assert session_sampler_weights(session_row).tolist() == [
        1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2, 1.0
    ]


def test_session_sampler_weights_give_every_session_the_same_total_mass():
    """The point of the weight: a 20-frame session and a 2-frame session are drawn equally
    often in expectation, so the loss is session-balanced without a second weighting."""
    session_row = np.array([0] * 20 + [1] * 2)
    weights = session_sampler_weights(session_row)
    assert weights[:20].sum() == pytest.approx(1.0)
    assert weights[20:].sum() == pytest.approx(1.0)


# --------------------------------------------------- frame -> session aggregation (hand)


def test_median_frame_to_session_is_the_frozen_median():
    """Odd count -> the middle value; even count -> the mean of the two middle values
    (numpy's median). A MEAN aggregation gives 5.25 for group 1 below, not 2.5."""
    values = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 15.0, 7.0])
    group = np.array([0, 0, 0, 1, 1, 1, 1, 2])
    assert median_frame_to_session(values, group, 3).tolist() == [2.0, 2.5, 7.0]


# ------------------------------------------------------------------- the frame spine


@pytest.mark.parametrize(
    "family, expected_shape",
    [
        ("cnn1d_raw", (100, 1, N_SLOWTIME)),
        ("cnn1d_matched", (100, 2, N_SLOWTIME)),
        ("spec2d_raw", (100, 1, 65, 13)),
        ("spec2d_matched", (100, 2, 65, 13)),
    ],
)
def test_build_frames_d_applies_each_familys_own_input_construction(
    store, config, family, expected_shape
):
    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", family, sessions, store_dir)

    assert frames.X.shape == expected_shape
    assert frames.subjects.shape == (100,)
    assert frames.session_row.shape == (100,)
    assert frames.frame_ids.tolist() == [
        i for _ in range(N_SUBJECTS * 5) for i in range(N_FRAMES_PER_SESSION)
    ]
    # the target is the session's Δm%, broadcast to its frames (training only; scoring is
    # session-level) — every frame of a session carries the identical value
    for row in range(len(frames.session_delta_m_pct)):
        rows = frames.session_row == row
        assert len(set(frames.y[rows].tolist())) == 1
        assert frames.y[rows][0] == frames.session_delta_m_pct[row]


def test_build_frames_d_reads_the_store_key_the_family_names(store, config):
    """`cnn1d_raw` must consume `sig__raw_slowtime` and `cnn1d_matched`
    `sig__matched_iq`; the fixture's two arrays differ, so a swapped key is visible."""
    store_dir, sessions = store
    raw = build_frames_d(config, "77ghz", "cnn1d_raw", sessions, store_dir)
    matched = build_frames_d(config, "77ghz", "cnn1d_matched", sessions, store_dir)

    assert raw.X.shape[1] == 1 and matched.X.shape[1] == 2
    assert raw.X[0, 0].tobytes() != matched.X[0, 0].tobytes()   # robust(raw) != robust(matched ch0)


def test_build_frames_d_refuses_an_unknown_family_or_band(store, config):
    store_dir, sessions = store
    with pytest.raises(ExpDError):
        build_frames_d(config, "77ghz", "physics", sessions, store_dir)
    with pytest.raises(ExpDError):
        build_frames_d(config, "23ghz", "cnn1d_raw", sessions, store_dir)


def test_build_frames_d_refuses_a_store_whose_frame_count_disagrees_with_the_spine(
    tmp_path, config
):
    """Frame-order alignment is the store's contract (§2.9); a silently shorter stored array
    would misalign every target."""
    sessions = _sessions(n_subjects=2)
    _write_store_77(tmp_path, sessions)
    sessions[0]["frame_ids"] = list(range(N_FRAMES_PER_SESSION + 1))
    with pytest.raises(ExpDError, match="frame"):
        build_frames_d(config, "77ghz", "cnn1d_raw", sessions, tmp_path)


# ------------------------------------------------------- the nested path, end to end


@pytest.fixture(scope="module")
def cnn_result(store, config):
    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", "cnn1d_raw", sessions, store_dir)
    fold = _fold(1)
    return frames, fold, run_cnn_family(config, "77ghz", "cnn1d_raw", fold, SEEDS, frames)


def test_run_cnn_family_produces_one_prediction_per_test_session_per_seed(cnn_result):
    frames, fold, result = cnn_result
    assert isinstance(result, CnnFoldResult)
    assert result.test_subject == 1
    assert result.test_session_idx.tolist() == [0, 1, 2, 3, 4]
    assert [o.seed for o in result.seed_outcomes] == list(SEEDS)
    for outcome in result.seed_outcomes:
        assert outcome.session_predictions.shape == (5,)
        assert math.isfinite(outcome.session_mae)
    # seeds are scored SEPARATELY, never ensembled (`:644-649`)
    assert (
        result.seed_outcomes[0].session_predictions.tobytes()
        != result.seed_outcomes[1].session_predictions.tobytes()
    )


def test_every_config_of_the_frozen_grid_is_scored_on_every_inner_fold_and_seed(cnn_result):
    _frames, fold, result = cnn_result
    assert result.n_inner_folds == len(fold.inner_folds) == 3
    assert len(result.per_config_scores) == 6
    assert len(result.inner_results) == 6 * 3 * len(SEEDS)


def test_config_selection_routes_through_selection_py(cnn_result):
    """The single-tie-break-source doctrine: the winner is whatever `select_candidate`
    returns on the recorded scores, with simplicity rank 0 and a constant feature dimension
    (one family, one input size — neither rung can decide anything here)."""
    _frames, _fold, result = cnn_result
    scores = result.per_config_scores
    assert all(isinstance(s, CandidateScore) for s in scores)
    assert {s.simplicity_rank for s in scores} == {0}
    assert len({s.feature_dimension for s in scores}) == 1
    assert scores[0].feature_dimension == 1 * N_SLOWTIME
    assert select_candidate(scores).candidate_id == scores[result.selected_config_index].candidate_id
    assert result.selected_config == result.config_grid[result.selected_config_index]


def test_per_config_score_is_the_mean_over_inner_folds_times_seeds(cnn_result):
    """Trap 7's population, at the scoring end: the score averages every (inner fold, seed)
    cell, and the tie-break variance is the population std over that same set — not over
    fold means, which would be a different number."""
    _frames, _fold, result = cnn_result
    for ci, score in enumerate(result.per_config_scores):
        cells = [c.val_session_mae for c in result.inner_results if c.config_index == ci]
        assert len(cells) == 3 * len(SEEDS)
        assert score.inner_val_mae == pytest.approx(float(np.mean(cells)))
        assert score.inner_fold_variance == pytest.approx(float(np.std(cells, ddof=0)))


def test_epoch_budget_is_the_median_over_the_winners_inner_folds_times_seeds(cnn_result):
    """`:650-655` written for one config x deterministic fits; with 5 seeds the median is
    over folds x seeds (trap 7). Taking it over the first seed only, or over fold means,
    is a different number — the population size is asserted, not just the value."""
    _frames, _fold, result = cnn_result
    winning_cells = [
        c for c in result.inner_results if c.config_index == result.selected_config_index
    ]
    assert sorted(result.selected_epoch_counts) == sorted(
        c.n_epochs_selected for c in winning_cells
    )
    assert len(result.selected_epoch_counts) == 3 * len(SEEDS)
    assert result.epoch_budget == int(np.median(result.selected_epoch_counts))
    assert result.epoch_budget >= 1


def test_epoch_budget_median_floors_an_even_count():
    """Hand-computed on the frozen rule itself: `int(np.median(...))`.
        [1, 2, 3, 4, 5]        -> 3
        [3, 3, 5, 7, 9, 9]     -> (5 + 7)/2 = 6.0 -> 6
        [3, 3, 5, 8, 9, 9]     -> (5 + 8)/2 = 6.5 -> 6   (floors, never rounds up)
    """
    from dehyd.eval.exp_d import epoch_budget_from

    assert epoch_budget_from([1, 2, 3, 4, 5]) == 3
    assert epoch_budget_from([3, 3, 5, 7, 9, 9]) == 6
    assert epoch_budget_from([3, 3, 5, 8, 9, 9]) == 6


def test_early_stopping_can_halt_before_max_epochs(store, config):
    """With patience 2 on a small fixture at least one inner fit stops before max_epochs —
    otherwise the early-stopping branch would be dead code and the budget meaningless."""
    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", "cnn1d_raw", sessions, store_dir)
    result = run_cnn_family(config, "77ghz", "cnn1d_raw", _fold(2), SEEDS, frames)
    selected = [c.n_epochs_selected for c in result.inner_results]
    assert max(selected) <= config.baselines.max_epochs
    assert min(selected) < config.baselines.max_epochs


def test_two_runs_of_the_same_fold_are_bit_identical(cnn_result, config):
    frames, fold, result = cnn_result
    again = run_cnn_family(config, "77ghz", "cnn1d_raw", fold, SEEDS, frames)
    assert again.epoch_budget == result.epoch_budget
    assert again.selected_config == result.selected_config
    assert _state_bytes(again.final_fits, "cnn_state") == _state_bytes(result.final_fits, "cnn_state")
    for a, b in zip(again.seed_outcomes, result.seed_outcomes, strict=True):
        assert a.session_predictions.tobytes() == b.session_predictions.tobytes()


# --------------------------------------------------- the CNN-path mutation property (T18)


def test_held_out_mutation_moves_only_the_held_out_subjects_predictions(cnn_result, config):
    """The T18 pattern over the REAL Exp D composition: mutating the held-out subject's
    frames and targets must leave config selection, the epoch budget, the sampler weights,
    every refit state and the recorded per-fit epoch counts bytewise unchanged."""
    frames, fold, base = cnn_result
    mutated = run_cnn_family(
        config, "77ghz", "cnn1d_raw", fold, SEEDS, _mutate_held_out(frames, 1)
    )

    assert mutated.selected_config == base.selected_config
    assert mutated.epoch_budget == base.epoch_budget
    assert mutated.selected_epoch_counts == base.selected_epoch_counts
    assert [s.inner_val_mae for s in mutated.per_config_scores] == [
        s.inner_val_mae for s in base.per_config_scores
    ]
    for mut_cell, base_cell in zip(mutated.inner_results, base.inner_results, strict=True):
        assert mut_cell.n_epochs_selected == base_cell.n_epochs_selected
        assert _state_bytes(mut_cell.fits, "cnn_state") == _state_bytes(base_cell.fits, "cnn_state")
        assert _state_bytes(mut_cell.fits, "sampler_weights") == _state_bytes(
            base_cell.fits, "sampler_weights"
        )
    assert _state_bytes(mutated.final_fits, "cnn_state") == _state_bytes(base.final_fits, "cnn_state")
    assert _state_bytes(mutated.final_fits, "sampler_weights") == _state_bytes(
        base.final_fits, "sampler_weights"
    )
    # ... and only the held-out subject's predictions move (the power companion)
    assert (
        mutated.seed_outcomes[0].session_predictions.tobytes()
        != base.seed_outcomes[0].session_predictions.tobytes()
    )


def test_spectrogram_norm_is_fit_inside_the_fold_and_audited_at_both_levels(store, config):
    """Trap 12: the per-frequency mean/std is the one FITTED input transform, so it must be
    a train-only quantity at both CV levels and it must move when the training rows move —
    while a held-out mutation leaves it bytewise identical."""
    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", "spec2d_raw", sessions, store_dir)
    fold = _fold(1)
    base = run_cnn_family(config, "77ghz", "spec2d_raw", fold, SEEDS, frames)
    mutated = run_cnn_family(
        config, "77ghz", "spec2d_raw", fold, SEEDS, _mutate_held_out(frames, 1)
    )

    inner_norm = next(f for f in base.inner_results[0].fits if f.quantity == "spectrogram_norm")
    outer_norm = next(f for f in base.final_fits if f.quantity == "spectrogram_norm")
    assert inner_norm.role == "inner_train" and outer_norm.role == "outer_train"
    assert 1 not in inner_norm.subjects and 1 not in outer_norm.subjects
    assert inner_norm.params["mean"].shape == (1, 65)          # [C, F]
    assert "n_zero_variance_cells" in inner_norm.params
    # train-only under the held-out mutation ...
    assert _state_bytes(mutated.final_fits, "spectrogram_norm") == _state_bytes(
        base.final_fits, "spectrogram_norm"
    )
    # ... and genuinely fitted: the outer-train norm differs from an inner-train one
    assert outer_norm.params["mean"].tobytes() != inner_norm.params["mean"].tobytes()


def test_fit_audit_covers_every_exp_d_fitted_quantity(cnn_result):
    """`harness.fit_audit` duck-types on `.inner_results`/`.final_fits`, so the CNN result
    must expose the same shape as the sklearn engine's."""
    _frames, _fold, result = cnn_result
    audit = harness.fit_audit([result])
    quantities = {row["quantity"] for row in audit}
    assert {"sampler_weights", "cnn_state", "selected_epochs"} <= quantities
    assert {"inner_train", "outer_train"} == {row["role"] for row in audit}
    for row in audit:
        assert result.test_subject not in row["fitted_on"]


# ------------------------------------------------------ the sampler / DataLoader contract


def test_sampled_batch_indices_match_a_weighted_draw_with_replacement(cnn_result, config):
    """(C20) The pinned trace. `WeightedRandomSampler` IS
    `torch.multinomial(weights, num_samples, replacement, generator=g)`, so the expected
    sequence is re-derived from that definition with an identically seeded generator — not
    read back from the implementation.

    This fails against: `replacement=False` (a permutation, no repeats), a different
    `num_samples`, `shuffle=True`, `drop_last=False` (a short tail batch), and a generator
    shared across fits (whose draw would depend on the fits before it).
    """
    _frames, fold, result = cnn_result
    cell = result.inner_results[0]
    weights = next(f for f in cell.fits if f.quantity == "sampler_weights").params["weights"]
    n_train = len(weights)
    steps = n_train // config.baselines.batch_size

    generator = torch.Generator()
    generator.manual_seed(
        fit_seed(config.run.seed, fold_id=fold.test_subject, config_index=cell.config_index,
                 inner_fold=cell.inner_fold, seed=cell.seed)
    )
    # ONE generator is shared between the sampler and the DataLoader (plan §2.8), and a
    # `num_workers=0` DataLoader draws its `_base_seed` from that generator before the
    # sampler iterates — so the realized draw is multinomial's, taken after that one draw.
    # Pinning it here means a torch change to that draw pattern surfaces as a failed test
    # rather than as silently different training batches.
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    expected = torch.multinomial(
        torch.as_tensor(weights, dtype=torch.double), n_train, True, generator=generator
    ).numpy()

    trace = cell.first_epoch_batch_indices
    assert trace.shape == (steps, config.baselines.batch_size)     # uniform 16-row batches
    assert steps == n_train // 16 and n_train % 16 != 0            # the tail IS dropped
    assert trace.reshape(-1).tolist() == expected[: steps * 16].tolist()
    assert len(set(trace.reshape(-1).tolist())) < steps * 16       # repeats => replacement=True


def test_the_same_seed_draws_the_same_training_batches_under_a_changed_held_out_set(
    cnn_result, config
):
    """A shared/global generator would make fit k's data order depend on fits 1..k-1, so a
    held-out mutation would shift TRAINING batches (§5 trap 8)."""
    frames, fold, base = cnn_result
    mutated = run_cnn_family(
        config, "77ghz", "cnn1d_raw", fold, SEEDS, _mutate_held_out(frames, 1)
    )
    for mut_cell, base_cell in zip(mutated.inner_results, base.inner_results, strict=True):
        assert (
            mut_cell.first_epoch_batch_indices.tobytes()
            == base_cell.first_epoch_batch_indices.tobytes()
        )


def test_a_fold_with_fewer_training_frames_than_one_batch_is_refused(store, config):
    """`drop_last=True` means such a fit would take zero optimizer steps — a silent no-op
    rather than a trained model."""
    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", "cnn1d_raw", sessions, store_dir)
    tiny = dataclasses.replace(config, baselines=dataclasses.replace(config.baselines, batch_size=1000))
    with pytest.raises(ExpDError, match="batch_size"):
        run_cnn_family(tiny, "77ghz", "cnn1d_raw", _fold(1), SEEDS, frames)


# --------------------------------------------- the optimizer never sees validation data


def test_train_trajectory_is_identical_over_the_common_prefix_under_a_val_mutation(
    store, config
):
    """T18's common-prefix property at the fit level: same train data, two different
    validation targets -> the per-epoch train predictions are bit-identical over the common
    prefix of executed epochs. Only the val-driven stop time / checkpoint may differ."""
    from dehyd.eval.exp_d import _train_cnn

    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", "cnn1d_raw", sessions, store_dir)
    train_rows = np.isin(frames.subjects, [2, 3, 4])
    val_rows = np.isin(frames.subjects, [1])
    x_train = np.ascontiguousarray(frames.X[train_rows], dtype=np.float32)
    y_train = np.ascontiguousarray(frames.y[train_rows], dtype=np.float32)
    weights = session_sampler_weights(frames.session_row[train_rows])

    val_sessions = np.unique(frames.session_row[val_rows])
    position = {int(k): i for i, k in enumerate(val_sessions)}
    group = np.array([position[int(k)] for k in frames.session_row[val_rows]])
    val_subjects = np.ones(len(val_sessions), dtype=int)
    truth = frames.session_delta_m_pct[val_sessions]

    def run(scale):
        return _train_cnn(
            x_train, y_train, weights, family="cnn1d_raw", lr=1e-3, weight_decay=0.0,
            max_epochs=6, batch_size=config.baselines.batch_size,
            betas=config.baselines.adam_betas, patience=2, min_delta=1e-4,
            seed_value=4242, device="cpu",
            val=(np.ascontiguousarray(frames.X[val_rows], dtype=np.float32), group,
                 val_subjects, truth * scale),
            trace_train=True,
        )

    a, b = run(1.0), run(50.0)
    common = min(len(a.train_pred_per_epoch), len(b.train_pred_per_epoch))
    assert common >= 1
    for epoch in range(common):
        assert a.train_pred_per_epoch[epoch].tobytes() == b.train_pred_per_epoch[epoch].tobytes()
    assert a.val_history[:common] != b.val_history[:common]      # the val side DID move


# ---------------------------------------------------- BatchNorm at prediction time (trap 9)


def test_prediction_is_repeatable_and_independent_of_the_predicted_sets_composition(
    store, config
):
    """Predicting with `model.train()` active keeps BatchNorm's running statistics updating,
    which silently makes a prediction depend on what else is in the batch. Predicting the
    same rows twice, and predicting a subset, must both be exact."""
    from dehyd.eval.exp_d import _predict, _predict_with_state, _train_cnn

    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", "cnn1d_raw", sessions, store_dir)
    rows = np.isin(frames.subjects, [2, 3, 4])
    x = np.ascontiguousarray(frames.X[rows], dtype=np.float32)
    y = np.ascontiguousarray(frames.y[rows], dtype=np.float32)
    outcome = _train_cnn(
        x, y, session_sampler_weights(frames.session_row[rows]), family="cnn1d_raw",
        lr=1e-3, weight_decay=0.0, max_epochs=3, batch_size=config.baselines.batch_size,
        betas=config.baselines.adam_betas, patience=2, min_delta=1e-4, seed_value=7,
        device="cpu",
    )
    first = _predict_with_state("cnn1d_raw", 1, outcome.state, x, "cpu")
    second = _predict_with_state("cnn1d_raw", 1, outcome.state, x, "cpu")
    subset = _predict_with_state("cnn1d_raw", 1, outcome.state, x[:7], "cpu")

    assert first.tobytes() == second.tobytes()
    assert subset.tobytes() == first[:7].tobytes()

    # and the choke point itself: `_predict` switches the model out of train mode, so
    # BatchNorm's running statistics can never be updated by a prediction.
    model = cnn.build_network("cnn1d_raw", 1)
    model.train()
    _predict(model, x[:4], "cpu")
    assert model.training is False


# ------------------------------------------------------------- protocol fail-closed checks


def test_an_off_grid_learning_rate_is_refused_before_any_fit(store, config):
    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", "cnn1d_raw", sessions, store_dir)
    drifted = dataclasses.replace(
        config,
        model_grid=dataclasses.replace(config.model_grid, baseline_learning_rate=(0.5,)),
    )
    with pytest.raises(ExpDProtocolError, match="baseline_learning_rate"):
        run_cnn_family(drifted, "77ghz", "cnn1d_raw", _fold(1), SEEDS, frames)


@pytest.mark.parametrize(
    "field, value",
    [("optimizer", "sgd"), ("loss", "mae"), ("frame_to_session_aggregation", "mean"),
     ("checkpoint_metric", "inner_val_frame_mae")],
)
def test_a_drifted_training_protocol_constant_is_refused(store, config, field, value):
    """The frozen training protocol is not re-implementable by a config edit: this path
    implements exactly `adam` + `mse` + median frame->session + inner-val session MAE."""
    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", "cnn1d_raw", sessions, store_dir)
    drifted = dataclasses.replace(
        config, baselines=dataclasses.replace(config.baselines, **{field: value})
    )
    with pytest.raises(ExpDProtocolError, match=field):
        run_cnn_family(drifted, "77ghz", "cnn1d_raw", _fold(1), SEEDS, frames)


def test_run_cnn_family_refuses_frames_built_for_another_family(store, config):
    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", "cnn1d_matched", sessions, store_dir)
    with pytest.raises(ExpDError, match="cnn1d_raw"):
        run_cnn_family(config, "77ghz", "cnn1d_raw", _fold(1), SEEDS, frames)


def test_run_cnn_family_refuses_a_non_selectable_fold(store, config):
    store_dir, sessions = store
    frames = build_frames_d(config, "77ghz", "cnn1d_raw", sessions, store_dir)
    fold = harness.nested_loso_splits([1, 2, 3])[0]
    assert not fold.selectable
    with pytest.raises(ExpDError, match="selectable"):
        run_cnn_family(config, "77ghz", "cnn1d_raw", fold, SEEDS, frames)


def test_the_held_out_subject_never_appears_in_any_recorded_fit(cnn_result):
    _frames, _fold, result = cnn_result
    for cell in result.inner_results:
        assert result.test_subject not in cell.inner_val
        for record in cell.fits:
            assert result.test_subject not in record.subjects
    for record in result.final_fits:
        assert result.test_subject not in record.subjects
