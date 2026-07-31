"""T-M9-cnnpath / T-M9-physics / T-M9-expd-shard / T-M9-expd-compare: Experiment D.

Milestone 9 steps 7 and 8: the frame spine and `run_cnn_family` (grid x early stopping x
epoch budget x per-seed refit) from step 7, then the physics and session-index baselines,
the four per-family merged artifacts, the GPU fold-array shard/merge and the frozen
comparison statistics from step 8.

Everything is asserted against the SPECIFICATION's arithmetic (plan §2.8, `:644-655`,
`:826-849`, `:917-919`, `:1263-1281`), never against a recorded run: the sampler weights,
the median aggregation, the epoch-budget median and the per-fit seed derivation are all
hand-computed; the sampler trace is re-derived from `torch.multinomial` (which is what
`WeightedRandomSampler` *is*); and the physics band bins and band powers are derived from
the frozen radar constants and from the closed-form DFT of a periodic Hann window rather
than read back from the implementation.

The CNN fixtures use the 77 GHz signal shapes (256-sample slow time) because they are the
smallest real ones; the 10 GHz shapes are pinned in `test_cnn.py`, which needs no training.
"""

import csv
import dataclasses
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from dehyd.config import load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_b, exp_c, exp_d, harness
from dehyd.eval import metrics as M
from dehyd.eval.exp_d import (
    COMPOSITE_MEMBERS,
    EXPD_FAMILIES,
    FIT_SEED_BASE,
    PHYSICS_EPS_SCALE,
    PHYSICS_SIGNAL_KEY,
    RNG_OFFSET_EXPD_BASE,
    BaselineFoldResult,
    CnnFoldResult,
    CnnSeedOutcome,
    ExpDError,
    ExpDProtocolError,
    FramesD,
    RadarReference,
    band_masks_from_frequencies,
    band_power_ratio_scalar,
    baseline_config_grid,
    build_frames_d,
    build_physics_spine,
    build_session_index_spine,
    cheap_prediction_rows,
    cheap_selection_row,
    cnn_prediction_rows,
    cnn_selection_row,
    cnn_config_score_statistics,
    expected_test_rows_by_fold,
    fit_seed,
    half_spectrum_power,
    load_exp_a_radar,
    load_family_artifacts,
    median_frame_to_session,
    merge_exp_d_folds,
    physics_band_masks,
    physics_frame_scalar,
    realized_fold_census,
    run_cheap_baseline,
    run_cnn_family,
    run_physics,
    run_session_index,
    selectable_folds,
    session_sampler_weights,
    summarize_exp_d,
    write_exp_d_comparison_reports,
    write_family_artifacts,
    write_fold_shard,
    write_noop_marker,
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
    mutated = dataclasses.replace(
        frames,
        X=frames.X.copy(),
        y=frames.y.copy(),
        session_delta_m_pct=frames.session_delta_m_pct.copy(),
    )
    frame_rows = mutated.subjects == subject
    mutated.X[frame_rows] = rng.normal(size=mutated.X[frame_rows].shape) * 10 + 5

    # `run_cnn_family` scores against the session-level target, not the broadcast frame
    # copy. Mutate that authoritative target and then keep the training copy coherent:
    # every frame in a session still carries exactly its session's Δm%.
    session_positions = np.flatnonzero(mutated.session_subjects == subject)
    new_targets = rng.normal(size=len(session_positions)) * 10 + 5
    for position, target in zip(session_positions, new_targets, strict=True):
        mutated.session_delta_m_pct[position] = target
        mutated.y[mutated.session_row == position] = target
    return mutated


def _record_bytes(record):
    return b"".join(k.encode() + v.tobytes() for k, v in sorted(record.params.items()))


def _state_bytes(fits, quantity):
    record = next(f for f in fits if f.quantity == quantity)
    return _record_bytes(record)


def _all_state_bytes(fits, quantity):
    return tuple(_record_bytes(f) for f in fits if f.quantity == quantity)


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


def test_per_config_score_averages_seeds_within_fold_before_inner_fold_variance(cnn_result):
    """The stochastic-model rule first averages seeds within each inner fold. The primary
    mean is then the mean of those fold scores; the tie-break is the population std across
    fold means, matching the shared harness rather than mixing seed spread into a quantity
    named `inner_fold_variance`."""
    _frames, _fold, result = cnn_result
    for ci, score in enumerate(result.per_config_scores):
        matrix = np.array([
            [
                c.val_session_mae
                for c in result.inner_results
                if c.config_index == ci and c.inner_fold == inner_fold
            ]
            for inner_fold in range(result.n_inner_folds)
        ])
        assert matrix.shape == (3, len(SEEDS))
        per_fold = matrix.mean(axis=1)
        assert score.inner_val_mae == pytest.approx(float(per_fold.mean()))
        assert score.inner_fold_variance == pytest.approx(float(per_fold.std(ddof=0)))


def test_fold_level_variance_does_not_confuse_seed_spread_with_fold_spread():
    """Same overall mean, opposite ordering under the correct and flattened variances.

    A has seed variability but identical fold means; B has no seed variability but
    different fold means. The frozen lower-inner-fold-variance tie-break must prefer A.
    """
    candidate_a = np.array([[0.0, 2.0], [0.0, 2.0]])
    candidate_b = np.array([[0.9, 0.9], [1.1, 1.1]])

    mean_a, variance_a = cnn_config_score_statistics(candidate_a)
    mean_b, variance_b = cnn_config_score_statistics(candidate_b)

    assert mean_a == pytest.approx(mean_b)
    assert variance_a == 0.0
    assert variance_b == pytest.approx(0.1)
    assert candidate_a.std(ddof=0) > candidate_b.std(ddof=0)  # flattened gives the wrong order


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
    assert _all_state_bytes(again.final_fits, "cnn_state") == _all_state_bytes(
        result.final_fits, "cnn_state"
    )
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
    mutated_states = _all_state_bytes(mutated.final_fits, "cnn_state")
    base_states = _all_state_bytes(base.final_fits, "cnn_state")
    assert len(mutated_states) == len(base_states) == len(SEEDS)
    assert mutated_states == base_states
    assert _state_bytes(mutated.final_fits, "sampler_weights") == _state_bytes(
        base.final_fits, "sampler_weights"
    )
    # ... while the authoritative held-out labels, every seed's predictions, and every
    # seed's score move. These are the only quantities scoring is allowed to depend on.
    assert mutated.test_session_truth.tobytes() != base.test_session_truth.tobytes()
    for mutated_seed, base_seed in zip(mutated.seed_outcomes, base.seed_outcomes, strict=True):
        assert (
            mutated_seed.session_predictions.tobytes()
            != base_seed.session_predictions.tobytes()
        )
        assert mutated_seed.session_mae != base_seed.session_mae


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
    from dehyd.eval.exp_d import _train_cnn

    frames = build_frames_d(config, "77ghz", "cnn1d_raw", sessions, store_dir)
    rows = np.isin(frames.subjects, [2, 3, 4])
    with pytest.raises(ExpDError, match="batch_size"):
        _train_cnn(
            np.ascontiguousarray(frames.X[rows], dtype=np.float32),
            np.ascontiguousarray(frames.y[rows], dtype=np.float32),
            session_sampler_weights(frames.session_row[rows]),
            family="cnn1d_raw", lr=1e-3, weight_decay=0.0, max_epochs=1,
            batch_size=1000, betas=config.baselines.adam_betas, patience=2,
            min_delta=1e-4, seed_value=7, device="cpu",
        )


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
    [
        ("optimizer", "sgd"),
        ("loss", "mae"),
        ("adam_betas", (0.5, 0.9)),
        ("weight_init", "custom"),
        ("batch_size", 8),
        ("frame_to_session_aggregation", "mean"),
        ("checkpoint_metric", "inner_val_frame_mae"),
        ("raw_matched_standardize", "none"),
        ("spectrogram_standardize", "global"),
        ("matched_reference_rx_index_77ghz", 7),
    ],
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


# =====================================================================================
# Milestone 9 step 8 — physics + session-index baselines, per-family artifacts,
# fold-array shard/merge, and the frozen comparison statistics.
# =====================================================================================


N_BEAT = 534          # the frozen 10 GHz raw chirp-mean beat length
N_SLOW = 256          # the frozen 77 GHz raw reduced slow-time length


@pytest.fixture(scope="module")
def fast_ci_config(config):
    """The same frozen config with a SMALL bootstrap B. Only the replicate count moves —
    every estimand, seed-collapse rule and CI method stays frozen; B=10000 x 3 metrics x 6
    families inside one test would cost minutes and prove nothing extra."""
    return dataclasses.replace(config, stats=dataclasses.replace(config.stats, bootstrap_b=200))


# ------------------------------------------------------- the physics band definitions


def test_physics_bands_are_the_hand_derived_bins_of_the_frozen_10ghz_constants(config):
    """Hand-derived from `:826-849` + the frozen radar constants, not read back from code.

        hz_per_m = 2*(B/T)/c = 2*(500e6 / 1024e-6) / 299792458 = 3257.4619 Hz/m
        target     [0.9, 1.5) m -> [2931.72, 4886.19) Hz
        background [1.5, 3.0] m -> [4886.19, 9772.39] Hz
        df = fs/n = 520834/534 = 975.3446 Hz, half-spectrum bin centres k*df:
            bin  3 = 2926.03  -> below target
            bins 4, 5 = 3901.38, 4876.72  -> TARGET
            bins 6..10 = 5852.07 .. 9753.45 -> BACKGROUND
            bin 11 = 10728.79 -> above background
    """
    target, background = physics_band_masks(config, "10ghz", N_BEAT)

    assert target.shape == background.shape == (N_BEAT // 2,)     # bins 0 .. n//2-1
    assert np.flatnonzero(target).tolist() == [4, 5]
    assert np.flatnonzero(background).tolist() == [6, 7, 8, 9, 10]
    assert not np.any(target & background)


def test_the_1_5_m_boundary_bin_belongs_to_the_background_band_only():
    """Half-open at 1.5 m (`:829-843`): a bin landing exactly on the shared edge is
    background, never target, and never both. Exercised on explicit frequencies because no
    bin of the real 534-point grid lands exactly on the 4886.19 Hz edge — a closed target
    band (`<=`) would double-count it, which this fixture would catch."""
    lo, edge, hi = 2931.72, 4886.19, 9772.39
    freqs = np.array([2000.0, lo, 4000.0, edge, 6000.0, hi, 10000.0])
    target, background = band_masks_from_frequencies(freqs, (lo, edge), (edge, hi))

    assert target.tolist() == [False, True, True, False, False, False, False]
    assert background.tolist() == [False, False, False, True, True, True, False]
    assert not np.any(target & background)


def test_77ghz_physics_partition_is_the_dc_bin_against_every_resolvable_doppler_bin(config):
    """A-M6-2 (iii): bin 0 alone vs bins 1..127 — a partition of the non-negative
    half-spectrum of the 256-point Doppler FFT, both bands nonempty and disjoint."""
    static, motion = physics_band_masks(config, "77ghz", N_SLOW)

    assert static.shape == motion.shape == (128,)
    assert np.flatnonzero(static).tolist() == [0]
    assert np.flatnonzero(motion).tolist() == list(range(1, 128))
    assert not np.any(static & motion)
    assert np.all(static | motion)                     # an exact partition, nothing dropped


def test_physics_band_masks_refuse_a_length_the_frozen_77ghz_partition_cannot_cover(config):
    """The frozen (1, 127) motion range assumes the 256-point FFT; on any other length the
    partition would silently stop covering the half-spectrum."""
    with pytest.raises(ExpDProtocolError, match="partition"):
        physics_band_masks(config, "77ghz", 512)


# ---------------------------------------------------------- the physics scalar (hand)


def test_physics_scalar_on_a_hand_computed_two_tone_10ghz_signal(config):
    """The periodic Hann window's DFT has exactly three nonzero bins,
    W[0] = N/2 and W[+-1] = -N/4, so a complex tone `A*exp(2*pi*i*k0*n/N)` windowed and
    transformed puts `0.5*A*N` at bin k0, `-0.25*A*N` at k0 +- 1, and nothing anywhere else.

    Tones at bin 5 (amplitude 2, inside the target band {4, 5}) and bin 8 (amplitude 1,
    inside the background band {6..10}) therefore give, with N = 534:

        P_target     = |-0.25*2*534|^2 + |0.5*2*534|^2          = 267^2 + 534^2
        P_background = |-0.25*2*534|^2 + |-0.25*534|^2
                       + |0.5*534|^2 + |-0.25*534|^2 + 0        = 267^2 + 133.5^2
                                                                  + 267^2 + 133.5^2

    an exact power ratio of 2, and the frozen scalar is
    `log10((P_t + eps)/(P_b + eps))` with `eps = 1e-12*(P_t + P_b)`.
    """
    n = np.arange(N_BEAT)
    signal = 2.0 * np.exp(2j * np.pi * 5 * n / N_BEAT) + np.exp(2j * np.pi * 8 * n / N_BEAT)

    p_target = 267.0**2 + 534.0**2
    p_background = 267.0**2 + 133.5**2 + 267.0**2 + 133.5**2
    assert p_target / p_background == 2.0
    eps = PHYSICS_EPS_SCALE * (p_target + p_background)
    expected = math.log10((p_target + eps) / (p_background + eps))

    assert physics_frame_scalar(config, "10ghz", signal) == pytest.approx(expected, rel=1e-9)


def test_physics_scalar_on_a_hand_computed_77ghz_constant_signal(config):
    """A constant slow-time series is pure DC, so the windowed spectrum is the Hann
    window's own DFT: 0.5*N at bin 0 and -0.25*N at bin 1 (bin 255 is outside the
    half-spectrum). With N = 256 that is P_dc = 128^2 against P_motion = 64^2 — a ratio of
    exactly 4, i.e. log10(4) up to the frozen eps coupling."""
    signal = np.ones(N_SLOW)
    p_dc, p_motion = 128.0**2, 64.0**2
    eps = PHYSICS_EPS_SCALE * (p_dc + p_motion)
    expected = math.log10((p_dc + eps) / (p_motion + eps))

    assert expected == pytest.approx(math.log10(4.0), abs=1e-11)   # eps moves it by ~1.6e-12
    assert physics_frame_scalar(config, "77ghz", signal) == pytest.approx(expected, rel=1e-9)


def test_physics_scalar_is_finite_when_the_target_band_holds_no_energy():
    """`:930-940`: eps is added to BOTH terms precisely so `P_target = 0` gives a finite
    floor instead of -inf. With P_t = 0 the value is log10(eps/(P_b + eps)) -> -12."""
    power = np.array([0.0, 0.0, 4.0, 6.0])
    target = np.array([True, True, False, False])
    background = np.array([False, False, True, True])

    value = band_power_ratio_scalar(power, target, background)
    assert math.isfinite(value)
    assert value == pytest.approx(math.log10(1e-12 / (1.0 + 1e-12)), rel=1e-9)


def test_physics_scalar_refuses_a_frame_with_no_energy_in_either_band():
    """Then eps is 0 too and the ratio is a genuine 0/0 — the one case the frozen eps
    cannot rescue, so it must stop rather than emit NaN into a fitted linear model."""
    power = np.zeros(4)
    with pytest.raises(ExpDError, match="no energy"):
        band_power_ratio_scalar(power, np.array([True, True, False, False]),
                                np.array([False, False, True, True]))


def test_physics_scalar_is_scale_invariant_only_through_the_frozen_eps_coupling(config):
    """Trap 13's positive half: eps is proportional to (P_t + P_b), so multiplying the
    signal by a constant leaves the scalar unchanged — which is exactly why the baseline
    may not be fed a robust-standardized signal (that changes the SHAPE, not the scale)."""
    n = np.arange(N_BEAT)
    signal = 2.0 * np.exp(2j * np.pi * 5 * n / N_BEAT) + np.exp(2j * np.pi * 8 * n / N_BEAT)
    assert physics_frame_scalar(config, "10ghz", 1e4 * signal) == pytest.approx(
        physics_frame_scalar(config, "10ghz", signal), rel=1e-12
    )


def test_half_spectrum_power_keeps_dc_and_drops_nyquist():
    """The repo's own half-spectrum convention (`qc/screens.py:147-154`): bins
    0 .. n//2 - 1, DC included, Nyquist excluded — which is what makes the 77 GHz
    `bins 0..127` of A-M6-2 the literal non-negative half-spectrum."""
    assert half_spectrum_power(np.ones(N_SLOW)).shape == (128,)
    assert half_spectrum_power(np.ones(N_BEAT)).shape == (267,)


# ------------------------------------------------- the physics spine and its LOSO run


def _write_store_10(store_dir, sessions, *, dc_offset=0.0):
    """A synthetic schema-v2 10 GHz store carrying only `sig__raw_beat`. The target-band
    tone's amplitude tracks the session, so the range-power scalar genuinely varies with
    the label and the per-fold linear fit has something to fit."""
    k = np.arange(N_BEAT)
    for s in sessions:
        n = len(s["frame_ids"])
        amplitude = 1.0 + 0.6 * s["session_idx"] + 0.1 * s["subject"]
        beat = amplitude * np.exp(2j * np.pi * 5 * k / N_BEAT) + np.exp(2j * np.pi * 8 * k / N_BEAT)
        signals = np.stack([beat * (1.0 + 0.02 * i) + dc_offset for i in range(n)])
        write_session_store("10ghz", s["subject"], s["session_name"],
                            {"sig__raw_beat": signals}, {"n_frames": n}, store_dir)


@pytest.fixture(scope="module")
def store_10(tmp_path_factory):
    path = tmp_path_factory.mktemp("expd_store_10")
    sessions = _sessions()
    _write_store_10(path, sessions)
    return path, sessions


def test_physics_reads_the_named_unstandardized_raw_store_key():
    """Trap 13: robust standardization destroys absolute power, so the physics path reads
    the raw keys — the same ones the raw CNN families consume, stated here independently of
    `cnn.FRAME_INPUT` so the two sources check each other."""
    assert PHYSICS_SIGNAL_KEY == {"10ghz": "sig__raw_beat", "77ghz": "sig__raw_slowtime"}
    for band in ("10ghz", "77ghz"):
        assert PHYSICS_SIGNAL_KEY[band] == cnn.FRAME_INPUT[(band, "cnn1d_raw")][0]


def test_physics_scalar_differs_from_the_robust_standardized_signals_scalar(config):
    """Trap 13, made concrete on the 77 GHz band where DC *is* the target band: robust
    standardization removes the median, so a standardized signal would give a completely
    different (and meaningless) DC-vs-motion ratio. The physics path must not do it."""
    from dehyd.preprocess.standardize import robust_standardize

    rng = np.random.default_rng(3)
    signal = 5.0 + rng.normal(size=N_SLOW)          # a strong DC pedestal
    raw = physics_frame_scalar(config, "77ghz", signal)
    standardized = physics_frame_scalar(config, "77ghz", robust_standardize(signal))
    assert abs(raw - standardized) > 1.0


def test_build_physics_spine_is_one_row_per_session_with_the_median_over_its_frames(
    store_10, config
):
    """O-M9-4: the per-frame scalar becomes a session value by the frozen
    `frame_to_session_aggregation: median` — the analysis unit is the session."""
    store_dir, sessions = store_10
    spine = build_physics_spine(config, "10ghz", sessions, store_dir)

    assert spine.family == "physics"
    assert len(spine.subjects) == len(sessions)
    assert spine.n_frames.tolist() == [N_FRAMES_PER_SESSION] * len(sessions)
    assert spine.delta_m_pct.tolist() == [s["delta_m_pct"] for s in sessions]

    # hand-recompute one session's value straight from the stored array
    row = 7
    target = sessions[row]
    store = np.load(store_dir / "features" / "10ghz" /
                    f"s{target['subject']}_{target['session_name']}.npz")
    per_frame = [physics_frame_scalar(config, "10ghz", f) for f in store["sig__raw_beat"]]
    assert spine.physics_scalar[row] == pytest.approx(float(np.median(per_frame)))


def test_the_physics_scalar_is_finite_on_every_frame_of_the_spine(store_10, config):
    """The frozen finite-output assertion (`:936-938`), over every QC-passed frame."""
    store_dir, sessions = store_10
    spine = build_physics_spine(config, "10ghz", sessions, store_dir)
    assert np.all(np.isfinite(spine.physics_scalar))


def _mutate_spine_held_out(spine, subject, *, seed=11):
    """Value mutation of the held-out subject's rows only — session membership untouched."""
    rng = np.random.default_rng(seed)
    mutated = dataclasses.replace(
        spine,
        physics_scalar=spine.physics_scalar.copy(),
        delta_m_pct=spine.delta_m_pct.copy(),
    )
    rows = mutated.subjects == subject
    mutated.physics_scalar[rows] = rng.normal(size=int(rows.sum())) * 20 + 50
    mutated.delta_m_pct[rows] = rng.normal(size=int(rows.sum())) * 20 + 50
    return mutated


def test_the_physics_linear_fit_is_train_only_at_both_cv_levels(store_10, config):
    """`:942-944` + O-M9-4: a per-fold least-squares line on the outer-TRAINING sessions.
    Mutating the held-out subject's scalars and labels must leave every fitted coefficient
    bytewise identical and move only that subject's predictions."""
    store_dir, sessions = store_10
    spine = build_physics_spine(config, "10ghz", sessions, store_dir)
    base = run_cheap_baseline(spine)
    mutated = run_cheap_baseline(_mutate_spine_held_out(spine, 1))

    fold_a = next(r for r in base if r.test_subject == 1)
    fold_b = next(r for r in mutated if r.test_subject == 1)
    for rec_a, rec_b in zip(fold_a.fits, fold_b.fits, strict=True):
        assert rec_a.quantity == rec_b.quantity == "physics_linear_fit"
        assert rec_a.role == rec_b.role
        assert 1 not in rec_a.subjects
        for key in rec_a.params:
            assert rec_a.params[key].tobytes() == rec_b.params[key].tobytes()
    assert fold_a.inner_scores == fold_b.inner_scores
    # the power companion: the held-out subject's own predictions DID move
    assert fold_a.test_predictions.tobytes() != fold_b.test_predictions.tobytes()

    # and other folds' fits, which train ON subject 1, are genuinely different
    other_a = next(r for r in base if r.test_subject == 2)
    other_b = next(r for r in mutated if r.test_subject == 2)
    assert (other_a.fits[-1].params["slope"].tobytes()
            != other_b.fits[-1].params["slope"].tobytes())


def test_run_physics_and_run_session_index_share_the_folds_and_the_inner_scoring(
    store_10, config
):
    """`:917-919`: every baseline runs under the identical outer folds, and each is scored
    on the inner folds too so the composite procedure has a real inner-CV score."""
    store_dir, sessions = store_10
    physics = run_physics(config, "10ghz", sessions, store_dir)
    session_index = run_session_index(config, "10ghz", sessions)

    folds = selectable_folds(sorted({s["subject"] for s in sessions}))
    assert [r.fold_id for r in physics] == list(range(len(folds)))
    assert [r.test_subject for r in physics] == [f.test_subject for f in folds]
    assert [r.test_subject for r in session_index] == [f.test_subject for f in folds]
    for a, b in zip(physics, session_index, strict=True):
        assert a.test_session_idx.tolist() == b.test_session_idx.tolist()
        assert a.n_inner_folds == b.n_inner_folds == len(folds[a.fold_id].inner_folds)
        assert len(a.inner_scores) == len(b.inner_scores) == a.n_inner_folds
        assert math.isfinite(a.inner_score) and math.isfinite(b.inner_score)


def test_session_index_baseline_reuses_the_shared_model_verbatim(store_10, config):
    """`:850-854`, `:915-916`: band-agnostic and shared — the fitted quantity is
    `models/baselines.fit_session_index_baseline`'s own record, not a re-implementation."""
    from dehyd.models import baselines as baselines_mod

    _store_dir, sessions = store_10
    results = run_session_index(config, "10ghz", sessions)
    fold = results[0]
    outer = fold.fits[-1]
    assert outer.quantity == "session_index_means"
    assert fold.test_subject not in outer.subjects

    expected = baselines_mod.fit_session_index_baseline(
        np.array([s["subject"] for s in sessions]),
        np.array([s["session_idx"] for s in sessions]),
        np.array([s["delta_m_pct"] for s in sessions]),
        sorted(set(s["subject"] for s in sessions) - {fold.test_subject}),
    )
    predicted = baselines_mod.predict_session_index(expected.model, fold.test_session_idx)
    assert fold.test_predictions.tolist() == predicted.tolist()


def test_assert_mechanism_ok_d_catches_a_fit_that_saw_the_held_out_subject(store_10, config):
    store_dir, sessions = store_10
    results = run_physics(config, "10ghz", sessions, store_dir)
    subjects = sorted({s["subject"] for s in sessions})
    exp_d.assert_mechanism_ok_d(results, subjects)          # the real run passes

    leaked = dataclasses.replace(
        results[0],
        fits=[dataclasses.replace(results[0].fits[-1],
                                  subjects=frozenset(subjects))],
    )
    with pytest.raises(AssertionError):
        exp_d.assert_mechanism_ok_d([leaked] + list(results[1:]), subjects)


# ----------------------------------------------------- the four per-family artifacts


def _artifact_lineage(run_group_id="rg_test"):
    return {"analysis_commit": "c0ffee1234", "config_hash": "cfg0001",
            "run_group_id": run_group_id}


def _write_cheap_family(config, band, family, results, out_dir, *, deterministic=True):
    return write_family_artifacts(
        band, family, out_dir,
        prediction_rows=[row for r in results for row in cheap_prediction_rows(r)],
        selection_rows=[cheap_selection_row(r) for r in results],
        deterministic=deterministic,
        bootstrap_b=config.stats.bootstrap_b,
        rng_seed=config.run.seed,
        skip_threshold_pct=config.stats.undefined_metric_skip_threshold_pct,
        lineage=_artifact_lineage(),
    )


def test_write_family_artifacts_emits_the_four_named_files(store_10, fast_ci_config, tmp_path):
    store_dir, sessions = store_10
    results = run_physics(fast_ci_config, "10ghz", sessions, store_dir)
    paths = _write_cheap_family(fast_ci_config, "10ghz", "physics", results, tmp_path)

    assert set(paths) == {"predictions", "metrics", "selection", "per_subject"}
    assert paths["predictions"].name == "predictions_physics_10ghz.csv"
    assert paths["metrics"].name == "metrics_physics_10ghz.json"
    assert paths["selection"].name == "selection_physics_10ghz.csv"
    assert paths["per_subject"].name == "per_subject_physics_10ghz.csv"

    header = next(csv.reader(paths["predictions"].open(encoding="utf-8")))
    assert header == ["fold_id", "subject", "session_idx", "seed", "y_true_delta_m_pct",
                      "y_pred", "n_frames_aggregated"]

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["conditional_exploratory"] is True
    assert metrics["deterministic"] is True
    assert metrics["n_seeds"] == 1
    assert set(metrics) >= {"per_subject_session_mae", "subject_balanced_mae", "session_rmse",
                            "pooled_pearson_r", "n_eval", "fold_ids", "lineage"}
    assert metrics["lineage"] == _artifact_lineage()


def test_a_deterministic_family_writes_one_seed_row_per_session_never_five(
    store_10, fast_ci_config, tmp_path
):
    """`:644-649` forbids ensembling and the schema forbids pretending a deterministic
    family has five observations: seed 1 once, `deterministic: true` in the metrics JSON."""
    store_dir, sessions = store_10
    results = run_session_index(fast_ci_config, "10ghz", sessions)
    paths = _write_cheap_family(fast_ci_config, "10ghz", "session_index", results, tmp_path)

    rows = list(csv.DictReader(paths["predictions"].open(encoding="utf-8")))
    assert {r["seed"] for r in rows} == {"1"}
    # every subject is the test subject of exactly one fold, so every session appears once
    assert len(rows) == len(sessions)
    assert json.loads(paths["metrics"].read_text(encoding="utf-8"))["deterministic"] is True


def test_load_family_artifacts_recomputes_the_per_subject_vector_from_the_predictions(
    store_10, fast_ci_config, tmp_path
):
    store_dir, sessions = store_10
    results = run_physics(fast_ci_config, "10ghz", sessions, store_dir)
    _write_cheap_family(fast_ci_config, "10ghz", "physics", results, tmp_path)
    loaded = load_family_artifacts(tmp_path, "10ghz", "physics")

    # independently: per subject, mean over seeds of the mean |error| over its sessions
    expected = {}
    for r in results:
        expected[r.test_subject] = float(
            np.mean(np.abs(r.test_truth - r.test_predictions))
        )
    assert loaded.per_subject_mae == pytest.approx(expected)
    assert loaded.deterministic is True
    assert loaded.inner_score_by_fold == pytest.approx({r.fold_id: r.inner_score for r in results})


@pytest.mark.parametrize(
    "victim", ["predictions_physics_10ghz.csv", "metrics_physics_10ghz.json",
               "selection_physics_10ghz.csv", "per_subject_physics_10ghz.csv"]
)
def test_load_family_artifacts_refuses_an_incomplete_family_naming_the_file(
    store_10, fast_ci_config, tmp_path, victim
):
    store_dir, sessions = store_10
    results = run_physics(fast_ci_config, "10ghz", sessions, store_dir)
    _write_cheap_family(fast_ci_config, "10ghz", "physics", results, tmp_path)
    (tmp_path / victim).unlink()
    with pytest.raises(ExpDError, match=victim):
        load_family_artifacts(tmp_path, "10ghz", "physics")


def test_load_family_artifacts_refuses_a_metrics_json_that_no_longer_recomputes(
    store_10, fast_ci_config, tmp_path
):
    """The merge acceptance rule: the metrics JSON's per-subject vector must recompute
    exactly from the predictions CSV, so a hand-edited or stale metrics file is rejected."""
    store_dir, sessions = store_10
    results = run_physics(fast_ci_config, "10ghz", sessions, store_dir)
    paths = _write_cheap_family(fast_ci_config, "10ghz", "physics", results, tmp_path)

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    first = sorted(metrics["per_subject_session_mae"])[0]
    metrics["per_subject_session_mae"][first] += 0.5
    paths["metrics"].write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ExpDError, match="per-subject"):
        load_family_artifacts(tmp_path, "10ghz", "physics")


def test_load_family_artifacts_refuses_a_per_subject_csv_that_no_longer_recomputes(
    store_10, fast_ci_config, tmp_path
):
    """All four advertised family artifacts are validated, not merely required to exist."""
    store_dir, sessions = store_10
    results = run_physics(fast_ci_config, "10ghz", sessions, store_dir)
    paths = _write_cheap_family(fast_ci_config, "10ghz", "physics", results, tmp_path)

    path = paths["per_subject"]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    fieldnames = list(rows[0])
    rows[0]["seed_averaged_session_mae"] = str(
        float(rows[0]["seed_averaged_session_mae"]) + 999.0
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ExpDError, match="per_subject_physics_10ghz.csv.*does not recompute"):
        load_family_artifacts(tmp_path, "10ghz", "physics")


@pytest.mark.parametrize(
    "field, changed_value",
    [
        ("y_true_delta_m_pct", "12345.0"),
        ("fold_id", "1"),
        ("n_frames_aggregated", "99"),
    ],
)
def test_prediction_matrix_refuses_session_metadata_that_changes_between_seeds(
    field, changed_value
):
    rows = [
        {
            "fold_id": "0", "subject": "1", "session_idx": "0", "seed": "1",
            "y_true_delta_m_pct": "-0.5", "y_pred": "-0.4", "n_frames_aggregated": "3",
        },
        {
            "fold_id": "0", "subject": "1", "session_idx": "0", "seed": "2",
            "y_true_delta_m_pct": "-0.5", "y_pred": "-0.6", "n_frames_aggregated": "3",
        },
    ]
    rows[1][field] = changed_value

    with pytest.raises(ExpDError, match=field):
        exp_d._prediction_matrix(rows)


def test_load_family_artifacts_refuses_a_selection_budget_that_is_not_its_own_median(
    tmp_path, fast_ci_config
):
    """The other merge acceptance rule, on the CNN schema: `epoch_budget` must equal the
    median of the per-(inner fold x seed) counts the row itself lists."""
    frames = _tiny_frames()
    folds = selectable_folds(sorted(set(frames.subjects.tolist())))
    results = [_fake_cnn_fold(frames, fold) for fold in folds]
    write_family_artifacts(
        "77ghz", "cnn1d_raw", tmp_path,
        prediction_rows=[row for i, r in enumerate(results) for row in cnn_prediction_rows(r, i)],
        selection_rows=[cnn_selection_row(r, i) for i, r in enumerate(results)],
        deterministic=False, bootstrap_b=fast_ci_config.stats.bootstrap_b,
        rng_seed=fast_ci_config.run.seed, skip_threshold_pct=5.0, lineage=_artifact_lineage(),
    )

    path = tmp_path / "selection_cnn1d_raw_77ghz.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    fieldnames = list(rows[0])
    rows[0]["epoch_budget"] = str(int(rows[0]["epoch_budget"]) + 3)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ExpDError, match="epoch budget"):
        load_family_artifacts(tmp_path, "77ghz", "cnn1d_raw")


def test_load_family_artifacts_refuses_a_selection_fold_mapped_to_the_wrong_subject(
    store_10, fast_ci_config, tmp_path
):
    store_dir, sessions = store_10
    results = run_physics(fast_ci_config, "10ghz", sessions, store_dir)
    paths = _write_cheap_family(fast_ci_config, "10ghz", "physics", results, tmp_path)

    path = paths["selection"]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    fieldnames = list(rows[0])
    rows[0]["test_subject"] = rows[1]["test_subject"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ExpDError, match="fold-to-subject mapping"):
        load_family_artifacts(tmp_path, "10ghz", "physics")


# ------------------------------------------------------ the CNN fold-array shard/merge


def _tiny_frames(n_subjects=4, n_sessions=2, n_frames=3):
    """A minimal `FramesD` — the census and the merge only ever read its membership."""
    subjects, session_row, frame_ids, y = [], [], [], []
    s_subjects, s_idx, s_delta = [], [], []
    row = 0
    for subject in range(1, n_subjects + 1):
        for session_idx in range(n_sessions):
            delta = -(0.5 * session_idx + 0.1 * subject)
            for frame in range(n_frames):
                subjects.append(subject)
                session_row.append(row)
                frame_ids.append(frame)
                y.append(delta)
            s_subjects.append(subject)
            s_idx.append(session_idx)
            s_delta.append(delta)
            row += 1
    return FramesD(
        band="77ghz", family="cnn1d_raw",
        subjects=np.array(subjects), session_row=np.array(session_row),
        frame_ids=np.array(frame_ids), X=np.zeros((len(subjects), 1, 4)),
        y=np.array(y), session_subjects=np.array(s_subjects),
        session_idx=np.array(s_idx), session_delta_m_pct=np.array(s_delta),
    )


SHARD_SEEDS = (1, 2, 3, 4, 5)


def _fake_cnn_fold(frames, fold, *, seeds=SHARD_SEEDS, bias=0.0):
    """A `CnnFoldResult` with fabricated predictions. The nested path itself is pinned by
    the step-7 tests; the shard/merge machinery only reads the result's shape."""
    positions = np.flatnonzero(frames.session_subjects == fold.test_subject)
    truth = frames.session_delta_m_pct[positions]
    n_frames = np.array([int(np.count_nonzero(frames.session_row == p)) for p in positions])
    outcomes = [
        CnnSeedOutcome(seed=s, session_predictions=truth + bias + 0.01 * s,
                       session_mae=abs(bias + 0.01 * s))
        for s in seeds
    ]
    grid = [{"lr": 3e-4, "weight_decay": 0.0}, {"lr": 1e-3, "weight_decay": 0.0}]
    return CnnFoldResult(
        band=frames.band, family=frames.family, test_subject=fold.test_subject,
        train_subjects=fold.train_subjects, n_inner_folds=len(fold.inner_folds),
        config_grid=grid,
        per_config_scores=[CandidateScore("cfg0", 0.5, 0, 4, 0.1),
                           CandidateScore("cfg1", 0.4, 0, 4, 0.1)],
        selected_config_index=1, selected_config=grid[1], epoch_budget=4,
        selected_epoch_counts=[3, 4, 4, 5, 6, 4], inner_results=[], final_fits=[],
        seed_outcomes=outcomes, test_session_subjects=frames.session_subjects[positions],
        test_session_idx=frames.session_idx[positions], test_session_truth=truth,
        test_n_frames_aggregated=n_frames,
    )


def _sha_lines(lines):
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def test_realized_fold_census_hashes_frame_and_session_identities_separately():
    """Two hashes, not one (§2.8): the frame hash cannot validate a session-level CSV, so a
    substituted session at unchanged `n_session_rows` would pass a count-only check. Both
    are re-derived here from the canonical string forms named in the plan."""
    frames = _tiny_frames()
    fold = selectable_folds([1, 2, 3, 4])[0]
    census = realized_fold_census(frames, fold, SHARD_SEEDS)

    frame_ids = sorted(f"{s}|{si}|{fid}" for s, si, fid in [
        (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2)])
    session_ids = sorted({"1|0", "1|1"})
    assert census == {
        "test_subject": 1,
        "n_frame_rows": 6,
        "n_session_rows": 2,
        "frame_rows_sha256": _sha_lines(frame_ids),
        "session_rows_sha256": _sha_lines(session_ids),
        "seed_set": [1, 2, 3, 4, 5],
    }
    assert census["frame_rows_sha256"] != census["session_rows_sha256"]


def test_expected_test_rows_by_fold_covers_exactly_the_selectable_folds_by_position():
    """Trap 14: fold ids are POSITIONS in the selectable-fold list, not subject ids."""
    frames = _tiny_frames()
    expected = expected_test_rows_by_fold(frames, SHARD_SEEDS)
    assert sorted(expected) == ["0", "1", "2", "3"]
    assert [expected[k]["test_subject"] for k in sorted(expected)] == [1, 2, 3, 4]


def _init_group(tmp_path, frames, *, band="77ghz", family="cnn1d_raw",
                commit="c0ffee1234", config_hash="cfg0001", seeds=SHARD_SEEDS):
    """The run-group provenance a `--init-run-group` task writes (the entrypoint that calls
    `record_run` is step 9; this is the same payload shape, `extra` NESTED as `record_run`
    nests it)."""
    run_dir = tmp_path / "rg_expd"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps({
        "git": {"commit": commit},
        "seed": 20260730,
        "config": {"stats": {"bootstrap_b": 200, "undefined_metric_skip_threshold_pct": 5.0}},
        "extra": {
            "stage": "exp-d-cnn-group", "band": band, "family": family,
            "config_hash": config_hash,
            "expected_subjects": sorted(set(frames.subjects.tolist())),
            "expected_test_rows_by_fold": expected_test_rows_by_fold(frames, seeds),
        },
    }, indent=2, sort_keys=True), encoding="utf-8")
    return run_dir


def _write_all_shards(run_dir, frames, *, commit="c0ffee1234", config_hash="cfg0001",
                      seeds=SHARD_SEEDS, biases=None):
    folds = selectable_folds(sorted(set(frames.subjects.tolist())))
    for fold_id, fold in enumerate(folds):
        bias = 0.0 if biases is None else biases[fold_id]
        write_fold_shard(
            _fake_cnn_fold(frames, fold, seeds=seeds, bias=bias), frames, fold, fold_id,
            run_dir, band=frames.band, family=frames.family, seeds=seeds,
            run_group_id=run_dir.name, analysis_commit=commit, config_hash=config_hash,
        )
    return folds


def test_merge_produces_the_family_summary_only_when_every_fold_is_present(tmp_path):
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)

    merged = merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)
    assert merged["complete"] is True
    assert merged["state"] == "complete"
    assert merged["completed_folds"] == [0, 1, 2, 3]
    assert merged["missing_folds"] == [] and merged["noop_folds"] == []
    for name in ("predictions", "metrics", "selection", "per_subject"):
        assert merged["artifacts"][name].exists()

    loaded = load_family_artifacts(run_dir, "77ghz", "cnn1d_raw")
    assert sorted(loaded.per_subject_mae) == [1, 2, 3, 4]
    assert loaded.deterministic is False


def test_a_partial_merge_is_a_named_non_reportable_state_not_a_smaller_cohort(tmp_path):
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)
    (run_dir / "exp_d_cnn1d_raw_77ghz_fold2.json").unlink()

    merged = merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)
    assert merged["complete"] is False
    assert merged["state"] == "partial_non_reportable"
    assert merged["missing_folds"] == [2]
    assert merged["artifacts"] is None
    assert not (run_dir / "metrics_cnn1d_raw_77ghz.json").exists()


def test_merge_distinguishes_a_noop_marker_from_a_missing_shard(tmp_path):
    """Trap 14: a task whose fold index exceeds the selectable list exits 0 with a NAMED
    marker; the merge must never confuse that with a crashed task."""
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)
    (run_dir / "exp_d_cnn1d_raw_77ghz_fold1.json").unlink()
    write_noop_marker(run_dir, band="77ghz", family="cnn1d_raw", fold_id=1,
                      reason="fold index beyond the selectable list")
    write_noop_marker(run_dir, band="77ghz", family="cnn1d_raw", fold_id=9,
                      reason="fold index beyond the selectable list")

    merged = merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)
    assert merged["noop_folds"] == [1]
    assert merged["noop_out_of_range_folds"] == [9]
    assert merged["missing_folds"] == []
    assert merged["complete"] is False


@pytest.mark.parametrize(
    "field, value",
    [("analysis_commit", "deadbeef"), ("config_hash", "other"), ("band", "10ghz"),
     ("family", "spec2d_raw"), ("fold_id", 3), ("run_group_id", "somewhere_else"),
     ("test_subject", 2)],
)
def test_every_lineage_mismatch_field_is_rejected_by_name(tmp_path, field, value):
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)

    path = run_dir / "exp_d_cnn1d_raw_77ghz_fold0.json"
    shard = json.loads(path.read_text(encoding="utf-8"))
    shard[field] = value
    path.write_text(json.dumps(shard, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ExpDError, match=field):
        merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)


@pytest.mark.parametrize("field, value", [("fold_id", 3), ("test_subject", 2)])
def test_a_shards_selection_row_must_match_its_authoritative_fold(
    tmp_path, field, value
):
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)

    path = run_dir / "exp_d_cnn1d_raw_77ghz_fold0.json"
    shard = json.loads(path.read_text(encoding="utf-8"))
    shard["selection"][field] = value
    path.write_text(json.dumps(shard, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ExpDError, match=rf"selection\.{field}"):
        merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)


def test_a_shard_that_silently_drops_one_test_row_is_rejected_by_the_census(tmp_path):
    """(C7) Perfect lineage — same commit, config hash, family, band, fold id — and a
    plausible count, but one expected test frame is gone. Only the row census and
    `frame_rows_sha256` can see it; nothing in `record_run`'s manifest can."""
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)

    path = run_dir / "exp_d_cnn1d_raw_77ghz_fold0.json"
    shard = json.loads(path.read_text(encoding="utf-8"))
    shard["census"]["n_frame_rows"] -= 1
    shard["census"]["frame_rows_sha256"] = _sha_lines(sorted(
        ["1|0|0", "1|0|1", "1|1|0", "1|1|1", "1|1|2"]))     # one frame dropped
    path.write_text(json.dumps(shard, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ExpDError, match="frame_rows_sha256|n_frame_rows"):
        merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)


def _rewrite_fold_predictions(run_dir, fold_id, transform, band="77ghz", family="cnn1d_raw"):
    path = run_dir / f"exp_d_{family}_{band}_fold{fold_id}_predictions.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    fieldnames = list(rows[0])
    rows = transform(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_a_predictions_csv_missing_a_row_the_shard_still_counts_is_rejected(tmp_path):
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)
    _rewrite_fold_predictions(run_dir, 0, lambda rows: rows[1:])

    with pytest.raises(ExpDError, match="session_rows_sha256|seed"):
        merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)


def test_a_csv_substituting_a_session_at_unchanged_row_count_is_rejected(tmp_path):
    """(C14) The count-only check's blind spot: the number of rows is right, but one
    session identity is wrong. `session_rows_sha256` is what sees it."""
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)

    def substitute(rows):
        for row in rows:
            if row["session_idx"] == "1":
                row["session_idx"] = "4"       # a session this fold never held out
        return rows

    _rewrite_fold_predictions(run_dir, 0, substitute)
    with pytest.raises(ExpDError, match="session_rows_sha256"):
        merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)


def test_a_csv_duplicating_a_session_at_unchanged_row_count_is_rejected(tmp_path):
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)

    def duplicate(rows):
        for row in rows:
            row["session_idx"] = "0"           # every row becomes session 0
        return rows

    _rewrite_fold_predictions(run_dir, 0, duplicate)
    with pytest.raises(ExpDError, match="session_rows_sha256|duplicate"):
        merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)


def test_a_csv_missing_one_seed_of_one_session_is_rejected_by_the_cross_product(tmp_path):
    """Seed is deliberately NOT part of the row identity hashes; it is validated as an
    exact `session identities x seed_set` cross product instead."""
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)
    _rewrite_fold_predictions(
        run_dir, 0,
        lambda rows: [r for r in rows if not (r["seed"] == "3" and r["session_idx"] == "1")],
    )

    with pytest.raises(ExpDError, match="seed"):
        merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)


def test_a_predictions_csv_with_the_wrong_fold_id_is_rejected(tmp_path):
    frames = _tiny_frames()
    run_dir = _init_group(tmp_path, frames)
    _write_all_shards(run_dir, frames)

    def change_fold(rows):
        for row in rows:
            row["fold_id"] = "1"
        return rows

    _rewrite_fold_predictions(run_dir, 0, change_fold)
    with pytest.raises(ExpDError, match="fold_id"):
        merge_exp_d_folds("77ghz", "cnn1d_raw", run_dir)


# ------------------------------------------------------------------ the comparisons


def test_exp_d_rng_offsets_are_pairwise_distinct_including_the_other_experiments():
    """Trap 10's doctrine: fixed named blocks, never a running counter. Exp A occupies
    seed+0..3, Exp B +100..134, Exp C +200..212, Exp D +300..373."""
    offsets = exp_d._all_rng_offsets()
    assert RNG_OFFSET_EXPD_BASE == 300
    assert min(offsets) == 300 and max(offsets) == 373
    assert len(set(offsets)) == len(offsets)
    assert not set(offsets) & set(exp_b._all_rng_offsets())
    assert not set(offsets) & set(exp_c._all_rng_offsets())
    assert not set(offsets) & {0, 1, 2, 3}


def test_the_composite_and_holm_families_are_the_three_primary_variants_only():
    """O-M9-3: the matched-preprocessing ablations are reported AS ablations and enter no
    comparison family, so the Holm denominator stays exactly 3."""
    assert COMPOSITE_MEMBERS == ("cnn1d_raw", "spec2d_raw", "physics")
    assert "cnn1d_matched" not in COMPOSITE_MEMBERS
    assert "spec2d_matched" not in COMPOSITE_MEMBERS
    assert set(EXPD_FAMILIES) == {"cnn1d_raw", "cnn1d_matched", "spec2d_raw",
                                  "spec2d_matched", "physics", "session_index"}


# The comparison fixture: 4 subjects x 2 sessions, every family's per-(subject, seed) error
# fixed by hand so every downstream quantity can be recomputed with plain numpy.
COMPARE_SUBJECTS = (1, 2, 3, 4)
COMPARE_SESSIONS = (1, 2)
COMPARE_TRUTH = {s: np.array([-1.0 * s, -2.0 * s]) for s in COMPARE_SUBJECTS}

# deliberately MIXED-SIGN across seeds, so the seed-averaged per-subject MAE and the MAE of
# a seed-ensembled prediction are genuinely different numbers (the C13 discriminator).
CNN_SEED_ERRORS = (-0.5, -0.3, 0.1, 0.4, 0.6)


def _family_errors(family, subject, seed):
    base = {"cnn1d_raw": 0.10, "spec2d_raw": 0.35, "physics": 0.20,
            "cnn1d_matched": 0.55, "spec2d_matched": 0.65, "session_index": 0.45}[family]
    if family in ("physics", "session_index"):
        return np.full(2, base + 0.05 * subject)
    return np.full(2, base + 0.05 * subject + CNN_SEED_ERRORS[seed - 1])


def _family_seeds(family):
    return (1,) if family in ("physics", "session_index") else (1, 2, 3, 4, 5)


def _expected_per_subject_mae(family):
    """The frozen additive collapse (`:1193-1199`): per subject, average the per-seed
    per-subject MAEs. A deterministic family averages ONE value, which is that value."""
    out = {}
    for subject in COMPARE_SUBJECTS:
        per_seed = [float(np.mean(np.abs(_family_errors(family, subject, seed))))
                    for seed in _family_seeds(family)]
        out[subject] = float(np.mean(per_seed))
    return out


# per-fold inner scores: folds 0 and 2 make the raw 1D-CNN the best member, folds 1 and 3
# make physics the best — so the composite genuinely alternates across folds.
COMPOSITE_INNER = {
    "cnn1d_raw": {0: 0.10, 1: 0.90, 2: 0.11, 3: 0.95},
    "spec2d_raw": {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50},
    "physics": {0: 0.80, 1: 0.20, 2: 0.85, 3: 0.15},
    "cnn1d_matched": {i: 0.01 for i in range(4)},     # would win if ablations were members
    "spec2d_matched": {i: 0.01 for i in range(4)},
    "session_index": {i: 0.70 for i in range(4)},
}
EXPECTED_COMPOSITE_WINNER = {0: "cnn1d_raw", 1: "physics", 2: "cnn1d_raw", 3: "physics"}


def _write_comparison_family(root, band, family, config, *, error_fn=_family_errors):
    out_dir = root / family
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows, selection_rows = [], []
    for fold_id, subject in enumerate(COMPARE_SUBJECTS):
        truth = COMPARE_TRUTH[subject]
        for seed in _family_seeds(family):
            pred = truth + error_fn(family, subject, seed)
            for k, session_idx in enumerate(COMPARE_SESSIONS):
                prediction_rows.append({
                    "fold_id": fold_id, "subject": subject, "session_idx": session_idx,
                    "seed": seed, "y_true_delta_m_pct": float(truth[k]),
                    "y_pred": float(pred[k]), "n_frames_aggregated": 3,
                })
        selection_rows.append({
            "fold_id": fold_id, "test_subject": subject, "selected_config": "n/a",
            "learning_rate": "", "weight_decay": "", "epoch_budget": "",
            "selected_epoch_counts": [], "per_config_inner_scores": [COMPOSITE_INNER[family][fold_id]],
            "inner_score": COMPOSITE_INNER[family][fold_id], "n_inner_folds": 3,
            "fitted_coefficients": {},
        })
    write_family_artifacts(
        band, family, out_dir, prediction_rows=prediction_rows, selection_rows=selection_rows,
        deterministic=family in ("physics", "session_index"),
        bootstrap_b=config.stats.bootstrap_b, rng_seed=config.run.seed,
        skip_threshold_pct=config.stats.undefined_metric_skip_threshold_pct,
        lineage=_artifact_lineage(),
    )
    return out_dir


RADAR_ERRORS = {1: 0.30, 2: 0.55, 3: 0.25, 4: 0.80}


def _write_exp_a_run(path, band, *, commit="c0ffee1234", config=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "provenance.json").write_text(json.dumps({
        "git": {"commit": commit},
        "config": config or {"paths": {"results_dir": str(path)}, "model_grid": {"ridge_alpha": [1.0]}},
    }, indent=2, sort_keys=True), encoding="utf-8")
    with (path / f"predictions_{band}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "seed", "y_true", "y_pred"])
        for subject in COMPARE_SUBJECTS:
            for seed in (1, 2, 3, 4, 5):
                for k, truth in enumerate(COMPARE_TRUTH[subject]):
                    writer.writerow([subject, seed, truth, truth + RADAR_ERRORS[subject]])
    return path


@pytest.fixture(scope="module")
def comparison_inputs(tmp_path_factory, fast_ci_config):
    root = tmp_path_factory.mktemp("expd_compare")
    family_runs = {
        family: _write_comparison_family(root, "10ghz", family, fast_ci_config)
        for family in EXPD_FAMILIES
    }
    m7 = _write_exp_a_run(root / "m7_run", "10ghz")
    m9 = _write_exp_a_run(root / "m9_run", "10ghz")
    return root, family_runs, m9, m7


@pytest.fixture(scope="module")
def comparison_summary(comparison_inputs, fast_ci_config):
    _root, family_runs, m9, m7 = comparison_inputs
    radar = load_exp_a_radar("10ghz", m9, m7, analysis_commit="c0ffee1234")
    return summarize_exp_d("10ghz", fast_ci_config, family_runs, radar)


def test_load_exp_a_radar_refuses_predictions_that_are_not_bit_identical_to_m7(
    comparison_inputs, tmp_path
):
    """O-M9-5 / trap 17: a mismatch STOPS the milestone. Comparing against the fresh
    predictions anyway would convert a detected fault into a silent protocol change."""
    _root, _family_runs, m9, m7 = comparison_inputs
    drifted = _write_exp_a_run(tmp_path / "drifted", "10ghz")
    path = drifted / "predictions_10ghz.csv"
    rows = list(csv.reader(path.open(encoding="utf-8")))
    rows[1][3] = str(float(rows[1][3]) + 1e-9)          # one prediction, one ULP-scale drift
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    with pytest.raises(ExpDProtocolError, match="bit-identical"):
        load_exp_a_radar("10ghz", drifted, m7, analysis_commit="c0ffee1234")
    # and the clean pair passes
    assert load_exp_a_radar("10ghz", m9, m7, analysis_commit="c0ffee1234").bit_identity_verified


def test_load_exp_a_radar_refuses_a_run_at_the_wrong_commit_or_config(
    comparison_inputs, tmp_path
):
    _root, _family_runs, _m9, m7 = comparison_inputs
    wrong_commit = _write_exp_a_run(tmp_path / "wrong_commit", "10ghz", commit="deadbeef")
    with pytest.raises(ExpDProtocolError, match="commit"):
        load_exp_a_radar("10ghz", wrong_commit, m7, analysis_commit="c0ffee1234")

    wrong_config = _write_exp_a_run(
        tmp_path / "wrong_config", "10ghz",
        config={"paths": {"results_dir": "elsewhere"}, "model_grid": {"ridge_alpha": [7.0]}},
    )
    with pytest.raises(ExpDProtocolError, match="model_grid"):
        load_exp_a_radar("10ghz", wrong_config, m7, analysis_commit="c0ffee1234")


def test_load_exp_a_radar_requires_the_m7_reference_provenance(tmp_path):
    m7 = _write_exp_a_run(tmp_path / "m7_without_provenance", "10ghz")
    m9 = _write_exp_a_run(tmp_path / "m9", "10ghz")
    (m7 / "provenance.json").unlink()

    with pytest.raises(ExpDProtocolError, match="M7 reference.*provenance.json"):
        load_exp_a_radar("10ghz", m9, m7, analysis_commit="c0ffee1234")


def test_summarize_exp_d_refuses_a_radar_input_without_the_bit_identity_precondition(
    comparison_inputs, fast_ci_config
):
    """The precondition is STRUCTURAL: `summarize_exp_d` accepts only a `RadarReference`
    that `load_exp_a_radar` produced, so a bare run directory cannot slip past O-M9-5."""
    _root, family_runs, m9, _m7 = comparison_inputs
    with pytest.raises(ExpDProtocolError, match="O-M9-5"):
        summarize_exp_d("10ghz", fast_ci_config, family_runs, m9)
    with pytest.raises(ExpDProtocolError, match="O-M9-5"):
        summarize_exp_d("10ghz", fast_ci_config, family_runs,
                        dataclasses.replace(
                            load_exp_a_radar("10ghz", m9, _m7, analysis_commit="c0ffee1234"),
                            bit_identity_verified=False))


def test_summarize_exp_d_refuses_an_incomplete_family_set_naming_the_family(
    comparison_inputs, fast_ci_config
):
    _root, family_runs, m9, m7 = comparison_inputs
    radar = load_exp_a_radar("10ghz", m9, m7, analysis_commit="c0ffee1234")
    partial = {f: p for f, p in family_runs.items() if f != "spec2d_matched"}
    with pytest.raises(ExpDError, match="spec2d_matched"):
        summarize_exp_d("10ghz", fast_ci_config, partial, radar)


def test_the_pre_registered_primary_is_radar_versus_the_session_index_baseline(
    comparison_summary, fast_ci_config
):
    """`:1263-1266`: fixed in advance, not chosen from outer-test scores — and numerically
    the same paired comparison M7 reported, recomputed here from the artifacts."""
    radar = {s: RADAR_ERRORS[s] for s in COMPARE_SUBJECTS}
    baseline = _expected_per_subject_mae("session_index")
    diffs = np.array([radar[s] - baseline[s] for s in COMPARE_SUBJECTS])
    wstat, wp = M.wilcoxon_signed_rank(diffs)

    primary = comparison_summary["primary_vs_session_index"]
    assert primary["comparison"] == "radar_vs_session_index"
    assert primary["n_eval"] == 4
    assert primary["wilcoxon_statistic"] == pytest.approx(wstat)
    assert primary["wilcoxon_p"] == pytest.approx(wp)
    assert primary["mean_difference_radar_minus_baseline"]["point"] == pytest.approx(
        float(np.mean(diffs))
    )


def test_the_composite_picks_its_per_fold_winner_by_inner_cv_score_alone(comparison_summary):
    """`:1267-1274` + O-M9-3: the best of {raw 1D-CNN, raw spectrogram, physics} INSIDE each
    outer fold, by inner CV. The ablations have the best inner scores in this fixture and
    must never be selected."""
    rows = {row["subject"]: row for row in comparison_summary["composite"]["per_fold"]}
    for fold_id, subject in enumerate(COMPARE_SUBJECTS):
        assert rows[subject]["selected_family"] == EXPECTED_COMPOSITE_WINNER[fold_id]
        assert rows[subject]["inner_score"] == pytest.approx(
            COMPOSITE_INNER[EXPECTED_COMPOSITE_WINNER[fold_id]][fold_id]
        )
    assert {r["selected_family"] for r in rows.values()} <= set(COMPOSITE_MEMBERS)


def test_the_composite_is_spliced_at_the_per_subject_seed_averaged_metric_level(
    comparison_summary
):
    """(C13) Each subject's composite value is its fold's winning family's OWN seed-averaged
    per-subject MAE — a deterministic family contributes one value, never five copies, and
    no prediction is ever averaged across seeds."""
    per_family = {f: _expected_per_subject_mae(f) for f in COMPOSITE_MEMBERS}
    expected = {
        subject: per_family[EXPECTED_COMPOSITE_WINNER[fold_id]][subject]
        for fold_id, subject in enumerate(COMPARE_SUBJECTS)
    }
    rows = {row["subject"]: row for row in comparison_summary["composite"]["per_fold"]}
    for subject, value in expected.items():
        assert rows[subject]["per_subject_mae"] == pytest.approx(value)
        assert rows[subject]["n_seeds_averaged"] == (
            1 if EXPECTED_COMPOSITE_WINNER[COMPARE_SUBJECTS.index(subject)] == "physics" else 5
        )

    diffs = np.array([RADAR_ERRORS[s] - expected[s] for s in COMPARE_SUBJECTS])
    assert comparison_summary["composite"]["mean_difference_radar_minus_composite"][
        "point"] == pytest.approx(float(np.mean(diffs)))


def test_a_prediction_level_splice_would_give_a_different_answer(comparison_summary):
    """The discriminator for C13: with mixed-sign per-seed errors, the MAE of the
    seed-ENSEMBLED prediction is a genuinely different number from the seed-averaged
    per-subject MAE, so an implementation that spliced predictions fails the test above."""
    subject = 1
    per_seed = [np.abs(_family_errors("cnn1d_raw", subject, seed))
                for seed in _family_seeds("cnn1d_raw")]
    seed_averaged = float(np.mean([v.mean() for v in per_seed]))
    ensembled = float(np.mean(np.abs(np.mean(
        [_family_errors("cnn1d_raw", subject, seed) for seed in _family_seeds("cnn1d_raw")],
        axis=0))))
    assert abs(seed_averaged - ensembled) > 0.1

    rows = {row["subject"]: row for row in comparison_summary["composite"]["per_fold"]}
    assert rows[subject]["per_subject_mae"] == pytest.approx(seed_averaged)
    assert rows[subject]["per_subject_mae"] != pytest.approx(ensembled)


def test_the_per_family_family_is_holm_corrected_at_exactly_three(comparison_summary,
                                                                 fast_ci_config):
    """`:1275-1277` + O-M9-3: a Holm family of exactly 3, pinned strictly stronger than a
    len-2 family would be on the same p-values."""
    block = comparison_summary["per_family_exploratory"]
    assert block["holm_family_size"] == fast_ci_config.stats.holm_family_baseline_per_family == 3
    assert sorted(block["families"]) == sorted(COMPOSITE_MEMBERS)

    raw_p = [block["families"][f]["wilcoxon_p"] for f in COMPOSITE_MEMBERS]
    expected = M.holm_adjusted(raw_p, family_size=3)
    assert [block["families"][f]["holm_p"] for f in COMPOSITE_MEMBERS] == pytest.approx(expected)
    weaker = M.holm_adjusted(raw_p, family_size=2)
    assert any(a > b for a, b in zip(expected, weaker, strict=True))


def test_the_ablations_are_descriptive_only_and_carry_no_comparison(comparison_summary):
    """O-M9-3: reported as ablations — their own MAE and CI, no p-value, no Holm slot."""
    ablations = comparison_summary["ablations_descriptive"]
    assert sorted(ablations) == ["cnn1d_matched", "spec2d_matched"]
    for block in ablations.values():
        assert "subject_balanced_mae" in block
        assert "wilcoxon_p" not in block and "holm_p" not in block
    assert set(comparison_summary["per_family_exploratory"]["families"]) == set(COMPOSITE_MEMBERS)


def test_every_comparison_ci_is_labelled_conditional_exploratory(comparison_summary):
    assert comparison_summary["conditional_exploratory"] is True
    assert comparison_summary["n_eval"] == 4


def test_write_exp_d_comparison_reports_writes_the_composite_audit_trail(
    comparison_summary, tmp_path
):
    paths = write_exp_d_comparison_reports(comparison_summary, tmp_path, "10ghz")
    assert set(paths) == {"metrics", "composite"}
    assert paths["metrics"].name == "metrics_exp_d_10ghz.json"

    rows = list(csv.DictReader(paths["composite"].open(encoding="utf-8")))
    assert list(rows[0]) == ["subject", "selected_family", "inner_score", "per_subject_mae",
                             "n_seeds_averaged"]
    assert [r["selected_family"] for r in rows] == [
        EXPECTED_COMPOSITE_WINNER[i] for i in range(4)
    ]
