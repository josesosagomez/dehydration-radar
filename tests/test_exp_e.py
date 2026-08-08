"""T-M10-E: Experiment E — leave-one-path-group-out attribution under outer LOSO (A-M10-1).

Structured as plan §5.2 asks: the column->path grouping, the fail-closed metadata gate, the
band-aware physics labels, the fold computation and its leakage properties, the planted-signal
and correlated-surrogate fixtures that show what attribution does and does not mean, the
deterministic aggregation, and the artifact contracts.

Deliberately FAST. E has no inner CV and no candidate search, so none of this pays the
113-candidate tax that makes `test_exp_g.py` a ~176 s file: every run below is on a 6-path
synthetic store, serial, and the only real Kymatio banks built are the two geometry pins.
"""

import csv
import json

import numpy as np
import pytest

from dehyd.config import beat_band_hz, load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_b, exp_e
from dehyd.eval.exp_e import ExpEError
from dehyd.eval.splits import nested_loso_splits
from dehyd.features.extraction_77 import apply_order_log_77
from dehyd.features.pooling import aggregate_session, pool_stats_batch, session_feature_layout
from dehyd.features.store import (
    order_key,
    prelog77_key,
    prelog_key,
    raw77_key,
    raw_key,
    vec77_key,
    vec_key,
    write_session_store,
)
from dehyd.features.wst import apply_order_log
from dehyd.models.regressors import build_estimator, fit_pipeline

P, T, NFR = 6, 4, 3               # tiny path/time/frame dims for a fast synthetic store
CN_10, CN_77 = 1, 2               # 10 GHz magnitude is one channel; 77 GHz is the I/Q pair
ORDER = np.array([0, 1, 1, 2, 2, 2])
SIGNAL_PATH, SURROGATE_PATH = 3, 4


@pytest.fixture(scope="module")
def config_10():
    return load_config("configs/exp_a_regression.yaml", "configs/exp_b.yaml", "configs/exp_e.yaml")


@pytest.fixture(scope="module")
def config_77():
    return load_config("configs/exp_a_regression_77ghz.yaml", "configs/exp_b.yaml",
                       "configs/exp_e.yaml")


# ------------------------------------------------------------------------------- fixtures


def _bank(n_paths=P, n_time=T, fs_hz=520834.0, n_in=470):
    """A hand-built stand-in for the reconstructed Kymatio bank.

    Kymatio pads the unused order slots of xi/sigma/j with NaN, and that padding is exactly
    what the artifact must render as BLANK, so the fixture reproduces it rather than filling
    zeros. Order-0 has no xi at all; order-1 has xi1 only; order-2 has both.
    """
    xi = np.full((n_paths, 2), np.nan)
    sigma = np.full((n_paths, 2), np.nan)
    j = np.full((n_paths, 2), np.nan)
    order = ORDER if n_paths == P else np.array([0] + [1] * (n_paths - 1))
    first = np.array([0.008, 0.20, 0.008, 0.20, 0.40])      # xi1 for paths 1..5
    xi[1:, 0] = first[: n_paths - 1]
    sigma[1:, 0] = 0.02
    j[1:, 0] = [0.0, 1.0, 0.0, 1.0, 2.0][: n_paths - 1]
    second = np.array([0.15, 0.08, 0.04])
    order2 = np.flatnonzero(order == 2)
    xi[order2, 1] = second[: order2.size]
    sigma[order2, 1] = 0.01
    j[order2, 1] = [1.0, 2.0, 3.0][: order2.size]
    return {"order": order, "xi": xi, "sigma": sigma, "j": j,
            "n_paths": n_paths, "n_time": n_time, "pad_left": 0, "pad_right": 0,
            "padded_len": 64, "fs_hz": fs_hz, "n_in": n_in, "q": (10, 4), "invariance_ms": 0.2}


def _sessions(n_subjects=6, session_indices=(1, 2, 3, 4)):
    """An Exp-B spine: S0 already excluded, target linear in session (the clock) and subject."""
    return [{"subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
             "delta_m_pct": float(-0.3 * i - 0.05 * s)}
            for s in range(1, n_subjects + 1) for i in session_indices]


def _raw_tensor(rng, subject, n_channels, *, signal=0.0, surrogate=0.0):
    """One session's pre-log tensor, optionally with a subject-linear signal planted in one
    path (and, for the surrogate fixture, a copy of it in a second path).

    The residual target is `delta_m_pct - train_session_mean`, which for this spine is a
    per-subject constant, so a feature proportional to the subject id is exactly the kind of
    between-subject structure a LOSO-honest model could actually use.
    """
    raw = np.abs(rng.normal(size=(NFR, n_channels, P, T))) + 0.01
    if signal:
        raw[:, :, SIGNAL_PATH, :] += signal * subject
    if surrogate:
        raw[:, :, SURROGATE_PATH, :] += surrogate * subject
    return raw


def _write_store_10(store_dir, sessions, config, *, seed=0, signal=0.0, surrogate=0.0):
    rng = np.random.default_rng(seed)
    wst = config.wst
    eps = {1: wst.log_epsilon, 2: wst.log_epsilon}
    meta = {"order": ORDER}
    for s in sessions:
        npz = {}
        for ti in range(len(wst.tilings)):
            npz[order_key(ti)] = ORDER
            for gi in range(len(config.search_10ghz.range_gate_m)):
                for r in config.search_10ghz.reduction:
                    for c in config.search_10ghz.channel:
                        channels = CN_10 if c == "mag" else 2
                        raw = _raw_tensor(rng, s["subject"], channels,
                                          signal=signal, surrogate=surrogate)
                        off = aggregate_session(pool_stats_batch(
                            apply_order_log(raw, meta, wst, log_on=False), meta))
                        frozen = aggregate_session(pool_stats_batch(
                            apply_order_log(raw, meta, wst, log_on=True, epsilon_by_order=eps), meta))
                        npz[vec_key(gi, r, c, ti, "off")] = off
                        npz[vec_key(gi, r, c, ti, "frozen")] = frozen
                        npz[raw_key(gi, r, c, ti)] = raw
                        npz[prelog_key(gi, r, c, ti)] = np.array([raw.mean()] * 3)
        write_session_store("10ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR},
                            store_dir)


def _write_store_77(store_dir, sessions, config, *, seed=0):
    """The 77 GHz half, whose stored tensor carries TWO channels (real/imag slow-time),
    so the layout's channel count is exercised rather than assumed."""
    rng = np.random.default_rng(seed)
    meta = {"order": ORDER}
    for s in sessions:
        npz = {}
        for ti in range(len(config.wst77.tilings)):
            npz[order_key(ti)] = ORDER
            raw = _raw_tensor(rng, s["subject"], CN_77)
            for name, branch in (("off", "off"), ("frozen", "on_frozen_eps")):
                logged = np.stack([
                    apply_order_log_77(raw[i], meta, config.wst77, log_branch=branch)
                    for i in range(NFR)
                ])
                npz[vec77_key(ti, name)] = aggregate_session(pool_stats_batch(logged, meta))
            npz[raw77_key(ti)] = raw
            npz[prelog77_key(ti)] = np.array([raw.mean()] * 3)
        write_session_store("77ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR},
                            store_dir)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory, config_10):
    """ONE complete 10 GHz Exp E run on a synthetic store, with a planted signal path.

    Returns `(results, bank, path_of_column, sessions, store_dir)`. Serial: the parallel path
    is `fold_parallel.run_folds_parallel`, already pinned bit-identical to serial elsewhere.
    """
    store_dir = tmp_path_factory.mktemp("store_e10")
    sessions = _sessions()
    _write_store_10(store_dir, sessions, config_10, signal=0.9)
    results, bank, path_of_column = exp_e.run_exp_e(
        config_10, "10ghz", sessions, store_dir, bank=_bank(), n_workers=1
    )
    return results, bank, path_of_column, sessions, store_dir


@pytest.fixture(scope="module")
def e2e_77(tmp_path_factory, config_77):
    store_dir = tmp_path_factory.mktemp("store_e77")
    sessions = _sessions()
    _write_store_77(store_dir, sessions, config_77)
    results, bank, path_of_column = exp_e.run_exp_e(
        config_77, "77ghz", sessions, store_dir, bank=_bank(fs_hz=1953.125, n_in=256), n_workers=1
    )
    return results, bank, path_of_column, sessions, store_dir


# --------------------------------------------------------------- the fixed model form


def test_the_fixed_candidate_is_the_pre_registered_anchor_not_a_selected_model(config_10, config_77):
    """E interprets the frozen anchor. If this ever became "the best A/B model", each fold
    would be interpreting a different estimator and the importances would not be comparable."""
    c10 = exp_e.fixed_candidate(config_10, "10ghz")
    active = dict(c10.active)
    assert c10.family == "ridge" and c10.params() == {"alpha": 1.0}
    assert active["reduction"] == "A" and active["channel"] == "mag"
    assert active["tiling"] == "T1" and active["log_branch"] == "off"
    assert tuple(active["range_gate_m"]) == (1.0, 2.0)
    assert c10.feature_key[1:] == ("A", "mag", 0, "off")

    c77 = exp_e.fixed_candidate(config_77, "77ghz")
    assert dict(c77.active)["tiling"] == "T1_77"
    assert c77.feature_key == (0, "off")


def test_a_tuned_log_branch_is_refused(config_10):
    """A tuned-ε matrix is refit per fold, so its columns would not be comparable across
    folds — the one thing a fixed interpretability model exists to guarantee."""
    import dataclasses

    tuned = dataclasses.replace(config_10, exp_e=dataclasses.replace(
        config_10.exp_e, log_10ghz="on_tuned_eps"))
    with pytest.raises(ExpEError, match="not be comparable across folds"):
        exp_e.fixed_candidate(tuned, "10ghz")


def test_a_gate_absent_from_the_search_is_refused_rather_than_silently_remapped(config_10):
    import dataclasses

    moved = dataclasses.replace(config_10, exp_e=dataclasses.replace(
        config_10.exp_e, gate_10ghz_m=(0.1, 0.2)))
    with pytest.raises(ExpEError, match="no column block"):
        exp_e.fixed_candidate(moved, "10ghz")


def test_the_dead_a_m10_1_config_is_never_read():
    """`ExpEConfig.n_folds` / `fold_assignment` describe the standalone 4-fold permutation CV
    that A-M10-1 replaced. They stay in the frozen M6 record, but nothing may consume them —
    a structural check, because the failure mode is silent (a 4-fold split that still runs)."""
    import re

    text = open(exp_e.__file__, encoding="utf-8").read()
    # attribute ACCESS only, so `run_folds_parallel` (which merely contains "n_folds") and the
    # docstring prose that explains the rule are not false positives
    reads = re.findall(r"^.*\.(?:n_folds|fold_assignment)\b.*$", text, flags=re.MULTILINE)
    assert reads, "the dead fields vanished entirely — they are a frozen M6 record, not litter"
    # the ONLY permitted mentions are the module docstring's explanation of A-M10-1 and the
    # metrics-JSON note that RECORDS the dead values; neither drives any fold construction
    for line in reads:
        assert "ExpEConfig" in line or "dead config" in line, line


# ------------------------------------------------------- grouping: columns -> path groups


def test_every_model_column_maps_to_exactly_one_path_group(e2e, config_10):
    _, bank, path_of_column, sessions, store_dir = e2e
    n_columns = exp_e.stored_column_count(config_10, "10ghz", sessions[0], store_dir)
    layout = session_feature_layout(bank, bank["n_time"], CN_10, family="pooled")

    assert len(layout) == n_columns == path_of_column.size
    assert sorted(set(path_of_column.tolist())) == list(range(P))
    # the grouping IS the layout's path field — not a reimplementation of it
    assert path_of_column.tolist() == [element[2] for element in layout]


def test_a_path_group_is_both_aggregates_every_channel_and_all_segment_statistics(e2e, config_10):
    """The frozen definition of a path group, checked against the layout rather than a count:
    every column sharing a path_id, across frame_mean/frame_median, channels, and the
    global/half x mean/std cells."""
    _, bank, path_of_column, _, _ = e2e
    layout = session_feature_layout(bank, bank["n_time"], CN_10, family="pooled")
    group = [layout[i] for i in np.flatnonzero(path_of_column == SIGNAL_PATH)]

    assert {element[0] for element in group} == {"frame_mean", "frame_median"}
    assert {element[1] for element in group} == set(range(CN_10))
    assert len({(element[3], element[4]) for element in group}) == len(group) // 2
    assert np.bincount(path_of_column).tolist() == [len(group)] * P


def test_a_layout_that_does_not_describe_the_stored_matrix_fails_closed(config_10):
    """The other half of the §1.2 gate: a bank can be right about paths and still be paired
    with the wrong channel count or output length."""
    with pytest.raises(ExpEError, match="does not"):
        exp_e.column_path_index(config_10, "10ghz", _bank(), 999)


def test_the_77ghz_layout_uses_two_channels(e2e_77, config_77):
    _, bank, path_of_column, sessions, store_dir = e2e_77
    assert exp_e.n_channels(config_77, "77ghz") == CN_77
    assert path_of_column.size == exp_e.stored_column_count(config_77, "77ghz", sessions[0], store_dir)
    assert np.bincount(path_of_column).tolist() == [path_of_column.size // P] * P


# --------------------------------------------------------------- the fail-closed gate


def test_the_gate_accepts_a_store_whose_order_matches(e2e, config_10):
    _, bank, _, sessions, store_dir = e2e
    exp_e.assert_bank_matches_store(config_10, "10ghz", sessions, store_dir, bank)


def test_one_mismatched_session_stops_the_run(tmp_path, config_10):
    """Not a warning, and not a per-session skip: if the rebuilt bank is not the bank the
    features came from, every physics label and every column block is attributed to the wrong
    path, silently. The run must not produce a table at all."""
    sessions = _sessions(n_subjects=4)
    _write_store_10(tmp_path, sessions, config_10)

    victim = sessions[2]
    path = tmp_path / "features" / "10ghz" / f"s{victim['subject']}_{victim['session_name']}.npz"
    npz = dict(np.load(path))
    npz[order_key(0)] = np.array([0, 1, 1, 1, 2, 2])       # one path relabelled
    np.savez(path, **npz)

    with pytest.raises(ExpEError, match=f"subject {victim['subject']}"):
        exp_e.assert_bank_matches_store(config_10, "10ghz", sessions, tmp_path, _bank())


def test_a_path_count_change_is_caught_even_when_every_session_agrees(tmp_path, config_10):
    sessions = _sessions(n_subjects=4)
    _write_store_10(tmp_path, sessions, config_10)
    with pytest.raises(ExpEError, match="rebuilt bank has shape"):
        exp_e.assert_bank_matches_store(config_10, "10ghz", sessions, tmp_path, _bank(n_paths=5))


def test_a_store_without_the_order_metadata_is_refused(tmp_path, config_10):
    sessions = _sessions(n_subjects=4)
    _write_store_10(tmp_path, sessions, config_10)
    path = tmp_path / "features" / "10ghz" / f"s{sessions[0]['subject']}_{sessions[0]['session_name']}.npz"
    npz = {k: v for k, v in dict(np.load(path)).items() if k != order_key(0)}
    np.savez(path, **npz)

    with pytest.raises(ExpEError, match="no 'order__t0'"):
        exp_e.assert_bank_matches_store(config_10, "10ghz", sessions, tmp_path, _bank())


def test_the_reconstructed_10ghz_bank_is_the_pinned_geometry(config_10):
    """The real rebuild, pinned. The stores persist only `order`, so if kymatio or a frozen
    tiling ever changed this geometry, E would relabel every path without any other signal."""
    bank = exp_e.reconstruct_bank(config_10, "10ghz")
    order = np.asarray(bank["order"])
    assert bank["n_paths"] == 742 and bank["n_time"] == 7
    assert [int((order == o).sum()) for o in (0, 1, 2)] == [1, 55, 686]
    assert bank["fs_hz"] == 520834.0 and bank["n_in"] == 470 and bank["q"] == (10, 4)
    assert np.asarray(bank["xi"]).shape == (742, 2)


def test_the_reconstructed_77ghz_bank_is_the_pinned_geometry(config_77):
    bank = exp_e.reconstruct_bank(config_77, "77ghz")
    order = np.asarray(bank["order"])
    assert bank["n_paths"] == 424 and bank["n_time"] == 8
    assert [int((order == o).sum()) for o in (0, 1, 2)] == [1, 38, 385]
    assert bank["fs_hz"] == pytest.approx(1953.125) and bank["n_in"] == 256


# ----------------------------------------------------------- band-aware physics labels


def test_order_zero_makes_no_frequency_range_or_level_claim(config_10):
    row = exp_e.path_metadata_rows(config_10, "10ghz", _bank())[0]
    assert row["scattering_order"] == 0
    # BLANK, not zero: a 0.0 here would read as a measured centre frequency at DC
    for field in ("xi1_normalized", "xi2_normalized", "sigma1_normalized", "sigma2_normalized",
                  "j1", "j2", "xi1_hz", "xi2_hz", "coarse_range_m"):
        assert row[field] == "", field
    assert "standardization removed absolute level" in row["claim_limit"]


def test_10ghz_order_one_maps_xi_to_a_beat_frequency_and_a_coarse_scene_range(config_10):
    rows = exp_e.path_metadata_rows(config_10, "10ghz", _bank())
    row = next(r for r in rows if r["scattering_order"] == 1)

    assert row["input_domain"] == "fast_time_beat"
    assert row["xi1_hz"] == pytest.approx(row["xi1_normalized"] * 520834.0)
    assert row["coarse_range_m"] == pytest.approx(
        exp_e.coarse_range_m(row["xi1_hz"], config_10))
    assert "beat centre" in row["physical_label"]
    assert "not tissue penetration depth" in row["claim_limit"]


def test_10ghz_order_two_keeps_one_range_and_calls_xi2_a_modulation(config_10):
    row = next(r for r in exp_e.path_metadata_rows(config_10, "10ghz", _bank())
               if r["scattering_order"] == 2)
    assert row["xi2_hz"] != "" and "envelope" in row["physical_label"]
    assert "not a\nsecond range".replace("\n", " ") in row["claim_limit"].replace("\n", " ")
    # exactly ONE range number on an order-2 row
    assert row["physical_label"].count("coarse scene range") == 1


def test_no_77ghz_path_ever_receives_a_range_label(config_77):
    """The slow-time axis carries Doppler/modulation frequency. A range reading there would be
    a category error, and a signed velocity is not supported by the acquisition either."""
    rows = exp_e.path_metadata_rows(config_77, "77ghz", _bank(fs_hz=1953.125, n_in=256))
    assert {r["input_domain"] for r in rows} == {"slow_time_doppler"}
    for row in rows:
        assert row["coarse_range_m"] == ""
        assert "range" not in row["physical_label"]
        if row["scattering_order"] >= 1:
            assert "not range" in row["claim_limit"]
            assert "velocity" in row["claim_limit"]
    order1 = next(r for r in rows if r["scattering_order"] == 1)
    assert order1["xi1_hz"] == pytest.approx(order1["xi1_normalized"] * 1953.125)


def test_j_is_a_subsampling_index_and_is_never_converted_to_hz(config_10):
    bank = _bank()
    rows = exp_e.path_metadata_rows(config_10, "10ghz", bank)
    for row in rows:
        if row["j1"] == "":
            continue
        assert isinstance(row["j1"], int)
        assert row["j1"] == int(bank["j"][row["path_id"], 0])
        assert row["j1"] != row["xi1_hz"]


def test_the_beat_mapping_round_trips_against_the_shared_forward_physics(config_10):
    """`coarse_range_m` is the inverse of `config.beat_band_hz`, which the QC mask and the
    config cross-validation already share. Pinning the round trip is what keeps the physics in
    one place instead of two that drift."""
    for metres in (0.5, 1.0, 2.0, 3.7):
        beat = beat_band_hz((metres, metres), config_10.preprocess.bandwidth_hz,
                            config_10.preprocess.chirp_time_s)[0]
        assert exp_e.coarse_range_m(beat, config_10) == pytest.approx(metres)


def test_a_path_outside_the_analysis_gate_band_says_so(config_10):
    """Most of the T1 bank tiles beat frequencies the 1-2 m range gate band-passed away. A
    reader seeing "coarse scene range 71.66 m" needs to know that describes a filter the gated
    signal has little energy in, not a target at 72 m."""
    rows = exp_e.path_metadata_rows(config_10, "10ghz", _bank())
    inside = next(r for r in rows if r["xi1_normalized"] == 0.008)     # ~4.2 kHz, in band
    outside = next(r for r in rows if r["xi1_normalized"] == 0.40)     # ~208 kHz, far outside
    assert "lies inside the 1.0-2.0 m analysis gate band" in inside["claim_limit"]
    assert "lies OUTSIDE the 1.0-2.0 m analysis gate band" in outside["claim_limit"]


# ------------------------------------------------------- fold computation and leakage


def test_every_reported_fold_is_an_outer_loso_fold_holding_out_one_subject(e2e):
    results, bank, _, sessions, _ = e2e
    folds = {f.test_subject: f for f in nested_loso_splits(exp_b.evaluable_subjects_b(sessions))}
    assert [r.test_subject for r in results] == sorted(folds)
    for result in results:
        assert result.test_subject not in folds[result.test_subject].train_subjects
        assert result.ablated_mae.size == bank["n_paths"]
        assert np.all(np.isfinite(result.ablated_mae))


def test_the_full_and_ablated_models_use_identical_rows_and_train_only_scalers(e2e, config_10):
    """The load-bearing leakage test. Recomputed by hand from the provider: the ablated fit
    must see exactly the full fit's training rows with one column block removed BEFORE
    scaling, and be scored on exactly the full fit's held-out rows."""
    results, _, path_of_column, sessions, store_dir = e2e
    result = results[0]
    fold = next(f for f in nested_loso_splits(exp_b.evaluable_subjects_b(sessions))
                if f.test_subject == result.test_subject)

    provider = exp_b.SessionResidualFeatures("10ghz", sessions, store_dir, config_10)
    bundle = provider.data_for(exp_e.fixed_candidate(config_10, "10ghz"), fold.train_subjects)
    train = np.isin(bundle.subjects, sorted(fold.train_subjects))
    test = bundle.subjects == result.test_subject

    full = build_estimator("ridge", {"alpha": 1.0}, seed=config_10.run.seed)
    fit_pipeline(full, bundle.X[train], bundle.y[train])
    from dehyd.eval.metrics import equal_session_residual_mae
    assert result.full_mae == pytest.approx(equal_session_residual_mae(
        bundle.subjects[test], bundle.y[test], full.predict(bundle.X[test]),
        bundle.session_idx[test]))

    keep = path_of_column != SIGNAL_PATH
    ablated = build_estimator("ridge", {"alpha": 1.0}, seed=config_10.run.seed)
    fit_pipeline(ablated, bundle.X[train][:, keep], bundle.y[train])
    assert result.ablated_mae[SIGNAL_PATH] == pytest.approx(equal_session_residual_mae(
        bundle.subjects[test], bundle.y[test], ablated.predict(bundle.X[test][:, keep]),
        bundle.session_idx[test]))

    # the ablated scaler saw the TRAINING rows of the kept columns, and nothing else
    assert ablated.named_steps["scaler"].mean_ == pytest.approx(
        bundle.X[train][:, keep].mean(axis=0))
    assert ablated.named_steps["scaler"].n_features_in_ == int(keep.sum())


def test_mutating_the_held_out_subject_leaves_every_fitted_quantity_unchanged(tmp_path, config_10):
    """The frozen leakage property, in E's shape: the held-out subject's features may change
    the SCORE, but must not move a single fitted coefficient or scaler statistic."""
    sessions = _sessions(n_subjects=5)
    _write_store_10(tmp_path, sessions, config_10, signal=0.9)
    bank = _bank()
    before, _, _ = exp_e.run_exp_e(config_10, "10ghz", sessions, tmp_path, bank=bank, n_workers=1)

    victim = before[0].test_subject
    rng = np.random.default_rng(1234)
    for s in [s for s in sessions if s["subject"] == victim]:
        path = tmp_path / "features" / "10ghz" / f"s{victim}_{s['session_name']}.npz"
        npz = dict(np.load(path))
        for key in [k for k in npz if k.startswith("vec__")]:
            npz[key] = npz[key] + rng.normal(scale=5.0, size=npz[key].shape)
        np.savez(path, **npz)

    after, _, _ = exp_e.run_exp_e(config_10, "10ghz", sessions, tmp_path, bank=bank, n_workers=1)
    fitted_before = {r.test_subject: r.coefficient_rows for r in before}
    fitted_after = {r.test_subject: r.coefficient_rows for r in after}

    # the mutated subject's OWN fold is fit on the other subjects -> bit-identical fitted state
    assert fitted_after[victim] == fitted_before[victim]
    # ... and its score DID move, so the fixture actually perturbed something
    victim_before = next(r for r in before if r.test_subject == victim)
    victim_after = next(r for r in after if r.test_subject == victim)
    assert victim_after.full_mae != pytest.approx(victim_before.full_mae)
    # every OTHER fold trains on the mutated subject, so those fits are expected to move
    other = next(r.test_subject for r in before if r.test_subject != victim)
    assert fitted_after[other] != fitted_before[other]


def test_mutating_the_held_out_subject_s_TARGET_moves_no_train_derived_fit(tmp_path, config_10):
    """The other half of §5.1's mutation property, and the one that reaches E's second fitted
    quantity: the train-only session means mu_s.

    E residualizes with `mu_s` computed from the outer-TRAINING subjects, so changing the
    held-out subject's delta_m_pct must leave mu_s, the scaler and the ridge untouched while
    changing only that subject's residual targets — and therefore its score.
    """
    sessions = _sessions(n_subjects=5)
    _write_store_10(tmp_path, sessions, config_10, signal=0.9)
    bank = _bank()
    before, _, _ = exp_e.run_exp_e(config_10, "10ghz", sessions, tmp_path, bank=bank, n_workers=1)

    victim = before[0].test_subject
    moved = [dict(s, delta_m_pct=s["delta_m_pct"] - 7.5) if s["subject"] == victim else dict(s)
             for s in sessions]
    after, _, _ = exp_e.run_exp_e(config_10, "10ghz", moved, tmp_path, bank=bank, n_workers=1)

    victim_before = next(r for r in before if r.test_subject == victim)
    victim_after = next(r for r in after if r.test_subject == victim)
    assert victim_after.coefficient_rows == victim_before.coefficient_rows
    assert not np.isclose(victim_after.full_mae, victim_before.full_mae)

    # mu_s itself: recomputed the way the provider does, it must not have moved either
    from dehyd.models.baselines import session_means
    fold = next(f for f in nested_loso_splits(exp_b.evaluable_subjects_b(sessions))
                if f.test_subject == victim)
    args = []
    for spine in (sessions, moved):
        subjects = np.array([s["subject"] for s in spine])
        session_idx = np.array([s["session_idx"] for s in spine])
        y_raw = np.array([s["delta_m_pct"] for s in spine], dtype=float)
        args.append(session_means(subjects, session_idx, y_raw, fold.train_subjects,
                                  min_train_subjects=2)[0])
    assert args[0] == args[1]


def test_a_fold_with_no_surviving_test_row_is_excluded_never_scored_as_zero(tmp_path, config_10):
    """A held-out subject whose ONLY session is dropped by the train-only session-mean rule
    contributes no held-out residual row at all.

    The spine is built to force exactly that: subject 6 appears only in session 4, which
    otherwise holds only subject 5 — so holding out 6 leaves session 4 with one training
    subject, below `session_means`' `min_train_subjects=2`, and the session is dropped. The
    fold must land in the exclusion ledger and contribute NO importance value, because a
    missing fold silently read as a zero importance is the failure that matters here.
    """
    sessions = _sessions(n_subjects=5, session_indices=(1, 2, 3))
    sessions += [{"subject": 5, "session_idx": 4, "session_name": SESSION_NAMES[4],
                  "delta_m_pct": -1.5},
                 {"subject": 6, "session_idx": 4, "session_name": SESSION_NAMES[4],
                  "delta_m_pct": -1.6}]
    _write_store_10(tmp_path, sessions, config_10)

    results, bank, path_of_column = exp_e.run_exp_e(
        config_10, "10ghz", sessions, tmp_path, bank=_bank(), n_workers=1)
    dropped = [r for r in results if r.reason is not None]
    assert [r.test_subject for r in dropped] == [6]
    assert dropped[0].reason == "no_surviving_test_rows"
    assert dropped[0].ablated_mae.size == 0        # no value, not a zero

    rows = exp_e.importance_fold_rows(results, "10ghz", bank, path_of_column)
    assert 6 not in {r["test_subject"] for r in rows}
    ledger = exp_e.exclusion_rows(results, "10ghz")
    assert [r["test_subject"] for r in ledger] == [6]
    # and the summary's denominator drops with it, rather than counting a phantom subject
    summary = exp_e.importance_summary_rows(
        rows, exp_e.path_metadata_rows(config_10, "10ghz", bank))
    assert {r["n_subjects"] for r in summary} == {len(results) - 1}


# ------------------------------------------- what attribution does and does not mean


def test_a_planted_signal_path_yields_positive_importance(e2e, config_10):
    """The mechanism check: a path the model genuinely relies on measures as positive (removing
    it makes held-out prediction worse)."""
    results, bank, path_of_column, _, _ = e2e
    rows = exp_e.importance_fold_rows(results, "10ghz", bank, path_of_column)
    summary = {int(r["path_id"]): r for r in exp_e.importance_summary_rows(
        rows, exp_e.path_metadata_rows(config_10, "10ghz", bank))}

    planted = float(summary[SIGNAL_PATH]["mean"])
    others = [float(summary[p]["mean"]) for p in range(P) if p != SIGNAL_PATH]
    assert planted > 0.0
    assert planted > max(others)


def test_a_correlated_surrogate_shows_attribution_is_model_specific_not_causal(
        tmp_path, config_10, e2e):
    """The interpretive fixture §5.2 asks for. Plant the SAME subject-linear signal in a second
    path: nothing about the first path's relationship to the target changed, but its measured
    importance collapses, because the information survives its removal. Low importance
    therefore means "this model did not need this column here", never "this path carries no
    signal" — which is exactly why A-M10-6 forbids labelling a path null."""
    sessions = _sessions()
    _write_store_10(tmp_path, sessions, config_10, signal=0.9, surrogate=0.9)
    bank = _bank()
    results, _, path_of_column = exp_e.run_exp_e(
        config_10, "10ghz", sessions, tmp_path, bank=bank, n_workers=1)
    rows = exp_e.importance_fold_rows(results, "10ghz", bank, path_of_column)
    duplicated = {int(r["path_id"]): float(r["mean"]) for r in exp_e.importance_summary_rows(
        rows, exp_e.path_metadata_rows(config_10, "10ghz", bank))}

    alone_results, alone_bank, alone_columns, _, _ = e2e
    alone_rows = exp_e.importance_fold_rows(alone_results, "10ghz", alone_bank, alone_columns)
    alone = {int(r["path_id"]): float(r["mean"]) for r in exp_e.importance_summary_rows(
        alone_rows, exp_e.path_metadata_rows(config_10, "10ghz", alone_bank))}

    assert duplicated[SIGNAL_PATH] < alone[SIGNAL_PATH]


# ------------------------------------------------------------------- aggregation


def test_path_aggregation_is_deterministic_and_counts_contributing_subjects(e2e, config_10):
    results, bank, path_of_column, _, _ = e2e
    metadata = exp_e.path_metadata_rows(config_10, "10ghz", bank)
    rows = exp_e.importance_fold_rows(results, "10ghz", bank, path_of_column)

    first = exp_e.importance_summary_rows(rows, metadata)
    second = exp_e.importance_summary_rows(list(reversed(rows)), metadata)
    assert first == second                                   # order of input rows is irrelevant
    assert [(r["scattering_order"], r["path_id"]) for r in first] == sorted(
        (r["scattering_order"], r["path_id"]) for r in first)

    n_evaluable = sum(1 for r in results if r.reason is None)
    assert {r["n_subjects"] for r in first} == {n_evaluable}
    row = first[0]
    values = [float(r["importance_delta_mae_pct_points"]) for r in rows
              if r["path_id"] == row["path_id"]]
    assert row["mean"] == pytest.approx(float(np.mean(values)))
    assert row["sd"] == pytest.approx(float(np.std(values, ddof=0)))   # population SD (O1)
    assert row["min"] == pytest.approx(min(values)) and row["max"] == pytest.approx(max(values))


def test_a_path_with_no_contributing_fold_gets_blank_statistics_not_zero(config_10):
    metadata = exp_e.path_metadata_rows(config_10, "10ghz", _bank())
    rows = exp_e.importance_summary_rows([], metadata)
    assert {r["n_subjects"] for r in rows} == {0}
    for name in ("mean", "median", "sd", "q25", "q75", "min", "max"):
        assert {r[name] for r in rows} == {""}


def test_importance_is_ablated_minus_full_in_residual_points(e2e):
    results, bank, path_of_column, _, _ = e2e
    for row in exp_e.importance_fold_rows(results, "10ghz", bank, path_of_column):
        assert row["importance_delta_mae_pct_points"] == pytest.approx(
            row["ablated_mae_pct_points"] - row["full_mae_pct_points"])


# --------------------------------------------------------------------- the artifacts


@pytest.fixture(scope="module")
def written(e2e, config_10, tmp_path_factory):
    results, bank, path_of_column, sessions, store_dir = e2e
    out = tmp_path_factory.mktemp("exp_e_reports")
    paths = exp_e.write_exp_e_reports(
        results, config_10, "10ghz", bank, path_of_column, sessions, store_dir, out)
    return paths, out


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_five_artifact_rows_are_written_with_the_exact_column_lists(written):
    paths, out = written
    for name in ("importance_folds_10ghz.csv", "path_metadata_10ghz.csv",
                 "importance_summary_10ghz.csv", "ridge_coefficients_10ghz.csv",
                 "exclusions_e_10ghz.csv", "metrics_exp_e_10ghz.json",
                 "interpretability_map_10ghz.png"):
        assert (out / name).exists(), name
    assert list(_read(paths["importance_folds"])[0]) == list(exp_e.IMPORTANCE_FOLDS_COLUMNS)
    assert list(_read(paths["path_metadata"])[0]) == list(exp_e.PATH_METADATA_COLUMNS)
    assert list(_read(paths["importance_summary"])[0]) == list(exp_e.IMPORTANCE_SUMMARY_COLUMNS)
    assert list(_read(paths["ridge_coefficients"])[0]) == list(exp_e.RIDGE_COEFFICIENT_COLUMNS)


def test_the_coefficient_table_is_the_full_model_only_never_an_ablation_refit(written, e2e):
    paths, _ = written
    results, bank, path_of_column, _, _ = e2e
    rows = _read(paths["ridge_coefficients"])
    n_evaluable = sum(1 for r in results if r.reason is None)

    assert {r["model_variant"] for r in rows} == {"full"}
    assert len(rows) == n_evaluable * path_of_column.size
    # every column carries its layout tuple, so a coefficient can be traced to a path group
    assert {r["frame_aggregate"] for r in rows} == {"frame_mean", "frame_median"}
    assert sorted({int(r["path_id"]) for r in rows}) == list(range(P))
    assert all(r["scaler_scale"] not in ("", "0.0") for r in rows)


def test_the_metrics_json_states_the_context_before_any_table_is_read(written, e2e):
    paths, _ = written
    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    results = e2e[0]

    assert payload["status"] == "descriptive"
    assert payload["band"] == "10ghz"
    # A-M10-6: the weak predictive context and the not-causal limit come FIRST, and no
    # ranking, winner or "physical" verdict appears anywhere in the summary.
    text = " ".join(payload["interpretation"]).lower()
    assert "before this table" in text
    assert "not causality" in text and "correlated paths share credit" in text
    assert "best model" in payload["fixed_model_note"]
    assert "a-m10-1" in payload["dead_config_note"].lower()
    assert "dead config" in payload["dead_config_note"]
    assert payload["n_paths"] == P and payload["n_evaluable_outer_folds"] == len(results)
    assert len(payload["store_fingerprints_sha256"]) == 64
    assert set(payload["artifact_sha256"]) == {
        "importance_folds_10ghz.csv", "path_metadata_10ghz.csv", "importance_summary_10ghz.csv"}


def test_the_figure_is_regenerated_from_the_saved_tables_alone(written, tmp_path):
    """§3: figures read saved tables; they do not recompute models or hidden statistics."""
    paths, _ = written
    rows = _read(paths["importance_summary"])
    redrawn = exp_e._interpretability_figure(rows, "10ghz", tmp_path / "again.png")
    assert redrawn.exists() and redrawn.stat().st_size > 0


def test_the_smoke_surfaces_no_importance_value(tmp_path, config_10, monkeypatch):
    sessions = _sessions(n_subjects=5)
    _write_store_10(tmp_path, sessions, config_10)
    # store validation is Exp A's contract, tested there; these synthetic sessions carry no
    # raw file to fingerprint, so both halves of it are stubbed to reach the reporting boundary
    monkeypatch.setattr(exp_e.store_mod, "validate_store", lambda *a, **k: None)
    monkeypatch.setattr(exp_e.exp_a, "expected_fingerprints", lambda *a, **k: {})

    run_dir = tmp_path / "run"
    outputs = exp_e.run_and_report_e(
        config_10, "10ghz", sessions, tmp_path, run_dir, mode="smoke",
        analysis_commit="deadbeef", bank=_bank(), n_workers=1)

    assert set(outputs) == {"run_log"}
    payload = json.loads(outputs["run_log"].read_text(encoding="utf-8"))
    assert payload["mode"] == "mechanism-only"
    assert payload["n_paths"] == P
    assert not list(run_dir.glob("importance_*.csv"))
    assert not list(run_dir.glob("metrics_exp_e_*.json"))
    flat = json.dumps(payload)
    assert "mae" not in flat.lower() and "importance_delta" not in flat
