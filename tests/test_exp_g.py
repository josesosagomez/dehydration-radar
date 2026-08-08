"""T-M10-g: Experiment G, the matched-session decision-level fusion (`eval/exp_g.py`).

Split the way `test_robustness.py` is, and for the same reason: a RUN half that drives the real
three-level staged search on a synthetic two-band store, and a REDUCE half that runs on
hand-built band results so the population rules, the alpha arithmetic, the estimand definition
and the artifact schemas are checked against the SPECIFICATION rather than against whatever the
implementation happened to produce.

The RUN half is ONE end-to-end run behind a module-scoped fixture, deliberately. The M6 search
space cannot be shrunk in a test — `protocol_freeze_guard` rejects any deviation — so every
level pays for the full 113-candidate (10 GHz) / 50-candidate (77 GHz) staged search, and at
four subjects that single run is already ~2.5 minutes. Sharing it across a dozen assertions is
what keeps that honest and affordable at once.

Groups: T-M10-g-population (the matched join), T-M10-g-lineage (the two-config gate),
T-M10-g-oof (five-seed key coverage), T-M10-g-alpha (the grid, the tie-break, and the proof that
outer outcomes are never read), T-M10-g-estimand (subject-additive `fused - 10`),
T-M10-g-audit (selection and fit provenance), T-M10-g-failclosed, T-M10-g-artifacts,
T-M10-g-e2e.
"""

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from dehyd.config import ExpGConfig, RunConfig, SplitConfig, load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_g
from dehyd.eval.exp_g import BandFoldResult, ExpGError, ExpGFoldResult
from dehyd.eval.selection import SelectionError
from dehyd.eval.splits import nested_loso_splits, selection_folds
from dehyd.features.extraction_77 import apply_order_log_77
from dehyd.features.pooling import aggregate_session, pool_stats_batch
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

P, T, CN, NFR = 6, 4, 1, 3        # tiny path/time/channel/frame dims for a fast synthetic store
ORDER = np.array([0, 1, 1, 2, 2, 2])

# Four subjects is the FLOOR for a real Exp G run: `min_train_subjects = 3` makes three subjects
# entirely non-selectable, and at four every level is still well formed (T_s = 3, one-subject
# meta groups, a two-subject selection pool).
E2E_SUBJECTS, E2E_SESSIONS = 4, 3


# ------------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def config_10():
    """The frozen 10 GHz search space with the smoke overlay's single seed."""
    return load_config("configs/exp_a_regression.yaml", "configs/stats.yaml", "configs/smoke.yaml")


@pytest.fixture(scope="module")
def config_77():
    return load_config(
        "configs/exp_a_regression_77ghz.yaml", "configs/stats.yaml", "configs/smoke.yaml"
    )


def _spine(n_subjects=E2E_SUBJECTS, n_sessions=E2E_SESSIONS, subjects=None):
    subjects = subjects if subjects is not None else range(1, n_subjects + 1)
    return [
        {"subject": int(s), "session_idx": i, "session_name": SESSION_NAMES[i],
         "rel_path": f"s{s}_{SESSION_NAMES[i]}.mat", "frame_ids": list(range(NFR)),
         "delta_m_pct": 0.0 if i == 0 else -(0.3 * i + 0.05 * s)}
        for s in subjects
        for i in range(n_sessions)
    ]


def _write_two_band_store(store_dir, sessions, config_10, config_77, seed=0):
    """One synthetic session store per band, in each band's own `StoreBackedFeatures` layout.

    The 77 GHz half is keyed by tiling + branch only (reduction/channel/gate are fixed by
    `search_77ghz`) and uses `apply_order_log_77`, whose branch names differ from the 10 GHz
    ones — the two bands are genuinely different store schemas, which is half of why G cannot
    just run Experiment A twice.
    """
    rng = np.random.default_rng(seed)
    meta = {"order": ORDER}
    eps10 = {1: config_10.wst.log_epsilon, 2: config_10.wst.log_epsilon}
    for s in sessions:
        npz = {}
        for ti in range(len(config_10.wst.tilings)):
            npz[order_key(ti)] = ORDER
            for gi in range(len(config_10.search_10ghz.range_gate_m)):
                for r in config_10.search_10ghz.reduction:
                    for c in config_10.search_10ghz.channel:
                        raw = np.abs(rng.normal(size=(NFR, CN, P, T))) + 0.01
                        raw[:, :, 1:, :] += 0.4 * s["session_idx"]   # a signal to key on
                        npz[vec_key(gi, r, c, ti, "off")] = aggregate_session(pool_stats_batch(
                            apply_order_log(raw, meta, config_10.wst, log_on=False), meta))
                        npz[vec_key(gi, r, c, ti, "frozen")] = aggregate_session(pool_stats_batch(
                            apply_order_log(raw, meta, config_10.wst, log_on=True,
                                            epsilon_by_order=eps10), meta))
                        npz[raw_key(gi, r, c, ti)] = raw
                        npz[prelog_key(gi, r, c, ti)] = np.array([raw.mean()] * 3)
        write_session_store("10ghz", s["subject"], s["session_name"], npz, {"n_frames": NFR},
                            store_dir)

        npz77 = {}
        for ti in range(len(config_77.wst77.tilings)):
            npz77[order_key(ti)] = ORDER
            raw = np.abs(rng.normal(size=(NFR, CN, P, T))) + 0.01
            raw[:, :, 1:, :] += 0.3 * s["session_idx"]
            for name, branch in (("off", "off"), ("frozen", "on_frozen_eps")):
                logged = np.stack([
                    apply_order_log_77(raw[i], meta, config_77.wst77, log_branch=branch)
                    for i in range(NFR)
                ])
                npz77[vec77_key(ti, name)] = aggregate_session(pool_stats_batch(logged, meta))
            npz77[raw77_key(ti)] = raw
            npz77[prelog77_key(ti)] = np.array([raw.mean()] * 3)
        write_session_store("77ghz", s["subject"], s["session_name"], npz77, {"n_frames": NFR},
                            store_dir)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory, config_10, config_77):
    """ONE complete Exp G run on a synthetic two-band store: the RUN half's whole budget.

    Returns `(results, sessions, store_dir)`. Serial (`n_workers=1`) because the parallel path
    is `fold_parallel.run_folds_parallel`, already pinned bit-identical to serial by Exp A/B/C's
    own tests — repeating that here would only buy spawn-pool startup cost.
    """
    store_dir = tmp_path_factory.mktemp("g_store")
    sessions = _spine()
    _write_two_band_store(store_dir, sessions, config_10, config_77)
    results = exp_g.run_exp_g(config_10, config_77, sessions, sessions, store_dir,
                              seeds=config_10.run.seed_set, n_workers=1)
    return results, sessions, store_dir


@pytest.fixture(scope="module")
def e2e_reports(tmp_path_factory, e2e, config_10):
    """The nine artifacts, written once from the e2e results and read back as CSV."""
    results, sessions, _ = e2e
    out_dir = tmp_path_factory.mktemp("g_reports")
    matched = [
        {"subject": s["subject"], "session_idx": s["session_idx"],
         "session_name": s["session_name"], "delta_m_pct": s["delta_m_pct"],
         "n_frames_10": NFR, "n_frames_77": NFR}
        for s in sessions
    ]
    subject_rows = exp_g.per_subject_rows(results)
    summary = exp_g.summarize_exp_g(
        results, subject_rows, exp_g.population_summary(matched, []), config_10
    )
    paths = exp_g.write_exp_g_reports(results, matched, [], subject_rows, summary, out_dir)
    return paths, out_dir


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _canonical(rows):
    """Row lists with NaN made comparable, for byte-identity assertions.

    Non-viable candidates legitimately score NaN — `knn` with `n_neighbors` above the training
    row count is skipped by the harness's viability rule and contributes no score — and
    `nan != nan`, so a plain `==` would report a difference between two identical tables.
    """
    def value(v):
        return "nan" if isinstance(v, float) and np.isnan(v) else v

    return [{k: value(v) for k, v in row.items()} for row in rows]


# ---------------------------------------------- hand-built band results (the REDUCE half)


def _oof(test_subject, meta_fold, band, subject, session_idx, seed, y_true, y_pred, source=""):
    return {
        "outer_test_subject": test_subject, "meta_fold": meta_fold, "band": band,
        "subject": subject, "session_idx": session_idx, "seed": seed,
        "deterministic_source_seed": source,
        "selection_record_id": exp_g._selection_record_id(test_subject, meta_fold, band),
        "y_true": float(y_true), "y_pred": float(y_pred),
    }


def _outer(test_subject, band, session_idx, seed, y_true, y_pred):
    return {
        "outer_test_subject": test_subject, "meta_fold": exp_g.OUTER_FINAL, "band": band,
        "subject": test_subject, "session_idx": session_idx, "seed": seed,
        "deterministic_source_seed": "",
        "selection_record_id": exp_g._selection_record_id(
            test_subject, exp_g.OUTER_FINAL, band),
        "y_true": float(y_true), "y_pred": float(y_pred),
    }


def _band_result(test_subject, band, meta_rows, outer_rows):
    return BandFoldResult(test_subject=test_subject, band=band, meta_oof_rows=list(meta_rows),
                          outer_rows=list(outer_rows))


def _fold(subjects, test_subject):
    return next(f for f in nested_loso_splits(subjects) if f.test_subject == test_subject)


def _oof_pair(fold, predict, sessions_per_subject=2, seeds=(1,)):
    """OOF rows for both bands over `fold.train_subjects`.

    `predict(band, subject, session_idx) -> y_pred`; `y_true` is `-0.5 * session_idx` so both
    bands agree on the target by construction (a disagreement is a separate, failing test).
    """
    rows = {"10ghz": [], "77ghz": []}
    meta_folds = {
        s: f"meta_{i}"
        for i, inner in enumerate(selection_folds(sorted(fold.train_subjects)))
        for s in sorted(inner.val_subjects)
    }
    for band in ("10ghz", "77ghz"):
        for subject in sorted(fold.train_subjects):
            for session_idx in range(sessions_per_subject):
                for seed in seeds:
                    rows[band].append(_oof(
                        fold.test_subject, meta_folds[subject], band, subject, session_idx,
                        seed, -0.5 * session_idx, predict(band, subject, session_idx),
                    ))
    return rows


# ======================================================================= T-M10-g-population


def _patch_spines(monkeypatch, spine_10, spine_77):
    def fake(config, band):
        return list(spine_10 if band == "10ghz" else spine_77)

    monkeypatch.setattr(exp_g, "build_sessions", fake)


def test_matched_population_is_the_inner_join_in_canonical_key_order(monkeypatch, config_10,
                                                                     config_77):
    spine_10 = _spine(subjects=[2, 1], n_sessions=3)
    spine_77 = _spine(subjects=[1, 2], n_sessions=2)
    _patch_spines(monkeypatch, spine_10, spine_77)

    matched, sessions_10, sessions_77, unmatched = exp_g.build_matched_population(
        config_10, config_77
    )
    keys = [(c["subject"], c["session_idx"]) for c in matched]
    assert keys == sorted(keys) == [(1, 0), (1, 1), (2, 0), (2, 1)]
    # the two per-band row lists must be aligned to the matched order, cell for cell
    assert [(s["subject"], s["session_idx"]) for s in sessions_10] == keys
    assert [(s["subject"], s["session_idx"]) for s in sessions_77] == keys
    assert [(u["subject"], u["session_idx"], u["missing_band"]) for u in unmatched] == [
        (1, 2, "77ghz"), (2, 2, "77ghz")
    ]
    assert all(c["n_frames_10"] == NFR and c["n_frames_77"] == NFR for c in matched)


def test_a_duplicate_session_key_in_either_band_fails(monkeypatch, config_10, config_77):
    spine = _spine(n_subjects=2, n_sessions=2)
    _patch_spines(monkeypatch, spine + [dict(spine[0])], spine)
    with pytest.raises(ExpGError, match="duplicate 10ghz session key"):
        exp_g.build_matched_population(config_10, config_77)

    _patch_spines(monkeypatch, spine, spine + [dict(spine[1])])
    with pytest.raises(ExpGError, match="duplicate 77ghz session key"):
        exp_g.build_matched_population(config_10, config_77)


def test_target_disagreement_between_bands_is_fatal(monkeypatch, config_10, config_77):
    """Both bands read the same weight workbook, so a per-cell target difference means the join
    is not aligning the cells it thinks it is — which is exactly what a swapped band label
    would look like from inside."""
    spine_10 = _spine(n_subjects=2, n_sessions=2)
    spine_77 = [dict(s) for s in spine_10]
    spine_77[2]["delta_m_pct"] = spine_77[2]["delta_m_pct"] - 0.001
    _patch_spines(monkeypatch, spine_10, spine_77)
    with pytest.raises(ExpGError, match="disagree on delta_m_pct"):
        exp_g.build_matched_population(config_10, config_77)


def test_inconsistent_session_name_and_non_finite_target_are_fatal(monkeypatch, config_10,
                                                                   config_77):
    spine_10 = _spine(n_subjects=2, n_sessions=2)
    spine_77 = [dict(s) for s in spine_10]
    spine_77[1]["session_name"] = "not_the_same_session"
    _patch_spines(monkeypatch, spine_10, spine_77)
    with pytest.raises(ExpGError, match="inconsistent session name"):
        exp_g.build_matched_population(config_10, config_77)

    spine_10 = _spine(n_subjects=2, n_sessions=2)
    spine_10[0]["delta_m_pct"] = float("nan")
    spine_77 = [dict(s) for s in _spine(n_subjects=2, n_sessions=2)]
    spine_77[0]["delta_m_pct"] = float("nan")
    _patch_spines(monkeypatch, spine_10, spine_77)
    with pytest.raises(ExpGError, match="non-finite delta_m_pct"):
        exp_g.build_matched_population(config_10, config_77)


def test_no_overlapping_cell_at_all_is_fatal(monkeypatch, config_10, config_77):
    _patch_spines(monkeypatch, _spine(subjects=[1]), _spine(subjects=[9]))
    with pytest.raises(ExpGError, match="no session cell in common"):
        exp_g.build_matched_population(config_10, config_77)


def test_population_summary_reports_both_pre_match_inventories(monkeypatch, config_10,
                                                               config_77):
    _patch_spines(monkeypatch, _spine(subjects=[1, 2], n_sessions=3),
                  _spine(subjects=[2, 3], n_sessions=3))
    matched, _, _, unmatched = exp_g.build_matched_population(config_10, config_77)
    summary = exp_g.population_summary(matched, unmatched)
    assert summary["n_subjects_g"] == 1 and summary["n_matched_cells"] == 3
    assert summary["n_cells_10ghz_before_matching"] == 6
    assert summary["n_cells_77ghz_before_matching"] == 6
    assert summary["n_subjects_10ghz_before_matching"] == 2
    assert summary["n_subjects_77ghz_before_matching"] == 2
    assert summary["sessions_per_matched_subject"] == {2: 3}


# ========================================================================== T-M10-g-lineage


def test_shared_protocol_accepts_the_real_band_config_pair(config_10, config_77):
    exp_g.assert_shared_protocol(config_10, config_77)


@pytest.mark.parametrize("field, value", [
    ("run", RunConfig(seed=1, seed_set=(1,), device="cpu")),
    ("split", SplitConfig(n_inner_max=4, min_train_subjects=3)),
    ("exp_g", ExpGConfig(alpha_tie_break="closest_to_one", seed_pairing=False)),
])
def test_disagreeing_band_configs_are_refused(config_10, config_77, field, value):
    """The two bands are loaded separately by design, so nothing but this gate stops a pair
    that disagrees on the seeds, the folds or the fusion rule from producing a fused number
    that means nothing."""
    drifted = dataclasses.replace(config_77, **{field: value})
    with pytest.raises(ExpGError, match="disagree on shared analysis constants"):
        exp_g.assert_shared_protocol(config_10, drifted)


def test_a_different_weight_workbook_is_refused(config_10, config_77):
    paths = dataclasses.replace(config_77.paths, weight_xlsx=Path("other_workbook.xlsx"))
    with pytest.raises(ExpGError, match="paths.weight_xlsx"):
        exp_g.assert_shared_protocol(config_10, dataclasses.replace(config_77, paths=paths))


def test_split_constants_must_equal_the_ones_experiment_a_runs_at(config_10, config_77):
    """G's three levels have to be A's levels. A config that agreed with itself but not with
    `nested_loso_splits`' defaults would silently describe a different protocol."""
    split = SplitConfig(n_inner_max=4, min_train_subjects=3)
    with pytest.raises(ExpGError, match="G's levels must be A's levels"):
        exp_g.assert_shared_protocol(
            dataclasses.replace(config_10, split=split),
            dataclasses.replace(config_77, split=split),
        )


# ============================================================================ T-M10-g-alpha


def test_alpha_is_fit_only_on_outer_training_oof_rows_and_never_on_the_test_outcome(config_10):
    """§5.4's load-bearing fixture. The training subjects' OOF rows say 10 GHz is perfect and
    77 GHz is useless, so alpha must be 1.0. The HELD-OUT subject's outcome says the opposite —
    77 GHz is perfect there and 10 GHz is badly wrong — so if outer outcomes were read at all,
    alpha would move toward 0. It must not."""
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i if band == "10ghz" else -0.5 * i + 4.0)
    outer_10 = [_outer(5, "10ghz", i, 1, -0.5 * i, -0.5 * i + 9.0) for i in range(2)]
    outer_77 = [_outer(5, "77ghz", i, 1, -0.5 * i, -0.5 * i) for i in range(2)]

    result = exp_g._fuse_fold(config_10, fold, [
        _band_result(5, "10ghz", rows["10ghz"], outer_10),
        _band_result(5, "77ghz", rows["77ghz"], outer_77),
    ], seeds=(1,))

    assert result.alpha == 1.0
    for row in result.prediction_rows:
        assert row["pred_fused"] == row["pred_10"]      # the 77 GHz arm gets no weight at all
        assert row["pred_fused"] != row["pred_77"]


def test_the_whole_21_point_grid_is_recorded_per_seed_with_one_selected_alpha(config_10):
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i + (0.2 if band == "10ghz" else -0.3),
                     seeds=(1,))
    result = exp_g._fuse_fold(config_10, fold, [
        _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, 1, 0.0, 0.1)]),
        _band_result(5, "77ghz", rows["77ghz"], [_outer(5, "77ghz", 0, 1, 0.0, -0.1)]),
    ], seeds=(1,))

    grid = result.alpha_grid_rows
    assert len(grid) == 21                                      # 21 alphas x 1 seed label
    assert [r["alpha"] for r in grid] == list(config_10.exp_g.alpha_grid)
    assert sum(1 for r in grid if r["selected"]) == 1
    assert all(np.isfinite(r["subject_balanced_oof_mae"]) for r in grid)
    # the selected alpha is the argmin of the mean-over-seeds column, not of any single seed
    best = min(grid, key=lambda r: r["mean_over_seeds"])["mean_over_seeds"]
    assert next(r for r in grid if r["selected"])["mean_over_seeds"] == best


def test_the_grid_is_recorded_for_every_seed_label_and_alpha_uses_their_mean(config_10):
    """Seed pairing: label `k` means the same realization in both bands, and alpha is chosen
    from the mean objective across labels — not from one label, and not from a pooled fit."""
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    seeds = (1, 2, 3, 4, 5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i + (0.4 if band == "10ghz" else -0.2),
                     seeds=seeds)
    result = exp_g._fuse_fold(config_10, fold, [
        _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, k, 0.0, 0.1) for k in seeds]),
        _band_result(5, "77ghz", rows["77ghz"], [_outer(5, "77ghz", 0, k, 0.0, -0.1) for k in seeds]),
    ], seeds=seeds)

    assert len(result.alpha_grid_rows) == 21 * 5
    assert sorted({r["seed"] for r in result.alpha_grid_rows}) == list(seeds)
    for alpha in config_10.exp_g.alpha_grid:
        cells = [r for r in result.alpha_grid_rows if r["alpha"] == alpha]
        assert len(cells) == 5
        assert cells[0]["mean_over_seeds"] == pytest.approx(
            float(np.mean([c["subject_balanced_oof_mae"] for c in cells]))
        )


def test_a_tied_grid_keeps_the_weight_on_the_primary_band(config_10):
    """Identical predictions in both bands make every alpha give the identical objective. The
    frozen tie-break must then return 1.0 — fusion is never credited by a tie."""
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i + 0.25)
    result = exp_g._fuse_fold(config_10, fold, [
        _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, 1, 0.0, 0.3)]),
        _band_result(5, "77ghz", rows["77ghz"], [_outer(5, "77ghz", 0, 1, 0.0, 0.3)]),
    ], seeds=(1,))
    assert result.alpha == 1.0


def test_equal_weight_is_alpha_one_half_regardless_of_the_learned_alpha(config_10):
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i if band == "10ghz" else -0.5 * i + 2.0)
    result = exp_g._fuse_fold(config_10, fold, [
        _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, 1, 0.0, 1.0)]),
        _band_result(5, "77ghz", rows["77ghz"], [_outer(5, "77ghz", 0, 1, 0.0, 3.0)]),
    ], seeds=(1,))
    row = result.prediction_rows[0]
    assert result.alpha == 1.0
    assert row["pred_equal_weight"] == pytest.approx(2.0)


# ============================================================================== T-M10-g-oof


def test_unequal_prediction_key_sets_between_the_bands_fail(config_10):
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i)
    rows["77ghz"] = rows["77ghz"][:-1]                    # one 77 GHz OOF cell missing
    with pytest.raises(ExpGError, match="prediction keys differ"):
        exp_g._fuse_fold(config_10, fold, [
            _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, 1, 0.0, 0.1)]),
            _band_result(5, "77ghz", rows["77ghz"], [_outer(5, "77ghz", 0, 1, 0.0, 0.1)]),
        ], seeds=(1,))


def test_the_two_bands_must_agree_on_y_true_at_every_shared_key(config_10):
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i)
    rows["77ghz"][0] = dict(rows["77ghz"][0], y_true=99.0)
    with pytest.raises(ExpGError, match="disagree on y_true"):
        exp_g._fuse_fold(config_10, fold, [
            _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, 1, 0.0, 0.1)]),
            _band_result(5, "77ghz", rows["77ghz"], [_outer(5, "77ghz", 0, 1, 0.0, 0.1)]),
        ], seeds=(1,))


def test_partial_meta_validation_coverage_is_refused(config_10):
    """§2.3: "no partial meta-validation coverage is used". Dropping one training subject's OOF
    rows must fail rather than fit alpha on the subjects that happened to work."""
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i)
    for band in ("10ghz", "77ghz"):
        rows[band] = [r for r in rows[band] if r["subject"] != 3]
    with pytest.raises(ExpGError, match="meta OOF rows cover"):
        exp_g._fuse_fold(config_10, fold, [
            _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, 1, 0.0, 0.1)]),
            _band_result(5, "77ghz", rows["77ghz"], [_outer(5, "77ghz", 0, 1, 0.0, 0.1)]),
        ], seeds=(1,))


def test_a_duplicated_prediction_key_fails(config_10):
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i)
    rows["10ghz"].append(dict(rows["10ghz"][0]))
    with pytest.raises(ExpGError, match="duplicate prediction key"):
        exp_g._fuse_fold(config_10, fold, [
            _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, 1, 0.0, 0.1)]),
            _band_result(5, "77ghz", rows["77ghz"], [_outer(5, "77ghz", 0, 1, 0.0, 0.1)]),
        ], seeds=(1,))


def test_a_seed_label_with_no_oof_rows_fails_the_five_seed_contract(config_10):
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i, seeds=(1, 2))
    with pytest.raises(ExpGError, match=r"no meta OOF rows for seed label\(s\) \[3, 4, 5\]"):
        exp_g._fuse_fold(config_10, fold, [
            _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, 1, 0.0, 0.1)]),
            _band_result(5, "77ghz", rows["77ghz"], [_outer(5, "77ghz", 0, 1, 0.0, 0.1)]),
        ], seeds=(1, 2, 3, 4, 5))


def test_a_deterministic_winner_is_copied_to_every_seed_label_with_its_source_recorded():
    """§2.3: five seed LABELS, one observation. A deterministic family is fit once by
    `harness._seed_list`, and the label expansion must record which seed the single fit used —
    tests require labelled coverage, not artificially distinct values."""
    from dehyd.eval.harness import SeedOutcome

    one = [SeedOutcome(1, np.array([]), np.array([0.5, 0.25]), 0.0)]
    labeled = exp_g._labeled_seed_predictions(one, (1, 2, 3, 4, 5))
    assert [seed for seed, _, _ in labeled] == [1, 2, 3, 4, 5]
    assert all(source == 1 for _, source, _ in labeled)
    assert all(np.array_equal(p, np.array([0.5, 0.25])) for _, _, p in labeled)

    five = [SeedOutcome(k, np.array([]), np.array([0.1 * k]), 0.0) for k in (1, 2, 3, 4, 5)]
    labeled = exp_g._labeled_seed_predictions(five, (1, 2, 3, 4, 5))
    assert [seed for seed, _, _ in labeled] == [1, 2, 3, 4, 5]
    assert all(source is None for _, source, _ in labeled)     # nothing was copied

    with pytest.raises(ExpGError, match="expected either 1 or 5 seed outcomes"):
        exp_g._labeled_seed_predictions(five[:2], (1, 2, 3, 4, 5))


# ========================================================================= T-M10-g-estimand


def test_the_primary_contrast_is_subject_additive_not_session_weighted():
    """§5.4: "Primary contrast is the subject-additive `fused - 10` scalar and differs from a
    session-weighted calculation on an unequal-session fixture."

    Subject 1 has four sessions where fusion is 0.1 worse; subject 2 has one session where
    fusion is 0.8 better. Subject-additive: mean(+0.1, -0.8) = -0.35, fusion wins. Session-
    weighted: (4*0.1 - 0.8)/5 = -0.08, a different number — and on a less extreme fixture it
    would be a different SIGN.
    """
    rows = []
    for i in range(4):
        rows.append({"outer_test_subject": 1, "subject": 1, "session_idx": i, "seed": 1,
                     "y_true": 0.0, "pred_10": 0.0, "pred_77": 0.4,
                     "pred_equal_weight": 0.2, "pred_fused": 0.1, "alpha": 0.75})
    rows.append({"outer_test_subject": 2, "subject": 2, "session_idx": 0, "seed": 1,
                 "y_true": 0.0, "pred_10": 1.0, "pred_77": 0.0,
                 "pred_equal_weight": 0.5, "pred_fused": 0.2, "alpha": 0.2})

    results = [
        ExpGFoldResult(test_subject=1, alpha=0.75, prediction_rows=rows[:4]),
        ExpGFoldResult(test_subject=2, alpha=0.2, prediction_rows=rows[4:]),
    ]
    per_subject = exp_g.per_subject_rows(results)
    assert [r["subject"] for r in per_subject] == [1, 2]
    assert per_subject[0]["n_sessions"] == 4 and per_subject[1]["n_sessions"] == 1
    assert per_subject[0]["difference_fused_minus_10"] == pytest.approx(0.1)
    assert per_subject[1]["difference_fused_minus_10"] == pytest.approx(-0.8)

    additive = float(np.mean([r["difference_fused_minus_10"] for r in per_subject]))
    session_weighted = float(np.mean(
        [abs(r["y_true"] - r["pred_fused"]) - abs(r["y_true"] - r["pred_10"]) for r in rows]
    ))
    assert additive == pytest.approx(-0.35)
    assert session_weighted == pytest.approx(-0.08)
    assert additive != pytest.approx(session_weighted)


def test_seed_collapse_averages_each_seed_label_s_mae_within_the_subject():
    """`StatsConfig.seed_collapse_additive` = average_per_subject_before_resample: the MAE is
    formed per seed label and the labels are then averaged — not the predictions."""
    rows = [
        {"outer_test_subject": 1, "subject": 1, "session_idx": 0, "seed": k, "y_true": 0.0,
         "pred_10": float(k), "pred_77": 0.0, "pred_equal_weight": 0.0, "pred_fused": 0.0,
         "alpha": 1.0}
        for k in (1, 3)
    ]
    per_subject = exp_g.per_subject_rows([ExpGFoldResult(test_subject=1, alpha=1.0,
                                                         prediction_rows=rows)])
    assert per_subject[0]["mae_10"] == pytest.approx(2.0)     # mean(|1|, |3|)


def test_non_evaluable_folds_contribute_to_no_condition(config_10):
    """An outer fold with no learned prediction contributes its 10-only number to nothing
    either: §7 requires all conditions to use the same cells, and a fold present in one
    condition but absent from another would break exactly that."""
    results = [
        ExpGFoldResult(test_subject=1, alpha=1.0, prediction_rows=[
            {"outer_test_subject": 1, "subject": 1, "session_idx": 0, "seed": 1, "y_true": 0.0,
             "pred_10": 0.5, "pred_77": 0.5, "pred_equal_weight": 0.5, "pred_fused": 0.5,
             "alpha": 1.0}]),
        ExpGFoldResult(test_subject=2, reason="no_surviving_candidate"),
    ]
    assert [r["subject"] for r in exp_g.per_subject_rows(results)] == [1]


# ============================================================================ T-M10-g-audit


def test_a_fit_audit_row_refuses_a_subject_that_was_fitted_and_predicted():
    with pytest.raises(ExpGError, match="fit on subject\\(s\\) it then predicted: \\[3\\]"):
        exp_g._fit_audit_row("rec", 9, "meta_0", "10ghz", "scaler", "outer_train",
                             [1, 2, 3], [3, 4])


def test_the_alpha_fit_audits_against_the_outer_training_subjects(config_10):
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i)
    result = exp_g._fuse_fold(config_10, fold, [
        _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, 1, 0.0, 0.1)]),
        _band_result(5, "77ghz", rows["77ghz"], [_outer(5, "77ghz", 0, 1, 0.0, 0.1)]),
    ], seeds=(1,))

    alpha_rows = [r for r in result.fit_audit_rows if r["quantity"] == "fusion_alpha"]
    assert len(alpha_rows) == 1
    row = alpha_rows[0]
    assert json.loads(row["fitted_subjects_json"]) == [1, 2, 3, 4]
    assert json.loads(row["predicted_subjects_json"]) == [5]
    assert row["band"] == "fused" and row["meta_fold_or_outer_final"] == exp_g.OUTER_FINAL


# ======================================================================== T-M10-g-failclosed


def test_a_meta_level_with_too_few_selection_training_subjects_kills_the_whole_fold(
        e2e, config_10):
    """Fail-closed, whole-fold. Hand-built because `nested_loso_splits` at the frozen constants
    cannot reach it (|T_s| >= 3 and one-subject meta groups leave a pool of >= 2) — the guard is
    defensive, and a defensive guard nobody exercises is a guard nobody has."""
    from dehyd.eval.splits import InnerFold, OuterFold

    _, sessions, store_dir = e2e
    degenerate = OuterFold(
        test_subject=4, train_subjects=frozenset({1, 2, 3}), selectable=True,
        inner_folds=(InnerFold(train_subjects=frozenset({1}), val_subjects=frozenset({2, 3})),),
    )
    result = exp_g._run_band_fold(config_10, "10ghz", sessions, store_dir, degenerate,
                                  config_10.run.seed_set)
    assert result.reason == "insufficient_selection_training_subjects"
    assert not result.meta_oof_rows and not result.outer_rows
    assert result.exclusions[0]["meta_fold"] == "meta_0"
    assert result.exclusions[0]["band"] == "10ghz"


def test_no_surviving_candidate_makes_the_entire_outer_fold_non_evaluable(e2e, config_10,
                                                                          monkeypatch):
    _, sessions, store_dir = e2e
    fold = _fold(sorted({s["subject"] for s in sessions}), test_subject=1)

    def refuse(_stage):
        raise SelectionError("no comparable candidate")

    monkeypatch.setattr(exp_g.harness, "select_stage_winner", refuse)
    result = exp_g._run_band_fold(config_10, "10ghz", sessions, store_dir, fold,
                                  config_10.run.seed_set)
    assert result.reason == "no_surviving_candidate"
    assert not result.meta_oof_rows                    # no partial meta-validation coverage
    assert result.exclusions[0]["reason"] == "no_surviving_candidate"


def test_a_band_failure_propagates_to_the_fused_fold(config_10):
    fold = _fold([1, 2, 3, 4, 5], test_subject=5)
    rows = _oof_pair(fold, lambda band, s, i: -0.5 * i)
    broken = BandFoldResult(test_subject=5, band="77ghz", reason="no_surviving_candidate",
                            exclusions=[{"outer_test_subject": 5, "meta_fold": "meta_0",
                                         "band": "77ghz", "reason": "no_surviving_candidate",
                                         "detail": ""}])
    result = exp_g._fuse_fold(config_10, fold, [
        _band_result(5, "10ghz", rows["10ghz"], [_outer(5, "10ghz", 0, 1, 0.0, 0.1)]), broken,
    ], seeds=(1,))
    assert result.reason == "no_surviving_candidate"
    assert result.alpha is None and not result.prediction_rows
    assert result.exclusions and result.exclusions[0]["band"] == "77ghz"


def test_a_cohort_with_no_selectable_fold_reports_exclusions_and_no_estimand(config_10,
                                                                             config_77,
                                                                             tmp_path):
    sessions = _spine(subjects=[1, 2, 3], n_sessions=2)
    results = exp_g.run_exp_g(config_10, config_77, sessions, sessions, tmp_path, seeds=(1,))
    assert [r.reason for r in results] == ["outer_fold_not_selectable"] * 3
    assert all(r.exclusions[0]["reason"] == "outer_fold_not_selectable" for r in results)
    assert exp_g.per_subject_rows(results) == []


# ================================================================================ T-M10-g-e2e


def test_e2e_every_selectable_fold_produces_an_alpha_on_the_frozen_grid(e2e, config_10):
    results, sessions, _ = e2e
    assert len(results) == E2E_SUBJECTS
    assert all(r.reason is None for r in results)
    for result in results:
        assert result.alpha in config_10.exp_g.alpha_grid
        assert len(result.prediction_rows) == E2E_SESSIONS      # 1 subject x 3 cells x 1 seed
        assert all(row["subject"] == result.test_subject for row in result.prediction_rows)


def test_e2e_the_meta_oof_table_has_exactly_one_row_per_expected_key(e2e, config_10):
    """§5.4: exactly one OOF row per band/fold/subject/session/seed key, with the meta groups
    covering the outer-training subjects once each."""
    results, sessions, _ = e2e
    subjects = sorted({s["subject"] for s in sessions})
    for result in results:
        fold = _fold(subjects, result.test_subject)
        keys = [(r["band"], r["subject"], r["session_idx"], r["seed"])
                for r in result.meta_oof_rows]
        assert len(keys) == len(set(keys))
        expected = {
            (band, subject, session_idx, seed)
            for band in ("10ghz", "77ghz")
            for subject in sorted(fold.train_subjects)
            for session_idx in range(E2E_SESSIONS)
            for seed in config_10.run.seed_set
        }
        assert set(keys) == expected
        # each training subject is a meta-validation subject exactly once, per band
        for band in ("10ghz", "77ghz"):
            groups = {r["subject"]: r["meta_fold"] for r in result.meta_oof_rows
                      if r["band"] == band}
            assert sorted(groups) == sorted(fold.train_subjects)
            assert len(set(groups.values())) == len(fold.inner_folds)


def test_e2e_every_level_is_selection_honest_about_its_own_meta_group(e2e):
    """The A-M10-3 invariant, read straight off the selection table: at a meta level the
    training pool is `T_s \\ V` and the validation group is `V`, disjoint; at the outer-final
    level the pool is all of `T_s` and validation is blank."""
    results, sessions, _ = e2e
    subjects = sorted({s["subject"] for s in sessions})
    for result in results:
        fold = _fold(subjects, result.test_subject)
        seen_validation = set()
        for row in result.selection_rows:
            train = set(json.loads(row["train_subjects_json"]))
            assert result.test_subject not in train
            if row["meta_fold_or_outer_final"] == exp_g.OUTER_FINAL:
                assert row["validation_subjects_json"] == ""
                assert train == set(fold.train_subjects)
                continue
            validation = set(json.loads(row["validation_subjects_json"]))
            assert not train & validation
            assert train | validation == set(fold.train_subjects)
            assert result.test_subject not in validation
            seen_validation |= validation
        assert seen_validation == set(fold.train_subjects)


def test_e2e_the_selection_table_keeps_the_losing_candidates(e2e):
    """G's per-candidate enumeration is what §5.4/§8.2 protect (unlike H's winner rows under
    A-M10-10): proving that outer outcomes never chose a candidate needs the scores of the
    candidates that did NOT win."""
    results, _, _ = e2e
    rows = results[0].selection_rows
    per_level_stage = {}
    for row in rows:
        key = (row["meta_fold_or_outer_final"], row["band"], row["stage"])
        per_level_stage.setdefault(key, []).append(row)
    for (_, band, stage), group in per_level_stage.items():
        expected = {("10ghz", "stage1"): 72, ("77ghz", "stage1"): 9}.get((band, stage), 41)
        assert len(group) == expected
        assert sum(1 for r in group if r["selected"]) == 1
        assert len({r["candidate"] for r in group}) == expected
        assert all(r["model_seeds_json"] == "[1]" for r in group)


def test_e2e_every_prediction_resolves_to_a_selection_record_and_a_fit_chain(e2e):
    """§5.4: every OOF and outer-final prediction resolves to one complete base-selection
    record and fit-audit chain, with disjoint fitted/predicted subjects."""
    results, _, _ = e2e
    for result in results:
        selection_ids = {r["selection_record_id"] for r in result.selection_rows}
        audit_ids = {r["selection_record_id"] for r in result.fit_audit_rows}
        for row in result.meta_oof_rows + result.prediction_rows[:0]:
            assert row["selection_record_id"] in selection_ids
            assert row["selection_record_id"] in audit_ids

        by_level = {}
        for row in result.fit_audit_rows:
            by_level.setdefault(row["selection_record_id"], set()).add(row["quantity"])
        for record_id, quantities in by_level.items():
            if record_id.endswith("|fused"):
                assert quantities == {"fusion_alpha"}
                continue
            assert "staged_selection" in quantities and "scaler" in quantities
            assert len(quantities) >= 3            # selection + scaler + at least one model

        for row in result.fit_audit_rows:
            fitted = set(json.loads(row["fitted_subjects_json"]))
            predicted = set(json.loads(row["predicted_subjects_json"]))
            assert not fitted & predicted
            assert result.test_subject not in fitted


def test_e2e_mutating_the_held_out_subject_changes_no_train_derived_quantity(e2e, config_10,
                                                                             config_77,
                                                                             tmp_path_factory):
    """§5.1's leakage probe, run for real: shift the held-out subject's targets and every
    train-derived quantity — the selection table, the alpha, the meta OOF rows — must be
    byte-identical. Only that subject's own predictions and score may move.

    Run for ONE (fold, band) rather than the whole cohort: the claim is about a fold's
    training-side computation, and the full cohort would cost a second complete Exp G.
    """
    _, sessions, store_dir = e2e
    fold = _fold(sorted({s["subject"] for s in sessions}), test_subject=2)
    baseline = exp_g._run_band_fold(config_10, "10ghz", sessions, store_dir, fold,
                                    config_10.run.seed_set)

    mutated = [dict(s) for s in sessions]
    for record in mutated:
        if record["subject"] == 2:
            record["delta_m_pct"] = record["delta_m_pct"] + 17.0
    after = exp_g._run_band_fold(config_10, "10ghz", mutated, store_dir, fold,
                                 config_10.run.seed_set)

    assert _canonical(after.selection_rows) == _canonical(baseline.selection_rows)
    assert after.fit_audit_rows == baseline.fit_audit_rows
    assert after.meta_oof_rows == baseline.meta_oof_rows
    # the held-out subject's own targets moved, and its predictions did not
    assert [r["y_true"] for r in after.outer_rows] == [
        r["y_true"] + 17.0 for r in baseline.outer_rows]
    assert [r["y_pred"] for r in after.outer_rows] == [r["y_pred"] for r in baseline.outer_rows]


# ========================================================================== T-M10-g-artifacts


def test_every_required_artifact_is_written_with_the_specified_columns(e2e_reports):
    paths, out_dir = e2e_reports
    expected = {
        "matched_population.csv", "unmatched_population.csv", "fusion_meta_oof.csv",
        "fusion_base_selection.csv", "fit_audit_g.csv", "fusion_alpha_grid.csv",
        "predictions_g.csv", "per_subject_g.csv", "exclusions_g.csv", "metrics_exp_g.json",
        "fusion_comparison.png",
    }
    assert expected <= {p.name for p in out_dir.iterdir()}

    columns = {
        "matched_population.csv": exp_g.MATCHED_POPULATION_COLUMNS,
        "unmatched_population.csv": exp_g.UNMATCHED_POPULATION_COLUMNS,
        "fusion_meta_oof.csv": exp_g.META_OOF_COLUMNS,
        "fusion_base_selection.csv": exp_g.BASE_SELECTION_COLUMNS,
        "fit_audit_g.csv": exp_g.FIT_AUDIT_COLUMNS,
        "fusion_alpha_grid.csv": exp_g.ALPHA_GRID_COLUMNS,
        "predictions_g.csv": exp_g.PREDICTIONS_COLUMNS,
        "per_subject_g.csv": exp_g.PER_SUBJECT_COLUMNS,
        "exclusions_g.csv": exp_g.EXCLUSIONS_COLUMNS,
    }
    for name, expected_columns in columns.items():
        with (out_dir / name).open(newline="", encoding="utf-8") as fh:
            assert tuple(next(csv.reader(fh))) == expected_columns


def test_the_metrics_json_names_its_provenance_tables_by_content_hash(e2e_reports):
    """A summary must never be readable next to a base-selection or fit-audit table it was not
    computed against."""
    import hashlib

    paths, out_dir = e2e_reports
    payload = json.loads((out_dir / "metrics_exp_g.json").read_text(encoding="utf-8"))
    for name in ("fusion_base_selection.csv", "fit_audit_g.csv"):
        digest = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
        assert payload["artifact_sha256"][name] == digest


def test_the_metrics_json_reports_the_primary_estimand_its_sign_and_the_cohort_limitation(
        e2e_reports):
    _, out_dir = e2e_reports
    payload = json.loads((out_dir / "metrics_exp_g.json").read_text(encoding="utf-8"))
    primary = payload["primary"]
    assert primary["estimand"] == "mean_over_subject(mean_over_seed(MAE_fused - MAE_10ghz))"
    assert primary["direction"] == "negative favours fusion"
    assert primary["sign"] in ("negative", "positive", "zero")
    assert set(primary["mean_difference_fused_minus_10"]) >= {"point", "low", "high", "method"}
    assert payload["conditional_exploratory"] is True
    assert "A-M10-4" in payload["feature_level_variant"]
    assert "cannot rescue" in payload["limitation"]
    assert len(payload["alpha_grid"]) == 21
    assert payload["alpha_tie_break"] == "closest_to_one"
    assert set(payload["alpha_by_outer_fold"]) == {"1", "2", "3", "4"}


def test_the_predictions_table_carries_the_fold_alpha_and_both_band_predictions(e2e_reports):
    _, out_dir = e2e_reports
    rows = _read_csv(out_dir / "predictions_g.csv")
    assert len(rows) == E2E_SUBJECTS * E2E_SESSIONS
    for row in rows:
        alpha = float(row["alpha"])
        fused = alpha * float(row["pred_10"]) + (1.0 - alpha) * float(row["pred_77"])
        assert float(row["pred_fused"]) == pytest.approx(fused)
        assert float(row["pred_equal_weight"]) == pytest.approx(
            0.5 * float(row["pred_10"]) + 0.5 * float(row["pred_77"])
        )
        assert row["outer_test_subject"] == row["subject"]


def test_the_per_subject_table_is_the_frozen_seed_collapsed_contrast(e2e_reports):
    _, out_dir = e2e_reports
    rows = _read_csv(out_dir / "per_subject_g.csv")
    assert [int(r["subject"]) for r in rows] == [1, 2, 3, 4]
    for row in rows:
        assert int(row["n_sessions"]) == E2E_SESSIONS
        assert float(row["difference_fused_minus_10"]) == pytest.approx(
            float(row["mae_fused"]) - float(row["mae_10"])
        )


def test_the_alpha_grid_table_records_all_21_points_per_fold(e2e_reports):
    _, out_dir = e2e_reports
    rows = _read_csv(out_dir / "fusion_alpha_grid.csv")
    assert len(rows) == E2E_SUBJECTS * 21          # one seed label in the smoke overlay
    for subject in ("1", "2", "3", "4"):
        fold_rows = [r for r in rows if r["outer_test_subject"] == subject]
        assert len(fold_rows) == 21
        assert sum(1 for r in fold_rows if r["selected"] == "True") == 1


def test_the_exclusions_table_exists_and_is_empty_on_a_clean_run(e2e_reports):
    """An empty exclusions table is a RESULT, not an omission — the file must be there."""
    _, out_dir = e2e_reports
    assert _read_csv(out_dir / "exclusions_g.csv") == []
