"""T-M10-refgate: the Experiment-A reference gate (MILESTONE_10_PLAN.md §5.3, §4.2 step 1).

The gate's whole job is to notice when a store rebuild silently changed Exp A's answers, so
every test here is a *deliberate* change to one evidence class that must be caught, plus the
matching control that an unchanged rebuild is approved. A gate that only ever sees identical
inputs proves nothing about the case it exists for.

Everything runs on a synthetic four-subject world — a hand-built store, a hand-built run
directory — because the real one lives on IBEX and the properties under test are structural,
not numerical.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from dehyd.config import load_config
from dehyd.eval import exp_a, reference_gate as gate
from dehyd.features import store as store_mod

BAND = "10ghz"
COMMIT = "a" * 40
REBUILD_COMMIT = "b" * 40

# One data-independent key and one fold-locally reconstructed key, so both halves of the
# branch-aware evidence are exercised by every snapshot the fixtures take.
FK_OFF = (0, "A", "mag", 0, "off")
FK_TUNED = (0, "A", "mag", 1, "tuned")

N_SUBJECTS, N_SESSIONS, N_FRAMES = 4, 2, 3
N_PATHS, N_TIME, N_CHANNELS = 5, 6, 1
ORDER = np.array([0, 1, 1, 2, 2])


# --------------------------------------------------------------------- synthetic world


def _sessions(frame_offset=0):
    records = []
    for subject in range(1, N_SUBJECTS + 1):
        for session_idx in range(N_SESSIONS):
            records.append(
                {
                    "subject": subject,
                    "session_idx": session_idx,
                    "session_name": f"s{session_idx}",
                    "rel_path": f"subject_{subject}_{session_idx}.mat",
                    "frame_ids": [frame_offset + i for i in range(N_FRAMES)],
                    "delta_m_pct": -0.1 * subject - 0.05 * session_idx,
                }
            )
    return records


def _raw_tensor(subject, session_idx, *, scale=1.0):
    """[N_FRAMES, C, n_paths, n_time], strictly positive so the order-1/2 log is defined."""
    rng = np.random.default_rng(100 * subject + session_idx)
    return scale * (1.0 + rng.random((N_FRAMES, N_CHANNELS, N_PATHS, N_TIME)))


def _write_store(store_dir, sessions, *, commit, vector_bump=0.0, prelog_scale=1.0, raw_scale=1.0):
    for s in sessions:
        raw = _raw_tensor(s["subject"], s["session_idx"], scale=raw_scale)
        rng = np.random.default_rng(7 * s["subject"] + s["session_idx"])
        npz = {
            store_mod.vec_key(0, "A", "mag", 0, "off"): rng.random(2 * N_PATHS) + vector_bump,
            store_mod.vec_key(0, "A", "mag", 0, "frozen"): rng.random(2 * N_PATHS),
            store_mod.raw_key(0, "A", "mag", 1): raw,
            store_mod.prelog_key(0, "A", "mag", 1): np.array([1.0, 2.0, 3.0]) * prelog_scale,
            store_mod.order_key(0): ORDER,
            store_mod.order_key(1): ORDER,
        }
        fingerprint = {
            "git": {"commit": commit, "dirty": False, "branch": "test"},
            "spec_hash": "spec", "qc_config_hash": "qc",
            "frame_selection": store_mod.FRAME_SELECTION,
            "frame_ids_sha256": store_mod.frame_ids_sha256(s["frame_ids"]),
            "n_frames": len(s["frame_ids"]),
            "raw_sha256": "raw", "session_eligible": True,
            "store_version": store_mod.STORE_VERSION,
        }
        store_mod.write_session_store(BAND, s["subject"], s["session_name"], npz,
                                      fingerprint, store_dir)


def _write_run_dir(run_dir, sessions, *, commit, selection=None, pred_noise=0.0,
                   y_true_bump=0.0, seeds=(1, 2, 3, 4, 5)):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted({s["subject"] for s in sessions})
    selection = selection or {s: FK_TUNED if s % 2 else FK_OFF for s in subjects}

    (run_dir / "provenance.json").write_text(
        json.dumps({
            "git": {"commit": commit, "dirty": False, "branch": "test"},
            "timestamp_utc": "2026-08-07T00:00:00+00:00",
            "seed": 20260721, "seed_set": list(seeds),
            "extra": {"stage": "exp-a-full", "band": BAND},
        }, indent=2) + "\n", encoding="utf-8")
    (run_dir / f"metrics_exp_a_{BAND}.json").write_text(
        json.dumps({"n_eval_subjects": len(subjects)}) + "\n", encoding="utf-8")

    with (run_dir / f"selection_table_{BAND}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["test_subject", "feature_key", "family", "params"])
        for subject in subjects:
            writer.writerow([subject, selection[subject], "ridge", {"alpha": 1.0}])

    with (run_dir / f"predictions_{BAND}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "seed", "y_true", "y_pred"])
        for subject in subjects:
            rows = [s for s in sessions if s["subject"] == subject]
            for seed in seeds:
                for i, s in enumerate(rows):
                    y_true = s["delta_m_pct"] + y_true_bump
                    writer.writerow([subject, seed, y_true, y_true + 0.01 * (i + 1) + pred_noise])
    return run_dir


class World:
    """A complete synthetic (config, sessions, store, run dir) the gate can be run against."""

    def __init__(self, root, monkeypatch, **kwargs):
        self.root = Path(root)
        self.sessions = _sessions(frame_offset=kwargs.pop("frame_offset", 0))
        results_dir = self.root / "results"
        base = load_config("configs/exp_a_regression.yaml")
        self.config = dataclasses.replace(
            base, paths=dataclasses.replace(base.paths, results_dir=results_dir)
        )
        self.commit = kwargs.pop("commit", COMMIT)
        _write_store(results_dir, self.sessions, commit=kwargs.pop("store_commit", self.commit),
                     **{k: kwargs.pop(k) for k in ("vector_bump", "prelog_scale", "raw_scale")
                        if k in kwargs})
        self.run_dir = _write_run_dir(self.root / "run", self.sessions, commit=self.commit, **kwargs)

        # The real spine reads the raw cohort and the QC manifest; the gate's behaviour under
        # test is what it does with the spine, not how the spine is built.
        monkeypatch.setattr(exp_a, "build_sessions", lambda config, band: list(self.sessions))
        monkeypatch.setattr(
            exp_a, "expected_fingerprints",
            lambda config, band, sessions: {
                (s["subject"], s["session_name"]): {
                    "spec_hash": "spec", "qc_config_hash": "qc",
                    "frame_selection": store_mod.FRAME_SELECTION,
                    "frame_ids_sha256": store_mod.frame_ids_sha256(s["frame_ids"]),
                    "n_frames": len(s["frame_ids"]), "session_eligible": True,
                    "store_version": store_mod.STORE_VERSION,
                }
                for s in sessions
            },
        )

    def snapshot(self, **kwargs):
        return gate.snapshot({BAND: self.config}, {BAND: self.run_dir},
                             hash_npz=False, **kwargs)

    def evidence(self, **kwargs):
        return gate.build_band_evidence(self.config, BAND, self.run_dir, hash_npz=False, **kwargs)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path / "reference", monkeypatch)


def _rebuild(tmp_path, monkeypatch, **kwargs):
    """A fresh world standing in for the post-rebuild state (new commit, new run dir)."""
    kwargs.setdefault("commit", REBUILD_COMMIT)
    return World(tmp_path / "rebuild", monkeypatch, **kwargs)


def _compare(manifest, rebuilt):
    return gate.compare(manifest, {BAND: rebuilt.config}, {BAND: rebuilt.run_dir}, hash_npz=False)


def _status(sources, evidence_class):
    rows = sources["bands"][BAND]["comparisons"]
    return next(r["status"] for r in rows if r["evidence_class"] == evidence_class)


# ------------------------------------------------------------------------- the snapshot


def test_snapshot_records_every_evidence_class(world):
    manifest = world.snapshot()
    assert manifest["schema_version"] == gate.MANIFEST_SCHEMA_VERSION
    assert manifest["reference_grade"] == gate.GRADE_AUTHORITATIVE

    band = manifest["bands"][BAND]
    assert band["population"]["n_subjects"] == N_SUBJECTS
    assert band["population"]["n_sessions"] == N_SUBJECTS * N_SESSIONS
    assert band["folds"]["n_selectable"] == N_SUBJECTS
    assert band["stage1_candidates"]["n_candidates"] == 72   # 2 gates x 2 red x 2 chan x 3 tilings x 3 branches
    assert set(band["feature_inputs"]) == {repr(FK_OFF), repr(FK_TUNED)}
    assert len(band["selected_folds"]) == N_SUBJECTS
    # Every row spans the whole session spine, and the width is the branch's own: `off`
    # reads its stored session vector verbatim, `tuned` reconstructs the pooled vector from
    # the raw tensor (6 segment statistics x 5 paths, doubled by the frame mean/median
    # aggregate). A row of the wrong width means the gate hashed the wrong array.
    widths = {"off": 2 * N_PATHS, "tuned": 2 * 6 * N_PATHS * N_CHANNELS}
    for row in band["selected_folds"]:
        assert row["feature_matrix_shape"] == [N_SUBJECTS * N_SESSIONS, widths[row["branch"]]]
        assert row["predictions"]["seeds"] == [1, 2, 3, 4, 5]
        assert row["test_subject"] not in row["train_subjects"]


def test_branch_aware_evidence_differs_by_branch(world):
    """off/frozen carry the stored vector; tuned carries raw/prelog/order plus a fold-local ε."""
    band = world.snapshot()["bands"][BAND]

    off = band["feature_inputs"][repr(FK_OFF)]
    assert off["branch"] == "off" and "vector_sha256" in off
    assert "raw_sha256" not in off

    tuned = band["feature_inputs"][repr(FK_TUNED)]
    assert tuned["branch"] == "tuned"
    assert {"raw_sha256", "prelog_sha256", "order_sha256"} <= set(tuned)

    by_subject = {r["test_subject"]: r for r in band["selected_folds"]}
    assert "tuned_epsilon" not in by_subject[2]                      # an off fold
    eps = by_subject[1]["tuned_epsilon"]                             # a tuned fold
    # ε = k * median over TRAINING subjects of the stored per-order pre-log scale.
    assert eps["order_1"] == pytest.approx(0.1 * 2.0)
    assert eps["order_2"] == pytest.approx(0.1 * 3.0)


def test_folds_are_built_the_way_exp_a_builds_them(world):
    """The gate must reproduce the folds Exp A used, which come from the module defaults —
    `exp_a.run_exp_a` passes no split kwargs, so reading `config.split` here would be a
    different (and silently wrong) fold set the day the two disagree."""
    from dehyd.eval.splits import DEFAULT_MIN_TRAIN_SUBJECTS, DEFAULT_N_INNER_MAX, nested_loso_splits
    from dehyd.provenance import fold_manifest

    band = world.snapshot()["bands"][BAND]
    assert band["folds"]["constructed_with"]["n_inner_max"] == DEFAULT_N_INNER_MAX
    assert band["folds"]["constructed_with"]["min_train_subjects"] == DEFAULT_MIN_TRAIN_SUBJECTS
    expected = fold_manifest(nested_loso_splits(sorted({s["subject"] for s in world.sessions})))
    assert band["folds"]["sha256"] == gate.json_sha256(expected)


def test_snapshot_refuses_a_store_that_did_not_back_the_reference_run(tmp_path, monkeypatch):
    """The evidence must come from the store the run used; another store's arrays are not
    evidence about that run, and `store._check_match` enforces the same equality."""
    other = World(tmp_path / "other", monkeypatch, store_commit="c" * 40)
    with pytest.raises(gate.ReferenceGateError, match="did not back that run"):
        other.snapshot()

    degraded = other.snapshot(allow_store_commit_divergence=True)
    assert degraded["reference_grade"] == gate.GRADE_DEGRADED_STORE_COMMIT


def test_a_run_dir_must_be_a_full_cohort_exp_a_run_for_that_band(world):
    provenance = json.loads((world.run_dir / "provenance.json").read_text(encoding="utf-8"))
    provenance["extra"]["stage"] = "exp-a-smoke"
    (world.run_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(gate.ReferenceGateError, match="not a full-cohort Exp A run"):
        world.evidence()


def test_missing_artifact_fails_closed(world):
    (world.run_dir / f"selection_table_{BAND}.csv").unlink()
    with pytest.raises(gate.ReferenceGateError, match="selection table"):
        world.evidence()


# -------------------------------------------------------------------------- the compare


def test_an_unchanged_rebuild_is_approved(world, tmp_path, monkeypatch):
    """The control. Same data, same store contents, a new commit and a new run directory —
    every evidence class must match, or none of the negative tests below mean anything."""
    manifest = world.snapshot()
    sources = _compare(manifest, _rebuild(tmp_path, monkeypatch))
    assert sources["status"] == "approved"
    assert sources["bands_approved"] == [BAND]
    assert {row["evidence_class"] for row in sources["bands"][BAND]["comparisons"]} == set(
        gate.EVIDENCE_CLASSES
    )
    assert all(row["status"] == "match" for row in sources["bands"][BAND]["comparisons"])


def test_compare_refuses_a_degraded_manifest(tmp_path, monkeypatch):
    other = World(tmp_path / "other", monkeypatch, store_commit="c" * 40)
    degraded = other.snapshot(allow_store_commit_divergence=True)
    with pytest.raises(gate.ReferenceGateError, match="not 'authoritative'"):
        _compare(degraded, _rebuild(tmp_path, monkeypatch))


def test_compare_refuses_a_band_the_snapshot_never_covered(world, tmp_path, monkeypatch):
    manifest = world.snapshot()
    manifest["bands"].pop(BAND)
    manifest["bands_covered"] = []
    with pytest.raises(gate.ReferenceGateError, match="does not cover band"):
        _compare(manifest, _rebuild(tmp_path, monkeypatch))


def test_changed_frame_membership_fails_the_population(world, tmp_path, monkeypatch):
    """C4's failure mode at the milestone scale: the same number of frames, different frames."""
    manifest = world.snapshot()
    sources = _compare(manifest, _rebuild(tmp_path, monkeypatch, frame_offset=1))
    assert sources["status"] == "not_approved"
    assert _status(sources, "population") == "mismatch"


def test_a_changed_stored_session_vector_is_caught(world, tmp_path, monkeypatch):
    """The data-independent branch: its stored vector IS its input, so a rebuild that moved
    it must fail both the input hash and the feature matrix built from it."""
    manifest = world.snapshot()
    sources = _compare(manifest, _rebuild(tmp_path, monkeypatch, vector_bump=1e-9))
    assert sources["status"] == "not_approved"
    assert _status(sources, "feature_inputs") == "mismatch"
    assert _status(sources, "feature_matrices") == "mismatch"


def test_a_changed_prelog_scale_moves_the_fold_local_epsilon_and_the_matrix(
    world, tmp_path, monkeypatch
):
    """The tuned branch's fitted quantity. ε is a train-only median of the stored pre-log
    scales, so a rebuilt store with different scales reconstructs a different matrix even
    though the raw tensors are untouched — the exact drift a vector-only check would miss."""
    manifest = world.snapshot()
    sources = _compare(manifest, _rebuild(tmp_path, monkeypatch, prelog_scale=1.5))
    assert sources["status"] == "not_approved"
    assert _status(sources, "tuned_epsilon") == "mismatch"
    assert _status(sources, "feature_matrices") == "mismatch"
    assert _status(sources, "feature_inputs") == "mismatch"


def test_a_changed_raw_tensor_is_caught_even_when_epsilon_is_unchanged(
    world, tmp_path, monkeypatch
):
    manifest = world.snapshot()
    sources = _compare(manifest, _rebuild(tmp_path, monkeypatch, raw_scale=1.001))
    assert sources["status"] == "not_approved"
    assert _status(sources, "tuned_epsilon") == "match"      # prelog untouched
    assert _status(sources, "feature_inputs") == "mismatch"
    assert _status(sources, "feature_matrices") == "mismatch"


def test_a_changed_model_selection_is_tolerance_free(world, tmp_path, monkeypatch):
    """O-M9-5's part 1. Which feature key a fold selects is a discrete outcome: no tolerance
    applies to it, and it is what makes the prediction tolerance safe rather than convenient."""
    manifest = world.snapshot()
    flipped = {s: FK_OFF for s in range(1, N_SUBJECTS + 1)}
    rebuilt = _rebuild(tmp_path, monkeypatch, selection=flipped)
    sources = _compare(manifest, rebuilt)
    assert sources["status"] == "not_approved"
    assert _status(sources, "selected_feature_keys") == "mismatch"
    assert _status(sources, "selection_table") == "mismatch"


def test_last_ulp_prediction_noise_passes_and_a_real_shift_does_not(
    world, tmp_path, monkeypatch
):
    """O-M9-5's part 2, at both sides of the frozen tolerance. The tolerance is imported from
    its one definition in `exp_d.py`, never re-declared here."""
    manifest = world.snapshot()
    tolerance = gate.pred_tolerance()

    tiny = _compare(manifest, _rebuild(tmp_path / "tiny", monkeypatch, pred_noise=tolerance / 100))
    assert tiny["status"] == "approved"
    row = next(r for r in tiny["bands"][BAND]["comparisons"] if r["evidence_class"] == "predictions")
    assert 0 < row["max_abs_pred_delta"] <= tolerance
    assert row["byte_identical"] is False       # it really did change, and was still accepted

    big = _compare(manifest, _rebuild(tmp_path / "big", monkeypatch, pred_noise=tolerance * 100))
    assert big["status"] == "not_approved"
    assert _status(big, "predictions") == "mismatch"
    assert _status(big, "scores") == "mismatch"


def test_a_changed_y_true_is_a_data_fault_not_float_noise(world, tmp_path, monkeypatch):
    """The tolerance exists for a fitted model's output. Different ground truth means
    different data or different splits, and must never be averaged into a delta."""
    manifest = world.snapshot()
    sources = _compare(manifest, _rebuild(tmp_path, monkeypatch, y_true_bump=1e-15))
    assert sources["status"] == "not_approved"
    row = next(
        r for r in sources["bands"][BAND]["comparisons"] if r["evidence_class"] == "predictions"
    )
    assert "y_true differs" in row["detail"]
    assert row["max_abs_pred_delta"] is None


def test_differing_seed_labels_are_a_fault(world, tmp_path, monkeypatch):
    manifest = world.snapshot()
    sources = _compare(manifest, _rebuild(tmp_path, monkeypatch, seeds=(1, 2, 3, 4, 6)))
    assert _status(sources, "predictions") == "mismatch"


# ---------------------------------------------------------------- what Exp F is handed


def test_exp_f_only_ever_receives_an_approved_source(world, tmp_path, monkeypatch):
    manifest = world.snapshot()
    sources = _compare(manifest, _rebuild(tmp_path, monkeypatch))
    path = gate.write_json(tmp_path / "exp_a_sources.json", sources)

    record = gate.load_approved_sources(path, BAND)
    assert record["status"] == "approved"
    assert Path(record["final_selection_table"]["path"]).name == f"selection_table_{BAND}.csv"

    rejected = _compare(manifest, _rebuild(tmp_path / "moved", monkeypatch, vector_bump=1.0))
    rejected_path = gate.write_json(tmp_path / "rejected.json", rejected)
    with pytest.raises(gate.ReferenceGateError, match="refuses to consume"):
        gate.load_approved_sources(rejected_path, BAND)


def test_sources_schema_is_checked(tmp_path):
    path = gate.write_json(tmp_path / "wrong.json", {"schema_version": "something_else"})
    with pytest.raises(gate.ReferenceGateError, match="schema is"):
        gate.load_approved_sources(path, BAND)


# ------------------------------------------------------------------------ the entrypoint


def _entrypoint():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "experiments" / "validate_exp_a_reference.py"
    spec = importlib.util.spec_from_file_location("validate_exp_a_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_requires_an_explicit_pointer_file_and_never_globs(tmp_path):
    """Plan §6: the final run directory is supplied, never discovered. A missing or empty
    pointer must stop the compare rather than fall back to whatever ran most recently."""
    module = _entrypoint()
    with pytest.raises(gate.ReferenceGateError, match="missing run-directory pointer"):
        module._read_pointer(tmp_path / "absent.txt")

    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(gate.ReferenceGateError, match="is empty"):
        module._read_pointer(empty)

    dangling = tmp_path / "dangling.txt"
    dangling.write_text(str(tmp_path / "no_such_run"), encoding="utf-8")
    with pytest.raises(gate.ReferenceGateError, match="not a directory"):
        module._read_pointer(dangling)

    good = tmp_path / "good.txt"
    good.write_text(f"{tmp_path}\n", encoding="utf-8")
    assert module._read_pointer(good) == tmp_path


def test_entrypoint_snapshot_writes_the_manifest(world, tmp_path, monkeypatch):
    module = _entrypoint()
    monkeypatch.setattr(module, "load_config", lambda *paths: world.config)
    output = tmp_path / "milestone10" / "reference_exp_a_manifest.json"
    code = module.main([
        "--snapshot", "--reference-10", str(world.run_dir),
        "--output", str(output), "--no-hash-npz",
    ])
    assert code == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["bands_covered"] == [BAND]
    assert manifest["reference_grade"] == gate.GRADE_AUTHORITATIVE


# ------------------------------------------- a superseded run whose store no longer exists


def test_a_superseded_run_is_recorded_artifact_only_and_proves_the_substitution_is_safe(
    world, tmp_path
):
    """M10's real situation: the plan named the `f0a46aa6` Exp A runs, but that store was
    already rebuilt away, so the reference had to move to the `3f465abc` pair. The manifest
    has to say *why that was safe* — which is a table comparison, not feature evidence."""
    superseded = _write_run_dir(tmp_path / "superseded", world.sessions, commit="f" * 40)
    manifest = gate.snapshot(
        {BAND: world.config}, {BAND: world.run_dir},
        hash_npz=False, superseded_run_dirs={BAND: [superseded]},
    )
    record = manifest["bands"][BAND]["superseded_runs"][0]

    assert record["store_recomputable"] is False       # its store is gone; only tables remain
    assert record["commit"] == "f" * 40
    assert record["vs_reference"]["status"] == "equivalent"
    assert record["vs_reference"]["selection_table_byte_identical"] is True
    assert record["vs_reference"]["max_abs_pred_delta"] == 0.0
    # It is provenance, not a second gate: an equivalent superseded run changes nothing.
    assert manifest["reference_grade"] == gate.GRADE_AUTHORITATIVE


def test_a_superseded_run_that_selected_different_features_is_reported_as_differing(
    world, tmp_path
):
    """The finding this exists to surface: if the two runs disagree on which feature key a
    fold selected, moving the reference silently would change what Exp F consumes."""
    flipped = {s: FK_OFF for s in range(1, N_SUBJECTS + 1)}
    superseded = _write_run_dir(tmp_path / "superseded", world.sessions,
                                commit="f" * 40, selection=flipped)
    manifest = gate.snapshot(
        {BAND: world.config}, {BAND: world.run_dir},
        hash_npz=False, superseded_run_dirs={BAND: [superseded]},
    )
    verdict = manifest["bands"][BAND]["superseded_runs"][0]["vs_reference"]
    assert verdict["status"] == "differs"
    assert verdict["selected_feature_keys_identical"] is False


def test_a_superseded_run_beyond_the_prediction_tolerance_differs(world, tmp_path):
    superseded = _write_run_dir(tmp_path / "superseded", world.sessions, commit="f" * 40,
                                pred_noise=gate.pred_tolerance() * 100)
    manifest = gate.snapshot(
        {BAND: world.config}, {BAND: world.run_dir},
        hash_npz=False, superseded_run_dirs={BAND: [superseded]},
    )
    verdict = manifest["bands"][BAND]["superseded_runs"][0]["vs_reference"]
    assert verdict["status"] == "differs"
    assert verdict["selected_feature_keys_identical"] is True    # same models, moved numbers
    assert verdict["max_abs_pred_delta"] > gate.pred_tolerance()


# ------------------------------------------------------------------------ the entrypoint

SBATCH = Path(__file__).resolve().parents[1] / "scripts" / "ibex" / "validate_exp_a_reference.sbatch"


def test_the_ibex_wrapper_parses_and_is_git_free():
    """The M8 step-10.5 lesson, paid once: a compute node cannot answer `git`, so a wrapper
    must take its revision from the environment.

    `bash -n` reads the script from STDIN rather than by path. The existing M9 wrapper tests
    pass a path, which on this Windows checkout reaches Git Bash with its backslashes eaten
    (`C:Usersjosemsosag...`) and fails for reasons that have nothing to do with the script —
    feeding the text avoids the whole class. It is fed as BYTES with LF forced: Python's
    `text=True` rewrites `\\n` to `os.linesep` on the way into the pipe, and a CRLF landing
    after a line-continuation backslash ends the command early — which is the very
    CRLF-breaks-the-shell failure `.gitattributes` (`scripts/ibex/* text eol=lf`) exists to
    prevent, and it would otherwise be reported here as a syntax error in the script.
    """
    import shutil
    import subprocess

    text = SBATCH.read_text(encoding="utf-8")
    assert text.isascii(), "keep the wrapper ASCII - it is read by a shell under an unknown locale"
    if shutil.which("bash") is not None:
        script = text.replace("\r\n", "\n").encode("utf-8")
        result = subprocess.run(["bash", "-n"], input=script, capture_output=True)
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")

    assert "DEHYD_GIT_COMMIT" in text
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            assert not line.lstrip().startswith("git "), f"wrapper shells out to git: {line!r}"


def test_the_ibex_wrapper_carries_its_own_resources_and_both_modes():
    text = SBATCH.read_text(encoding="utf-8")
    header = "\n".join(line for line in text.splitlines() if line.startswith("#SBATCH"))
    for directive in ("--cpus-per-task", "--mem", "--time", "--output", "--error"):
        assert directive in header

    # The exact payloads, so a wrapper edit that drops a flag fails here rather than on IBEX.
    assert "experiments/validate_exp_a_reference.py --snapshot" in text
    assert "experiments/validate_exp_a_reference.py --compare" in text
    for flag in ("--reference-10", "--reference-77", "--reference-manifest",
                 "--final-10-file", "--final-77-file", "--config configs/ibex.yaml",
                 "--superseded-10", "--superseded-77"):
        assert flag in text
    # Fail closed on a mistyped MODE rather than silently doing neither.
    assert "exit 2" in text

    # The reference defaults must be the runs the LIVE stores back (3f465abc), with the
    # plan-named f0a46aa6 pair demoted to superseded — the whole point of the correction.
    assert "REF10=${REF10:-results/runs/20260804T150841445054Z_3f465abc}" in text
    assert "REF77=${REF77:-results/runs/20260804T171005433711Z_3f465abc}" in text
    assert "SUP10=${SUP10:-results/runs/20260803T143704568296Z_f0a46aa6}" in text
    assert "SUP77=${SUP77:-results/runs/20260803T151715023672Z_f0a46aa6}" in text


def test_entrypoint_returns_nonzero_for_a_degraded_snapshot(tmp_path, monkeypatch):
    other = World(tmp_path / "other", monkeypatch, store_commit="c" * 40)
    module = _entrypoint()
    monkeypatch.setattr(module, "load_config", lambda *paths: other.config)
    code = module.main([
        "--snapshot", "--reference-10", str(other.run_dir),
        "--output", str(tmp_path / "degraded.json"),
        "--no-hash-npz", "--allow-store-commit-divergence",
    ])
    assert code == 3
