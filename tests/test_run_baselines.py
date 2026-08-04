"""T-M9-entrypoints + T-M9-expd-shard (CLI half): `experiments/run_baselines.py`.

Three things pytest can hold this file to without a GPU or the real cohort:

  * the FLAG MATRIX — a CNN family runs as a run group (init -> fold array -> merge) or as a
    local smoke, never as one sequential full-cohort job (the C11 mistake); the cheap
    families and `comparisons` take --subset XOR --full-cohort;
  * the RUN-LEVEL PROTOCOL — outside the smoke the seed set is the frozen (1,2,3,4,5) and
    only a CNN family may name a GPU (§2.11, §5 trap 11);
  * the REAL `record_run` contract of `--init-run-group` (C19-C22): a real manifest hashed
    against the band-correct root, a real fold-role manifest, and the authoritative per-fold
    ROW CENSUS the merge validates every shard against — plus the lineage refusals that must
    fire BEFORE any GPU work.

The training itself is covered by test_exp_d.py; nothing here trains a network.
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from dehyd.config import load_config
from dehyd.data.sessions import SESSION_NAMES
from dehyd.eval import exp_a, exp_b, exp_d
from dehyd.features.store import write_session_store

import experiments.run_baselines as rb

CONFIG_ARGS = ["--config", "configs/exp_a_regression.yaml", "--config", "configs/baselines.yaml"]
N_SUBJECTS = 4
N_FRAMES = 2


@pytest.fixture(scope="module")
def config():
    return load_config("configs/exp_a_regression.yaml", "configs/baselines.yaml")


def _sessions(n_subjects=N_SUBJECTS):
    return [
        {"subject": s, "session_idx": i, "session_name": SESSION_NAMES[i],
         "rel_path": f"subject_{s}_{SESSION_NAMES[i]}.mat",
         "frame_ids": list(range(N_FRAMES)), "delta_m_pct": 0.0 if i == 0 else -(0.4 * i + 0.1 * s)}
        for s in range(1, n_subjects + 1)
        for i in range(5)
    ]


def _write_store(store_dir, sessions, seed=0):
    """A schema-v2 store carrying only the 10 GHz signal Exp D reads."""
    rng = np.random.default_rng(seed)
    for s in sessions:
        n = len(s["frame_ids"])
        write_session_store(
            "10ghz", s["subject"], s["session_name"],
            {"sig__raw_beat": (rng.standard_normal((n, 534))
                               + 1j * rng.standard_normal((n, 534)))},
            {"n_frames": n}, store_dir,
        )


def _manifest(data_dir, sessions):
    rows = []
    for s in sessions:
        (data_dir / s["rel_path"]).write_bytes(f"radar {s['rel_path']}".encode())
        rows.append({"subject": s["subject"], "session_idx": s["session_idx"],
                     "rel_path": s["rel_path"]})
    return pd.DataFrame(rows)


def _overlay(tmp_path):
    data_10 = tmp_path / "data10"
    results = tmp_path / "results"
    for directory in (data_10, results):
        directory.mkdir(exist_ok=True)
    (tmp_path / "weights.xlsx").write_bytes(b"pretend workbook")
    path = tmp_path / "paths_overlay.yaml"
    path.write_text(
        "paths:\n"
        f"  data_10ghz_dir: {data_10.as_posix()}\n"
        f"  weight_xlsx: {(tmp_path / 'weights.xlsx').as_posix()}\n"
        f"  results_dir: {results.as_posix()}\n",
        encoding="utf-8",
    )
    return path, data_10, results


def _patch(monkeypatch, sessions, data_dir):
    monkeypatch.setattr(rb.exp_a, "build_sessions", lambda *a, **k: sessions)
    monkeypatch.setattr(rb, "_build_manifest_qc", lambda config, band: _manifest(data_dir, sessions))
    monkeypatch.setattr(rb.exp_a, "expected_fingerprints", lambda *a, **k: {})
    monkeypatch.setattr(rb.store_mod, "validate_store", lambda *a, **k: None)


# ---------------------------------------------------------------------- the flag matrix


def test_cnn_family_has_no_single_process_full_cohort_mode():
    """(Step 0 item 3 / C11) 16 outer folds run sequentially cost their sum; the fold array
    is the only full-cohort path, so the flag that would invite the mistake is refused."""
    with pytest.raises(SystemExit):
        rb.main(CONFIG_ARGS + ["--family", "cnn1d_raw", "--full-cohort"])


def test_cnn_family_requires_exactly_one_mode():
    for extra in ([],
                  ["--init-run-group", "--merge-folds", "--run-dir", "x"],
                  ["--subset", "6subjects", "--init-run-group"]):
        with pytest.raises(SystemExit):
            rb.main(CONFIG_ARGS + ["--family", "spec2d_raw"] + extra)


def test_fold_and_merge_require_a_run_dir():
    with pytest.raises(SystemExit):
        rb.main(CONFIG_ARGS + ["--family", "cnn1d_raw", "--fold", "0"])
    with pytest.raises(SystemExit):
        rb.main(CONFIG_ARGS + ["--family", "cnn1d_raw", "--merge-folds"])


def test_a_negative_fold_position_is_refused(tmp_path):
    """A fold id is a POSITION in the selectable-fold list; `folds[-1]` would run the LAST
    fold while writing the shard under the requested (negative) name."""
    with pytest.raises(SystemExit):
        rb.main(CONFIG_ARGS + ["--family", "cnn1d_raw", "--fold", "-1",
                               "--run-dir", str(tmp_path)])


def test_cheap_families_reject_the_run_group_flags():
    with pytest.raises(SystemExit):
        rb.main(CONFIG_ARGS + ["--family", "physics", "--init-run-group"])
    with pytest.raises(SystemExit):
        rb.main(CONFIG_ARGS + ["--family", "physics"])
    with pytest.raises(SystemExit):
        rb.main(CONFIG_ARGS + ["--family", "session_index", "--subset", "6subjects",
                               "--full-cohort"])


def test_comparisons_require_the_radar_side_and_every_family_dir(tmp_path):
    base = CONFIG_ARGS + ["--family", "comparisons", "--full-cohort"]
    with pytest.raises(SystemExit):
        rb.main(base)
    with pytest.raises(SystemExit):    # exp-a given, families missing
        rb.main(base + ["--exp-a-run-dir", str(tmp_path), "--m7-reference-dir", str(tmp_path)])
    with pytest.raises(SystemExit):    # one family short of the six
        rb.main(base + ["--exp-a-run-dir", str(tmp_path), "--m7-reference-dir", str(tmp_path)]
                + [f"--family-run-dir={f}={tmp_path}" for f in exp_d.EXPD_FAMILIES[:-1]])
    with pytest.raises(SystemExit):    # an unknown family name in NAME=PATH
        rb.main(base + [f"--family-run-dir=exp_a={tmp_path}"])


# ------------------------------------------------------- the frozen run-level protocol


def test_full_cohort_refuses_a_seed_set_that_is_not_the_frozen_one(tmp_path, monkeypatch):
    """A reporting run is frozen at seeds (1,2,3,4,5) exactly — a same-SIZE but different set
    is the case `load_config`'s own "exactly 5 distinct seeds" rule cannot catch, so it has to
    be caught here.

    NB: `load_config` (config.py:852) already refuses any seed set that is not of size 5, so
    the `seed_set=[1]` smoke overlay §2.11 describes is not expressible in a config file at
    all — flagged for the owner; this entrypoint's contract is only that it does not ADD a
    frozen-set requirement in smoke mode (asserted directly below).
    """
    overlay, data_10, _ = _overlay(tmp_path)
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text("run:\n  seed_set: [1, 2, 3, 4, 6]\n", encoding="utf-8")
    _patch(monkeypatch, _sessions(), data_10)
    with pytest.raises(SystemExit, match="seed_set"):
        rb.main(CONFIG_ARGS + ["--config", str(overlay), "--config", str(seeds),
                               "--family", "physics", "--full-cohort"])


def test_smoke_mode_imposes_no_seed_set_or_device_requirement(config):
    """The mechanism-only smoke differs from a full run by SUBSET, SEEDS and DEVICE and by
    nothing else (CLAUDE.md's smoke rule), so the run-level guard must be inert there."""
    import dataclasses

    reduced = dataclasses.replace(
        config, run=dataclasses.replace(config.run, seed_set=(1,), device="cuda")
    )
    rb._require_frozen_run_protocol(reduced, "physics", "smoke")          # no raise
    with pytest.raises(SystemExit, match="seed_set"):
        rb._require_frozen_run_protocol(reduced, "physics", "full")
    # ...and a CNN family may name a GPU even in a full run
    gpu = dataclasses.replace(
        config, run=dataclasses.replace(config.run, device="cuda")
    )
    rb._require_frozen_run_protocol(gpu, "spec2d_raw", "full")            # no raise


def test_only_a_cnn_family_may_name_a_gpu(tmp_path, monkeypatch):
    """(§5 trap 11) The DL baselines are the one authorized GPU path; the WST/numpy
    canonical-backend policy is untouched, so every other family runs on CPU."""
    overlay, data_10, _ = _overlay(tmp_path)
    device = tmp_path / "device.yaml"
    device.write_text("run:\n  device: cuda\n", encoding="utf-8")
    _patch(monkeypatch, _sessions(), data_10)
    with pytest.raises(SystemExit, match="authorized GPU path"):
        rb.main(CONFIG_ARGS + ["--config", str(overlay), "--config", str(device),
                               "--family", "session_index", "--full-cohort"])


def test_the_smoke_surfaces_no_performance_value(tmp_path, monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    sessions = _sessions(n_subjects=6)
    _write_store(results, sessions)
    _patch(monkeypatch, sessions, data_10)

    assert rb.main(CONFIG_ARGS + ["--config", str(overlay),
                                  "--family", "physics", "--subset", "6subjects"]) == 0

    run_dir = sorted((results / "runs").iterdir())[0]
    written = sorted(p.name for p in run_dir.iterdir())
    assert written == ["provenance.json", "run_log_physics_10ghz.json"]
    log = json.loads((run_dir / "run_log_physics_10ghz.json").read_text(encoding="utf-8"))
    assert log["mode"] == "mechanism-only"
    assert set(log) == {"stage", "band", "family", "mode", "n_folds", "n_sessions", "note"}
    assert not list(results.rglob("metrics_physics_*"))
    assert not list(results.rglob("predictions_physics_*"))


def test_full_cohort_cheap_family_writes_the_four_merged_artifacts(tmp_path, monkeypatch):
    overlay, data_10, results = _overlay(tmp_path)
    sessions = _sessions()
    _write_store(results, sessions)
    _patch(monkeypatch, sessions, data_10)

    assert rb.main(CONFIG_ARGS + ["--config", str(overlay), "--family", "session_index",
                                  "--full-cohort"]) == 0

    run_dir = sorted((results / "runs").iterdir())[0]
    for name in ("predictions", "metrics", "selection", "per_subject"):
        suffix = "json" if name == "metrics" else "csv"
        assert (run_dir / f"{name}_session_index_10ghz.{suffix}").is_file()
    # the comparison stage's own reader must accept what this wrote
    artifacts = exp_d.load_family_artifacts(run_dir, "10ghz", "session_index")
    assert artifacts.deterministic is True and artifacts.seeds == [1]
    loaded = load_config(*CONFIG_ARGS[1::2], str(overlay))
    assert artifacts.metrics["lineage"]["config_hash"] == exp_b.config_fingerprint(loaded)


# ------------------------------------------- --init-run-group: the REAL record_run contract


def _init_group(tmp_path, monkeypatch, sessions, family="cnn1d_raw"):
    overlay, data_10, results = _overlay(tmp_path)
    _write_store(results, sessions)
    _patch(monkeypatch, sessions, data_10)
    assert rb.main(CONFIG_ARGS + ["--config", str(overlay), "--family", family,
                                  "--init-run-group"]) == 0
    run_dir = sorted((results / "runs").iterdir())[0]
    return run_dir, results, overlay


def test_init_run_group_records_the_authoritative_per_fold_row_census(tmp_path, monkeypatch):
    """(C19-C22 + the one field the M8 contract had no need for) Nothing else recorded can
    catch a truncated or stale shard: `record_run`'s manifest holds cohort totals and the
    fold manifest holds roles, so a shard that silently dropped test rows would have no
    reference to be rejected by. The census is that reference."""
    sessions = _sessions()
    run_dir, _, _ = _init_group(tmp_path, monkeypatch, sessions)
    payload = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    extra = payload["extra"]

    assert extra["stage"] == "exp-d-cnn-group"
    assert extra["band"] == "10ghz" and extra["family"] == "cnn1d_raw"
    assert extra["seed_set"] == [1, 2, 3, 4, 5]
    assert payload["inputs"]["radar_files"]          # a REAL manifest was hashed
    assert len(payload["folds"]) == N_SUBJECTS       # a REAL fold-role manifest

    census = extra["expected_test_rows_by_fold"]
    folds = exp_d.selectable_folds(sorted({s["subject"] for s in sessions}))
    assert sorted(int(k) for k in census) == list(range(len(folds)))
    for fold_id, fold in enumerate(folds):
        entry = census[str(fold_id)]
        assert entry["test_subject"] == fold.test_subject
        assert entry["n_frame_rows"] == 5 * N_FRAMES     # 5 sessions of the held-out subject
        assert entry["n_session_rows"] == 5
        assert len(entry["frame_rows_sha256"]) == 64 and len(entry["session_rows_sha256"]) == 64
        assert entry["frame_rows_sha256"] != entry["session_rows_sha256"]
        assert entry["seed_set"] == [1, 2, 3, 4, 5]


def test_init_run_group_hashes_against_the_77ghz_root(tmp_path, monkeypatch):
    """(C22) The init stage is the one Exp D stage that writes provenance, so it is the one
    that must pass `require_77ghz_dir(config)` — `_hash_inputs` would otherwise silently hash
    10 GHz bytes under a 77 GHz label."""
    data_10 = tmp_path / "data10"
    data_77 = tmp_path / "data77"
    results = tmp_path / "results"
    for directory in (data_10, data_77, results):
        directory.mkdir()
    (tmp_path / "weights.xlsx").write_bytes(b"pretend workbook")
    overlay = tmp_path / "paths_overlay.yaml"
    overlay.write_text(
        "paths:\n"
        f"  data_10ghz_dir: {data_10.as_posix()}\n"
        f"  data_77ghz_dir: {data_77.as_posix()}\n"
        f"  weight_xlsx: {(tmp_path / 'weights.xlsx').as_posix()}\n"
        f"  results_dir: {results.as_posix()}\n",
        encoding="utf-8",
    )

    sessions = _sessions()
    rng = np.random.default_rng(1)
    for s in sessions:
        write_session_store("77ghz", s["subject"], s["session_name"],
                            {"sig__raw_slowtime": rng.standard_normal((N_FRAMES, 256))},
                            {"n_frames": N_FRAMES}, results)
    _patch(monkeypatch, sessions, data_10)
    for s in sessions:
        (data_77 / s["rel_path"]).write_bytes(f"SEVENTY-SEVEN {s['rel_path']}".encode())

    assert rb.main(CONFIG_ARGS + ["--config", str(overlay), "--band", "77ghz",
                                  "--family", "cnn1d_raw", "--init-run-group"]) == 0

    from dehyd.provenance import sha256_file

    payload = json.loads(
        (sorted((results / "runs").iterdir())[0] / "provenance.json").read_text(encoding="utf-8")
    )
    assert payload["extra"]["band"] == "77ghz"
    for entry in payload["inputs"]["radar_files"]:
        assert entry["sha256"] == sha256_file(data_77 / entry["rel_path"])
        assert entry["sha256"] != sha256_file(data_10 / entry["rel_path"])


def test_init_prints_the_run_dir_as_its_last_stdout_line(tmp_path, monkeypatch, capsys):
    """The machine-readable handoff `submit_exp_d_cnn.sh` reads with `tail -n1`."""
    run_dir, _, _ = _init_group(tmp_path, monkeypatch, _sessions())
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(run_dir)


def test_fold_task_refuses_a_run_group_it_does_not_belong_to(tmp_path, monkeypatch):
    """Every lineage field, by name, BEFORE any training — a wrong `--run-dir` must cost a
    diagnosis, not eight GPU-hours and a corrupt merge."""
    sessions = _sessions()
    run_dir, results, overlay = _init_group(tmp_path, monkeypatch, sessions)

    args = CONFIG_ARGS + ["--config", str(overlay), "--fold", "0", "--run-dir", str(run_dir)]
    with pytest.raises(SystemExit, match="family"):
        rb.main(args + ["--family", "spec2d_raw"])
    with pytest.raises(SystemExit, match="band"):
        rb.main(args + ["--family", "cnn1d_raw", "--band", "77ghz"])

    payload = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    payload["extra"]["config_hash"] = "drifted"
    (run_dir / "provenance.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="config_hash"):
        rb.main(args + ["--family", "cnn1d_raw"])


def test_a_fold_task_naming_a_different_device_than_init_is_refused(tmp_path, monkeypatch):
    """`run.device` is inside `config_to_dict`, so it is hashed into `config_fingerprint` —
    which means a GPU overlay applied to the FOLD stage alone (the obvious way to wire
    `device: cuda`, §5 trap 11) makes every array task fail lineage validation AFTER init
    already succeeded. This test pins that failure so the sbatch's uniform-overlay structure
    cannot be "simplified" back into the broken shape.
    """
    sessions = _sessions()
    run_dir, _, overlay = _init_group(tmp_path, monkeypatch, sessions)   # init: device cpu
    device = tmp_path / "gpu.yaml"
    device.write_text("run:\n  device: cuda\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="config_hash"):
        rb.main(CONFIG_ARGS + ["--config", str(overlay), "--config", str(device),
                               "--family", "cnn1d_raw", "--fold", "0",
                               "--run-dir", str(run_dir)])


def test_a_fold_index_beyond_the_selectable_list_is_a_named_no_op(tmp_path, monkeypatch):
    """(§5 trap 14) The array is a fixed 16 tasks while N_eval can be smaller; the extra tasks
    must exit 0 with a marker the merge can tell apart from a crash."""
    sessions = _sessions()
    run_dir, _, overlay = _init_group(tmp_path, monkeypatch, sessions)

    assert rb.main(CONFIG_ARGS + ["--config", str(overlay), "--family", "cnn1d_raw",
                                  "--fold", "15", "--run-dir", str(run_dir)]) == 0
    marker = run_dir / "exp_d_cnn1d_raw_10ghz_fold15.noop.json"
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["state"] == "noop"

    merged = exp_d.merge_exp_d_folds("10ghz", "cnn1d_raw", run_dir)
    assert merged["noop_out_of_range_folds"] == [15]
    assert merged["state"] == "partial_non_reportable" and merged["artifacts"] is None


def test_merge_writes_the_named_partial_state_rather_than_a_smaller_cohort(tmp_path,
                                                                          monkeypatch):
    sessions = _sessions()
    run_dir, _, overlay = _init_group(tmp_path, monkeypatch, sessions)

    assert rb.main(CONFIG_ARGS + ["--config", str(overlay), "--family", "cnn1d_raw",
                                  "--merge-folds", "--run-dir", str(run_dir)]) == 0
    merged = json.loads(
        (run_dir / "exp_d_cnn1d_raw_10ghz_merged.json").read_text(encoding="utf-8")
    )
    assert merged["state"] == "partial_non_reportable"
    assert merged["missing_folds"] == list(range(N_SUBJECTS))
    assert merged["artifacts"] is None


def test_comparisons_refuse_a_lineage_mismatched_family_directory(tmp_path, monkeypatch):
    """Named by family AND by field: the comparison consumes six directories, so "something
    is stale" is not an actionable message."""
    overlay, data_10, results = _overlay(tmp_path)
    sessions = _sessions()
    _patch(monkeypatch, sessions, data_10)

    family_dirs = {}
    for family in exp_d.EXPD_FAMILIES:
        directory = tmp_path / f"run_{family}"
        directory.mkdir()
        (directory / f"metrics_{family}_10ghz.json").write_text(
            json.dumps({"lineage": {"analysis_commit": "not-this-commit",
                                    "config_hash": "not-this-config"}}),
            encoding="utf-8",
        )
        family_dirs[family] = directory

    args = (CONFIG_ARGS + ["--config", str(overlay), "--family", "comparisons", "--full-cohort",
                           "--exp-a-run-dir", str(tmp_path), "--m7-reference-dir", str(tmp_path)]
            + [f"--family-run-dir={f}={p}" for f, p in family_dirs.items()])
    with pytest.raises(SystemExit) as excinfo:
        rb.main(args)
    message = str(excinfo.value)
    assert "lineage.analysis_commit" in message
    assert sorted(exp_d.EXPD_FAMILIES)[0] in message


def test_the_six_families_are_validated_against_their_own_authorized_device(tmp_path):
    """The CNN families run at `device: cuda` and the deterministic ones are REFUSED unless
    they are at `device: cpu`, while `config_fingerprint` hashes `run.device` along with
    everything else. So the six families cannot all carry one `config_hash`, and a plain
    equality check against the comparison's own hash could never pass on a real cohort — it
    would fail on four families or on two, whichever half the comparison was configured like.

    This pins the per-family normalization: each family is checked against the hash it is
    ALLOWED to have, and against nothing looser. Everything outside `run.device` still has to
    match exactly.
    """
    config = load_config(*CONFIG_ARGS[1::2])
    assert str(config.run.device) == "cpu", "the base experiment config pins device: cpu"

    cpu_hash = exp_b.config_fingerprint(config)
    gpu_config = dataclasses.replace(
        config, run=dataclasses.replace(config.run, device="cuda")
    )
    gpu_hash = exp_b.config_fingerprint(gpu_config)
    assert cpu_hash != gpu_hash, "run.device must be inside the fingerprint (trap 11)"

    for family in exp_d.DETERMINISTIC_FAMILIES:
        assert rb._expected_family_config_hash(config, family) == cpu_hash
        assert rb._expected_family_config_hash(gpu_config, family) == cpu_hash
    for family in exp_d.CNN_FAMILIES:
        assert rb._expected_family_config_hash(config, family) == gpu_hash
        assert rb._expected_family_config_hash(gpu_config, family) == gpu_hash

    # and the normalization is ONLY of the device: a genuinely different config still differs
    other = dataclasses.replace(config, run=dataclasses.replace(config.run, seed=config.run.seed + 1))
    assert rb._expected_family_config_hash(other, "physics") != cpu_hash
    assert rb._expected_family_config_hash(other, "cnn1d_raw") != gpu_hash
