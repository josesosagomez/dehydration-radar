"""Config loading, composition, path rules and validation.

The mandatory tests here must pass on a clean checkout with NO private data. The
canonical configs point at the real data/ tree and load_config requires input paths to
exist, so every mandatory test appends a final overlay whose input paths point into
tmp_path. That still exercises include composition, merge precedence, both path rules,
and every validation branch. Loading the canonical config unmodified is the separate
realdata test at the bottom.
"""

from pathlib import Path

import pytest
import yaml

from dehyd.config import REPO_ROOT, Config, ConfigError, config_to_dict, load_config

CONFIGS = REPO_ROOT / "configs"
EXP_A = CONFIGS / "exp_a_regression.yaml"


@pytest.fixture
def fake_data(tmp_path):
    """A minimal stand-in for the private dataset: the paths just have to exist."""
    data_dir = tmp_path / "radar"
    data_dir.mkdir()
    xlsx = tmp_path / "weights.xlsx"
    xlsx.write_bytes(b"")
    return data_dir, xlsx


@pytest.fixture
def overlay(tmp_path, fake_data):
    """Final overlay redirecting the required INPUT paths into tmp_path."""
    data_dir, xlsx = fake_data
    path = tmp_path / "overlay.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_10ghz_dir": str(data_dir),
                    "weight_xlsx": str(xlsx),
                    "results_dir": str(tmp_path / "results"),
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ------------------------------------------------------------ composition & merging


def test_loads_canonical_experiment_config_with_overlay(overlay, fake_data):
    cfg = load_config(EXP_A, overlay)
    data_dir, xlsx = fake_data

    assert isinstance(cfg, Config)
    assert cfg.paths.data_10ghz_dir == data_dir.resolve()
    assert cfg.paths.weight_xlsx == xlsx.resolve()
    # Values from the included files survived composition:
    assert cfg.preprocess.butter_order == 4
    assert cfg.preprocess.edge_trim == 32
    assert cfg.qc.histogram_bins == 200
    assert cfg.wst.max_order == 2
    # ...as did this file's own keys:
    assert cfg.run.seed_set == (1, 2, 3, 4, 5)
    assert cfg.split.min_train_subjects == 3


def test_results_dir_need_not_exist(overlay):
    """results_dir is an OUTPUT: creatable, not required to pre-exist."""
    cfg = load_config(EXP_A, overlay)
    assert not cfg.paths.results_dir.exists()


def test_later_file_wins(tmp_path, overlay):
    later = write_yaml(tmp_path / "later.yaml", {"run": {"seed": 999}})
    cfg = load_config(EXP_A, overlay, later)
    assert cfg.run.seed == 999
    assert cfg.run.seed_set == (1, 2, 3, 4, 5)  # untouched keys survive


def test_lists_are_replaced_not_concatenated(tmp_path, overlay):
    later = write_yaml(tmp_path / "later.yaml", {"run": {"seed_set": [7, 8, 9, 10, 11]}})
    cfg = load_config(EXP_A, overlay, later)
    assert cfg.run.seed_set == (7, 8, 9, 10, 11)


def test_include_resolves_against_declaring_file_not_cwd(tmp_path, overlay, monkeypatch):
    """Both path rules hold from an unrelated CWD.

    The experiment config is passed by ABSOLUTE path — a relative top-level config path
    would be unresolvable from a different CWD, which is not what this test probes.
    """
    from_root = load_config(EXP_A, overlay)
    monkeypatch.chdir(tmp_path)
    from_elsewhere = load_config(EXP_A.resolve(), overlay.resolve())

    assert config_to_dict(from_root) == config_to_dict(from_elsewhere)


def test_nested_include_rejected(tmp_path, overlay):
    inner = write_yaml(tmp_path / "inner.yaml", {"include": ["data.yaml"]})
    outer = write_yaml(tmp_path / "outer.yaml", {"include": [inner.name]})
    with pytest.raises(ConfigError, match="nested include"):
        load_config(outer, overlay)


# ------------------------------------------------------------------- path validation


def test_missing_input_path_raises(tmp_path):
    bad = write_yaml(
        tmp_path / "bad.yaml",
        {
            "paths": {
                "data_10ghz_dir": str(tmp_path / "nope"),
                "weight_xlsx": str(tmp_path / "nope.xlsx"),
                "results_dir": str(tmp_path / "results"),
            },
            "run": {"seed": 1, "seed_set": [1, 2, 3, 4, 5], "device": "cpu"},
        },
    )
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(bad)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


# ------------------------------------------------------------------ schema validation


def test_unknown_key_rejected(tmp_path, overlay):
    bad = write_yaml(tmp_path / "bad.yaml", {"run": {"sed": 1}})
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(EXP_A, overlay, bad)


def test_unknown_section_rejected(tmp_path, overlay):
    bad = write_yaml(tmp_path / "bad.yaml", {"modelz": {"kind": "ridge"}})
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(EXP_A, overlay, bad)


def test_missing_required_key_raises(tmp_path, fake_data):
    data_dir, xlsx = fake_data
    bad = write_yaml(
        tmp_path / "bad.yaml",
        {
            "paths": {"data_10ghz_dir": str(data_dir), "weight_xlsx": str(xlsx)},
            "run": {"seed": 1, "seed_set": [1, 2, 3, 4, 5], "device": "cpu"},
        },
    )
    with pytest.raises(ConfigError, match="missing required key"):
        load_config(bad)


def test_wrong_type_raises(tmp_path, overlay):
    bad = write_yaml(tmp_path / "bad.yaml", {"run": {"seed": "not-an-int"}})
    with pytest.raises(ConfigError, match="run.seed must be an integer"):
        load_config(EXP_A, overlay, bad)


# ----------------------------------------------------------- numeric protocol floors


def test_seed_set_must_have_five_entries(tmp_path, overlay):
    bad = write_yaml(tmp_path / "bad.yaml", {"run": {"seed_set": [1, 2, 3]}})
    with pytest.raises(ConfigError, match="exactly 5 seeds"):
        load_config(EXP_A, overlay, bad)


def test_seed_set_must_be_distinct(tmp_path, overlay):
    """Duplicates would silently give fewer than 5 effective stochastic repeats."""
    bad = write_yaml(tmp_path / "bad.yaml", {"run": {"seed_set": [1, 1, 2, 3, 4]}})
    with pytest.raises(ConfigError, match="DISTINCT"):
        load_config(EXP_A, overlay, bad)


def test_device_must_be_valid(tmp_path, overlay):
    bad = write_yaml(tmp_path / "bad.yaml", {"run": {"device": "tpu"}})
    with pytest.raises(ConfigError, match="run.device"):
        load_config(EXP_A, overlay, bad)


def test_n_inner_max_floor(tmp_path, overlay):
    bad = write_yaml(tmp_path / "bad.yaml", {"split": {"n_inner_max": 1}})
    with pytest.raises(ConfigError, match="n_inner_max must be >= 2"):
        load_config(EXP_A, overlay, bad)


def test_min_train_subjects_floor_is_protocol_not_library(tmp_path, overlay):
    """2 is GroupKFold's mechanical floor but violates the approved nested-CV rule."""
    bad = write_yaml(tmp_path / "bad.yaml", {"split": {"min_train_subjects": 2}})
    with pytest.raises(ConfigError, match="min_train_subjects must be >= 3"):
        load_config(EXP_A, overlay, bad)


def test_wst_tilings_cannot_be_overridden(tmp_path, overlay):
    bad = write_yaml(tmp_path / "bad.yaml", {"wst": {"tilings": [{"q": [2, 2]}]}})
    with pytest.raises(ConfigError, match="frozen constants"):
        load_config(EXP_A, overlay, bad)


# ------------------------------------------------------------------------ provenance


def test_config_to_dict_is_json_serializable(overlay):
    import json

    payload = config_to_dict(load_config(EXP_A, overlay))
    json.dumps(payload)  # must not raise
    assert payload["preprocess"]["butter_order"] == 4
    assert isinstance(payload["paths"]["data_10ghz_dir"], str)


# -------------------------------------------------------------------------- realdata


@pytest.mark.realdata
def test_canonical_config_points_at_the_real_data(real_data_paths):
    """The committed run config really does resolve to the real dataset."""
    cfg = load_config(EXP_A)
    assert cfg.paths.data_10ghz_dir == real_data_paths["data_10ghz_dir"]
    assert cfg.paths.weight_xlsx == real_data_paths["weight_xlsx"]
