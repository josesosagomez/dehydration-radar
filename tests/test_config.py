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


# ------------------------------------------------------- QC / preprocess validation
# M2 actually consumes these numbers, so every field it activates is validated here
# rather than surfacing later as a confusing numpy error inside a screen.


def test_in_band_margin_default_and_override(tmp_path, overlay):
    assert load_config(EXP_A, overlay).qc.in_band_margin_hz == 1000.0
    override = write_yaml(tmp_path / "m.yaml", {"qc": {"in_band_margin_hz": 250.0}})
    assert load_config(EXP_A, overlay, override).qc.in_band_margin_hz == 250.0


def test_yaml_list_gates_are_normalised_to_tuples(tmp_path, overlay):
    """A frozen dataclass must not carry a mutable list (and provenance wants one type)."""
    override = write_yaml(
        tmp_path / "g.yaml",
        {"qc": {"qc_gate_m": [0.8, 2.8]}, "preprocess": {"model_gate_m": [1.1, 1.9]}},
    )
    config = load_config(EXP_A, overlay, override)
    assert config.qc.qc_gate_m == (0.8, 2.8)
    assert config.preprocess.model_gate_m == (1.1, 1.9)
    assert isinstance(config.qc.qc_gate_m, tuple)
    assert isinstance(config.preprocess.model_gate_m, tuple)


@pytest.mark.parametrize(
    "section, key, value, match",
    [
        # --- out-of-range numbers ----------------------------------------------
        ("qc", "histogram_bins", 0, "must be >= 1"),
        ("qc", "histogram_bins", -5, "must be >= 1"),
        ("qc", "flatline_max_bin_fraction", 0.0, r"must be in \(0.0, 1.0\]"),
        ("qc", "flatline_max_bin_fraction", 1.5, r"must be in \(0.0, 1.0\]"),
        ("qc", "min_in_band_energy_ratio", -0.1, r"must be in \[0.0, 1.0\]"),
        ("qc", "min_in_band_energy_ratio", 1.1, r"must be in \[0.0, 1.0\]"),
        ("qc", "rms_robust_z_threshold", 0.0, "must be in"),
        ("qc", "in_band_margin_hz", -1.0, "must be in"),
        ("qc", "min_frame_fraction", 0.0, r"must be in \(0.0, 1.0\]"),
        ("qc", "min_frame_fraction", 1.5, r"must be in \(0.0, 1.0\]"),
        ("preprocess", "butter_order", 0, "must be >= 1"),
        ("preprocess", "edge_trim", -1, "must be an integer >= 0"),
        ("preprocess", "fs_hz", 0.0, "must be in"),
        ("preprocess", "bandwidth_hz", -1.0, "must be in"),
        ("preprocess", "chirp_time_s", 0.0, "must be in"),
        # --- wrong types (bool is an int in Python; it must not slip through) ---
        ("qc", "histogram_bins", "200", "must be an integer"),
        ("qc", "histogram_bins", True, "must be an integer"),
        ("qc", "min_in_band_energy_ratio", "0.3", "must be a number"),
        ("qc", "min_in_band_energy_ratio", True, "must be a number"),
        ("preprocess", "edge_trim", 32.5, "must be an integer >= 0"),
        # --- malformed gates ---------------------------------------------------
        ("qc", "qc_gate_m", [0.9, 2.0, 3.0], "exactly 2 entries"),
        ("qc", "qc_gate_m", [3.0, 0.9], "strictly increasing"),
        ("qc", "qc_gate_m", [2.0, 2.0], "strictly increasing"),
        ("qc", "qc_gate_m", 0.9, "two-element"),
        ("qc", "qc_gate_m", "0.9,3.0", "two-element"),
        ("qc", "qc_gate_m", [0.0, 3.0], "must be in"),
        ("qc", "qc_gate_m", [-1.0, 3.0], "must be in"),
        ("preprocess", "model_gate_m", [2.0, 1.0], "strictly increasing"),
    ],
)
def test_field_validation_rejects_bad_values(tmp_path, overlay, section, key, value, match):
    bad = write_yaml(tmp_path / "bad.yaml", {section: {key: value}})
    with pytest.raises(ConfigError, match=match):
        load_config(EXP_A, overlay, bad)


def test_non_finite_value_rejected(tmp_path, overlay):
    """YAML spells these `.nan` / `.inf`; they would poison the band arithmetic."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("qc:\n  in_band_margin_hz: .inf\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be finite"):
        load_config(EXP_A, overlay, bad)


def test_radar_constants_load_as_floats_not_strings(overlay):
    """YAML 1.1 only reads an exponent as a float when it is SIGNED.

    `bandwidth_hz: 500.0e6` silently loads as the string "500.0e6" — which M1 never
    noticed because nothing consumed it. Pin the parsed types and values here so the
    canonical config cannot regress into string arithmetic.
    """
    pre = load_config(EXP_A, overlay).preprocess
    for name in ("fs_hz", "bandwidth_hz", "chirp_time_s"):
        assert isinstance(getattr(pre, name), float), name
    assert pre.bandwidth_hz == 500e6
    assert pre.chirp_time_s == 1024e-6


def test_qc_gate_above_nyquist_rejected(tmp_path, overlay):
    """~3257.5 Hz/m against a 260 kHz Nyquist: a 100 m gate has nothing to measure."""
    bad = write_yaml(tmp_path / "bad.yaml", {"qc": {"qc_gate_m": [100.0, 200.0]}})
    with pytest.raises(ConfigError, match="above Nyquist"):
        load_config(EXP_A, overlay, bad)


def test_margin_covering_whole_spectrum_rejected(tmp_path, overlay):
    """A screen that can never fire is a config error, not a valid configuration."""
    bad = write_yaml(tmp_path / "bad.yaml", {"qc": {"in_band_margin_hz": 300_000.0}})
    with pytest.raises(ConfigError, match="entire spectrum"):
        load_config(EXP_A, overlay, bad)


# --------------------------------------------------- M3 preprocessing fields (§2.1)


def test_m3_preprocess_defaults(overlay):
    """The frozen M3 values, present without being written in any YAML."""
    pre = load_config(EXP_A, overlay).preprocess
    assert pre.gate_method == "butterworth"
    assert pre.standardize == "robust"
    assert pre.peak_neighbors == 1
    assert pre.mask_taper is True
    assert pre.fft_gate_transition_hz == 500.0


def test_m3_preprocess_yaml_overrides_honoured(tmp_path, overlay):
    override = write_yaml(
        tmp_path / "pp.yaml",
        {
            "preprocess": {
                "gate_method": "fft",
                "standardize": "meanstd",
                "peak_neighbors": 0,
                "mask_taper": False,
                "fft_gate_transition_hz": 250.0,
            }
        },
    )
    pre = load_config(EXP_A, overlay, override).preprocess
    assert (pre.gate_method, pre.standardize) == ("fft", "meanstd")
    assert pre.peak_neighbors == 0  # keeping only the peak bin is legitimate
    assert pre.mask_taper is False
    assert pre.fft_gate_transition_hz == 250.0


@pytest.mark.parametrize(
    "section, message",
    [
        ({"gate_method": "butterwurth"}, "must be one of"),
        ({"gate_method": 4}, "must be a string"),
        ({"standardize": "zscore"}, "must be one of"),
        ({"mask_taper": 1}, "must be true or false"),  # 0/1 is a typo, not a bool
        ({"peak_neighbors": -1}, "must be >= 0"),
        ({"peak_neighbors": True}, "must be an integer"),
        ({"fft_gate_transition_hz": -5.0}, "must be in"),
    ],
)
def test_m3_preprocess_bad_values_rejected(tmp_path, overlay, section, message):
    bad = write_yaml(tmp_path / "bad.yaml", {"preprocess": section})
    with pytest.raises(ConfigError, match=message):
        load_config(EXP_A, overlay, bad)


def test_model_gate_straddling_nyquist_rejected(tmp_path, overlay):
    """The case an "only the lower edge" check would miss.

    At ~3257.5 Hz/m against a 260.4 kHz Nyquist, a 70-90 m gate starts below Nyquist
    (228 kHz) and ends above it (293 kHz). The QC gate is widened to the same band so
    the QC check passes first and this test really exercises the MODEL band rule --
    without it, the failure would surface inside scipy.signal.butter instead.
    """
    bad = write_yaml(
        tmp_path / "bad.yaml",
        {"qc": {"qc_gate_m": [70.0, 90.0]}, "preprocess": {"model_gate_m": [70.0, 90.0]}},
    )
    with pytest.raises(ConfigError, match="whole band must be below Nyquist"):
        load_config(EXP_A, overlay, bad)


def test_model_gate_outside_qc_gate_rejected(tmp_path, overlay):
    """QC fixed the population on the wider gate; the model may not reach outside it."""
    bad = write_yaml(tmp_path / "bad.yaml", {"preprocess": {"model_gate_m": [0.5, 2.0]}})
    with pytest.raises(ConfigError, match="not contained in"):
        load_config(EXP_A, overlay, bad)


def test_model_gate_equal_to_qc_gate_accepted(tmp_path, overlay):
    """The 0.9-3.0 m inner-CV candidate is exactly the QC gate — containment is
    inclusive, so this must load."""
    ok = write_yaml(tmp_path / "ok.yaml", {"preprocess": {"model_gate_m": [0.9, 3.0]}})
    assert load_config(EXP_A, overlay, ok).preprocess.model_gate_m == (0.9, 3.0)


def test_fft_gate_covering_whole_spectrum_rejected(tmp_path, overlay):
    """Skirts wide enough to pass everything = a filter that filters nothing."""
    bad = write_yaml(
        tmp_path / "bad.yaml",
        {"preprocess": {"gate_method": "fft", "fft_gate_transition_hz": 300_000.0}},
    )
    with pytest.raises(ConfigError, match="entire spectrum"):
        load_config(EXP_A, overlay, bad)


def test_wide_fft_transition_allowed_on_the_butterworth_path(tmp_path, overlay):
    """The vacuity rule is about the FFT mask; it must not fire on the primary path."""
    ok = write_yaml(tmp_path / "ok.yaml", {"preprocess": {"fft_gate_transition_hz": 300_000.0}})
    assert load_config(EXP_A, overlay, ok).preprocess.gate_method == "butterworth"


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
