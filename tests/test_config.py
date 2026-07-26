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

from dehyd.config import (
    REPO_ROOT,
    BaselineConfig,
    Config,
    ConfigError,
    ExpBConfig,
    ExpCConfig,
    ExpEConfig,
    ExpFConfig,
    ExpGConfig,
    M6_SECTIONS,
    ModelGridConfig,
    Preprocess77Config,
    ProtocolFreezeConfig,
    QC77Config,
    SearchSpace10GHzConfig,
    SearchSpace77GHzConfig,
    StatsConfig,
    WST77Config,
    config_to_dict,
    load_config,
    require_77ghz_dir,
)

CONFIGS = REPO_ROOT / "configs"
EXP_A = CONFIGS / "exp_a_regression.yaml"
EXP_A_77 = CONFIGS / "exp_a_regression_77ghz.yaml"
EXP_77 = CONFIGS / "exp_77ghz.yaml"


def test_exp_a_entrypoint_configs_exist_and_expose_their_search_space():
    """Both Exp A entrypoint configs (referenced by run_regression / the sbatch scripts) must
    exist and wire in their band's frozen search space — catches a missing/renamed config
    before it fails deep in an IBEX run."""
    from dehyd.config import load_config

    c10 = load_config(EXP_A)
    assert c10.search_10ghz.tiling == ("T1", "T2", "T3")

    assert EXP_A_77.is_file(), f"missing 77 GHz Exp A entrypoint config: {EXP_A_77}"
    c77 = load_config(EXP_A_77)
    assert c77.search_77ghz.tiling == ("T1_77", "T2_77", "T3_77")
    assert c77.paths.data_77ghz_dir is not None  # data77.yaml is composed in


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


# --------------------------------------------------------------- WST field validation
# Consumed from M4 (features/wst.py), so validated at load per the M2 rule.


def test_wst_backend_default_and_override(tmp_path, overlay):
    assert load_config(EXP_A, overlay).wst.backend == "numpy"
    override = write_yaml(tmp_path / "b.yaml", {"wst": {"backend": "torch"}})
    assert load_config(EXP_A, overlay, override).wst.backend == "torch"


def test_wst_backend_rejects_unknown(tmp_path, overlay):
    bad = write_yaml(tmp_path / "b.yaml", {"wst": {"backend": "jax"}})
    with pytest.raises(ConfigError, match="wst.backend"):
        load_config(EXP_A, overlay, bad)


@pytest.mark.parametrize("value", [0, 3, -1])
def test_wst_max_order_rejects_out_of_range(tmp_path, overlay, value):
    bad = write_yaml(tmp_path / "mo.yaml", {"wst": {"max_order": value}})
    with pytest.raises(ConfigError, match="wst.max_order must be 1 or 2"):
        load_config(EXP_A, overlay, bad)


def test_wst_max_order_rejects_non_int(tmp_path, overlay):
    # A float (2.0) and a bool (YAML true) are both "not an integer" here.
    for value in (2.0, True):
        bad = write_yaml(tmp_path / "mo.yaml", {"wst": {"max_order": value}})
        with pytest.raises(ConfigError, match="wst.max_order must be an integer"):
            load_config(EXP_A, overlay, bad)


@pytest.mark.parametrize("value", [0, -1e-6])
def test_wst_log_epsilon_rejects_non_positive(tmp_path, overlay, value):
    bad = write_yaml(tmp_path / "eps.yaml", {"wst": {"log_epsilon": value}})
    with pytest.raises(ConfigError, match="wst.log_epsilon"):
        load_config(EXP_A, overlay, bad)


def test_wst_log_epsilon_override(tmp_path, overlay):
    override = write_yaml(tmp_path / "eps.yaml", {"wst": {"log_epsilon": 1.0e-8}})
    assert load_config(EXP_A, overlay, override).wst.log_epsilon == 1.0e-8


def test_wst_provenance_carries_backend(overlay):
    d = config_to_dict(load_config(EXP_A, overlay))
    assert d["wst"]["backend"] == "numpy"
    assert d["wst"]["max_order"] == 2


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


# ------------------------------------------- M5 77 GHz config fields (T-C77, §2.1)
# Three parallel top-level sections (qc77 / preprocess77 / wst77) — different physics,
# their own frozen defaults. Always built (via default_factory), so every 10 GHz base
# config exercises them at their defaults; overrides ride on EXP_A + overlay.


def test_77ghz_sections_default_on_a_10ghz_config(overlay):
    """The *77 sections appear at their frozen defaults even on a 10 GHz-only config."""
    cfg = load_config(EXP_A, overlay)
    assert cfg.qc77 == QC77Config()
    assert cfg.preprocess77 == Preprocess77Config()
    assert cfg.wst77 == WST77Config()


def test_77ghz_frozen_defaults_pinned():
    """One literal-pinning test for the frozen 77 GHz defaults group (the M4 exception).

    If step 6 replaces the flatline rule (outcome b), THESE literals change here in
    lockstep with QC77Config, the YAML, and the canonical guard — the test is the tripwire
    that a stale value was left behind.
    """
    qc, pre, wst = QC77Config(), Preprocess77Config(), WST77Config()
    assert (qc.histogram_bins, qc.flatline_max_bin_fraction, qc.flatline_skip_leading_bins,
            qc.min_in_band_energy_ratio, qc.in_band_margin_hz, qc.min_frame_fraction) == (
        128, 0.25, 1, 0.30, 1953.125, 0.5)
    assert (pre.butter_order, pre.gate_m, pre.fs_hz, pre.bandwidth_hz, pre.chirp_time_s,
            pre.standardize) == (4, (2.0, 4.0), 500e3, 2e9, 512e-6, "robust")
    assert wst.max_order == 2 and wst.log_epsilon == 1e-6 and wst.backend == "numpy"
    assert wst.tilings == (
        type(wst.tilings[0])(q=(8, 4), invariance_ms=20.0),
        type(wst.tilings[0])(q=(6, 4), invariance_ms=40.0),
        type(wst.tilings[0])(q=(4, 2), invariance_ms=60.0),
    )


def test_qc77_override_honoured(tmp_path, overlay):
    override = write_yaml(tmp_path / "q.yaml", {"qc77": {"min_in_band_energy_ratio": 0.5}})
    assert load_config(EXP_A, overlay, override).qc77.min_in_band_energy_ratio == 0.5


def test_preprocess77_override_honoured(tmp_path, overlay):
    override = write_yaml(tmp_path / "p.yaml", {"preprocess77": {"standardize": "meanstd"}})
    cfg = load_config(EXP_A, overlay, override)
    assert cfg.preprocess77.standardize == "meanstd"


def test_preprocess77_gate_normalised_to_tuple(tmp_path, overlay):
    override = write_yaml(tmp_path / "g.yaml", {"preprocess77": {"gate_m": [2.5, 3.5]}})
    gate = load_config(EXP_A, overlay, override).preprocess77.gate_m
    assert gate == (2.5, 3.5) and isinstance(gate, tuple)


def test_wst77_backend_default_and_override(tmp_path, overlay):
    assert load_config(EXP_A, overlay).wst77.backend == "numpy"
    override = write_yaml(tmp_path / "b.yaml", {"wst77": {"backend": "torch"}})
    assert load_config(EXP_A, overlay, override).wst77.backend == "torch"


def test_wst77_tilings_cannot_be_overridden(tmp_path, overlay):
    bad = write_yaml(tmp_path / "t.yaml", {"wst77": {"tilings": [{"q": [2, 2]}]}})
    with pytest.raises(ConfigError, match="frozen constants"):
        load_config(EXP_A, overlay, bad)


@pytest.mark.parametrize("value", [0, 3, -1])
def test_wst77_max_order_rejects_out_of_range(tmp_path, overlay, value):
    bad = write_yaml(tmp_path / "mo.yaml", {"wst77": {"max_order": value}})
    with pytest.raises(ConfigError, match="wst77.max_order must be 1 or 2"):
        load_config(EXP_A, overlay, bad)


@pytest.mark.parametrize(
    "section, key, value, match",
    [
        ("qc77", "histogram_bins", 128.0, "must be an integer"),
        ("qc77", "histogram_bins", 0, "must be >= 1"),
        ("qc77", "flatline_skip_leading_bins", -1, "must be an integer >= 0"),
        ("qc77", "flatline_skip_leading_bins", 1.0, "must be an integer >= 0"),
        ("qc77", "flatline_max_bin_fraction", 1.5, r"must be in \(0.0, 1.0\]"),
        ("qc77", "min_in_band_energy_ratio", 1.1, r"must be in \[0.0, 1.0\]"),
        ("qc77", "min_frame_fraction", 0.0, r"must be in \(0.0, 1.0\]"),
        ("preprocess77", "butter_order", 0, "must be >= 1"),
        ("preprocess77", "fs_hz", 0.0, "must be in"),
        ("preprocess77", "standardize", "zscore", "preprocess77.standardize"),
        ("wst77", "backend", "jax", "wst77.backend"),
        ("wst77", "log_epsilon", 0.0, "wst77.log_epsilon"),
    ],
)
def test_77ghz_bad_values_rejected(tmp_path, overlay, section, key, value, match):
    bad = write_yaml(tmp_path / "bad.yaml", {section: {key: value}})
    with pytest.raises(ConfigError, match=match):
        load_config(EXP_A, overlay, bad)


def test_qc77_gate_above_nyquist_rejected(tmp_path, overlay):
    """A syntactically valid but physically out-of-band 77 GHz gate fails the cross-check."""
    bad = write_yaml(tmp_path / "bad.yaml", {"preprocess77": {"gate_m": [10.0, 12.0]}})
    with pytest.raises(ConfigError, match="above Nyquist"):
        load_config(EXP_A, overlay, bad)


def test_preprocess77_gate_must_be_increasing(tmp_path, overlay):
    bad = write_yaml(tmp_path / "bad.yaml", {"preprocess77": {"gate_m": [4.0, 2.0]}})
    with pytest.raises(ConfigError, match="strictly increasing"):
        load_config(EXP_A, overlay, bad)


def test_data_77ghz_dir_optional(overlay):
    """A 10 GHz-only config loads fine with data_77ghz_dir unset (None)."""
    assert load_config(EXP_A, overlay).paths.data_77ghz_dir is None


def test_data_77ghz_dir_existence_checked_when_present(tmp_path, overlay):
    bad = write_yaml(
        tmp_path / "d.yaml", {"paths": {"data_77ghz_dir": str(tmp_path / "nope")}}
    )
    with pytest.raises(ConfigError, match="data_77ghz_dir does not exist"):
        load_config(EXP_A, overlay, bad)


def test_require_77ghz_dir_raises_when_absent(overlay):
    cfg = load_config(EXP_A, overlay)
    with pytest.raises(ConfigError, match="data_77ghz_dir is not set"):
        require_77ghz_dir(cfg)


def test_require_77ghz_dir_returns_when_present(tmp_path, overlay):
    d77 = tmp_path / "data77"
    d77.mkdir()
    override = write_yaml(tmp_path / "d.yaml", {"paths": {"data_77ghz_dir": str(d77)}})
    cfg = load_config(EXP_A, overlay, override)
    assert require_77ghz_dir(cfg) == d77.resolve()


def test_10ghz_config_dict_unchanged_except_additive_77_defaults(overlay):
    """Existing 10 GHz sections are byte-identical; the only new keys are the *77 defaults."""
    d = config_to_dict(load_config(EXP_A, overlay))
    # The pre-M5 sections are untouched.
    assert d["qc"]["histogram_bins"] == 200
    assert d["preprocess"]["butter_order"] == 4
    assert d["wst"]["backend"] == "numpy"
    assert d["paths"]["data_77ghz_dir"] is None
    # The additive sections carry exactly the frozen 77 GHz defaults.
    assert d["qc77"] == config_to_dict(QC77Config())
    assert d["preprocess77"] == config_to_dict(Preprocess77Config())
    assert d["wst77"] == config_to_dict(WST77Config())


# ------------------------------------------------------------------------ provenance


def test_config_to_dict_is_json_serializable(overlay):
    import json

    payload = config_to_dict(load_config(EXP_A, overlay))
    json.dumps(payload)  # must not raise
    assert payload["preprocess"]["butter_order"] == 4
    assert isinstance(payload["paths"]["data_10ghz_dir"], str)


# ---------------------------------------------------------------- milestone 6 (freeze)
# The M6 sections are FROZEN RECORDS: a run YAML may restate a default but not change it,
# and modelling entrypoints re-validate via protocol_freeze_guard. These tests pin the
# frozen values (the tripwire that a stale value was left behind if a future amendment
# changes one), prove restating is allowed and changing is rejected, and prove the two
# band search spaces cannot express each other's candidates (C6-03).


def test_m6_sections_default_on_a_10ghz_config(overlay):
    """Every M6 section appears at its frozen default even on a plain 10 GHz config."""
    cfg = load_config(EXP_A, overlay)
    for name, cls in M6_SECTIONS.items():
        assert getattr(cfg, name) == cls(), name


def test_m6_search_10ghz_frozen_defaults_pinned():
    """Literal-pinning tripwire for the 10 GHz search space + model grid (T-C6-search)."""
    s = SearchSpace10GHzConfig()
    assert s.reduction == ("A", "B")
    assert s.channel == ("mag", "iq")
    assert s.tiling == ("T1", "T2", "T3")
    assert s.log_branches == ("off", "on_frozen_eps", "on_tuned_eps")
    assert s.range_gate_m == ((1.0, 2.0), (0.9, 3.0))
    assert s.model_families == ("ridge", "svr", "rf", "gbm", "knn")
    assert s.budget_k == 12
    assert (s.stage1_anchor_model, s.stage1_anchor_ridge_alpha, s.tuned_eps_k) == (
        "ridge", 1.0, 0.1)


def test_m6_search_77ghz_fixes_reduction_channel_gate_as_scalars():
    """77 GHz reduction/channel/gate are FIXED scalars, not candidate sets (C6-03)."""
    s = SearchSpace77GHzConfig()
    assert s.reduction == "slow_time_iq_primary" and isinstance(s.reduction, str)
    assert s.channel == "iq" and isinstance(s.channel, str)
    assert s.gate_m == (2.0, 4.0)
    assert s.tiling == ("T1_77", "T2_77", "T3_77")
    assert s.log_branches == ("off", "on_frozen_eps", "on_tuned_eps")


def test_m6_cross_band_candidates_are_structurally_inexpressible():
    """A 10 GHz-only candidate cannot be named on the 77 GHz space and vice versa (C6-03).

    This is a dataclass-shape guarantee, not a runtime check: the 77 GHz space has no
    `reduction`/`channel` TUPLE to hold {A, B}/{mag} and no `range_gate_m` axis at all, so
    the leak the shared-schema draft risked cannot be expressed.
    """
    s10, s77 = SearchSpace10GHzConfig(), SearchSpace77GHzConfig()
    assert isinstance(s10.reduction, tuple) and isinstance(s77.reduction, str)
    assert isinstance(s10.channel, tuple) and isinstance(s77.channel, str)
    assert hasattr(s10, "range_gate_m") and not hasattr(s77, "range_gate_m")
    # 77 GHz has a single fixed gate_m instead of a candidate set.
    assert hasattr(s77, "gate_m") and not hasattr(s10, "gate_m")


def test_m6_frozen_record_rejects_a_changed_value(tmp_path, overlay):
    bad = write_yaml(tmp_path / "b.yaml", {"search_10ghz": {"budget_k": 8}})
    with pytest.raises(ConfigError, match="frozen protocol constant"):
        load_config(EXP_A, overlay, bad)


def test_m6_frozen_record_allows_restating_a_default(tmp_path, overlay):
    restate = write_yaml(
        tmp_path / "r.yaml",
        {"search_10ghz": {"budget_k": 12, "tuned_eps_k": 0.1}, "stats": {"bootstrap_b": 10000}},
    )
    cfg = load_config(EXP_A, overlay, restate)
    assert cfg.search_10ghz.budget_k == 12 and cfg.stats.bootstrap_b == 10000


def test_m6_frozen_record_rejects_unknown_key(tmp_path, overlay):
    bad = write_yaml(tmp_path / "u.yaml", {"exp_c": {"bogus": 1}})
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(EXP_A, overlay, bad)


def test_m6_frozen_record_rejects_bool_as_int(tmp_path, overlay):
    """YAML has true/false, so `1` for a frozen bool is a typo, not a restatement."""
    bad = write_yaml(tmp_path / "bl.yaml", {"exp_b": {"reuse_exp_a_search_space": 1}})
    with pytest.raises(ConfigError, match="frozen protocol constant"):
        load_config(EXP_A, overlay, bad)


def test_m6_frozen_record_rejects_changed_tuple(tmp_path, overlay):
    bad = write_yaml(tmp_path / "t.yaml", {"search_10ghz": {"tiling": ["T1", "T2"]}})
    with pytest.raises(ConfigError, match="frozen protocol constant"):
        load_config(EXP_A, overlay, bad)


def test_m6_baseline_both_band_defaults_pinned():
    """Literal-pin every prediction-affecting BaselineConfig constant, both bands (T-C6-baseline)."""
    b = BaselineConfig()
    # Shared architecture / optimizer.
    assert b.cnn_channels == (16, 32, 64) and b.cnn_kernel == 7 and b.cnn_pool == 4
    assert b.cnn2d_channels == (16, 32) and b.cnn2d_kernel == 3 and b.cnn2d_pool == 2
    assert (b.optimizer, b.lr, b.loss) == ("adam", 1e-3, "mse")
    assert b.adam_betas == (0.9, 0.999)
    assert (b.batch_size, b.max_epochs) == (16, 200)
    assert (b.early_stopping_patience, b.early_stopping_min_delta) == (15, 1e-4)
    assert b.frame_to_session_aggregation == "median"
    assert b.raw_matched_standardize == "robust_per_channel"
    assert b.spectrogram_standardize == "train_only_per_frequency_mean_std"
    assert (b.spectrogram_hann, b.spectrogram_hop, b.spectrogram_nfft) == (64, 16, 128)
    # 10 GHz physics.
    assert b.physics_target_range_m_10ghz == (0.9, 1.5)
    assert b.physics_background_range_m_10ghz == (1.5, 3.0)
    # 77 GHz raw/matched/spectrogram (A-M6-2): raw keeps the chirp axis, matched is one Rx.
    assert b.raw_reduction_77ghz == "mean_over_fast_time_and_rx" and b.raw_channels_77ghz == 1
    assert b.matched_input_77ghz == "chain_steps_1_5_single_rx_range_bin_mean"
    assert b.matched_reference_rx_index_77ghz == 0 and b.matched_channels_77ghz == 2
    assert b.spectrogram_primary_channels_77ghz == 1
    assert b.spectrogram_ablation_channels_77ghz == 2
    # 77 GHz physics: DC bin vs any resolvable motion (a 2 Hz cutoff is unrepresentable).
    assert b.physics_static_band_bins_77ghz == (0, 0)
    assert b.physics_motion_band_bins_77ghz == (1, 127)
    assert b.physics_prf_hz_77ghz == 1953.125


def test_m6_exp_c_records_frank_hall_not_ordered_model():
    """A-M6-5: family (b) is Frank-Hall, chosen after OrderedModel lacked sample_weight."""
    c = ExpCConfig()
    assert c.proportional_odds_impl == "frank_hall_ordinal_decomposition_sklearn_logisticregression"
    assert c.cutpoint_source == "family_a_regressor_in_sample_predictions_inner_train"
    assert c.cutpoint_quantiles == (0.2, 0.4, 0.6, 0.8)
    assert c.class_weight_formula == "inverse_frequency_inner_train"
    assert c.class_weight_unsupported_families == ("knn",)
    assert c.proportional_odds_c_grid == (0.1, 1.0, 10.0)
    assert (c.selection_metric_primary, c.selection_metric_secondary) == ("class_unit_mae", "qwk")


def test_m6_sklearn_logistic_regression_supports_sample_weight():
    """The capability that justifies the Frank-Hall choice (A-M6-5): sklearn's
    LogisticRegression really does accept per-sample weights, unlike statsmodels'
    OrderedModel. A cheap smoke fit documents the verified premise as an assertion."""
    import inspect

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    assert "sample_weight" in inspect.signature(LogisticRegression.fit).parameters
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    w = np.array([1.0, 3.0, 3.0, 1.0])  # the inverse-frequency-style weights Exp C needs
    LogisticRegression().fit(X, y, sample_weight=w)  # must not raise


def test_m6_exp_e_interpretability_config_is_concrete():
    e = ExpEConfig()
    assert (e.reduction_10ghz, e.channel_10ghz, e.tiling_10ghz, e.log_10ghz) == (
        "A", "mag", "T1", "off")
    assert e.gate_10ghz_m == (1.0, 2.0)
    assert (e.tiling_77ghz, e.log_77ghz) == ("T1_77", "off")
    assert (e.model, e.ridge_alpha, e.n_folds) == ("ridge", 1.0, 4)
    assert e.fold_assignment == "sorted_subject_id_array_split"


def test_m6_exp_b_f_g_defaults_pinned():
    assert ExpBConfig() == ExpBConfig(
        reuse_exp_a_search_space=True,
        objective="equal_session_residual_mae",
        session_specific_variant_enabled=True,
    )
    f = ExpFConfig()
    assert f.radar_representation_rule == "exp_a_selected_feature_config_per_fold"
    assert f.covariates_primary == ("age", "height", "baseline_mass", "bmi")
    assert f.covariates_sensitivity == ("age", "height")
    assert f.target_sensitivity == "signed_kg_change"
    g = ExpGConfig()
    assert g.alpha_grid == tuple(round(0.05 * i, 2) for i in range(21))
    assert g.alpha_grid[0] == 0.0 and g.alpha_grid[-1] == 1.0 and len(g.alpha_grid) == 21
    assert g.alpha_tie_break == "closest_to_one" and g.seed_pairing is True


def test_m6_stats_covers_the_full_protocol():
    """StatsConfig transcribes §Statistics; pin the fields code will consume (C6-20)."""
    s = StatsConfig()
    assert (s.confidence_level, s.bootstrap_b, s.ci_method, s.ci_fallback) == (
        0.95, 10000, "bca", "percentile")
    assert s.resample_unit == "subject"
    assert s.undefined_metric_skip_threshold_pct == 5.0
    assert s.per_subject_pearson_r_min_sessions == 3
    assert (s.holm_family_expb_per_session, s.holm_family_expf_primary) == (4, 2)
    # Exp F's exploratory covariate contrasts get NO invented Holm family (C6-38).
    assert s.expf_exploratory_correction == "none_reported_individually"
    assert (s.robustness_replicates_r, s.robustness_min_distinct_subjects) == (200, 4)
    assert s.robustness_min_successful_replicates == 100 and s.robustness_ordinal_min_classes == 5


def test_m6_protocol_freeze_constants_pinned():
    p = ProtocolFreezeConfig()
    assert (p.option_b_peak_neighbors, p.option_b_mask_taper) == (1, True)
    assert p.fft_gate_transition_hz == 500.0
    assert p.qc77_min_in_band_energy_ratio == 0.30  # Step 0 kept it frozen


def test_m6_model_grids_fit_under_budget_k():
    """Every family's model-hyperparameter combination count is <= budget_k (T-C6-budget)."""
    g = ModelGridConfig()
    k = SearchSpace10GHzConfig().budget_k
    counts = {
        "ridge": len(g.ridge_alphas),
        "svr": len(g.svr_c) * len(g.svr_epsilon),
        "rf": len(g.rf_n_estimators) * len(g.rf_max_depth),
        "gbm": len(g.gbm_n_estimators) * len(g.gbm_learning_rate) * len(g.gbm_max_depth),
        "knn": len(g.knn_n_neighbors),
        "baseline": len(g.baseline_learning_rate) * len(g.baseline_weight_decay),
    }
    assert all(c <= k for c in counts.values()), counts
    assert SearchSpace77GHzConfig().budget_k == k  # both bands share the cap


def test_m6_sections_round_trip_through_config_to_dict(overlay):
    """Every M6 section survives config_to_dict as JSON-serializable data (C6-20)."""
    import json

    payload = config_to_dict(load_config(EXP_A, overlay))
    json.dumps(payload)  # must not raise
    for name, cls in M6_SECTIONS.items():
        assert payload[name] == config_to_dict(cls()), name


def test_m6_committed_config_files_all_load(overlay):
    """Each committed M6 config file composes cleanly onto the base config."""
    configs = REPO_ROOT / "configs"
    for name in (
        "search_10ghz", "search_77ghz", "baselines", "stats",
        "exp_b", "exp_c", "exp_e", "exp_f", "exp_g_fusion", "protocol_freeze",
    ):
        cfg = load_config(EXP_A, configs / f"{name}.yaml", overlay)
        assert isinstance(cfg, Config)


# -------------------------------------------------------------------------- realdata


@pytest.mark.realdata
def test_canonical_config_points_at_the_real_data(real_data_paths):
    """The committed run config really does resolve to the real dataset."""
    cfg = load_config(EXP_A)
    assert cfg.paths.data_10ghz_dir == real_data_paths["data_10ghz_dir"]
    assert cfg.paths.weight_xlsx == real_data_paths["weight_xlsx"]


@pytest.mark.realdata
def test_canonical_77ghz_config_points_at_the_real_data(real_data_77_paths):
    """The committed 77 GHz entry config resolves to the real 77 GHz cohort."""
    cfg = load_config(EXP_77)
    assert require_77ghz_dir(cfg) == real_data_77_paths["data_77ghz_dir"]
