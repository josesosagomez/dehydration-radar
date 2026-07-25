"""Config loading and validation.

One validated, frozen config object per run; it is also exactly what provenance.py
records. Plain dataclasses (no pydantic) so the schema is readable in one screen.

Two path rules, deliberately different because they answer different questions:

  * `include:` entries resolve against the DECLARING YAML's own directory — ordinary
    "import a sibling file" semantics, so configs/exp_a_regression.yaml can say
    `include: [data.yaml, ...]` and find configs/data.yaml.
  * Path VALUES (data_10ghz_dir, weight_xlsx, results_dir) resolve against the
    REPOSITORY ROOT, so a data root means the same thing regardless of which file
    declared it or where the run was launched from.

Merge order is "later wins", with lists/scalars replaced wholesale (never
concatenated): a later config states the entire intended value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID_DEVICES = ("cpu", "cuda")

# Band-gate implementations and standardization methods. The first of each is the
# primary; the second is a PRE-DECLARED ABLATION (never an inner-CV candidate).
GATE_METHODS = ("butterworth", "fft")
STANDARDIZE_METHODS = ("robust", "meanstd")
# kymatio frontends. numpy is primary/canonical; torch is validated by the M4
# cross-backend equivalence test and used only for unreported feature work.
BACKENDS = ("numpy", "torch")

# Exact by definition (SI), so it is written here rather than pulled from scipy —
# config validation should not depend on a numerics package being importable.
SPEED_OF_LIGHT_M_S = 299_792_458.0


class ConfigError(ValueError):
    """Raised for any malformed or invalid configuration."""


# --------------------------------------------------------------------------- schema


@dataclass(frozen=True)
class PathsConfig:
    data_10ghz_dir: Path
    weight_xlsx: Path
    results_dir: Path
    # Optional so existing 10 GHz-only configs load unchanged. Required (and existence-
    # checked) only for the 77 GHz entrypoints, via require_77ghz_dir(config).
    data_77ghz_dir: Path | None = None


@dataclass(frozen=True)
class RunConfig:
    seed: int
    seed_set: tuple[int, ...]
    device: str


@dataclass(frozen=True)
class SplitConfig:
    n_inner_max: int = 5
    min_train_subjects: int = 3


@dataclass(frozen=True)
class QCConfig:
    """Frozen QC screen thresholds (implementation_plan.md 'QC screens & thresholds').

    Used from M2. Pinned here because they are frozen constants of the design, not
    tunable parameters — if one ever becomes data-adaptive it moves inside inner CV.
    """

    histogram_bins: int = 200
    flatline_max_bin_fraction: float = 0.25
    min_in_band_energy_ratio: float = 0.30
    rms_robust_z_threshold: float = 4.5
    # QC uses ONE fixed (wider) gate for all candidates so the QC-passing population
    # never varies with the model gate chosen later in inner CV.
    qc_gate_m: tuple[float, float] = (0.9, 3.0)
    # The reference's BandMarginHz default. At df = fs/534 ~ 975.3 Hz this is ~1 FFT
    # bin, so a target sitting at a band edge is not rejected for Hann leakage alone;
    # the 0.30 ratio threshold was defined together with this margin.
    in_band_margin_hz: float = 1000.0
    min_frame_fraction: float = 0.5


@dataclass(frozen=True)
class PreprocessConfig:
    """Frozen preprocessing parameters (implementation_plan.md 'Preprocessing').

    The fields fall into three classes, fixed before M6 so no alternative can quietly
    become an undeclared search axis (MILESTONE_3_PLAN.md §0):

      * inner-CV search axis: `model_gate_m` (1-2 m default vs the 0.9-3.0 m
        candidate), selected per outer fold on inner folds only;
      * ablation switches: `gate_method`, `standardize` -- their non-default values
        are PRE-DECLARED ABLATIONS, never inner-CV candidates and never able to
        displace the primary path;
      * frozen protocol constants: `butter_order`, `edge_trim`, `peak_neighbors`,
        `mask_taper`, `fft_gate_transition_hz` and the radar constants. They are
        configurable only so a run's YAML is a complete record and so tests can drive
        boundary behaviour; non-default values are rejected by modelling/artifact
        entrypoints (at M3 by run_preprocess.py's canonical-spec guard).

    Reduction {A, B} and channel {mag, iq} are the other two inner-CV axes; they are
    call arguments rather than config, so one config can serve every variant.
    """

    butter_order: int = 4
    model_gate_m: tuple[float, float] = (1.0, 2.0)
    edge_trim: int = 32
    fs_hz: float = 520834.0
    bandwidth_hz: float = 500e6
    chirp_time_s: float = 1024e-6
    # Band gate: the time-domain zero-phase Butterworth is primary; the FFT tapered
    # mask (filter_gpt_fft.m) is the pre-declared ablation.
    gate_method: str = "butterworth"
    fft_gate_transition_hz: float = 500.0
    # Option B: "+/-1-bin two-sided Hann-tapered mask" IS peak_neighbors=1 with
    # mask_taper=True -- the two together are the frozen form.
    peak_neighbors: int = 1
    mask_taper: bool = True
    # Robust median/MAD z is primary; plain mean/std is the pre-declared ablation.
    standardize: str = "robust"


@dataclass(frozen=True)
class WSTTiling:
    q: tuple[int, int]
    invariance_ms: float


@dataclass(frozen=True)
class WSTConfig:
    """Frozen WST tilings (implementation_plan.md 'WST parameterization').

    Consumed from M4. `backend` selects the kymatio frontend and is an implementation
    choice validated by the cross-backend equivalence test — never a search axis or
    ablation; **numpy is the canonical backend for every reported WST feature** (a torch
    frontend may back only unreported feature work, and only after the cross-backend
    check passes). J, T and the output shape are DERIVED and MEASURED from the
    instantiated filter bank at build time (features/wst.py), never precomputed here.
    """

    tilings: tuple[WSTTiling, ...] = (
        WSTTiling(q=(10, 4), invariance_ms=0.20),
        WSTTiling(q=(8, 2), invariance_ms=0.30),
        WSTTiling(q=(6, 2), invariance_ms=0.40),
    )
    max_order: int = 2
    log_epsilon: float = 1e-6
    backend: str = "numpy"


# --------------------------------------------------------------- 77 GHz (band 2)
# The 77 GHz arm (milestone 5) is DIFFERENT PHYSICS, not overrides of the 10 GHz
# defaults, so it gets its own three frozen dataclasses and three parallel top-level
# config sections (qc77 / preprocess77 / wst77) — never a nested band block. Each band
# then has its own canonical spec, so canonical_spec_guard_77 can compare against
# Preprocess77Config()/QC77Config()/WST77Config() exactly as the 10 GHz guard does.
# All values are the audited reference constants (matlab/77ghz_code/*, the M2 audit).


@dataclass(frozen=True)
class Preprocess77Config:
    """Frozen 77 GHz preprocessing parameters (implementation_plan.md Exp G chain).

    ONE range gate (2-4 m) serves both the chain crop and the QC in-band mask — the
    Exp G spec freezes a single gate for both, so a separate qc77 gate would only be a
    drift channel. No model_gate_m/edge_trim/peak_neighbors: the Doppler slow-time chain
    has no reduction stage and no gate search (that would be a plan amendment, not a
    config edit). `standardize` selects the per-channel robust z applied to each
    slow-time series before WST (A-M5-3); robust is primary, meanstd the ablation.
    """

    butter_order: int = 4
    gate_m: tuple[float, float] = (2.0, 4.0)
    fs_hz: float = 500e3
    bandwidth_hz: float = 2e9
    chirp_time_s: float = 512e-6  # PRF = 1/chirp_time_s = 1953.125 Hz (derived)
    standardize: str = "robust"


@dataclass(frozen=True)
class QC77Config:
    """Frozen 77 GHz QC screen thresholds (implementation_plan.md Exp G QC).

    Exactly three screens (NaN/Inf, flatline, in-band): no RMS robust-z diagnostic
    (that is a 10 GHz-only diagnostic, not specified for 77 GHz). The gate itself is NOT
    here — it lives once in Preprocess77Config.gate_m and screens_77 reads it there.

    Milestone-5 step-6 flatline rule (owner outcome (b), mechanism-corrected):
    `flatline_skip_leading_bins = 1` excludes fast-time index 0 (range bin 0) from the
    flatline screen. The M5 mechanism analysis (HISTORY 2026-07-23) found range bin 0 of
    every (Rx, chirp) trace carries an EMBEDDED FRAME COUNTER (value ~256*frame, per-chirp
    increment, universal across files, ~20-90x the ~27 echo), NOT echo. That single
    outlier stretched the per-trace [min,max] histogram range so the ~255 genuine samples
    piled into the first bins and false-tripped the >=25% concentration rule (M2's 7/10
    'flatline'). Both Hann windows zero fast[0] and the gate crop (bins 27..53) excludes
    bin 0, so the counter never reaches the WST features — it corrupted only this
    raw-magnitude screen. Excluding it restores the proven 128-bin/0.25 rule on the echo
    samples; a genuinely dead/constant channel is still flagged (degenerate spread).
    """

    histogram_bins: int = 128
    flatline_max_bin_fraction: float = 0.25  # any bin >= 25% of the screened samples flags a trace
    flatline_skip_leading_bins: int = 1  # exclude range bin 0 (embedded frame counter); M5 step 6b
    min_in_band_energy_ratio: float = 0.30
    in_band_margin_hz: float = 1953.125  # one FFT bin = fs/256, frozen a priori
    min_frame_fraction: float = 0.5


@dataclass(frozen=True)
class WST77Config:
    """Frozen 77 GHz WST tilings (implementation_plan.md Exp G / wst_extract77.m).

    Doppler tilings at fs = PRF = 1953.125 Hz. `tilings` is code-frozen (YAML override
    rejected) exactly like the 10 GHz WSTConfig; J, T and the output shape are DERIVED
    and MEASURED from the instantiated bank at build time, never precomputed here.
    numpy is the canonical backend for every reported 77 GHz feature.
    """

    tilings: tuple[WSTTiling, ...] = (
        WSTTiling(q=(8, 4), invariance_ms=20.0),
        WSTTiling(q=(6, 4), invariance_ms=40.0),
        WSTTiling(q=(4, 2), invariance_ms=60.0),
    )
    max_order: int = 2
    log_epsilon: float = 1e-6
    backend: str = "numpy"


# ---------------------------------------------- milestone 6 (the config-freeze gate)
# Every section below is a FROZEN RECORD of an owner-approved protocol decision
# (plans/MILESTONE_6_PLAN.md, Step 0 resolved 2026-07-25). None is a live tuning knob:
# a run's YAML may restate a scalar default (a complete record) but tuple-valued design
# constants are code-frozen like wst.tilings, and modelling entrypoints additionally
# validate every value at its frozen default via protocol_freeze_guard (features/
# protocol_freeze.py). The dataclasses exist so the search-space whitelist, the baseline
# specs, and the statistical protocol are one validated, provenance-recorded object
# rather than prose the M7 harness has to re-derive.

# Enumerated string domains, so a YAML typo fails at load rather than deep in M7.
REDUCTIONS = ("A", "B")
CHANNELS = ("mag", "iq")
TILINGS_10GHZ = ("T1", "T2", "T3")
TILINGS_77GHZ = ("T1_77", "T2_77", "T3_77")
LOG_BRANCHES = ("off", "on_frozen_eps", "on_tuned_eps")
MODEL_FAMILIES = ("ridge", "svr", "rf", "gbm", "knn")


@dataclass(frozen=True)
class SearchSpace10GHzConfig:
    """The frozen Exp A/WST inner-CV search space for 10 GHz (implementation_plan.md
    §"LOSO harness"). The tuple fields ARE the whitelist protocol_freeze_guard checks a
    call-time protocol record against; they are code-frozen (YAML override rejected).
    The log axis carries all three branches unconditionally (A-M6-1, owner-approved)."""

    reduction: tuple[str, ...] = REDUCTIONS
    channel: tuple[str, ...] = CHANNELS
    tiling: tuple[str, ...] = TILINGS_10GHZ
    log_branches: tuple[str, ...] = LOG_BRANCHES
    range_gate_m: tuple[tuple[float, float], ...] = ((1.0, 2.0), (0.9, 3.0))
    model_families: tuple[str, ...] = MODEL_FAMILIES
    budget_k: int = 12
    stage1_anchor_model: str = "ridge"
    stage1_anchor_ridge_alpha: float = 1.0
    tuned_eps_k: float = 0.1


@dataclass(frozen=True)
class SearchSpace77GHzConfig:
    """The 77 GHz search space. reduction/channel/gate are FIXED (Exp G freezes the
    primary slow-time I/Q chain and the 2-4 m gate) — scalars, not candidate sets, so a
    10 GHz-only candidate (a second reduction or a model gate) cannot be expressed here.
    Only tiling and the log branch are inner-CV axes."""

    reduction: str = "slow_time_iq_primary"
    channel: str = "iq"
    gate_m: tuple[float, float] = (2.0, 4.0)
    tiling: tuple[str, ...] = TILINGS_77GHZ
    log_branches: tuple[str, ...] = LOG_BRANCHES
    model_families: tuple[str, ...] = MODEL_FAMILIES
    budget_k: int = 12
    stage1_anchor_model: str = "ridge"
    stage1_anchor_ridge_alpha: float = 1.0
    tuned_eps_k: float = 0.1


@dataclass(frozen=True)
class ModelGridConfig:
    """Per-family hyperparameter grids, enumerated so budget_k is a literal len() check
    (T-C6-budget). Each family's total combination count is <= budget_k = 12; the
    baseline learning-rate/weight-decay grid gives the budget-parity rule a real
    candidate set to count on the baseline side (owner-approved, Step 0 item 3)."""

    ridge_alphas: tuple[float, ...] = (0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)  # 8
    svr_c: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    svr_epsilon: tuple[float, ...] = (0.01, 0.1, 0.3)  # 4 x 3 = 12
    rf_n_estimators: tuple[int, ...] = (100, 300)
    rf_max_depth: tuple[int | None, ...] = (3, 5, None)  # 2 x 3 = 6
    gbm_n_estimators: tuple[int, ...] = (100, 300)
    gbm_learning_rate: tuple[float, ...] = (0.01, 0.1)
    gbm_max_depth: tuple[int, ...] = (2, 3)  # 2 x 2 x 2 = 8
    knn_n_neighbors: tuple[int, ...] = (3, 5, 7, 9, 11, 13, 15)  # 7
    baseline_learning_rate: tuple[float, ...] = (3e-4, 1e-3, 3e-3)
    baseline_weight_decay: tuple[float, ...] = (0.0, 1e-4)  # 3 x 2 = 6


@dataclass(frozen=True)
class BaselineConfig:
    """The frozen Exp D baseline specs, both bands (implementation_plan.md §D + A-M6-2).
    Architecture/optimizer constants are shared; band-specific fields carry the 10 GHz
    range-power split and the 77 GHz axis-consistent raw/matched/spectrogram tensor
    definitions and the DC-vs-any-motion Doppler physics split (A-M6-2, owner-approved)."""

    # Shared CNN / 2D-CNN architecture.
    cnn_channels: tuple[int, int, int] = (16, 32, 64)
    cnn_kernel: int = 7
    cnn_pool: int = 4
    cnn2d_channels: tuple[int, int] = (16, 32)
    cnn2d_kernel: int = 3
    cnn2d_pool: int = 2
    # Optimizer / training (owner-confirmed provisional defaults, §3).
    optimizer: str = "adam"
    lr: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_weight_decay_default: float = 0.0
    weight_init: str = "framework_default"
    loss: str = "mse"
    batch_size: int = 16
    max_epochs: int = 200
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    checkpoint_metric: str = "inner_val_session_mae"
    checkpoint_direction: str = "minimize"
    frame_to_session_aggregation: str = "median"
    # Two distinct normalization rules (implementation_plan.md "Fit-on-train-only").
    raw_matched_standardize: str = "robust_per_channel"
    spectrogram_standardize: str = "train_only_per_frequency_mean_std"
    spectrogram_hann: int = 64
    spectrogram_hop: int = 16
    spectrogram_nfft: int = 128
    # 10 GHz physics baseline (range-power ratio).
    physics_target_range_m_10ghz: tuple[float, float] = (0.9, 1.5)
    physics_background_range_m_10ghz: tuple[float, float] = (1.5, 3.0)
    # 77 GHz raw/matched/spectrogram tensors (A-M6-2). The raw reduction retains the
    # chirp/slow-time axis; the matched input is one fixed Rx, never coherently fused.
    raw_reduction_77ghz: str = "mean_over_fast_time_and_rx"
    raw_channels_77ghz: int = 1
    matched_input_77ghz: str = "chain_steps_1_5_single_rx_range_bin_mean"
    matched_reference_rx_index_77ghz: int = 0
    matched_channels_77ghz: int = 2
    spectrogram_primary_channels_77ghz: int = 1
    spectrogram_ablation_channels_77ghz: int = 2
    # 77 GHz Doppler physics baseline: DC bin vs any resolvable motion. A rate-specific
    # cutoff is unrepresentable (df = PRF/256 ~ 7.63 Hz), so this is a bin-partition.
    physics_static_band_bins_77ghz: tuple[int, int] = (0, 0)
    physics_motion_band_bins_77ghz: tuple[int, int] = (1, 127)
    physics_prf_hz_77ghz: float = 1953.125


@dataclass(frozen=True)
class ExpBConfig:
    """Clock-decoupling (implementation_plan.md §B, A-M6-3). Reuses Exp A's search space
    under Exp B's own equal-session residual-MAE objective."""

    reuse_exp_a_search_space: bool = True
    objective: str = "equal_session_residual_mae"
    session_specific_variant_enabled: bool = True


@dataclass(frozen=True)
class ExpCConfig:
    """Ordinal 5-class (implementation_plan.md §C). Family (a) thresholds family (a)'s
    OWN in-sample inner-training predictions at fixed quantiles; family (b) is the
    Frank-Hall ordinal decomposition (A-M6-5, owner-approved after statsmodels
    OrderedModel was verified to lack sample_weight support). Both consume train-only
    inverse-frequency class weights; KNN is excluded from weighted fitting."""

    cutpoint_source: str = "family_a_regressor_in_sample_predictions_inner_train"
    cutpoint_quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    cutpoint_min_separation: float = 1e-9
    class_weight_formula: str = "inverse_frequency_inner_train"
    class_weight_unsupported_families: tuple[str, ...] = ("knn",)
    proportional_odds_impl: str = "frank_hall_ordinal_decomposition_sklearn_logisticregression"
    proportional_odds_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    selection_metric_primary: str = "class_unit_mae"
    selection_metric_secondary: str = "qwk"


@dataclass(frozen=True)
class ExpEConfig:
    """The single pre-registered interpretability configuration (implementation_plan.md
    §E), fixed a priori and non-performance-based so importances are comparable."""

    reduction_10ghz: str = "A"
    channel_10ghz: str = "mag"
    tiling_10ghz: str = "T1"
    log_10ghz: str = "off"
    gate_10ghz_m: tuple[float, float] = (1.0, 2.0)
    tiling_77ghz: str = "T1_77"
    log_77ghz: str = "off"
    model: str = "ridge"
    ridge_alpha: float = 1.0
    n_folds: int = 4
    fold_assignment: str = "sorted_subject_id_array_split"


@dataclass(frozen=True)
class ExpFConfig:
    """Confound check (implementation_plan.md §F, A-M6-4). Four nested ridge models on
    one fixed clock encoding; models 3 and 4 reuse that fold's Exp A-selected feature
    configuration (refit with ridge)."""

    radar_representation_rule: str = "exp_a_selected_feature_config_per_fold"
    ridge_alphas: tuple[float, ...] = (0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    clock_encoding: str = "session_index_one_hot"
    covariates_primary: tuple[str, ...] = ("age", "height", "baseline_mass", "bmi")
    covariates_sensitivity: tuple[str, ...] = ("age", "height")
    target_sensitivity: str = "signed_kg_change"


@dataclass(frozen=True)
class ExpGConfig:
    """Cross-band fusion combiner (implementation_plan.md §G). Records the already-frozen
    convex-weight selection rule as config; the alpha value itself is only fit in M7's CV."""

    alpha_grid: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(21))
    alpha_tie_break: str = "closest_to_one"
    seed_pairing: bool = True
    objective: str = "subject_balanced_oof_mae"


@dataclass(frozen=True)
class StatsConfig:
    """The full statistical protocol (implementation_plan.md §Statistics), transcribed
    into validated config — no redesign, every field names a choice that section makes."""

    confidence_level: float = 0.95
    bootstrap_b: int = 10000
    ci_method: str = "bca"
    ci_fallback: str = "percentile"
    resample_unit: str = "subject"
    seed_collapse_additive: str = "average_per_subject_before_resample"
    seed_collapse_pooled: str = "average_metric_across_seeds_within_resample"
    undefined_metric_skip_threshold_pct: float = 5.0
    per_subject_pearson_r_min_sessions: int = 3
    expb_aggregate_estimand: str = "session_weighted_equal_weight_per_session"
    expb_paired_test_estimand: str = "subject_weighted_complete_case_s1_s4"
    paired_test: str = "wilcoxon_signed_rank"
    holm_family_expb_per_session: int = 4
    holm_family_expf_primary: int = 2
    # Exp F's two covariate contrasts are "exploratory" per implementation_plan.md §F,
    # which names NO Holm family for them — so they are reported individually, uncorrected.
    expf_exploratory_correction: str = "none_reported_individually"
    holm_family_baseline_per_family: int = 3
    composite_baseline_comparison: str = "single_uncorrected"
    robustness_replicates_r: int = 200
    robustness_min_distinct_subjects: int = 4
    robustness_min_successful_replicates: int = 100
    robustness_ordinal_min_classes: int = 5


@dataclass(frozen=True)
class ProtocolFreezeConfig:
    """The frozen protocol constants a modelling entrypoint validates (beyond the WST
    feature guards): the Option-B mask form, the FFT-gate transition, and the 77 GHz
    in-band QC threshold that Step 0 kept frozen at 0.30."""

    option_b_peak_neighbors: int = 1
    option_b_mask_taper: bool = True
    fft_gate_transition_hz: float = 500.0
    qc77_min_in_band_energy_ratio: float = 0.30


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    run: RunConfig
    split: SplitConfig = field(default_factory=SplitConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    wst: WSTConfig = field(default_factory=WSTConfig)
    # 77 GHz (band 2), milestone 5. default_factory so every existing 10 GHz config
    # loads unchanged: the new sections simply take defaults and appear in provenance.
    qc77: QC77Config = field(default_factory=QC77Config)
    preprocess77: Preprocess77Config = field(default_factory=Preprocess77Config)
    wst77: WST77Config = field(default_factory=WST77Config)
    # Milestone 6 (the config-freeze gate). Same additive default_factory pattern: an
    # existing config loads unchanged; the frozen protocol sections appear at their
    # owner-approved defaults and are recorded in provenance.
    search_10ghz: SearchSpace10GHzConfig = field(default_factory=SearchSpace10GHzConfig)
    search_77ghz: SearchSpace77GHzConfig = field(default_factory=SearchSpace77GHzConfig)
    model_grid: ModelGridConfig = field(default_factory=ModelGridConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    exp_b: ExpBConfig = field(default_factory=ExpBConfig)
    exp_c: ExpCConfig = field(default_factory=ExpCConfig)
    exp_e: ExpEConfig = field(default_factory=ExpEConfig)
    exp_f: ExpFConfig = field(default_factory=ExpFConfig)
    exp_g: ExpGConfig = field(default_factory=ExpGConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    protocol_freeze: ProtocolFreezeConfig = field(default_factory=ProtocolFreezeConfig)


# ---------------------------------------------------------------------- yaml loading


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a mapping at top level: {path}")
    return data


def _merge(base: dict, overlay: dict) -> dict:
    """Recursive merge; scalars and lists are REPLACED wholesale, never concatenated."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_with_includes(path: Path, *, is_included: bool = False) -> dict:
    """Load one YAML, merging its `include:` files first (declaring file wins)."""
    raw = _read_yaml(path)
    includes = raw.pop("include", [])

    if includes and is_included:
        raise ConfigError(
            f"nested include is not allowed (in {path}); composition is kept flat so "
            "the resolution order stays followable"
        )
    if not isinstance(includes, list):
        raise ConfigError(f"'include' must be a list of paths (in {path})")

    merged: dict = {}
    for entry in includes:
        # Include paths resolve against the DECLARING file's directory.
        merged = _merge(merged, _load_with_includes(path.parent / entry, is_included=True))
    return _merge(merged, raw)


# ------------------------------------------------------------------------ validation


def _resolve_path(value, key: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"paths.{key} must be a string path, got {type(value).__name__}")
    path = Path(value)
    # Path VALUES resolve against the repo root, regardless of CWD or declaring file.
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_keys(section: dict, required: tuple[str, ...], name: str) -> None:
    missing = sorted(set(required) - set(section))
    if missing:
        raise ConfigError(f"{name} is missing required key(s): {', '.join(missing)}")


def _reject_unknown(section: dict, allowed: tuple[str, ...], name: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{name} has unknown key(s): {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(allowed))})"
        )


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    return value


# --------------------------------------------------------------- field validators
# These exist because M2 actually *consumes* the QC/preprocess numbers: a bad value
# would otherwise surface as a confusing numpy error deep inside a screen, or worse,
# as a screen that silently never fires.


def _number(value, name: str, *, low: float, high: float, low_open=False, high_open=False):
    """A finite number inside the stated interval. bool is rejected (it is an int)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite, got {value}")
    if (value <= low if low_open else value < low) or (
        value >= high if high_open else value > high
    ):
        interval = f"{'(' if low_open else '['}{low}, {high}{')' if high_open else ']'}"
        raise ConfigError(f"{name} must be in {interval}, got {value}")
    return value


def _float_field(section: dict, key: str, default: float, name: str, **bounds) -> float:
    return _number(section.get(key, default), f"{name}.{key}", **bounds)


def _int_field(section: dict, key: str, default: int, name: str, *, minimum: int = 1) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name}.{key} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ConfigError(f"{name}.{key} must be >= {minimum}, got {value}")
    return value


def _bool_field(section: dict, key: str, default: bool, name: str) -> bool:
    """A real boolean. 0/1 are rejected: YAML has true/false, so an int here is a typo."""
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{name}.{key} must be true or false, got {type(value).__name__}")
    return value


def _choice_field(section: dict, key: str, default: str, name: str, allowed: tuple[str, ...]) -> str:
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{name}.{key} must be a string, got {type(value).__name__}")
    if value not in allowed:
        raise ConfigError(f"{name}.{key} must be one of {allowed}, got {value!r}")
    return value


def _gate_field(section: dict, key: str, default, name: str) -> tuple[float, float]:
    """A [min_m, max_m] range gate -> tuple.

    YAML hands us a list; the frozen dataclass must not carry a mutable value (and
    provenance should record the same type every run).
    """
    value = section.get(key, default)
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigError(
            f"{name}.{key} must be a two-element [min_m, max_m] list, "
            f"got {type(value).__name__}"
        )
    if len(value) != 2:
        raise ConfigError(f"{name}.{key} must have exactly 2 entries, got {len(value)}")
    bounds = tuple(
        _number(v, f"{name}.{key}[{i}]", low=0.0, high=math.inf, low_open=True, high_open=True)
        for i, v in enumerate(value)
    )
    if bounds[0] >= bounds[1]:
        raise ConfigError(
            f"{name}.{key} must be strictly increasing, got [{bounds[0]}, {bounds[1]}]"
        )
    return bounds


def beat_band_hz(gate_m, bandwidth_hz: float, chirp_time_s: float) -> tuple[float, float]:
    """FMCW range gate (metres) -> beat-frequency band (Hz).

    A target at range r returns a beat tone at f = HzPerM * r, with
    HzPerM = 2 * slope / c and slope = B / Tchirp. Lives here because both the config
    cross-validation below and the QC in-band mask need it, and duplicating the
    physics is how the two drift apart.
    """
    hz_per_m = 2.0 * (bandwidth_hz / chirp_time_s) / SPEED_OF_LIGHT_M_S
    return (hz_per_m * gate_m[0], hz_per_m * gate_m[1])


def _check_qc_band(qc: QCConfig, preprocess: PreprocessConfig) -> None:
    """The frozen QC gate must map to a real, non-vacuous beat-frequency band.

    Positive-and-increasing metres is not enough. Two ways a syntactically valid gate
    still breaks the screen, both config errors rather than valid configurations,
    because a silently disabled QC screen is far worse than a loud failure:

      * the band maps entirely at/above Nyquist -> nothing to measure;
      * the margin widens it to cover the whole represented spectrum -> the ratio is
        identically 1 and the screen can never fire.

    (An "empty band after clamping" check would be dead code: the margin only widens,
    so lo <= f_lo < nyquist <= hi holds whenever the Nyquist check passes.) The
    bin-level guards -- at least one bin of support, and not *every* bin at the actual
    fast-time length -- live in qc.screens.in_band_mask.
    """
    f_lo, f_hi = beat_band_hz(
        qc.qc_gate_m, preprocess.bandwidth_hz, preprocess.chirp_time_s
    )
    nyquist = preprocess.fs_hz / 2.0
    if f_lo >= nyquist:
        raise ConfigError(
            f"qc.qc_gate_m {qc.qc_gate_m} m maps to {f_lo:.1f}-{f_hi:.1f} Hz, which "
            f"starts at or above Nyquist ({nyquist:.1f} Hz) — no representable band"
        )
    lo = max(0.0, f_lo - qc.in_band_margin_hz)
    hi = min(nyquist, f_hi + qc.in_band_margin_hz)
    if lo <= 0.0 and hi >= nyquist:
        raise ConfigError(
            f"qc gate {qc.qc_gate_m} m widened by margin {qc.in_band_margin_hz} Hz "
            f"covers the entire spectrum [0, {nyquist:.1f}] Hz — the in-band ratio "
            "would be identically 1 and the screen could never fire"
        )


def _check_model_band(qc: QCConfig, preprocess: PreprocessConfig) -> None:
    """The model gate must be filterable, inside the QC gate, and not a no-op.

    Three separate failures, all config errors rather than valid configurations:

      * **The whole band must sit below Nyquist** (0 < f_lo < f_hi < fs/2). This is
        deliberately STRICTER than `_check_qc_band`, which only rejects a band
        *starting* at/above Nyquist -- the QC screen is an FFT mask whose upper edge is
        legitimately Nyquist-clamped (frozen at M2), whereas `butter` raises on
        Wn >= 1, so a gate straddling Nyquist would pass config load and blow up deep
        inside the filter. Clamping the model band instead would make the two gate
        methods filter different bands under one config.
      * **model_gate_m must lie inside qc_gate_m.** The QC gate was frozen wider
        precisely so the QC-passing population is identical for every model-gate
        candidate (implementation_plan.md, "One fixed QC range gate"). A model gate
        reaching outside it would use energy QC never guaranteed.
      * **The FFT gate must not pass everything.** With skirts wide enough to cover the
        whole represented spectrum the "gate" is a no-op -- the same doctrine as the QC
        vacuity guards.
    """
    f_lo, f_hi = beat_band_hz(
        preprocess.model_gate_m, preprocess.bandwidth_hz, preprocess.chirp_time_s
    )
    nyquist = preprocess.fs_hz / 2.0
    if not (0.0 < f_lo < f_hi < nyquist):
        raise ConfigError(
            f"preprocess.model_gate_m {preprocess.model_gate_m} m maps to "
            f"{f_lo:.1f}-{f_hi:.1f} Hz, which is not strictly inside "
            f"(0, {nyquist:.1f}) Hz — the whole band must be below Nyquist "
            "(scipy.signal.butter raises on a normalized cutoff >= 1)"
        )

    if preprocess.model_gate_m[0] < qc.qc_gate_m[0] or preprocess.model_gate_m[1] > qc.qc_gate_m[1]:
        raise ConfigError(
            f"preprocess.model_gate_m {preprocess.model_gate_m} m is not contained in "
            f"qc.qc_gate_m {qc.qc_gate_m} m — QC fixed the frame population on the "
            "wider gate, so a model gate reaching outside it would use energy QC never "
            "screened for"
        )

    if preprocess.gate_method == "fft":
        transition = preprocess.fft_gate_transition_hz
        if f_lo - transition <= 0.0 and f_hi + transition >= nyquist:
            raise ConfigError(
                f"preprocess: the fft gate {preprocess.model_gate_m} m widened by "
                f"{transition} Hz skirts covers the entire spectrum [0, {nyquist:.1f}] Hz "
                "— it would pass everything and filter nothing"
            )


def _check_qc77_band(qc77: QC77Config, pre77: Preprocess77Config) -> None:
    """The frozen 77 GHz gate must map to a real, non-vacuous beat-frequency band.

    The exact analog of _check_qc_band for band 2: the single 2-4 m gate (in
    Preprocess77Config) mapped to beat frequency must start below Nyquist and, once
    widened by the margin, must not cover the whole represented spectrum (or the in-band
    ratio would be identically 1 and the screen could never fire). Reuses beat_band_hz so
    the physics is defined once.
    """
    f_lo, f_hi = beat_band_hz(pre77.gate_m, pre77.bandwidth_hz, pre77.chirp_time_s)
    nyquist = pre77.fs_hz / 2.0
    if f_lo >= nyquist:
        raise ConfigError(
            f"preprocess77.gate_m {pre77.gate_m} m maps to {f_lo:.1f}-{f_hi:.1f} Hz, "
            f"which starts at or above Nyquist ({nyquist:.1f} Hz) — no representable band"
        )
    lo = max(0.0, f_lo - qc77.in_band_margin_hz)
    hi = min(nyquist, f_hi + qc77.in_band_margin_hz)
    if lo <= 0.0 and hi >= nyquist:
        raise ConfigError(
            f"preprocess77.gate_m {pre77.gate_m} m widened by qc77 margin "
            f"{qc77.in_band_margin_hz} Hz covers the entire spectrum [0, {nyquist:.1f}] "
            "Hz — the in-band ratio would be identically 1 and the screen could never fire"
        )


def _build_paths(raw: dict) -> PathsConfig:
    section = _section(raw, "paths")
    required = ("data_10ghz_dir", "weight_xlsx", "results_dir")
    optional = ("data_77ghz_dir",)
    _require_keys(section, required, "paths")
    _reject_unknown(section, required + optional, "paths")

    resolved = {key: _resolve_path(section[key], key) for key in required}

    # Required INPUTS must exist. results_dir is an OUTPUT: it need only be creatable,
    # and its writers create it on demand.
    for key in ("data_10ghz_dir", "weight_xlsx"):
        if not resolved[key].exists():
            raise ConfigError(f"paths.{key} does not exist: {resolved[key]}")

    # data_77ghz_dir is optional (10 GHz configs omit it). When present it is an INPUT
    # and must exist; require_77ghz_dir raises later if a 77 GHz entrypoint needs it and
    # it was never set.
    if "data_77ghz_dir" in section:
        resolved["data_77ghz_dir"] = _resolve_path(section["data_77ghz_dir"], "data_77ghz_dir")
        if not resolved["data_77ghz_dir"].exists():
            raise ConfigError(
                f"paths.data_77ghz_dir does not exist: {resolved['data_77ghz_dir']}"
            )

    return PathsConfig(**resolved)


def require_77ghz_dir(config: Config) -> Path:
    """The 77 GHz data root, or a pointed ConfigError if it was never configured.

    Every 77 GHz entrypoint calls this before any I/O, so a config that forgot
    `paths.data_77ghz_dir` fails with a clear message instead of an AttributeError or a
    None reaching h5py.
    """
    data_dir = config.paths.data_77ghz_dir
    if data_dir is None:
        raise ConfigError(
            "paths.data_77ghz_dir is not set — a 77 GHz run needs it; add it via "
            "configs/data77.yaml (or an ibex.yaml overlay). See configs/exp_77ghz.yaml."
        )
    return data_dir


def _build_run(raw: dict) -> RunConfig:
    section = _section(raw, "run")
    required = ("seed", "seed_set", "device")
    _require_keys(section, required, "run")
    _reject_unknown(section, required, "run")

    seed = section["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("run.seed must be an integer")

    seed_set = section["seed_set"]
    if not isinstance(seed_set, list) or not all(
        isinstance(s, int) and not isinstance(s, bool) for s in seed_set
    ):
        raise ConfigError("run.seed_set must be a list of integers")
    if len(seed_set) != 5:
        raise ConfigError(
            f"run.seed_set must contain exactly 5 seeds, got {len(seed_set)} "
            "(the protocol fixes a 5-seed set for stochastic models)"
        )
    if len(set(seed_set)) != 5:
        raise ConfigError(
            f"run.seed_set must contain 5 DISTINCT seeds, got {seed_set} — "
            "duplicates would silently give fewer than 5 effective repeats"
        )

    device = section["device"]
    if device not in VALID_DEVICES:
        raise ConfigError(f"run.device must be one of {VALID_DEVICES}, got {device!r}")

    return RunConfig(seed=seed, seed_set=tuple(seed_set), device=device)


def _build_split(raw: dict) -> SplitConfig:
    section = _section(raw, "split")
    allowed = ("n_inner_max", "min_train_subjects")
    _reject_unknown(section, allowed, "split")

    defaults = SplitConfig()
    n_inner_max = section.get("n_inner_max", defaults.n_inner_max)
    min_train_subjects = section.get("min_train_subjects", defaults.min_train_subjects)

    for key, value in (
        ("n_inner_max", n_inner_max),
        ("min_train_subjects", min_train_subjects),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"split.{key} must be an integer")

    if n_inner_max < 2:
        raise ConfigError(f"split.n_inner_max must be >= 2 (GroupKFold), got {n_inner_max}")
    if min_train_subjects < 3:
        raise ConfigError(
            f"split.min_train_subjects must be >= 3, got {min_train_subjects} — "
            "the protocol requires >=3 training subjects before inner CV is selectable "
            "(implementation_plan.md, 'Inner loop'); this is stricter than GroupKFold's "
            "mechanical floor of 2 on purpose"
        )

    return SplitConfig(n_inner_max=n_inner_max, min_train_subjects=min_train_subjects)


def _known_section(raw: dict, name: str, cls) -> dict:
    section = _section(raw, name)
    _reject_unknown(section, tuple(f.name for f in fields(cls)), name)
    return section


def _build_qc(raw: dict) -> QCConfig:
    section = _known_section(raw, "qc", QCConfig)
    d = QCConfig()
    return QCConfig(
        histogram_bins=_int_field(section, "histogram_bins", d.histogram_bins, "qc"),
        flatline_max_bin_fraction=_float_field(
            section, "flatline_max_bin_fraction", d.flatline_max_bin_fraction, "qc",
            low=0.0, high=1.0, low_open=True,
        ),
        min_in_band_energy_ratio=_float_field(
            section, "min_in_band_energy_ratio", d.min_in_band_energy_ratio, "qc",
            low=0.0, high=1.0,
        ),
        rms_robust_z_threshold=_float_field(
            section, "rms_robust_z_threshold", d.rms_robust_z_threshold, "qc",
            low=0.0, high=math.inf, low_open=True, high_open=True,
        ),
        qc_gate_m=_gate_field(section, "qc_gate_m", d.qc_gate_m, "qc"),
        in_band_margin_hz=_float_field(
            section, "in_band_margin_hz", d.in_band_margin_hz, "qc",
            low=0.0, high=math.inf, high_open=True,
        ),
        min_frame_fraction=_float_field(
            section, "min_frame_fraction", d.min_frame_fraction, "qc",
            low=0.0, high=1.0, low_open=True,
        ),
    )


def _build_preprocess(raw: dict) -> PreprocessConfig:
    section = _known_section(raw, "preprocess", PreprocessConfig)
    d = PreprocessConfig()
    positive = dict(low=0.0, high=math.inf, low_open=True, high_open=True)
    edge_trim = section.get("edge_trim", d.edge_trim)
    if isinstance(edge_trim, bool) or not isinstance(edge_trim, int) or edge_trim < 0:
        raise ConfigError(f"preprocess.edge_trim must be an integer >= 0, got {edge_trim!r}")
    return PreprocessConfig(
        butter_order=_int_field(section, "butter_order", d.butter_order, "preprocess"),
        model_gate_m=_gate_field(section, "model_gate_m", d.model_gate_m, "preprocess"),
        edge_trim=edge_trim,
        fs_hz=_float_field(section, "fs_hz", d.fs_hz, "preprocess", **positive),
        bandwidth_hz=_float_field(section, "bandwidth_hz", d.bandwidth_hz, "preprocess", **positive),
        chirp_time_s=_float_field(section, "chirp_time_s", d.chirp_time_s, "preprocess", **positive),
        gate_method=_choice_field(
            section, "gate_method", d.gate_method, "preprocess", GATE_METHODS
        ),
        fft_gate_transition_hz=_float_field(
            section, "fft_gate_transition_hz", d.fft_gate_transition_hz, "preprocess", **positive
        ),
        # minimum=0: keeping only the peak bin is a legitimate (test-only) setting.
        peak_neighbors=_int_field(
            section, "peak_neighbors", d.peak_neighbors, "preprocess", minimum=0
        ),
        mask_taper=_bool_field(section, "mask_taper", d.mask_taper, "preprocess"),
        standardize=_choice_field(
            section, "standardize", d.standardize, "preprocess", STANDARDIZE_METHODS
        ),
    )


def _build_wst(raw: dict) -> WSTConfig:
    """Validate the WST fields consumed from M4 (the M2 rule: a bad value must fail at
    config load, not deep inside kymatio's filter-bank construction).

    `tilings` stays un-overridable (frozen design constants). `max_order` and
    `log_epsilon` are frozen protocol constants but validated so a typo fails loudly;
    `backend` is a bounded choice like gate_method/standardize.
    """
    section = _known_section(raw, "wst", WSTConfig)
    if "tilings" in section:
        raise ConfigError(
            "wst.tilings cannot be overridden in YAML — the three tilings are frozen "
            "constants of the design (see implementation_plan.md)"
        )
    d = WSTConfig()

    max_order = section.get("max_order", d.max_order)
    if isinstance(max_order, bool) or not isinstance(max_order, int):
        raise ConfigError(f"wst.max_order must be an integer, got {type(max_order).__name__}")
    if max_order not in (1, 2):
        raise ConfigError(
            f"wst.max_order must be 1 or 2, got {max_order} — order 0 keeps no wavelet "
            "paths and >2 is unsupported by the design (the plan fixes max_order = 2)"
        )

    return WSTConfig(
        max_order=max_order,
        log_epsilon=_float_field(
            section, "log_epsilon", d.log_epsilon, "wst",
            low=0.0, high=math.inf, low_open=True, high_open=True,
        ),
        backend=_choice_field(section, "backend", d.backend, "wst", BACKENDS),
    )


def _build_preprocess77(raw: dict) -> Preprocess77Config:
    """Validate the 77 GHz preprocessing fields (M2 rule: a bad value fails at load)."""
    section = _known_section(raw, "preprocess77", Preprocess77Config)
    d = Preprocess77Config()
    positive = dict(low=0.0, high=math.inf, low_open=True, high_open=True)
    return Preprocess77Config(
        butter_order=_int_field(section, "butter_order", d.butter_order, "preprocess77"),
        gate_m=_gate_field(section, "gate_m", d.gate_m, "preprocess77"),
        fs_hz=_float_field(section, "fs_hz", d.fs_hz, "preprocess77", **positive),
        bandwidth_hz=_float_field(section, "bandwidth_hz", d.bandwidth_hz, "preprocess77", **positive),
        chirp_time_s=_float_field(section, "chirp_time_s", d.chirp_time_s, "preprocess77", **positive),
        standardize=_choice_field(
            section, "standardize", d.standardize, "preprocess77", STANDARDIZE_METHODS
        ),
    )


def _build_qc77(raw: dict) -> QC77Config:
    """Validate the 77 GHz QC fields. Exactly three screens; no RMS diagnostic."""
    section = _known_section(raw, "qc77", QC77Config)
    d = QC77Config()
    skip_leading = section.get("flatline_skip_leading_bins", d.flatline_skip_leading_bins)
    if isinstance(skip_leading, bool) or not isinstance(skip_leading, int) or skip_leading < 0:
        raise ConfigError(
            f"qc77.flatline_skip_leading_bins must be an integer >= 0, got {skip_leading!r}"
        )
    return QC77Config(
        histogram_bins=_int_field(section, "histogram_bins", d.histogram_bins, "qc77"),
        flatline_max_bin_fraction=_float_field(
            section, "flatline_max_bin_fraction", d.flatline_max_bin_fraction, "qc77",
            low=0.0, high=1.0, low_open=True,
        ),
        flatline_skip_leading_bins=skip_leading,
        min_in_band_energy_ratio=_float_field(
            section, "min_in_band_energy_ratio", d.min_in_band_energy_ratio, "qc77",
            low=0.0, high=1.0,
        ),
        in_band_margin_hz=_float_field(
            section, "in_band_margin_hz", d.in_band_margin_hz, "qc77",
            low=0.0, high=math.inf, high_open=True,
        ),
        min_frame_fraction=_float_field(
            section, "min_frame_fraction", d.min_frame_fraction, "qc77",
            low=0.0, high=1.0, low_open=True,
        ),
    )


def _build_wst77(raw: dict) -> WST77Config:
    """Validate the 77 GHz WST fields. `tilings` stays code-frozen like the 10 GHz WST."""
    section = _known_section(raw, "wst77", WST77Config)
    if "tilings" in section:
        raise ConfigError(
            "wst77.tilings cannot be overridden in YAML — the three Doppler tilings are "
            "frozen constants of the design (see implementation_plan.md Exp G)"
        )
    d = WST77Config()

    max_order = section.get("max_order", d.max_order)
    if isinstance(max_order, bool) or not isinstance(max_order, int):
        raise ConfigError(f"wst77.max_order must be an integer, got {type(max_order).__name__}")
    if max_order not in (1, 2):
        raise ConfigError(
            f"wst77.max_order must be 1 or 2, got {max_order} — order 0 keeps no wavelet "
            "paths and >2 is unsupported by the design (the plan fixes max_order = 2)"
        )

    return WST77Config(
        max_order=max_order,
        log_epsilon=_float_field(
            section, "log_epsilon", d.log_epsilon, "wst77",
            low=0.0, high=math.inf, low_open=True, high_open=True,
        ),
        backend=_choice_field(section, "backend", d.backend, "wst77", BACKENDS),
    )


# ---------------------------------------------------------- milestone 6 frozen records
# Every milestone-6 section is a FROZEN RECORD (Step 0 resolved 2026-07-25): a run YAML
# may RESTATE a field at its owner-approved default (so a config is a complete record)
# but may not CHANGE it. protocol_freeze_guard re-checks every value at modelling time;
# this builder is the load-time half of the same contract. One builder serves all 11
# sections because, unlike qc/preprocess/wst (which carry live inner-CV axes and
# pre-declared ablations), nothing in these sections is a knob.


def _normalize_frozen(value):
    """YAML gives lists; frozen defaults are tuples. Normalize recursively so a restated
    value compares equal to its tuple default (e.g. [[1,2],[0.9,3]] -> ((1,2),(0.9,3)))."""
    if isinstance(value, list):
        return tuple(_normalize_frozen(v) for v in value)
    return value


def _frozen_matches(value, default) -> bool:
    """Type-aware equality for a frozen field. A bool is NOT interchangeable with an int
    (YAML writes true/false, so `1` for a bool is a typo, and `True == 1` must not slip
    through); tuples compare elementwise after list->tuple normalization."""
    value = _normalize_frozen(value)
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(default, bool) and value == default
    if isinstance(default, tuple):
        return (
            isinstance(value, tuple)
            and len(value) == len(default)
            and all(_frozen_matches(v, d) for v, d in zip(value, default))
        )
    return value == default


def _build_frozen_record(raw: dict, name: str, cls):
    """Build a milestone-6 frozen-record section. Unknown keys and any *changed* value
    are ConfigErrors; restating a default is allowed. Returns the canonical `cls()`
    (safe, since every present value was proven equal to its default)."""
    section = _known_section(raw, name, cls)
    default = cls()
    for key, value in section.items():
        want = getattr(default, key)
        if not _frozen_matches(value, want):
            raise ConfigError(
                f"{name}.{key} is a frozen protocol constant and cannot be changed from "
                f"{want!r} (got {value!r}) — it is fixed at the milestone-6 config-freeze "
                "gate (plans/MILESTONE_6_PLAN.md, Step 0). Change the code default and its "
                "amendment record, not a run YAML."
            )
    return default


# The milestone-6 section name -> dataclass map, used by load_config and reused by
# features/protocol_freeze.py so the two never disagree on what a "frozen section" is.
M6_SECTIONS = {
    "search_10ghz": SearchSpace10GHzConfig,
    "search_77ghz": SearchSpace77GHzConfig,
    "model_grid": ModelGridConfig,
    "baselines": BaselineConfig,
    "exp_b": ExpBConfig,
    "exp_c": ExpCConfig,
    "exp_e": ExpEConfig,
    "exp_f": ExpFConfig,
    "exp_g": ExpGConfig,
    "stats": StatsConfig,
    "protocol_freeze": ProtocolFreezeConfig,
}


def load_config(*yaml_paths: str | Path) -> Config:
    """Load, merge and validate one or more YAML files (later files win).

    Typical use:
        load_config("configs/exp_a_regression.yaml")                  # local
        load_config("configs/exp_a_regression.yaml", "configs/ibex.yaml")  # on IBEX
    """
    if not yaml_paths:
        raise ConfigError("load_config requires at least one YAML path")

    merged: dict = {}
    for path in yaml_paths:
        merged = _merge(merged, _load_with_includes(Path(path).resolve()))

    known_sections = (
        "paths", "run", "split", "qc", "preprocess", "wst",
        "qc77", "preprocess77", "wst77",
        *M6_SECTIONS,
    )
    _reject_unknown(merged, known_sections, "config")

    qc = _build_qc(merged)
    preprocess = _build_preprocess(merged)
    # Cross-section: both bands are only meaningful against this radar's fs/B/Tchirp,
    # and the model gate is additionally constrained by the (wider) frozen QC gate.
    _check_qc_band(qc, preprocess)
    _check_model_band(qc, preprocess)

    qc77 = _build_qc77(merged)
    preprocess77 = _build_preprocess77(merged)
    # Same cross-check for band 2's single gate (always present via defaults).
    _check_qc77_band(qc77, preprocess77)

    m6 = {name: _build_frozen_record(merged, name, cls) for name, cls in M6_SECTIONS.items()}

    return Config(
        paths=_build_paths(merged),
        run=_build_run(merged),
        split=_build_split(merged),
        qc=qc,
        preprocess=preprocess,
        wst=_build_wst(merged),
        qc77=qc77,
        preprocess77=preprocess77,
        wst77=_build_wst77(merged),
        search_10ghz=m6["search_10ghz"],
        search_77ghz=m6["search_77ghz"],
        model_grid=m6["model_grid"],
        baselines=m6["baselines"],
        exp_b=m6["exp_b"],
        exp_c=m6["exp_c"],
        exp_e=m6["exp_e"],
        exp_f=m6["exp_f"],
        exp_g=m6["exp_g"],
        stats=m6["stats"],
        protocol_freeze=m6["protocol_freeze"],
    )


# ----------------------------------------------------------------------- provenance


def config_to_dict(config) -> dict:
    """Fully-resolved config as plain JSON-serializable data (for provenance.py)."""
    if is_dataclass(config):
        return {f.name: config_to_dict(getattr(config, f.name)) for f in fields(config)}
    if isinstance(config, (list, tuple)):
        return [config_to_dict(v) for v in config]
    if isinstance(config, Path):
        return str(config)
    return config
