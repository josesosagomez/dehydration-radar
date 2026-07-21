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

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID_DEVICES = ("cpu", "cuda")


class ConfigError(ValueError):
    """Raised for any malformed or invalid configuration."""


# --------------------------------------------------------------------------- schema


@dataclass(frozen=True)
class PathsConfig:
    data_10ghz_dir: Path
    weight_xlsx: Path
    results_dir: Path


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
    min_frame_fraction: float = 0.5


@dataclass(frozen=True)
class PreprocessConfig:
    """Frozen preprocessing parameters (implementation_plan.md 'Preprocessing')."""

    butter_order: int = 4
    model_gate_m: tuple[float, float] = (1.0, 2.0)
    edge_trim: int = 32
    fs_hz: float = 520834.0
    bandwidth_hz: float = 500e6
    chirp_time_s: float = 1024e-6


@dataclass(frozen=True)
class WSTTiling:
    q: tuple[int, int]
    invariance_ms: float


@dataclass(frozen=True)
class WSTConfig:
    """Frozen WST tilings (implementation_plan.md 'WST parameterization')."""

    tilings: tuple[WSTTiling, ...] = (
        WSTTiling(q=(10, 4), invariance_ms=0.20),
        WSTTiling(q=(8, 2), invariance_ms=0.30),
        WSTTiling(q=(6, 2), invariance_ms=0.40),
    )
    max_order: int = 2
    log_epsilon: float = 1e-6


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    run: RunConfig
    split: SplitConfig = field(default_factory=SplitConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    wst: WSTConfig = field(default_factory=WSTConfig)


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


def _build_paths(raw: dict) -> PathsConfig:
    section = _section(raw, "paths")
    required = ("data_10ghz_dir", "weight_xlsx", "results_dir")
    _require_keys(section, required, "paths")
    _reject_unknown(section, required, "paths")

    resolved = {key: _resolve_path(section[key], key) for key in required}

    # Required INPUTS must exist. results_dir is an OUTPUT: it need only be creatable,
    # and its writers create it on demand.
    for key in ("data_10ghz_dir", "weight_xlsx"):
        if not resolved[key].exists():
            raise ConfigError(f"paths.{key} does not exist: {resolved[key]}")

    return PathsConfig(**resolved)


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


def _build_frozen_section(raw: dict, name: str, cls):
    """Build a section whose fields all have defaults (QC / preprocess / WST).

    Values may be overridden in YAML, but unknown keys are still rejected.
    """
    section = _section(raw, name)
    allowed = tuple(f.name for f in fields(cls))
    _reject_unknown(section, allowed, name)
    if not section:
        return cls()
    if name == "wst" and "tilings" in section:
        raise ConfigError(
            "wst.tilings cannot be overridden in YAML at milestone 1 — the three "
            "tilings are frozen constants of the design (see implementation_plan.md)"
        )
    return cls(**section)


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

    known_sections = ("paths", "run", "split", "qc", "preprocess", "wst")
    _reject_unknown(merged, known_sections, "config")

    return Config(
        paths=_build_paths(merged),
        run=_build_run(merged),
        split=_build_split(merged),
        qc=_build_frozen_section(merged, "qc", QCConfig),
        preprocess=_build_frozen_section(merged, "preprocess", PreprocessConfig),
        wst=_build_frozen_section(merged, "wst", WSTConfig),
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
