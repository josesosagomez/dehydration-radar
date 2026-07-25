"""The milestone-6 protocol-freeze guard for MODELLING entrypoints (M7+).

`canonical_spec_guard` / `canonical_spec_guard_77` protect the one canonical curated WST
*feature* artifact: they demand `preprocess.model_gate_m == (1.0, 2.0)` exactly, which is
right for the artifact but wrong for model selection, where `(0.9, 3.0)` is an APPROVED
search candidate. This guard is the modelling-time counterpart:

  * it reuses `canonical_spec_guard_77` unchanged (no 77 GHz feature field is a search
    axis — tiling/log are call arguments to extraction_77, not stored config);
  * it re-validates the 10 GHz `preprocess`/`wst` fields locally, with the ONE exception
    that `model_gate_m` is checked for MEMBERSHIP in the frozen `search_10ghz.range_gate_m`
    whitelist rather than equality to the single canonical gate;
  * it validates every milestone-6 frozen section equals its canonical default;
  * given an `active` call-time protocol record, it validates each axis value against the
    band's search-space whitelist — because which reduction / channel / tiling / log
    branch / gate / model family is actually used for a fit is a CALL ARGUMENT to the
    extraction/modelling functions, never a stored `Config` field, so a config-only guard
    cannot see it (a typo'd tiling would pass and still reach evaluation).

The local ~dozen-line re-validation of the 10 GHz feature fields is deliberate duplication
of `canonical_spec_guard`'s loop: it keeps `features/extraction.py` — frozen front-end code
this milestone does not touch — unrefactored. A future session may collapse the two behind a
shared private helper if it prefers; that is a behaviour-preserving cleanup, flagged in
MILESTONE_6_PLAN.md §6.
"""

from __future__ import annotations

import dataclasses

from ..config import Config
from .extraction import CANONICAL_PREPROCESS, CANONICAL_WST
from .extraction_77 import canonical_spec_guard_77


class ProtocolFreezeError(ValueError):
    """Raised when a config or a call-time protocol record leaves its frozen whitelist."""


def _as_tuple(value):
    """Normalize a list (e.g. a gate from JSON/YAML) to a tuple for whitelist comparison."""
    if isinstance(value, list):
        return tuple(_as_tuple(v) for v in value)
    return value


def _check_10ghz_feature_fields(config: Config, deviations: list[str]) -> None:
    """Re-validate preprocess/wst against canonical, EXCEPT model_gate_m (a search axis).

    model_gate_m must be a member of the frozen range_gate_m whitelist; every other
    preprocess/wst field must equal its canonical default (the ablation switches
    gate_method/standardize included — a modelling run uses the primary path).
    """
    allowed_gates = tuple(_as_tuple(g) for g in config.search_10ghz.range_gate_m)
    for section, canonical in (("preprocess", CANONICAL_PREPROCESS), ("wst", CANONICAL_WST)):
        actual = getattr(config, section)
        for f in dataclasses.fields(canonical):
            got = getattr(actual, f.name)
            if section == "preprocess" and f.name == "model_gate_m":
                if _as_tuple(got) not in allowed_gates:
                    deviations.append(
                        f"preprocess.model_gate_m={got!r} is not in the frozen "
                        f"search_10ghz.range_gate_m whitelist {allowed_gates}"
                    )
                continue
            want = getattr(canonical, f.name)
            if got != want:
                deviations.append(f"{section}.{f.name}={got!r} (canonical: {want!r})")


def _check_m6_sections(config: Config, deviations: list[str]) -> None:
    """Every milestone-6 frozen section must equal its canonical default.

    The loader already enforces this for any config that came through load_config, but the
    guard also defends configs built programmatically (tests, or a future direct-construct
    path) — the frozen sections are the whitelist tuples the `active` check reads.
    """
    from ..config import M6_SECTIONS

    for name, cls in M6_SECTIONS.items():
        actual = getattr(config, name)
        canonical = cls()
        if actual != canonical:
            for f in dataclasses.fields(cls):
                got, want = getattr(actual, f.name), getattr(canonical, f.name)
                if got != want:
                    deviations.append(f"{name}.{f.name}={got!r} (canonical: {want!r})")


# Per-band call-time axis validators: key -> (kind, reference). "member" means the value
# must be in the whitelist tuple; "equal" means it must equal the fixed (non-axis) value.
def _active_rules(config: Config, band: str):
    if band == "10ghz":
        s = config.search_10ghz
        return {
            "reduction": ("member", s.reduction),
            "channel": ("member", s.channel),
            "tiling": ("member", s.tiling),
            "log_branch": ("member", s.log_branches),
            "range_gate_m": ("member", tuple(_as_tuple(g) for g in s.range_gate_m)),
            "model_family": ("member", s.model_families),
        }
    if band == "77ghz":
        s = config.search_77ghz
        return {
            "reduction": ("equal", s.reduction),  # fixed at 77 GHz, not an axis
            "channel": ("equal", s.channel),
            "gate_m": ("equal", _as_tuple(s.gate_m)),
            "tiling": ("member", s.tiling),
            "log_branch": ("member", s.log_branches),
            "model_family": ("member", s.model_families),
        }
    return None


def _check_active(config: Config, active: dict, deviations: list[str]) -> None:
    band = active.get("band")
    if band not in ("10ghz", "77ghz"):
        deviations.append(f"active.band must be '10ghz' or '77ghz', got {band!r}")
        return
    rules = _active_rules(config, band)
    for key, value in active.items():
        if key == "band":
            continue
        rule = rules.get(key)
        if rule is None:
            deviations.append(f"active.{key} is not a recognized {band} protocol axis")
            continue
        kind, ref = rule
        checked = _as_tuple(value)
        if kind == "member" and checked not in ref:
            deviations.append(f"active.{key}={value!r} is not in the frozen whitelist {ref}")
        elif kind == "equal" and checked != ref:
            deviations.append(
                f"active.{key}={value!r} must equal the fixed {band} value {ref!r}"
            )


def protocol_freeze_guard(config: Config, active: dict | None = None) -> None:
    """Validate that `config` — and, if given, one call-time protocol record — stays inside
    the frozen milestone-6 whitelists. Raises `ProtocolFreezeError` naming every deviation.

    `active`, when given, is the exact protocol for ONE fit, e.g.::

        {"band": "10ghz", "reduction": "A", "channel": "mag", "tiling": "T1",
         "log_branch": "on_frozen_eps", "range_gate_m": (1.0, 2.0), "model_family": "ridge"}

    (77 GHz keys use gate_m, and reduction/channel/gate are fixed, not axes.) Every present
    axis is checked for membership in — or equality to — the band's frozen whitelist,
    regardless of how the caller produced it, so this is defense-in-depth over the harness's
    own enumeration, not a substitute for it. `active=None` validates only the config-level
    frozen sections (a pre-flight check with no fit in progress) and is NOT a substitute for
    passing `active` on every real fit. M7's harness MUST call this immediately before any
    model fit or result write.
    """
    canonical_spec_guard_77(config)  # 77 GHz feature fields; unchanged, safe to reuse

    deviations: list[str] = []
    _check_10ghz_feature_fields(config, deviations)
    _check_m6_sections(config, deviations)
    if active is not None:
        _check_active(config, active, deviations)

    if deviations:
        raise ProtocolFreezeError(
            "config/protocol left the frozen milestone-6 whitelist; deviating: "
            + "; ".join(deviations)
        )
