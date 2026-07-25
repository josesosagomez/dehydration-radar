"""The modelling-time protocol-freeze guard (`features/protocol_freeze.py`) — T-C6-guard.

Proves: it composes the existing `canonical_spec_guard_77` unchanged; it validates the
10 GHz feature fields locally with the ONE range-gate exception (so both APPROVED gates
pass while an out-of-whitelist gate fails); it validates a call-time `active` record even
when the stored config is canonical (C6-29); and a stub entrypoint runs it BEFORE any
result I/O. A regression check confirms the strict `canonical_spec_guard` on the artifact
write path is untouched.
"""

import dataclasses

import pytest
import yaml

from dehyd.config import REPO_ROOT, load_config
from dehyd.features.protocol_freeze import ProtocolFreezeError, protocol_freeze_guard

EXP_A = REPO_ROOT / "configs" / "exp_a_regression.yaml"


@pytest.fixture
def cfg(tmp_path):
    data_dir = tmp_path / "radar"
    data_dir.mkdir()
    xlsx = tmp_path / "w.xlsx"
    xlsx.write_bytes(b"")
    overlay = tmp_path / "o.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_10ghz_dir": str(data_dir),
                    "weight_xlsx": str(xlsx),
                    "results_dir": str(tmp_path / "results"),
                }
            }
        )
    )
    return load_config(EXP_A, overlay)


def active_10ghz(**overrides):
    base = {
        "band": "10ghz",
        "reduction": "A",
        "channel": "mag",
        "tiling": "T1",
        "log_branch": "on_frozen_eps",
        "range_gate_m": (1.0, 2.0),
        "model_family": "ridge",
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ config-level guard


def test_canonical_config_passes_with_no_active(cfg):
    protocol_freeze_guard(cfg)  # must not raise


def test_non_canonical_m6_section_is_rejected(cfg):
    bad = dataclasses.replace(cfg, stats=dataclasses.replace(cfg.stats, bootstrap_b=999))
    with pytest.raises(ProtocolFreezeError, match="stats.bootstrap_b=999"):
        protocol_freeze_guard(bad)


def test_composes_canonical_spec_guard_77(cfg):
    """A non-canonical 77 GHz feature field is caught via the reused 77 GHz guard."""
    from dehyd.features.extraction_77 import CanonicalSpecError77

    bad = dataclasses.replace(cfg, wst77=dataclasses.replace(cfg.wst77, backend="torch"))
    with pytest.raises(CanonicalSpecError77):
        protocol_freeze_guard(bad)


# ------------------------------------------------------- range-gate exception (C6-24)


@pytest.mark.parametrize("gate", [(1.0, 2.0), (0.9, 3.0)])
def test_both_approved_gates_pass_via_active(cfg, gate):
    protocol_freeze_guard(cfg, active_10ghz(range_gate_m=gate))  # must not raise


def test_out_of_whitelist_gate_fails(cfg):
    with pytest.raises(ProtocolFreezeError, match="range_gate_m"):
        protocol_freeze_guard(cfg, active_10ghz(range_gate_m=(0.5, 1.5)))


def test_config_model_gate_off_whitelist_is_rejected(cfg):
    """The 10 GHz feature-field check flags a stored model gate outside the whitelist,
    while still accepting the second approved gate (0.9, 3.0) — i.e. it does NOT demand
    the single canonical (1.0, 2.0) the artifact guard requires."""
    ok = dataclasses.replace(
        cfg, preprocess=dataclasses.replace(cfg.preprocess, model_gate_m=(0.9, 3.0))
    )
    protocol_freeze_guard(ok)  # (0.9, 3.0) is whitelisted -> passes
    bad = dataclasses.replace(
        cfg, preprocess=dataclasses.replace(cfg.preprocess, model_gate_m=(0.8, 1.2))
    )
    with pytest.raises(ProtocolFreezeError, match="model_gate_m"):
        protocol_freeze_guard(bad)


# ------------------------------------------------ call-time axis validation (C6-29)


@pytest.mark.parametrize(
    "override, needle",
    [
        ({"tiling": "T9"}, "tiling"),
        ({"reduction": "C"}, "reduction"),
        ({"channel": "phase"}, "channel"),
        ({"log_branch": "on"}, "log_branch"),
        ({"model_family": "xgboost"}, "model_family"),
    ],
)
def test_invalid_call_time_axis_rejected_even_when_config_canonical(cfg, override, needle):
    """The config is canonical, but a bad call-time value must still be caught (C6-29)."""
    with pytest.raises(ProtocolFreezeError, match=needle):
        protocol_freeze_guard(cfg, active_10ghz(**override))


def test_unknown_active_key_is_rejected(cfg):
    with pytest.raises(ProtocolFreezeError, match="not a recognized"):
        protocol_freeze_guard(cfg, active_10ghz(tilng="T1"))  # typo'd key


def test_bad_active_band_is_rejected(cfg):
    with pytest.raises(ProtocolFreezeError, match="active.band"):
        protocol_freeze_guard(cfg, {"band": "24ghz"})


def test_77ghz_active_fixed_axes_and_membership(cfg):
    ok = {
        "band": "77ghz",
        "reduction": "slow_time_iq_primary",
        "channel": "iq",
        "gate_m": (2.0, 4.0),
        "tiling": "T2_77",
        "log_branch": "on_tuned_eps",
        "model_family": "gbm",
    }
    protocol_freeze_guard(cfg, ok)  # must not raise
    with pytest.raises(ProtocolFreezeError, match="reduction"):
        protocol_freeze_guard(cfg, {**ok, "reduction": "A"})  # 77 GHz reduction is fixed
    with pytest.raises(ProtocolFreezeError, match="tiling"):
        protocol_freeze_guard(cfg, {**ok, "tiling": "T1"})  # 10 GHz tiling name, wrong band


# --------------------------------------------- entrypoint runs guard before I/O (C6-29)


def test_guard_runs_before_result_write(cfg, tmp_path):
    """A stub M7-style entrypoint validates BEFORE writing — a rejected protocol leaves no
    output file behind."""
    out = tmp_path / "result.json"

    def stub_entrypoint(config, active):
        protocol_freeze_guard(config, active)  # gate first
        out.write_text("{}")  # ...only then any result I/O

    with pytest.raises(ProtocolFreezeError):
        stub_entrypoint(cfg, active_10ghz(tiling="T9"))
    assert not out.exists()  # nothing was written

    stub_entrypoint(cfg, active_10ghz())  # a valid protocol does write
    assert out.exists()


# --------------------------------------------- the artifact guard stays strict (C6-24)


def test_canonical_spec_guard_still_rejects_the_second_gate_on_artifact_path(cfg):
    """The strict feature-artifact guard is unchanged: it still demands the single
    canonical (1.0, 2.0) model gate, so the modelling guard's laxer gate rule did not
    weaken the artifact write path."""
    from dehyd.features.extraction import CanonicalSpecError, canonical_spec_guard

    canonical_spec_guard(cfg)  # canonical config passes both guards
    second_gate = dataclasses.replace(
        cfg, preprocess=dataclasses.replace(cfg.preprocess, model_gate_m=(0.9, 3.0))
    )
    with pytest.raises(CanonicalSpecError):
        canonical_spec_guard(second_gate)  # artifact guard still rejects it
