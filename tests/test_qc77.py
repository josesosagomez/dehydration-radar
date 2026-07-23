"""77 GHz QC screens and the semantic axis check (T-Q77, no private data).

The screens are shape-generic, so flatline / in-band / per-Rx behaviour is tested on small
frames; the mask-bin pinning uses the real n_fast=256. The axis check is tested on small
real-float tone cubes (the loaded cube is real float64 in production). A noiseless tone is
avoided where it would collide with the flatline rule.
"""

import numpy as np
import pytest
from types import SimpleNamespace

from dehyd.config import Preprocess77Config, QC77Config
from dehyd.qc.axis_check_77 import (
    axis_metrics,
    axis_spec_hash,
    axis_verdict,
    certify_axis,
    range_gate_bins,
)
from dehyd.qc.screens import in_band_mask
from dehyd.qc.screens_77 import FrameQC77, _flatline_per_rx, run_qc_cube_77, run_qc_frame_77

QC = QC77Config()
PRE = Preprocess77Config()

N_FAST_S = 64
N_CHIRP_S = 4
N_RX_S = 2


def noise_frame(seed=0, n_fast=N_FAST_S, n_chirp=N_CHIRP_S, n_rx=N_RX_S):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_fast, n_chirp, n_rx))


def in_gate_tone_frame(bin_idx=10, seed=1, noise=0.05):
    """A real fast-time tone at `bin_idx` (n_fast=64), constant across chirps + a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_FAST_S)
    tone = np.cos(2 * np.pi * bin_idx * t / N_FAST_S)
    frame = np.broadcast_to(tone[:, None, None], (N_FAST_S, N_CHIRP_S, N_RX_S)).copy()
    return frame + noise * rng.standard_normal(frame.shape)


# ------------------------------------------------------------------------- flatline


def test_quantised_trace_flatlines():
    frame = noise_frame()
    frame[:, 0, 0] = np.tile([0.0, 1.0], N_FAST_S // 2)  # two magnitude levels -> one bin piles up
    result = run_qc_frame_77(frame, QC, PRE)
    assert result.flatline is True
    assert result.per_rx_flatline[0] >= 1


def test_constant_trace_flatlines_via_degenerate_spread():
    frame = noise_frame()
    frame[:, 1, 1] = 5.0  # constant -> min==max -> degenerate spread == flatline
    result = run_qc_frame_77(frame, QC, PRE)
    assert result.flatline is True
    assert result.per_rx_flatline[1] >= 1


def test_noise_frame_does_not_flatline():
    result = run_qc_frame_77(noise_frame(seed=3), QC, PRE)
    assert result.flatline is False
    assert result.n_flatline_traces == 0
    assert result.per_rx_flatline == (0, 0)


def test_any_trace_rule_one_bad_of_many_fails_the_frame():
    """A single flatlined trace out of n_chirp*n_rx flags the whole frame (the any-trace rule)."""
    frame = noise_frame(seed=4)
    frame[:, 2, 1] = 7.0  # exactly one constant trace
    result = run_qc_frame_77(frame, QC, PRE)
    assert result.flatline is True
    assert result.n_flatline_traces == 1


def test_per_rx_counts_are_correct():
    frame = noise_frame(seed=5)
    frame[:, 0, 0] = 3.0            # rx0: constant
    frame[:, 1, 0] = np.tile([0.0, 2.0], N_FAST_S // 2)  # rx0: quantised
    frame[:, 0, 1] = 1.0           # rx1: constant
    result = run_qc_frame_77(frame, QC, PRE)
    assert result.per_rx_flatline == (2, 1)
    assert result.n_flatline_traces == 3


# ------------------------------------------------- M5 step-6 counter-outlier correction


def test_counter_outlier_does_not_false_flatline():
    """The step-6 fix: a huge value at fast[0] (the embedded frame counter) must NOT flag.

    Without the exclusion the single outlier stretches the [min,max] range so all the echo
    piles into the first bins and false-trips the rule — this reproduces M2's 7/10, then
    shows the production screen (skip_leading=1) no longer fires.
    """
    frame = noise_frame(seed=8)
    frame[0, :, :] = 5000.0  # counter at range bin 0, every (Rx, chirp) trace

    old = _flatline_per_rx(frame, bins=QC.histogram_bins,
                           max_bin_fraction=QC.flatline_max_bin_fraction, skip_leading=0)
    assert sum(old) == N_CHIRP_S * N_RX_S  # the OLD any-trace rule false-flags every trace

    result = run_qc_frame_77(frame, QC, PRE)  # production rule (skip_leading=1)
    assert result.flatline is False
    assert result.n_flatline_traces == 0


def test_skip_leading_still_catches_a_dead_channel():
    """Excluding bin 0 must not hide a genuinely dead channel (echo constant, not just bin 0)."""
    frame = noise_frame(seed=9)
    frame[1:, 0, 0] = 4.0  # the ECHO samples are constant -> degenerate -> flatline
    result = run_qc_frame_77(frame, QC, PRE)
    assert result.flatline is True
    assert result.per_rx_flatline[0] >= 1


# ------------------------------------------------------------------------- in-band


def test_in_gate_tone_passes_in_band():
    result = run_qc_frame_77(in_gate_tone_frame(bin_idx=10), QC, PRE)
    assert result.low_in_band is False
    assert result.in_band_ratio >= QC.min_in_band_energy_ratio


def test_out_of_gate_tone_is_low_in_band():
    # bin 2 at n_fast=64 -> ~15.6 kHz, below the 2-4 m band (mask bins 7..13 here).
    result = run_qc_frame_77(in_gate_tone_frame(bin_idx=2), QC, PRE)
    assert result.low_in_band is True
    assert result.in_band_ratio < QC.min_in_band_energy_ratio


def test_nan_inf_short_circuits():
    frame = in_gate_tone_frame()
    frame[0, 0, 0] = np.nan
    result = run_qc_frame_77(frame, QC, PRE)
    assert result.nan_inf is True
    assert result.flatline is False and result.low_in_band is False
    assert np.isnan(result.in_band_ratio)
    assert result.n_flatline_traces == 0
    assert result.passed is False


def test_mask_bins_pinned_at_256():
    """The frozen 77 GHz QC mask is bins 26..54 (matches the M2 audit)."""
    mask = np.flatnonzero(
        in_band_mask(256, PRE.fs_hz, PRE.bandwidth_hz, PRE.chirp_time_s, PRE.gate_m,
                     QC.in_band_margin_hz)
    )
    assert (mask[0], mask[-1]) == (26, 54)


# ------------------------------------------------------------------- per-frame independence


def test_run_qc_cube_is_per_frame_independent():
    """Permuting the other frames must not change any frame's verdict (no cross-frame stat)."""
    frames = [in_gate_tone_frame(seed=s) for s in range(4)]
    frames[2][:, 0, 0] = 9.0  # make frame 2 flatline
    cube = np.stack(frames, axis=0)
    baseline = run_qc_cube_77(cube, QC, PRE)

    permuted = np.stack([frames[3], frames[0], frames[2], frames[1]], axis=0)
    reordered = run_qc_cube_77(permuted, QC, PRE)
    # Frame 2 sits at index 2 in both; its verdict is identical.
    assert reordered[2].flatline == baseline[2].flatline is True
    assert reordered[2].n_flatline_traces == baseline[2].n_flatline_traces


def test_passed_property_cannot_be_violated():
    r = FrameQC77(nan_inf=False, flatline=True, low_in_band=False, in_band_ratio=0.9,
                  n_flatline_traces=1, per_rx_flatline=(1, 0))
    assert r.passed is False


# ----------------------------------------------------------------- semantic axis check


def real_tone_cube(axis, bin_idx=40, n_fast=64, n_chirp=64, n_rx=2, n_frames=2):
    """Real cosine tone along `axis` at `bin_idx`, constant along the other. [frame,fast,chirp,rx]."""
    _, fast, chirp, _ = np.indices((n_frames, n_fast, n_chirp, n_rx))
    index = fast if axis == "fast" else chirp
    return np.cos(2 * np.pi * bin_idx * index / n_fast).astype(np.float64)


def test_axis_check_accepts_proposed_mapping():
    # Range energy along fast at a gate bin (27..53), constant across chirp -> D_chirp high.
    verdict, metrics = certify_axis(real_tone_cube("fast", bin_idx=40), PRE)
    assert verdict == "ACCEPTED"
    assert metrics["G_fast"] > metrics["G_chirp"]


def test_axis_check_rejects_swapped_mapping():
    verdict, _ = certify_axis(real_tone_cube("chirp", bin_idx=40), PRE)
    assert verdict == "REJECTED"


def test_axis_check_inconclusive_on_noise():
    rng = np.random.default_rng(7)
    cube = rng.standard_normal((2, 64, 64, 2))
    verdict, _ = certify_axis(cube, PRE)
    assert verdict == "INCONCLUSIVE"


def test_range_gate_bins_pinned():
    gb = range_gate_bins(256, PRE.bandwidth_hz, PRE.gate_m)
    assert (gb[0], gb[-1]) == (27, 53)


# ---------------------------------------------------------------------- axis_spec_hash


def test_axis_spec_hash_ignores_environment_and_tracks_axis_fields():
    base = SimpleNamespace(preprocess77=Preprocess77Config())
    h = axis_spec_hash(base)

    # A path-only "overlay" changes nothing the hash reads -> the certificate stays valid.
    same = SimpleNamespace(preprocess77=Preprocess77Config(),
                           paths=SimpleNamespace(results_dir="/ibex/whatever"))
    assert axis_spec_hash(same) == h

    # Changing an axis-relevant constant (gate/bandwidth) invalidates the certificate.
    assert axis_spec_hash(SimpleNamespace(preprocess77=Preprocess77Config(gate_m=(1.0, 3.0)))) != h
    assert axis_spec_hash(SimpleNamespace(preprocess77=Preprocess77Config(bandwidth_hz=1e9))) != h
    # ...but a NON-axis field (butter_order) does not.
    assert axis_spec_hash(SimpleNamespace(preprocess77=Preprocess77Config(butter_order=6))) == h


# ---------------------------------------------------------------------- realdata


@pytest.mark.realdata
def test_real_counter_frame_no_longer_false_flatlines(real_data_77_paths):
    """On the audited file, the corrected rule fixes M2's flatline false positive.

    Frame 9 of subject_1_8am carries an embedded counter ~2560 at range bin 0. The M2 rule
    (skip_leading=0) flagged ~3981/4096 traces; the corrected rule flags none — the traces
    are healthy (~230 distinct magnitude levels), only the bin-0 outlier tripped the screen.
    """
    from dehyd.data.loader_77ghz import load_77ghz_file

    cube = load_77ghz_file(real_data_77_paths["data_77ghz_dir"] / "subject_1_8am.mat")
    frame9 = cube[9]

    old = _flatline_per_rx(frame9, bins=QC.histogram_bins,
                           max_bin_fraction=QC.flatline_max_bin_fraction, skip_leading=0)
    assert sum(old) > 3000  # the M2 false positive is reproduced

    result = run_qc_frame_77(frame9, QC, PRE)  # corrected production rule
    assert result.flatline is False
    assert result.passed is True
