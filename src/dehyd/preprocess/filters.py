"""The band gate -- step 3 of the preprocessing sequence.

The paper says "window -> range FFT -> SOS bandpass", but the reference *code*
bandpasses the time-domain complex chirp directly and uses windowed FFTs only for QC
and peak detection. We follow the code, and state the domain explicitly: this is a
zero-phase IIR filter along **fast time, per chirp, per frame**, on the complex signal.
Beat-frequency banding *is* the range gate -- no FFT is taken in the primary path.

**No window before the filter** (a deliberate ROADMAP §3.2 departure). Windowing
suppresses FFT spectral leakage, which is irrelevant to a time-domain IIR filter, and
pre-tapering a fast-time chirp would attenuate real signal energy at its edges.
filtfilt's edge transients are handled by EdgeTrim after reduction instead.

Everything here is shape- and fs-agnostic so the 77 GHz chain (M9, fs = 500 kHz,
N = 256) can reuse it: the physical constants arrive as arguments, never as module
constants.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..config import PreprocessConfig, beat_band_hz

# scipy's sosfiltfilt default. Frozen by being passed EXPLICITLY (see default_padlen):
# padding shapes the edges of a 534-sample record, so it must be pinned by our code
# rather than inherited from a library default that could change under us.
PADTYPE = "odd"


class FilterError(ValueError):
    """Raised when a filter design or application is structurally invalid."""


def design_bandpass_sos(
    f_lo_hz: float, f_hi_hz: float, fs_hz: float, order: int
) -> np.ndarray:
    """Order-`order` Butterworth bandpass as second-order sections.

    scipy's `order` N gives 2N poles for a bandpass, matching MATLAB
    `butter(4, ..., 'bandpass')` -> 8 poles -> 4 sections.

    SOS rather than (b, a) is mandatory, not stylistic: the normalized cutoffs here are
    ~0.0125-0.025, where the transfer-function form loses its poles to floating-point
    error. (scipy folds the overall gain into the sections; MATLAB returns it separately
    as `g`. Same filter, and there is no second gain to apply.)
    """
    nyquist = fs_hz / 2.0
    if not (0.0 < f_lo_hz < f_hi_hz < nyquist):
        raise FilterError(
            f"band {f_lo_hz:.1f}-{f_hi_hz:.1f} Hz is not strictly inside "
            f"(0, {nyquist:.1f}) Hz — cannot design a bandpass"
        )
    return butter(order, [f_lo_hz / nyquist, f_hi_hz / nyquist], btype="bandpass", output="sos")


def default_padlen(sos: np.ndarray) -> int:
    """scipy's documented `sosfiltfilt` default pad length, from the sos alone.

    `3 * (2 * n_sections + 1 - min(#zero b2, #zero a2))` -- a pure function of the
    design, so it can be recorded from config without running the filter. For the
    default 10 GHz design (order 4 -> 4 sections) this is 27; T-PP1 pins both that
    value and its bit-identity with the library default, so a scipy change surfaces as
    a failing test rather than as silently different edges.
    """
    sos = np.asarray(sos)
    n_sections = sos.shape[0]
    n_zero_b2 = int((sos[:, 2] == 0).sum())
    n_zero_a2 = int((sos[:, 5] == 0).sum())
    return 3 * (2 * n_sections + 1 - min(n_zero_b2, n_zero_a2))


def bandpass_filtfilt(x: np.ndarray, sos: np.ndarray, axis: int = 0) -> np.ndarray:
    """Zero-phase forward-backward filtering along `axis`.

    Complex input is filtered as real and imaginary parts SEPARATELY through the same
    real SOS filter. That is mathematically identical to filtering the complex signal
    (the filter is real and linear), but it is written out because the main plan
    specifies it and because scipy's complex handling is not something to rely on.

    Note the effective response is |H|^2 -- filtfilt applies the filter twice -- so the
    -3 dB design corners are -6 dB in practice (T-PP2), and on a finite 534-sample
    record the realized energy retention is well below the steady-state figure (T-PP6).
    """
    x = np.asarray(x)
    padlen = default_padlen(sos)
    if np.iscomplexobj(x):
        real = sosfiltfilt(sos, x.real, axis=axis, padtype=PADTYPE, padlen=padlen)
        imag = sosfiltfilt(sos, x.imag, axis=axis, padtype=PADTYPE, padlen=padlen)
        return real + 1j * imag
    return sosfiltfilt(sos, x, axis=axis, padtype=PADTYPE, padlen=padlen)


def fft_gate(
    x: np.ndarray,
    f_lo_hz: float,
    f_hi_hz: float,
    fs_hz: float,
    transition_hz: float,
    axis: int = 0,
) -> np.ndarray:
    """Frequency-domain tapered-mask range gate -- the pre-declared ablation.

    Ported from `filter_gpt_fft.m`: FFT along `axis`, multiply by a passband mask that
    is flat inside [f_lo, f_hi] with raised-cosine (Hann) skirts of width
    `transition_hz`, then IFFT. The mask is symmetric in |f| so it treats the complex
    signal's negative frequencies identically.

    Built directly in unshifted bin space -- the reference's fftshift/ifftshift
    round-trip produces the same mask with more opportunity to get an index wrong.
    """
    x = np.asarray(x)
    n = x.shape[axis]
    freqs = np.abs(np.fft.fftfreq(n, d=1.0 / fs_hz))

    mask = np.zeros(n, dtype=np.float64)
    mask[(freqs >= f_lo_hz) & (freqs <= f_hi_hz)] = 1.0
    if transition_hz > 0:
        lower = (freqs >= f_lo_hz - transition_hz) & (freqs < f_lo_hz)
        mask[lower] = 0.5 * (1.0 - np.cos(np.pi * (freqs[lower] - (f_lo_hz - transition_hz)) / transition_hz))
        upper = (freqs > f_hi_hz) & (freqs <= f_hi_hz + transition_hz)
        mask[upper] = 0.5 * (1.0 + np.cos(np.pi * (freqs[upper] - f_hi_hz) / transition_hz))

    shape = [1] * x.ndim
    shape[axis] = n
    spectrum = np.fft.fft(x, axis=axis) * mask.reshape(shape)
    gated = np.fft.ifft(spectrum, axis=axis)
    return gated if np.iscomplexobj(x) else gated.real


def apply_band_gate(x: np.ndarray, pre: PreprocessConfig, axis: int = 0) -> np.ndarray:
    """The band gate as configured: the primary Butterworth, or the FFT ablation.

    The band always comes from the MODEL gate (`preprocess.model_gate_m`), never the QC
    gate: QC froze the frame population on the wider 0.9-3.0 m band so it stays
    identical for every model-gate candidate later searched in inner CV.
    """
    f_lo, f_hi = beat_band_hz(pre.model_gate_m, pre.bandwidth_hz, pre.chirp_time_s)
    if pre.gate_method == "butterworth":
        sos = design_bandpass_sos(f_lo, f_hi, pre.fs_hz, pre.butter_order)
        return bandpass_filtfilt(x, sos, axis=axis)
    if pre.gate_method == "fft":
        return fft_gate(x, f_lo, f_hi, pre.fs_hz, pre.fft_gate_transition_hz, axis=axis)
    raise FilterError(f"unknown gate_method {pre.gate_method!r}")  # unreachable via config


def filter_spec(pre: PreprocessConfig) -> dict:
    """The realized design, as plain data for provenance and HISTORY.md.

    Everything is derived from the config -- including the explicit padlen, which is a
    function of the sos and so needs no signal to observe.
    """
    f_lo, f_hi = beat_band_hz(pre.model_gate_m, pre.bandwidth_hz, pre.chirp_time_s)
    nyquist = pre.fs_hz / 2.0
    spec = {
        "gate_method": pre.gate_method,
        "model_gate_m": list(pre.model_gate_m),
        "f_lo_hz": f_lo,
        "f_hi_hz": f_hi,
        "fs_hz": pre.fs_hz,
    }
    if pre.gate_method == "butterworth":
        sos = design_bandpass_sos(f_lo, f_hi, pre.fs_hz, pre.butter_order)
        spec.update(
            butter_order=pre.butter_order,
            n_sections=int(sos.shape[0]),
            wn=[f_lo / nyquist, f_hi / nyquist],
            padtype=PADTYPE,
            padlen=default_padlen(sos),
        )
    else:
        spec.update(fft_gate_transition_hz=pre.fft_gate_transition_hz)
    return spec
