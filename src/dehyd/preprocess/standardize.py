"""Channel mapping and per-signal standardization -- steps 6 and 7 of the sequence.

Each signal is standardized **from its own statistics**. That is what keeps
preprocessing free of fitted quantities: nothing is estimated on one set of frames and
applied to another, so there is no train/test leakage vector here and no reason for
this stage to live inside the CV loop.

**Deliberate departure from the reference** (settled in implementation_plan.md
"Deliberate departures"): `standardize_robust` in `wst_extract.m` centres by the *mean*
but scales by the *MAD* -- an internally inconsistent mix of a non-robust location with
a robust scale. We use the coherent robust z: median-centred, MAD-scaled. The 77 GHz
reference already uses this form.
"""

from __future__ import annotations

import numpy as np

# The MAD -> sigma consistency constant for Gaussian data: 1/Phi^-1(3/4). It makes the
# denominator comparable to a standard deviation, so a robust-z of 4.5 means roughly
# "4.5 sigma" (the same constant the QC RMS screen uses).
MAD_TO_SIGMA = 1.4826

# Guards a zero denominator. Frozen as float64 machine epsilon, and placed OUTSIDE the
# scale factor: `1.4826*MAD + eps`, not the reference's `1.4826*(MAD + eps)`. The
# difference is numerically irrelevant -- this is a division guard, not a tuning
# constant -- but one form has to be fixed for bit-reproducibility.
EPS = float(np.finfo(np.float64).eps)


class StandardizeError(ValueError):
    """Raised for an unknown standardization method or channel."""


def robust_standardize(x: np.ndarray) -> np.ndarray:
    """y = (x - median) / (1.4826 * MAD + eps). The primary form.

    A constant signal has MAD = 0 and therefore maps to all zeros -- finite, never NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return (x - median) / (MAD_TO_SIGMA * mad + EPS)


def meanstd_standardize(x: np.ndarray) -> np.ndarray:
    """y = (x - mean) / (std + eps) with **ddof = 0**. The pre-declared ablation.

    ddof is stated because it is a real fork: numpy defaults to the population form
    (ddof=0) and MATLAB's `std` to the sample form (ddof=1). No reference constrains
    the choice -- the reference never used mean/std scaling at all -- so the population
    convention is frozen here and pinned by the hand-computation test.
    """
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std(ddof=0) + EPS)


def standardize(x: np.ndarray, method: str) -> np.ndarray:
    if method == "robust":
        return robust_standardize(x)
    if method == "meanstd":
        return meanstd_standardize(x)
    raise StandardizeError(f"unknown standardize method {method!r}")


def to_channels(signal: np.ndarray, channel: str, method: str) -> np.ndarray:
    """Complex signal -> real standardized channels, shape [n_channels x n_samples].

      * `mag` -> |s|, standardized                                  -> [1, N]
      * `iq`  -> real and imag, standardized SEPARATELY (each from   -> [2, N]
                 its own statistics, never a shared scale)

    Which channel is used is an inner-CV choice at M6, so both are built the same way
    and neither is preferred here.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise StandardizeError(f"expected a 1-D signal, got shape {signal.shape}")

    if channel == "mag":
        parts = [np.abs(signal)]
    elif channel == "iq":
        parts = [signal.real, signal.imag]
    else:
        raise StandardizeError(f"unknown channel {channel!r} (expected 'mag' or 'iq')")

    return np.stack([standardize(part, method) for part in parts])
