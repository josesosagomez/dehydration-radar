"""Experiment D's frozen CNN architectures and per-frame input constructions.

Deterministic given (weights, input). There is **no CV logic here** — no fold, no seed
schedule, no selection. The one genuinely fitted quantity in this file is
`SpectrogramNorm`, and it is fitted only where its caller (`eval/exp_d.py`) hands it
training rows; everything else (the robust median/MAD z) is per-signal and therefore
unfitted, which is why it can live outside the CV loop at all (plan §5 trap 12).

**Where the numbers come from.** Every architecture/STFT constant below is the frozen
`BaselineConfig` / `configs/baselines.yaml` value (`implementation_plan.md` §D (i)/(ii),
A-M6-2, O-M9-6). They are written here as literals because the frozen §2.7 API takes no
config (`Cnn1d(in_channels)`, `spectrogram(x_1d)`); `assert_frozen_constants` compares the
two independently-written sources fail-closed at run time, and `test_cnn.py` pins them.

**Which signal each variant consumes** — the part that is easy to get wrong, because "raw"
and "matched" differ here. `BaselineConfig`'s two normalization rules
(`raw_matched_standardize: robust_per_channel` vs
`spectrogram_standardize: train_only_per_frequency_mean_std`) do NOT mean spectrograms
never see robust standardization; they mean the *time-domain* robust step belongs to the
raw/matched input definitions while the *spectral* per-frequency step is the fitted one. So:

  * 10 GHz primary  = STFT of the raw complex 534 beat, **no** robust step, real/imag
    separately -> 2 channels (O-M9-6);
  * 10 GHz ablation = STFT of the stored `[2, 470]` matched I/Q, which is ALREADY
    robust-standardized in the store (`preprocess/pipeline.py:73`) — nothing is applied at
    load, and a second standardization would not be the frozen matched signal;
  * 77 GHz primary  = STFT of the raw real 256 slow-time series, **no** robust step -> 1 ch;
  * 77 GHz ablation = STFT of `matched_input_77`'s output, i.e. the robust per-channel z
    applied at load, because the 77 GHz store deliberately keeps that tensor
    pre-standardization while `implementation_plan.md:877-892` defines the matched input as
    robust-standardized.

The physics baseline is unaffected: it reads `sig__raw_beat` / `sig__raw_slowtime`, which
carry no standardization at all (trap 13).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal.windows import hann
from torch import nn

from ..preprocess.standardize import robust_standardize, to_channels

# ------------------------------------------------------------------- frozen constants

CNN1D_CHANNELS = (16, 32, 64)
CNN1D_KERNEL = 7
CNN1D_POOL = 4
CNN2D_CHANNELS = (16, 32)
CNN2D_KERNEL = 3
CNN2D_POOL = 2

SPECTROGRAM_HANN = 64
SPECTROGRAM_HOP = 16
SPECTROGRAM_NFFT = 128

# The log guard. A pure REPRESENTABILITY floor against log(0) — data-independent, with no
# magnitude chosen by anyone. A relative floor would make the transform data-dependent and
# is rejected; `1e-30` is neither the float64 tiny nor a principled level and would clip
# genuinely low-energy bins (plan §2.7).
SPECTROGRAM_EPS = float(np.finfo(np.float64).tiny)

# PERIODIC Hann, matching this repo's own split: the chain window is symmetric
# (`preprocess/pipeline_77.py:24`, MATLAB `hann(N)`) while every *spectral analysis* window
# is periodic (`qc/screens.py:147-154`, `qc/axis_check_77.py:60`). An STFT is spectral
# analysis, and the periodic window is also the standard STFT convention.
_HANN_WINDOW = hann(SPECTROGRAM_HANN, sym=False)

CNN_FAMILIES = ("cnn1d_raw", "cnn1d_matched", "spec2d_raw", "spec2d_matched")


class CnnError(ValueError):
    """A drifted frozen constant, or a family with no architecture."""


class CnnInputError(ValueError):
    """An input array that does not match the stored signal contract for its variant."""


def assert_frozen_constants(baselines) -> None:
    """Fail closed if this module's literals and `BaselineConfig` ever disagree.

    Two separately written sources compared against each other (the doctrine
    `exp_c.FRANK_HALL_MAX_ITER` uses), so a config edit cannot silently change the network
    while the code keeps the old shape, or the reverse.
    """
    for name, here, there in (
        ("cnn_channels", CNN1D_CHANNELS, tuple(baselines.cnn_channels)),
        ("cnn_kernel", CNN1D_KERNEL, baselines.cnn_kernel),
        ("cnn_pool", CNN1D_POOL, baselines.cnn_pool),
        ("cnn2d_channels", CNN2D_CHANNELS, tuple(baselines.cnn2d_channels)),
        ("cnn2d_kernel", CNN2D_KERNEL, baselines.cnn2d_kernel),
        ("cnn2d_pool", CNN2D_POOL, baselines.cnn2d_pool),
        ("spectrogram_hann", SPECTROGRAM_HANN, baselines.spectrogram_hann),
        ("spectrogram_hop", SPECTROGRAM_HOP, baselines.spectrogram_hop),
        ("spectrogram_nfft", SPECTROGRAM_NFFT, baselines.spectrogram_nfft),
    ):
        if here != there:
            raise CnnError(
                f"frozen constant {name} is {here!r} in models/cnn.py but {there!r} in "
                "BaselineConfig — one of the two drifted; refusing to build a network"
            )


# --------------------------------------------------------------------- architectures


class Cnn1d(nn.Module):
    """3 x (Conv1d(k=7, stride 1) -> BatchNorm1d -> ReLU -> MaxPool1d(4)), channels
    16/32/64 -> global average pool over time -> Linear -> one scalar per frame.

    Convolutions are unpadded (PyTorch's default): the freeze names the kernel, the stride
    and the pool and says nothing about padding, so the literal reading is padding 0 —
    'same' padding would be an unstated choice of its own.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        blocks: list[nn.Module] = []
        c_in = int(in_channels)
        for c_out in CNN1D_CHANNELS:
            blocks += [
                nn.Conv1d(c_in, c_out, CNN1D_KERNEL, stride=1),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(CNN1D_POOL),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(CNN1D_CHANNELS[-1], 1)

    def forward(self, x):                       # x: [batch, channels, samples]
        return self.head(self.features(x).mean(dim=2)).squeeze(-1)


class Cnn2d(nn.Module):
    """2 x (Conv2d(3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)), channels 16/32 ->
    global average pool over (frequency, time) -> Linear -> one scalar per frame."""

    def __init__(self, in_channels: int):
        super().__init__()
        blocks: list[nn.Module] = []
        c_in = int(in_channels)
        for c_out in CNN2D_CHANNELS:
            blocks += [
                nn.Conv2d(c_in, c_out, CNN2D_KERNEL, stride=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(CNN2D_POOL),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(CNN2D_CHANNELS[-1], 1)

    def forward(self, x):                       # x: [batch, channels, frequency, time]
        return self.head(self.features(x).mean(dim=(2, 3))).squeeze(-1)


def build_network(family: str, in_channels: int) -> nn.Module:
    """The family -> architecture map. `spec2d_*` are the 2-D spectrogram nets, `cnn1d_*`
    the 1-D time-domain nets; nothing else has an Exp D architecture."""
    if family in ("cnn1d_raw", "cnn1d_matched"):
        return Cnn1d(in_channels)
    if family in ("spec2d_raw", "spec2d_matched"):
        return Cnn2d(in_channels)
    raise CnnError(f"no Exp D architecture for family {family!r} (expected one of {CNN_FAMILIES})")


# ------------------------------------------------------------------- 1-D input builders


def raw_beat_input_10(sig) -> np.ndarray:
    """`sig__raw_beat`: one complex 534-sample chirp-mean beat -> `[2, N]` {real, imag},
    each robust-standardized from its OWN statistics (nothing shared across frames).

    `to_channels(..., "iq", "robust")` is reused rather than reimplemented so the CNN's raw
    input is literally the same robust z the preprocessing chain applies at its step 7.
    """
    sig = np.asarray(sig)
    if sig.ndim != 1 or not np.iscomplexobj(sig):
        raise CnnInputError(
            f"raw_beat_input_10 expects a 1-D complex beat signal, got shape {sig.shape} "
            f"dtype {sig.dtype}"
        )
    return to_channels(sig, "iq", "robust")


def matched_input_10(sig) -> np.ndarray:
    """`sig__matched_iq` (10 GHz): the stored `[2, 470]` matched I/Q, returned untouched.

    It IS `preprocess_cube(..., channel="iq")`, whose final step is already the robust
    per-channel z (`preprocess/pipeline.py:73`), so standardizing again here would produce
    something that is not the frozen matched signal.
    """
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[0] != 2:
        raise CnnInputError(
            f"matched_input_10 expects the stored [2, N] matched I/Q, got shape {sig.shape}"
        )
    return sig.copy()


def raw_input_77(sig) -> np.ndarray:
    """`sig__raw_slowtime`: one real 256-sample slow-time series (mean over fast time and
    Rx, A-M6-2 (i)) -> `[1, N]`, robust-standardized per signal."""
    sig = np.asarray(sig)
    if sig.ndim != 1 or np.iscomplexobj(sig):
        raise CnnInputError(
            f"raw_input_77 expects a 1-D real slow-time signal, got shape {sig.shape} "
            f"dtype {sig.dtype}"
        )
    return robust_standardize(sig)[None, :]


def matched_input_77(sig) -> np.ndarray:
    """`sig__matched_iq` (77 GHz): the Rx-0 gate-mean slow-time series -> `[2, N]`
    {real, imag}, robust-standardized **per channel at load**.

    Unlike the 10 GHz matched signal, the 77 GHz store keeps this tensor
    pre-standardization (§2.9) while `implementation_plan.md:877-892` defines the matched
    77 GHz input as robust-standardized per channel — so the step happens here. Accepts
    either the stored `[2, N]` real/imag pair or the equivalent complex `[N]` series.
    """
    sig = np.asarray(sig)
    if sig.ndim == 1 and np.iscomplexobj(sig):
        parts = (sig.real, sig.imag)
    elif sig.ndim == 2 and sig.shape[0] == 2 and not np.iscomplexobj(sig):
        parts = (sig[0], sig[1])
    else:
        raise CnnInputError(
            "matched_input_77 expects the stored [2, N] real/imag pair or a complex [N] "
            f"series, got shape {sig.shape} dtype {sig.dtype}"
        )
    return np.stack([robust_standardize(p) for p in parts])


# ------------------------------------------------------------------------ the STFT


def _stft_log_magnitude(x: np.ndarray) -> np.ndarray:
    """One real 1-D signal -> `[frequency, time]` log-magnitude STFT.

    Hann 64 / hop 16 / nfft 128, so frames start at 0, 16, 32, ... and the frequency axis is
    the non-redundant `nfft//2 + 1 = 65` real-FFT bins. LITERAL log-magnitude
    `log(|STFT| + eps)`: an earlier draft's `log(|STFT|^2 + 1e-30)` is log-*power*, a factor
    2 away from the frozen wording at `implementation_plan.md:821-825`.
    """
    n = int(x.shape[0])
    if n < SPECTROGRAM_HANN:
        raise CnnInputError(
            f"signal of length {n} is shorter than the frozen Hann window {SPECTROGRAM_HANN}"
        )
    n_frames = (n - SPECTROGRAM_HANN) // SPECTROGRAM_HOP + 1
    frames = np.stack(
        [x[s : s + SPECTROGRAM_HANN] for s in range(0, n_frames * SPECTROGRAM_HOP, SPECTROGRAM_HOP)]
    )
    coefficients = np.fft.rfft(frames * _HANN_WINDOW, n=SPECTROGRAM_NFFT, axis=1)
    return np.log(np.abs(coefficients) + SPECTROGRAM_EPS).T


def spectrogram(x) -> np.ndarray:
    """`[channel, frequency, time]` log-magnitude spectrogram of a per-frame signal.

    A complex 1-D input has its real and imag parts transformed **separately and stacked**
    (O-M9-6, mirroring A-M6-2's own convention for the 77 GHz complex ablation); a real 1-D
    input stays 1-channel; a `[C, N]` input is transformed channel by channel. The output is
    asserted finite — a non-finite value here would otherwise reach BatchNorm and poison a
    whole fit silently.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        channels = (x.real, x.imag) if np.iscomplexobj(x) else (x,)
    elif x.ndim == 2 and not np.iscomplexobj(x):
        channels = tuple(x)
    else:
        raise CnnInputError(
            f"spectrogram expects a 1-D (real or complex) or [C, N] real signal, got shape "
            f"{x.shape} dtype {x.dtype}"
        )
    # C-contiguous, deliberately: `_stft_log_magnitude` returns a transposed (F-ordered)
    # view, and `np.stack` of those yields a non-C-contiguous array whose numpy reductions
    # sum in a different ORDER — so `SpectrogramNorm` fit on bytewise-identical data could
    # differ in the last bits purely by memory layout, which would break the bit-identity
    # claims this project's mutation tests rest on.
    out = np.ascontiguousarray(
        np.stack([_stft_log_magnitude(np.asarray(c, dtype=np.float64)) for c in channels])
    )
    if not np.all(np.isfinite(out)):
        raise CnnInputError("spectrogram produced a non-finite value — check the input signal")
    return out


# ------------------------------------------------ per-variant spectrogram input builders


def spec_input_10_raw(sig) -> np.ndarray:
    """10 GHz primary: STFT of the RAW complex beat — no robust step (§2.7)."""
    return spectrogram(sig)


def spec_input_10_matched(sig) -> np.ndarray:
    """10 GHz ablation: STFT of the stored matched I/Q, already robust-standardized in the
    store, with nothing applied at load."""
    return spectrogram(matched_input_10(sig))


def spec_input_77_raw(sig) -> np.ndarray:
    """77 GHz primary: STFT of the RAW real slow-time series — no robust step, 1 channel."""
    return spectrogram(sig)


def spec_input_77_matched(sig) -> np.ndarray:
    """77 GHz ablation: STFT of `matched_input_77`'s robust-standardized output."""
    return spectrogram(matched_input_77(sig))


# (band, family) -> (store v2 signal key, per-frame input builder). ONE table, so "which
# signal does this variant consume" has a single answer that a test can read.
FRAME_INPUT = {
    ("10ghz", "cnn1d_raw"): ("sig__raw_beat", raw_beat_input_10),
    ("10ghz", "cnn1d_matched"): ("sig__matched_iq", matched_input_10),
    ("10ghz", "spec2d_raw"): ("sig__raw_beat", spec_input_10_raw),
    ("10ghz", "spec2d_matched"): ("sig__matched_iq", spec_input_10_matched),
    ("77ghz", "cnn1d_raw"): ("sig__raw_slowtime", raw_input_77),
    ("77ghz", "cnn1d_matched"): ("sig__matched_iq", matched_input_77),
    ("77ghz", "spec2d_raw"): ("sig__raw_slowtime", spec_input_77_raw),
    ("77ghz", "spec2d_matched"): ("sig__matched_iq", spec_input_77_matched),
}


# ------------------------------------------------------- the one fitted input transform


class SpectrogramNorm:
    """Train-only per-`(channel, frequency)` mean/std of the spectrogram tensor.

    'Per-frequency' is not executable on a `[frame, channel, frequency, time]` tensor
    without naming the axes, so they are named: statistics are kept per `(channel,
    frequency)` — parameter shape `[C, F]` — and reduced over **frames x time** of the
    training frames only. Per channel rather than shared across channels because the real
    and imag parts of a complex beat have genuinely different per-frequency scales, and
    sharing would let one channel's statistics standardize the other.

    The zero-variance fallback is `scale = 1.0`, not `std + tiny`: a constant training bin
    (a dead frequency bin, a padded edge — both plausible) would otherwise amplify any
    differing test value by ~1e308. `np.where(std == 0.0, 1.0, std)` is already this repo's
    convention for exactly this case (`models/torch_fit.py::_normalize_stats`), so this is a
    reuse rather than a new rule. The count of substituted cells travels in `params()` so a
    fold where the fallback fired is visible in the fit audit.
    """

    def __init__(self, mean: np.ndarray, scale: np.ndarray, n_zero_variance_cells: int):
        self.mean = mean
        self.scale = scale
        self.n_zero_variance_cells = int(n_zero_variance_cells)

    @classmethod
    def fit(cls, x_train) -> SpectrogramNorm:
        # contiguous, so the reduction order (and hence the last bits) is a function of the
        # DATA and not of how the caller happened to slice it — see `spectrogram`.
        x = np.ascontiguousarray(x_train, dtype=np.float64)
        if x.ndim != 4:
            raise CnnInputError(
                f"SpectrogramNorm expects [frames, channels, frequency, time], got {x.shape}"
            )
        mean = x.mean(axis=(0, 3))
        std = x.std(axis=(0, 3), ddof=0)
        scale = np.where(std == 0.0, 1.0, std)
        return cls(mean, scale, int(np.count_nonzero(std == 0.0)))

    def transform(self, x) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.ndim != 4:
            raise CnnInputError(
                f"SpectrogramNorm expects [frames, channels, frequency, time], got {x.shape}"
            )
        return (x - self.mean[:, :, None]) / self.scale[:, :, None]

    def params(self) -> dict:
        """The fit-record payload. The count is stored as a 0-d int array so every value in
        a `FitRecord.params` dict is a numpy array and the audit's bytewise comparisons work
        uniformly across quantities."""
        return {
            "mean": self.mean.copy(),
            "scale": self.scale.copy(),
            "n_zero_variance_cells": np.asarray(self.n_zero_variance_cells, dtype=np.int64),
        }


def flattened_input_dimension(x) -> int:
    """The per-frame input size (`C*N` or `C*F*T`) — the harness's `feature_dimension` for
    the Exp D tie-break. Constant across the 6 configs of a family, so it can never decide a
    comparison; recorded because `select_candidate`'s frozen key includes it."""
    return int(np.prod(np.asarray(x).shape[1:]))


def torch_module_state_to_numpy(state) -> dict:
    """`state_dict` -> plain numpy, for the `FitRecord`. Keeps BatchNorm's running mean /
    variance / `num_batches_tracked` as well as the weights — they are fitted quantities of
    the training rows and belong in the audit."""
    return {k: v.detach().cpu().numpy().copy() for k, v in state.items()}


def enable_gpu_determinism(device) -> None:
    """GPU kernels are not bit-deterministic; ask for deterministic algorithms where they
    exist and let the rest be covered by per-seed reporting (plan §0, §5 trap 10). No
    cross-run bit-identity is claimed on GPU — every bit-assert in this project is CPU."""
    if torch.device(device).type != "cpu":
        torch.use_deterministic_algorithms(True, warn_only=True)
