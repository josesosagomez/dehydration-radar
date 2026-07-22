# MILESTONE 3 PLAN — Preprocessing: the executable sequence

_Task-level execution plan for milestone 3 **only** (ROADMAP §7.3; implementation_plan.md
"Build order" §3)._

_**Status: APPROVED AND IMPLEMENTED (2026-07-23, branch `v1_milestone_3`).** All seven
build steps executed; definition of done §4 D1–D7 met in full — `uv run pytest` →
319 passed / 11 skipped, `--realdata` → 329 passed / 1 skipped (T18), and
`tests/test_no_leakage.py` is byte-for-byte unmodified since M1 and green. This
document is now a **record of what was built and why**, not a proposal; see HISTORY.md
for the per-step log and the four facts discovered during the build (the
arithmetic-vs-geometric mid-band ambiguity, the finite-record energy reality, the
windowed-vs-unwindowed zero-ROI counterexample, and the measured 1.50 m target range —
none of which changed a frozen parameter)._

_Review rounds 2026-07-22/23: **4 rounds, 15 comments, all applied, none disputed** —
strict model-band Nyquist rule; Option-B mask anti-vacuity guard and nb boundary
battery; `OptionBDetection` as the single canonical detection result; frozen diagnostic
formulas; the three-way classification of every preprocessing alternative; padlen frozen
by explicit passing; mean/std ddof; the quadratic-energy mixture expectation; the full
canonical-spec artifact guard; the mode tie rule; frozen protocol constants; and the
removal of the false zero-ROI output claim. Amendments A-M3-6 and A-M3-7 were propagated
to `plans/implementation_plan.md` during review; A-M3-1..A-M3-5 at milestone close._

This document adds the execution detail the main plan intentionally omits. It does not
restate design decisions; where a decision is needed it cites
`plans/implementation_plan.md` (the source of truth — for M3 chiefly §"Preprocessing —
executable sequence", §"Deliberate departures from the reference" and Build order §3),
`CLAUDE.md`, or `ROADMAP.md`. Anything here that goes beyond those documents is flagged
in §6.

Milestones 1–2 are done, committed and pushed (`f3fbade`, `395eb62`): config system,
10 GHz loader, ground truth + cross-checks, manifest, frozen QC screens, session
eligibility, nested-LOSO splitter, provenance, the 77 GHz audit — 260 tests green
(269 with `--realdata`). M3 builds on those components and duplicates none of them.

**The input population is fixed by M2:** `eligible_frames(manifest_qc)` = **7168
QC-passed frames across 73 eligible sessions and 16 evaluable subjects**. M3's
functions are population-agnostic pure functions; the wiring of the eligible-frame
population into feature extraction happens at M4 — M3 touches real data through its
`--realdata` tests and the cohort diagnostic run (§2.6).

**Owner decisions already made (recorded here so they are not re-litigated):**

1. **M3 work happens on branch `v1_milestone_3`** (already created from
   `v1_milestone_2` and checked out; HANDOFF.md is its bootstrap).
2. **The cohort-level diagnostic script `experiments/run_preprocess.py` is IN scope**
   (owner decision 2026-07-22): frozen preprocessing meets the full eligible cohort
   once, findings recorded — the same first-contact pattern as `run_qc.py` at M2.

---

## §0 Scope and ground rules

**In scope:**

- `src/dehyd/preprocess/__init__.py`, `filters.py`, `reduce.py`, `standardize.py`,
  `pipeline.py` — the executable sequence of implementation_plan.md §"Preprocessing"
  steps 3–7 (§2.2–§2.5).
- New `PreprocessConfig` fields + field-level and cross-field validation in
  `src/dehyd/config.py`, mirrored in `configs/preprocess.yaml` (§2.1).
- `experiments/run_preprocess.py` — one-command cohort diagnostic pass; writes a
  curated artifact under `<results_dir>/preprocess/` (§2.6).
- `tests/test_preprocess.py` (synthetic, private-data-free) + extensions to
  `tests/test_config.py` + `realdata` one-file pipeline tests (§3).
- Journal upkeep: HISTORY.md as steps/attempts resolve (departures logged);
  SECOND_CHAPTER.md §2 "Preprocessing" at milestone close.

**Explicitly out of scope (deferred to their milestones):**

- WST / kymatio feature extraction (`features/wst.py`, `features/pooling.py`) — **M4**.
- torch (enters the environment at M4 for the cross-backend WST test; the T18 torch
  mutation leg stays skip-marked until M6 — owner decision, HANDOFF §Do-NOT-re-litigate).
- `eval/harness.py`, `eval/metrics.py`, any modeling — **M6+**.
- 77 GHz preprocessing (`loader_77ghz.py`, MTI, range FFT) — **M9**. But §2.2's filter
  functions are written shape/fs-agnostic precisely so M9 reuses them (the 77 GHz
  chain step 2 is the same Butterworth zero-phase bandpass at fs = 500 kHz, N = 256).
- `configs/ibex.yaml`, `scripts/ibex/` — first IBEX milestone. M3 is entirely
  CPU-local: the heaviest operation (cohort diagnostic, §2.6) is ≈ 1–3 min dominated
  by MAT decompression, same order as `run_qc.py`.
- **No caching of preprocessed signals.** M4 recomputes preprocessing on the fly per
  run — the full-cohort cost is minutes, and a cache with hash invalidation is
  complexity deferred until it is actual friction (same reasoning as the M2 no-QC-cache
  decision).

**Environment: no new dependencies.** `scipy` 1.16.3 (pinned `<1.17`) already provides
`butter`, `sosfiltfilt`, `sosfreqz`, `windows.hann`. No torch, no new packages;
`uv.lock` unchanged.

**The milestone-3 invariant, protected above all (CLAUDE.md §Hard invariants;
implementation_plan.md §Preprocessing, §Deliberate departures):**

> **Preprocessing is a deterministic per-frame function of (one frame, frozen config).
> It contains NO fitted quantities and NO cross-frame, cross-session, cross-subject,
> or cross-role statistics.**

Concretely: the only data-dependent numbers anywhere in the sequence are computed
**within one signal from itself** — Option B's peak bin from that frame's own chirps,
and the robust z from that signal's own median/MAD. Per-signal normalization is not a
fitted transform (nothing is estimated on one set and applied to another), so
preprocessing introduces **no leakage vector** and nothing here enters the CV loops.
Where this could silently break, and is forbidden to:

- standardizing with statistics pooled over a session, subject, or the dataset
  (that would be a fitted scaler and would have to move inside the CV harness — it is
  not what the plan specifies; the per-signal form is frozen);
- selecting Option B's peak bin from any frame other than the one being reduced
  (e.g. a session-median peak bin) — detection is per frame, from that frame's own
  20 chirps;
- letting any filter parameter adapt to data (the band comes from config; the design
  is a pure function of frozen constants).

The choice axes that ARE searched later — reduction {A, B}, channel {mag, iq}, model
gate {1–2 m, 0.9–3.0 m} — are **explicit function arguments / config alternatives**,
selected only inside inner CV at M6. M3 implements the alternatives; it never picks
between them based on data.

**Every preprocessing alternative is classified NOW, before M6 (and locked at the M5
config freeze).** Two kinds, and nothing may migrate between them after results exist:

- **Inner-CV search axes** (selected per outer fold on inner folds only): reduction
  {A, B} × channel {mag, iq} × model gate {1–2 m, 0.9–3.0 m} — exactly the main
  plan's search space, no additions.
- **Pre-declared ablations** (never inner-CV candidates, can never displace the
  primary, reported only as explicitly-labeled ablation results):
  `gate_method: fft` (the primary is `butterworth`; the main plan offers the FFT
  gate "for ablation") and `standardize: meanstd` (the primary is `robust`). A run
  using either carries the ablation label in its provenance; an attractive
  full-cohort ablation number can therefore never become unrecorded test-set tuning.
- **Frozen protocol constants** (a third class — neither searched nor ablated):
  `peak_neighbors = 1` and `mask_taper = true` (the main plan's "±1-bin two-sided
  Hann-tapered mask" *is* these two values) and `fft_gate_transition_hz = 500.0`
  (the FFT ablation is one fixed variant — its transition is not itself varied).
  They are config fields only so a run's YAML is a complete record and so tests can
  drive boundary behavior: **non-default values are exercised only inside tests to
  validate function behavior** (e.g. T-PP12's nb battery) and are **rejected by
  every modeling/artifact entrypoint** — at M3 by the §2.6 full-canonical-spec
  guard, from M6 by the harness validating these constants at their frozen values
  before any outer-fold evaluation, with the whitelist recorded at the M5 config
  freeze (main-plan amendment A-M3-7). A supported low-level function argument is
  not, by itself, a licensed experimental axis.

`tests/test_no_leakage.py` stays **byte-for-byte unmodified since M1** and green
throughout.

**Ground rules:** work on `v1_milestone_3`; commits only when the owner asks — build
steps *write and verify* artifacts, they never commit them; HISTORY.md written
continuously as attempts resolve (failures kept, newest-first); superseded material to
`archive/` with a note.

---

## §1 Build sequence — exact order and why

Tests land in the same step as their module. HISTORY.md gets **at least** one entry per
resolved step; every failed or superseded attempt inside a step gets its own entry.

| # | Step | Why this position |
|---|------|-------------------|
| 1 | Config: new `PreprocessConfig` fields (`gate_method`, `fft_gate_transition_hz`, `peak_neighbors`, `mask_taper`, `standardize`) + field-level and cross-field validation (§2.1) + `configs/preprocess.yaml` mirror + `tests/test_config.py` extensions | The modules read these; schema and validation first, exactly as M1/M2 did |
| 2 | `src/dehyd/preprocess/filters.py` + filter tests | Pure functions, no I/O; everything downstream consumes filtered chirps |
| 3 | `src/dehyd/preprocess/reduce.py` + reduction/trim tests | Consumes step 2's output shape; Option B needs the band arithmetic pinned in step 1 |
| 4 | `src/dehyd/preprocess/standardize.py` + tests | Last per-signal stage; independent of 2–3 but ordered by the sequence it implements |
| 5 | `src/dehyd/preprocess/pipeline.py` (the executable sequence as one linear function) + composition tests + the one-file `--realdata` test | Only meaningful once 2–4 exist; the composition test is the guard against silent reordering |
| 6 | `experiments/run_preprocess.py`; **run on the real eligible cohort**; write and verify the diagnostics artifact; record actual numbers in HISTORY.md | First contact of the frozen sequence with all 7168 frames; produces the §2.6 findings SECOND_CHAPTER §2 needs |
| 7 | Journal close-out: SECOND_CHAPTER.md §2 "Preprocessing"; final HISTORY.md entry; §6 amendments applied to `plans/implementation_plan.md` | CLAUDE.md write-cadence rules; closing the milestone requires the distilled account |

---

## §2 Per-file specifications

Format per file: **Responsibility** (single) · **Public API** · **Frozen parameters** ·
**Acceptance criteria**.

### 2.1 Config: `src/dehyd/config.py` + `configs/preprocess.yaml`

**Responsibility.** The M3-consumed constants, validated at build time — M2's rule:
once a field is actually consumed, a bad value must fail loudly at config load, not
surface as a confusing numpy error inside a filter.

**New `PreprocessConfig` fields (defaults are the frozen values):**

```python
gate_method: str = "butterworth"      # "butterworth" (primary) | "fft" (ablation)
fft_gate_transition_hz: float = 500.0 # FFT-gate Hann skirt width (filter_gpt_fft.m tw_hz)
peak_neighbors: int = 1               # Option B: ±bins kept around the detected peak
mask_taper: bool = True               # Option B: Hann-tapered vs rectangular mask
standardize: str = "robust"           # "robust" (median/MAD, primary) | "meanstd"
```

`configs/preprocess.yaml` mirrors all five with provenance comments (reference values
from `filter_gpt_fft.m` and `wst_extract.m`; the robust/meanstd switch is the
main-plan departure's "configurable; plain mean/std available"). The comments state
each field's §0 classification: `gate_method` and `standardize` are **ablation
switches** (non-default values are pre-declared ablations only, never inner-CV
axes); `peak_neighbors`, `mask_taper`, and `fft_gate_transition_hz` are **frozen
protocol constants** (non-default values are test-only and rejected by modeling/
artifact entrypoints).

**Field-level validation (build-time, `ConfigError` on violation):**

- `gate_method` ∈ {"butterworth", "fft"} (string; anything else rejected);
- `fft_gate_transition_hz` > 0, finite;
- `peak_neighbors` integer ≥ 0, not bool (0 is valid: keep only the peak bin);
- `mask_taper` strictly bool (not 0/1 integers — YAML supplies real booleans);
- `standardize` ∈ {"robust", "meanstd"}.

**Cross-field validation (new, mirrors `_check_qc_band`):**

- **Model band ENTIRELY below Nyquist:** `beat_band_hz(model_gate_m, ...)` must
  satisfy `0 < f_lo < f_hi < fs/2` strictly — for **both** gate methods. This is
  deliberately stricter than the QC-gate check (which only rejects a band *starting*
  at/above Nyquist, because the QC screen is an FFT mask whose upper edge is
  legitimately Nyquist-clamped — semantics frozen at M2). The Butterworth design
  raises inside `butter` on `Wn ≥ 1`, so a straddling gate (lower edge below, upper
  edge at/above Nyquist) would otherwise pass config loading and fail deep in the
  filter — exactly what the fail-at-config-load rule forbids; and silently
  Nyquist-truncating the *model* band for the FFT path would make the two gate
  methods filter different bands under one config, so the same strict rule applies
  to both.
- **`model_gate_m ⊆ qc_gate_m`.** The QC gate was frozen *wider* precisely so the
  QC-passing population is identical for every model-gate candidate
  (implementation_plan.md §"One fixed QC range gate"). A model gate outside the QC
  gate would select model energy in a band QC never guaranteed — a silent breach of
  that design, therefore a hard `ConfigError`, not a warning.
- **FFT-gate non-vacuity:** when `gate_method: fft`, the tapered mask must not cover
  the whole spectrum — require `f_lo − transition > 0` **or**
  `f_hi + transition < fs/2` (otherwise the "gate" passes everything; a silently
  disabled filter is a config error, same doctrine as the QC vacuity guards).

**Not config-validated (needs the fast-time length, which is a loader constant):**
the `edge_trim`-vs-length guard lives in the trim function (§2.3): it **raises** if
`2·edge_trim ≥ N` or the effective length falls below 32 (the reference's own floor)
— never clamps, because the reference's silent `min(EdgeTrim, N/4)` clamp would hide
a config error.

**Acceptance.** `tests/test_config.py`: defaults present; YAML overrides honored;
every bound/type above rejected; both cross-field checks fire (model gate above
Nyquist; model gate ⊄ QC gate; whole-spectrum FFT mask); provenance
(`config_to_dict`) carries the new fields automatically.

### 2.2 `src/dehyd/preprocess/filters.py`

**Responsibility.** The band gate — implementation_plan.md sequence step 3. Pure,
**shape/fs-agnostic** functions (M9 reuses them at fs = 500 kHz, N = 256): design,
zero-phase application, and the FFT-gate ablation alternative. No I/O, no pandas,
no state.

**Public API.**

```python
def design_bandpass_sos(f_lo_hz, f_hi_hz, fs_hz, order) -> np.ndarray
    # scipy.signal.butter(order, [f_lo, f_hi]/(fs/2), btype='bandpass',
    # output='sos'). order=4 -> 8 poles -> sos shape (4, 6), matching MATLAB
    # butter(4, ..., 'bandpass'). RAISES if f_hi >= fs/2 or f_lo <= 0.

def bandpass_filtfilt(x, sos, axis) -> np.ndarray
    # Zero-phase sosfiltfilt along `axis`. Complex input is filtered as
    # real and imag parts SEPARATELY through the identical real SOS filter
    # (mathematically equal to filtering the complex signal; kept explicit per
    # the main plan: "filter real and imag"). padtype='odd' — FROZEN.
    # padlen is FROZEN BY EXPLICIT PASSING: default_padlen(sos) (below) is
    # computed once from the designed sos and passed explicitly, so the padding
    # is pinned by our code, not by a library default that could drift.

def default_padlen(sos) -> int
    # scipy's documented sosfiltfilt default, computed from the sos alone (a
    # pure function of the design — no signal needed):
    # 3 * (2*n_sections + 1 - min(#zero b2 coeffs, #zero a2 coeffs)).
    # For the default-config design (order 4, 4 sections, scipy 1.16.3) this is
    # 27 — VERIFIED 2026-07-22: passing padlen=27 explicitly is bit-identical
    # to the library default. A regression test pins both facts (T-PP1), so a
    # scipy behavior change is caught loudly instead of silently changing edges.

def fft_gate(x, f_lo_hz, f_hi_hz, fs_hz, transition_hz, axis) -> np.ndarray
    # The filter_gpt_fft.m alternative: per-signal FFT along `axis`, tapered
    # passband mask symmetric in |f| (raised-cosine skirts of width
    # transition_hz; passband 1; stopband 0), IFFT. Built directly in
    # UNSHIFTED bin space (no fftshift round-trip — same mask, less code).

def filter_spec(pre: PreprocessConfig) -> dict
    # The recorded design: gate_method, model_gate_m, f_lo/f_hi (via
    # beat_band_hz — REUSED from config.py, never re-derived), Wn, order,
    # padtype, and the frozen explicit padlen = default_padlen(designed sos)
    # (derivable from config alone — no execution context needed), fft
    # transition. Goes into provenance extras (run_preprocess.py) and
    # HISTORY.md.
```

**Frozen semantics.**

- **`Wn` comes from the MODEL gate** (`preprocess.model_gate_m`, default 1–2 m →
  **3257.5–6514.9 Hz** at HzPerM ≈ 3257.5 Hz/m; Wn ≈ (0.01251, 0.02502)) — **never**
  the QC gate. The QC/model gate separation is the M2 design: QC fixed the population
  on 0.9–3.0 m; the model band is a config choice searched in inner CV.
- **No window before the time-domain filter.** The primary path applies no taper to
  the chirp — the logged ROADMAP §3.2 departure (windowing suppresses FFT leakage,
  which is irrelevant to a time-domain IIR filter, and would taper real signal energy
  at the chirp edges); filtfilt edge transients are handled by EdgeTrim instead.
- **Vectorization:** the cube path reshapes `[534 × 20 × N]` → `[534 × 20·N]` and
  calls `sosfiltfilt` once per real/imag part. A test asserts batched ≡ per-chirp
  (T-PP5), so the speedup can never silently change semantics.
- **Finite-record behavior ≠ steady-state design response (measured 2026-07-22,
  scipy 1.16.3, the exact planned design, padlen = 27).** The band is narrow
  (≈ 3.3 kHz at fs = 520.8 kHz), so the filter's transient occupies a large share of
  a 534-sample record: a mid-band (4886.2 Hz) complex exponential retains
  **0.7595** of its energy over the full chirp and **0.8313** after the 32-sample
  trim (the trim removing transient edges is empirical support for EdgeTrim; the
  residual shortfall is the narrowband ringing — expected, not a bug); a 50 kHz
  tone is attenuated **−17.2 dB** full-record and **−20.1 dB** after trim (finite
  records leak broadband edge energy into the passband, so steady-state stopband
  figures are unreachable). The steady-state `|H|²` from `sosfreqz` (T-PP2) and
  these finite-record numbers are **different claims, asserted by different
  tests** (T-PP2 vs T-PP6) and must never be conflated. **No filter parameter is
  changed to chase the steady-state ideal** — any such change would be an explicit,
  logged design decision, never a test-fixing tweak.

**Acceptance.** §3 filter tests green; the designed filter is stable (all poles
strictly inside the unit circle — low normalized Wn is numerically delicate, which is
exactly why the SOS form is mandatory); `filter_spec` reports every value from config,
none re-hardcoded.

### 2.3 `src/dehyd/preprocess/reduce.py`

**Responsibility.** Sequence steps 4–5: collapse 20 filtered chirps to one complex
534-sample signal per frame (Options A and B), then EdgeTrim. Pure functions.

**Public API.**

```python
def reduce_option_a(frame) -> np.ndarray
    # frame: complex [534 x 20] -> mean across the 20 chirps -> complex [534].

def option_b_roi_bins(pre: PreprocessConfig, n_fast: int) -> np.ndarray
    # Detection ROI: bins k of the non-negative half-spectrum (0..floor(N/2)-1,
    # DC in, Nyquist out — the QC convention) with k*df inside
    # beat_band_hz(model_gate_m). NO margin: the QC ±1000 Hz margin is a QC
    # constant and does not apply here (the reference wst_extract.m ROI has no
    # margin either). Default config: df = 520834/534 ≈ 975.34 Hz ->
    # ROI bins 4..6 (3901.4/4876.7/5852.1 Hz). The 0.9–3.0 m alternative gate
    # gives bins 4..10 (bin 3 = 2926.0 Hz misses 2931.7 Hz by 5.7 Hz).
    # RAISES if the ROI is empty (the reference errors too).

@dataclass(frozen=True)
class OptionBDetection:
    peak_bin: int          # argmax of `power` restricted to `roi_bins`
    power: np.ndarray      # THE periodic-Hann, chirp-averaged half-spectrum power
    roi_bins: np.ndarray   # the ROI bin indices the argmax was restricted to

def detect_option_b_peak(frame, pre: PreprocessConfig) -> OptionBDetection
    # The detection stage ALONE, returned as one structured result — THE single
    # source of everything detection-derived: reduce_option_b builds its mask
    # from .peak_bin, and the §2.6 diagnostics compute roi_to_total and
    # peak_share from the SAME result's .power/.roi_bins (and the realdata test
    # consumes it too) — never a re-implementation, never a second FFT with a
    # subtly different convention (T-PP11 asserts all reported values derive
    # from one result).

def reduce_option_b(frame, pre: PreprocessConfig) -> np.ndarray
    # Per implementation_plan.md step 4b, exact semantics below; mask centred
    # on detect_option_b_peak(frame, pre).peak_bin.

def edge_trim(signal, n_trim) -> np.ndarray
    # signal[n_trim : len-n_trim]; 534 - 2*32 = 470. RAISES (never clamps) if
    # 2*n_trim >= len(signal) or the result would be < 32 samples.
```

**Frozen Option-B semantics (each sub-step stated because the reference is buggy
here — see the departure below):**

1. **Detect:** per chirp, multiply by **periodic** Hann
   (`scipy.signal.windows.hann(534, sym=False)` — the QC/reference detection
   convention), 534-pt FFT, power on the half-spectrum; average the 20 per-chirp
   power spectra; **peak bin = argmax restricted to the ROI bins**. Zero detection
   power across the ROI (possible: QC guaranteed energy in 0.9–3.0 m, not in
   1–2 m) → `np.argmax` returns the **first ROI bin** — deterministic, in-ROI,
   documented. **No claim that the output is then zero:** detection power is
   Hann-windowed while the mask applies to the *unwindowed* FFT, and the
   frequency-domain periodic-Hann kernel [−¼, ½, −¼] can annihilate the windowed
   ROI bins while unwindowed bins under the mask stay nonzero (adversarial
   construction: unwindowed bins 3..7 = [1, 0, −1, −2, −3] makes windowed bins
   4..6 exactly zero, yet the mask around bin 4 retains nonzero bins 3 and 5).
   The frozen behavior is precisely "mask centred on the first ROI bin, applied
   to the unwindowed FFT, whatever that yields" — finite and deterministic,
   pinned by T-PP13's adversarial fixture; `peak_share` is NaN in this case
   because no detected peak exists (§2.6). No special-case zero branch is added:
   an exact-zero test on float spectra would be a discontinuous knife-edge, and
   the NaN `peak_share` already flags the situation. (The reference's `max` over
   a zeroed full half-spectrum would give the DC bin — ours stays in the ROI by
   construction.)
2. **Mask (the corrected "±1-bin two-sided Hann-tapered mask"):** in unshifted bin
   space, keep positive-frequency bins `peak−nb .. peak+nb` (clamped to
   `[0, N/2]`) with taper weights
   `w(k) = 0.5·(1 + cos(π·k/(nb+1)))` for offset k ∈ [−nb, +nb]
   — the **interior of `hann(2·nb+3, sym=True)`**; for nb = 1 that is
   **[0.5, 1.0, 0.5]** with full weight on the detected peak (nb = 0 degenerates
   to weight [1.0] on the peak bin alone; nb = 2 gives [0.25, 0.75, 1.0, 0.75,
   0.25]). Mirror every kept bin onto its conjugate `(N − k) mod N` with the same
   weight ("two-sided"); if a bin is its own mirror (DC or Nyquist — cannot occur
   at the default config, where positive bins are 3..7), take the **max** of the
   contributions, never the sum. `mask_taper: false` → rectangular weights 1 on
   the same bins.
   **Anti-vacuity guard (the `in_band_mask` doctrine applied here):** config
   validation cannot bound `peak_neighbors` because it does not know the
   fast-time length, so the guard lives where `n_fast` and the peak are known —
   mask construction **raises** if the clamped, mirrored mask has **nonzero
   weight on every FFT bin** (Option B would silently become a pass-through of
   the already-bandpassed signal: a disabled reduction is a config error, never
   a valid configuration).
3. **Apply** the mask to each chirp's **unwindowed** FFT (the Hann is detection-only),
   IFFT per chirp, **mean across the 20 chirps** → one complex 534-sample signal.

**EdgeTrim comes AFTER reduction** (deliberate, matches `wst_extract.m`: the
reduction — in particular Option B's 534-pt FFT — operates on the full chirp; the
reduced 1-D signal is trimmed only afterward). `edge_trim` is a separate function so
the pipeline test can assert the order structurally.

**Departure from the reference, to be logged in HISTORY.md at implementation
(discovered while specifying against `wst_extract.m`; the main plan's "±1-bin
two-sided Hann-tapered mask" is already the corrected form — this pins its exact
values and records why the reference's own code cannot be followed literally):**

- The reference keeps only `posBins = peakBin + (0:nb)` — **one-sided upward**,
  contradicting its own docstring "±bins kept around peak".
- Worse, its taper `M(idx) = hann(numel(idx))` spans the sorted kept indices
  (positive block + mirrored block treated as one contiguous window) with MATLAB's
  **endpoint-zero symmetric Hann** — for nb = 1 that is `hann(4) = [0, .75, .75, 0]`,
  which **zeroes the detected peak bin itself** (and its mirror), leaving only bin
  peak+1 at 75% weight. Internally inconsistent in the same way as the reference's
  mean-centre/MAD-scale standardization, and resolved the same way: implement the
  form the docstring and the main plan describe (symmetric ±nb around the peak, peak
  at full weight, Hann-shaped shoulders), record the reference's actual behavior.

**Acceptance.** §3 reduction tests green; ROI bins 4..6 at default config verified by
independent arithmetic; the mask energy audit (T-PP12) shows energy only at the kept
bins with the specified weights; trim order and length pinned by test.

### 2.4 `src/dehyd/preprocess/standardize.py`

**Responsibility.** Sequence steps 6–7: channel mapping and per-signal
standardization. Pure functions.

**Public API.**

```python
def robust_standardize(x) -> np.ndarray
    # y = (x - median(x)) / (1.4826 * MAD(x) + eps)
    # MAD = median(|x - median(x)|); eps = np.finfo(np.float64).eps (~2.22e-16).
    # Constant input -> MAD = 0 -> y = 0/eps = all zeros: finite, never NaN.

def meanstd_standardize(x) -> np.ndarray
    # y = (x - mean(x)) / (std(x, ddof=0) + eps) — the pre-declared ablation
    # alternative (§0 classification). ddof=0 (population convention, numpy's
    # default) is FROZEN and stated in the docstring: MATLAB's std defaults to
    # ddof=1, and no reference constraint exists (the reference never used
    # mean/std scaling), so the convention is pinned here and by the
    # hand-computation test (T-PP16) exactly as the robust denominator is.

def to_channels(s, channel, method) -> np.ndarray
    # channel="mag": standardize(|s|)                 -> float64 [1, 470]
    # channel="iq":  standardize(real), standardize(imag) SEPARATELY
    #                                                  -> float64 [2, 470]
    # method from preprocess.standardize; anything else raises.
```

**Frozen semantics — the settled departure, made exact.** The reference's
`standardize_robust` centers by the **mean** but scales by the MAD — internally
inconsistent (main plan §Deliberate departures; already settled, not re-litigated).
We use the coherent robust z: **median-centred, MAD-scaled**. Two small points the
main plan leaves open are pinned here (§6 A-M3-5):

- **eps placement:** denominator = `1.4826·MAD + eps` (the main plan's formula as
  written), with **eps = float64 machine epsilon** — not the reference's
  `1.4826·(MAD + eps)`; the difference is negligible (a guard against MAD = 0, not a
  tuning constant) but one form must be frozen for bit-reproducibility.
- Channels are standardized **each from its own statistics** (mag from |s|; real and
  imag each from themselves) — per-signal, so no leakage vector (§0 invariant).

**Acceptance.** §3 standardization tests green; formula matches a hand computation
exactly; constant and near-constant inputs produce finite output.

### 2.5 `src/dehyd/preprocess/pipeline.py`

**Responsibility.** The executable sequence as **one linear, followable function** —
the plan's steps 3–7 in code order, no indirection (CLAUDE.md code style: the
signal-processing steps visible, close to how the plan states them).

**Public API.**

```python
def preprocess_frame(frame, pre: PreprocessConfig, *, reduction, channel) -> np.ndarray
    # frame: complex128 [534 x 20] (validated EXACTLY against the loader
    # constants N_FAST_TIME, N_CHIRPS; non-finite input raises — the input
    # contract is QC-passed frames only).
    # reduction: "a" | "b"      channel: "mag" | "iq"    (explicit arguments,
    # NOT config — these are inner-CV search axes at M6, §0)
    # Sequence: band gate (butterworth sosfiltfilt | fft gate, per config)
    #   -> reduce (A | B) -> edge_trim(32) -> to_channels -> [C x 470] float64.

def preprocess_cube(cube, pre, *, reduction, channel) -> np.ndarray
    # complex128 [534 x 20 x N] -> float64 [N x C x 470]; a documented loop /
    # batched-filter application over frames with identical per-frame results
    # (T-PP20 asserts frame-in-cube == frame-alone, the M2 T-QC7 pattern).
```

**Frozen parameters.** Everything from `PreprocessConfig`; output length
534 − 2·32 = **470**; channel count C = 1 (mag) or 2 (iq).

**Acceptance.** Pipeline output equals the manual composition of the §2.2–§2.4
functions in the documented order for all four (reduction × channel) variants
(T-PP19); bit-identical across runs; `run_qc`-style exact shape validation.

### 2.6 `experiments/run_preprocess.py` — cohort diagnostic (first contact)

**Responsibility.** The one-command pass of the frozen sequence over the real
eligible cohort, and the curated diagnostics artifact. Thin CLI, same repeatable
`--config` pattern as `run_qc.py`. **Diagnostic only — it selects nothing**: every
constant is frozen before it runs, and a surprising distribution is a *finding* for
HISTORY.md and the owner, never a license to retune (the M2 doctrine).

**Behavior.** config → `load_ground_truth` → `build_manifest` → `apply_qc` →
`eligible_frames` → for each eligible session, load its file once and preprocess its
QC-passed frames (both reductions; channels are cheap to derive) → per-session
diagnostics → print a cohort summary → write
**`<results_dir>/preprocess/preprocess_diagnostics_10ghz.csv`** (config is the single
output-path authority) → `record_run(config, manifest_qc, folds=None,
extra={"stage": "milestone-3-preprocess", "analysis_role": "primary",
"filter_spec": filter_spec(pre), <headline stats>})`.

**Primary-only guard (no artifact collision — the FULL canonical spec, not just the
ablation switches).** The curated CSV is the **primary** first-contact artifact, and
§0 promises non-primary results are always explicitly labeled — so
`run_preprocess.py` **refuses to run** (a loud error before any I/O) unless the
consumed `config.preprocess` **equals the complete canonical primary spec =
`PreprocessConfig()`** (the frozen dataclass defaults: butterworth + robust, model
gate 1–2 m, order 4, edge_trim 32, peak_neighbors 1, mask_taper on, and the 10 GHz
sampling constants). Checking only `gate_method`/`standardize` would not be enough:
`model_gate_m: [0.9, 3.0]` is an inner-CV *candidate*, not the primary, and a run
with it (or with any altered filter/reduction constant) would overwrite the same
CSV under `analysis_role: "primary"` with different peak-ROI/filter diagnostics.
Frozen-dataclass equality makes the guard one comparison, listing the deviating
fields in its error. If non-primary diagnostics are ever wanted, they get their own
explicitly-labeled artifact path as a separate, recorded decision — deliberately
not built at M3. The guard is a pure helper (script structure = pure helpers +
thin `main`, the M2 audit pattern) so it is testable without a cohort run
(T-PP24).

**Per-session columns (one row per subject × eligible session), each with a frozen
definition — two correct-looking implementations must not be able to produce
incomparable numbers:**

- `n_eligible_frames`.
- **Option-B peak diagnostics** — peak-bin mode / min / max and median peak Hz,
  obtained from `detect_option_b_peak(...).peak_bin` (§2.3 — the same structured
  result the reduction uses, never a re-implementation). **Mode tie rule (frozen):**
  when two or more bins are equally frequent, the reported mode is the **lowest
  tied bin** — implemented deterministically (`np.unique` returns sorted bins;
  first argmax of the counts is the lowest tie), never delegated to a
  pandas/scipy `mode` whose tie behavior can shift across versions (T-PP23 pins
  it with a crafted tie). "Does the dominant beat sit where a seated-at-~1 m
  subject should?"
- **Filter energy-retention ratio (frozen formula):** per frame,
  `E_post / E_pre` where `E = Σ|x|²` over the **full complex 534 × 20 cube in the
  time domain** (Parseval makes a spectral convention unnecessary — no FFT, no
  window enters this number), `E_pre` on the raw frame, `E_post` immediately after
  the band gate, **before reduction and trim**. Session value = median across its
  eligible frames, Butterworth path. Zero-denominator behavior: an eligible frame
  cannot have `E_pre = 0` (the QC in-band screen requires ratio ≥ 0.30, impossible
  at zero total power), so the helper **raises** on `E_pre = 0` — an input-contract
  violation, never silently guarded to 0.
- **Peak-concentration diagnostics (frozen formulas)** — Option-B ROI membership is
  true by construction, so it is NOT evidence a meaningful dominant peak exists;
  these are. Both computed from **one `OptionBDetection` result per frame** (§2.3)
  — its `.power` (the periodic-Hann, chirp-averaged half-spectrum power `P` of the
  post-filter frame) and `.roi_bins`, never a recomputation:
  `roi_to_total = Σ_ROI P / Σ_half P` (guarded denominator as in QC: an all-zero
  spectrum → 0) and `peak_share = P[peak] / Σ_ROI P` (Σ_ROI P = 0 — the documented
  zero-ROI case — → recorded as NaN/absent, never fabricated).
  **Session aggregation of missing values (frozen):** session `peak_share` =
  **median over the frames where it is defined** (nanmedian / skip-missing); a
  session where every frame lacks `peak_share` gets an **empty CSV cell** (pandas
  NaN → empty field) — deterministic either way, never a fabricated 0. All other
  diagnostics are defined on every eligible frame and use the plain median.
- Output finiteness confirmation for all four variants.

These are the concrete cohort numbers SECOND_CHAPTER §2 needs (e.g. "the detected
beat concentrated at bins 4–5 ≈ 1.2–1.5 m across N sessions, with median peak_share
X"). Each formula gets a synthetic test before the cohort run (T-PP23).

**Acceptance.** Runs end-to-end on the real cohort in ≈ 1–3 min; the CSV is written
and verified (re-read, row count = 73 eligible sessions); invoking with any non-canonical preprocess config (an ablation switch, the
0.9–3.0 m candidate gate, or any altered filter/reduction constant) fails loudly
before any I/O, naming the deviating fields;
actual distributions recorded in HISTORY.md; committed only on explicit owner
request.

---

## §3 Tests

All constants are read from the config object — a test that re-hardcodes 4/32/[1,2]/
500 would pass vacuously when config and code drift apart. Synthetic fixtures use
seeded RNG noise (`numpy.random.default_rng(<seed in the test>)`); **a noiseless tone
is not a valid clean-frame fixture** (M2 lesson: degenerate histograms — and here,
degenerate MAD).

**`tests/test_preprocess.py` (synthetic; no real data):**

| ID | Test | What it proves |
|----|------|----------------|
| T-PP1 | `design_bandpass_sos` from config: band = `beat_band_hz(model_gate_m)` — NOT the QC gate (the two differ at defaults); order 4 → sos shape (4, 6); all poles strictly inside the unit circle; `default_padlen(sos)` = 27 at the default config AND `sosfiltfilt(..., padlen=27)` is bit-identical to the library default (a scipy behavior change fails loudly) | Right band, right structure, numerically stable; the frozen explicit padlen is pinned |
| T-PP2 | Effective forward-backward magnitude response via `sosfreqz` (\|H\|², since filtfilt applies H twice): ≥ 0.99 at mid-band, ≈ 0.5 at the gate corners (−3 dB → −6 dB effective, the filtfilt-squaring fact), ≤ 1e-4 at 0.5·f_lo and 2·f_hi | The filter passes in [f_lo, f_hi] and stops outside — the handoff's response check, with the squaring accounted for |
| T-PP3 | Zero-phase: a Gaussian-enveloped in-band tone burst keeps its envelope-peak sample index after `bandpass_filtfilt` (cross-correlation of in/out peaks at lag 0) | Forward-backward is zero-phase / zero group delay on a test tone |
| T-PP4 | Complex handling: `filter(x) == filter(real) + 1j·filter(imag)` and `filter(conj(x)) == conj(filter(x))`; an in-band complex exponential keeps its frequency (spectral peak bin unchanged); output dtype complex128. (Gain/energy claims live in T-PP6 — finite-record, not unit gain) | Real/imag-separate filtering is correct for complex signals |
| T-PP5 | Batched (reshaped-cube) filtering ≡ per-chirp filtering, bit-identical | Vectorization cannot change semantics |
| T-PP6 | Energy sanity, **finite-record regression** (the §2.2 measured values, pinned with tolerance — these document actual 534-sample `sosfiltfilt` behavior, NOT design targets): mid-band complex-exponential retention = 0.7595 full record / 0.8313 after trim (±0.005); 50 kHz tone ≤ −17 dB full record and ≤ −20 dB after trim; Parseval: time-domain energy equals `norm="ortho"` spectral energy on a test signal | Finite-record behavior is pinned so a scipy/design drift is caught — and kept distinct from T-PP2's steady-state claims |
| T-PP7 | `fft_gate`: mask = 1 across the passband, raised-cosine skirts exactly `transition_hz` wide, symmetric in \|f\|, 0 beyond; in-band tone unchanged, out-of-band removed; complex-safe | The ablation alternative matches `filter_gpt_fft.m` semantics |
| T-PP8 | Changing `model_gate_m` in config moves the passband (a tone at the old band edge is attenuated under the new gate); `gate_method` switches the implementation | Parameters genuinely come from config |
| T-PP9 | `reduce_option_a` equals a hand-computed chirp mean on a crafted cube | Option A exact |
| T-PP10 | `option_b_roi_bins` at default config = {4, 5, 6} by independent arithmetic (df ≈ 975.34 Hz); at the 0.9–3.0 m gate = {4..10}; empty ROI raises | Exact ROI arithmetic; no margin applied |
| T-PP11 | Option B detection: an in-ROI tone (bin 5) is chosen over a STRONGER out-of-ROI tone (bin 20); detection uses periodic Hann; **single-sourcing** — one `OptionBDetection` result carries `peak_bin`/`power`/`roi_bins`; the mask is centred on exactly `.peak_bin` (asserted via the mask energy audit) and the §2.6 `roi_to_total`/`peak_share` values recomputed by hand from that same result's `.power`/`.roi_bins` match what the diagnostics report | ROI restriction is real; detection convention pinned; one canonical detection result, no duplicated logic and no second FFT convention |
| T-PP12 | Option B mask audit: output spectrum has energy only at peak±1 and conjugate mirrors; taper weights = [0.5, 1.0, 0.5] (peak at FULL weight — the reference's peak-zeroing bug is absent); `mask_taper: false` gives rectangular weights; an exactly-on-bin tone at the peak survives at ≈ unit amplitude. **Boundary battery:** nb = 0 → weight [1.0] on the peak alone; nb = 2 → [0.25, 0.75, 1.0, 0.75, 0.25]; a peak near the half-spectrum edge → clamped bins, mask still valid; an nb large enough that the clamped+mirrored mask covers **every** FFT bin → **raises** (the anti-vacuity guard) | The corrected mask made executable across the whole nb domain, and a pass-through Option B is an error, not a configuration |
| T-PP13 | Zero-ROI-detection-power cases, two fixtures: (a) trivial — a frame with no in-ROI content → peak = first ROI bin, output finite, bit-identical across runs; (b) **adversarial windowed-null** — unwindowed bins 3..7 = [1, 0, −1, −2, −3] (Hann-windowed detection bins 4..6 exactly zero, unwindowed mask bins nonzero) → peak = first ROI bin (4) and the output equals the hand-computed mask-applied IFFT of the unwindowed spectrum (nonzero — the documented no-zero-claim behavior), `peak_share` NaN | The deterministic fallback is pinned by a fixture that would expose a false "output ≈ 0" assumption, not one that passes trivially |
| T-PP14 | `edge_trim`: output = manual `[32:502]` slice, length 470; raises when `2·n ≥ len` or result < 32 (raise, never clamp) | Trim semantics + the anti-clamp guard |
| T-PP15 | Pipeline order: Option B on a crafted frame gives a DIFFERENT result if trim were applied before reduction (fixture built so the orders disagree); pipeline matches trim-after-reduction | "EdgeTrim AFTER reduction" is structural, not a comment |
| T-PP16 | `robust_standardize` equals the hand formula; median(y) ≈ 0 and 1.4826·MAD(y) ≈ 1 on seeded random data; constant input → all zeros, no NaN/Inf; `meanstd_standardize` equals its hand formula **with ddof = 0 pinned** (a ddof = 1 implementation fails the exact comparison) | The frozen robust z, and the frozen plain-z convention |
| T-PP17 | One extreme outlier barely changes the bulk scale under robust, but wrecks meanstd (comparative assertion) | Robustness is real, and the two methods genuinely differ |
| T-PP18 | `to_channels`: mag = standardized \|s\| with shape (1, 470); iq = real/imag standardized SEPARATELY (each matches its own manual computation), shape (2, 470); method honored from config | Channel contract |
| T-PP19 | `preprocess_frame` ≡ manual composition filter→reduce→trim→channels for all 4 (reduction × channel) variants, both gate methods | The pipeline is exactly the documented sequence |
| T-PP20 | `preprocess_frame(frame)` ≡ `preprocess_cube(cube)[i]` regardless of companion frames | Per-frame independence (the T-QC7 pattern; §0 invariant executable) |
| T-PP21 | Exact shape validation raises on anything ≠ (534, 20)/(534, 20, N); non-finite input raises | Input contract |
| T-PP22 | Two full-pipeline runs on seeded fixtures are bit-identical | Determinism (there is no RNG in the pipeline; the seed governs fixtures) |
| T-PP23 | Diagnostic formulas (§2.6) on synthetic frames: for a known in-band + out-of-band tone mix, the reported energy-retention ratio equals the hand computation **on the actual summed mixture** — `Σ\|filt(x_mix)\|² / Σ\|x_mix\|²` with `filt` the same public filter function (component-energy sums are NOT a valid expectation: energy is quadratic, and finite-record transients make the filtered components non-orthogonal — measured 0.39608 direct vs 0.38919 component-sum on the equal-amplitude 4886.2 Hz + 50 kHz fixture). Signal-level linearity `filt(x₁+x₂) == filt(x₁) + filt(x₂)` is asserted **separately** (float tolerance) and never substituted for the mixture energy. The "adding out-of-band energy lowers retention" assertion is pinned to **this exact fixture** (a fixture property, not a general law — the cross-term can reverse it for other amplitudes/phases). `E_pre = 0` raises; `roi_to_total` and `peak_share` match hand computation from the detection result's own `power`/`roi_bins`; zero-ROI-power frame → `peak_share` NaN/absent, never fabricated; session aggregation skips missing `peak_share` values (nanmedian), an all-missing session yields an empty cell, and a crafted **tied peak-bin mode** resolves to the lowest tied bin (§2.6 tie rule) regardless of library version | The cohort numbers are the frozen formulas, testably — with the quadratic-energy reality in the expectation |
| T-PP24 | `run_preprocess.py` primary-only guard: the guard helper accepts exactly the canonical `PreprocessConfig()` and raises — naming the deviating field — on `gate_method: fft`, `standardize: meanstd`, **`model_gate_m: [0.9, 3.0]`** (an inner-CV candidate is not the primary), and a non-default filter/reduction constant (e.g. `edge_trim: 16`) (pure-function test, no cohort run) | A non-primary run can never overwrite the primary curated CSV under a primary label |

**`tests/test_config.py` additions:** the five new fields' defaults, YAML overrides,
type/bound rejections (string gate_method typo, negative transition, bool
peak_neighbors, integer mask_taper, unknown standardize name); cross-field: model
gate entirely above Nyquist rejected AND a **straddling** gate (f_lo below Nyquist,
f_hi at/above it) rejected — the strict `0 < f_lo < f_hi < fs/2` rule, both gate
methods; `model_gate_m ⊄ qc_gate_m` rejected; whole-spectrum FFT mask rejected.

**`realdata` (in `tests/test_preprocess.py`):** one real file (`subject_1_8am.mat`):
run QC (`run_qc_cube`) to get its passing frames, then all four pipeline variants over
those frames — every output finite with shape (C, 470); re-run bit-identical; every
Option-B peak bin (via `detect_option_b_peak` — the canonical helper, §2.3) ∈ ROI;
print the peak-Hz and peak-concentration distributions. **No expected-distribution
assertion** — distributions are unknown until this runs, and asserting them would be
tuning by the back door (M2 doctrine).

**`tests/test_no_leakage.py`: zero changes.** Preprocessing is per-frame and unfitted
(§0); T1–T19 and the reference procedure remain untouched and green.

---

## §4 Definition of done

| ID | Criterion |
|----|-----------|
| D1 | `uv run pytest` green on a checkout with no private data (all new synthetic tests included) |
| D2 | `uv run pytest --realdata` green, including the one-file pipeline test |
| D3 | `uv run python experiments/run_preprocess.py --config configs/exp_a_regression.yaml` runs over the full eligible cohort, **writes and verifies** `<results_dir>/preprocess/preprocess_diagnostics_10ghz.csv` (73 rows); actual distributions recorded in HISTORY.md; committed only on explicit owner request |
| D4 | `tests/test_no_leakage.py` **byte-for-byte unmodified since M1** and green |
| D5 | HISTORY.md entries per resolved step, including the departure logs: no-window-in-primary-path (ROADMAP §3.2), median/MAD form + eps placement, and the Option-B mask correction (reference one-sided/peak-zeroing quirk, §2.3) |
| D6 | SECOND_CHAPTER.md §2 "Preprocessing" written at close: provenance of every parameter (order 4, gate 1–2 m, trim 32, taper values, robust z), the paper-vs-code domain resolution, and the cohort diagnostic findings |
| D7 | §6 amendments applied to `plans/implementation_plan.md`; the two documents consistent |

---

## §5 What could go wrong (known traps, pre-paid)

- **`sosfiltfilt` on complex input:** don't rely on scipy's complex handling — filter
  real and imag explicitly (also what the main plan literally specifies).
- **filtfilt squares the magnitude response:** corner attenuation is −6 dB effective,
  not −3 dB; T-PP2 asserts ≈ 0.5 at the corners so nobody "fixes" it later.
- **Very low normalized Wn (0.0125–0.025):** transfer-function (b, a) form would be
  numerically fragile here — SOS is mandatory, and T-PP1's pole check makes stability
  executable rather than assumed.
- **scipy vs MATLAB gain placement:** MATLAB returns `[sos, g]` with a separate gain;
  scipy folds the gain into the sections. Equivalent — but don't port `g` twice.
- **Periodic vs symmetric Hann:** Option-B *detection* uses periodic Hann (QC
  convention, `hann(N,'periodic')`); the mask *taper* values come from the interior
  of a **symmetric** Hann. Deliberate, stated in §2.3 — never mixed silently.
- **`np.argmax` ties/zero ROI:** first-index behavior is the documented deterministic
  fallback (T-PP13); do not add data-dependent tie-breaking.
- **Mirror-bin arithmetic:** conjugate bin = `(N − k) mod N`; DC and Nyquist are
  their own mirrors — collision takes max weight, never sum (cannot occur at default
  config; guard anyway).
- **Raise, never clamp, on `edge_trim`:** the reference's `min(EdgeTrim, N/4)` hides
  config errors; ours fails loudly (§2.1).
- **YAML 1.1 signed exponents (M1 trap):** any new float written in exponent form
  needs a signed exponent; `fft_gate_transition_hz: 500.0` avoids the issue, but the
  rule holds for any future edit of `preprocess.yaml`.
- **No window before the time-domain filter** — resist the reflex to "add the
  Hamming from the paper"; the departure is deliberate and logged (main plan
  §Preprocessing).
- **Noiseless-tone fixtures are degenerate** (M2 lesson — flatline histograms; here
  also MAD = 0): always add small seeded noise to "clean" fixtures.
- **Do not "fix" surprising cohort distributions.** If `run_preprocess.py` shows odd
  peak-bin or retention patterns, that is a finding for HISTORY.md and the owner —
  not a license to nudge the gate, taper, or trim (§0 invariant; M2 doctrine).
- The repo-root `.pytest_cache/` ACL issue stands; pytest cache stays redirected to
  `.cache/pytest` (pyproject) — leave it alone.

---

## §6 Flagged gaps in `implementation_plan.md` + proposed amendments

To be applied to the main plan on approval of this document (keeping it the single
source of truth), exactly as M2's A1–A7 were:

**A-M3-1..A-M3-5 were applied to `plans/implementation_plan.md` at milestone close
(2026-07-23); A-M3-6 and A-M3-7 were applied earlier, during review rounds 1 and 4.**

- **A-M3-1 — Option-B mask, exact taper (gap: "±1-bin two-sided Hann-tapered mask"
  has no values).** Amend §Preprocessing step 4b to state: taper weights = interior
  of `hann(2·nb+3, sym=True)` (nb = 1 → [0.5, 1.0, 0.5], full weight on the peak),
  mirrored onto conjugate bins, max-weight on self-mirror collision. Also record (as
  a departure note) that the reference code is one-sided (`peak..peak+nb`) and its
  endpoint-zero taper zeroes the detected peak — the main plan's two-sided form is a
  correction, not a port.
- **A-M3-2 — Option-B ROI (ambiguity: "the range ROI").** Make explicit: ROI = the
  **model** gate band (`preprocess.model_gate_m`) on the non-negative half-spectrum,
  **no margin** (the ±1000 Hz margin is a QC constant; the reference ROI carries no
  margin). Default gate → bins 4..6.
- **A-M3-3 — New config fields + gate checks.** `gate_method`,
  `fft_gate_transition_hz`, `peak_neighbors`, `mask_taper`, `standardize` join the
  frozen `preprocess:` section; cross-field validation now enforces the strict model
  band rule `0 < f_lo < f_hi < fs/2` (both gate methods — no straddling-Nyquist gate
  reaches `butter`), `model_gate_m ⊆ qc_gate_m` (the "one fixed QC gate" design made
  executable), and FFT-gate non-vacuity. The Option-B mask carries a runtime
  anti-vacuity guard (raises when the clamped+mirrored mask covers every bin) — the
  `in_band_mask` doctrine applied to reduction.
- **A-M3-6 — Ablation classification (APPLIED to implementation_plan.md
  2026-07-22, review round 1).** The main plan offered the FFT gate "as a config
  alternative for ablation" and the mean/std standardization as "configurable"
  without classifying either against the inner-CV search space — a gap that could
  let an attractive ablation number become unrecorded test-set tuning. Amended the
  main plan (§Preprocessing step 3; §Deliberate departures, robust-standardization
  bullet) to state both are **pre-declared ablations only: never inner-CV
  candidates, never able to displace the primary path**, matching this plan's §0
  classification. Applied immediately (not deferred to approval) because it
  corrects the main plan itself, per the cross-document consistency rule.
- **A-M3-4 — Repo-tree additions (F-item, like M2's F2).**
  `src/dehyd/preprocess/pipeline.py` (the sequence as one linear function) and
  `experiments/run_preprocess.py` (+ its diagnostics artifact under
  `results/preprocess/`) are not in the main plan's tree; they follow its thin-CLI
  and composition patterns. Note in the tree.
- **A-M3-5 — Standardization eps (gap: "eps" unspecified).** Freeze
  eps = float64 machine epsilon (≈ 2.22e-16) in the denominator form
  `1.4826·MAD + eps` (as the main plan's formula is written; the reference's
  `1.4826·(MAD + eps)` differs immaterially but one form must be frozen).
- **A-M3-7 — Frozen protocol constants enter the config-freeze gate (APPLIED to
  implementation_plan.md 2026-07-23, review round 4).** The main plan's M5 freeze
  enumerated experiment designs and search spaces but did not state that
  low-level-supported-but-unsearched parameters (Option-B `peak_neighbors`/
  `mask_taper`, the FFT-gate transition) are validated at their frozen values in
  every modeling config. Amended Build order §5 so a non-whitelisted value cannot
  reach outer-fold evaluation merely because a function supports it. Applied
  immediately (corrects the main plan itself, cross-document consistency rule).
- **F-M3-1 — `filter_spec` recording** (padtype/padlen/Wn actually realized) is
  execution detail satisfying the main plan's "padtype/padlen fixed and recorded";
  no main-plan change needed.

---

## §7 Open items this milestone resolves or carries

| Item | Status after M3 |
|------|-----------------|
| Preprocessing modules (Build order §3) | **Resolved** — the executable sequence implemented and self-consistency-tested |
| Cohort-level preprocessing characterization | **Resolved** — `run_preprocess.py` diagnostics feed SECOND_CHAPTER §2 |
| torch + kymatio WST | **Next (M4)** — consumes `preprocess_cube` output [N × C × 470]; kymatio border-effect warning at J=7/470 to be measured there, not assumed |
| `configs/ibex.yaml`, `scripts/ibex/` | **Still deferred** — no GPU work in M3; first IBEX milestone |
| 77 GHz flatline multiplicity (7/10 audited frames) | **Still parked** — owner decision at M5, never retuned from data |
| Preprocessed-signal caching | **Deliberately not built** — M4 recomputes on the fly; revisit only on real friction |
| Branch | M3 work on `v1_milestone_3` (pushed branches: `v1_milestone_1`, `v1_milestone_2`; nothing merged to `main`) |

---

_Review round 1 (2026-07-22): all 7 Codex comments accepted and applied — strict
model-band Nyquist rule (both gate methods) + straddling test; Option-B mask
anti-vacuity guard + nb boundary battery; `detect_option_b_peak` as the single
canonical peak source; frozen formulas for the §2.6 diagnostics (time-domain
energy-retention, roi_to_total, peak_share); explicit ablation classification of
`gate_method`/`standardize` (A-M3-6, propagated to implementation_plan.md);
padlen frozen by explicit passing (27 at default config, verified bit-identical to
the scipy default on 2026-07-22); mean/std ddof = 0 pinned._

_Review round 2 (2026-07-23): all 3 Codex comments accepted and applied — (1) the
finite-record vs steady-state distinction: Codex's measured numbers were
**independently reproduced** (mid-band retention 0.7595 full / 0.8313 trimmed;
50 kHz −17.2 dB / −20.1 dB) and are now pinned in §2.2 and T-PP4/T-PP6/T-PP23 as
regression values, with an explicit no-parameter-chasing rule; (2) detection is
single-sourced as a structured `OptionBDetection` (peak_bin, power, roi_bins)
consumed by reduction, diagnostics, and tests alike, with nanmedian/empty-cell
aggregation frozen for missing `peak_share`; (3) `run_preprocess.py` gained a
primary-only guard (`butterworth + robust` or loud failure, `analysis_role:
"primary"` in provenance) so ablation runs can never overwrite the primary curated
CSV (T-PP24)._

_Review round 3 (2026-07-23): all 3 Codex comments accepted and applied — (1) the
T-PP23 expectation now hand-computes on the **actual summed mixture** (energy is
quadratic; the cross-term claim was **independently reproduced**: 0.39608 direct vs
0.38919 component-sum), signal-level linearity is asserted separately, and the
directional assertion is pinned to the exact fixture; (2) the primary-only guard
now compares the consumed `config.preprocess` against the **complete canonical
`PreprocessConfig()`** (any deviation fails loudly, named — including the 0.9–3.0 m
inner-CV candidate gate), T-PP24 extended accordingly; (3) the peak-bin mode tie
rule is frozen to the lowest tied bin with a deterministic `np.unique`-based
implementation and a crafted-tie test._

_Review round 4 (2026-07-23): both Codex comments accepted and applied — (1) §0
gained the third classification class, **frozen protocol constants**
(`peak_neighbors = 1`, `mask_taper = true`, `fft_gate_transition_hz = 500`:
non-defaults are test-only and rejected by modeling/artifact entrypoints), with the
M5-freeze whitelist commitment propagated to the main plan's config-freeze gate
(A-M3-7); (2) the false "zero-ROI ⇒ output ≈ zero" claim was removed — the
windowed-detection vs unwindowed-mask counterexample is valid (Hann kernel
[−¼, ½, −¼] annihilates windowed bins 4..6 for unwindowed bins [1, 0, −1, −2, −3])
— the deterministic first-ROI-bin fallback is now pinned by that adversarial
fixture in T-PP13 with `peak_share = NaN`, and no exact-zero special branch was
added (knife-edge on floats; the NaN diagnostic already flags the case)._
