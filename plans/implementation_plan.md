# Implementation Plan — Dehydration Radar (Python rebuild & extension)

## Context

We are rebuilding, in Python, the analysis behind the paper *"From Radio Waves to
Wellness"* (originally MATLAB) and extending it into a methodologically rigorous
study for a PhD chapter / T-IM submission. The published headline (~96–98%
frame-level, subject-dependent 5-class accuracy) is inflated by session-block
leakage and is not defensible. The rebuild replaces it with an **honest pipeline**
whose headline is **regression of fluid loss (Δm%)** validated against measured
body-mass change under **leave-one-subject-out (LOSO)** CV, with the 5-class task
demoted to a secondary *ordinal* problem. Primary modality is the 10 GHz CN0566
radar; 77 GHz is used only for a cross-band fusion section on the same 16-subject
cohort.

**The MATLAB code in `matlab/` is reference material only** — a guide to how the
original processing was done. **All reported results come from Python alone**;
MATLAB is not mentioned in the paper or thesis chapter, and we neither expect nor
check for matching numeric values. Where the reference makes a choice that is
arbitrary, internally inconsistent, or improvable, we take the better option and
record the reason in HISTORY.md / SECOND_CHAPTER.md. This is explicitly a
**feasibility study** (single objective reference, 16 subjects), and is framed as
such throughout.

This plan is the pre-implementation design for approval. No implementation code is
written yet. It follows the milestones in ROADMAP §7 and the non-negotiables in
CLAUDE.md / ROADMAP §1.

## Confirmed data facts (verified against the raw files, not the paper)

- **10 GHz**: `data/10ghz/subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat`, 80 files.
  MAT v5, little-endian, zlib-compressed → readable with `scipy.io.loadmat`.
  Each file holds `framesRadar`, a **MATLAB `double` array of shape
  [534 fast-time × 20 chirps × 100 frames]**, complex. On disk the values are
  stored in the compact `miINT16` element type (a MAT-file space optimization), but
  the array *class* is double, so `scipy.io.loadmat` returns **`complex128`** — we
  work in complex128, not int16. Also `framesRadarIQ` [20834×2×100] = raw
  pre-arrangement IQ, **unused** by the reference pipeline (we ignore it too).
  → One file = one subject/session = 100 frames. 16×5×100 = 8000 frames pre-QC.
- **77 GHz**: MAT **v7.3 / HDF5**, ~285 MB each (~23 GB total) → `h5py`, not
  `loadmat`. Deferred to the fusion milestone. Frame count differs from 10 GHz
  (≈125/session), which forces session-level fusion (see Exp G).
  **Axis order (decided now, asserted at load).** MAT v7.3 stores dimensions in
  reverse of the MATLAB-logical order, so `h5py` presents the dataset as
  `(16, 256, 256, 125) = (Nrx, Nchirps, Nfast, Nframes)`. The loader applies a full
  axis reversal → **`(Nframes, Nfast, Nchirps, Nrx) = (125, 256, 256, 16)`**, matching
  the shape the reference (`chirpavg_and_fuse_batch.m`) expects. Because the two 256
  axes (fast-time, chirps) are indistinguishable by size, the shape assertion **cannot
  by itself** rule out a fast-time↔chirp interchange; disambiguation is done by the
  **signal-domain semantic check in Exp G** (range structure on the fast-time axis,
  near-zero-Doppler on the chirp axis, cross-checked against
  `chirpavg_and_fuse_batch.m`) before any fusion. The loader asserts the observed
  `(16,256,256,125)` on read and records the mapping. *(This `h5py` shape is the
  reviewer's sampled value; it is re-confirmed on the first real load, since h5py is
  not yet installed in the local env — but the axis decision is fixed here, not
  deferred to coding.)*
- **Ground truth**: `data/weight/metadata_subjects_info.xlsx`, sheet `MetaData`,
  rows 3–18 (Subject 1–16). The header spans **two rows** (row 1 column labels; row 2
  fractional-time subheaders 0.333=8am … 0.667=4pm) with merged cells — it must be
  parsed by **fixed cell addresses**, not by header-name inference. Columns:
  B Name, C Age, D Height(cm), **E–I = weights at 8am,10am,12pm,2pm,4pm**,
  J kg lost, K "% of body weight" note.
  **Signed target**: `Δm%(subj, session) = (m(session) − m(S0)) / m(S0) × 100`
  (S0 = 8am, col E); **negative = fluid lost**. Near-monotone-decreasing, ≈0 to
  ≈ −2%. `ground_truth.py` cross-checks the computed S4 values two ways, each
  sign-aware and with an explicit tolerance, failing loudly on disagreement:
  (i) **column J is a signed kg change** (an `=I−E` formula; read via cached values)
  → compare against computed `m(S4) − m(S0)` with `|diff| ≤ 0.05 kg`;
  (ii) **column K is positive text** ("Loss of 1.74%…") → parse the positive
  percentage and compare against `abs(Δm%(S4))` with `≤ 0.05%` tolerance.
  Tolerances are conservative bounds justified by direct workbook inspection — most
  weights are recorded to 0.1 kg but Subject 15 uses 0.05-kg increments, and column K
  is not always conventional two-decimal rounding (truncation observed; max deviation
  ≈0.01 pct-points) — not claims about uniform recording precision.
  **No heart-rate column** — confirmed absent (see Exp F).
- **Subject identity — confirmed by the dataset owner.** Radar `subject_N` is the
  same person as workbook "Subject N" (direct identity mapping). The historical MATLAB
  5–20 numbering was **renumbered to 1–16 by the dataset owner purely for
  cleanliness** — same subjects, same order, no re-mapping — which is why the old
  scripts read 5–20 while the delivered files and workbook read 1–16. This provenance
  is recorded (not merely assumed). `manifest.py` additionally **fails on any missing,
  duplicate, or unmatched record** (every radar file matches exactly one weight row;
  every expected subject×session cell is present), so a mislabeled or absent file is
  still caught structurally.
- Session index S0..S4 = 8am/10am/12pm/2pm/4pm = paper's DeHydL0..L4.
- Radar params (reference values from the MATLAB, adopted unless a better choice is
  justified): fs=520834 Hz, B=500e6, Tchirp=1024e-6.

## Repo structure

```
pyproject.toml / uv.lock        # pinned env, Python 3.11+
HISTORY.md                      # implementation log, newest-first (written continuously)
SECOND_CHAPTER.md               # thesis chapter material (written at each milestone)
HANDOFF.md                      # new-session bootstrap (ONLY when you ask for it)
archive/code/  archive/results/ # retired/superseded code, stale/invalidated artifacts only
configs/                        # one YAML per run (seeds, paths, subset, device, tiling, model)
  data.yaml, preprocess.yaml, wst.yaml, exp_a_regression.yaml, ...
src/dehyd/
  config.py                     # load/validate YAML, resolve seeds & device
  data/
    loader_10ghz.py             # loadmat -> complex128 framesRadar cube; parse subject/session
    loader_77ghz.py             # h5py; full axis reversal -> (Nframes,Nfast,Nchirps,Nrx); asserts on-disk shape (fusion milestone)
    ground_truth.py             # fixed-cell xlsx parse -> signed Δm%, 5-class label, covariates; cross-check
    manifest.py                 # frame index table; FAILS on missing/dup/unmatched; per-frame QC reason codes
  qc/screens.py                 # flatline / robust RMS / in-band energy (pre-filter) screens with frozen thresholds
  preprocess/
    filters.py                  # SOS Butterworth zero-phase bandpass on complex fast-time; optional FFT-gate
    reduce.py                   # Option A (chirp mean) & Option B (Hann-detected peak isolation + tapered mask)
    standardize.py              # per-signal robust standardization
  features/
    wst.py                      # kymatio Scattering1D; ms->samples->(J,T) mapping; 3 tilings; mag & I/Q
    pooling.py                  # mean/std over global+halves; raw-flatten; session-level aggregation
  eval/
    splits.py                   # THE single source of folds: nested LOSO API yielding (train/val/test) SUBJECTS
    harness.py                  # fit-on-train-only runner for BOTH sklearn and torch; session-level inference
    metrics.py                  # regression (MAE/RMSE/r), ordinal (adj-acc, MAE-class, QWK), subject-cluster CIs
  models/
    regressors.py               # Ridge/SVR/RF/GBM (+ optional MLP)
    baselines.py                # 1D-CNN, spectrogram+2D-CNN, physics power-ratio, session-index-only
  provenance.py                 # per-run: raw-file hashes, resolved config, fold manifest, versions, git rev, device, seed, Slurm ID
experiments/                    # thin CLI entry points, one per ROADMAP experiment A..H
  run_regression.py, run_clock_decoupling.py, run_ordinal.py, run_baselines.py, ...
scripts/ibex/                   # sbatch templates for GPU jobs (DL baselines / NN)
tests/
  test_no_leakage.py            # subject-disjoint train/val/test + fit-on-train-only (sklearn AND torch)
  test_preprocess.py            # filter response, zero-phase, energy, Option-B ROI, determinism
  test_wst.py                   # WST path structure, shift-stability, numpy/torch cross-backend equivalence
  test_loader.py, test_manifest.py, test_ground_truth.py, test_metrics.py
results/  figures/              # regenerable artifacts, one command each
```

(`matlab/` stays as read-only reference. No Python test depends on MATLAB output,
and nothing MATLAB-derived is reported.)

## Build order (ROADMAP §7 milestones)

1. **Scaffold + config + `test_no_leakage.py`.** Pinned env (uv), config loader,
   manifest builder (with the fail-on-mismatch checks above), nested-LOSO splitter,
   provenance recorder, leakage test — green before any modeling. Also creates
   `HISTORY.md`, `SECOND_CHAPTER.md` and `archive/{code,results}/`.
2. **Loader + QC screens** (10 GHz) **plus a minimal 77 GHz audit.** Deterministic
   xlsx parse + cross-check; 10 GHz QC screens with frozen thresholds and per-frame
   reason codes; log per-subject/session removal counts; in-band energy computed
   **before** bandpass filtering. **The milestone-5 freeze fixes 77 GHz QC/tilings/
   input-domain/fusion, so those choices must rest on a real 77 GHz file, not
   assumptions** — therefore a *minimal* 77 GHz audit happens here (full extraction
   still deferred to milestone 9): install `h5py`, load one real file, confirm
   dtype / shape / complex representation, run the **raw-data axis semantic check**
   (range structure vs near-zero-Doppler, before clutter subtraction), and verify the
   proposed QC + range-Doppler operations produce **non-degenerate (nonzero-energy)**
   data. Findings are logged and feed the milestone-5 freeze.
3. **Preprocessing** — the executable sequence below, validated by self-consistency
   checks (filter response, zero-phase, energy, Option-B ROI) on Python output.
4. **WST feature extraction** — the kymatio parameterization below, validated by
   path-structure, shift-stability, and numpy/torch cross-backend checks.
5. **Config-freeze gate — the COMPLETE A–G protocol, before any outer results are
   inspected.** Because B–G reuse the same 16 subjects, any protocol choice made after
   seeing Exp A's outer-fold results is indirectly informed by later "test" subjects.
   So the freeze covers **every experiment's design**, committed to versioned
   `configs/` and git before modeling: the Exp A/WST search space **and its staged-
   search order**, the baseline specs (raw-beat and matched-preprocessing
   CNN/spectrogram, physics bands), the per-family budget K and seed set, the full
   statistical protocol, **Exp B** (residualization + equal-session objective),
   **Exp C** (ordinal family, objective, sign, fold-viability), **Exp E** (importance
   protocol), **Exp F** (single learner + covariate/collinearity rule), and **Exp G**
   (77 GHz QC thresholds, session-eligibility, WST tilings, input domain, fusion).
   Anything genuinely decided after outer results appear is **explicitly labeled
   exploratory**. This makes the "no config tuned on outer-test" chronology real at the
   cohort level, not just for Exp A.
6. **LOSO harness + fluid-loss regression (Exp A)** — the headline, session-level.
7. **Clock-decoupling analysis (Exp B)** — design locked (below) before Exp A results
   are examined.
8. **Ordinal 5-class (Exp C)** + **baselines (Exp D)** under the same harness (specs
   already frozen at milestone 5).
9. **Fusion (G)**, **interpretability (E)**, **confound check (F)**, **stats (H)**.
10. **Figure/table generation** for the chapter (one command each).

## Project journal & file hygiene (CLAUDE.md §Project journal files, §File hygiene)

Three living documents at the repo root, each with a different write cadence:

- **`HISTORY.md` — written continuously**, at the resolution of individual
  experiments/attempts. Each entry records what was tried, whether it failed or
  succeeded **and why**, and the concrete parameter values with reasoning (filter
  order/cutoffs, EdgeTrim, WST tiling Q and invariance scale, model
  hyperparameters, seeds, subset sizes). **Failures stay in the log.** Newest-first
  with dated headers. Every deliberate departure from the reference is logged here.
- **`SECOND_CHAPTER.md` — at each ROADMAP §7 milestone.** The distilled, chapter-
  ready account with the provenance of every choice; no MATLAB reference appears.
- **`HANDOFF.md` — only when you explicitly ask.** Max 200 lines.

**File hygiene.** `src/`, `experiments/`, `results/`, `figures/` contain only
current, valid, working material. `archive/` is for **superseded code and stale or
invalidated artifacts only**. A scientifically valid ablation or negative result —
Option A vs B, mag vs I/Q, a tiling or model that simply scored worse — is a
**current, reported result** and stays in `results/`; it is *not* archived merely
for losing. Only material that is broken, wrong, or belongs to an abandoned version
of the pipeline is moved to `archive/`, with the move noted in HISTORY.md.

## Library choices

- **WST**: `kymatio` (Scattering1D). numpy backend for local smoke tests; torch
  backend (CPU or CUDA) selectable by config. If both backends can produce reported
  features, a cross-backend equivalence test (below) must pass.
- **Classical models**: scikit-learn (Ridge, SVR, RandomForest, GradientBoosting,
  KNN, SVM) — CPU, local. Fit-on-train-only enforced via `Pipeline` inside the fold.
- **DL baselines / NN**: PyTorch, GPU on IBEX via sbatch. All fitted quantities are
  train-only through the common fold API (below), not via `Pipeline`.
- **I/O**: scipy.io (v5), h5py (v7.3), pandas+openpyxl (xlsx), PyYAML, pytest.

## Relationship to the MATLAB reference code

MATLAB is a **design reference, not a validation oracle.** We reproduce the *intent*
of the reference pipeline and establish correctness with Python-native checks
(filter response, zero-phase, energy/ROI, WST path structure, shift-stability,
determinism, cross-backend equivalence, fixed-seed reproducibility) — never numeric
diffs against MATLAB.

### Deliberate departures from the reference (with reasons)

- **Robust standardization → a proper robust z-score.** The 10 GHz reference centers
  by the *mean* but scales by the *MAD* — internally inconsistent. We use
  median-centering with MAD scaling, `y = (x − median)/(1.4826·MAD + eps)` (the
  coherent form the 77 GHz reference already uses). Applied per signal to its own
  statistics (per-frame normalization → no train/test leakage vector). Configurable;
  plain mean/std available.
- **Range gate is a parameter.** ROADMAP §3 specifies a 1–2 m gate (≈3.25–6.51 kHz);
  the reference used 0.9–3.0 m (≈2.93–9.77 kHz). Subject seated ~1 m from the radar →
  the tighter physically-motivated gate is the default, but it is config-driven and,
  if treated as a choice, selected inside inner CV (never on test subjects).
- **WST log transform** is a configurable modeling choice selected inside inner CV.
- **EdgeTrim** (32 samples/end) drops filtfilt edge transients; config parameter.

## Preprocessing — executable sequence (resolves the paper-vs-code ambiguity)

The paper describes "window → range FFT → SOS bandpass", but the reference *code*
bandpasses the time-domain complex chirp directly and uses windowed FFTs only for
QC and peak detection. We follow the code (the authoritative "how"), and state each
operation's domain, axis, and taper explicitly. **Edge trimming happens after signal
reduction, exactly as in `wst_extract.m`** (the reduction operates on the full
534-sample chirp; the reduced 1-D signal is trimmed only afterward):

1. **Load** `framesRadar` → complex128 cube `[534 fast-time × 20 chirps × 100 frames]`.
2. **QC screens** (see next section) run on the **raw** cube; the in-band energy
   screen uses a Hann-windowed 534-pt range FFT **before** any filtering.
3. **Band gate (primary):** SOS **Butterworth bandpass, order 4** designed with
   scipy `butter(4, Wn, btype='bandpass', output='sos')` (scipy order `N`=4 →
   `2N`=8 poles, matching MATLAB `butter(4,...,'bandpass')`), applied **zero-phase**
   with `sosfiltfilt` along the **fast-time (534) axis, per chirp per frame**, on the
   **complex** signal (filter real and imag; `padtype`/`padlen` fixed and recorded).
   `Wn` from the range gate via `HzPerM = 2·(B/Tchirp)/c`. No window and no FFT in
   this path — beat-frequency banding *is* the range gate. (An FFT-domain tapered-mask
   gate, per `filter_gpt_fft.m`, is available as a config alternative for ablation.)
4. **Signal reduction** on the full 534-sample filtered chirps (two branches):
   - **Option A** — mean across the 20 chirps → one complex 534-sample signal/frame.
   - **Option B** — per chirp: Hann-window + 534-pt FFT to **detect** the dominant
     beat bin in the range ROI, build a ±1-bin two-sided Hann-tapered mask, apply it
     to the (unwindowed) chirp FFT, IFFT, then average across chirps → one 534-sample
     signal/frame.
5. **EdgeTrim** the reduced signal by 32 samples each end → effective length 470.
6. **Channel**: `mag` = |s|, or `iq` = {real(s), imag(s)} standardized separately.
7. **Robust standardize** (median/MAD, above), per signal.
8. → WST (on the 470-sample signal).

**Deliberate ROADMAP departure — no Hamming window in the primary path.** ROADMAP
§3.2 lists "Hamming window; range FFT; SOS Butterworth bandpass" for preprocessing.
We apply a window only where an **FFT** is taken (the QC in-band-energy screen, Option
B's peak detection, the spectrogram baseline, and the optional FFT-gate variant) —
using a Hann (periodic) taper, matching the reference code. The primary Option-A path
takes **no** window before the time-domain zero-phase IIR bandpass, because
pre-windowing a fast-time signal ahead of `sosfiltfilt` tapers real signal energy at
the chirp edges (windowing suppresses FFT spectral leakage, which is only relevant to
the FFT-based steps); filtfilt edge transients are instead handled by EdgeTrim. This
departure is logged in HISTORY.md with this justification.

Self-consistency tests target this exact sequence: designed-filter magnitude
response passes in [f_lo,f_hi] and stops outside; forward-backward filter is
zero-phase (linear-phase/zero group delay on a test tone); Parseval energy sanity;
Option B's isolated peak lands in the ROI; fixed-seed determinism.

## QC screens & thresholds (frozen, computed before filtering)

Ported from `wst_integrity_check_dataset.m`, thresholds **frozen and justified
before evaluation** (not performance-tuned, so they do not enter CV). Each screen
writes a per-frame **reason code**; `manifest.py` stores per-frame codes and
per-subject/session removal counts.

| Screen | Stage | Rule (reject frame if…) |
|---|---|---|
| NaN/Inf | raw | any non-finite sample |
| Flatline/saturation | raw | any chirp's magnitude histogram (**200 bins** over that chirp's magnitude range) has a bin containing ≥ 25% of the 534 samples (≥134) |
| In-band energy ratio | raw, Hann+FFT **pre-filter** | in-band(gate) / total power < 0.30 |
| Robust RMS outlier | raw | robust-z of per-chirp RMS > 4.5 → **flag/log** (diagnostic; not sole reject) |

Rejection rule = NaN/Inf **or** flatline **or** low in-band energy (bin count 200 and
the 25%/0.30/4.5 thresholds are the reference values, frozen in `configs/`). If we
ever decide a threshold should be data-adaptive, it moves **inside inner CV**; until
then it is frozen.

**One fixed QC range gate for the shared population.** The in-band energy screen uses
a **single frozen gate — the wider 0.9–3.0 m band** — for **all** candidates and
baselines, so the QC-passing frame/session population is identical regardless of the
1–2 m vs 0.9–3.0 m *model* gate later chosen in inner CV. QC never varies with the
model gate. (The wider band is used for QC so a frame is not rejected for energy that
a wider candidate model gate would legitimately use.)

**Session eligibility after QC (pre-defined, no imputation).** A session is retained
only if **at least `ceil(0.5 × actual_frame_count)` of its frames survive QC** — the
threshold is computed from the **actual per-file frame count**, not a hard-coded 100
(10 GHz) or 125 (77 GHz), since files vary. Otherwise the whole session is dropped.
Dropped sessions are simply **absent** — never imputed from other subjects or
sessions. A subject may therefore have fewer than 5 sessions; the manifest records the
actual frame count, surviving-frame count and eligibility flag per session. If a
subject's **held-out** session is ineligible it is not scored (excluded from that
subject's session set), and if a subject loses so many sessions that it cannot serve
as an informative outer-test fold, that is recorded rather than back-filled. Per-
subject/session retained-frame counts and drop reasons are reported.

## WST parameterization (MATLAB invariance scale → kymatio)

MATLAB's `waveletScattering` takes an InvarianceScale in **seconds** and
`QualityFactors=[Q1,Q2]`; kymatio `Scattering1D(J, shape, Q, T, max_order)` takes an
integer octave count `J` and an averaging support `T` in **samples**. Mapping, per
tiling, at fs=520834 Hz on the trimmed length N=470:

- `T_samples = round(InvScale_ms · 1e-3 · fs)` → 0.20 ms→104, 0.30 ms→156,
  0.40 ms→208 samples (realized invariance within <0.2% of requested, since we round
  to the nearest sample). Set kymatio `T` to these directly.
- `J = ceil(log2(T_samples))` so the largest wavelet scale covers the averaging
  support → T1 J=7, T2/T3 J=8. Record the (requested ms, realized samples, J) triple
  and the approximation error for each tiling in HISTORY.md.
- `Q = (Q1, Q2)` from the three tilings: (10,4), (8,2), (6,2). `max_order = 2`
  (orders 0/1/2 kept).
- **Padding and output shape are measured, never assumed.** We do **not** hard-code a
  512-sample padded length or estimate `n_time` as `padded_len / 2^J` — kymatio's
  padding depends on the instantiated filter bank and is exposed via `pad_left` /
  `pad_right`, and it does not guarantee next-power-of-two. At build time we
  instantiate the **pinned** kymatio `Scattering1D` for each tiling, read back the
  actual padding, output time length, `n_paths`, and per-path metadata (order, `j`,
  `xi`) from the object / `scattering.meta()`, and **record those observed values**.
  The tests assert the *observed* padding and output shape (API ref:
  https://www.kymat.io/codereference.html), not a formula.
- **Averaging / log (mathematically defined).** Local averaging is by the low-pass of
  support `T`. The optional log transform is **order-aware**, because in kymatio
  **order-0** `S0 = x ⋆ φ` is a signed low-pass of the (median/MAD-standardized) input
  and **can be negative** (for both magnitude and I/Q inputs), while **orders 1–2 are
  modulus-based and ≥ 0**. Frozen rule when log is on: **orders 1 and 2 →
  `log(S + ε)`** with **ε = 1e-6** (fixed; the coefficients live on an O(1) standardized
  scale); **order 0 is left linear (never logged)**. (Signed `log1p` on order 0 is the
  documented alternative but is *not* used unless explicitly switched.) A test asserts
  **every branch — mag and I/Q, all three orders, log on and off — produces finite
  values.** log on/off remains an inner-CV-selected choice; this formula is what "on"
  means.
- **Feature families**: (a) pooled statistics — mean/std over global + first/second
  half per path; (b) raw-flattened scattering series. Both preserve a fixed,
  documented element order derived from the recorded path metadata.
- **Cross-backend test**: numpy vs torch WST agree to ≤1e-4 relative on a shared
  sample; only then may either back a reported run.

## Analysis unit — session-level primary (fixes pseudo-replication)

The label is one weight target per (subject, session); the ~100 frames within a
session are highly correlated. Broadcasting the target to every frame does not leak
subjects, but it **pseudo-replicates** and lets sessions with more QC-surviving
frames dominate. Therefore:

- **Primary classical analysis is session-level.** Per-session, aggregate the
  QC-passing frames' per-frame pooled-WST features into **one feature vector per
  session** by **concatenating the per-frame mean and the per-frame median** (both
  statistics, fixed — not competing aggregation choices and not selected by
  performance, so no extra CV axis). This gives **up to 80 observations** (16×5, minus
  any QC-ineligible sessions per the eligibility rule above), one per (subject,
  session), matched 1:1 to the target. LOSO is over subjects.
- The **raw-flattened** WST family and any per-frame view are **diagnostic only**
  (and feed the DL paths); they are session-aggregated before any classical
  session-level metric.
- **CNNs** train with **session-balanced sampling** (equal total weight per
  subject/session) and their per-frame outputs are aggregated to session level for
  inference.
- **All headline metrics are session/subject-level.** Per-frame numbers are labeled
  diagnostic and **never** carry frame-IID confidence intervals or headline claims.

## LOSO harness, nested-CV protocol, and no-leakage guarantee

**Single fold source.** `eval/splits.py` is the only place train/val/test indices
are created. It exposes a **nested LOSO API** that yields, per outer fold, the
**subject id sets** `(train_subjects, val_subjects, test_subject)`; `harness.py`
consumes this identically for sklearn and torch paths. No other module constructs
splits.

**Outer loop.** `LeaveOneGroupOut` over the **evaluable** subjects → **N_eval outer
folds** (N_eval ≤ 16; equal to 16 in the clean case where every subject keeps enough
sessions). The single held-out subject is touched only for final scoring.

**Evaluability after QC (predefined; drives N_eval).** Session eligibility (≥50% of
frames surviving QC) can remove sessions, so scoring counts are stated in terms of
evaluable subjects, not a hard 16:
- **Exp A (one unambiguous rule):** a subject is evaluable — `N_eval,A` — iff it has
  **≥1 eligible session**. A per-subject session-level MAE is well-defined from a
  single session, so one-session subjects are included **consistently** in every Exp A
  result and outer fold. Only subjects with **0 eligible sessions are dropped before
  outer splitting** (they never produce an empty fold). The only thing a one-session
  subject cannot contribute to is per-subject **correlation** (next bullet).
- **Exp B:** per-session, only subjects whose session s∈{S1..S4} is eligible enter
  that session's analysis; a subject is evaluable for the B aggregate if it has **≥1
  eligible S1–S4 session**. Report N_eval,B per session and for the aggregate.
- **Per-subject Pearson r** (descriptive) is computed only for subjects with **≥3
  eligible sessions**.
- **Bootstrap and Wilcoxon operate over N_eval subjects, not automatically 16**; the
  effective N is reported alongside every CI and test.
- **Missingness is reported by subject and session** (retained-frame counts, dropped
  sessions, reason codes), because QC failure may itself correlate with hydration or
  acquisition quality and must be visible, not silently absorbed.

**Inner loop (model/config selection), built exclusively from the outer-training
subjects** (15 when N_eval=16, otherwise N_eval−1)**:**
- Splitter: subject-grouped `GroupKFold` with an **adaptive fold count**
  `n_inner = min(5, n_train_subjects)` (5 in the full run: 15 training subjects; fewer
  when QC shrinks `N_eval` or in the smoke test). Inner CV **requires ≥3 training
  subjects**; below that the fold cannot select and the outer fold is reported as
  non-selectable rather than run with a degenerate split. Each inner-val subject set is
  disjoint from inner-train; the outer-test subject never appears inner.
- Selection metric: **session-level MAE** (aggregate to session, mean over inner-val
  subjects) — the same unit as the headline.
- Search space (bounded, enumerated in config): reduction branch {A,B} × channel
  {mag, I/Q} × tiling {T1,T2,T3} × {log on/off} × range-gate {1–2 m default,
  0.9–3.0 m} × model family × that model's small hyperparameter grid. The space is
  kept modest for tractability; if needed it is searched in a fixed staged order,
  but **every** data-dependent choice is made on inner folds only.
- Tie-break: lower session-level MAE, then simpler model (fewer effective
  parameters / smaller feature dim), then lower inner-fold variance.
- Stochastic models (incl. NN): a fixed seed set (5 seeds); inner metric = mean over
  seeds; selection on that mean. At outer scoring, **each seed's refit model predicts
  the held-out subject and the per-seed outer metric is computed; we report the mean
  over seeds ± sd** (seeds are scored separately and averaged — **not** ensembled into
  one prediction — so the outer procedure matches the inner selection rule and seed
  sensitivity stays visible).
- **Final refit for early-stopped NN/CNN** (there is no held-out validation subject
  once we train on all outer-training subjects): we take the **fixed epoch budget =
  median of the epochs selected across the inner folds**, then train the selected
  configuration on **all outer-training subjects for that fixed number of epochs**
  (no early stopping, no validation subject sacrificed). sklearn models simply refit
  on all outer-training subjects. Then score the held-out subject.
- The reported headline uses each outer-test subject's score **once**; a single
  configuration is **never** chosen by its outer-test scores.

**Fit-on-train-only — sklearn *and* torch.** Every fitted quantity is estimated from
training frames of training subjects only. sklearn: transforms live in a `Pipeline`
fit within the fold. torch: **CNN input normalization / spectrogram scaling
statistics, class weights, sampler weights, and early-stopping/checkpoint selection**
are computed from inner-train and monitored on inner-val subjects — never on the
outer-test subject. `harness.py` emits a **fit-audit artifact** per fold listing
every fitted quantity and the subject set it was estimated from.

**`tests/test_no_leakage.py`** (green from milestone 1) asserts, for every outer
fold the API yields: (a) `train/val/test` subject sets are pairwise disjoint and no
subject occupies two roles; (b) every frame maps to exactly one subject and no
held-out subject's frames appear in training for any session; (c) a **strong mutation
property test** — mutating the outer-test subject's **features and labels** must leave
**everything determined before scoring** bit-for-bit unchanged: the selected
configuration, the inner-CV scores, the chosen epoch budget, every fitted transform
parameter (sklearn and torch), the training-set predictions, and the final model
parameters. **Only the held-out subject's prediction/score may change.** Two
conditions keep this test valid: it runs **after QC/eligibility is frozen** and uses
**eligibility-preserving mutations** (values change but the frame/session membership
does not), so a mutation cannot legitimately alter the fold composition itself; and
the bit-for-bit comparison runs on a **deterministic CPU fixture** (fixed seeds,
single-threaded) rather than relying on GPU determinism. This catches any dependence
of selection or training on outer-test data, not just leaked scaler statistics.
The mutation property is asserted at **both CV levels**: in addition to the
outer-test mutation above, mutating an **inner-validation** subject must leave that
inner fold's fitted transforms and model parameters (functions of inner-train only)
bit-identical — only its validation predictions/scores, and hence possibly the
selected configuration, may change; folds where that subject is inner-*train* change
legitimately and are not constrained. The fit-audit distinguishes roles: every
inner-selection fit is estimated from exactly that fold's inner-train subjects, the
final refit from exactly the full outer-training set.
*Staging:* at milestone 1 the test runs against a test-local sklearn reference
procedure that defines the selection/refit contract `harness.py` must satisfy; at
milestone 6 it rebinds to the real `harness.py`. torch enters the environment at
milestone 4 (WST cross-backend validation), but the torch-path mutation assertions
stay skip-marked until the torch fit path exists in `harness.py` (milestone 6), and
must be green before any torch result is reported.

**Reporting the selection, not a single "winning model."** Nested CV may select
different reduction branches, tilings, gates, normalization, or model families in
different outer folds — the outer predictions are the output of the *procedure*, not
of one fixed pipeline. Every experiment therefore reports a **selection-frequency /
stability table** (how often each branch/tiling/gate/model was chosen across the
N_eval outer folds). The prose says "the selection procedure" rather than implying a
single pipeline produced all predictions; if one configuration does dominate, that is
shown by the table rather than assumed.

## Experiments

**A — Fluid-loss regression (headline).** Session-level LOSO regression of signed
Δm% from 10 GHz WST features. Report session-level MAE, RMSE, predicted-vs-actual r,
pooled and **per-subject**, with subject-cluster CIs (Stats below). Reported against
the **session-index-only baseline** (predict Δm% from time of day alone), which is
the number the radar must beat given the fasting clock/hydration confound.

**B — Clock-decoupling (crucial evidence; design locked before Exp A is examined).**
Tests whether radar predicts **between-subject** Δm% variation at a **fixed** time of
day, i.e. signal not attributable to the clock.
- **Exclude S0** (its Δm% is identically 0 → no between-subject variation).
- **Model structure — one pooled S1–S4 model on session-mean-residualized targets**
  (primary). A single regressor is trained on the pooled S1–S4 sessions of the
  training subjects, with the per-session mean removed from the target so what remains
  is between-subject variation; session identity is carried as the residualization,
  not as separate models. (Four session-specific models — with only the eligible
  training subjects per session, a variable count ≤15 — are a thin secondary
  robustness variant only.)
- **Residualization is a fitted quantity and is therefore computed train-only at
  every CV level** (fixes the leak of holding μ_s fixed across inner folds). The
  training population is **whatever subjects are eligible for session s** among the
  outer/inner-training set — never a hard-coded 15, since QC and per-session
  eligibility vary:
  - *Inner CV:* for each inner fold, compute each `μ_s` from the **inner-training
    subjects that have session s eligible**, then subtract it from both inner-training
    and inner-validation targets. Inner-validation/test labels are **never** used to
    form μ_s.
  - *Outer scoring:* after selection, recompute each `μ_s` from **all outer-training
    subjects eligible for s** and apply it to the outer held-out subject's target.
  - **Degenerate-fold rule:** a session s with **< 2 eligible training subjects** in a
    given fold has an undefined/unstable μ_s → that session is **dropped for that fold**
    (excluded from residualization, the objective, and reporting for that fold), and
    the drop is logged. It is **never** filled from validation/test labels or from
    other subjects.
  - The **session-mean baseline** predicts residual 0 (i.e. μ_s) using the same
    train-only μ_s at each level, and carries no subject-specific information.
- **Exp B has its own inner-CV selection objective** — the **equal-session
  residual-MAE** (mean over S1–S4 of the per-session residual MAE on inner-validation
  subjects), matching the reported aggregate — not Exp A's fluid-loss MAE.
- **Report** out-of-fold radar-vs-baseline performance **separately for S1–S4** and a
  **pre-specified aggregate** = equal-weight mean of the four per-session MAEs (see
  Statistics for how the aggregate CI and the paired test are each defined under
  missing sessions).

**C — Ordinal 5-class (secondary).** S0–S4 as an ordered task, session-level, ordinal
metrics only (adjacent-accuracy, MAE-in-class-units, quadratic-weighted κ) and the
LOSO confusion matrix. The model must be genuinely **ordinal** — a plain multiclass
classifier is not.
- **Sign convention:** work in **positive loss magnitude `L = −Δm%`** so class order
  increases monotonically with L (S0 lowest, S4 highest); this fixes the orientation so
  larger predictions → higher dehydration class (equivalently, reverse the cutpoint
  order — we choose the positive-L form).
- **Ordinal inner-CV objective:** selection uses an **ordinal** metric — primary
  **class-unit MAE** (mean |predicted class − true class|), with **QWK** as the
  secondary tie-break — **not** Exp A's fluid-loss MAE (the generic harness objective
  does not define ordinal selection).
- Two ordinal families: **(a)** threshold the continuous `L` predictor into 5 ordered
  bins with **monotone cutpoints** — the *config* is chosen on inner folds, then the
  **cutpoints are refit on all outer-training subjects only** before the outer
  prediction; **(b)** a proportional-odds / cumulative-link ordinal regressor as a
  comparison.
- **Class weighting** is inverse-frequency computed **per fold on inner-training data
  after QC** (never global, never using the held-out subject).
- **Fold-viability rules (predefined).** QC can leave a fold missing one of S0–S4,
  which makes cutpoints, inverse-frequency weights, or proportional-odds fitting
  undefined, and QWK is undefined on a single-class validation set. A fold/config is
  **non-evaluable** for the ordinal task if its **inner-training set lacks any of the 5
  classes**; such configs are skipped in ordinal selection (recorded), and if all
  configs are non-evaluable the fold contributes no ordinal score. When **QWK is
  undefined** on a validation set (one class present), selection **falls back to the
  primary class-unit MAE** for that fold rather than erroring. These rules are frozen
  at milestone 5.

**D — Baselines (fair by construction).** All baselines share **identical** outer
folds, inner subject-validation, QC-passed population, session weighting/aggregation,
seed set, and model-selection budget as Exp A. Concrete, frozen specifications (locked
at the config-freeze gate below, before any results are viewed):
- **(i) Raw-beat 1D-CNN (ROADMAP primary).** Input = the **QC-passed, chirp-mean beat
  signal with *no* bandpass, *no* trim, *no* WST** — 534 samples, **2 channels
  {real, imag}**, per-signal standardized only. This is the ROADMAP "1D-CNN on the raw
  beat signal": it tests whether a network learns from minimally-processed data what
  the WST+preprocessing pipeline extracts. Architecture: 3 conv blocks
  (Conv1d→BN→ReLU→MaxPool), channels 16/32/64, kernel 7, stride 1, pool 4, global
  average pool → FC → 1 scalar (Δm%). Adam, lr 1e-3, session-balanced sampler; inner
  early stopping / final-refit epoch budget per the harness rule above. **Frame→session
  aggregation is frozen as the median** of the frame-level predictions within a
  session (matching the session-level primary), applied identically to every CNN
  baseline.
  - **(i-ablation) Matched-preprocessing 1D-CNN** — identical net on the bandpassed,
    trimmed, Option-A reduced 470-sample I/Q signal, to isolate what the preprocessing
    contributes vs the raw input.
- **(ii) Spectrogram + small 2D-CNN.** Primary = STFT of the **raw** chirp-mean beat
  signal; ablation = STFT of the matched-preprocessed signal. **Hann window 64, hop
  16, nfft 128**, log-magnitude, per-frequency mean/std normalization fit
  **train-only**. 2D-CNN: 2 conv blocks (3×3, channels 16/32, 2×2 pool) → global
  average pool → FC → 1 scalar. Same optimizer / refit rule as (i).
- **(iii) Physics range-power baseline** (signal-domain, correctly labeled). In FMCW
  the beat-frequency axis maps to **range**, not to the 10 GHz RF/dielectric band, so
  this is a **target-range vs farther-range (background) energy ratio**, not a
  dielectric-frequency split. **Input = the QC-passed raw chirp-mean signal (534
  samples, *unfiltered* — no bandpass, no Option-B, no trim)**, so the range-FFT power
  is not pre-shaped by the range gate (bandpassing would suppress the background band
  it needs). Frozen definition: from the Hann-windowed 534-pt range FFT of that raw
  signal, **target band = range [0.9, 1.5) m (≈ 2.93–4.88 kHz beat)**
  ≈ the seated subject (~1 m); **background band = range [1.5, 3.0] m
  (≈ 4.88–9.77 kHz)** — **half-open at 1.5 m so the boundary bin belongs to exactly one
  band** (assigned to background), never double-counted. Scalar feature =
  **`log10( (P_target + ε) / (P_background + ε) )`** — ε added to **both** terms so the
  ratio is always finite (QC guarantees energy somewhere in the combined gate but
  **not** necessarily nonzero target-band energy, so `P_target = 0` must not yield
  `−∞`), with `P_·` = summed squared magnitude over the band's bins and
  **ε = 1e-12·(P_target + P_background) > 0**. A **finite-output test** asserts the
  feature is finite on every QC-passing session. (The symmetric log-ratio is the frozen
  definition, not a tuned option.) Rationale (kept honest):
  hydration changes tissue reflectivity and attenuation, altering how reflected energy
  distributes **across coarse target/scene range** (≈0.3 m resolution at 500 MHz — *not*
  tissue layers) and the total in-band power — a legitimate 1-D signal-domain scalar,
  with **no claim** that it separates the GHz dielectric spectrum or localizes hydration
  within tissue. The scalar → Δm% mapping is a **per-fold linear fit** (inside each
  outer-training fold, never global).
- **(iv) Session-index-only** baseline — frozen as the **categorical train-subject
  session mean**: predict `Δm%` for a held-out subject's session s as the mean `Δm%` of
  the outer-training subjects at that same session s (a 5-level lookup, S0–S4), **not**
  a fitted linear time trend. This is the pure clock/confound reference and matches Exp
  B's session-mean baseline.
- **Budget parity**: each learned-baseline family and the WST classical models are
  given the **same inner-CV configuration budget** (≤ K configs each, K fixed in
  config) and the same seed set, so "WST wins" is not an artifact of unequal search.

All baseline specs above are locked at the **config-freeze gate (milestone 5)**,
before any Exp A/D results are inspected; nothing here is chosen or adjusted after
seeing outer-test scores.

**E — Physics-grounded interpretability (beat/range domain, honest about limits).**
Identify which scattering paths drive the prediction.
- **Interpret the clock-decoupled (Exp B) model as primary — not Exp A.** Exp A
  predicts Δm%, which is confounded with time of day, so A-path importance can flag
  paths that encode **session/clock** rather than the between-subject signal. The
  **primary interpretability analysis therefore runs on the Exp B (residualized,
  clock-decoupled) model**, whose signal is **between-subject body-mass-loss variation
  at a fixed session** (Exp B reduces the clock confound but does **not** isolate
  hydration causally). Exp A importance may be shown for comparison but is **explicitly
  labelled not hydration-specific**.
- **A separate, pre-registered grouped interpretability CV — not permutation over the
  pooled OOF predictions.** The pooled LOSO predictions come from N_eval *different*
  refit models, so there is no single estimator to attribute importance to; worse, a
  donor subject permuted into a recipient's slot was generally in that recipient fold's
  training set (contamination). Instead we run a **dedicated interpretability analysis**,
  descriptive-only and fully separate from the reported A/B predictions:
  - A **single pre-registered interpretability configuration** (one tiling + one model
    family, fixed at milestone 5) is used throughout, so importances are comparable.
  - **Grouped subject CV with multiple held-out subjects per fold** (**4-fold
    subject-grouped CV**, ~4 validation subjects each). In each fold the config is fit on
    that fold's training subjects; permutation happens **only among the held-out
    validation subjects**, all excluded from that fold's training — so a permuted
    trajectory is never a training subject's.
  - **Structure-preserving, path-grouped permutation:** a path's **complete per-subject
    trajectory is permuted across validation subjects with session alignment kept**
    (S1↔S1, S2↔S2, …); the fold's fitted transform is applied **after** permutation; the
    model is re-scored on the **equal-session residual MAE**. Permutation **unit = one
    WST path's whole column group** (global/half mean-std pooling × session mean/median ×
    I/Q moved together). **Repeats = 50**; report mean ± sd of the importance drop,
    aggregated across the 4 folds. (**Leave-one-path-group-out refitting under LOSO** —
    drop a path group, refit, measure the MAE increase — is the documented alternative.)
  - Because this is a standalone descriptive CV that never touches model selection or the
    headline predictions, it **cannot influence selection or retuning**.
  - **Model-native coefficients** are reported additionally as descriptive quantities.
    Everything here is **descriptive, never confirmatory**.
- **Order-aware physical mapping (kept in beat/range space).**
  - **Order 0** (scaling coefficient): the low-frequency / scaling coefficient of the
    **per-signal median/MAD-standardized** signal — **no frequency/range band**, and
    **not** overall reflected level, since standardization has already removed absolute
    reflected power. Interpret only as residual low-frequency structure of the
    normalized signal.
  - **Order 1** (`xi_1`): a **beat/modulation frequency (kHz)** in the fast-time
    signal → **coarse target/scene range `r = c·f_b/(2·slope)`** (≈0.3 m resolution at
    500 MHz — a scene distance, **not** a tissue layer/depth).
  - **Order 2** (`xi_1, xi_2`): `xi_1` still → coarse range, but **`xi_2` is the
    modulation rate of the first-order envelope, not a second range** — a
    structural/temporal modulation descriptor, interpreted as such.
  A WST path frequency does **not** map to the instantaneous 10–10.5 GHz transmit
  frequency (that is tied to fast-time *position*, which the time-pooled features
  average out and cannot recover). Consistency with the Cole-Cole water-driven
  expectation (paper §II) is discussed **qualitatively** as supporting — not
  confirmatory — evidence; any RF-frequency-level claim would need a separate,
  time-localized analysis specified and validated on its own, and is out of scope.

**F — Confound check (clock-adjusted nested-model comparison).** Static covariates
(age, height, baseline mass, BMI) do not change within a subject, while the target
changes by session, so a covariate-only model cannot be a fair alternative to a radar
model that can encode time. The comparison instead **holds the clock fixed in every
model** and asks what radar adds on top of it — nested LOSO models:
1. **session index only** (the clock),
2. **session index + subject covariates**,
3. **session index + radar**,
4. **session index + radar + covariates**.
Only **nested contrasts** (differing by exactly one component) are interpreted:
**radar beyond clock = 3 vs 1**; **radar beyond clock+covariates = 4 vs 2**;
**covariates beyond clock = 2 vs 1**; **covariates beyond clock+radar = 4 vs 3**.
(3 vs 2 is *not* used — it adds radar and drops covariates at once.) If radar adds
nothing beyond the clock (± covariates), that is reported honestly.
- **The four models are genuinely nested only because the learner is fixed.** All four
  use **the same learner — ridge regression** — with the **same clock encoding**
  (session index as one-hot), **fold-local standardization**, **inner-CV-selected ridge
  λ**, and train-only preprocessing throughout; the models differ *only* by which
  feature block (covariates, radar) is present. Ridge's L2 penalty absorbs the
  collinearity among height / baseline mass / BMI (mass/height²), so **no separate VIF
  pruning is used** — this resolves the earlier "ridge or VIF" ambiguity in favor of
  ridge for all four. (Were the learner or preprocessing to differ across the four,
  these would be *matched predictive comparisons*, not nested models — which we avoid.)
- **Algebraic-coupling caveat + sensitivity analysis.** Baseline mass `m0` sits in the
  **denominator** of `Δm% = (m − m0)/m0`, and `m0` and BMI (∝ `m0/h²`) are covariates —
  so part of any apparent covariate contribution may be **algebraic, not
  physiological**. Keeping Δm% as the **primary** target, we pre-specify a **sensitivity
  analysis**: (a) covariates restricted to **age + height only** (drop mass and BMI),
  and (b) target switched to **signed kg change `m − m0`** (negative = loss, same
  direction as Δm%; not "absolute"). Agreement of the radar conclusion across these is
  required before any covariate claim.
- **Same radar representation in 3 and 4.** Within each outer fold, models 3 and 4 use
  the **identical selected radar feature set**, so the covariate contrast (4 vs 3) is
  not confounded by a different radar representation.
- **Contrast status & correction.** **Pre-specified primary family (Holm over 2):**
  radar beyond clock (3 vs 1) and radar beyond clock+covariates (4 vs 2) — the questions
  the study is built to answer. **Exploratory:** the covariate contrasts (2 vs 1,
  4 vs 3). (Per Statistics, all p-values/CIs remain conditional — "primary" denotes
  pre-registration and emphasis, not a confirmatory inferential guarantee.)
**Heart rate was reportedly collected but is not in the delivered data**; skin
temperature and glucose are stated as uncontrolled (ROADMAP §8).

**G — Cross-band 10+77 GHz fusion (original cohort only).** 10 and 77 GHz have
different, unsynchronized frame counts (100 vs ≈125/session), so fusion is at
**subject-session level** (each band aggregated to session) unless synchronized frame
correspondence can be demonstrated.
- **Primary 77 GHz feature = Doppler (slow-time) content, on I/Q, fused per-Rx in
  feature space.** The reference removes slow-time clutter (`filter_gpt_butterworth77.m`
  subtracts the per-fast-time-bin **mean over chirps**) and then averages over chirps
  (`chirpavg_and_fuse_batch.m`) — two linear steps whose composition drives `chirpAvg`
  and `fused_mean` to **≈ 0**, so a chirp-mean fast-time signal is **not** the feature.
  We scatter the **per-range-bin slow-time signal** (Doppler content). Two consequences
  drive the exact form:
  - **Channel = I/Q, not magnitude.** A moving scatterer's slow-time return is ≈ a
    complex sinusoid `A·e^{j2π f_D nT}`, whose **magnitude is nearly constant** — taking
    magnitude before WST discards the **phase rotation that carries the Doppler
    frequency**. So the primary channel is **I/Q** (real & imag scattered separately);
    magnitude is not claimed to preserve Doppler.
  - **Rx fusion in feature space, not coherent complex averaging.** Coherently averaging
    complex data across the 16 **uncalibrated** receivers can cause **phase
    cancellation**. So we do **per-Rx WST** and fuse **in feature space** (mean primary,
    median secondary), never a coherent complex Rx mean before WST.
  A **nonzero-energy assertion** guards every signal fed to WST. If a **fast-time** WST
  branch is used at all (secondary), it must **either omit the slow-time mean
  subtraction** or **use a nonzero across-chirp statistic (RMS/magnitude)** — never the
  signed chirp mean.
- **Exact primary chain (executable, per subject·session) — WST on the *slow-time*
  signal so its ms invariance scales are physically real.** After a Doppler FFT the 256
  samples are indexed by Doppler *frequency*, not time, so `fs = PRF`/ms scales would be
  meaningless there; we scatter the **pre-FFT slow-time (across-chirp) signal** at
  `fs = PRF`, where 20/40/60 ms are genuine time scales (256 chirps span ≈131 ms). Chain,
  per frame **and per Rx**: **(1)** MTI — subtract the per-fast-time-bin mean over the
  256 chirps; **(2)** fast-time Butterworth bandpass (2–4 m gate); **(3)** Hann window on
  the fast-time axis; **(4)** range FFT (256-pt, fast-time); **(5)** crop range to the
  2–4 m bins. Then **slow-time I/Q WST, per Rx**: **(6)** for each retained range bin,
  WST the 256-point complex slow-time series as **two channels {real, imag}**
  (`fs = PRF`, invariance 20/40/60 ms); **(7)** average the scattering matrices across
  range bins → one per-Rx feature; **(8)** **feature-space Rx fusion** = mean over the 16
  per-Rx features (median secondary); **(9)** pool (mean/std over global + halves) →
  per-frame vector; **(10)** session pooling = concatenated per-frame **mean + median**.
  *(A spectral variant — Doppler FFT then WST the magnitude spectrum as a shape
  descriptor — is allowed only as a secondary branch, with scales expressed in
  Doppler-frequency/bin units and no temporal interpretation.)*
- **Frozen 77 GHz pipeline** (locked at milestone 5; concrete, no data-dependent
  fitting):
  - **QC:** same rule structure as 10 GHz with **fixed** numbers for 256-sample
    fast-time — flatline histogram **128 bins**, reject a chirp if any bin ≥ 25% of 256
    (≥64); in-band energy ratio < 0.30 on the 2–4 m gate; NaN/Inf. Frozen, not tuned.
  - **Session eligibility:** retained iff **≥ `ceil(0.5 × actual_frame_count)`** frames
    survive QC (from the file's real frame count, not an assumed 125).
  - **WST tilings — re-parameterized from fixed milliseconds, not fitted.** Use the
    reference 77 GHz tilings (`wst_extract77.m`): fast-time `Q=[8 4],[6 4],[4 2]` at
    invariance 0.08/0.16/0.20 ms; Doppler `Q=[8 4],[6 4],[4 2]` at 20/40/60 ms; each
    `T_samples = round(ms·1e-3·fs_axis)` at the axis's own rate — **derived, no
    outcome-based fitting** (this is what "re-parameterized" means).
  - **Input domain:** slow-time (Doppler) **I/Q WST with per-Rx feature-space fusion is
    primary**; the spectral Doppler-FFT WST variant and fast-time WST (with the nonzero
    fix) are **secondary variants**. (Coherent complex Rx averaging is explicitly *not*
    used.)
- **Population.** Distinguish **`N_subjects,G`** = number of subjects with ≥1 matched
  cell, from the **number of matched subject-session cells** = (subject, session) cells
  where **both** bands have an eligible session (the intersection). Both are reported.
- **Semantic 77 GHz axis check (shape assertion is not enough), run on RAW data.**
  Because fast-time and chirps are both length 256, asserting `(16,256,256,125)` cannot
  detect their interchange. Before fusion we add a **signal-domain disambiguation on the
  raw cube *before* any clutter subtraction** (MTI would remove the static subject and
  defeat the check): a range-FFT along the *proposed fast-time* axis must concentrate
  energy at the expected 2–4 m target range, while an FFT along the *proposed chirp*
  axis must concentrate near **zero-Doppler** (a mostly-static seated subject). The axis
  assignment is accepted only if this structure appears on the expected axes
  (cross-checked against the MATLAB dimension usage in `chirpavg_and_fuse_batch.m`);
  otherwise the loader fails.
- **Models & fusion (matched population, attributable delta).** All of **10-only,
  77-only, and fused** are **trained and scored on the exact matched subject-session
  intersection** (cells where both bands are eligible) — not merely identical subject
  folds — so no model sees more data than another and the delta is attributable. The
  **pre-specified primary fusion contrast is fused vs 10-only** (10 GHz is the primary,
  extensible modality; the question is whether adding 77 GHz helps), reported with a
  subject-cluster CI.
- **Primary combiner = a constrained convex weight** (not open-ended stacking):
  `pred_fused = α·pred_10 + (1−α)·pred_77`, `α ∈ [0,1]`, fit on **cross-fitted
  (out-of-fold) base-model predictions of the outer-training subjects** — *never* base
  predictions on their own training data. **Objective = subject-balanced OOF MAE** over
  the matched-session population: the objective averages **per-subject** MAE (equal
  weight per subject), so subjects with more surviving sessions do not dominate `α`.
  **Deterministic selection:** evaluate α on a **fixed grid {0, 0.05, …, 1.0}** (21
  points), pick minimum subject-balanced OOF MAE, and **break ties toward the α closest
  to 1.0** (max weight on the primary 10 GHz band); the chosen α is **recorded per outer
  fold**. **Stochastic base models — seed pairing (no silent ensembling):** pair seed
  `k` across bands (10 GHz seed `k` with 77 GHz seed `k`); compute the inner objective
  per paired seed; select **one α from the mean inner objective across the 5 paired
  seeds**; then produce **5 fused predictions (one per paired seed) scored separately**
  and summarized per the Statistics seed-collapse rule — never ensembled into one
  prediction. Base models are then refit on all outer-training subjects and the frozen α
  applied. **Feature-level fusion** (concatenated session-level features, scaler fit
  train-only) is a secondary variant; every scaling/fusion quantity is fit train-only
  and never touched by outer-test scores.
- No claim of generalization beyond the cohort.

**H — Statistics (pre-specified below).**

## Statistics (16-subject cohort; scoring over N_eval evaluable subjects, pre-specified)

- **Resample subjects, not frames or sessions.** Uncertainty comes from a
  **subject-level cluster bootstrap**: resample the **N_eval evaluable subjects** with
  replacement, **B=10000**, carrying each resampled subject's full set of sessions. CI
  type: BCa, falling back to percentile if BCa is unstable at small N (recorded). The
  effective N_eval is reported with every CI.
- **Seeds are collapsed before inference — they do not inflate N — and the collapse
  is metric-type-aware.** Seeds are never treated as observations (never `N_eval × 5`
  independent rows).
  - *Per-subject additive metrics (per-subject MAE):* average each subject's 5 per-seed
    values into one value per subject, **then** bootstrap/Wilcoxon over the N_eval
    subject values.
  - *Pooled / nonlinear metrics (pooled Pearson r, RMSE, class-unit MAE, QWK, and
    **adjacent accuracy**):* these are **not** averages of per-subject values, so within
    **each** bootstrap resample of subjects we **recompute the metric separately for
    each of the 5 seeds on the resampled data, then average the metric across seeds** to
    get one value per resample; the bootstrap distribution is over resamples. Seeds are
    summarized, never counted as observations.
- **CIs for every headline metric across A–G, not just MAE.** Each headline metric gets
  a subject-cluster bootstrap 95% CI computed the same way: Exp A **MAE, RMSE, and
  pooled predicted-vs-actual r**; Exp B per-session S1–S4 and aggregate MAE; Exp C
  ordinal **adjacent-accuracy, MAE-in-class-units, quadratic-weighted κ**; **Exp F the
  four nested incremental contrasts** (per-subject MAE differences); **Exp G the primary
  fusion delta (fused − 10-only)** on the matched intersection. The per-subject summary
  statistic for the primary regression metric is that subject's session-level MAE;
  headline = mean over the N_eval per-subject values with its CI, and the **per-subject
  distribution** is always shown, not just the mean.
- **Undefined metrics inside a resample are skipped and counted, never silently
  dropped.** A resample can make **QWK** undefined (a single class present) or **Pearson
  r** undefined (zero variance in predictions or targets). Such a replicate is **skipped
  for that metric only** and the **skipped count is reported**; if **> 5%** of replicates
  are skipped for a metric, its CI is flagged **unreliable** rather than quietly
  narrowed. MAE/RMSE are always defined and are not affected.
- **Per-subject Pearson r is descriptive only.** Each subject has ≤5 sessions
  including the fixed-zero S0 anchor, so per-subject correlation is unstable — reported
  as a descriptive spread with **no CI and no headline claim**. Correlation as a
  headline is the **pooled** predicted-vs-actual r across session points (with a
  subject-cluster CI), and it is additionally reported on **S1–S4 only** because the
  common S0=0 anchor inflates the pooled value.
- **Exp B — the aggregate and the paired test are different estimands, defined
  separately** (they diverge when subjects have different eligible sessions):
  - *Aggregate CI (session-weighted):* within each subject-resample, **recompute the
    four per-session residual MAEs and average them with equal weight**; the bootstrap
    is over that session-weighted aggregate. This matches the reported primary aggregate.
  - *Paired test (subject-weighted):* the per-subject scalar for Wilcoxon is each
    subject's **mean residual-MAE over its eligible S1–S4 sessions** (radar minus
    baseline); to keep the test's estimand clean this uses **complete cases** (subjects
    with all four S1–S4 sessions eligible), with the complete-case N reported. The two
    are presented as answering two different questions, not conflated.
  - *Status & correction:* the **equal-session aggregate is the single pre-specified
    primary test** for Exp B (radar vs baseline). The **four per-session S1–S4 results
    are exploratory**, reported as a **Holm family of 4** — never elevated to primary by
    picking the best session. (All Exp B p-values/CIs remain conditional per Statistics.)
- **How the "strongest baseline" is chosen — no double-dipping.** Two comparisons,
  both defined in advance:
  - **Pre-registered primary:** radar vs the **session-index-only** baseline (the
    clock/confound null). Fixed now; not chosen from outer-test scores.
  - **Secondary #1 (the composite procedure — one comparison):** radar vs a **single
    composite learned-baseline procedure** that selects the best learned family (CNN /
    spectrogram / physics) **inside each outer fold by inner CV**, exactly as the radar
    model is selected. This is "radar selection procedure vs baseline selection
    procedure" — one comparison, **not** Holm-corrected on its own.
  - **Secondary #2 (per-family, exploratory):** radar vs each learned family separately
    — CNN, spectrogram, physics — a **Holm family of exactly 3 comparisons**, all
    labeled exploratory. (The session-index primary is outside this family.) No family
    is chosen by its outer-test score.
  - Test for each comparison: per-subject metric differences over the N_eval
    subjects, **Wilcoxon signed-rank** plus a cluster-bootstrap CI on the mean
    difference (effective N reported).
- **Selection hygiene**: the reported model is the nested-CV winner selected on inner
  folds; outer-test scores are used once for reporting and never to re-select. These
  rules are fixed now even though execution is milestone H.
- **CIs/p-values are labeled conditional / exploratory.** The cluster bootstrap
  resamples **already-generated LOSO predictions**, so it captures sampling variability
  of the fixed, selected models but **not** model-refitting or model-selection
  uncertainty; and LOSO subject errors share heavily overlapping training sets, which
  weakens the independence assumption behind Wilcoxon. We therefore (a) label all CIs
  and p-values **conditional/exploratory**, and let **effect sizes and the per-subject
  distribution carry the scientific interpretation**, not a significance threshold; and
  (b) as a robustness check for the **classical (cheap) models only**, repeat the full
  nested select+refit procedure **inside** the resampling to show how much selection
  variance widens the intervals — with two required safeguards:
  - **Grouped resampling, no split subjects.** Resample subjects with replacement to a
    bootstrap cohort and run LOSO over the **distinct** subjects; **all copies of a
    drawn subject keep one shared group ID and occupy the same role** (never one copy
    training while another is held out). Multiplicity is applied **after roles are
    assigned**: estimators that support `sample_weight` (Ridge, SVR, GBM, …) receive
    multiplicity as a **weight**; estimators that do **not** (e.g. KNN) receive
    **duplicated training rows** carrying the same group ID — and no estimator for which
    duplication is meaningless is silently weighted. Any estimator where neither weight
    nor duplication is well-defined is **excluded from this robustness analysis** (and
    that exclusion is stated).
  - **Frozen replicate/viability rules.** `R = 200` replicates (separate from the
    B=10000 conditional bootstrap). A replicate is **skipped** if the resampled cohort
    has **< 4 distinct subjects** (LOSO not runnable) or, for ordinal, **< all 5 classes
    present**; skipped replicates are **counted and reported**, never silently dropped.
    A CI is reported only if **≥ 100 of 200** replicates succeed; otherwise the
    robustness bootstrap is reported as inconclusive and the conditional-CI wording
    stands on its own.
  The DL baselines are too expensive to refit inside the bootstrap and stay explicitly
  conditional.

## Reproducibility / run provenance

`provenance.py` writes, with **every run**: raw-file SHA-256 hashes, the fully
resolved config, the fold manifest (subject role per fold), package versions, git
commit, device, seed(s), and (on IBEX) the Slurm job ID. Every figure and table is
regenerable by one command from saved intermediate artifacts.

## Compute / IBEX

- CPU/local: loading, QC, preprocessing, WST extraction, all classical
  regression/classification, Exp A/B/C, stats. Smoke tests use a **≥6-subject**
  subset (see Verification): with one outer subject held out, ≥5 remain so the
  adaptive `GroupKFold(min(5, n_train))` inner loop runs a genuine multi-fold split
  rather than collapsing.
- IBEX (Slurm, GPU): DL baselines (1D-CNN, 2D-CNN) and any NN, as `sbatch` batch jobs
  under `scripts/ibex/`. Same Python entry points; only config differs (device,
  subset size, epochs, data root). No separate GPU-only code paths; no GPU training
  in interactive runs.

## Risks flagged

1. **Small-sample regression**: up to 80 session points (fewer after QC-ineligible
   sessions), 16 subjects, Δm% span ≈2%. LOSO
   MAE and subject-cluster CIs will be wide. Inherent, not fixable — reported
   honestly (per-subject spread + cluster-bootstrap CIs). Consistent with the
   feasibility framing; sets a realistic bar for the J-BHI upgrade path.
2. **Clock/hydration confound is structural** to the fasting design. Exp B mitigates
   the interpretation but cannot fully break it; no causal claim is made.
3. **ROADMAP §2 frame bookkeeping** ("20 frames/file, 5 files/session") does not match
   the on-disk layout (100 frames in one file per session). Loader follows the actual
   files; net 100/session is unchanged.
4. **77 GHz volume/compute** (~23 GB, 4-D filtering + RD maps + per-Rx WST) is heavy;
   confined to the late fusion milestone, at session level, no generalization claim.
5. **Reference-guided, not reference-matched.** Numbers will differ from MATLAB; that
   is expected. Correctness rests on Python-native self-consistency checks and the
   no-leakage protocol; every deliberate departure is logged.
6. **Search-space size vs N=16.** A broad inner search risks selection variance on
   few subjects. Mitigated by a modest, enumerated, staged search space and by
   reporting the selection's stability across outer folds.

## Verification (how we'll prove it works end to end)

- `pytest` green: `test_no_leakage.py` (sklearn + torch fit-audit), `test_preprocess.py`,
  `test_wst.py` (path structure + numpy/torch cross-backend), loader/manifest/
  ground-truth/metrics tests.
- Preprocessing self-consistency: filter magnitude response matches the target band;
  forward-backward filter is zero-phase; energy/Parseval sanity; Option B peak in ROI.
- WST self-consistency: correct path structure per tiling; near-invariance to small
  time shifts; deterministic under fixed seed; numpy≈torch.
- **Smoke run (≥6 subjects)**: `experiments/run_regression.py --config
  configs/exp_a_regression.yaml --subset 6subjects` completes on CPU, actually runs
  the nested outer+inner loops (a real multi-fold inner split via the adaptive
  `GroupKFold`, or singleton grids on the same fold code), emits a predicted-vs-actual
  scatter and a metrics JSON with seeds and provenance, and **asserts no subject
  occupies more than one of train/val/test**.
- IBEX dry run: one `sbatch` script submits a DL baseline with only config changed
  from the local smoke config; the fold API and fit-audit artifact are identical.
- Full Exp A: session-level LOSO MAE/RMSE/r, pooled and per-subject, against the
  session-index-only baseline, with subject-cluster CIs, as a regenerable figure/table.
- Journal upkeep is part of "done": **HISTORY.md** gains an entry per resolved attempt
  (what was tried, pass/fail and the real reason, concrete parameter values and their
  rationale — failures kept, newest-first); **SECOND_CHAPTER.md** gains a distilled,
  provenance-complete section as each ROADMAP §7 milestone closes; **superseded or
  invalidated code/results are moved to `archive/`** with the move noted in HISTORY.md;
  and **HANDOFF.md is refreshed only when explicitly requested** (never automatically).
