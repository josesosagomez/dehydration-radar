# Dehydration Radar — Rebuild & Extension Roadmap

## 0. What this project is

Rebuild the analysis behind the paper *"From Radio Waves to Wellness: Radar and
Machine Learning for Noncontact Dehydration Assessment"* from scratch in Python
(the original is MATLAB), and extend it into a methodologically rigorous study
suitable for a PhD thesis chapter and a journal submission.

**Primary target venue:** IEEE Transactions on Instrumentation and Measurement
(T-IM). **Upgrade path:** IEEE J-BHI, reachable only if the fluid-loss
regression comes out genuinely strong under leave-one-subject-out evaluation.

**Primary modality:** the 10 GHz CN0566 radar (the one still available for future
work). The 77 GHz Inras data exists only for the original 16-subject fasting
cohort. **Originally used solely for cross-band fusion, it is promoted to a full
parallel primary arm: Experiments A–F run on 77 GHz too (band-appropriate
parameters), while 10 GHz remains the sole headline and the cross-band fusion
(Experiment G) is retained.** *(A-M5-1, 2026-07-23; owner-approved.)*

**The core reframe:** the published paper reports frame-level, subject-dependent
5-class accuracy (~96–98%). That number is inflated by session-block leakage and
is not defensible at a serious venue. This rebuild replaces it with an honest
pipeline whose headline result is **regression of fluid loss**, validated against
measured body-mass change, under **leave-one-subject-out** cross-validation.

## 1. Non-negotiable methodological principles

These are invariants. Every experiment must satisfy them. There should be an
automated test that fails the build if any is violated.

1. **Group-aware leave-one-subject-out (LOSO).** Splits happen at the *subject*
   level. No frame — of any session — from a held-out subject may appear in
   training. Frame-level random splitting is forbidden as an evaluation protocol.
2. **No leakage in preprocessing.** Any fitted transform (standardization, PCA,
   feature selection, scaler statistics, class weights) is fit on the training
   folds only, inside the CV loop, and applied to the held-out subject.
3. **No test-set tuning.** WST tilings, model hyperparameters, and thresholds are
   selected by nested CV or a held-out *subject* validation split — never by
   picking the configuration that scores best on the test subjects.
4. **The primary target is continuous.** Fluid loss as a percentage of baseline
   body mass, `Δm%(subject, session) = (m(session) − m(S0)) / m(S0) × 100`.
   The 5-class task is secondary and is evaluated with ordinal metrics.
5. **Honest reporting.** Report the per-subject distribution of performance, not
   only the pooled mean; report confidence intervals; state confounds explicitly.

## 2. Data inventory

| Item | Detail |
|---|---|
| Subjects | 16 (fasting cohort, Ramadan 2024) |
| Sessions | 5 per subject: S0 (8am) … S4 (4pm), 2 h spacing |
| 10 GHz frames | 534 × 20 complex per frame; 20 frames/file; 5 files/session → 100/session; ~7500 after QC |
| 77 GHz frames | 256 × 256 × 16 per frame; 25 frames/file; 5 files/session → 125/session; ~9500 after QC |
| Ground truth | Body mass per session → Δm; **this is the only objective hydration reference** |
| Lost | Temperature/humidity logs are no longer available |
| Also recorded | Heart rate before each acquisition (use for a confound check) |

**Before writing any loading code**, inspect one raw 10 GHz file and the original
MATLAB loader to confirm on-disk layout, dtype, endianness, and axis order. Do
not assume the format from the paper alone.

## 3. Pipeline to port (10 GHz primary)

Reproduce the *intent* of the paper's processing, then improve where noted. The
MATLAB in `matlab/` is a **design reference only** — a guide to how the original
processing was done. All reported results come from Python alone; MATLAB is not
mentioned in the paper or thesis chapter, and matching its numeric values is not a
goal. Where the reference makes a choice that is arbitrary, internally inconsistent,
or improvable, take the better option and log the reason.

1. **QC / integrity screen** — flatline detection, robust outlier test, in-band
   energy ratio screen. Log how many frames each screen removes, per subject.
2. **Preprocessing** — Hamming window; range FFT (534-point, no zero-padding);
   4th-order SOS Butterworth bandpass, zero-phase (`filtfilt` equivalent), gating
   the 1–2 m range (≈3.25–6.51 kHz beat). Implement both signal-reduction
   branches from the paper: (a) mean across the 20 chirps, and (b) frequency-domain
   peak isolation with a tapered mask. For each, keep both the Magnitude branch
   and the Real/Imag (I/Q) branch.
3. **WST features** — port the three tilings (Q=[10,4]@0.20 ms, Q=[8,2]@0.30 ms,
   Q=[6,2]@0.40 ms). Produce both feature families: pooled statistics (mean/std
   over global + two halves) and raw flattened scattering time series. Use
   `kymatio` (Python). Validate the WST on Python's own terms — correct path
   structure per tiling, near-invariance/stability to small time shifts, and
   determinism under a fixed seed — not by numeric comparison to MATLAB.

## 4. Experiments (in priority order)

**A — Fluid-loss regression (headline).** Predict `Δm%` from 10 GHz WST features
under LOSO. Report MAE (in % body mass), RMSE, and predicted-vs-actual
correlation, pooled and per-subject. This is the paper's main result.

**B — Clock-decoupling analysis (the crucial evidence).** Within a *fixed*
session, subjects have lost different amounts of fluid (Fig. 10 spread), but the
time of day is identical. Test whether radar features predict *between-subject
variation in Δm at a fixed session* better than a session-mean baseline. A
positive result is the strongest available argument that the signal tracks fluid
loss and not merely time-of-day, without needing a new experiment. Make this a
headline analysis, not a footnote.

**C — Ordinal 5-class classification (secondary).** Keep S0–S4 as an ordered
task. Report ordinal metrics: adjacent-accuracy, mean absolute error in class
units, quadratic-weighted kappa — not plain accuracy. Show the confusion matrix
under LOSO (expect adjacent-class confusion).

**D — Baselines.** Contest "WST wins" with: (i) a 1D-CNN on the raw beat signal,
(ii) a spectrogram + small 2D-CNN, (iii) a simple physics baseline (reflected
in-band power / two-band power ratio). All under the same LOSO harness.

**E — Physics-grounded interpretability.** Identify which scattering paths /
frequency bands drive the prediction, and map them onto the Cole-Cole dielectric
response set up in Section II of the paper. Alignment between the informative
band and the expected water-driven permittivity shift is supporting evidence the
signal is physical.

**F — Confound check.** Show the radar prediction is not simply explained by the
recorded heart rate. State plainly that skin temperature and glucose could not be
controlled (temperature logs lost; glucose never measured).

**G — Cross-band 10 + 77 GHz fusion (original cohort only).** Feature- or
decision-level fusion on the 16-subject cohort, presented as complementary
evidence. 10 GHz remains the primary, extensible modality; do not claim the
fusion result generalizes beyond this cohort.

**H — Statistics.** Per-subject performance spread, bootstrap CIs, and a
significance test comparing the best model against the strongest baseline.

## 5. Engineering & reproducibility standards

- Config-driven experiments (one config file per run; no hard-coded params).
- Global seed control; log seeds with every result.
- A `test_no_leakage.py` that asserts train/test subject disjointness for every
  split the harness produces. This test must exist early and stay green.
- Every figure and table in the results is regenerable by a single command from
  saved intermediate artifacts.
- Environment pinned (`uv`/`conda` + lockfile). Python 3.11+.
- Clear separation: `src/` (library), `configs/`, `experiments/`, `results/`,
  `figures/`, `tests/`.

## 6. Deliverables mapped to the paper

- A results table replacing the paper's Table IV (now LOSO, regression-led).
- A revised related-work comparison (paper's Table V) reframed as
  protocol/granularity, not a raw-accuracy leaderboard.
- Regenerable figures: LOSO predicted-vs-actual scatter, per-subject error,
  clock-decoupling result, ordinal confusion matrix, interpretability map,
  fusion result.

## 7. Suggested build order (milestones)

1. Repo scaffold + config system + `test_no_leakage.py`.
2. Data loader + QC screens, verified against a sample file and the MATLAB code.
3. Preprocessing, validated by self-consistency checks (filter response, zero-phase,
   energy sanity) — not by diffing MATLAB.
4. WST feature extraction + path-structure / shift-stability checks.
5. **77 GHz front-end: loader → QC → preprocessing → slow-time I/Q WST — the
   parallel-arm build (first IBEX milestone).** *(A-M5-2, 2026-07-23; owner-approved.
   Note: `plans/implementation_plan.md` inserts a config-freeze gate before Experiment A
   that this list does not, so from here its milestone numbers run one ahead.)*
6. LOSO harness + fluid-loss regression (Experiment A) — on both bands.
7. Clock-decoupling analysis (Experiment B).
8. Ordinal classification (C) + baselines (D).
9. Fusion (G), interpretability (E), confound check (F), stats (H).
10. Figure/table generation for the chapter.

## 8. Honesty guardrails (do not violate)

- Do **not** report frame-level random-split accuracy as a headline number.
- Do **not** claim the study causally isolates hydration from time-of-day. The
  fasting design confounds hydration with the clock; the paper must say so.
- Do **not** overclaim clinical readiness. This is a feasibility/measurement
  study with a single objective reference (body mass) on 16 subjects.
