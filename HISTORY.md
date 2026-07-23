# HISTORY — implementation log

Running record of every attempt, newest-first. Each entry: what was tried, whether it
succeeded/failed **and why**, and the concrete parameter values + reasoning. Failures
stay in the log. A new session reads only the most recent entries to orient.

---

## 2026-07-23 — **MILESTONE 4 COMPLETE.** Definition of done met in full.

**D1 — mandatory suite, no private data.** `uv run pytest` → **396 passed, 12 skipped**
(was 319/11 at M3 close; +77, of which 66 are in `test_wst.py`).
**D2 — real-cohort acceptance.** `uv run pytest --realdata` → **407 passed, 1 skipped**
(T18 only, as designed until M6).
**D3 —** `experiments/run_wst.py` wrote and re-verified
`results/wst/wst_diagnostics_10ghz.csv` (73 rows) + provenance; ~12 min cohort pass.
**D4 —** `tests/test_no_leakage.py` **byte-for-byte unmodified since M1**
(`git diff f3fbade HEAD -- tests/test_no_leakage.py` empty) and green.
**D5 —** HISTORY.md carries an entry per resolved step, including the dtype-fork
resolution (step 1), the measured geometry (steps 3–4), and the cohort finding + pooling
performance fix (step 7).
**D6 —** SECOND_CHAPTER.md §3 "WST features" written: the ms→(J,T) mapping, the measured
padding/border decision, the order-aware log with the corrected ε rationale, the pooling
degenerate-std departure, the session unit, the single-backend policy, and the cohort
characterisation.
**D7 —** amendments **A-M4-1..A-M4-6** are live in `plans/implementation_plan.md`
(§Library choices, §WST parameterization, §Feature families, the repo tree, Build order §4);
the two plan documents agree.

**The invariant held.** WST + pooling is a per-frame function of one frame plus frozen
constants: batched extraction is bit-identical to single-frame (T-W16), the whole extraction
is deterministic (T-W8), and the only cross-frame step is the declared session
mean+median. Nothing is fitted, so nothing enters the CV loop — `test_no_leakage.py` is
untouched.

**Milestone-4 scoreboard.** 3 new source modules
(`features/{wst,pooling,extraction}.py`) + `features/__init__.py`, 1 new experiment entry
point, 1 new test module (66 tests), 1 new config field (`wst.backend`) validated at load.
torch entered the env (float32-only frontend); the strict cross-backend tolerance was kept,
not loosened. Two facts discovered empirically — the falsified O(1)-coefficient assumption
and the pooling hotspot — neither of which changed a frozen parameter.

**Open for M5 (config freeze):** the complete A–G protocol freeze, including the WST search
space (tiling × log on/off) and the 77 GHz decisions; the parked 77 GHz flatline rule; the
first IBEX configs/scripts. torch's T18 mutation leg still activates at M6. Nothing committed
yet — awaiting the owner's word.

## 2026-07-23 — M4 follow-up diagnostic: **is the order-2 scale stable enough that a
## data-derived ε would be leakage-safe? Yes fold-to-fold (<1%), but subjects vary ~14%.**

Read-only diagnostic on the existing cohort CSV (no WST recompute, no frozen parameter
touched) prompted by an owner question: if ε were computed per LOSO fold from training
subjects, would the value move much fold to fold? A near-constant ε means adapting it is
essentially leakage-free (the number barely depends on which subjects you use).

**Method.** For each held-out subject, take the median order-2 pre-log scale over the other
15 subjects' sessions (the fold's training statistic); look at the spread of those 16 fold
values. Separately, the spread across the 16 *individual* subjects, to tell "median is
robust" apart from "subjects genuinely alike".

```
                 across the 16 FOLDS            across the 16 SUBJECTS
T1 order2   max/min 1.01  CV 0.45%         max/min 1.73  CV 13.8%
T2 order2   max/min 1.02  CV 0.80%         max/min 1.78  CV 14.7%
T3 order2   max/min 1.02  CV 0.69%         max/min 1.83  CV 14.9%
```

**Reading.** (1) A per-fold ε would be near-identical across folds (<1%), so a fold-local
vs global choice differs by <1% — the leakage cost of adapting ε here is negligible. (2)
But individuals differ by up to ~1.8×, so part of the fold-stability is the median being
robust to dropping ~4 of 68 sessions, *not* subject homogeneity — and extrapolation to a
genuinely different setup (not this project's continuation, which reuses the same radar/
distance) stays an assumption. (3) **Tuning ε *to* the order-2 scale returns ~1e-6 — i.e.
today's value** (ε is already ~64% of the T1 order-2 median); *un-flooring* order-2 needs ε
much smaller than the scale, which trades flattening for near-zero noise amplification —
a two-sided tradeoff whose sweet spot is genuinely uncertain. Stability tells us adapting ε
is safe/cheap; it does **not** tell us it improves prediction. Recorded, ε unchanged —
this motivates the M5-pre-registered third log branch (implementation_plan.md §WST
parameterization / §LOSO harness search space), decided at M6, never now.

## 2026-07-23 — M4 step 7: `features/extraction.py` + `run_wst.py` cohort run.
## **All 73 sessions finite in ~12 min; but ε=1e-6 is NOT negligible vs the tiny order-2
## coefficient scale — the plan's "O(1) coefficients" assumption is measured false.**

`src/dehyd/features/extraction.py` (the reusable manifest→features wiring, in `src/` so
the M6 harness never imports a CLI script) + `experiments/run_wst.py` (thin CLI over it) +
extraction tests (T-W14 guard, T-W18 variants≡single-variant + call-count, pre-log scale).

**Cohort result — `results/wst/wst_diagnostics_10ghz.csv` (73 rows, matches M2/M3):**
7168 eligible frames, 73 sessions, 16 subjects. **`all_variants_finite = True` for every
session** across all (reduction × channel × tiling × log × family) branches — the
finiteness battery holds on real data at cohort scale. Total WST wall-clock **722.7 s
(~12 min)**, matching the ~14-min projection.

**Feature dimensions (constant across sessions), nominal / effective / raw per the
≥2-sample segment-std rule:** T1 mag 4452/4452/5194, T1 iq 8904/8904/10388; T2 mag
2796/**2330**/1398 (effective < nominal — the 1-sample first half drops one std), T2 iq
5592/4660/2796; T3 mag 2094/**1745**/1047, T3 iq 4188/3490/2094. Effective = nominal only
for T1 (n_time = 7); T2/T3 (n_time = 3) lose one std/path exactly as A-M4-6 intends.

**THE FINDING — pre-log coefficient scale (cohort medians):**
```
tiling   order0        order1     order2      eps/|order2|
T1     -4.39e-02     7.98e-04    1.56e-06    0.64
T2     -3.99e-02     1.14e-03    4.79e-06    0.21
T3     -3.46e-02     1.28e-03    8.23e-06    0.12
```
- **order 0 is signed and negative** (median ≈ −0.04) — confirms empirically that order 0
  is a signed low-pass and MUST stay linear; logging it would be `log(negative)`.
- **The plan's rationale for ε — "the coefficients live on an O(1) standardized scale" —
  is FALSE.** The standardized *input* is O(1), but scattering coefficients are far
  smaller: order 1 ≈ 1e-3, order 2 ≈ 1e-6. ε = 1e-6 is ~3 decades below order 1
  (negligible there), but **12–64 % of the median order-2 scale** — so when log is on,
  `log(S + ε)` on order 2 is materially ε-floored: for the (many) below-median order-2
  paths, ε dominates and compresses them toward log(ε) ≈ −13.8.
- **Action per the M2/M3 doctrine: none to the parameter.** ε stays frozen at 1e-6 —
  changing it on seeing this would be the forbidden data-driven retune. The pipeline
  already carries the mechanism that resolves whether this matters: **log on/off is an
  inner-CV axis at M6**, so the CV empirically decides per fold whether ε-floored order-2
  logging helps. The finding *strengthens* the case for keeping log selectable rather than
  always-on, and is recorded in SECOND_CHAPTER §3 as such. Flagged to the owner, not acted
  on. (The misleading console line "eps << order1/2" was corrected to print the eps/scale
  ratio; the CSV always carried the true numbers.)

**Performance failure and fix (kept in the log).** The first cohort attempt ran at
**~96 s/session → ~2 h projected**, not ~14 min. Profiling one session isolated the cause:
`pool_stats` was **7.82 s** for T1 iq (100 frames) vs 2.31 s for the scattering itself —
its per-path Python loop runs C·n_paths·segments = 4452 iterations/frame with tiny
`.mean()`/`.std()` slices. Vectorizing `pool_stats` over channels and paths (assemble each
segment×stat as a `[C, n_paths]` column, stack, transpose to channel→path→stat, flatten)
cut it to **0.053 s (150×)** with bit-identical values and order (the hand-computation
tests pin both). Full session 96 s → **11.7 s**; cohort ~12 min. build_scattering is *not*
the bottleneck (≈0.01–0.04 s), so no bank caching was added.

**Cross-backend on real frames** (in the realdata test): numpy-f64 vs torch-f32 max
elementwise ratio 0.36–0.47 (< 1), rel L2 ≈ 4e-7 — passes the strict "float64" policy on
actual data, where more near-zero coefficients push the ratio up but well inside bounds.

## 2026-07-23 — M4 steps 3–4: `features/wst.py`. **Measured geometry pins the reviewer
## values exactly; order-aware log, batched transform, and cross-backend gate all green.**

`src/dehyd/features/{__init__,wst.py}` + `tests/test_wst.py` (41 tests). The kymatio
parameterization, all shape/fs-agnostic (M9 reuse), constants from `WSTConfig` only.

**Measured filter-bank geometry (numpy frontend, n_in = 470, fs = 520834), pinned as
T-W2 regression values — matches the reviewer-sampled values exactly:**

```
tiling   Q       ms    T    J   n_paths  n_time   pad_left  pad_right  padded_len
T1    (10, 4)  0.20  104   7    742      7        277       277        1024
T2    ( 8, 2)  0.30  156   8    466      3        277       277        1024
T3    ( 6, 2)  0.40  208   8    349      3        277       277        1024
```

- **Padding is MEASURED, never assumed** — `pad_left`/`pad_right` read back from the
  object; `padded_len = n_in + pad_left + pad_right = 1024` (which *is* 2^10 here, but
  that is observed, not hard-coded — T3 at J=8 still pads to 1024, not 2^18). The
  deprecated `.N` attribute is avoided; padded length comes from the pad math.
- **n_time = 3 for T2/T3** — this is the measured fact that makes A-M4-6's ≥2-sample
  segment-std rule necessary (a 1-sample first half → identically-zero std). n_time = 7
  for T1 keeps all 6 stats.
- **Path order counts:** order 0 → 1 path, order 1 → 55, order 2 → 686 (T1); order-1 `xi`
  is strictly decreasing (kymatio convention), asserted in T-W2.

**Order-aware log (`apply_order_log`).** Orders 1–2 → `log(S + 1e-6)`; **order 0 stays
linear** — a crafted negative order-0 coefficient (S0 is a signed low-pass) passes through
untouched and finite (T-W6), where logging it would give NaN. A path-count mismatch vs
`meta` raises. The finiteness battery (mag/iq × orders 0/1/2 × log on/off, all 3 tilings)
is green (T-W5).

**Batched transform (`scatter_frames`).** [N, C, 470] folded into kymatio's leading batch
dim in one call; **bit-identical to stacked single-frame calls** for all three tilings
(T-W16). `scatter_channels` is defined as `scatter_frames(x[None])[0]`, so the two paths
cannot diverge.

**Cross-backend gate (`backend_agreement` + `AgreementResult`).** Two frozen policies in a
table, no caller tolerances; raises on shape/empty/non-finite/dtype violations; returns the
measured `max_elementwise_ratio` and `rel_l2`. **numpy-f64 vs torch-f32 passes the strict
"float64" policy** on raw AND logged tensors, both channels, all three tilings (T-W9) —
consistent with the step-1 finding. **Suite: test_wst 41 passed.**

## 2026-07-23 — M4 step 1: `uv add torch`. **kymatio's torch frontend is float32-only —
## the planned float64-vs-float64 cross-backend comparison is impossible; strict
## tolerances kept anyway (owner-approved), no fallback needed.**

`uv add torch` → **torch 2.13.0+cpu**, pulling filelock, fsspec, jinja2, markupsafe,
mpmath, networkx, setuptools, sympy, typing-extensions. **The scipy pin held:** the
resolver kept **scipy 1.16.3** (< 1.17), so `from kymatio.numpy import Scattering1D`
still imports — the M1 kymatio-breaks-on-scipy-≥1.17 trap did not fire. numpy stayed
2.4.6. Added `packaging>=21.0` to the dev group (test_env now imports it) and a torch
import + `test_scipy_pin_survives_torch` (parsed-version comparison, not string —
`Version(scipy.__version__) < Version("1.17")`). **Env suite: 3 passed.**

**Finding that triggered the pre-declared dtype fork (MILESTONE_4_PLAN §2.2).**
kymatio's **torch** `Scattering1D` runs its filter bank in **float32**: a float64 input
raises `TypeError: Input and filter must be of the same dtype`. So the plan's assumed
"convert to a float64 torch tensor, compare float64-vs-float64" is **not achievable in
the pinned stack** — an environment fact, exactly the contingency §2.2 pre-declared.

**Measured the only achievable comparison — numpy-float64 vs torch-float32** — on a
`default_rng(0)` standard-normal 470-sample signal through all three tilings:

```
tiling      Q       T    J   out shape   rel L2     max elemwise ratio (float64 policy)
T1      (10, 4)   104   7   (742, 7)   6.57e-08   0.0437   PASS
T2      ( 8, 2)   156   8   (466, 3)   8.59e-08   0.0090   PASS
T3      ( 6, 2)   208   8   (349, 3)   6.72e-08   0.0039   PASS
```

Two things fall out of this and are recorded now:
1. **The measured output shapes exactly match the reviewer-sampled values** T1 (742, 7),
   T2 (466, 3), T3 (349, 3) — the T-W2 regression pins (confirmed at build, not assumed),
   and the n_time = 3 for T2/T3 that motivates the ≥2-sample segment-std rule (A-M4-6).
2. **numpy-f64 vs torch-f32 passes the STRICT "float64" policy** (rtol 1e-4, atol 1e-8):
   max elementwise ratio ≤ 0.044 (< 1), rel L2 ≈ 1e-7. float32 accumulation across the
   scattering depth stays comfortably inside the 1e-4 relative bound.

**Owner decision (2026-07-23): keep the strict tolerances; do NOT adopt the float32
fallback.** The fork the plan pre-declared required owner approval to loosen; the owner
chose *not* to loosen, because the data clears the tight bar. The `backend_agreement`
"float32-fallback" policy (rtol 1e-3, atol 1e-5) stays defined but unused — invoking it
would need a fresh owner decision. Propagated to MILESTONE_4_PLAN §2.2 (dtype policy),
the `scatter_frames`/§5 wording, and pyproject's dependency comment; the cross-backend
formula in implementation_plan.md is unchanged (still the strict "float64" policy).

## 2026-07-23 — **MILESTONE 3 COMPLETE.** Definition of done met in full.

**D1 — mandatory suite, no private data.** `uv run pytest` → **319 passed, 11 skipped**
(was 260/10 at M2 close; +59 tests, 45 of them in `test_preprocess.py`).
**D2 — real-cohort acceptance.** `uv run pytest --realdata` → **329 passed, 1 skipped**
(T18 only, as designed until M6).
**D3 —** `experiments/run_preprocess.py` wrote and re-verified
`results/preprocess/preprocess_diagnostics_10ghz.csv` (73 rows); distributions recorded
in the step-6 entry below.
**D4 —** `tests/test_no_leakage.py` is **byte-for-byte unmodified since M1**
(`git diff f3fbade HEAD -- tests/test_no_leakage.py` empty) and green — 24 passed,
2 skipped.
**D5 —** HISTORY.md carries an entry per resolved step, including all three required
departure logs: no-window-in-the-primary-path, the median/MAD form with its eps
placement, and the Option-B mask correction.
**D6 —** SECOND_CHAPTER.md §2 "Preprocessing" written: the paper-vs-code ambiguity
resolved as a methodological argument, the provenance of every parameter, the
finite-record measurements, the three cohort findings, and the correctness argument.
**D7 —** amendments **A-M3-1..A-M3-7** are live in `plans/implementation_plan.md`
(§Preprocessing steps 3–4, §Deliberate departures, the repo tree, Build order §3 and
§5); the two plan documents agree.

**The invariant held.** Preprocessing is a per-frame function of one frame plus frozen
constants: T-PP20 asserts a frame's output is identical whether processed alone or
beside an arbitrary companion — including one scaled ×1000 — and T-PP22 asserts
bit-identical repeats. Nothing here is fitted, so nothing enters the CV loop.

**Milestone-3 scoreboard.** 5 new source modules
(`preprocess/{__init__,filters,reduce,standardize,pipeline}.py`), 1 new experiment
entry point, 1 new test module, **59 new tests** (319 vs 260). Five config fields added,
each classified as search axis / pre-declared ablation / frozen protocol constant before
any result existed. Three deliberate departures from the reference logged with reasons.
**Four facts discovered empirically rather than assumed** — the arithmetic-vs-geometric
mid-band ambiguity, the finite-record energy reality, the windowed-vs-unwindowed
zero-ROI counterexample, and the 1.50 m target range — none of which changed a frozen
parameter.

**Open for M4:**
- Feature extraction consumes `preprocess_cube(...)` → float64 [n_frames × C × 470].
  The kymatio border-effect warning at `Scattering1D(J=7, shape=(470,))` is an M4
  concern: measure padding and output shape from the instantiated filter bank, never
  assume them.
- torch enters the environment at M4 (cross-backend WST check); the T18 torch mutation
  leg stays skip-marked until M6 — owner decision, unchanged.
- `configs/ibex.yaml`, `scripts/ibex/` still deferred to the first IBEX milestone.
- The 77 GHz any-trace flatline rule remains parked for an owner decision at M5.
- Nothing committed yet — awaiting the owner's word, per the ground rules.

---

## 2026-07-23 — M3 step 6: `run_preprocess.py` over the full cohort. **The beat sits
## at 1.50 m and the ROI peak is genuinely dominant (peak_share 0.51 vs 0.33 uniform).**

Thin CLI, all logic in pure helpers (M2 audit pattern) so the diagnostics are testable
without a cohort run. **Primary-only guard:** the script compares the consumed
`config.preprocess` against the whole canonical `PreprocessConfig()` and refuses to run
otherwise, naming the deviating fields — checking just the two ablation switches would
have let `model_gate_m: [0.9, 3.0]` (an inner-CV *candidate*) overwrite the primary CSV
under a "primary" label.

### Cohort result (73 sessions, 7168 eligible frames, 16 subjects — matches M2 exactly)

```
peak bin (mode)   5           bin 4: 1 session, bin 5: 41, bin 6: 31
peak Hz median    4876.7  ->  1.50 m
energy retention  median 0.407   [0.061, 0.644]
roi/total         median 0.930   [0.775, 0.977]
peak_share        median 0.512   [0.410, 0.739]
missing cells     0            all four variants finite in every session
```

**Findings, recorded not acted on:**

1. **The target range is ~1.50–1.80 m, not the ~1 m the config comment assumed.**
   72 of 73 sessions put the dominant beat in bin 5 or 6 (4877 / 5852 Hz = 1.50 /
   1.80 m). The frozen 1–2 m model gate contains this comfortably — the gate is better
   justified by the data than by the assumption that motivated it — but the *stated
   reason* ("subject seated ~1 m") is now known to be off by half a metre. Left as is:
   changing the gate on seeing this would be exactly the data-driven retune the
   milestone forbids. The correct reading goes in SECOND_CHAPTER §2.
2. **The peak is genuinely dominant.** This is the measure Codex's round-1 comment 4
   asked for, and it answers it: with 3 ROI bins, a flat spectrum would give
   `peak_share = 0.333`; the observed median is **0.512** (min 0.410), so one bin
   really does carry the return. Option-B's premise holds on this data.
3. **Caveat on `roi_to_total` (0.930) — it is measured POST-filter, as the plan
   specifies, so it is largely a filter-selectivity descriptor and cannot be low by
   construction.** It is not evidence of target presence; `peak_share`, which compares
   *within* the ROI, is. Recorded so the chapter does not over-read it.
4. **Per-session peak stability is high:** 45 of 73 sessions have every frame on one
   bin, 22 span 1 bin, 6 span 2. The detection is not jittering frame to frame.
5. **Energy retention varies 10× across sessions** (0.061 to 0.644, median 0.407) —
   far below the 0.76 a pure mid-band tone retains (T-PP6), as expected since real
   frames carry energy across and outside the band. The three weakest (s11 10am 0.061,
   s12 2pm 0.070, s5 4pm 0.095) still pass QC and still show high roi/total, so this is
   overall signal level, not a band mismatch. Noted for M4.

**One test fixture was impossible and revealed a real property.** T-PP23's aggregation
case originally fed all-zero frames to `session_diagnostics` to force a missing
`peak_share` — but `energy_retention` raises on a zero-energy frame *by contract*
(QC's in-band ratio ≥ 0.30 is impossible at zero power). Chasing that showed the
"all-missing session → empty CSV cell" path is **unreachable for eligible frames**:
`peak_share` is undefined only at exactly-zero ROI power, and any frame with energy
leaves float-positive power there. The rule was factored into a
`median_skipping_missing` helper and tested directly; the guard stays so "undefined"
remains distinguishable from "zero", but it is documented as not expected to fire —
and the cohort run confirms it, with **0 missing cells in 73 sessions**.

Artifact written and re-verified: `results/preprocess/preprocess_diagnostics_10ghz.csv`
(73 rows). Provenance carries `analysis_role: "primary"`, the full `filter_spec`
(padlen 27, Wn, band edges) and the ROI bins. **Suite: 319 passed / 11 skipped;
`--realdata` 329 passed / 1 skipped (T18).** Staged-file list checked with `git add -An`
per the M1 lesson: all five `src/dehyd/preprocess/*.py` appear — the `.gitignore`
package trap did not recur.

---

## 2026-07-23 — M3 steps 3–5: reduction, standardization, pipeline. **First real-data
## contact: the dominant beat sits at ~1.50 m, not the assumed ~1 m.**

**Step 3 — `reduce.py`.** Option A (chirp mean), `detect_option_b_peak` →
`OptionBDetection(peak_bin, power, roi_bins)`, `option_b_mask`, `reduce_option_b`,
`edge_trim`. ROI = model gate, **no margin**, half-spectrum bins 0..266 → **bins 4,5,6**
at the default config (df = 975.34 Hz), verified against independent arithmetic; the
0.9–3.0 m candidate gives bins 4..10 (bin 3 = 2926.0 Hz misses the 2931.7 Hz edge by
5.7 Hz). `edge_trim` **raises rather than clamps** — the reference's silent
`min(EdgeTrim, N/4)` would hide a mis-set config.

**Departure logged — the Option-B mask is a correction, not a port.** `wst_extract.m`
keeps only `peakBin + (0:nb)` (**one-sided**, contradicting its own "±bins" docstring)
and then applies MATLAB's endpoint-zero `hann(numel(idx))` across the concatenated
positive+mirror block: at nb = 1 that is `[0, .75, .75, 0]`, which **zeroes the detected
peak itself**, leaving only bin peak+1 at 75%. We implement the form the docstring and
the main plan describe — symmetric ±nb, **full weight on the peak**, Hann shoulders:
weights = interior of `hann(2nb+3)` → nb=0 [1.0], nb=1 **[0.5, 1.0, 0.5]**, nb=2
[0.25, 0.75, 1.0, 0.75, 0.25]. Mirrors take the same weight; a self-mirroring bin
(DC/Nyquist) takes the **max**, never the sum. A mask that would keep every bin raises
(a pass-through Option B is a disabled reduction, not a configuration).

**The zero-ROI claim was wrong in an earlier draft and is now pinned by an adversarial
fixture** (Codex review round 4). Detection is Hann-windowed; the mask is applied
*unwindowed*. The frequency-domain periodic-Hann kernel [−¼, ½, −¼] can annihilate the
windowed ROI while the unwindowed bins under the mask stay nonzero. Constructed and
**confirmed numerically**: unwindowed bins 3..7 = [1, 0, −1, −2, −3] → windowed ROI
power ≈ 1e-34 (exactly zero to float precision) yet the reduced output carries
9.4e-4 of energy. So the frozen behaviour is "mask the first ROI bin, whatever that
yields" — finite, deterministic, and flagged downstream by `peak_share = NaN`. An
exactly-zero frame is tested separately, where the argmax tie-break genuinely does
return the first ROI bin.

**Step 4 — `standardize.py`.** `robust_standardize` = `(x − median)/(1.4826·MAD + eps)`
with **eps = float64 machine epsilon placed OUTSIDE the scale factor** (the reference
uses `1.4826·(MAD + eps)`; numerically irrelevant — it is a division guard, not a
tuning constant — but one form must be frozen for bit-reproducibility). This is the
settled departure from the reference's mean-centre/MAD-scale mix. `meanstd_standardize`
uses **ddof = 0** (numpy's population convention; MATLAB's `std` defaults to ddof = 1
and no reference constrains the choice), pinned by an exact hand-computation test that
a ddof = 1 implementation fails. Channels are standardized **each from its own
statistics**.

**Step 5 — `pipeline.py`.** `preprocess_frame` / `preprocess_cube`, the sequence
readable in one screen: gate → reduce → trim → channel → standardize → float64
[C × 470]. `reduction`/`channel` are **call arguments, not config** (they are inner-CV
axes at M6, so one config must produce every variant). T-PP15 makes the
trim-after-reduction ordering structural rather than a comment: trimming first would
change the FFT bin grid (470-pt → df = 1108 Hz, ROI bins 3..5), so the two orders
genuinely disagree and the test can tell them apart.

### First contact with real data (`subject_1_8am.mat`, 35 QC-passing frames)

```
option-B peak bins  {4: 1, 5: 34}      df = 975.3 Hz
peak Hz median      4876.7   ->  4876.7 / 3257.5 Hz/m = 1.50 m
peak_share median   0.460
```

**The dominant beat sits at ≈1.50 m, not the ≈1 m the plan assumed** when it called the
1–2 m gate "physically motivated (subject seated ~1 m)". The value lands comfortably
inside the 1–2 m model gate — near its centre, in fact — so the frozen gate is
*better* justified by the data than by the assumption behind it. Nothing was changed:
this is recorded as a finding. (Note s1 8am is one of the 7 QC-**ineligible** sessions
from M2 at 35/100; the frames used here are still genuine QC-passing frames, and the
test is about pipeline mechanics, not about that session's eligibility.)

**Suite: 309 passed, 11 skipped** (36 in `test_preprocess.py`, one of them realdata).
`tests/test_no_leakage.py` is **byte-for-byte unmodified** (`git diff` empty) and green
— 24 passed, 2 skipped.

---

## 2026-07-23 — M3 step 2: `preprocess/filters.py` — the band gate. Three test
## fixtures were wrong on first contact; all three taught something.

`design_bandpass_sos` / `bandpass_filtfilt` / `fft_gate` / `apply_band_gate` /
`filter_spec`, written **shape- and fs-agnostic** so the 77 GHz chain (M9, fs = 500 kHz,
N = 256) reuses them unchanged. Band from the **model** gate (1–2 m →
3257.5–6514.9 Hz, Wn ≈ 0.0125–0.0250), never the QC gate. `padlen = 27` is passed
**explicitly** (not left to scipy's default) and T-PP1 pins both the value and its
bit-identity with the library default. **9 tests pass.**

**Departure logged — no window before the time-domain filter (ROADMAP §3.2).** The
ROADMAP lists "Hamming window; range FFT; SOS Butterworth". A window suppresses FFT
spectral leakage, which is meaningless for a time-domain IIR filter, and pre-tapering
the chirp would attenuate real signal energy at its edges. Windows are applied only
where an FFT is actually taken (QC in-band screen, Option-B detection, the FFT-gate
ablation). filtfilt's edge transients are handled by EdgeTrim instead.

**Three fixture failures, each a real fact about the filter — the module was not
changed to accommodate any of them (§5's "no parameter chasing" rule):**

1. **"Mid-band" was ambiguous.** I probed at the *geometric* mean (4606.7 Hz) while the
   plan's regression values were measured at the *arithmetic* centre (4886.2 Hz). Both
   are inside the passband; only one matches the recorded numbers. Fixed by defining
   mid-band **once**, as the arithmetic centre, in a fixture every probe now uses.
2. **The zero-phase fixture was too narrow.** A σ = 40-sample Gaussian burst has ~2 kHz
   of bandwidth against a 3.3 kHz passband, so the filter distorts it enough to move
   the envelope peak by one sample. σ = 80 fits inside the band and lands the peak
   exactly at centre. (σ = 120 is *worse* — the envelope becomes so flat that its
   argmax is noise-dominated, 273 vs 267.) The test now also asserts a **single causal
   pass shifts the same burst by ~131 samples**, so it demonstrably has the power to
   catch a non-zero-phase implementation instead of passing vacuously.
3. **The stopband assertion assumed steady state.** A 2.6 m tone under the 1–2 m gate
   retains **3.5%** of its energy, not < 1% — the same finite-record leakage the plan
   already pins in T-PP6 (a 534-sample record cannot reach the design stopband). The
   assertion now states the honest claim: kept ≈ 0.81, rejected < 0.1, and kept is
   > 10× rejected.

**Also measured (recorded, not asserted):** filtfilt with `padtype='odd'` is **not**
exactly time-reversal-symmetric on a 534-sample noise record (max |diff| ≈ 0.49 in the
interior). I had considered using that identity as the zero-phase test; it is not a
property scipy guarantees under edge padding, so the cross-correlation-lag test is used
instead. Noted so nobody re-derives it.

---

## 2026-07-23 — M3 step 1: preprocessing config fields + cross-field band validation.

`plans/MILESTONE_3_PLAN.md` was approved after **4 review rounds (15 comments, all
applied, none disputed)**; amendments A-M3-6 and A-M3-7 were propagated into
`implementation_plan.md` during review. Implementation starts here.

**Five new `PreprocessConfig` fields**, each classified before M6 so no alternative can
become an undeclared search axis:
`gate_method="butterworth"` and `standardize="robust"` are **ablation switches**
(non-default = pre-declared ablation only); `peak_neighbors=1`, `mask_taper=True`,
`fft_gate_transition_hz=500.0` are **frozen protocol constants** (non-default is
test-only, rejected by modelling entrypoints). `peak_neighbors=1 + mask_taper=True`
*is* the plan's "±1-bin two-sided Hann-tapered mask" — they are constants, not knobs.

**Validation added.** Field level: `gate_method`/`standardize` from fixed choice
tuples; `mask_taper` strictly bool (0/1 rejected — YAML has real booleans, so an int
is a typo); `peak_neighbors` integer ≥ 0 (0 = keep the peak bin alone, legitimate in
tests). This forced `_int_field` to take a `minimum` (it hard-coded `> 0`); its
message became "must be >= N", and three existing M1/M2 parametrised cases had their
expected regex updated — same assertions, new wording.

**New cross-field check `_check_model_band`,** three failures, all hard errors:
1. **whole band strictly below Nyquist** (`0 < f_lo < f_hi < fs/2`). Deliberately
   stricter than `_check_qc_band`, which only rejects a band *starting* above Nyquist:
   the QC screen is an FFT mask whose upper edge is legitimately Nyquist-clamped
   (frozen at M2), but `scipy.signal.butter` raises on `Wn ≥ 1`, so a *straddling*
   gate would pass config load and fail deep inside the filter — exactly what the
   fail-at-config-load rule forbids. Clamping instead would make the two gate methods
   filter different bands under one config.
2. **`model_gate_m ⊆ qc_gate_m`** (inclusive, so the 0.9–3.0 m inner-CV candidate —
   which equals the QC gate — still loads). QC fixed the frame population on the wider
   gate; a model gate reaching outside it would use energy QC never screened for.
3. **FFT-gate non-vacuity** (skirts covering the whole spectrum = a filter that
   filters nothing), checked only on the `fft` path.

`configs/preprocess.yaml` mirrors all five with their classification stated in the
comments. **Result: `tests/test_config.py` 67 passed, 1 skipped** — including the
straddling-Nyquist case, which needed the QC gate widened to 70–90 m so the QC check
passes first and the *model* rule is what actually fires.

**Commit `395eb62`** on `v1_milestone_2` — 20 files, pushed to
`origin/v1_milestone_2` (new upstream). Staged list checked file-by-file before
committing, per the M1 lesson: both `src/dehyd/qc/__init__.py` and
`src/dehyd/qc/screens.py` are present (the `.gitignore`-swallows-a-new-package trap did
**not** recur), both curated artifacts (`results/qc/qc_survival_10ghz.csv`,
`results/qc/audit_77ghz.json`) are in, and `results/runs/` is correctly excluded —
verified with `git add -An` rather than assumed.

**Branch `v1_milestone_3`** created from `v1_milestone_2` and checked out. HANDOFF.md
rewritten for the milestone-3 bootstrap (owner-requested — it is never updated
automatically). Nothing is merged to `main` yet; `v1_milestone_1`, `v1_milestone_2`
are pushed.

**M3 starts from:** `eligible_frames(manifest_qc)` = 7168 frames across 73 sessions and
16 evaluable subjects, and implementation_plan.md §"Preprocessing — executable
sequence" (order-4 SOS Butterworth zero-phase on the complex fast-time axis using the
**model** gate, Options A/B reduction, EdgeTrim 32 **after** reduction → 470 samples,
median/MAD robust standardisation).

---

## 2026-07-21 — **MILESTONE 2 COMPLETE.** Definition of done met in full.

**D1 — mandatory suite, no private data.** `uv run pytest` → **260 passed, 10 skipped**
(the 10 skips are 9 `realdata` tests plus T18). Was 151/8 at M1 close.
**D2 — real-cohort acceptance.** `uv run pytest --realdata` → **269 passed, 1 skipped**
(T18 only).
**D3 —** `experiments/run_qc.py` writes and re-verifies
`results/qc/qc_survival_10ghz.csv`; survival recorded in the step 3–4 entry below.
**D4 —** `run_regression.py` builds its folds from the post-QC evaluable subjects;
provenance carries the QC config including the new margin.
**D5 —** the 77 GHz audit ran on one real file; all five verdicts recorded in a
provenance-complete `results/qc/audit_77ghz.json`.
**D6 —** HISTORY.md has a per-step entry (steps 1–7); SECOND_CHAPTER.md §1 "Data &
ground truth" now carries the QC section.
**D7 —** amendments **A1–A7** are live in `plans/implementation_plan.md`; the two plan
documents agree.

**The invariant held.** `tests/test_no_leakage.py` is **byte-for-byte unmodified since
M1** (`git diff HEAD` empty) and green — 24 passed, 1 skipped. QC never enters CV: it
is a per-frame function of one frame plus frozen constants, applied once before any
split exists, and T-QC7 asserts a frame's verdict is identical whether screened alone
or beside arbitrary companions.

**Milestone-2 scoreboard.** 2 new source modules (`qc/screens.py`, QC section of
`manifest.py`), 2 new experiment entry points, 2 new test modules, **101 new tests**
(260 vs 159). Three genuine facts discovered empirically rather than assumed — numpy's
histogram behaviour on a degenerate range, YAML 1.1's signed-exponent rule, and the
77 GHz real-valued storage — each of which would have surfaced later as a confusing
failure. Two frozen thresholds met the real data for the first time; **neither was
touched**, and both surprising results (670 in-band rejections at 10 GHz, 7/10 flatline
rejections at 77 GHz) are recorded as findings for the milestone-5 freeze.

**Open for M3 / M5:**
- **Owner decision at M5:** the 77 GHz any-trace flatline rule rejects 7 of 10 audited
  frames because of ADC quantisation, not a dead receiver. Frozen as-is for now.
- Preprocessing (M3) consumes `eligible_frames`; the analysis population is 7168 frames
  across 73 sessions and 16 evaluable subjects.
- `configs/ibex.yaml`, `scripts/ibex/` still deferred to the first IBEX milestone.
- Nothing committed yet — awaiting the owner's word, per the ground rules.

---

## 2026-07-21 — M2 steps 5–6: evaluability hookup + **the 77 GHz audit.**
## **Axis hypothesis CONFIRMED. Two findings that change milestone-5 planning.**

**Step 5** (`run_regression.py`): folds now come from `evaluable_subjects` after QC.
Smoke on real data → 7330/8000 frames, 73/80 sessions, **16 outer folds, 5 inner
each** — the clean full-cohort case, unchanged from M1 because no subject lost all
five sessions.

**Step 6**: `uv add h5py` (3.16.0), `experiments/audit_77ghz.py` (pure parameterised
helpers + thin `main`), `tests/test_audit_77ghz.py` (23 synthetic tests, no private
data). Audit ran on `data/77ghz/subject_1_8am.mat` in **8.3 s**.

### Verdicts — all green

```
H1_shape   ACCEPTED    (16, 256, 256, 125) exactly as predicted; chunks (16,4,1,125), gzip
H1_storage ACCEPTED    but NOT the representation the plan froze — see finding 1
H1_axes    ACCEPTED    G_fast=0.2260  G_chirp=6.70e-06  D_chirp=0.9999  D_fast=0.4939
qc_smoke   NON_DEGEN.  3/10 frames pass; median in-band ratio 0.382 — see finding 2
chain      NON_DEGEN.  final/raw energy 2.98e-05 (gated) and 8.50e-06 (range-Doppler)
```

**The axis question is settled, and not marginally.** A1 needed D_chirp ≥ 0.5 → got
0.9999. A2 needed G_fast ≥ 0.05 → got 0.226. A3 needed G_fast ≥ 10·G_chirp → the
actual ratio is **≈34 000×**. The mirrored swapped-axis hypothesis fails on its very
first criterion (D_fast = 0.494 < 0.5), so the result is not a coin-flip resolved by
threshold placement. The check ran on the **raw** slab, before MTI, exactly as
required — MTI would have removed the near-zero-Doppler static-subject energy that
makes D_chirp ≈ 1 the discriminator it is.

### Finding 1 — the 77 GHz raw is REAL float64, not complex

The plan froze the accepted storage as "compound `real`/`imag`". The real files are
**plain `float64`**, `dtype.names is None`: ADC-like values quantised to 1/16,
|x| ≤ 2560. Cross-checked against the reference — `wst_extract77.m` and
`filter_gpt_butterworth77.m` never call `real()`/`imag()` on raw 77 GHz data, which is
consistent with a real-sampled capture. (The 10 GHz files genuinely are complex, which
is where the assumption came from.)

Handled as a **correction of a wrong a-priori assumption about the file format, not a
threshold chosen from data**: it is visible in HDF5 metadata alone, it changes no
frame's screen verdict, and the observed dtype descriptor is recorded either way.
`H1-storage` now accepts real-float *or* compound-complex and rejects everything else;
both plan documents were amended. **Consequence for Exp G, recorded in
implementation_plan.md:** the primary chain's "I/Q" comes from the **complex range-FFT
output** (chain step 4), not the raw cube. The chain itself is unaffected — it already
scatters the post-range-FFT slow-time series — but nothing before step 4 may assume
complex input.

### Finding 2 — the frozen 77 GHz flatline rule rejects 7 of 10 frames

Per-Rx flagged-trace counts over the 10-frame slab: **169–1601 of 2560 traces per Rx**,
spread fairly evenly across all 16 receivers — so this is **not** a dead channel. Cause
is the rule meeting heavily quantised ADC data: with 128 bins across a 256-sample
trace's own range, quantisation piles ≥64 samples into one bin easily. The ≈205×
multiplicity the reviewer flagged **does bite**, empirically.

Per the milestone invariant, **nothing was changed**: the rule was frozen a priori
(owner decision, 2026-07-21), the audit's job was to make the multiplicity visible
before the M5 freeze, and it has. The in-band screen by contrast looks healthy
(ratios 0.373–0.392, all above the 0.30 threshold, so 0 low-in-band rejections). **This
is an owner decision for milestone 5**, not something to be quietly retuned here.

### Other recorded facts

- **Chunk layout `(16, 4, 1, 125)` spans the entire frame axis**, so a 10-frame read
  still decompresses ~1.05 GB. Memory stays bounded (84 MB retained) and it costs
  ~8 s, which is why the bounded-slab contract is about *memory*, not I/O volume.
- **MTI removes 99.7 % of the energy** (ratio 2.70e-03) — exactly the "legitimate
  physical attenuation" the plan predicted when it rejected a bare `> 0` test. The
  1e-9 floor sits ~4 orders below the smallest observed stage ratio, so it discriminates
  true degeneracy without firing on real clutter suppression.
- Derived constants confirmed against independent computation: dr = 0.0749 m, range
  gate **bins 27..53**, QC mask **bins 26..54**, PRF 1953.125 Hz.

### Test-fixture bugs the tests caught in themselves

Two of my own fixtures wrote a `(frame, fast, chirp, rx)` cube straight to disk
instead of the on-disk `(rx, chirp, fast, frame)` layout, so the round-trip test was
asserting against the wrong shape. Fixed by making `write_fixture` apply the same full
reversal the audit uses on read (it is its own inverse) — which makes the round-trip
test genuinely round-trip. The end-to-end fixtures were also shrunk from the real
`(16,256,256,·)` to `(2,32,32,·)`; 32 fast-time bins still yield a non-empty range
gate (27..31) and QC mask (4..6), and the audit-test module dropped from ~6 s to 1.7 s.

**Tests:** `uv run pytest` → **260 passed, 10 skipped**.

**Next:** step 7 — journal close-out.

---

## 2026-07-21 — M2 steps 3–4: manifest QC bookkeeping + **first real-cohort QC pass.**
## **Success. 7330/8000 frames survive; 7 of 80 sessions dropped; N_eval = 16.**

**What was built.** `apply_qc` / `session_qc_report` / `eligible_frames` /
`evaluable_subjects` in `manifest.py` (§2.2), `experiments/run_qc.py` (§2.4), 20 new
manifest tests and 2 realdata tests.

### The real numbers (frozen screens, first contact with the cohort)

```
frames    : 7330 pass / 670 fail of 8000  (91.6% survive)
reasons   : nan/inf 0, flatline 0, low in-band 670   (rms flagged 2752, diagnostic only)
sessions  : 73 eligible / 7 dropped of 80
dropped   : s1 8am 35/100, s1 4pm 1/100, s3 10am 37/100, s4 2pm 35/100,
            s5 2pm 39/100, s6 8am 0/100, s16 10am 15/100   (all needed 50)
N_eval    : 16 evaluable subjects — every subject keeps >= 1 eligible session
analysis  : 7168 frames (the 162 passing frames inside dropped sessions are excluded)
```

**Independent corroboration.** ROADMAP §2 states "~7500 after QC" for 10 GHz, written
from the original study and never used to tune anything here. We get **7330** from
thresholds frozen before looking. The agreement is evidence the ported screens behave
like the originals; it is *not* a target that was fitted to.

**What the failures actually are.** Every rejection is the in-band energy screen —
**zero** NaN/Inf and **zero** flatline across all 8000 frames. So the dropped sessions
are acquisitions where the return simply is not in the 0.9–3.0 m gate (subject 6's 8am
session: 0/100 frames in-band; subject 1's 4pm: 1/100), not corrupted files. Recorded
as a finding; **no threshold was touched** in response, per the §0 invariant.

**The RMS diagnostic is trigger-happy on this data — interpret accordingly.** 2752
frames (34%) carry the flag, concentrated in a few sessions (subject 16: 4pm 99/100,
2pm 82/100; subject 11 and 13: mostly 0). Cause is structural, the same effect the
unit tests hit: the robust z is taken across a frame's own **20** chirps, which are
near-identical, so the MAD is tiny and the z is very sensitive. This is the
reference's own definition and it rejects nothing — but the count must be read as
"chirp-to-chirp variability is non-uniform here", not as "2752 anomalous frames".
Kept diagnostic-only exactly as the plan freezes it.

**Fail-closed join (§2.2), and why it is not paranoia.** `_join_qc` asserts key
uniqueness on both sides, merges with `validate="one_to_one"` plus an indicator check
for left/right-only keys, asserts the row count is unchanged, and restores `SORT_KEYS`
order. Four tests inject duplicate / missing / extra / duplicate-manifest keys and
require a loud `ManifestError`. The M1 trap this defends against is real:
`subject_1_10am` sorts *before* `subject_1_8am`, so any positional join silently
attributes one session's QC verdicts to another — `test_join_is_by_key_not_row_order`
constructs exactly that scenario (failure on 10am, none on 8am) and would catch it.

**Reconciliation.** Per-reason columns are non-additive incidence counts; the identity
asserted everywhere is `n_pass + n_fail_any == n_frames`. The all-zero-frame test
pins the overlap case (flatline **and** low in-band: counted in both reason columns,
once in `n_fail_any`).

**Artifact.** `results/qc/qc_survival_10ghz.csv` — written under
`config.paths.results_dir` (the config is the single output-path authority), re-read
and reconciliation-checked after writing, and verified **not** gitignored. Per the
ground rules it is written and verified but **not committed** until asked.

**Tests:** `uv run pytest` → **237 passed, 8 skipped**; `--realdata` survival +
determinism tests green. The survival test asserts structure only (80 cells present,
ratios in [0,1], reconciliation, `min_pass == ceil(0.5 × actual count)`) and
deliberately makes **no** expected-rate assertion.

**Next:** step 5 — `run_regression.py` folds over post-QC evaluable subjects.

---

## 2026-07-21 — M2 step 2: `src/dehyd/qc/screens.py` + `tests/test_qc.py`.
## **Success**, after two wrong assumptions of mine were corrected by the tests.

**What was built.** The four frozen 10 GHz screens as pure per-frame functions:
`FrameQC`, `run_qc_frame`, `run_qc_cube`, `in_band_mask`. Implements
MILESTONE_2_PLAN §2.1. 34 tests (T-QC1..15).

**Frozen semantics as implemented.** NaN/Inf → reject; flatline = any chirp whose
200-bin magnitude histogram (over that chirp's own range) has a bin ≥ 0.25·534 = 133.5
i.e. ≥134; in-band = periodic-Hann 534-pt FFT, half-spectrum bins 0..266 (DC in,
Nyquist out), averaged over the 20 chirps, band 2931.7–9772.4 Hz widened by ±1000 Hz
→ **mask bins 2..11 (10 of 267)** — measured, not assumed; reject below 0.30.
Robust-RMS z across the frame's own 20 chirps, >4.5 → flag only.

**`passed` is a property, not a stored field** (a small departure from the plan's
sketch): the rejection rule then cannot be violated by construction, and `rms_flag`
can never leak into it. T-QC14 stays meaningful by asserting the battery actually
contains frames that are RMS-flagged *and* passing.

**Wrong assumption #1 — numpy does NOT expand a zero-width histogram range.** The
plan (and my code comment) claimed a constant chirp yields a single populated bin.
In fact `np.histogram(x, bins=200)` **raises** `ValueError: Too many bins for data
range` whenever the span is too narrow for 200 distinct float64 edges — and that
includes any *near*-constant chirp, not just an exactly constant one: a noiseless CW
tone `exp(2πift)` has |x| constant to ~1e-16. Ten tests crashed. Fix: build the edges
with `linspace` and check `edges[:-1] < edges[1:]` (numpy's own criterion) — if the
span is degenerate, flag flatline directly and skip the histogram. That is also the
*correct physics*: constant magnitude is exactly what the screen exists to catch, and
MATLAB's `histcounts` reaches the same verdict by choosing its own bin width. Pinned
by `test_qc3_degenerate_magnitude_spread_is_flatline_not_a_crash`.

**Wrong assumption #2 — a perfect CW tone is not a valid "clean frame" fixture.**
Following from the above, the test helper now adds small seeded noise
(`tone_frame(f, noise=0.01, seed=…)`), with `pure_tone_frame` kept only where the
degenerate case is the thing under test. This is more realistic anyway — real
acquisitions always carry receiver noise.

**Two test claims corrected to what is actually true** (rather than forcing the code
to fit them):
- *RMS threshold.* Chirps differ only by noise, so the MAD is tiny and the robust z is
  very sensitive: an unperturbed frame already sits at z ≈ 2.6, and a ×1.5 chirp gives
  z ≈ 1350. The "not flagged" case therefore needs **no** perturbation at all.
- *Margin.* Removing the ±1000 Hz margin does **not** flip the verdict for a
  10 300 Hz tone (ratio 0.972 → 0.454, still above 0.30) because Hann leakage keeps
  energy inside the bare gate. The test now asserts what is true and load-bearing: the
  ratio drops by >40% and the mask shrinks 10 → 7 bins. Asserting a flip would have
  been asserting something false.

**Also verified empirically before writing tests** (rather than trusting arithmetic):
mask bins 2..11; in-band ratios 1.0 m→0.9995, 2.5 m→1.0000, 10.3 kHz→0.972,
12 kHz→0.0546, 50 kHz→0.0000. F_BEYOND_MARGIN=12 kHz is one bin past the mask edge and
correctly fails, which makes T-QC10 a tight test rather than a trivial one.

**Tests:** `uv run pytest tests/test_qc.py` → **34 passed**. T-QC7 (companion-frame
independence) and T-QC9 (QC gate ≠ model gate) are the two that carry the leakage
guarantee; T-QC15 covers the `in_band_mask` zero-bin / all-bin guards.

**Next:** step 3 — manifest QC columns, eligibility, fail-closed join.

---

## 2026-07-21 — M2 step 1: QC config margin + field/cross-field validation.
## **Success — and it immediately caught a latent config bug.**

**What was built.** `QCConfig.in_band_margin_hz = 1000.0` (owner decision: the
reference `BandMarginHz` code default, ~1 FFT bin at df = 520834/534 ≈ 975.3 Hz);
`configs/preprocess.yaml` mirror; and real validation for every field M2 consumes,
replacing M1's generic `_build_frozen_section` for the `qc` and `preprocess` sections
(`_build_qc`, `_build_preprocess`, `_build_wst`). Implements MILESTONE_2_PLAN §2.3.

**Values and why.** `histogram_bins` positive int; `flatline_max_bin_fraction` and
`min_frame_fraction` in (0,1]; `min_in_band_energy_ratio` in [0,1];
`rms_robust_z_threshold` > 0; `in_band_margin_hz` >= 0; gates = exactly two finite,
positive, strictly increasing metres, **normalised list -> tuple** (a frozen dataclass
must not carry a mutable value, and provenance should record one type). `bool` is
rejected everywhere it would otherwise pass as an `int`/number.

**Cross-field check, and one branch deliberately NOT written.** `_check_qc_band`
rejects (i) a gate mapping at/above Nyquist and (ii) a margin that widens the band
across the whole represented spectrum (the ratio would be identically 1 — a screen
that can never fire). The plan also listed an "empty band after clamping" check; while
implementing it I proved it **unreachable** — the margin only widens, so
`lo <= f_lo < nyquist <= hi` holds whenever the Nyquist check passes. Dead code is
worse than absent code, so it was dropped and the reasoning recorded in the docstring.
The remaining bin-level guards (>=1 bin of support; not *every* bin) need the
fast-time length and belong in `in_band_mask` (step 2).

**The bug it caught immediately.** With types actually checked, 19 tests failed on
`preprocess.bandwidth_hz must be a number, got str`. Cause: **YAML 1.1 only parses an
exponent as a float when the exponent carries a sign.** `bandwidth_hz: 500.0e6` was
loading as the *string* `"500.0e6"` — and had been since M1. It was invisible because
nothing consumed the value until now; `chirp_time_s: 1024.0e-6` was fine only by
luck of its `-`. Fixed to `500.0e+6` with a warning comment, and pinned by
`test_radar_constants_load_as_floats_not_strings` (asserts parsed *type* and value),
so the canonical config cannot regress into string arithmetic. Had this survived, the
first symptom would have been a bizarre failure inside the QC band mapping.

**Also added.** `beat_band_hz(gate_m, B, Tchirp)` — the FMCW range->beat-frequency
mapping (`HzPerM = 2*(B/Tchirp)/c`) in `config.py`, shared by the cross-validation and
(step 2) the QC mask, so the physics is not duplicated. `SPEED_OF_LIGHT_M_S` is
written out rather than imported from scipy: it is exact by SI definition, and config
validation should not depend on a numerics package importing.

**Tests:** `uv run pytest` → **185 passed, 8 skipped** (was 151/8 at M1 close);
34 new config tests, mostly a parametrised bad-value table covering out-of-range
numbers, wrong types (incl. `bool` and `.inf`), malformed gates, and both cross-field
branches.

**Next:** step 2 — `src/dehyd/qc/screens.py` + `tests/test_qc.py`.

---

## 2026-07-21 — M1 commit: `.gitignore` was silently excluding `src/dehyd/data/`.
## **Bug found and fixed at commit time.**

**What happened.** Staging the milestone-1 commit, the file list was missing **all five
files** of `src/dehyd/data/` — `sessions.py`, `loader_10ghz.py`, `ground_truth.py`,
`manifest.py`, `__init__.py`. Everything else staged normally.

**Cause.** The `.gitignore` inherited from the initial commit contained `data*/`, which
in gitignore syntax is **unanchored** — a pattern without a leading slash matches at
*any* directory depth, so it excluded `src/dehyd/data/` along with the intended raw-data
tree. Confirmed with `git check-ignore -v src/dehyd/data/manifest.py` →
`.gitignore:4:data*/`.

**Fix.** Anchored the rule to the repository root: **`/data*/`**. Both directions
re-verified — `data/10ghz/*.mat` and `data/weight/*.xlsx` are still ignored, and
`src/dehyd/data/*` is now tracked.

**Why this matters and how it was caught.** The local working tree and the full test
suite were completely unaffected — the files existed on disk, so all 159 tests passed
either way. The failure would only have appeared on a **fresh clone or on IBEX**, as a
`dehyd.data` package that imports nothing, with the original machine looking healthy.
It was caught only by *reading the staged file list* before committing rather than
trusting `git add -A`. Lesson recorded: for a commit that introduces a new package
directory, check the staged list against the intended tree, especially when the
repository ignores a directory whose name is a common word.

**Commit:** `f3fbade` — 34 files, 5783 insertions, working tree clean afterwards.

---

## 2026-07-21 — **MILESTONE 1 COMPLETE.** Definition of done met in full.

**D1 — mandatory suite, no private data.** `uv run pytest` → **151 passed, 8 skipped**
on a checkout with no `data/` tree required (the 8 skips are the 7 `realdata` tests plus
T18).
**D1 count check.** `uv run pytest tests/test_no_leakage.py -m "not realdata"` →
**24 passed, 1 skipped, 1 deselected** — T18 is the *only* skipped non-`realdata`
leakage test, verified as a count so a mis-scoped skip cannot hide the suite.
**D2 — real-cohort acceptance.** `uv run pytest --realdata` → **158 passed, 1 skipped**
(T18). Manifest builds and validates on the real 80 files: 8000 frames, 16 subjects,
80 sessions.
**D3 —** both ground-truth cross-checks pass on the real workbook for all 16 subjects.
**D4 —** every fold anywhere originates from `eval/splits.py`; nothing else constructs
splits.
**D5 —** the smoke runs end to end on real data and writes provenance with 80 radar
hashes + workbook hash + git rev.
**D6 —** HISTORY.md has a per-step entry (steps 1–9); SECOND_CHAPTER.md §0.1 written.
**D7 —** `plans/implementation_plan.md` and `plans/MILESTONE_1_PLAN.md` agree (the A1
and A7 amendments were applied during the review rounds).

**Milestone-1 scoreboard.** 8 source modules, 8 test modules, **159 tests**. Two
genuine environment/dependency facts discovered empirically rather than assumed
(kymatio↔scipy `sph_harm`; openpyxl formula caches). Four test-scaffolding bugs found
and fixed by the tests themselves during the build (openpyxl `value=None` no-op;
`rel_path` string ordering vs session ordering; `tests/` not a package; `FoldResult`
missing `train_subjects`). The leakage suite was validated **adversarially** — a
deliberately leaky procedure fails it — so its green state carries evidence, not just
absence of failure.

**Open for M2:** install `h5py` and run the minimal 77 GHz audit; QC screens with frozen
thresholds and per-frame reason codes; add QC/eligibility columns to the manifest;
`configs/ibex.yaml` when the cluster roots are known.

---

## 2026-07-21 — M1 step 9: entry-point stub + end-to-end smoke on real data.
## **Success.**

**What was built.** `experiments/run_regression.py`. Implements MILESTONE_1_PLAN §2.9.

**Smoke result (real data, this machine).**
```
config       : configs/exp_a_regression.yaml
device       : cpu   seed: 20260721
ground truth : 16 subjects, 80 sessions
               Delta m% range -2.02 .. 0.00
manifest     : 8000 frames, 16 subjects, 80 sessions
folds        : 16 outer (16 selectable), 5 inner each
provenance   : results/runs/20260721T094017375792Z_2a26fff2/provenance.json
```
The Δm% range **−2.02 … 0.00** matches the expected ≈0 to ≈−2% from the workbook
inspection, and N_eval = 16 with 15 training subjects and 5 inner folds per outer fold
is exactly the full-cohort case the protocol describes.

**Provenance artifact verified:** 80 radar hashes + the workbook hash, all as logical
`rel_path`s; resolved config; 16 fold records with roles; git commit/branch/dirty;
package versions (with `torch: null`, `h5py: null` positively recorded as absent);
platform; `slurm_job_id: null` locally.

**Decisions.**
- The stub deliberately **stops after the data spine**; modeling lands at M6 on top of
  exactly these folds. It prints what it built rather than pretending to model.
- `sys.path` insertion of `src/` so the script runs without an editable install.
- **`results/runs/` is gitignored.** Every invocation writes a new timestamped run
  directory, so committing them all would be noise; `results/` itself stays available
  for curated, reported artifacts added deliberately. Reversible if the owner prefers
  full provenance history in git.
- **`configs/ibex.yaml` not created.** The overlay *mechanism* is implemented and
  tested (`--config` is repeatable, later files win, and a test asserts an overlay
  replaces the data root), but a committed IBEX file would have to name paths that do
  not exist yet and would fail input-path validation locally. It is written at the
  first IBEX milestone, when the real roots are known.

**Next:** step 10 — journal close-out (SECOND_CHAPTER.md milestone-1 section).

---

## 2026-07-21 — M1 step 8: `tests/test_no_leakage.py` (T1–T19) + reference procedure.
## **Success**, and verified adversarially.

**What was built.** `tests/reference_procedure.py` (the contract `harness.py` must
satisfy) and `tests/test_no_leakage.py` (25 tests covering T1–T19). Implements
MILESTONE_1_PLAN §4.

**The reference procedure.** Deterministic nested select-and-refit over
`StandardScaler → Ridge`, α ∈ {0.1, 1, 10}, folds taken **only** from
`eval/splits.py`. Selection metric is the **subject-balanced** session-level MAE. It
returns an auditable bundle per fold — selected config, full inner score table,
per-inner-fold *and* final fitted parameters, training/val/test predictions, and the
fit audit — so tests verify **roles**, not implementation trust. At M6 the leakage
tests rebind to the real harness and this module is deleted.

**Determinism mechanism (the corrected one).** `Ridge` has **no `n_jobs`**, and BLAS
thread env vars set inside a test arrive too late (NumPy/SciPy are already imported at
collection). So the numeric work runs inside `threadpool_limits(1)` with an explicitly
deterministic `solver="cholesky"` instead of leaving `solver="auto"` free to switch
algorithm. A test **asserts the achieved limit** via `threadpool_info()` rather than
documenting an intention. T10 (two unmutated runs bit-identical) is the precondition
that makes every later bit-for-bit comparison non-vacuous.

**Both CV levels are tested, which is the point.**
- **T11–T15 (outer):** mutating the held-out subject's features/labels/both leaves
  selected α, the inner score table, every fitted parameter (inner *and* final), and
  the training predictions **bit-identical** (`.tobytes()`); power checks confirm
  feature mutation moves the held-out prediction and label-only mutation moves the
  score but not the prediction.
- **T16 (inner):** the outer mutation **cannot** detect fitting on
  `inner_train + inner_val`, because inner-val subjects *are* outer-training subjects.
  So a separate test mutates a training subject and asserts that, for the folds where
  it is **validation**, that fold's fits are bit-identical — while its own validation
  predictions do move. Scope is deliberate and documented: folds where it is
  inner-train legitimately change, as may the selected config.
- **T17:** with equal session counts, subject-balanced and pooled MAE are numerically
  identical, so a pooled implementation would pass unnoticed. Tested against a
  **hand-calculated** value on a deliberately unequal fixture (5/2 sessions → 5.5, not
  25/7), plus an end-to-end run with counts {5,5,4,2,5,3,5,4}.
- **T19:** the audit must map every fitted quantity to the subject set it came from —
  inner fits from exactly that fold's `inner_train`, the final refit from exactly the
  full `outer_train`, no audited set ever containing the test subject — plus a test
  that the audit **covers every fitted quantity** (an audit with silent omissions
  would be worthless).

**T18 skip scope (the subtle one).** Both guards — `pytest.importorskip("torch")` and
the static marker — are **inside the T18 function**. A module-level `importorskip`
would skip T1–T17 and T19 too, letting the file report green with nothing verified.
Verified by the acceptance command:
`uv run pytest tests/test_no_leakage.py -m "not realdata"` → **24 passed, 1 skipped
(T18 only), 1 deselected (R1)** — checked as a *count*, not as "no failures".

**Adversarial verification (does the test actually catch a leak?).** A passing test
proves nothing on its own, so the reference procedure was monkeypatched to fit the
scaler on **train + held-out** — the classic leak — and T13 was re-run. Result: clean
procedure passes, **leaky procedure fails**. The test has teeth.

**Two failures during the step, both mine, both in test scaffolding.**
1. `from .reference_procedure import ...` → `ImportError: attempted relative import
   with no known parent package`. `tests/` is not a package; switched to an absolute
   import (pytest's default import mode puts the test dir on `sys.path`).
2. T19 → `AttributeError: 'FoldResult' has no attribute 'train_subjects'`. The audit
   needs the outer-training set to check against; added `train_subjects` to
   `FoldResult`. This is a genuine improvement to the harness contract, not a
   workaround — the real `harness.py` must expose it too.

**Verification.** `uv run pytest` → 151 passed, 8 skipped.
`uv run pytest --realdata` → 158 passed, 1 skipped (T18).

**Next:** step 9 — `experiments/run_regression.py` stub and the M1 end-to-end smoke.

---

## 2026-07-21 — M1 step 7: provenance recorder. **Success.**

**What was built.** `src/dehyd/provenance.py`, `tests/test_provenance.py` (14 tests).
Implements MILESTONE_1_PLAN §2.8.

**Concrete decisions and why.**
- **`results_dir` is the single output authority.** `record_run(config, manifest,
  folds, extra)` has **no `out_dir` parameter**, so the destination cannot be given two
  ways and disagree. Tests supply a `Config` whose `results_dir` is `tmp_path`, which
  also keeps every test write outside the repo — a test run therefore cannot alter the
  git-dirty flag that a later assertion reads.
- **The ground-truth workbook is hashed alongside the 80 radar files.** Hashing only
  radar data would let the labels change without provenance noticing.
- **Logical identity + physical hash.** Entries are `{rel_path, sha256}`: `rel_path`
  from the manifest (portable across Windows/IBEX), hash computed on the resolved file.
  Tests assert no absolute paths and no `..` segments.
- **Canonical serialization** — radar entries sorted by `rel_path`, folds sorted by
  test subject, subject sets as sorted lists, `json.dumps(sort_keys=True)`. A test
  asserts two runs on unchanged inputs differ **only** in `timestamp_utc`.
- **Windows-safe run directories.** `results_dir/runs/<YYYYMMDDTHHMMSSffffffZ>_<rev>/`
  — no colons (invalid in Windows paths) and microsecond precision so two runs in the
  same second cannot collide; a real ISO-8601 timestamp is kept *inside* the JSON.
  Existing `provenance.json` → raises rather than overwriting (tested by pinning the
  stamp format so two runs collide on purpose).
- Package versions include `torch` and `h5py` as `None` — a positive record that they
  were absent, rather than silence.

**One failure during the step, and what it was.** `test_hash_changes_when_data_changes`
failed with two identical hashes. Cause was in the **test**: it compared
`radar_files[0]`, but that list is sorted by `rel_path`, where
`"subject_1_10am.mat"` sorts **before** `"subject_1_8am.mat"` — so index 0 was not the
file the test had modified. Fixed by looking entries up **by path**, and strengthened
to also assert every untouched file's hash is unchanged. Worth recording because the
same string-vs-session ordering trap will recur wherever `rel_path` order is mistaken
for session order.

**Verification.** `uv run pytest` → 127 passed, 6 skipped.

**Next:** step 8 — `tests/test_no_leakage.py` (T1–T19), the milestone capstone.

---

## 2026-07-21 — M1 step 6: nested-LOSO splitter (the single fold source). **Success.**

**What was built.** `src/dehyd/eval/splits.py`, `tests/test_splits.py` (23 tests).
Implements MILESTONE_1_PLAN §3.

**Concrete decisions and why.**
- **`GroupKFold` over one row per subject, not per frame.** The inner splitter is fed
  a `(n_train_subjects, 1)` array with `groups = subject_ids`, so it is literally
  splitting *subjects*; frame-level selection happens downstream by filtering on the
  returned subject sets. This makes it structurally impossible for the splitter to
  emit a frame-level split, which is the invariant it exists to protect.
- **Adaptive inner count `min(n_inner_max, n_train)`** — 5 folds at the full cohort
  (15 training subjects), 5 at the 6-subject smoke subset, 3 at n_train=3.
- **`min_train_subjects` constrains the outer-training pool** (owner decision 4). At
  the boundary `n_train == 3`, `GroupKFold(3)` fits each inner model on 2 subjects;
  a test asserts exactly this so the accepted consequence is visible in the suite
  rather than buried in prose. Below the floor the fold is returned with
  `selectable=False` and **no** inner folds — reported as non-selectable, never run
  degenerate.
- **Frozen dataclasses** (`OuterFold`, `InnerFold`, `frozenset` members) so a consumer
  cannot mutate a fold in place; tested.
- **No RNG anywhere.** Subjects are sorted on entry and GroupKFold's assignment is
  deterministic, so `nested_loso_splits(x) == nested_loso_splits(x)` and input order is
  irrelevant — both tested (S7).
- **Duplicate subject ids raise.** A subject appearing twice would let one copy train
  while another is held out — the exact failure LOSO exists to prevent.
- `iter_triples()` provides the flat `(inner_train, inner_val, test)` view, reconciling
  the main plan's "(train, val, test)" phrasing with several inner folds per outer fold.

**Verification.** All seven documented invariants S1–S7 are unit tests:
S1 test∉train; S2 inner disjoint and ⊆ outer-train with test in neither; S3 inner val
sets **partition** outer-train (asserted, not assumed from the GroupKFold docs); S4
each subject held out exactly once; S5 non-empty when selectable; S6 adaptive count at
n∈{16,6,4,3}; S7 determinism. Full suite: 113 passed, 6 skipped.

**Next:** step 7 — `provenance.py` + `tests/test_provenance.py`.

---

## 2026-07-21 — M1 step 5: frame manifest + structural gate. **Success.**

**What was built.** `src/dehyd/data/manifest.py`, `tests/test_manifest.py` (17 tests).
Implements MILESTONE_1_PLAN §2.6.

**Concrete decisions and why.**
- **Logical file identity (`rel_path`), not repo-relative.** The manifest stores the
  path **relative to `data_10ghz_dir`** (`subject_1_8am.mat`), resolved against that
  root for I/O and hashing via `resolve_path()`. A repo-relative path would break on
  IBEX, whose data root lives outside the repository — the same file would then carry
  machine-specific `..` segments and a different identity per machine. A test asserts
  no `..`, no leading `/`, no drive letters.
- **Deterministic ordering + fixed dtypes.** Sorted by `(subject, session_idx,
  frame_idx)` with the index reset, and every column dtype asserted. Verified by a test
  that **monkeypatches `Path.glob` to return reversed order** and asserts the two builds
  are frame-for-frame identical — so filesystem enumeration order can never reach
  training order, hashes, or saved artifacts.
- **All six checks fail loudly and name every offender**, not just the first: C1
  completeness, C2 duplicates, C3 unparseable/stray files, C4 bijection **in both
  directions** (file with no workbook row *and* workbook row with no file), C5 per-file
  structure (shape and MATLAB class), C6 actual frame counts. A test with two missing
  cells asserts both are named.
- **Mandatory tests build `GroundTruth` directly in memory** (it is two DataFrames)
  rather than round-tripping a synthetic workbook — sidesteps the openpyxl
  formula-cache limitation from step 4 and keeps these tests about the manifest.
- **Frame counts come from the file.** A synthetic session with counts {3,7,2,5,4}
  confirms per-file `n_frames_in_file` and contiguous `frame_idx` — the M2 eligibility
  rule `ceil(0.5 × actual_frame_count)` depends on this not being a hard-coded 100.
- QC columns (reason codes, eligibility) deliberately **not stubbed**; they arrive at M2.

**Verification.** `uv run pytest` → 90 passed, 6 skipped. `uv run pytest --realdata`
→ 96 passed. On the real data the manifest builds to **8000 rows** (16×5×100), subjects
exactly {1..16}, every session exactly 100 frames, all dtypes as specified.

**Next:** step 6 — `eval/splits.py` + `tests/test_splits.py`.

---

## 2026-07-21 — M1 step 4: ground truth (fixed-cell parse + cross-checks). **Success.**

**What was built.** `src/dehyd/data/ground_truth.py`, `tests/test_ground_truth.py`
(31 tests). Implements MILESTONE_1_PLAN §2.5.

**Module split forced by an openpyxl limitation (verified, not assumed).** openpyxl
writes formulas but never evaluates them, so a synthetic workbook can hold **either**
an `=I-E` formula in column J **or** a cached number — never both. No synthetic fixture
can therefore exercise the full dual-view load. The module is split so each view is
independently testable:
- `_validate_layout(ws_formula)` — headers, column-B subject identity, J formula
  structure. Formula-view fixtures.
- `_read_values(ws_data_only)` — masses, covariates, cached J, K text. Literal-value
  fixtures.
- `check_targets(...)` — pure math + both cross-checks, no I/O. Array-level tests,
  including tolerance boundary behaviour and the sign convention.
- `load_ground_truth()` — the only place the two views meet; exercised on the **real**
  workbook (which genuinely has formulas *and* Excel-written caches) in a `realdata`
  test.

**Concrete decisions and why.**
- **Identity parsed from column B** (`^Subject (\d+)$`), asserted unique and exactly
  {1..16} — not inferred from row position. The owner-confirmed radar↔workbook identity
  is thus *checked*, not assumed.
- **Extra-subject guard scans all of column B**, not just the rows below the block; a
  `Subject 17` planted at row 400 is caught (tested).
- **Covariates validated before BMI** is computed from them (age 15–80, height
  120–220 cm) — a metres-instead-of-cm height is caught before it silently produces a
  BMI of ~24000.
- **Tolerances kept at 0.05 kg / 0.05 pct-points but re-justified from the observed
  workbook**, not from an assumed recording precision: Subject 15 uses 0.05-kg
  increments and column K truncates (Subject 13: 0.5997% → "0.59"), worst observed
  deviation ≈0.0097 pct-points, so 0.05 is ≈5× the worst case. Recorded in the module
  docstring so the number is never mistaken for a claim about the instrument.
- **All problems are reported at once**, not just the first — a test asserts two
  corrupted J cells both appear in the error.

**One failure during the step, and what it was.** `test_missing_weight_detected`
initially did not raise. Cause was in the **test**, not the parser:
`ws.cell(row, col, value=None)` is a **no-op** in openpyxl (None is the sentinel for
"don't set"), so the cell was never blanked. Fixed by assigning `.value = None`
directly, and a `"n/a"` string case was added alongside. The mass check was factored
into `_is_plausible_mass()` while fixing it (also rejects `bool`, which is an `int`
subclass).

**Verification.** `uv run pytest` → 74 passed, 5 skipped. `uv run pytest --realdata`
→ 79 passed. On the real workbook: 80 session rows, 16 subjects, **both cross-checks
pass for all 16**; S0 is identically zero, all S4 deltas negative and > −3%, all BMIs
in 15–45.

**Next:** step 5 — `manifest.py` + `tests/test_manifest.py`.

---

## 2026-07-21 — M1 step 3: sessions + minimal 10 GHz loader. **Success.**

**What was built.** `src/dehyd/data/sessions.py` (the single definition of session
order), `src/dehyd/data/loader_10ghz.py` (filename parse, header inspect, full load),
`tests/test_loader.py` (22 tests). Implements MILESTONE_1_PLAN §2.3–2.4.

**Concrete decisions and why.**
- **Header-only inspection via `scipy.io.whosmat`.** Measured on a real file:
  **0.017 s**, so all 80 files cost ≈1.4 s instead of decompressing ≈1.4 GB. The
  planned fallback (full `loadmat` per file) was therefore **not** needed.
- **`whosmat` returns the MATLAB class**, confirmed `('framesRadar', (534, 20, 100),
  'double')` on the real data — so the class assertion the plan asked for is checkable
  without loading. An `int16` array of the correct shape is rejected (tested).
- **`loadmat(..., variable_names=["framesRadar"])`** so the unused
  `framesRadarIQ` [20834×2×100] is never decompressed. A test writes a file containing
  both and confirms loading succeeds regardless.
- **Frame count is read from the file, never assumed 100** — session eligibility at M2
  is `ceil(0.5 × actual_frame_count)`, so an assumed constant would silently corrupt it.
  Tested with a 42-frame synthetic file.
- **Strict filename regex.** An unparseable name raises rather than being skipped,
  because "unmatched file" is a manifest failure condition, not a benign case. Seven
  malformed-name variants tested, including `.MAT` case and trailing junk.
- Complex-dtype check lives in `load_10ghz_file` (whosmat cannot report complexity);
  a real-valued double cube of the right shape is rejected.

**Verification.** `uv run pytest` → 43 passed, 3 skipped. `uv run pytest --realdata`
→ 46 passed: **all 80 real files** inspect as `(534, 20, 100)` MATLAB-class `double`,
subjects exactly {1..16}, and a full real load is complex128 and all-finite.

**Next:** step 4 — `ground_truth.py` + `tests/test_ground_truth.py`.

---

## 2026-07-21 — M1 step 2: config system. **Success.**

**What was built.** `src/dehyd/config.py` (frozen dataclass schema + `load_config`),
`configs/{data,preprocess,wst,exp_a_regression}.yaml`, `tests/test_config.py`
(21 tests). Implements MILESTONE_1_PLAN §2.2.

**Concrete decisions and why.**
- **Two path rules, deliberately different.** `include:` entries resolve against the
  **declaring YAML's directory** (so `exp_a_regression.yaml` can say `data.yaml` and
  find its sibling); path **values** resolve against the **repo root** (so a data root
  means the same thing from any CWD or declaring file). Both are covered by a test that
  loads from an unrelated CWD via `monkeypatch.chdir` and compares the fully resolved
  configs.
- **Merge = later wins, lists replaced wholesale**, never concatenated — a later config
  states the entire intended value. Tested directly (`seed_set` replacement).
- **`include:` may not nest.** Flat composition keeps the resolution order followable;
  nesting raises.
- **Numeric floors enforced at the config layer, not just documented:** `seed_set` must
  be exactly 5 **distinct** seeds (duplicates would silently reduce effective repeats);
  `n_inner_max >= 2` (GroupKFold); **`min_train_subjects >= 3`** — deliberately stricter
  than GroupKFold's mechanical floor of 2, because the approved protocol requires ≥3
  training subjects before an outer fold is selectable. A permissive floor here would
  let an overlay YAML weaken the nested-CV rule while staying syntactically valid.
- **`wst.tilings` cannot be overridden in YAML** — the three tilings are frozen design
  constants; J and output shape are derived/measured at M4, never hard-coded.
- **`results_dir` is not required to exist** (output, created on demand) while
  `data_10ghz_dir` / `weight_xlsx` are (required inputs). This distinction is what makes
  the mandatory suite runnable on a clean checkout.
- **Mandatory tests never touch the private data:** each appends a final `tmp_path`
  overlay redirecting the input paths, so composition/merge/path-rules/validation are
  all exercised without `data/`. That the *canonical* config resolves to the real
  dataset is a separate `realdata` test.

**Verification.** `uv run pytest` → 21 passed, 1 skipped (the `realdata` test).
`uv run pytest --realdata` → 22 passed. Both gate directions confirmed working.

**Next:** step 3 — `sessions.py` + minimal `loader_10ghz.py` + `tests/test_loader.py`.

---

## 2026-07-21 — M1 step 1: environment + repo skeleton. **Success**, with one real
## dependency conflict found (scipy, not numpy).

**What was tried.** Created the pinned uv environment and package skeleton per
`plans/MILESTONE_1_PLAN.md` §1 step 1 / §2.1: `pyproject.toml` (package `dehyd`, src
layout, `requires-python >=3.11`), `.python-version` = 3.11 (uv fetched CPython
3.11.15), `uv lock` + `uv sync`, `src/dehyd/{data,eval}/` skeleton, `tests/test_env.py`,
`tests/conftest.py` (the `--realdata` gate), `.gitignore` additions.

**The env unknown resolved — but it was not the anticipated one.** The plan flagged a
possible **kymatio vs numpy 2.x** conflict, with the contingency "pin numpy<2". That
conflict does **not** exist: kymatio 0.3.0 imports and runs fine on numpy 2.4.6.
The actual conflict is **kymatio 0.3.0 vs scipy ≥1.17**: `kymatio/scattering3d/
filter_bank.py` imports `scipy.special.sph_harm`, which scipy **removed in 1.17**
(superseded by `sph_harm_y`). Symptom is subtle and would have surfaced at M4, not
here: top-level `import kymatio` **succeeds** (so a naive import smoke passes), but
`from kymatio.numpy import Scattering1D` raises `ImportError` because the 1-D entry
point pulls in the 3-D filter bank.

- **Resolution:** pin **`scipy>=1.11,<1.17`** in `pyproject.toml` with the reason in a
  comment; revisit when kymatio ships a release using `sph_harm_y`.
- **Resolved versions:** python 3.11.15, numpy 2.4.6, scipy 1.16.3, kymatio 0.3.0,
  scikit-learn 1.9.0, pandas 2.3.3, openpyxl 3.1.5, PyYAML 6.0.3, pytest 9.1.1,
  threadpoolctl 3.6.0 (arrives via scikit-learn — needed for the M1 determinism
  fixture, §4 Part C).
- **Verified after the pin:** `Scattering1D(J=7, shape=(470,), Q=(10,4), T=104,
  max_order=2)` instantiates and transforms, output shape `(742, 7)`. kymatio emits
  `UserWarning: Signal support is too small to avoid border effects` for J=7 on 470
  samples — **noted for M4**, where the plan already requires padding/output shape to
  be *measured* from the instantiated filter bank rather than assumed. Not an M1 issue.

**Why the plan's ordering paid off.** §2.1 put env resolution first precisely so an
unknown like this fails before any code depends on it. It did — and it was a different
unknown than predicted, which is the argument for resolving it empirically rather than
assuming the documented risk was the only one.

**Incidental.** `environment.yml` (planning-phase conda export) moved to
`archive/code/` per the file-hygiene rule (owner decision 2) — `uv` is now the sole
local env manager. A stale `.pytest_cache/` at the repo root has an unreadable ACL on
this machine (cannot be read, `takeown`'d, or removed without elevation) and made
pytest warn on every run; worked around by setting `cache_dir = ".cache/pytest"` in
`[tool.pytest.ini_options]` rather than leaving permanent noise in the test output.

**Outcome:** `uv run pytest` green (2 passed, no warnings).
**Next:** step 2 — `configs/data.yaml` + `src/dehyd/config.py` + `tests/test_config.py`.

---

## 2026-07-21 — Planning phase complete; plan approved and hardened. Pre-implementation.

**State:** No implementation code written yet. The design is locked in
`plans/implementation_plan.md` and is the spec milestone 1 builds against.

**What was done.** Read CLAUDE.md/AGENTS.md + ROADMAP.md in full, the paper
(`paper/`), and the MATLAB reference (`matlab/`). Inspected a real 10 GHz file
byte-for-byte (not assumed from the paper) and parsed the weight workbook. Produced the
implementation plan, then hardened it across **7 rounds of independent (Codex) review**
— every comment resolved; reviewer's final verdict was "no further comments,
implementation-ready."

**Verified data facts (not assumed).**
- 10 GHz: `data/10ghz/subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat`, 80 files, MAT v5,
  little-endian, zlib. Var `framesRadar` = MATLAB **double** `[534 fast-time × 20 chirps
  × 100 frames]`, complex; on disk the elements are `miINT16` (space optimization) so
  `scipy.io.loadmat` returns **complex128**. Also `framesRadarIQ` [20834×2×100] (raw IQ,
  unused). One file = one subject/session = 100 frames.
- 77 GHz: MAT **v7.3/HDF5**, ~285 MB each (~23 GB), needs `h5py` (not yet installed —
  milestone-2 audit installs it). h5py-reported shape (reviewer-sampled)
  `(16,256,256,125)=(Nrx,Nchirps,Nfast,Nframes)`; full axis reversal →
  `(Nframes,Nfast,Nchirps,Nrx)`. Fast-time↔chirp (both 256) disambiguated by a raw-data
  signal-domain check, not shape alone.
- Ground truth: `data/weight/metadata_subjects_info.xlsx`, sheet `MetaData`, rows 3–18.
  Two-row merged header → parse by fixed cell addresses. Cols E–I = weights 8am→4pm.
  Signed target `Δm% = (m(s) − m(S0))/m(S0)×100` (negative = loss), ≈0 to ≈−2%.
- Subject identity: radar `subject_N` = workbook "Subject N" (owner-confirmed; old
  MATLAB 5–20 numbering was renumbered to 1–16 for cleanliness, same subjects/order).

**Key locked decisions & why (see plan for full detail).**
- MATLAB is a **design reference only** — Python is the sole source of all reported
  numbers; correctness via Python-native self-consistency checks, not numeric diffs.
- Headline = **fluid-loss (Δm%) regression under LOSO**; 5-class demoted to secondary
  **ordinal**. Analysis unit is **session-level** (aggregate per-frame WST features to
  one vector/session) to kill pseudo-replication; per-frame is diagnostic only.
- Deliberate departures from the reference (logged here as they're implemented):
  robust standardize = median/MAD (not the reference's mean/MAD mix); range gate is a
  config parameter (default 1–2 m); WST log transform = order-aware
  (`log(S+ε)` on orders 1–2, ε=1e-6; order 0 left linear); EdgeTrim=32 **after**
  reduction.
- Scoring counts use **N_eval** (evaluable subjects), never a hard 16; session
  eligibility = `≥ ceil(0.5 × actual_frame_count)` QC-passing frames, no imputation.
- 77 GHz primary feature = **slow-time (Doppler) I/Q WST, per-Rx, feature-space fused**
  (magnitude discards Doppler phase; coherent Rx averaging risks phase cancellation).
- Stats: subject-cluster bootstrap (B=10000), seeds collapsed (metric-type-aware),
  all CIs/p-values labeled **conditional/exploratory**; effect sizes + per-subject
  spread carry interpretation.

**Outcome:** success (planning). **Next:** milestone 1 — repo scaffold, config system,
manifest + nested-LOSO splitter + provenance, and `tests/test_no_leakage.py` green
before any modeling.

