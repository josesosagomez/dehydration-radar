# REPORT_RESULTS — where the signal isn't

A walkthrough of the pipeline, the frozen parameters, the negative results, and an argument
about which part of the study is actually responsible. Written for a reviewer who has not seen
this code.

- **Cohort** — 16 subjects, 5 sessions, 80 acquisitions
- **Bands** — 10 GHz FMCW (primary), 77 GHz (cross-band section only)
- **Protocol** — nested leave-one-subject-out, frozen before any result existed
- **Status at time of writing** — Experiments A–G complete and folded in; H's robustness
  bootstrap (R = 200 per experiment and band) is running

Companion artifact (same content, browsable):
<https://claude.ai/code/artifact/f26208c6-a71b-4491-a7d8-45025acac9c3>

---

## 1. The verdict, stated first

The question was whether the fault lies with the model, the scattering transform, or its
parameters. The evidence discriminates between these more cleanly than expected, and it does not
point at any of them.

| Candidate cause | Verdict | Why |
|---|---|---|
| **The model** | Not the cause | Five classical families searched per fold (ridge, SVR, RF, GBM, k-NN) plus two networks trained end to end. All land within 0.45–0.57 MAE. Best-to-worst spread is smaller than the gap to the baseline. |
| **Wavelet scattering as the feature method** | Not the cause | Exp D ran two representations that learn their own features from the raw signal and inherit no scattering assumption. Both lose to the same baseline by the same margin. |
| **Q values and tiling parameters** | Not the cause | 3 tilings × 3 log branches × 2 gates × 2 reductions × 2 channels searched inside every fold — 72 chances per fold to find a tiling that works. Selection frequency is scattered, not peaked. |
| **Any individual scattering path** | Not the cause | Exp E ablated all 742 path groups one at a time. Importances centre on zero (median −0.0001), and the single most important group is worth less than the margin by which the model loses to a trivial constant. |
| **Measurement and target** | **Where the evidence points** | A predictor that sees only the time of day beats every radar method in both bands. At 10 GHz it wins for *all sixteen* subjects. |

**The single most informative number.** At 10 GHz the Wilcoxon statistic for radar versus the
session-index baseline is **exactly 0.0**. That is not "the baseline wins on average" — it means
every one of the sixteen held-out subjects was predicted better by a lookup table of the clock
than by the radar. A feature-engineering problem does not usually produce a clean sweep.

---

## 2. What is being measured, and what is being predicted

Sixteen fasting subjects recorded at 08:00, 10:00, 12:00, 14:00 and 16:00 (S0–S4), giving 80
acquisitions of 100 frames each. A 10 GHz FMCW radar (520.834 kHz sampling, 500 MHz sweep,
1024 µs chirp) produces a complex 534 fast-time × 20 chirp matrix per frame — 8000 frames before
quality control. A 77 GHz sensor recorded the same sessions and is used only for the cross-band
work.

The target is body-mass change as a signed percentage of each subject's own 08:00 baseline:
**negative for loss, identically zero at S0 by construction**. Across the cohort it spans 0 to
−2.02%, with most sessions well under 1%.

Two structural properties of this target matter more than anything else in this report:

- **It is tied to the clock.** Subjects fast and dehydrate progressively across the day, so Δm%
  is correlated with session index by design. Any model that decodes the time of day scores well
  without using the radar at all.
- **It is the only objective reference available.** No osmolality, no temperature, no blood
  measurements; the environmental logs were lost. Mass change over a fasting day conflates fluid
  loss with other mass fluxes, so it is a proxy with a known direction, not a gold standard.

---

## 3. The pipeline, end to end

Every stage before the model is a deterministic function of one frame and a frozen constant.
Nothing is estimated from a population, so no quantity computed before the model can carry
information between subjects.

1. **Quality control and eligibility.** Frame-level screening, then a session-eligibility rule
   expressed as a fraction of each acquisition's own frame count. Survivors: **73 of 80 sessions
   at 10 GHz** (7168 analysable frames), **72 at 77 GHz**. Frame counts are read per acquisition
   rather than assumed constant.
2. **Band gate — the range gate, in the time domain.** In FMCW the beat frequency is proportional
   to range, so restricting to a band of beat frequencies *is* the range gate. The published
   description's window→transform→filter ordering was deliberately not followed: a window before
   a time-domain recursive filter tapers real signal and suppresses nothing. Windows are applied
   only where a transform is actually taken.
3. **Chirp collapse, edge trim, normalisation.** The frame's 20 chirps collapse to one trace,
   filter edge transients are discarded, and the result is normalised — leaving a complex trace of
   **470 samples**, taken either as magnitude or as separate real/imaginary channels.
4. **Wavelet scattering.** Three frozen tilings, orders 0–2, with an order-aware logarithm:
   orders 1 and 2 are moduli and can be logged; order 0 is a signed low-pass of a mean-centred
   signal and is always left linear. Geometry is measured from the instantiated filter bank, not
   derived on paper.
5. **Per-session feature store.** Features extracted once into a per-session store, each shard
   carrying a fingerprint binding the exact QC frame membership, config hashes, raw file hash and
   build commit. A run validates the store fail-closed before reading it.
6. **Nested leave-one-subject-out evaluation.** Outer = leave one subject out. Inner =
   subject-grouped `GroupKFold`. Selection metric = session-level subject-balanced MAE. Every
   fitted transform is fit on training folds only, and a protocol guard runs before every fit.
7. **Statistics.** Subject-cluster BCa bootstrap (B = 10 000) with percentile fallback recorded;
   radar-versus-baseline by Wilcoxon signed-rank plus a cluster-bootstrap CI on the per-subject
   difference. All intervals labelled conditional/exploratory.

---

## 4. The frozen parameters

All of this was fixed at a config-freeze gate, tagged in git, **before any outer-fold result
existed**. A reviewer's first instinct is to ask whether parameters were tuned toward the answer;
here they provably could not have been.

### Scattering tilings — reference quality factors, measured geometry

| Tiling | Q | Invariance | Support | Octaves J | Paths |
|---|---|---|---|---|---|
| T1 | (10, 4) | 0.20 ms | 104 smp | 7 | 742 × 7 |
| T2 | (8, 2) | 0.30 ms | 156 smp | 8 | 466 × 3 |
| T3 | (6, 2) | 0.40 ms | 208 smp | 8 | 349 × 3 |

Q values are the reference paper's. Invariance in milliseconds converts to support as
`round(scale · f_s)`, realising each requested scale to better than two parts in a thousand; J is
the smallest integer covering that support. Path counts are read back from the instantiated bank
and pinned as regression values.

### What the inner cross-validation searched, per outer fold

| Stage | Axes | 10 GHz | 77 GHz |
|---|---|---|---|
| Stage 1 (features) | reduction × channel × tiling × log branch × range gate | 72 | 9 |
| Stage 2 (models) | 5 families × grid, each ≤ 12 points | 41 | 41 |
| **Total candidates** | — | **113** | **50** |

Stage 1 runs at a fixed ridge anchor (α = 1.0); Stage 2 searches families at the Stage-1 winner.
Five seeds; the inner metric is the mean over seeds, and each seed is scored separately at the
outer level — never ensembled.

Only one fitted quantity exists inside the feature path: the tuned-ε log branch, where
ε_o = 0.1 · scale_o is computed fold-locally from training subjects only and recorded in the fit
audit like any other fitted quantity.

**Measured, and worth a reviewer's attention.** The range gate was chosen in advance from the
assumption that a seated subject sits about a metre from the antenna. The subject actually sits
at **1.50–1.80 m** — wrong by roughly half a metre, and the 1–2 m gate happens to be well placed
anyway, with the measured range near its centre. The gate was *not* adjusted after this was
observed. Band-gate retention nevertheless varies from 0.06 to 0.64 across sessions (median 0.41),
which describes how unequal these recordings are before any feature exists.

---

## 5. Results

Four independent analyses on the same frozen protocol. Positive radar−baseline differences mean
the radar is **worse**.

### Experiment A — fluid-loss regression

| Metric | 10 GHz | 77 GHz |
|---|---|---|
| Session MAE, subject-balanced | 0.469 [0.409, 0.568] | 0.495 [0.404, 0.646] |
| Session RMSE | 0.593 [0.509, 0.747] | 0.581 [0.483, 0.721] |
| Pooled predicted-vs-actual r | −0.138 [−0.286, 0.075] | −0.153 [−0.407, 0.174] |
| **Radar − baseline** | **+0.200 [0.145, 0.260]** | **+0.216 [0.127, 0.296]** |
| Wilcoxon p | 3.05 × 10⁻⁵ | 7.6 × 10⁻⁴ |

The radar regressor loses to a session-index-only baseline in both bands; the mean-difference CI
excludes zero by a wide margin. Pooled correlation is indistinguishable from zero. Per-subject r
ranges from −0.99 to +0.90 — inconsistent in sign, which is itself the finding: whatever each fold
fits does not generalise as a stable subject-level relationship.

### Experiment B — clock decoupling

Exp A cannot separate "no radar signal" from "signal swamped by the fasting-clock confound".
Exp B removes the train-only session mean and asks whether radar tracks *between-subject*
variation within a fixed session, where every subject was measured at the same clock time.

| Quantity | 10 GHz | 77 GHz |
|---|---|---|
| Radar residual MAE | 0.389 [0.310, 0.523] | 0.341 [0.268, 0.443] |
| Zero-residual baseline MAE | 0.341 [0.270, 0.481] | 0.316 [0.247, 0.397] |
| **Radar − baseline (primary)** | **+0.0475 [0.0230, 0.0749]** | +0.0246 [−0.0066, 0.0756] |

10 GHz still loses, significantly, after the clock is removed — so its Exp A defeat is not fully
explained by the confound. 77 GHz becomes indistinguishable from the trivial constant in either
direction, which is more consistent with a confound explanation. Four independently-fitted
single-session models at 10 GHz all agree in direction with the pooled result.

### Experiment C — five-class ordinal (S0–S4)

| Band | Arm | QWK | Adjacent acc. | Class-unit MAE |
|---|---|---|---|---|
| 10 GHz | regress-then-threshold | **−0.212 [−0.365, −0.030]** | 0.534 | 1.553 |
| 10 GHz | Frank–Hall | **−0.197 [−0.312, −0.075]** | 0.521 | 1.658 |
| 77 GHz | regress-then-threshold | **−0.278 [−0.461, −0.077]** | 0.558 | 1.492 |
| 77 GHz | Frank–Hall | +0.025 [−0.281, 0.243] | 0.611 | 1.347 |

Three of four arms have kappa intervals entirely below zero. The correct reading is *no usable
ordinal signal*, not inverse predictive ability: predictions collapse toward the middle classes
and then run counter to truth, which is what a no-signal predictor looks like under a
chance-corrected metric.

### Experiment D — six representations against the clock

| Family | 10 GHz MAE | 77 GHz MAE |
|---|---|---|
| **session_index** *(knows only the time of day)* | **0.269** | **0.278** |
| physics | 0.446 | 0.479 |
| cnn1d_matched | 0.451 | 0.497 |
| radar (WST + classical) | 0.469 | 0.495 |
| cnn1d_raw *(learns its own features)* | 0.468 | 0.492 |
| spec2d_matched | 0.528 | 0.478 |
| spec2d_raw *(learns its own features)* | 0.569 | 0.531 |

Lower is better. Every radar-based representation lands between 0.45 and 0.57 and every one loses
to the clock. This is the strongest single piece of evidence that the negative result is not an
artifact of the feature choice.

**A detail that should not be glossed over.** Radar "beats the composite" at 10 GHz — but the
composite's inner CV selected `spec2d_raw`, the *worst* of the six families, in 16 of 16 folds. So
the composite *is* spec2d_raw, and the win reduces to beating the weakest alternative. An inner-CV
procedure that consistently selects the worst family is itself evidence that inner scores carry no
information about outer performance — which is what you would expect if there is nothing to select
on.

### Experiment E — leave-one-path-group-out ablation

The most direct test of "it's the features". A **fixed pre-registered anchor model** — T1,
Q = (10,4), magnitude channel, reduction A, log off, 1–2 m gate, ridge α = 1.0 — is refit under
outer LOSO with one scattering path group removed at a time, against the Exp B residual target.
The anchor is deliberately *not* the best model from Exp A/B and is never swapped for one.
Positive importance means the full model predicted that held-out subject better with the group
present.

| | 10 GHz | 77 GHz |
|---|---|---|
| Path groups ablated | 742 | 424 |
| Median importance | −0.0001 | −0.0001 |
| SD across groups | 0.0053 | 0.0006 |
| Range | −0.047 … **+0.034** | −0.004 … **+0.005** |
| Groups with positive mean | 265 / 742 (36%) | 184 / 424 (43%) |
| Groups helpful across subjects (IQR above zero) | **22 / 742 (3%)** | 34 / 424 (8%) |

By scattering order at 10 GHz: order 0 (1 group) −0.023; order 1 (55 groups) mean +0.0002, max
+0.028; order 2 (686 groups) mean −0.0001, max +0.034.

**The decisive comparison.** The single most important path group at 10 GHz is worth **+0.034**
residual Δm% points. The margin by which the radar model *loses* to a trivial constant in Exp B is
**+0.0475**. No individual path group contributes enough to close the gap to doing nothing — and
36% of groups having positive mean means most groups are, if anything, mildly harmful to remove
nothing at all: their removal *improves* prediction.

Two caveats the experiment states about itself, both of which I accept: attribution here is model
reliance, not causality, and it cannot establish or refute a dielectric mechanism; and correlated
paths share credit, so a group whose information survives in its neighbours can measure as
unimportant while carrying the same signal. The aggregate picture — a distribution centred on zero
with 3% of groups consistently helpful, which is about what chance would give — is what carries
the weight, not any single row.

### Supplementary frozen WST-order trajectory diagnostic

An independently frozen diagnostic tested the complete first- and second-order trajectories
directly rather than asking whether a fitted predictor relied on them. It analyzed every path and
retained all null and inconsistent rows; sensitivity banks could test persistence but could not
create or rescue a primary candidate.

| | 10 GHz headline | 77 GHz secondary |
|---|---|---|
| Population | A65: 65 sessions; 12 fixed complete-trajectory subjects in the candidate rule | QC72_77: 72 sessions; 13 fixed complete-trajectory subjects in the candidate rule |
| Primary order 1 | **Path 40 descriptive candidate**; positive majority sign; QC and boundary persistence; Holm-adjusted p = **1.0** | No candidate |
| Primary order 2 | No candidate | No candidate |
| Sensitivity result | Several predeclared flags, none eligible to replace the primary result | No flags in any bank |

The attractive-looking 10 GHz path is not a corrected statistical result. Persistence shows that
its descriptive direction survives the declared QC and boundary checks; it does not overcome the
family size. With 12 fixed complete-trajectory subjects the smallest possible two-sided sign
p-value is 0.00048828125, which makes Holm support structurally impossible for nearly every large
10 GHz order-2 family. With 13 subjects, every 77 GHz order-2 family is structurally incapable.
That is a power limitation, not a proof that coefficient structure is absent.

The diagnostic therefore sharpens rather than changes Experiment E's interpretation: there is one
hypothesis-generating 10 GHz trajectory, but no multiplicity-corrected order evidence and no
independent 77 GHz counterpart. Path 40 is not selected for ML, no bank is ranked, and no paths are
matched or fused across bands. The complete report is bound to normalized SHA-256
`d7eab500560e8f49519647032cd8998c90464082f8160b9b3f142c063bcd1994`.

### Experiment F — confound sensitivity

Direction convention: **positive means adding that component makes prediction worse.** 73 sessions
(10 GHz) / 72 (77 GHz), 16 subjects, 16 outer folds.

| Contrast | 10 GHz | 77 GHz |
|---|---|---|
| **radar given clock** | **+0.364 [0.245, 0.485]**, Holm p = 0.0003 | **+0.257 [0.166, 0.347]**, Holm p = 0.0006 |
| **radar given clock + covariates** | **+0.301 [0.185, 0.428]**, Holm p = 0.0003 | **+0.174 [0.014, 0.282]**, Holm p = 0.025 |
| covariates given clock | +0.063 [0.025, 0.128] | +0.081 [0.030, 0.235] |
| covariates given clock + radar | +0.000 [−0.017, 0.010] | −0.002 [−0.004, 0.000] |

Both primary milestone contrasts are Holm-significant in both bands and in the wrong direction:
**adding radar on top of a clock model significantly degrades prediction**, with or without age,
height, baseline mass and BMI.

Heart rate is confirmed `not_estimable_missing_heart_rate` — no HR column in the workbook, no HR
file in the data roots, zero observations, and no proxy substituted. The artifact carries its own
constraint: this is a limited clock/static-covariate sensitivity result, it is **not** an HR
adjustment, and temperature and glucose remain uncontrolled.

### Experiment G — matched-session cross-band fusion

| | Value |
|---|---|
| Matched population | 65 cells (of 73 / 72), 16 subjects, 15 unmatched |
| **Primary: fused − 10 GHz** | **+0.018 [−0.004, +0.061]** — CI crosses zero |
| Subject-balanced MAE | 10 GHz 0.482 · 77 GHz 0.480 · equal-weight 0.476 · fused 0.501 |

Fusion does not help, and the learned weight says so directly: the fold-local α (combiner
`α·pred_10 + (1−α)·pred_77`, selected on out-of-fold MAE) is **1.0 in 9 of 16 folds** — the
procedure collapses to "use 10 GHz alone" in most folds. Per the pre-registered stance, fusion was
never required to beat 10 GHz and is reported as observed. The artifact carries its own limitation:
conditional and exploratory, generalizes to no other cohort, and cannot rescue the single-band
outcome.

---

## 6. Why the fault is unlikely to be the features or the model

Seven independent lines converge, each capable of failing separately:

- **Model class is ruled out by breadth.** Five classical families with grids, selected per fold,
  plus two networks trained end to end. Best and worst differ by about 0.12 MAE; the gap to the
  baseline is 0.20.
- **The scattering assumption is ruled out by construction.** `cnn1d_raw` and `spec2d_raw` never
  touch the WST features — they consume the signal and learn their own representation. They land
  in the same band and lose by the same margin.
- **The parameter grid is ruled out by search depth.** 72 feature combinations per fold at
  10 GHz, chosen on inner folds only. Selection frequency is scattered across tilings, branches and
  families rather than concentrating on one configuration, and the per-band patterns even reverse
  between Exp A and Exp B.
- **Selection carries no information.** The composite selecting the worst family 16/16 times means
  inner-fold score is uncorrelated with outer performance. Where a real signal exists, selection
  normally transfers at least weakly.
- **No individual scattering path carries it either.** Exp E ablated all 742 groups one at a time;
  importances centre on zero, only 3% are consistently helpful across subjects, and the best single
  group is worth less than the margin by which the model loses to a constant. If the representation
  were hiding a usable signal in some subset of paths, this is the experiment that would have
  surfaced it.
- **Complete WST-order trajectories do not supply confirmatory path evidence.** The frozen
  trajectory diagnostic retained one descriptive 10 GHz order-1 pattern, but its Holm-adjusted
  p-value is 1.0 and 77 GHz has no primary candidate in either order. Persistence is useful for
  hypothesis generation; it does not rescue the predictive result.
- **Adding radar to the clock actively hurts.** Exp F's two primary contrasts are Holm-significant
  in both bands with the radar increment *positive* — the radar term does not merely fail to add
  information, it degrades a clock-only model.

What remains is the measurement problem itself. The study asks a cross-subject model to detect a
permittivity change corresponding to a fraction of a percent of body mass, from sessions that
differ in retained power by a factor of ten, against a target substantially explained by the time
of day, with sixteen subjects and no physiological reference beyond mass.

**The bounded claim.** In this cohort, at this dehydration range, with this hardware, the radar
features carry no recoverable information about fluid loss beyond what the time of day already
supplies. That is *not* evidence that radar-based hydration sensing is infeasible — it is evidence
that this study cannot demonstrate it, plus an account of why an earlier analysis of the same data
appeared to.

---

## 7. The leakage demonstration

Run deliberately, labelled unreportable, never a result about hydration. The same features,
models, data and sessions were re-evaluated with frames shuffled into five random folds instead of
held out by subject:

| 10 GHz arm | LOSO QWK | Frame-split QWK | Frame-split accuracy |
|---|---|---|---|
| regress-then-threshold | −0.212 | +0.405 | 0.307 |
| Frank–Hall | −0.197 | **+0.819** | **0.803** |

Identical pipeline, identical feature key, identical 73 sessions. Only the split changed. Frank–Hall
swings by more than a full unit of kappa.

The frame split trains on 5734 rows against LOSO's 73 sessions, but those frames come from the same
sessions and are near-duplicates sharing one label. That is pseudo-replication — the leakage
itself, not a data advantage. It does not reproduce the original 96–98% exactly; 80.3% is lower,
and the residual gap is plausibly a different classifier and feature pipeline. What is unambiguous
is the regime change.

---

## 8. Threats to validity a reviewer should press on

Written against the conclusion above, not in support of it.

- **Preprocessing could destroy the signal before any model sees it.** The chain is frozen and
  outside the CV loop — which protects against leakage but means a wrong constant is never
  discovered by the search. Band gate, chirp collapse and normalisation each discard information
  irreversibly. The strongest counter-evidence is that `cnn1d_raw` consumes the raw beat signal and
  also fails, but its preprocessing is not identical.
- **Session-level aggregation may be the wrong unit.** Frames within a session are pooled into
  session statistics. If the informative quantity were a within-session temporal dynamic rather
  than a session-level summary, this design would not see it.
- **Sixteen subjects is a hard ceiling on power.** With LOSO, every reported number rests on 16
  held-out estimates. A weak but real effect could plausibly hide here, and the CIs are wide enough
  to admit it.
- **The target is a proxy.** Mass change conflates fluid loss with other fluxes. A genuine
  radar–hydration relationship could be masked by target noise rather than absent.
- **All intervals are labelled conditional/exploratory.** They are conditional on the selection
  procedure and should not be read as unconditional coverage.
- **The software is extensively tested and author-reviewed, but not independently reviewed.** That
  was an explicit, recorded decision (A-M10-12), and it is precisely the gap this document exists
  to help close.

---

## 9. What I would check first, in this order

1. Confirm no frame from a held-out subject can reach training, for any session.
   — `tests/test_no_leakage.py`, byte-unchanged since milestone 7
2. Confirm every fitted transform — scaler, selector, ε, class weights — is fit inside the fold on
   training subjects only. — the per-fold fit audit emitted by each experiment
3. Re-derive the baseline. It is the load-bearing comparison; if it is wrong, everything inverts.
   — session-index-only predictor, K = 1, global-train-mean fallback
4. Check the preprocessing constants against the physics rather than the paper's wording.
   — band gate, chirp collapse, edge trim, normalisation
5. Verify the scattering geometry is read from the instantiated bank, not hard-coded.
   — 742 / 466 / 349 paths, pinned as regression values
6. Re-run one fold by hand and compare against the recorded selection table.
   — selection tables are byte-reproducible per machine
7. Read the frame-split demonstration as a protocol measurement, and confirm it is excluded from
   every reported artifact. — outputs tagged `frameSplit_leaked_exploratory` / `never_report`

---

## 10. Open items at the time of writing

Experiments E, F and G are complete and folded in above. **Experiment H's robustness bootstrap is
still running** — R = 200 replicates for each of Exp A, B and C in both bands, run as 16 shards
plus 6 merges. It asks whether the selection procedure itself is stable under resampling, which
bears directly on the "selection carries no information" observation in §6: if selection is
unstable under resampling, that is a second, independent symptom of the same underlying absence.

Nothing in H can change the sign of A–G. It quantifies how much of the observed selection
behaviour is reproducible, and its inconclusive verdict rule (`min_successful = 100`, never scaled
to fit a short run) is deliberately strict.

The assembly step then builds the final numerical tables from an explicit run map — every source
named on the command line, no globbing and no "latest" directory — and validates every registered
artifact's SHA-256 before writing anything.

**One reproducibility finding worth recording.** The 10 GHz feature pipeline is bit-reproducible on
a fixed CPU generation but *not* across generations, for a minority of sessions. This surfaced when
a store rebuild changed three sessions' bytes while leaving seventy identical. Every discrete
outcome — folds, candidates, selected keys, the selection table — was unaffected, and predictions
agreed to 2 × 10⁻¹⁴. It changes no result, but byte-level reproduction of this store requires
matching hardware, and that belongs in any reproducibility statement.

---

Prepared from the project's own frozen artifacts: the config-freeze tag, the per-session
feature-store fingerprints, the committed reference manifests and the run provenance records. Every
number above is traceable to a named run directory and commit. Where this document states an
opinion — §1 and §6 — it is labelled as one, and §8 is the argument against it.
