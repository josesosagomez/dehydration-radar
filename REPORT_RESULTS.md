# REPORT_RESULTS — where the signal isn't

A walkthrough of the pipeline, the frozen parameters, the negative results, and an argument
about which part of the study is actually responsible. Written for a reviewer who has not seen
this code.

- **Cohort** — 16 subjects, 5 sessions, 80 acquisitions
- **Bands** — 10 GHz FMCW (primary), 77 GHz (cross-band section only)
- **Protocol** — nested leave-one-subject-out, frozen before any result existed
- **Status at time of writing** — Experiments A–D complete; E/F/G completed hours before this
  was written and are **not yet folded in**; H's robustness bootstrap is running

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

---

## 6. Why the fault is unlikely to be the features or the model

Four independent lines converge, each capable of failing separately:

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

Experiments E (path-group ablation), F (confound sensitivity) and G (cross-band fusion) completed
on the full cohort hours before this was written, and their numbers are **not yet folded in**.
They bear directly on this document's argument:

- **E** ablates scattering path groups. If any group carried recoverable information, this is where
  it would show — the most direct test of the "it's the features" hypothesis argued against above.
- **F** tests sensitivity to available covariates. Heart rate is *not estimable* in this dataset
  and temperature and glucose stay uncontrolled, so F is a limited sensitivity result by
  construction, not an adjustment.
- **G** tests whether combining bands recovers anything. It is not required to beat 10 GHz, and
  reporting it as observed is the pre-registered stance.

Experiment H's robustness bootstrap (R = 200 per experiment and band) is running now. It asks
whether the selection procedure itself is stable under resampling — relevant to the "selection
carries no information" observation in §6.

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
