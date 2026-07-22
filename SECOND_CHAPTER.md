# SECOND_CHAPTER — thesis chapter material

Distilled, chapter-ready account, written **at each ROADMAP §7 milestone completion**
(not a duplicate of HISTORY.md). Capture the provenance of every choice so nothing is
unexplained: why each parameter value, why one processing choice over an alternative,
what a result means, and how it ties back to the paper's method and physics. MATLAB is
**not** mentioned here — all results are from Python.

> Status: **milestones 1–2 complete** (§0.1 and §1 below). Later sections fill in as
> their milestones close. The methodological framing (§0) is locked and is the spine of
> the Methods section.

## 0. Framing (locked at planning)

- **The reframe.** The original paper reports frame-level, subject-dependent 5-class
  accuracy (~96–98%), inflated by session-block leakage. This chapter replaces it with
  an honest pipeline whose headline is **regression of fluid loss (Δm%)** validated
  against measured body-mass change under **leave-one-subject-out** CV. It is a
  **feasibility study**: one objective reference (body mass), 16 subjects.
- **Why regression, not classification.** The physiological quantity is continuous
  (fluid loss as % of baseline mass); 5-class S0–S4 is a coarsening, kept as a secondary
  **ordinal** task with ordinal metrics only.
- **Why LOSO + session-level.** Hydration is a subject-level trait per session; frames
  within a session are highly correlated, so frame-level splitting leaks and per-frame
  training pseudo-replicates. Splits are at the subject level; the analysis unit is the
  session (one aggregated WST feature vector per subject·session).
- **The clock confound (honesty guardrail).** Fasting confounds hydration with time of
  day; the study does **not** causally isolate hydration. Experiment B (clock-decoupled,
  between-subject at fixed session) is the strongest available evidence, but is framed
  as reducing — not removing — the confound.

## 0.1 Evaluation protocol and its enforcement  *(milestone 1 — complete)*

This section supplies the Methods text asserting that the reported evaluation cannot
leak subject information, and the reproducibility apparatus behind every number in the
chapter.

### The evaluation protocol

**Nested leave-one-subject-out.** The outer loop holds out one subject at a time; the
held-out subject is touched only for final scoring. The inner loop — used for every
data-dependent choice (preprocessing branch, channel, WST tiling, log transform, range
gate, model family and its hyperparameters) — is a subject-grouped `GroupKFold` built
**exclusively from the outer-training subjects**, with an adaptive fold count
`min(5, n_train)`. On the full cohort this is 16 outer folds, each with 15 training
subjects and 5 inner folds.

**Why nested, and why grouped.** A single CV loop that both selects a configuration and
reports its score is optimistically biased, because the reported score has seen the
selection decision. Grouping at the subject level is required because frames within a
session — and sessions within a subject — are strongly correlated; a frame-level split
would place near-duplicate frames on both sides of the partition, which is the specific
flaw that inflated the original study's headline number.

**Small-sample rule.** Where cross-validation shrinks the training pool below **three
subjects**, the fold is reported as *non-selectable* rather than executed with a
degenerate split, so no reported number ever rests on a selection made from two
subjects. This is a stated protocol floor, enforced in configuration: values below three
are rejected at load time rather than merely discouraged.

**Analysis unit.** Splits, scoring, and all reported uncertainty are at subject/session
level; per-frame quantities are diagnostic only and never carry frame-level confidence
intervals.

### How the protocol is enforced rather than intended

Three structural properties make subject-level leakage detectable rather than a matter
of discipline:

1. **A single fold source.** One module constructs every train/validation/test
   partition used anywhere in the study. It splits *subjects* — it is handed one row per
   subject, so it is not capable of emitting a frame-level partition — and downstream
   code selects frames by filtering on the returned subject sets. It contains no random
   number generation, so folds are identical on every machine and every run.
2. **A mutation property test at both cross-validation levels.** Everything determined
   before scoring — the selected configuration, the inner-validation score table, every
   fitted transform and model parameter, and the training-set predictions — must be
   **bit-for-bit unchanged** when the held-out subject's features and labels are
   replaced with different values. Only that subject's own prediction and score may
   move. A second mutation, applied to a *training* subject, verifies the analogous
   property one level down: fits from folds in which that subject serves as
   inner-validation are unchanged, which detects the otherwise-invisible error of
   fitting on inner-training and inner-validation data together. Paired power checks
   confirm the test can fail: feature mutation must move the held-out prediction, and
   label mutation must move the score while leaving the prediction untouched.
3. **A fit audit.** Every fitted quantity is recorded together with the subject set it
   was estimated from, and the audit is checked by role: inner-selection fits come from
   exactly that fold's inner-training subjects, the final refit from exactly the full
   outer-training set, and no fitted quantity anywhere derives from the held-out
   subject.

Because bit-for-bit comparison is only meaningful against a deterministic reference,
the numeric work runs single-threaded with an explicitly deterministic linear solver,
and a precondition test establishes that two unmutated runs agree bitwise before any
mutation comparison is made. The enforcement suite was itself validated adversarially:
a deliberately leaky variant, in which the scaler is fitted on training and held-out
data together, fails the test as required.

### Data integrity established before any analysis

**Cohort.** 16 subjects × 5 sessions (08:00, 10:00, 12:00, 14:00, 16:00) = 80
acquisitions, 100 frames each, giving 8000 frames before quality control. Each frame is
a 534 fast-time × 20 chirp complex matrix.

**Target.** Fluid loss is the signed body-mass change relative to that subject's 08:00
baseline, Δm% = (m(s) − m(S0))/m(S0) × 100, negative for loss. Across the cohort it
spans 0 to −2.02%, with S0 identically zero by construction.

**Verification of the reference.** Body mass is the only objective hydration reference
available, so its transcription is checked rather than trusted. Each subject's computed
08:00→16:00 change is cross-checked two independent ways against separately recorded
quantities in the source record — a signed mass difference in kilograms and an
independently written percentage — with tolerances of 0.05 kg and 0.05 percentage
points. Both tolerances are conservative bounds derived from inspecting the source: most
masses are recorded to 0.1 kg, but one subject's are recorded to 0.05 kg, and the
percentage column is not consistently rounded to two decimals (at least one entry is
truncated). The largest observed discrepancy is 0.01 percentage points, so the tolerance
is roughly five times the worst observed case. All 16 subjects pass both checks.
Subject identity is parsed and asserted rather than inferred from row position, and any
additional subject record anywhere in the source is treated as an error.

**Structural completeness.** The frame index is built under six checks that fail loudly
and enumerate every offender: every subject × session cell present exactly once; no
duplicate acquisition; no unrecognised or stray file; a one-to-one correspondence
between acquisitions and reference records in both directions; per-file structural
validation; and frame counts read from each file rather than assumed. The real cohort
passes all six, yielding exactly 8000 indexed frames. Frame counts are read per file
because the session-eligibility rule at the quality-control stage is defined as a
fraction of each acquisition's actual frame count.

### Reproducibility apparatus

Every run writes a provenance record: SHA-256 hashes of all 80 acquisition files **and**
of the ground-truth source (so labels cannot change undetected), the fully resolved
configuration, the fold assignment with each subject's role, package versions, git
commit and working-tree state, device, seeds, and — on the compute cluster — the job
identifier. Runs are config-driven with a fixed seed and a fixed five-seed set for
stochastic models; file identities are recorded logically, relative to the configured
data root, so a record is identical whether produced on a workstation or the cluster.

*Provenance of the numbers in this section: HISTORY.md entries for milestone 1 steps
1–9 (2026-07-21), and `plans/MILESTONE_1_PLAN.md` §§2–5 for the specifications.*

## 1. Data & ground truth  *(milestone 2 — complete)*

**Cohort and acquisition.** Sixteen fasting subjects were recorded at five times of day
— 08:00, 10:00, 12:00, 14:00, 16:00, denoted S0–S4 — giving 80 acquisitions. Each
acquisition contains 100 frames, and each frame is a complex 534 fast-time × 20 chirp
matrix from a 10 GHz frequency-modulated continuous-wave radar (sampling rate
520 834 Hz, 500 MHz sweep bandwidth, 1024 µs chirp). The complete indexed dataset is
therefore 8000 frames before quality control.

**Target definition and sign.** The hydration reference is body mass. Fluid loss is
expressed as a signed percentage of each subject's own 08:00 baseline,

  Δm%(subject, s) = [m(subject, s) − m(subject, S0)] / m(subject, S0) × 100,

which is **negative for loss** and identically zero at S0 by construction. Normalising
to the subject's own baseline rather than using absolute mass change removes
between-subject body-size differences from the target, which matters at this sample
size. Across the cohort Δm% spans 0 to −2.02%, a narrow dynamic range that is itself a
central difficulty of the problem and is reflected in how results are reported.

**Why body mass, and its limits.** Short-term body-mass change is the only objective
hydration reference available for this cohort: no temperature, osmolality, or blood
measurements exist, and the environmental logs were lost. Mass change over a fasting
day conflates fluid loss with other mass fluxes, so it is treated as a *proxy* with a
known direction rather than a gold standard. This is the principal reason the study is
framed as a feasibility assessment.

**Verification of the reference.** Because the entire study rests on this one
reference, its transcription is verified rather than assumed. Each subject's computed
08:00→16:00 mass change is cross-checked against two independently recorded quantities
in the source record — a signed change in kilograms, and a separately written
percentage — with tolerances of 0.05 kg and 0.05 percentage points respectively. The
tolerances are conservative bounds established by inspecting the source rather than
assumed from nominal precision: most masses are recorded to 0.1 kg but one subject's
to 0.05 kg, and the percentage entries are not consistently rounded to two decimals
(at least one is truncated). The largest observed discrepancy across the cohort is
0.01 percentage points, roughly five times smaller than the tolerance. All sixteen
subjects pass both checks. Subject identity is parsed from the record and asserted to
be exactly the sixteen expected subjects, rather than inferred from position, and any
additional record anywhere in the source is treated as an error.

**Structural completeness.** The correspondence between acquisitions and reference
records is verified as a one-to-one mapping in both directions, together with checks
for duplicate, missing, unrecognised, and structurally invalid acquisitions, and with
frame counts read from each file rather than assumed constant. The cohort passes all
of these, giving exactly 8000 indexed frames. Frame counts are read per acquisition
because the session-eligibility rule applied after quality control is defined as a
fraction of each acquisition's own frame count.

### Quality control  *(milestone 2 — complete)*

**Why quality control is fixed before anything is learned.** The screens described here
are a fixed, per-frame measurable function of a single frame and a set of constants
chosen in advance. They are applied once, to the raw data, before any split is
constructed, and they take no part in cross-validation. This is a deliberate design
constraint rather than an implementation convenience: were any threshold estimated from
the data — a percentile of the cohort's energy distribution, say — the set of surviving
frames would depend on the whole dataset, and the held-out subject would have
influenced the training population through the back door. Every quantity a screen
computes is therefore derived from the frame it is judging: histogram limits from the
chirp being examined, an energy ratio normalised by that frame's own total power, a
dispersion statistic taken across that frame's own chirps. An automated test asserts
this directly, requiring a frame's verdict to be identical whether it is screened alone
or alongside arbitrary companion frames.

**The four screens, and what each is for.** A frame is examined on the **raw** signal,
before filtering.

1. *Non-finite samples.* Any frame containing a non-finite value is rejected. Because
   the remaining screens cannot be evaluated meaningfully on such data, they are
   skipped and their numeric diagnostics reported as unavailable rather than fabricated.
2. *Flatline / saturation.* A chirp whose magnitudes concentrate into a single narrow
   band is either saturated or dead. Each chirp's magnitudes are histogrammed into 200
   bins spanning that chirp's own observed range, and the chirp is flagged if any bin
   holds at least 25% of its 534 samples. A frame containing any flagged chirp is
   rejected. A chirp whose magnitude is constant to numerical precision is treated as
   flagged, which is the same verdict the limiting case of the histogram rule gives.
3. *In-band energy ratio.* This is the screen that asks whether the acquisition
   actually contains a return from the subject. The beat-frequency axis of a
   frequency-modulated continuous-wave radar maps linearly to range, at
   2·(B/T_chirp)/c ≈ 3257.5 Hz per metre for this instrument. A tapered 534-point
   spectrum is formed per chirp and averaged over the 20 chirps, and the fraction of
   non-negative-frequency power falling inside the range gate is compared against a
   floor of 0.30. Frames below it are rejected.
4. *Dispersion outliers.* The robust z-score of each chirp's root-mean-square
   amplitude, taken across the frame's own 20 chirps, is recorded and flagged above
   4.5. This is **diagnostic only and never rejects a frame** — see the interpretation
   note below.

A frame is rejected if it fails (1), (2) or (3). These three indicators are independent
and may fire together, so counts of individual reasons do not sum to the number of
rejections; the reported reconciliation is between passing frames, frames failing at
least one screen, and the total.

**The quality-control range gate is deliberately wider than the analysis gate, and
fixed.** The subject sits roughly one metre from the radar, and the analysis gate is
selected later, inside cross-validation, from physically motivated candidates. Quality
control instead uses a **single frozen 0.9–3.0 m gate for every candidate**. The reason
is structural: if screening used whichever gate a model happened to be using, then
changing a modelling hyperparameter would change which frames exist, and the population
being evaluated would move with the model — the evaluation would no longer be a fixed
target. The wider band is chosen so that a frame is never discarded for energy that a
wider candidate analysis gate would legitimately have used. The band is additionally
widened by one frequency bin (1000 Hz) on each side before the ratio is computed, so
that a target lying exactly at a gate edge is not penalised for the spectral spreading
the taper necessarily introduces; the 0.30 floor is defined together with this margin.
For this instrument the resulting band is 2931.7–9772.4 Hz, widened to
1931.7–10772.4 Hz, which is bins 2 through 11 of the non-negative half-spectrum.

**Session eligibility, and the decision not to impute.** A single acquisition is
retained only if at least half of its frames survive, the threshold being computed from
that acquisition's own frame count rather than an assumed constant. An acquisition
falling below it is dropped in its entirety, and **is then simply absent**: it is never
reconstructed from the subject's other sessions or from other subjects. Imputing it
would manufacture a hydration measurement that was never observed, and at this sample
size a small number of fabricated points could visibly move a result. The cost of this
choice is an unbalanced design, which the reporting absorbs by stating the effective
number of subjects and sessions alongside every figure.

**Survival on this cohort.** Of the 8000 indexed frames, **7330 (91.6%) pass**.
Every rejection came from the in-band energy screen: there were **no non-finite frames
and no flatlined chirps anywhere in the cohort**. Seven of the eighty acquisitions fell
below the eligibility threshold and were dropped —

| Subject | Session | Frames surviving |
|---|---|---|
| 1 | 08:00 | 35 / 100 |
| 1 | 16:00 | 1 / 100 |
| 3 | 10:00 | 37 / 100 |
| 4 | 14:00 | 35 / 100 |
| 5 | 14:00 | 39 / 100 |
| 6 | 08:00 | 0 / 100 |
| 16 | 10:00 | 15 / 100 |

— leaving **73 eligible acquisitions and 7168 analysable frames**. Because every
subject retained at least one eligible session, **all sixteen subjects remain
evaluable**, and the outer cross-validation is the full sixteen folds. The failures are
therefore not corrupted recordings but acquisitions in which the returned energy simply
does not lie in the target range interval, consistent with a subject seated outside the
gate or an acquisition begun before the subject was in position. Their distribution is
reported rather than absorbed, since quality-control failure could itself correlate
with hydration state or with acquisition conditions.

**Interpreting the dispersion diagnostic.** The robust dispersion flag fires on 34% of
frames, concentrated in a few acquisitions. This is a property of the statistic rather
than evidence of widespread anomaly: with only 20 chirps per frame, all recorded within
a few tens of milliseconds, the chirp-to-chirp spread is very small, the median absolute
deviation is correspondingly small, and the normalised score is therefore extremely
sensitive to mild variation. The quantity is retained because it usefully describes
where chirp-to-chirp variability is non-uniform, but it is reported as a descriptor and
never used to discard data — which is precisely why it does not enter the rejection
rule.

**Cross-band note.** The 77 GHz recordings, used only in the later cross-band section,
were audited at this stage on a single acquisition to establish their layout before any
processing choices depending on it were locked. The dimension ordering was confirmed on
raw data by a signal-domain test rather than by shape alone — the two ambiguous axes are
both of length 256 — by requiring range structure to appear on the proposed fast-time
axis and near-stationary content on the proposed slow-time axis. The test was run
before clutter removal, since subtracting the static component would have suppressed
exactly the stationary content that identifies the slow-time axis. Both expectations
were met decisively, and the alternative assignment was contradicted. The audit also
established that these recordings are stored as **real-valued** samples, so the complex
representation used by the cross-band feature chain arises at the range transform and
not in the raw data.

## 2. Preprocessing  *(milestone 3 — complete)*

Between a quality-controlled frame and the feature extractor sits a short, fixed
sequence: restrict the signal to the range band where the subject is, collapse the
frame's repeated chirps into a single trace, discard the filter's edge transients, and
normalise what remains. Every step is a deterministic function of one frame and a
frozen constant. Nothing is estimated from a population, so no quantity computed here
can carry information between subjects, and the whole sequence sits outside the
cross-validation loop by construction rather than by convention.

### Resolving an ambiguity in the published description

The published account of this work describes preprocessing as a window, a range
transform, and a bandpass filter. That ordering is not what the analysis actually
requires, and taken literally it is self-defeating. In a frequency-modulated
continuous-wave radar the beat frequency of a return is proportional to the target's
range, so restricting the signal to a band of beat frequencies **is** the range gate:
once the filter is expressed in the time domain there is no transform to take. A window
serves only to suppress spectral leakage, which is a property of a finite Fourier
transform; applying one before a time-domain recursive filter would taper genuine
signal energy at the ends of the chirp while suppressing nothing. The sequence adopted
here therefore applies a window **only where a transform is actually taken** — the
quality-control energy screen, the peak detection described below, and the
spectrogram-based comparison model — and filters in the time domain everywhere else.
The departure from the published wording is deliberate and is recorded as such.

### The band gate

The gate is a fourth-order Butterworth bandpass, realised as second-order sections and
applied **forward and backward** along fast time, independently per chirp, to the
complex signal. Three properties of that sentence carry weight.

*Fourth order, in second-order sections.* The band is narrow relative to the sampling
rate — roughly 3.3 kHz within a 520.8 kHz band — so the normalised cutoffs are of order
0.01. A direct transfer-function realisation loses its poles to floating-point error at
that scale; the factored form is a numerical necessity, not a stylistic preference, and
pole stability is asserted by test rather than assumed.

*Forward and backward.* Running the filter in both directions cancels its phase
response exactly, which matters because the features extracted downstream are sensitive
to the temporal position of structure within the trace. A single causal pass would
displace a mid-band pulse in this configuration by roughly 131 samples — a quarter of
the record — so the property is verified against that alternative rather than merely
asserted. The cost is that the magnitude response is applied twice, so the nominal
half-power edges of the design sit at a quarter power in use; the reported band edges
are stated with that in mind.

*On the complex signal.* Real and imaginary parts are filtered separately through the
same real filter, which is equivalent to filtering the complex signal and is written
out explicitly so the equivalence is visible rather than delegated.

The padding used to initialise the recursion is fixed by the code rather than inherited
from a library default, so the treatment of the record's ends cannot change beneath the
analysis when a dependency is upgraded.

**A finite record does not behave like the design.** The frequency response of a filter
describes its steady state; a 534-sample record of a narrowband filter is dominated by
its transient. Measured on this configuration, a mid-band tone retains 76% of its
energy over the full chirp and 83% after edge trimming, and an out-of-band tone is
suppressed by 17 dB and 20 dB respectively — not the far larger figures the design
response would suggest. Both facts are recorded as measurements and pinned by
regression tests. **No filter parameter was adjusted to make the realised behaviour
approach the idealised one**; the discrepancy is a property of short records, and the
improvement from edge trimming is itself the empirical case for that step.

### Two range gates, deliberately different

Quality control screens frames on a **wider** band (0.9–3.0 m) than the band the model
uses (1–2 m by default). The separation is what makes the analysis population
independent of a modelling choice: were the screen tied to the model's gate, changing
that gate would change which frames exist, and a hyperparameter would silently alter
the dataset. The wider screening band also avoids rejecting a frame for energy that a
wider candidate gate would legitimately use. The relationship is enforced — a model
gate that reached outside the screening band would be rejected before any data were
read.

### Collapsing the chirps

Each frame contains twenty repetitions of the same measurement, and two ways of
combining them are carried forward as alternatives, resolved later by validation on
training subjects only.

The first simply averages them. For a stationary target the returns add coherently
while noise averages down, and nothing about the range profile is presumed.

The second isolates the dominant return. The frame's own chirps are transformed, their
power spectra averaged, and the strongest bin **within the range gate** located; a
narrow tapered neighbourhood of that bin is retained and the rest discarded before
transforming back. Two details are stated because they are easy to get wrong. The
window used to *find* the peak is not applied to the signal that is *reconstructed* —
detection and reconstruction are separate operations, and tapering twice would distort
the result. And the retained neighbourhood is symmetric about the peak with **full
weight on the peak itself**, tapering only on its shoulders; a taper that reached zero
at the edges of the retained block would annihilate the very bin the procedure exists
to isolate.

The detection region is the model's range band with **no tolerance margin**. The margin
used by the quality-control screen exists to keep a frame from being rejected over
leakage at a boundary; it has no bearing on where a peak may legitimately be found, and
importing it here would widen the search for no reason. At the default configuration
the region is three frequency bins.

One degenerate case is defined rather than assumed away. If the detection region
carries no power, the search returns its first bin and the reconstruction proceeds from
there. It is tempting to state that the output is then zero, but that is **false**:
detection is windowed and reconstruction is not, and the windowing operation can cancel
the detection region exactly while leaving the reconstructed bins non-zero. The
behaviour is therefore specified as what it is, and such a frame is flagged in the
diagnostics rather than silently described by a fabricated value.

### Trimming, and why it comes last

Edge trimming removes 32 samples from each end, leaving 470. It is applied **after**
the chirps are combined, not before: the peak-isolation branch transforms the full
chirp, and trimming first would change the frequency grid the search runs on — at 470
samples the bin spacing is 1108 Hz rather than 975 Hz, and the detection region shifts
accordingly. The ordering is verified by a test that would fail if the two steps were
exchanged. An over-large trim is refused rather than quietly reduced, so a mis-set
value surfaces immediately instead of producing a shorter signal than intended.

### Normalisation

Each trace is standardised against its own statistics: centred on its median and scaled
by its median absolute deviation, with the customary consistency factor that makes the
scale comparable to a standard deviation. Using a robust centre with a robust scale is
a correction — pairing a mean with a median absolute deviation, as an earlier
implementation did, mixes a statistic that an outlier moves with one that it does not,
and the combination is neither robust nor interpretable. A single extreme sample shifts
the bulk of a normalised trace by under 0.05 under this form and by more than an order
of magnitude more under mean-and-standard-deviation scaling.

Because each trace is normalised by its own statistics, normalisation is not a fitted
transform: nothing is estimated on one set of frames and applied to another. This is
what allows a step that would ordinarily have to live inside the cross-validation loop
to sit safely outside it. The magnitude channel and the two quadrature channels are
each normalised separately, from their own statistics.

### What the sequence does to this dataset

Run over all 7168 analysable frames of the 73 eligible sessions, the sequence produces
finite output of the expected length for every frame and every combination of the
alternatives above. Three measurements characterise it.

**The subject sits at 1.50–1.80 m, not at 1 m.** The dominant return falls in one of
two adjacent frequency bins in 72 of the 73 sessions, corresponding to 1.50 m and
1.80 m. The default 1–2 m gate was chosen in advance from the reasoning that a seated
subject would be about a metre from the antenna; that reasoning was **wrong by roughly
half a metre**, and the gate happens to be well chosen anyway — the measured range sits
near its centre. The gate was not adjusted after this was observed, and the
justification recorded here is the measurement rather than the original assumption.

**The dominant return is genuinely dominant.** That a detected peak lies inside the
search region is true by construction and therefore evidence of nothing. The
informative quantity is how much of the region's power the peak actually holds: with
three bins in the region, an unstructured spectrum would place a third of the power in
the strongest bin, and the observed median is **0.512**, with no session below 0.410.
The premise of the peak-isolation branch — that there is a single dominant return to
isolate — holds on this data. Detection is also stable within a session: in 45 of 73
sessions every frame agrees on the same bin, and no session spans more than three.

**Band-gate retention varies widely between sessions**, with a median of 0.41 and a
range from 0.06 to 0.64. The low end reflects overall signal level rather than a
mismatched band — the weakest sessions still concentrate their remaining power in the
gate — but the spread is recorded here because it describes how unequal these
recordings are before any feature is computed.

One quantity is deliberately *not* over-read. The fraction of spectral power falling
inside the range gate is measured after filtering, where it cannot be low: it describes
the selectivity of the filter, not the presence of a target. It is reported for
completeness and carries no evidential weight.

### How correctness was established

No numerical comparison against the earlier implementation is made anywhere. Instead
the sequence is pinned by properties that are true of a correct implementation and
false of plausible incorrect ones: the designed filter passes its band and stops
outside it; the two-directional application has zero group delay, demonstrated against
a single-pass alternative that does not; the batched application over a whole session
is bit-identical to filtering each chirp alone; the detection region matches
independently computed arithmetic; the retained mask has the stated weights, keeps the
peak at full weight, and refuses to expand until it keeps everything; trimming happens
after combination and cannot be exchanged with it; normalisation matches its formula
exactly, including the convention for the divisor; a frame's output is identical
whether it is processed alone or beside arbitrary companions; and repeated runs are
bit-identical. The last two together are what make the claim of a per-frame,
population-independent sequence executable rather than asserted.

## 3. WST features  *(fill at milestone 4)*

## 4. Fluid-loss regression — Experiment A  *(fill at milestone 5–6)*

## 5. Clock-decoupling — Experiment B  *(fill at milestone 6)*

## 6. Ordinal classification & baselines — Experiments C, D  *(fill at milestone 7–8)*

## 7. Fusion, interpretability, confounds, statistics — G, E, F, H  *(fill at milestone 8)*

## Provenance index

For each locked parameter, the "where did this number come from?" answer lives in
`plans/implementation_plan.md` and, once implemented, in the corresponding HISTORY.md
entry. Key ones to carry into the chapter: robust-standardization form (median/MAD),
range gate, WST tiling Q + invariance scales and their kymatio (J,T) mapping, the
order-aware log transform, QC thresholds, session-eligibility rule, and the 77 GHz
Doppler/I-Q/per-Rx feature choice.

**Settled at milestone 3** (detail in HISTORY.md, 2026-07-23 entries): filter order 4
and its second-order-section realisation; forward-backward application and the
consequent quarter-power band edges; the explicit padding length; the absence of a
window in the primary path; the separation of the screening gate from the model gate;
the peak-detection region as the model gate without margin; the retained-mask weights
and their full weight on the peak; edge trimming after combination; the robust
standardisation form and the placement of its numerical guard; and the measured
1.50–1.80 m target range that now justifies the default gate in place of the original
seating assumption.
