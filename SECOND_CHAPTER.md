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

## 3. WST features  *(milestone 4 — complete)*

The preprocessed trace — a normalised complex signal of 470 samples, taken either as its
magnitude or as its two real and imaginary channels — is turned into a feature vector by a
wavelet scattering transform. Scattering was chosen for the same reason the original work
chose it: it produces a representation that is stable to small time shifts and to smooth
deformations while preserving the high-frequency structure that a plain low-pass average
would discard, and it does so with a fixed filter bank rather than anything learned from
the data. That last property is what matters most for this study. The transform contains
no fitted quantity, so like the preprocessing before it, it sits outside the
cross-validation loop by construction: the same three design questions the published work
left implicit — how finely to tile frequency, how much time-invariance to impose, and how
to summarise the result — are made explicit here, frozen before any model is fit, and
where they are genuinely choices they are resolved inside the inner cross-validation rather
than on the held-out subjects.

### From an invariance in milliseconds to a filter bank in samples

The reference expresses each tiling as a pair of quality factors and an invariance scale
in milliseconds; the Python library takes an integer octave count and an averaging support
in samples. The translation is deterministic and is recorded rather than assumed. An
invariance scale in milliseconds becomes an averaging support of `round(scale · f_s)`
samples — 0.20, 0.30 and 0.40 ms at the 520.834 kHz sampling rate give 104, 156 and 208
samples, each realising the requested scale to better than two parts in a thousand — and
the octave count is the smallest integer whose largest wavelet scale covers that support,
`ceil(log2 T)`, giving seven octaves for the first tiling and eight for the other two. The
three tilings retain the reference quality factors (10 and 4, 8 and 2, 6 and 2) and keep
scattering orders zero, one and two.

Everything downstream of these two numbers — how much the signal is padded, how many
scattering paths result, how long the averaged output is — is a property of the
instantiated filter bank, not a formula to be guessed, and is read back from the bank and
recorded. This matters because the natural guess is wrong in an instructive way: a
470-sample signal is not a power of two, and the library pads it symmetrically by 277
samples each side to a length of 1024, which happens to be a power of two here but is not
guaranteed to be and is not computed as one. The measured geometry is 742 paths of length
7 for the first tiling and 466 and 349 paths of length 3 for the other two. These numbers
are pinned as regression values, so a future change in the library that altered padding or
path structure would be caught rather than silently changing every feature.

At these signal lengths the library warns that the support is too small to fully avoid
edge effects, and it does so for all three tilings, not only the shortest. The response is
deliberate and is fixed before any measurement: the native padding is accepted for all
three, because the trace length and the tilings are both frozen upstream and no mitigation
is on the table that would not itself be a data-driven change. The size of the effect is
measured for the record — zeroing the outermost 32 samples of a test signal moves the
averaged coefficients by roughly two-thirds of their norm, confirming the warning describes
something real — but that measurement gates nothing; it is descriptive, and the decision it
accompanies was made in advance.

### An order-aware logarithm

A logarithm is optionally applied to compress the dynamic range of the coefficients, and
whether it helps is left to the inner cross-validation. What "applying the logarithm" means
is fixed here, and it is not uniform across the scattering orders. The first- and
second-order coefficients are moduli and therefore non-negative, so `log(S + ε)` is
well-defined; the zeroth-order coefficient is a signed low-pass of an already
mean-centred signal and is routinely negative, so logging it would be undefined. The
zeroth order is therefore always left linear and only orders one and two are logged. A test
constructs a deliberately negative zeroth-order coefficient and confirms it passes through
finite and unchanged — the exact failure the rule exists to prevent.

The floor ε is fixed at one part in a million, and the cohort measurement turned the stated
reason for that value into something more honest. The original expectation was that the
coefficients live on a scale of order one, having come from a normalised signal, so that a
floor of a millionth would be a pure numerical guard. The standardised *input* is indeed of
order one, but the scattering coefficients are not: measured across the cohort, the
first-order coefficients sit near a thousandth and the second-order coefficients near a
millionth. Against the first order the floor is three decades down and genuinely negligible;
against the second order it is between a tenth and two-thirds of the coefficient scale, so
for the many second-order paths below that median the floor dominates and the logarithm
compresses them toward a constant. This is a real property of the representation, not a
defect to be tuned away, and it is treated as one: the floor stays frozen at the declared
value, and because the logarithm is a selectable option rather than a fixed step, the
cross-validation is free to decline it fold by fold if the second-order compression costs
more than the range compression buys. The measurement is what makes the earlier assumption
correctable; the parameter is left exactly where it was declared.

### Summarising a session

Two families of summary are produced from the scattering paths. The first pools each path
over time into low-order statistics — its mean and, where a segment is long enough to have
one, its standard deviation, taken over the whole path and over each half. The second keeps
the raw averaged series and is reserved for the network baselines and diagnostics; it is
never a classical feature on its own. The pooled family carries a subtlety that only became
visible once the output lengths were measured. Two of the three tilings produce an averaged
series of length three, whose first half is a single sample; a standard deviation over one
sample is identically zero, so pooling those tilings by the literal "mean and standard
deviation over each half" would ship a column that is zero for every path of every frame of
every subject — a structurally dead feature that a later fitting step would have to discover
and handle. The rule adopted instead makes a segment contribute its standard deviation only
when it has at least two samples, which depends only on the fixed output length and so
introduces no dependence on the data. This produces five statistics per path for the two
short tilings and six for the long one, and because it departs from the literal description
in the reference it is recorded as a deliberate departure rather than a silent
simplification. The exact column layout, including which standard deviations are present, is
emitted as machine-readable metadata so that the modelling and interpretation stages consume
the true layout rather than reconstructing it.

The analysis unit is the session, not the frame. A session's surviving frames are each
turned into a pooled vector and then combined into one vector per session by concatenating
their per-frame mean and per-frame median. This is the point at which the many correlated
frames of a session stop being treated as independent observations: the label is one
weight measurement per session, and the features are matched to it one-to-one, so a session
with more surviving frames cannot come to dominate. The combination is a fixed pair of
statistics, not a further choice to be searched.

### Two implementations, one set of reported numbers

The transform is available through two computational backends, and the study commits to one
of them for every reported number. The reason is not correctness but reproducibility: the
two backends agree only to a tolerance, not to the last bit, so allowing either to produce a
reported artifact would make two runs of the same pipeline disagree in their last digits for
no scientific reason. The numerical backend is therefore canonical for all reported WST
features, and the alternative backend's role is to validate it and to serve throughput-only
work that never reaches a reported table. Establishing that the alternative is a faithful
stand-in surfaced a concrete constraint: its filter bank runs in single precision and cannot
be coerced to double, so the two backends can only ever be compared as double against
single. That comparison passes the strict tolerance regardless — the largest relative
discrepancy across all three tilings, both channels, and both logarithm states stays a small
fraction of the allowed bound, on synthetic and on real frames alike — so no loosening of
the tolerance was needed, and the possibility of one was declined rather than exercised. The
agreement criterion itself is a single frozen formula with an absolute floor for
coefficients near zero, applied identically wherever the two backends are compared, so the
gate cannot drift between one test and another.

### What the transform costs, and that it never fails

Run once over the whole eligible cohort, the transform and its pooling produce a finite
feature vector for every one of the 73 sessions, in every combination of reduction, channel,
tiling and logarithm state — the finiteness that each unit test asserts on a single frame
holds at cohort scale. The full pass takes about twelve minutes on one machine, which fixes
the cost of recomputing features inside a model search and confirms that no cached
intermediate is needed. That figure is only real because a first attempt was five times
slower and the cause was found rather than tolerated: the per-path pooling loop, not the
transform, was dominating, and rewriting it to work over all paths at once restored the
transform to its rightful place as the expense. The measured feature dimensions, the pre-log
coefficient scales, and the cohort-wide finiteness are recorded alongside the run so that the
representation the models receive is fully characterised before any model is fit.

### How correctness was established

As with the preprocessing, nothing here is checked against the earlier implementation.
Correctness rests on properties that a correct transform has and plausible mistakes do not:
the millisecond-to-sample-to-octave arithmetic is verified against independent computation;
the padding, path count and output length are asserted at their measured values so a library
change cannot pass unnoticed; the edge-effect warning is required to appear for every tiling,
so silencing it would fail the test rather than hide the issue; every combination of channel,
order and logarithm state is shown to be finite, and the one case that could produce a
non-finite value — logging a negative zeroth-order coefficient — is constructed on purpose
and shown not to; a small time shift is shown to move the averaged coefficients by less than
half of the shift it was given, a bound argued before the measurement rather than fitted to
it; the batched transform is bit-identical to transforming each frame alone; the two backends
agree to the frozen tolerance; the pooled layout matches a hand computation element by
element for both the five- and six-statistic cases, and a deliberately reordered reference is
shown to fail that comparison; and the session vector is the concatenation of mean and median
it claims to be. The two independence properties — batched equals single-frame, and the whole
extraction is deterministic under a fixed seed — are again what make the claim of an unfitted,
per-session, population-independent representation executable rather than merely asserted.

### 10 GHz session-quality audit  *(milestone 5a — complete)*

The WST features are useful only when the recording itself is usable. A separate target-free
audit therefore examined all 80 10 GHz subject-session files. It reports a component card—not
one weighted “quality score”—covering frame availability and existing QC failures, stored-index
block coverage, signal level and preprocessing diagnostics, and within-session WST repeatability.
This avoids inventing a magic number whose weights would be hard to defend.

Existing QC eligibility is a hard, frozen condition: seven sessions failed it and remain visible
as `INELIGIBLE_EXISTING_QC`. Block coverage and WST repeatability are soft review signals: two
sessions were labelled `REVIEW_BLOCK_COVERAGE`, while 71 were `REPEATABILITY_ANALYSABLE`. The
WST check uses the frozen preprocessing and all three tilings, with separate within-path-shape
and across-path-energy-composition views. It fits no population transform, uses no body mass,
and removes no sample. Body mass is used only afterward for a separate recorded-equal-mass
comparison, which is a sanity aid—not a quality label or training rule.

The clean run used commit `bc5832b582e2d705c97bf7f445ba48fd38a4b2d3` with `dirty=false` and wrote
80 session-card rows, 400 block rows, 2,880 WST rows, 25,920 relative-comparison rows, and 288
recorded-equal-mass rows. Subject 10 passed 100/100 frames in every session: 12pm was strong and
stable, while 4pm was the clearest review candidate. Subject 10's equal-mass pair was 8am–10am. These are
recording-consistency observations, not proof of hydration validity; a review flag is not proof
of equipment failure. Verification passed 32 focused tests with one real-data test skipped, the
mass-quarantine integration test, and 383 critical tests with 10 skipped. The 317 frozen files
were unchanged. Earlier pre-fix outputs remain in
`archive/results/quality_10ghz_dirty_provenance_20260829/`.

## 4. The 77 GHz front-end  *(milestone 5 — complete)*

The 77 GHz Inras band was promoted from a fusion-only afterthought to a full parallel primary
arm, so the same honest pipeline built for 10 GHz — loader, QC, preprocessing, WST features —
had to exist for band 2 before the config-freeze gate could pin its numbers. The front-end
mirrors milestones 1–4 for the second band and stops at the same boundary (features and cohort
diagnostics; no modelling). This section records where each 77 GHz number came from and, most
importantly, a data artifact discovered here that a survival-tuned QC rule would have hidden.

### Different physics, not overridden defaults

The 77 GHz radar is a genuinely different instrument: a 500 kHz slow-time sample rate, a 2 GHz
sweep over a 512 µs chirp, and a raw cube of 16 receivers × 256 chirps × 256 fast-time samples
per frame, stored real-valued (I/Q arises only after the range FFT). Because these are different
physics rather than variations of the 10 GHz constants, the configuration keeps three parallel
frozen sections (`qc77`, `preprocess77`, `wst77`) rather than nesting band-2 overrides — each
band then owns a complete canonical specification, so a run's YAML is a full record and the
artifact guard can compare a config against a single frozen default. The two size-256 axes
(fast-time and chirps) are indistinguishable by shape, which is not a cosmetic risk: a silent
fast↔chirp interchange would pass every shape assertion and invert range and Doppler. A semantic
axis check — range-gate energy must concentrate along the assumed fast axis and near-zero-Doppler
energy along the assumed chirp axis, on the raw pre-clutter-removal cube — certifies the mapping
per file, fails closed on anything short of acceptance, and its certificate is keyed to a hash of
exactly the axis-relevant constants so a paths-only cluster overlay leaves it valid.

### The chain, and why the counter never reaches the features

The executable chain is the paper's Doppler method made concrete: clutter removal by subtracting
each frame's own per-fast-bin mean over chirps (a within-frame operation, no cross-frame
statistic), a fast-time Butterworth bandpass over the 2–4 m beat band, a symmetric Hann window,
a 256-point range FFT, and a crop to range bins 27–53. Then, per frame, the 16 × 27 = 432 complex
slow-time series are each split into real and imaginary channels, robust-standardised from their
own median/MAD, scattered as one batch, averaged over the range bins, fused across receivers
(mean primary, median a labelled secondary), order-aware-logged on the fused tensor, pooled, and
aggregated to one vector per session (frame mean concatenated with frame median). The scattering
geometry was measured, not assumed: at the PRF of 1953.125 Hz the three Doppler tilings realise
averaging supports of 39, 78 and 117 samples (J = 6, 7, 7), pad to 512, and realise the requested
millisecond invariances to within 0.16 %; the border-effect warning fires and is asserted rather
than silenced.

### An artifact the mechanism analysis caught — and a survival rule would not have

The single most consequential finding of this milestone concerns the flatline QC screen. Ported
literally from the 10 GHz screen, it flagged 7 of 10 frames in the milestone-2 single-file audit,
which — left unexamined — would have collapsed 77 GHz eligibility. The rule was **not** re-tuned
to restore survival; that would be cohort-level leakage. Instead the mechanism was traced. The
flag tracked the per-frame maximum magnitude, which grew as exactly 256 × frame-index, while the
traces themselves stayed richly varied (≈ 230 distinct magnitude levels; no dead channels). The
cause is an **embedded frame counter in range bin 0**: the first fast-time sample of every trace
carries a per-chirp counter (≈ 256 × frame, resetting periodically), universal across files and
20–90× the genuine echo. That one outlier stretched the per-trace histogram range so the real
samples piled into the first bins and tripped the 25 %-in-one-bin rule. Crucially, both Hann
windows are zero at index 0 and the 2–4 m gate excludes range bin 0, so the counter never reaches
the WST features, the in-band ratio, or the axis check — it corrupted only the raw-magnitude
flatline screen, which is exactly why the audit's in-band ratios and axis verdict were healthy
while flatline false-fired. The correction, specified from this mechanism (not from survival) and
frozen before the cohort run, simply excludes range bin 0 from the flatline screen and keeps the
proven rule on the echo samples; a genuinely dead channel is still caught by the degenerate-spread
branch. On the audited file the corrected screen flags none of frame 9's 4096 traces (the old
rule flagged 3981) and the session survives at 100 %. The general lesson for the chapter: a QC
screen inherited across bands must be validated against the new band's data-generating quirks,
and a false positive is fixed by understanding it, never by tuning a threshold until the numbers
look acceptable.

### Leakage safety and reproducibility

The whole front-end is a deterministic per-frame function of one frame and frozen constants:
nothing is fitted, so nothing enters the cross-validation loop and the no-leakage test is
untouched. The two data-independent log branches (off, on + frozen ε) are realised end to end;
the third (on + tuned ε) is a train-only branch whose per-order ε the milestone-7 harness will
fit and re-extract with — this milestone provides and tests the application path but never
computes a data-dependent ε. numpy backs every reported feature; the torch frontend is admitted
only after a cross-backend agreement test passes over both log states. The heavy cohort passes
run on IBEX as CPU batch jobs (a single QC job; a job array over the 80 cells for features, each
task re-certifying its own file's axis and fingerprinting its shard), differing from the local
smoke only by a paths-only overlay — the same code, as the compute policy requires.

## 5. The config freeze — the whole protocol, before any result  *(milestone 6 — complete)*

The evaluation protocol (§0.1) makes leakage impossible *within* a single experiment: no
held-out subject's frame reaches training, and every fitted quantity is estimated inside the
fold. But the study runs seven experiments (A–G) on one 16-subject cohort. A subtler leak sits
between them: if a modelling choice — a search space, a baseline's architecture, an ordinal
family, a statistical test — is decided *after* looking at Experiment A's per-subject
predictions, that choice has been informed by subjects who will later serve as held-out test
subjects in B–G. The information travels across experiments even though no single experiment
violates its own protocol. The remedy is chronological: commit the complete A–G design, for
both radar bands, to versioned configuration and to git **before any outer-fold result from any
experiment is inspected**. This milestone is that commitment.

### What "frozen" means mechanically

The design is not merely documented; it is a set of validated configuration objects the code
refuses to run outside of. Eleven dataclasses hold the two band-specific search spaces, the
per-family hyperparameter grids, the baseline specifications for both bands, the five
downstream experiment designs (B, C, E, F, G), the full statistical protocol, and a small set
of protocol constants. Each is a *frozen record*: a run's configuration file may restate a
value at its approved default — so the file is a complete, self-describing record — but may not
change it, and a change is a load-time error, not a silent override. This is stricter than the
earlier preprocessing and WST sections, which legitimately carry live inner-CV axes and
pre-declared ablations; nothing in the milestone-6 sections is a knob, because every one of
their values was fixed before results existed.

A second guard runs at modelling time. The single canonical feature artifact is written only
under one exact preprocessing/WST specification, enforced since earlier milestones. But model
*selection* legitimately explores approved alternatives — for instance either of the two range
gates — so a separate guard validates that a run stays inside the *whitelist* rather than at a
single canonical point: every axis value actually used for a fit (which reduction, channel,
tiling, log branch, range gate, model family) is checked for membership in the frozen search
space. Crucially, those axis values are call arguments to the extraction and modelling
functions, not stored configuration fields, so a configuration-only check could never see
them; the guard therefore takes the active per-fit protocol as an explicit argument and
validates it against the same whitelist, regardless of how the caller produced it. This is
defence in depth over the harness's own enumeration, not a substitute for it.

### The three decisions the freeze forced into the open

A design freeze is only honest if the choices being frozen are themselves free of hindsight.
Three had to be settled here on grounds that do not depend on cohort performance, and each was
resolved by owner decision before the freeze closed.

*The third log branch.* The WST log transform carries an optional data-adaptive variant whose
per-order ε is fit on the training fold. An earlier plan proposed to admit this branch only if
a preliminary check showed the second-order scattering coefficients carried predictive value —
but that check, as specifiable, is a full leave-one-subject-out comparison across the whole
cohort, whose aggregate result would then set a global modelling option applied to the same
subjects later held out. That is exactly the cross-experiment leak the freeze exists to
prevent, and it echoes the milestone-5 precedent that leakage-sensitive choices are decided on
mechanism, never on cohort-wide performance. The check was retracted; the branch is instead an
unconditional inner-CV candidate for both bands, leakage-safe by construction and a strictly
smaller change to the search space than standing up a separate cohort-wide test would have
been.

*The ordinal comparison model.* The secondary five-class task specifies a proportional-odds /
cumulative-link regressor with per-fold inverse-frequency class weighting. The natural library
implementation, verified directly against the pinned version, has no mechanism for observation
weights at all — its fitting routine simply does not accept them — so it cannot satisfy the
weighting requirement. Rather than quietly drop the weighting (which the protocol requires) or
quietly swap in a different model (which would change what is being compared without saying
so), the substitution was made explicit: the comparison family is a Frank–Hall ordinal
decomposition over binary logistic regressions, which does accept the required weights, and
which is recorded as a deliberate, documented departure from a literal cumulative-link model
rather than presented as one.

*The 77 GHz physics baseline.* Promoting the 77 GHz arm to a full parallel set of experiments
required a physics baseline analogous to the 10 GHz reflected-power ratio. A first proposal
split the Doppler spectrum at a physiologically motivated 2 Hz boundary — but the system's
Doppler resolution, set by the chirp count and pulse-repetition frequency, is coarser than that
boundary, so no such cut is representable and no individual physiological rate is resolvable in
the recorded aperture at all. The baseline was redefined honestly as a static-versus-any-motion
energy ratio (the zero-Doppler bin against all resolvable motion bins), and the chapter states
that limitation plainly rather than letting the "physics baseline" label imply a specificity
the measurement cannot support.

### Provenance of the frozen values

Every value that the main design left unstated was proposed on non-performance grounds —
standard small hyperparameter grids, conventional small-dataset training defaults, an arbitrary
non-tuned anchor for the staged feature search — and confirmed by owner decision as a group,
with none derived from running anything against the cohort. The one piece of executable logic
the milestone adds is the model-selection tie-break, made concrete as a pure comparison over
already-computed scores: lower validation error, then a frozen simplicity ranking over the
model families, then smaller feature dimension, then lower inner-fold variance, with
non-evaluable candidates filtered before the comparison so that an undefined score can never
decide a winner by accident. It fits nothing and reads no data; the fitting it will later serve
belongs to the harness milestone.

### Correctness

The milestone introduces no computation on cohort data, so its correctness is entirely a matter
of the frozen configuration loading, rejecting changes, and round-tripping into the provenance
record, and of the two guards accepting exactly the approved space and no more. Fifty tests
establish this: that each frozen section loads at its pinned values and rejects any change
(including the subtle case of a boolean written as an integer); that the two band search spaces
cannot express each other's candidates, so a 10 GHz-only option cannot leak onto the 77 GHz
arm; that every family's grid fits under the shared search budget; that the tie-break honours
its ordering and excludes undefined scores; and that the modelling guard admits both approved
range gates while rejecting an out-of-whitelist gate or a mistyped call-time axis, all without
weakening the strict single-point guard on the artifact path. The no-leakage test remains
byte-for-byte unchanged, as it must: this milestone adds no cross-validation code for it to
exercise.

## 6. Fluid-loss regression — Experiment A  *(complete — full-cohort results below)*

The milestone-7 harness and the Experiment-A driver are complete; the method, every design
choice, and the full-cohort results are recorded here. The full-cohort `MODE=full` job on IBEX
produced the first outer-fold results and spent the config freeze — reported below from
`metrics_exp_a_{10,77}ghz.json`.

### Method
- **The harness (`eval/harness.py`).** One generic nested-LOSO engine used by both the reported
  Exp A path and the frozen leakage suite. Outer = leave-one-subject-out over the evaluable
  subjects; inner = subject-grouped `GroupKFold(min(5, n_train))`; selection metric = session-level
  subject-balanced MAE (`eval/metrics.py::subject_balanced_mae`, the same statistic the leakage
  suite pins to 5.5). Folds come only from `eval/splits.py`; the tie-break is only
  `eval/selection.py::select_candidate`; `protocol_freeze_guard(config, active=…)` runs before
  every fit, with a fail-closed completeness check on the per-fit protocol record.
- **The staged search (per outer fold).** Stage 1 searches the feature axes at a fixed ridge
  anchor (α = 1.0) — 10 GHz: reduction{A,B}×channel{mag,iq}×tiling{T1,T2,T3}×log{off,frozen,tuned}×
  gate{(1,2),(0.9,3)} = 72 combos; 77 GHz: 1×1×3×3×1 = 9. Stage 2 searches model family × grid
  (each ≤ budget_k = 12) at the Stage-1 winner. Seeds: the 5-seed set; inner metric = mean over
  seeds; outer = each seed scored separately (mean ± sd, never ensembled).
- **The tuned-ε branch — the one genuinely fitted WST quantity.** ε_o = 0.1·scale_o for orders
  1,2, where scale_o = median-over-training-subjects of the per-subject mean of the stored
  per-session pre-log scale; computed **fold-locally, train-only** (non-finite/non-positive →
  fallback 1e-6). It is recorded in the fit-audit like any fitted quantity.
- **The baseline.** Session-index-only (predict Δm% from time of day alone), K = 1, fit on
  outer-training subjects; the absent-time-index rule is the owner-decided global-training-mean
  fallback (O2). Reported as the pre-registered primary comparison.
- **Statistics (`eval/metrics.py`).** Session-level MAE/RMSE, pooled and per-subject predicted-vs-
  actual r (pooled r additionally on S1–S4), with a self-implemented subject-cluster **BCa**
  bootstrap (B = 10000, percentile fallback recorded, undefined-metric skip-and-count with the
  >5% unreliable flag) and the metric-type-aware seed-collapse rules; radar-vs-baseline via
  Wilcoxon signed-rank + a cluster-bootstrap CI on the per-subject difference. All CIs labeled
  conditional/exploratory.
- **Reproducibility + performance.** All numeric work is single-threaded (`threadpool_limits(1)`)
  and per-machine bit-reproducible. The 16 outer folds are independent, so they run in parallel
  worker processes (each single-threaded) — proven byte-for-byte identical to the serial run, so
  parallelism buys speed without touching the numbers.

### Provenance of the choices
- The whole A–G protocol was **frozen at the milestone-6 config-freeze gate** (tag
  `config-freeze-v1`) *before* any outer-fold result existed; Exp A consumes that frozen search
  space + statistics unchanged. Three protocol-gap completions were owner-decided at M7 (Step 0b):
  the inner-fold-variance estimator = population std (O1, A-M7-2); the baseline absent-index rule
  = global train mean (O2); the K=1 baseline guard path = config-level (O3).
- **Feature store.** WST features are extracted once into a per-session store (npz + fingerprint
  binding the exact QC frame membership + build commit) and validated fail-closed before a run —
  10 GHz locally / job-array, 77 GHz on IBEX. The tuned-ε branch reconstructs from the stored raw
  pre-log tensors per fold (why the store keeps them).
- **Mechanism before results.** The pipeline was proven on both bands with mechanism-only smokes
  (no performance value surfaced) before the freeze was spent — the deliberate M7 checkpoint.

### Results

All 16 subjects evaluable both bands (73 sessions, 10 GHz; 72 sessions, 77 GHz), 5-seed set,
commit `f36c4fb2`. All CIs are self-implemented subject-cluster BCa bootstraps, labelled
`conditional_exploratory: true`.

| | 10 GHz | 77 GHz |
|---|---|---|
| Session-level MAE (subject-balanced) | 0.469 [0.409, 0.568] | 0.495 [0.404, 0.646] |
| Session RMSE | 0.593 [0.509, 0.747] | 0.581 [0.483, 0.721] |
| Pooled predicted-vs-actual r | −0.138 [−0.286, 0.075] | −0.153 [−0.407, 0.174] |
| Radar − baseline (mean difference) | **+0.200 [0.145, 0.260]** | **+0.216 [0.127, 0.296]** |
| Wilcoxon p (radar vs baseline) | 3.05×10⁻⁵ | 7.6×10⁻⁴ |

**The headline result is negative, and decisively so, in both bands.** The radar regressor
*loses* to the trivial session-index-only (time-of-day) baseline — the mean-difference CI
excludes zero by a wide margin in both bands, and the Wilcoxon test is highly significant. Pooled
predicted-vs-actual r sits at essentially zero (both CIs straddle zero, and are wide — 77 GHz's
particularly so), so there is no pooled linear trend between predicted and actual Δm% either. Per-
subject r values are noisy and inconsistent in sign (10 GHz ranges from −0.99 to +0.90 across
subjects; 77 GHz from −0.96 to +0.91), which is itself informative: whatever the model is fitting
per fold does not generalize as a stable subject-level relationship.

Selection frequency differs somewhat by band — 10 GHz favours `knn` (7/16 folds) and the tuned-ε
branch (13/16); 77 GHz favours `svr` (8/16) and the off-ε branch (8/16, vs. 5/16 tuned) — but
neither band's selected-model distribution rescues the aggregate result.

**The key read, and why Experiment B exists.** Exp A alone cannot distinguish two explanations for
this negative result: (a) radar carries no usable fluid-loss signal in this cohort, or (b) it
does, but that signal is swamped by the fasting-clock confound — Δm% is structurally correlated
with time of day, because subjects fast and dehydrate progressively across the measurement day,
and the session-index-only baseline captures exactly that trend. A model that *only* decoded the
clock would already beat a radar model that ignored it, independent of whether radar carries any
real hydration signal at all. Experiment B (§7) is the pre-registered analysis designed to
separate these two explanations, by residualizing out the session mean and testing whether
*between-subject* variation within a fixed session is trackable.

## 7. Clock-decoupling — Experiment B  *(complete — full-cohort + session-specific variant results below)*

Experiment B is the analysis §6 motivated: Exp A's negative result cannot distinguish "no radar
signal" from "signal present but swamped by the fasting-clock confound", because Δm% is
structurally correlated with time of day. Within a *fixed* session every subject was measured at
the same clock time but lost different amounts of fluid, so predicting the **session-mean-
residualized** target — Δm%(subj, session) − μ_s, where μ_s is the train-only session mean —
tests whether radar tracks between-subject fluid-loss variation rather than decoding the clock.
ROADMAP.md calls this "the crucial evidence… a headline analysis, not a footnote."

### Method
- **Reuses Exp A's engine and search space, unchanged.** Composed on the same generic harness
  (`eval/harness.py`) via a `score_fn`/`FeatureBundle.session_idx` hook added specifically so Exp B
  never needs a second copy of Exp A's search-space enumeration or store-backed feature path
  (`stage1_candidates`/`stage2_candidates`/`StoreBackedFeatures`) — the frozen search space is
  enumerated in exactly one place in the codebase, used by both experiments.
- **S0 excluded at the source.** S0's Δm% is identically 0 by construction (baseline session), so
  including it would give every fold a free, perfectly-"predicted" session that deflates every
  MAE. `build_sessions_b` filters it out before any downstream code sees session data at all.
- **The residualizing provider (`SessionResidualFeatures`).** Wraps (does not subclass) Exp A's
  feature path. Computes μ_s — the train-only per-session mean of the raw target — via
  `baselines.session_means`, **fold-locally, train-only, cached per (fold, train_subjects) pair**
  so it is computed exactly once regardless of how many candidates/stages consume it. A session
  with fewer than 2 eligible training subjects is *dropped from that fold entirely*, never
  imputed with a global fallback (deliberately different from Exp A's baseline, which does have a
  global-mean fallback) — a degenerate per-subject residual would be meaningless, and Exp B's own
  invariant is that a dropped session must be visible in `dropped_sessions`, never silently
  smoothed over. μ_s is emitted via `extra_fits`, so it is fit-audited exactly like any other
  fitted quantity, at both CV levels, for free.
- **The baseline.** Zero residual, by construction — μ_s statistically saturates the mean of the
  target it is being subtracted from, so a baseline predicting "no deviation from the session
  mean" is the correct trivial comparison, not an afterthought.
- **The objective (`equal_session_residual_mae`).** Equal-session-weighted, not equal-subject-
  weighted: averages the four per-session residual MAEs, then averages those — provably different
  from a naive subject-weighted mean whenever per-session subject counts are unequal (proved
  directly on a synthetic fixture, not asserted). This is what makes S1-S4 contribute equally to
  the primary aggregate regardless of how many subjects happen to be eligible in each.
- **Statistics — two pre-defined estimands, both always computed and reported (A-M8-1, below).**
  A `session_weighted_bootstrap` CI on `aggregate(radar) − aggregate(baseline)`, matching the
  aggregate's own equal-session-weighting; and a subject-weighted, complete-case Wilcoxon signed-
  rank test + CI as a companion answering a different question (only subjects with usable S1-S4
  data contribute). A bootstrap replicate that empties a session out is skipped-and-counted
  (`n_skipped`, `unreliable` flag), never silently averaged over the surviving sessions (A-M8-2,
  below). Per-session breakdown for the primary model uses Holm-4 (family size 4, S1-S4). A run
  where any session is *globally* absent from the out-of-fold data sets `primary_aggregate=null`
  with a named `primary_unavailable_reason`, rather than a silently-degraded 3-session mean.
- **The session-specific secondary variant.** Four *independently fitted* single-session models
  (one per S1-S4), reusing the same objective (which degenerates naturally to plain single-session
  residual MAE with no special-casing) and the same execution strategy as the pooled model —
  designed from the start as four independent units of work, run as a real 4-task SLURM array, not
  a sequential loop (a sequential loop's wall-time would be ≈ the sum of all four searches, not
  the max). Reported descriptively only — effect size + CI, no p-value — because the frozen
  protocol's Holm-4 is defined only for the pooled model's own per-session breakdown and
  authorizes no multiplicity rule for four independently-fitted secondary models; inventing one
  would itself be an undisclosed post-Exp-A protocol completion.
- **Reproducibility + performance.** Single-threaded per fit (`threadpool_limits(1)`), bit-
  reproducible; the outer folds are independent and run in parallel worker processes, proven
  bit-identical to the serial result on the test fixture.

### Provenance of the choices
- **Exp B's core design — the residualization rule, search-space reuse, objective, baseline, the
  existence of both estimands, the Holm-4 family — was locked as part of `config-freeze-v1`
  *before any Exp A result was examined*.** That pre-registration is the entire scientific value
  of this experiment, and is exactly why it survives being run *after* seeing Exp A's negative
  headline: nothing about what runs, what data it uses, or how candidates are selected was
  informed by Exp A's outcome.
- **Two narrow, explicitly disclosed completions were decided on 2026-07-27 — after Exp A's full-
  cohort results were already visible — and that chronology is stated plainly here, not folded
  into "frozen before Exp A":**
  - **A-M8-1 (which pre-specified quantity is labelled "primary").** The frozen protocol text is
    internally ambiguous: one passage calls the equal-session aggregate "the single pre-specified
    primary test", but that aggregate is session-weighted, while the Statistics section's test
    form (Wilcoxon) needs per-subject pairs — the subject-weighted, complete-case estimand. Both
    cannot be the primary at once, and the frozen text never says which wins. Decided: primary =
    the session-weighted bootstrap CI, matching the aggregate's own words; the subject-weighted
    Wilcoxon is a companion, never conflated with the primary. This decision is **computation-
    neutral** — both quantities were already fully specified by the frozen text and are computed
    and reported in every run regardless of which gets the "primary" label; the decision changes
    prose, not numbers.
  - **A-M8-2 (empty-session bootstrap replicates).** The frozen aggregate presupposes all four
    S1-S4 sessions contribute to every bootstrap replicate; it does not say what happens when a
    subject-resample happens to omit every subject holding one session's only remaining rows.
    Decided: use the pre-registered skip-and-count machinery, never a silent fallback to a
    3-session mean. Unlike A-M8-1, this is **not** computation-neutral — it changes what a
    degenerate replicate contributes — which is exactly why it required the same owner-approval
    gate as any other protocol-gap completion, not a quiet implementation default.
  - Neither decision is a data-use choice, and neither reopens the blinding question `config-
    freeze-v1` protects: they are statistics-labelling and edge-case-reporting completions of an
    already-fully-specified protocol, decided in the open, after the fact that made them necessary
    (seeing the frozen text's own internal contradiction) became visible — not decisions about
    what to run or what data to use in response to Exp A's outcome.
- **Gating.** Exp A's full run already spent the `config-freeze-v1` blinding guarantee; Exp B's
  core design was itself frozen before Exp A was seen, so there was nothing left to blind for this
  compute step — the mechanism-only smoke ran as a cheap pre-flight, and the full-cohort run
  followed immediately with no owner pause in between (a deliberate single-phase DoD, contrasting
  with Exp A/M7's two-phase smoke-then-pause-then-full sequence, because that pause's entire
  purpose was already spent).
- **Feature store.** Same per-session store Exp A uses, rebuilt and `--validate`d fail-closed
  before any Exp B run — commit `30c6d907ca6f293f72db73517dc585bc39ec8e66` (a docs-only commit on
  top of the code commit, carrying identical M8 source; the store/analysis commit-match rule
  treats it as a genuinely different revision regardless, and enforced that strictly in practice —
  see HISTORY.md's step 10.5 entry for a case where it did).

### Results

All 16 subjects evaluable both bands, 59 evaluable S1-S4 sessions both bands, 5-seed set
(`seed=20260721`, `seed_set=[1,2,3,4,5]`), commit `30c6d907ca6f293f72db73517dc585bc39ec8e66`. All
CIs are the same subject-cluster BCa bootstrap machinery as Exp A, labelled
`conditional_exploratory: true`. `primary_viable=true` both bands (no session globally absent from
the out-of-fold data).

**Primary pooled model (full cohort):**

| | 10 GHz | 77 GHz |
|---|---|---|
| Radar residual MAE | 0.389 [0.310, 0.523] | 0.341 [0.268, 0.443] |
| Baseline (zero residual) MAE | 0.341 [0.270, 0.481] | 0.316 [0.247, 0.397] |
| **Primary: radar − baseline (session-weighted, A-M8-1)** | **+0.0475 [0.0230, 0.0749]** | **+0.0246 [−0.0066, 0.0756]** |
| Companion: subject-weighted complete-case Wilcoxon | p=0.00488 (n=11), diff +0.0592 [0.0297, 0.0906] | p=0.542 (n=13), diff +0.0195 [−0.0142, 0.0780] |

Per-session Holm-4 breakdown: 10 GHz — S1 holm_p=0.0736 (n=14), **S2 holm_p=0.00305 (n=16,
significant)**, S3 holm_p=0.482 (n=14), S4 holm_p=0.482 (n=15), all four point estimates positive
(radar worse); 77 GHz — all four holm_p=1.0 (none significant).

**10 GHz shows a statistically clear residual signal, in the wrong direction: radar loses to the
trivial train-only session mean even after clock-decoupling**, both by the primary aggregate CI
(excludes zero) and its significant companion Wilcoxon. **77 GHz shows no distinguishable
difference either way** (CI crosses zero, companion not significant, no session individually
significant). Selection frequency differs by band exactly as in Exp A: 10 GHz favours `svr`
(10/16 folds) and the frozen-ε branch (11/16); 77 GHz also favours `svr` (12/16) but the tuned-ε
branch (14/16) — a reversal of Exp A's own per-band pattern, worth noting but not adjudicating
without a dedicated interpretability pass (§9).

**Session-specific secondary variant** (four independently-fitted single-session models,
descriptive only — no p-value, by design; `completed_sessions=[1,2,3,4]` both bands):

| Session | 10 GHz radar − baseline | 77 GHz radar − baseline |
|---|---|---|
| S1 | +0.0612 [0.0105, 0.1486] (n=14) | +0.0058 [−0.0209, 0.0309] (n=13) |
| S2 | +0.0793 [0.0056, 0.1681] (n=16) | −0.0077 [−0.1015, 0.0773] (n=16) |
| S3 | +0.2127 [0.0448, 0.4214] (n=14) | −0.0430 [−0.0925, 0.0062] (n=14) |
| S4 | +0.1360 [0.0006, 0.3349] (n=15) | +0.0024 [−0.0403, 0.0500] (n=16) |

10 GHz's four independent single-session models agree in direction with its own pooled result —
all four positive (radar worse), consistent rather than an artifact of pooling. 77 GHz's four are
mixed in sign and every CI crosses zero — no consistent signal in either direction, consistent
with its own pooled null result.

**Overall reading.** Clock-decoupling does not rescue radar in this cohort. 10 GHz's Exp A loss to
the time-of-day baseline is *not* fully explained by the fasting-clock confound alone — even after
removing the session-mean trend, the residual radar signal is still measurably worse than a
trivial constant, both pooled and (independently) per session. 77 GHz's Exp A loss is more
consistent with a confound explanation: once decoupled, 77 GHz shows no significant difference
from the residual baseline in either direction, at any level of analysis. Taken together with §6,
neither band demonstrates a radar-based fluid-loss signal that outperforms simple time-of-day or
session-mean statistics in this cohort — a genuinely negative result for the study's central
hypothesis, reported here in full rather than as a footnote, per the plan's own framing.

## 8. Ordinal classification & baselines — Experiments C, D  *(complete — full-cohort results below)*

Experiments C and D close the question §6 and §7 opened. Exp A found the radar regressor losing
to a time-of-day baseline; Exp B found it losing to a session mean even *within* a fixed session.
What remained was whether the negative result was an artifact of posing the problem as
regression (C), or of the WST feature choice (D). Neither turns out to be the case. All results
below are at commit `3f465ab`, full cohort, 16 subjects.

### Experiment C — method

- **The task.** The paper's 5-class S0–S4 staging, kept as an **ordered** task. The target is
  two-column, `y = [L, class]`, with the frozen sign convention `L = −Δm%` so that class order
  increases monotonically with fluid loss. `class_idx` **is** the session index: S0 = 08:00
  through S4 = 16:00. That identity is not incidental and is returned to below.
- **One search space, not a second copy** (A-M9-1). Family (a) reuses Exp A's frozen enumeration
  and store-backed feature path unchanged; only the target, the objective and the estimator head
  differ. Nothing about the feature search was re-opened for this experiment.
- **Two frozen arms sharing one Stage-1 feature key.** Arm (a) is regress-then-threshold: the
  five base families wrapped in the frozen thresholding rule. Arm (b) is Frank-Hall — K−1 = 4
  independent binary logistic fits on `1[class > k]`, each carrying train-only inverse-frequency
  class weights, over the frozen C grid.
- **The objective** is pooled class-unit MAE as the harness `score_fn`, with the §2.3 inner-fold
  aggregation over *evaluable folds only*. That aggregation lives in `exp_c.py` rather than the
  harness deliberately: the class-coverage viability predicate is candidate-independent, so a
  plain `np.mean` over all inner folds would go NaN for every candidate whenever one inner
  training set lost a class — silently promoting "one inner fold lost a class" into "this outer
  fold produced no ordinal result", which is stricter than the frozen rule.
- **Metrics are ordinal only**: quadratic-weighted kappa, adjacent accuracy, class-unit MAE, plus
  the confusion matrix. Plain accuracy is not reported, per the framing locked in §0.
- **No baseline comparison** (plan §5 trap 16). The session-index baseline predicts the Exp C
  class *perfectly* by construction — the class is the session index — so any radar-vs-baseline
  framing would be degenerate. Exp C reports its ordinal metrics absolutely.

### Experiment C — provenance of the choices

- **The QWK-undefinedness rule (O-M9-8, decision 8a, owner-approved 2026-07-30).** Undefinedness
  is decided by the actual denominator, not by a class-count pre-check: kappa is NaN iff the
  input is empty or the expected disagreement is exactly zero, which on this fixed 5×5 grid
  happens only when both marginals concentrate on the *same* single class. A single-class truth
  side alone does not trigger it as long as the other side varies. This was a contested design
  point, so the instrumentation counting how often it fires is reported below rather than
  assumed negligible.
- **`max_iter = 1000` in Frank-Hall is a solver convergence bound, not a tuned quantity**, and a
  `ConvergenceWarning` from lbfgs is promoted to an exception: a non-converged threshold fit must
  stop the run rather than contribute coefficients to a reported result. `exp_c.py` asserts the
  bound against an independently-stated constant so the guard cannot drift.

### Experiment C — results

16 subjects evaluable in both bands (73 sessions at 10 GHz, 72 at 77 GHz). Seed counts are
**realized per arm**: arm (a) reaches the full 5-seed set on folds selecting a seed-sensitive
family, while arm (b) is Frank-Hall throughout and is deterministic, so it realizes 1.

| band | arm | realized seeds | QWK | adjacent accuracy | class-unit MAE |
|---|---|---|---|---|---|
| 10 GHz | a (regress-then-threshold) | 5 | **−0.212 [−0.365, −0.030]** | 0.534 [0.432, 0.631] | 1.553 [1.369, 1.772] |
| 10 GHz | b (Frank-Hall) | 1 | **−0.197 [−0.312, −0.075]** | 0.521 [0.459, 0.597] | 1.658 [1.477, 1.833] |
| 77 GHz | a | 5 | **−0.278 [−0.461, −0.077]** | 0.558 [0.465, 0.636] | 1.492 [1.317, 1.700] |
| 77 GHz | b | 1 | +0.025 [−0.281, 0.243] | 0.611 [0.487, 0.712] | 1.347 [1.137, 1.680] |

**Three of the four arms have kappa confidence intervals lying entirely below zero.** The correct
reading is *no usable ordinal signal*, not inverse predictive ability: the predictions collapse
toward the middle classes and then run counter to truth, which is what a no-signal predictor
looks like under a chance-corrected metric. The fourth arm (77 GHz, Frank-Hall) sits at
essentially zero with a CI spanning it.

**Adjacent accuracy must not be quoted alone.** Values of 0.52–0.61 look respectable, but that is
precisely what middle-collapsed predictions produce: a prediction parked near S2 lands within ±1
of a large share of the truth without tracking it. QWK is the headline; adjacent accuracy is
reported for completeness and interpreted only alongside it.

Two supporting observations. Arm (a)'s selected family is heterogeneous across folds (10 GHz:
ridge 8, gbm 4, rf 2, svr 2; 77 GHz: gbm 8, ridge 4, rf 2, svr 2) with no family dominating —
when a real signal exists one family tends to win consistently. And the O-M9-8 instrumentation
is a **clean zero in both bands**: `n_qwk_nan = 0` and `n_single_class_truth_val_folds = 0` across
9280 (10 GHz) and 4240 (77 GHz) inner evaluation cells, and across 56 and 72 outer cells. The
contested undefinedness rule never fired, so no result below depends on which side of that
decision was taken.

### Experiment D — method

Experiment D contests "WST wins" by putting the frozen pipeline against five alternatives under
the identical LOSO harness, on both bands.

- **The six families.** Four deep-learning families — `cnn1d_raw` and `cnn1d_matched` (1D CNN on
  the raw beat signal), `spec2d_raw` and `spec2d_matched` (spectrogram + small 2D CNN) — plus
  `physics` (in-band reflected-power / two-band power ratio) and `session_index` (time-of-day
  lookup, no radar data at all). The CNN families are the **only** authorized GPU consumers; the
  frozen numpy-backend policy for reported WST features is untouched, and every non-CNN family is
  refused if it names a device other than CPU.
- **The run-group architecture.** Each CNN family × band runs as three stages: a single-task
  `--init-run-group` that validates the store, builds the frame spine and writes the
  authoritative per-fold row census; a 16-task GPU array, one outer fold per task; and a merge
  gated on the array. A fold index beyond the selectable list exits zero with a named no-op
  marker, and a partial merge is a **named non-reportable state** — a subset of the selectable
  folds is never silently treated as a smaller cohort.
- **The primary comparison** is radar versus `session_index`, pre-registered, on per-subject
  session MAE, via Wilcoxon signed-rank plus a subject-cluster bootstrap CI on the mean
  difference.
- **The composite** is one uncorrected comparison against a per-fold best-of-three
  (`cnn1d_raw`, `spec2d_raw`, `physics`) chosen by inner CV. It splices at the **per-subject
  metric** level, not the prediction level: a prediction-level splice is undefined across
  families of different seed multiplicity, and averaging predictions across seeds is forbidden.
- **Per-family comparisons** form an exploratory Holm family of exactly 3. The two matched-
  preprocessing families are **ablations**: reported descriptively, entering no comparison
  family, with no p-value and no Holm slot (O-M9-3).

### Experiment D — results

| | 10 GHz | 77 GHz |
|---|---|---|
| radar MAE | 0.469 [0.409, 0.568] | 0.495 [0.403, 0.648] |
| **session-index MAE** | **0.269 [0.212, 0.377]** | **0.278 [0.216, 0.372]** |
| radar − session-index | **+0.200 [0.145, 0.261]** | **+0.216 [0.129, 0.294]** |
| Wilcoxon p | 3.05×10⁻⁵ | 7.63×10⁻⁴ |
| Wilcoxon statistic | **0.0** | 8.0 |

**A baseline that never looks at the radar — it knows only which session of the day a recording
came from — beats every radar-based method in both bands.** At 10 GHz the Wilcoxon statistic is
exactly zero, meaning *all sixteen* subjects were better predicted by the baseline, not a
majority. This reproduces on the full cohort what §6 found for Exp A at `f36c4fb2`, and it is not
a one-off.

Per-family subject-balanced MAE (lower is better):

| family | 10 GHz | 77 GHz |
|---|---|---|
| session_index | **0.269** | **0.278** |
| physics | 0.446 | 0.479 |
| cnn1d_matched *(ablation)* | 0.451 | 0.497 |
| radar (WST + classical) | 0.469 | 0.495 |
| cnn1d_raw | 0.468 | 0.492 |
| spec2d_matched *(ablation)* | 0.528 | 0.478 |
| spec2d_raw | 0.569 | 0.531 |

Every radar-based representation — including two that **learn their own features end to end from
the raw signal**, inheriting no WST assumption — lands between 0.45 and 0.57, and every one loses
to the clock. That is the strongest available evidence that the negative result is not an
artifact of the feature choice.

Against the composite, radar wins at 10 GHz (−0.099 [−0.192, −0.050]) and is indistinguishable at
77 GHz (−0.037 [−0.093, +0.019]). In the exploratory Holm family of three, radar ties `cnn1d_raw`
(10 GHz +0.002, Holm p = 0.94) and `physics` (+0.024, Holm p = 0.70), and beats `spec2d_raw`
(−0.099, Holm p = 0.0064); at 77 GHz nothing separates (all Holm p ≥ 0.76).

**These wins should not be led with, and one detail shows why.** At 10 GHz the composite's inner
CV selected `spec2d_raw` in **16 of 16 folds** — the *worst* of the six families — so the
composite is simply `spec2d_raw`, and radar "beating the composite" reduces to radar beating the
weakest alternative. An inner-CV procedure that consistently selects the worst-performing family
is itself evidence that inner scores carry no information about outer performance. Being the best
of several approaches that all lose to a time-of-day lookup is not a positive result.

### Reproducibility, and a post-hoc amendment disclosed in full

Experiment D's radar side is Exp A's own output, so the comparison is only meaningful if that
output is the one §6 reported. Acceptance criterion **O-M9-5** required the M9 Exp A re-run to
reproduce the milestone-7 artifacts.

**The criterion was amended after it failed, and the chapter states this plainly.** As originally
written, O-M9-5 required `predictions_{band}.csv` to be bit-identical to the M7 artifact. On the
first M9 re-run that failed for 10 GHz while 77 GHz passed: 11 of 149 rows differed, by at most
5.14×10⁻¹⁴, with `Δy_true` exactly zero, and `selection_table_10ghz.csv` byte-identical. The
cause was pursued to exhaustion — raw data, splits, seeds, model selection, node hardware,
run-to-run nondeterminism, declared package versions, the store rebuild, the Exp A code path and
core count were each eliminated by direct test, and M7's own code was shown to reproduce the
current feature store bit-for-bit across all five SVR-selected subjects (2277 arrays, 23
sessions). What remained was floating-point summation order in the fit path, amplified by
libsvm's SMO convergence tolerance.

The amended criterion is a **conjunction**, ordered so the second part is unreachable unless the
first passes: `selection_table_{band}.csv` byte-identical, **and** `max |Δy_pred| ≤ 10⁻¹⁰` with
the observed value recorded on every run. The tolerance is four orders above the largest
difference ever observed and about nine below anything that could reach a reported digit for a
Δm% of order 0.1–1. The selection table is what makes this safe rather than merely convenient:
which model a fold selects is a discrete, tolerance-free outcome that any genuine drift must
change. **A limitation to state honestly:** part 1 detects any change in a discrete decision,
whatever its origin — not only real drift.

Two findings from that investigation stand on their own. First, the pipeline is **bit-reproducible
across CPU microarchitectures** for Exp C in both bands and for Exp A at 77 GHz — verified by an
AMD Turin versus Intel Skylake comparison of byte-identical selection tables and predictions.
Second, 10 GHz Exp A is *not*: it is sensitive both to CPU microarchitecture and to the realized
software environment, at the 10⁻¹³ level, and in no case did that ever change a single model
selection. The final reported runs satisfy the amended criterion on both bands
(2.33×10⁻¹³ and 0.0 against the 10⁻¹⁰ tolerance).

### A measurement of the protocol itself — the frame-split demonstration

The chapter's opening claim is that the published ~96–98% five-class accuracy is a product of
frame-level, subject-dependent splitting rather than of a hydration signal. That claim can be
measured rather than asserted, and the sanctioned exploratory frame split does so.

**The numbers in this subsection are leaky by construction and are not results.** They appear
once, here, as a measurement of the evaluation protocol. They are not comparable with any LOSO
figure in this chapter, they appear in no results table, and no conclusion about hydration rests
on them.

Holding the features, the models, the selected configurations and the data fixed, and changing
*only* the split — subject-level hold-out replaced by 5-fold random assignment of the 7168
individual frames — the 10 GHz ordinal task moves as follows:

| arm | LOSO (reported) | frame-level split (leaky) |
|---|---|---|
| a (regress-then-threshold) | QWK −0.212 | QWK +0.405, accuracy 0.307 |
| b (Frank-Hall) | QWK −0.197 | QWK **+0.819**, accuracy **0.803** |

Frank-Hall swings by more than a full unit of kappa, and from no usable signal to 80.3% five-class
accuracy, on the split alone. The obvious objection — that the frame split simply has more
training rows — is the leakage restated: those 5734 training frames come from the same 73
sessions and carry no new information, because frames within a session are near-duplicates
sharing one label.

This does **not** reproduce the published 96–98% exactly; 80.3% is lower, and the residual gap is
plausibly the original's different classifier and feature pipeline. What it establishes is the
*regime*: an honest protocol on this data says no signal, and frame-level splitting on the same
machinery says strong classifier.

One further observation, offered as a hypothesis rather than a conclusion. The arm that exploits
the leakage hardest is the discriminative one — Frank-Hall reaches 0.803 accuracy where
regress-then-threshold reaches 0.307 — and Frank-Hall at 77 GHz is precisely the one cell of the
sixteen that could not be computed, because its logistic fits failed to converge at the frozen
bound and the guard refused to return coefficients from an unconverged fit. Logistic regression
fails to converge when classes become near-perfectly separable, which is what leakage
manufactures. If that is the mechanism, the refusal is a symptom of the leakage rather than a
solver accident. The 10 GHz arm converged, so this is not universal and is not claimed as
established.

### What Experiments C and D establish

Four independent lines now agree: Exp A's null pooled correlation and defeat by a time-of-day
baseline (§6), Exp B's failure to beat a session mean within a fixed session (§7), Exp C's ordinal
collapse, and Exp D's defeat by the same baseline across six representations including two
learned end to end.

**In this cohort, at this dehydration range, with this hardware, the radar features carry no
recoverable information about fluid loss beyond what the time of day already supplies.**

The claim is bounded deliberately, and the bounds are the study's own. Sixteen subjects. A Δm%
range from 0.000 to −2.020, with most sessions well under 1%, so the physical effect being asked
for is a permittivity change from a fraction of a percent of body mass. Body mass as the single
objective reference, with no osmolality and no temperature record. And a fasting design that ties
hydration to time of day, which §7 reduces but cannot remove.

This is not evidence that radar-based hydration sensing is infeasible. It is evidence that this
study cannot demonstrate it, and — through the frame-split measurement above — an account of why
an earlier analysis of the same data appeared to.

### Supplementary mechanism check: frozen WST-order trajectories

A separate, outcome-independent diagnostic asked a narrower question after the predictive
experiments: do complete first- or second-order scattering trajectories contain a path whose
direction is shared across subjects and robust to the predeclared boundaries? This is not
Experiment E. Experiment E measures how much a fixed predictive model relies on a path group;
the trajectory diagnostic instead analyzes every saved path directly, without fitting a hydration
predictor, choosing a bank, or using one band to select or explain the other.

The diagnostic was frozen before its real-data run and executed through four guarded stages in a
separate repository. For 10 GHz, all 80 mapped cells were accounted for, with 73 eligible sessions
and seven frozen-QC skips; for 77 GHz the corresponding counts were 80, 72 and eight. The complete
M3 tables contain 100,624 subject-effect and 6,289 group-summary rows at 10 GHz, and 51,040 and
3,190 at 77 GHz. Before publishing the two-band report, M4 rehashed all 80 raw files in each band.
The two analysis hashes are `9eb12b83…38e19` and `74cdb2b9…92dec`; the normalized report hash is
`d7eab500…d1994`. The final report and no-leakage gate exited zero. No ML or cross-band fusion ran.

The result is compact:

| arm | primary order 1 | primary order 2 |
|---|---|---|
| 10 GHz, A65 | path 40 met the descriptive shared-candidate rule; positive majority sign; QC73 and boundary persistence; Holm-adjusted p = 1.0 | no descriptive candidate |
| 77 GHz, QC72_77 | no descriptive candidate | no descriptive candidate |

Path 40 is therefore a hypothesis, not evidence that survives multiplicity. Its persistence says
the descriptive pattern is not created by the QC73 inclusion or the tested path boundaries; it
does not make the association statistically confirmatory. The sensitivity banks preserve several
10 GHz flags, as the frozen protocol required, but those flags cannot create, replace or rescue a
primary candidate. All 77 GHz sensitivity banks are null. No paths are matched across banks or
bands, and no bank is ranked.

The multiplicity ceiling is severe and must qualify the null. With 12 A65 subjects, the minimum
two-sided sign p-value is 0.00048828125; Holm significance is mathematically unavailable to the
129-, 303- and 686-path 10 GHz order-2 families, while only the 95-path `s_q1` family is even
theoretically capable. With 13 QC72_77 subjects the minimum is 0.000244140625, leaving every
77 GHz order-2 family (211–561 paths) structurally incapable of Holm significance. This is low
confirmatory power, not proof that no trajectory structure exists.

The appropriate synthesis with §§6–8 is nevertheless unchanged. The trajectory diagnostic finds
no multiplicity-corrected path evidence, and its one descriptive 10 GHz pattern has no independent
77 GHz counterpart. It therefore does not rescue the failed LOSO predictions, establish a
water-driven dielectric mechanism, or justify selecting path 40 for a later model. It is retained
because a complete null-sensitive mechanism analysis is more informative than reporting only an
attractive path.

## 9. Fusion, interpretability, confounds, statistics — G, E, F, H  *(fill at milestone 10)*

Not yet written. **No milestone-10 result exists as of 2026-08-08** — implementation is in progress
(steps 1–4 of the plan's §4.2 are built and tested: the Exp-A reference gate, the multiplicity
foundation, the H robustness driver, and Experiment G) but **no full-cohort E/F/G/H job has been
run**, so this section still contains no number by design. It is written in full only after verified
full-cohort artifacts exist. Recorded here so the section's scope and its known framing problems are
fixed *before* any result exists, in the same spirit as the milestone-6 config freeze.
**`plans/MILESTONE_10_PLAN.md` (independently reviewed and accepted 2026-08-07, amended during
implementation) is the authoritative design for all four experiments; this section is a
chapter-facing summary of it, not a second source of design decisions.** Where anything below and
the plan appear to disagree, the plan governs and this section is stale and needs updating.

**What the section will contain.** Experiment G (matched-session decision-level fusion: independent
per-band Exp-A staged selection refit on the matched cohort, a constrained convex combiner with α on
a 21-point grid fit from *selection-honest* nested cross-fitted out-of-fold predictions — never the
harness's ordinary first-seed-only inner predictions — primary contrast fused vs 10-only); Experiment
E (a fixed, pre-registered ridge model on Exp-B residual targets, scored under ordinary outer LOSO,
with per-path importance from **leave-one-path-group-out refit and rescore**, not a permutation CV);
Experiment F (the heart-rate confound check ROADMAP §4 describes is not estimable — no HR observation
exists anywhere in the delivered data — reported as such, alongside four nested ridge models on the
clock plus static covariates as a separately named available-covariate sensitivity analysis, plus the
algebraic-coupling sensitivity variants); Experiment H (per-subject performance spread, the
pre-specified comparisons with subject-cluster CIs and Holm correction, and a selection-variance
**empirical percentile range** — not a BCa interval — from full-procedure subject resampling). All
four designs were frozen at milestone 6 and are transcribed in `configs/exp_e.yaml`, `exp_f.yaml`,
`exp_g_fusion.yaml` and `stats.yaml`; **twelve** explicit post-freeze protocol amendments
(**A-M10-1..12**, `plans/MILESTONE_10_PLAN.md` §0.2) are disclosed here by reference and will be
restated with their full reasoning, and with their true chronology, when this section is written in
full. The chronology matters and must not be smoothed over: **A-M10-1..6 were made after A–D's
results were visible** but before any milestone-10 code existed, as the condition of accepting the
plan; **A-M10-7..12 were raised during implementation** (7–9 in steps 1–2, 10 in step 3, 11 in
step 4, 12 before step 8). A-M10-7..11 were each found by testing the plan against its own stated
requirements: one is a provenance correction that changes no estimand (A-M10-7), two are the
bootstrap-multiplicity mechanism and the Exp-C weighting that preserves byte-neutrality (A-M10-8,
A-M10-9), and two are artifact-granularity decisions about which provenance rows are written
(A-M10-10 for Experiment H, A-M10-11 for Experiment G's fit audit) — **none of those five changes
an estimand, a metric, or an acceptance criterion**, and the chapter must say so explicitly rather
than leaving a reader to count amendments and infer drift.

**A-M10-12 is the exception and must not be grouped with them.** It is an owner decision about
process, taken before step 8, and it *does* change an acceptance criterion: there is no
independent code review, because no second person is available. Step 9 became an author
self-review. The chapter must state this plainly in its own right — see limitation 6 below — and
must never describe the milestone-10 software as peer-reviewed.

**Things this section must not paper over.**

1. **Experiment E's design changed under review (A-M10-1) and its reporting stance is outcome-neutral
   by design (A-M10-6).** The milestone-6 freeze specified a standalone 4-fold permutation CV; review
   found this violated the project-wide LOSO requirement and was undefined for incomplete validation
   trajectories. The accepted replacement — leave-one-path-group-out refit under ordinary outer LOSO —
   is the *documented alternative* the frozen text already named, so no new method was invented
   post-hoc, only the already-specified fallback was promoted to primary. Separately: E's stated
   original purpose was alignment between informative scattering paths and the Cole-Cole water-driven
   permittivity expectation — supporting evidence a real signal is physical — and §6–§8 show no such
   signal. The accepted resolution is **not** to pre-label the per-path result "null" or "physical"
   before it exists (A-M10-6: "a desired narrative must not be encoded as a software acceptance
   criterion"); instead, the chapter states plainly, before presenting the path table, that the fixed
   model's predictive context is already known to be weak (§6–§8), so any attribution describes what
   the model relied on, not a validated physical mechanism — then reports the table as measured, with
   no framing that predetermines whether a path's importance reads as "signal" or "noise structure."
   This supersedes the 2026-08-06/2026-08-07 "report as a null attribution" framing recorded in an
   earlier draft of this section, which the review correctly identified as pre-encoding an outcome.
2. **Experiment F is not the heart-rate confound check the paper's framing implies, and the software
   says so explicitly rather than silently substituting.** Heart rate was reportedly collected but
   **zero HR observations exist anywhere in the delivered data** (verified: no HR file, no HR column
   in the weight/subject workbook). The accepted design (A-M10-2) makes this a first-class, machine-checked
   status — `status="not_estimable_missing_heart_rate"`, `n_hr_observations=0` — never a silent
   substitution and never a proxy correlation. The clock-plus-static-covariate analysis (age, height,
   baseline mass, BMI) that *does* run is reported as a separately named available-covariate
   sensitivity analysis, not relabelled as the HR check. Skin temperature and glucose remain
   uncontrolled (temperature logs lost, glucose never measured).
3. **Experiment G fuses two arms that both already lost to the clock.** The pre-registered primary
   contrast is therefore a comparison between two failures, and the reading of a *positive* fused
   result is pre-committed here, before the result is seen: with α selected over 21 grid points on a
   16-subject matched population, a small fused improvement is far more consistent with selection
   noise than with complementary information across bands, and would be reported as such unless its
   subject-cluster CI excludes zero by a margin comparable to the §6 effects. Feature-level fusion is
   explicitly deferred (A-M10-4) and is not a milestone-10 completion criterion.

4. **The milestone-10 code was reviewed only by its author (A-M10-12).** The accepted plan
   specified an independent read-only code review as a completion criterion; no second person
   was available, so step 9 became an author self-review and the criterion was amended rather
   than quietly treated as met. The chapter states this as a limitation of the *evidence*, not
   as a caveat about effort: what independent review defends against is the assumption the
   author never thought to question, which is by construction encoded identically in the code
   and in the test written to check it, and no amount of author diligence reaches it. What does
   still hold, because it does not depend on who read the code: `tests/test_no_leakage.py` was
   frozen at milestone 7 and is byte-identical throughout milestone 10; the multiplicity work is
   pinned byte-for-byte against the already-reported M8/M9 artifacts; every store, schema and
   lineage check fails closed; and the full and real-data suites pass on the final tree. Those
   catch drift, leakage and broken lineage. They do not catch a wrong-but-self-consistent idea.
   The software behind these results is therefore **extensively tested and author-reviewed, and
   is not peer-reviewed** — a reader assessing the numbers is entitled to weigh that.

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
