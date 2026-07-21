# SECOND_CHAPTER — thesis chapter material

Distilled, chapter-ready account, written **at each ROADMAP §7 milestone completion**
(not a duplicate of HISTORY.md). Capture the provenance of every choice so nothing is
unexplained: why each parameter value, why one processing choice over an alternative,
what a result means, and how it ties back to the paper's method and physics. MATLAB is
**not** mentioned here — all results are from Python.

> Status: **milestone 1 complete** (§0.1 below). Later sections fill in as their
> milestones close. The methodological framing (§0) is locked and is the spine of the
> Methods section.

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

## 1. Data & ground truth  *(fill at milestone 2)*

## 2. Preprocessing  *(fill at milestone 3)*

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
