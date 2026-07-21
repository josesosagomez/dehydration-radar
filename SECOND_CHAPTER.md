# SECOND_CHAPTER — thesis chapter material

Distilled, chapter-ready account, written **at each ROADMAP §7 milestone completion**
(not a duplicate of HISTORY.md). Capture the provenance of every choice so nothing is
unexplained: why each parameter value, why one processing choice over an alternative,
what a result means, and how it ties back to the paper's method and physics. MATLAB is
**not** mentioned here — all results are from Python.

> Status: scaffold only. No milestone complete yet; sections below fill in as
> milestones close. The methodological framing (below) is already locked and is the
> spine of the Methods section.

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
