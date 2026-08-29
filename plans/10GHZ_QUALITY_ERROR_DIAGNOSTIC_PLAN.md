# ACCEPTED PLAN — 10 GHz quality versus LOSO error diagnostic

## Status and question

**Status: implemented and verified on 2026-08-29.**

This diagnostic asks whether sessions with worse radar-quality measurements also have
larger errors in the already-completed, leakage-safe Experiment A LOSO predictions.  It
is exploratory association analysis only.  It does not retrain a model, change QC,
exclude or weight a sample, choose a threshold, or recommend that a session be removed.

## Frozen inputs and population

- Predictions are reconstructed only from the authoritative Milestone-10 Experiment A
  manifest after the approval/source comparison gates pass.  Old local prediction CSVs
  are not inputs.
- The gate requires schemas `reference_exp_a_manifest_v1` and `exp_a_sources_v1`, and
  requires each of the reference gate's 11 evidence classes exactly once.  It recomputes
  the ordered population relations using the reference gate's canonical JSON hashing:
  session keys `ef937439...`, targets `1f0530dd...`, and frame population `e2d0ea2c...`.
  Thus even equal-target sessions cannot be silently reordered.
- The reconstructed bytes must equal the frozen `predictions_10ghz.csv` SHA-256
  `78d6076c5c5fcd79cf7b994c5f7ad508832228f6cf9baca70bbdc39cf1cebf9e`.
- Each fold is aligned to its subject's sessions in the manifest's recorded population
  order, with exact target equality.  The 73 eligible session cells occur once each.
- Random-forest/gradient-boosting seeds are repeated realizations, not independent
  samples.  Session MAE is the mean absolute error over realized seeds.
- The quality card contributes 73 complete rows.  WST repeatability contributes 71;
  its two missing rows must be exactly the two block-coverage review sessions.

## Fixed metrics and estimand

Seven worse-oriented metrics are analysed separately; no composite is constructed:

1. `1 - pass_fraction`;
2. `20 - minimum_block_n_pass`;
3. `-in_band_ratio_p10_margin`;
4. `peak_bin_iqr`;
5. to 7. magnitude, order-2, within-path-shape
   `block_to_session_distance_maximum`, separately for tilings 0, 1, and 2.

For each analytic population, freeze the observed metric mean and sample SD (`ddof=1`).
Fit ordinary least squares with an intercept, the metric expressed as one sample-SD
worse, subject fixed effects, and session-index fixed effects, omitting one reference
level for each.  The coefficient is reported in absolute-error percentage-body-mass
points per one-SD worse quality.  Partial correlation residualizes both error and the
metric against the same fixed-effect-only design.  Full rank and finite inputs are
required.

## Uncertainty and influence

- Attempt exactly 10,000 subject-cluster bootstrap draws with the fixed configured seed.
  Draw 16 subjects with replacement and give every repeated draw its own cluster label.
  Refit fixed effects while retaining the original metric mean and SD.  Use percentile
  95% intervals over valid draws.  More than 5% invalid draws makes the metric unreliable
  and fails the strict production run.
- Refit deterministically after leaving out each original subject.  Report every
  coefficient and the resulting range.
- Report finite-row, subject, unique-value, and non-best-value support counts.

No p-values, learned cutoffs, exclusions, weights, causal claims, or sample-quality
recommendations are produced.

## Outputs and safeguards

The runner writes only beneath `results/quality_error_10ghz/` and
`figures/quality_error_10ghz/`.  It captures a valid clean Git commit at the beginning,
authenticates every frozen input and source relationship, refuses existing output roots,
requires those exact repository-root destinations, and verifies that all pre-existing
result/figure bytes remain unchanged.

Tables:

- `session_quality_vs_loso_error.csv`
- `quality_metric_catalog.csv`
- `association_summary.csv`
- `leave_one_subject_out_influence.csv`
- `population_flow.csv`
- `provenance.json`

Figures:

- `association_forest.png`
- `quality_error_residual_panels.png`
- `population_flow.png`

Every figure is labelled “exploratory association — no exclusion rule.”  A later
preprocessing change or sensitivity retraining needs a separate owner-approved plan.
