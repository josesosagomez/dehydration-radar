# 10 GHz quality-aware training sensitivity (implemented design)

This is an exploratory, post-hoc sensitivity analysis.  It does not replace the frozen
Experiment A or C results and writes only below
`results/quality_training_sensitivity_10ghz/`.

The single quality value is the authenticated radar-only
`in_band_ratio_p10_margin`.  The 73 eligible session keys must align exactly and its five
negative rows must be `(4,S2), (8,S0), (8,S2), (12,S0), (16,S3)`.  The threshold is fixed
at zero; no outcome, prediction error, or body-mass-derived quality value chooses it.
The source is pinned to the approved quality-audit commit and CSV hash; all 80 status rows
and the formula `margin = in_band_ratio_p10 - 0.3` are reconciled before use.

For every fixed learner, compare: (1) baseline, (2) remove negative-margin rows from
training only, and (3) append the raw margin before the train-only scaler.  Held-out rows
are identical.  Filtering precedes tuned WST epsilon, scaler, ordinal weights/cutpoints,
and model fitting.  There is no quality weighting.

LOSO is primary.  Regression reuses each authenticated Exp-A fold's selected learner and
feature key; Experiment C performs its frozen nested baseline selection for both ordinal
arms, then holds each winner fixed across treatments.  The second protocol is a 5-fold,
seed-20260829, session-level stratified split.  It has subject overlap by design, performs
baseline-only nested selection with explicit row masks, and is always labelled optimistic,
post-hoc, and not new-subject generalization.  Frame splitting is never used.

Regression reports subject-balanced MAE/RMSE and paired subject deltas. Classification
reports class-unit MAE, adjacent accuracy, and quadratic-weighted kappa; exact accuracy is
secondary only.  Full execution is CPU-based on IBEX and requires a store rebuilt at the
implementation commit.  The local synthetic run is mechanism-only and suppresses scores.

Before any result is written, the unmodified LOSO regression treatment must replay the
authenticated Exp-A fold selections, effective seeds, tuned epsilon, and predictions within
an absolute `1e-10` tolerance.  A run is assembled in a unique sibling staging directory;
only a complete run with hashes and provenance is atomically renamed to the fixed final root.
