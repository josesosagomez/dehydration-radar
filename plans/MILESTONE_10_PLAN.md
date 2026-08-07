# MILESTONE 10 PLAN — Experiments E, F, G, and H

## 0. Purpose, authority, and planning-only boundary

Milestone 10 implements the four remaining analyses:

- **E — physics-grounded model interpretability**;
- **F — confound assessment**;
- **G — decision-level 10 GHz + 77 GHz fusion on the dual-band cohort**;
- **H — final subject-level statistical assembly and selection-variance robustness analysis**.

This is ROADMAP §7 item 9, called Milestone 10 because
`plans/implementation_plan.md` inserted the completed configuration-freeze gate. The number is
therefore correct for the repository even though ROADMAP's compact list calls the same work item 9.

This file is the only deliverable of the present planning task. No source, tests, configs, results,
HISTORY.md, SECOND_CHAPTER.md, or environment files are changed now. During future implementation,
HISTORY is updated continuously after each resolved attempt; SECOND_CHAPTER waits for a verified
milestone, exactly as the project journal rules require.

Experiments A–D are complete. Their negative results do not cancel E–H and do not justify tuning E–H
to obtain a more favorable result. Implementation passes when the registered questions are answered
correctly and transparently; it does **not** require an informative path, an HR result, improved
fusion, or statistical significance.

### 0.1 Governing invariants

1. Every reported predictive or attribution result uses outer leave-one-subject-out folds from
   `dehyd.eval.splits.nested_loso_splits`.
2. Every fitted quantity is estimated from the applicable training subjects only. This includes
   scalers, residual means, ridge penalties, base-model selection, fusion weights, bootstrap
   multiplicities, and any feature ablation refit.
3. Outer-test outcomes are used once for reporting. They never select features, paths, models,
   fusion weights, thresholds, or narrative categories.
4. The primary target remains signed fluid loss in percentage points of baseline mass, `delta_m_pct`.
5. Subject is the independent inferential unit. Sessions remain clustered within subject; frames are
   never inferential observations.
6. 10 GHz remains the headline modality. Any fusion claim is restricted to the original evaluable
   dual-band cohort.
7. All effects, confidence intervals, and p-values are conditional/exploratory in this 16-subject
   feasibility study. No causal hydration or clinical-readiness claim is permitted.

### 0.2 Explicit Milestone-10 protocol amendments

These decisions are made after A–D outcomes were available. They must be described with that true
chronology when the implementation plan and thesis chapter are later updated. Acceptance of this plan
is the authorization gate; implementation must not begin if any amendment is rejected.

| ID | Decision | Reason |
|---|---|---|
| **A-M10-1** | Replace Exp E's frozen standalone 4-fold permutation CV with the already-documented **leave-one-path-group-out refit under outer LOSO**. | The 4-fold design violates the project-wide requirement that every reported result use LOSO and is undefined for incomplete validation trajectories. The replacement is the documented alternative in `implementation_plan.md` §E and preserves the fixed Exp-E feature/model configuration. |
| **A-M10-2** | Record the ROADMAP heart-rate analysis as **not estimable from the delivered data**. Run the frozen clock/static-covariate nested ridge analysis as a separately named available-covariate sensitivity analysis, not as proof that HR was controlled. | The repository contains only radar files and the weight workbook. The workbook has name, age, height, five masses, loss, and notes; no HR field or external HR file is delivered. Temperature logs are lost and glucose was never measured. |
| **A-M10-3** | Exp G's base OOF predictions used to select `alpha` are produced by **selection-honest nested cross-fitting** inside each outer-training set. | `InnerResult.val_predictions` stores only first-seed predictions, and the eventual winner was selected using all inner-validation outcomes. Reusing those rows would violate the five-seed contract and contaminate the meta-training inputs with candidate-selection information. |
| **A-M10-4** | Exp G implements the pre-registered constrained decision-level combiner only. The unspecified feature-level variant is explicitly deferred and is not a Milestone-10 completion criterion. | No feature-level learner, reduction rule, or search budget was frozen. Inventing one after observing two null single-band arms would create post-hoc tuning and unnecessary complexity. |
| **A-M10-5** | The `R=200` full-procedure resamples produce a **selection-variance robustness distribution** summarized by its empirical 2.5th and 97.5th percentiles, not a BCa CI. | BCa requires the observed statistic and an original-subject delete-one jackknife. Applying BCa to an arbitrary vector of already-bootstrapped estimates is invalid. Existing `B=10000` subject-cluster BCa intervals remain the formal conditional intervals. |
| **A-M10-6** | Exp E reporting is outcome-neutral. The fixed model's weak predictive context is stated before interpreting attribution, but the path table is not pre-labelled as “null” or “physical.” | Attribution describes model reliance/predictive contribution; it cannot prove or disprove a dielectric mechanism. A desired narrative must not be encoded as a software acceptance criterion. |

## 1. Verified repository contracts and dependencies

The implementation must use the repository that exists, not an imagined uniform experiment API.

### 1.1 Data and analysis keys

- `exp_a.build_sessions(config, band)` returns one eligible record per
  `(subject, session_idx)`, sorted by that key, with `session_name`, `rel_path`, `frame_ids`, and
  `delta_m_pct`.
- The canonical session key in this codebase is **`subject, session_idx`**. New tables use those
  exact names. `session_name` is carried as a readable label but is not a join key.
- `GroundTruth.sessions` provides `mass_kg`, `delta_m_kg`, and `delta_m_pct`.
- `GroundTruth.subjects` provides `age`, `height_cm`, `baseline_mass_kg`, and `bmi`.
- Exp F's config-to-data mapping is frozen as:
  `age -> age`, `height -> height_cm`, `baseline_mass -> baseline_mass_kg`, `bmi -> bmi`.
- The delivered data inventory contains 80 10 GHz radar files, 80 77 GHz radar files, and
  `data/weight/metadata_subjects_info.xlsx`. No heart-rate file or column is present.

### 1.2 Feature representation and WST metadata

- Exp E interprets the **pooled WST session vector**, not raw flattened scattering. One model column
  is identified by the existing `session_feature_layout` tuple:
  `(frame_aggregate, channel, path_id, segment, statistic)`.
- One **path group** is every column sharing one Kymatio canonical `path_id`: both
  `frame_mean`/`frame_median` aggregates, every channel, and all global/half × mean/std columns.
- Existing session stores persist the path `order` array but do **not** persist `xi`, `j`, or
  `sigma`. Exp E therefore reconstructs the pinned Kymatio filter bank from the resolved config and
  calls the existing `wst.scattering_shape`. Before attribution, reconstructed `order` must equal the
  stored `order__{tiling}` array for every consumed session and its path count must match the model
  layout. Any mismatch stops the run.
- Kymatio 0.3.0 `xi` is normalized in cycles/sample. A finite center frequency is
  `xi_hz = xi * fs_hz`; `j` is a dyadic subsampling index and is not converted to Hz.

### 1.3 Existing evaluation and result contracts

- All folds come from `eval/splits.py`. Existing `Dataset`/`FeatureBundle` rows are session-level.
  M10 adds public `selection_folds(subject_ids, n_inner_max=5)`: it validates unique sorted
  subjects, requires at least two, and returns the same deterministic `InnerFold` objects now made
  by private `_inner_folds`. `nested_loso_splits` calls this public helper. No experiment constructs
  `GroupKFold` indices itself.
- `StoreBackedFeatures` implements frozen/tuned feature access. `SessionResidualFeatures` adds
  train-only session means and Exp-B residual targets.
- `InnerResult.val_predictions` has no session keys and preserves only the first seed. Exp G must not
  use it as its five-seed meta-training table.
- Exp A prediction CSVs have `subject, seed, y_true, y_pred` and no `session_idx`; they cannot be
  joined for fusion.
- Exp B prediction CSVs use `subject, session_idx, seed, y_true_residual,
  y_pred_residual, baseline_pred_residual`.
- Exp C has experiment-specific arm/ordinal schemas. Exp D comparison directories contain
  `metrics_exp_d_{band}.json` and `composite_{band}.csv`, not a generic prediction table.
- `metrics.py` already provides subject-cluster BCa/percentile fallback, pooled metric bootstrap,
  session-weighted bootstrap, mean-difference CI, Holm adjustment, Wilcoxon signed rank, regression
  metrics, and ordinal metrics. H reuses these results and functions; it does not rebuild them.
- M9's validated Exp-A reruns are reference controls for fold-specific feature selection:
  `results/runs/20260803T143704568296Z_f0a46aa6` (10 GHz) and
  `results/runs/20260803T151715023672Z_f0a46aa6` (77 GHz). Because M10 changes the harness and
  rebuilds both stores, F's authoritative source is instead a full Exp-A rerun at the final M10
  commit on each rebuilt store. Before any store rebuild, `validate_exp_a_reference.py --snapshot`
  validates the current M9 stores/runs and writes
  `results/milestone10/reference_exp_a_manifest.json`. It records canonical
  subject/session/frame-population hashes, fold/candidate enumeration, selection/prediction table
  hashes, and branch-aware feature evidence for every selected key: stored session-vector hashes for
  `off`/`frozen`; for `tuned`, raw-tensor/prelog/order hashes plus the fold-local training subjects,
  recomputed epsilon, reconstructed matrix hash, and prediction/score evidence.

  After the final M10 A runs, `validate_exp_a_reference.py --compare` reads only that immutable
  snapshot plus explicit final-run pointer files. It recreates the same branch-aware evidence on the
  final stores and writes `results/milestone10/exp_a_sources.json` containing approved absolute final
  and reference run paths/hashes. Any population, input, reconstructed-matrix, fold, candidate,
  selection, prediction, or score mismatch stops the milestone for scientific review; it is never
  excused as byte-neutral drift. `run_confound.py` requires `--exp-a-sources` and refuses glob/latest
  discovery. F records approved final-run paths and hashes only.
- The current workspace has no complete authoritative Exp-B result directory. This absence is a
  resolved inventory result, not a late discovery: M10 must run primary Exp B for both bands at the
  final commit and add those explicit directories to `run_manifest.json` before H assembly.
- Existing authoritative sources retained for assembly are Exp C
  `results/runs/20260803T143705048534Z_f0a46aa6` (10 GHz) and
  `results/runs/20260803T160645780475Z_f0a46aa6` (77 GHz), and Exp D
  `results/runs/20260806T104207854321Z_3f465abc` (10 GHz) and
  `results/runs/20260806T110156650286Z_3f465abc` (77 GHz). The additional C 10 GHz directory
  `20260803T172827484892Z_f0a46aa6` is a cross-vendor determinism control, not a headline source.
  The early inventory hashes and validates these paths; a failure requires retrieval/rerun, never
  fallback glob selection.

## 2. Frozen scientific designs

### 2.1 Experiment E — LOSO path-group ablation

**Question.** For the fixed, pre-registered Exp-E ridge model on Exp-B residual targets, how much
does held-out-subject residual MAE change when one WST path group is unavailable and the entire
fold-local pipeline is refit?

**Models and bands.** Run separately for 10 GHz and 77 GHz. Each band uses the existing `ExpEConfig`
feature/model anchor: 10 GHz `gate=(1,2)m`, reduction A, magnitude, T1, log off; 77 GHz T1_77,
I/Q, mean Rx fusion, log off; ridge `alpha=1.0`. This is a fixed model-form analysis, not the
“best model” selected from A/B outer results. It reuses Exp B's train-only residualization logic but
does not consume or rerun Exp B result artifacts.

**Fold computation.** For each selectable outer fold from `nested_loso_splits`:

1. Build `SessionResidualFeatures` using only outer-training subjects to fit session means.
2. Fit the full fixed ridge pipeline on all retained outer-training rows. Its StandardScaler is fit
   only on those rows. Score the held-out subject with equal-session residual MAE.
3. For each canonical `path_id`, delete its complete path-group column block **before** scaling,
   refit a new StandardScaler and ridge model on the identical outer-training rows, and score the
   identical held-out rows.
4. Store
   `importance_delta_mae_pct_points = ablated_mae - full_mae`.
   Positive means the full model predicted that subject better with the group; negative means
   removal improved prediction. One value is one path × one held-out subject, in residual
   `delta_m_pct` percentage points.
5. A fold with no evaluable held-out residual rows is recorded in the exclusion ledger and
   contributes no importance value. No other row is silently dropped.

There is no path selection, permutation repeat, p-value, or use of Exp-E attribution to retune a
model. The path ranking is descriptive and is produced only after all folds finish.

**Physical mapping.** Mapping is band- and order-aware:

- order 0: normalized low-frequency/scaling structure; no frequency, range, or reflected-level
  claim because per-signal standardization removed absolute level;
- 10 GHz order 1: `xi_1 * 520834` Hz is a fast-time beat-frequency center. Report the coarse scene
  range `c * f_b / (2 * slope)`, with `slope = bandwidth_hz / chirp_duration_s`. This is scene
  distance, not tissue penetration depth;
- 10 GHz order 2: `xi_1` retains the coarse beat/range label; `xi_2` is a first-order envelope
  modulation center, not a second range;
- 77 GHz order 1: `xi_1 * PRF` Hz is a slow-time Doppler/modulation-frequency magnitude. It is not
  range. Radial velocity is not reported because the I/Q channels are scattered separately and the
  acquisition/geometry do not support a signed velocity claim;
- 77 GHz order 2: `xi_2 * PRF` is an envelope-modulation center, not range or a second velocity.

Cole–Cole/dielectric discussion is limited to modality-level qualitative plausibility. A pooled,
standardized WST path does not identify instantaneous RF frequency, tissue water, or causality.

### 2.2 Experiment F — unavailable HR question plus available-covariate sensitivity

**Registered HR question.** The intended question was whether radar predictions are simply explained
by heart rate recorded before acquisition. The required HR observations are absent. The software
must therefore emit `status="not_estimable_missing_heart_rate"`, the inventory evidence, and
`n_hr_observations=0`. It must not correlate radar with a proxy, fabricate values, or label the
static-covariate analysis as an HR adjustment. If an HR file is supplied before implementation, stop
and amend this plan with its schema/alignment/QC rules before reading it.

**Available analysis.** Separately run four nested ridge models under outer LOSO, on S0–S4 eligible
session rows:

1. clock only;
2. clock + static covariates;
3. clock + radar;
4. clock + radar + static covariates.

Clock is session-index one-hot. Primary covariates are ordered exactly
`age, height_cm, baseline_mass_kg, bmi`. The config-to-workbook map is exactly
`{age: age, height: height_cm, baseline_mass: baseline_mass_kg, bmi: bmi}`; height is cm and mass is
kg.

Exactly three non-factorial analysis variants run:

- `pct_full`: percentage target, full four-covariate block, models 1–4;
- `pct_reduced`: percentage target, `age,height_cm` only; reuse the byte-identical `pct_full`
  predictions/selections for models 1 and 3 and rerun models 2 and 4 only;
- `kg_full`: signed `delta_m_kg = mass_kg - baseline_mass_kg` (negative remains loss), full
  covariate block, models 1–4. Its radar block retains the pre-specified percentage-target Exp-A
  feature key; radar features are not reselected on kg outcomes.

There is no combined kg-plus-reduced-covariate variant.

For each outer fold, models 3 and 4 use the same Exp-A-selected **feature key** from the authoritative
final-M10 selection table described in §1.3. They do not reuse an Exp-A fitted estimator. Each of
models 1–4 selects its own
ridge alpha inside the outer-training subjects:

- candidates are the ordered `ExpFConfig.ridge_alphas` tuple;
- each inner fold fits a fresh StandardScaler and ridge pipeline on inner-training rows only;
- inner scoring is subject-balanced MAE over validation session rows;
- each alpha becomes a `CandidateScore` with identical simplicity rank and that model's fixed feature
  dimension; `select_candidate` applies mean MAE, feature dimension, and inner-fold variance. A full
  tie resolves to the first alpha in the frozen ordered tuple, matching `select_candidate`'s stable
  input-order rule.

Models 3 and 4 share byte-identical radar columns within a fold. All four models are refit on the
outer-training subjects and predict only the held-out subject.

**Contrasts.** For each subject, compute mean session MAE and paired difference “with component minus
without component”; negative means improvement.

- the **only primary family** is `pct_full`: Holm-2 over model 3 − model 1 and model 4 − model 2;
- `pct_full` model 2 − model 1 and model 4 − model 3 are exploratory and reported individually;
- every `pct_reduced` and `kg_full` contrast is a sensitivity result. Report its unadjusted
  two-sided p-value/effect/CI with `multiplicity_family="none_sensitivity"`; do not create a new
  Holm family, pool variants, or call any of the six radar contrasts primary.

Report mean difference with subject-cluster CI, median difference, two-sided Wilcoxon statistic and
p-value, number of paired subjects, number of nonzero pairs, and ties. The conclusion remains a
limited clock/static-covariate sensitivity result. Temperature and glucose remain uncontrolled.

### 2.3 Experiment G — matched-session decision fusion

**Population.** Build the two band session spines independently, then inner-join on unique
`(subject, session_idx)`. Fail on duplicates, unequal `delta_m_pct`, inconsistent session names, or
non-finite targets. Canonically sort by `subject, session_idx`. Record:

- subjects and cells in each band before matching;
- `n_subjects_g` with at least one matched cell;
- `n_matched_cells`;
- every unmatched cell with `subject, session_idx, missing_band, reason`;
- sessions per matched subject.

Every 10-only, 77-only, equal-weight, and learned-fusion model is trained and scored on this exact
matched cell population. No frame-to-frame alignment is attempted; the radar front ends and frame
counts differ.

**Outer LOSO.** For each outer test subject `s`, let `T_s` be the other matched subjects.

**Selection-honest meta-training predictions for alpha.** Use the `inner_folds` already attached to
outer fold `s`; each `val_subjects` set is one meta-validation group `V`, so the meta partition is
identical to the repository's ordinary outer-fold inner partition. For each `V` and each band:

1. Call public `selection_folds(sorted(T_s - V), n_inner_max=5)` and run the complete Exp-A staged
   selection over those further folds. Stage scores aggregate exactly as ordinary Exp A: the
   subject-balanced validation MAE is averaged across all returned folds for each seed and then
   across configured seeds before `select_candidate`. No fold indices are built in `exp_g.py`.
2. Refit that selected pipeline on every matched session of `T_s \ V`.
3. Predict the matched sessions of `V` for every configured seed. A deterministic winning family is
   evaluated once and copied to the five configured seed **labels**, with
   `deterministic_source_seed` recorded; it is not counted as five independent observations.
4. Emit exactly one row per `(band, outer_test_subject, subject, session_idx, seed)`.

Concatenate meta-validation groups and require exact key equality between bands. Thus neither fitting
nor candidate selection saw a subject whose OOF prediction is used to fit alpha.

`selection_folds` fails closed if fewer than two selection-training subjects remain. Candidate-level
viability uses the existing harness rules. If no candidate survives any required further fold, the
entire outer fold `s` is non-evaluable for learned fusion; no partial meta-validation coverage is
used. The exclusion record names `outer_test_subject`, `meta_fold`, `band`, and reason.

For each alpha in `{0, .05, ..., 1}` and seed `k`, calculate subject-balanced OOF MAE across `T_s`.
Select one alpha from the mean objective across the five paired seed labels. Ties choose the value
closest to 1.0. Record the entire objective grid and selected alpha for the outer fold.

**Outer-test base predictions.** Separately, for each band, run the ordinary full Exp-A staged
selection on all of `T_s`, refit its winner on `T_s`, and predict `s` per seed. Apply the frozen alpha:

`pred_fused = alpha * pred_10 + (1 - alpha) * pred_77`.

Also save `pred_equal_weight` for alpha 0.5. Keep five labeled seed prediction sets; never average
predictions before scoring.

**Primary fusion estimand.** For subject `s`:

`d_s = mean_over_seed(MAE_s(fused) - MAE_s(10_only))`.

The headline is `mean_over_subject(d_s)` with `mean_difference_ci`; negative favors fusion. Report
the full per-subject distribution. 77-only and equal-weight comparisons are descriptive secondary
outputs with CIs and no additional p-value family. A positive or negative fusion result is reported
as observed; it cannot rescue the failed single-band validation or generalize beyond this cohort.

### 2.4 Experiment H — statistics and robustness

**Primary model-versus-baseline comparison.** H does not choose a “best” model from outer results.
The pre-registered primary comparison remains Exp A's radar selection procedure versus the
session-index-only baseline, using the M9 Exp-D comparison artifact on the identical subject folds.
The paired scalar is each subject's seed-averaged session MAE difference `radar - baseline`.
Alternative is two-sided because either direction is scientifically relevant. Report the existing
Wilcoxon result plus mean difference/BCa CI, median difference, ties, nonzero N, total N, and sign.
The composite learned baseline is secondary; the three per-family comparisons retain their frozen
Holm-3 exploratory treatment. H adds no cross-experiment omnibus p-value and performs no
outer-score-based model selection.

**Conditional confidence intervals.** Reuse or validate the existing `B=10000`, seed-controlled,
subject-cluster BCa intervals (percentile fallback recorded) for A–D. Compute F/G intervals under
their definitions above. Undefined Pearson/QWK replicates are skipped per metric and counted; over
5% skipped marks the interval unreliable. Correlation remains pooled with subject-cluster resampling;
per-subject correlation is descriptive only, and constant input is explicitly NaN. Every row reports
contributing subjects and sessions.

**Selection-variance robustness distribution.** Apply `R=200` full-procedure subject resamples to:

- Exp A, both bands: selected radar subject-balanced MAE and radar − session-index MAE difference;
- Exp B, both bands: the primary equal-session aggregate radar − baseline difference;
- Exp C, both bands: class-unit MAE for arm a and arm b. Adjacent accuracy and QWK retain their
  existing conditional CIs and are not given a refit-robustness range.

One replicate draws `N` subject IDs with replacement and stores multiplicity `m_s`. LOSO roles are
constructed over distinct drawn subjects; every copy of one original subject always has one role.
Replicates with fewer than four distinct subjects are skipped. C additionally requires all five
classes across its resampled cohort. These are coarse prechecks, not permission to summarize a
partial run. For each requested experiment/band/arm or contrast, every required distinct outer
subject must produce its complete OOF result. If any nested selection has no surviving candidate,
any outer prediction is absent, or Exp B's four-session primary aggregate is unavailable, that
whole result replicate is skipped with the first canonical reason; it is never computed over the
remaining easier folds. Reasons are counted. Fewer than 100 successful replicates makes the
robustness result inconclusive.

Multiplicity must reach the complete procedure:

- the provider supplies the original unique session rows plus `m_s`;
- tuned-WST epsilon repeats each training subject's per-subject scale exactly `m_s` times before
  taking the median; this equals an explicitly duplicated bootstrap cohort;
- weighted families pass row multiplicity to both
  `StandardScaler.fit(sample_weight=...)` and estimator `fit(sample_weight=...)`;
- KNN deterministically repeats training rows by `m_s` after role assignment; row order is original
  canonical order with each row repeated contiguously; its scaler and model see the same expansion;
- Exp A's session-index baseline fits multiplicity-weighted per-session and global training means
  and audits both effective denominators;
- Exp C recomputes class weights from effective bootstrap counts: with
  `n_eff=sum_rows(m_s)` and `n_c_eff=sum_rows_in_class_c(m_s)`, each unique training row receives
  estimator weight `m_s * n_eff / (K_present * n_c_eff)`; its scaler uses `m_s` only;
- Exp C arm-a cutpoints use `np.quantile` on in-sample predictions repeated contiguously by `m_s`
  with `method="linear"`, then apply the frozen strictly-increasing correction;
- Exp B's train-only session means use `m_s` as subject-copy weights and preserve the existing
  distinct-subject minimum-viability rule;
- inner-validation objectives and outer replicate summaries weight each subject by `m_s`; for
  pooled/ordinal metrics, evaluation rows are deterministically repeated by `m_s` before metric
  calculation;
- fit-audit records include the distinct fitted subject set, multiplicity map, weighting mode, and
  effective weighted row count. No held-out subject appears in any fitted map.

Report the original full-cohort point estimate separately, plus successful/attempted counts, skip
reasons, replicate mean, median, SD, and empirical 2.5th–97.5th percentile range. Label the latter
`selection_variance_empirical_95pct_range`, not `ci_method=bca`.

**RNG and percentile freeze.** For each `(experiment, band, replicate)`, construct a NumPy
`SeedSequence` from the integer tuple
`[robustness_seed, experiment_code, band_code, replicate]`, where the versioned enum maps
`a,b,c -> 1,2,3` and `10ghz,77ghz -> 10,77`. The resulting subject multiplicity draw is shared by
all arms/contrasts of that experiment-band replicate. Candidate model seeds remain the configured
model seeds and are not derived from the resampling seed. Save the tuple and generated 128-bit seed
state. Endpoints are exactly
`np.quantile(successful_estimates, [0.025, 0.975], method="linear")` after sorting by replicate ID.

## 3. Exact saved-artifact contracts

Every numerical artifact is UTF-8 CSV or JSON, finite unless an explicitly defined metric is
undefined, and accompanied by `provenance.json`. Figures read saved tables; they do not recompute
models or hidden statistics.

| Stage | Required artifact | Required content |
|---|---|---|
| A reference gate | `reference_exp_a_manifest.json` | pre-rebuild M9 run/store paths and SHA-256; population/fold/candidate/table evidence; per-selected-key branch-aware off/frozen vector hashes or tuned raw/prelog/order/epsilon/matrix evidence |
| A reference gate | `exp_a_sources.json` | approved explicit final/reference run paths and hashes for both bands, comparison status for every evidence class, final store/commit/config hashes; F refuses any non-approved status |
| E | `importance_folds_{band}.csv` | `band, outer_fold, test_subject, path_id, scattering_order, feature_group_size, n_test_sessions, full_mae_pct_points, ablated_mae_pct_points, importance_delta_mae_pct_points` |
| E | `path_metadata_{band}.csv` | `band, input_domain, path_id, scattering_order, xi1_normalized, xi2_normalized, sigma1_normalized, sigma2_normalized, j1, j2, xi1_hz, xi2_hz, coarse_range_m, physical_label, claim_limit`; non-applicable numeric fields are blank, not invented |
| E | `importance_summary_{band}.csv` | metadata join plus `n_subjects, mean, median, sd, q25, q75, min, max` of path importance; deterministic sort `scattering_order, path_id` |
| E | `ridge_coefficients_{band}.csv` | **full fixed model only**, never ablation refits: `model_variant="full", outer_fold, test_subject, feature_index, frame_aggregate, channel, path_id, segment, statistic, coefficient_per_training_sd, scaler_mean, scaler_scale`; coefficients are descriptive |
| E | `metrics_exp_e_{band}.json`, `interpretability_map_{band}.png`, `exclusions_e_{band}.csv` | fixed design/config, subject/path counts, source store fingerprints, descriptive status, figure regenerated only from the two CSVs |
| F | `confound_availability.csv` | one row per HR/temperature/glucose/static-covariate source: `variable, availability, source_checked, observation_unit, n_values, reason`; HR status is missing/not estimable |
| F | `predictions_f_{band}.csv` | `band, outer_fold, subject, session_idx, model_id, analysis_variant, target_name, seed, y_true, y_pred`; deterministic ridge uses the first configured seed label once |
| F | `selection_f_{band}.csv` | `outer_fold, test_subject, source_exp_a_final_run, source_selection_sha256, feature_key, model_id, analysis_variant, selected_alpha, inner_score, inner_score_variance, n_inner_folds` |
| F | `contrasts_f_{band}.csv` | `subject, contrast_id, analysis_variant, target_name, n_sessions, mae_with, mae_without, difference_with_minus_without` |
| F | `metrics_exp_f_{band}.json`, `exclusions_f_{band}.csv` | HR status; primary Holm-2 and exploratory contrast summaries; N/ties/nonzero N; CI/Wilcoxon fields; exclusion reasons |
| G | `matched_population.csv` | one row per matched `subject, session_idx` with `session_name, delta_m_pct, n_frames_10, n_frames_77`; unique sorted keys |
| G | `unmatched_population.csv` | `subject, session_idx, missing_band, reason` |
| G | `fusion_meta_oof.csv` | `outer_test_subject, meta_fold, band, subject, session_idx, seed, deterministic_source_seed, selection_record_id, y_true, y_pred`; exact five-seed label/key coverage |
| G | `fusion_base_selection.csv` | one row per `outer_test_subject, meta_fold_or_outer_final, band, stage, candidate`; stable `selection_record_id`, train/validation subject JSON and SHA-256, feature key, active axes JSON, family, params JSON, candidate score/variance/fold count, configured seed set, selected flag; outer-final rows have validation blank and identify the winner used on the outer test subject |
| G | `fit_audit_g.csv` | `outer_test_subject, meta_fold_or_outer_final, band, quantity, role, fitted_subjects_json, fitted_subjects_sha256, predicted_subjects_json, selection_record_id`; every scaler/model/selection/alpha fit is represented |
| G | `fusion_alpha_grid.csv` | `outer_test_subject, alpha, seed, subject_balanced_oof_mae, mean_over_seeds, selected` |
| G | `predictions_g.csv` | `outer_test_subject, subject, session_idx, seed, y_true, pred_10, pred_77, pred_equal_weight, pred_fused, alpha` |
| G | `per_subject_g.csv` | `subject, n_sessions, mae_10, mae_77, mae_equal_weight, mae_fused, difference_fused_minus_10` after frozen seed collapse |
| G | `metrics_exp_g.json`, `fusion_comparison.png`, `exclusions_g.csv` | `n_subjects_g`, matched cells, alpha by fold, primary CI/sign, secondary summaries, cohort-only limitation, and SHA-256 of the base-selection/fit-audit tables; plot from saved tables |
| H robustness | `robustness_replicates.csv` | `experiment, band, arm_or_contrast, replicate, robustness_seed_tuple_json, generated_seed_state, multiplicity_json, n_distinct_subjects, status, skip_reason, estimate` |
| H robustness | `robustness_selection.csv` | one row per `experiment, band, arm_or_contrast, replicate, outer_test_subject, stage, candidate`; selected feature key, active axes, family/params, score/variance/fold count, model seeds, selected flag, multiplicity SHA-256 |
| H robustness | `fit_audit_robustness.csv` | one row per real fit: `experiment, band, arm_or_contrast, replicate, outer_test_subject, stage, quantity, role, fitted_subjects_sha256, multiplicity_sha256, weighting_mode, effective_weighted_row_count`; companion JSON stores canonical subject/multiplicity maps keyed by hash |
| H robustness | `robustness_summary.csv`, `metrics_robustness.json` | original point, R, successes, skip counts, mean/median/SD, empirical percentile endpoints, label `selection_variance_empirical_95pct_range` |
| H assembly | `run_manifest.json` | explicit mapping from each A–G experiment/band to one authoritative run directory, required relative artifacts, SHA-256, source commit, resolved-config hash; no glob discovery |
| H assembly | `headline_metrics.csv` | `experiment, band, model_or_contrast, metric, estimate, ci_low, ci_high, ci_method, n_subjects, n_sessions, status, primary_or_secondary, source_run, source_artifact` |
| H assembly | `per_subject_results.csv` | `experiment, band, subject, model_or_contrast, metric, value, n_sessions, source_run`; no frame rows |
| H assembly | `paired_comparisons.csv`, `analysis_exclusions.csv`, `metrics_milestone10.json` | primary/secondary status, direction, p/effect fields, N and exclusions; exact source lineage |

All tables additionally include a schema version in their companion JSON. Writers fail closed on
duplicates, unknown columns where strict schemas apply, missing required artifacts, non-finite core
values, target disagreement, or lineage mismatch.

## 4. Future implementation structure and sequence

Favor plain functions and small dataclasses for returned records. Do not introduce a plugin system,
generic experiment framework, inheritance hierarchy, database, or caching layer.

### 4.1 Planned files

- `src/dehyd/eval/exp_e.py` — LOSO fixed-model full/path-ablated fits, path metadata, E summaries.
- `src/dehyd/eval/exp_f.py` — HR inventory report, four nested ridge models, contrasts.
- `src/dehyd/eval/exp_g.py` — matched population, nested meta-cross-fitting, alpha selection, G summary.
- `src/dehyd/eval/robustness.py` — multiplicity-aware A/B/C resampling and empirical range.
- `src/dehyd/eval/assembly.py` — explicit experiment-specific adapters and final tables.
- `src/dehyd/eval/splits.py` — public deterministic `selection_folds` used by ordinary nested LOSO
  and G's further selection folds; remains the only index-construction module.
- `src/dehyd/eval/harness.py` — one backward-compatible optional subject-multiplicity path; defaults
  must leave A–D byte-neutral.
- `src/dehyd/eval/exp_a.py`, `exp_b.py`, `exp_c.py` — propagate optional subject/row multiplicity
  through the existing canonical feature providers and A/B/C orchestration; robustness never copies
  their candidate enumeration or selection logic.
- `src/dehyd/models/baselines.py` — optional multiplicity for the session-index baseline and Exp-B
  session means.
- `src/dehyd/models/ordinal.py` — optional raw row multiplicity for effective arm-a/arm-b class
  weights and arm-a cutpoint quantiles.
- `src/dehyd/models/regressors.py` — estimator sample-weight capability table and explicit weighted
  pipeline fit dispatch.
- `src/dehyd/eval/selection.py` — small `select_alpha` grid argmin with closest-to-one tie-break.
- Five thin entrypoints: `run_interpretability.py`, `run_confound.py`, `run_fusion.py`,
  `run_robustness.py`, `run_stats_assembly.py`; plus `validate_exp_a_reference.py`.
- `experiments/run_regression.py` and `run_clock_decoupling.py` gain optional `--run-dir-out PATH`:
  after a successful complete run they atomically write the absolute run directory, and never write
  it on failure. No score or model behavior changes.
- `scripts/ibex/run_exp_e.sbatch`, `run_exp_f.sbatch`, `run_exp_g.sbatch`,
  `run_robustness.sbatch`, and `run_stats_assembly.sbatch`, following the existing git-free
  `REVISION` pattern and executing the payloads frozen in §6.
- Existing `scripts/ibex/run_exp_a.sbatch` and `run_exp_b.sbatch` pass optional `RUN_DIR_OUT` through
  to their entrypoints; absent values preserve their current commands byte-for-byte.

The backward-compatible multiplicity signatures are explicit:

- provider/baseline/session-mean functions add keyword-only `subject_multiplicity: Mapping[int,int]
  | None = None`;
- harness fit/scoring calls add keyword-only `row_multiplicity: ndarray | None = None`;
- `ThresholdedOrdinalRegressor.fit` and `CumulativeOrdinalClassifier.fit` add keyword-only
  `row_multiplicity=None`, passed as `model__row_multiplicity` by weighted pipeline dispatch;
- an arm-a KNN path is expanded before the pipeline and receives `row_multiplicity=None` thereafter,
  preventing double weighting;
- every `None` default executes the current statements and produces byte-identical A–D outputs.

Arm a derives its multiclass effective weights from `row_multiplicity`; arm b recomputes each
threshold's binary inverse-frequency weights from the two effective counts before multiplying by
row multiplicity. Arm-a in-sample predictions are repeated only for cutpoint estimation. These
rules live in `models/ordinal.py`, not in a robustness-only estimator clone.

### 4.2 Ordered implementation milestones

1. **Protocol, inventory, and schema pin.** Apply A-M10-1..6 to the authoritative implementation
   plan/configs, add artifact-schema constants/tests, pin current A–D byte behavior, and write a
   source-artifact inventory before structural edits. The inventory must record that the two M9 A
   runs are reference controls and that B has no currently available authoritative run, so final
   M10 A and B reruns are scheduled rather than discovered ad hoc. While the M9 stores are still
   present, create and validate the immutable branch-aware `reference_exp_a_manifest.json`; absence
   or mismatch of that snapshot blocks every later store edit/rebuild.
2. **Multiplicity foundation.** Add optional multiplicity to the provider/harness/scoring path and
   estimator dispatch. Test scaler/model weights, KNN expansion, Exp-B weighted residual means,
   Exp-C combined weights, fit audit, and byte-neutral default behavior. Land this as an isolated
   green commit if desired; do not build stores yet.
3. **H robustness driver.** Reuse A/B/C candidate enumeration and orchestration with the exact
   multiplicity contract. A smoke with `R=8` must intentionally report inconclusive under the real
   `min_successful=100`; the threshold is never scaled.
4. **Exp G.** First expose/test `selection_folds` in `splits.py`; then implement matched keys,
   selection-honest nested cross-fitting, five-seed OOF and base-selection/audit tables, alpha
   selection, outer refits, additive per-subject contrast, and artifacts.
5. **Exp E.** Implement LOSO path-group ablation and band-aware reconstructed metadata.
6. **Exp F.** Implement availability record, required approved-final-M10 Exp-A source manifest,
   fold-local alpha selection, four models, exact sensitivities, and contrasts.
7. **Drivers and assembly.** Implement experiment-specific artifact readers and explicit run map;
   no filename globbing or assumed uniform schema.
8. **Independent tests.** A tester independent of the sole implementation writer runs targeted
   risk tests, unchanged `tests/test_no_leakage.py`, the full suite, and the real-data suite on the
   candidate tree. All must be green before handoff to review.
9. **Independent read-only code review.** Review the same tested candidate for scientific,
   leakage, lineage, and statistical defects. Any blocker/high finding returns to the original
   writer; the reviewer does not implement corrections.
10. **Corrections and mandatory retest.** The original writer fixes every accepted finding and adds
    targeted regression coverage. An independent tester reruns the affected tests plus
    `test_no_leakage.py`, the full suite, and the real-data suite. The final-analysis gate accepts
    only this post-correction green tree, not the earlier pre-review result.
11. **Final analysis commit and stores.** After every source/config/test correction lands, stamp
   `REVISION` once, rebuild both feature stores once, and validate them. Multiple green commits are
   allowed; the optimization is one final store rebuild, not one risky uncommitted code wave.
12. **Reference controls and missing inputs.** On the final stores/commit, run full Exp A and B for
    both bands. Make final A the authoritative F selection source, enforce the §1.3 real-data
    comparison with M9 A, and register the new B directories. A mismatch stops downstream jobs.
13. **Mechanism-only local smokes**, then full E/F/G and `R=200` robustness jobs on IBEX CPU.
14. **Assembly.** Supply exact run directories in `run_manifest.json`, validate every source, write
    final numerical tables, and only then update SECOND_CHAPTER.md.

HISTORY.md is different: during future implementation, append a checkpoint immediately after every
resolved edit/experiment, smoke, failed or successful IBEX job, rerun, and review correction, with
the exact parameters and reason. It is never deferred to assembly. This planning-only review does
not edit the journal.

Exp E has no Exp-B artifact rerun dependency. It reuses code for residualization. F uses the
final-commit Exp-A selection artifacts after the M9 comparison gate. G must refit on the matched
population. H uses explicitly inventoried A–D numerical artifacts and does not select a new winner
from their outer scores.

Workflow stop conditions are fail-closed: no structural edit/store rebuild without the validated
immutable M9 A snapshot; no final A/B/E/F/G/H job before the post-review retest gate; no F or
assembly after an A-reference mismatch, missing B source, invalid store, or schema/lineage failure.
The IBEX launch matrix remains serialized in the displayed dependency order because jobs create
lineage-bearing result state. Independent read-only review/exploration may run in parallel; jobs
that consume one another's artifacts may not.

## 5. Risk-based test plan

### 5.1 Leakage and fold tests

- Every E/F/G reported outer fold has exactly one test subject from `nested_loso_splits`.
- Mutating an outer-test subject's target/features/covariates leaves every train-derived fit,
  feature key, alpha candidate score, fusion alpha, and robustness training state unchanged.
- E ablated scalers are refit on outer train only; the full and ablated models use identical rows.
- F models 3/4 share byte-identical radar matrices; models 1/2 never read the store.
- G meta-validation subject outcomes cannot affect candidate selection that generated their base OOF
  predictions. A toy subject that flips a winner when included must still be predicted by the winner
  selected without that subject.
- `selection_folds` alone constructs G's second/third levels; at every level train/validation/test
  subject sets are disjoint and each expected validation subject appears exactly once.
- Learned alpha never sees the outer-test subject. Fit audit includes selection, scaler, model, and
  alpha subject sets.
- Robustness copies of one subject never occupy different roles; multiplicity affects transforms,
  fits, selection scoring, and final estimands exactly once.

### 5.2 E tests

- `len(session_feature_layout) == X.shape[1]`; every column maps to exactly one path group.
- Reconstructed order/path count equals every consumed store's order metadata.
- Hand-built order-0/1/2 metadata maps correctly for 10 GHz fast-time and 77 GHz slow-time; no 77 GHz
  path receives a range label.
- Synthetic signal path yields positive LOPGO importance; a correlated surrogate fixture shows why
  attribution is predictive/model-specific rather than causal.
- Path aggregation is deterministic and reports the correct contributing-subject count.

### 5.3 F tests

- Repository/workbook inventory yields zero HR observations and the exact not-estimable status.
- Config-to-column map and covariate order/units are exact.
- Every inner alpha fit has its own inner-train scaler; scoring is subject-balanced.
- Full alpha ties choose the first frozen alpha through `select_candidate` input order.
- Primary/sensitivity feature blocks and signed-kg target differ only as specified.
- The Exp-A source gate snapshots before rebuild and compares after rebuild: real
  subject/session/frame hashes; off/frozen stored vectors; tuned raw/prelog/order inputs,
  train-subject epsilon, reconstructed matrices; fold/candidate enumeration; selections,
  predictions, and scores. Any deliberate mismatch or missing explicit run pointer fails before F.
- Missing covariate or non-finite value fails with a named subject/column; no silent complete-case drop.

### 5.4 G tests

- Matched population accepts only unique equal-target keys; swapped band labels, duplicate keys,
  target disagreement, or unequal prediction-key sets fail.
- Exactly one OOF row exists per expected band/fold/subject/session/seed key.
- Every OOF and outer-final prediction resolves to one complete base-selection record and fit-audit
  chain containing feature axes, family/params/seeds, and disjoint fitted/predicted subjects.
- Deterministic winners may have equal numerical values across seed labels; tests require labeled
  coverage, not artificially distinct values.
- Alpha grid has 21 values and closest-to-one tie behavior.
- A fixture where test outcomes would choose another alpha proves those outcomes are never read.
- Primary contrast is the subject-additive `fused - 10` scalar and differs from a session-weighted
  calculation on an unequal-session fixture.

### 5.5 H/statistical tests

- Subject bootstrap samples subject clusters, not frames/sessions; fixed seeds reproduce endpoints.
- Paired inputs join by subject, preserve direction, and report N/ties; equal methods return zero
  effect and an explicitly undefined Wilcoxon p-value rather than raising.
- Constant predictions/targets make correlation explicitly undefined and increment skip counts.
- Robustness percentile endpoints equal hand-computed quantiles of successful replicate estimates;
  no fake jackknife or second bootstrap is called.
- `R=8` smoke is inconclusive with `min_successful=100`.
- Direct-equivalence fixtures compare the multiplicity implementation with an explicitly duplicated
  cohort for tuned epsilon, weighted/KNN scaler and model fits, the session-index baseline, Exp-B
  session means, Exp-C effective class weights, and arm-a cutpoint quantiles.
- Fixtures that pass the coarse distinct-subject/class check but fail one nested fold or Exp-B's
  four-session aggregate skip the whole requested result replicate; no partial-fold estimate exists.
- Fixed seed tuples reproduce multiplicity draws; all arms in one experiment-band replicate share
  the draw; percentile endpoints match NumPy's linear method exactly.
- Every successful real robustness estimate resolves to complete winner/feature and fit-audit rows;
  fitted-subject and multiplicity hashes agree with the replicate table, and effective row counts
  match explicit duplication.
- Assembly round-trips each actual A–D schema plus synthetic E–G schemas; missing/mismatched source
  artifacts fail closed. Run directories are supplied explicitly, never discovered by glob.

## 6. Exact future validation commands

Targeted local gates after each implementation stage:

```text
uv run pytest tests/test_metrics.py tests/test_regressors.py tests/test_harness.py tests/test_no_leakage.py
uv run pytest tests/test_exp_e.py tests/test_run_interpretability.py
uv run pytest tests/test_exp_f.py tests/test_run_confound.py
uv run pytest tests/test_exp_g.py tests/test_run_fusion.py
uv run pytest tests/test_robustness.py tests/test_run_robustness.py
uv run pytest tests/test_assembly.py tests/test_run_stats_assembly.py
uv run pytest tests/test_no_leakage.py -m "not realdata"
uv run pytest
uv run pytest --realdata
git diff --exit-code -- tests/test_no_leakage.py
```

Mechanism-only smokes after the final store rebuild:

```text
uv run python experiments/run_interpretability.py --config configs/exp_a_regression.yaml --config configs/exp_e.yaml --config configs/stats.yaml --band 10ghz --subset 6subjects
uv run python experiments/run_interpretability.py --config configs/exp_a_regression_77ghz.yaml --config configs/exp_e.yaml --config configs/stats.yaml --band 77ghz --subset 6subjects
uv run python experiments/run_confound.py --config configs/exp_a_regression.yaml --config configs/exp_f.yaml --config configs/stats.yaml --exp-a-sources results/milestone10/exp_a_sources.json --band 10ghz --subset 6subjects
uv run python experiments/run_confound.py --config configs/exp_a_regression_77ghz.yaml --config configs/exp_f.yaml --config configs/stats.yaml --exp-a-sources results/milestone10/exp_a_sources.json --band 77ghz --subset 6subjects
uv run python experiments/run_fusion.py --config-10 configs/exp_a_regression.yaml --config-77 configs/exp_a_regression_77ghz.yaml --shared-config configs/exp_g_fusion.yaml --shared-config configs/stats.yaml --subset 6subjects
uv run python experiments/run_robustness.py --config configs/exp_a_regression.yaml --config configs/stats.yaml --experiment a --band 10ghz --replicates 8 --subset 6subjects
```

`run_fusion.py` loads the two band configs separately, applies shared Exp-G/stats overlays to both,
and asserts shared run seeds, target definition, split constants, and weight workbook. It never tries
to merge two top-level configs through the config loader.

Before structural edits or a store rebuild, run the exact reference snapshot gate once from the
validated M9 tree/store state:

```text
uv run python experiments/validate_exp_a_reference.py --snapshot --reference-10 results/runs/20260803T143704568296Z_f0a46aa6 --reference-77 results/runs/20260803T151715023672Z_f0a46aa6 --output results/milestone10/reference_exp_a_manifest.json
```

The full-cohort launch matrix below is exact. Run it from the final stamped tree; heavy work is
always submitted to Slurm, never executed on a login shell. Each job must exit zero and produce the
stage's schema-listed artifacts before its directory is entered in the manifest:

```text
sbatch --wait scripts/ibex/extract10.sbatch
uv run --no-sync python experiments/extract_features.py --config configs/exp_a_regression.yaml --config configs/ibex.yaml --band 10ghz --validate
sbatch --wait scripts/ibex/extract77.sbatch
uv run --no-sync python experiments/extract_features.py --config configs/exp_a_regression_77ghz.yaml --config configs/ibex.yaml --band 77ghz --validate

sbatch --wait --export=ALL,BAND=10ghz,MODE=full,RUN_DIR_OUT=results/milestone10/sources/exp_a_10.txt scripts/ibex/run_exp_a.sbatch
sbatch --wait --export=ALL,BAND=77ghz,MODE=full,RUN_DIR_OUT=results/milestone10/sources/exp_a_77.txt scripts/ibex/run_exp_a.sbatch
uv run --no-sync python experiments/validate_exp_a_reference.py --compare --reference-manifest results/milestone10/reference_exp_a_manifest.json --final-10-file results/milestone10/sources/exp_a_10.txt --final-77-file results/milestone10/sources/exp_a_77.txt --output results/milestone10/exp_a_sources.json

sbatch --wait --export=ALL,BAND=10ghz,MODE=full,RUN_DIR_OUT=results/milestone10/sources/exp_b_10.txt scripts/ibex/run_exp_b.sbatch
sbatch --wait --export=ALL,BAND=77ghz,MODE=full,RUN_DIR_OUT=results/milestone10/sources/exp_b_77.txt scripts/ibex/run_exp_b.sbatch

sbatch --wait --export=ALL,BAND=10ghz,MODE=full scripts/ibex/run_exp_e.sbatch
sbatch --wait --export=ALL,BAND=77ghz,MODE=full scripts/ibex/run_exp_e.sbatch
sbatch --wait --export=ALL,BAND=10ghz,MODE=full,EXP_A_SOURCES=results/milestone10/exp_a_sources.json scripts/ibex/run_exp_f.sbatch
sbatch --wait --export=ALL,BAND=77ghz,MODE=full,EXP_A_SOURCES=results/milestone10/exp_a_sources.json scripts/ibex/run_exp_f.sbatch
sbatch --wait --export=ALL,MODE=full scripts/ibex/run_exp_g.sbatch

sbatch --wait --export=ALL,EXPERIMENT=a,BAND=10ghz,MODE=full,REPLICATES=200 scripts/ibex/run_robustness.sbatch
sbatch --wait --export=ALL,EXPERIMENT=a,BAND=77ghz,MODE=full,REPLICATES=200 scripts/ibex/run_robustness.sbatch
sbatch --wait --export=ALL,EXPERIMENT=b,BAND=10ghz,MODE=full,REPLICATES=200 scripts/ibex/run_robustness.sbatch
sbatch --wait --export=ALL,EXPERIMENT=b,BAND=77ghz,MODE=full,REPLICATES=200 scripts/ibex/run_robustness.sbatch
sbatch --wait --export=ALL,EXPERIMENT=c,BAND=10ghz,MODE=full,REPLICATES=200 scripts/ibex/run_robustness.sbatch
sbatch --wait --export=ALL,EXPERIMENT=c,BAND=77ghz,MODE=full,REPLICATES=200 scripts/ibex/run_robustness.sbatch

sbatch --wait --export=ALL,RUN_MANIFEST=results/milestone10/run_manifest.json,VALIDATE_ONLY=1 scripts/ibex/run_stats_assembly.sbatch
sbatch --wait --export=ALL,RUN_MANIFEST=results/milestone10/run_manifest.json,VALIDATE_ONLY=0 scripts/ibex/run_stats_assembly.sbatch
```

The wrappers map only the displayed environment variables to the direct Python payloads already
shown by the mechanism-smoke interface; wrapper tests assert the exact argv and nonzero propagation.
`run_exp_a.sbatch` and `run_exp_b.sbatch` pass `RUN_DIR_OUT` to the new CLI option. The other wrappers
print and atomically save their successful run directory under `results/milestone10/sources/` for
manifest construction; no wrapper infers a latest directory. Assembly's `--validate-only` performs schema,
SHA-256, config/store/commit, matched-population, and source-lineage checks and writes no numerical
summary. The second call must create all three H assembly rows listed in §3 plus
`metrics_milestone10.json`; omission of any matrix cell is fatal.

## 7. Objective acceptance criteria

### E passes when

- all reported folds are outer LOSO folds and all fitted quantities audit to outer train;
- every model column maps to verified metadata and every path has per-subject results;
- band-aware frequency/range labels and limitations are correct;
- required CSV/JSON/figure artifacts are reproducible from saved numerical tables;
- no desired sign, ranking, or “physical” story is required.

### F passes when

- HR absence is recorded as not estimable with inventory evidence;
- the four available-covariate models and two sensitivities obey nested LOSO/train-only fitting;
- contrast direction, Holm-2 primary family, exploratory status, missing-data policy, and N are saved;
- substantial covariate effects or no radar increment are both valid outcomes.

### G passes when

- matched keys and exclusions are auditable and all conditions use the same cells;
- meta OOF predictions are session-keyed, seed-keyed, and selection-honest;
- alpha uses only outer-training OOF rows and outer scoring remains one-subject-held-out;
- primary/additional comparisons and artifacts use the frozen estimands;
- fusion is not required to beat 10 GHz.

### H passes when

- every headline row traces to an explicit source run/artifact/config/commit and reports N;
- no frame is treated as independent and no new winner is chosen from outer results;
- conditional CIs, paired tests, effect sizes, multiplicity families, undefined metrics, and signs are
  explicit;
- robustness multiplicity is end-to-end, `R=200` rules are honored, and its empirical range is not
  mislabeled BCa;
- non-significance or an inconclusive robustness distribution is a valid scientific result.

### Milestone 10 passes when

all targeted/full/real-data tests are green, `tests/test_no_leakage.py` is unchanged, both rebuilt
stores validate at the final analysis commit, every required artifact exists and passes schema/
lineage checks, and an independent code review has no unresolved blocker/high scientific finding.

## 8. Known limitations and deferred work

- Heart-rate confounding is not tested because the measurement file is unavailable. Temperature and
  glucose are also uncontrolled.
- Exp E attributes a fixed predictive procedure; correlated WST paths and a weak underlying model
  limit interpretation. It does not establish dielectric causality.
- Fusion applies only to subjects/sessions with both eligible bands in the original cohort.
- Feature-level fusion is deferred under A-M10-4; it requires a new pre-specified design before use.
- The empirical robustness range has only 200 draws (roughly five observations per 2.5% tail) and is
  descriptive. Formal refit-aware BCa would require full delete-one-subject procedure refits.
- The fasting design remains confounded with time of day; Exp B/F reduce specific alternatives but do
  not causally isolate hydration.

### 8.1 Methodological basis for the amendments

- Group removal/refitting estimates loss of predictive value available from a feature group, not a
  causal mechanism: Williamson et al. (2021), DOI
  [10.1111/biom.13392](https://doi.org/10.1111/biom.13392), and Williamson et al. (2023), DOI
  [10.1080/01621459.2021.2003200](https://doi.org/10.1080/01621459.2021.2003200).
- Nested selection must exclude the validation observation whose prediction enters a higher-level
  learner: Varma & Simon (2006), DOI
  [10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91), and Cawley & Talbot (2010),
  [JMLR 11:2079–2107](https://jmlr.org/papers/v11/cawley10a.html).
- BCa needs an observed statistic, bootstrap distribution, and original-unit influence/jackknife
  information: Efron (1987), DOI
  [10.1080/01621459.1987.10478410](https://doi.org/10.1080/01621459.1987.10478410).
- Kymatio 0.3.0 defines `xi` as a normalized wavelet center frequency below the 0.5 Nyquist bound;
  the implementation uses the pinned
  [official filter-bank source](https://github.com/kymatio/kymatio/blob/v0.3.0/kymatio/scattering1d/filter_bank.py#L13-L16)
  rather than treating `j` as a physical frequency.

## 9. Workflow gate after plan acceptance

The future workflow is:

`accepted plan -> one implementation writer -> independent risk-based tests -> independent code review -> corrections and rerun -> documentation only after verification`.

The implementation writer should keep calculations visible in plain functions and inspectable
tables. The independent test pass should target leakage, key alignment, multiplicity, and statistical
edge cases. HISTORY checkpoints occur continuously as specified in §4.2; SECOND_CHAPTER and final
user-facing documentation wait for verified full-cohort artifacts. Both must disclose A-M10-1..6
and the observed results without outcome-based reframing.
