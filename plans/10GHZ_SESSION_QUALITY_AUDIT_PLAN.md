# PROPOSED PLAN — 10 GHz session-quality and repeatability audit

## 0. Status, purpose, and approval boundary

**Status: implemented and verified on 2026-08-29.**

This audit asks a question that the current binary QC does not answer:

> Is a 10 GHz recording internally healthy and repeatable, and how does it compare with
> the same subject's other recordings?

The first phase is descriptive only. It does **not** change the frozen preprocessing,
WST search space, eligible-session population, or Experiments A–D. It does not remove or
weight samples, tune a threshold, retrain a model, or replace any reported result.

Approval of this plan authorizes only the audit described here. Any later sensitivity
retraining requires a separate plan and approval gate.

## 1. Fixed decisions proposed for approval

1. **Use a component card, not one magic score.** A session receives separate values for
   availability, signal integrity, range/peak stability, absolute signal level, and WST
   repeatability. No arbitrary weighted 0–100 score is created.
2. **Audit all 80 raw subject-session files through a target-free inventory.** The radar-only
   path parses filenames, inspects/loads radar files, and runs the existing per-frame QC helpers
   directly. It must not call `build_manifest`, `apply_qc`, or `GroundTruth`, because the normal
   project manifest carries `delta_m_pct` and `class_label`. The existing 73 eligible sessions keep their
   current status. The seven ineligible sessions remain visible with their failure reasons;
   they are never imputed or silently dropped from the audit table.
3. **Use five stored-index blocks.** The real files contain 100 frames, so the canonical
   blocks are frame indices 0–19, 20–39, 40–59, 60–79, and 80–99. The loader preserves
   on-disk order, but no timestamp metadata independently proves chronology. Reports therefore
   say `stored-index block`, not `time block`, unless stronger provenance is found.
4. **Preserve failed-frame locations.** Blocks are formed from raw frame indices first and
   QC is applied second. Passing frames are never repacked into five equal groups, because that
   would hide a cluster of failures in one part of an acquisition.
5. **Use one fixed diagnostic WST family, never the fold-selected model feature.** The proposed
   primary diagnostic is reduction A (chirp mean), **magnitude**, pooled coefficients, log off,
   with all three already-frozen tilings reported separately. Magnitude is phase-insensitive;
   reduction A has no data-dependent peak isolation; log off avoids making the known order-2
   epsilon floor part of a quality definition; reporting all three tilings avoids declaring one
   the winner after seeing results. I/Q is a separately labelled sensitivity view and can never be
   combined with magnitude into one status.
6. **Keep body mass completely separate.** The quality card consumes radar and frozen constants
   only. Recorded-equal-mass pairs are a later sanity view and can never set a quality status,
   threshold, WST choice, exclusion, or training weight. Radar-only provenance and
   recorded-equal-mass provenance are different files; only the latter hashes the workbook.
7. **Use an audit-specific config loader.** `quality_audit_10ghz.yaml` names unchanged base project
   configs and audit outputs/constants. A small `load_quality_audit_config` validates it separately;
   the global `Config` schema and existing serialized config hashes are not extended or changed.
8. **Run locally on CPU.** This is a deterministic diagnostic over the 10 GHz cohort and does not
   require GPU training or IBEX. Provenance records the commit, configuration, package versions,
   machine/CPU information, raw-file hashes, and source-artifact hashes.
9. **Snapshot frozen inputs before work.** Hash every locally available existing file below
   `results/`, including feature-store `.npz` arrays, before the audit. Remote IBEX paths are not
   dereferenced; their local pointer/manifest files and embedded approved hashes are preserved.
   New code has a fail-closed output-root guard and may write only below
   `results/quality_10ghz/` and `figures/quality_10ghz/`. Recheck the snapshot afterward.

## 2. Existing evidence this audit builds on

The present QC already records, per frame, NaN/Inf, flatline/saturation, in-band energy ratio,
and a diagnostic chirp-RMS robust-z flag. A session is eligible when at least 50% of its actual
frames pass. The existing preprocessing diagnostic also defines band-gate energy retention,
ROI-to-total energy, detected peak bin, and peak share.

Those checks are useful but incomplete: the saved session report mainly says how many frames
passed. It does not preserve a graded session trust card, within-session frame dispersion, or
five-block repeatability. Session WST aggregation then reduces surviving frames to a mean and a
median, so that instability is not visible to the final model.

The audit reuses existing formulas; it does not alter their thresholds or reinterpret them as
hydration measurements.

## 3. Outputs and exact questions

### 3.1 Radar-only session quality card

Write one row for each of the 80 `(subject, session_idx)` cells. Proposed columns:

- identity: `subject`, `session_idx`, `session_name`, `rel_path`;
- frozen status: `n_raw`, `n_pass`, `pass_fraction`, existing eligibility, and failure counts
  by reason;
- block coverage: passing count for each of the five stored-index blocks and their minimum;
- signal integrity: RMS-flag fraction and the median and 10th percentile of in-band energy
  ratio; the 10th-percentile margin above the existing 0.30 threshold is also recorded;
- absolute level: per-frame raw RMS
  `sqrt(mean(abs(frame)^2))` over all 534 fast-time samples × 20 chirps, summarized by the
  session median and unscaled MAD, clearly labelled as hardware/geometry-sensitive rather than
  hydration-specific;
- preprocessing behaviour: median and robust spread of energy retention, ROI-to-total energy,
  peak share, detected peak bin, and detected-peak range; every "robust spread" in this plan means
  unscaled MAD unless the column name explicitly says IQR;
- stability: peak-bin mode share and peak-bin interquartile range;
- audit status and an explicit missing-reason field.

The population contract is fixed per component:

| component | frames consumed |
|---|---|
| QC incidence and failure counts | all raw frames |
| raw RMS | finite raw frames; non-finite count stays visible |
| in-band ratio, RMS flag | frames for which the existing QC diagnostic is defined |
| energy retention, ROI/total, peak share, peak/range | QC-passing frames, whether or not the session is eligible |
| WST repeatability | QC-passing frames of existing-QC-eligible sessions only |

Statuses are deliberately modest:

- `INELIGIBLE_EXISTING_QC`: the existing frozen eligibility rule failed;
- `REVIEW_BLOCK_COVERAGE`: the session is eligible overall, but at least one block contains
  fewer than 10 passing frames, so five-block repeatability is weakly supported;
- `REPEATABILITY_ANALYSABLE`: all five blocks contain at least 10 passing frames.

The value 10 is an owner-approved display/sufficiency policy (half of a nominal 20-frame block),
not a statistically learned cutoff and not a new sample-rejection threshold. It does not alter the
model population. A session receives the five-block WST metrics only when all five blocks meet this
rule; otherwise those session metrics are missing with reason `insufficient_block_coverage`.

### 3.2 WST within-session repeatability

For each eligible session with sufficient block coverage, form per-frame pooled WST vectors under
the fixed diagnostic family. Report each tiling and scattering order separately so large order-0
coefficients cannot hide order-1 or order-2 instability. Results from different tilings, orders, or
the I/Q sensitivity view must never be numerically ranked against each other.

Raw Euclidean distance over thousands of coefficients is not used directly. Within each frame,
tiling, and scattering order, group the pooled statistics belonging to one canonical WST path and
construct **two separate dimensionless views**:

1. **Within-path shape.** Normalize each path block to unit L2 norm, then concatenate the
   normalized paths with `1/sqrt(n_paths)` weighting. This asks whether the pattern inside a path
   changed while preventing a strong path from hiding every weak path.
2. **Across-path energy composition.** Form the vector of the original path-block L2 norms and
   normalize that whole vector once to unit L2 norm. This asks whether energy moved between paths;
   it must change when one of at least two active paths alone becomes stronger even if total raw
   RMS is unchanged. When an order has fewer than two paths (notably order 0), this view is
   `not_applicable_single_path`, not a fabricated constant-repeatability result.

The two views are reported independently and never averaged into a status. A common positive unit
change applied to the whole tiling/order leaves both views unchanged; a path-specific amplitude
change is expected to change the composition view. Absolute signal strength remains available
separately in the quality card. Define
`epsilon_path = 1e-12 * median(positive path norms in that session/tiling/order)`. A path norm at
or below this value maps to an all-zero normalized block and increments `n_near_zero_paths`; if no
positive path norm exists, the representation is missing with a named reason. Record vector
dimension, path count, near-zero-path count, and `view={within_path_shape,path_energy_composition}`
in every output row.

For stored-index block `b` and each view, let `c_b` be the componentwise median dimensionless vector and
let `v_b` be the median Euclidean distance of that block's passing frames from `c_b`. Define
`epsilon_distance = 1e-12 * median(positive frame-to-session-centroid distances)` in the same
dimensionless space. If that median does not exist because every distance is exactly zero, set
`epsilon_distance=0` and apply the exact-zero rule below. For every tiling/order, report both
typical and worst-block behaviour:

- median and **minimum** cosine similarity across the ten block-centroid pairs;
- median and **maximum** `separation_to_wobble`, where each pair is
  `||c_a-c_b|| / sqrt(v_a^2 + v_b^2 + epsilon_distance^2)`;
- **maximum block-to-session-centroid distance** across the five blocks;
- `max_leave_one_block_influence`: the largest change in the all-frame median centroid when one
  block is omitted, divided by
  `sqrt(session_wobble^2 + epsilon_distance^2)`;
- usable block count and all numerical/missing guards.

Median metrics describe typical behaviour; the maximum/minimum metrics are the checks for one bad
block. Leave-one-block influence is a robustness description, not an anomaly detector. For any
ratio, exact numerator=0 and exact denominator=0 is defined as 0 with `exactly_identical=true`;
nonzero numerator with exact denominator=0 is missing with reason
`zero_within_block_wobble_nonzero_between`. Raw numerators and denominators are always retained.

No population scaler, PCA, covariance estimate, learned anomaly detector, target, prediction, or
model error enters these calculations. Both normalizations are fixed, per-frame geometric
definitions, not fitted population transforms. The high-dimensional vectors are never treated as
independent inferential observations.

### 3.3 Within-subject comparison

Create a subject-by-session view that answers questions such as “How does subject 10 at 12pm
compare with subject 10's other sessions?” It shows the raw component values and deterministic
within-subject ranks only for metrics with an unambiguous audit direction. Ties receive the same
rank. Higher-is-steadier ranks are allowed for pass fraction, the low-quantile in-band margin,
peak-mode share, and cosine similarity. Lower-is-steadier ranks are allowed for failure/RMS-flag
fractions, peak-bin IQR, separation-to-wobble, block-to-session distance, and leave-one-block
influence. Raw RMS, energy retention, ROI/total, peak share, peak bin/range, near-zero-path count,
and every absolute location/level value are shown but **not ranked as better/worse**, because they
have no universal good direction.

Ranks are descriptive only. They do not define good/bad thresholds and do not feed any model.

### 3.4 Recorded-equal-mass sanity view — separate output

After the radar-only files are finalized and hashed, identify **adjacent sessions of the same
subject with exactly equal recorded mass**. Call them `recorded_equal_mass`, not physiologically
unchanged: the workbook is rounded and body mass is an imperfect proxy for local tissue water.

For each pair, diagnostic WST tiling/order, and view, define
`within_s = median(||c_a-c_b||)` over that session's ten stored-index block-centroid pairs.

Path validity is made common before comparing sessions. For the within-path-shape view, a path is
`session_active` when its median raw path norm over passing frames exceeds that session's
`epsilon_path`; the pair uses the intersection of the two sessions' active path sets and recomputes
both sessions' centroids, `within_s`, and `epsilon_distance_s` in that identical path space. An
empty intersection is missing with a named reason. The path-energy-composition view uses the complete canonical path set;
each session's below-floor path norms are zero before whole-vector normalization.

For each pair report:

- dimensionless between-session centroid distance under the same view-specific definition
  as §3.2, plus its raw numerator;
- `within_a` and `within_b` exactly as defined above;
- `between_to_within_ratio = between_session_distance /
  sqrt(within_a^2 + within_b^2 + epsilon_distance_a^2 + epsilon_distance_b^2)`.

The exact-zero and missing rules from §3.2 apply unchanged. Ratios are never compared across
tilings/orders/views and carry their raw numerator, denominator, both session-specific epsilon
values, common vector dimension/path count, and both near-zero-path counts.

A large ratio means “review this acquisition pair,” not “discard it” and not “the radar is
wrong.” Posture, body position, respiration, hardware geometry, and unmeasured physiology remain
possible explanations. Subject 10's required worked example is 8am to 10am; 10am to 12pm is not
an equal-mass pair.

Changing ground-truth values in a test fixture must change only this separate output. It must not
change any radar-only quality or repeatability artifact byte.

## 4. Planned files and artifacts

### 4.1 New implementation files

- `src/dehyd/quality/session_audit.py` — plain functions for block assignment, component
  summaries, WST repeatability, and subject-relative views;
- `src/dehyd/quality/radar_inventory_10ghz.py` — target-free filename/header inventory and
  direct `run_qc_cube` wiring; it has no ground-truth field or import;
- `src/dehyd/quality/config.py` — audit-specific frozen dataclass/schema and
  `load_quality_audit_config`, which composes an unchanged base project config without extending
  the global `Config`;
- `src/dehyd/quality/__init__.py`;
- `experiments/run_session_quality_audit.py` — thin, config-driven CLI;
- `configs/quality_audit_10ghz.yaml` — paths plus the fixed diagnostic declaration;
- `tests/test_session_quality.py`;
- `tests/test_run_session_quality_audit.py`.

Existing QC, preprocessing, WST, feature-store, experiment, and frozen configuration files are
inputs only. This audit must not modify their behaviour.

### 4.2 Regenerable outputs

- `results/quality_10ghz/session_quality_all_80.csv`;
- `results/quality_10ghz/block_quality.csv`;
- `results/quality_10ghz/wst_block_repeatability.csv`;
- `results/quality_10ghz/subject_relative_quality.csv`;
- `results/quality_10ghz/recorded_equal_mass_sanity.csv`;
- `results/quality_10ghz/frozen_inputs_before.json`;
- `results/quality_10ghz/provenance_radar.json`;
- `results/quality_10ghz/provenance_recorded_equal_mass.json`;
- `figures/quality_10ghz/session_component_heatmap.png`;
- `figures/quality_10ghz/subject_10_quality_card.png`;
- `figures/quality_10ghz/recorded_equal_mass_sanity.png`.

The radar-only CSVs contain no mass, `delta_m`, class, prediction, or model-error columns.
`provenance_radar.json` contains no workbook path/hash. The recorded-equal-mass provenance is the
only new artifact that names and hashes the weight workbook.

## 5. Sequential implementation plan

### Step 0 — Snapshot frozen inputs and enforce write scope

1. Before creating the new audit root, enumerate **every locally available regular file** below
   `results/`, sorted by repository-relative path. This concrete local inventory currently
   includes the two local Exp-A run directories, QC/WST artifacts, milestone-10 pointer/manifest
   files, logs, and all feature-store fingerprint and `.npz` files. Hash every file in full,
   including the multi-gigabyte feature arrays; do not treat a fingerprint as proof that its array
   bytes were unchanged.
2. Write paths, sizes, modification times, and SHA-256 values to
   `frozen_inputs_before.json` under the new audit output root. IBEX paths embedded in local
   milestone pointer/manifest files are **not dereferenced** because they do not exist locally; the
   local pointer/manifest file and its embedded approved hashes are what the local audit preserves.
   Authoritative C/D directories that are absent locally are recorded as `not_locally_present`,
   not silently replaced with an older local run. The neutrality claim is therefore exact for all
   locally available result bytes and makes no false claim to rehash remote IBEX bytes.
3. Add a fail-closed path guard: every audit write must resolve below one of the two approved new
   output roots. Existing files may be opened only read-only.

### Step 1 — Freeze contracts and verify stored-index blocks

1. Add the audit-specific config loader, target-free radar inventory, and pure block-assignment
   helper. Prove loading the audit config does not change the serialized mapping/hash of the base
   project config.
2. Build the census from filenames and radar headers only—never from `GroundTruth` or the normal
   target-bearing manifest. Assert the canonical real-data census is 80 files, 16 subjects × 5
   sessions, 100 frames per session, with generated unique contiguous `frame_idx=0..99`.
3. Prove that five blocks are non-overlapping, cover every raw frame exactly once, and are formed
   before QC filtering.
4. Record that NumPy/scipy loading preserves stored array-axis order, but the MAT file contains no
   independent timestamp with which to detect a hypothetical reorder inside that axis. The audit
   verifies deterministic stored order only, not acquisition chronology.

**Stop gate:** any missing, duplicated, non-contiguous, or incorrectly assigned generated frame
index stops the audit before a repeatability value is produced. Internal acquisition chronology
cannot be a stop gate because the delivered MAT arrays contain no independent timestamps.

### Step 2 — Build the 80-session radar-only component table

1. Run `run_qc_cube` directly and recompute/reconcile frozen QC fields for all 8,000 frames without
   constructing the target-bearing project manifest.
2. Compute raw-level and preprocessing components for every session and block.
3. Reconcile the eligible subset exactly to the existing 73-session population.
   The preprocessing reconciliation authority is the existing pure helper code in
   `experiments/run_preprocess.py`; the plan does not assume that its historical CSV is present.
4. Write and reread the component artifacts; fail on row loss, duplicate keys, unexplained
   non-finite values, or changed eligibility.

### Step 3 — Add WST five-block repeatability

1. Reuse the existing preprocessing and WST library without changing it.
2. Compute the fixed **magnitude-primary** diagnostic family for QC-passing frames of eligible
   sessions meeting the five-block coverage rule. Compute I/Q separately as a sensitivity view.
3. Compute both repeatability views separately for all three tilings and orders; never merge or
   rank across those cells/views.
4. Write explicit missing reasons where block coverage or a zero-norm vector makes a metric
   undefined; never fabricate zero.
5. Confirm a second run on the same machine/config produces byte-identical CSVs under fixed
   column/float serialization; separately apply the existing numerical tolerance to any
   cross-machine/backend comparison. Record hashes and machine details.

### Step 4 — Produce the human-readable views

1. Generate the subject-by-session component heatmap.
2. Generate within-subject comparisons without combining components into one score.
3. Produce the subject 10 worked example, including the existing finding that all five sessions
   pass 100/100 frames while their graded diagnostics differ.

### Step 5 — Run the separated recorded-equal-mass sanity check

1. Hash/freeze the radar-only artifacts and `provenance_radar.json` first.
2. Identify adjacent exact-equality pairs from recorded `mass_kg`.
3. Join pair identities to already-written radar metrics without recomputing or changing them.
4. Generate the pair table and figure with the non-QC/non-exclusion warning embedded in metadata
   and captions. Write the workbook hash and radar-artifact hashes only to
   `provenance_recorded_equal_mass.json`.

### Step 6 — Verification and documentation

1. Run the focused, full non-real-data, real-data, and leakage test suites.
2. Confirm `tests/test_no_leakage.py` stays green and unchanged.
3. Rehash the Step-0 frozen input list and require exact agreement. Also assert no new or modified
   file exists outside the two approved audit output roots and the explicitly planned new source,
   config, test, plan, and journal files.
4. Append `HISTORY.md` after each resolved implementation attempt, including failures and exact
   formulas/guards.
5. Add a concise chapter-ready subsection to `SECOND_CHAPTER.md` only after the full audit passes.
6. Do not update `HANDOFF.md` unless explicitly requested.

## 6. Required tests and acceptance criteria

1. **Block identity:** synthetic and real sessions map each frame to exactly one expected block;
   QC failures remain in their original block.
2. **Permutation guard:** reordering target-free inventory/QC rows cannot change results because
   `frame_idx`, not row position, defines membership. No claim is made that the MAT file proves
   acquisition chronology.
3. **Census:** the all-session card contains exactly 80 unique cells; frozen eligibility agrees
   exactly with the existing 73 eligible cells.
4. **Reconciliation:** reused QC and preprocessing formulas match their existing helpers/artifacts.
5. **Repeatability controls:** five identical synthetic blocks yield zero/near-zero separation and
   influence. Perturbing block 0, 1, 2, 3, or 4 in separate parameterized cases must increase at
   least one predeclared **worst-block** metric; typical medians are not required to catch an
   isolated block.
6. **Dimensionless geometry:** multiplying every coefficient in one whole tiling/order by a common
   positive unit-conversion constant leaves both views unchanged. Multiplying one path alone must
   change the path-energy-composition view when at least two active paths exist, while proportional
   within-path shape may remain unchanged. Single-path composition is explicitly not applicable.
   Vector dimension, path count, and near-zero-path counts are verified. Equal-mass tests pin that
   centroids, within-session distances, and both epsilon values are recomputed in the identical
   pair-common active-path space.
   Cross-tiling/order/view ranking is rejected by schema/API tests.
7. **No cross-session borrowing:** changing any other session's frames cannot change a session's
   radar-only card.
8. **No outcome access:** the target-free radar inventory and radar-only computation
   import/accept no `GroundTruth`, normal target-bearing manifest, workbook, or prediction input;
   their schemas forbid target-like columns.
9. **Ground-truth quarantine:** changing mass values may change only
   `recorded_equal_mass_sanity.csv`, `recorded_equal_mass_sanity.png`, and
   `provenance_recorded_equal_mass.json`; every radar-only CSV/PNG and
   `provenance_radar.json` remains byte-identical.
10. **Missingness:** every undefined metric has a named reason; no NaN is silently replaced with
   zero and no ineligible session receives an imputed WST vector.
11. **Config neutrality:** audit-specific config loading rejects unknown fields, records every
    audit constant, and leaves existing global-config serialization/fingerprints byte-identical.
12. **Determinism/provenance:** distinguish three checks: keys/order/schema are exact; same-machine
    CSVs are byte-identical under fixed column/float serialization; cross-machine numeric values
    need only satisfy the existing backend tolerance. Timestamped provenance and PNGs are excluded
    from byte-identity claims. Raw full-precision files keep ordinary SHA-256 hashes; any optional
    normalized numeric digest states its rounding rule explicitly.
13. **Frozen-results neutrality:** the before/after digest manifest agrees exactly; no existing
    result, feature store, config, or experiment file is overwritten; every runtime write resolves
    under an approved audit output root.
14. **Leakage gate:** `tests/test_no_leakage.py` remains byte-unchanged and green.

Suggested verification commands during implementation:

```powershell
uv run pytest tests/test_session_quality.py tests/test_run_session_quality_audit.py -q
uv run pytest -q
uv run pytest --realdata -q
uv run python experiments/run_session_quality_audit.py --config configs/quality_audit_10ghz.yaml
```

## 7. Interpretation rules fixed before results

- High repeatability means the recorded radar pattern is steady within that acquisition; it does
  not prove hydration validity.
- Low repeatability means the acquisition deserves review; it does not prove equipment failure.
- Similar radar for recorded-equal mass supports repeatability, but different radar does not prove
  a bad sample because other physiology and geometry may change.
- A clean quality card cannot rescue a failed hydration model, and a poor card cannot be used
  post hoc to erase a difficult subject.
- Cohort percentiles and within-subject ranks are descriptive labels only, not universal quality
  limits.

## 8. Separate future decision gate: sensitivity retraining

No retraining is part of this plan. A later proposal may be considered only if the audit reveals a
clear, radar-defined acquisition failure pattern and all of the following are accepted separately:

1. the rule uses radar measurements only and is written before viewing model improvements;
2. thresholds are physically fixed or fitted on outer-training subjects only inside LOSO;
3. the held-out subject never helps define its own exclusion/weighting rule;
4. original A–D results remain primary and unchanged;
5. the rerun is labelled post-training sensitivity analysis, not a replacement headline result;
6. sample removal counts and reasons are reported per subject and session.

## 9. Approval checklist

Owner approval is requested for the nine decisions in §1, especially:

- component card rather than a single score;
- stored-index wording rather than an unsupported chronology claim;
- fixed WST diagnostic family: reduction A, magnitude primary, I/Q sensitivity, pooled, log off,
  all three tilings, with separate within-path-shape and across-path-energy-composition views;
- the 10-passing-frame block sufficiency flag;
- a target-free radar inventory plus separate radar/ground-truth provenance;
- an audit-specific config loader that leaves the global config untouched;
- worst-block metrics so one bad 20-frame block cannot hide behind a median;
- strict separation of recorded-equal-mass analysis;
- local CPU execution;
- no exclusions or retraining in this audit.
