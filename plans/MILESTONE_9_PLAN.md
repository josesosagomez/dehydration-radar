# MILESTONE 9 PLAN — Exp C (ordinal 5-class) + Exp D (baselines), both bands, plus the owner's sanctioned exploratory frame-split

## §0 Scope and ground rules

**Why this milestone exists (implementation_plan.md §Build order step 9, `:234-235`; ROADMAP.md
§4 C/D, `:101-108`).** Experiments A (M7) and B (M8) are complete on the full cohort, with a
negative headline in both bands: the radar regressor lost to the session-index (time-of-day)
baseline in Exp A, and to its own train-only session mean in Exp B at 10 GHz (77 GHz: no
significant Exp B difference either way) — `SECOND_CHAPTER.md` §6-§7. Milestone 9 builds the two
remaining core-experiment arms on the same harness:

- **Exp C — ordinal 5-class classification (secondary task).** S0-S4 as an ordered, session-level
  task, ordinal metrics only (class-unit MAE, adjacent accuracy, quadratic-weighted κ) plus the
  LOSO confusion matrix. Spec frozen at M6: `implementation_plan.md:758-801`, `configs/exp_c.yaml`,
  `ExpCConfig` (config.py:383-398), amendment A-M6-5 (Frank-Hall).
- **Exp D — baselines.** Raw-beat and matched-preprocessing 1D-CNNs, raw and matched
  spectrogram 2D-CNNs, the physics power-ratio baseline, and the session-index-only baseline —
  both bands, under the identical LOSO harness, folds, seed set, and budget as Exp A. Spec frozen
  at M6: `implementation_plan.md:803-923`, `configs/baselines.yaml`, `BaselineConfig`
  (config.py:320-369), amendment A-M6-2 (77 GHz definitions), plus the pre-registered
  radar-vs-baseline comparison rules in §Statistics (`:1263-1281`).
- **The owner's exploratory frame-level random split** (decided 2026-07-30, recorded at
  `implementation_plan.md:925-941` and HISTORY.md 2026-07-30): a deliberately-leaky 5-fold
  random split over pooled frames for every C/D result, exploratory-only, never reported.

The scientific position mirrors M8's: **the C/D core designs were frozen at `config-freeze-v1`
(357f734) before any outer result existed**, and are now executed after two negative results are
known. That chronology is the milestone's central asset and its central risk: nothing in C/D may
be chosen, tuned, or re-scoped in response to the A/B outcomes beyond the **named, disclosed
protocol completions** enumerated in Step 0/0b and §6 — exactly the A-M8-1/A-M8-2 discipline.

**Review.** This plan goes through the Codex⇄Claude review loop (`review/PROMPT_codex_review.md`,
`review/PROMPT_claude_review.md`) before any source is written, per the owner's authoring prompt
and the M6-M8 precedent. The review block is appended at the end of this file with
`REVIEW-STATUS: AWAITING_CODEX`.

## Step 0 — owner decisions (RESOLVED 2026-07-30)

Three decisions were settled with the owner in the planning session, before authoring. All three
were made **after Exp A's and Exp B's full-cohort results were visible**; the chronology is stated
plainly here and must be restated in SECOND_CHAPTER.md §8 (the A-M8-1 disclosure discipline).

1. **A-M9-1 — Exp C family (a)'s search space = Exp A's frozen space, reused.** The frozen text
   defines family (a) as "threshold the continuous `L` predictor into 5 ordered bins"
   (`:770-773`) but — unlike Exp B, where A-M6-3 froze "reuse the identical enumerated search
   space as Exp A" — never states what feature/model space that continuous regressor searches.
   **Decision: mirror A-M6-3.** Family (a) reuses the identical frozen per-band staged search
   (Stage 1 feature axes at the ridge anchor α=1.0, Stage 2 the five families × frozen grids —
   `configs/search_10ghz.yaml` / `search_77ghz.yaml`, `ModelGridConfig`), with each candidate fit
   on **L = −Δm%** under the mandated inverse-frequency class weights and selected by the frozen
   **ordinal objective** (class-unit MAE primary, QWK secondary), not Exp A's fluid-loss MAE.
   Concretely (how the reuse maps onto Exp C's two families): **Stage 1 runs once per outer fold**
   with the family-(a) ridge anchor (`ord_a_ridge`, α=1.0, weighted, thresholded), scored
   ordinally; its winning feature key is then shared by **two Stage-2 arms** — arm (a): the five
   base families × frozen grids as thresholded ordinal regressors; arm (b): Frank-Hall over the
   frozen C grid (0.1, 1.0, 10.0). Arm (a) is the primary ordinal model; arm (b) is reported "as
   a comparison" per the frozen text (`:770-775`). Rationale for a shared Stage 1: the frozen
   `stage1_anchor_model: ridge` names ridge as the only Stage-1 anchor; inventing a second
   (Frank-Hall) anchor would be new, unfrozen design. This is a computation-affecting completion
   decided post-A/B; it is propagated into `plans/implementation_plan.md` at step 0.5 with its
   chronology, and listed in §6.
2. **The exploratory frame-split runs as a modal-config refit, not a nested search.** Per band,
   the exploratory path refits the single configuration each LOSO run selected most often (and
   each CNN family's LOSO-selected lr/weight-decay with its median final-refit epoch budget) on
   80% of pooled shuffled frames and scores the held-out 20%, five times (k=5). No inner search
   inside the exploratory path. Rationale: the owner wants paper-comparable numbers for private
   comparison, not a second validated procedure; a full nested search inside each leaky fold
   would cost its own IBEX allocation while producing numbers that are forbidden from every
   report regardless of their quality.
3. **Exp D CNN full-cohort layout = fold-array GPU jobs.** One SLURM array task per outer fold
   (16 tasks × 1 GPU) per family×band, with the M8 session-specific variant's proven
   init → array → merge shard machinery (fail-closed lineage validation, C14-C25 doctrine, and
   the step-10.5 git-free/`REVISION` wrapper lesson baked in from the start). Rationale: a single
   job iterating 16 folds sequentially has wall-time ≈ their sum (the C11 mistake); the array
   gives real cross-fold concurrency.
4. **Review process: the full Codex⇄Claude loop**, per the owner's authoring prompt — same as
   M7 (21 comments) and M8 (26 comments).

## Step 0b — owner-approval items (RESOLVED 2026-07-30)

Seven protocol completions the frozen text does not settle, each raised as an open question
during planning and **decided by the owner one by one on 2026-07-30** (the planning session's
question round) — all after Exp A's and Exp B's full-cohort results were visible, per the same
process rule as M8's Step 0b. None reopens a frozen decision; each fills a gap the freeze left
open. All are propagated into `plans/implementation_plan.md` at step 0.5 with this chronology.

- **O-M9-1 — ordinal tie-break completion.** The freeze names the primary (class-unit MAE) and
  secondary (QWK) selection metrics (`exp_c.yaml`, `:766-769`) but not the full deterministic
  order, nor how a per-candidate QWK is aggregated. **Decision:** `select_candidate_ordinal`
  orders by lower mean inner-val class-unit MAE → **higher** mean inner-val QWK (mean over the
  inner folds where QWK is defined; a candidate with QWK defined nowhere ranks below any finite
  QWK) → lower simplicity rank → lower feature dimension → lower inner-fold variance (the frozen
  Exp A tail, unchanged). Per-candidate QWK is computed from the stored **first-seed**
  validation predictions (`InnerResult.val_predictions`) — the primary MAE stays seed-averaged;
  only the tie-break secondary uses first-seed values (deterministic families are identical
  across seeds anyway; rf/gbm differ only here). Computation-affecting on ties.
- **O-M9-2 — Frank-Hall decision rule.** A-M6-5 froze probability recovery ("successive
  differences of the cumulative probabilities") but no class-decision rule, and successive
  differences can be negative (the classifiers are unlinked). **Decision:** floor recovered
  probabilities at 0, predicted class = argmax, ties broken toward the lower class. (The
  cumulative-threshold rule `class = Σ_k 1[P(>k) > 0.5]` was the documented alternative and was
  not chosen.)
- **O-M9-3 — composite-baseline membership.** §Statistics defines the composite procedure as
  "selects the best learned family (CNN / spectrogram / physics) inside each outer fold by inner
  CV" (`:1267-1271`) without saying whether the matched-preprocessing ablations are members.
  **Decision:** members are the three **primary** variants only — raw 1D-CNN, raw spectrogram
  2D-CNN, physics — matching the Holm-3 per-family list; ablations are reported as ablations,
  never entering any comparison family.
- **O-M9-4 — physics scalar → session mapping.** The freeze fixes the per-frame scalar and "a
  per-fold linear fit" (`:826-849`) but not the frame→session step. **Decision:** per-frame
  scalar → session value = **median** over the session's QC-passed frames (the frozen
  `frame_to_session_aggregation: median`, applied "identically to every CNN baseline" — extended
  to the physics scalar) → one-dimensional least-squares fit (slope+intercept) on outer-training
  **sessions** → predict test sessions. Session-level, matching the analysis unit (`:567-588`).
- **O-M9-5 — the radar side of every Exp D comparison.** The comparisons need Exp A's radar
  predictions paired on identical folds. **Decision:** re-run Exp A at the M9 commit against
  the rebuilt stores (`experiments/run_regression.py`, unchanged) and **assert its predictions
  are bit-identical to the M7 artifacts** (`results/runs/*_f36c4fb2/predictions_{band}.csv`)
  before any comparison is computed; the re-run's artifacts are then the comparison input with
  unified M9 provenance. A mismatch stops the milestone (it would mean the store rebuild or code
  drifted) rather than silently comparing against either version. Cost ≈ 1 h/band on IBEX CPU.
- **O-M9-6 — 10 GHz spectrogram channel convention.** `:821-825` fixes STFT constants but not
  channel count; the 10 GHz beat signal is complex. **Decision:** apply A-M6-2's own
  convention for complex inputs (77 GHz ablation, `:887-892`): real and imag parts STFT'd
  separately and stacked — **2-channel** log-magnitude spectrograms for both the 10 GHz primary
  (raw complex beat) and ablation (matched complex I/Q). (77 GHz stays as frozen: 1-channel
  primary, 2-channel ablation.) **Computation-affecting** — the channel convention fixes the
  input tensor itself, hence `in_channels`, the first conv layer's parameter count, the fitted
  `SpectrogramNorm` state (per-frequency statistics per channel), and the predictions. It is
  labelled as such everywhere it is summarized (§6, step 0.5, SECOND_CHAPTER.md §8).
- **O-M9-7 — inverse-frequency weight normalization.** `class_weight_formula:
  inverse_frequency_inner_train` fixes the form but not the scale, and scale interacts with
  every regularized fit (ridge α, SVR C, logistic C). **Decision:** for class c with count
  `n_c` among n inner-training rows and `K_present` classes,
  `w(c) = n / (K_present · n_c)` — sklearn's own "balanced" convention, mean weight ≈ 1, so the
  frozen grids' regularization strengths keep their Exp A meaning. Same per-row weights feed
  family (a) fits (except knn, frozen unsupported) and every Frank-Hall binary fit.

---

**In scope:**

- `src/dehyd/eval/metrics.py` — ordinal metrics: `class_unit_mae`, `adjacent_accuracy`,
  `quadratic_weighted_kappa`, `confusion_counts` (pure functions, hand-computed tests).
- `src/dehyd/models/ordinal.py` (new) — `inverse_frequency_class_weights`,
  `ThresholdedOrdinalRegressor` (family a), `FrankHallOrdinal` (family b), and their
  bit-comparable fitted-state extractors.
- `src/dehyd/models/regressors.py` — `build_estimator`/`fitted_state_params` dispatch for the
  `ord_a_*`/`ord_b_frank_hall` families (a `_bare_model` factoring; existing families
  byte-unchanged); `SEED_SENSITIVE` gains `ord_a_rf`/`ord_a_gbm`.
- `src/dehyd/eval/selection.py` — `SIMPLICITY_RANK` entries for the ordinal families;
  `OrdinalCandidateScore` + `select_candidate_ordinal` (O-M9-1). Single-tie-break-source
  doctrine kept: every Exp C selection routes through this module.
- `src/dehyd/eval/harness.py` — the **one** structural edit: `_viability_reason` generalized
  (knn check by param key; ordinal class-coverage check for 2-column y), byte-neutral for every
  existing path (pinned).
- `src/dehyd/eval/fold_parallel.py` (new) — the generic fold-parallel pool + heartbeat,
  extracted from `exp_b._run_folds_parallel`; exp_b delegates (bit-identity re-asserted).
- `src/dehyd/eval/exp_c.py` (new) — the Exp C composition mirroring `exp_b.py`'s shape.
- `src/dehyd/models/cnn.py` (new) — the frozen 1D/2D CNN architectures, input builders
  (raw/matched, both bands), spectrogram transform, normalizations.
- `src/dehyd/eval/exp_d.py` (new) — the CNN nested torch path (torch_fit's T18-protected
  algorithm + the frozen 6-config grid), the physics and session-index baselines' LOSO runs,
  per-family reporting, shard/merge for the GPU fold-array, and the frozen comparison
  statistics.
- `src/dehyd/features/store.py` — schema v2: per-frame signal arrays for Exp D
  (`STORE_VERSION = 2`), producers extended, validation unchanged in shape.
- `src/dehyd/eval/frame_split.py` (new) + `experiments/run_frame_split_exploratory.py` (new) —
  the sanctioned exploratory path, structurally isolated.
- `src/dehyd/provenance.py` — one byte-neutral extraction: the public
  `build_provenance_payload(...)` factored out of `record_run` (which then calls it and writes), so
  the exploratory writer reuses the real payload builder instead of private helpers (§2.10, C21).
- `experiments/run_ordinal.py`, `experiments/run_baselines.py` (new CLIs);
  `scripts/ibex/run_exp_c.sbatch`, `run_exp_d_cheap.sbatch`, `run_exp_d_cnn.sbatch` +
  `submit_exp_d_cnn.sh` (git-free from day one).
- Tests: `test_exp_c.py`, `test_ordinal.py`, `test_cnn.py`, `test_exp_d.py`,
  `test_frame_split.py`, `test_run_ordinal.py`, `test_run_baselines.py` (new), plus additions to
  `test_metrics.py`, `test_selection.py`, `test_harness.py`, `test_regressors.py`,
  `test_store.py`, `test_exp_b_ibex_scripts.py`-style sbatch checks.
- Store rebuild (both bands) at the clean M9 commit; full-cohort runs: Exp C (both bands, CPU),
  Exp A re-run (both bands, CPU, bit-identity assert), Exp D cheap baselines + comparisons
  (CPU), Exp D CNN fold-arrays (8 = 4 families × 2 bands, GPU); the exploratory frame-split
  runs; HISTORY.md per step; SECOND_CHAPTER.md §8 from the full LOSO results.

**Explicitly out of scope (deferred to M10+):**

- Experiment E (interpretability — runs on the Exp B model, own milestone), Experiment F
  (confound check), Experiment G (cross-band fusion), Experiment H (the cross-experiment
  statistics chapter and figure/table generation milestone).
- The **selection-variance robustness bootstrap** (R=200 refit-inside-the-resample,
  `:1282-1310`) — classical-only, cross-experiment, explicitly "milestone H" work.
- Any change to the frozen search spaces, grids, budget K, seed set, statistical constants,
  QC/eligibility rules, or the M6 guard whitelists; any 77 GHz secondary WST variant (A-M5-7).
- Promotion of the exploratory frame-split beyond its sanctioned, unreported role.
- Ordinal-task variants of the Exp D baselines (the freeze defines D as Δm% regression
  baselines for Exp A; no ordinal baseline is pre-registered — see §5 trap 16).

**The milestone-9 invariant, protected above all:**

> Exp C's and Exp D's **core designs** — the ordinal families, objective, sign convention,
> cutpoint source, class weighting, fold-viability rules; the baseline input definitions,
> architectures, optimizer constants, physics band definitions, budget-parity rule; the
> comparison families and their multiplicity treatment — were frozen at `config-freeze-v1`
> before any outer result existed, and are implemented exactly as written. The completions
> decided after Exp A/B's negative results were visible are **named and enumerated** — A-M9-1
> and O-M9-1..8 (O-M9-8 review-derived, owner-approved 2026-07-30 as option 8a) — each labelled
> computation-affecting or not, each propagated into
> `plans/implementation_plan.md` with its true chronology (step 0.5), none silently folded into
> "frozen before results". The exploratory frame-split is the one sanctioned exception to the
> LOSO reporting protocol: it exists **in addition to** LOSO, in a structurally separate path,
> and appears in no reported result. A disappointing C/D number is not a reason to revisit any
> frozen choice.

**Not reopened at M9** (fixed at M6-M8; each would need a prior authoritative amendment):
the fold source (`eval/splits.py` only), the tie-break source (`eval/selection.py` only — M9
adds a second pure function there, it does not bypass the module), the guard-before-every-fit
contract, the fit-audit shape, the tuned-ε rule (k=0.1, fallback 1e-6), the store fingerprint
doctrine and commit-match rule, `tests/test_no_leakage.py`'s frozen body, the five model
families + grids, `bootstrap_b=10000`/BCa/0.95, the A-M8-1/A-M8-2 resolutions, Exp B's frozen
config, and every `BaselineConfig`/`ExpCConfig`/`StatsConfig` constant.

**Ground rules.** Branch `v1_milestone9` — **already created off `v1_milestone8` and checked out at
`b6e7ba1`** (both branches point there; nothing has been committed on it yet). The M8-closing
journal edits an earlier draft of this line called uncommitted have in fact landed:
`SECOND_CHAPTER.md` and `plans/implementation_plan.md` are clean at `c7b6b83`. What is still
uncommitted is this plan (untracked), `HISTORY.md`, `HANDOFF.md`, and the owner's `.gitignore` edge
(`review*/`) — those plus step 0.5's `implementation_plan.md` edits form the owner-triggered
pre-implementation commit. Nothing is committed without the owner's explicit request. HISTORY.md entries per resolved step, not batched. Superseded material moves
to `archive/`. Tests are written alongside each §2 spec; a step is not done until its acceptance
tests are green. Bit-identity claims are **CPU-scoped** throughout: GPU training is not
bit-deterministic and is covered instead by per-seed reporting and CPU-fixture contract tests.

---

## §1 Build sequence — exact order and why

| # | Step | Why this position |
|---|------|--------------------|
| 0 | Write this plan; run the Codex⇄Claude review loop to closure (Step 0b already owner-resolved pre-review) | No source until the protocol reading is adversarially checked — Exp C/D are the last core-experiment arms and everything downstream (E/F/G/H) consumes them |
| 0.5 | Propagate A-M9-1 and every decided O-M9-* completion into `plans/implementation_plan.md` **at each authoritative location that contains the resolved ambiguity — including §Statistics, which an earlier draft wrongly declared "untouched"**, with the destination fixed per item so D0 can verify none was lost: A-M9-1 → §C (search-space reuse); O-M9-1 → §C (selection order + QWK aggregation) ; O-M9-2 → §C (Frank-Hall decision rule); **O-M9-3 → §Statistics `:1267-1274` (the composite/Holm family membership ambiguity lives there, not in §D) and cross-referenced in §D**; O-M9-4 → §D (iii) (physics frame→session mapping); **O-M9-5 → §Statistics `:1275-1281`** (the comparison's radar artifact + bit-identity precondition); O-M9-6 → §D (ii) (10 GHz spectrogram channels, **labelled computation-affecting**); **O-M9-7 → §C** (class-weight scale); O-M9-8 → §C (QWK undefinedness: annotate `:798-800` that the single-class clause is motivation, the operative trigger is zero expected disagreement on the fixed 5-class grid, owner-approved 2026-07-30 as 8a, with (8b) recorded as the rejected alternative); plus the ordinal inner-fold aggregation rule → §C. Each annotation states the post-A/B chronology and the computation-affecting label plainly; commit with the plan in the owner-triggered pre-implementation commit | The C9 lesson from M8: the doc fix must precede all source so no later doc-only commit invalidates the store commit-match at the milestone's end |
| 1 | Pin current behaviour: `test_m8_pin.py` re-run + a new pin capturing `_viability_reason`'s current outputs and a full `run_nested_candidates` byte-trace on the synthetic fixture | Step 4 claims byte-neutrality for every existing path; unverifiable without a pre-edit pin |
| 2 | `metrics.py` — the four ordinal pure functions | No dependencies; they define Exp C's objective and reports; hand-computed fixtures first |
| 3 | `models/ordinal.py` + `regressors.py` dispatch + `selection.py` additions | The estimators and tie-break exp_c consumes; pure/CPU; existing-family byte-neutrality asserted against step-1 pins |
| 4 | `harness.py` — the `_viability_reason` generalization | **The one risky edit.** Own commit between two green states; step-1 pin + full `test_harness.py` + `test_no_leakage.py` re-run immediately |
| 5 | `eval/fold_parallel.py` extraction; exp_b delegates | Zero-behaviour-change move proven by the existing serial-vs-parallel bit-identity tests; exp_c/exp_d need it next |
| 6 | `eval/exp_c.py` run half (spine, provider, worker, staged two-arm search) then report half (summaries, CSVs, confusion matrix, mechanism assertions) | Needs steps 2-5. Imports `stage1_candidates`-style enumeration from `exp_a.py` machinery — never copies the frozen space (A-M9-1 = one enumeration) |
| 7 | `models/cnn.py` + `eval/exp_d.py`'s torch nested path (grid × early stopping × epoch budget × refit), CPU-tested end to end on synthetic stores | The dominant new machinery; contract-tested before any real data or GPU |
| 8 | `eval/exp_d.py` remaining: physics + session-index LOSO runs, per-family reports, fold-shard/merge, comparison statistics | Needs step 7 + the frozen comparison rules; comparisons also need the Exp A re-run artifacts (step 12) but are testable on fixtures now |
| 9 | Store schema v2 (`build_session_npz_*` + `STORE_VERSION=2`), entrypoints, sbatch scripts + git-free wrapper, `eval/frame_split.py` + its entrypoint | Everything above is import-stable; producers change last so the clean commit follows immediately |
| 9.5 | Owner-triggered clean M9 implementation commit on `v1_milestone9` (branch already exists at `b6e7ba1`; all code green, tree clean) | `validate_store` requires store commit == analysis commit; every M9 edit moves the commit (M7/M8 step-8.5 precedent) |
| 9.6 | Rebuild both feature stores from the 9.5 commit (`extract10.sbatch` / `extract77.sbatch`) and `--validate` both | Real runs need v2 stores whose commit matches the M9 revision; v1 stores now fail closed on `store_version` |
| 10 | Local synthetic-store test suite green end to end; mechanism-only smokes: Exp C both bands (CPU, `--subset 6subjects`); Exp D per family both bands (CPU, `--subset 6subjects`, `seed_set=[1]` smoke overlay — run-level config only, frozen M6 sections untouched); one GPU array-task smoke (`--fold` on one fold) per CNN family | Cheap correctness gates before ~16-fold IBEX spends; the CNN smoke path differs from full only by run-level config (subset, seeds, device) per CLAUDE.md's smoke rule |
| 11 | Full-cohort Exp C, both bands, on IBEX CPU (`run_exp_c.sbatch`) | Classical, ~Exp B-scale wall-time; no owner pause (nothing left to blind — same C13 logic as M8 step 10) |
| 12 | Exp A re-run at the M9 commit, both bands (`run_exp_a.sbatch`, unchanged code) + **bit-identity assert vs the M7 prediction artifacts** (O-M9-5) | The radar side of every comparison; a mismatch stops the milestone |
| 13 | Exp D cheap baselines (physics, session-index), both bands, single CPU job; then the 8 CNN fold-array groups via `submit_exp_d_cnn.sh` (init → 16-task GPU array → merge, per family × band, `ARRAY_TIME` sized from the step-10 GPU smoke measurement); then the comparison stage against the step-12 artifacts | Cheap first (fast feedback), arrays sized from measurement not guesses (the C8 lesson), comparisons last since they consume everything |
| 14 | Exploratory frame-split runs — the full 16-run §2.10 matrix (4 ordinal: both C arms × 2 bands, classical on CPU; 12 regression: the six D families × 2 bands, the four CNN families as one small GPU job, physics/session-index on CPU) — **after** all LOSO results exist | Modal/reduced configs are only defined once the LOSO selection tables and merged CNN summaries exist (Step 0 item 2); the matrix is exactly the owner's sanctioned set, no more |
| 15 | HISTORY.md per-step entries (continuous, already written by now); SECOND_CHAPTER.md §8 from the full LOSO results — including every A-M9/O-M9 chronology; frame-split results appear **nowhere** in §8 | CLAUDE.md journal rules; the frame-split exclusion is constraint (3) of the owner's 2026-07-30 decision |

---

## §2 Per-file specifications

Format per file: **Responsibility** · **Public API / content** · **Frozen values** ·
**Acceptance criteria**. Signatures are the contract; bodies are written at implementation time.
Test-group IDs map to §3.

### 2.1 `src/dehyd/eval/metrics.py` (additions)

**Responsibility.** The four ordinal pure functions. No fitting, no I/O.

**Public API.**
- `class_unit_mae(y_class_true, y_class_pred) -> float` — plain pooled mean
  `|predicted − true|` in class units. Deliberately **pooled**, not subject-balanced: the frozen
  inner objective is "mean |predicted class − true class|" (`:766-769`) and §Statistics
  classifies class-unit MAE among the pooled/nonlinear metrics with the pooled seed-collapse
  (`:1199-1204`). NaN on empty input.
- `adjacent_accuracy(y_class_true, y_class_pred) -> float` — pooled fraction with
  `|ŷ − y| ≤ 1`. NaN on empty.
- `quadratic_weighted_kappa(y_class_true, y_class_pred, *, n_classes=5) -> float` — Cohen's κ
  with quadratic weights over the fixed 5×5 class grid (weights `(i−j)²/(K−1)²`, expected
  matrix from marginal outer product). **Undefinedness is decided by the actual denominator,
  not by a class-count pre-check:** returns NaN iff the input is empty or the expected
  disagreement `Σ_ij w_ij E_ij` is exactly 0 (which on the fixed grid happens only when both
  marginals concentrate on the same single class). It never raises, so the frozen per-fold
  fallback (`:798-800`) and the bootstrap skip-and-count machinery consume it directly.
  *Disclosed divergence from the frozen text's motivating parenthetical* — `:798-800` says
  "QWK is undefined on a single-class validation set", which holds when the label set is
  inferred from the data (a 1×1 matrix gives 0/0) but **not** on the fixed 5-class grid this
  task mandates: with truth all-S0 and a varying predictor the denominator is non-zero and κ is
  defined (κ = 0 for an uninformative predictor; verified against
  `sklearn.metrics.cohen_kappa_score(..., weights="quadratic", labels=[0,1,2,3,4])`, which
  returns 0.0 for true `[0,1,2,3,4]`/pred all-0 and for true `[0,0]`/pred `[0,1]`, and NaN only
  for both-sides-constant-and-equal). The frozen *behaviour* — never error, fall back to
  class-unit MAE whenever QWK is undefined — is implemented exactly; only the trigger condition
  is the mathematically correct one. Recorded as **O-M9-8** in §6 — computation-affecting on
  tie-breaks, **owner-approved 2026-07-30 (option 8a)**. The count of single-class-truth validation
  folds and of NaN QWK values is reported alongside the metric (§6), so the choice's empirical size
  is visible rather than assumed negligible.
- `confusion_counts(y_class_true, y_class_pred, *, n_classes=5) -> np.ndarray` — 5×5 integer
  counts, rows = true class.

**Frozen values.** `n_classes = 5` (S0-S4); the three metric names/definitions from
`ExpCConfig`/`:758-769`; no new constants.

**Acceptance criteria.** Hand-computed fixtures for all four (including a QWK case checked
against the standard formula by hand); **QWK returns the defined value (0.0), not NaN, for
single-class truth against a varying predictor and for a constant predictor against multi-class
truth — both asserted against `cohen_kappa_score(..., weights="quadratic",
labels=[0,1,2,3,4])`; NaN only for empty input and for both-sides-constant-and-equal (the
zero-denominator case)**; class-unit MAE
provably ≠ subject-balanced MAE on an unequal-sessions fixture (guards the pooled choice);
`confusion_counts` sums to n and is orientation-checked (an asymmetric fixture). (T-M9-metrics)

### 2.2 `src/dehyd/models/ordinal.py` (new) + `regressors.py` dispatch

**Responsibility.** The two frozen ordinal families as sklearn-compatible estimators, so the
existing harness engine (`_fit_once` → `Pipeline(scaler, model).fit(X, y)`) runs them unchanged.
The 2-column target convention: `y[:, 0] = L = −Δm%` (continuous), `y[:, 1] = class = session
index` — the scaler ignores y; only these estimators read it.

**Public API.**
- `inverse_frequency_class_weights(class_labels) -> np.ndarray` — per-row weights, O-M9-7
  formula `w(c) = n / (K_present · n_c)`. Pure; a fitted quantity when computed on a fold's
  training rows (recorded, below).
- `class ThresholdedOrdinalRegressor(base_family, base_params, *, quantiles, min_separation,
  weighted, seed)` — family (a). `fit(X, y2)`: compute weights from `y2[:, 1]` (skipped for
  `weighted=False`, i.e. knn per the frozen `class_weight_unsupported_families`); fit the bare
  base regressor (from `regressors._bare_model`) on `(X, y2[:, 0])` with `sample_weight` where
  supported; cutpoints = the four `cutpoint_quantiles` (0.2, 0.4, 0.6, 0.8) of the **in-sample
  predictions on its own training rows** (the frozen `cutpoint_source`), then enforced strictly
  increasing by nudging each tied/inverted cutpoint up by `cutpoint_min_separation` (1e-9).
  `predict(X)`: continuous L̂ → `np.searchsorted(cutpoints, L̂, side="right")` → class in
  {0..4} as float. Stores `cutpoints_`, `class_weights_` (or None), and the fitted base.
- `class FrankHallOrdinal(C, *, max_iter=1000)` — family (b), A-M6-5. `fit(X, y2)`: per-row
  weights from `y2[:, 1]`; four binary `LogisticRegression(C=C, solver="lbfgs",
  max_iter=max_iter)` fits with `sample_weight`, targets `1[class > k]`, k = 0..3. Raises a
  typed `OrdinalViabilityError` if any binary target is single-class (unreachable when the
  harness viability check passed — defense in depth, never silent). `predict(X)`: cumulative
  probabilities → successive differences → floor at 0 → argmax, ties to the lower class
  (O-M9-2). `predict_proba(X)` returns the recovered (floored, renormalized) matrix for the
  reports.
- `fitted_state_params_ordinal(family, model) -> dict[str, np.ndarray]` — bit-comparable state:
  ord_a_* = the base family's `fitted_state_params` + `cutpoints_` + `class_weights_`; ord_b =
  stacked per-threshold `coef_`/`intercept_` + `class_weights_`.
- `regressors.py`: `_bare_model(family, params, seed)` factored out of `build_estimator`
  (existing five families byte-unchanged — asserted against step-1 pins); `build_estimator`
  dispatches `ord_a_<fam>` → `Pipeline(scaler, ThresholdedOrdinalRegressor(...))` and
  `ord_b_frank_hall` → `Pipeline(scaler, FrankHallOrdinal(...))`; `fitted_state_params`
  dispatches to `fitted_state_params_ordinal`; `SEED_SENSITIVE` += {`ord_a_rf`, `ord_a_gbm`}.

**Frozen values.** All from `ExpCConfig`: quantiles (0.2, 0.4, 0.6, 0.8), min_separation 1e-9,
weight formula (+ O-M9-7 normalization), knn-unweighted, C grid (0.1, 1.0, 10.0), lbfgs
LogisticRegression (the A-M6-5 verified-API substitution). `max_iter=1000` is a solver
convergence bound, not a tuned quantity (recorded; a ConvergenceWarning fails the fit loudly
rather than being suppressed).

**Acceptance criteria.** (T-M9-ordinal) Cutpoints are a **train-only fitted quantity**: mutating
any non-training row's L or class leaves `cutpoints_`/`class_weights_`/base state bytewise
identical; cutpoints from in-sample predictions, not targets (fixture where the two differ);
strict-increase nudge on a constant-prediction fixture (all cutpoints tied) yields
searchsorted-safe monotone cutpoints; knn fits unweighted; weight formula hand-computed;
Frank-Hall recovers hand-computed cumulative/differences on a tiny fixture; negative-difference
case exercised (floor+argmax); `OrdinalViabilityError` on a missing-class fixture;
deterministic across repeated fits (fixed seed).

### 2.3 `src/dehyd/eval/selection.py` (additions)

**Responsibility.** The Exp C tie-break, in the single tie-break module.

**Public API.** `SIMPLICITY_RANK` gains `ord_a_ridge=0, ord_a_knn=1, ord_a_svr=2, ord_a_rf=3,
ord_a_gbm=4` (the base family's frozen rank — the wrapper adds no capacity) and
`ord_b_frank_hall=0` (linear; sole family in its arm, so the value can never decide across
families — comment says so). `OrdinalCandidateScore(candidate_id, inner_val_class_mae,
inner_val_qwk, simplicity_rank, feature_dimension, inner_fold_variance, n_evaluable_inner_folds)`;
`select_candidate_ordinal(scores)` — comparability requires finite MAE and variance (as
`select_candidate`) **and `n_evaluable_inner_folds >= 1`**; ordering per O-M9-1 with NaN-QWK
ranked below any finite QWK; stable min. Raises the existing `SelectionError` when no score is
comparable, with a message naming the count of non-evaluable cells (the caller adds the missing
classes — §2.6).

**Ordinal inner-fold aggregation (the rule `select_candidate_ordinal` consumes).** The MAE and
variance fed to `OrdinalCandidateScore` are aggregated **over that candidate's evaluable inner
folds only** — `np.nanmean` / `np.nanstd(ddof=0)` over the finite entries of the candidate's
`inner_scores` row, with `n_evaluable_inner_folds` = the count of finite entries. This lives in
`exp_c`'s score assembly (§2.6), **not** in the harness: `harness._score_candidates_on_fold`
keeps its `np.mean`/`np.std` over all folds (`harness.py:329-337`) so Exp A/B stay byte-identical,
and Exp C reads `StageOutcome.inner_scores` and does its own reduction. Why this is required and
not a nicety: the class-coverage predicate is candidate-independent, so a single missing-class
inner fold NaNs **every** candidate's plain mean and would make the whole outer fold
non-selectable — whereas `implementation_plan.md:793-800` makes the outer fold contribute no
ordinal score only "if **all** configs are non-evaluable". The stated minimum is therefore
**one** evaluable inner fold (the frozen text's own threshold; no new constant is invented, and
deliberately not a tighter minimum like 2, which would be unfrozen design). `n_evaluable_inner_folds`
and the per-cell reasons are written into `selection_table_{band}.csv` and the metrics JSON for
every fold, so a fold selected on fewer than the full 5 inner folds is visible, never silent.

**Frozen values.** Primary/secondary metric names from `ExpCConfig`; the tail order is the
O-M9-1 completion; the evaluable-fold minimum (1) is `implementation_plan.md:793-800`'s own
threshold.

**Acceptance criteria.** (T-M9-selection) Hand-built score lists prove each rung of the order
decides exactly when all higher rungs tie; QWK direction (higher wins) asserted; NaN-QWK loses
to finite-QWK at equal MAE; a score with `n_evaluable_inner_folds = 0` is incomparable even when
its MAE/variance are finite; **the existing five-family ranks and `select_candidate`'s behaviour
remain pinned** — `select_candidate`'s body is byte-unchanged and its existing tests pass unedited,
while `tests/test_selection.py:79`'s exact-dict assertion on `SIMPLICITY_RANK` **is updated** to the
five base families plus the six ordinal keys (an exact-equality pin cannot survive adding keys, and
that file is not frozen — only `tests/test_no_leakage.py` is). The updated assertion keeps the frozen
`ridge < knn < svr < rf < gbm` ordering check and adds `ord_a_*` mirroring it.

### 2.4 `src/dehyd/eval/harness.py` (the one structural edit)

**Responsibility.** Generalize `_viability_reason` so the engine can mark ordinal-non-evaluable
cells with the existing `InnerResult.reason` mechanism — the frozen fold-viability rule
(`:793-801`) — without any other engine change.

**Content.** New signature `_viability_reason(candidate, bundle, train_rows) -> str | None`:
(a) the knn row-count check keyed on `"n_neighbors" in candidate.params()` (covers `knn` and
`ord_a_knn` identically); (b) **iff `bundle.y.ndim == 2`**: the inner-training rows must contain
**all five S0-S4 classes**, compared against the **constant** class set
`ORDINAL_CLASSES = (0, 1, 2, 3, 4)` (`range(n_classes)`, frozen `n_classes = 5`) — *never*
against `set(y[:, 1])` or any other data-derived set; a missing class returns
`f"ordinal_missing_class_{c}_in_inner_train"` (lowest missing class named; the reason string
lists all of them when several are absent). Two reasons the constant is mandatory, not
stylistic: (i) `implementation_plan.md:793-797` fixes the rule as "its inner-training set lacks
any of the 5 classes" — a bundle-relative predicate would silently stop requiring a class that
QC removed cohort-wide, which is a weaker rule than the frozen one; (ii) `OrdinalFeatures`
mirrors `StoreBackedFeatures`, whose bundles carry **all** session rows (`exp_a.py`
`StoreBackedFeatures.data_for`; the row mask is applied only afterwards in
`_score_candidates_on_fold`), so `set(y[:, 1])` includes inner-validation *and outer-test*
labels — comparing against it would make control flow (which cells are fit at all) a function
of held-out labels. The constant predicate is a pure function of inner-training rows.
The check is candidate-independent by construction (rows only) but recorded per cell, matching
"such configs are skipped in ordinal selection (recorded)". One call-site update in
`_score_candidates_on_fold`. `_score`'s default
(`score_fn=None` → `subject_balanced_mae`) is never reached with 2-column y — exp_c always
supplies its score_fn, and a new fail-fast assert in `_score` raises on 2-D y with
`score_fn=None` rather than silently feeding a matrix to the frozen metric.

**Fitted quantities.** None introduced; this edit computes nothing from data beyond a set
membership on training rows.

**Acceptance criteria.** (T-M9-harness) The step-1 pin (`test_m8_pin.py` + the new
`_viability_reason` pin) is bytewise unchanged for every 1-D-y fixture; the knn reason string
identical to today's on the same fixture; a 2-D-y fixture with a class absent from inner-train
yields the reason (no fit attempted, `inner_scores` NaN there); the 2-D-y + `score_fn=None`
assert fires; full `test_harness.py` + `tests/test_no_leakage.py` green, the latter proven
unchanged by `git diff --exit-code`. Two dedicated C1 tests: (i) **non-training-label
independence** — rewriting the class labels of every inner-validation and outer-test row of a
2-D-y bundle (any permutation, including ones that delete a class from the non-training rows)
leaves the per-cell reason strings and the whole viability decision map bytewise identical;
(ii) **globally-absent class still blocks** — a fixture where class 3 appears nowhere in the
bundle still yields `ordinal_missing_class_3_in_inner_train` for every cell (the bundle-relative
predicate would have returned `None` here — the test fails against that implementation).

### 2.5 `src/dehyd/eval/fold_parallel.py` (new)

**Responsibility.** The generic fold-parallel pool with the heartbeat (`_POLL_INTERVAL_S=1`,
`_PROGRESS_INTERVAL_S=60`), extracted verbatim from `exp_b._run_folds_parallel` and
parameterized `run_folds_parallel(worker, tasks, n_workers, label)`; `exp_b` delegates.
spawn-context, single-threaded workers, canonical result ordering by the caller — unchanged.

**Acceptance criteria.** (T-M9-parallel) exp_b's existing serial-vs-parallel bit-identity test
green unchanged; exp_c's and exp_d's equivalents added; the heartbeat interval constants are
the M8-committed values (provenance: HISTORY.md 2026-07-29 fix 2, commit e88fd33).

### 2.6 `src/dehyd/eval/exp_c.py` (new)

**Responsibility.** The Exp C composition, mirroring `exp_b.py`'s shape: spine → provider →
picklable per-fold worker (staged two-arm search) → fold-parallel orchestration → out-of-fold
assembly → summaries/reports → mechanism assertions → `run_and_report_c`.

**Public API / content.**
- `build_sessions_c(config, band)` — `exp_a.build_sessions` records + per-record `loss_l =
  -delta_m_pct` and `class_idx = session_idx` (the frozen sign convention `:762-765`; the class
  IS the session stage). All S0-S4 sessions; evaluability = Exp A's rule (≥1 eligible session,
  `:605-610`) via `evaluable_subjects_c`.
- `class OrdinalFeatures` — wraps `exp_a.StoreBackedFeatures` (composition, not copy): same X
  paths (off/frozen vectors, tuned-ε reconstruction with the identical train-only
  `tuned_epsilons` — the same fitted quantity, audited identically), but `y` is the 2-column
  `[L, class]` matrix and `session_idx` is populated. X bytewise identical to Exp A's on every
  branch (asserted).
- `ordinal_class_mae_score(subjects, y_true2, y_pred, session_idx) -> float` — the harness
  `score_fn`: pooled `class_unit_mae(y_true2[:, 1], y_pred)`.
- `_run_single_fold_c(config, band, sessions, store_dir, fold, seeds) -> ExpCFoldResult` —
  top-level, picklable. Stage 1: the band's frozen feature-axis enumeration (72 / 9 candidates,
  reusing `exp_a.stage1_candidates`'s enumeration with family swapped to `ord_a_ridge`,
  α=1.0) scored by `ordinal_class_mae_score`; winner selected via
  **`select_candidate_ordinal`** over `OrdinalCandidateScore`s that exp_c assembles from the
  `StageOutcome` via `_ordinal_candidate_scores(stage, sessions)`: MAE/variance are the
  **evaluable-inner-folds-only** `nanmean`/`nanstd` of the candidate's `inner_scores` row with
  `n_evaluable_inner_folds` alongside (the §2.3 aggregation rule — the harness's own
  `candidate_scores` are ignored for Exp C, never modified); QWK is the mean over the evaluable
  folds whose QWK is defined, recomputed from each cell's stored first-seed `val_predictions`
  against the spine classes (O-M9-1, and the frozen "QWK undefined → fall back to MAE for that
  fold" rule, `:798-800`, is exactly this skip). If a candidate has zero evaluable inner folds
  it is incomparable; if **every** candidate does, the fold contributes no ordinal score via the
  existing `SelectionError` path, and exp_c re-raises it with the fold's test subject and the set
  of missing classes named (§5 trap 3). Stage 2, **arm (a)**: the
  five `ord_a_*` families × the frozen grids (41 candidates) at the Stage-1 feature key; Stage
  2, **arm (b)**: `ord_b_frank_hall` × C grid (3 candidates) at the same feature key. Each arm's
  winner refit via `harness._final_refit` with the ordinal score_fn. `before_fit` guards every
  fit with **three** checks, in this order:
  1. `protocol_freeze_guard(config, active=...)` — for `ord_a_*`, the full Exp A-shaped `active`
     record with `model_family` = the **base** family (the fitted regressor genuinely is that
     family) validated by `require_complete_active` as today; for `ord_b_frank_hall`, an `active`
     record of band + feature axes **without** `model_family`, validated for completeness by
     exp_c's own `REQUIRED_ACTIVE_KEYS_C` (= the band set minus `model_family`) since
     `harness.require_complete_active` demands exactly the band set and would reject it.
  2. **`assert_exp_c_fit_authorized(candidate, config)` (new, in `exp_c.py`)** — the check that
     binds the computation actually about to run, because neither existing guard does:
     `protocol_freeze_guard`'s `_check_active` only validates the keys *present* in `active`
     (`protocol_freeze.py:116-136`) and `REQUIRED_ACTIVE_KEYS_C` only checks feature-key
     completeness, so without this an unauthorized family or an off-grid `C` would reach `.fit()`
     with every guard passing. It asserts, raising `ExpCProtocolError` naming the offending field:
     (a) `candidate.family ∈ ORDINAL_FAMILIES` = the six authorized ids
     (`ord_a_{ridge,knn,svr,rf,gbm}`, `ord_b_frank_hall`) and matches the arm being run;
     (b) for `ord_a_<fam>`: the wrapper's `base_family` is exactly `<fam>`, its `base_params`
     are a member of that family's frozen `ModelGridConfig` grid (the same enumerated tuples
     Exp A uses — `ridge_alphas`, `svr_c`×`svr_epsilon`, `rf_*`, `gbm_*`, `knn_n_neighbors`;
     at Stage 1, `base_params == {"alpha": 1.0}`, the frozen `stage1_anchor` value), and the
     wrapper constants equal `ExpCConfig`'s (`cutpoint_quantiles`, `cutpoint_min_separation`,
     `weighted == (fam not in class_weight_unsupported_families)`);
     (c) for `ord_b_frank_hall`: `C ∈ ExpCConfig.proportional_odds_c_grid`, the implementation
     tag equals `ExpCConfig.proportional_odds_impl`, and `max_iter` equals the recorded bound.
  3. exp_c's `active`-completeness check (per arm, as in 1).
  So family (b)'s authorization is a *checked* property of the candidate, not an inference from
  the config section's existence. See §5 trap 2 and Review focus item 2.
- `ExpCFoldResult` — test_subject, per-arm: selected feature key/family/params, test class
  predictions (float classes), per-seed outcomes, final fits; test classes/targets; reason.
- `run_exp_c(...)` — fold-parallel over selectable folds via `fold_parallel`.
- `summarize_exp_c(results, config)` — per arm: pooled class-unit MAE, adjacent accuracy, QWK
  with subject-cluster CIs (pooled seed-collapse via `subject_cluster_bootstrap_pooled`,
  QWK/adjacent skip-and-count per `:1214-1219`); per-subject class-MAE (descriptive
  distribution); 5×5 confusion matrix = per-seed `confusion_counts` averaged across seeds
  (descriptive); selection-frequency table (feature axes × family × arm), carrying per fold the
  `n_evaluable_inner_folds` and the per-cell viability reason counts (§2.3); **the O-M9-8 exposure
  counters — `n_single_class_truth_val_folds` (validation folds whose true classes are
  single-valued, i.e. the folds option 8b would have skipped) and `n_qwk_nan` (folds where QWK was
  genuinely undefined) at both CV levels, plus the bootstrap's existing skip-and-count for the QWK
  CI** — so §8 can state how much the (8a) choice actually changed rather than asserting it was
  negligible;
  `conditional_exploratory: true`. **No baseline comparison field** — see §5 trap 16.
  RNG offsets: named constants `RNG_OFFSET_EXPC_BASE = 200` (arm a: 200-202 for
  MAE/adjacent/QWK; arm b: 210-212); the pairwise-distinct offsets test extends over Exp B's
  and Exp D's.
- `write_exp_c_reports(...)` — `metrics_exp_c_{band}.json`, `predictions_{band}.csv`
  (subject, arm, seed, true class, predicted class, session), `selection_table_{band}.csv`,
  `confusion_{band}.png` + CSV. `_assert_mechanism_ok_c` — fold-role disjointness, audit
  coverage (every fit record excludes the held-out subject), **S0 present** (unlike Exp B — all
  five classes are the task), predicted classes within {0..4}.
- `run_and_report_c(...)` — `validate_store` first (fail-closed, commit-match); smoke mode =
  mechanism-only (structural run-log, no class metric, no confusion matrix, no selection
  values).

**Frozen values.** Everything cited above from `ExpCConfig`, `search_*.yaml`,
`ModelGridConfig`, `StatsConfig`; seeds = `run.seed_set` (5); inner folds
`min(5, n_train)`; A-M9-1 for the space itself.

**Acceptance criteria.** (T-M9-expc-provider, T-M9-expc-leak, T-M9-expc-mutation,
T-M9-expc-viability, T-M9-expc-report — §3.) Highlights: X bytewise equal to
`StoreBackedFeatures` per branch; the **end-to-end synthetic-store outer-mutation property over
the real Exp C composition** (mutate the held-out subject's stored tensors + targets + class →
both arms' inner scores, winners, cutpoints, class weights, tuned-ε, every fitted parameter and
training prediction bytewise unchanged; only the held-out predictions/score move) and the
**inner-val mutation** analog (T16 pattern); a fabricated inner fold missing one class is
non-evaluable with the named reason **while the remaining folds still produce a selected
winner** (the one-missing-fold case), and only an all-inner-folds-non-evaluable fold raises the
re-labelled `SelectionError` (the all-missing case); the `assert_exp_c_fit_authorized` negative
matrix — an unauthorized family id, an `ord_a_svr` with off-grid `(C, ε)`, an `ord_a_ridge` whose
`base_family` disagrees with its id, a mismatched cutpoint quantile, and an off-grid Frank-Hall
`C` each raise `ExpCProtocolError` naming the field, and no `.fit()` is reached (spy on
`build_estimator`); smoke surfaces no performance token; serial vs `n_workers=2` bit-identical.

### 2.7 `src/dehyd/models/cnn.py` (new)

**Responsibility.** The frozen Exp D architectures and input constructions — deterministic
given (weights, input); no CV logic here.

**Public API / content.**
- `Cnn1d(in_channels)` — 3 × (Conv1d(k=7, stride 1) → BatchNorm1d → ReLU → MaxPool1d(4)),
  channels (16, 32, 64) → global average pool → Linear → 1 scalar.
- `Cnn2d(in_channels)` — 2 × (Conv2d(3×3) → BatchNorm2d → ReLU → MaxPool2d(2×2)), channels
  (16, 32) → global average pool → Linear → 1 scalar.
- Input builders (all per-frame, unfitted unless stated):
  - `raw_beat_input_10(sig)` — stored complex 534 beat → 2-ch {real, imag} → robust
    (median/MAD) per-channel per-signal standardization.
  - `matched_input_10(sig)` — the stored `[2, 470]` matched I/Q (already the preprocess-chain
    output, robust-standardized there).
  - `raw_input_77(sig)` — stored real 256 slow-time → 1-ch → robust per-signal.
  - `matched_input_77(sig)` — stored complex 256 Rx-0 gate-mean slow-time → 2-ch {real, imag} →
    robust per channel.
  - `spectrogram(x_1d)` — Hann 64 / hop 16 / nfft 128, **literal log-magnitude
    `log(|STFT| + eps_mag)`** — magnitude, not power. An earlier draft wrote
    `log(|STFT|² + 1e-30)`, which is log-*power* (`= 2·log|STFT|` away from the floor) and so
    contradicts the frozen wording "log-magnitude" at `:821-825` and `:887-892`; `1e-30` was also
    not a float64 tiny guard (`np.finfo(np.float64).tiny ≈ 2.2e-308`) but an arbitrary floor that
    can clip genuinely low-energy bins. Neither the squaring nor that floor appears in
    `BaselineConfig`, so both are corrected to the frozen text rather than promoted to an
    amendment. `eps_mag = np.finfo(np.float64).tiny` — a pure representability guard against
    `log(0)`, data-independent, with no magnitude chosen by anyone (a *relative* floor would make
    the transform data-dependent and is therefore rejected). Output is asserted finite.
    Complex inputs: real and imag transformed separately and stacked (O-M9-6); 77 GHz primary
    stays 1-channel.
  - **Which signal each spectrogram variant transforms — per variant, because "raw" and "matched"
    differ here.** `BaselineConfig`'s "two distinct normalization rules"
    (`raw_matched_standardize: robust_per_channel` vs `spectrogram_standardize:
    train_only_per_frequency_mean_std`) does **not** mean spectrograms never see robust
    standardization; it means the *time-domain* robust step belongs to the raw/matched input
    definitions while the *spectral* per-frequency step is the fitted one. The **raw** spectrogram
    branches therefore bypass robust standardization (they are defined as the STFT of the raw
    signal), but a **matched** spectrogram must be the STFT of the fully matched-preprocessed
    signal — that is what the ablation is *for* — and the matched definition includes the robust
    per-channel step. An earlier draft's blanket "all four STFT the stored unstandardized arrays"
    was wrong, and self-contradictory: it also disagreed with §2.9 (the stored 10 GHz
    `sig__matched_iq` is `preprocess_cube(..., channel="iq")`, whose last step is
    `to_channels(trimmed, channel, pre.standardize)` — `preprocess/pipeline.py:73` — so it is
    *already* robust-standardized in the store, exactly as this section's `matched_input_10` says).
    The four variants:
    - 10 GHz primary = STFT of the raw complex 534 beat (`sig__raw_beat`, **no** robust step),
      real/imag separately → 2 ch;
    - 10 GHz ablation = STFT of the stored `[2, 470]` `sig__matched_iq`, each channel separately
      → 2 ch. **Already robust-standardized in the store**; nothing is applied at load, and no
      second standardization is applied (double-standardizing would not be the frozen matched
      signal);
    - 77 GHz primary = STFT of the raw real 256 slow-time (`sig__raw_slowtime`, **no** robust
      step) → 1 ch;
    - 77 GHz ablation = STFT of the stored `[2, 256]` matched real/imag **after** applying the
      robust per-channel step at load — the 77 GHz store deliberately keeps this tensor
      pre-standardization (§2.9), while `implementation_plan.md:877-892` defines the matched 77 GHz
      input as robust-standardized per channel, so the step happens in `matched_input_77` and the
      spectrogram branch consumes that same builder's output → 2 ch.
    Every branch then receives train-only `SpectrogramNorm`. The physics path is unaffected: it
    reads `sig__raw_beat` / `sig__raw_slowtime`, which carry no standardization at all (trap 13).
  - `SpectrogramNorm` — the **fitted** train-only normalization, with its axes named because
    "per-frequency" is not executable on a `[frame, channel, freq, time]` tensor without them:
    statistics are kept **per `(channel, frequency)`** (parameter shape `[C, F]`, one mean and one
    std each) and reduced over **frames × time** of the training frames only
    (`X_train.mean(axis=(0, 3))`, `std(axis=(0, 3), ddof=0)`), applied as
    `(x − mean[:, :, None]) / scale[:, :, None]` where **`scale = np.where(std == 0.0, 1.0, std)`**.
    The zero-variance fallback is `1.0`, not `std + tiny`: a constant training bin under
    `std + np.finfo(np.float64).tiny` would amplify any differing test value by ~1e308, overflowing
    or swamping the net — and zero-variance spectrogram bins are plausible (a dead frequency bin, a
    padded edge). `scale = 1.0` leaves such a bin as a centered raw value, and it is the convention
    already used in this repo for exactly this case (`models/torch_fit.py::_normalize_stats`, which
    does `np.where(std == 0.0, 1.0, std)`), so this is a reuse, not a new rule. Both `mean` and the
    substituted `scale` (plus a count of substituted cells) go into the `FitRecord`, so a fold where
    the fallback fired is visible in the audit. Per-`(channel, frequency)` rather than
    frequency-shared-across-channels because the real and imag parts of a complex beat have
    genuinely different per-frequency scales, and sharing would let one channel's statistics
    standardize the other. Emitted as a `FitRecord("spectrogram_norm", ...)` at both CV levels.

**Frozen values.** All architecture/optimizer/STFT constants from `BaselineConfig` /
`configs/baselines.yaml` (kernel 7/pool 4/channels 16-32-64; 3×3/2×2/16-32; Hann 64/hop
16/nfft 128; robust vs per-frequency normalization split; A-M6-2 tensor definitions;
`matched_reference_rx_index_77ghz = 0`).

**Acceptance criteria.** (T-M9-cnn) Output shapes for every (band, family, channel-count)
input; parameter counts asserted once (architecture pin); robust standardization is per-signal
(two frames never share statistics — fixture) **and each of the four spectrogram variants consumes
exactly the input §2.7 names — the two raw branches STFT the unstandardized stored array (feeding a
scaled copy scales the pre-norm spectrogram), the 10 GHz matched branch consumes the store's
already-standardized `sig__matched_iq` with no second standardization (bytewise equal to
`matched_input_10`'s output), and the 77 GHz matched branch consumes `matched_input_77`'s
robust-standardized output (bytewise equal, and a scaled input copy leaves it unchanged)**;
**a `(channel, frequency)` cell with zero training variance normalizes with `scale = 1.0` — a
differing finite validation value stays finite and O(1), which the `std + tiny` form fails — and the
substituted-cell count appears in the `FitRecord`**; **the spectrogram formula is hand-computed on
a fixture with both zero and nonzero STFT coefficients and asserted to equal
`log(|STFT| + tiny)` — a log-power implementation fails it by the factor 2, and a `1e-30` floor
fails it on the zero bin**; `SpectrogramNorm` fit on train rows only (mutation fixture)
**with parameter shape `[C, F]` and hand-computed means/stds on an asymmetric 2-channel fixture
whose two channels have different per-frequency scales — proving statistics are not shared across
channels and that the reduction is over frames × time**; spectrogram shape/orientation pinned;
CPU determinism under fixed seed.

### 2.8 `src/dehyd/eval/exp_d.py` (new)

**Responsibility.** Exp D end to end: the CNN nested torch path, the two cheap baselines under
LOSO, per-family reporting, the GPU fold-array shard/merge, and the frozen comparison
statistics.

**Public API / content.**
- **Frame spine.** `build_frames_d(config, band, family)` — per QC-passed frame: subject,
  session_idx, frame_id, the family's stored signal (store v2 keys), target = the session's
  Δm% broadcast to frames (training only; scoring is session-level).
- **CNN nested path** — `run_cnn_family(config, band, family, fold, seeds)` per outer fold
  (one fold = one array task = the unit of work):
  - Inner loop: for each of the **6 frozen configs** (`ModelGridConfig.baseline_learning_rate ×
    baseline_weight_decay`) × 5 inner folds × 5 seeds: train with Adam(betas 0.9/0.999), MSE,
    batch 16, `max_epochs` 200, early stopping patience 15 / min-delta 1e-4, checkpoint on
    **inner-val session-MAE** (median frame→session aggregation of val predictions →
    `subject_balanced_mae` over session rows — the frozen `checkpoint_metric`), session-balanced
    sampling via a seeded `WeightedRandomSampler` (per-row weight `1/frames_in_session`,
    `num_samples=len(train)`, generator seeded from (run seed, fold, config, inner fold,
    seed) — a named derivation, recorded). **The full sampler/DataLoader contract, pinned because
    each option left open below changes BatchNorm statistics, the optimizer-step count, early
    stopping and hence the median epoch budget:** `WeightedRandomSampler(weights,
    num_samples=len(train), replacement=True, generator=g)` — *with* replacement, since drawing
    `len(train)` rows without replacement would merely permute the training set and discard the
    session balancing entirely; `DataLoader(ds, batch_size=16, sampler=sampler, shuffle=False,
    drop_last=True, num_workers=0, generator=g)` — `shuffle=False` (it is mutually exclusive with
    `sampler`, which already fixes the order) and `num_workers=0` so no worker RNG or completion
    order enters the trace; `drop_last=True`, so every batch is exactly 16 rows and **one epoch =
    `floor(len(train) / 16)` optimizer steps** — that is the epoch definition early stopping and the
    budget median count in. `drop_last=True` rather than keeping the short tail because BatchNorm in
    train mode is undefined on a 1-row batch (it would raise mid-run on any fold with
    `len(train) % 16 == 1`) and because under replacement sampling the dropped remainder is a random
    tail, not a systematically excluded subset. Loss `MSELoss(reduction="mean")` with **no**
    additional per-row weighting — the balancing lives in the sampler and is never applied twice.
    Each fit constructs its own `torch.Generator` from the named derivation and shares it between
    sampler and loader; nothing is drawn from the global RNG (trap 8). These are recorded
    implementation constants, not owner-gated: the freeze fixes `batch_size = 16` and says nothing
    about the loader, and each choice is justified in place rather than taken silently.
    This is `torch_fit.py`'s T18-protected algorithm —
    optimizer never sees validation data; per-epoch train trajectory a pure function of train
    data + seed — extended with batching/sampler and the config grid. Every fitted quantity
    (spectrogram norm where applicable, sampler weights, model state, selected epochs) emitted
    as `FitRecord`s at inner level.
  - Config selection: per-config score = mean over inner folds × seeds of the best-checkpoint
    inner-val session-MAE; winner via **`select_candidate`** (CandidateScore: MAE; simplicity
    rank 0 for all — same family; feature_dimension = flattened input size, constant; variance
    breaks residual ties) — the single tie-break source, unchanged.
  - Epoch budget = **median over the winning config's (inner fold × seed) selected epoch
    counts** (the harness rule `:650-655`, seed dimension included since batching makes every
    seed's trajectory distinct — stated, not hidden).
  - Final refit: per seed, all outer-training frames, exactly the budget, no early stopping, no
    validation subject; test frames → median per session → per-seed session predictions and
    scores (seeds scored separately, never ensembled — `:644-649`).
  - Determinism scope: CPU fixture tests are bit-asserted; GPU runs enable
    `torch.use_deterministic_algorithms(True, warn_only=True)` + seeded generators and report
    per-seed spread — no cross-run bit-identity claim on GPU (§0 ground rules).
- **Physics baseline** — `run_physics(config, band, sessions, store_dir)`: per frame, from the
  stored **unstandardized** raw signal: 10 GHz — Hann-windowed 534-pt FFT of the raw chirp-mean
  beat, target band [0.9, 1.5) m / background [1.5, 3.0] m (bins derived from the frozen
  constants via `beat_band_hz`; the 1.5 m boundary bin belongs to background only — half-open,
  `:829-843`); 77 GHz — Hann-windowed 256-pt Doppler FFT of the raw reduced slow-time signal,
  DC bin 0 vs bins 1..127 (`:893-914`). Scalar = `log10((P_t + ε)/(P_b + ε))`,
  `ε = 1e-12·(P_t + P_b)`; finite-output assertion over every QC-passed frame. Session value =
  median over frames (O-M9-4); per-fold 1-D least-squares on outer-training sessions → test
  predictions; also scored on inner folds (fit inner-train, score inner-val session-MAE) so the
  composite procedure has a real inner score (K=1 config; no selection). `FitRecord`s for the
  linear fit at both levels.
- **Session-index baseline** — reuses `models/baselines.fit_session_index_baseline` /
  `predict_session_index` verbatim (band-agnostic, shared — `:850-854, :915-916`), under the
  same folds; inner-scored the same way for the composite.
- **Shard/merge (CNN fold-array).** `--init-run-group` (per family × band): `validate_store`,
  `record_run` with a genuine QC manifest, `folds=fold_manifest(nested_loso_splits(...))`,
  `data_dir` per band, `extra={stage, band, family, config_hash (exp_b.config_fingerprint
  pattern), expected_subjects, expected_test_rows_by_fold}` — the M8 C19/C20/C21/C22 contract,
  reused not reinvented, **plus one new field the M8 contract had no need for**.
  `expected_test_rows_by_fold` is the authoritative per-fold row census the merge validates
  against: `{fold_id: {"test_subject": s, "n_session_rows": int, "n_frame_rows": int,
  "frame_rows_sha256": <hash>, "session_rows_sha256": <hash>, "seed_set": [1,2,3,4,5]}}`, where
  `frame_rows_sha256` is the SHA-256 of the canonical newline-joined
  `f"{subject}|{session_idx}|{frame_id}"` list of that fold's QC-passed **test** frames, sorted
  canonically, and **`session_rows_sha256` is the same construction over the distinct
  `f"{subject}|{session_idx}"` session identities**. Two hashes, not one, because the artifacts they
  guard are different files: the frame hash cannot validate the session-level predictions CSV (it is
  opaque to it), so a CSV that **substitutes** a wrong or duplicated session while preserving
  `n_session_rows` would pass a count-only check. **Seed is deliberately not part of the row
  identity** — it is the CSV's third column and is validated separately: the predictions CSV must
  contain exactly the cross product `session_identities × seed_set` for a CNN family (5 rows per
  session, no duplicates, no absences) and `session_identities × {seed 1}` for a deterministic
  family, with the seed convention recorded in the shard JSON. Both hashes and the census are
  needed because nothing already recorded can catch a truncated or stale
  shard: `record_run`'s manifest holds only cohort totals (`n_frames`/`n_subjects`/`n_sessions`,
  `provenance.py:214-228`), `fold_manifest` holds fold roles (subject ids), and
  `extra.expected_subjects` is a cohort-level list — so a shard that silently dropped test rows
  would have no reference to be rejected by, and a "n rows" check against group provenance
  (as this plan previously claimed) has no such number to read.
  Each `--fold N --run-dir P` task: independent `validate_store`, runs exactly one
  `run_cnn_family` fold, writes `exp_d_{family}_{band}_fold{N}.json` +
  per-fold predictions CSV via shared helpers, **each carrying its own realized
  `n_session_rows`/`n_frame_rows`/`frame_rows_sha256`/`session_rows_sha256`/`seed_set` computed the
  same way**; an unexpected exception propagates (C12). `merge_exp_d_folds(band, family, run_dir)`:
  fail-closed lineage validation of every shard (analysis_commit, config_hash, family, band, fold
  id, run_group_id) **and an exact match of the shard's realized census and both hashes — and of the
  predictions CSV's own recomputed session identities and seed cross product — against
  `expected_test_rows_by_fold[fold_id]`**; `completed_folds` computed from what is present and
  valid; **the family's summary is only produced when `completed_folds` = every selectable fold** —
  a partial merge is a named, non-reportable state, not a silently smaller cohort.
- **Per-family merged artifacts (the schema every comparison and §2.10 consume).** Named here
  because "per-family reporting" alone would let D10 be satisfied by a comparison-only summary that
  is neither independently auditable nor regenerable, and because §2.10's modal-config reduction
  needs a guaranteed source artifact. Every Exp D family — CNN and deterministic alike, one schema —
  emits under its run dir:
  - `predictions_{family}_{band}.csv` — session-level out-of-fold predictions, one row per
    `(subject, session_idx, seed)`: `fold_id, subject, session_idx, seed, y_true_delta_m_pct,
    y_pred, n_frames_aggregated`. Deterministic families (physics, session-index) write `seed = 1`
    once and set `"deterministic": true` in the metrics JSON — never five copies.
  - `metrics_{family}_{band}.json` — per-subject seed-averaged session-MAE vector (the input to
    every comparison), plus the family's own **MAE, RMSE and pooled Pearson r** with subject-cluster
    BCa CIs computed under the frozen metric-type-aware seed collapse (`:1193-1204`: additive for
    MAE, recompute-per-seed-then-average for RMSE/r), `n_eval`, `conditional_exploratory: true`,
    and the run's provenance/lineage fields.
  - `selection_{family}_{band}.csv` — per outer fold: selected `(lr, wd)`, the epoch budget and the
    per-(inner fold × seed) selected epoch counts it was the median of, the per-config inner scores,
    and `n_inner_folds`. This is the artifact §2.10's CNN modal reduction reads; deterministic
    families write the fitted linear coefficients instead of a grid selection (`K=1`, `selected
    config = "n/a"`), so the file exists for every family and the reader needs no per-family
    special-casing.
  - `per_subject_{family}_{band}.csv` — subject, per-seed MAEs, seed-averaged MAE, n_sessions —
    the descriptive error distribution and the composite's join key.
  Merge acceptance: all four files exist, are internally consistent (the metrics JSON's per-subject
  vector recomputes exactly from the predictions CSV; the selection CSV's budget equals the median
  of its own listed epoch counts), and cover exactly the selectable folds — **checked before the
  comparison stage is allowed to read them** (the comparison entrypoint validates this and refuses
  otherwise, naming the missing or inconsistent file).
- **Comparisons** — `summarize_exp_d(band, config, family_runs, exp_a_run)` implementing
  §Statistics `:1263-1281` exactly: per-subject session-MAE per family (seed-averaged,
  additive collapse); **pre-registered primary**: radar vs session-index (Wilcoxon +
  `mean_difference_ci`) — numerically the same comparison M7 reported, recomputed from the
  step-12 artifacts for one self-contained report; **composite procedure** (one comparison,
  uncorrected): per outer fold pick the best of {raw 1D-CNN, raw spectrogram, physics}
  (O-M9-3) by their inner-CV scores. **The splice happens at the per-subject metric level, not at
  the prediction level** — the earlier draft's "splice that family's test predictions" is undefined
  across families of different seed multiplicity (each CNN has 5 per-seed prediction sets; physics
  is deterministic with 1). Concretely: every family already produces, per test subject, its
  **seed-averaged per-subject session-MAE** (the frozen additive collapse, `:1193-1199`; a
  deterministic family averages one value, which is that value — never replicated into 5 pseudo
  observations, and no cross-seed prediction averaging anywhere, which `:644-649` forbids). Each
  subject is the test subject of exactly one outer fold, so the composite's per-subject vector is
  that fold's winning family's per-subject value, and the Wilcoxon + BCa comparison against radar
  runs on the N_eval paired per-subject values exactly as every other comparison does. Serialized
  as `composite_{band}.csv` with columns `subject, selected_family, inner_score, per_subject_mae,
  n_seeds_averaged`, so the per-fold winner is auditable and the radar pairing is one join on
  `subject`;
  **per-family exploratory Holm-3** (raw 1D-CNN, raw spectrogram, physics) via
  `holm_adjusted(family_size=3)`; ablations reported descriptively (their own MAEs + CIs, no
  comparison family). Radar predictions loaded from the step-12 run dir only after its
  provenance (commit, config) validates and the O-M9-5 bit-identity assert against the M7
  artifacts has passed. Every CI subject-cluster BCa; all `conditional_exploratory: true`;
  RNG offsets `RNG_OFFSET_EXPD_BASE = 300`, one named block of 10 per family, distinctness
  test extended.

**Frozen values.** All of `BaselineConfig`, `ModelGridConfig.baseline_*` (budget parity: 6 ≤
K=12 configs per learned family, same seed set — `:917-919`), `StatsConfig`
(`holm_family_baseline_per_family=3`, `composite_baseline_comparison=single_uncorrected`),
physics constants, A-M6-2, the harness refit rule.

**Acceptance criteria.** (T-M9-cnnpath, T-M9-physics, T-M9-expd-shard, T-M9-expd-compare —
§3.) Highlights: the CNN-path mutation property on a CPU fixture (mutating the held-out
subject's frames/targets leaves config selection, epoch budgets, spectrogram norms, sampler
weights, and refit states bytewise unchanged; only its predictions move — the T18 pattern over
the real Exp D composition); **a sampled-index trace fixture: for a fixed seed the exact sequence of
batch row indices and the optimizer-step count per epoch are pinned, proving `replacement=True`,
`floor(len(train)/16)` steps, uniform 16-row batches, and that two fits with the same seed but
different held-out data draw identical training batches** (C20); early stopping demonstrably
early-stops and the budget median is hand-checked; physics scalar finite on a zero-target-band fixture and hand-computed on a
synthetic two-tone signal; boundary-bin membership asserted; merge lineage negative fixtures
(each mismatched field raises, naming it) **including a valid-lineage shard that silently drops
one expected test row — same commit, config_hash, family, band, fold id, plausible counts — which
the `frame_rows_sha256`/census check must reject by name; a predictions CSV that drops a row the
shard JSON still counts; a CSV that substitutes or duplicates a session while preserving
`n_session_rows` (rejected by `session_rows_sha256`, the count-only check's blind spot); and a CSV
missing one seed of one session (rejected by the seed cross-product check)**; the four per-family
merged artifacts exist and are internally consistent, and the comparison entrypoint refuses to read
an incomplete or inconsistent family set naming the file; the composite CSV is written with its
per-fold winner and a fixture whose selected family alternates CNN/physics across folds reproduces
the hand-computed seed-averaged per-subject differences (and differs from the MAE of an ensembled
prediction, so a prediction-level splice fails the test); composite selection uses inner scores only (mutation
fixture); Holm family_size=3 pinned strictly stronger than len-2 on identical p-values.

### 2.9 `src/dehyd/features/store.py` (schema v2) + producers

**Responsibility.** Extend the per-session store with the Exp D per-frame signal arrays;
everything else unchanged.

**Content.** `STORE_VERSION = 2`. New keys (canonical QC-passed frame order, aligned with the
existing raw WST tensors): 10 GHz — `sig__raw_beat` `[N, 534] complex128` (chirp-mean via
`reduce_option_a` of the **raw, ungated** QC-passed frame — no bandpass, no trim, unstandardized:
both the CNN raw input's source and the physics input) and `sig__matched_iq` `[N, 2, 470]
float64` (= `preprocess_cube(sub, pre_default_gate, reduction="a", channel="iq")` — exactly the
frozen matched definition). 77 GHz — `sig__raw_slowtime` `[N, 256] float64` (mean over
fast-time and Rx axes of the raw cube — A-M6-2 (i)) and `sig__matched_iq` `[N, 2, 256] float64`
(real/imag of `preprocess_frame_77(frame)[:, :, 0].mean(axis=0)` — chain steps 1-5, Rx 0, mean
over the 27 gate bins — A-M6-2 (i-ablation); robust-standardized at input-build time, stored
raw). `build_session_npz_10ghz` / `build_session_npz_77ghz` extended in place;
`compute_fingerprint`/`validate_store` unchanged in shape — the `store_version` bump makes every
v1 store fail closed, which is the intended behaviour.

**Provenance.** Input definitions: `implementation_plan.md:807-820` (10 GHz raw/matched),
`:869-886` (77 GHz, A-M6-2), `:826-849`/`:893-914` (physics inputs). Store-size delta ≈ 1.6 MB
(10 GHz) / 0.8 MB (77 GHz) per session — negligible against the existing raw WST tensors.

**Acceptance criteria.** (T-M9-store) v1 sidecar fails `_check_match` on `store_version`; new
keys present with the exact shapes/dtypes above; `sig__matched_iq` (10 GHz) bytewise equal to
the WST chain's own preprocessed frames on a fixture cube (one definition, two consumers);
77 GHz matched equals the hand-composed chain on a fixture; frame-order alignment with
`frame_ids` asserted.

### 2.10 `src/dehyd/eval/frame_split.py` (new, exploratory-only) + entrypoint

**Responsibility.** The owner's sanctioned leaky evaluation, structurally isolated per the three
hard constraints (`implementation_plan.md:925-941`).

**The exact required matrix.** The owner's decision covers "both Exp C and every Exp D baseline"
(`:925-928`) — nothing else. D11 is therefore **16 exploratory runs**, enumerated so no
implementer has to infer them:

| Task | Unit | Bands | Runs |
|------|------|-------|------|
| ordinal (Exp C) | arm (a) thresholded-ordinal, arm (b) Frank-Hall | 10, 77 | 4 |
| regression (Exp D) | `cnn1d_raw`, `cnn1d_matched`, `spec2d_raw`, `spec2d_matched`, `physics`, `session_index` | 10, 77 | 12 |

**Exp A's radar regressor is deliberately NOT in this matrix.** An earlier draft of this section
named "Exp A's [modal config] for the regression task"; Exp A is not an Exp D baseline and the
owner's sanction does not reach it, so running it would be an unauthorized addition of exactly
the kind §0's invariant forbids. Flagged in §6 as available for separate owner authorization; not
implemented, not run. (`session_index` under a frame split is near-degenerate — every test frame's
session is trained on — and is run only because the owner's text says "every Exp D baseline"; its
output carries a `degenerate_by_construction: true` note.)

**Content.** `run_frame_split(config, band, task, unit, k=5)`: build the pooled frame table —
- **classical (both Exp C arms):** per-frame pooled WST vectors reconstructed from the stored raw
  tensors at that arm's **modal LOSO configuration** — the most-selected (feature key, family,
  params) triple read from the arm's `selection_table_{band}.csv` **artifact** (never recomputed),
  ties broken toward the configuration selected by the lowest fold id;
- **CNN families:** store v2 signals at the family's cross-fold reduced hyperparameters. The
  reduction from per-outer-fold selections to one configuration, stated because "the LOSO-selected
  (lr, wd)" is not single-valued: **(lr, wd)** = the modal `(lr, wd)` pair over the outer folds,
  ties broken toward the pair selected by the lowest fold id; **epoch budget** = the integer
  `int(np.floor(np.median(b)))` over the per-fold budgets `b` of *only the folds that selected the
  modal pair* (floor, so an even count is resolved downward rather than by numpy's mean-of-middle,
  which can be non-integral). Both read from the family's merged LOSO summary artifact;
- **physics / session_index:** no hyperparameters; the frozen scalar/estimator refit per leaky fold.

`sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=config.run.seed + 900)` over
frames **inside this module only** — `splits.py` is never imported here (asserted by test). Per
fold: fit on 80%, score 20% — frame-level ordinal metrics for the C arms (plain accuracy included,
deliberately: the paper-comparable number) and frame-level regression MAE for the D families;
report per-fold values + mean ± sd.

**tuned-ε in this path is computed from training *frames*, never from the stored session-level
scales.** An earlier draft said "ε is computed from the training frames' sessions only", which is
not implementable without leaking: the store holds one `prelog__*` tuple per **session**, and that
tuple is already a median over *all* of that session's frames (`extraction.py::_prelog_scale`,
written per session at `store.py:246`; 77 GHz analogous). Under a pooled frame KFold nearly every
session straddles the split, so consuming the stored tuple would fit ε on held-out frame values —
a *second*, unsanctioned leak. The owner sanctioned subject-overlapping frame splits (`:929-931`);
fit-on-train-only still governs every fitted transform, and nothing in the sanction licenses
fitting on the scored rows. So when a modal config's log branch is `tuned`, `frame_split.py`
recomputes the pre-log scale from the store's **raw scattering tensors** (`raw_key`, `order_key` —
already present, the same arrays `_prelog_scale` consumes) restricted to the fold's training frame
rows, **keeping the frozen subject-balanced hierarchy intact** (`:477-500`) — only the innermost
population narrows from "the session" to "that session's training frames". Per order o ∈ {1, 2}:
`_per_frame_prelog(S, meta)[o]` reproduces `_prelog_scale`'s per-frame intermediate exactly
(time-mean → mean over order-o paths → mean over channels, *before* its median-over-frames step);
**session scale** = median over that session's **training** frames of those per-frame values (the
same median `_prelog_scale` takes, over a subset); **per-subject value** = mean over that subject's
sessions that have ≥ 1 training frame; **`scale_o`** = median over subjects with ≥ 1 such session;
`ε_o = k · scale_o` with the frozen `k = 0.1` and the 1e-6 non-finite/non-positive fallback. Order 0
stays linear. Pooling all training frames into one median was the earlier draft's choice and is
**rejected**: it silently overweights subjects and sessions contributing more training frames,
discarding the subject-balancing `:485-489` states as the reason for the two-stage form, and it
would be a new computation-affecting post-A/B estimator that the frame-split sanction does not
authorize. The narrowed innermost population is recorded in the output JSON as
`{"tuned_eps_aggregation": "frozen_hierarchy_training_frames_only"}`.

**Output isolation is structural, and does not go through `record_run`.** `provenance.record_run`
unconditionally creates `results_dir/runs/<stamp>_<rev>/provenance.json`
(`src/dehyd/provenance.py:190-208`), so calling it here would violate §2.10 and D11 no matter how
the metrics files are named. Instead `frame_split.py` owns
`write_exploratory_provenance(config, band, task, unit, manifest, out_dir, *, data_dir)` — the
manifest and band-correct data root are **parameters, not implied**, because `_hash_inputs(config,
manifest, data_dir=None)` needs the QC manifest to know which files to hash and silently defaults to
the 10 GHz root (`provenance.py:169-173`); an exploratory 77 GHz run must pass
`require_77ghz_dir(config)` or it would hash 10 GHz files under a 77 GHz label — the M8 C19/C22
failure mode. The entrypoint builds the same QC manifest the LOSO runs build (the pooled QC-passed
frame table for that band, which the run needs anyway) and passes it through. To avoid depending on
private helpers, step 9 extracts a **public** `provenance.build_provenance_payload(config, manifest,
folds=None, extra=None, data_dir=None) -> dict` from `record_run`'s body — `record_run` then calls it
and writes, so its output stays byte-identical (pinned by the existing provenance tests) — and
`write_exploratory_provenance` calls the same function with `folds=None` and
`extra={...}`, then writes `provenance_frameSplit_leaked_exploratory.json` **under the exploratory
root only** — same payload builder, different root, no `runs/` directory created or touched.
`folds=None` is deliberate (there are no LOSO folds here), but the seeded KFold's `k_folds` +
`random_state` are **not** by themselves a record of what ran: nothing in the payload pins the frame
order the KFold indexes into (the manifest summary stores only `n_frames`/`n_subjects`/`n_sessions`,
`provenance.py:214-228`), nor the LOSO artifact whose modal configuration and epoch budget *define*
this computation — so two runs over a different frame order or a different source artifact could
carry indistinguishable provenance. `extra` therefore records all of:
`{"leaky_protocol": true, "never_report": true, "task", "unit", "band", "k_folds": 5,
"kfold_random_state": config.run.seed + 900, "tuned_eps_aggregation",
"frame_order_sha256"` — SHA-256 of the canonical newline-joined `subject|session_idx|frame_id`
list **in the exact order the KFold indexes**, so a permutation changes it —
`"fold_assignment_sha256"` — the same list annotated `…|fold{j}` after the split, pinning the
realized assignment rather than the recipe — `"source_run": {"run_dir", "analysis_commit",
"config_hash", "artifact_rel_path", "artifact_sha256"}` for the LOSO selection-table / merged-summary
artifact the modal configuration was read from, and `"resolved_config"` — the modal
`(feature_key, family, params)` or `(lr, wd, epoch_budget)` actually used, so the exploratory number
is traceable to the LOSO selection it claims to mirror`}`. Every output path
passes `_require_exploratory_path(p)`, an **allowlist**: the resolved path must be inside
`Path(config.paths.results_dir) / "exploratory_frame_split"` and its filename must contain
`frameSplit_leaked_exploratory` — anything else raises (an allowlist, not a `runs/`-substring
refusal, so a path that is merely spelled differently cannot slip through). Layout:
`results/exploratory_frame_split/{band}/{task}_{unit}_frameSplit_leaked_exploratory.{json,csv}`;
every JSON carries `{"leaky_protocol": true, "never_report": true}` at the top level.

**Acceptance criteria.** (T-M9-frame-split) No import of `eval.splits` (AST/text assert); the
allowlist guard raises on a `results/runs/...` target, on a path outside the exploratory root, and
on an untagged filename inside it; filenames tagged; KFold seeded/deterministic; **the tuned-ε
mutation property — perturbing the raw scattering coefficients of any held-out frame (including
frames of a session that is otherwise in training) leaves ε and every fitted state bytewise
identical, while perturbing a training frame's coefficients moves ε (the power companion), and
`frame_split.py` never reads a `prelog__*` store key (text/AST assert)**; `_per_frame_prelog`
agrees with `_prelog_scale` when the frame subset is a whole session (median over that session's
frames reproduces the stored tuple bytewise — one definition, two frame populations); **the
subject-balanced hierarchy is preserved — on a fixture with deliberately unequal subject/session/
frame counts (one subject with many training frames, one with few) ε equals the hand-computed
session-median → subject-mean → subject-median value and differs from the pooled-frame median, so
a pooled implementation fails; a session with zero training frames drops out of its subject's mean
and a subject with no such session drops out of the median**; the CNN
modal-pair + floor-median-budget reduction hand-computed on a fixture with an even fold count and
a tie (lowest-fold-id resolution asserted); the modal classical config is read from the selection
table artifact and a fixture where the artifact and a plausible recomputation disagree proves the
artifact wins; **a complete end-to-end invocation of `run_frame_split_exploratory.py` on a
synthetic store creates no new file or directory anywhere under `results/runs/`** (snapshot the
tree before and after and assert equality) while the exploratory provenance file exists and
validates; **`record_run`'s output is byte-identical after the `build_provenance_payload`
extraction (same-inputs comparison against a pre-extraction pin, plus the existing provenance
tests); an exploratory 77 GHz invocation hashes against the 77 GHz root in a two-distinct-data-roots
fixture and a missing/short manifest is refused rather than silently hashing nothing** (C21);
**exploratory provenance distinguishes runs that differ in what actually ran: permuting the frame
order changes `frame_order_sha256` and `fold_assignment_sha256` (a same-content different-order run
is never provenance-identical), pointing at a different selection artifact changes
`source_run.artifact_sha256`, a source artifact whose `analysis_commit`/`config_hash` fail lineage
validation is refused before any fitting, and `resolved_config` equals the modal configuration the
artifact records** (C24); the 16-row matrix above is enumerated in one module-level table and a test asserts the
CLI accepts exactly those `(task, unit)` pairs and rejects `exp_a`; outputs never appear in any
`metrics_exp_*` file; `tests/test_no_leakage.py` untouched (`git diff --exit-code`).

### 2.11 Entrypoints — `experiments/run_ordinal.py`, `run_baselines.py`, `run_frame_split_exploratory.py`

**Content.** `run_ordinal.py` and `run_baselines.py` follow `run_clock_decoupling.py`'s pattern
(multi `--config`, `--band`, `load_config` → `protocol_freeze_guard(config)` pre-flight,
`record_run` with band-correct `data_dir`, `SLURM_CPUS_PER_TASK` workers, results under
`config.paths.results_dir`). `run_frame_split_exploratory.py` follows the same CLI/pre-flight
shape but **must not call `record_run`** — it calls
`frame_split.write_exploratory_provenance` instead, for the reason given in §2.10 (`record_run`
always creates `results/runs/<stamp>_<rev>/`, which D11 forbids for this path).
- `run_ordinal.py`: `--subset 6subjects` XOR `--full-cohort`; composes
  `configs/exp_a_regression{,_77ghz}.yaml + configs/exp_c.yaml`.
- `run_baselines.py`: `--family {cnn1d_raw, cnn1d_matched, spec2d_raw, spec2d_matched, physics,
  session_index, comparisons}`. CNN families take `--init-run-group` XOR `--fold N --run-dir P`
  XOR `--merge-folds --run-dir P` (the M8 variant CLI pattern); cheap families and
  `comparisons` take `--subset` XOR `--full-cohort` (comparisons additionally
  `--exp-a-run-dir`, `--family-run-dir` args, validated for lineage). **Full-cohort mode
  validates `run.seed_set == (1,2,3,4,5)` and `run.device` appropriate to the family** (cuda
  allowed only for CNN families; the WST/numpy canonical-backend policy is untouched — DL
  baselines are the one authorized GPU path, `implementation_plan.md:1326-1329`). Smoke mode =
  mechanism-only (no performance token in stdout or files, tested), allows `seed_set=[1]` and
  `device=cpu` — run-level config only; every M6-frozen section stays at its frozen value in
  both modes (the guard enforces this; smoke differs by subset/seeds/device, per CLAUDE.md's
  smoke rule, and max_epochs stays 200 with early stopping doing the shortening).
- `run_frame_split_exploratory.py`: `--band {10ghz,77ghz}`, `--task {ordinal, regression}`,
  `--unit` restricted per task to exactly the §2.10 matrix (`ordinal` → `arm_a`|`arm_b`;
  `regression` → the six D families; `exp_a` and any other value rejected by the parser) — a thin
  wrapper over `frame_split.py`; refuses to run unless the referenced LOSO run dirs exist and
  validate; writes only through `_require_exploratory_path`.

**Acceptance criteria.** (T-M9-entrypoints) Flag mutual-exclusivity matrix; smoke emits no
performance token; full-cohort seed-set/device validation fires; comparisons refuse a
lineage-mismatched input dir naming the field; 77 GHz paths hash against the 77 GHz root
(two-roots fixture — the C22 lesson).

### 2.12 `scripts/ibex/` — `run_exp_c.sbatch`, `run_exp_d_cheap.sbatch`, `run_exp_d_cnn.sbatch`, `submit_exp_d_cnn.sh`

**Content.**
- `run_exp_c.sbatch` — clone of `run_exp_b.sbatch` (CPU 16 / 64 G / 04:00 — justified: same
  store, comparable candidate count per fold, Exp B measured ≈ 1-1.3 h), `BAND`/`MODE` env.
- `run_exp_d_cheap.sbatch` — physics + session-index + (separately invoked) comparisons; small
  CPU job (4 cores / 16 G / 01:00).
- `run_exp_d_cnn.sbatch` — **one file, `STAGE=init|fold|merge` dispatch, carrying NO `#SBATCH`
  resource directives at all** (the C24 doctrine: the header is parsed before `STAGE` exists);
  every resource is a CLI flag on the specific `sbatch` invocation in the wrapper. The fold
  stage requests `--gres=gpu:1`.
- `submit_exp_d_cnn.sh` — **git-free from day one** (the M8 step-10.5 lesson, commit e88fd33):
  repo root from `${BASH_SOURCE[0]}`, requires and reads `REVISION`, unsets `DEHYD_GIT_*`;
  per (FAMILY, BAND): submits `STAGE=init` sized as its own batch allocation and blocks
  (`sbatch --wait`, `set -e` — a failed init never reaches the array, C23); captures `RUN_DIR`
  from the init stage's final stdout line; submits `STAGE=fold` as `--array=1-16` (SLURM task
  id → fold index; a task whose fold is non-selectable exits 0 having written a named
  no-op marker) with `ARRAY_TIME` env-sized from the step-10 GPU smoke measurement (default
  08:00:00 until measured — recorded, then re-sized); submits `STAGE=merge` with
  `--dependency=afterany:<array-id>`; **every** `sbatch --parsable` output normalized with
  `${var%%;*}` before any use (C25).

**Acceptance criteria.** (T-M9-sbatch — the `test_exp_b_ibex_scripts.py` patterns re-applied)
`run_exp_d_cnn.sbatch` contains no `#SBATCH` resource directive (file inspection); wrapper
normalizes job ids on a `"12345;ibex"` fixture; a failing init aborts before array submission
(fixture); `RUN_DIR` parsing robust to preflight noise; the wrapper never invokes `git`.

---

## §3 Tests

| Group | File | What it proves |
|-------|------|-----------------|
| T-M9-metrics | test_metrics.py | Hand-computed class-unit MAE / adjacent accuracy / QWK / confusion counts; QWK **defined (0.0) for single-class truth vs a varying predictor and for a constant predictor vs multi-class truth, cross-checked against `cohen_kappa_score(weights="quadratic", labels=[0..4])`; NaN only for empty input and both-sides-constant-and-equal** (O-M9-8); pooled ≠ subject-balanced on an unequal fixture; confusion orientation |
| T-M9-ordinal | test_ordinal.py | Cutpoints/weights are train-only fitted quantities (mutation fixtures); cutpoints from in-sample predictions not targets; strict-increase nudge on all-tied cutpoints; knn unweighted; O-M9-7 weights hand-computed; Frank-Hall cumulative/difference recovery hand-checked; negative-difference floor+argmax; `OrdinalViabilityError` on missing class; determinism |
| T-M9-selection | test_selection.py | Each rung of `select_candidate_ordinal`'s order decides exactly when higher rungs tie; higher-QWK-wins; NaN-QWK loses at equal MAE; `n_evaluable_inner_folds = 0` is incomparable even with finite MAE/variance; `select_candidate` byte-unchanged and its existing tests pass unedited, while the `SIMPLICITY_RANK` exact-dict assertion (`tests/test_selection.py:79`) is updated to include the six ordinal keys and keeps the frozen base ordering |
| T-M9-harness | test_harness.py, test_m8_pin.py | Step-1 pins bytewise intact after the `_viability_reason` edit on every 1-D-y path; knn reason string unchanged; 2-D-y missing-class cell → named reason, no fit, NaN score; **coverage predicate is a pure function of inner-training rows — permuting/deleting classes in inner-val and outer-test rows leaves every cell's reason bytewise identical; a class absent from the whole bundle still blocks the fit (`ordinal_missing_class_3_in_inner_train`), i.e. the constant `{0..4}` predicate, never `set(y[:,1])`** (C1); 2-D-y with `score_fn=None` raises; `git diff --exit-code tests/test_no_leakage.py` |
| T-M9-parallel | test_exp_b.py, test_exp_c.py, test_exp_d.py | exp_b serial-vs-parallel bit-identity green through the `fold_parallel` extraction; exp_c/exp_d equivalents |
| T-M9-expc-provider | test_exp_c.py | `OrdinalFeatures` X bytewise == `StoreBackedFeatures` per branch; y columns [L, class] correct; tuned-ε identical to Exp A's on the same (fk, train set); session_idx aligned; **`assert_exp_c_fit_authorized` negative matrix (unauthorized family id, off-grid `ord_a_svr` `(C, ε)`, `base_family` disagreeing with the candidate id, mismatched cutpoint quantiles, off-grid Frank-Hall `C`) each raises `ExpCProtocolError` naming the field with no `.fit()` reached; arm (b)'s `active` record carries no `model_family` and arm (a)'s carries the base family** (C3, trap 2) |
| T-M9-expc-leak | test_exp_c.py | T16-pattern fit-record property over the real composition: inner-val label/class mutation leaves every inner-train fit (cutpoints, weights, scaler, base state, tuned-ε) bytewise identical; inner-train mutation moves them (power companion) |
| T-M9-expc-mutation | test_exp_c.py | End-to-end synthetic-store outer-mutation over the REAL two-arm Exp C composition (the M8 T-M8-outer-mutation pattern): only the held-out subject's predictions/scores may change |
| T-M9-expc-viability | test_exp_c.py | Fabricated fold with a class missing from **one** inner-train set: those cells carry the named reason and every candidate is still scored and a winner still selected, from the remaining evaluable folds only (`nanmean`/`nanstd`, `n_evaluable_inner_folds = 4`, recorded in the selection table) — the plain-mean aggregation would NaN every candidate here and the test fails against it; **all** inner folds missing → the fold contributes no score via the existing `SelectionError` path, re-raised naming the test subject and missing classes; QWK-undefined val fold falls back to MAE-only ranking (C2) |
| T-M9-expc-report | test_exp_c.py, test_run_ordinal.py | Arm (a)/(b) both summarized; pooled seed-collapse used for all three CIs; confusion = per-seed counts averaged; per-subject distribution present; `conditional_exploratory` true; no baseline-comparison field; **the O-M9-8 counters (`n_single_class_truth_val_folds`, `n_qwk_nan`) present at both CV levels and consistent with a fixture built to contain exactly one single-class-truth validation fold**; smoke surfaces no performance token; RNG offsets pairwise distinct incl. Exp B's and D's |
| T-M9-cnn | test_cnn.py | Architecture shapes/param-count pins; per-signal robust standardization shares nothing across frames; **per-variant spectrogram inputs (matching §2.7 exactly, not a blanket claim): the two **raw** branches bypass the robust step (a scaled stored array scales the pre-norm spectrogram), while both **matched** branches consume robust-preprocessed signals — 10 GHz the store's already-standardized `sig__matched_iq` bytewise equal to `matched_input_10`'s output with no second standardization, 77 GHz `matched_input_77`'s robust-standardized output bytewise equal and invariant to scaling the stored input**; **the spectrogram formula hand-computed as `log(\|STFT\| + finfo.tiny)` on zero and nonzero coefficients (log-power or a `1e-30` floor fails it)**; `SpectrogramNorm` train-only (mutation fixture) **with `[C, F]` parameter shape, frames×time reduction, and hand-computed statistics on an asymmetric 2-channel fixture proving no cross-channel sharing**; spectrogram shape/channel conventions (O-M9-6) pinned |
| T-M9-cnnpath | test_exp_d.py | CPU-fixture contract: early stopping stops early; epoch budget = median over (fold × seed); optimizer never sees val (T18 common-prefix property); the full CNN-path mutation property (selection, budgets, norms, sampler weights, refit states invariant under held-out mutation); sampler weight formula; **the sampled-index trace: exact batch index sequence and optimizer-step count pinned for a fixed seed, proving `replacement=True`, uniform 16-row batches, `floor(len(train)/16)` steps per epoch, and identical training batches under a changed held-out set** (C20); median frame→session aggregation |
| T-M9-physics | test_exp_d.py | Hand-computed scalar on a synthetic two-tone signal; finite on zero-target-band; boundary bin (1.5 m) in background only; 77 GHz DC-vs-motion partition exact; per-fold linear fit train-only (mutation) |
| T-M9-expd-shard | test_exp_d.py, test_run_baselines.py | Init/record_run real-contract checks (manifest, folds manifest, data_dir per band, extra fields at `provenance["extra"]` incl. `expected_test_rows_by_fold` with per-fold `rows_sha256`); every lineage-mismatch field raises by name; **a shard with perfect lineage that silently drops one expected test row is rejected by the census/`frame_rows_sha256` check, and so is a predictions CSV missing a row the shard JSON still counts** (C7); **a CSV that substitutes or duplicates a session at unchanged `n_session_rows` is rejected by `session_rows_sha256`, and one missing a single seed by the seed cross-product check** (C14); **the four per-family merged artifacts (predictions/metrics/selection/per-subject) exist, recompute consistently from each other, cover exactly the selectable folds, and the comparison stage refuses an incomplete or inconsistent family set by name** (C15); partial `completed_folds` → named non-reportable state, never a smaller cohort; unexpected task exception propagates |
| T-M9-expd-compare | test_exp_d.py | Composite selects by inner scores only (mutation fixture) over exactly {raw 1D-CNN, raw spectrogram, physics}; **the composite is spliced at the per-subject seed-averaged metric level — a fixture whose winner alternates CNN (5 seeds) / physics (deterministic) across folds reproduces the hand-computed per-subject differences, deterministic values are never replicated into 5 pseudo-observations, and the MAE-of-ensembled-predictions alternative fails the test** (C13); Holm family_size=3 pinned; primary vs session-index reproduces M7's comparison on a fixture; radar input refused without the O-M9-5 bit-identity precondition |
| T-M9-store | test_store.py | v1 store fails closed on `store_version`; new key shapes/dtypes; 10 GHz matched == WST-chain frames bytewise; 77 GHz matched == hand-composed chain steps 1-5/Rx 0/bin-mean; frame-order alignment |
| T-M9-frame-split | test_frame_split.py | No `eval.splits` import; **output-path allowlist** raises on a `results/runs/` target, on any path outside `results/exploratory_frame_split/`, and on an untagged filename; tagged filenames; seeded KFold determinism; **a complete CLI invocation creates nothing under `results/runs/` (before/after tree snapshot) while writing its own exploratory provenance** (C6); **the 16-row `(task, unit, band)` matrix is the CLI's accepted set and `exp_a` is rejected**; CNN modal-pair + floor-median-budget reduction hand-computed incl. an even count and a lowest-fold-id tie; classical modal config read from the selection-table artifact, not recomputed (C5); **frame order and fold assignment are hashed into provenance, and the source LOSO artifact's identity/lineage is recorded and validated (C24)**; **tuned-ε is a function of training frames only — perturbing any held-out frame's raw coefficients (including one whose session is otherwise in training) leaves ε and every fitted state bytewise identical, perturbing a training frame moves it, no `prelog__*` store key is read, and `_per_frame_prelog` reproduces the stored session tuple bytewise when the subset is a whole session** (C10); outputs absent from every `metrics_exp_*` artifact |
| T-M9-entrypoints | test_run_ordinal.py, test_run_baselines.py | Flag exclusivity; full-cohort seed-set/device validation; smoke no-performance-token; two-data-roots 77 GHz hashing; comparisons lineage refusal |
| T-M9-sbatch | test_exp_d_ibex_scripts.py | No `#SBATCH` resource directives in the STAGE-dispatch file; `--parsable` `;cluster` normalization; failed-init abort; `RUN_DIR` parse robustness; wrapper is git-free (no `git` invocation in the script text) |

`tests/test_no_leakage.py`: **zero changes** — verified by `git diff --exit-code` as an
acceptance step. Expected total: the M8 baseline (767 passed / 16 skips) plus the T-M9-* groups.

---

## §4 Definition of done

| ID | Criterion |
|----|-----------|
| D0 | Plan reviewed through the loop to closure; every Step 0b item carries an owner decision; **O-M9-8 decided as (8a) by the owner on 2026-07-30 — gate discharged, and the single-class-fold / NaN-QWK counts are reported so the choice's empirical size is visible**; decided completions propagated into `plans/implementation_plan.md` at step 0.5 (before any other M9 source) with their post-A/B chronology and computation-affecting label stated plainly |
| D1 | Full suite green (`uv run python -m pytest`) including every T-M9-* group; `git diff --exit-code tests/test_no_leakage.py` clean |
| D2 | Step-1 pins bytewise intact after the harness edit; the 2-D-y viability path proven on fixtures; existing Exp A/B behaviour byte-unchanged (pins + full suite) |
| D3 | Exp C: cutpoints, class weights, tuned-ε, scalers, and model states audited as fitted quantities at both CV levels; T-M9-expc-leak and T-M9-expc-mutation green; both arms selected only via `selection.py`; fold-viability rules implemented exactly as frozen (`:793-801`) with named reasons — the coverage predicate constant-`{0..4}` and provably independent of non-training labels, aggregation over evaluable inner folds with `n_evaluable_inner_folds` recorded per fold, and only an all-non-evaluable fold non-selectable; every Exp C fit passes `assert_exp_c_fit_authorized` |
| D4 | Exp D CNN path: T18-pattern mutation property green on the CPU fixture; config selection via `select_candidate`; epoch-budget rule implemented and hand-checked; spectrogram norm / sampler weights / model states audited at both levels — the norm's `[C, F]` shape, frames×time reduction and train-only fit all pinned, and the transform is the literal `log(\|STFT\| + tiny)` magnitude form; budget parity holds (6 ≤ K=12 per learned family, same seed set) |
| D5 | Physics and session-index baselines run under the identical folds with train-only fits and inner scores; physics finite-output assertion green over both real cohorts |
| D6 | Store v2 built and `--validate`-clean for both bands from the clean M9 commit (steps 9.5/9.6); v1 stores fail closed |
| D7 | Mechanism-only smokes green: Exp C both bands (CPU), every Exp D family both bands (CPU, seed_set [1]), one GPU array-task smoke per CNN family; no performance value surfaced by any smoke |
| D8 | Full-cohort Exp C, both bands, on IBEX: metrics/predictions/selection/confusion artifacts exist and are regenerable by one command |
| D9 | Exp A re-run at the M9 commit, both bands, **bit-identical to the M7 prediction artifacts** (O-M9-5); a mismatch stops the milestone and is escalated, not papered over |
| D10 | All 8 CNN fold-array groups merged with `completed_folds` = every selectable fold, fail-closed lineage **and per-fold row-census (frame + session `*_rows_sha256`, seed cross-product)** validation throughout, **each family emitting its full four-artifact set (predictions / metrics / selection / per-subject) and the comparison stage refusing to run on an incomplete or inconsistent set**; cheap baselines complete; the comparison report (primary, composite, Holm-3, ablations-descriptive) written for both bands with subject-cluster CIs and `conditional_exploratory` labels |
| D11 | Exploratory frame-split complete for **all 16 runs of the §2.10 matrix** (both Exp C arms × 2 bands; all six Exp D families × 2 bands) and no others — Exp A absent unless separately authorized; outputs only under `results/exploratory_frame_split/` with tagged filenames and `never_report` markers, written through the allowlist and its own provenance writer; **`results/runs/` gains no file or directory from any exploratory invocation** |
| D12 | HISTORY.md per-step entries written as work happened; SECOND_CHAPTER.md §8 written from the full LOSO results, disclosing every A-M9/O-M9 completion with its true chronology; frame-split absent from §8 |

---

## §5 What could go wrong (known traps)

1. **2-column y meets sklearn validation.** Some sklearn internals call `check_X_y(y,
   ...)` with 1-D enforcement. The scaler ignores y and the ordinal wrappers do their own
   validation, but any future harness path that touches `bundle.y` assuming 1-D will break or —
   worse — silently flatten. Mitigation: the `_score` fail-fast assert (§2.4), plus tests that
   pass 2-D y through the full engine.
2. **`ord_b_frank_hall` has no legal `model_family` value for the guard.** The frozen whitelist
   is the five regressor families; forcing a fake value would corrupt the protocol record, and
   extending the whitelist would reopen M6. The §2.6 design (feature-axes-only `active` +
   config-level `ExpCConfig` validation + exp_c's own completeness contract) is the narrow
   path — any drift here (e.g. someone "helpfully" adding `model_family: ridge` to the (b)
   record) makes the audit lie. Asserted directly in T-M9-expc-provider.
3. **Class-coverage viability differs between smoke and full cohorts.** With 6 subjects and
   5 inner folds, an inner-training set can genuinely lack a class (a subject with few eligible
   sessions), making many cells non-evaluable — the smoke must *report* this structurally
   (reason counts) and still complete; if every candidate is non-evaluable in a fold the
   existing `SelectionError` path must name the fold and the missing classes, not die on a
   bare "non-finite MAE" (the M8 trap-5 lesson, applied to classes). The sharper trap is the
   **aggregation**: because the predicate is candidate-independent, one non-evaluable inner fold
   makes the harness's plain `np.mean` NaN for *every* candidate, which silently escalates "one
   inner fold lost a class" into "this outer fold produced no ordinal result" — stricter than the
   frozen rule. §2.3's evaluable-folds-only reduction is the fix, and both cases are tested
   separately (T-M9-expc-viability).
4. **Cutpoints from targets instead of predictions.** Quantiles of the training *targets* look
   almost right and leak nothing — but they are not the frozen rule
   (`family_a_regressor_in_sample_predictions_inner_train`) and behave differently on a biased
   regressor. A fixture where target- and prediction-quantiles differ pins the correct source.
5. **Searchsorted orientation.** `side="right"` vs `"left"` moves boundary predictions between
   classes. Pinned by a hand fixture (predictions exactly on cutpoints).
6. **QWK tie-break computed from re-predicted rather than stored predictions.** Recomputing
   val predictions after selection would double-fit and can drift (rf/gbm seeds); the O-M9-1
   rule reads the *stored* first-seed `val_predictions` from the same `InnerResult`s the MAE
   came from. Asserted by making a recompute drift detectable in the fixture.
7. **The epoch-budget median's population.** "Median of the epochs selected across the inner
   folds" (`:650-655`) was written for one config × deterministic fits; with 5 seeds × 5 folds
   for the winning config the median is over 25 values. Taking it over folds-of-seed-means (or
   the first seed only) is a different number. §2.8 pins folds × seeds; the test hand-computes
   it.
8. **Sampler RNG bleeding across fits.** A shared torch generator would make fit k's data
   order depend on fits 1..k-1, breaking the mutation property (a held-out mutation would
   shift *training* batches). Every fit constructs its own generator from the named derivation
   (§2.8); the mutation test would catch a violation.
9. **BatchNorm in eval vs train mode at prediction time.** Predicting with `model.train()`
   active (running stats updating) silently makes predictions depend on prediction-set
   composition — a leakage-adjacent bug the mutation property catches only if the test
   predicts twice. Explicit `model.eval()` at every predict; a test predicts the same rows
   twice and asserts identity.
10. **GPU nondeterminism vs the project's bit-identity language.** No cross-run bit-identity
    claim on GPU; per-seed spread is the reported uncertainty; CPU fixtures carry every
    bit-assert. Any claim stronger than that in SECOND_CHAPTER.md §8 would be wrong.
11. **`ibex.yaml` is paths-only; nothing sets `device=cuda` implicitly.** The CNN full-cohort
    config must set it explicitly, and the entrypoint validates family-vs-device (§2.11). The
    frozen numpy-backend policy applies to reported WST *features*; the DL baselines are the
    authorized GPU consumers (`:1326-1329`) — SECOND_CHAPTER.md §8 states this split.
12. **Spectrogram normalization is the one *fitted* input transform.** Robust per-signal
    standardization is unfitted (per frame); the per-frequency mean/std is fit on training
    frames and is a leakage vector if ever computed on the pooled set. It is a `FitRecord` at
    both levels and mutation-tested (T-M9-cnn / T-M9-cnnpath).
13. **Physics baseline fed the standardized signal.** Robust standardization destroys absolute
    power — the ratio would be meaningless. The physics path reads the *unstandardized* stored
    signals (which is why the store keeps them raw, §2.9); a test computes the scalar on a
    scaled copy and asserts the ratio is scale-invariant only through the frozen ε coupling.
14. **The 16-task array with N_eval < 16.** Fold indices are positions in the selectable-fold
    list, not subject ids; a task whose index exceeds the list exits 0 with a named no-op
    marker (never an error, never a silent absence the merge can't distinguish from a crash).
    Merge distinguishes "no-op marker present" from "shard missing".
15. **Modal-config definition for the frame split.** "Most-selected configuration" needs a
    deterministic tie-break (lowest fold id, §2.10) and must be computed from the *selection
    table artifact*, not recomputed from memory — otherwise the exploratory run silently
    diverges from what the LOSO run actually selected. The CNN families need a second reduction
    the classical arms don't: per-fold `(lr, wd)` **and** per-fold epoch budgets both have to
    collapse to one value, and "the LOSO-selected budget" is not single-valued — §2.10 fixes
    modal pair + `floor(median)` over the folds that chose that pair, because a plain
    `np.median` over an even count returns a non-integral epoch count.
16. **The temptation to give Exp C a baseline comparison.** The session-index baseline predicts
    the class *perfectly* (the class IS the session index), so any radar-vs-baseline framing
    for Exp C is degenerate, and the freeze registers none. Exp C reports its ordinal metrics
    absolutely; the paper-comparable number lives only in the unreported frame split. Writing
    any C-vs-baseline p-value would be an undisclosed protocol invention (the M8 C16 lesson).
17. **Comparisons against a drifted Exp A.** If the step-12 bit-identity assert fails, the
    *tempting* move is to compare against the fresh predictions anyway ("the code is newer").
    That converts a detected fault into a silent protocol change — D9 makes the failure a
    milestone-stopping event with owner escalation.
18. **Store rebuild ordering.** Any code change after step 9.5 moves the commit and invalidates
    the freshly built v2 stores (the M8 e88fd33 lesson — HISTORY.md 2026-07-29). Nothing merges
    after 9.5 except run artifacts and journal files; if a code fix is unavoidable, the stores
    are rebuilt again from the fix commit, cost acknowledged, never waived.
19. **The exploratory path inheriting a *second* leak by reusing a session-level fitted
    quantity.** The sanctioned leak is subject overlap across the frame split — nothing more. The
    stored `prelog__*` scales are per-session medians over *all* of a session's frames, so
    consuming them under a pooled frame split fits ε on the very rows being scored. §2.10
    recomputes the pre-log scale from the raw tensors restricted to training frame rows for exactly
    this reason. The general form of the trap: **any** store key that is already an
    aggregate-over-a-session is a leakage vector in a frame split even though it is perfectly safe
    under LOSO — the frame-split module must reach for raw arrays, not summaries.
20. **Frame-split outputs drifting toward reportability.** The isolation is structural
    (path guard, tags, markers, test) — but the last line of defense is §8's author. D12 makes
    "absent from §8" an explicit acceptance criterion, checked at review of the chapter text.

---

## §6 Flagged gaps in implementation_plan.md + proposed amendments

- **A-M9-1 (family (a) search space) — OWNER-APPROVED 2026-07-30, decided AFTER Exp A's and
  Exp B's full-cohort results were visible.** Gap: §C never names the continuous L-predictor's
  search space. Resolution: reuse Exp A's frozen per-band space under the ordinal objective,
  concretized as one shared Stage 1 (ord_a_ridge anchor) + two Stage-2 arms (§0 Step 0 item 1).
  Computation-affecting. Propagated at step 0.5 with chronology.
- **O-M9-1 … O-M9-7 — OWNER-APPROVED, Step 0b, decided one by one 2026-07-30, AFTER Exp A's
  and Exp B's results were visible.** Each a genuine gap in the frozen text (selection-order
  tail and QWK aggregation; Frank-Hall decision rule; composite membership; physics session
  mapping; comparison pairing source; 10 GHz spectrogram channels; weight normalization); the
  decisions are recorded in Step 0b and propagated at step 0.5 with this chronology, **each to
  the section that actually contains the ambiguity it resolves — O-M9-3 and O-M9-5 land in
  §Statistics (`:1267-1281`), not only §D, and O-M9-7 lands in §C; the step-0.5 row now
  enumerates all of A-M9-1 and O-M9-1..8 with destinations so D0 can check for omissions.**
  **Computation-affecting: O-M9-1, -2, -3, -4, -6, -7.** O-M9-6 belongs in that list — the
  real/imag 2-channel convention determines the spectrogram input tensor, `in_channels`, the
  first conv layer's parameters, the fitted per-frequency normalization state, and therefore the
  predictions; an earlier draft of this bullet grouped it with O-M9-5 as merely fixing
  "procedure/inputs", which understated it and is corrected here and at step 0.5.
  **Not computation-affecting: O-M9-5 only** — it fixes which artifacts the comparison is paired
  against and adds a bit-identity precondition; the numbers it admits are, by that very assert,
  the M7 numbers.
- **O-M9-8 — OWNER-APPROVED 2026-07-30 (option 8a), review-derived, decided AFTER Exp A's and
  Exp B's results were visible and after the review loop closed — QWK's
  undefinedness condition.** `:798-800` motivates the MAE fallback with "QWK is undefined on a
  single-class validation set", which is true only when the label set is inferred from the data;
  on the fixed 5-class grid this task mandates, single-class truth against a varying predictor has
  a defined κ (= 0), and only zero expected disagreement is undefined. Resolution: implement the
  frozen *behaviour* (never error; fall back to class-unit MAE when QWK is undefined) with the
  mathematically correct trigger — NaN iff empty input or zero expected disagreement (§2.1).
  **Computation-affecting on tie-breaks and on the QWK CI's skip-and-count**, since folds the
  looser reading would have skipped now contribute a defined value. Flagged rather than silently
  implemented because it is the one place M9 reads a frozen sentence as motivation rather than as
  specification. **The two alternatives were put to the owner and (8a) was chosen (2026-07-30):**
  - **(8a) — CHOSEN.** Undefined ⟺ empty input or zero expected disagreement on the fixed 5×5 grid.
    Matches `sklearn.metrics.cohen_kappa_score(..., weights="quadratic", labels=[0,1,2,3,4])`;
    single-class validation folds contribute a defined κ (usually 0) to the secondary tie-break and
    to the QWK CI. Owner's stated reason for preferring it: reproducibility against the reference
    implementation an examiner would check the numbers against.
  - **(8b) — not chosen.** Undefined ⟺ (8a) **or** either side has a single distinct class.
    Single-class validation folds would be skipped by the MAE fallback and by the CI's
    skip-and-count, as `:798-800` reads on its face. Its strongest arguments, recorded because the
    chapter must state what was rejected and why: it keeps M9's record of never reinterpreting the
    freeze, a single-class fold carries no ordinal-agreement information so averaging its κ = 0
    dilutes the mean, and under (8a) two candidates on such a fold are treated differently (a
    varying predictor scores 0 while a constant one is skipped as genuinely undefined).
  **Gate satisfied.** The fail-closed gate that would have stopped the milestone at step 0.5 is
  discharged by this decision; step 2 may write `metrics.py` implementing (8a). Because (8a) admits
  folds (8b) would skip, §2.6 and §2.8 **must report how often it fires**: the count of
  validation folds whose true classes are single-valued, and of QWK values that came back NaN, goes
  into `metrics_exp_c_{band}.json` and the bootstrap's skip-and-count field — so SECOND_CHAPTER §8
  can state the empirical size of the choice rather than only its rationale. (Kept in the plan body
  rather than the review block's `Deferred to owner`, which the loop protocol reserves for threads
  deadlocked after their debate rounds; O-M9-8 was never a reviewer⇄author disagreement — both sides
  agreed on the mathematics.)
- **A gap deliberately NOT filled: an ordinal baseline for Exp C.** The freeze defines no
  baseline for the ordinal task, and the natural candidate (session-index) is degenerate
  (trap 16). Inventing one now would be an undisclosed post-hoc addition with no textual
  anchor. Exp C reports absolute ordinal metrics; nothing more. (No amendment.)
- **A gap deliberately NOT filled: Exp A's radar regressor under the exploratory frame split.**
  The owner's 2026-07-30 sanction covers "both Exp C and every Exp D baseline" (`:925-928`); Exp A
  is neither. A frame-split Exp A number would be the most directly paper-comparable of all, which
  is precisely why adding it without authorization would be the wrong call — it is the one number
  most likely to be *wanted* as a headline, and the freeze's sanction stops short of it. Available
  for a one-line owner authorization; not implemented, not run, and §2.10's matrix rejects it at
  the CLI. (No amendment.)
- **A second gap deliberately NOT filled: ordinal-task variants of the D baselines.** §D
  defines the baselines as Δm% regressors contesting Exp A; running them as classifiers too
  would be new design. The owner's frame-split covers the classification-comparison curiosity,
  unreported. (No amendment.)
- No other scientific-protocol gap is proposed; every constant Exp C/D consumes is taken as
  frozen at M6.

---

## §7 Open items this milestone resolves or carries

**Resolves:** the ordinal estimator family and its train-only cutpoint/weight machinery; the
ordinal selection path routed through `selection.py`; the harness's ordinal viability
completion; the CNN nested path (grid × early-stopping × budget × refit) as reusable machinery;
the store's signal schema (v2); the Exp D baseline results and the pre-registered comparisons
on both bands; the sanctioned exploratory frame-split, built and quarantined; full Exp C
results on both bands.

**Fixed here, NOT open at M10+:** the 2-column-y convention and the `_viability_reason`
contract; `fold_parallel.py` as the single fold-parallel implementation; the ordinal families'
fitted-state extractors; the CNN-path FitRecord vocabulary (`spectrogram_norm`,
`sampler_weights`, `cnn_state`, `epoch_budget`); the A-M9/O-M9 resolutions once decided —
reopening any needs a prior authoritative amendment.

**Carries to M10+:** Exp E (interpretability, on the Exp B model), Exp F (confound check —
reuses the Exp A-selected feature rule A-M6-4), Exp G (fusion — consumes both bands' stores),
Exp H (cross-experiment statistics, the R=200 robustness bootstrap, figures/tables); the
optional run-startup QC caching flagged since M7.

---

## Review focus

Four places this milestone most plausibly hides a real flaw; press hardest here.

1. **The ordinal estimators' fitted quantities.** Cutpoints and class weights are *new* fitted
   quantities computed inside `.fit()` — verify the plan's contract actually forces them
   train-only at both CV levels (T-M9-expc-leak/-mutation), and that the cutpoint source is the
   frozen in-sample-predictions rule, not targets.
2. **The family-(b) guard path (§2.6, trap 2).** The `active`-without-`model_family` design is
   the one place a fit runs without a whitelist-validated family axis. The compensating control is
   now `assert_exp_c_fit_authorized` (§2.6 step 2), which validates the candidate's family id,
   base-family mapping, grid membership and wrapper constants before every fit — check it actually
   covers every path that reaches `.fit()` (both stages, both arms, `_final_refit` included) and
   that its negative tests can't pass vacuously.
3. **The CNN path's selection/budget/refit chain (§2.8).** This generalizes the T18 contract
   with batching, a sampler, a config grid, and a fitted input normalization — each a new place
   validation data could touch training. The mutation property's coverage (does it really pin
   selection, budgets, norms, sampler weights, and refit states?) is the load-bearing check.
4. **Frame-split isolation (§2.10) and the M9 completions' chronology (§0/§6).** Verify the
   isolation is structural rather than promised, and that every post-A/B completion is named,
   labelled for computation-effect, and scheduled for step-0.5 propagation — nothing folded
   into "frozen before results".

---

## Plan review (reviewer ⇄ author)

<!-- REVIEW-STATUS: REVIEW_COMPLETE -->
Reviewer: NO MORE COMMENTS (2026-07-30)
<!-- REVIEW-TURN: 9 -->

### Comment ledger (append-only)

- **C1** (blocking | turn 1 | section: "§2.4 `harness.py` — `_viability_reason`") — The proposed coverage predicate is both leakage-sensitive and weaker than the frozen rule: it compares inner-train classes with `set(y[:, 1])`, so control flow can depend on labels outside inner-train (including inner-validation and, because `OrdinalFeatures` mirrors the all-row `StoreBackedFeatures` bundle, the outer-test subject), and if class c is absent from the full bundle it silently stops requiring c. `implementation_plan.md:793-800` instead fixes the requirement as all five S0–S4 classes, independent of any validation/test labels. Make the predicate compare inner-training labels against the constant `{0,1,2,3,4}` (or `range(configured_n_classes)` whose frozen value is 5), never against the full bundle, and add mutation tests showing that changing non-training class labels cannot change the reason while a globally absent class still makes the inner fit non-evaluable. I verified the current provider/harness contract in `src/dehyd/eval/exp_a.py::StoreBackedFeatures.data_for` and `src/dehyd/eval/harness.py::_score_candidates_on_fold`: bundles contain all session rows and the row mask is applied only after `data_for`.
- **C2** (major | turn 1 | section: "§2.4/§2.6 ordinal fold viability") — The plan does not define an aggregation path that implements “the fold contributes no ordinal score.” The current harness constructs each candidate with `np.mean(per_fold)` and `np.std(per_fold)` (`src/dehyd/eval/harness.py:329-337`), so one candidate-independent missing-class cell makes every candidate’s aggregate MAE/variance NaN and `select_candidate_ordinal` rejects the entire outer fold. That conflicts with the frozen wording at `implementation_plan.md:793-800`, which distinguishes a non-evaluable inner fold from configs and says the fold contributes no score, and it contradicts T-M9-expc-viability’s “all-folds-missing” threshold. Specify the exact rule (normally aggregate each candidate over its evaluable inner folds, require a stated minimum number/count, and make only an all-non-evaluable outer search non-selectable), implement it in the ordinal composition without changing Exp A/B aggregation, and test both one-missing-fold and all-missing-fold cases.
- **C3** (major | turn 1 | section: "§2.6 family-(b) guard path") — The claimed compensating control does not validate the computation that actually runs. `protocol_freeze_guard(config, active=...)` checks only keys present in `active`, while `REQUIRED_ACTIVE_KEYS_C = ... minus model_family` checks only feature-key completeness; neither binds `candidate.family == "ord_b_frank_hall"` nor constrains its fitted C to `ExpCConfig.proportional_odds_c_grid`. Thus an unauthorized family or arbitrary C could pass immediately before fit even though the plan calls this the guard-before-every-fit contract. Add a named Exp-C fit guard that validates the ordinal wrapper identity, base-family mapping for every `ord_a_*`, Frank-Hall implementation, and candidate parameters against the frozen grids before every fit; add negative tests that a wrong family and off-grid C are rejected. I checked `src/dehyd/features/protocol_freeze.py::_check_active`/`protocol_freeze_guard` and `src/dehyd/eval/harness.py::require_complete_active`; both have exactly the present-key behavior the plan describes.
- **C4** (major | turn 1 | section: "§2.1 `quadratic_weighted_kappa`") — “NaN when a single distinct class is present on either side” discards valid QWK values and changes both the O-M9-1 tie-break and reported/bootstrap metrics. A constant predictor against multi-class truth has a defined κ (typically 0); only the genuinely zero-denominator/no-common-variation case is undefined. I ran `.venv/Scripts/python.exe` with sklearn’s fixed labels `[0,1,2,3,4]`: true `[0,1,2,3,4]`, pred all 0 returned `0.0`; true `[0,0]`, pred `[0,1]` also returned `0.0`; only both all 0 returned `nan`. Define undefinedness from the actual expected-disagreement denominator (with the fixed 5-class grid), remove the blanket “single-class either side” pre-check, and update T-M9-metrics accordingly.
- **C5** (major | turn 1 | section: "§2.10 exploratory frame split") — The implementation surface does not cover the owner-approved scope “for every C/D result.” The classical paragraph names an Exp-C modal configuration and then an **Exp-A** regression configuration (Exp A is not D), while it never specifies separate runs for both Exp-C arms or the Exp-D physics and session-index baselines; the CNN sentence also leaves undefined how per-outer-fold `(lr, wd, epoch_budget)` selections collapse to one modal configuration/budget. Enumerate the exact `(task, arm/family, band)` matrix required by D11, including both C arms and all six D variants, remove the unintended Exp-A exploratory model unless separately authorized, and define/test the cross-fold modal/tie/epoch-budget reduction for each CNN family.
- **C6** (major | turn 1 | section: "§2.10–§2.11 frame-split isolation") — §2.11 says **all three** entrypoints call `record_run` following `run_clock_decoupling.py`; the real `record_run` always creates `results/runs/<stamp>_<rev>/provenance.json` (`src/dehyd/provenance.py:190-239`). That directly violates §2.10 and D11 (“outputs only under `results/exploratory_frame_split/`” and absent from every runs artifact), even if metrics use tagged filenames elsewhere. Give the exploratory entrypoint a dedicated provenance writer rooted under `results/exploratory_frame_split/` (or explicitly call the existing payload helpers without `record_run`), make the output root an allowlist rather than merely refusing paths containing `runs/`, and test that a complete invocation creates no new directory or file under `results/runs/`.
- **C7** (major | turn 1 | section: "§2.8 CNN shard/merge") — The merger promises to validate each shard’s “n rows” against group provenance, but the proposed init record contains only the ordinary aggregate manifest counts, fold roles, and `extra.expected_subjects`; none is an authoritative per-fold expected frame/session-row count. The current `record_run` schema stores only total `n_frames/n_subjects/n_sessions`, not row identities or per-fold counts (`src/dehyd/provenance.py:214-228`). A truncated/stale shard could therefore self-report a plausible row count with no stated reference to reject it. Snapshot `expected_test_rows_by_fold` (preferably canonical `(subject, session, frame_id)` identities or hashes) into group provenance at init, validate every shard/predictions file against it, and add a negative test where a valid-lineage shard silently drops one expected test row.
- **C8** (major | turn 1 | section: "§0b O-M9-6 / §6 chronology classification") — O-M9-6 is computation-affecting, contrary to §6’s classification of it with O-M9-5 as merely fixing “procedure/inputs.” Choosing two separate real/imag spectrogram channels rather than another channel convention changes the input tensor, parameterization (`in_channels`), fitted normalization state, and potentially predictions. Since this choice was made after A/B results and the milestone invariant requires every completion to be labelled for computation effect, classify O-M9-6 explicitly as computation-affecting everywhere it is summarized and carry that wording into the step-0.5 source-of-truth amendment and SECOND_CHAPTER chronology.
- **C9** (major | turn 3 | section: "§1 step 0.5 / §6 source-of-truth propagation") — Step 0.5 still says “§Statistics untouched,” so it cannot faithfully propagate the completions the plan says it propagates. O-M9-3 explicitly resolves an ambiguity in `implementation_plan.md:1267-1274` (whether ablations enter the composite/Holm family), O-M9-5 defines the comparison artifact/precondition, and O-M9-7 supplies the missing class-weight scale in §C; the parenthetical propagation list omits O-M9-5/-7 and places O-M9-3 only in §D while leaving the contradictory Statistics text unchanged. Amend every authoritative location that contains the resolved ambiguity—especially §Statistics for O-M9-3—and enumerate O-M9-1..8 (or their exact destination sections) in step 0.5 so D0 can verify that none was lost.
- **C10** (blocking | turn 3 | section: "§2.10 tuned-ε frame split") — The proposed tuned-log implementation can read held-out frame values through the existing session-level `prelog__*` entries. The repository stores one `_prelog_scale` tuple per whole session (`src/dehyd/features/extraction.py:76-86`, written by `src/dehyd/features/store.py:246`; 77 GHz is analogous), not a per-frame scale. In a pooled frame KFold, almost every session appears on both sides; selecting “training frames’ sessions” and consuming that stored tuple therefore incorporates the same session’s held-out frames into a fitted ε. The owner sanctioned subject-overlapping frame splits, not additional fit-on-test leakage, and the hard fit-on-train-only invariant still governs fitted transforms. Compute ε directly from the raw scattering tensors indexed to the fold’s training frame rows only (with an explicitly defined aggregation for this exploratory frame population), apply it unchanged to test frames, and add a mutation test proving held-out-frame raw coefficients cannot change ε or any fitted state.
- **C11** (major | turn 3 | section: "§2.7 `spectrogram`") — The transform called “log-magnitude” is actually `log(|STFT|² + 1e-30)`, while the frozen Exp-D text says **log-magnitude** (`implementation_plan.md:821-825,887-892`). Squaring changes the transform, and `1e-30` is not a float64 “tiny” guard (`np.finfo(float).tiny ≈ 2.2e-308`) but a new arbitrary floor that can clip low-energy bins; neither choice appears in `BaselineConfig` or the post-A/B completion ledger. Implement the literal `log(|STFT| + ε)` with a genuinely numerical zero guard whose exact rule is recorded, or treat the power/floor choice as a computation-affecting owner amendment with chronology. Pin the formula on hand-computed zero and nonzero STFT coefficients, not only output shape.
- **C12** (major | turn 3 | section: "§2.7 `SpectrogramNorm`") — “Per-frequency mean/std” is not executable for multi-channel `[frame, channel, frequency, time]` tensors without naming the reduction and retention axes. It can mean one statistic per frequency shared across real/imag channels, or one per `(channel, frequency)`, and it can average over frames only or frames×time; these produce different fitted states and predictions. The section also does not explicitly say whether the raw spectrogram branches bypass the robust per-signal builders (the frozen config calls robust raw/matched normalization and train-only spectrogram normalization “two distinct” rules). Specify the exact input to each of the four spectrogram variants and the exact `SpectrogramNorm` parameter shape/reduction axes, then test the hand-computed statistics and channel non-sharing on an asymmetric two-channel fixture.
- **C13** (major | turn 3 | section: "§2.8 composite comparison") — “Splice that family’s test predictions” leaves the frozen seed-collapse undefined when outer folds select families with different seed multiplicity: each CNN has five prediction sets, while physics is deterministic. The Statistics invariant requires per-subject additive metrics to average each subject’s per-seed metric before inference and forbids implicit ensembling (`implementation_plan.md:1193-1204,644-649`). Define the composite at the per-subject metric level (or define a common seed axis, explicitly repeating deterministic predictions without treating copies as observations), and specify how it is serialized and paired with radar. Add a fixture whose selected family alternates CNN/physics across folds and prove the result equals the hand-computed seed-averaged per-subject differences rather than MAE of an ensembled prediction.
- **C14** (major | turn 3 | section: "§2.8 `expected_test_rows_by_fold`") — C7’s frame census is improved, but the session-level predictions CSV still has no authoritative identity reference: group provenance stores only `n_session_rows`, not which `(subject, session_idx)` rows are expected. A CSV can replace one expected session with a duplicate/wrong session while preserving `n_session_rows`; the frame `rows_sha256` is opaque and cannot validate a session-level file unless the merger also has the underlying identity list. Add a separate canonical `session_rows_sha256` (define whether seed is part of the row identity) or store the expected session identities, validate the CSV exactly against it, and add the same-count substitution negative test—not only a missing-row test.
- **C15** (major | turn 3 | section: "§2.8 per-family reporting") — The plan promises “per-family reporting” but never defines the merged result/artifact schema or acceptance criteria for any Exp-D family. The shard section names per-fold files and the comparison section consumes `family_runs`, yet there is no contract requiring session-level OOF predictions for all seeds, inner scores/selected `(lr, wd)` and epoch budgets, selection-frequency/stability, per-subject errors, or each family’s own MAE/RMSE/r and subject-cluster CIs. Without this, D10 can be satisfied by a comparison-only summary that is not independently auditable or regenerable, and §2.10’s modal CNN configuration has no guaranteed source artifact. Specify common per-family outputs (including deterministic-family conventions), metrics and CIs, selection tables, and merge acceptance tests before comparisons are allowed to run.
- **C16** (question | turn 3 | section: "§6 O-M9-8 / review state") — O-M9-8 is now explicitly “OWNER SIGN-OFF PENDING,” but the review block’s `Deferred to owner` section remains `_(none)_` and no automatic fallback step says when the literal frozen rule is restored if no owner response arrives. Because this loop is unattended and deferred items do not block review closure, should O-M9-8 be recorded in `Deferred to owner` with the two exact alternatives and a fail-closed implementation gate (no source work until decided), rather than living only as an untracked pending note in the plan body?
- **C17** (major | turn 5 | section: "§2.10 tuned-ε aggregation", re: C10) — The applied leak fix unnecessarily changes the tuned-ε estimator from the frozen subject-balanced hierarchy to a pooled-frame median. Subject and session identities still exist in the frame KFold, so the frozen hierarchy remains meaningful and executable without reading test frames: compute each training session’s scale from **that session’s training-frame subset only**, take the mean of those training-session scales per subject, then the median over training subjects, exactly as `implementation_plan.md:477-500` defines. Pooling frames instead overweights subjects/sessions with more training frames and is a new computation-affecting post-A/B rule not authorized by the frame-split decision. Preserve the frozen hierarchy with train-only frame subsets, or list the pooled-frame replacement as a new owner decision; update the equivalence/mutation test to cover unequal subjects/session/frame counts where the two estimators differ.
- **C18** (major | turn 5 | section: "§2.7 spectrogram inputs", re: C12) — The applied “all spectrogram inputs are stored unstandardized” edit contradicts the actual store contract and the plan itself. For 10 GHz, `sig__matched_iq` is specified as `preprocess_cube(..., channel="iq")`; the current `preprocess_frame` returns `to_channels(..., pre.standardize)` (`src/dehyd/preprocess/pipeline.py:73`), so this tensor is already robust-standardized (as §2.7 line 571 also says). For 77 GHz the v2 matched tensor is stored pre-standardization, but the frozen matched definition says robust-standardize per channel before use (`implementation_plan.md:877-892`). Thus neither matched spectrogram branch can simply STFT an “unstandardized matched” tensor while still representing the frozen matched-preprocessing ablation. Specify per variant: raw spectrograms bypass robust normalization; matched spectrograms consume the fully matched-preprocessed signal (10 GHz already standardized in store; 77 GHz standardized after load), then receive train-only `SpectrogramNorm`. Correct the scaled-copy tests accordingly.
- **C19** (major | turn 5 | section: "§2.7 `SpectrogramNorm` zero variance") — Dividing by `std + np.finfo(float).tiny` is unsafe for a `(channel, frequency)` cell with zero training variance: a non-identical test value is amplified by ~1e308 and can overflow or dominate the CNN. The existing torch normalization precedent replaces zero std with 1.0 (`src/dehyd/models/torch_fit.py::_normalize_stats`), and zero-variance spectrogram bins are plausible. Define the frozen fallback explicitly (`scale = 1.0` where std == 0, otherwise std, or another owner-approved rule), record it in the `FitRecord`, and test a constant training bin with a different finite validation value. A finiteness assertion only before normalization does not cover this failure.
- **C20** (major | turn 5 | section: "§2.8 CNN nested path") — The minibatch training algorithm is still not reproducibly specified. `WeightedRandomSampler` leaves material choices open: `replacement`, DataLoader `shuffle`, `drop_last`, `num_workers`, whether the final short batch is kept, loss reduction/any additional weighting after sampling, and what exactly constitutes one epoch. These affect BatchNorm statistics, optimizer steps, early stopping, and the median epoch budget. Pin the actual DataLoader/sampler contract (including generator ownership and an epoch = exactly `num_samples` sampled rows, with the chosen last-batch behavior) and add a small index-trace fixture proving the sampled batches and optimizer-step count for a fixed seed.
- **C21** (major | turn 5 | section: "§2.10 `write_exploratory_provenance`") — The promised “same provenance content” is not supported by the stated signature `write_exploratory_provenance(config, band, task, unit)`: the real `_hash_inputs` requires the QC manifest and a band-correct `data_dir`, and defaults to the 10 GHz root (`src/dehyd/provenance.py:153-188`). The exploratory writer neither accepts/builds the manifest in its contract nor states `require_77ghz_dir(config)` for 77 GHz, recreating the M8 C19/C22 failure mode. Define the exact manifest/fold metadata it records, pass the band-correct data root, avoid relying on private helpers without a public extraction, and add the two-distinct-data-roots fixture for an exploratory 77 GHz invocation.
- **C22** (minor | turn 5 | section: "§2.3 selection acceptance") — “`test_selection.py` unchanged and green” is impossible after adding keys to `SIMPLICITY_RANK`: the current test asserts exact dict equality at `tests/test_selection.py:79`. Change the acceptance claim to “the existing five-family ranks and `select_candidate` behavior remain pinned,” update that expected mapping to include the ordinal keys, and keep a byte/behavior pin on `select_candidate`; only `tests/test_no_leakage.py`, not `test_selection.py`, is frozen against edits.
- **C23** (major | turn 7 | section: "§3 T-M9-cnn", re: C18) — The applied per-variant correction is contradicted by the test matrix, which still says robust per-signal standardization “is absent from the spectrogram branches.” That assertion would require both matched spectrogram variants to bypass the robust step, directly opposing §2.7's corrected contract (10 GHz matched is already robust-standardized in store; 77 GHz matched is robust-standardized by `matched_input_77` before STFT) and could make the implementation satisfy one section only by failing the other. Rewrite T-M9-cnn to state absence only for the two **raw** spectrogram branches and explicitly pin the robust-preprocessed inputs for both **matched** branches, matching §2.7's per-variant acceptance criteria. This was found on the required full fresh plan-body read.
- **C24** (major | turn 7 | section: "§2.10 exploratory provenance", re: C21) — The claim that frame-fold membership is “fully determined by `k_folds` + `kfold_random_state` + the recorded frame order” is not supported by the proposed payload: neither `write_exploratory_provenance`'s stated `extra` nor the current `record_run` manifest summary records frame identities/order, and `src/dehyd/provenance.py:214-228` stores only `n_frames`, `n_subjects`, and `n_sessions`. It also does not identify/hash the LOSO selection artifact whose modal configuration and epoch budget define the exploratory computation. Consequently, two runs can carry indistinguishable provenance while using a different frame order/fold assignment or different modal-config source artifact. Add canonical ordered-frame and per-fold assignment hashes (or explicit identities) plus the validated source run/artifact lineage and SHA-256/selected modal configuration to exploratory provenance; test that permuting frame order or changing the referenced selection artifact changes or invalidates provenance. I checked the current `_hash_inputs`/`record_run` payload construction in `src/dehyd/provenance.py:153-239`; raw-file hashes do not encode either derived dependency.

### Thread state

| ID | State | Rounds |
| --- | --- | --- |
| C1 | applied | — |
| C2 | applied | — |
| C3 | applied | — |
| C4 | applied | — |
| C5 | applied | — |
| C6 | applied | — |
| C7 | applied | — |
| C8 | applied | — |
| C9 | applied | — |
| C10 | applied | — |
| C11 | applied | — |
| C12 | applied | — |
| C13 | applied | — |
| C14 | applied | — |
| C15 | applied | — |
| C16 | applied | — |
| C17 | applied | — |
| C18 | applied | — |
| C19 | applied | — |
| C20 | applied | — |
| C21 | applied | — |
| C22 | applied | — |
| C23 | applied | — |
| C24 | applied | — |

### Debate

_(none)_

### Deferred to owner

_(none)_

### Resolved

- C1 (applied, turn 2): bundle-relative class-coverage predicate was both leakage-sensitive and weaker than the freeze → §2.4 now compares inner-training labels against the constant `ORDINAL_CLASSES = (0,1,2,3,4)`, never `set(y[:,1])`, with the two reasons spelled out (frozen wording; all-row bundles) and two new tests: non-training-label independence, and globally-absent class still blocks. §3 T-M9-harness and §4 D3 updated. [checked: `plans/implementation_plan.md:793-800`; `src/dehyd/eval/exp_a.py` `StoreBackedFeatures.data_for`; `src/dehyd/eval/harness.py:297-322` (`_score_candidates_on_fold` applies the row mask after `data_for`)]
- C2 (applied, turn 2): confirmed `harness.py:329-337` builds `CandidateScore` with `np.mean`/`np.std` over all inner folds, so one candidate-independent missing-class cell NaNs every candidate and rejects the whole outer fold — stricter than `:793-800`, which makes the fold contribute no score only when *all* configs are non-evaluable. §2.3 now specifies the rule explicitly: exp_c aggregates `nanmean`/`nanstd(ddof=0)` over each candidate's evaluable inner folds, carries `n_evaluable_inner_folds` on `OrdinalCandidateScore`, requires ≥ 1 (the frozen text's own threshold — deliberately not a tighter invented minimum), and lives in exp_c so Exp A/B aggregation is byte-unchanged; §2.6 wires it into `_ordinal_candidate_scores` and re-raises `SelectionError` naming subject + missing classes; counts published in the selection table. §3 T-M9-expc-viability / T-M9-selection now test one-missing-fold and all-missing separately; §5 trap 3 and §4 D3 updated. [checked: `src/dehyd/eval/harness.py:297-340`; `plans/implementation_plan.md:793-800`]
- C3 (applied, turn 2): verified `protocol_freeze._check_active` only validates keys *present* in `active` (`protocol_freeze.py:116-136`) and `harness.require_complete_active` only checks the key set (`harness.py:163-183`) — so nothing bound the candidate's family or params for arm (b). §2.6's `before_fit` is now a three-check sequence with a new `assert_exp_c_fit_authorized(candidate, config)` validating family id ∈ the six authorized ids, base-family/id agreement, `base_params` ∈ the frozen `ModelGridConfig` grid (Stage-1 anchor `alpha=1.0`), wrapper constants == `ExpCConfig`, and Frank-Hall `C` ∈ `proportional_odds_c_grid` + impl tag + `max_iter`; raises `ExpCProtocolError` naming the field. Negative matrix added to §2.6 acceptance + §3 T-M9-expc-provider (spy on `build_estimator` proves no fit is reached); Review focus item 2 rewritten. [checked: `src/dehyd/features/protocol_freeze.py:116-168`; `src/dehyd/eval/harness.py:41-44,163-183`; `src/dehyd/config.py:300-317,383-398`]
- C4 (applied, turn 2): reviewer's sklearn result reproduces by hand on the fixed 5×5 grid — true `[0,1,2,3,4]` vs pred all-0 gives numerator = denominator = 30/16, hence κ = 0, not NaN; the blanket single-class pre-check would have discarded defined values from the O-M9-1 tie-break and the QWK CI. §2.1 now defines undefinedness solely as empty input or zero expected disagreement, keeps the frozen *behaviour* (never error; MAE fallback whenever NaN), and records the divergence from `:798-800`'s motivating parenthetical as **O-M9-8** in §6 — computation-affecting on tie-breaks and on the QWK skip-and-count, flagged for owner sign-off at step 0.5 with the revert path stated, since it is the one place M9 reads a frozen sentence as motivation rather than specification. §3 T-M9-metrics, §4 D0 and the §0 invariant updated. [checked: `plans/implementation_plan.md:798-800`; hand-computed QWK on the reviewer's two fixtures]
- C5 (applied, turn 2): the owner's sanction is "both Exp C and every Exp D baseline" (`implementation_plan.md:925-928`) — Exp A is neither, so the draft's Exp-A regression modal config was an unauthorized addition. §2.10 now opens with the explicit 16-run matrix (both C arms × 2 bands; the six D families × 2 bands), removes Exp A and makes the CLI reject it, flags it in §6 as available for separate one-line owner authorization, notes `session_index`'s frame-split degeneracy, and defines the CNN cross-fold reduction that was undefined: modal `(lr, wd)` with lowest-fold-id tie-break, epoch budget `int(floor(median(...)))` over the folds that chose the modal pair. §1 step 14, §3 T-M9-frame-split, §4 D11, §5 trap 15 updated. [checked: `plans/implementation_plan.md:925-941`]
- C6 (applied, turn 2): confirmed `record_run` unconditionally creates `results_dir/runs/<stamp>_<rev>/provenance.json` and raises rather than overwriting (`provenance.py:190-208`), so §2.11's "all three follow `run_clock_decoupling.py`'s pattern" contradicted §2.10/D11. §2.11 now excludes the exploratory entrypoint from `record_run` and §2.10 gives it `write_exploratory_provenance` reusing the same payload helpers under the exploratory root; the path check is now an **allowlist** (must resolve inside `results/exploratory_frame_split/` *and* carry the `frameSplit_leaked_exploratory` tag) rather than a `runs/` refusal. Acceptance adds a before/after tree snapshot proving a complete invocation creates nothing under `results/runs/`; §4 D11 and §3 updated. [checked: `src/dehyd/provenance.py:190-239`]
- C7 (applied, turn 2): confirmed `record_run`'s manifest stores only cohort totals (`n_frames`/`n_subjects`/`n_sessions`, `provenance.py:214-228`) — there was no per-fold reference for the merge's promised "n rows" check, so a truncated shard was undetectable. §2.8's init now snapshots `extra.expected_test_rows_by_fold = {fold: {test_subject, n_session_rows, n_frame_rows, rows_sha256}}` with `rows_sha256` over the canonical sorted `subject|session_idx|frame_id` identities of that fold's QC-passed test frames; each shard reports its realized census and the merge requires an exact match for the shard JSON *and* the predictions CSV. Negative tests added for the drop-one-row cases (shard, and CSV-vs-JSON mismatch); §4 D10 updated. [checked: `src/dehyd/provenance.py:190-239`]
- C8 (applied, turn 2): conceded — the real/imag 2-channel convention fixes the input tensor, `in_channels`, the first conv layer's parameters and the fitted per-frequency norm state, so it is computation-affecting; §6's grouping of it with O-M9-5 understated it. Reclassified in Step 0b (O-M9-6 entry), in §6 (now "computation-affecting: O-M9-1,-2,-3,-4,-6,-7; not computation-affecting: O-M9-5 only", with the earlier draft's error stated rather than quietly dropped), and carried into the §1 step-0.5 propagation line for the source-of-truth amendment and SECOND_CHAPTER §8. [checked: plan §0b O-M9-6, §2.7 `spectrogram`/`SpectrogramNorm`, §6]
- C9 (applied, turn 4): confirmed the composite/Holm family-membership wording O-M9-3 resolves is in §Statistics (`:1263-1274`), not §D, so "§Statistics untouched" made step 0.5 unable to propagate what the plan says it propagates. §1 step 0.5 now enumerates all of A-M9-1 and O-M9-1..8 with an explicit destination each (O-M9-3 → §Statistics `:1267-1274` + §D cross-ref; O-M9-5 → §Statistics `:1275-1281`; O-M9-7 → §C; O-M9-6 → §D (ii) labelled computation-affecting; the ordinal aggregation rule → §C), so D0 can check for omissions; §6's propagation bullet updated to match. [checked: `plans/implementation_plan.md:1263-1281`]
- C10 (applied, turn 4): verified the leak — `extraction.py::_prelog_scale` returns one tuple **per session**, itself a `median over frames`, stored per session at `store.py:246`; `harness.tuned_epsilons` (`harness.py:186-209`) consumes those session tuples. Under a pooled frame KFold nearly every session straddles the split, so "training frames' sessions" would fit ε on scored rows — a second leak the owner's sanction (`:929-931`, subject overlap) does not cover, and fit-on-train-only governs fitted transforms regardless. §2.10 now recomputes the pre-log scale inside `frame_split.py` from the store's **raw** `raw_key`/`order_key` tensors restricted to the fold's training frame rows, via `_per_frame_prelog` (exactly `_prelog_scale`'s pre-median intermediate), with `scale_o = median over training frames pooled across sessions` — the aggregation unit stated and recorded in the output JSON, since the frozen subject-balanced median-of-subject-means has no meaning at frame granularity — then the frozen `k = 0.1` / 1e-6 fallback. Acceptance adds the held-out-frame mutation property, the training-frame power companion, a no-`prelog__*`-read assert, and a whole-session equivalence check against the stored tuple; new §5 trap 19 generalizes the lesson (any session-aggregate store key is a frame-split leakage vector). [checked: `src/dehyd/features/extraction.py:76-92`; `src/dehyd/features/store.py:235-249`; `src/dehyd/eval/harness.py:186-209`; `plans/implementation_plan.md:925-941`]
- C11 (applied, turn 4): the frozen text says "log-magnitude" at both `:821-825` and `:887-892`; the draft's `log(|STFT|² + 1e-30)` is log-power, and `1e-30` is not the float64 tiny (`≈2.2e-308`) but an arbitrary floor. Corrected to the literal `log(|STFT| + np.finfo(np.float64).tiny)` — i.e. back to the frozen text rather than promoted to an amendment — with the floor's rule recorded as a pure representability guard (a relative floor rejected as data-dependent) and finiteness asserted. Acceptance/§3/§4 D4 now pin the formula on hand-computed zero and nonzero coefficients so both the power form and the `1e-30` floor fail. [checked: `plans/implementation_plan.md:821-825,887-892`; `src/dehyd/config.py` `BaselineConfig` (no power/floor field exists)]
- C12 (applied, turn 4): §2.7 now names the reduction and retention axes — `SpectrogramNorm` keeps statistics **per `(channel, frequency)`** (shape `[C, F]`) reduced over **frames × time** of training frames, with the reason (real and imag parts have different per-frequency scales; sharing would let one channel standardize the other) — and specifies each of the four spectrogram variants' exact input, resolving the second half of the comment: the spectrogram branches STFT the **stored unstandardized** arrays and bypass the robust per-signal builders, which is what `BaselineConfig`'s "two distinct normalization rules" (`raw_matched_standardize` vs `spectrogram_standardize`) means. Tests: hand-computed statistics on an asymmetric 2-channel fixture, channel non-sharing, and a scaled-copy check proving no robust step precedes the STFT. [checked: `src/dehyd/config.py:346-350` (`raw_matched_standardize`/`spectrogram_standardize`); `plans/implementation_plan.md:821-825,887-892`]
- C13 (applied, turn 4): agreed — a prediction-level splice is undefined across families with 5 seeds vs 1. §2.8's composite is now defined at the per-subject metric level: each family's seed-averaged per-subject session-MAE (frozen additive collapse `:1193-1199`), spliced by outer fold (each subject is exactly one fold's test subject), no replication of deterministic values into pseudo-observations and no cross-seed prediction averaging (`:644-649`); serialized as `composite_{band}.csv` with `subject, selected_family, inner_score, per_subject_mae, n_seeds_averaged` and paired with radar by a join on subject. Test: a fixture alternating CNN/physics winners reproduces the hand-computed differences and fails under the ensembled-prediction alternative. [checked: `plans/implementation_plan.md:1193-1204,644-649,1263-1274`]
- C14 (applied, turn 4): correct — the frame hash cannot validate a session-level CSV, so a same-count session substitution would pass. `expected_test_rows_by_fold` now carries **both** `frame_rows_sha256` and `session_rows_sha256` (over distinct `subject|session_idx` identities) plus `seed_set`; seed is deliberately not part of the row identity but is validated as an exact `session_identities × seed_set` cross product (× `{1}` for deterministic families). Negative tests added for same-count substitution/duplication and for a single missing seed, alongside C7's missing-row cases. [checked: `src/dehyd/provenance.py:190-239`]
- C15 (applied, turn 4): real gap — nothing required an auditable per-family artifact set, and §2.10's modal reduction had no guaranteed source. §2.8 now specifies one schema for every family (CNN and deterministic alike): `predictions_*` (session-level OOF rows per `(subject, session_idx, seed)`, deterministic families writing seed 1 once with a `deterministic` flag), `metrics_*` (per-subject seed-averaged MAE vector + the family's own MAE/RMSE/pooled r with subject-cluster BCa CIs under the frozen metric-type-aware collapse `:1193-1204`, `n_eval`, `conditional_exploratory`), `selection_*` (per-fold `(lr, wd)`, epoch budget and the per-(inner fold × seed) counts it is the median of, per-config inner scores; deterministic families write their fitted coefficients so the file exists for every family), and `per_subject_*`. Merge acceptance requires all four, internal recomputation consistency, and exact fold coverage **before** the comparison stage may read them; D10 and §3 updated. [checked: `plans/implementation_plan.md:1193-1204`; plan §2.10's modal-config reader]
- C16 (applied, turn 4): answered and strengthened. On the question itself: `Deferred to owner` is reserved by the loop protocol for threads deadlocked after their debate rounds, and O-M9-8 is not a reviewer⇄author disagreement — both sides agree on the mathematics — so filing it there would misuse the section and the owner would read it as an unresolved argument. The substantive concern is right, though, so §6 now states the **two exact alternatives** ((8a) denominator-based, recommended; (8b) the literal frozen sentence) and a **fail-closed gate**: O-M9-8 is decided before step 0.5 closes and therefore before step 2 writes any source, and absent a decision the milestone **stops at step 0.5** rather than defaulting in either direction. D0 carries the gate, and the item is carried to the owner explicitly in the end-of-review decision brief. [checked: plan §6, §1 step 0.5, §4 D0; `review/PROMPT_claude_review.md` section-ownership rules]
- C17 (applied, turn 6): conceded — `implementation_plan.md:477-500` defines the two-stage subject-balanced aggregation *and* its reason (`:485-489`: a subject with more sessions must not get more weight), and that hierarchy is executable at frame granularity, so my pooled-frame median replaced a frozen estimator for no gain. §2.10 now keeps the frozen shape and narrows only the innermost population: session scale = median over **that session's training frames**, → mean over the subject's sessions with ≥ 1 training frame, → median over subjects with ≥ 1 such session, then frozen `k = 0.1` / 1e-6 fallback; the JSON tag becomes `frozen_hierarchy_training_frames_only`. Test updated to a deliberately unequal subject/session/frame fixture where the two estimators differ (the pooled version now fails), plus the empty-session and empty-subject drop-out cases. [checked: `plans/implementation_plan.md:477-500`; `src/dehyd/eval/harness.py:186-209`]
- C18 (applied, turn 6): correct, and it caught a self-contradiction — `preprocess_frame` ends in `to_channels(trimmed, channel, pre.standardize)` (`preprocess/pipeline.py:73`), so the stored 10 GHz `sig__matched_iq` is already robust-standardized, exactly as §2.7's own `matched_input_10` line said while my C12 edit claimed all four variants STFT unstandardized arrays. §2.7 now specifies the four variants individually: raw branches bypass the robust step (10 GHz `sig__raw_beat`, 77 GHz `sig__raw_slowtime`); the 10 GHz matched branch consumes the store's already-standardized tensor with no second standardization; the 77 GHz matched branch applies the robust per-channel step at load via `matched_input_77` (the store keeps it pre-standardization, while `:877-892` defines the matched input as robust-standardized), so both matched spectrograms really are the matched-preprocessing ablation. The "two distinct rules" reading is corrected in place: it separates the time-domain from the spectral step, it does not exclude spectrograms from the former. Scaled-copy tests rewritten per variant. [checked: `src/dehyd/preprocess/pipeline.py:60-73`; `plans/implementation_plan.md:877-892`; plan §2.9]
- C19 (applied, turn 6): right, and the repo already had the answer — `models/torch_fit.py::_normalize_stats` does `np.where(std == 0.0, 1.0, std)`. `SpectrogramNorm` now divides by `scale = np.where(std == 0.0, 1.0, std)` instead of `std + tiny` (which would amplify a differing test value in a constant training bin by ~1e308), with `mean`, the substituted `scale`, and the substituted-cell count all in the `FitRecord` so a fold where it fired is auditable. Test added for a constant training bin with a different finite validation value. [checked: `src/dehyd/models/torch_fit.py:86-91`]
- C20 (applied, turn 6): agreed — the loader contract was unspecified and every open option moves BatchNorm stats, step counts, early stopping and the budget median. §2.8 now pins: `WeightedRandomSampler(..., replacement=True, generator=g)` (without replacement it would merely permute the training set and discard the session balancing); `DataLoader(batch_size=16, sampler=sampler, shuffle=False, drop_last=True, num_workers=0, generator=g)`; **one epoch = `floor(len(train)/16)` optimizer steps** with uniform 16-row batches; `drop_last=True` justified by BatchNorm being undefined on a 1-row batch (it would raise on any fold with `len(train) % 16 == 1`) and by the dropped remainder being a random tail under replacement sampling; `MSELoss(reduction="mean")` with no second per-row weighting; a per-fit generator from the named derivation shared by sampler and loader, nothing from the global RNG. Recorded as implementation constants with in-place justification (the freeze fixes `batch_size` only). A sampled-index trace fixture pins the batch index sequence and step count for a fixed seed and that a changed held-out set leaves training batches identical. [checked: `src/dehyd/models/torch_fit.py:93-105,190-210`; `src/dehyd/config.py` `BaselineConfig.batch_size`]
- C21 (applied, turn 6): correct — `_hash_inputs(config, manifest, data_dir=None)` needs the manifest and defaults to `data_10ghz_dir` (`provenance.py:169-173`), so my four-argument signature could not produce "the same provenance content" and a 77 GHz exploratory run would have hashed 10 GHz files. Signature is now `write_exploratory_provenance(config, band, task, unit, manifest, out_dir, *, data_dir)` with `require_77ghz_dir(config)` for 77 GHz and the entrypoint passing the band's QC-passed frame manifest it already builds; step 9 extracts a **public** `provenance.build_provenance_payload(...)` from `record_run` (which then calls it and writes, output byte-identical and pinned) so no private helper is reached into; `folds=None` with the KFold's `k_folds`/`kfold_random_state` recorded in `extra` instead, since there are no LOSO folds here. Acceptance adds the byte-identity pin, the two-distinct-data-roots 77 GHz fixture, and refusal on a missing manifest. [checked: `src/dehyd/provenance.py:153-239`]
- C22 (applied, turn 6): confirmed `tests/test_selection.py:79` asserts `SIMPLICITY_RANK == {"ridge": 0, "knn": 1, "svr": 2, "rf": 3, "gbm": 4}`, so "unchanged and green" was impossible once six ordinal keys are added. §2.3's acceptance now says the existing five-family ranks and `select_candidate`'s behaviour stay pinned — body byte-unchanged, its own tests unedited — while that one exact-dict assertion is updated to the eleven keys and keeps the frozen `ridge < knn < svr < rf < gbm` ordering with `ord_a_*` mirroring it; only `tests/test_no_leakage.py` is frozen against edits. §3's row updated to match. [checked: `tests/test_selection.py:70-81`]
- C23 (applied, turn 8): correct and my own drift — the C18 fix landed in §2.7 but §3's T-M9-cnn row still carried the blanket "absent from the spectrogram branches" from turn 4, which would have forced the matched branches to bypass a step §2.7 requires (an implementation could satisfy one section only by failing the other). The row now states absence for the two **raw** branches only and pins both matched branches' robust-preprocessed inputs explicitly, mirroring §2.7's per-variant acceptance criteria. [checked: plan §2.7 variant list and acceptance, §3 T-M9-cnn row]
- C24 (applied, turn 8): conceded — `k_folds` + `random_state` pin the *recipe*, not the frame order the KFold indexes into, and `record_run`'s manifest summary carries only `n_frames`/`n_subjects`/`n_sessions` (`provenance.py:214-228`), so my "fully determined" claim was unsupported; nor was the LOSO artifact that *defines* the modal configuration identified at all. §2.10's `extra` now records `frame_order_sha256` (canonical `subject|session_idx|frame_id` list in the exact indexing order, so a permutation changes it), `fold_assignment_sha256` (the same list annotated with its realized fold, pinning the assignment rather than the recipe), `source_run` = {run_dir, analysis_commit, config_hash, artifact_rel_path, artifact_sha256} for the selection-table/merged-summary artifact read, and `resolved_config` = the modal `(feature_key, family, params)` or `(lr, wd, epoch_budget)` actually used. Acceptance adds: permuted frame order changes both hashes, a different referenced artifact changes `source_run.artifact_sha256`, a lineage-failing source artifact is refused before any fitting, and `resolved_config` matches what the artifact records. [checked: `src/dehyd/provenance.py:153-239`]
- O-M9-8 (owner, turn 9, post-closure): owner decided **option 8a** — QWK is undefined only on empty input or zero expected disagreement on the fixed 5×5 grid (matching `sklearn.metrics.cohen_kappa_score(..., weights="quadratic", labels=[0,1,2,3,4])`), not whenever either side has a single distinct class. Plan updated: §6's O-M9-8 bullet now reads OWNER-APPROVED with (8b) recorded as the rejected alternative and its strongest arguments preserved; the fail-closed gate is discharged in §6 and D0; §2.1's cross-reference and the §0 invariant's completion list updated; step 0.5 now names the exact `:798-800` annotation to write. Added on my own initiative, because (8a) admits folds (8b) would skip: §2.6 must report `n_single_class_truth_val_folds` and `n_qwk_nan` at both CV levels so SECOND_CHAPTER §8 states the choice's empirical size instead of assuming it negligible (T-M9-expc-report extended with a fixture containing exactly one such fold).
