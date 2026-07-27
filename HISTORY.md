# HISTORY — implementation log

Running record of every attempt, newest-first. Each entry: what was tried, whether it
succeeded/failed **and why**, and the concrete parameter values + reasoning. Failures
stay in the log. A new session reads only the most recent entries to orient.

## 2026-07-27 — M8 step 0.5 done: A-M8-1/A-M8-2 propagated into `implementation_plan.md`.

Per the M8 plan's own build sequence (step 0.5, forced by review comment C9 — this must land
*before* any other M8 source, so it is already part of the eventual clean-commit lineage and no
later doc-only commit can invalidate it), added both amendments to the source of truth itself,
not just the milestone plan: **A-M8-1** rewrites the Exp B Statistics bullet that named the
equal-session aggregate "the single pre-specified primary test" while a different bullet defined
the actual test form (Wilcoxon) on a different estimand — now resolved (session-weighted
aggregate CI = primary, subject-weighted complete-case Wilcoxon = companion), with the
2026-07-27/post-Exp-A chronology stated inline, not hidden. **A-M8-2** documents the
previously-unstated empty-session bootstrap-replicate rule (skip-and-count, matching every other
undefined-metric case in the same section). No other Exp B protocol text changed. Ready for
implementation to begin at plan step 1.

## 2026-07-27 — M8 plan (Exp B, clock-decoupling) written; Codex⇄Claude review closed — 25/25 comments applied, no debates, nothing deferred.

`plans/MILESTONE_8_PLAN.md` written from `implementation_plan.md`'s frozen §B design
(residualization, search-space reuse via A-M6-3, both estimands, Holm-4) plus two owner-approved
completions decided today, **after** M7's full-cohort results were already visible: **A-M8-1**
(primary = session-weighted aggregate difference CI, subject-weighted complete-case Wilcoxon as a
companion — resolving a genuine contradiction between `implementation_plan.md:1218-1219` and
`:1213-1217`) and **A-M8-2** (an empty-session bootstrap replicate is skipped-and-counted, not
averaged over the survivors). Both completions are disclosed with their real post-Exp-A
chronology throughout the plan, not folded into "frozen before Exp A" language — a distinction
the review itself had to force through three separate rounds (C3, C10, C13) before every summary
claim in the document was actually honest about it.

Full Codex⇄Claude adversarial review loop ran to `REVIEW_COMPLETE`: **25 comments (C1–C25), all
applied, zero withdrawn, zero debated, zero deferred to owner.** Substantive corrections, grouped:
- **Protocol integrity** (C2–C4, C16): added an end-to-end synthetic-store outer-mutation
  property for the real Exp B composition (not just fit-record mutation, which the first draft's
  leakage test alone would have missed); added a run-level viability check so a globally-missing
  session can't silently degrade the four-session primary to an unlabelled three-session mean;
  made the session-specific variant's four p-values descriptive-only rather than quietly inventing
  an undisclosed *third* post-Exp-A multiplicity completion on top of A-M8-1/A-M8-2.
- **Reproducibility/provenance** (C1, C7, C9, C14–C15, C17, C19–C22): added a clean-commit +
  store-rebuild gate before any run; moved the `implementation_plan.md` amendment-propagation to
  *before* the clean commit so no later doc-only commit invalidates the store's commit-match
  lineage; gave the session-specific variant's array/merge design `validate_store` enforcement
  before every fit and fail-closed shard-lineage validation against a run-group's own provenance —
  and, after two rounds of catching my own from-memory mistakes against `provenance.record_run`'s
  actual source (wrong return value, missing required argument, an invented field, wrong `extra`
  nesting), landed on a design verified line-by-line against the real code: correct `extra`
  nesting, a genuine per-session fold manifest (satisfying `implementation_plan.md:1273-1276`'s
  every-run requirement), and correct `data_dir` handling for 77 GHz.
- **Compute/regenerability** (C6, C8, C11–C12, C18, C23–C25): redesigned the variant's execution
  from a sequential loop into a genuine 4-task SLURM array with real cross-session concurrency;
  removed all in-process exception-catching in favour of fail-loud propagation (only pre-defined
  non-evaluability may degrade gracefully — the frozen C6/C21 doctrine, unchanged); gave the
  variant the same regenerable-from-intermediate-artifacts outputs (predictions/selection-table/
  dropped-folds CSVs, selection-frequency) every other experiment already produces; fixed two
  SLURM-mechanics bugs in the submission wrapper (`#SBATCH` resource sizing can't vary by a
  runtime `STAGE` in one shared file — parsed before the shell ever runs; `sbatch --parsable`'s
  job-ID output needs normalizing before use in a path or `--dependency`).

No comment was debated — every one was a genuine, applicable catch, several after I had already
"fixed" a prior round's mistake and needed a second (or third) correction in the same area. Plan
is ready for implementation pending owner approval; `v1_milestone_8` branches off
`v1_milestone_7` @ `bda8e45`.

## 2026-07-27 — M7 CLOSES: full-cohort Exp A complete on both bands — radar loses to the session-index-only baseline (negative result, config freeze spent).

Owner ran `MODE=full` on IBEX, both bands, at commit `f36c4fb2` (clean, fold-parallel harness).
Both completed cleanly — no errors, full cohort (73 sessions/16 subjects 10 GHz; 72 sessions/16
subjects 77 GHz), 5 seeds — writing `metrics_exp_a_{band}.json` into `results/runs/`. **The
config freeze is now spent; this is the first real outer-fold result inspected.**

**Headline result: radar-based Exp A regression loses to the trivial session-index-only baseline,
significantly, in both bands** — not the result the milestone hoped for, but a clean, valid,
honestly-obtained one (freeze respected right up to this exact moment):
- 10 GHz: subject-balanced MAE 0.469 [0.409, 0.568]; mean difference (radar − baseline)
  **+0.200** [0.145, 0.260]; Wilcoxon p=3.05e-5.
- 77 GHz: subject-balanced MAE 0.495 [0.404, 0.646]; mean difference **+0.216** [0.127, 0.296];
  Wilcoxon p=7.6e-4.
- Pooled predicted-vs-actual r centred near zero in both bands (10 GHz: −0.138 [−0.286, 0.075];
  77 GHz: −0.153 [−0.407, 0.174]) — no reliable linear relationship either.
- Selection tables show the search wasn't degenerate (knn/svr/gbm mixed across folds, mostly
  log-tuned features) — this reads as a genuine negative result from a working harness, not a
  pipeline bug.

Because Exp A's target (raw Δm%) is structurally confounded with time-of-day (the fasting
protocol), this result alone can't distinguish "no radar signal" from "signal present but
swamped by the clock" — which is exactly what Experiment B (clock-decoupling,
session-mean-residualized) was designed, and pre-registered *before* this result was seen, to
test. Decided (with the owner) to hold off on writing SECOND_CHAPTER §6 until Exp B's result is
in, so both experiments can be reported together with full context. M8 (Exp B) planned
immediately after — see the entry above.

## 2026-07-26 — M7 post-checkpoint: full-cohort runs timed out (4 h) → fold-level parallelism added (bit-identical, ~8-16× faster); still awaiting the full run.

After the checkpoint, the owner launched both `MODE=full` runs; **both hit TIMEOUT at the 4 h
limit**. They died mid-search, before writing any metrics/scatter — so **no outer-fold result
was inspected and the config freeze is still intact** (the re-run will be the first real number).

**Why so slow:** the fold-local tuned-ε reconstruction cost scales with sessions × folds, so the
full run (73/72 sessions × 16 folds) is ~8× the 6-subject smoke search — on the order of 8-16 h
serial. Not a bug (full suite green, D5 + T18 pass); a genuine compute cost.

**Fix — fold-level parallelism (f36c4fb):** the 16 outer folds are independent and each
deterministic, so `run_exp_a` was refactored to run each fold in its own worker process
(`_run_single_fold`, top-level/picklable, builds its own single-threaded store-backed provider),
serial when `n_workers=1` (test/CI default) or a `multiprocessing` **spawn** Pool when >1; results
reassembled in test-subject order. A new test asserts **parallel(n_workers=2) == serial
byte-for-byte**, with the D5 held-out-mutation property still green — so it is faster with
*identical* results. `run_regression` reads `SLURM_CPUS_PER_TASK`; `run_exp_a.sbatch` now defaults
to 16 cores / 64 G (12610af), so all 16 folds run in one wave (~a single fold's wall-time).
Extraction/store code untouched → store data unchanged.

**IBEX operational fixes this session (all committed):** the `REVISION`-file provenance fallback
for copied non-git trees (5677de9); the missing `configs/exp_a_regression_77ghz.yaml` entrypoint
(e8145cb); shard-mode `extract_features` QCing only its own file, not the whole cohort per array
task (3823611); `.gitignore` the regenerable feature store (4f003aa); the tuned-ε reconstruction
cache (b6a72c8). Stores build as IBEX job arrays (`extract10/77.sbatch`), runs as single CPU jobs
(`run_exp_a.sbatch`); `submit_ibex.sh` captures the clean commit (git checkout) or use `REVISION`
(copied tree). **Store-vs-run commit-match (C16) forces a store rebuild after any code change** —
the recurring friction this session; the store build is a fast parallel array, so it's cheap.

**State:** M7 code complete + all committed (`v1_milestone_7` @ 12610af); full suite green; freeze
intact. Blocking on the owner: launch the parallel `MODE=full` runs on IBEX (both bands), then
send the two `metrics_exp_a_{band}.json` so we can sanity-check radar-vs-baseline and write
SECOND_CHAPTER §6 to close the milestone.

## 2026-07-26 — M7 OWNER CHECKPOINT REACHED: both mechanism-only smokes GREEN on IBEX, freeze intact.

The 10 GHz and 77 GHz Exp A mechanism-only smokes ran to completion on IBEX compute nodes
(the workload that OOM-killed locally — peak RSS ~3.6 GB against 32 G reserved, so ample
headroom). Both wrote the expected `run_log_{band}.json` and **nothing else** — no
metrics/predictions/scatter — confirming the full staged search executed end to end on real
data with **no outer-fold performance value surfaced**:
- 10 GHz: `n_folds=6, n_sessions=24, mode=mechanism-only`.
- 77 GHz: `n_folds=6, n_sessions=29, mode=mechanism-only`.

The owner-gated real-data path was reached only after several IBEX-specific fixes surfaced and
were resolved: the `REVISION`-file provenance fallback for copied (non-git) trees (5677de9);
the missing `configs/exp_a_regression_77ghz.yaml` entrypoint (e8145cb); and shard-mode
`extract_features` QCing only its own file instead of the whole cohort per array task
(3823611). The store builds ran as IBEX job arrays; the runs as single 32 G CPU jobs.

**Checkpoint status:** M7 code complete + committed; full suite 682 passed / 16 skipped; T18
green; frozen `test_no_leakage.py` = single T18 hunk; both stores build + validate on real
data; both mechanism-only smokes complete. **No outer-fold result inspected — the config
freeze is intact.** The only remaining step is the owner-gated full-cohort run
(`run_exp_a.sbatch MODE=full`, both bands) which produces the first real numbers and spends
the freeze; it awaits explicit go-ahead.

## 2026-07-26 — M7 real-data run venue: the mechanism-only smoke is memory-bound locally → moved to IBEX (added `extract10.sbatch` + `run_exp_a.sbatch` + generic `submit_ibex.sh`).

Committed the M7 code (da7bdce, + gitignore 4f003aa, + tuned-ε caching b6a72c8) and built the
10 GHz store for the 6 smoke subjects **locally** — the BUILD works fine (~30 s/session). The
mechanism-only **smoke run**, however, repeatedly reached the staged search and was then killed
**silently — no Python traceback, no run_log** (finding: an OS-level kill, not a logic error;
the full test suite incl. the D5 real-store-backed mutation property and T18 all pass, so the
code is correct). Cause: the 72-combo staged search holds ~24 memory-mapped ~72 MB npz files and
materialises large logged tensors during the fold-local tuned-ε reconstruction — memory pressure
on a shared local Windows box (compounded earlier by several stuck python processes from repeated
attempts, now cleared). Added a tuned-ε/raw reconstruction cache (b6a72c8) that removes redundant
recompute (D5 still green) but the peak footprint remains too large for the local box.

**Resolution (consistent with the plan, which already routes the 77 GHz store to IBEX):** run the
heavy real-data work on IBEX. `scripts/ibex/extract10.sbatch` builds the 10 GHz store as an 80-task
job array (parallel, off the local disk); `scripts/ibex/run_exp_a.sbatch` runs the staged search as
a single CPU job (32 G / 8 cores, no OOM, no local contention) — `MODE=smoke` for the mechanism
check, `MODE=full` for the owner-gated full-cohort run; `scripts/ibex/submit_ibex.sh` captures the
clean commit and refuses a dirty tree. **This is a compute-venue decision, not a code fix.** The
milestone code is complete, committed, and fully green; the real-data smoke + full run are the
owner-gated steps, now targeted at IBEX.

## 2026-07-26 — MILESTONE 7 IMPLEMENTED to the owner checkpoint: the LOSO harness + Exp A. Full suite 682 passed / 16 skipped; T18 GREEN; frozen `test_no_leakage.py` changed only in the pre-registered T18 hunk. Real-data store build + smoke await the owner's clean commit (step 10.5).

M7's code is complete and every test is green. Built test-first on `v1_milestone_7` (from the
config-freeze base). Nothing committed yet (commit only when the owner asks) — which is exactly
why the real-data store build + smoke are the remaining owner-gated steps (a store must be built
from a clean revision; `assert_clean_tree` refuses a dirty tree, C7/C16).

**What landed (build order steps 0–10, each green before the next):**
- **Step 0–1** — pinned `matplotlib` (scipy stays 1.16.3 < 1.17; added to `TRACKED_PACKAGES`);
  `provenance._git_info` gained a `DEHYD_GIT_COMMIT/_BRANCH/_DIRTY` env fallback (per-field, only
  where live git returns None) so IBEX compute nodes self-attest their revision; stale
  "milestone 6" strings fixed. **T-M7-provenance** green.
- **Step 2 — `eval/metrics.py`**: `subject_balanced_mae` byte-compatible with the M1 definition
  (T17's 5.5 pin holds through the shim re-export); own **BCa** cluster bootstrap (B=10000,
  percentile fallback recorded, skip-and-count + >5% unreliable flag), metric-type-aware seed
  collapse, Wilcoxon. No `scipy.stats.bootstrap` (pin). 13 tests.
- **Step 3 — `models/regressors.py`**: the five families in a scaler pipeline; grid enumeration
  (ridge 8 / svr 12 / knn 7 / rf 6 / gbm 8, each ≤ budget_k); per-family auditable fitted state
  as ndarrays incl. SVR `support_vectors_` (C15) and an rf/gbm sha256 **ensemble digest binding
  `init_` + combining hyperparameters** (C20). 26 tests.
- **Step 4 — `eval/harness.py`**: ONE generic nested-LOSO engine — candidate-major execution but
  fold-major/candidate-minor assembly (the frozen flat order); selection ONLY via
  `select_candidate` (`inner_fold_variance = np.std(ddof=0)`, owner O1); `before_fit` guard hook
  with a **fail-closed `active`-completeness check** (C5); **pre-fit fold-viability** predicates
  with reason codes, unexpected exceptions propagate (C6/C21); per-seed outer outcomes kept
  separate; fit-audit incl. `tuned_epsilon`; `tuned_epsilons` train-only. 25 tests incl. the
  per-family held-out mutation property.
- **Step 5 — the rebind**: `reference_procedure.py` rewritten as a thin adapter over the harness
  (zero sklearn), returning a 9-field VIEW by reference. Frozen `test_no_leakage.py` **T1–T17,
  T19 now exercise the real engine**.
- **Step 6 — `models/torch_fit.py`** (TinyMLP, **true patience/min-delta early stopping**, C19)
  + **T18 activated** — the one sanctioned frozen-file edit (A-M7-1): outer-test mutation
  contract green; a separate inner-val test compares the train trajectory over the **common
  prefix** (C13/C18). Frozen diff = single T18 hunk; T1–T17/T19 byte-identical (D4).
- **Step 7** — `apply_order_log(epsilon_by_order=…)` (bit-identical to frozen at ε=1e-6) +
  `pool_stats_batch` (== looped) + `tuned_epsilons` bit-equivalence. 7 tests.
- **Step 8 — `features/store.py`**: per-session `.npz` + fingerprint (spec_hash, **frame_ids_sha256**
  binding QC frame membership (C4), qc_config_hash, git); fail-closed `validate_store` with
  **store/analysis commit match** (C16); `keep_raw` on both extractions; **store-vs-direct
  frozen-ε reconstruction equivalence** green; `extract_features.py` producer (both bands, refuses
  a dirty tree) + `extract77.sbatch` + `submit_extract77.sh` (captures the clean commit). 11 tests.
- **Step 9 — `models/baselines.py`**: session-index-only baseline, **global-train-mean fallback**
  (owner O2), FitRecord for the audit, config-level guard path (owner O3). 4 tests.
- **Step 10 — `eval/exp_a.py`** (StoreBackedFeatures, staged Stage-1→Stage-2 run, guard before
  every fit, baseline, summarize + Agg scatter) + rewritten `experiments/run_regression.py`
  (`--band`, `--subset 6subjects` XOR `--full-cohort`; **mechanism-only smoke surfaces no
  performance value**, C9/C14). **D5 end-to-end synthetic-store outer-mutation property GREEN** —
  the real store-backed staged path leaks nothing under held-out mutation. Entrypoint + plotting
  tests green.

**Owner decisions folded in:** O1 `ddof=0`, O2 baseline global-mean fallback, O3 config-level
baseline guard — all as the owner chose (A-M7-2 recorded).

**Remaining (owner-gated, the checkpoint):** step 10.5 = a clean owner-triggered commit of this
code, then step 11 = build the 10 GHz store locally (`extract_features.py --band 10ghz`, ~8–12 GB)
+ the 77 GHz store on IBEX (`submit_extract77.sh`), then step 12 = the mechanism-only smokes, then
STOP. The full-cohort run (step 13, spends the freeze) is a separate `--full-cohort` invocation on
explicit go-ahead. **No outer-fold result has been inspected.**

## 2026-07-25 — MILESTONE 7 PLANNED + Codex-reviewed: `plans/MILESTONE_7_PLAN.md` written and taken through the file-mediated Codex⇄Claude review loop to `REVIEW_COMPLETE`. 21 comments over 3 rounds, all applied; 3 owner decisions (O1–O3) parked for the owner. No code yet.

M7 (the LOSO harness + Exp A) is **planned, not implemented**. This entry logs the planning +
review; implementation starts later on a new `v1_milestone_7` branch after O1–O3 are decided.

**The plan.** `plans/MILESTONE_7_PLAN.md`, in the M4–M6 template spine (§0 scope → Step 0/0b
owner decisions → §1 build-sequence table → §2 per-file specs → §3 tests → §4 DoD → §5 traps →
§6 flagged gaps/amendments → §7 open items). Scope = `eval/harness.py` (one generic nested-LOSO
engine, sklearn + torch fit paths, fit-audit, tuned-ε fold-local) + `eval/metrics.py` (MAE/RMSE/
r + subject-cluster BCa bootstrap B=10000 + Wilcoxon) + `models/{regressors,baselines,torch_fit}`
+ `features/store.py` + `experiments/extract_features.py` + extend `run_regression.py`; run Exp A
session-level LOSO on BOTH bands vs the session-index-only baseline; rebind `reference_procedure.py`
to the real harness and activate **T18**. Three owner decisions taken up front: T18 activation is
the one sanctioned edit to frozen `test_no_leakage.py` (A-M7-1, pre-registered at M1); the 77 GHz
feature store is built on IBEX; an explicit owner checkpoint precedes the first full-cohort run.

**The review setup.** Two reusable prompt files (`plans/review_prompt_codex.md`,
`plans/review_prompt_claude.md`) drive a turn-based loop mediated entirely through a `## Plan
review` block appended to the plan file: a `Status:` line is the turn token
(AWAITING_CODEX/AWAITING_CLAUDE/REVIEW_COMPLETE); Codex writes `C#` comments, Claude applies (edit
+ delete + log) or debates (verbatim thread + rebuttal, ≥3 rounds → Deferred to owner). Claude
waited on each turn via a file-watch monitor. No debates or escalations arose — every comment was
judged correct on the merits and applied.

**Round 1 (C1–C13, all applied).** The substantive catches: **C1** the frozen suite only
exercises the Ridge shim, so the real store-backed staged path was unproven → added an end-to-end
**synthetic-store outer-mutation** test; **C5** `protocol_freeze_guard` validates only *present*
`active` keys (`{"band":"10ghz"}` passes) → added a fail-closed completeness check; **C9** the
6-subject real smokes were themselves exposing cohort-member outer scores → made the real-data
smoke **mechanism-only**; **C2/C3/C12** three unapproved protocol completions (baseline
absent-index rule, the `ddof=0` tie-break estimator, the K=1 baseline guard path) → de-frozen and
elevated to owner-approval items (the M6 C6-16 process rule). Plus C4 (fingerprint must bind exact
QC frame membership), C6 (KNN `k>rows` fold-viability), C7 (clean commit before store builds),
C8 (DoD split into pre-checkpoint / post-approval), C10 (matplotlib unpinned), C11 (per-family
fitted-state capture), C13 (outer-test vs inner-val torch mutation contracts disentangled).

**Round 2 (C14–C18, all applied).** Second-order refinements: **C14** "smoke computes no
performance value" is infeasible (a nested search must compute scores to select) → reworded to
"no performance value *leaves the process*", one code path, branch only at the reporting boundary;
**C15** the fitted-state contract wasn't executable as typed (string hash in a `.tobytes()`-
compared ndarray field; SVR missing `support_vectors_`) → all params ndarray, digest as uint8,
separate JSON audit artifact; **C16** the dirty-tree guard was IBEX-only → both producers +
store/analysis commit-match; **C17** moved O1–O3 into the "Deferred to owner" section proper;
**C18** early stopping makes the inner-val trajectory variable-length.

**Round 3 (C19–C21, all applied).** **C19 (blocking) reverted my own C18 fix**: I had changed the
torch trainer to run-to-fixed-max-epochs to make the inner-val test clean, but that diverged from
the frozen early-stopping protocol (`patience`/`min_delta`) and would make T18 protect a
*different* algorithm than the DL baselines actually run — reverted to true early stopping with a
**common-prefix** inner-val comparison (stop-time/checkpoint may differ). **C20**: the GBM digest
must bind `init_` + combining hyperparameters, not just tree nodes. **C21**: fold-viability must
be explicit pre-fit predicates with reason codes, not a broad try/except that could swallow real
bugs — unexpected fit/predict exceptions now propagate loudly.

**Outcome.** `Status: REVIEW_COMPLETE`, `Codex: NO MORE COMMENTS`. Plan test estimate ≈84 new
tests (576 → ~660), T18 skip→pass. **Owner cleared O1–O3 (2026-07-25), all recommended options:**
O1 inner-fold-variance = **population std `np.std(ddof=0)`**; O2 baseline absent-index =
**global training-fold mean** (keeps every test session scored, so radar/baseline share the
session set the paired Wilcoxon needs); O3 K=1 baseline guard = **config-level
`protocol_freeze_guard(config, active=None)`** (no WST axes to validate; frozen M6 guard code
untouched). Folded into the plan body (§2.3/§2.4/§6) and the Deferred-to-owner threads.
**The plan is final.** Implementation is the next session's work, on a new `v1_milestone_7`
branch from `v1_milestone_6` (`357f734`); step 1 = the git-provenance fix + matplotlib pin, then
metrics → regressors → harness → the frozen-suite rebind, test-first. Commit only when asked.

## 2026-07-25 — MILESTONE 6 IMPLEMENTED: the config-freeze gate. 11 frozen config sections + 2 new src modules + 50 tests; full suite 576 passed / 17 skipped; `test_no_leakage.py` untouched.

M6 is code-complete. It is a pure config/validation milestone — **zero computation on cohort
data** — so the whole thing is deterministic schema + two small guards, and the no-leakage
test needed no changes (nothing enters a CV loop).

**Config layer (`src/dehyd/config.py`, additive only).** 11 new frozen dataclasses:
`SearchSpace10GHzConfig` / `SearchSpace77GHzConfig` (band-keyed — 77 GHz's reduction/channel/
gate are FIXED scalars, not candidate tuples, so a 10 GHz-only candidate is structurally
inexpressible there, C6-03), `ModelGridConfig`, `BaselineConfig` (both bands), `ExpBConfig`..
`ExpGConfig`, `StatsConfig`, `ProtocolFreezeConfig`. All wired into `Config` via
`default_factory` (existing configs load unchanged) and into `known_sections`.

**Design decision — one generic frozen-record builder, not 11 verbose validators.** Unlike
qc/preprocess/wst (which carry live inner-CV axes + pre-declared ablations, hence per-field
validation), every M6 field is frozen post-Step-0. So `_build_frozen_record` implements one
contract for all 11 sections: a run YAML may RESTATE a field at its default (complete record)
but any CHANGED value is a ConfigError. Needed a type-aware equality (`_frozen_matches`):
list→tuple normalization so a restated tuple compares equal, and bool-vs-int strictness so
`reuse_exp_a_search_space: 1` is rejected (YAML has true/false; `True == 1` must not slip
through). This is stricter than the qc/preprocess pattern *because the sections are genuinely
all-frozen*, not an inconsistency.

**`src/dehyd/eval/selection.py` (the ONLY executable selection code M6 adds, C6-27).**
`select_candidate(list[CandidateScore])` — a pure tie-break over already-computed scores (no
data, no fitting): lower `inner_val_mae` → lower `simplicity_rank` (frozen ridge=0<knn=1<svr=2
<rf=3<gbm=4, C6-30 — an ordinal ranking, not a parameter count) → lower `feature_dimension` →
lower `inner_fold_variance`. Non-finite MAE / non-finite-or-negative variance candidates are
filtered *before* the tie-break (C6-34, so NaN ordering can't decide a winner); all-non-finite
raises. `min()` gives deterministic first-in-order tie resolution. The fit/score behaviour is
M7's; this is the tie-break half only.

**`src/dehyd/features/protocol_freeze.py` — `protocol_freeze_guard(config, active=None)`.**
Composes `canonical_spec_guard_77` unchanged (no 77 GHz feature field is a search axis);
re-validates the 10 GHz preprocess/wst fields locally with the one range-gate exception
(`model_gate_m` checked for MEMBERSHIP in the frozen `range_gate_m` whitelist, so BOTH approved
gates pass — the strict artifact guard `canonical_spec_guard` that demands the single (1.0,2.0)
is untouched on the write path, C6-24); validates every M6 section equals its canonical
default; and, given an `active` per-fit record, validates each axis against the band whitelist
(C6-29 — those axes are call arguments, never stored config, so a config-only guard is blind to
them). Deliberately duplicates ~12 lines of `canonical_spec_guard`'s loop to keep
`extraction.py` untouched (flagged as a future-refactor option in the plan §6).

**Configs.** 10 new YAML files (`search_10ghz`, `search_77ghz`, `baselines`, `stats`,
`exp_{b,c,e,f,g_fusion}`, `protocol_freeze`), wst77.yaml-style: restate salient scalars,
document code-frozen tuples in comments. `exp_a_regression.yaml` now `include:`s
`search_10ghz.yaml` and its stale "milestone-5 config-freeze" comment is fixed (A-M5-2
renumber). **YAML-1.1 trap bitten again:** `log_10ghz: off` parsed as boolean `False` →
quoted to `"off"` in exp_e.yaml.

**`implementation_plan.md` A-M6-1/A-M6-2/A-M6-5 all read APPLIED** (owner-approved 2026-07-25);
the log-branch pre-check retraction, the 77 GHz baselines, and the Frank-Hall substitution are
reflected in both documents consistently.

**Tests: +50 (526→576).** test_config.py M6 group (frozen-record load/restate/reject-change/
reject-unknown/bool-safety; band literal-pins; cross-band inexpressibility; budget-fits;
config_to_dict round-trip; all 10 committed files load); test_selection.py (14, tie-break +
NaN/empty/all-non-finite); test_protocol_freeze.py (guard composition, both-gates-pass/
bad-gate-fails, call-time axis rejection with canonical config, 77 GHz fixed-axis equality,
entrypoint-runs-guard-before-I/O, artifact-guard-still-strict). A capability test asserts
sklearn `LogisticRegression.fit` accepts `sample_weight` — the executable premise behind
A-M6-5. **Full suite 576 passed / 17 skipped** (same 17 realdata/torch-mutation skips as M5).
`git diff f3fbade -- tests/test_no_leakage.py` clean. Skipped `--realdata` deliberately: M6
touches no data-loading code and has no realdata tests.

**Freeze committed + tagged (DoD D7 met).** Owner-triggered on 2026-07-25: commit **`357f734`**
on `v1_milestone_6` (21 files, +2749/−31), annotated tag **`config-freeze-v1`** ("the complete
A–G protocol… before any outer-fold result was inspected"). Everything decided after this tag
that depends on outer-fold results is exploratory by definition. Not pushed; nothing on `main`.
**All seven DoD items (D0–D7) met. Milestone 6 is closed.**

## 2026-07-25 — M6 PLAN, Step 0 RESOLVED: all five owner decisions recorded; `statsmodels.OrderedModel` verified and rejected, Frank-Hall approved as an explicit substitution (A-M6-5).

Following the closed Codex review (38 comments, 5 rounds), the owner ruled on all five Step
0 items in one pass:

1. **A-M6-1 (log-axis pre-check) — retracted.** The `on+tuned-ε` log branch is now an
   unconditional inner-CV candidate for both bands, never gated by a cohort-wide predictive
   check. `implementation_plan.md` A-M6-1 status changed from PROPOSED to **APPLIED**.
2. **A-M6-2 (77 GHz Exp D baselines) — approved as corrected**, including the DC-bin-vs-
   any-motion physics baseline: the owner confirmed it's worth reporting despite being
   unable to distinguish breathing from any other motion (a coarser contrast than the
   10 GHz physics baseline, which rests on a real M3 measurement). Status: **APPLIED**.
3. **Budget-parity interpretation — approved as proposed**: representation-level choices
   (feature axes / raw-vs-matched input) exempt from the search budget K; model/training
   hyperparameter grids capped at K=12 uniformly across every family, WST or baseline.
4. **Every §3 provisional constant — accepted as-is**, no changes requested.
5. **Exp C's proportional-odds implementation — resolved by actual verification, not
   guesswork.** Per the owner's instruction ("go with A, if wrong then B"), candidate A
   (`statsmodels.miscmodels.ordinal_model.OrderedModel`) was checked directly rather than
   assumed: `uv run --with statsmodels` (an ephemeral install, no change to the project's
   pinned `pyproject.toml`/`uv.lock`) confirmed its `__init__`/`.fit()` signatures carry
   **no `sample_weight`/`freq_weights` parameter anywhere**, and it inherits from
   `GenericLikelihoodModel` — a generic MLE optimizer with no observation-weighting
   mechanism at all. Candidate A cannot implement the inverse-frequency class weighting
   `implementation_plan.md` §C requires, full stop. **Moved to candidate B**: the Frank-Hall
   ordinal decomposition (4 independent binary `sklearn.LogisticRegression` classifiers,
   guaranteed `sample_weight` support), recorded as **`implementation_plan.md` A-M6-5,
   APPLIED** — an explicit, documented substitution of a genuinely different statistical
   model for the approved one, not a silent swap (the exact process the round-3 self-
   correction, C6-28→C6-31, existed to prevent from happening quietly).

`plans/MILESTONE_6_PLAN.md`'s Step 0 section is now a resolved record, not an open gate;
every "[provisional]"/"pending"/"PROPOSED" marker in the document has been updated to
reflect the settled state. `implementation_plan.md`'s A-M6-1/A-M6-2/A-M6-5 tags all read
APPLIED. M6 config implementation can now proceed (§1 step 2 onward).

## 2026-07-25 — M6 PLAN, Codex review CLOSED: "NO MORE COMMENTS" after 5 rounds, 38 comments total (C6-01..C6-38).

Codex's closing note: the plan is now "methodologically coherent, executable once its
explicit Step 0 gate clears, and faithful to the authoritative design wherever it does not
openly request an amendment," and recommends removing the leaking cohort-wide pre-check
(Step 0 item 1) when the owner reviews it. `plans/MILESTONE_6_PLAN.md` is left clean of
review-comment markers; every one of the 38 comments across 5 rounds ended either
applied-and-fixed or escalated to the explicit **Step 0 owner-approval gate** at the top of
the plan — none were disputed.

**What Step 0 now asks the owner to decide, in one place:**
1. Retract the order-2-usefulness pre-check (A-M6-1)? Codex's own recommendation: yes —
   remove it, since it's genuine cohort-level leakage as specified.
2. Approve the corrected 77 GHz Exp D baseline design (A-M6-2), including whether the
   now-honest DC-vs-any-motion physics contrast (not the physiologically-invalid 2 Hz
   version) is worth reporting at all.
3. Approve the budget-parity interpretation (representation-level choices exempt from K,
   model-hyperparameter grids capped at K uniformly) or direct a different reconciliation.
4. Confirm, replace, or accept every §3 provisional constant (model grids, `k=0.1`, DL
   training hyperparameters).
5. Choose Exp C's proportional-odds implementation — a genuine cumulative-link model
   (`statsmodels.OrderedModel`, weighting support to be verified before `ExpCConfig` is
   written) or an explicitly-amended Frank-Hall substitution — with the weighting-support
   check itself required as part of clearing this item, not deferred to M7.

**The single most valuable lesson from this whole review, worth remembering past this
milestone:** the review caught not just bugs (dimensional errors, unrealizable physics
constants, missing test scope) but two instances of the *same* process failure — treating a
technically-sound fix to previously-approved design as already settled, once at the A-M4-7
pre-check (round 2, C6-16) and once again mid-fix at the Exp C library substitution (round 3,
C6-28→C6-31). Both times the fix itself was reasonable; the mistake was applying it silently
instead of proposing it. That is now structurally prevented by the Step 0 gate rather than
relying on remembering to ask each time.

## 2026-07-25 — M6 PLAN, Codex review round 4: 4 comments, mostly consistency lapses from round 3's own rapid-fire fixes.

**C6-35 (blocking)** — Step 0 item 2 still described the just-retracted 2 Hz Doppler cutoff
and asked the owner to approve it "as drafted," even though the rest of the document (§2.1,
`implementation_plan.md`'s A-M6-2) had already been corrected to the DC-bin-vs-motion design
in the same round. Left uncorrected, the owner could have approved the wrong physics
baseline. Fixed: Step 0 item 2 now describes the current, axis-consistent proposal.

**C6-36 (blocking)** — Step 0 item 5 said Exp C's proportional-odds implementation needs
verifying, but §7's carry-forward still deferred that verification to M7 — too late, since
an unverified required capability (does `statsmodels.OrderedModel` actually accept the
mandatory inverse-frequency weights?) shouldn't be allowed to cross the freeze gate at all.
Fixed: verification is now an explicit, required action *within* Step 0 itself — reading the
pinned `statsmodels` source/docs or running a two-line smoke fit with weights, with the
answer recorded — and `ExpCConfig` is not written until that's actually done.

**C6-37 (should-fix)** — the build-sequence table still said "owner approval on the four
items above" after Step 0 grew to five items in round 3. Fixed.

**C6-38 (should-fix)** — `StatsConfig` had quietly invented a `holm_family_expf_exploratory
= 2` correction for Exp F's two exploratory covariate contrasts (2v1, 4v3). Checked against
`implementation_plan.md` §F directly: it labels those two contrasts "exploratory" but never
says they form a Holm-corrected family together — only the two PRIMARY contrasts (3v1, 4v2)
get an explicit "Holm over 2." Inventing a second correction scheme for the exploratory pair
would have been exactly the kind of undisclosed protocol addition this whole freeze exists to
prevent. Fixed: removed the invented correction; the exploratory contrasts are now recorded
as reported individually and uncorrected, matching the approved text exactly.

**No comments were disputed.** The pattern across rounds 3 and 4 is worth naming explicitly:
fixing one comment can introduce or leave behind an inconsistency elsewhere in the same
document (a stale cross-reference, an outdated Step-0 description, a table that didn't get
its count updated) — worth a deliberate consistency pass after any batch of edits, not just
trusting that each fix was locally correct.

## 2026-07-25 — M6 PLAN, Codex review round 3: 9 comments (arriving in two waves mid-fix), including a self-correction caught within the round itself.

**The most interesting one: C6-28 → C6-31, a fix that was itself wrong.** Round 2 flagged
that the proposed `mord.LogisticAT` ordinal library had unverified `sample_weight` support
with a silent unweighted-fallback escape hatch — not acceptable, since
`implementation_plan.md` §C requires inverse-frequency class weighting, full stop. The first
attempt to fix this substituted a Frank-Hall binary decomposition over
`sklearn.LogisticRegression` (guaranteed weighting support, no new dependency) — but a
follow-up comment in the same round (C6-31) correctly caught that Frank-Hall is a
**genuinely different statistical model** from the approved proportional-odds/cumulative-link
family (separate coefficients per threshold vs. one shared slope with ordered cutpoints).
Silently swapping the approved family to solve a library problem is exactly the kind of
unapproved design change Step 0 exists to catch. **Un-applied and re-escalated**: Step 0
gained a fifth item presenting both candidates — a genuine cumulative-link implementation
(e.g. `statsmodels.OrderedModel`, weighting support still unverified) or the Frank-Hall
substitution (needs its own `implementation_plan.md` amendment, not a config choice) — for
the owner to decide between, rather than picking one and calling it settled.

**C6-26 (blocking) — the 77 GHz baseline proposal (A-M6-2) was dimensionally broken.** Its
"raw" CNN input averaged away the chirp/slow-time axis, then its own physics baseline tried
to run a Doppler FFT over that same (now axis-less) signal; its "matched" input claimed a
pre-WST Rx fusion that Exp G's own no-coherent-complex-Rx-averaging rule explicitly forbids
(fusion only exists post-WST). Fixed in both documents: the raw input now averages over
fast-time and Rx instead (retaining chirp/slow-time), and the matched input uses one fixed
representative Rx (index 0) with no cross-Rx averaging at all.

**C6-33 (blocking) — the proposed 2 Hz Doppler cutoff for that same physics baseline was not
physically realizable.** The system's actual Doppler frequency resolution is
`PRF/256 ≈ 7.63 Hz` — coarser than the proposed 2 Hz cutoff itself, meaning `|f_D|<2Hz`
would have silently selected only the DC bin regardless of the stated number, and no
physiological rate (breathing ≈0.2–0.5 Hz, heart rate ≈1–1.5 Hz) is resolvable at all in the
≈131 ms chirp-burst aperture this system captures. This is a basic arithmetic check I should
have run before proposing the value. Redefined honestly in both documents as a DC-bin-vs-
any-motion split (bin 0 vs. bins 1–127) — a coarser, non-rate-specific feature, flagged for
the owner to judge whether it's still worth reporting.

**C6-27 (blocking) — the staged-selection section implied real fitting code existed to
test, when M6 (deliberately) builds none.** `T-C6-stage` claimed to run the Stage-1→Stage-2
algorithm and inspect fitted parameters, but no file/function for it lived anywhere in §2,
and `eval/harness.py` is explicitly M7's. Added exactly one small, genuinely non-modeling
helper — `src/dehyd/eval/selection.py`'s `select_candidate`, a pure tie-break comparison over
already-computed scores, no data or fitting involved — and explicitly deferred the real
fit/score behavioral claim to M7.

**C6-30 (blocking) — a direct consequence of C6-27's fix: the prose tie-break and the new
`select_candidate`'s `effective_params: int` field were two different, inconsistent
definitions of "simpler model," with no frozen mapping between them** (and "effective
parameters" isn't even comparable across ridge/knn/svr/rf/gbm). Replaced with two
deterministic components used identically by both: a frozen `simplicity_rank` family
ordering (ridge=0 < knn=1 < svr=2 < rf=3 < gbm=4) and `feature_dimension` (computed from
already-recorded WST geometry, not fit-time data).

**C6-29 (blocking) — the whitelist guard could only validate what `Config` statically
declares as allowed; reduction/channel/tiling/log-branch are call arguments, never stored in
`Config` at all**, so an out-of-whitelist call-time value would pass the guard silently.
Added a required `active=` parameter carrying the exact per-fit protocol record, checked
against the same whitelists regardless of how the caller produced it (defense-in-depth, not
"trust the entrypoint's loop").

**C6-34 (should-fix) — `select_candidate` had no defined behavior for non-finite scores**,
and Python's NaN comparison semantics could make the result silently order-dependent. Added
an explicit finite-value filter before the tie-break runs, raising if nothing survives it —
mirroring Exp C's existing "non-evaluable configs are skipped" convention.

**No comments were disputed.** All 9 were correct, including the one (C6-31) that caught my
own immediately-prior fix. The lesson from this round, on top of round 2's: even a
technically-motivated substitution (swap library A for library B to solve a real problem)
can silently change WHAT is being evaluated, not just HOW — that distinction needs the same
Step-0 gate as any other change to previously-approved design, not just changes that look
like design changes on their face.

## 2026-07-25 — M6 PLAN, Codex review round 2: 10 follow-up comments — the most important one a process correction, not a technical one.

**The headline finding (C6-16, agreed in full — a process error, not a technical one).**
Round 1 fixed a real leakage bug (the order-2 pre-check, see below) by rewriting
`implementation_plan.md`'s A-M4-7 gating rule and labeling it "retracted" — but
`implementation_plan.md` is this task's explicit authoritative base, and a milestone plan
was told not to override it. Reversing a previously-approved mechanism and simply asserting
the reversal, even with a sound technical argument attached, is exactly the kind of change
this project's own established practice (A-M5-1/A-M5-2) requires an **owner-approved
prerequisite** for, not a narrative applied after the fact. **Fixed:** both `A-M6-1` (the
pre-check retraction) and the new `A-M6-2` (77 GHz Exp D baselines, genuinely new design
content, not an ambiguity fix) are now marked **PROPOSED — PENDING OWNER APPROVAL** in
`implementation_plan.md` itself, and `MILESTONE_6_PLAN.md` gained an explicit **"Step 0 —
owner decisions required before implementation"** section, gating all config-writing behind
four items: the A-M6-1 approval, the A-M6-2 approval, a budget-parity interpretation
(below), and every remaining §3 provisional constant. The technical analysis behind A-M6-1
is left standing (Codex did not dispute its correctness) — only its premature "already
settled" framing was wrong.

**Other blocking follow-ups, all agreed and applied:**
- **C6-17** — the tuned-ε mechanism defined how to reduce ONE session's coefficients to a
  scalar (`_prelog_scale`) but never how multiple training sessions' scalars become one
  fold-level `scale_o`. Fixed in `implementation_plan.md`: subject-balanced two-stage
  aggregation (mean within each training subject's eligible sessions, then median across
  training subjects — matching the "equal weight per subject" convention already used for
  Exp B/G), a non-finite/non-positive fallback to the frozen `log_epsilon=1e-6`, and an
  explicit statement that the resulting ε is applied as a fixed constant to train/validation/
  outer-test frames alike (the same fit-on-train-only pattern as every other fitted quantity).
- **C6-18** — the staged-selection algorithm said Stage 1/2 score candidates on **inner-
  training** MAE, which scores a candidate on the data used to fit it — contrary to
  `implementation_plan.md`'s own "mean over inner-**val** subjects" selection rule. Fixed:
  both stages now fit on inner-training and score on inner-validation, with an added
  mutation-property acceptance test (mutate inner-validation labels → selection can change,
  fitted parameters must not).
- **C6-19** — WST-classical's Stage 1 evaluated up to 72 feature-axis configs uncapped, while
  CNN/spectrogram baselines had no comparable grid at all — violating `implementation_plan.md`
  §D's "same inner-CV configuration budget (≤K) each" rule. This is a genuine interpretive
  question (not a numeric fix), so it's proposed, not applied: representation-level choices
  (WST's feature axes; a baseline's raw-vs-matched input) are symmetric and exempt from K on
  both sides; model/training-hyperparameter grids are capped at K=12 uniformly, with a new
  small `baseline_learning_rate × baseline_weight_decay` grid added so baselines have
  something real to cap. Flagged as Step 0 item 3 — if rejected, Stage 1 needs a bounded
  redesign, not just a config change.
- **C6-20** — `StatsConfig` only had 6 fields; the actual `implementation_plan.md`
  §Statistics protocol is long and detailed (confidence level, resample unit, BCa fallback,
  metric-aware seed collapse, the >5%-skip unreliability flag, Exp B's two distinct
  estimands, the Holm families, the robustness-bootstrap safeguards). Expanded to ~20 fields,
  pure transcription of an already-approved protocol, no redesign.
- **C6-21** — round 1's 77 GHz baseline fix added only the physics-cutoff *field*; the raw/
  matched/spectrogram tensor definitions lived only in `implementation_plan.md` prose and
  were never mirrored into `BaselineConfig` — meaning the planned tests claimed to pin fields
  that didn't exist. Added the missing fields (`raw_reduction_77ghz`, `matched_input_77ghz`,
  channel counts for each 77 GHz baseline variant).
- **C6-23** — `ExpCConfig`'s cutpoint source was unstated (true labels? predictions? whose?).
  Fixed: cutpoints are quantiles of family (a)'s own regressor's **in-sample predictions on
  inner-training sessions** (never validation/outer-test — that would leak). Added the
  inverse-frequency class-weight formula as an executable rule, and flagged — not silently
  assumed — that `mord.LogisticAT`'s `sample_weight` support is **unverified** against the
  pinned version; if absent, family (b) needs an owner-approved substitute before M7.
- **C6-24** — the whitelist guard's round-1 design called `canonical_spec_guard(config)`
  unchanged, which **factually would reject** the approved `(0.9,3.0)` 10 GHz range-gate
  candidate (verified: `canonical_spec_guard` checks `preprocess.model_gate_m` against the
  single canonical `(1.0,2.0)` default). Fixed: the guard now re-implements a small local
  field comparison for the 10 GHz `preprocess`/`wst` sections with one exception (range-gate
  membership, not equality), while `canonical_spec_guard`/`canonical_spec_guard_77` stay
  exactly as-is on the curated-artifact write path — deliberately avoiding a touch to
  `extraction.py`'s frozen front-end code, flagged as a duplication tradeoff in case the
  owner would rather see a shared-helper refactor there instead.
- **C6-22 (should-fix)** — spectrogram inputs need **train-only per-frequency mean/std**,
  distinct from the raw/matched inputs' robust per-channel standardization; Adam betas,
  weight init, loss function, and checkpoint metric/direction were all implicit. All named
  explicitly now (flagged provisional where `implementation_plan.md` doesn't state them).
- **C6-25 (blocking)** — nothing previously stopped the plan from writing/testing every
  provisional constant and declaring the milestone done without explicit owner sign-off.
  Folded into the same Step 0 gate as C6-16/C6-19: D0 now requires Step 0 cleared, and §3
  must be empty or every row explicitly owner-accepted before D2 can be satisfied.

**No comments were disputed** — all 10 were correct. The structural lesson (C6-16) is the one
worth remembering going forward: a correction to previously-approved design, however
technically sound, gets proposed and gated for approval — it does not get applied and
back-filled with a rationale.

## 2026-07-25 — M6 PLAN, Codex review round 1: 15 comments (5 blocking-tier resolved as blocking), one genuine leakage conflict traced back to `implementation_plan.md` itself and fixed there.

**The headline finding (C6-01, agreed in full).** The plan's draft order-2-usefulness
pre-check ran a full-cohort LOSO comparison (order-{0,1} vs order-{0,1,2} features) to decide
whether the `on+tuned-ε` log branch enters the search space. Codex correctly identified that
this is leakage: a full LOSO run uses every subject as an outer-test subject at some point, so
using its aggregate result to set a *global* search-space bit — later applied to the SAME 16
subjects' Exp A/B/etc. evaluation — is indistinguishable from "a configuration chosen by
outer-test scores," directly conflicting with CLAUDE.md/ROADMAP invariants 2–3 and the main
plan's own "outer-test subject is touched only for final scoring." It also broke the
precedent the project already set at M5 (A-M5-6): decide leakage-sensitive protocol
questions on mechanism/physics grounds, never on cohort-wide predictive/survival evidence.
**Retracted.** `implementation_plan.md`'s WST-parameterization section amended (A-M6-1): the
log axis's third branch is now an **unconditional** inner-CV candidate for both bands,
selected fold-locally exactly like tiling/model family — leakage-safe by construction, and a
strictly smaller search-space change (one axis 2→3 values) than the retracted pre-check would
have been. Direct consequence: M6 now performs **zero predictive computation** — it is pure
config specification, which is a stronger, simpler milestone invariant than the original draft
had.

**Other blocking comments, all agreed and applied by rewriting the affected sections:**
- **C6-02** (ordering: freeze everything before running anything) — resolved structurally by
  the C6-01 fix; nothing in M6 computes anything anymore, so there's no "before/after" to get
  wrong.
- **C6-03** — the shared `SearchSpaceConfig` couldn't represent 10 GHz vs 77 GHz's genuinely
  different search axes (77 GHz has no reduction/channel/gate axes) and had no explicit tiling
  field. Split into `SearchSpace10GHzConfig`/`SearchSpace77GHzConfig`, band-keyed, tiling added.
- **C6-04** — the tuned-ε mechanism's `k` and "coefficient scale" were never actually specified
  in `implementation_plan.md` (a real gap in the authoritative document, not just the milestone
  plan). Fixed there (A-M6-1): `k = 0.1` [provisional], `scale_o` reuses the existing
  `_prelog_scale` function restricted to the fold's training sessions, zero-scale falls back to
  the frozen `log_epsilon = 1e-6`.
- **C6-05** — `budget_k` was a cap with no enumerated grids or staged-selection algorithm.
  Added concrete per-family grids (Ridge/SVR/RF/GBM/KNN, each ≤12 combos) and a literal
  two-stage algorithm (Stage 1: ridge-anchored feature-axis search; Stage 2: model
  family/grid search on the Stage-1 winner; tie-break = MAE → simplicity → inner-fold
  variance).
- **C6-06** — `config.py`'s `_reject_unknown` rejects unknown YAML keys and ignores comments,
  so the draft's "YAML comments/literals" for Exp B–G could never actually be loaded or
  validated. Added real dataclasses (`ExpBConfig`..`ExpGConfig`) for every experiment.
- **C6-07** — 77 GHz Exp D baselines were entirely unaddressed despite 77 GHz being promoted to
  a full parallel A–F arm. Added a complete 77 GHz baseline spec to `implementation_plan.md`
  (A-M6-2): raw/matched 1D-CNN (1-channel real / 2-channel complex, since 77 GHz raw ADC is
  real, not complex), spectrogram+2D-CNN, and a **Doppler-domain** physics baseline (quasi-static
  vs motion energy ratio, cut at 2 Hz [provisional] — physiologically motivated, not derived
  from an audited file, flagged for confirmation).
- **C6-08** — 10 GHz `BaselineConfig` was missing 2D-CNN architecture, session-balanced
  sampling formula, and training hyperparameters (batch size, epoch ceiling, early-stopping
  patience/min-delta). All added, batch/epoch/patience values marked [provisional].
- **C6-09** — the Exp E interpretability config was a placeholder ("one tiling + one model,
  named later"). Named concretely and non-performance-based: 10 GHz = reduction A / mag / T1 /
  log off / 1–2 m gate; 77 GHz = T1_77 / log off; both ridge (α=1.0, the same non-tuned anchor
  as Stage 1); deterministic 4-fold subject assignment via sorted-ID array split.
- **C6-10** — Exp C's cutpoint-fitting algorithm and proportional-odds implementation were
  unspecified. Added: cutpoints = training quantiles at {0.2,0.4,0.6,0.8} with a tiny
  degenerate-tie separation; proportional-odds = `mord.LogisticAT` — **flagged as a new
  third-party dependency needing owner sign-off** before it enters `pyproject.toml`/`uv.lock`.
- **C6-11** — Exp B never stated whether it reuses Exp A's search space or invents its own.
  Clarified in `implementation_plan.md` (A-M6-3): Exp B reuses Exp A's identical search space,
  scored on Exp B's own equal-session residual-MAE objective.
- **C6-12** — Exp F never stated how "the identical selected radar feature set" for models 3/4
  is obtained. Clarified in `implementation_plan.md` (A-M6-4): reuse that outer fold's
  Exp A-selected *feature* configuration (not model family), refit with ridge.
- **C6-13/C6-14** (should-fix; smoke-test subject count, pre-check IBEX provenance) — both moot
  after C6-01's retraction removed the pre-check script and its IBEX job entirely.
- **C6-15** — the whitelist validator only checked 3 constants, missing the rest of the freeze
  and never composing the existing `canonical_spec_guard`/`canonical_spec_guard_77`. Redesigned
  to call both existing guards and validate every new frozen section, plus an entrypoint-order
  test.

**No comments were disputed** — all 15 were correct on inspection; every fix landed in both
`MILESTONE_6_PLAN.md` and, where the gap traced back to the authoritative document,
`implementation_plan.md` itself (A-M6-1..4), so the two stay consistent. A short list of
provisional constants (`k`, the model grids, DL training hyperparameters, the `mord`
dependency, the 77 GHz physics-baseline Doppler cut) is now collected in the plan's §3,
flagged for explicit owner confirmation before the freeze closes — none were derived from
running anything against the cohort.

## 2026-07-25 — M6 PLANNED: `plans/MILESTONE_6_PLAN.md` drafted; two owner decisions settled ahead of the freeze.

**Decision 1 — 77 GHz in-band QC threshold (0.30) stays FROZEN, not moved inside inner CV.**
Despite sitting at percentile 9.6 of a unimodal in-band distribution (0.28 vs 0.32 would give
a materially different population — see the 2026-07-24 M5 cohort-QC entry), the owner chose
to keep it frozen rather than declare it data-adaptive. Reasoning: the confound check was
negative (median in-band ratio flat across S0–S4, 0.365/0.358/0.365/0.363/0.374 — QC is not
behaving in a hydration-dependent way), and moving it inside CV would make session
eligibility fold-dependent, which breaks the frozen-eligibility precondition the
`test_no_leakage.py` mutation property and the whole manifest/harness design currently rest
on — a large structural cost for a confound that isn't there. A labeled-**exploratory**
sensitivity re-run at 0.28/0.32 is pre-registered for **after** primary results, so the
threshold-sensitivity finding isn't simply dropped, only deferred to a place where it can't
leak.

**Decision 2 — the order-2-usefulness pre-check runs on BOTH bands, gated per band.** Both
10 GHz (M4) and 77 GHz (M5) show the same ~1.8× across-subject order-2 pre-log scale spread
and <1% fold-to-fold stability, but the two bands' physical content differs (fast-time beat
vs slow-time Doppler), so each band's `on+tuned-ε` log branch is confirmed or dropped on its
**own** evidence rather than borrowing one band's verdict for both. 10 GHz features already
exist as session-level diagnostics; 77 GHz needs one small CPU IBEX pass over the 72 eligible
cells (reusing `extraction_77.py` as-is — no new library code) since only per-cell
diagnostics, not full feature vectors, exist for 77 GHz so far.

**Plan drafted**: `plans/MILESTONE_6_PLAN.md`, §0–§7, mirroring the M2–M5 template. M6 is
mostly config + one whitelist validator + the one pre-check computation above — no harness,
no outer-fold evaluation, `tests/test_no_leakage.py` untouched. Two values the plan proposes
that implementation_plan.md leaves unstated are flagged for Codex/owner review rather than
silently fixed: the per-family search-space budget **K = 12** (a proposal, not yet
authoritative), and the pre-check's own ridge λ grid `{0.1, 1.0, 10.0}`. Awaiting Codex
review + owner approval before implementation starts on `v1_milestone_6`.

---

## 2026-07-24 — M5 cohort QC on IBEX: **90.4% frame survival, 72/80 sessions, ZERO flatline** — the bin-0 correction validated at scale.

Job 49383703, all 80 files, `flatline skip_leading 1` confirmed in the log header.
**Axis: ACCEPTED on all 80 files.** **Flatline: 0 across 10,000 frames × 4096 traces (~41M trace
evaluations)** — the M2 rule would have destroyed this cohort; the mechanism-corrected rule fires
on nothing. NaN/Inf: 0. **Every one of the 964 rejections is the in-band screen.**
Survival 9036/10000 (90.4%); **72/80 sessions eligible; all 16 subjects evaluable**; analysis
population 8966 frames. (10 GHz for comparison: 73/80 sessions, 91.6% frames — the two bands land
almost on top of each other. **Exp G matched population = 65 sessions**, spanning all 16 subjects.)

**Dropout structure (label-blind).** 8 dropped cells across only 5 subjects (2, 7, 8, 9, 15); the
other 11 subjects are 125/125 everywhere. The axis diagnostics corroborate the cause independently:
dropped sessions have lower `G_fast` (median 0.173 vs 0.220) — genuinely less energy in the 2–4 m
gate, consistent with subject positioning rather than broken acquisition.
**Confound check — negative, which is the reassuring answer:** median in-band ratio by session is
flat (0.365 / 0.358 / 0.365 / 0.363 / 0.374 for S0..S4), so signal quality carries no time-of-day
trend and QC is not behaving in a hydration-dependent way. Eligibility does skew slightly against
baseline (S0/S1 lost 6 of 32 session points; S2/S4 lost none), driven by a few bad cells, not a trend.

**Carried to M6 — the in-band threshold is threshold-sensitive.** 0.30 sits at **percentile 9.6**
of the cohort's in-band distribution (p1 0.254, p10 0.304, median 0.364, max 0.520): it slices the
lower tail of ONE unimodal distribution, with no bimodal gap between "dead" and "healthy" sessions —
the 8 dropped cells (medians 0.250–0.298) are simply its weakest end. So 0.28 or 0.32 would give a
materially different population. **Not touched** — re-tuning after seeing survival is exactly the
cohort-level leakage the plan forbids. Recorded for the M6 freeze, where the documented remedy for a
genuinely data-adaptive threshold is to move it inside the inner CV loop.

## 2026-07-25 — M5 cohort WST re-run on IBEX: **curated `wst_diagnostics_77ghz.csv` produced and validated — D3 satisfied for band 2.**

Array re-run after the eligible-frame fix; 80/80 tasks, no cache race this time. Validation
before merge: 72/72 eligible cells populated, 8 empty shards == the 8 ineligible cells, all
finite, 6 rows/cell, geometry uniform (n_paths 424/453/182, n_time 8/4/4), 80/80 fingerprints
consistent, and — the point of the fix — **every cell used exactly its QC-passing frame count**
(subject 7 12pm now 102, subject 9 12pm now 114, all others 125), summing to **8966 analysis
frames = the QC analysis population exactly**.

**A THIRD bug, in the merge itself, caught here.** The first merge attempt aborted with "merged
72 cells but expected 5 eligible". Cause: the staleness check I added in the previous fix did
`expected = _fingerprint(...)`, **shadowing** the eligible-cells set (72) with a 5-key fingerprint
dict, so the final `n_cells != len(expected)` compared 72 against `len(dict)`. Renamed to
`expected_fp`. Root cause of why BOTH this and the earlier "self-consistent stale set" bug slipped
through: **no test ran `run_merge` to completion** — every prior check stopped at an early abort.
Added `tests/test_run_wst77_merge.py` (4 tests) driving the real function: a clean merge writes the
curated CSV with the right cell count (would have caught the shadowing), a stale fingerprint aborts,
inter-shard disagreement aborts, a missing shard aborts.

**Curated artifact:** `results/wst/wst_diagnostics_77ghz.csv` — 72 cells × 6 = 432 rows, 16
subjects, all finite. Pre-log order-2 scale (mean fusion) medians 1.80e-4 / 1.97e-4 / 6.22e-4 by
tiling, across-subject max/min ≈1.85 — the same ~1.8× subject spread the 10 GHz cohort showed at
M4, so the tuned-ε stability question carries over to band 2 unchanged (an M6 item, not decided here).

**Provenance gap persists:** `git.commit` is still `None` in the fingerprints — the
`safe.directory` fix did not take on the compute nodes. The set is internally consistent and
matches the current code by fingerprint, but does not self-attest the revision; the producing
commit is recorded here manually as the pushed head at run time. Worth resolving before M7 if the
harness re-extracts.

## 2026-07-24 — M5 cohort WST on IBEX: **BUG FOUND AND FIXED — the curated run extracted all frames of eligible sessions instead of the eligible FRAMES.** Shards archived; re-run required.

Array 49399759 completed 79/80; task 22 (subject 5, 12pm) died on a **uv shared-cache race**
(`failed to rename .../interpreter-v4/...: Text file busy`) — 80 tasks hitting `~/.cache/uv` at once,
not a data or code fault. Rerun as `--array=22` (job 49399874) succeeded. Validation of the completed
set was clean: 72/72 eligible cells populated, the 8 empty shards exactly matching the 8 ineligible
cells, all finite, 6 rows/cell, **geometry identical cohort-wide and identical to the local run**
(n_paths 424/453/182, n_time 8/4/4), 80/80 fingerprints consistent, ~168 s median per cell.

**Then the defect.** `n_eligible_frames` read 125 for every cell — but `run_wst77.py`'s curated mode
filtered only at the SESSION level and then extracted **all 125 frames of each eligible session**,
rather than that session's QC-**passing** frames. The analysis population is `eligible_frames`
(passing frames of eligible sessions) — documented as the only view modelling may consume — and the
10 GHz `run_wst.py` does this correctly (`population = eligible_frames(...)` → `cube[:,:,frame_indices]`).
Impact: **34 QC-failing frames (0.38%) entered the features, in exactly two sessions** — subject 7
12pm (used 125, should be 102) and subject 9 12pm (used 125, should be 114). The other 70 eligible
sessions were 125/125 passing, so unaffected numerically. Small, but a specification violation that
contaminates those two session vectors (a frame-mean/median over failing frames) and breaks
cross-band consistency for Exp G.

**Fixes applied:** `run_curated` now reads `qc_frames_77ghz.csv`, subsets each cube to that session's
passing `frame_idx`, and records the column as `n_eligible_frames` (the 10 GHz name); it fails closed
if the frames CSV is absent. **A second, worse gap surfaced while testing the fix:** `--merge-shards`
compared shard fingerprints only against EACH OTHER, so a wholly-stale set is self-consistent and
merged silently — it accepted the pre-fix shards without complaint. The merge now also compares them
against the fingerprint THIS config/code produces over the semantic fields
(`wst77_backend`, `axis_spec_hash`, `frame_selection`), excluding `git` (unreadable on the compute
nodes) and the per-file `raw_sha256`. `frame_selection=qc_pass_frames_of_eligible_sessions` was added
to the fingerprint precisely so pre-fix shards can never merge with post-fix ones — verified: the
merge now rejects them by name. Regression test added to `test_wst77.py`.

**Artifacts retired** to `archive/results/m5_wst77_prefilter_shards/` (80 shards + sidecars + the
invalid curated CSV) with a README stating why they are invalid and what they still evidence.
**The array must be re-run**; no valid 77 GHz feature artifact exists yet.

**Provenance gap noted:** every shard fingerprint and the QC provenance dir carry
`git: {commit: None, branch: None, dirty: None}` (dir named `..._nogit`) — git metadata is
unreadable from the compute nodes, most likely the `safe.directory` ownership check on
`/ibex/user/...`. The fingerprints still prove all tasks ran identical code, but they cannot attest
WHICH revision. Fix before the re-run:
`git config --global --add safe.directory /ibex/user/sosagojm/dehy_radar`.

## 2026-07-23 — **MILESTONE 5 — code complete; definition of done met except D3 (the cohort runs), which is staged for IBEX.**

**D0 — ✅** prerequisite satisfied before implementation (A-M5-1/A-M5-2 in `implementation_plan.md`
**and** `ROADMAP.md`; committed as `189ad35`).
**D1 — ✅** `uv run pytest` → **521 passed, 17 skipped** on a checkout with no private data
(was 396/12 at M4 close; **+125** tests this milestone).
**D2 — ✅** `uv run pytest --realdata` → **537 passed, 1 skipped** (only T18, the torch mutation
leg, still deferred — unchanged from M4).
**D3 — ⏳ STAGED FOR IBEX, not produced locally.** The code and scaffolding are complete and were
verified end-to-end on real subsets (see the steps 7+10 entry), but the *cohort* artifacts require
all 80 files: the QC pass is histogram-bound at ~2–3 s/frame × ~10⁴ frames ≈ **9 h locally**, which
is exactly the job `scripts/ibex/qc77.sbatch` exists for. Order: `sbatch qc77.sbatch` →
inspect `qc_survival_77ghz.csv` → `sbatch wst77.sbatch` (array) → rsync back →
`run_wst77.py --merge-shards`. **No cohort feature artifact may predate the frozen QC rule** — that
ordering is enforced by `run_wst77`'s curated mode failing closed without the survival CSV.
**D4 — ✅** `tests/test_no_leakage.py` byte-for-byte unmodified since M1 (`git diff --exit-code
f3fbade -- …` clean, working-tree-aware) and green. **M5 touched no 10 GHz file** (empty diff vs the
M4 commit for every frozen 10 GHz module/config); the only shared-file edits are the sanctioned
additive `config.py` (+231) and `provenance.py` (+14).
**D5 — ✅** `run_wst77.py --smoke --subject 1 --session 8am` → **D=10176 finite features, 7.8 s**;
the same entrypoint runs the curated array mode on IBEX differing only by `--config configs/ibex.yaml`
plus shard args.
**D6 — ✅** the flatline rule was decided on **M2-audit mechanism grounds, independent of cohort
survival**, recorded (HISTORY + A-M5-6), and applied uniformly; the `QC77Config` /
`canonical_spec_guard_77` / `screens_77` / YAML / tests were all revised and T-C77/T-Q77 rerun green
**before** any cohort QC could execute.
**D7 — ✅** A-M5-3..8 applied to `plans/implementation_plan.md` (Exp G chain, Compute/IBEX,
the frozen 77 GHz QC block, the repo tree, the experiments list); **SECOND_CHAPTER.md §4 "The
77 GHz front-end"** written (and the later placeholder sections renumbered §5–§8 to match the
A-M5-2 milestone renumber).

**Milestone-5 scoreboard.** 7 new source modules (`data/{loader_77ghz,manifest_77}`,
`qc/{screens_77,axis_check_77}`, `preprocess/pipeline_77`, `features/extraction_77`), 3 new CLIs,
5 new config files + the IBEX overlay, 4 sbatch/README scaffolding files, 5 new test modules
(**+125 tests**). Two facts discovered empirically, neither of which was fixed by moving a
threshold: the **range-bin-0 frame counter** (which explains M2's flatline false positive and is
harmless downstream because both Hann windows zero it and the gate excludes it), and the measured
WST geometry (T=39/78/117, J=6/7/7, pad 512) confirming the plan's predictions.

**The invariant held.** The whole 77 GHz front-end is a deterministic per-frame function of one
frame plus frozen constants — nothing fitted, so nothing enters the CV loop and
`test_no_leakage.py` is untouched. The tuned-ε log branch ships its *application* path only; the
fold-local ε remains M7's, computed train-only.

**Open for M6 (config freeze):** confirm the on+tuned-ε branch via the order-2-usefulness pre-check;
freeze both bands' A–F design and the 77 GHz protocol-constant whitelist. Nothing committed since
`189ad35` — awaiting the owner's word.

## 2026-07-23 — M5 steps 7 + 10: **three 77 GHz CLIs + the axis-cert guard + provenance `data_dir` + IBEX scaffolding.** Verified on real subsets; cohort runs staged for IBEX.

**Provenance (§2.8):** `record_run(..., data_dir=None)` and `_hash_inputs(..., data_dir)` — defaults
to the 10 GHz root so every existing call site is unchanged; 77 GHz entrypoints pass
`require_77ghz_dir(config)`, and an array task passes its **single-session manifest slice** so
only that one file is hashed (hashing 22 GB in each of 80 tasks would be pure waste).

**Axis-cert guard (`require_accepted_axis`, C5-08):** the hard per-file guard every extraction/
preprocess entrypoint and the smoke calls. It accepts a matching **ACCEPTED** record in
`qc_survival_77ghz.csv` keyed to the raw **sha256 + axis_spec_hash**, otherwise runs the semantic
check **inline**; any non-ACCEPTED verdict, or a record whose sha256/spec-hash disagrees, aborts
before a feature is written. A sha or spec-hash mismatch falls through to the inline check rather
than trusting a stale certificate.

**CLIs.** `run_qc77.py` — authoritative cohort QC: loads each file once, certifies the axis on the
**raw pre-MTI cube** (fails closed on non-ACCEPTED), runs the frozen screens, joins via the imported
`_join_qc`, finalizes eligibility, and writes `qc_survival_77ghz.csv` (+ axis certificate columns)
and `qc_frames_77ghz.csv`, re-reading and reconciling both. `run_preprocess77.py` — chain-energy
diagnostics as a **single** cohort job (no shard/merge race). `run_wst77.py` — three modes over the
same library code: curated cohort / **array shard** (deterministic shard + fingerprint sidecar),
`--merge-shards` (rejects a missing shard or any fingerprint disagreement, then writes the curated
CSV), and **`--smoke`** — non-curated, fixed frame indices, NaN/Inf + axis guards only (never the
flatline rule), so it is outcome-independent and can never reintroduce eligibility through the CLI.
`canonical_spec_guard_77` runs first in curated modes.

**Verified locally on real data (subsets; the artifacts were removed afterwards — they are not the
authoritative cohort run):**
- `run_qc77 --subject 1 --session 8am` → axis **ACCEPTED**, **125/125 frames pass (100 % survival)**,
  flatline 0, low-in-band 0 — the corrected rule's end-to-end confirmation on real data (the M2 rule
  would have failed most of these frames).
- `run_wst77 --smoke --subject 1 --session 8am` → **D=10176 finite features in 7.8 s** (3 frames,
  tiling Q=(8,4)) — the D5 deliverable and the sbatch time-limit calibration.
- `run_preprocess77 --subject 1 --session 8am` → MTI removes ~100 % (static clutter dominates for a
  seated subject), gate/raw energy ratio 3.4e-6.

**IBEX scaffolding (step 10, A-M5-5):** `configs/ibex.yaml` (**paths-only** overlay — device stays
cpu, so an axis certificate survives the rsync), `scripts/ibex/{qc77,preprocess77,wst77}.sbatch`
(single job / single job / **array over the 80 cells**, `HDF5_USE_FILE_LOCKING=FALSE` for GPFS),
a self-contained `scripts/ibex/README.md` (every command literal — the owner runs them), and
`.gitattributes` pinning `scripts/ibex/* eol=lf` so CRLF never breaks a shebang.

**Tests: +23** — T-A77 axis-guard (record accepted without loading; non-ACCEPTED record rejected;
sha/spec-hash mismatch falls to inline; inline accepts/rejects/inconclusive) and provenance
(`data_dir` selects the hashed root; a sliced manifest hashes one file). **Full suite: 521 passed,
17 skipped.** `test_no_leakage.py` byte-for-byte unmodified since M1; **M5 touched no 10 GHz file**
(empty diff vs the M4 commit) — only the sanctioned additive `config.py` / `provenance.py`.

## 2026-07-23 — M5 steps 8 + 9: **`pipeline_77.py` (T-P77)** and **`extraction_77.py` (T-W77 + T-R77)**. Green; geometry measured, matches the plan.

**`pipeline_77.py`** — chain steps 1–5 on one real frame `[n_fast,n_chirp,n_rx]`: MTI
(subtract this frame's per-fast-bin chirp mean) → fast-time zero-phase Butterworth over the
2–4 m beat band (reuses `filters.design_bandpass_sos`/`bandpass_filtfilt`, `beat_band_hz`) →
**symmetric Hann** (its zero endpoints also neutralise the bin-0 counter) → 256-pt range FFT
(where I/Q first exists) → crop to gate bins **27..53** (`range_gate_bins` from
`axis_check_77`, single home). Shape-generic; `preprocess_cube_77` a plain per-frame loop;
`chain_stages_77` for the run_preprocess77 energy diagnostics (Parseval convention).

**`extraction_77.py`** — chain steps 6–10, a linear composition of the fs/shape-agnostic
`wst.py`/`pooling.py`/`standardize.py`. Per frame: `slow_time_signal_batch` builds the **432**
complex slow-time series (16 rx × 27 gate, **rx-major/bin-minor** frozen fold), splits real/imag
and **vectorized-robust-standardizes** all 864 channels at once (bit-equivalent to stacked
`to_channels`, pinned) — **raises** on an all-zero (constant) channel; `scatter_frames([432,2,256])`
→ reshape `[16,27,2,P,t]` → mean over gate bins → per-Rx `[16,2,P,t]` → Rx fusion mean(primary)/
median(secondary) → `apply_order_log_77` on the **fused** tensor (branches off / on+frozen-ε /
**on+tuned-ε** with a caller-supplied `epsilon_by_order` — M5 applies but never computes ε) →
pool/flatten → `aggregate_session`. `extract_session_variants_77` scatters once per tiling,
derives every (log × fusion × family) + `(tiling,fusion)` pre-log scale from the shared raw
tensor. `canonical_spec_guard_77` checks all three `*77` sections equal their frozen defaults
(covers the step-6 flatline field). **Measured geometry** at n_in=256, fs=PRF=1953.125:
**T=39/78/117, J=6/7/7, padded=512** (as predicted), n_paths=424/453/182, n_time=8/4/4, invariance
error <0.16%; border warning fires and is asserted.

**Tests: +30 (T-W77 +19, T-P77 +11) + T-R77 realdata.** T-P77: MTI kills static / preserves
Doppler, gate bins 27..53, range peak lands at the expected bin (no shift), fast↔chirp swap
yields >50× less gate energy, zero-phase, Parseval, determinism. T-W77: geometry regression-pinned,
border warning present, batch `[432,2,256]` + fold order bit-equivalent to `to_channels`,
range-bin averaging vs independent scatter, fusion mean≠median, fuse-then-log≠log-then-fuse,
order-0 stays linear, zero-energy fires, per-channel standardization, pre-log scale keyed by
fusion + hand-checked, **tuned-ε applied before pooling & role-independent**, variants==single-variant
across all combos, pooled dims == layout, canonical guard (incl. stale-flatline rejection), and
the **numpy-vs-torch cross-backend agreement** over both log states (raw + logged) — the
precondition for `backend: torch`. **T-R77 realdata**: subject_1_8am 3-frame extract → finite,
zero-energy guard does not false-fire.

## 2026-07-23 — M5 step 6: **flatline-rule GATE resolved — mechanism found (embedded frame-counter, not ADC quantisation); owner chose exclude-bin-0; frozen + T-C77/T-Q77 rerun green.**

**Mechanism investigation (M2 single-file evidence, label-blind — never cohort survival).**
The M2 audit's 7/10 "flatline" turned out NOT to be ADC quantisation. On `subject_1_8am`:
`abs_max` grows as **exactly 256×(frame+1)** while `distinct` magnitude levels stay ~230 and
**no trace is dead** (`n_distinct≤3` = 0). Cause: **fast-time index 0 (range bin 0) of every
(Rx,chirp) trace holds an embedded FRAME COUNTER** — value ~256×frame, increments per chirp
(2305→2560 across frame 9's 256 chirps), resets periodically (frame 50→1), universal across
files (subject_1/5/12: 100% of traces, value ~2561 vs echo ~27 median / ~100 max). That single
~20–90× outlier stretches the per-trace `[min,max]` histogram range so the ~255 real samples
pile into the first bins and false-trip the ≥25% rule (flatline count tracks `abs_max`:
0,0,0,84,442,1609,2419,3191,3672,3981 across frames 0–9).
**Both Hann windows zero fast[0]** (`hann(256,sym=True)[0]=0`, periodic `[0]=0`) and the gate
crop keeps bins 27..53 (excludes bin 0), so the counter **never reaches the WST features or the
in-band/axis screens** — which is exactly why M2's in-band ratios (0.38) and axis verdict
(ACCEPTED) were healthy while flatline false-fired. It corrupts ONLY the raw-magnitude flatline
screen.

**Decision (owner gate).** Presented the finding + three mechanism-corrected options; **owner
chose "exclude range-bin-0"** — the most faithful exact replacement: drop fast[0] (known
non-echo counter), keep the frozen 128-bin / 0.25 rule on the 255 echo samples. A genuinely
dead/constant channel is still flagged (degenerate spread). This refines the plan's stated
"ADC-quantisation" premise (A-M5-6) to the actual counter mechanism; still leakage-safe
(specified from M2 mechanism + physics, not cohort survival).

**Step-6b revisions (all before the step-7 gate, so no stale rule can run):** new frozen field
`QC77Config.flatline_skip_leading_bins = 1` (validated ≥0); `screens_77._flatline_per_rx` gains
`skip_leading` (default 0 preserves the audit's M2 `qc_smoke_frame` semantics; `run_qc_frame_77`
passes 1) and screens `frame[skip_leading:]` with the threshold over `n_screened`;
`configs/preprocess77.yaml` documents it; T-C77 literal-pin + validation updated; T-Q77 gains
the counter-false-positive fix, dead-channel-still-caught, and a **realdata regression**
(frame 9: old rule flags >3000/4096, corrected rule flags 0 and the frame passes).
`canonical_spec_guard_77` (step 9) checks `QC77Config()` equality, so the field is auto-covered.
**Full suite: 482 passed, 16 skipped.**

## 2026-07-23 — M5 steps 5 + 4: **`screens_77.py` + `axis_check_77.py` (T-Q77)**, then **`manifest_77.py` (T-M77)**. Green.

**Reorder (noted):** built step 5 (QC screens + axis check) **before** step 4 (manifest),
because `manifest_77.apply_qc_77` imports `run_qc_cube_77` from `screens_77` — the module
dependency runs screens → manifest, so screens must exist first. The plan's step *numbering*
is unchanged; only the build order within this session swapped.

**`axis_check_77.py`** — promoted `range_gate_bins`, `_mean_power_spectrum`, `axis_metrics`,
`axis_verdict` (+ `AXIS_*` constants) from the audit; added `certify_axis(cube, pre77)` (the
per-file entrypoint, fast_axis=1/chirp_axis=2 on the loaded `[frame,fast,chirp,rx]` cube;
returns `(verdict, metrics)`) and **`axis_spec_hash(config)`** — a sha256 over exactly the
axis-relevant inputs (algo version, the four thresholds, expected shape+representation,
gate/bandwidth/fs) and **nothing environment-specific**, so a path-only `ibex.yaml` overlay
keeps a certificate valid while any axis-relevant change invalidates it (C5-16). Range gate
bins **27..53** confirmed.

**`screens_77.py`** — `FrameQC77` (nan_inf/flatline/low_in_band + in_band_ratio +
n_flatline_traces + per_rx_flatline; `passed` is a property so the rule can't be violated by
construction), `run_qc_frame_77`/`run_qc_cube_77` (shape-generic, plain per-frame loop,
structurally leak-proof), reusing `screens.in_band_mask` as-is (QC mask **bins 26..54**
confirmed). Flatline is per-(Rx,chirp) trace with the **any-trace rule** (one bad trace of
4096 fails the frame). Promoted the audit's `qc_smoke_frame` (dict) + `qc_in_band_mask_77`
here; both share the flatline core (`_flatline_per_rx`) with `run_qc_frame_77`. **Audit
re-imports** all promoted names; `test_audit_77ghz.py` still green (23).

**`manifest_77.py`** — mirrors `manifest.py` for band 2, **reusing by import** the subtle
pieces (`_join_qc` fail-closed one-to-one join, `eligible_frames`, `evaluable_subjects`, base
`COLUMN_DTYPES`/`SORT_KEYS`/`_describe`). New `QC77_COLUMN_DTYPES` (three screens, no RMS; +
`qc_rx_max_flatline`), `build_manifest_77`/`resolve_path_77`/`apply_qc_77`/
`session_qc_report_77`. Eligibility factored into `_finalize_qc_77(merged, min_frame_fraction)`
so the `ceil(0.5·n_frames)` arithmetic is unit-testable **without** loading 1 GB cubes; same
ground truth as 10 GHz (the shared 16-subject Δm). Real cohort manifest builds + validates
over all **80 files in 2.3 s** (metadata-only inspect).

**Tests: +39** — **T-Q77 (+16)** flatline (quantised / degenerate-spread / any-trace /
per-Rx counts), in-gate vs out-of-gate in-band, NaN/Inf short-circuit, mask bins 26..54 &
gate bins 27..53 pinned, per-frame independence, axis ACCEPTED/REJECTED/INCONCLUSIVE on real
tone cubes, `axis_spec_hash` env-independence. **T-M77 (+23)** C1–C6 bijection, actual frame
counts, determinism, imported `_join_qc` fail-closed, `_finalize_qc_77` ceil-eligibility +
`n_pass+n_fail_any==n_frames`, one small full-shape `apply_qc_77` integration (real screens),
realdata cohort build. **Full suite: 478 passed, 15 skipped.**

## 2026-07-23 — M5 step 3: **`loader_77ghz.py`** + promoted `reverse_axes`/`to_numeric`. Green; real file confirms the contract.

Created `src/dehyd/data/loader_77ghz.py` (h5py). `inspect_77ghz_file` reads **HDF5 metadata
only** (no chunk decompress) and enforces the M2 contract in a fixed order — **compound →
real-float → 8-byte → little-endian → shape** — so a tiny wrong-dtype fixture is rejected on
dtype, not shape. Rejections (each a stop-and-report, C5-18): compound `(real,imag)` (cites
the M2 real-float64 finding — MTI/pre-FFT steps assume real ADC data), non-float kind,
float32/other widths, **big-endian** (`dtype.byteorder` accept set `('<','=','|')`; empirically
h5py reports `'<'` for native LE float64 and `'>'` for big-endian — both dev and IBEX are LE),
and any on-disk shape ≠ `(16,256,256,n_frames)`. Frame count read from the file, never assumed
125. `load_77ghz_file` does a **whole-file read** (`dset[()]`) → `to_numeric` → `reverse_axes`
→ `[n_frames,256,256,16]` float64, forced by the chunk layout spanning all frames.

- **Promotion**: `reverse_axes`/`to_numeric` moved from `experiments/audit_77ghz.py` into the
  loader; the audit now **imports them back** (single copy). `tests/test_audit_77ghz.py` still
  green (23 passed) via `audit.reverse_axes`/`audit.to_numeric`.
- **Real-file check**: `inspect` → `(16,256,256,125)`; `load` → `(125,256,256,16)` float64,
  **1.05 GB, 3.0 s** — matches the plan's memory/geometry expectations exactly.

**Tests (T-L77): +16 in `tests/test_loader77.py`** (filename parse + rejects; `reverse_axes`
round-trip with distinct-per-coordinate small dims; `to_numeric` real/compound; **one
full-shape `(16,256,256,2)` fixture** — zeros + distinct markers, gzip-chunked so it stays a
few KB — inspected and loaded bit-for-bit with markers at the reversed coordinates; missing-var
/ compound / float32 / big-endian / wrong-shape / non-HDF5 all rejected; a realdata inspect on
`subject_1_8am.mat`). **Full suite: 439 passed, 14 skipped** (+16 pass, +1 realdata skip).

## 2026-07-23 — M5 step 2: **77 GHz config layer** (`qc77`/`preprocess77`/`wst77` + `data_77ghz_dir`). Green.

Added the three frozen band-2 dataclasses to `src/dehyd/config.py` — **`Preprocess77Config`**
(`butter_order=4`, single `gate_m=(2.0,4.0)` for both chain crop and QC mask, `fs_hz=500e3`,
`bandwidth_hz=2e9`, `chirp_time_s=512e-6` → PRF 1953.125 Hz, `standardize="robust"`),
**`QC77Config`** (`histogram_bins=128`, `flatline_max_bin_fraction=0.25`,
`min_in_band_energy_ratio=0.30`, `in_band_margin_hz=1953.125` = one FFT bin, `min_frame_fraction=0.5`),
**`WST77Config`** (code-frozen Doppler tilings `Q=(8,4)/(6,4)/(4,2)` at 20/40/60 ms,
`max_order=2`, `log_epsilon=1e-6`, `backend="numpy"`). Design choice (not re-litigated):
**three parallel top-level sections, not a nested `band77` block** — reuses the existing
`_known_section`/`_reject_unknown` machinery verbatim and gives each band its own canonical spec.

- `PathsConfig.data_77ghz_dir: Path | None = None` (optional so 10 GHz configs load unchanged;
  existence-checked only when present). New `require_77ghz_dir(config)` raises a pointed
  `ConfigError` when a 77 GHz entrypoint needs it and it was never set.
- `Config` gains `qc77`/`preprocess77`/`wst77` via `default_factory` → **every existing 10 GHz
  config loads byte-identically except the additive `*77` defaults now appear in provenance**
  (a completeness gain). `known_sections` extended by the three names.
- Validators `_build_{qc77,preprocess77,wst77}` reuse `_int/_float/_choice/_gate_field`;
  `wst77.tilings` override rejected like the 10 GHz WST; `_check_qc77_band` mirrors
  `_check_qc_band` (reuses `beat_band_hz`): the 2–4 m gate → beat band **52.1–104.2 kHz**,
  below Nyquist 250 kHz, non-vacuous. Confirmed the YAML signed-exponent trap is avoided
  (`500.0e+3` etc. load as floats, not strings).
- Four YAML files: `configs/{data77,preprocess77,wst77,exp_77ghz}.yaml` (values stated
  explicitly, matching code defaults; `exp_77ghz` includes the four + `run`/`split` parity
  with `exp_a_regression`).

**Tests (T-C77): +27 in `tests/test_config.py`** (defaults/overrides per section; `wst77.tilings`
override rejected; `max_order` 0/3/-1 rejected; bad values across all three sections;
`data_77ghz_dir` optional + existence-when-present; `require_77ghz_dir` raises/returns;
`_check_qc77_band` rejects an out-of-band gate; **one literal-pinning test for the frozen
defaults group** — the tripwire that flags a stale value if step 6 replaces the flatline rule;
the additive-only 10 GHz-dict check). New `real_data_77_paths` conftest fixture (separate from
`real_data_paths` so 10 GHz realdata tests don't couple to the 22 GB 77 GHz tree) + a realdata
config test. **Full suite: 423 passed, 13 skipped** (was 396/12 at M4 close; +27 pass, +1 realdata
skip). Realdata config tests pass with the cohort present. `test_no_leakage.py` untouched.

---

## 2026-07-23 — **MILESTONE 5 PLAN drafted, Codex-reviewed, and owner-approved (scope + amendments).**

**Scope decision (owner).** 77 GHz is promoted from fusion-only (Exp G) to a **full parallel
primary arm**: Experiments A–F run on 77 GHz too (16-subject cohort, band-appropriate
parameters); 10 GHz stays the sole headline; Exp G fusion retained. A new **milestone 5 =
77 GHz front-end** (loader → QC → preprocessing → slow-time I/Q WST → cohort diagnostics;
first IBEX milestone) is inserted before the config-freeze gate; downstream milestones
renumber +1 (config-freeze → 6, harness/Exp A → 7 … figures → 11). Applied to
`implementation_plan.md` (§Context, §Build order, all milestone cross-refs) and `ROADMAP.md`
(§0, §7) as amendments **A-M5-1/A-M5-2** — the owner-approved step-0 prerequisite (Codex
C5-01/C5-02: authorities must be amended before implementation, not by assertion).

**`plans/MILESTONE_5_PLAN.md` written** in the M4 template style (config → loader → manifest
→ QC/axis-check → preprocessing → WST → 3 CLIs → IBEX; ≈85–90 new tests). **Codex review: 24
comments over 5 rounds (C5-01…C5-24), all accepted and applied, none disputed/escalated.**
Notable fixes: batch shape corrected 864→**432** complex series (`[432,2,256]`); axis check
**fails closed on any non-ACCEPTED** verdict + per-entrypoint cert guard keyed to
`sha256`+`axis_spec_hash`; the flatline decision reframed as **mechanism-based (M2 audit),
never cohort-survival** (leakage); executable **tuned-ε handoff** (M7 re-extracts with a
train-fitted `epsilon_by_order`); outcome-(c) data-adaptive path forks M5 (stops before cohort
features, emits a non-authoritative `qc_characterization_77ghz.csv`); shard fingerprinting;
8-byte-float loader check; vectorized batch standardization; non-curated fixed-frame smoke.

**Owner decisions (2026-07-23):**
- **Flatline rule → outcome (b)** (mechanism-corrected exact replacement of the **77 GHz**
  screen; the 10 GHz screen is untouched and stays frozen since M2). Exact corrected rule to
  be specified at M5 step 6 from the ADC-quantisation mechanism, not cohort survival. M5 thus
  proceeds to the cohort feature runs; (a)/(c) retained as fallbacks. (A-M5-6.)
- **A-M5-3 (standardization) and A-M5-4 (log placement) confirmed and APPLIED** to the
  `implementation_plan.md` Exp G chain: step (6) robust-standardizes real/imag separately
  (10 GHz-consistent); the order-aware log applies to the fused per-frame tensor at step (8),
  keeping the three-branch log axis (off / on+frozen-ε / on+tuned-ε).

Next: implement M5 per the plan (still awaiting go); step-6 flatline rule and the M6 freeze
are the open design gates. `test_no_leakage.py` untouched throughout (plan is per-frame,
unfitted).

---

## 2026-07-23 — **MILESTONE 4 COMPLETE.** Definition of done met in full.

**D1 — mandatory suite, no private data.** `uv run pytest` → **396 passed, 12 skipped**
(was 319/11 at M3 close; +77, of which 66 are in `test_wst.py`).
**D2 — real-cohort acceptance.** `uv run pytest --realdata` → **407 passed, 1 skipped**
(T18 only, as designed until M6).
**D3 —** `experiments/run_wst.py` wrote and re-verified
`results/wst/wst_diagnostics_10ghz.csv` (73 rows) + provenance; ~12 min cohort pass.
**D4 —** `tests/test_no_leakage.py` **byte-for-byte unmodified since M1**
(`git diff f3fbade HEAD -- tests/test_no_leakage.py` empty) and green.
**D5 —** HISTORY.md carries an entry per resolved step, including the dtype-fork
resolution (step 1), the measured geometry (steps 3–4), and the cohort finding + pooling
performance fix (step 7).
**D6 —** SECOND_CHAPTER.md §3 "WST features" written: the ms→(J,T) mapping, the measured
padding/border decision, the order-aware log with the corrected ε rationale, the pooling
degenerate-std departure, the session unit, the single-backend policy, and the cohort
characterisation.
**D7 —** amendments **A-M4-1..A-M4-6** are live in `plans/implementation_plan.md`
(§Library choices, §WST parameterization, §Feature families, the repo tree, Build order §4);
the two plan documents agree.

**The invariant held.** WST + pooling is a per-frame function of one frame plus frozen
constants: batched extraction is bit-identical to single-frame (T-W16), the whole extraction
is deterministic (T-W8), and the only cross-frame step is the declared session
mean+median. Nothing is fitted, so nothing enters the CV loop — `test_no_leakage.py` is
untouched.

**Milestone-4 scoreboard.** 3 new source modules
(`features/{wst,pooling,extraction}.py`) + `features/__init__.py`, 1 new experiment entry
point, 1 new test module (66 tests), 1 new config field (`wst.backend`) validated at load.
torch entered the env (float32-only frontend); the strict cross-backend tolerance was kept,
not loosened. Two facts discovered empirically — the falsified O(1)-coefficient assumption
and the pooling hotspot — neither of which changed a frozen parameter.

**Open for M5 (config freeze):** the complete A–G protocol freeze, including the WST search
space (tiling × log on/off) and the 77 GHz decisions; the parked 77 GHz flatline rule; the
first IBEX configs/scripts. torch's T18 mutation leg still activates at M6. Nothing committed
yet — awaiting the owner's word.

## 2026-07-23 — M4 follow-up diagnostic: **is the order-2 scale stable enough that a
## data-derived ε would be leakage-safe? Yes fold-to-fold (<1%), but subjects vary ~14%.**

Read-only diagnostic on the existing cohort CSV (no WST recompute, no frozen parameter
touched) prompted by an owner question: if ε were computed per LOSO fold from training
subjects, would the value move much fold to fold? A near-constant ε means adapting it is
essentially leakage-free (the number barely depends on which subjects you use).

**Method.** For each held-out subject, take the median order-2 pre-log scale over the other
15 subjects' sessions (the fold's training statistic); look at the spread of those 16 fold
values. Separately, the spread across the 16 *individual* subjects, to tell "median is
robust" apart from "subjects genuinely alike".

```
                 across the 16 FOLDS            across the 16 SUBJECTS
T1 order2   max/min 1.01  CV 0.45%         max/min 1.73  CV 13.8%
T2 order2   max/min 1.02  CV 0.80%         max/min 1.78  CV 14.7%
T3 order2   max/min 1.02  CV 0.69%         max/min 1.83  CV 14.9%
```

**Reading.** (1) A per-fold ε would be near-identical across folds (<1%), so a fold-local
vs global choice differs by <1% — the leakage cost of adapting ε here is negligible. (2)
But individuals differ by up to ~1.8×, so part of the fold-stability is the median being
robust to dropping ~4 of 68 sessions, *not* subject homogeneity — and extrapolation to a
genuinely different setup (not this project's continuation, which reuses the same radar/
distance) stays an assumption. (3) **Tuning ε *to* the order-2 scale returns ~1e-6 — i.e.
today's value** (ε is already ~64% of the T1 order-2 median); *un-flooring* order-2 needs ε
much smaller than the scale, which trades flattening for near-zero noise amplification —
a two-sided tradeoff whose sweet spot is genuinely uncertain. Stability tells us adapting ε
is safe/cheap; it does **not** tell us it improves prediction. Recorded, ε unchanged —
this motivates the M5-pre-registered third log branch (implementation_plan.md §WST
parameterization / §LOSO harness search space), decided at M6, never now.

## 2026-07-23 — M4 step 7: `features/extraction.py` + `run_wst.py` cohort run.
## **All 73 sessions finite in ~12 min; but ε=1e-6 is NOT negligible vs the tiny order-2
## coefficient scale — the plan's "O(1) coefficients" assumption is measured false.**

`src/dehyd/features/extraction.py` (the reusable manifest→features wiring, in `src/` so
the M6 harness never imports a CLI script) + `experiments/run_wst.py` (thin CLI over it) +
extraction tests (T-W14 guard, T-W18 variants≡single-variant + call-count, pre-log scale).

**Cohort result — `results/wst/wst_diagnostics_10ghz.csv` (73 rows, matches M2/M3):**
7168 eligible frames, 73 sessions, 16 subjects. **`all_variants_finite = True` for every
session** across all (reduction × channel × tiling × log × family) branches — the
finiteness battery holds on real data at cohort scale. Total WST wall-clock **722.7 s
(~12 min)**, matching the ~14-min projection.

**Feature dimensions (constant across sessions), nominal / effective / raw per the
≥2-sample segment-std rule:** T1 mag 4452/4452/5194, T1 iq 8904/8904/10388; T2 mag
2796/**2330**/1398 (effective < nominal — the 1-sample first half drops one std), T2 iq
5592/4660/2796; T3 mag 2094/**1745**/1047, T3 iq 4188/3490/2094. Effective = nominal only
for T1 (n_time = 7); T2/T3 (n_time = 3) lose one std/path exactly as A-M4-6 intends.

**THE FINDING — pre-log coefficient scale (cohort medians):**
```
tiling   order0        order1     order2      eps/|order2|
T1     -4.39e-02     7.98e-04    1.56e-06    0.64
T2     -3.99e-02     1.14e-03    4.79e-06    0.21
T3     -3.46e-02     1.28e-03    8.23e-06    0.12
```
- **order 0 is signed and negative** (median ≈ −0.04) — confirms empirically that order 0
  is a signed low-pass and MUST stay linear; logging it would be `log(negative)`.
- **The plan's rationale for ε — "the coefficients live on an O(1) standardized scale" —
  is FALSE.** The standardized *input* is O(1), but scattering coefficients are far
  smaller: order 1 ≈ 1e-3, order 2 ≈ 1e-6. ε = 1e-6 is ~3 decades below order 1
  (negligible there), but **12–64 % of the median order-2 scale** — so when log is on,
  `log(S + ε)` on order 2 is materially ε-floored: for the (many) below-median order-2
  paths, ε dominates and compresses them toward log(ε) ≈ −13.8.
- **Action per the M2/M3 doctrine: none to the parameter.** ε stays frozen at 1e-6 —
  changing it on seeing this would be the forbidden data-driven retune. The pipeline
  already carries the mechanism that resolves whether this matters: **log on/off is an
  inner-CV axis at M6**, so the CV empirically decides per fold whether ε-floored order-2
  logging helps. The finding *strengthens* the case for keeping log selectable rather than
  always-on, and is recorded in SECOND_CHAPTER §3 as such. Flagged to the owner, not acted
  on. (The misleading console line "eps << order1/2" was corrected to print the eps/scale
  ratio; the CSV always carried the true numbers.)

**Performance failure and fix (kept in the log).** The first cohort attempt ran at
**~96 s/session → ~2 h projected**, not ~14 min. Profiling one session isolated the cause:
`pool_stats` was **7.82 s** for T1 iq (100 frames) vs 2.31 s for the scattering itself —
its per-path Python loop runs C·n_paths·segments = 4452 iterations/frame with tiny
`.mean()`/`.std()` slices. Vectorizing `pool_stats` over channels and paths (assemble each
segment×stat as a `[C, n_paths]` column, stack, transpose to channel→path→stat, flatten)
cut it to **0.053 s (150×)** with bit-identical values and order (the hand-computation
tests pin both). Full session 96 s → **11.7 s**; cohort ~12 min. build_scattering is *not*
the bottleneck (≈0.01–0.04 s), so no bank caching was added.

**Cross-backend on real frames** (in the realdata test): numpy-f64 vs torch-f32 max
elementwise ratio 0.36–0.47 (< 1), rel L2 ≈ 4e-7 — passes the strict "float64" policy on
actual data, where more near-zero coefficients push the ratio up but well inside bounds.

## 2026-07-23 — M4 steps 3–4: `features/wst.py`. **Measured geometry pins the reviewer
## values exactly; order-aware log, batched transform, and cross-backend gate all green.**

`src/dehyd/features/{__init__,wst.py}` + `tests/test_wst.py` (41 tests). The kymatio
parameterization, all shape/fs-agnostic (M9 reuse), constants from `WSTConfig` only.

**Measured filter-bank geometry (numpy frontend, n_in = 470, fs = 520834), pinned as
T-W2 regression values — matches the reviewer-sampled values exactly:**

```
tiling   Q       ms    T    J   n_paths  n_time   pad_left  pad_right  padded_len
T1    (10, 4)  0.20  104   7    742      7        277       277        1024
T2    ( 8, 2)  0.30  156   8    466      3        277       277        1024
T3    ( 6, 2)  0.40  208   8    349      3        277       277        1024
```

- **Padding is MEASURED, never assumed** — `pad_left`/`pad_right` read back from the
  object; `padded_len = n_in + pad_left + pad_right = 1024` (which *is* 2^10 here, but
  that is observed, not hard-coded — T3 at J=8 still pads to 1024, not 2^18). The
  deprecated `.N` attribute is avoided; padded length comes from the pad math.
- **n_time = 3 for T2/T3** — this is the measured fact that makes A-M4-6's ≥2-sample
  segment-std rule necessary (a 1-sample first half → identically-zero std). n_time = 7
  for T1 keeps all 6 stats.
- **Path order counts:** order 0 → 1 path, order 1 → 55, order 2 → 686 (T1); order-1 `xi`
  is strictly decreasing (kymatio convention), asserted in T-W2.

**Order-aware log (`apply_order_log`).** Orders 1–2 → `log(S + 1e-6)`; **order 0 stays
linear** — a crafted negative order-0 coefficient (S0 is a signed low-pass) passes through
untouched and finite (T-W6), where logging it would give NaN. A path-count mismatch vs
`meta` raises. The finiteness battery (mag/iq × orders 0/1/2 × log on/off, all 3 tilings)
is green (T-W5).

**Batched transform (`scatter_frames`).** [N, C, 470] folded into kymatio's leading batch
dim in one call; **bit-identical to stacked single-frame calls** for all three tilings
(T-W16). `scatter_channels` is defined as `scatter_frames(x[None])[0]`, so the two paths
cannot diverge.

**Cross-backend gate (`backend_agreement` + `AgreementResult`).** Two frozen policies in a
table, no caller tolerances; raises on shape/empty/non-finite/dtype violations; returns the
measured `max_elementwise_ratio` and `rel_l2`. **numpy-f64 vs torch-f32 passes the strict
"float64" policy** on raw AND logged tensors, both channels, all three tilings (T-W9) —
consistent with the step-1 finding. **Suite: test_wst 41 passed.**

## 2026-07-23 — M4 step 1: `uv add torch`. **kymatio's torch frontend is float32-only —
## the planned float64-vs-float64 cross-backend comparison is impossible; strict
## tolerances kept anyway (owner-approved), no fallback needed.**

`uv add torch` → **torch 2.13.0+cpu**, pulling filelock, fsspec, jinja2, markupsafe,
mpmath, networkx, setuptools, sympy, typing-extensions. **The scipy pin held:** the
resolver kept **scipy 1.16.3** (< 1.17), so `from kymatio.numpy import Scattering1D`
still imports — the M1 kymatio-breaks-on-scipy-≥1.17 trap did not fire. numpy stayed
2.4.6. Added `packaging>=21.0` to the dev group (test_env now imports it) and a torch
import + `test_scipy_pin_survives_torch` (parsed-version comparison, not string —
`Version(scipy.__version__) < Version("1.17")`). **Env suite: 3 passed.**

**Finding that triggered the pre-declared dtype fork (MILESTONE_4_PLAN §2.2).**
kymatio's **torch** `Scattering1D` runs its filter bank in **float32**: a float64 input
raises `TypeError: Input and filter must be of the same dtype`. So the plan's assumed
"convert to a float64 torch tensor, compare float64-vs-float64" is **not achievable in
the pinned stack** — an environment fact, exactly the contingency §2.2 pre-declared.

**Measured the only achievable comparison — numpy-float64 vs torch-float32** — on a
`default_rng(0)` standard-normal 470-sample signal through all three tilings:

```
tiling      Q       T    J   out shape   rel L2     max elemwise ratio (float64 policy)
T1      (10, 4)   104   7   (742, 7)   6.57e-08   0.0437   PASS
T2      ( 8, 2)   156   8   (466, 3)   8.59e-08   0.0090   PASS
T3      ( 6, 2)   208   8   (349, 3)   6.72e-08   0.0039   PASS
```

Two things fall out of this and are recorded now:
1. **The measured output shapes exactly match the reviewer-sampled values** T1 (742, 7),
   T2 (466, 3), T3 (349, 3) — the T-W2 regression pins (confirmed at build, not assumed),
   and the n_time = 3 for T2/T3 that motivates the ≥2-sample segment-std rule (A-M4-6).
2. **numpy-f64 vs torch-f32 passes the STRICT "float64" policy** (rtol 1e-4, atol 1e-8):
   max elementwise ratio ≤ 0.044 (< 1), rel L2 ≈ 1e-7. float32 accumulation across the
   scattering depth stays comfortably inside the 1e-4 relative bound.

**Owner decision (2026-07-23): keep the strict tolerances; do NOT adopt the float32
fallback.** The fork the plan pre-declared required owner approval to loosen; the owner
chose *not* to loosen, because the data clears the tight bar. The `backend_agreement`
"float32-fallback" policy (rtol 1e-3, atol 1e-5) stays defined but unused — invoking it
would need a fresh owner decision. Propagated to MILESTONE_4_PLAN §2.2 (dtype policy),
the `scatter_frames`/§5 wording, and pyproject's dependency comment; the cross-backend
formula in implementation_plan.md is unchanged (still the strict "float64" policy).

## 2026-07-23 — **MILESTONE 3 COMPLETE.** Definition of done met in full.

**D1 — mandatory suite, no private data.** `uv run pytest` → **319 passed, 11 skipped**
(was 260/10 at M2 close; +59 tests, 45 of them in `test_preprocess.py`).
**D2 — real-cohort acceptance.** `uv run pytest --realdata` → **329 passed, 1 skipped**
(T18 only, as designed until M6).
**D3 —** `experiments/run_preprocess.py` wrote and re-verified
`results/preprocess/preprocess_diagnostics_10ghz.csv` (73 rows); distributions recorded
in the step-6 entry below.
**D4 —** `tests/test_no_leakage.py` is **byte-for-byte unmodified since M1**
(`git diff f3fbade HEAD -- tests/test_no_leakage.py` empty) and green — 24 passed,
2 skipped.
**D5 —** HISTORY.md carries an entry per resolved step, including all three required
departure logs: no-window-in-the-primary-path, the median/MAD form with its eps
placement, and the Option-B mask correction.
**D6 —** SECOND_CHAPTER.md §2 "Preprocessing" written: the paper-vs-code ambiguity
resolved as a methodological argument, the provenance of every parameter, the
finite-record measurements, the three cohort findings, and the correctness argument.
**D7 —** amendments **A-M3-1..A-M3-7** are live in `plans/implementation_plan.md`
(§Preprocessing steps 3–4, §Deliberate departures, the repo tree, Build order §3 and
§5); the two plan documents agree.

**The invariant held.** Preprocessing is a per-frame function of one frame plus frozen
constants: T-PP20 asserts a frame's output is identical whether processed alone or
beside an arbitrary companion — including one scaled ×1000 — and T-PP22 asserts
bit-identical repeats. Nothing here is fitted, so nothing enters the CV loop.

**Milestone-3 scoreboard.** 5 new source modules
(`preprocess/{__init__,filters,reduce,standardize,pipeline}.py`), 1 new experiment
entry point, 1 new test module, **59 new tests** (319 vs 260). Five config fields added,
each classified as search axis / pre-declared ablation / frozen protocol constant before
any result existed. Three deliberate departures from the reference logged with reasons.
**Four facts discovered empirically rather than assumed** — the arithmetic-vs-geometric
mid-band ambiguity, the finite-record energy reality, the windowed-vs-unwindowed
zero-ROI counterexample, and the 1.50 m target range — none of which changed a frozen
parameter.

**Open for M4:**
- Feature extraction consumes `preprocess_cube(...)` → float64 [n_frames × C × 470].
  The kymatio border-effect warning at `Scattering1D(J=7, shape=(470,))` is an M4
  concern: measure padding and output shape from the instantiated filter bank, never
  assume them.
- torch enters the environment at M4 (cross-backend WST check); the T18 torch mutation
  leg stays skip-marked until M6 — owner decision, unchanged.
- `configs/ibex.yaml`, `scripts/ibex/` still deferred to the first IBEX milestone.
- The 77 GHz any-trace flatline rule remains parked for an owner decision at M5.
- Nothing committed yet — awaiting the owner's word, per the ground rules.

---

## 2026-07-23 — M3 step 6: `run_preprocess.py` over the full cohort. **The beat sits
## at 1.50 m and the ROI peak is genuinely dominant (peak_share 0.51 vs 0.33 uniform).**

Thin CLI, all logic in pure helpers (M2 audit pattern) so the diagnostics are testable
without a cohort run. **Primary-only guard:** the script compares the consumed
`config.preprocess` against the whole canonical `PreprocessConfig()` and refuses to run
otherwise, naming the deviating fields — checking just the two ablation switches would
have let `model_gate_m: [0.9, 3.0]` (an inner-CV *candidate*) overwrite the primary CSV
under a "primary" label.

### Cohort result (73 sessions, 7168 eligible frames, 16 subjects — matches M2 exactly)

```
peak bin (mode)   5           bin 4: 1 session, bin 5: 41, bin 6: 31
peak Hz median    4876.7  ->  1.50 m
energy retention  median 0.407   [0.061, 0.644]
roi/total         median 0.930   [0.775, 0.977]
peak_share        median 0.512   [0.410, 0.739]
missing cells     0            all four variants finite in every session
```

**Findings, recorded not acted on:**

1. **The target range is ~1.50–1.80 m, not the ~1 m the config comment assumed.**
   72 of 73 sessions put the dominant beat in bin 5 or 6 (4877 / 5852 Hz = 1.50 /
   1.80 m). The frozen 1–2 m model gate contains this comfortably — the gate is better
   justified by the data than by the assumption that motivated it — but the *stated
   reason* ("subject seated ~1 m") is now known to be off by half a metre. Left as is:
   changing the gate on seeing this would be exactly the data-driven retune the
   milestone forbids. The correct reading goes in SECOND_CHAPTER §2.
2. **The peak is genuinely dominant.** This is the measure Codex's round-1 comment 4
   asked for, and it answers it: with 3 ROI bins, a flat spectrum would give
   `peak_share = 0.333`; the observed median is **0.512** (min 0.410), so one bin
   really does carry the return. Option-B's premise holds on this data.
3. **Caveat on `roi_to_total` (0.930) — it is measured POST-filter, as the plan
   specifies, so it is largely a filter-selectivity descriptor and cannot be low by
   construction.** It is not evidence of target presence; `peak_share`, which compares
   *within* the ROI, is. Recorded so the chapter does not over-read it.
4. **Per-session peak stability is high:** 45 of 73 sessions have every frame on one
   bin, 22 span 1 bin, 6 span 2. The detection is not jittering frame to frame.
5. **Energy retention varies 10× across sessions** (0.061 to 0.644, median 0.407) —
   far below the 0.76 a pure mid-band tone retains (T-PP6), as expected since real
   frames carry energy across and outside the band. The three weakest (s11 10am 0.061,
   s12 2pm 0.070, s5 4pm 0.095) still pass QC and still show high roi/total, so this is
   overall signal level, not a band mismatch. Noted for M4.

**One test fixture was impossible and revealed a real property.** T-PP23's aggregation
case originally fed all-zero frames to `session_diagnostics` to force a missing
`peak_share` — but `energy_retention` raises on a zero-energy frame *by contract*
(QC's in-band ratio ≥ 0.30 is impossible at zero power). Chasing that showed the
"all-missing session → empty CSV cell" path is **unreachable for eligible frames**:
`peak_share` is undefined only at exactly-zero ROI power, and any frame with energy
leaves float-positive power there. The rule was factored into a
`median_skipping_missing` helper and tested directly; the guard stays so "undefined"
remains distinguishable from "zero", but it is documented as not expected to fire —
and the cohort run confirms it, with **0 missing cells in 73 sessions**.

Artifact written and re-verified: `results/preprocess/preprocess_diagnostics_10ghz.csv`
(73 rows). Provenance carries `analysis_role: "primary"`, the full `filter_spec`
(padlen 27, Wn, band edges) and the ROI bins. **Suite: 319 passed / 11 skipped;
`--realdata` 329 passed / 1 skipped (T18).** Staged-file list checked with `git add -An`
per the M1 lesson: all five `src/dehyd/preprocess/*.py` appear — the `.gitignore`
package trap did not recur.

---

## 2026-07-23 — M3 steps 3–5: reduction, standardization, pipeline. **First real-data
## contact: the dominant beat sits at ~1.50 m, not the assumed ~1 m.**

**Step 3 — `reduce.py`.** Option A (chirp mean), `detect_option_b_peak` →
`OptionBDetection(peak_bin, power, roi_bins)`, `option_b_mask`, `reduce_option_b`,
`edge_trim`. ROI = model gate, **no margin**, half-spectrum bins 0..266 → **bins 4,5,6**
at the default config (df = 975.34 Hz), verified against independent arithmetic; the
0.9–3.0 m candidate gives bins 4..10 (bin 3 = 2926.0 Hz misses the 2931.7 Hz edge by
5.7 Hz). `edge_trim` **raises rather than clamps** — the reference's silent
`min(EdgeTrim, N/4)` would hide a mis-set config.

**Departure logged — the Option-B mask is a correction, not a port.** `wst_extract.m`
keeps only `peakBin + (0:nb)` (**one-sided**, contradicting its own "±bins" docstring)
and then applies MATLAB's endpoint-zero `hann(numel(idx))` across the concatenated
positive+mirror block: at nb = 1 that is `[0, .75, .75, 0]`, which **zeroes the detected
peak itself**, leaving only bin peak+1 at 75%. We implement the form the docstring and
the main plan describe — symmetric ±nb, **full weight on the peak**, Hann shoulders:
weights = interior of `hann(2nb+3)` → nb=0 [1.0], nb=1 **[0.5, 1.0, 0.5]**, nb=2
[0.25, 0.75, 1.0, 0.75, 0.25]. Mirrors take the same weight; a self-mirroring bin
(DC/Nyquist) takes the **max**, never the sum. A mask that would keep every bin raises
(a pass-through Option B is a disabled reduction, not a configuration).

**The zero-ROI claim was wrong in an earlier draft and is now pinned by an adversarial
fixture** (Codex review round 4). Detection is Hann-windowed; the mask is applied
*unwindowed*. The frequency-domain periodic-Hann kernel [−¼, ½, −¼] can annihilate the
windowed ROI while the unwindowed bins under the mask stay nonzero. Constructed and
**confirmed numerically**: unwindowed bins 3..7 = [1, 0, −1, −2, −3] → windowed ROI
power ≈ 1e-34 (exactly zero to float precision) yet the reduced output carries
9.4e-4 of energy. So the frozen behaviour is "mask the first ROI bin, whatever that
yields" — finite, deterministic, and flagged downstream by `peak_share = NaN`. An
exactly-zero frame is tested separately, where the argmax tie-break genuinely does
return the first ROI bin.

**Step 4 — `standardize.py`.** `robust_standardize` = `(x − median)/(1.4826·MAD + eps)`
with **eps = float64 machine epsilon placed OUTSIDE the scale factor** (the reference
uses `1.4826·(MAD + eps)`; numerically irrelevant — it is a division guard, not a
tuning constant — but one form must be frozen for bit-reproducibility). This is the
settled departure from the reference's mean-centre/MAD-scale mix. `meanstd_standardize`
uses **ddof = 0** (numpy's population convention; MATLAB's `std` defaults to ddof = 1
and no reference constrains the choice), pinned by an exact hand-computation test that
a ddof = 1 implementation fails. Channels are standardized **each from its own
statistics**.

**Step 5 — `pipeline.py`.** `preprocess_frame` / `preprocess_cube`, the sequence
readable in one screen: gate → reduce → trim → channel → standardize → float64
[C × 470]. `reduction`/`channel` are **call arguments, not config** (they are inner-CV
axes at M6, so one config must produce every variant). T-PP15 makes the
trim-after-reduction ordering structural rather than a comment: trimming first would
change the FFT bin grid (470-pt → df = 1108 Hz, ROI bins 3..5), so the two orders
genuinely disagree and the test can tell them apart.

### First contact with real data (`subject_1_8am.mat`, 35 QC-passing frames)

```
option-B peak bins  {4: 1, 5: 34}      df = 975.3 Hz
peak Hz median      4876.7   ->  4876.7 / 3257.5 Hz/m = 1.50 m
peak_share median   0.460
```

**The dominant beat sits at ≈1.50 m, not the ≈1 m the plan assumed** when it called the
1–2 m gate "physically motivated (subject seated ~1 m)". The value lands comfortably
inside the 1–2 m model gate — near its centre, in fact — so the frozen gate is
*better* justified by the data than by the assumption behind it. Nothing was changed:
this is recorded as a finding. (Note s1 8am is one of the 7 QC-**ineligible** sessions
from M2 at 35/100; the frames used here are still genuine QC-passing frames, and the
test is about pipeline mechanics, not about that session's eligibility.)

**Suite: 309 passed, 11 skipped** (36 in `test_preprocess.py`, one of them realdata).
`tests/test_no_leakage.py` is **byte-for-byte unmodified** (`git diff` empty) and green
— 24 passed, 2 skipped.

---

## 2026-07-23 — M3 step 2: `preprocess/filters.py` — the band gate. Three test
## fixtures were wrong on first contact; all three taught something.

`design_bandpass_sos` / `bandpass_filtfilt` / `fft_gate` / `apply_band_gate` /
`filter_spec`, written **shape- and fs-agnostic** so the 77 GHz chain (M9, fs = 500 kHz,
N = 256) reuses them unchanged. Band from the **model** gate (1–2 m →
3257.5–6514.9 Hz, Wn ≈ 0.0125–0.0250), never the QC gate. `padlen = 27` is passed
**explicitly** (not left to scipy's default) and T-PP1 pins both the value and its
bit-identity with the library default. **9 tests pass.**

**Departure logged — no window before the time-domain filter (ROADMAP §3.2).** The
ROADMAP lists "Hamming window; range FFT; SOS Butterworth". A window suppresses FFT
spectral leakage, which is meaningless for a time-domain IIR filter, and pre-tapering
the chirp would attenuate real signal energy at its edges. Windows are applied only
where an FFT is actually taken (QC in-band screen, Option-B detection, the FFT-gate
ablation). filtfilt's edge transients are handled by EdgeTrim instead.

**Three fixture failures, each a real fact about the filter — the module was not
changed to accommodate any of them (§5's "no parameter chasing" rule):**

1. **"Mid-band" was ambiguous.** I probed at the *geometric* mean (4606.7 Hz) while the
   plan's regression values were measured at the *arithmetic* centre (4886.2 Hz). Both
   are inside the passband; only one matches the recorded numbers. Fixed by defining
   mid-band **once**, as the arithmetic centre, in a fixture every probe now uses.
2. **The zero-phase fixture was too narrow.** A σ = 40-sample Gaussian burst has ~2 kHz
   of bandwidth against a 3.3 kHz passband, so the filter distorts it enough to move
   the envelope peak by one sample. σ = 80 fits inside the band and lands the peak
   exactly at centre. (σ = 120 is *worse* — the envelope becomes so flat that its
   argmax is noise-dominated, 273 vs 267.) The test now also asserts a **single causal
   pass shifts the same burst by ~131 samples**, so it demonstrably has the power to
   catch a non-zero-phase implementation instead of passing vacuously.
3. **The stopband assertion assumed steady state.** A 2.6 m tone under the 1–2 m gate
   retains **3.5%** of its energy, not < 1% — the same finite-record leakage the plan
   already pins in T-PP6 (a 534-sample record cannot reach the design stopband). The
   assertion now states the honest claim: kept ≈ 0.81, rejected < 0.1, and kept is
   > 10× rejected.

**Also measured (recorded, not asserted):** filtfilt with `padtype='odd'` is **not**
exactly time-reversal-symmetric on a 534-sample noise record (max |diff| ≈ 0.49 in the
interior). I had considered using that identity as the zero-phase test; it is not a
property scipy guarantees under edge padding, so the cross-correlation-lag test is used
instead. Noted so nobody re-derives it.

---

## 2026-07-23 — M3 step 1: preprocessing config fields + cross-field band validation.

`plans/MILESTONE_3_PLAN.md` was approved after **4 review rounds (15 comments, all
applied, none disputed)**; amendments A-M3-6 and A-M3-7 were propagated into
`implementation_plan.md` during review. Implementation starts here.

**Five new `PreprocessConfig` fields**, each classified before M6 so no alternative can
become an undeclared search axis:
`gate_method="butterworth"` and `standardize="robust"` are **ablation switches**
(non-default = pre-declared ablation only); `peak_neighbors=1`, `mask_taper=True`,
`fft_gate_transition_hz=500.0` are **frozen protocol constants** (non-default is
test-only, rejected by modelling entrypoints). `peak_neighbors=1 + mask_taper=True`
*is* the plan's "±1-bin two-sided Hann-tapered mask" — they are constants, not knobs.

**Validation added.** Field level: `gate_method`/`standardize` from fixed choice
tuples; `mask_taper` strictly bool (0/1 rejected — YAML has real booleans, so an int
is a typo); `peak_neighbors` integer ≥ 0 (0 = keep the peak bin alone, legitimate in
tests). This forced `_int_field` to take a `minimum` (it hard-coded `> 0`); its
message became "must be >= N", and three existing M1/M2 parametrised cases had their
expected regex updated — same assertions, new wording.

**New cross-field check `_check_model_band`,** three failures, all hard errors:
1. **whole band strictly below Nyquist** (`0 < f_lo < f_hi < fs/2`). Deliberately
   stricter than `_check_qc_band`, which only rejects a band *starting* above Nyquist:
   the QC screen is an FFT mask whose upper edge is legitimately Nyquist-clamped
   (frozen at M2), but `scipy.signal.butter` raises on `Wn ≥ 1`, so a *straddling*
   gate would pass config load and fail deep inside the filter — exactly what the
   fail-at-config-load rule forbids. Clamping instead would make the two gate methods
   filter different bands under one config.
2. **`model_gate_m ⊆ qc_gate_m`** (inclusive, so the 0.9–3.0 m inner-CV candidate —
   which equals the QC gate — still loads). QC fixed the frame population on the wider
   gate; a model gate reaching outside it would use energy QC never screened for.
3. **FFT-gate non-vacuity** (skirts covering the whole spectrum = a filter that
   filters nothing), checked only on the `fft` path.

`configs/preprocess.yaml` mirrors all five with their classification stated in the
comments. **Result: `tests/test_config.py` 67 passed, 1 skipped** — including the
straddling-Nyquist case, which needed the QC gate widened to 70–90 m so the QC check
passes first and the *model* rule is what actually fires.

**Commit `395eb62`** on `v1_milestone_2` — 20 files, pushed to
`origin/v1_milestone_2` (new upstream). Staged list checked file-by-file before
committing, per the M1 lesson: both `src/dehyd/qc/__init__.py` and
`src/dehyd/qc/screens.py` are present (the `.gitignore`-swallows-a-new-package trap did
**not** recur), both curated artifacts (`results/qc/qc_survival_10ghz.csv`,
`results/qc/audit_77ghz.json`) are in, and `results/runs/` is correctly excluded —
verified with `git add -An` rather than assumed.

**Branch `v1_milestone_3`** created from `v1_milestone_2` and checked out. HANDOFF.md
rewritten for the milestone-3 bootstrap (owner-requested — it is never updated
automatically). Nothing is merged to `main` yet; `v1_milestone_1`, `v1_milestone_2`
are pushed.

**M3 starts from:** `eligible_frames(manifest_qc)` = 7168 frames across 73 sessions and
16 evaluable subjects, and implementation_plan.md §"Preprocessing — executable
sequence" (order-4 SOS Butterworth zero-phase on the complex fast-time axis using the
**model** gate, Options A/B reduction, EdgeTrim 32 **after** reduction → 470 samples,
median/MAD robust standardisation).

---

## 2026-07-21 — **MILESTONE 2 COMPLETE.** Definition of done met in full.

**D1 — mandatory suite, no private data.** `uv run pytest` → **260 passed, 10 skipped**
(the 10 skips are 9 `realdata` tests plus T18). Was 151/8 at M1 close.
**D2 — real-cohort acceptance.** `uv run pytest --realdata` → **269 passed, 1 skipped**
(T18 only).
**D3 —** `experiments/run_qc.py` writes and re-verifies
`results/qc/qc_survival_10ghz.csv`; survival recorded in the step 3–4 entry below.
**D4 —** `run_regression.py` builds its folds from the post-QC evaluable subjects;
provenance carries the QC config including the new margin.
**D5 —** the 77 GHz audit ran on one real file; all five verdicts recorded in a
provenance-complete `results/qc/audit_77ghz.json`.
**D6 —** HISTORY.md has a per-step entry (steps 1–7); SECOND_CHAPTER.md §1 "Data &
ground truth" now carries the QC section.
**D7 —** amendments **A1–A7** are live in `plans/implementation_plan.md`; the two plan
documents agree.

**The invariant held.** `tests/test_no_leakage.py` is **byte-for-byte unmodified since
M1** (`git diff HEAD` empty) and green — 24 passed, 1 skipped. QC never enters CV: it
is a per-frame function of one frame plus frozen constants, applied once before any
split exists, and T-QC7 asserts a frame's verdict is identical whether screened alone
or beside arbitrary companions.

**Milestone-2 scoreboard.** 2 new source modules (`qc/screens.py`, QC section of
`manifest.py`), 2 new experiment entry points, 2 new test modules, **101 new tests**
(260 vs 159). Three genuine facts discovered empirically rather than assumed — numpy's
histogram behaviour on a degenerate range, YAML 1.1's signed-exponent rule, and the
77 GHz real-valued storage — each of which would have surfaced later as a confusing
failure. Two frozen thresholds met the real data for the first time; **neither was
touched**, and both surprising results (670 in-band rejections at 10 GHz, 7/10 flatline
rejections at 77 GHz) are recorded as findings for the milestone-5 freeze.

**Open for M3 / M5:**
- **Owner decision at M5:** the 77 GHz any-trace flatline rule rejects 7 of 10 audited
  frames because of ADC quantisation, not a dead receiver. Frozen as-is for now.
- Preprocessing (M3) consumes `eligible_frames`; the analysis population is 7168 frames
  across 73 sessions and 16 evaluable subjects.
- `configs/ibex.yaml`, `scripts/ibex/` still deferred to the first IBEX milestone.
- Nothing committed yet — awaiting the owner's word, per the ground rules.

---

## 2026-07-21 — M2 steps 5–6: evaluability hookup + **the 77 GHz audit.**
## **Axis hypothesis CONFIRMED. Two findings that change milestone-5 planning.**

**Step 5** (`run_regression.py`): folds now come from `evaluable_subjects` after QC.
Smoke on real data → 7330/8000 frames, 73/80 sessions, **16 outer folds, 5 inner
each** — the clean full-cohort case, unchanged from M1 because no subject lost all
five sessions.

**Step 6**: `uv add h5py` (3.16.0), `experiments/audit_77ghz.py` (pure parameterised
helpers + thin `main`), `tests/test_audit_77ghz.py` (23 synthetic tests, no private
data). Audit ran on `data/77ghz/subject_1_8am.mat` in **8.3 s**.

### Verdicts — all green

```
H1_shape   ACCEPTED    (16, 256, 256, 125) exactly as predicted; chunks (16,4,1,125), gzip
H1_storage ACCEPTED    but NOT the representation the plan froze — see finding 1
H1_axes    ACCEPTED    G_fast=0.2260  G_chirp=6.70e-06  D_chirp=0.9999  D_fast=0.4939
qc_smoke   NON_DEGEN.  3/10 frames pass; median in-band ratio 0.382 — see finding 2
chain      NON_DEGEN.  final/raw energy 2.98e-05 (gated) and 8.50e-06 (range-Doppler)
```

**The axis question is settled, and not marginally.** A1 needed D_chirp ≥ 0.5 → got
0.9999. A2 needed G_fast ≥ 0.05 → got 0.226. A3 needed G_fast ≥ 10·G_chirp → the
actual ratio is **≈34 000×**. The mirrored swapped-axis hypothesis fails on its very
first criterion (D_fast = 0.494 < 0.5), so the result is not a coin-flip resolved by
threshold placement. The check ran on the **raw** slab, before MTI, exactly as
required — MTI would have removed the near-zero-Doppler static-subject energy that
makes D_chirp ≈ 1 the discriminator it is.

### Finding 1 — the 77 GHz raw is REAL float64, not complex

The plan froze the accepted storage as "compound `real`/`imag`". The real files are
**plain `float64`**, `dtype.names is None`: ADC-like values quantised to 1/16,
|x| ≤ 2560. Cross-checked against the reference — `wst_extract77.m` and
`filter_gpt_butterworth77.m` never call `real()`/`imag()` on raw 77 GHz data, which is
consistent with a real-sampled capture. (The 10 GHz files genuinely are complex, which
is where the assumption came from.)

Handled as a **correction of a wrong a-priori assumption about the file format, not a
threshold chosen from data**: it is visible in HDF5 metadata alone, it changes no
frame's screen verdict, and the observed dtype descriptor is recorded either way.
`H1-storage` now accepts real-float *or* compound-complex and rejects everything else;
both plan documents were amended. **Consequence for Exp G, recorded in
implementation_plan.md:** the primary chain's "I/Q" comes from the **complex range-FFT
output** (chain step 4), not the raw cube. The chain itself is unaffected — it already
scatters the post-range-FFT slow-time series — but nothing before step 4 may assume
complex input.

### Finding 2 — the frozen 77 GHz flatline rule rejects 7 of 10 frames

Per-Rx flagged-trace counts over the 10-frame slab: **169–1601 of 2560 traces per Rx**,
spread fairly evenly across all 16 receivers — so this is **not** a dead channel. Cause
is the rule meeting heavily quantised ADC data: with 128 bins across a 256-sample
trace's own range, quantisation piles ≥64 samples into one bin easily. The ≈205×
multiplicity the reviewer flagged **does bite**, empirically.

Per the milestone invariant, **nothing was changed**: the rule was frozen a priori
(owner decision, 2026-07-21), the audit's job was to make the multiplicity visible
before the M5 freeze, and it has. The in-band screen by contrast looks healthy
(ratios 0.373–0.392, all above the 0.30 threshold, so 0 low-in-band rejections). **This
is an owner decision for milestone 5**, not something to be quietly retuned here.

### Other recorded facts

- **Chunk layout `(16, 4, 1, 125)` spans the entire frame axis**, so a 10-frame read
  still decompresses ~1.05 GB. Memory stays bounded (84 MB retained) and it costs
  ~8 s, which is why the bounded-slab contract is about *memory*, not I/O volume.
- **MTI removes 99.7 % of the energy** (ratio 2.70e-03) — exactly the "legitimate
  physical attenuation" the plan predicted when it rejected a bare `> 0` test. The
  1e-9 floor sits ~4 orders below the smallest observed stage ratio, so it discriminates
  true degeneracy without firing on real clutter suppression.
- Derived constants confirmed against independent computation: dr = 0.0749 m, range
  gate **bins 27..53**, QC mask **bins 26..54**, PRF 1953.125 Hz.

### Test-fixture bugs the tests caught in themselves

Two of my own fixtures wrote a `(frame, fast, chirp, rx)` cube straight to disk
instead of the on-disk `(rx, chirp, fast, frame)` layout, so the round-trip test was
asserting against the wrong shape. Fixed by making `write_fixture` apply the same full
reversal the audit uses on read (it is its own inverse) — which makes the round-trip
test genuinely round-trip. The end-to-end fixtures were also shrunk from the real
`(16,256,256,·)` to `(2,32,32,·)`; 32 fast-time bins still yield a non-empty range
gate (27..31) and QC mask (4..6), and the audit-test module dropped from ~6 s to 1.7 s.

**Tests:** `uv run pytest` → **260 passed, 10 skipped**.

**Next:** step 7 — journal close-out.

---

## 2026-07-21 — M2 steps 3–4: manifest QC bookkeeping + **first real-cohort QC pass.**
## **Success. 7330/8000 frames survive; 7 of 80 sessions dropped; N_eval = 16.**

**What was built.** `apply_qc` / `session_qc_report` / `eligible_frames` /
`evaluable_subjects` in `manifest.py` (§2.2), `experiments/run_qc.py` (§2.4), 20 new
manifest tests and 2 realdata tests.

### The real numbers (frozen screens, first contact with the cohort)

```
frames    : 7330 pass / 670 fail of 8000  (91.6% survive)
reasons   : nan/inf 0, flatline 0, low in-band 670   (rms flagged 2752, diagnostic only)
sessions  : 73 eligible / 7 dropped of 80
dropped   : s1 8am 35/100, s1 4pm 1/100, s3 10am 37/100, s4 2pm 35/100,
            s5 2pm 39/100, s6 8am 0/100, s16 10am 15/100   (all needed 50)
N_eval    : 16 evaluable subjects — every subject keeps >= 1 eligible session
analysis  : 7168 frames (the 162 passing frames inside dropped sessions are excluded)
```

**Independent corroboration.** ROADMAP §2 states "~7500 after QC" for 10 GHz, written
from the original study and never used to tune anything here. We get **7330** from
thresholds frozen before looking. The agreement is evidence the ported screens behave
like the originals; it is *not* a target that was fitted to.

**What the failures actually are.** Every rejection is the in-band energy screen —
**zero** NaN/Inf and **zero** flatline across all 8000 frames. So the dropped sessions
are acquisitions where the return simply is not in the 0.9–3.0 m gate (subject 6's 8am
session: 0/100 frames in-band; subject 1's 4pm: 1/100), not corrupted files. Recorded
as a finding; **no threshold was touched** in response, per the §0 invariant.

**The RMS diagnostic is trigger-happy on this data — interpret accordingly.** 2752
frames (34%) carry the flag, concentrated in a few sessions (subject 16: 4pm 99/100,
2pm 82/100; subject 11 and 13: mostly 0). Cause is structural, the same effect the
unit tests hit: the robust z is taken across a frame's own **20** chirps, which are
near-identical, so the MAD is tiny and the z is very sensitive. This is the
reference's own definition and it rejects nothing — but the count must be read as
"chirp-to-chirp variability is non-uniform here", not as "2752 anomalous frames".
Kept diagnostic-only exactly as the plan freezes it.

**Fail-closed join (§2.2), and why it is not paranoia.** `_join_qc` asserts key
uniqueness on both sides, merges with `validate="one_to_one"` plus an indicator check
for left/right-only keys, asserts the row count is unchanged, and restores `SORT_KEYS`
order. Four tests inject duplicate / missing / extra / duplicate-manifest keys and
require a loud `ManifestError`. The M1 trap this defends against is real:
`subject_1_10am` sorts *before* `subject_1_8am`, so any positional join silently
attributes one session's QC verdicts to another — `test_join_is_by_key_not_row_order`
constructs exactly that scenario (failure on 10am, none on 8am) and would catch it.

**Reconciliation.** Per-reason columns are non-additive incidence counts; the identity
asserted everywhere is `n_pass + n_fail_any == n_frames`. The all-zero-frame test
pins the overlap case (flatline **and** low in-band: counted in both reason columns,
once in `n_fail_any`).

**Artifact.** `results/qc/qc_survival_10ghz.csv` — written under
`config.paths.results_dir` (the config is the single output-path authority), re-read
and reconciliation-checked after writing, and verified **not** gitignored. Per the
ground rules it is written and verified but **not committed** until asked.

**Tests:** `uv run pytest` → **237 passed, 8 skipped**; `--realdata` survival +
determinism tests green. The survival test asserts structure only (80 cells present,
ratios in [0,1], reconciliation, `min_pass == ceil(0.5 × actual count)`) and
deliberately makes **no** expected-rate assertion.

**Next:** step 5 — `run_regression.py` folds over post-QC evaluable subjects.

---

## 2026-07-21 — M2 step 2: `src/dehyd/qc/screens.py` + `tests/test_qc.py`.
## **Success**, after two wrong assumptions of mine were corrected by the tests.

**What was built.** The four frozen 10 GHz screens as pure per-frame functions:
`FrameQC`, `run_qc_frame`, `run_qc_cube`, `in_band_mask`. Implements
MILESTONE_2_PLAN §2.1. 34 tests (T-QC1..15).

**Frozen semantics as implemented.** NaN/Inf → reject; flatline = any chirp whose
200-bin magnitude histogram (over that chirp's own range) has a bin ≥ 0.25·534 = 133.5
i.e. ≥134; in-band = periodic-Hann 534-pt FFT, half-spectrum bins 0..266 (DC in,
Nyquist out), averaged over the 20 chirps, band 2931.7–9772.4 Hz widened by ±1000 Hz
→ **mask bins 2..11 (10 of 267)** — measured, not assumed; reject below 0.30.
Robust-RMS z across the frame's own 20 chirps, >4.5 → flag only.

**`passed` is a property, not a stored field** (a small departure from the plan's
sketch): the rejection rule then cannot be violated by construction, and `rms_flag`
can never leak into it. T-QC14 stays meaningful by asserting the battery actually
contains frames that are RMS-flagged *and* passing.

**Wrong assumption #1 — numpy does NOT expand a zero-width histogram range.** The
plan (and my code comment) claimed a constant chirp yields a single populated bin.
In fact `np.histogram(x, bins=200)` **raises** `ValueError: Too many bins for data
range` whenever the span is too narrow for 200 distinct float64 edges — and that
includes any *near*-constant chirp, not just an exactly constant one: a noiseless CW
tone `exp(2πift)` has |x| constant to ~1e-16. Ten tests crashed. Fix: build the edges
with `linspace` and check `edges[:-1] < edges[1:]` (numpy's own criterion) — if the
span is degenerate, flag flatline directly and skip the histogram. That is also the
*correct physics*: constant magnitude is exactly what the screen exists to catch, and
MATLAB's `histcounts` reaches the same verdict by choosing its own bin width. Pinned
by `test_qc3_degenerate_magnitude_spread_is_flatline_not_a_crash`.

**Wrong assumption #2 — a perfect CW tone is not a valid "clean frame" fixture.**
Following from the above, the test helper now adds small seeded noise
(`tone_frame(f, noise=0.01, seed=…)`), with `pure_tone_frame` kept only where the
degenerate case is the thing under test. This is more realistic anyway — real
acquisitions always carry receiver noise.

**Two test claims corrected to what is actually true** (rather than forcing the code
to fit them):
- *RMS threshold.* Chirps differ only by noise, so the MAD is tiny and the robust z is
  very sensitive: an unperturbed frame already sits at z ≈ 2.6, and a ×1.5 chirp gives
  z ≈ 1350. The "not flagged" case therefore needs **no** perturbation at all.
- *Margin.* Removing the ±1000 Hz margin does **not** flip the verdict for a
  10 300 Hz tone (ratio 0.972 → 0.454, still above 0.30) because Hann leakage keeps
  energy inside the bare gate. The test now asserts what is true and load-bearing: the
  ratio drops by >40% and the mask shrinks 10 → 7 bins. Asserting a flip would have
  been asserting something false.

**Also verified empirically before writing tests** (rather than trusting arithmetic):
mask bins 2..11; in-band ratios 1.0 m→0.9995, 2.5 m→1.0000, 10.3 kHz→0.972,
12 kHz→0.0546, 50 kHz→0.0000. F_BEYOND_MARGIN=12 kHz is one bin past the mask edge and
correctly fails, which makes T-QC10 a tight test rather than a trivial one.

**Tests:** `uv run pytest tests/test_qc.py` → **34 passed**. T-QC7 (companion-frame
independence) and T-QC9 (QC gate ≠ model gate) are the two that carry the leakage
guarantee; T-QC15 covers the `in_band_mask` zero-bin / all-bin guards.

**Next:** step 3 — manifest QC columns, eligibility, fail-closed join.

---

## 2026-07-21 — M2 step 1: QC config margin + field/cross-field validation.
## **Success — and it immediately caught a latent config bug.**

**What was built.** `QCConfig.in_band_margin_hz = 1000.0` (owner decision: the
reference `BandMarginHz` code default, ~1 FFT bin at df = 520834/534 ≈ 975.3 Hz);
`configs/preprocess.yaml` mirror; and real validation for every field M2 consumes,
replacing M1's generic `_build_frozen_section` for the `qc` and `preprocess` sections
(`_build_qc`, `_build_preprocess`, `_build_wst`). Implements MILESTONE_2_PLAN §2.3.

**Values and why.** `histogram_bins` positive int; `flatline_max_bin_fraction` and
`min_frame_fraction` in (0,1]; `min_in_band_energy_ratio` in [0,1];
`rms_robust_z_threshold` > 0; `in_band_margin_hz` >= 0; gates = exactly two finite,
positive, strictly increasing metres, **normalised list -> tuple** (a frozen dataclass
must not carry a mutable value, and provenance should record one type). `bool` is
rejected everywhere it would otherwise pass as an `int`/number.

**Cross-field check, and one branch deliberately NOT written.** `_check_qc_band`
rejects (i) a gate mapping at/above Nyquist and (ii) a margin that widens the band
across the whole represented spectrum (the ratio would be identically 1 — a screen
that can never fire). The plan also listed an "empty band after clamping" check; while
implementing it I proved it **unreachable** — the margin only widens, so
`lo <= f_lo < nyquist <= hi` holds whenever the Nyquist check passes. Dead code is
worse than absent code, so it was dropped and the reasoning recorded in the docstring.
The remaining bin-level guards (>=1 bin of support; not *every* bin) need the
fast-time length and belong in `in_band_mask` (step 2).

**The bug it caught immediately.** With types actually checked, 19 tests failed on
`preprocess.bandwidth_hz must be a number, got str`. Cause: **YAML 1.1 only parses an
exponent as a float when the exponent carries a sign.** `bandwidth_hz: 500.0e6` was
loading as the *string* `"500.0e6"` — and had been since M1. It was invisible because
nothing consumed the value until now; `chirp_time_s: 1024.0e-6` was fine only by
luck of its `-`. Fixed to `500.0e+6` with a warning comment, and pinned by
`test_radar_constants_load_as_floats_not_strings` (asserts parsed *type* and value),
so the canonical config cannot regress into string arithmetic. Had this survived, the
first symptom would have been a bizarre failure inside the QC band mapping.

**Also added.** `beat_band_hz(gate_m, B, Tchirp)` — the FMCW range->beat-frequency
mapping (`HzPerM = 2*(B/Tchirp)/c`) in `config.py`, shared by the cross-validation and
(step 2) the QC mask, so the physics is not duplicated. `SPEED_OF_LIGHT_M_S` is
written out rather than imported from scipy: it is exact by SI definition, and config
validation should not depend on a numerics package importing.

**Tests:** `uv run pytest` → **185 passed, 8 skipped** (was 151/8 at M1 close);
34 new config tests, mostly a parametrised bad-value table covering out-of-range
numbers, wrong types (incl. `bool` and `.inf`), malformed gates, and both cross-field
branches.

**Next:** step 2 — `src/dehyd/qc/screens.py` + `tests/test_qc.py`.

---

## 2026-07-21 — M1 commit: `.gitignore` was silently excluding `src/dehyd/data/`.
## **Bug found and fixed at commit time.**

**What happened.** Staging the milestone-1 commit, the file list was missing **all five
files** of `src/dehyd/data/` — `sessions.py`, `loader_10ghz.py`, `ground_truth.py`,
`manifest.py`, `__init__.py`. Everything else staged normally.

**Cause.** The `.gitignore` inherited from the initial commit contained `data*/`, which
in gitignore syntax is **unanchored** — a pattern without a leading slash matches at
*any* directory depth, so it excluded `src/dehyd/data/` along with the intended raw-data
tree. Confirmed with `git check-ignore -v src/dehyd/data/manifest.py` →
`.gitignore:4:data*/`.

**Fix.** Anchored the rule to the repository root: **`/data*/`**. Both directions
re-verified — `data/10ghz/*.mat` and `data/weight/*.xlsx` are still ignored, and
`src/dehyd/data/*` is now tracked.

**Why this matters and how it was caught.** The local working tree and the full test
suite were completely unaffected — the files existed on disk, so all 159 tests passed
either way. The failure would only have appeared on a **fresh clone or on IBEX**, as a
`dehyd.data` package that imports nothing, with the original machine looking healthy.
It was caught only by *reading the staged file list* before committing rather than
trusting `git add -A`. Lesson recorded: for a commit that introduces a new package
directory, check the staged list against the intended tree, especially when the
repository ignores a directory whose name is a common word.

**Commit:** `f3fbade` — 34 files, 5783 insertions, working tree clean afterwards.

---

## 2026-07-21 — **MILESTONE 1 COMPLETE.** Definition of done met in full.

**D1 — mandatory suite, no private data.** `uv run pytest` → **151 passed, 8 skipped**
on a checkout with no `data/` tree required (the 8 skips are the 7 `realdata` tests plus
T18).
**D1 count check.** `uv run pytest tests/test_no_leakage.py -m "not realdata"` →
**24 passed, 1 skipped, 1 deselected** — T18 is the *only* skipped non-`realdata`
leakage test, verified as a count so a mis-scoped skip cannot hide the suite.
**D2 — real-cohort acceptance.** `uv run pytest --realdata` → **158 passed, 1 skipped**
(T18). Manifest builds and validates on the real 80 files: 8000 frames, 16 subjects,
80 sessions.
**D3 —** both ground-truth cross-checks pass on the real workbook for all 16 subjects.
**D4 —** every fold anywhere originates from `eval/splits.py`; nothing else constructs
splits.
**D5 —** the smoke runs end to end on real data and writes provenance with 80 radar
hashes + workbook hash + git rev.
**D6 —** HISTORY.md has a per-step entry (steps 1–9); SECOND_CHAPTER.md §0.1 written.
**D7 —** `plans/implementation_plan.md` and `plans/MILESTONE_1_PLAN.md` agree (the A1
and A7 amendments were applied during the review rounds).

**Milestone-1 scoreboard.** 8 source modules, 8 test modules, **159 tests**. Two
genuine environment/dependency facts discovered empirically rather than assumed
(kymatio↔scipy `sph_harm`; openpyxl formula caches). Four test-scaffolding bugs found
and fixed by the tests themselves during the build (openpyxl `value=None` no-op;
`rel_path` string ordering vs session ordering; `tests/` not a package; `FoldResult`
missing `train_subjects`). The leakage suite was validated **adversarially** — a
deliberately leaky procedure fails it — so its green state carries evidence, not just
absence of failure.

**Open for M2:** install `h5py` and run the minimal 77 GHz audit; QC screens with frozen
thresholds and per-frame reason codes; add QC/eligibility columns to the manifest;
`configs/ibex.yaml` when the cluster roots are known.

---

## 2026-07-21 — M1 step 9: entry-point stub + end-to-end smoke on real data.
## **Success.**

**What was built.** `experiments/run_regression.py`. Implements MILESTONE_1_PLAN §2.9.

**Smoke result (real data, this machine).**
```
config       : configs/exp_a_regression.yaml
device       : cpu   seed: 20260721
ground truth : 16 subjects, 80 sessions
               Delta m% range -2.02 .. 0.00
manifest     : 8000 frames, 16 subjects, 80 sessions
folds        : 16 outer (16 selectable), 5 inner each
provenance   : results/runs/20260721T094017375792Z_2a26fff2/provenance.json
```
The Δm% range **−2.02 … 0.00** matches the expected ≈0 to ≈−2% from the workbook
inspection, and N_eval = 16 with 15 training subjects and 5 inner folds per outer fold
is exactly the full-cohort case the protocol describes.

**Provenance artifact verified:** 80 radar hashes + the workbook hash, all as logical
`rel_path`s; resolved config; 16 fold records with roles; git commit/branch/dirty;
package versions (with `torch: null`, `h5py: null` positively recorded as absent);
platform; `slurm_job_id: null` locally.

**Decisions.**
- The stub deliberately **stops after the data spine**; modeling lands at M6 on top of
  exactly these folds. It prints what it built rather than pretending to model.
- `sys.path` insertion of `src/` so the script runs without an editable install.
- **`results/runs/` is gitignored.** Every invocation writes a new timestamped run
  directory, so committing them all would be noise; `results/` itself stays available
  for curated, reported artifacts added deliberately. Reversible if the owner prefers
  full provenance history in git.
- **`configs/ibex.yaml` not created.** The overlay *mechanism* is implemented and
  tested (`--config` is repeatable, later files win, and a test asserts an overlay
  replaces the data root), but a committed IBEX file would have to name paths that do
  not exist yet and would fail input-path validation locally. It is written at the
  first IBEX milestone, when the real roots are known.

**Next:** step 10 — journal close-out (SECOND_CHAPTER.md milestone-1 section).

---

## 2026-07-21 — M1 step 8: `tests/test_no_leakage.py` (T1–T19) + reference procedure.
## **Success**, and verified adversarially.

**What was built.** `tests/reference_procedure.py` (the contract `harness.py` must
satisfy) and `tests/test_no_leakage.py` (25 tests covering T1–T19). Implements
MILESTONE_1_PLAN §4.

**The reference procedure.** Deterministic nested select-and-refit over
`StandardScaler → Ridge`, α ∈ {0.1, 1, 10}, folds taken **only** from
`eval/splits.py`. Selection metric is the **subject-balanced** session-level MAE. It
returns an auditable bundle per fold — selected config, full inner score table,
per-inner-fold *and* final fitted parameters, training/val/test predictions, and the
fit audit — so tests verify **roles**, not implementation trust. At M6 the leakage
tests rebind to the real harness and this module is deleted.

**Determinism mechanism (the corrected one).** `Ridge` has **no `n_jobs`**, and BLAS
thread env vars set inside a test arrive too late (NumPy/SciPy are already imported at
collection). So the numeric work runs inside `threadpool_limits(1)` with an explicitly
deterministic `solver="cholesky"` instead of leaving `solver="auto"` free to switch
algorithm. A test **asserts the achieved limit** via `threadpool_info()` rather than
documenting an intention. T10 (two unmutated runs bit-identical) is the precondition
that makes every later bit-for-bit comparison non-vacuous.

**Both CV levels are tested, which is the point.**
- **T11–T15 (outer):** mutating the held-out subject's features/labels/both leaves
  selected α, the inner score table, every fitted parameter (inner *and* final), and
  the training predictions **bit-identical** (`.tobytes()`); power checks confirm
  feature mutation moves the held-out prediction and label-only mutation moves the
  score but not the prediction.
- **T16 (inner):** the outer mutation **cannot** detect fitting on
  `inner_train + inner_val`, because inner-val subjects *are* outer-training subjects.
  So a separate test mutates a training subject and asserts that, for the folds where
  it is **validation**, that fold's fits are bit-identical — while its own validation
  predictions do move. Scope is deliberate and documented: folds where it is
  inner-train legitimately change, as may the selected config.
- **T17:** with equal session counts, subject-balanced and pooled MAE are numerically
  identical, so a pooled implementation would pass unnoticed. Tested against a
  **hand-calculated** value on a deliberately unequal fixture (5/2 sessions → 5.5, not
  25/7), plus an end-to-end run with counts {5,5,4,2,5,3,5,4}.
- **T19:** the audit must map every fitted quantity to the subject set it came from —
  inner fits from exactly that fold's `inner_train`, the final refit from exactly the
  full `outer_train`, no audited set ever containing the test subject — plus a test
  that the audit **covers every fitted quantity** (an audit with silent omissions
  would be worthless).

**T18 skip scope (the subtle one).** Both guards — `pytest.importorskip("torch")` and
the static marker — are **inside the T18 function**. A module-level `importorskip`
would skip T1–T17 and T19 too, letting the file report green with nothing verified.
Verified by the acceptance command:
`uv run pytest tests/test_no_leakage.py -m "not realdata"` → **24 passed, 1 skipped
(T18 only), 1 deselected (R1)** — checked as a *count*, not as "no failures".

**Adversarial verification (does the test actually catch a leak?).** A passing test
proves nothing on its own, so the reference procedure was monkeypatched to fit the
scaler on **train + held-out** — the classic leak — and T13 was re-run. Result: clean
procedure passes, **leaky procedure fails**. The test has teeth.

**Two failures during the step, both mine, both in test scaffolding.**
1. `from .reference_procedure import ...` → `ImportError: attempted relative import
   with no known parent package`. `tests/` is not a package; switched to an absolute
   import (pytest's default import mode puts the test dir on `sys.path`).
2. T19 → `AttributeError: 'FoldResult' has no attribute 'train_subjects'`. The audit
   needs the outer-training set to check against; added `train_subjects` to
   `FoldResult`. This is a genuine improvement to the harness contract, not a
   workaround — the real `harness.py` must expose it too.

**Verification.** `uv run pytest` → 151 passed, 8 skipped.
`uv run pytest --realdata` → 158 passed, 1 skipped (T18).

**Next:** step 9 — `experiments/run_regression.py` stub and the M1 end-to-end smoke.

---

## 2026-07-21 — M1 step 7: provenance recorder. **Success.**

**What was built.** `src/dehyd/provenance.py`, `tests/test_provenance.py` (14 tests).
Implements MILESTONE_1_PLAN §2.8.

**Concrete decisions and why.**
- **`results_dir` is the single output authority.** `record_run(config, manifest,
  folds, extra)` has **no `out_dir` parameter**, so the destination cannot be given two
  ways and disagree. Tests supply a `Config` whose `results_dir` is `tmp_path`, which
  also keeps every test write outside the repo — a test run therefore cannot alter the
  git-dirty flag that a later assertion reads.
- **The ground-truth workbook is hashed alongside the 80 radar files.** Hashing only
  radar data would let the labels change without provenance noticing.
- **Logical identity + physical hash.** Entries are `{rel_path, sha256}`: `rel_path`
  from the manifest (portable across Windows/IBEX), hash computed on the resolved file.
  Tests assert no absolute paths and no `..` segments.
- **Canonical serialization** — radar entries sorted by `rel_path`, folds sorted by
  test subject, subject sets as sorted lists, `json.dumps(sort_keys=True)`. A test
  asserts two runs on unchanged inputs differ **only** in `timestamp_utc`.
- **Windows-safe run directories.** `results_dir/runs/<YYYYMMDDTHHMMSSffffffZ>_<rev>/`
  — no colons (invalid in Windows paths) and microsecond precision so two runs in the
  same second cannot collide; a real ISO-8601 timestamp is kept *inside* the JSON.
  Existing `provenance.json` → raises rather than overwriting (tested by pinning the
  stamp format so two runs collide on purpose).
- Package versions include `torch` and `h5py` as `None` — a positive record that they
  were absent, rather than silence.

**One failure during the step, and what it was.** `test_hash_changes_when_data_changes`
failed with two identical hashes. Cause was in the **test**: it compared
`radar_files[0]`, but that list is sorted by `rel_path`, where
`"subject_1_10am.mat"` sorts **before** `"subject_1_8am.mat"` — so index 0 was not the
file the test had modified. Fixed by looking entries up **by path**, and strengthened
to also assert every untouched file's hash is unchanged. Worth recording because the
same string-vs-session ordering trap will recur wherever `rel_path` order is mistaken
for session order.

**Verification.** `uv run pytest` → 127 passed, 6 skipped.

**Next:** step 8 — `tests/test_no_leakage.py` (T1–T19), the milestone capstone.

---

## 2026-07-21 — M1 step 6: nested-LOSO splitter (the single fold source). **Success.**

**What was built.** `src/dehyd/eval/splits.py`, `tests/test_splits.py` (23 tests).
Implements MILESTONE_1_PLAN §3.

**Concrete decisions and why.**
- **`GroupKFold` over one row per subject, not per frame.** The inner splitter is fed
  a `(n_train_subjects, 1)` array with `groups = subject_ids`, so it is literally
  splitting *subjects*; frame-level selection happens downstream by filtering on the
  returned subject sets. This makes it structurally impossible for the splitter to
  emit a frame-level split, which is the invariant it exists to protect.
- **Adaptive inner count `min(n_inner_max, n_train)`** — 5 folds at the full cohort
  (15 training subjects), 5 at the 6-subject smoke subset, 3 at n_train=3.
- **`min_train_subjects` constrains the outer-training pool** (owner decision 4). At
  the boundary `n_train == 3`, `GroupKFold(3)` fits each inner model on 2 subjects;
  a test asserts exactly this so the accepted consequence is visible in the suite
  rather than buried in prose. Below the floor the fold is returned with
  `selectable=False` and **no** inner folds — reported as non-selectable, never run
  degenerate.
- **Frozen dataclasses** (`OuterFold`, `InnerFold`, `frozenset` members) so a consumer
  cannot mutate a fold in place; tested.
- **No RNG anywhere.** Subjects are sorted on entry and GroupKFold's assignment is
  deterministic, so `nested_loso_splits(x) == nested_loso_splits(x)` and input order is
  irrelevant — both tested (S7).
- **Duplicate subject ids raise.** A subject appearing twice would let one copy train
  while another is held out — the exact failure LOSO exists to prevent.
- `iter_triples()` provides the flat `(inner_train, inner_val, test)` view, reconciling
  the main plan's "(train, val, test)" phrasing with several inner folds per outer fold.

**Verification.** All seven documented invariants S1–S7 are unit tests:
S1 test∉train; S2 inner disjoint and ⊆ outer-train with test in neither; S3 inner val
sets **partition** outer-train (asserted, not assumed from the GroupKFold docs); S4
each subject held out exactly once; S5 non-empty when selectable; S6 adaptive count at
n∈{16,6,4,3}; S7 determinism. Full suite: 113 passed, 6 skipped.

**Next:** step 7 — `provenance.py` + `tests/test_provenance.py`.

---

## 2026-07-21 — M1 step 5: frame manifest + structural gate. **Success.**

**What was built.** `src/dehyd/data/manifest.py`, `tests/test_manifest.py` (17 tests).
Implements MILESTONE_1_PLAN §2.6.

**Concrete decisions and why.**
- **Logical file identity (`rel_path`), not repo-relative.** The manifest stores the
  path **relative to `data_10ghz_dir`** (`subject_1_8am.mat`), resolved against that
  root for I/O and hashing via `resolve_path()`. A repo-relative path would break on
  IBEX, whose data root lives outside the repository — the same file would then carry
  machine-specific `..` segments and a different identity per machine. A test asserts
  no `..`, no leading `/`, no drive letters.
- **Deterministic ordering + fixed dtypes.** Sorted by `(subject, session_idx,
  frame_idx)` with the index reset, and every column dtype asserted. Verified by a test
  that **monkeypatches `Path.glob` to return reversed order** and asserts the two builds
  are frame-for-frame identical — so filesystem enumeration order can never reach
  training order, hashes, or saved artifacts.
- **All six checks fail loudly and name every offender**, not just the first: C1
  completeness, C2 duplicates, C3 unparseable/stray files, C4 bijection **in both
  directions** (file with no workbook row *and* workbook row with no file), C5 per-file
  structure (shape and MATLAB class), C6 actual frame counts. A test with two missing
  cells asserts both are named.
- **Mandatory tests build `GroundTruth` directly in memory** (it is two DataFrames)
  rather than round-tripping a synthetic workbook — sidesteps the openpyxl
  formula-cache limitation from step 4 and keeps these tests about the manifest.
- **Frame counts come from the file.** A synthetic session with counts {3,7,2,5,4}
  confirms per-file `n_frames_in_file` and contiguous `frame_idx` — the M2 eligibility
  rule `ceil(0.5 × actual_frame_count)` depends on this not being a hard-coded 100.
- QC columns (reason codes, eligibility) deliberately **not stubbed**; they arrive at M2.

**Verification.** `uv run pytest` → 90 passed, 6 skipped. `uv run pytest --realdata`
→ 96 passed. On the real data the manifest builds to **8000 rows** (16×5×100), subjects
exactly {1..16}, every session exactly 100 frames, all dtypes as specified.

**Next:** step 6 — `eval/splits.py` + `tests/test_splits.py`.

---

## 2026-07-21 — M1 step 4: ground truth (fixed-cell parse + cross-checks). **Success.**

**What was built.** `src/dehyd/data/ground_truth.py`, `tests/test_ground_truth.py`
(31 tests). Implements MILESTONE_1_PLAN §2.5.

**Module split forced by an openpyxl limitation (verified, not assumed).** openpyxl
writes formulas but never evaluates them, so a synthetic workbook can hold **either**
an `=I-E` formula in column J **or** a cached number — never both. No synthetic fixture
can therefore exercise the full dual-view load. The module is split so each view is
independently testable:
- `_validate_layout(ws_formula)` — headers, column-B subject identity, J formula
  structure. Formula-view fixtures.
- `_read_values(ws_data_only)` — masses, covariates, cached J, K text. Literal-value
  fixtures.
- `check_targets(...)` — pure math + both cross-checks, no I/O. Array-level tests,
  including tolerance boundary behaviour and the sign convention.
- `load_ground_truth()` — the only place the two views meet; exercised on the **real**
  workbook (which genuinely has formulas *and* Excel-written caches) in a `realdata`
  test.

**Concrete decisions and why.**
- **Identity parsed from column B** (`^Subject (\d+)$`), asserted unique and exactly
  {1..16} — not inferred from row position. The owner-confirmed radar↔workbook identity
  is thus *checked*, not assumed.
- **Extra-subject guard scans all of column B**, not just the rows below the block; a
  `Subject 17` planted at row 400 is caught (tested).
- **Covariates validated before BMI** is computed from them (age 15–80, height
  120–220 cm) — a metres-instead-of-cm height is caught before it silently produces a
  BMI of ~24000.
- **Tolerances kept at 0.05 kg / 0.05 pct-points but re-justified from the observed
  workbook**, not from an assumed recording precision: Subject 15 uses 0.05-kg
  increments and column K truncates (Subject 13: 0.5997% → "0.59"), worst observed
  deviation ≈0.0097 pct-points, so 0.05 is ≈5× the worst case. Recorded in the module
  docstring so the number is never mistaken for a claim about the instrument.
- **All problems are reported at once**, not just the first — a test asserts two
  corrupted J cells both appear in the error.

**One failure during the step, and what it was.** `test_missing_weight_detected`
initially did not raise. Cause was in the **test**, not the parser:
`ws.cell(row, col, value=None)` is a **no-op** in openpyxl (None is the sentinel for
"don't set"), so the cell was never blanked. Fixed by assigning `.value = None`
directly, and a `"n/a"` string case was added alongside. The mass check was factored
into `_is_plausible_mass()` while fixing it (also rejects `bool`, which is an `int`
subclass).

**Verification.** `uv run pytest` → 74 passed, 5 skipped. `uv run pytest --realdata`
→ 79 passed. On the real workbook: 80 session rows, 16 subjects, **both cross-checks
pass for all 16**; S0 is identically zero, all S4 deltas negative and > −3%, all BMIs
in 15–45.

**Next:** step 5 — `manifest.py` + `tests/test_manifest.py`.

---

## 2026-07-21 — M1 step 3: sessions + minimal 10 GHz loader. **Success.**

**What was built.** `src/dehyd/data/sessions.py` (the single definition of session
order), `src/dehyd/data/loader_10ghz.py` (filename parse, header inspect, full load),
`tests/test_loader.py` (22 tests). Implements MILESTONE_1_PLAN §2.3–2.4.

**Concrete decisions and why.**
- **Header-only inspection via `scipy.io.whosmat`.** Measured on a real file:
  **0.017 s**, so all 80 files cost ≈1.4 s instead of decompressing ≈1.4 GB. The
  planned fallback (full `loadmat` per file) was therefore **not** needed.
- **`whosmat` returns the MATLAB class**, confirmed `('framesRadar', (534, 20, 100),
  'double')` on the real data — so the class assertion the plan asked for is checkable
  without loading. An `int16` array of the correct shape is rejected (tested).
- **`loadmat(..., variable_names=["framesRadar"])`** so the unused
  `framesRadarIQ` [20834×2×100] is never decompressed. A test writes a file containing
  both and confirms loading succeeds regardless.
- **Frame count is read from the file, never assumed 100** — session eligibility at M2
  is `ceil(0.5 × actual_frame_count)`, so an assumed constant would silently corrupt it.
  Tested with a 42-frame synthetic file.
- **Strict filename regex.** An unparseable name raises rather than being skipped,
  because "unmatched file" is a manifest failure condition, not a benign case. Seven
  malformed-name variants tested, including `.MAT` case and trailing junk.
- Complex-dtype check lives in `load_10ghz_file` (whosmat cannot report complexity);
  a real-valued double cube of the right shape is rejected.

**Verification.** `uv run pytest` → 43 passed, 3 skipped. `uv run pytest --realdata`
→ 46 passed: **all 80 real files** inspect as `(534, 20, 100)` MATLAB-class `double`,
subjects exactly {1..16}, and a full real load is complex128 and all-finite.

**Next:** step 4 — `ground_truth.py` + `tests/test_ground_truth.py`.

---

## 2026-07-21 — M1 step 2: config system. **Success.**

**What was built.** `src/dehyd/config.py` (frozen dataclass schema + `load_config`),
`configs/{data,preprocess,wst,exp_a_regression}.yaml`, `tests/test_config.py`
(21 tests). Implements MILESTONE_1_PLAN §2.2.

**Concrete decisions and why.**
- **Two path rules, deliberately different.** `include:` entries resolve against the
  **declaring YAML's directory** (so `exp_a_regression.yaml` can say `data.yaml` and
  find its sibling); path **values** resolve against the **repo root** (so a data root
  means the same thing from any CWD or declaring file). Both are covered by a test that
  loads from an unrelated CWD via `monkeypatch.chdir` and compares the fully resolved
  configs.
- **Merge = later wins, lists replaced wholesale**, never concatenated — a later config
  states the entire intended value. Tested directly (`seed_set` replacement).
- **`include:` may not nest.** Flat composition keeps the resolution order followable;
  nesting raises.
- **Numeric floors enforced at the config layer, not just documented:** `seed_set` must
  be exactly 5 **distinct** seeds (duplicates would silently reduce effective repeats);
  `n_inner_max >= 2` (GroupKFold); **`min_train_subjects >= 3`** — deliberately stricter
  than GroupKFold's mechanical floor of 2, because the approved protocol requires ≥3
  training subjects before an outer fold is selectable. A permissive floor here would
  let an overlay YAML weaken the nested-CV rule while staying syntactically valid.
- **`wst.tilings` cannot be overridden in YAML** — the three tilings are frozen design
  constants; J and output shape are derived/measured at M4, never hard-coded.
- **`results_dir` is not required to exist** (output, created on demand) while
  `data_10ghz_dir` / `weight_xlsx` are (required inputs). This distinction is what makes
  the mandatory suite runnable on a clean checkout.
- **Mandatory tests never touch the private data:** each appends a final `tmp_path`
  overlay redirecting the input paths, so composition/merge/path-rules/validation are
  all exercised without `data/`. That the *canonical* config resolves to the real
  dataset is a separate `realdata` test.

**Verification.** `uv run pytest` → 21 passed, 1 skipped (the `realdata` test).
`uv run pytest --realdata` → 22 passed. Both gate directions confirmed working.

**Next:** step 3 — `sessions.py` + minimal `loader_10ghz.py` + `tests/test_loader.py`.

---

## 2026-07-21 — M1 step 1: environment + repo skeleton. **Success**, with one real
## dependency conflict found (scipy, not numpy).

**What was tried.** Created the pinned uv environment and package skeleton per
`plans/MILESTONE_1_PLAN.md` §1 step 1 / §2.1: `pyproject.toml` (package `dehyd`, src
layout, `requires-python >=3.11`), `.python-version` = 3.11 (uv fetched CPython
3.11.15), `uv lock` + `uv sync`, `src/dehyd/{data,eval}/` skeleton, `tests/test_env.py`,
`tests/conftest.py` (the `--realdata` gate), `.gitignore` additions.

**The env unknown resolved — but it was not the anticipated one.** The plan flagged a
possible **kymatio vs numpy 2.x** conflict, with the contingency "pin numpy<2". That
conflict does **not** exist: kymatio 0.3.0 imports and runs fine on numpy 2.4.6.
The actual conflict is **kymatio 0.3.0 vs scipy ≥1.17**: `kymatio/scattering3d/
filter_bank.py` imports `scipy.special.sph_harm`, which scipy **removed in 1.17**
(superseded by `sph_harm_y`). Symptom is subtle and would have surfaced at M4, not
here: top-level `import kymatio` **succeeds** (so a naive import smoke passes), but
`from kymatio.numpy import Scattering1D` raises `ImportError` because the 1-D entry
point pulls in the 3-D filter bank.

- **Resolution:** pin **`scipy>=1.11,<1.17`** in `pyproject.toml` with the reason in a
  comment; revisit when kymatio ships a release using `sph_harm_y`.
- **Resolved versions:** python 3.11.15, numpy 2.4.6, scipy 1.16.3, kymatio 0.3.0,
  scikit-learn 1.9.0, pandas 2.3.3, openpyxl 3.1.5, PyYAML 6.0.3, pytest 9.1.1,
  threadpoolctl 3.6.0 (arrives via scikit-learn — needed for the M1 determinism
  fixture, §4 Part C).
- **Verified after the pin:** `Scattering1D(J=7, shape=(470,), Q=(10,4), T=104,
  max_order=2)` instantiates and transforms, output shape `(742, 7)`. kymatio emits
  `UserWarning: Signal support is too small to avoid border effects` for J=7 on 470
  samples — **noted for M4**, where the plan already requires padding/output shape to
  be *measured* from the instantiated filter bank rather than assumed. Not an M1 issue.

**Why the plan's ordering paid off.** §2.1 put env resolution first precisely so an
unknown like this fails before any code depends on it. It did — and it was a different
unknown than predicted, which is the argument for resolving it empirically rather than
assuming the documented risk was the only one.

**Incidental.** `environment.yml` (planning-phase conda export) moved to
`archive/code/` per the file-hygiene rule (owner decision 2) — `uv` is now the sole
local env manager. A stale `.pytest_cache/` at the repo root has an unreadable ACL on
this machine (cannot be read, `takeown`'d, or removed without elevation) and made
pytest warn on every run; worked around by setting `cache_dir = ".cache/pytest"` in
`[tool.pytest.ini_options]` rather than leaving permanent noise in the test output.

**Outcome:** `uv run pytest` green (2 passed, no warnings).
**Next:** step 2 — `configs/data.yaml` + `src/dehyd/config.py` + `tests/test_config.py`.

---

## 2026-07-21 — Planning phase complete; plan approved and hardened. Pre-implementation.

**State:** No implementation code written yet. The design is locked in
`plans/implementation_plan.md` and is the spec milestone 1 builds against.

**What was done.** Read CLAUDE.md/AGENTS.md + ROADMAP.md in full, the paper
(`paper/`), and the MATLAB reference (`matlab/`). Inspected a real 10 GHz file
byte-for-byte (not assumed from the paper) and parsed the weight workbook. Produced the
implementation plan, then hardened it across **7 rounds of independent (Codex) review**
— every comment resolved; reviewer's final verdict was "no further comments,
implementation-ready."

**Verified data facts (not assumed).**
- 10 GHz: `data/10ghz/subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat`, 80 files, MAT v5,
  little-endian, zlib. Var `framesRadar` = MATLAB **double** `[534 fast-time × 20 chirps
  × 100 frames]`, complex; on disk the elements are `miINT16` (space optimization) so
  `scipy.io.loadmat` returns **complex128**. Also `framesRadarIQ` [20834×2×100] (raw IQ,
  unused). One file = one subject/session = 100 frames.
- 77 GHz: MAT **v7.3/HDF5**, ~285 MB each (~23 GB), needs `h5py` (not yet installed —
  milestone-2 audit installs it). h5py-reported shape (reviewer-sampled)
  `(16,256,256,125)=(Nrx,Nchirps,Nfast,Nframes)`; full axis reversal →
  `(Nframes,Nfast,Nchirps,Nrx)`. Fast-time↔chirp (both 256) disambiguated by a raw-data
  signal-domain check, not shape alone.
- Ground truth: `data/weight/metadata_subjects_info.xlsx`, sheet `MetaData`, rows 3–18.
  Two-row merged header → parse by fixed cell addresses. Cols E–I = weights 8am→4pm.
  Signed target `Δm% = (m(s) − m(S0))/m(S0)×100` (negative = loss), ≈0 to ≈−2%.
- Subject identity: radar `subject_N` = workbook "Subject N" (owner-confirmed; old
  MATLAB 5–20 numbering was renumbered to 1–16 for cleanliness, same subjects/order).

**Key locked decisions & why (see plan for full detail).**
- MATLAB is a **design reference only** — Python is the sole source of all reported
  numbers; correctness via Python-native self-consistency checks, not numeric diffs.
- Headline = **fluid-loss (Δm%) regression under LOSO**; 5-class demoted to secondary
  **ordinal**. Analysis unit is **session-level** (aggregate per-frame WST features to
  one vector/session) to kill pseudo-replication; per-frame is diagnostic only.
- Deliberate departures from the reference (logged here as they're implemented):
  robust standardize = median/MAD (not the reference's mean/MAD mix); range gate is a
  config parameter (default 1–2 m); WST log transform = order-aware
  (`log(S+ε)` on orders 1–2, ε=1e-6; order 0 left linear); EdgeTrim=32 **after**
  reduction.
- Scoring counts use **N_eval** (evaluable subjects), never a hard 16; session
  eligibility = `≥ ceil(0.5 × actual_frame_count)` QC-passing frames, no imputation.
- 77 GHz primary feature = **slow-time (Doppler) I/Q WST, per-Rx, feature-space fused**
  (magnitude discards Doppler phase; coherent Rx averaging risks phase cancellation).
- Stats: subject-cluster bootstrap (B=10000), seeds collapsed (metric-type-aware),
  all CIs/p-values labeled **conditional/exploratory**; effect sizes + per-subject
  spread carry interpretation.

**Outcome:** success (planning). **Next:** milestone 1 — repo scaffold, config system,
manifest + nested-LOSO splitter + provenance, and `tests/test_no_leakage.py` green
before any modeling.

