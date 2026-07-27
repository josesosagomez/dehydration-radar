# HANDOFF — resume point for a new chat (Milestone 7: LOSO harness + Exp A — code done, full run pending)

_Written 2026-07-26. **M7 is code-complete and fully committed**; the only remaining work is the
owner-gated **full-cohort Exp A run on IBEX** (which spends the config freeze), then reading the
results and writing SECOND_CHAPTER §6. **The config freeze is INTACT — no outer-fold result has
been inspected.**_

## TL;DR

Branch **`v1_milestone_7`**, head **`12610af`** (descends from `config-freeze-v1` = `357f734`).
Nothing pushed; nothing on `main`. M7 built the fit-on-train-only nested-LOSO harness + Experiment
A (session-level Δm% regression, both bands, vs the session-index-only baseline). **Full test
suite green** (682 passed / 16 skipped at code-complete; additive tests since — REVISION fallback,
77 GHz entrypoint config, fold-parallel equivalence — each re-run green; a fresh full run is
advisable but nothing indicates breakage). **T18 is green**; `tests/test_no_leakage.py` changed
only in the one pre-registered T18 hunk (byte-identical elsewhere). Both **mechanism-only smokes
passed on IBEX** (the owner checkpoint), so the mechanism is proven end-to-end on real data with
no performance value surfaced. **Next: the owner runs `MODE=full` on IBEX** (now fold-parallel),
then send the two `metrics_exp_a_{band}.json` back to sanity-check and fill SECOND_CHAPTER §6.

## Read first (in this order)

1. `CLAUDE.md` / `AGENTS.md` — hard invariants (LOSO subject-level; fit-on-train-only; no
   test-set tuning; continuous primary target; frozen `test_no_leakage.py`), code style, journals.
2. `HISTORY.md` — **the newest entries** (post-checkpoint parallelism + IBEX fixes; the checkpoint;
   the M7 implementation entry). They carry the concrete why.
3. `plans/MILESTONE_7_PLAN.md` — the reviewed plan (Codex⇄Claude loop closed); its §2 per-file
   specs + the Step-0b owner decisions O1–O3 are what the code implements.

## What M7 built (all in `src/dehyd/` unless noted)

- **`eval/harness.py`** — the ONE generic nested-LOSO engine (sklearn). Folds only from
  `eval/splits.py`; tie-break only via `eval/selection.py::select_candidate` (inner_fold_variance
  = `np.std(ddof=0)`, owner O1); fail-closed `active`-completeness guard hook; pre-fit
  fold-viability (unexpected errors propagate); per-seed outer outcomes; fit-audit incl.
  `tuned_epsilon`; `tuned_epsilons(...)` train-only. Dataclasses `Dataset/FitRecord/InnerResult/
  FoldResult`.
- **`eval/metrics.py`** — `subject_balanced_mae` (M1-compatible, T17's 5.5 pin); own BCa
  subject-cluster bootstrap (B=10000, percentile fallback, skip/unreliable bookkeeping); Wilcoxon.
- **`eval/exp_a.py`** — the Exp A composition: `stage1/stage2_candidates`, `StoreBackedFeatures`,
  `_run_single_fold` (picklable worker), `run_exp_a(config, band, sessions, store_dir, *, seeds,
  session_index, n_workers)` (serial or `multiprocessing` spawn Pool — **bit-identical**),
  `build_sessions`, `run_and_report` (validate → run → mechanism-only smoke vs full reporting),
  `summarize_exp_a`, `write_exp_a_reports` (Agg scatter).
- **`models/regressors.py`** (5 families + grids ≤ budget_k + per-family auditable fitted state,
  incl. SVR support vectors + rf/gbm ensemble digest binding init_/lr), **`models/baselines.py`**
  (session-index-only; O2 global-mean fallback; O3 config-level guard), **`models/torch_fit.py`**
  (TinyMLP, true early stopping; the T18 target).
- **`features/store.py`** + **`experiments/extract_features.py`** — per-session `.npz` store +
  fingerprint (binds QC frame membership + build commit); fail-closed `validate_store` (incl.
  store/analysis commit-match, C16); both producers refuse a dirty tree. `keep_raw` on both
  extractions; `wst.apply_order_log(epsilon_by_order=…)`; `pooling.pool_stats_batch`.
- **`tests/reference_procedure.py`** — rewritten as a thin adapter over `harness.py` (the frozen
  leakage suite now exercises the real engine). **T18 activated** in `tests/test_no_leakage.py`.
- **`experiments/run_regression.py`** — the Exp A entrypoint (`--band`, `--subset 6subjects` XOR
  `--full-cohort`; mechanism-only smoke; reads `SLURM_CPUS_PER_TASK` → fold parallelism).
- **`scripts/ibex/`** — `extract10.sbatch` / `extract77.sbatch` (store job arrays),
  `run_exp_a.sbatch` (single CPU job, **defaults 16 cores / 64 G**, `BAND`/`MODE` env),
  `submit_ibex.sh` (generic submit; captures clean commit, refuses dirty), `submit_extract77.sh`.
- **`provenance.py`** — `DEHYD_GIT_COMMIT/_BRANCH/_DIRTY` env fallback + a **`REVISION`-file**
  fallback (for copied non-git IBEX trees). `matplotlib` pinned (scipy stays <1.17).

## The owner decisions folded in (Step 0b)

- **O1**: inner-fold-variance = population std `np.std(ddof=0)` (A-M7-2).
- **O2**: baseline absent-time-index → global training-fold mean.
- **O3**: K=1 baseline guarded at the config level (`protocol_freeze_guard(config)`), not a per-fit
  WST `active` record.
- **A-M7-1**: T18 activation is the one sanctioned edit to the frozen `test_no_leakage.py`.

## IBEX run state + the workflow

- **Stores:** built on IBEX for the smoke (6 subjects). **The FULL run needs the complete store**
  (73 sessions 10 GHz / 72 sessions 77 GHz) — confirm/build with `extract10/77.sbatch` before
  `MODE=full`.
- **Smokes:** both bands PASSED (mechanism-only; `run_log_{band}.json`, no metrics). Checkpoint met.
- **Full runs:** first attempt TIMED OUT at 4 h (serial ~8-16 h) → **fold parallelism added**;
  16-core parallel run is ~1 fold's wall-time. **Not yet completed** — this is the pending step.
- **Copied-tree gotcha:** the user copies folders to IBEX (private repo, no `git pull`), so:
  create `REVISION` locally (`git rev-parse HEAD > REVISION`), copy it along, and run `.sbatch`
  **directly** (not `submit_ibex.sh`). Or set `DEHYD_GIT_COMMIT` in the env. The `.venv` must have
  **matplotlib** (the smokes didn't need it; the full run's scatter does).
- **Commit-match friction (important):** `validate_store` requires the store's recorded commit ==
  the run's. **Any code change moves the commit → the store must be rebuilt at the new commit**
  (fast — parallel array). Sbatch-file-only changes do NOT affect this (commit comes from
  REVISION/env, not the sbatch). This bit repeatedly; consider it before making code changes.

**Run the full cohort (both bands):**
```
BAND=10ghz MODE=full sbatch --export=ALL scripts/ibex/run_exp_a.sbatch
BAND=77ghz MODE=full sbatch --export=ALL scripts/ibex/run_exp_a.sbatch
```
Each writes `metrics_exp_a_{band}.json`, `predictions_{band}.csv`, `selection_table_{band}.csv`,
`scatter_{band}.png` into `results/runs/<stamp>_<commit>/`. **This spends the freeze.**

## Next step / open items

1. **Owner runs `MODE=full` on IBEX** (both bands, parallel). Confirm complete stores first.
2. **Read the results** — headline is radar session-balanced MAE vs the session-index baseline
   (Wilcoxon + CI), selection-frequency table, per-subject r. Sanity-check they're sane.
3. **Write SECOND_CHAPTER §6 "Results"** — the method/provenance is already written there; only
   the numbers are pending.
4. Then M7 DoD D10–D13 are met and the milestone closes. **Exp B (M8)** reuses this exact harness.
5. **Optional efficiency** (before M8's many reruns): cache eligibility to skip the run-startup
   full-cohort QC (like M5's survival CSV). Flagged, not done.

## Hard invariants (never violate — a failing check stops the build)

LOSO at subject level; fit-on-train-only at both CV levels; no test-set tuning; primary target
continuous Δm%, session-level headline; folds only from `splits.py`; tie-break only via
`select_candidate`; numpy backs all reported features; `protocol_freeze_guard` before every
fit/write; `tests/test_no_leakage.py` unchanged except the one T18 hunk.

## Journal & hygiene

**HISTORY.md** newest-first (post-checkpoint entry current). **SECOND_CHAPTER.md** §0–§5 written;
**§6 method+provenance written, Results pending the full run**. **MILESTONE_7_PLAN.md** reviewed +
closed. Branches `v1_milestone_1..7` local; `v1_milestone_7` @ `12610af`; nothing pushed, nothing
on `main`. Commit only when the owner asks. Superseded material → `archive/`.
