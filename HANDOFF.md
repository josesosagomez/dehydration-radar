# HANDOFF — resume point for a new chat (planning milestone 7: the LOSO harness + Exp A)

_Written 2026-07-25, at **milestone 6 close**. M6 (the config-freeze gate) is fully
implemented, committed, and tagged. Purpose: let a fresh Claude Code session **plan
milestone 7 — the LOSO harness + fluid-loss regression (Experiment A)** without re-deriving
context. **M7 is not yet planned** — no `plans/MILESTONE_7_PLAN.md` exists._

## TL;DR

**M6 is DONE.** Branch **`v1_milestone_6`**, head **`357f734`**, annotated tag
**`config-freeze-v1`** ("the complete A–G protocol… before any outer-fold result was
inspected"). Full suite **576 passed / 17 skipped**; `test_no_leakage.py` byte-for-byte
unchanged since M1 (`f3fbade`). Nothing pushed; nothing on `main`. **Next: plan M7** (write
`plans/MILESTONE_7_PLAN.md` in the M2–M6 template style), then implement on a new
`v1_milestone_7` branch. **M7 is the first modelling milestone** — it builds `eval/harness.py`
+ `eval/metrics.py`, runs Exp A on both bands, and makes the torch mutation leg green. This is
where "no outer-fold result inspected before the freeze" starts to be spent: after M7 produces
outer results, later design changes are exploratory by definition (per the `config-freeze-v1`
tag).

## Read first (in this order)

1. `CLAUDE.md` / `AGENTS.md` — hard invariants, code style, journal + file-hygiene rules.
2. `plans/implementation_plan.md` — the source of truth. For M7: **§"LOSO harness, nested-CV
   protocol, and no-leakage guarantee"** (the whole harness contract — outer/inner loops,
   evaluability/N_eval, the staged search, seed handling, final refit, the fit-audit
   artifact, and the `test_no_leakage.py` staging note), **§"Analysis unit — session-level
   primary"**, **§Experiments A**, and **§Statistics** (the subject-cluster CIs Exp A reports).
3. `plans/MILESTONE_6_PLAN.md` — what the freeze committed that M7 now consumes (the search
   spaces, `eval/selection.py`'s tie-break, `protocol_freeze_guard`, the tuned-ε mechanics).
4. `HISTORY.md` — **the newest entries** (the M6 implementation entry + the M6 Step-0 /
   review entries). They carry the concrete M6 API M7 wires to.

## What M7 is (implementation_plan.md §Build order step 7)

**LOSO harness + fluid-loss regression (Exp A), session-level, on BOTH bands.** Build:
- **`src/dehyd/eval/harness.py`** — the fit-on-train-only runner for sklearn **and** torch;
  session-level inference; emits a **fit-audit artifact** per fold (every fitted quantity +
  the subject set it was estimated from). Consumes folds ONLY from `eval/splits.py`.
- **`src/dehyd/eval/metrics.py`** — regression MAE/RMSE/pooled-r, and the subject-cluster
  bootstrap CI machinery Exp A reports (the full Exp H stats live at M10, but Exp A's headline
  CIs are here).
- **`experiments/run_regression.py`** — the Exp A entrypoint (session-level LOSO, both bands,
  vs the session-index-only baseline), regenerable figure + metrics JSON + provenance.

## The M6 machinery M7 must consume (do NOT re-derive)

- **Search spaces** `config.search_10ghz` / `search_77ghz` (band-keyed;
  `SearchSpace{10,77}GHzConfig`) and `config.model_grid` (`ModelGridConfig`, per-family grids
  each ≤ `budget_k = 12`).
- **The staged selection algorithm** (MILESTONE_6_PLAN.md §2.1): **Stage 1** searches the
  feature axes (reduction × channel × tiling × log × gate) with a **fixed ridge anchor**
  (`stage1_anchor_ridge_alpha = 1.0`), fit on inner-training, **scored on inner-validation**;
  **Stage 2** searches model family × grid on the Stage-1 winner. Tie-break is
  **`eval/selection.py`'s `select_candidate`** (lower inner-val MAE → `simplicity_rank`
  ridge<knn<svr<rf<gbm → `feature_dimension` → `inner_fold_variance`; non-finite filtered).
  **Use it — do not re-implement the tie-break inline** (C6-30 caught exactly that drift).
- **`protocol_freeze_guard(config, active=...)`** (`features/protocol_freeze.py`) — the
  harness MUST call it immediately before any model fit or result write, with the `active`
  per-fit protocol record populated **from the same enumeration loop that produced the fit**.
  It composes the 77 GHz feature guard and validates each call-time axis against the whitelist.
- **The tuned-ε log branch** (`on_tuned_eps`): compute the fold-local `ε_o = k·scale_o`
  **train-only** (`k = 0.1`; `scale_o` = median-over-training-subjects of the per-subject mean
  of `extraction._prelog_scale` over that subject's eligible training sessions; non-finite/
  non-positive → fall back to `log_epsilon = 1e-6`). M5 ships the *application* path
  (`apply_order_log_77`'s `epsilon_by_order`); M7 computes ε. **This is the one genuinely
  fitted WST quantity — it must be train-only at every CV level.**
- **Seed handling** (implementation_plan.md §"Inner loop"): 5 seeds; inner metric = mean over
  seeds; outer = each seed scored separately, report mean ± sd, **never ensembled**.

## The no-leakage rebind — the subtle M7 trap (read carefully)

`tests/test_no_leakage.py` is **byte-for-byte frozen since M1** and imports its procedure
from `tests/reference_procedure.py` (`run_nested_loso`, `fit_audit`, `Dataset`,
`subject_balanced_mae`, `ALPHA_GRID`). At M1–M6 `reference_procedure.py` IS the procedure
under test (a sklearn contract). **At M7 the leakage tests must rebind to the real
`harness.py` WITHOUT editing the frozen test** — i.e. `reference_procedure.py` is rewritten
to re-export/delegate to `harness.py`, so `harness.py` must satisfy that exact public
contract. **Caveat:** both `test_no_leakage.py` and `reference_procedure.py` say "**M6**" for
this rebind and for the torch leg — those comments predate the A-M5-2 renumber and mean the
**current milestone 7**; the frozen test cannot be edited to fix the stale number, so treat
"M6" there as "M7". The **torch fit path** enters `harness.py` at M7 and makes **T18** (the
torch mutation leg, currently the only skipped mandatory test) green — required **before any
torch result is reported**. The bit-for-bit mutation property must hold on a deterministic
single-threaded CPU fixture at **both** CV levels.

## Hard invariants (never violate — a failing check stops the build)

- **LOSO** at subject level; no frame of a held-out subject in training, any session.
- **Fit-on-train-only** (sklearn Pipeline AND torch: normalization/class-weights/sampler-
  weights/early-stop all from inner-train, monitored on inner-val); **no test-set tuning**.
- **Primary target continuous** (Δm%); session-level headline; per-frame numbers never carry
  frame-IID CIs.
- **`tests/test_no_leakage.py` stays byte-for-byte unmodified since M1** — the rebind happens
  in `reference_procedure.py`/`harness.py`, never in the test.
- **numpy backs ALL reported features**; torch WST frontend only after the cross-backend test.

## Fixed at M6, NOT reopened at M7 (would need a prior authoritative amendment)

- The order-2 log branch is **unconditional** for both bands (A-M6-1) — no pre-check.
- 77 GHz Exp D baselines + the DC-vs-any-motion Doppler physics baseline (A-M6-2) — used at
  **M9** (Exp D), not M7, but frozen now.
- Exp C's family (b) = **Frank-Hall decomposition** over sklearn LogisticRegression (A-M6-5;
  `statsmodels.OrderedModel` verified to lack `sample_weight`) — used at **M9**.
- Budget-parity rule, every §3 constant (budget_k=12, k=0.1, the grids, DL hyperparameters).

## Cohort state (what modelling rests on)

- **10 GHz**: 73/80 sessions eligible, all 16 subjects; `results/wst/wst_diagnostics_10ghz.csv`.
- **77 GHz**: 72/80 sessions eligible, **8966 analysis frames**, all 16 subjects;
  `results/wst/wst_diagnostics_77ghz.csv`. Zero flatline cohort-wide.
- **Exp G matched population = 65 sessions**, all 16 subjects (both bands eligible) — for M10.
- The **77 GHz feature vectors themselves are not yet extracted** to a persistent store — only
  the diagnostics CSV exists. Exp A on 77 GHz needs the session-level features; deciding
  extract-on-the-fly vs a persistent cache (and where it runs — CPU/IBEX) is an M7 design item.

## Environment / IBEX (working, established)

- Env: `uv sync --frozen` (scipy pinned `<1.17`; run pytest via `uv run python -m pytest` —
  `uv run pytest` hit a transient trampoline error this session). IBEX = CPU batch jobs;
  `configs/ibex.yaml` is a paths-only overlay; the owner runs IBEX (no ssh from Claude).
- **Known gap to fix BEFORE M7 re-extracts on the cluster:** `git.commit` records as `None` in
  provenance on the compute nodes (the `safe.directory` fix didn't take). Resolve in the sbatch
  env so M7's runs self-attest their revision.

## Journal & hygiene

- **HISTORY.md** newest-first, an entry per resolved step (failures kept). **SECOND_CHAPTER.md**
  — §0–§4 (through 77 GHz front-end) + **§5 "The config freeze"** written; §6 = Exp A, filled
  at M7. **HANDOFF.md** — update only when asked. Superseded material → `archive/`.
- Branches `v1_milestone_1..6` local; **`v1_milestone_6` current at `357f734`, tag
  `config-freeze-v1`**; nothing pushed, nothing on `main`. Commit only when the owner asks.
  Start M7 on a new `v1_milestone_7` branch.
