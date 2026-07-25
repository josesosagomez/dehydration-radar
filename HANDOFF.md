# HANDOFF — resume point for a new chat (planning milestone 6: the config-freeze gate)

_Written 2026-07-25, at **milestone 5 close**. M5 (the 77 GHz front-end) is fully implemented,
its cohort QC + WST runs completed on IBEX, and the results are valid and committed. Purpose:
let a fresh Claude Code session **plan milestone 6 — the config-freeze gate** without
re-deriving context. **M6 is not yet planned** — no `plans/MILESTONE_6_PLAN.md` exists._

## TL;DR

**M5 is DONE.** Branch **`v1_milestone_5`**, head **`4c54e25`** (pushed). Mandatory suite
**526 passed / 17 skipped**; `--realdata` **537 passed / 1 skipped** (only T18, the torch
mutation leg, still deferred to M7). `test_no_leakage.py` byte-for-byte unmodified since M1.
Nothing merged to `main`. **Next: plan M6** (write `plans/MILESTONE_6_PLAN.md` in the M2–M5
template style), then implement on a new `v1_milestone_6` branch. M6 is a **design/freeze
milestone** — mostly configs, whitelist validators, and one pre-check — **no modelling, no
heavy compute.**

## Read first (in this order)

1. `CLAUDE.md` / `AGENTS.md` — hard invariants, code style, journal + file-hygiene rules.
2. `plans/implementation_plan.md` — the source of truth. For M6: **§Build order step 6**
   (what the freeze covers), §"LOSO harness"/"Search space" (the Exp A search space + the
   `on+tuned-ε` branch + the order-2-usefulness pre-check ~lines 480–590), §"Frozen 77 GHz
   pipeline" (~line 929, now carrying A-M5-6), and the baseline/stats/Exp B/C/E/F specs.
3. `HISTORY.md` — **the four newest entries** (M5 close, the cohort QC finding, the two WST
   bug fixes). They carry the specific numbers M6 must weigh.
4. `plans/MILESTONE_5_PLAN.md` §6/§7 — the "carries to M6" list, authoritative.

## What M6 is (from implementation_plan.md §Build order step 6)

**The config-freeze gate: commit the COMPLETE A–G protocol design for BOTH bands to versioned
`configs/` + git BEFORE any outer-fold results are inspected.** Rationale: B–G reuse the same
16 subjects, so any protocol choice made after seeing Exp A's outer results is indirectly
informed by later "test" subjects. The freeze makes "no config tuned on outer-test" real at the
cohort level, not just for Exp A. It covers: the Exp A/WST **search space + staged-search
order**; baseline specs (raw-beat + matched-preprocessing CNN/spectrogram, physics bands);
per-family budget K + seed set; the full statistical protocol; **Exp B** (residualization +
equal-session objective), **Exp C** (ordinal family/objective/sign/fold-viability), **Exp E**
(importance protocol), **Exp F** (single learner + covariate/collinearity rule), **Exp G**
(77 GHz QC thresholds, eligibility, WST tilings, input domain, fusion); and the **frozen
protocol-constant whitelist** that modelling entrypoints validate at their frozen values.
Anything decided after outer results appear is **explicitly labeled exploratory**.

## Specific decisions M6 must make (surfaced by M1–M5 — the real agenda)

1. **The `on+tuned-ε` third log branch — confirm or drop, via the order-2-usefulness
   pre-check.** The log axis is `{off / on+frozen-ε(1e-6) / on+tuned-ε}`; the third is a
   pre-registered candidate that enters the search space **only if** order 2 adds predictive
   value (compare order-{0,1} vs order-{0,1,2} features). If not, drop it so no dead option
   widens the N=16 search. Evidence in hand: **both bands show ~1.8× across-subject order-2
   pre-log scale spread** (10 GHz at M4; 77 GHz at M5 — medians 1.80e-4/1.97e-4/6.22e-4 by
   tiling in `results/wst/wst_diagnostics_77ghz.csv`), and fold-to-fold the median is stable
   to <1%. Stability makes a fold-local ε cheap/safe; it does NOT prove order 2 helps — that
   is exactly what the pre-check tests. The M5 code **ships the tuned-ε application path**
   (`epsilon_by_order`) but never computes ε; M7's harness computes it train-only.
2. **The 77 GHz in-band threshold (0.30) — keep frozen, or declare data-adaptive?** The M5
   cohort QC showed it sits at **percentile 9.6 of a UNIMODAL in-band distribution** (p1 0.254,
   p10 0.304, median 0.364), slicing the lower tail with **no bimodal gap** — so 0.28 vs 0.32
   gives a materially different population (it currently drops 8/80 sessions). Per the plan's
   "QC screens & thresholds" doctrine, a genuinely data-adaptive threshold **moves inside the
   inner CV loop** (fit per training fold). M6 decides: keep 0.30 frozen (the current state),
   or move it inside CV. **Do NOT re-tune it to a fixed new value after seeing survival — that
   is the cohort-level leakage the plan forbids.** The confound check was negative (in-band
   ratio is flat across sessions → not hydration-correlated), which supports keeping it simple.
3. **Confirm the A-M4-7 third-log-branch machinery is consistent across both bands** and the
   10 GHz search space (reduction {A,B} × channel {mag,iq} × tiling × log × gate × model × grid)
   is frozen with its staged order and per-family budget K.

## Fixed at M5, NOT reopened at M6 (would need a prior authoritative amendment)

- 77 GHz **Rx fusion = mean primary/frozen; median a labeled secondary variant.**
- 77 GHz **feature family = pooled classical; flat is diagnostic/DL-only.**
- The 77 GHz frozen protocol constants incl. **`qc77.flatline_skip_leading_bins = 1`** (the
  exclude-range-bin-0 flatline correction, A-M5-6), tilings (Q, invariance_ms), max_order=2,
  log_epsilon=1e-6, the 2–4 m gate, `min_in_band_energy_ratio` structure. M6 *whitelists* these;
  it does not redesign them.

## Hard invariants (never violate — a failing check stops the build)

- **LOSO** at subject level; no frame of a held-out subject in training, any session.
- **Fit-on-train-only**; **no test-set tuning** (incl. QC thresholds — see decision 2).
- **Primary target continuous** (Δm%); 5-class secondary, ordinal metrics.
- **`tests/test_no_leakage.py` byte-for-byte unmodified since M1.** M6 is config/whitelist +
  a pre-check; if it touches the harness at all, this file stays untouched and green.
- **numpy kymatio backs ALL reported features**; torch only after the cross-backend test.

## Cohort state (what the freeze rests on)

- **77 GHz**: 72/80 sessions eligible, **8966 analysis frames**, all 16 subjects evaluable.
  Artifacts: `results/qc/qc_survival_77ghz.csv` (+ axis certs), `qc_frames_77ghz.csv`,
  **`results/wst/wst_diagnostics_77ghz.csv`** (72 cells × 6, committed). Zero flatline
  cohort-wide; all 964 QC failures are the in-band screen.
- **10 GHz**: 73/80 sessions eligible (M2/M4 baseline); `wst_diagnostics_10ghz.csv` committed.
- **Exp G matched population = 65 sessions**, all 16 subjects (eligible in both bands).

## Environment / IBEX (working, established)

- Env: `uv sync --frozen` (scipy pinned `<1.17`; torch CPU-vs-CUDA differs by platform but
  numpy backs all reported features). IBEX = **CPU** batch jobs; `configs/ibex.yaml` is a
  paths-only overlay. The owner runs IBEX (no ssh from Claude); `scripts/ibex/README.md` is
  self-contained. sbatch scripts prefer `.venv/bin/python`, add uv to PATH, and preflight the
  environment. **Known gap:** `git.commit` records as `None` in provenance on the compute nodes
  (`safe.directory` fix didn't take) — resolve before M7 re-extracts, likely in the sbatch env.

## Lesson from M5 to carry into M7 (harness wraps this same code)

The library functions were solid, but the **CLI/run-script wiring was under-tested** — M5's
cohort runs surfaced three bugs in the run scripts (eligible-frames selection; a silently
self-consistent stale-shard merge; a shadowed cell-count check), none in the library. The
merge path now has end-to-end tests (`tests/test_run_wst77_merge.py`). When M7's harness wraps
`extraction_77`/`extraction`, **test the wiring to completion**, not just the primitives.

## Journal & hygiene

- **HISTORY.md** newest-first, an entry per resolved step (failures kept). **SECOND_CHAPTER.md**
  — §1–3 (10 GHz) + §4 (77 GHz front-end) written; M6 adds nothing until it produces the freeze.
  **HANDOFF.md** — update only when asked. Superseded material → `archive/` (M5 retired the
  pre-fix WST shards to `archive/results/m5_wst77_prefilter_shards/`).
- Branches `v1_milestone_1..5` pushed; **`v1_milestone_5` current at `4c54e25`**; nothing on
  `main`. Commit only when the owner asks. Start M6 on a new `v1_milestone_6` branch.
