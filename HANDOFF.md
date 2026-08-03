# HANDOFF — resume point for a new chat (M9 step 11 DONE; step 12 BLOCKED on an owner decision)

_Written 2026-08-03. The next chat's job: **get the owner's decision on the O-M9-5 gate failure
below, then continue `plans/MILESTONE_9_PLAN.md` from step 12**. Do not read the whole 1474-line
plan — §1 rows 12-15, §4 D9-D12, §5 traps 10, 11, 17, 18, 20. Anything needing more detail than
fits here is in HISTORY.md's four newest entries: search it, don't guess._

## TL;DR — read this first

- **Step 11 (full-cohort Exp C, both bands) is DONE.** Negative result, cleanly obtained.
- **Step 12 is BLOCKED.** The O-M9-5 bit-identity gate **FAILED on 10 GHz** (77 GHz passed). It was
  investigated to exhaustion over 2026-08-01/02: every controllable cause was eliminated by direct
  test, the effect is characterized and benign (max 5.14e-14, zero effect on model selection), but
  the root cause is NOT fully identified and bit-identity with the M7 artifacts is **not
  achievable**. **An owner decision is required before anything else happens.** Trap 17 forbids
  proceeding, weakening the gate, or re-pointing the comparison — none of which was done.
- **Analysis commit is still `f9dee54e0cef11c92f0d932d33a51710e098bd26`.** IBEX's `REVISION` must
  keep reading exactly this. Local HEAD has moved past it for journal-only commits — see below.
- **Nothing about steps 13-15 has changed.** They are unblocked the moment step 12 resolves.

## The blocking decision (this is the whole handoff)

**What happened.** Step 12's Exp A re-runs completed cleanly at `f9dee54` (jobs 49779546 / 49779550,
cohorts 73 / 72). `load_exp_a_radar` then gave:

    10ghz: *** FAILED *** predictions_10ghz.csv is NOT bit-identical to the M7 artifact
    77ghz: BIT-IDENTICAL OK | n_seeds=5 | subjects=16 | verified=True

10 GHz: 11 of 149 rows differ, `max |Δy_pred| = 5.14e-14`, `max |Δy_true| = 0`.

**What was eliminated, each by direct test (full detail in HISTORY.md):**

| candidate | how it was killed |
|---|---|
| raw data | `provenance.inputs` sha256 identical; `manifest`, `folds`, seeds identical |
| model selection drift | `selection_table_10ghz.csv` **byte-identical** — no tie-break flipped |
| IBEX node hardware | both jobs on `cpu_amd_epyc_9655` (Turin), identical feature strings |
| run-to-run nondeterminism | 3 independent runs at `f9dee54` → one hash |
| package / env drift | only `f9dee54` touched `uv.lock` (torch cu126 + `nvidia-*`); numpy/scipy/kymatio/sklearn untouched |
| the store rebuild | **M7's own code (`f36c4fb2`, throwaway worktree) reproduces the current M9 store bit-exactly** |
| Exp A code path | `exp_a.py`/`splits.py`/`extraction.py` unchanged; harness `score_fn=None` is the identical call; regressors is pure extraction; `SIMPLICITY_RANK` base entries unchanged |
| sbatch resources (8→16 cores) | 8-core control run (job 49848293, `workers : 8` confirmed) → same hash as the 16-core runs |

**The one structural finding.** The divergence correlates *perfectly* with one estimator family:

| selected family | folds | differing |
|---|---|---|
| **svr** | 5 | **5** |
| knn | 7 | 0 |
| gbm | 3 | 0 |
| rf | 1 | 0 |

`SVR` is libsvm's iterative SMO solver with a convergence tolerance — the only family in the grid
that amplifies a sub-ULP difference into visible output. **But 77 GHz has 8 SVR folds and still
matched M7 exactly**, so SVR is the amplifier, not the source. The 10 GHz-specific sub-ULP source
is unidentified. The M7-era store no longer exists and the M7 venv's BLAS build was never recorded,
so it is not reconstructible.

**Known weakness in the evidence:** the session used for the store-reproducibility check
(`s10_10am`) belongs to subject 10, which selected **knn** — insensitive by construction. Re-running
that check on IBEX with M7 code against the **SVR** subjects' sessions (1, 7, 8, 11, 14) is the one
test that could still localize the source. ~15 min. It would not change the recommendation.

**The recommendation put to the owner (NOT yet accepted — do not implement unprompted):** amend
O-M9-5 to a conjunction —
  1. `selection_table_{band}.csv` byte-identical (already true, both bands — this is the real gate:
     any genuine drift changes which model is picked), **and**
  2. `max |Δy_pred| ≤ 1e-10`, with the observed maximum recorded in the run log.

1e-10 is ~4 orders above the observed noise and ~9 below anything meaningful for Δm% of order
0.1-1. **Cost if accepted:** it changes `src/`, so the commit moves → trap 18 → both stores rebuilt
→ steps 11 and 12 re-run (~4 CPU-hours). There is no local-only path: step 13's comparison stage
calls `load_exp_a_radar` on IBEX. The amendment is **post-hoc**, made after seeing a failure, and
carries the same §8 disclosure obligation as the A-M9/O-M9 completions.

## Step 11 results (done, no action needed)

Full-cohort Exp C, ordinal S0-S4, both bands, 16 LOSO folds, `seed_set=[1,2,3,4,5]`, cohorts 73/72.

| band | arm | QWK | adjacent acc. | class-unit MAE |
|---|---|---|---|---|
| 10 GHz | a | −0.213 [−0.365, −0.030] | 0.534 | 1.553 |
| 10 GHz | b | −0.198 [−0.317, −0.075] | 0.534 | 1.644 |
| 77 GHz | a | −0.278 [−0.461, −0.077] | 0.558 | 1.492 |
| 77 GHz | b | +0.025 [−0.281, +0.243] | 0.611 | 1.347 |

**Negative result — write it as one.** Three of four CIs sit entirely below zero, but this is "no
usable ordinal signal", NOT inverse predictive ability: predictions collapse into the middle classes
and then run counter to truth. Adjacent accuracy (0.53-0.61) looks fine precisely because
middle-collapsed predictions land within ±1 of much of the truth — QWK stays the headline, adjacent
accuracy is never quoted alone. Supporting no-signal evidence: arm-a family choice is heterogeneous
across folds (no family dominates); `n_evaluable_inner_folds = 5` everywhere with empty
`viability_reason_counts`; and the O-M9-8/8a QWK-undefinedness instrumentation is a **clean zero**
in both bands (`n_qwk_nan = 0`, `n_single_class_truth_val_folds = 0`) — worth stating in §8 because
it was a contested design point.

**Known cosmetic defect, deliberately NOT fixed** (fixing it costs a store rebuild for one metadata
integer): `metrics_exp_c_*.json`'s top-level `n_seeds` reads `1` for both bands but is arm-b-only —
`exp_c.py:745-750` overwrites `cohort` inside the per-arm loop. Arm a realizes 5 seeds when it picks
rf/gbm. No metric is affected. **§8 must state realized seed count per arm and never quote that
field.**

## Where things are

**Run dirs** (IBEX paths are `results/runs/<stamp>_f9dee54e/`; the owner's local copies sit under
the extra folders `results/runs/step11/` and `results/runs/validate_step11/`, which is NOT where
`record_run` writes — don't let those nested paths get cited):

    20260801T133513697136Z_f9dee54e   Exp C 10 GHz   (step 11)
    20260801T141211973935Z_f9dee54e   Exp C 77 GHz   (step 11)
    20260801T162357722390Z_f9dee54e   Exp A 10 GHz   (step 12, gate FAILED)
    20260801T165841260326Z_f9dee54e   Exp A 77 GHz   (step 12, gate PASSED)
    20260801T190220062895Z_f9dee54e   Exp A 10 GHz   (determinism control A)
    20260801T190223956644Z_f9dee54e   Exp A 10 GHz   (determinism control B)
    20260802T093302331078Z_f9dee54e   Exp A 10 GHz   (8-core control, job 49848293)

M7 references: `results/runs/20260727T111437230187Z_f36c4fb2/` (10 GHz) and
`.../20260727T115046533408Z_f36c4fb2/` (77 GHz). Present both locally and on IBEX.

Hashes worth carrying: M7 10 GHz `4bd21201cb87a62aed32b19e7f5fbb478fd7354a6a2c08040cfad6a377145c57`;
every M9 10 GHz run `453a22ba2e6ac06ea846037dba551587d9c0f36ef58498162fc2dee51a18ef8f`.

**Git state.** Branch `v1_milestone9`. Local HEAD is `d1c531d` (journal-only commit) plus
uncommitted HISTORY/HANDOFF edits from this session. **IBEX's `REVISION` still reads `f9dee54` and
must NOT be re-stamped for journal-only commits** — the source there is byte-identical to `f9dee54`,
and re-stamping would invalidate both v2 stores and force a rebuild for zero code change. Trap 18's
blanket "land a commit → re-stamp" rule needs this carve-out; it explicitly permits journal files to
merge after 9.5. Re-stamp only when actual code ships.

Both v2 stores remain valid at `f9dee54` (10 GHz 73 sessions, 77 GHz 72). Store building is
confirmed bit-reproducible on both IBEX and locally.

## Next steps once the decision lands

- **If the amendment is accepted:** implement the two-part criterion in `load_exp_a_radar`, re-stamp
  `REVISION`, sync to IBEX, rebuild both stores, re-run steps 11 and 12, then proceed to step 13.
- **If not:** the owner names the alternative; do not invent one.
- **Step 13** — Exp D cheap baselines both bands, then the 8 CNN fold-array groups
  (`ARRAY_TIME`: `cnn1d_raw` `03:00:00`, other three `02:00:00`, measured at step 10), then
  comparisons last. `submit_exp_d_cnn.sh` already loads `configs/gpu.yaml` uniformly across
  init/fold/merge — a partial overlay fails `_validate_group_lineage` on `config_hash`.
- **Step 14** — the 16-run exploratory frame split, after every LOSO result exists. Output only
  under `results/exploratory_frame_split/`, `never_report` markers, absent from §8.
- **Step 15** — `SECOND_CHAPTER.md` §8 from the real LOSO results, disclosing every A-M9/O-M9
  completion's true chronology (now including the O-M9-5 amendment if it happens).

`SECOND_CHAPTER.md` §8 is still deliberately empty: CLAUDE.md says write at milestone *completion*,
and M9 is mid-build (step 12 of 15). `AGENTS.md` / `ROADMAP.md` are static reference.

## Hard invariants (unchanged, never violate)

LOSO at subject level for every reported result; fit-on-train-only at both CV levels; no test-set
tuning; primary target continuous Δm%; ordinal metrics only for the 5-class task; folds only from
`splits.py`; tie-breaks only via `eval/selection.py`; numpy backs all reported WST features (GPU
authorized only for the Exp D DL baselines, confirmed working on IBEX); `protocol_freeze_guard`
before every fit/write; `tests/test_no_leakage.py` frozen (`git diff --exit-code` is an acceptance
step). GPU training is never claimed bit-deterministic. The frame split is the one sanctioned
reporting exception: structurally quarantined, never reported, absent from §8.

## Process traps worth carrying forward

- **Stamping `REVISION` breaks `test_provenance.py::test_git_degrades_to_none_without_env`** —
  fixed in that one test, not the shared fixture. If a "no git, no env" test fails right after a
  stamp, check this before assuming a regression.
- **`provenance.platform` records OS/Python/machine but no CPU model or `NodeList`.** Had it
  recorded either, this session's milestone stop would have been a two-minute diagnosis. Fixing it
  is a code change (trap 18) — post-M9 list, not now.
- **Store sidecars record what a store was built FROM (`git`, `spec_hash`, `qc_config_hash`,
  `raw_sha256`) but not what it was built WITH** — no package versions. That gap is exactly why the
  M7 divergence is unreconstructible. Same post-M9 list.
- **Don't wholesale-replace local `results/runs/`** — it is gitignored with no safety net. A
  previous sync deleted both M7 reference dirs (recovered from the Recycle Bin). Pull specific dirs.
