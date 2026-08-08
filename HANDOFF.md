# HANDOFF — resume point for a new chat (STEPS 1–4 DONE; next job: **step 5, Experiment E**)

_Written 2026-08-08. The next chat's job: **implement `plans/MILESTONE_10_PLAN.md` §4.2 step 5 —
Experiment E, the LOSO path-group ablation**, in `src/dehyd/eval/exp_e.py`. Read this file, then the
plan's §2.1 (E's frozen design, in full), §1.2 (feature layout + WST metadata — E is the only
experiment that reconstructs the filter bank), §3's five E artifact rows, §4.2 step 5, §5.1/§5.2,
and §6. Steps 1–4 are complete, tested and committed; do not re-litigate them._

## TL;DR

- **Branch `v1_milestone_10`**, 15 commits since `fee9172` (step 4 is `9d49a7a`; this doc-sync
  commit is HEAD). Working tree clean except untracked `.codex/` (owner's tooling — leave it).
  **Push if `git status` says ahead.**
- **Step 1 done.** Exp-A reference gate; `results/milestone10/reference_exp_a_manifest.json` is
  `reference_grade: authoritative` for both bands and version-controlled. **The fail-closed
  precondition on every later structural edit and store rebuild is satisfied** — do not repeat it.
- **Step 2 done.** Multiplicity reaches models, harness, providers and A/B/C orchestration,
  byte-neutral by default for Experiments A–D.
- **Step 3 done.** The H robustness driver: `eval/robustness.py`, `experiments/run_robustness.py`,
  three IBEX shell artifacts.
- **Step 4 done** (`9d49a7a`). Experiment G: `eval/splits.py::selection_folds`,
  `eval/selection.py::select_alpha`, `eval/exp_g.py`, `experiments/run_fusion.py`,
  `scripts/ibex/run_exp_g.sbatch`, 71 new tests.
- **Suite baseline is now `1376 passed, 5 failed, 16 skipped`** (~20 min). The 5 are the
  pre-existing Windows-only failures in `tests/test_exp_b_ibex_scripts.py` /
  `tests/test_exp_d_ibex_scripts.py`. **A 6th failure is yours.**
- **Step 5 is the next job and is unblocked.**

## NO OPEN DECISIONS — A-M10-11 was accepted on 2026-08-08

**All eleven amendments are accepted and in force** (A-M10-1..6 at plan acceptance, A-M10-7..11
during implementation). A-M10-11, the last one outstanding, was accepted by the owner as implemented:

> Exp G's `fit_audit_g.csv` records the fit chain **behind every reported prediction** (per level
> and band: the staged selection, then that level's tuned-ε / scaler / model refit, plus one
> `fusion_alpha` row per outer fold) and **not** the inner-CV fits inside a staged selection. Scoped
> to that table only — `fusion_base_selection.csv` keeps its full per-candidate enumeration.

Full text in plan §0.2, basis in §8.2, §3's G row already matches, and so does the committed code —
acceptance changed no file under `src/`. **Do not reopen it**, and do not carry it into step 9's
review as a pending item: the step-9 reviewer inherits an empty open-decisions list.

## Step 5 = Exp E. What it must REUSE, never reimplement

| Need | Use |
|---|---|
| Every outer fold | `eval/splits.py::nested_loso_splits` (E has **no** inner CV — the model is fixed, nothing is selected) |
| Residual targets + train-only session means | `exp_b.SessionResidualFeatures` (wraps `StoreBackedFeatures`; emits μ_s via `extra_fits` so it audits like any other fitted quantity). E **reuses Exp B's code, and consumes no Exp B artifact** |
| The S0-excluded spine | `exp_b.build_sessions_b` / `exp_b.evaluable_subjects_b` |
| Residual scoring | `metrics.equal_session_residual_mae` (equal weight per session) |
| Column → path metadata | `features.pooling.session_feature_layout(meta, n_time, n_channels, family="pooled")` → `(frame_aggregate, channel, path_id, segment, statistic)` |
| The filter bank's `xi`/`sigma`/`j` | Rebuild the pinned Kymatio bank from the resolved config and call `features.wst.scattering_shape` — the stores persist `order` **only** |
| Fold-parallel execution | `eval/fold_parallel.py::run_folds_parallel(..., unit=)` |
| CIs (if any are reported) | `eval/metrics.py` — E is descriptive; do **not** invent a p-value |

## Step 5's own requirements (read them in the plan, in full)

- **A-M10-1 is in force:** the frozen standalone 4-fold permutation CV is **replaced** by
  leave-one-path-group-out refit under ordinary outer LOSO. Consequence to notice early:
  `ExpEConfig.n_folds = 4` and `fold_assignment` are now **dead config**. Do not read them, do not
  delete them (M6 sections are frozen records), and say so in a comment.
- **Per selectable outer fold** (§2.1): build `SessionResidualFeatures` on outer-training subjects
  only → fit the full fixed ridge on all retained outer-training rows (its `StandardScaler` fit on
  those rows only) → score the held-out subject with equal-session residual MAE → then, **for each
  canonical `path_id`**, delete that path group's complete column block **BEFORE scaling**, refit a
  fresh scaler + ridge on the **identical** rows, and score the **identical** held-out rows.
- `importance_delta_mae_pct_points = ablated_mae − full_mae`. Positive = the group helped. One value
  is one path × one held-out subject, in residual Δm% points.
- **Fixed model form, both bands, run separately.** 10 GHz: `gate=(1,2) m`, reduction A, magnitude,
  T1, log off. 77 GHz: T1_77, I/Q, mean-Rx fusion, log off. Ridge `alpha=1.0`. This is **not** the
  "best model" from A/B outer results and must never be swapped for one.
- **A path group** = every column sharing one Kymatio canonical `path_id`: both `frame_mean` and
  `frame_median` aggregates, every channel, and all global/half × mean/std columns.
- **Fail-closed metadata gate (§1.2).** Reconstructed `order` must equal the stored `order__{tiling}`
  array for **every consumed session**, and its path count must match the model layout. Any mismatch
  **stops the run** — it is not a warning.
- **Band-aware physics labels (§2.1), and the limits are part of the artifact.** Order 0: no
  frequency/range/level claim (per-signal standardization removed absolute level). 10 GHz order 1:
  `xi_1 * 520834` Hz is a fast-time **beat** frequency → coarse scene range `c·f_b/(2·slope)` with
  `slope = bandwidth_hz / chirp_duration_s`; that is **scene distance, not penetration depth**.
  10 GHz order 2: `xi_2` is an envelope-modulation centre, not a second range. 77 GHz order 1:
  `xi_1 * PRF` is a slow-time Doppler/modulation magnitude — **not range**, and **no signed
  velocity** is reported. 77 GHz order 2: envelope modulation, not range or a second velocity.
  Kymatio 0.3.0 `xi` is cycles/sample so `xi_hz = xi * fs_hz`; `j` is a dyadic subsampling index and
  is **never** converted to Hz.
- **A-M10-6: reporting is outcome-neutral.** State the fixed model's weak predictive context
  (§6–§8) *before* the path table, then report the table as measured. Do **not** pre-label paths
  "null" or "physical", and do not make a desired sign an acceptance criterion.
- **Artifacts** (§3, five rows): `importance_folds_{band}.csv`, `path_metadata_{band}.csv`
  (non-applicable numeric fields **blank, not invented**), `importance_summary_{band}.csv`
  (deterministic sort `scattering_order, path_id`), `ridge_coefficients_{band}.csv`
  (**full fixed model only, never ablation refits**), `metrics_exp_e_{band}.json` +
  `interpretability_map_{band}.png` + `exclusions_e_{band}.csv`. Column lists are in §3.
- Also in scope: `experiments/run_interpretability.py` and `scripts/ibex/run_exp_e.sbatch`. Unlike
  G, E takes **one** `--config` list per band (§6 shows the exact smoke argv) plus `--band`.

## Things to check EARLY in step 5

1. **Sizing — E is the CHEAP one, and that is worth confirming rather than assuming.** There is no
   inner CV and no search: per outer fold it is `1 + n_paths` ridge fits on ~50–60 rows. Get
   `n_paths` for T1 / T1_77 from `scattering_shape` first, then size the sbatch from
   `16 folds × (1 + n_paths) × 2 bands`. Expect minutes, not hours — do **not** clone
   `run_exp_g.sbatch`'s 32-core/24 h header out of habit.
2. **E is the only experiment that rebuilds the Kymatio filter bank.** The stores never persisted
   `xi`/`sigma`/`j`. Get the reconstruction + the `order` equality gate working on a synthetic store
   before writing a single physics label.
3. **The 77 GHz half cannot be smoked locally** — the local 77 GHz store is 1 of 72 sessions. Test it
   on a synthetic store; the real smoke is step 13, on IBEX, after the step-11 rebuild.
4. **Test cost.** E has no search, so its tests do **not** pay the 113-candidate tax that made
   `test_exp_g.py` a 176 s file. There is no excuse for a slow E suite; keep it fast.

## Process traps that are still live

- **Do not rebuild the stores.** Steps 5–10 land as green commits with **no** store rebuild; step 11
  stamps `REVISION` once and rebuilds both bands once. A commit move invalidates both stores
  (`store._check_match`, strict commit equality) — that is why the work is batched this way.
- **The local 10 GHz store is complete but built at `dab8f708`**, which is *not* the analysis commit,
  so `validate_store` refuses it: a local real-data smoke fails on provenance, not mechanism.
- **IBEX** sits on `v1_milestone_9a` at `3f465ab` unless someone moved it; it needs
  `git fetch origin && git checkout v1_milestone_10`. Both former dirty-tree offenders (`dcgm/`,
  `results/exploratory_frame_split/`) are gitignored, so `submit_ibex.sh` will not refuse.
- **Heavy work is always `sbatch`**, never a login-shell run.
- **`tests/test_no_leakage.py` is frozen** — untouched through steps 1–4; keep it that way, and keep
  `git diff --exit-code -- tests/test_no_leakage.py` as an acceptance step.
- **`core.autocrlf=true` here** — a byte-exact reference artifact needs a `-text` entry in
  `.gitattributes`. `scripts/ibex/*` is already `text eol=lf`, so new sbatch files are safe.
- **The two stale IBEX-script test files**: `tests/test_run_robustness.py` and
  `tests/test_run_fusion.py` show the fix those two still lack — pipe the script's **bytes** to
  `bash -n -` with LF forced, never a path. Use that pattern for `run_exp_e.sbatch`'s syntax gate.

## The five amendments raised during implementation

| ID | One line | Status |
|---|---|---|
| **A-M10-7** | Reference Exp-A runs are the `*_3f465abc` pair, not `*_f0a46aa6`. Provenance only; selection tables byte-identical, **no estimand changed**. | accepted |
| **A-M10-8** | Multiplicity is **contiguous row duplication for every family**, not `sample_weight` (not duplication-equivalent for `svr`'s `gamma="scale"` or `rf`'s bootstrap `n_samples`). | accepted |
| **A-M10-9** | Exp C arm (b) keeps the frozen **multiclass** O-M9-7 weights; the per-threshold binary rule would break byte-neutrality at multiplicity one. | accepted |
| **A-M10-10** | H's `robustness_selection.csv` records the **selected** candidate per stage, and `fit_audit_robustness.csv` the **outer-level** fits. Scoped to H. | accepted |
| **A-M10-11** | G's `fit_audit_g.csv` records the fit chain behind every **reported prediction**, not the inner-CV fits. Scoped to G's fit audit; its per-candidate selection table is unchanged. | accepted (2026-08-08) |

A-M10-1..6 were accepted at plan acceptance. §1.3, §2.4, §3, §4.1, §5.5, §6, §8.2 and §9.1 were
revised for all of them, so **the plan is internally consistent — trust its current text over any
memory of it.**

## Hard invariants (unchanged, never violate)

LOSO at subject level for every reported result; fit-on-train-only at every CV level; no test-set
tuning; primary target continuous Δm%; folds only from `eval/splits.py`; tie-breaks only via
`eval/selection.py`; `protocol_freeze_guard` before every fit/write; E/F/G/H are entirely CPU.
Do not report frame-level accuracy as a headline, do not claim causal isolation of hydration from
time of day, do not overclaim clinical readiness, and **do not tune E/F/G/H toward a more favourable
result because A–D came out negative**. Attribution is not causality; fusion is not required to beat
10 GHz.

## Remaining sequence after step 5 (plan §4.2)

6. Exp F · 7. drivers + assembly · 8. independent tests · 9. independent code review ·
10. corrections + mandatory retest · 11. final commit + **one** store rebuild · 12. Exp A/B reruns at
the final commit, then `validate_exp_a_reference.py --compare` against the committed manifest ·
13. local smokes then the full E/F/G + `R=200` IBEX jobs · 14. assembly, and **only then**
`SECOND_CHAPTER.md` §9.

**Step 13 sizing already decided:** Exp G = `run_exp_g.sbatch` at 32 cores / 24 h / 128 G (~192
core-hours, ~6 h wall). Robustness: A and B submit `run_robustness.sbatch` with
`--cpus-per-task=64`; Exp C uses `submit_robustness_sharded.sh` (array + merge) with `ARRAY_TIME`
sized from one measured shard. Both are in §6's launch matrix.

## Chapter state

`SECOND_CHAPTER.md` §0–§8 complete. §9 is still the pre-registration stub, now reconciled to the
plan as amended: it records that **no milestone-10 result exists**, and discloses **A-M10-1..11**
with their true chronology (1–6 at plan acceptance, 7–11 during implementation) and the fact that
none of 7–11 changes an estimand. It is written in full **only after** verified full-cohort M10
artifacts exist, never before.
