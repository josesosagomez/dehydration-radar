# HANDOFF — resume point for a new chat (STEPS 1–3 DONE; next job: **step 4, Experiment G**)

_Written 2026-08-08. The next chat's job: **implement `plans/MILESTONE_10_PLAN.md` §4.2 step 4 —
Experiment G, the matched-session decision fusion**, in `src/dehyd/eval/exp_g.py`. Read this file,
then the plan's §2.3 (G's frozen design, in full), §1.3's G-relevant contracts, §3's nine G artifact
rows, §4.2 step 4, §5.1/§5.4, and §6. Steps 1–3 are complete, tested and committed; do not
re-litigate them._

## TL;DR

- **Branch `v1_milestone_10`**, HEAD `51ade69`, pushed to `origin`. 12 commits since `fee9172`.
  Working tree clean except untracked `.codex/` (owner's tooling — leave it).
- **Step 1 done.** Exp-A reference gate built; `results/milestone10/reference_exp_a_manifest.json`
  is `reference_grade: authoritative` for both bands and version-controlled. **The fail-closed
  precondition on every later structural edit and store rebuild is satisfied** — do not repeat it.
- **Step 2 done.** Multiplicity reaches models, harness, providers and A/B/C orchestration,
  byte-neutral by default for Experiments A–D.
- **Step 3 done** (`51ade69`). The H robustness driver: `eval/robustness.py`,
  `experiments/run_robustness.py`, three IBEX shell artifacts, 77 tests.
- **Four amendments are in force and accepted: A-M10-7, -8, -9, -10** (plan §0.2). They are ordinary
  parts of the design now. §1.3, §2.4, §3, §4.1, §5.5, §6, §8.2 and §9.1 were revised to match, so
  **the plan is internally consistent — trust its current text over any memory of it**.
- **Step 4 is the next job and is unblocked.**

## Step 4 = Exp G. What it must REUSE, never reimplement

| Need | Use |
|---|---|
| Every fold index, at every level | `eval/splits.py`. **Step 4's FIRST task**: expose public `selection_folds(subject_ids, n_inner_max=5)` — validates unique sorted subjects, requires ≥ 2, returns the same deterministic `InnerFold`s private `_inner_folds` makes — and make `nested_loso_splits` call it. `exp_g.py` constructs **no** indices. |
| Exp-A staged selection at every level | `exp_a.stage1_candidates` / `stage2_candidates` / `StoreBackedFeatures`, driven through `harness._score_candidates_on_fold` + `select_stage_winner` + `_final_refit` |
| The alpha grid argmin | New small `select_alpha` in `eval/selection.py` (§4.1) — closest-to-1.0 tie-break. Never inline a tie-break; that module is the single source. |
| The 21-point grid + tie-break rule | `config.exp_g` (`ExpGConfig`, M6-frozen): `alpha_grid` = 0.00…1.00 step 0.05, `alpha_tie_break="closest_to_one"`, `seed_pairing=True`, `objective="subject_balanced_oof_mae"` |
| CIs / mean-difference | `eval/metrics.py::mean_difference_ci` — already complete |
| Fold-parallel execution | `eval/fold_parallel.py::run_folds_parallel(..., unit=)` |

## Step 4's own requirements (read them in the plan, in full)

- **Matched population** (§2.3): build both band spines independently, inner-join on unique
  `(subject, session_idx)`. Fail on duplicates, unequal `delta_m_pct`, inconsistent session names,
  non-finite targets. Every model — 10-only, 77-only, equal-weight, learned — trains and scores on
  this exact cell set. No frame-to-frame alignment is attempted.
- **Selection-honest meta-training** (**A-M10-3**): for outer fold `s`, each attached
  `inner_folds[i].val_subjects` is one meta-validation group `V`. For each `V` and band, call
  `selection_folds(sorted(T_s - V))` and run the **complete** Exp-A staged selection over those
  further folds, refit on `T_s \ V`, predict `V` per seed. **Never** reuse
  `InnerResult.val_predictions` — it keeps only the first seed and its rows carry no session keys.
- **Five-seed labels, not five observations**: a deterministic winning family is evaluated once and
  copied to the five configured seed labels with `deterministic_source_seed` recorded.
- **Fail-closed, whole-fold**: `selection_folds` refuses < 2 selection-training subjects; if no
  candidate survives any required further fold, the **entire** outer fold is non-evaluable for
  learned fusion. No partial meta-validation coverage is ever used.
- **Alpha**: subject-balanced OOF MAE across `T_s` per alpha per seed; select from the mean over the
  five paired seed labels; ties → closest to 1.0. Record the whole grid.
- **Primary estimand**: `d_s = mean_over_seed(MAE_s(fused) − MAE_s(10_only))`, headline
  `mean_over_subject(d_s)` with `mean_difference_ci`. Negative favours fusion. 77-only and
  equal-weight are descriptive secondary, **no** extra p-value family.
- **A-M10-4**: the constrained decision-level combiner ONLY. The feature-level variant is deferred
  and is not a completion criterion. Do not invent one.
- **Artifacts** (§3, nine of them): `matched_population.csv`, `unmatched_population.csv`,
  `fusion_meta_oof.csv`, `fusion_base_selection.csv`, `fit_audit_g.csv`, `fusion_alpha_grid.csv`,
  `predictions_g.csv`, `per_subject_g.csv`, `metrics_exp_g.json` + `fusion_comparison.png` +
  `exclusions_g.csv`. Column lists are in §3. **Note G's `fusion_base_selection.csv` genuinely IS
  per-candidate** — unlike H's (A-M10-10) — because §5.4 needs the losing candidates' scores to
  prove outer outcomes are never read. Plan §8.2 explains the distinction; keep it.
- Also in scope: `experiments/run_fusion.py` and `scripts/ibex/run_exp_g.sbatch`. `run_fusion.py`
  loads the two band configs **separately** (`--config-10` / `--config-77`), applies
  `--shared-config` overlays to both, and asserts shared run seeds, target definition, split
  constants and weight workbook. It never merges two top-level configs through the loader.

## Things to check EARLY in step 4 (learned the hard way in step 3)

1. **Sizing.** G runs a complete Exp-A staged selection at *three* levels: per (outer fold × meta
   fold × band), plus per (outer fold × band) for the outer-final winner. At 16 subjects that is
   roughly `16×5×2 + 16×2 = 192` staged selections against Exp A's 16 — order 10× a full Exp A run.
   Estimate it against the measured anchor (HISTORY 2026-07-28: Exp B 01:04:20 on 16 cores, 16 folds
   in one wave ⇒ ~1 core-hour/fold) **before** writing the sbatch header, and size `--time`/
   `--cpus-per-task` from that. Step 3's header was under-sized by 2–3× and it cost a redesign.
2. **The M6 search space cannot be shrunk in tests.** `protocol_freeze_guard._check_m6_sections`
   rejects any deviation, so every G test that drives the real search pays for 113 candidates per
   selection. Keep end-to-end tests to ONE small synthetic-store case and put everything else on
   hand-built fold results — the split `tests/test_robustness.py` uses.
3. **G needs BOTH bands, and the local 77 GHz store is 1 of 72 sessions.** There is no meaningful
   local real-data G smoke. Tests must build a synthetic two-band store; the real smoke is step 13,
   on IBEX, after the step-11 rebuild.
4. **Touching `splits.py` touches the frozen leakage suite's dependency.** `selection_folds` must be
   a pure extraction: `nested_loso_splits` output stays byte-identical.
   `git diff --exit-code -- tests/test_no_leakage.py` is an acceptance step and it must stay green.

## The four amendments in force (full text + evidence in plan §0.2 and §8.2)

| ID | One line |
|---|---|
| **A-M10-7** | Reference Exp-A runs are the `*_3f465abc` pair, not `*_f0a46aa6`. Provenance only; selection tables byte-identical, **no estimand changed**. Exp C/D `f0a46aa6` assembly sources remain valid. |
| **A-M10-8** | Multiplicity is **contiguous row duplication for every family**, not `sample_weight` (not duplication-equivalent for `svr`'s data-dependent `gamma="scale"` or `rf`'s bootstrap `n_samples`). Both mechanisms pinned by test. |
| **A-M10-9** | Exp C arm (b) keeps the frozen **multiclass** O-M9-7 weights; the per-threshold binary rule would break byte-neutrality at multiplicity one. |
| **A-M10-10** | H's `robustness_selection.csv` records the **selected** candidate per stage, and `fit_audit_robustness.csv` the **outer-level** fits, because A/B/C fold workers discard their `StageOutcome`/`InnerResult`. **Scoped to H — Exp G's per-candidate table is unaffected and stays.** |

## Process traps that are still live

- **Do not rebuild the stores.** Steps 4–10 land as green commits with **no** store rebuild; step 11
  stamps `REVISION` once and rebuilds both bands once. A commit move invalidates both stores
  (`store._check_match`, strict commit equality) — that is why the work is batched this way.
- **The local 77 GHz store is 1 of 72 sessions.** Any real-data 77 GHz work runs on IBEX. The local
  10 GHz store is complete but built at `dab8f708`, which is *not* the analysis commit, so
  `validate_store` refuses it — a local real-data smoke fails on provenance, not mechanism.
- **IBEX** sits on `v1_milestone_9a` at `3f465ab` unless someone moved it; it needs
  `git fetch origin && git checkout v1_milestone_10`. Both former dirty-tree offenders (`dcgm/`,
  `results/exploratory_frame_split/`) are gitignored, so `submit_ibex.sh` will not refuse.
- **Heavy work is always `sbatch`**, never a login-shell run.
- **`tests/test_no_leakage.py` is frozen** — untouched through steps 1–3; keep it that way.
- **5 pre-existing test failures on this Windows machine**, all in
  `tests/test_exp_b_ibex_scripts.py` / `tests/test_exp_d_ibex_scripts.py`: Git Bash eats the
  backslashes in a Windows path and inline `bash -c` strings return empty. **Current baseline is
  `1305 passed, 5 failed, 16 skipped`** (of which 77 are step 3's). A 6th failure is yours.
  `tests/test_run_robustness.py::test_every_shell_artifact_parses_with_bash_dash_n` shows the fix
  for those two files: pipe the script's **bytes** to `bash -n -` with LF forced, never a path.
- **`core.autocrlf=true` here** — a byte-exact reference artifact needs a `-text` entry in
  `.gitattributes`. `scripts/ibex/*` is already `text eol=lf`, so new sbatch files are safe.

## Hard invariants (unchanged, never violate)

LOSO at subject level for every reported result; fit-on-train-only at every CV level; no test-set
tuning; primary target continuous Δm%; folds only from `eval/splits.py`; tie-breaks only via
`eval/selection.py`; `protocol_freeze_guard` before every fit/write; E/F/G/H are entirely CPU.
Do not report frame-level accuracy as a headline, do not claim causal isolation of hydration from
time of day, do not overclaim clinical readiness, and **do not tune E/F/G/H toward a more favourable
result because A–D came out negative**. Fusion is not required to beat 10 GHz.

## Remaining sequence after step 4 (plan §4.2)

5. Exp E · 6. Exp F · 7. drivers + assembly · 8. independent tests · 9. independent code review ·
10. corrections + mandatory retest · 11. final commit + **one** store rebuild · 12. Exp A/B reruns at
the final commit, then `validate_exp_a_reference.py --compare` against the committed manifest ·
13. local smokes then the full E/F/G + `R=200` IBEX jobs · 14. assembly, and **only then**
`SECOND_CHAPTER.md` §9.

**Step 13 note:** robustness sizing is decided — A and B submit `run_robustness.sbatch` with
`--cpus-per-task=64`; Exp C uses `submit_robustness_sharded.sh` (array + merge), with `ARRAY_TIME`
sized from one measured shard. Both paths are in §6's launch matrix.

## Chapter state

`SECOND_CHAPTER.md` §0–§8 complete. §9 is still the pre-registration stub reconciled to the accepted
plan; it is written in full **only after** verified full-cohort M10 artifacts exist, never before.
It must disclose **A-M10-1..10**.
