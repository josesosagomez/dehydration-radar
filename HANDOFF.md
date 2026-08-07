# HANDOFF — resume point for a new chat (STEPS 1–2 DONE, RE-REVIEW PASSED; next job: **step 3**)

_Written 2026-08-08. The next chat's job: **implement `plans/MILESTONE_10_PLAN.md` §4.2 step 3 — the
H robustness driver, `src/dehyd/eval/robustness.py`.** Read this file, then the plan's §2.4 (the
resampling rules — **amended, read the current text, not your memory of it**), §3's four robustness
artifact rows, §4.2 step 3, §5.5, and §6. Steps 1 and 2 are complete, tested, pushed, and their
amendments have passed an independent re-review; do not re-litigate them._

## TL;DR

- **Branch `v1_milestone_10`**, HEAD `a3354fd`, pushed to `origin`. 10 commits since `fee9172`.
  Working tree clean except untracked `.codex/` (owner's tooling — leave it).
- **Step 1 done.** The Exp-A reference gate is built and the authoritative snapshot is taken:
  `results/milestone10/reference_exp_a_manifest.json` (725 KB, `reference_grade: authoritative`,
  both bands, version-controlled with a `-text` attribute). **The fail-closed precondition on every
  later structural edit and store rebuild is satisfied** — you do not need to repeat it.
- **Step 2 done.** The multiplicity foundation reaches models, harness, providers and A/B/C
  orchestration, byte-neutral by default for Experiments A–D.
- **Three amendments were raised during implementation and have PASSED re-review** (A-M10-7/8/9,
  `plans/MILESTONE_10_PLAN.md` §0.2). They are now ordinary parts of the design. The plan's §1.3,
  §2.4, §4.1, §5.5, §6, §8.2 and §9.1 were all revised to match, so **the plan is internally
  consistent — trust its current text**.
- **Step 3 is the next job and is unblocked.**

## What already exists that step 3 must REUSE, never reimplement

`plans/MILESTONE_10_PLAN.md` §4.2 step 3 is explicit: "Reuse A/B/C candidate enumeration and
orchestration with the exact multiplicity contract." All of this is built and tested:

| Need | Use |
|---|---|
| Run one experiment under a bootstrap draw | `exp_a.run_exp_a(..., subject_multiplicity=)`, `exp_b.run_exp_b(...)`, `exp_c.run_exp_c(...)` — all take it, including through their spawn-context workers |
| Apply multiplicity to a fit | `models.regressors.fit_pipeline(pipe, X, y, row_multiplicity=)` — the single dispatch; expands rows contiguously (A-M10-8) |
| Expand rows | `models.regressors.expand_by_multiplicity` |
| Per-row copy counts from a `{subject: m_s}` map | `harness.subject_row_multiplicity(subjects, mapping)` |
| Weighted baselines / session means | `models.baselines.fit_session_index_baseline(..., subject_multiplicity=)`, `session_means(...)`, `fit_session_mean_baseline(...)` |
| Tuned-ε under a draw | `harness.tuned_epsilons(..., subject_multiplicity=)` — repeats each subject's scale `m_s` times **before** the median |
| CIs, Wilcoxon, Holm, ordinal metrics | `eval/metrics.py` — already complete; H reuses it, does not rebuild it |

**Pass the `{subject: m_s}` MAPPING down, never a row-aligned array.** The harness expands it against
each bundle's own rows because Exp B's provider drops degenerate sessions, so a row array built once
outside would attach multiplicities to the wrong rows silently. See plan §4.1 and
`tests/test_multiplicity.py::test_multiplicity_stays_aligned_when_a_provider_drops_rows`.

## Step 3's own requirements (from the plan — read them there in full)

- **Estimands** (§2.4): Exp A both bands — selected radar subject-balanced MAE **and** radar −
  session-index MAE difference; Exp B both bands — the primary equal-session aggregate radar −
  baseline difference; Exp C both bands — class-unit MAE for arm a and arm b (adjacent accuracy and
  QWK get no refit-robustness range).
- **Skip rules, in order**: < 4 distinct drawn subjects; Exp C additionally needs all 5 classes.
  These are *coarse prechecks, not permission to summarize a partial run* — if any nested selection
  has no surviving candidate, any outer prediction is missing, or Exp B's four-session aggregate is
  unavailable, the **whole result replicate** is skipped with the first canonical reason. Never
  computed over the remaining easier folds. Reasons are counted.
- **`R=200`**; fewer than **100** successful replicates ⇒ the result is **inconclusive**. The
  threshold is never scaled — the `R=8` smoke must *intentionally* report inconclusive.
- **RNG freeze**: `SeedSequence([robustness_seed, experiment_code, band_code, replicate])` with
  `a,b,c → 1,2,3` and `10ghz,77ghz → 10,77`. One draw is **shared by all arms/contrasts of that
  experiment-band replicate**. Model seeds stay the configured model seeds and are *not* derived
  from the resampling seed. Save the tuple and the generated 128-bit state.
- **Percentiles**: exactly `np.quantile(successful_estimates, [0.025, 0.975], method="linear")`
  after sorting by replicate ID. Label `selection_variance_empirical_95pct_range` — **never**
  `ci_method=bca` (A-M10-5).
- **Artifacts** (§3): `robustness_replicates.csv`, `robustness_selection.csv`,
  `fit_audit_robustness.csv`, `robustness_summary.csv`, `metrics_robustness.json`. Column lists are
  in §3; the fit-audit companion JSON stores canonical subject/multiplicity maps keyed by hash.
- Also in step 3's scope per §4.1: `experiments/run_robustness.py` and
  `scripts/ibex/run_robustness.sbatch`.

## The one genuinely open decision

**Where does `robustness_seed` come from?** `StatsConfig` has `robustness_replicates_r`,
`robustness_min_distinct_subjects`, `robustness_min_successful_replicates` and
`robustness_ordinal_min_classes` — but **no `robustness_seed`**. The M6 sections are frozen records,
so adding a field is a config change, not a free choice. The obvious candidate is `config.run.seed`
(20260721). Decide it explicitly, record it in HISTORY, and pin it by test — do not let it be an
accident of whatever the first implementation reached for.

## The three amendments in force (full text + evidence in plan §0.2 and §8.2)

| ID | One line |
|---|---|
| **A-M10-7** | Reference Exp-A runs are the `*_3f465abc` pair, not `*_f0a46aa6` — the latter's stores were rebuilt away. Provenance only: selection tables byte-identical, **no estimand changed**. Scoped to Exp A; Exp C/D `f0a46aa6` assembly sources remain valid. |
| **A-M10-8** | Multiplicity is applied by **contiguous row duplication for every family**, not `sample_weight` — which is not duplication-equivalent for `svr` (data-dependent `gamma="scale"`) or `rf` (bootstrap draws `n_samples`). Both mechanisms are pinned by test. |
| **A-M10-9** | Exp C arm (b) keeps the frozen **multiclass** O-M9-7 weights; the plan's per-threshold binary rule would change Exp C at multiplicity one and break byte-neutrality. |

## Process traps that are still live

- **Do not rebuild the stores.** Steps 3–10 land as green commits with **no** store rebuild; step 11
  stamps `REVISION` once and rebuilds both bands once. A commit move invalidates both stores
  (`store._check_match`, strict commit equality), which is why the work is batched this way.
- **The local 77 GHz store is 1 of 72 sessions.** Any real-data 77 GHz work must run on IBEX. The
  local 10 GHz store is complete but built at `dab8f708`, which is *not* the reference commit.
- **IBEX** sits on `v1_milestone_9a` at `3f465ab` unless someone has moved it; it needs
  `git fetch origin && git checkout v1_milestone_10`. Both former dirty-tree offenders (`dcgm/`,
  `results/exploratory_frame_split/`) are now gitignored, so `submit_ibex.sh` will not refuse.
- **Heavy work is always `sbatch`**, never a login-shell run.
- **`tests/test_no_leakage.py` is frozen** — `git diff --exit-code` on it is an acceptance step at
  every stage. It has stayed untouched through steps 1–2; keep it that way.
- **5 pre-existing test failures on this Windows machine**, all in
  `tests/test_exp_b_ibex_scripts.py` / `tests/test_exp_d_ibex_scripts.py`: Git Bash eats the
  backslashes in a Windows path (`C:UsersjosemsosagDesktop…`) and inline `bash -c` strings return
  empty. They touch no M10 file. Baseline for the full suite is therefore
  **1225 passed, 5 failed, 16 skipped** — if you see a 6th failure, it is yours.
  (When feeding a script to `bash -n`, pass **bytes** with LF forced: `text=True` rewrites `\n` to
  `\r\n` and a CRLF after a line-continuation backslash breaks the parse.)
- **`core.autocrlf=true` here** — any byte-exact reference artifact committed needs a `-text` entry
  in `.gitattributes` (`results/milestone10/*.json` already has one).

## Hard invariants (unchanged, never violate)

LOSO at subject level for every reported result; fit-on-train-only at every CV level; no test-set
tuning; primary target continuous Δm%; folds only from `eval/splits.py`; tie-breaks only via
`eval/selection.py`; `protocol_freeze_guard` before every fit/write; E/F/G/H are entirely CPU.
Do not report frame-level accuracy as a headline, do not claim causal isolation of hydration from
time of day, do not overclaim clinical readiness, and **do not tune E/F/G/H toward a more favourable
result because A–D came out negative**.

## Remaining sequence after step 3 (plan §4.2)

4. Exp G (`selection_folds` in `splits.py` first) · 5. Exp E · 6. Exp F · 7. drivers + assembly ·
8. independent tests · 9. independent code review · 10. corrections + mandatory retest ·
11. final commit + **one** store rebuild · 12. Exp A/B reruns at the final commit, then
`validate_exp_a_reference.py --compare` against the committed manifest · 13. local smokes then the
full E/F/G + `R=200` IBEX jobs · 14. assembly, and **only then** `SECOND_CHAPTER.md` §9.

## Chapter state

`SECOND_CHAPTER.md` §0–§8 complete. §9 is still the pre-registration stub reconciled to the accepted
plan; it is written in full **only after** verified full-cohort M10 artifacts exist, never before.
It must disclose **A-M10-1..9**.
