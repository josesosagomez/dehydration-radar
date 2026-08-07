# HANDOFF — resume point for a new chat (MILESTONE 10 PLAN ACCEPTED; next job: **implement** it)

_Written 2026-08-07. The next chat's job: **implement `plans/MILESTONE_10_PLAN.md`** (787 lines,
independently reviewed and accepted 2026-08-07 — this is the authoritative design, not a summary
you should re-derive from ROADMAP.md/`implementation_plan.md`, which it amends in six disclosed
places). Read this file, then `plans/MILESTONE_10_PLAN.md` in full — it is dense and every section is
load-bearing; do not skim it and start coding from memory of what M10 "should" contain. No code
exists yet. One action is time-sensitive — see "Do this first," below, before anything else._

## TL;DR

- **Milestone 9 is closed** (unchanged since the last handoff): branch `v1_milestone_9a`, HEAD
  `fee9172`, working tree clean except this session's journal edits. The four-legged negative result
  (Exp A-D) stands; M10 is not going to rescue it.
- **Milestone 10 now has an accepted, independently-reviewed implementation plan and zero code.**
  `plans/MILESTONE_10_PLAN.md` was drafted this morning, then went through this project's
  review-before-code discipline, which found real design problems (not prose issues) and rewrote it
  substantially — six disclosed amendments, **A-M10-1..6** (`plans/MILESTONE_10_PLAN.md` §0.2). One
  of them, A-M10-3, corrects a genuine leakage-shaped bug in what I (the planning session) originally
  proposed for Exp G — worth reading, not just trusting.
- **`SECOND_CHAPTER.md` §9 has been reconciled to the accepted plan** in this session — an earlier
  draft of that section recorded the pre-review design and a reporting stance the review superseded
  (A-M10-1, A-M10-6); it now matches the accepted plan and says so.
- **Nothing is implemented.** No `src/dehyd/eval/exp_e.py`/`exp_f.py`/`exp_g.py`/`robustness.py`/
  `assembly.py`, no `v1_milestone_10` branch, no `results/milestone10/` directory. Verified this
  session by direct filesystem check.

## Do this first — time-sensitive, before any other M10 work

`plans/MILESTONE_10_PLAN.md` §4.2 step 1 requires snapshotting a **reference Exp-A manifest** from the
M9 stores and run artifacts **while they still exist**, before any structural edit or store rebuild:

```
uv run python experiments/validate_exp_a_reference.py --snapshot \
  --reference-10 results/runs/20260803T143704568296Z_f0a46aa6 \
  --reference-77 results/runs/20260803T151715023672Z_f0a46aa6 \
  --output results/milestone10/reference_exp_a_manifest.json
```

(`validate_exp_a_reference.py` does not exist yet — writing it is part of step 1 itself, §4.1/§3's "A
reference gate" row.) **Why this can't wait:** M10 will rebuild both feature stores, and
`store._check_match` enforces strict git-commit equality — once the stores move, the M9-era
evidence (stored session-vector hashes, tuned-branch raw/prelog/order data, fold/candidate/selection/
prediction hashes) is gone and cannot be reconstructed from the rebuilt stores. Verified this session:
both M9 stores (`results/features/{10ghz,77ghz}/`) and every M9 run directory the plan cites
(`results/runs/*_f0a46aa6`, `results/runs/*_3f465abc`) are still present on disk right now. This
snapshot is what later lets `validate_exp_a_reference.py --compare` prove the M10 rebuild didn't
silently change Exp A's answers before Exp F trusts its feature-selection output.

## The six amendments (A-M10-1..6) — read `plans/MILESTONE_10_PLAN.md` §0.2 for full reasoning

| ID | What changed | One-line why |
|---|---|---|
| A-M10-1 | Exp E: 4-fold permutation CV → **leave-one-path-group-out refit under outer LOSO** | The 4-fold design broke the project's "every reported result is outer LOSO" rule; LOPGO was already the frozen text's documented fallback |
| A-M10-2 | Exp F: HR check → **machine-checked `not_estimable_missing_heart_rate`** status, never a proxy or silent relabel | Zero HR observations exist anywhere in the delivered data |
| A-M10-3 | Exp G: OOF predictions for α come from **selection-honest nested cross-fitting**, not `InnerResult.val_predictions` | `val_predictions` is first-seed-only and reflects selection-time folds — reusing it leaks selection info into the meta-learner. **This corrects an error in my own original Exp G design** — see HISTORY.md 2026-08-07 for why |
| A-M10-4 | Exp G: feature-level fusion **explicitly deferred**, not a milestone-10 completion criterion | No learner/reduction/budget was ever frozen for it |
| A-M10-5 | Robustness bootstrap: **empirical 2.5th/97.5th percentile range**, not BCa | BCa needs an original-statistic jackknife an already-bootstrapped estimate vector doesn't have |
| A-M10-6 | Exp E reporting: **outcome-neutral**, no pre-labelled "null"/"physical" framing | Pre-labelling the result before it exists encodes a desired narrative as an acceptance criterion — supersedes the "report as null attribution" stance recorded earlier the same day |

## Where everything is

**The plan.** `plans/MILESTONE_10_PLAN.md`, 787 lines, all sections load-bearing:
§0 invariants + the six amendments · §1 verified repository contracts (read before assuming any
function signature) · §2 the four frozen designs in full mechanistic detail · §3 exact artifact
schemas (every CSV/JSON column, per experiment) · §4 planned files + the **14-step ordered
implementation sequence** (§4.2) · §5 the risk-based test plan by category · §6 exact validation
commands and the exact full-cohort IBEX launch matrix (copy-pasteable, already sequenced with
`--wait`) · §7 objective pass/fail criteria per experiment · §8 known limitations + the methodological
citations behind each amendment · §9 the workflow gate (below).

**Git.** Branch `v1_milestone_9a` (main branch `main`), HEAD `fee9172`. No `v1_milestone_10` branch
exists yet — create it off this HEAD at implementation start, following the M7-M9 branch-per-milestone
precedent (not specified by name in the plan itself, so this is a carried-forward convention, not a
plan requirement).

**Stores and runs.** M9 stores present but stale-by-commit (`5b5ff06`/`c523266` moved `src/` after the
last rebuild — unchanged fact from the last handoff). M10 rebuilds both once, per plan §4.2 step 11,
**after** the reference snapshot (above) and **after** the independent-review retest gate (below) —
not at the start. Referenced M9 run directories (all still present):

    20260803T143704568296Z_f0a46aa6   Exp A 10 GHz  ] the M9 reference controls §1.3 cites;
    20260803T151715023672Z_f0a46aa6   Exp A 77 GHz  ] superseded as F's authoritative source by a
                                                        fresh M10-commit Exp-A rerun once stores rebuild
    20260803T143705048534Z_f0a46aa6   Exp C 10 GHz   — authoritative source for H assembly
    20260803T160645780475Z_f0a46aa6   Exp C 77 GHz   — authoritative source for H assembly
    20260803T172827484892Z_f0a46aa6   Exp C 10 GHz (cross-vendor determinism control, not headline)
    20260806T104207854321Z_3f465abc   Exp D 10 GHz   — authoritative source for H assembly
    20260806T110156650286Z_3f465abc   Exp D 77 GHz   — authoritative source for H assembly

**No current authoritative Exp B run directory exists** (plan §1.3) — this is a resolved inventory
fact, not something to search harder for. M10 runs primary Exp B fresh for both bands at the final
commit (plan §4.2 step 12) and registers those directories before H assembly.

## Implementation sequence (condensed from plan §4.2 — read it there for the full detail)

1. **Protocol/inventory/schema pin** — apply A-M10-1..6 to configs/docs, pin current A-D byte
   behavior, take the reference snapshot above **while the M9 stores still exist**.
2. **Multiplicity foundation** — optional `subject_multiplicity`/`row_multiplicity` threaded through
   providers/harness/estimators, byte-neutral by default (every existing A-D call site must stay
   byte-identical — this is the one edit that touches shared code every other experiment depends on).
3. **H robustness driver** (`eval/robustness.py`) — reuses A/B/C's own candidate enumeration, never
   reimplements it.
4. **Exp G** (`eval/exp_g.py`) — `selection_folds` in `splits.py` first, then matched population,
   selection-honest nested cross-fitting, alpha selection, outer refits.
5. **Exp E** (`eval/exp_e.py`) — LOSO path-group ablation, band-aware physical metadata.
6. **Exp F** (`eval/exp_f.py`) — HR-availability record, four nested models, sensitivities, contrasts.
7. **Drivers/assembly** (`eval/assembly.py` + five entrypoints) — explicit run-directory maps, no glob
   discovery anywhere.
8. **Independent tests** — a tester who did not write the implementation runs the targeted + full +
   real-data suites on the candidate tree.
9. **Independent code review** — a reviewer who did not implement it checks the tested candidate for
   leakage/lineage/statistical defects; blockers return to the writer.
10. **Corrections + mandatory retest** — original writer fixes findings, adds regression coverage;
    independent tester reruns everything. **Only this post-correction green tree is eligible for the
    final commit** — not the pre-review result.
11. **Final commit + one store rebuild**, validated.
12. **Exp A/B reruns** at the final commit, both bands; bit-identity/lineage assert against the
    reference snapshot; register new Exp B directories.
13. **Mechanism-only local smokes, then full E/F/G + `R=200` robustness on IBEX CPU** — every job is
    `sbatch`, never a login-shell run (plan §6's launch matrix is exact and copy-pasteable).
14. **Assembly** — explicit `run_manifest.json`, final tables, **only then** update
    `SECOND_CHAPTER.md` §9 with real results.

**The workflow gate is a hard sequencing constraint, not a suggestion** (plan §9): one implementation
writer → independent risk-based tests → independent code review → corrections + mandatory retest →
documentation only after verification. No structural edit or store rebuild without the validated
reference snapshot; no full E/F/G/H IBEX job before the post-review retest gate is green; no Exp F or
assembly after any reference/lineage/schema mismatch.

## Planned new files (plan §4.1 has the full list with responsibilities)

`eval/exp_e.py`, `exp_f.py`, `exp_g.py`, `robustness.py`, `assembly.py` (new); `eval/splits.py` gains
public `selection_folds`; `eval/harness.py` gains one optional, byte-neutral-by-default
subject-multiplicity path; `eval/exp_a.py`/`exp_b.py`/`exp_c.py`, `models/baselines.py`,
`models/ordinal.py`, `models/regressors.py`, `eval/selection.py` (new `select_alpha`) all get small,
additive changes — no rewrites. Five entrypoints (`run_interpretability.py`, `run_confound.py`,
`run_fusion.py`, `run_robustness.py`, `run_stats_assembly.py`) + `validate_exp_a_reference.py`. Five
new `scripts/ibex/*.sbatch`, git-free/`REVISION`-wrapped from the start (the M9 C14-C25 lesson —
learned once, not relearned).

## Process traps carried forward from M9 (still true, still costly if ignored)

- **A commit move invalidates both stores** (`store._check_match`, strict commit equality) — why the
  reference snapshot above must happen before any structural edit, and why steps 1-10 land as green
  commits without a store rebuild until step 11.
- **Every real M9 bug lived in an untested success path while component tests stayed green** (the
  comparison stage, the merge summary, the M7 reference CSVs — three separate incidents). Plan §5 is
  the risk-based test plan by category; use it, and give each new driver the end-to-end
  real-lineage test it specifies, not only component tests.
- **`core.autocrlf=true` on this machine** — any byte-exact reference artifact committed needs a
  `-text` entry in `.gitattributes`.
- **Don't wholesale-replace local `results/runs/`** — gitignored except the `*_f36c4fb2/`/`*_f0a46aa6/`/
  `*_3f465abc/` references; pull specific dirs.
- **Heavy work is always `sbatch`, never a login-shell run** (plan §6 says this explicitly for M10;
  it was already true for M7-M9).

## Hard invariants (unchanged, never violate)

LOSO at subject level for every reported result; fit-on-train-only at every CV level; no test-set
tuning; primary target continuous Δm% (Exp F's signed-kg sensitivity variant is a disclosed secondary
check, not a redefinition); folds only from `eval/splits.py` (including the new `selection_folds`);
tie-breaks only via `eval/selection.py`; `protocol_freeze_guard` before every fit/write;
`tests/test_no_leakage.py` frozen (`git diff --exit-code` on it is an acceptance step at every stage);
GPU claims never apply here (E/F/G/H are entirely CPU). Do not report frame-level accuracy as a
headline, do not claim causal isolation of hydration from time of day, do not overclaim clinical
readiness, do not tune E/F/G toward a more favorable result because A-D came out negative (plan §0).

## Chapter state

`SECOND_CHAPTER.md` §0-§8 complete. §9 is reconciled to the accepted plan (this session) but still a
pre-registration stub, not the real chapter section — per plan §4.2 step 14 and CLAUDE.md's journal
rules, it is written in full **only after** verified full-cohort M10 artifacts exist, never before.
