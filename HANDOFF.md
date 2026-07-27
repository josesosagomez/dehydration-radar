# HANDOFF — resume point for a new chat (Milestone 8: Exp B implementation starts here)

_Written 2026-07-27. **M8's plan is fully written and reviewed; zero code has been written.**
This chat's job is to implement `plans/MILESTONE_8_PLAN.md` starting at its step 1 (after the
one-time setup below)._

## TL;DR

M7 closed with a **negative** headline result: full-cohort Exp A (radar-based fluid-loss
regression) lost to the trivial session-index-only (time-of-day) baseline, significantly, in
both bands. That result can't distinguish "no radar signal" from "signal present but swamped by
the fasting-clock confound" — which is exactly what **Experiment B (clock-decoupling,
session-mean-residualized)** was pre-registered, before Exp A's result was seen, to test.

`plans/MILESTONE_8_PLAN.md` (Exp B) is written and went through a **full Codex⇄Claude adversarial
review to `REVIEW_COMPLETE`: 25 comments (C1–C25), all applied, zero debated, zero deferred to the
owner.** The plan is dense (~1450 lines) because the review caught real issues across many
rounds — read it in full, not skimmed; the resolved-comment log at the bottom explains *why*
almost every non-obvious design choice is the way it is. **No M8 source code exists yet.**

## Read first (in this order)

1. `CLAUDE.md` — hard invariants (LOSO subject-level; fit-on-train-only; no test-set tuning;
   continuous primary target; frozen `test_no_leakage.py`), code style, journal rules.
2. `HISTORY.md` — the newest three entries (M8 step-0.5 done; M8 plan+review closed; M7's actual
   Exp A numbers, logged only just now since they'd been sitting unrecorded).
3. `plans/MILESTONE_8_PLAN.md` — **the plan, in full.** §0 (scope, invariant, owner decisions),
   §1 (build sequence — follow it in order), §2 (per-file specs — the actual contract to
   implement), §3 (tests), §4 (DoD), §5 (traps — read before you hit them, not after), §6
   (amendments), and the **resolved review log** at the very end (C1–C25) for the reasoning
   behind anything that looks surprising in §0–§6.
4. `plans/implementation_plan.md` §B and §Statistics — Exp B's frozen core design, now amended
   (A-M8-1, A-M8-2) to match the plan.

## One-time setup before step 1 (not yet done — do this first)

The working tree currently has **uncommitted changes on `v1_milestone_7`** (still checked out,
HEAD `bda8e45`): `HISTORY.md`, `plans/implementation_plan.md`, `plans/review_prompt_claude.md`,
`plans/review_prompt_codex.md` (modified), `plans/MILESTONE_8_PLAN.md` (new, untracked). These
are all planning/journal edits — no source code. I deliberately did **not** create the
`v1_milestone_8` branch or commit, since that's a git action the owner should trigger explicitly.

1. Confirm with the owner, then: `git checkout -b v1_milestone_8` (off `v1_milestone_7` @
   `bda8e45`, per the plan's own ground rules), commit the planning docs as the baseline (e.g.
   "M8: plan written and reviewed (REVIEW_COMPLETE, 25/25 applied)").
2. Then follow `plans/MILESTONE_8_PLAN.md` §1's build sequence starting at **step 1** — step 0
   (write+review the plan) and step 0.5 (propagate A-M8-1/A-M8-2 into `implementation_plan.md`)
   are both **already done** in this session's uncommitted changes, about to become that baseline
   commit.

## What M8 (Exp B) builds — one-paragraph shape

A residualized-target sibling of Exp A's harness: predict `Δm%(subj, session) − μ_s` (μ_s = the
train-only session mean) instead of raw `Δm%`, so a good score requires tracking *between-subject*
fluid-loss variation at a *fixed* clock time, not decoding the clock itself. Reuses Exp A's exact
search space (A-M6-3) and most of its machinery unchanged; the one harness edit is a pluggable
`score_fn` (`src/dehyd/eval/harness.py`) so Exp B's `equal_session_residual_mae` objective can
drive selection without touching Exp A's path. New `src/dehyd/eval/exp_b.py` composes it. A
frozen four-session-specific-models secondary variant runs last, as a genuine 4-task SLURM array
(not a sequential loop) with its own run-group provenance and fail-closed shard validation.

## Owner decisions already baked into the plan (do not re-litigate)

- **A-M8-1**: primary = session-weighted aggregate difference CI
  (`aggregate(radar) − aggregate(baseline)`); subject-weighted complete-case Wilcoxon = a
  companion, never conflated with the primary. Resolves a genuine textual contradiction in
  `implementation_plan.md`'s frozen Statistics section (now fixed there too, step 0.5).
- **A-M8-2**: a bootstrap replicate that empties a session is skipped-and-counted, not averaged
  over the survivors.
- Both A-M8-1/A-M8-2 were decided **2026-07-27, after Exp A's results were visible** — disclosed
  with that chronology throughout the plan, not folded into "frozen before Exp A" language. This
  distinction is load-bearing; don't casually rephrase it back to "frozen before Exp A."
- **Gating**: single-phase DoD, no owner-checkpoint pause before the full-cohort run — nothing
  left to blind, since Exp B's *core* design (what runs, what data is used) was frozen before
  Exp A, and A-M8-1/A-M8-2 are reporting/labelling completions, not data-use choices.
- **Session-specific variant**: build it, but last (owner decision, Step 0 item 3) — honours the
  frozen `session_specific_variant_enabled: true` config flag without letting it delay the
  primary pooled result.
- **Review**: full Codex⇄Claude loop, now closed. Do not reopen A-M8-1/A-M8-2 or any
  already-resolved comment without a new, explicit owner decision.

## Traps worth knowing before you start (full list: plan §5, 26 traps)

The three most likely to bite early: **(trap 2)** Exp B's evaluable-subject rule (≥1 eligible
S1–S4 session) is not Exp A's (≥1 eligible session) — reusing Exp A's helper will crash once S0
rows are filtered. **(trap 3)** S0 must be excluded at the session-spine level, not via Exp A's
`abs(y_true) > 1e-9` heuristic, which is meaningless on residualized targets. **(traps 20–26)**
every call into `provenance.record_run` was checked against its *actual* source after getting it
wrong from memory twice during review — read `src/dehyd/provenance.py` yourself before writing
the `--init-run-group` code, don't reconstruct its API from this handoff or from memory.

## Next steps, in order

1. **One-time setup** (above): branch, commit the planning baseline.
2. Plan §1 **step 1**: pin current Exp A/harness behaviour (`inner_scores.tobytes()`, selected
   candidate, `test_score` on a fixed synthetic-store fixture; `subject_cluster_bootstrap_pooled`'s
   CI on a fixed fixture) — **before** touching anything, so "no behaviour change to Exp A's
   path" is provable, not asserted, at steps 2 and 4.
3. Steps 2–11 per the plan's build sequence table, in order. Each step names its acceptance tests
   (§3) — keep them green before moving on.
4. Full-cohort Exp B run, both bands, on IBEX (step 10) — no owner pause needed this time.
5. Session-specific variant via `scripts/ibex/submit_exp_b_variant.sh` (step 10.5).
6. HISTORY.md entries per resolved step (continuously, not batched) — a pattern already
   established across M1–M7 and the M8 planning entries just added.
7. Only once Exp B's real numbers are in: **write SECOND_CHAPTER.md together for both §6 (Exp A)
   and §7 (Exp B)** — explicit owner decision (this session) to hold off on §6 alone despite Exp
   A's numbers already being available, so both experiments get reported with full context
   at once. Do not write §6 in isolation before Exp B lands.

## Exp A's actual numbers (for §6, once you get there)

10 GHz: subject-balanced MAE 0.469 [0.409, 0.568]; mean difference (radar − baseline) **+0.200**
[0.145, 0.260], Wilcoxon p=3.05e-5; pooled r −0.138 [−0.286, 0.075]. 77 GHz: MAE 0.495 [0.404,
0.646]; mean difference **+0.216** [0.127, 0.296], p=7.6e-4; pooled r −0.153 [−0.407, 0.174]. Full
detail and selection tables: `HISTORY.md`'s "M7 CLOSES" entry and
`results/runs/20260727T11{14,50}*_f36c4fb2/metrics_exp_a_{10,77}ghz.json`.

## Hard invariants (never violate — a failing check stops the build)

LOSO at subject level; fit-on-train-only at both CV levels; no test-set tuning; primary target
continuous Δm%, session-level headline; folds only from `splits.py`; tie-break only via
`select_candidate`; numpy backs all reported features; `protocol_freeze_guard` before every
fit/write; `tests/test_no_leakage.py` unchanged except its one pre-registered T18 hunk (from M7 —
**M8 makes zero changes to this file**, per its own §3).

## Journal & hygiene

**HISTORY.md** newest-first, current through M8's plan/review closure and step 0.5.
**SECOND_CHAPTER.md** §0–§5 written; §6 (Exp A) and §7 (Exp B) both **deliberately pending** —
see "Next steps" item 7. **MILESTONE_8_PLAN.md** written, reviewed, `REVIEW_COMPLETE`, not yet
implemented. **implementation_plan.md** amended (A-M8-1, A-M8-2) to match. Branches
`v1_milestone_1..7` local; `v1_milestone_8` **not yet created** (see one-time setup). Nothing
pushed, nothing on `main`. Commit only when the owner asks. Superseded material → `archive/`.
