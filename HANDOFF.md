# HANDOFF — resume point for a new chat (Milestone 8 CLOSED; Milestone 9 not yet planned)

_Written 2026-07-30. **Milestone 8 (Exp B, clock-decoupling) is fully complete — plan steps 1-11
all done, both bands, real IBEX results in hand and written up.** Picking this up almost certainly
means: start planning Milestone 9 (Experiments C/D — ordinal classification + baselines), the next
unplanned work. There is one pre-M9 decision already made and recorded that MUST be folded into
that plan (see "Next steps" item 2) — do not miss it by planning from ROADMAP.md/
implementation_plan.md alone without reading the note called out below._

## TL;DR

M8 is done. Both Experiment A (fluid-loss regression, full cohort) and Experiment B
(clock-decoupling, full cohort + session-specific variant) have real, final results, both bands,
written into `SECOND_CHAPTER.md` §6+§7 and logged step-by-step in `HISTORY.md`. **The headline is
negative for both experiments, in both bands:** radar does not beat the trivial time-of-day
(Exp A) or session-mean (Exp B) baselines in this cohort — 10 GHz loses significantly in both
experiments; 77 GHz loses significantly in Exp A but shows no significant difference either way
in Exp B (more consistent with "Exp A's loss was mostly the clock confound" for that band
specifically). This is reported in full as the plan required, not softened.

Along the way, two real IBEX-only operational bugs were found and fixed (git-free
`submit_exp_b_variant.sh` for the owner's copied-tree — no `.git` — deployment; a progress
heartbeat added to `_run_folds_parallel` so long jobs' logs don't look hung) — both are in commit
`e88fd33`. A **separate, not-yet-resolved wrinkle**: that commit doesn't match the feature stores'
build commit (`30c6d907ca6f293f72db73517dc585bc39ec8e66`), so the *deployed* `src/dehyd/eval/
exp_b.py` on IBEX right now is deliberately the older, no-heartbeat version (see HISTORY.md's
2026-07-29 step-10.5 entry for the full reasoning) — the heartbeat feature is real, committed, and
in the codebase, just not live on IBEX until the stores are next rebuilt at a commit that includes
it. Nothing to do about this now; it's just a fact worth knowing before assuming what code is
actually running there.

There's also a **pre-M9 planning decision already made**, not yet acted on: the owner wants an
additional, deliberately-leaky, exploratory frame-level random-split (k=5) evaluation alongside
LOSO for every Exp C/D result — see "Next steps" item 2.

**Working tree has uncommitted changes** (`HISTORY.md`, `SECOND_CHAPTER.md`,
`plans/implementation_plan.md`) — the M8-closing journal entries and the pre-M9 note. **Not
committed** — commit only on explicit owner request, per standing project rule. (There is also a
long-standing unrelated deletion, `results/preprocess/preprocess_diagnostics_10ghz.csv`, present
in git status since before this chat started — not something this session touched; leave it
alone unless the owner raises it.)

## Read first (in this order)

1. `CLAUDE.md` — hard invariants, code style, journal rules (unchanged).
2. `HISTORY.md` — the newest ~6 entries (2026-07-30's two entries, then M8 steps 10.5/10/9/8.6)
   for exactly what happened, what broke and why, and the real numbers. Older M8 entries below
   that only if you need earlier reasoning (e.g. why A-M8-1/A-M8-2 were decided the way they
   were).
3. `SECOND_CHAPTER.md` §6 (Exp A) and §7 (Exp B) — the full, final, real-numbers write-up. This is
   the authoritative account of what M8 (and M7's Exp A) actually found; read it before citing any
   number from memory.
4. `plans/implementation_plan.md` — the note right after the Exp D baseline specs (before §E),
   headed "Owner-requested addition, pre-M9 (decided 2026-07-30)" — the frame-split decision, see
   below. This is the one thing most likely to get missed if M9 planning starts from ROADMAP.md
   alone.
5. This file's "Next steps" below.

## What's actually done

- **M8 plan steps 1-11, all complete** (`plans/MILESTONE_8_PLAN.md`, `REVIEW_COMPLETE`). Code
  committed at `81cec63` (steps 1-8); operational fixes at `e88fd33` (git-free wrapper + progress
  logging, step 10.5's own detour). Full repo test suite green (767 passed, 16 pre-existing skips)
  as of `e88fd33`; `tests/test_no_leakage.py` unchanged.
- **Real IBEX results, both experiments, both bands** — in `results/runs/` locally (synced back
  via `rsync` from `/ibex/user/floresge/dehy_radar/`) and written up in `SECOND_CHAPTER.md` §6+§7:
  - Exp A full-cohort: `20260727T111437230187Z_f36c4fb2` (10 GHz), `20260727T115046533408Z_f36c4fb2`
    (77 GHz).
  - Exp B mechanism-only smoke: `20260727T224535326693Z_30c6d907` (10 GHz),
    `20260727T231856775480Z_30c6d907` (77 GHz).
  - Exp B full-cohort: `20260727T233312448972Z_30c6d907` (10 GHz),
    `20260728T000714076071Z_30c6d907` (77 GHz).
  - Exp B session-specific variant (merged, `completed_sessions=[1,2,3,4]` both bands):
    `20260728T224133954370Z_30c6d907` (10 GHz), `20260729T004647423767Z_30c6d907` (77 GHz).
  - Three empty `*_nogit` run dirs (provenance.json only, no results) are dead-end artifacts from
    early failed `init` attempts before the git-free wrapper/REVISION fix — harmless, candidates
    for `archive/` cleanup per CLAUDE.md file hygiene, not urgent.
- **HISTORY.md** fully current through M8's close (newest-first: pre-M9 frame-split decision, M8
  CLOSES, then steps 10.5/10/9/8.6/8.5/... down to M8's start).

## Next steps, in order

1. **Do not reopen anything M7/M8 already settled** (A-M7-1/2/3, A-M8-1/A-M8-2, any resolved
   `MILESTONE_7_PLAN.md`/`MILESTONE_8_PLAN.md` review comment, the Exp A/B results themselves)
   without a new, explicit owner decision — same standing rule as every prior milestone.
2. **Start Milestone 9 planning (Experiments C/D — ordinal 5-class classification + baselines),
   and fold in the pre-M9 decision already made:** alongside the required LOSO protocol, also
   plan for a **5-fold random split over pooled frames** (k=5, ~80/20 per fold, frames from all
   subjects shuffled together — matching the original paper's frame-as-sample splitting) for
   every Exp C/D result. This is deliberately leaky and **exploratory-only — never for the thesis
   or paper, never a headline number** (owner's own words). Three hard constraints when
   implementing it (already in `plans/implementation_plan.md`'s note): (a) a clearly separate
   code/output path, own tagged filenames, never merged into the LOSO metrics files; (b) must
   never touch or weaken `tests/test_no_leakage.py`, `splits.py`'s LOSO machinery, or
   `config-freeze-v1`; (c) never surfaces in `SECOND_CHAPTER.md`'s actual findings.
3. Read `ROADMAP.md`'s Exp C/D framing and `plans/implementation_plan.md`'s full C/D spec
   (ordinal sign convention, Frank-Hall decomposition amendment A-M6-5, fold-viability rules,
   the three D baselines + budget-parity rule) before drafting `plans/MILESTONE_9_PLAN.md` — this
   is a large, detailed, already-mostly-specified section; the plan mostly needs to sequence
   implementation, not invent design.
4. Follow the same process as M6-M8: draft the plan, run the Codex⇄Claude review loop to
   closure, then implement — per CLAUDE.md and the owner's established preference (confirmed
   explicitly at M8 Step 0 item 4).
5. Commit only when the owner asks — including the currently-uncommitted M8-closing journal
   changes (`HISTORY.md`, `SECOND_CHAPTER.md`, `plans/implementation_plan.md`), which are still
   sitting unstaged from this session.

## Hard invariants (unchanged, never violate)

LOSO at subject level for every REPORTED result; fit-on-train-only at both CV levels; no
test-set tuning; primary target continuous Δm%, session-level headline; 5-class S0-S4 is
secondary, ordinal metrics only; folds only from `splits.py`; tie-break only via
`select_candidate`; numpy backs all reported features; `protocol_freeze_guard` before every
fit/write; `tests/test_no_leakage.py` unchanged. The one deliberate, disclosed exception: the
pre-M9 exploratory frame-split addition above — never a substitute for LOSO, never reported.

## Journal & hygiene

**HISTORY.md** newest-first, current through M8's close + the pre-M9 decision (2026-07-30).
**SECOND_CHAPTER.md** §0-§7 all complete and final; §8 (Exp C/D) and §9 (fusion/interpretability/
statistics) pending their own milestones. **MILESTONE_8_PLAN.md** `REVIEW_COMPLETE`, all 11 steps
done. No `MILESTONE_9_PLAN.md` exists yet. Branch `v1_milestone8`, HEAD `e88fd33` at the time this
was written, 3 commits ahead of `origin/v1_milestone8` (not pushed — not verified whether the
owner has separately synced to IBEX by another route). Working tree has `HISTORY.md`,
`SECOND_CHAPTER.md`, and `plans/implementation_plan.md` modified but uncommitted (see TL;DR).
Commit only when the owner asks.
