# HANDOFF — resume point for a new chat (Milestone 8: Exp B — code complete, IBEX runs in progress)

_Written 2026-07-28. **All of M8's local code (plan steps 1-8) is implemented, tested, and
committed. Steps 8.6-10.5 (rebuild feature stores, mechanism-only smoke, full-cohort run,
session-specific variant) are IBEX-cluster-dependent and were handed to the owner to run
directly — this chat has no ssh access.** Picking this up almost certainly means: check whether
the owner's IBEX runs have finished and, if so, do step 11 (HISTORY.md entries for 8.6-10.5 from
the real run logs, then SECOND_CHAPTER.md §6+§7 together — a standing owner decision, not
optional)._

## TL;DR

M8 (Exp B, clock-decoupling) plan steps 1-8 are done: the `harness.py` `score_fn` hook,
`metrics.py`/`baselines.py` additions, the full `eval/exp_b.py` composition (residualizing
provider, staged search, A-M8-1/A-M8-2 reporting, the session-specific secondary variant), and
`experiments/run_clock_decoupling.py` + the three `scripts/ibex/*` artifacts. Committed as
`81cec63` on `v1_milestone8` ("M8 steps 1-8: Exp B (clock-decoupling) implementation complete and
green"). **Full repo suite: 767 passed, 16 pre-existing skips**, `tests/test_no_leakage.py`
`git diff --exit-code`-clean.

Two real bugs were caught by the tests during implementation, both fixed and logged in
HISTORY.md: `models/baselines.session_means` silently omitted a session from `dropped` when it
had ZERO (not just <2) eligible training subjects, because it only iterated sessions present
among the *training* rows; and a bash quoting quirk — an apostrophe inside a `${VAR:?message}`
parameter expansion breaks bash's parser even within double quotes
(`scripts/ibex/submit_exp_b_variant.sh`).

**The owner is now running steps 8.6-10.5 on IBEX directly** (rebuild+validate both feature
stores from commit `81cec63`; mechanism-only smoke both bands; full-cohort Exp B both bands; the
session-specific variant both bands, via `scripts/ibex/submit_exp_b_variant.sh`). This chat
stopped here — mid-IBEX-run — specifically to update the journal and prepare this handoff, per
explicit owner request; it was not asked to wait for the runs to finish.

## Read first (in this order)

1. `CLAUDE.md` — hard invariants, code style, journal rules (unchanged from prior milestones).
2. `HISTORY.md` — the newest ~7 entries (M8 steps 1 through 8.5) for exactly what was built, what
   broke and why, and the reasoning behind every non-obvious choice. Older M8 planning entries
   below that only if you need the plan's own pre-implementation reasoning.
3. `plans/MILESTONE_8_PLAN.md` — the plan (`REVIEW_COMPLETE`, 25/25 review comments applied), if
   you need to check an implementation detail against its spec. §1's build-sequence table, §2's
   per-file specs, §4's DoD, and §5's 26 traps are the sections you're most likely to actually
   need; the resolved-review log (C1-C25) at the end explains *why* almost every non-obvious
   design choice is the way it is.
4. This file's "Next steps" below.

## What's actually done (implemented AND tested, not just planned)

- `src/dehyd/eval/harness.py`: keyword-only `score_fn`/`FeatureBundle.session_idx` hook, proven
  bit-identical to pre-M8 behaviour via `tests/test_m8_pin.py` (a pin captured BEFORE any M8
  edit, re-asserted after both the `metrics.py` and `harness.py` changes).
- `src/dehyd/eval/metrics.py`: `per_session_residual_mae`, `equal_session_residual_mae`,
  `holm_adjusted`, `_cluster_bootstrap_over_rows` (extracted from
  `subject_cluster_bootstrap_pooled`, bit-identical), `session_weighted_bootstrap` (A-M8-1's
  primary CI, A-M8-2's skip-and-count built in).
- `src/dehyd/models/baselines.py`: `session_means`/`fit_session_mean_baseline`/
  `predict_session_mean` — deliberately NO Exp-A-style global-mean fallback; a degenerate session
  is dropped, never imputed.
- `src/dehyd/eval/exp_b.py` (new, ~900 lines): the full composition — `build_sessions_b`
  (S0-excluded), `evaluable_subjects_b`, `SessionResidualFeatures` (wraps
  `exp_a.StoreBackedFeatures`), `run_exp_b`, `summarize_exp_b` (C4 run-level viability, C5
  per-session `baseline_mae` as a CI, Holm-4), `write_exp_b_reports`, `run_and_report_b`, and the
  session-specific variant (`run_exp_b_one_session`, `run_exp_b_session_specific`,
  `summarize_variant_session`, `merge_session_specific_reports` with fail-closed shard-lineage
  validation, `config_fingerprint`, `eligible_subjects_for_session`).
- `src/dehyd/provenance.py`: `_fold_manifest` → public `fold_manifest` (docstring-only change
  beyond the name; zero behaviour change, confirmed by `test_provenance.py`).
- `experiments/run_clock_decoupling.py` (new CLI): the primary path (mirrors
  `run_regression.py`) plus the entirely separate `--session-specific` path with its three
  sub-flags (`--init-run-group`/`--session S --run-dir`/`--merge-sessions --run-dir`).
- `scripts/ibex/run_exp_b.sbatch` (primary path, cloned from `run_exp_a.sbatch`),
  `scripts/ibex/run_exp_b_variant.sbatch` (`STAGE=init|array|merge`, zero `#SBATCH` resource
  directives — C24), `scripts/ibex/submit_exp_b_variant.sh` (the init→array→merge orchestration
  wrapper, job-ID normalization — C25).
- Tests: `test_m8_pin.py`, `test_exp_b.py` (34 tests — run half, report half, variant),
  `test_run_clock_decoupling.py` (15), `test_exp_b_ibex_scripts.py` (5, incl. a `bash -n` syntax
  gate over all three new shell artifacts), plus additions to `test_metrics.py`/
  `test_baselines.py`/`test_harness.py`. All green individually and together with the full
  pre-existing suite (767 passed, 16 pre-existing skips).

## Next steps, in order

1. **Check whether the owner's IBEX runs (steps 8.6-10.5) have finished.** Ask, or look for new
   `results/runs/<stamp>_81cec63*/` directories containing `metrics_exp_b_{10,77}ghz.json` and,
   after the merge stage, `session_specific_{10,77}ghz.json`. If they haven't finished, there is
   genuinely nothing to do here yet except wait — do not re-derive, estimate, or guess numbers.
2. Once real results exist: **write per-step HISTORY.md entries for 8.6/9/10/10.5** — store
   rebuild+validate confirmation, the smoke pass (confirm no performance value surfaced), the
   actual full-cohort numbers per band (primary aggregate CI, paired Wilcoxon, per-session
   breakdown, `primary_viable`), and the variant's `completed_sessions`/per-session numbers —
   continuously as each resolves, per CLAUDE.md, not batched into one entry.
3. **Then, and only then, write SECOND_CHAPTER.md §6 (Exp A) and §7 (Exp B) TOGETHER.** This is
   an explicit standing owner decision from M7/M8 (see HISTORY.md's "M7 CLOSES" entry), not
   optional or reopenable without a new owner decision. §6's method/provenance subsections are
   already written; only its "Results" subsection and all of §7 are pending — fill both from the
   real numbers, in one pass. State the A-M8-1/A-M8-2 chronology plainly (both decided
   2026-07-27, *after* Exp A's full-cohort results were already visible — plan §0/Step 0 item 1,
   Step 0b, and resolved-review-log entries C3/C10 explain why this distinction is load-bearing
   and must not be elided or folded into "frozen before Exp A" language).
4. Do **not** reopen A-M8-1, A-M8-2, or any resolved plan comment (C1-C25) without a new,
   explicit owner decision — same standing rule as every prior milestone's frozen choices.
5. If a store `--validate`, a smoke run, or a full/variant run fails on IBEX: read the actual
   error against `plans/MILESTONE_8_PLAN.md` §5's 26 traps before guessing at a fix. Traps 20-26
   (the `record_run`/provenance contract) and 17/18 (`validate_store` enforcement) are the most
   IBEX-orchestration-specific; everything this session's own test suite could exercise for those
   paths is green, so a failure there most likely means something genuinely IBEX-specific (a
   stale store, a wrong commit checked out, a resource/time-limit issue) rather than a logic bug.

## Hard invariants (unchanged, never violate)

LOSO at subject level; fit-on-train-only at both CV levels; no test-set tuning; primary target
continuous Δm%, session-level headline; folds only from `splits.py`; tie-break only via
`select_candidate`; numpy backs all reported features; `protocol_freeze_guard` before every
fit/write; `tests/test_no_leakage.py` unchanged (confirmed `git diff --exit-code` clean as of
commit `81cec63`, M8's own step-4 diff included).

## Journal & hygiene

**HISTORY.md** newest-first, current through M8 step 8.5. **SECOND_CHAPTER.md** §0-§5 written,
§6 method/provenance written, §6 Results + all of §7 deliberately pending (see "Next steps" item
3 — do not write §6 in isolation). **MILESTONE_8_PLAN.md** `REVIEW_COMPLETE`, plan steps 1-8
implemented and committed, steps 8.6-11 not yet done (owner-run/IBEX-dependent). Branch
`v1_milestone8`, HEAD `81cec63` at the time this was written, working tree clean. Whether
`81cec63` has been pushed to `origin` is not verified from this session — the owner may have
synced it to IBEX by push, pull, or a copied-tree workflow (`scripts/ibex/README.md`'s
alternative for repos without git access on the cluster); check before assuming. Commit only when
the owner asks.
