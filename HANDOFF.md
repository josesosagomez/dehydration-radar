# HANDOFF — resume point for a new chat (M9 plan REVIEWED and FINAL; implementation starts now)

_Written 2026-07-30, closing the M9 plan-review chat. The next chat's job is: **implement
milestone 9 by following `plans/MILESTONE_9_PLAN.md`**, starting at its step 0.5. The plan is
the specification — read it in full before touching source, and do not re-litigate its
decisions. The review loop is DONE (24 comments, all applied, no open threads); there is no
reviewer to talk to and nothing left to adjudicate._

## TL;DR

- **M8 is fully closed.** Negative headline in both experiments, both bands. The authoritative
  write-up is `SECOND_CHAPTER.md` §6-§7 — never re-derive those numbers from memory.
- **`plans/MILESTONE_9_PLAN.md` is final** (still untracked). It covers Exp C (ordinal 5-class,
  both bands), Exp D (raw/matched 1D-CNN, raw/matched spectrogram 2D-CNN, physics power-ratio,
  session-index + the frozen comparison statistics), store schema v2, and the owner's sanctioned
  exploratory frame-split (deliberately leaky, never reported). Review block at the end reads
  `REVIEW-STATUS: REVIEW_COMPLETE`, turn 9, reviewer `NO MORE COMMENTS`.
- **Every decision is resolved. Nothing is pending.** Step 0 (A-M9-1, frame-split = modal-config
  refit, CNN = 16-task GPU fold-array), Step 0b (O-M9-1..7), and the review-derived **O-M9-8
  (owner-approved 2026-07-30 as option 8a)**. The one optional item the owner has NOT authorized:
  Exp A's radar regressor in the exploratory frame split — excluded by design, do not add it.
- **All of those were decided AFTER Exp A/B's negative results were visible.** The plan discloses
  that chronology deliberately (the A-M8-1 discipline). Never fold it into "frozen before
  results" — not in code comments, not in the journal, not in SECOND_CHAPTER §8.
- **Step 0.5 has NOT been done.** It is the first action of the next chat: propagate A-M9-1 and
  O-M9-1..8 into `plans/implementation_plan.md` before any source. The plan's step-0.5 row names
  the exact destination section for each item.

## Read first (in this order)

1. `plans/MILESTONE_9_PLAN.md` — the whole thing. §1 is the build order, §2 the per-file specs
   (signatures are the contract), §3 the test groups, §4 the definition of done, §5 the 20 known
   traps, §6 the amendments/completions with their chronology.
2. `CLAUDE.md` — invariants, code style (readable research code, no premature abstraction),
   journal rules.
3. HISTORY.md — the two newest entries (the review loop + O-M9-8; and the M9 planning entry).
   Nothing older unless you need a specific value.
4. `plans/implementation_plan.md` §C / §D / §Statistics — only the sections you are implementing,
   as you reach them. The plan cites exact line ranges throughout.

## The review changed the plan substantively — do not implement from a stale mental model

The 24 applied comments are logged in the plan's `### Resolved` section and summarized in
HISTORY.md. The ones most likely to bite if you skim:

- **Ordinal viability predicate** is the constant `ORDINAL_CLASSES = (0,1,2,3,4)`, never
  `set(y[:, 1])` — bundles carry all session rows, so a bundle-relative check would let held-out
  labels decide which cells get fit.
- **Ordinal inner-fold aggregation** is `nanmean`/`nanstd` over *evaluable* inner folds, computed
  in `exp_c`, with `n_evaluable_inner_folds` recorded; the harness's own `np.mean`/`np.std` stay
  byte-unchanged for Exp A/B.
- **`assert_exp_c_fit_authorized`** is a real new guard (family id, base-family mapping, grid
  membership, wrapper constants, Frank-Hall C/impl/max_iter) — the existing
  `protocol_freeze_guard` validates only keys *present* in `active` and binds nothing about the
  candidate.
- **QWK** is undefined only on empty input or zero expected disagreement (O-M9-8 / 8a), and the
  run must report `n_single_class_truth_val_folds` and `n_qwk_nan`.
- **Spectrograms** are literal log-magnitude `log(|STFT| + finfo.tiny)`; `SpectrogramNorm` keeps
  `[C, F]` statistics reduced over frames × time with `scale = where(std == 0, 1.0, std)`; the two
  *raw* branches bypass robust standardization while both *matched* branches consume the
  matched-preprocessed signal (10 GHz already standardized in the store, 77 GHz at load).
- **Composite baseline** splices per-subject seed-averaged metrics, never predictions.
- **CNN loader contract** is pinned: `replacement=True`, `shuffle=False`, `drop_last=True`,
  `num_workers=0`, one epoch = `floor(len(train)/16)` steps, per-fit generator.
- **Frame split** recomputes tuned-ε from raw tensors on training frames only, preserving the
  frozen subject-balanced hierarchy; it never calls `record_run`, writes through an allowlist, and
  hashes frame order + fold assignment + the source LOSO artifact.
- **Shard merge** validates against `expected_test_rows_by_fold` (frame *and* session hashes +
  seed cross-product), and every Exp D family emits a four-artifact set.

## Facts verified against source (the plan depends on these; re-verify before contradicting one)

- `protocol_freeze_guard` (`src/dehyd/features/protocol_freeze.py:116-168`) checks only the keys
  present in `active`, against the frozen five-family whitelist → `ord_b_frank_hall` has no legal
  `model_family` value → the §2.6 three-check design.
- `harness.require_complete_active` (`:163-183`) demands *exactly* the band key set, so arm (b)
  needs exp_c's own `REQUIRED_ACTIVE_KEYS_C`.
- `harness._score_candidates_on_fold` (`:297-340`) applies the row mask only after `data_for`;
  `CandidateScore` uses `np.mean`/`np.std` over all inner folds.
- `_prelog_scale` (`features/extraction.py:76-92`) is a **per-session** median over that session's
  frames, stored at `store.py:246` — the frame-split leak C10 caught.
- `preprocess_frame` ends in `to_channels(trimmed, channel, pre.standardize)`
  (`preprocess/pipeline.py:73`) → the stored 10 GHz `sig__matched_iq` is already
  robust-standardized.
- `provenance.record_run` (`:190-239`) always creates `results/runs/<stamp>_<rev>/` and stores only
  cohort totals — hence the public `build_provenance_payload` extraction and the per-fold census.
- `torch_fit._normalize_stats` (`:86-91`) uses `where(std == 0, 1.0, std)` — the precedent
  `SpectrogramNorm` reuses.
- `tests/test_selection.py:79` asserts exact dict equality on `SIMPLICITY_RANK` → that assertion
  gets updated (only `tests/test_no_leakage.py` is frozen).
- `BaselineConfig` (incl. `max_epochs: 200`) is in `M6_SECTIONS` → a smoke config CANNOT override
  epochs; smokes differ only by subset / `seed_set=[1]` / device.
- Store commit-match (`store.py::_check_match`) forces both stores rebuilt at the M9 commit
  regardless — the basis of O-M9-5 and of schema v2 riding along.
- The `exp_b.py` deployed on IBEX is still the `30c6d907` version (no heartbeat there). Unchanged,
  nothing to do.

## Working tree / git state at handoff

Branch **`v1_milestone9`**, already created off `v1_milestone8` and checked out; both point at
`b6e7ba1`, and nothing has been committed on it yet. `SECOND_CHAPTER.md` and
`plans/implementation_plan.md` are clean (they landed in `c7b6b83`). Uncommitted:
`plans/MILESTONE_9_PLAN.md` (untracked), `HISTORY.md`, `HANDOFF.md`, `.gitignore` (owner's
`review*/` line), and the owner's deletion of `plans/review_prompt_{claude,codex}.md` — the owner
erased those deliberately as redundant; do not restore or archive them. **Commit only on explicit
owner request** — the next gate is the pre-implementation commit carrying this plan, the journal
files, and step 0.5's `implementation_plan.md` edits.

## Next steps, in order (plan §1 is authoritative)

1. **Step 0.5** — propagate A-M9-1 + O-M9-1..8 into `plans/implementation_plan.md`, each to the
   section named in the plan's step-0.5 row (O-M9-3 and O-M9-5 → §Statistics, O-M9-7 → §C, etc.),
   stating the post-A/B chronology and the computation-affecting label. Then ask the owner for the
   pre-implementation commit.
2. **Step 1** — pins: `test_m8_pin.py` re-run plus a new `_viability_reason` / byte-trace pin
   (step 4 claims byte-neutrality and is unverifiable without them).
3. **Steps 2-9** — `metrics.py` → `models/ordinal.py` + dispatch + selection → the one
   `harness.py` edit (own commit, between two green states) → `fold_parallel.py` extraction →
   `exp_c.py` → `models/cnn.py` + `exp_d.py` torch path → `exp_d.py` remainder → store v2 +
   entrypoints + sbatch + `frame_split.py`. Tests alongside each spec; a step is not done until
   its acceptance tests are green. HISTORY.md entry per resolved step, not batched.
4. **9.5 / 9.6** — owner-triggered clean commit, then rebuild + `--validate` both stores from it.
5. **10** — local synthetic suite + mechanism-only smokes (Exp C both bands; every Exp D family
   both bands; one GPU array-task smoke per CNN family, to size `ARRAY_TIME` from measurement).
6. **11-14** — full-cohort Exp C; Exp A re-run + bit-identity assert (a mismatch STOPS the
   milestone, escalate — do not compare against the fresh predictions instead); cheap baselines;
   8 CNN fold-array groups; comparisons; then the 16-run exploratory frame split.
7. **15** — SECOND_CHAPTER §8 from the real LOSO results, disclosing every A-M9/O-M9 completion
   with its true chronology. The frame-split appears nowhere in §8.

## Hard invariants (unchanged, never violate)

LOSO at subject level for every REPORTED result; fit-on-train-only at both CV levels; no test-set
tuning; primary target continuous Δm%; ordinal metrics only for the 5-class task; folds only from
`splits.py`; tie-breaks only via `eval/selection.py`; numpy backs all reported features (GPU is
authorized ONLY for the Exp D DL baselines, `implementation_plan.md:1326-1329`);
`protocol_freeze_guard` before every fit/write; `tests/test_no_leakage.py` frozen (`git diff
--exit-code` is an acceptance step). Bit-identity claims are CPU-scoped — GPU training is not
bit-deterministic and is covered by per-seed reporting plus CPU-fixture contract tests. The one
sanctioned exception to the reporting protocol: the exploratory frame-split — in ADDITION to LOSO,
structurally quarantined, never a substitute, never reported, absent from §8.

## Journal & hygiene

HISTORY.md is current through the 2026-07-30 review-loop entry (newest-first; the O-M9-8 decision
is recorded inside it). SECOND_CHAPTER.md §0-§7 complete and final; §8 is written only from real M9
results. `MILESTONE_8_PLAN.md` and now `MILESTONE_9_PLAN.md` are both REVIEW_COMPLETE. Superseded
code/results move to `archive/code/` or `archive/results/`, noted in HISTORY — never left in `src/`
and never silently deleted. This HANDOFF was refreshed on explicit owner request (chat close,
2026-07-30).
