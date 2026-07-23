# HANDOFF — resume point for a new chat (implementing milestone 5)

_Written 2026-07-23, after the milestone-5 plan was drafted, Codex-reviewed (24 comments,
all applied), and owner-approved, and the authoritative-document amendments (A-M5-1..4) were
applied. Purpose: let a fresh Claude Code session **start implementing milestone 5 — the
77 GHz front-end** without re-deriving context._

## TL;DR

**Milestone 5 is fully planned and approved; no M5 code exists yet.** You are on branch
**`v1_milestone_5`**. 396 tests pass (407 with `--realdata`; only T18 skipped) — that is the
M4 baseline; M5 adds to it. **Next: implement M5 per `plans/MILESTONE_5_PLAN.md` §1 build
sequence** (start at step 2 — step 0/step 1 are done: authoritative docs amended, plan +
HISTORY written).

## Read first (in this order)

1. `CLAUDE.md` / `AGENTS.md` — hard invariants, code style, journal + file-hygiene rules.
2. `plans/MILESTONE_5_PLAN.md` — **THE execution plan for M5; source of truth for this
   milestone.** §0 scope, §1 build sequence (the 12 steps), §2 per-file specs, §3 tests
   (T-IDs), §4 definition of done, §5 traps, §6 amendments.
3. `plans/implementation_plan.md` — the approved main design. For M5: the **Exp G** block
   (the 77 GHz chain, now amended by A-M5-3/A-M5-4), §"WST parameterization", §"Analysis
   unit", §"Compute / IBEX".
4. `HISTORY.md` — **newest entry only** (the M5-plan + decisions entry at the top).

## What M5 is

77 GHz promoted from fusion-only to a **full parallel primary arm** (Exps A–F run on both
bands; 10 GHz stays the sole headline; Exp G fusion retained). M5 builds the 77 GHz
front-end — **loader → QC → preprocessing → slow-time I/Q WST → cohort diagnostics** —
mirroring M1–M4 for band 2, stopping at features (no modelling). **First IBEX milestone.**

## Milestone numbering (renumbered — A-M5-2, already applied)

M5 = 77 GHz front-end (this). **M6 = config-freeze gate. M7 = LOSO harness + Exp A.** M8 =
Exp B. M9 = Exp C/D. M10 = fusion(G)/interp(E)/confound(F)/stats(H). M11 = figures.
(`implementation_plan.md` §Build order + all cross-refs and `ROADMAP.md` §7 amended; ROADMAP's
list runs one behind because it has no config-freeze milestone.)

## Hard invariants (never violate — a failing check stops the build)

- **LOSO** at subject level; no frame of a held-out subject in training, any session.
- **Fit-on-train-only**: every fitted transform inside the CV loop on training folds only.
- **No test-set tuning** (incl. QC thresholds — see the flatline decision below).
- **Primary target continuous** (Δm%); 5-class secondary, ordinal metrics.
- **`tests/test_no_leakage.py` byte-for-byte unmodified since M1** — M5 is per-frame and
  unfitted, so it does not touch it. DoD checks `git diff --exit-code f3fbade -- <file>`.
- **numpy kymatio backs ALL reported features**; torch only after the 77 GHz cross-backend
  test passes.

## Owner decisions locked (do NOT re-litigate)

1. **77 GHz = full parallel primary arm** (A-M5-1, applied to plan + ROADMAP).
2. **Flatline rule → outcome (b): a mechanism-corrected exact replacement of the *77 GHz*
   screen** (the 10 GHz screen is untouched, frozen since M2). The exact corrected rule is
   **specified at M5 step 6 from the ADC-quantisation mechanism (M2 single-file audit +
   physics), NEVER from full-cohort survival** (that would be cohort-level leakage). M5 thus
   proceeds to the cohort feature runs; (a) keep-frozen and (c) data-adaptive-in-CV remain
   documented fallbacks only. (A-M5-6.)
3. **A-M5-3 + A-M5-4 applied** to the Exp G chain: step (6) robust-standardizes real/imag
   **separately** (median/MAD, 10 GHz-consistent) before WST; the order-aware log applies to
   the **fused per-frame tensor** at step (8), before pooling, keeping the three-branch log
   axis (off / on+frozen-ε / on+tuned-ε).
4. **IBEX access confirmed working**; created at step 10.

## Build sequence (`plans/MILESTONE_5_PLAN.md` §1) — start at step 2

0/1 ✅ done (authoritative amendments applied; plan + HISTORY written). Then:
**2** config (`qc77`/`preprocess77`/`wst77` + `data_77ghz_dir`; T-C77) → **3** `loader_77ghz`
(T-L77) → **4** `manifest_77` (T-M77) → **5** `screens_77` + `axis_check_77` (T-Q77) →
**6** specify the corrected flatline rule (owner (b)) + a substep that revises the
step-2/5 artifacts and reruns T-C77/T-Q77 **before** step 7 → **7** cohort QC run →
**8** `pipeline_77` (T-P77) → **9** `extraction_77` (T-W77) + non-curated `--realdata` smoke
(T-R77) → **10** IBEX scaffolding (`configs/ibex.yaml`, `scripts/ibex/`) → **11** cohort
feature runs (IBEX array; blocked by step 6) → **12** close-out (A-M5-5..8, SECOND_CHAPTER §4,
DoD). Steps 8–10 run in parallel with 6–7.

## Key design (from the plan — don't re-derive)

- **Config**: three parallel top-level sections `qc77`/`preprocess77`/`wst77` + optional
  `paths.data_77ghz_dir`; 10 GHz YAML byte-untouched. Doppler tilings `Q=(8,4)/(6,4)/(4,2)` at
  20/40/60 ms, **code-frozen** (YAML override rejected). New YAML `configs/{data77,
  preprocess77,wst77,exp_77ghz}.yaml`; signed exponents (`2.0e+9`).
- **Loader** (`loader_77ghz`): h5py; asserts `(16,256,256,N)` **8-byte real float64** (rejects
  float32/compound/endianness); **whole-file read** (chunk layout `(16,4,1,125)` spans all
  frames) then `reverse_axes` → `[N,256,256,16]`. Promote `reverse_axes`/`to_numeric` from the
  audit.
- **Chain** (`pipeline_77` steps 1–5, `extraction_77` steps 6–10): MTI → Butterworth bandpass
  → Hann → 256-pt range FFT → crop bins 27..53; then per frame a batch of **432 complex
  series** (16 rx × 27 bins) → `scatter_frames([432,2,256])` → mean over bins → per-Rx →
  **Rx-fusion mean (primary)/median (secondary)** → order-log on the **fused** tensor →
  pool_stats → session concat mean+median. Bin/Rx loops **folded into the batch dim, never
  Python loops**. Geometry (T≈39/78/117, J≈6/7/7, pad) **MEASURED, never assumed**.
- **Axis check** (`axis_check_77`): runs once per file in the cohort QC run on the raw
  pre-MTI cube; **fails closed on any non-`ACCEPTED` verdict**. Every extraction/preprocess
  entrypoint + the smoke re-verify an `ACCEPTED` record keyed to `sha256`+`axis_spec_hash`, or
  run the check inline.
- **Reuse as-is**: `sessions.py`, `preprocess/{filters,reduce,standardize}.py`,
  `features/{wst,pooling}.py`, `provenance.record_run` (+ new `data_dir` param), and the
  imported `manifest._join_qc`/`eligible_frames`/`evaluable_subjects`.
- **CLIs**: `run_qc77` (two modes — authoritative under (a)/(b); characterization-only under
  (c)), `run_preprocess77` (single cohort job), `run_wst77` (sharded array + `--merge-shards`
  with fingerprint sidecars; `--smoke` non-curated mode).

## Open design gates (later — not M5-implementation blockers except step 6)

- **M5 step 6**: the exact mechanism-corrected 77 GHz flatline rule.
- **M6 config-freeze**: confirm the A-M4-7 third log branch (on+tuned-ε) enters the search
  space; freeze both bands' A–F design + the protocol-constant whitelist.

## Data / environment

- `data/77ghz/` — 80 files (`subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat`), ~285 MB each,
  22 GB. `data/10ghz/` present. **Env complete** (h5py since M2; scipy **pinned `<1.17`**;
  torch CPU) — no new deps for M5. `uv sync --frozen` on IBEX.
- **IBEX**: CPU, numpy backend (frozen policy); array over the 80 (subject,session) cells;
  `configs/ibex.yaml` is a paths overlay only. The owner runs IBEX commands (no ssh from
  Claude) — `scripts/ibex/README.md` must be self-contained.

## Traps already paid for (plan §5)

- Chunk layout spans all frames → whole-file reads (per-frame reads cost ~125× I/O).
- kymatio pad at n_in=256 **measured**, not assumed (M4 off-by-one lesson); border warning
  asserted present, never silenced.
- 27-bin loop is the hotspot in disguise — fold into batch dim (M4 `pool_stats` 54× lesson).
- YAML 1.1 signed-exponent trap; CRLF breaks sbatch (`.gitattributes: scripts/ibex/* eol=lf`).
- Flatline outcome can swing 77 GHz eligibility hard — the rule is fixed on mechanism grounds
  before the cohort run, never tuned to survival.

## Journal & hygiene

- **HISTORY.md** newest-first, an entry per resolved step (failures kept). **SECOND_CHAPTER.md
  §4 "77 GHz front-end"** at milestone close. **HANDOFF.md** — update only when asked.
  Superseded code/results → `archive/{code,results}/`, noted in HISTORY.
- Branches `v1_milestone_1..4` pushed; **`v1_milestone_5` current** (plan + amendments +
  HISTORY added; no M5 code yet); nothing merged to `main`. Commit only when the owner asks.
