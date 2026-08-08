# HANDOFF — resume point for a new chat (STEPS 1–10 DONE; next job: **step 11, the final commit + ONE store rebuild**)

_Written 2026-08-09. The next chat's job: **`plans/MILESTONE_10_PLAN.md` §4.2 step 11 — stamp
`REVISION` once, rebuild both feature stores once, validate them** — then steps 12–13. Read this
file, then the plan's §4.2 steps 11–14, §6's launch matrix (it is exact and ordered), and §1.3.
**Step 11 onward is IBEX work, not local work.** Steps 1–10 are complete, tested and committed;
do not re-litigate them._

## TL;DR

- **Branch `v1_milestone_10`, HEAD `11f327e`**, pushed, working tree clean except untracked
  `.codex/` (owner's tooling — leave it).
- **Steps 1–10 are DONE.** All eight milestone-10 code artifacts exist, are tested, and are
  committed: the Exp-A reference gate, multiplicity, the H robustness driver, Exp G, Exp E,
  Exp F, assembly + drivers, and the test/self-review/correction cycle.
- **The full suite is `1531 passed, 0 failed, 16 skipped` (22 min) — fully green.** There is no
  longer any "expected failure" footnote. **Any failure you see is real and is yours.**
- Real-data suite (post-correction): `1542 passed, 0 failed` — `--realdata` unskips the 16
  real-data tests and all pass. ~45 min.
- `tests/test_no_leakage.py` is byte-unchanged since M7. Keep it that way; keep
  `git diff --exit-code -- tests/test_no_leakage.py` as an acceptance step.
- **No store has been rebuilt yet. That is step 11 and it is the next thing that happens.**

## ⚠ The single most important fact for step 11

**Moving the commit invalidates BOTH feature stores** (`store._check_match` uses strict commit
equality). That is why every code change in steps 1–10 was deliberately batched ahead of this
point. Step 11 is: stamp `REVISION` **once**, rebuild **both** stores **once**, validate. Then
steps 12–13 must follow on that same commit — any further source edit re-invalidates the stores
and costs a full re-extraction.

- `REVISION` currently holds `f9dee54e0cef11c92f0d932d33a51710e098bd26` — **stale** (HEAD is
  `11f327e`). Step 11 restamps it: `git rev-parse HEAD > REVISION`.
- **IBEX is on `v1_milestone_9a` @ `3f465ab`** unless someone moved it. It needs
  `git fetch origin && git checkout v1_milestone_10 && git pull`.
- Both former dirty-tree offenders (`dcgm/`, `results/exploratory_frame_split/`) are gitignored,
  so `submit_ibex.sh` will not refuse.
- **Heavy work is always `sbatch`**, never a login-shell run.

## What steps 1–10 delivered (do not rebuild any of it)

| Step | Delivered |
|---|---|
| 1 | `eval/reference_gate.py`, `experiments/validate_exp_a_reference.py`, the **authoritative** `results/milestone10/reference_exp_a_manifest.json` (both bands, version-controlled) |
| 2 | Multiplicity through models/harness/providers/A-B-C orchestration, byte-neutral by default |
| 3 | `eval/robustness.py`, `experiments/run_robustness.py`, 3 IBEX artifacts (incl. the sharded path) |
| 4 | `eval/splits.py::selection_folds`, `eval/selection.py::select_alpha`, `eval/exp_g.py`, `experiments/run_fusion.py`, `scripts/ibex/run_exp_g.sbatch` |
| 5 | `eval/exp_e.py`, `experiments/run_interpretability.py`, `scripts/ibex/run_exp_e.sbatch` |
| 6 | `eval/exp_f.py`, `experiments/run_confound.py`, `scripts/ibex/run_exp_f.sbatch` |
| 7 | `eval/assembly.py`, `experiments/run_stats_assembly.py`, `scripts/ibex/run_stats_assembly.sbatch`, **`--run-dir-out` on all six drivers + `RUN_DIR_OUT` on all six wrappers** |
| 8–10 | Test gate, author self-review (4 findings, all fixed + regression coverage), retest |

## Step 11–13, concretely

**Step 11 (IBEX).** `git rev-parse HEAD > REVISION`, commit it, then rebuild both stores once
(`extract10.sbatch`, `extract77.sbatch`) and validate each with
`experiments/extract_features.py --validate`. §6's launch matrix has the exact commands, in
order — follow it rather than improvising.

**Step 12.** Full Exp A **and** Exp B, both bands, on the final stores/commit, each with
`RUN_DIR_OUT=results/milestone10/sources/<name>.txt`. Then
`validate_exp_a_reference.py --compare` against the committed manifest → writes
`results/milestone10/exp_a_sources.json`. **A mismatch stops the milestone** for scientific
review; it is never excused as byte-neutral drift.

**Step 13.** Local mechanism-only smokes first, then the full E/F/G and `R=200` robustness jobs.
Sizing is already decided and written into each header:

- **Exp E** — `run_exp_e.sbatch`, 16 cores / 1 h / 32 G. Measured: 742 paths (10 GHz) / 424
  (77 GHz), ~12.4 s and ~7.1 s per fold. Minutes, not hours.
- **Exp F** — `run_exp_f.sbatch`, 16 cores / 1 h / 32 G. Measured ~4 s per fold. **Requires
  `EXP_A_SOURCES=results/milestone10/exp_a_sources.json`** (step 12's output); the wrapper
  refuses to launch without it.
- **Exp G** — `run_exp_g.sbatch`, 32 cores / 24 h / 128 G (~192 core-hours, ~6 h wall). If the
  partition cannot give 32 cores, **halve the cores, not the wall**.
- **Robustness** — A and B: `run_robustness.sbatch` with `--cpus-per-task=64`. Exp C: the
  sharded array-plus-merge path (`submit_robustness_sharded.sh`), `ARRAY_TIME` sized from one
  measured shard. `R=200` is never lowered to fit a wall.

**Step 14.** Assembly (two calls: `VALIDATE_ONLY=1`, then `0`), and **only then**
`SECOND_CHAPTER.md` §9.

## Things that will bite you

- **`run_exp_f.sbatch` cannot run until step 12 exists.** That is by design, not a bug.
- **Exp F has no local real-data path either** — it needs an approved `exp_a_sources.json`.
- **The 77 GHz half cannot be smoked locally**: the local 77 GHz store is 1 of 72 sessions.
- **The local 10 GHz store was built at `dab8f708`**, not the analysis commit, so
  `validate_store` refuses it — a local real-data smoke fails on *provenance*, not mechanism.
  After step 11 this stops being true on IBEX.
- **`core.autocrlf=true` here.** A byte-exact reference artifact needs a `-text` entry in
  `.gitattributes`; `scripts/ibex/*` is already `text eol=lf`.
- **Shell tests: never pass a script to bash as an argument.** `shutil.which("bash")` resolves
  to the WindowsApps **WSL stub**, which swallows `bash -c` output and cannot read a `C:\` path.
  Use `test_exp_b_ibex_scripts.run_bash` / `bash_syntax_check`, which pipe bytes on stdin. (This
  corrects the old "Git Bash eats backslashes" note, which named the symptom, not the cause.)

## The twelve amendments — all accepted, none open

A-M10-1..6 at plan acceptance; **A-M10-7..12 during implementation**. All are in force and the
plan text is internally consistent — **trust its current text over any memory of it**.

| ID | One line |
|---|---|
| A-M10-7 | Reference Exp-A runs are the `*_3f465abc` pair. Provenance only; no estimand changed. |
| A-M10-8 | Multiplicity is contiguous row duplication for every family, not `sample_weight`. |
| A-M10-9 | Exp C arm (b) keeps the frozen multiclass O-M9-7 weights. |
| A-M10-10 | H's selection table records the **selected** candidate; fit audit records outer-level fits. |
| A-M10-11 | G's `fit_audit_g.csv` records the chain behind every **reported prediction**. |
| **A-M10-12** | **No independent code review** (owner, 2026-08-08). Step 9 became an **author self-review**; §7's criterion amended and the absence disclosed as a limitation. **This is the one amendment that WEAKENS a criterion — it must reach the chapter.** |

## Chapter state

`SECOND_CHAPTER.md` §0–§8 complete. §9 is still the pre-registration stub: it records that **no
milestone-10 result exists**, discloses A-M10-1..12 with their true chronology, and carries the
A-M10-12 limitation ("extensively tested and author-reviewed, **not** peer-reviewed"). It is
written in full **only after** verified full-cohort M10 artifacts exist — i.e. after step 13.

## Hard invariants (unchanged, never violate)

LOSO at subject level for every reported result; fit-on-train-only at every CV level; no
test-set tuning; primary target continuous Δm%; folds only from `eval/splits.py`; tie-breaks only
via `eval/selection.py`; `protocol_freeze_guard` before every fit/write; E/F/G/H are entirely
CPU. Do not report frame-level accuracy as a headline, do not claim causal isolation of hydration
from time of day, do not overclaim clinical readiness, and **do not tune E/F/G/H toward a more
favourable result because A–D came out negative**. Attribution is not causality; fusion is not
required to beat 10 GHz; a non-significant or inconclusive result is a valid scientific outcome.
