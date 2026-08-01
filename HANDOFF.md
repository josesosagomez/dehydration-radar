# HANDOFF — resume point for a new chat (M9 step 10 CLOSED; step 11 starts now)

_Written 2026-08-01, closing a step-10 chat. The next chat's job: **continue
`plans/MILESTONE_9_PLAN.md` at step 11** (§1 build sequence). Read the plan's §1/§2/§4/§5 for
whatever step you're on — do not read the whole 1474-line file. For anything below that needs
more detail than fits here, the answer is in HISTORY.md, not memory: search it, don't guess._

## TL;DR

- **Step 10 is CLOSED.** D1 (suite), D2 (pins), D6 (both v2 stores rebuilt + validated), D7
  (every mechanism-only smoke green, both bands) all satisfied. Full detail, including the
  measured GPU wall-times, in HISTORY.md's newest entry ("M9 step 10 CLOSED").
- **Current analysis commit: `f9dee54e0cef11c92f0d932d33a51710e098bd26`.** Branch
  `v1_milestone9`, working tree clean. This is what both IBEX stores attest and what every run
  from here must match — `REVISION` at the IBEX repo root must read exactly this hash before
  anything is submitted (`cat REVISION`, gitignored, not part of the checkout).
- **Two real fixes landed at step 10, both load-bearing for everything downstream:**
  1. `run.seed_set` now legally accepts `[1]` (the smoke overlay, `configs/smoke.yaml`) *or* the
     frozen `[1,2,3,4,5]` — pinned to the literal value, not a length rule.
  2. `torch` is pinned to the **cu126** CUDA variant on linux (`pyproject.toml` /
     `tool.uv.sources`), because IBEX's driver is 570.86.15 = CUDA 12.8 and PyPI's default
     linux wheel is cu130 — confirmed dead on a real GPU node before being fixed. Verified: only
     `torch` and CUDA runtime packages moved in the relock; `numpy`/`scipy`/`kymatio` (the
     WST-feature-affecting stack) are byte-identical. `configs/gpu.yaml` (`run.device: cuda`)
     must be loaded by **every** run-group stage uniformly (init/fold/merge) — it's inside
     `config_to_dict`, hence hashed into `config_fingerprint`, so a partial overlay fails
     `_validate_group_lineage` on `config_hash` after init already succeeded. Already wired into
     `run_exp_d_cnn.sbatch`'s `GROUP_ARGS`; nothing to redo.
  3. Confirmed on real GPU hardware: `2.13.0+cu126 True` inside `srun --gres=gpu:1`.
- **GPU fold wall-times measured** (real hardware, frozen 5-seed set, 10 GHz, 1 fold each):
  `cnn1d_raw` 1:19:34, `cnn1d_matched` 0:57:05, `spec2d_raw` 0:51:23, `spec2d_matched` 0:46:51.
  **`ARRAY_TIME` for step 13** (measurement + margin, same value reused for 77 GHz — disclosed
  assumption, not a second measurement): `cnn1d_raw` → `03:00:00`; the other three →
  `02:00:00`. If a 77 GHz fold times out, this reuse is what to revisit first.
- **Not touched this session, deliberately:** `SECOND_CHAPTER.md` §8 stays empty — CLAUDE.md's
  rule is "write at each milestone[completion]," and M9 is mid-build (step 10 of 15), not done.
  `AGENTS.md` / `ROADMAP.md` are static reference and had nothing to update.

## Read first (in this order)

1. HISTORY.md's newest entry (step 10 close) — the smoke matrix, the two fixes, the timing
   table. Go further back only for a specific earlier value (e.g. why cu126 over cu128/cu129
   — that reasoning is in the "part 2" entry just below the closing one).
2. `plans/MILESTONE_9_PLAN.md` §1 (build sequence) rows 11-15 — your remaining steps. §2.8/§2.11
   if you touch `exp_d.py` or the entrypoints; §4 D8-D12; §5 traps 10, 11, 17, 18, 20 (GPU
   nondeterminism, the GPU-authorization split, comparing against a drifted Exp A, store-rebuild
   ordering, frame-split leaking into §8).
3. `CLAUDE.md` — invariants, code style, journal rules. Unchanged from every prior handoff.

## Working tree / git state at handoff

Branch `v1_milestone9` at `f9dee54e0cef11c92f0d932d33a51710e098bd26`, tree clean, nothing
staged. `REVISION` exists locally (gitignored) and must be copied to IBEX with this exact hash
before any further job — it was already updated there for step 10's smokes, so this is a
statement of current fact, not a pending action, *unless* you land more commits, in which case
re-stamp and re-copy before submitting anything (trap 18, no exceptions).

Both v2 stores are rebuilt and validated at this commit: 10 GHz confirmed by a standalone
`--validate` (expect 73 sessions); 77 GHz confirmed transitively — every 77 GHz smoke in step 10
calls `store_mod.validate_store` internally and fails closed, and all of them succeeded (expect
72 sessions on a standalone `--validate` if you want the direct confirmation).

**`results/runs/` was wholesale-replaced with an IBEX copy after step 10** (owner sync, to get
the 16 smoke run dirs locally) — this deleted the two M7 reference dirs (`*_f36c4fb2`, step 12's
O-M9-5 input) from the local tree, since `results/runs/` is gitignored and IBEX's own copy of
them was apparently already gone/never-synced. **Recovered from the Windows Recycle Bin this
session** (both were soft-deleted, not purged) and verified byte-identical to the values this
project already had on record (`predictions_10ghz.csv` sha256 `4bd21201...`, `predictions_77ghz.csv`
sha256 `c8000...`, both `provenance.json` reading `git.commit = f36c4fb2428127f590c415d0799fe677faa12c14`).
Both dirs are back at `results/runs/20260727T111437230187Z_f36c4fb2/` (10 GHz) and
`results/runs/20260727T115046533408Z_f36c4fb2/` (77 GHz). **Confirmed by the owner: both are
also present on IBEX's own `results/runs/`** — the local wholesale-replace simply didn't reach
that path; nothing was actually lost on IBEX. No open question here, just a reminder for any
future local-runs sync: pull specific dirs rather than replacing the whole tree, since local
`results/runs/` is gitignored with no safety net beyond whatever Windows keeps.

## Next steps, in order (plan §1 rows 11-15 are authoritative)

### Step 11 — full-cohort Exp C, both bands, IBEX CPU

```bash
BAND=10ghz MODE=full sbatch scripts/ibex/run_exp_c.sbatch
BAND=77ghz MODE=full sbatch scripts/ibex/run_exp_c.sbatch
```
No owner pause (Exp C was frozen before Exp A/B's results, same C13 logic as M8). Outputs:
metrics/predictions/selection/confusion artifacts under a new `results/runs/<stamp>_f9dee54e/`.

### Step 12 — Exp A re-run + bit-identity assert (O-M9-5) — **a real gate, check it before step 13**

```bash
BAND=10ghz MODE=full sbatch scripts/ibex/run_exp_a.sbatch
BAND=77ghz MODE=full sbatch scripts/ibex/run_exp_a.sbatch
```
Each job's stdout prints `provenance: <run_dir>/provenance.json` as its last relevant line —
that `<run_dir>` is what feeds the check below and, later, `--exp-a-run-dir` for comparisons.

**Verify bit-identity immediately**, per band, rather than waiting for step 13's comparisons to
discover a drift after spending the GPU arrays. `load_exp_a_radar` is a real function, callable
standalone — it needs no other Exp D family to exist yet:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from dehyd.eval.exp_d import load_exp_a_radar
load_exp_a_radar('10ghz', '<step-12 10ghz run_dir>', '<M7 10ghz reference run_dir>',
                 analysis_commit='f9dee54e0cef11c92f0d932d33a51710e098bd26')
print('10ghz bit-identical: OK')
"
# repeat for 77ghz
```

The M7 reference run dirs are `results/runs/*_f36c4fb2/` (locally: the `...111437230187Z...`
one has `predictions_10ghz.csv`, the `...115046533408Z...` one has `predictions_77ghz.csv`).
**Confirm these same two directories exist on IBEX** before running the check there — they are
M7-era artifacts, not something this session recreated, so their presence on the IBEX result
tree needs a quick `ls`, not an assumption. A mismatch here is **milestone-stopping** (trap 17):
escalate to the owner, do not compare against the fresh predictions instead, and do not let
step 13 proceed until resolved.

### Step 13 — Exp D: cheap baselines full-cohort, then the 8 CNN fold-array groups, then comparisons

```bash
# cheap, both bands
BAND=10ghz MODE=full FAMILY=cheap sbatch scripts/ibex/run_exp_d_cheap.sbatch
BAND=77ghz MODE=full FAMILY=cheap sbatch scripts/ibex/run_exp_d_cheap.sbatch

# the 8 fold-array groups (init -> 16-task GPU array -> merge, each), using the measured/sized
# ARRAY_TIME above; submit_exp_d_cnn.sh already loads configs/gpu.yaml uniformly, nothing else
# to pass. Leave ARRAY_SPEC unset (defaults to the full 1-16).
FAMILY=cnn1d_raw      BAND=10ghz ARRAY_TIME=03:00:00 bash scripts/ibex/submit_exp_d_cnn.sh
FAMILY=cnn1d_matched  BAND=10ghz ARRAY_TIME=02:00:00 bash scripts/ibex/submit_exp_d_cnn.sh
FAMILY=spec2d_raw     BAND=10ghz ARRAY_TIME=02:00:00 bash scripts/ibex/submit_exp_d_cnn.sh
FAMILY=spec2d_matched BAND=10ghz ARRAY_TIME=02:00:00 bash scripts/ibex/submit_exp_d_cnn.sh
FAMILY=cnn1d_raw      BAND=77ghz ARRAY_TIME=03:00:00 bash scripts/ibex/submit_exp_d_cnn.sh
FAMILY=cnn1d_matched  BAND=77ghz ARRAY_TIME=02:00:00 bash scripts/ibex/submit_exp_d_cnn.sh
FAMILY=spec2d_raw     BAND=77ghz ARRAY_TIME=02:00:00 bash scripts/ibex/submit_exp_d_cnn.sh
FAMILY=spec2d_matched BAND=77ghz ARRAY_TIME=02:00:00 bash scripts/ibex/submit_exp_d_cnn.sh
```
Each `submit_exp_d_cnn.sh` call blocks on its own `init` stage then returns (array + merge run
async) — the 8 calls can be fired one after another without waiting between them. Merge output
gives `completed_folds`; every family needs the full selectable-fold set before comparisons.

**Comparisons come last**, once cheap-full and all 8 merges are complete — collect each job's
printed run dir (they don't exist yet, so don't assemble this command until you have them):
```bash
sbatch --export=ALL,BAND=10ghz,MODE=full,FAMILY=comparisons,\
EXP_A_RUN_DIR=<step-12 dir>,M7_REFERENCE_DIR=<M7 10ghz dir>,\
FAMILY_RUN_DIRS="physics=<P> session_index=<S> cnn1d_raw=<Q> cnn1d_matched=<R> spec2d_raw=<T> spec2d_matched=<U>" \
scripts/ibex/run_exp_d_cheap.sbatch
```
(repeat for 77 GHz). `run_exp_d_cheap.sbatch`'s `comparisons` case is what dispatches this — see
its script body for the exact env-var contract if this shorthand doesn't match at the time.

### Step 14 — the exploratory frame-split (16 runs), after every LOSO result above exists

Both Exp C arms x 2 bands (classical, CPU) + all six Exp D families x 2 bands (CNNs as one
small GPU job, physics/session-index CPU). §2.10's modal-config reduction reads the selection
tables the steps above just produced — it cannot run before them. Output only under
`results/exploratory_frame_split/`, tagged filenames, `never_report` markers — never
`results/runs/`.

### Step 15 — journal

HISTORY.md entries per resolved step (already the practice). `SECOND_CHAPTER.md` §8 written
from the real full LOSO results once steps 11-13 are complete, disclosing every A-M9/O-M9
completion's true chronology. The frame-split (step 14) appears **nowhere** in §8.

## Hard invariants (unchanged, never violate)

LOSO at subject level for every reported result; fit-on-train-only at both CV levels; no
test-set tuning; primary target continuous Δm%; ordinal metrics only for the 5-class task;
folds only from `splits.py`; tie-breaks only via `eval/selection.py`; numpy backs all reported
WST features (GPU is authorized only for the Exp D DL baselines, and is now confirmed WORKING
on IBEX, not merely permitted); `protocol_freeze_guard` before every fit/write;
`tests/test_no_leakage.py` frozen (`git diff --exit-code` is an acceptance step). Bit-identity
claims are CPU-scoped for everything except the one deliberate CPU-vs-CPU check in step 12 — GPU
training is never claimed bit-deterministic. The frame-split is the one sanctioned exception to
the reporting protocol: in addition to LOSO, structurally quarantined, never reported, absent
from §8.

## A process trap worth carrying forward

Stamping `REVISION` for an IBEX hand-off gives `_revision_file_commit()` something real to
find, which broke `test_provenance.py::test_git_degrades_to_none_without_env` (fixed this
session, in that one test, not the shared fixture — see HISTORY). If a "no git, no env" test
ever fails right after a `REVISION` stamp, check this first before assuming a real regression.
