# HANDOFF — resume point for a new chat (starting milestone 1)

_Written 2026-07-21. Purpose: let a fresh Claude Code session start implementing
**milestone 1** without re-deriving context._

## TL;DR
Planning is **done and locked**. The full spec is `plans/implementation_plan.md`
(hardened over 7 rounds of independent review — reviewer's final word:
"implementation-ready"). No project code exists yet. Next action: **build milestone 1**
(scaffold + config + green `tests/test_no_leakage.py`) before any modeling.

## Read first (in this order)
1. `CLAUDE.md` / `AGENTS.md` — hard invariants, style, journal + file-hygiene rules.
2. `ROADMAP.md` — the study spec (§1 invariants, §7 milestones).
3. `plans/implementation_plan.md` — **the detailed design to implement.** This is the
   source of truth for every decision below; when in doubt, follow it.
4. `HISTORY.md` — newest entry has the planning record + verified data facts.

## Hard invariants (never violate — a failing check should stop the build)
- **LOSO**: splits at the subject level; no frame of any session from a held-out
  subject in training. No frame-level random splitting as an evaluation protocol.
- **Fit-on-train-only**: every fitted transform (scaler/PCA/selector/class weights/CNN
  norm/early-stopping) fit inside the CV loop on training folds only — sklearn **and**
  torch paths.
- **No test-set tuning**: tilings/hyperparameters/thresholds via nested CV or held-out
  subject validation, never chosen on test subjects.
- **Primary target continuous** (Δm% fluid loss); 5-class secondary, ordinal metrics.
- Keep `tests/test_no_leakage.py` green at all times.

## Verified data facts (already confirmed — don't re-assume from the paper)
- **10 GHz** `data/10ghz/subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat` (80 files): MAT
  v5, `scipy.io.loadmat`. Var `framesRadar` = double `[534 fast × 20 chirps × 100
  frames]`, complex, **loads as complex128** (stored int16 on disk). Ignore
  `framesRadarIQ`. One file = one subject/session = 100 frames.
- **77 GHz**: MAT v7.3/HDF5, ~285 MB each, needs `h5py`. Deferred; a **minimal audit**
  happens at milestone 2 (installs h5py, confirms dtype/shape/complex, raw-axis check).
- **Ground truth** `data/weight/metadata_subjects_info.xlsx` sheet `MetaData` rows 3–18:
  two-row merged header → parse by **fixed cell addresses**. Cols E–I = weights
  8am→4pm. Signed `Δm% = (m(s)−m(S0))/m(S0)×100` (negative = loss). Cross-check S4 vs
  col J (signed kg, ±0.05 kg) and col K (positive % text, ±0.05%).
- **Subject identity** confirmed by owner: radar `subject_N` = workbook "Subject N"
  (old MATLAB 5–20 renumbered to 1–16, same subjects/order).
- Radar params (reference): fs=520834 Hz, B=500e6, Tchirp=1024e-6.

## Milestone 1 — the task for the new chat
Deliverables (see plan "Repo structure" + "Build order" §1 and the LOSO/no-leakage
section):
1. **Env**: `pyproject.toml` + lockfile via **uv**, Python 3.11+, pinned. Deps: numpy,
   scipy, pandas, openpyxl, PyYAML, scikit-learn, kymatio, pytest (torch/h5py can wait).
   NOTE: scipy/h5py are **not installed** in the current shell — the env must be created.
2. **`src/dehyd/config.py`** — load/validate YAML, resolve seeds & device.
3. **`src/dehyd/data/ground_truth.py`** — fixed-cell xlsx parse → signed Δm%, 5-class
   label, covariates; the two sign-aware cross-checks; fail loudly on disagreement.
4. **`src/dehyd/data/manifest.py`** — frame index table (subject, session, frame_idx,
   label); **fails on any missing / duplicate / unmatched** file↔weight-row record.
5. **`src/dehyd/eval/splits.py`** — the **single** source of folds: nested-LOSO API
   yielding `(train_subjects, val_subjects, test_subject)` **subject id sets**; adaptive
   inner `GroupKFold(min(5, n_train))`, ≥3 training subjects required.
6. **`src/dehyd/provenance.py`** — per-run: raw-file hashes, resolved config, fold
   manifest, versions, git rev, device, seed(s), (IBEX) Slurm ID.
7. **`tests/test_no_leakage.py`** — asserts (a) train/val/test subject sets pairwise
   disjoint; (b) every frame → one subject, no held-out subject's frames in training;
   (c) the strong mutation property test (see plan): mutating outer-test features/labels
   leaves selected config, inner scores, epoch budget, fitted params, training preds &
   model params bit-identical — only the held-out score may change. Runs **after
   eligibility is frozen**, eligibility-preserving mutations, **deterministic CPU
   fixture**.
8. `configs/` scaffolding (data/preprocess/wst/exp YAMLs), `experiments/` entry-point
   stubs. `archive/{code,results}/` already exist.
9. Loader `loader_10ghz.py` can be minimal here or lead into milestone 2 — follow the
   plan's milestone split (loader + QC screens is milestone 2).

**Definition of done for M1:** `pytest` green on `test_no_leakage.py` (+ loader/manifest/
ground-truth/metrics tests as they land), manifest builds and validates on the real 80
files, and nested-LOSO splits are produced only by `eval/splits.py`.

## Do NOT re-litigate (decided over 7 review rounds — in the plan)
- MATLAB is **reference-only**; Python is the sole source of reported numbers.
- Analysis unit is **session-level** (aggregate frames → 1 vector/session; concat
  mean+median). Per-frame is diagnostic only, never headline / never frame-IID CIs.
- Scoring counts use **N_eval**; session eligibility `≥ ceil(0.5×actual_frame_count)`.
- Departures from reference: median/MAD standardize; range gate = config (default
  1–2 m); order-aware WST log (`log(S+ε)` orders 1–2, ε=1e-6; order 0 linear);
  EdgeTrim=32 after reduction.
- Stats: subject-cluster bootstrap B=10000, seeds collapsed (metric-type-aware), CIs/
  p-values labeled conditional/exploratory.
- 77 GHz primary = slow-time (Doppler) **I/Q** WST, **per-Rx → feature-space** fusion.
If you think one of these is wrong, raise it explicitly — don't silently change it.

## Environment / compute
- Local (this machine, Windows, git-bash + PowerShell): scaffolding, CPU smoke tests
  (**≥6-subject** subset so nested CV actually runs), all classical models, stats.
- **IBEX** (KAUST Slurm, GPU): DL baselines / any NN as `sbatch` jobs under
  `scripts/ibex/`; same code, config-only differences (device/epochs/subset). No GPU
  training in interactive runs.

## Journal & hygiene (keep doing)
- **HISTORY.md**: append an entry per resolved attempt (what/why/params), newest-first,
  failures kept. Log each reference-departure with its reason.
- **SECOND_CHAPTER.md**: fill the relevant section as each milestone closes.
- **HANDOFF.md**: update **only when asked**.
- Superseded/broken code or stale results → `archive/{code,results}/`, noted in HISTORY.
  Valid negative results / ablations are current results — they stay in `results/`.

## Open items (none blocking M1)
- `h5py` install + 77 GHz axis confirmation → milestone 2 (not needed for M1).
- `AGENTS.md` is a parallel context file that differs slightly from `CLAUDE.md`; both
  encode the same invariants — follow them, no action needed.
