# HANDOFF — resume point for a new chat (starting milestone 5)

_Written 2026-07-23, after milestone 4 was completed (but NOT yet committed). Purpose: let
a fresh Claude Code session start **milestone 5 — the config-freeze gate** without
re-deriving context._

## TL;DR

**Milestone 4 (WST features) is DONE and green, but NOT COMMITTED.** The last commit is
`a27d8ce` (M3); all M4 work sits in the working tree on branch **`v1_milestone_4`**. 396
tests pass (407 with `--realdata`; only T18 skipped). **First order of business is an owner
decision: commit M4 (and whether to open `v1_milestone_5`).** Then: **milestone 5 — freeze
the COMPLETE A–G protocol into versioned `configs/` + git before any modelling.**

## Read first (in this order)

1. `CLAUDE.md` / `AGENTS.md` — hard invariants, code style, journal + file-hygiene rules.
2. `plans/implementation_plan.md` — **the approved design; source of truth.** For M5:
   Build order **§5** (the freeze scope), and the **Experiments A–H** + **Statistics**
   sections (each experiment's design is what gets frozen). Also §"LOSO harness" (search
   space) and §"WST parameterization" (the A-M4-7 third log branch).
3. `HISTORY.md` — **newest entries only.** The M4 close + the ε/stability diagnostic are
   at the top; don't read it all.
4. `plans/MILESTONE_4_PLAN.md` — the shape an approved milestone plan takes (now a record);
   use it as the template for `plans/MILESTONE_5_PLAN.md`.

## Hard invariants (never violate — a failing check stops the build)

- **LOSO**: splits at the subject level; no frame of any session from a held-out subject in
  training. Frame-level random splitting is not a valid protocol.
- **Fit-on-train-only**: every fitted transform fit inside the CV loop on training folds
  only (sklearn **and** torch paths). Anything data-derived (incl. the A-M4-7 tuned ε) is
  computed per fold on training subjects, selected on inner-val, never on the outer test.
- **No test-set tuning**: tilings/hyperparameters/thresholds via nested CV or held-out
  subject validation, never chosen on test subjects. **This is the whole point of M5** —
  because B–G reuse the same 16 subjects, every experiment's design is frozen to git
  *before* any outer result is seen.
- **Primary target continuous** (Δm% fluid loss); 5-class secondary, ordinal metrics.
- Keep `tests/test_no_leakage.py` green — **byte-for-byte unmodified since M1** (verified at
  each close, still unmodified through M4).

## What exists now (working tree; all tested; NOT yet committed)

**Env** (`pyproject.toml` / `uv.lock`): python 3.11; numpy, scipy **PINNED <1.17**
(kymatio 0.3.0 breaks on ≥1.17), kymatio 0.3.0, sklearn, h5py, pandas, openpyxl, PyYAML,
pytest, **torch (CPU, added at M4; float32-only kymatio frontend)**.

```
configs/          data.yaml, preprocess.yaml, wst.yaml, exp_a_regression.yaml
src/dehyd/
  config.py       load_config(*paths)->frozen Config; QC/preprocess/WST validation;
                  WSTConfig.backend {numpy|torch}; beat_band_hz()
  data/           loader_10ghz, ground_truth, manifest, sessions
  qc/screens.py   frozen QC screens
  preprocess/     filters, reduce, standardize, pipeline (-> [N, C, 470] float64)
  features/       wst.py (ms->J/T, MEASURED geometry, batched scatter, order-log,
                  backend_agreement), pooling.py (pool_stats + >=2-sample std rule,
                  feature_layout/session_feature_layout, aggregate_session=concat
                  mean+median), extraction.py (extract_session_features /
                  extract_session_variants -> SessionVariantResult, canonical_spec_guard)
  eval/splits.py  nested_loso_splits(...)  (harness.py/metrics.py DO NOT EXIST YET -> M6)
  provenance.py   record_run(...)
experiments/      run_qc, run_preprocess, run_wst, audit_77ghz
results/{qc,preprocess,wst}/  curated CSVs (committed artifacts); runs/ is gitignored
tests/            +test_wst.py (66 tests); test_no_leakage.py unchanged since M1
```

**Commands:** `uv run pytest` -> 396 passed / 12 skipped; `--realdata` -> 407 / 1 (T18).
Every experiment script takes `--config configs/exp_a_regression.yaml`.

## M4 outcomes a new session should NOT re-derive

- **WST geometry (measured, pinned):** T1 742 paths × 7 time, T2 466 × 3, T3 349 × 3;
  pad 277/277 → 1024 (NOT 512). J = 7/8/8, T = 104/156/208 from 0.20/0.30/0.40 ms.
- **Cohort run** (`results/wst/wst_diagnostics_10ghz.csv`, 73 sessions, ~12 min): every
  (reduction × channel × tiling × log × family) variant finite.
- **THE ε finding:** ε = 1e-6 is negligible vs order-1 (~1e-3) but **12–64 % of the order-2
  scale (~1e-6)** — the plan's "O(1) coefficients" assumption is false. **ε left frozen**
  (finding, not retune). A LOSO diagnostic showed the fold-to-fold order-2 scale stable to
  **<1 %** (per-subject ~14 %) → a data-derived ε would be near-leakage-free here.
- **Pooling departure (A-M4-6):** the ≥2-sample segment-std rule drops the 1-sample half's
  std for T2/T3 → 5 stats/path (T1 gets 6). A deliberate ROADMAP §3.3 departure.
- **Backend policy:** numpy frontend backs ALL reported WST features; torch = validation +
  unreported work only (does NOT constrain torch as the CNN modelling framework at M6).

## Milestone 5 — the task (config-freeze gate, Build order §5)

**Freeze the COMPLETE A–G protocol to versioned `configs/` + git BEFORE any modelling.**
It is mostly design/config/documentation, not new modelling code. Follow the working
pattern: write `plans/MILESTONE_5_PLAN.md` first, get it reviewed/approved, then implement.
The freeze must cover **every experiment's design** (so no choice is informed by later
"test" subjects):

- **Exp A / WST search space + its staged-search order**, including the **A-M4-7 third log
  branch** (`log off / on+frozen-ε / on+tuned-ε`) and its **order-2-usefulness pre-check**
  gate — confirm/keep or drop as an owner decision.
- **Exp D baselines** — concrete frozen specs: raw-beat 1D-CNN, matched-preproc 1D-CNN,
  spectrogram+2D-CNN, physics range-power ratio, session-index-only.
- **Per-family budget K + the 5-seed set**; **Exp H full statistical protocol**
  (subject-cluster bootstrap B=10000, seed-collapse rules, Holm families).
- **Exp B** (session-mean residualization + equal-session objective), **Exp C** (ordinal
  family/objective/sign/fold-viability), **Exp E** (grouped interpretability CV),
  **Exp F** (nested clock/covariate models), **Exp G** (77 GHz QC/eligibility/tilings/
  input-domain/fusion).
- **Frozen protocol-constants whitelist** — the values modelling entrypoints validate
  (peak_neighbors=1, mask_taper, butter_order=4, edge_trim=32, fft_gate_transition=500,
  wst max_order=2/log_epsilon=1e-6, backend=numpy for reported features).

## Owner decisions to surface at M5 start (do NOT decide unilaterally)

1. **Commit M4** (uncommitted now) and whether to open branch `v1_milestone_5`.
2. **77 GHz any-trace flatline rule** — the frozen rule rejects **7 of 10** audited frames
   (ADC quantisation, not a dead channel). Parked since M2 explicitly for an owner decision
   at the M5 freeze; revisable only as an explicit decision, never retuned from data.
3. **A-M4-7 third log branch** — confirm it enters the frozen search space (with the
   order-2-usefulness gate), or defer/drop it.

## Do NOT re-litigate (settled; in the plan or owner-decided)

- MATLAB is **reference-only**; Python is the sole source of reported numbers. No numeric
  diffing against MATLAB — correctness rests on Python-native checks.
- Analysis unit is **session-level** (aggregate frames → 1 vector/session; concat
  mean+median). Scoring uses **N_eval**; eligibility `≥ ceil(0.5 × actual_frame_count)`.
- Three classes of preprocessing/WST parameter fixed before M6: inner-CV search axes
  (reduction {A,B} × channel {mag,iq} × model gate {1–2, 0.9–3.0 m} × tiling {T1,T2,T3} ×
  log branch); pre-declared ablations (`gate_method: fft`, `standardize: meanstd`); frozen
  protocol constants (whitelist above). Non-whitelisted values rejected by modelling
  entrypoints.
- **T18 torch mutation leg** activates at **M6** (with the harness), not M4. torch is in
  the env but the torch *fit path* it guards doesn't exist until `harness.py`.
- 77 GHz primary = slow-time (Doppler) **I/Q** WST, **per-Rx → feature-space** fusion.
  Axis assignment ACCEPTED; QC/tilings re-parameterised from fixed ms.
- If you think one of these is wrong, raise it explicitly — don't silently change it.

## Traps already paid for (don't rediscover)

- **scipy pinned <1.17** (kymatio breaks on ≥1.17); torch must not drag it forward.
- **kymatio torch frontend is float32-only**; the cross-backend check compares numpy-f64
  vs torch-f32→f64 and clears the strict tolerance. No fallback used.
- **WST padding/shape MEASURED, never assumed** — the naive `padded/2^J` gives 8, the real
  n_time is 7 (off by one). `pad_left/pad_right`/`meta()` read from the instantiated bank.
- **ε=1e-6 is NOT negligible vs order-2** — do not "fix" it; it's a frozen finding.
- **YAML 1.1 only parses a SIGNED exponent as float** (`1.0e-6` ok, `1.0e6` → string).
- **`.gitignore` matches at any depth** — check `git add -An` when adding a package dir
  (features/ staged cleanly at M4).
- **`test_no_leakage.py` byte-identical since M1** — keep it so unless the protocol changes.
- `tests/` is not a package (absolute imports); pytest cache redirected to `.cache/pytest`
  (repo-root `.pytest_cache/` has an unreadable ACL) — leave it.

## Journal & hygiene · environment

- **HISTORY.md**: an entry per resolved attempt (what/why/params, failures kept),
  newest-first. **SECOND_CHAPTER.md**: §0.1, §1, §2, **§3 (WST) written**; §4 (Exp A) is
  next real content, but M5 is a freeze so it mostly updates plans/configs. **HANDOFF.md**:
  update only when asked. Superseded code/stale results → `archive/{code,results}/`, noted
  in HISTORY; valid negative results/ablations stay in `results/`.
- **Local (Windows, git-bash + PowerShell):** QC, preprocessing, WST, classical models,
  stats. CPU smoke tests use a **≥6-subject** subset so nested CV genuinely runs.
- **IBEX (KAUST Slurm, GPU):** DL baselines / any NN as `sbatch` jobs under `scripts/ibex/`
  (NOT created yet — first IBEX milestone). Same code, config-only differences.
  `configs/ibex.yaml`, `scripts/ibex/` still deferred.
- Branches `v1_milestone_1..3` pushed; **`v1_milestone_4` current with uncommitted M4 work**;
  nothing merged to `main`.
