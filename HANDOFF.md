# HANDOFF — resume point for a new chat (starting milestone 3)

_Written 2026-07-21, after milestone 2 was completed, committed and pushed. Purpose:
let a fresh Claude Code session start **milestone 3** without re-deriving context._

## TL;DR

**Milestone 2 is done, committed and pushed** (`395eb62` on branch `v1_milestone_2`):
frozen 10 GHz QC screens, per-frame reason codes, session eligibility, the full-cohort
survival report, and a minimal 77 GHz audit that **confirmed the axis hypothesis**.
260 tests pass (269 with `--realdata`). You are on branch **`v1_milestone_3`**, which
is `v1_milestone_2` plus this handoff.
**Next: milestone 3 — preprocessing (the executable sequence).**

## Read first (in this order)

1. `CLAUDE.md` / `AGENTS.md` — hard invariants, code style, journal + file-hygiene rules.
2. `plans/implementation_plan.md` — **the approved design; source of truth.** For M3 read
   §"Preprocessing — executable sequence", §"Deliberate departures from the reference",
   and Build order §3.
3. `HISTORY.md` — **newest entries only.** The M2 step log is at the top; do not read it all.
4. `plans/MILESTONE_2_PLAN.md` — the shape an approved milestone plan takes (now a
   record, not a proposal). Use it as the template for `plans/MILESTONE_3_PLAN.md`.

## Hard invariants (never violate — a failing check stops the build)

- **LOSO**: splits at the subject level; no frame of any session from a held-out subject
  in training. Frame-level random splitting is not a valid protocol.
- **Fit-on-train-only**: every fitted transform fit inside the CV loop on training folds
  only — sklearn **and** torch paths.
- **No test-set tuning**: tilings/hyperparameters/thresholds via nested CV or held-out
  subject validation, never chosen on test subjects.
- **Primary target continuous** (Δm% fluid loss); 5-class secondary, ordinal metrics.
- Keep `tests/test_no_leakage.py` green at all times. It is **byte-for-byte unmodified
  since M1** — keep it that way unless the protocol itself changes.

## What exists now (all committed, all tested)

```
pyproject.toml / uv.lock     python 3.11.15; numpy 2.4.6, scipy 1.16.3 (PINNED <1.17),
                             kymatio 0.3.0, sklearn 1.9.0, h5py 3.16.0, pandas,
                             openpyxl, PyYAML, pytest.  NO torch yet -> add at M4.
configs/                     data.yaml, preprocess.yaml, wst.yaml, exp_a_regression.yaml
src/dehyd/
  config.py                  load_config(*paths) -> frozen Config; include-composition;
                             per-field + cross-field validation; beat_band_hz();
                             SPEED_OF_LIGHT_M_S
  data/sessions.py           SESSION_NAMES = ("8am","10am","12pm","2pm","4pm") == S0..S4
  data/loader_10ghz.py       parse/inspect/load -> complex128 [534, 20, N]
  data/ground_truth.py       load_ground_truth(xlsx) -> GroundTruth(sessions, subjects)
  data/manifest.py           build_manifest; apply_qc; session_qc_report;
                             eligible_frames; evaluable_subjects; resolve_path
  qc/screens.py              FrameQC, run_qc_frame, run_qc_cube, in_band_mask
  eval/splits.py             nested_loso_splits(...) -> [OuterFold]; iter_triples
  provenance.py              record_run(config, manifest, folds, extra) -> Path
experiments/                 run_qc.py, run_regression.py, audit_77ghz.py
results/qc/                  qc_survival_10ghz.csv, audit_77ghz.json  (curated, committed)
tests/                       conftest.py (--realdata gate), reference_procedure.py,
                             test_{env,config,loader,ground_truth,manifest,qc,splits,
                             provenance,audit_77ghz,no_leakage}.py
```

**Commands:**
```
uv run pytest                                              # 260 passed, 10 skipped
uv run pytest --realdata                                   # 269 passed, 1 skipped (T18)
uv run pytest tests/test_no_leakage.py -m "not realdata"   # 24 passed, 1 skipped = T18
uv run python experiments/run_qc.py         --config configs/exp_a_regression.yaml
uv run python experiments/run_regression.py --config configs/exp_a_regression.yaml
uv run python experiments/audit_77ghz.py    --config configs/exp_a_regression.yaml
```

## Verified data facts (confirmed against the real files — don't re-derive)

- **10 GHz**: 80 files, MAT v5, `framesRadar` = **[534 fast × 20 chirps × 100 frames]**,
  loads as **complex128**. 8000 frames pre-QC. Ignore `framesRadarIQ`.
- **Ground truth**: 16 subjects, both cross-checks pass, Δm% spans **−2.02 … 0.00**.
- **QC survival (M2, frozen thresholds, first contact):** **7330/8000 frames pass
  (91.6%)**; **zero** NaN/Inf and **zero** flatline — every rejection is the in-band
  energy screen. **7 of 80 sessions dropped** (s1 8am 35/100, s1 4pm 1/100, s3 10am
  37/100, s4 2pm 35/100, s5 2pm 39/100, s6 8am 0/100, s16 10am 15/100).
  → **73 eligible sessions, 7168 analysable frames, N_eval = 16** (16 outer folds,
  5 inner each). ROADMAP independently estimated "~7500" — corroboration, not a target.
  The RMS diagnostic fires on 34% of frames and **rejects nothing**; it is sensitive
  because MAD across only 20 near-identical chirps is tiny. Read it as a descriptor.
- **QC band arithmetic (measured, not assumed):** HzPerM ≈ 3257.5 Hz/m → 0.9–3.0 m gate
  = 2931.7–9772.4 Hz, +±1000 Hz margin → **mask bins 2..11** of the 267-bin
  non-negative half-spectrum (DC in, Nyquist out).
- **77 GHz**: MAT v7.3/HDF5, ~276 MB each (~21.5 GB), 80 files.
  **Shape `(16,256,256,125)` CONFIRMED**, gzip chunks `(16,4,1,125)` (which span the
  whole frame axis, so a 10-frame read still decompresses ~1 GB).
  **`framesRadar` is plain real `float64`, NOT complex** — ADC-like, quantised to 1/16,
  |x| ≤ 2560. Exp G's "I/Q" therefore arises at the **range FFT**, not the raw cube.
  **Axis assignment ACCEPTED** on raw data before MTI: D_chirp=0.9999, G_fast=0.226,
  G_fast/G_chirp ≈ 34000×. Derived: dr=0.0749 m, range gate **bins 27..53**, QC mask
  **bins 26..54**, PRF 1953.125 Hz.
- Radar params 10 GHz: fs=520834 Hz, B=500e6, Tchirp=1024e-6.
  77 GHz: fs=500e3, B=2e9, Tchirp=512e-6.

## Milestone 3 — the task

Per implementation_plan.md Build order §3 and §"Preprocessing — executable sequence".
Input population is `eligible_frames(manifest_qc)` — 7168 frames, 73 sessions.

1. **`src/dehyd/preprocess/filters.py`** — SOS Butterworth **order 4**
   (`butter(4, Wn, btype='bandpass', output='sos')`), applied **zero-phase**
   (`sosfiltfilt`) along the **fast-time (534) axis, per chirp per frame**, on the
   **complex** signal (filter real and imag; record `padtype`/`padlen`). `Wn` from the
   **model** gate (`preprocess.model_gate_m`, default 1–2 m) — *not* the QC gate.
   No window in this path. An FFT-domain tapered-mask gate is a config alternative.
2. **`src/dehyd/preprocess/reduce.py`** — **Option A** (mean across the 20 chirps) and
   **Option B** (per chirp: Hann + 534-pt FFT to *detect* the dominant beat bin in the
   ROI, ±1-bin two-sided Hann-tapered mask applied to the *unwindowed* chirp FFT,
   IFFT, then average across chirps).
3. **EdgeTrim 32 samples/end AFTER reduction** → length **470** (this order is
   deliberate and matches `wst_extract.m`).
4. **`src/dehyd/preprocess/standardize.py`** — robust z per signal:
   `y = (x − median)/(1.4826·MAD + eps)` (median-centred, a deliberate departure from
   the reference's mean-centre/MAD-scale inconsistency). Channels: `mag` = |s|, or
   `iq` = {real, imag} standardised separately.
5. **`tests/test_preprocess.py`** — filter magnitude response passes in [f_lo,f_hi] and
   stops outside; forward-backward is zero-phase (zero group delay on a test tone);
   Parseval/energy sanity; Option-B isolated peak lands in the ROI; determinism under
   a fixed seed.

**Suggested working pattern** (worked for M1 and M2): write `plans/MILESTONE_3_PLAN.md`
first, get it reviewed/approved, then implement step by step, appending a HISTORY.md
entry as each step resolves.

## Do NOT re-litigate (settled; in the plan or owner-decided)

- MATLAB is **reference-only**; Python is the sole source of reported numbers. Never
  diff against MATLAB numerics — correctness is established by Python-native checks.
- Analysis unit is **session-level** (aggregate frames → 1 vector/session; concat
  mean+median). Per-frame is diagnostic only, never headline, never frame-IID CIs.
- Scoring counts use **N_eval**; eligibility `≥ ceil(0.5 × actual_frame_count)`.
- Departures from reference: median/MAD standardize; range gate = config (default
  1–2 m); order-aware WST log (`log(S+ε)` orders 1–2, ε=1e-6; order 0 linear);
  EdgeTrim=32 **after** reduction; no Hamming window in the primary path; QC on the
  **raw** cube with low in-band as a **rejection** criterion.
- **Owner decisions:** T18 (torch mutation leg) activates at **M6** with the harness,
  not at M4 when torch first enters the env. `min_train_subjects` constrains the
  **outer-training pool**, not each inner fit (config floor is 3). 10 GHz QC margin
  ±1000 Hz; 77 GHz margin = one FFT bin (1953.125 Hz); 77 GHz flatline =
  any-(Rx,chirp)-trace-fails-frame.
- 77 GHz primary = slow-time (Doppler) **I/Q** WST, **per-Rx → feature-space** fusion.

If you think one of these is wrong, raise it explicitly — don't silently change it.

## Traps already paid for (don't rediscover)

- **YAML 1.1 only parses an exponent as a float when it is SIGNED.**
  `bandwidth_hz: 500.0e6` silently loads as the **string** `"500.0e6"`; it must be
  `500.0e+6`. This survived all of M1 because nothing consumed the value. Pinned by
  `test_radar_constants_load_as_floats_not_strings`.
- **`np.histogram(x, bins=200)` RAISES on a degenerate range** ("Too many bins for data
  range") — not only for an exactly constant chirp but for any *near*-constant one (a
  noiseless CW tone spans ~1e-16). `qc/screens.py` builds the edges with `linspace` and
  checks `edges[:-1] < edges[1:]` first; the degenerate case *is* flatline.
  Consequence for tests: **a noiseless tone is not a valid "clean frame" fixture** —
  add small seeded noise.
- **`.gitignore` patterns without a leading slash match at any depth.** `data*/` was
  silently excluding `src/dehyd/data/`; it is now `/data*/`. **Check the staged file
  list** when a commit adds a new package directory (done for `src/dehyd/qc/`).
- **kymatio 0.3.0 breaks on scipy ≥1.17** (`scipy.special.sph_harm` removed). scipy is
  pinned `<1.17`. Also `Scattering1D(J=7, shape=(470,))` warns "signal support too small
  to avoid border effects" — an **M4** concern; measure padding/output shape from the
  instantiated filter bank rather than assuming.
- **openpyxl never evaluates formulas**: a written formula has no cached value.
- **`rel_path` string order ≠ session order**: `subject_1_10am.mat` sorts before
  `subject_1_8am.mat`. Join **by key, never by index** — `manifest._join_qc` enforces
  this with `validate="one_to_one"` plus unmatched-key and row-count checks.
- **Per-frame QC reason flags overlap** (an all-zero frame is flatline *and* low
  in-band). Never assert per-reason counts sum to rejections; the identity is
  `n_pass + n_fail_any == n_frames`.
- `tests/` is not a package — use absolute imports. `tests/test_audit_77ghz.py` imports
  the audit script by inserting `experiments/` on `sys.path`.
- The repo-root `.pytest_cache/` has an unreadable ACL on this machine; pytest's cache
  is redirected to `.cache/pytest` in `pyproject.toml`. Leave it alone.

## Environment / compute

- **Local (Windows, git-bash + PowerShell):** scaffolding, QC, preprocessing, WST,
  all classical models, stats. CPU smoke tests use a **≥6-subject** subset so nested CV
  genuinely runs.
- **IBEX (KAUST Slurm, GPU):** DL baselines / any NN as `sbatch` jobs under
  `scripts/ibex/` (not created yet). Same code, **config-only** differences via an
  overlay YAML passed as a later `--config` (mechanism implemented and tested).
  No GPU training in interactive runs.

## Journal & hygiene (keep doing)

- **HISTORY.md**: append an entry per resolved attempt (what/why/params, failures kept),
  newest-first. Log each reference-departure with its reason.
- **SECOND_CHAPTER.md**: §0.1 (protocol + M1 integrity) and §1 (data, ground truth and
  the full QC account) are written; fill §2 "Preprocessing" as M3 closes.
- **HANDOFF.md**: update **only when asked**.
- Superseded code / stale results → `archive/{code,results}/`, noted in HISTORY.
  Valid negative results and ablations are current results — they stay in `results/`.

## Open items

- **Owner decision deferred to M5:** the frozen 77 GHz any-trace flatline rule rejects
  **7 of 10** audited frames — ADC quantisation against a 128-bin histogram, spread
  evenly over all 16 Rx (169–1601 of 2560 traces each), *not* a dead channel. Recorded,
  deliberately not retuned; revisable before the M5 freeze only as an explicit owner
  decision, never from audited data.
- `configs/ibex.yaml`, `scripts/ibex/` → first IBEX milestone.
- `results/runs/` is gitignored (per-run provenance regenerates each invocation);
  `results/qc/` holds the committed curated artifacts.
- Branches `v1_milestone_1` and `v1_milestone_2` are pushed; `v1_milestone_3` is the
  current working branch. Nothing is merged to `main` yet.
