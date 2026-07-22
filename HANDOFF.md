# HANDOFF — resume point for a new chat (starting milestone 4)

_Written 2026-07-23, after milestone 3 was completed, committed and pushed. Purpose: let
a fresh Claude Code session start **milestone 4** without re-deriving context._

## TL;DR

**Milestone 3 is done, committed and pushed** (`v1_milestone_3`): the preprocessing
sequence — zero-phase Butterworth band gate, Options A/B chirp reduction, EdgeTrim,
robust standardization — plus the cohort diagnostic that measured **the subject at
1.50 m**. 319 tests pass (329 with `--realdata`). You are on branch **`v1_milestone_4`**
= `v1_milestone_3` + this handoff. **Next: milestone 4 — WST features (kymatio).**

## Read first (in this order)

1. `CLAUDE.md` / `AGENTS.md` — hard invariants, code style, journal + file-hygiene rules.
2. `plans/implementation_plan.md` — **the approved design; source of truth.** For M4:
   §"WST parameterization", §"Analysis unit — session-level primary", Build order §4.
3. `HISTORY.md` — **newest entries only.** The M3 log is at the top; don't read it all.
4. `plans/MILESTONE_3_PLAN.md` — the shape an approved milestone plan takes (now a
   record); use it as the template for `plans/MILESTONE_4_PLAN.md`.

## Hard invariants (never violate — a failing check stops the build)

- **LOSO**: splits at the subject level; no frame of any session from a held-out subject
  in training. Frame-level random splitting is not a valid protocol.
- **Fit-on-train-only**: every fitted transform fit inside the CV loop on training folds
  only — sklearn **and** torch paths.
- **No test-set tuning**: tilings/hyperparameters/thresholds via nested CV or held-out
  subject validation, never chosen on test subjects.
- **Primary target continuous** (Δm% fluid loss); 5-class secondary, ordinal metrics.
- Keep `tests/test_no_leakage.py` green — **byte-for-byte unmodified since M1**
  (verified at each close); keep it so unless the protocol itself changes.

## What exists now (all committed, all tested)

**Env** (`pyproject.toml` / `uv.lock`): python 3.11.15; numpy 2.4.6, scipy 1.16.3
(**PINNED <1.17**), kymatio 0.3.0, sklearn 1.9.0, h5py 3.16.0, pandas, openpyxl, PyYAML,
pytest. **No torch yet — it arrives at M4.**

```
configs/                   data.yaml, preprocess.yaml, wst.yaml, exp_a_regression.yaml
src/dehyd/
  config.py                load_config(*paths) -> frozen Config; field + cross-field
                           validation; beat_band_hz()
  data/                    sessions.py (S0..S4 names); loader_10ghz.py (-> complex128
                           [534,20,N]); ground_truth.py; manifest.py (build_manifest,
                           apply_qc, session_qc_report, eligible_frames,
                           evaluable_subjects, resolve_path)
  qc/screens.py            FrameQC; run_qc_frame; run_qc_cube; in_band_mask
  preprocess/filters.py    design_bandpass_sos; bandpass_filtfilt; default_padlen;
                           fft_gate; apply_band_gate; filter_spec
  preprocess/reduce.py     reduce_option_a; option_b_roi_bins; OptionBDetection;
                           detect_option_b_peak; option_b_mask; reduce_option_b; edge_trim
  preprocess/standardize.py  robust_standardize; meanstd_standardize; to_channels
  preprocess/pipeline.py   preprocess_frame; preprocess_cube -> [N, C, 470] float64
  eval/splits.py           nested_loso_splits(...) -> [OuterFold]; iter_triples
  provenance.py            record_run(config, manifest, folds, extra) -> Path
experiments/               run_qc.py, run_preprocess.py, run_regression.py, audit_77ghz.py
results/{qc,preprocess}/   qc_survival_10ghz.csv, audit_77ghz.json,
                           preprocess_diagnostics_10ghz.csv  (curated, committed)
tests/                     conftest.py (--realdata gate), reference_procedure.py,
                           test_{env,config,loader,ground_truth,manifest,qc,splits,
                           provenance,audit_77ghz,preprocess,no_leakage}.py
```

**Commands:** `uv run pytest` → 319 passed / 11 skipped; `uv run pytest --realdata` →
329 / 1 (T18). Every experiment script takes
`--config configs/exp_a_regression.yaml`.

## Verified data facts (confirmed against the real files — don't re-derive)

- **10 GHz**: 80 files, MAT v5, `framesRadar` = **[534 fast × 20 chirps × 100 frames]**,
  complex128. 8000 frames pre-QC. Ignore `framesRadarIQ`. fs=520834 Hz, B=500e6,
  Tchirp=1024e-6 → HzPerM ≈ 3257.5 Hz/m.
- **Ground truth**: 16 subjects, both cross-checks pass, Δm% spans **−2.02 … 0.00**.
- **QC survival (M2, frozen thresholds):** **7330/8000 frames pass (91.6%)**; zero
  NaN/Inf and zero flatline — every rejection is the in-band energy screen. **7 of 80
  sessions dropped** → **73 eligible sessions, 7168 analysable frames, N_eval = 16**
  (16 outer folds, 5 inner each). QC gate 0.9–3.0 m + ±1000 Hz → mask bins 2..11 of 267.
  The RMS diagnostic fires on 34% of frames and rejects nothing.
- **M3 preprocessing (measured on all 7168 frames):** model gate 1–2 m →
  **3257.5–6514.9 Hz**, Wn ≈ (0.01251, 0.02502), sos (4,6), **padlen 27** (explicit).
  Option-B ROI = model gate, **no margin** → **bins 4,5,6** (df ≈ 975.34 Hz); the
  0.9–3.0 m candidate gives bins 4..10. Output length **470**.
  **The dominant beat sits at 1.50–1.80 m** (bin 5 in 41 sessions, bin 6 in 31) — *not*
  the ~1 m the gate's rationale assumed; the gate was **not** changed.
  **peak_share median 0.512** (a flat 3-bin ROI would be 0.333) — a genuinely dominant
  return. Retention median 0.407 [0.061, 0.644]. `roi_to_total` (0.930) is measured
  **post-filter** — filter selectivity, **not** target presence; do not over-read it.
  **Finite-record ≠ steady-state:** a mid-band tone retains 0.7595 (0.8313 after trim),
  a 50 kHz tone only −17.2 dB (−20.1 dB after trim). Pinned as regression values.
- **77 GHz** (fs=500e3, B=2e9, Tchirp=512e-6): MAT v7.3/HDF5, ~276 MB each, 80 files.
  Shape **`(16,256,256,125)` CONFIRMED**, gzip chunks `(16,4,1,125)`. **`framesRadar` is
  plain real `float64`, NOT complex** — Exp G's "I/Q" arises at the **range FFT**. Axis
  assignment **ACCEPTED** (D_chirp=0.9999, G_fast/G_chirp ≈ 34000×). dr=0.0749 m, gate
  bins 27..53, QC mask bins 26..54, PRF 1953.125 Hz.

## Milestone 4 — the task

Per implementation_plan.md Build order §4 and §"WST parameterization". Input is
`preprocess_cube(...)` → **float64 [n_frames × C × 470]** (C = 1 for `mag`, 2 for `iq`).

1. **`uv add torch`** (CPU wheel locally; the cross-backend check needs it). scipy stays
   pinned `<1.17` — **kymatio 0.3.0 breaks on scipy ≥1.17** (`sph_harm` removed).
2. **`src/dehyd/features/wst.py`** — kymatio `Scattering1D(J, shape=(470,), Q, T,
   max_order=2)`. `T_samples = round(InvScale_ms · 1e-3 · fs)` → 0.20 ms→**104**,
   0.30 ms→**156**, 0.40 ms→**208**; `J = ceil(log2(T_samples))` → **T1 J=7, T2/T3 J=8**;
   `Q` = (10,4), (8,2), (6,2). **Padding and output shape are MEASURED from the
   instantiated filter bank (`pad_left`/`pad_right`, `scattering.meta()`), never
   assumed** — no hard-coded 512 or `padded_len / 2^J`.
3. **Order-aware log** (config-selected, inner CV at M6): orders 1–2 → `log(S + ε)`,
   **ε = 1e-6**; **order 0 stays linear** (a signed low-pass, can be negative). A test
   must assert every branch — mag/iq × orders 0/1/2 × log on/off — is finite.
4. **`src/dehyd/features/pooling.py`** — (a) pooled stats: mean/std over global +
   first/second half per path; (b) raw-flattened series. Fixed, documented element order
   from the recorded path metadata. Session aggregation = **concat(per-frame mean,
   per-frame median)** → one vector per session.
5. **`tests/test_wst.py`** — path structure per tiling (from `meta()`), near-invariance
   to small time shifts, determinism, **numpy vs torch ≤1e-4 relative** (only then may
   either back a reported run).

**Known M4 concern:** `Scattering1D(J=7, shape=(470,))` warns *"signal support is too
small to avoid border effects"*. Measure the real padding/output shape and decide
explicitly; do not silence the warning without recording the reasoning.
**Working pattern** (worked for M1–M3): write `plans/MILESTONE_4_PLAN.md` first, get it
reviewed/approved, then implement step by step, appending a HISTORY.md entry per step.

## Do NOT re-litigate (settled; in the plan or owner-decided)

- MATLAB is **reference-only**; Python is the sole source of reported numbers. Never
  diff against MATLAB numerics — correctness rests on Python-native checks.
- Analysis unit is **session-level** (aggregate frames → 1 vector/session; concat
  mean+median). Per-frame is diagnostic only, never headline, never frame-IID CIs.
  Scoring uses **N_eval**; eligibility `≥ ceil(0.5 × actual_frame_count)`.
- **Three classes of preprocessing parameter, fixed before M6** (MILESTONE_3_PLAN §0):
  *inner-CV axes* = reduction {A,B} × channel {mag,iq} × model gate {1–2, 0.9–3.0 m};
  *pre-declared ablations* = `gate_method: fft`, `standardize: meanstd`; *frozen
  protocol constants* = `peak_neighbors=1`, `mask_taper=true`, `butter_order=4`,
  `edge_trim=32`, `fft_gate_transition_hz=500` (non-defaults are test-only and
  rejected by modelling entrypoints).
- Departures from reference: median/MAD standardize (eps outside the scale factor);
  range gate = config; order-aware WST log; EdgeTrim=32 **after** reduction; no window
  in the primary path; QC on the **raw** cube with low in-band as a **rejection**;
  Option-B mask two-sided with **full weight on the peak** (the reference is one-sided
  and zeroes the peak).
- **Owner decisions:** T18 (torch mutation leg) activates at **M6** with the harness,
  **not at M4** when torch enters the env. `min_train_subjects` constrains the
  **outer-training pool**, not each inner fit (floor 3). 10 GHz QC margin ±1000 Hz;
  77 GHz margin = one FFT bin (1953.125 Hz); 77 GHz flatline = any-trace-fails-frame.
  77 GHz primary = slow-time (Doppler) **I/Q** WST, **per-Rx → feature-space** fusion.
- If you think one of these is wrong, raise it explicitly — don't silently change it.

## Traps already paid for (don't rediscover)

- **YAML 1.1 only parses an exponent as a float when it is SIGNED.** `500.0e6` loads as
  the **string** `"500.0e6"`; it must be `500.0e+6`. Pinned by a test.
- **`np.histogram(x, bins=200)` RAISES on a degenerate range** — not only for a constant
  chirp but any *near*-constant one. So **a noiseless tone is not a valid fixture**
  (also zero MAD) — add small seeded noise.
- **`.gitignore` patterns without a leading slash match at any depth.** Check the staged
  list with `git add -An` when a commit adds a new package directory.
- **kymatio 0.3.0 breaks on scipy ≥1.17**; scipy is pinned `<1.17`.
- **openpyxl never evaluates formulas**: a written formula has no cached value.
- **`rel_path` string order ≠ session order** (`subject_1_10am` < `subject_1_8am`) —
  join **by key, never by index**. **Per-frame QC reason flags overlap**: the only
  identity is `n_pass + n_fail_any == n_frames`.
- **filtfilt squares the magnitude response** (−3 dB corners → −6 dB effective), and a
  534-sample record cannot reach the design stopband. Don't "fix" either.
- **Option-B detection is windowed; the mask is applied UNWINDOWED.** Zero ROI detection
  power does **not** imply zero output (the Hann kernel [−¼,½,−¼] can annihilate the
  windowed ROI while unwindowed mask bins stay nonzero).
- `tests/` is not a package — use absolute imports; `experiments/` goes on `sys.path` to
  import script helpers. The repo-root `.pytest_cache/` has an unreadable ACL, so the
  cache is redirected to `.cache/pytest` in `pyproject.toml` — leave it alone.

## Environment / compute · journal & hygiene

- **Local (Windows, git-bash + PowerShell):** QC, preprocessing, WST, all classical
  models, stats. CPU smoke tests use a **≥6-subject** subset so nested CV genuinely runs.
- **IBEX (KAUST Slurm, GPU):** DL baselines / any NN as `sbatch` jobs under
  `scripts/ibex/` (not created yet). Same code, **config-only** differences via an
  overlay YAML passed as a later `--config`. No GPU training in interactive runs.
- **HISTORY.md**: an entry per resolved attempt (what/why/params, failures kept),
  newest-first; log each reference-departure with its reason. **SECOND_CHAPTER.md**:
  §0.1, §1 (data + QC), §2 (preprocessing) written; fill §3 "WST features" as M4
  closes. **HANDOFF.md**: update **only when asked**. Superseded code / stale results →
  `archive/{code,results}/`, noted in HISTORY — but valid negative results and ablations
  are current results and stay in `results/`.

## Open items

- **Owner decision deferred to M5:** the frozen 77 GHz any-trace flatline rule rejects
  **7 of 10** audited frames — ADC quantisation, *not* a dead channel. Recorded,
  deliberately not retuned; revisable before the M5 freeze only as an explicit owner
  decision, never from audited data.
- `configs/ibex.yaml`, `scripts/ibex/` → first IBEX milestone. `results/runs/` is
  gitignored (provenance regenerates); `results/{qc,preprocess}/` hold the curated
  artifacts. Branches `v1_milestone_1..3` pushed, `v1_milestone_4` current, nothing
  merged to `main`.
