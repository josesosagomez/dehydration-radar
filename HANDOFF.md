# HANDOFF — resume point for a new chat (starting milestone 2)

_Written 2026-07-21, after milestone 1 was completed and committed. Purpose: let a
fresh Claude Code session start **milestone 2** without re-deriving context._

## TL;DR

**Milestone 1 is done and committed** (`f3fbade` on branch `v1_milestone_1`): pinned uv
env, config system, 10 GHz loader, ground-truth parser, frame manifest, nested-LOSO
splitter, provenance recorder, and a green `tests/test_no_leakage.py`. 159 tests.
**Next: milestone 2 — 10 GHz QC screens + a minimal 77 GHz audit.**

## Read first (in this order)

1. `CLAUDE.md` / `AGENTS.md` — hard invariants, code style, journal + file-hygiene rules.
2. `plans/implementation_plan.md` — **the approved design; source of truth.** For M2 read
   §"QC screens & thresholds", §"Confirmed data facts" (77 GHz bullet), and Build order §2.
3. `HISTORY.md` — **newest entries only.** The M1 step log is there; do not read it all.
4. `plans/MILESTONE_1_PLAN.md` — what M1 built and why (now a record, not a proposal).
   Useful as the template if you want an M2 plan document of the same shape.

## Hard invariants (never violate — a failing check stops the build)

- **LOSO**: splits at the subject level; no frame of any session from a held-out subject
  in training. Frame-level random splitting is not a valid protocol.
- **Fit-on-train-only**: every fitted transform fit inside the CV loop on training folds
  only — sklearn **and** torch paths.
- **No test-set tuning**: tilings/hyperparameters/thresholds via nested CV or held-out
  subject validation, never chosen on test subjects.
- **Primary target continuous** (Δm% fluid loss); 5-class secondary, ordinal metrics.
- Keep `tests/test_no_leakage.py` green at all times.

## What exists now (all committed, all tested)

```
pyproject.toml / uv.lock     python 3.11.15; numpy 2.4.6, scipy 1.16.3 (PINNED <1.17),
                             kymatio 0.3.0, sklearn 1.9.0, pandas, openpyxl, PyYAML, pytest
configs/                     data.yaml, preprocess.yaml, wst.yaml, exp_a_regression.yaml
src/dehyd/
  config.py                  load_config(*paths) -> frozen Config; include-composition;
                             path VALUES resolve to repo root, `include:` to declaring file
  data/sessions.py           SESSION_NAMES = ("8am","10am","12pm","2pm","4pm") == S0..S4
  data/loader_10ghz.py       parse_10ghz_filename / inspect_10ghz_file (whosmat, 17ms) /
                             load_10ghz_file -> complex128 [534, 20, N]
  data/ground_truth.py       load_ground_truth(xlsx) -> GroundTruth(sessions, subjects);
                             helpers _validate_layout / _read_values / check_targets
  data/manifest.py           build_manifest(paths, gt) -> per-frame DataFrame; resolve_path
  eval/splits.py             nested_loso_splits(...) -> [OuterFold]; iter_triples
  provenance.py              record_run(config, manifest, folds, extra) -> Path
experiments/run_regression.py  M1 smoke: config -> gt -> manifest -> folds -> provenance
tests/                       conftest.py (--realdata gate), reference_procedure.py,
                             test_{env,config,loader,ground_truth,manifest,splits,
                             provenance,no_leakage}.py
```

**Commands:**
```
uv run pytest                                              # 151 passed, 8 skipped
uv run pytest --realdata                                   # 158 passed, 1 skipped (T18)
uv run pytest tests/test_no_leakage.py -m "not realdata"   # 24 passed, 1 skipped = T18 only
uv run python experiments/run_regression.py --config configs/exp_a_regression.yaml
```

## Verified data facts (confirmed against the real files — don't re-derive)

- **10 GHz**: `data/10ghz/subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat`, 80 files, MAT v5.
  `framesRadar` = MATLAB class **double**, shape **[534 fast × 20 chirps × 100 frames]**,
  loads as **complex128**. Ignore `framesRadarIQ`. All 80 verified. 8000 frames total.
- **Ground truth**: all 16 subjects parse; **both cross-checks pass**; Δm% spans
  **−2.02 … 0.00** (negative = loss), S0 identically 0.
- **Workbook quirks** (already handled): sheet reports 1000×113 from stray formatting;
  col J is an `=I-E` **formula** (needs `data_only=True` for the cached value); row-2
  header mixes `datetime.time` cells with the literal string `'12 Noon'` (G2); Subject 15
  uses 0.05-kg increments; col K truncates rather than rounds (Subject 13).
- **77 GHz**: MAT **v7.3/HDF5**, ~285 MB each (~23 GB). `h5py` **not yet installed**.
  Reviewer-sampled h5py shape `(16,256,256,125) = (Nrx,Nchirps,Nfast,Nframes)`; loader
  must apply a full axis reversal → `(Nframes,Nfast,Nchirps,Nrx)`. **Re-confirm on the
  first real load.** Fast-time↔chirp (both 256) cannot be disambiguated by shape —
  needs the raw-data semantic check (below).
- Radar params: fs=520834 Hz, B=500e6, Tchirp=1024e-6.

## Milestone 2 — the task

Per implementation_plan.md Build order §2. **Note the xlsx parse + cross-check listed
there was already built in M1** — M2 inherits it complete.

1. **`src/dehyd/qc/screens.py`** — 10 GHz QC on the **raw** cube, thresholds frozen in
   `configs/preprocess.yaml` (already present as `QCConfig`): NaN/Inf; flatline
   (200-bin magnitude histogram per chirp, reject if any bin ≥25% of 534 samples);
   in-band energy ratio <0.30 computed with a Hann-windowed 534-pt FFT **before any
   filtering**; robust-RMS z>4.5 as a **diagnostic flag only**, not a sole reject.
   Reject = NaN/Inf **or** flatline **or** low in-band energy.
2. **QC uses ONE fixed gate — the wider 0.9–3.0 m** (`config.qc.qc_gate_m`) for all
   candidates, so the QC-passing population never varies with the model gate chosen
   later in inner CV.
3. **Per-frame reason codes + session eligibility into `manifest.py`.** A session is
   retained iff **≥ `ceil(0.5 × actual_frame_count)`** frames survive (use the real
   per-file count, already in `n_frames_in_file` — never a hard-coded 100). Dropped
   sessions are **absent, never imputed**. Record per-subject/session retained counts
   and drop reasons.
4. **Minimal 77 GHz audit** (not full extraction — that is M9): add `h5py`, load **one**
   real file, confirm dtype/shape/complex representation, run the **raw-data axis
   semantic check** (range structure on the proposed fast-time axis; near-zero-Doppler
   on the proposed chirp axis — **before** any clutter subtraction, since MTI would
   remove the static subject and defeat the check), and verify the proposed QC +
   range-Doppler ops give **non-degenerate (nonzero-energy)** data. Log findings; they
   feed the milestone-5 freeze.
5. **Tests**: `tests/test_qc.py` (each screen fires on a crafted frame and passes a clean
   one; thresholds read from config), manifest eligibility tests, and a `realdata` test
   reporting real per-subject/session QC survival. Keep `test_no_leakage.py` green —
   **QC must be frozen and data-independent, so it does not enter CV.**

**Suggested working pattern** (worked well in M1): write a
`plans/MILESTONE_2_PLAN.md` first, get it reviewed/approved, then implement step by
step, appending a HISTORY.md entry as each step resolves.

## Do NOT re-litigate (settled; in the plan or owner-decided)

- MATLAB is **reference-only**; Python is the sole source of reported numbers.
- Analysis unit is **session-level** (aggregate frames → 1 vector/session; concat
  mean+median). Per-frame is diagnostic only, never headline, never frame-IID CIs.
- Scoring counts use **N_eval**; eligibility `≥ ceil(0.5 × actual_frame_count)`.
- Departures from reference: median/MAD standardize; range gate = config (default
  1–2 m); order-aware WST log (`log(S+ε)` orders 1–2, ε=1e-6; order 0 linear);
  EdgeTrim=32 **after** reduction; no Hamming window in the primary path.
- **Owner decisions:** T18 (torch mutation leg) activates at **M6** with the harness,
  not at M4 when torch first enters the env. `min_train_subjects` constrains the
  **outer-training pool**, not each inner fit (config floor is 3).
- 77 GHz primary = slow-time (Doppler) **I/Q** WST, **per-Rx → feature-space** fusion.

If you think one of these is wrong, raise it explicitly — don't silently change it.

## Traps already paid for (don't rediscover)

- **`.gitignore` patterns without a leading slash match at any depth.** `data*/` was
  silently excluding `src/dehyd/data/`; it is now `/data*/`. **Check the staged file
  list** when a commit adds a new package directory.
- **kymatio 0.3.0 breaks on scipy ≥1.17** (`scipy.special.sph_harm` removed).
  `import kymatio` still succeeds — only `from kymatio.numpy import Scattering1D` fails.
  scipy is pinned `<1.17`; revisit when kymatio ships a `sph_harm_y` release.
  Also: `Scattering1D(J=7, shape=(470,), ...)` warns "signal support too small to avoid
  border effects" — **an M4 concern**; the plan already requires measuring padding and
  output shape from the instantiated filter bank rather than assuming.
- **openpyxl never evaluates formulas**: a written formula has **no** cached value
  (`data_only=True` → `None`). A synthetic workbook can hold a formula *or* a number,
  never both — hence the three-helper split in `ground_truth.py`.
- **`ws.cell(row, col, value=None)` is a no-op** in openpyxl; assign `.value = None`.
- **`rel_path` string order ≠ session order**: `subject_1_10am.mat` sorts before
  `subject_1_8am.mat`. Look provenance/manifest entries up **by path, never by index**.
- `tests/` is not a package — use absolute imports (`from reference_procedure import …`).
- The repo-root `.pytest_cache/` has an unreadable ACL on this machine; pytest's cache is
  redirected to `.cache/pytest` in `pyproject.toml`. Leave it alone.

## Environment / compute

- **Local (Windows, git-bash + PowerShell):** scaffolding, QC, preprocessing, WST,
  all classical models, stats. CPU smoke tests use a **≥6-subject** subset so nested CV
  genuinely runs.
- **IBEX (KAUST Slurm, GPU):** DL baselines / any NN as `sbatch` jobs under
  `scripts/ibex/` (not created yet). Same code, **config-only** differences via an
  overlay YAML passed as a later `--config` (mechanism implemented and tested;
  `configs/ibex.yaml` is written when the real cluster roots are known). No GPU training
  in interactive runs.

## Journal & hygiene (keep doing)

- **HISTORY.md**: append an entry per resolved attempt (what/why/params, failures kept),
  newest-first. Log each reference-departure with its reason.
- **SECOND_CHAPTER.md**: §0.1 (evaluation protocol + M1 data integrity) is written;
  fill §1 "Data & ground truth" as M2 closes.
- **HANDOFF.md**: update **only when asked**.
- Superseded code / stale results → `archive/{code,results}/`, noted in HISTORY.
  Valid negative results and ablations are current results — they stay in `results/`.

## Open items

- `h5py` not installed; 77 GHz axis order unconfirmed on a real file → **milestone 2**.
- `configs/ibex.yaml`, `scripts/ibex/` → first IBEX milestone.
- `results/runs/` is gitignored (per-run provenance regenerates each invocation);
  `results/` stays for curated artifacts. Reversible if you want full history in git.
- Nothing is pushed — no upstream is configured for `v1_milestone_1`.
