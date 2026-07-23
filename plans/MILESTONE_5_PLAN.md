# MILESTONE 5 PLAN — 77 GHz front-end (loader → QC → decision gate → preprocessing → slow-time I/Q WST → cohort diagnostics → IBEX)

_Task-level execution plan for milestone 5 **only** (the new milestone inserted ahead of the
config-freeze gate; ROADMAP §7 renumbered — see §6/A-M5-2). Status: **DRAFT for owner
review.** Work happens on branch `v1_milestone_5` (already current; HANDOFF.md is its
bootstrap). This document adds the execution detail the main plan intentionally omits; where
a design decision is needed it cites `plans/implementation_plan.md` (the source of truth —
for M5 chiefly the **Exp G** block, lines ~864–974, plus §"WST parameterization",
§"Analysis unit", §"Compute / IBEX"), `CLAUDE.md`, or `ROADMAP.md`. Anything here that goes
beyond those documents is flagged in §6 as an amendment._

**Why this milestone exists.** The owner decided to promote the 77 GHz band from
**fusion-only (Exp G)** to a **full parallel primary arm**: Experiments A–F will each run on
77 GHz with band-appropriate parameters, **10 GHz remains the sole headline**, and Exp G
keeps its cross-band fusion contrast on the matched population. The config-freeze gate
(formerly milestone 5) becomes **milestone 6** and now freezes the 77 GHz arm too — which is
exactly why the front-end must be built first: the freeze cannot pin 77 GHz
QC/eligibility/tilings numbers that have never met the full cohort. M5 mirrors, for 77 GHz,
what M1–M4 built for 10 GHz, stopping at the same boundary (features + cohort diagnostics;
**no modelling, no selection**).

**Owner decisions already made (recorded so they are not re-litigated):**

1. **77 GHz becomes a full parallel primary arm** (A–F on both bands; 10 GHz sole headline;
   Exp G fusion retained). This is the premise of the milestone (A-M5-1). Because it rescopes
   the authoritative documents, it is an **owner-approved prerequisite written into
   `implementation_plan.md` + `ROADMAP.md` before implementation** (§1 step 0), not an
   assertion the plan makes on its own.
2. **IBEX access is confirmed working.** The heavy 77 GHz cohort extraction runs on IBEX as
   a Slurm job array; a local smoke path differs only by config. This is the **first IBEX
   milestone** — `configs/ibex.yaml` and `scripts/ibex/` are created here (A-M5-5).
3. **The flatline QC rule is finalized on ADC-quantisation *mechanism* grounds from the M2
   single-file audit — before the cohort run, never from cohort survival** (it rejected 7/10
   audited frames at M2 on benign quantisation). **Owner pre-selected outcome (b) — a
   mechanism-corrected exact replacement of the 77 GHz screen** (2026-07-23, for now); the
   exact rule is specified at step 6 from the quantisation mechanism, the 10 GHz screen is
   untouched, and M5 proceeds to the cohort feature runs. Keep-frozen (a) and
   move-inside-inner-CV (c) remain fallbacks (§1 step 6; A-M5-6).
4. **numpy kymatio remains the canonical backend for every reported feature** (the frozen M4
   policy). CPU on IBEX; no GPU/torch-WST for reported artifacts is assumed.

**The full dataset is present and confirmed:** `data/77ghz/` holds **80 files**
(`subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat`, ~285 MB each, ~22 GB), MAT v7.3 / HDF5,
dataset `framesRadar`, on-disk shape `(16, 256, 256, n_frames)`.

---

## §0 Scope and ground rules

**In scope:**

- **Config**: new frozen dataclasses `Preprocess77Config`, `QC77Config`, `WST77Config` in
  `src/dehyd/config.py`; optional `paths.data_77ghz_dir`; the section whitelist, validators,
  and cross-checks; the mirror YAML files `configs/{data77,preprocess77,wst77,exp_77ghz}.yaml`
  (§2.1).
- **Loader**: `src/dehyd/data/loader_77ghz.py` (h5py; promotes the audited reader/axis
  reversal) (§2.2).
- **Manifest**: `src/dehyd/data/manifest_77.py` — a mirror of `manifest.py` that **reuses by
  import** the subtle shared pieces (`_join_qc`, `eligible_frames`, `evaluable_subjects`)
  (§2.3).
- **QC**: `src/dehyd/qc/screens_77.py` (frozen any-trace flatline + in-band screens) and
  `src/dehyd/qc/axis_check_77.py` (semantic axis check) (§2.4).
- **Preprocessing**: `src/dehyd/preprocess/pipeline_77.py` — chain steps 1–5 (MTI → bandpass
  → Hann → range FFT → gate crop) (§2.5).
- **Features**: `src/dehyd/features/extraction_77.py` — chain steps 6–10 (per-Rx slow-time
  I/Q WST → range-bin averaging → feature-space Rx fusion → pool → session aggregate)
  reusing the fs/shape-agnostic `wst.py`/`pooling.py` (§2.6).
- **Experiment CLIs**: `experiments/{run_qc77,run_preprocess77,run_wst77}.py` — thin cohort
  passes over the library modules, with the QC run producing the **label-blind flatline
  diagnostics** for the owner gate (§2.7).
- **Provenance**: one minimal extension (`record_run(..., data_dir=None)`) so 77 GHz runs
  and array tasks hash the right inputs (§2.8).
- **IBEX scaffolding**: `configs/ibex.yaml`, `scripts/ibex/{qc77,preprocess77,wst77}.sbatch`
  + `scripts/ibex/README.md`, `.gitattributes` (LF for sbatch) (§2.9).
- **Tests**: `tests/test_{config,loader77,manifest77,qc77,preprocess77,wst77}.py` additions
  (synthetic, private-data-free) + one-file `--realdata` end-to-end (§3).
- **Journal upkeep**: HISTORY.md continuously (departures + measured values, newest-first);
  SECOND_CHAPTER.md §4 "77 GHz front-end" at milestone close.

**Explicitly out of scope (deferred to their milestones):**

- **Any modelling, model selection, harness, or fusion.** M5 *implements* the 77 GHz
  tiling/log/fusion **alternatives**; it never *selects* among them. `eval/harness.py`,
  `eval/metrics.py`, and Exp G fusion (the α combiner) remain **M7+**.
- **The config-freeze gate itself** — that is the renumbered **M6**. M5 produces the real
  cohort artifacts the freeze will rest on; it does not perform the freeze.
- **Secondary 77 GHz variants** — the Doppler-FFT-spectrum WST branch and the fast-time WST
  branch (with the nonzero-statistic fix) from `wst_extract77.m` are **deferred past M5**
  (A-M5-7). Only the **primary slow-time I/Q chain** is built now.
- **10 GHz files are byte-untouched.** `configs/{data,preprocess,wst,exp_a_regression}.yaml`,
  `loader_10ghz.py`, `manifest.py`, `qc/screens.py`, `preprocess/pipeline.py`,
  `features/extraction.py` are **not edited** (the promoted helpers are moved from the audit
  script, not from the 10 GHz pipeline). The one shared-file exception is the additive
  `config.py` schema and the additive `provenance.py` parameter, both back-compatible.
- **`tests/test_no_leakage.py`** stays **byte-for-byte unmodified since M1** and green.

**The milestone-5 invariant, protected above all** (CLAUDE.md §Hard invariants;
implementation_plan.md §"Analysis unit", §"Fit-on-train-only"):

> **The 77 GHz QC, preprocessing, and WST + pooling are deterministic per-frame functions of
> (one frame's raw cube, frozen constants). They contain NO fitted quantities and NO
> cross-frame, cross-session, cross-subject, or cross-role statistics — except the single
> declared session aggregation (concat of per-frame mean + median), a fixed pair of
> statistics, not a fitted transform.**

Concretely, mirroring the M4 argument: MTI subtracts a *within-frame* mean over that frame's
own chirps; the Butterworth/Hann/FFT are fixed operators; the QC histograms are per-(Rx,
chirp) trace within one frame; the scattering transform is a fixed filter bank; the
order-aware log is a fixed pointwise map; range-bin averaging and Rx fusion are fixed moments
over fixed axes; pooling is fixed moments over fixed segments; the session aggregation is two
fixed statistics with a frozen concat order. Nothing is estimated on one set and applied to
another → **M5 introduces no leakage vector** and nothing here enters a CV loop. Forbidden,
exactly as at M4: fitting any scaler/PCA on the 77 GHz coefficients (that lives in the M7 CV
harness); computing a pooling statistic across frames before the declared aggregation;
selecting a tiling / log switch / fusion statistic / QC threshold from data anywhere in M5.

**Every 77 GHz alternative is classified NOW, before M6's freeze.** Three kinds, and nothing
may migrate between them after results exist:

- **Inner-CV search axes** (selected per outer fold on inner folds only, at M7): **tiling
  {T1,T2,T3}** and the **log axis** — which, mirroring the 10 GHz plan (§"WST
  parameterization"/§"LOSO harness", A-M4-7), carries **three** mutually exclusive branches:
  **off / on+frozen-ε (1e-6) / on+tuned-ε** (a fold-local, scale-relative *per-order* ε
  computed **train-only** in the M7 CV harness, gated by the pre-registered
  order-2-usefulness pre-check, confirmed/frozen at M6). Call arguments, not config — one
  config serves every variant (the M4 reduction/channel precedent). **M5 preserves the
  pre-log / order-separated information and API the tuned-ε branch needs** (the
  `prelog_scale` of §2.6) but **does not compute the tuned ε** — that is fold-local and lives
  in the M7 harness, never in M5 (computing it here would be leakage). M6 applies the
  order-2-usefulness gate.
- **Primary modeling path vs labeled secondary variant** (fixed by Exp G, **not** search
  axes): **Rx fusion = mean is primary and frozen; median is a labeled secondary variant**
  (implementation_plan.md Exp G fixes "mean primary, median secondary" in both the rationale
  and the executable chain). **Feature family = pooled statistics is the classical modeling
  family; raw-flattened ("flat") is diagnostic/DL-only**, session-aggregated before any
  classical metric (§"Analysis unit": raw-flattened WST is "diagnostic only"). M5 *computes*
  both fusion statistics and both families (for diagnostics + the DL path), but neither the
  median nor the flat family is an inner-CV search axis; promoting either to a competing axis
  at M6 would require a prior authoritative amendment.
- **Frozen protocol constants** (neither searched nor ablated): `max_order = 2`,
  `log_epsilon = 1e-6`, the order-aware log rule, the Doppler tilings (Q, invariance_ms), the
  2–4 m gate, the QC thresholds (histogram_bins 128, flatline_max_bin_fraction 0.25,
  min_in_band_energy_ratio 0.30, in_band_margin_hz 1953.125), pooling definitions, session
  concat order, kymatio options (`out_type='array'`). Configurable only so a run's YAML is a
  complete record and tests can drive behaviour; non-default values rejected by the §2.7
  canonical-spec guard.
- **Implementation choice validated by equivalence**: `wst77.backend ∈ {numpy, torch}`,
  default `numpy` — the M4 policy, unchanged. numpy backs all reported 77 GHz features; the
  torch frontend is validation + unreported work only, and **passing a 77 GHz cross-backend
  agreement test (§3, T-W77) is the precondition for any torch-frontend use** (the
  precondition rule of implementation_plan.md §"WST parameterization").

**Ground rules:** work on `v1_milestone_5`; commits only when the owner asks — build steps
*write and verify* artifacts, they never commit them; HISTORY.md written continuously
(failures kept, newest-first); superseded material → `archive/` with a note.

---

## §1 Build sequence — exact order and why

Tests land in the same step as their module. HISTORY.md gets **at least** one entry per
resolved step; every failed or superseded attempt inside a step gets its own entry.

**Two gates govern the order.** (i) An **authoritative-amendment prerequisite** (step 0):
because this milestone rescopes and renumbers the authoritative schedules, A-M5-1 (scope) and
A-M5-2 (renumber) must be **owner-approved and written into `implementation_plan.md` and
`ROADMAP.md` before any implementation begins** — a task plan cannot renumber/rescope the
authorities by assertion and repair them at close-out (§6). (ii) A **flatline-rule
specification decision** (step 6): the rule's final form is decided on **ADC-quantisation
mechanism grounds from the M2 single-file audit — never from full-cohort survival** — and
frozen **before** the cohort QC run, so no reported feature artifact and no eligibility number
feeds back into a preprocessing choice (C5-03; §2.4/§2.7/A-M5-6). Steps 8–10 proceed in
parallel with 6–7; only the cohort feature runs (step 11) sit behind both gates.

| # | Step | Gate? | Why this position |
|---|------|:---:|-------------------|
| 0 | **Authoritative-amendment prerequisite (owner-approved).** Apply A-M5-1 (scope: 77 GHz → full parallel arm) + A-M5-2 (build-order renumber) to `implementation_plan.md` **and** `ROADMAP.md` (§§0/2/7) | **BLOCKS all** | The authorities must reflect the rescope/renumber before code is written against them (C5-01/C5-02); ROADMAP edits need explicit owner approval per CLAUDE.md |
| 1 | Plan doc (this file) + HISTORY.md opening entry for M5 | – | Establishes the approved design before code (M2–M4 pattern) |
| 2 | **Config**: the three `*77` dataclasses + validators + `require_77ghz_dir` + section whitelist + 4 YAML files + `tests/test_config.py` (T-C77) | – | Every module reads these; schema/validation first (M1–M4 rule). Env already complete (h5py since M2; scipy `<1.17` pinned) |
| 3 | **Loader**: `loader_77ghz.py` (+ promote `reverse_axes`/`to_numeric`; audit re-imports) + `tests/test_loader77.py` (T-L77) | – | Everything downstream consumes the loaded cube shape/dtype; the real-float finding must be enforced first |
| 4 | **Manifest**: `manifest_77.py` + `tests/test_manifest77.py` (T-M77) | – | The cohort passes need the frame inventory + eligibility columns |
| 5 | **QC screens + axis check**: `screens_77.py`, `axis_check_77.py` + `tests/test_qc77.py` (T-Q77) | – | The cohort QC run (step 7) needs both; per-frame functions have no upstream deps |
| 6 | **Flatline-rule specification decision (owner; mechanism-based). [Owner pre-selected outcome (b), 2026-07-23.]** Specify the exact mechanism-corrected replacement of the **77 GHz** flatline rule on ADC-quantisation mechanism grounds using the **M2 single-file audit** (the sanctioned evidence, implementation_plan.md §"Frozen 77 GHz pipeline") + quantisation physics — **not** cohort survival (the 10 GHz screen is untouched). Three admissible outcomes: **(a) keep the frozen rule; (b) a mechanism-corrected exact replacement**, specified a priori and independent of the cohort outcome — **under (b) a substep first revises every already-built artifact that encoded the old any-trace rule (the `QC77Config` defaults, `canonical_spec_guard_77`, `screens_77.py`, the `configs/*77.yaml`/provenance representation) and reruns T-C77/T-Q77 green BEFORE the step-7 gate opens**, so step 7 can never execute the stale rule (C5-21); **both (a) and (b) then proceed to the cohort feature runs (step 11)**; **(c) declare it data-adaptive → it moves inside inner CV** (fit per training fold at M7, §"QC screens"), which means there is **no single frozen eligibility**, so **under (c) M5 STOPS before the cohort feature artifacts** — QC-dependent extraction defers to the M7 harness (C5-14), and M5 delivers only the code + a characterization QC run. Recorded in HISTORY + A-M5-6, applied uniformly | **GATE** | Pins the rule the cohort run executes (a/b) or forks M5's scope (c); keeps the choice leakage-safe (no cohort-derived preprocessing selection) |
| 7 | **Cohort QC run**: `run_qc77.py` over all 80 files. Under **(a)/(b)** → authoritative `qc_survival_77ghz.csv` (survival + `qc_pass`/eligibility + axis cert) + per-frame CSV + label-blind flatline diagnostics; under **(c)** → threshold-independent `qc_characterization_77ghz.csv` (raw flatline/histogram stats + NaN/Inf + in-band + axis cert, **no** authoritative eligibility; C5-22) + provenance (local ~3–5 h, or a single IBEX job) | – | First contact with the whole cohort; eligibility is a downstream consequence of the step-6 rule under (a)/(b), and is deferred to M7 under (c) |
| 8 | **Preprocessing**: `pipeline_77.py` + `tests/test_preprocess77.py` (T-P77) | – (parallel with 6–7) | Chain steps 1–5; code independent of the flatline outcome |
| 9 | **WST extraction**: `extraction_77.py` + `tests/test_wst77.py` (T-W77) + one-file non-curated `--realdata` smoke (T-R77) | – (parallel) | Steps 6–10; independent of the flatline outcome |
| 10 | **IBEX scaffolding**: `configs/ibex.yaml`, `scripts/ibex/*.sbatch` + `README.md`, `.gitattributes`; local `run_wst77.py --smoke --subject 1 --session 8am` **timed** (non-curated, all outcomes; calibrates sbatch limits) | – | Needs steps 8–9 to exist to smoke; the timing sets the array time limit |
| 11 | **Cohort feature runs**: `run_preprocess77.py` (single cohort job) + `run_wst77.py` (sharded array) over the **eligible** cohort (IBEX; local fallback), WST shard merge, curated CSVs + provenance | **BLOCKED by 6; reached only under outcomes (a)/(b)** | No reported feature artifact may predate the frozen QC rule; under (c) this step does not run — extraction defers to M7 |
| 12 | **Close-out**: remaining A-M5-* (A-M5-3..8) applied to `implementation_plan.md`, HISTORY close entry, SECOND_CHAPTER §4, DoD checks | after 11 **(a/b)** or after 10 + characterization QC **(c)** | CLAUDE.md write-cadence; reachable under every step-6 outcome — (a)/(b) close on the cohort feature artifacts, (c) closes on the code + characterization-QC deliverables (C5-20) |

---

## §2 Per-file specifications

Format per file: **Responsibility** (single) · **Public API** · **Frozen parameters** ·
**Acceptance criteria**. Constants stated here are the audited/reference values from
`experiments/audit_77ghz.py`, `results/qc/audit_77ghz.json`, and
`matlab/77ghz_code/{filter_gpt_butterworth77,chirpavg_and_fuse_batch,wst_extract77}.m` — to
be **re-confirmed at build**, never assumed.

### 2.1 Config: `src/dehyd/config.py` + `configs/{data77,preprocess77,wst77,exp_77ghz}.yaml`

**Responsibility.** The M5-consumed 77 GHz constants, validated at build time (M2's rule: a
bad value fails at config load, not deep inside h5py/kymatio). **Decision: three parallel
top-level sections (`qc77`, `preprocess77`, `wst77`), not a nested `band77` block.** Reasons:
the whole config machinery (`_known_section`, per-section field whitelists, `_reject_unknown`
on `known_sections`) is built per top-level section — parallel sections reuse it verbatim,
nesting needs a second-level schema walker (indirection CLAUDE.md forbids); and the 77 GHz
values are *different physics*, not overrides of the 10 GHz defaults — separate frozen
dataclasses give each band its own canonical spec, so `canonical_spec_guard_77` can compare
against `Preprocess77Config()`/`WST77Config()` exactly as the 10 GHz guard does.

**New dataclasses (frozen; defaults ARE the frozen spec):**

```python
@dataclass(frozen=True)
class Preprocess77Config:
    butter_order: int = 4
    gate_m: tuple[float, float] = (2.0, 4.0)   # ONE gate: chain crop AND QC mask
    fs_hz: float = 500e3
    bandwidth_hz: float = 2e9
    chirp_time_s: float = 512e-6               # PRF = 1/chirp_time_s = 1953.125 Hz (derived)
    standardize: str = "robust"                # per-channel robust z before WST (A-M5-3)

@dataclass(frozen=True)
class QC77Config:
    histogram_bins: int = 128
    flatline_max_bin_fraction: float = 0.25    # any bin >= 64 of 256 flags a (Rx,chirp) trace
    min_in_band_energy_ratio: float = 0.30
    in_band_margin_hz: float = 1953.125        # one FFT bin = fs/256, frozen a priori
    min_frame_fraction: float = 0.5

@dataclass(frozen=True)
class WST77Config:
    tilings: tuple[WSTTiling, ...] = (         # Doppler tilings at fs = PRF (wst_extract77.m)
        WSTTiling(q=(8, 4), invariance_ms=20.0),
        WSTTiling(q=(6, 4), invariance_ms=40.0),
        WSTTiling(q=(4, 2), invariance_ms=60.0),
    )
    max_order: int = 2
    log_epsilon: float = 1e-6
    backend: str = "numpy"
```

**`PathsConfig`** gains `data_77ghz_dir: Path | None = None` (optional so existing 10 GHz
configs load unchanged); `_build_paths` allows it and checks existence only when present. New
helper `require_77ghz_dir(config) -> Path` raises `ConfigError` (pointed message) when it is
`None` — called by every 77 GHz entrypoint before any I/O.

**Frozen design decisions embedded above (each stated so it is not re-litigated):**

- **One gate, in `preprocess77.gate_m`.** At 10 GHz the QC gate (0.9–3.0 m) deliberately
  differs from the model gate (1–2 m) to fix the evaluable population; the Exp G spec freezes
  a **single 2–4 m gate** for both the chain crop and the QC mask, so a separate
  `qc77.qc_gate_m` would only create a drift channel. `screens_77` reads the gate from
  `Preprocess77Config`. A 77 GHz gate *search* would be a plan amendment, not a config edit.
- **No RMS robust-z screen at 77 GHz.** The frozen Exp G QC lists exactly three screens
  (NaN/Inf, flatline, in-band); the 10 GHz RMS *diagnostic* is not specified for 77 GHz and
  is not added.
- **The flatline literals (`histogram_bins 128`, `flatline_max_bin_fraction 0.25`, the
  any-trace rule) are the outcome-(a) defaults** (§1 step 6). Under step-6 outcome (b) they
  are **replaced wholesale** by the mechanism-corrected rule in the step-6b substep — the
  `QC77Config` defaults, `canonical_spec_guard_77`, `screens_77.py`, the YAML/provenance, and
  the T-C77/T-Q77 literal-pinning tests are all updated and rerun **before** the cohort QC
  gate (C5-21), so no artifact keeps a stale value.
- **`wst77.tilings` is code-frozen** — `_build_wst77` rejects a YAML `tilings` key with the
  same error style as `_build_wst`. `max_order` (∈ {1,2}), `log_epsilon` (>0, finite),
  `backend` (∈ `BACKENDS`) validated with the existing `_number`/`_choice_field` helpers.
- **Cross-check** `_check_qc77_band(qc77, pre77)` mirrors `_check_qc_band`: the 2–4 m gate →
  beat band ≈ 52.1–104.2 kHz, below Nyquist 250 kHz, non-vacuous. Reuses `beat_band_hz`.
- `known_sections` becomes
  `("paths","run","split","qc","preprocess","wst","qc77","preprocess77","wst77")`. `Config`
  gains the three fields via `default_factory`, so existing configs load unchanged (the new
  sections take defaults and simply appear in provenance from now on — a completeness gain,
  noted in HISTORY).

**New YAML (values stated explicitly, matching code defaults — the `data.yaml` convention;
signed exponents mandatory per the YAML-1.1 trap):**

- `configs/data77.yaml` — `paths: {data_77ghz_dir: data/77ghz}`.
- `configs/preprocess77.yaml` — a `qc77:` block + a `preprocess77:` block (mirroring how
  `preprocess.yaml` carries `qc` + `preprocess`); `fs_hz: 500.0e+3`, `bandwidth_hz: 2.0e+9`,
  `chirp_time_s: 512.0e-6`, `in_band_margin_hz: 1953.125`.
- `configs/wst77.yaml` — `wst77: {max_order: 2, log_epsilon: 1.0e-6, backend: numpy}` with a
  header note that `tilings` is code-frozen (override rejected).
- `configs/exp_77ghz.yaml` — `include: [data.yaml, data77.yaml, preprocess77.yaml, wst77.yaml]`
  + the same `run` (seed 20260721, seed_set [1..5], device cpu) and `split` sections as
  `exp_a_regression.yaml`. This is the top-level entry config the 77 GHz CLIs load.

**Acceptance.** `tests/test_config.py` (T-C77): each new section's defaults present +
overridable; bad values rejected (`histogram_bins` float, `flatline_max_bin_fraction` >1,
`butter_order` 0, `backend` typo, `max_order` 0/3); `wst77.tilings` override rejected;
`data_77ghz_dir` optional, existence-checked when present; `require_77ghz_dir` raises when
absent; `_check_qc77_band` rejects an inverted gate; **existing 10 GHz configs still load and
their `config_to_dict` is unchanged except for the additive `*77` defaults**.

### 2.2 `src/dehyd/data/loader_77ghz.py`

**Responsibility.** Load one 77 GHz `.mat` (v7.3/HDF5) file to a canonical in-memory cube,
asserting the on-disk contract. Promotes the audited reader; no QC, no preprocessing.

**Public API.**

```python
N_RX = 16; N_CHIRPS = 256; N_FAST = 256; RADAR_VAR = "framesRadar"

def parse_77ghz_filename(path) -> tuple[int, int]
    # (subject, session_idx) from subject_<n>_<session>.mat, regex built from
    # SESSION_NAMES exactly as loader_10ghz. Raises LoaderError77 on any non-match.

def inspect_77ghz_file(path) -> FileInfo77
    # h5py METADATA ONLY (no chunk decompress): dataset RADAR_VAR present; dtype is a
    # REAL float of EXACTLY 8 bytes (kind == "f" AND itemsize == 8 -- float64, the
    # confirmed on-disk contract), with byte order native or explicitly handled; on-disk
    # shape (16, 256, 256, n_frames) with n_frames > 0. float32 / other float widths /
    # unexpected endianness are REJECTED (not silently accepted): format drift is a
    # stop-and-report (C5-18). A COMPOUND (real, imag) dtype is likewise REJECTED, message
    # citing the M2 finding: every production assumption (MTI on real ADC data; I/Q only
    # after the range FFT) rests on the real-float64 finding, so neither is ever coerced.

def load_77ghz_file(path) -> np.ndarray   # float64 [n_frames, 256 fast, 256 chirps, 16 rx]
    # Reads the WHOLE dataset in one call, then reverse_axes. Forced by the chunk layout
    # (16,4,1,125): every chunk spans all frames, so any frame-subset read decompresses
    # the entire file anyway; per-frame reads would decompress it n_frames times.
    # ~1.05 GB in memory. Asserts loaded shape agrees with inspect_77ghz_file.

def reverse_axes(cube) -> np.ndarray      # np.transpose(cube, (3, 2, 1, 0))  [PROMOTED]
def to_numeric(raw) -> np.ndarray         # real float pass-through; compound->complex [PROMOTED]
```

**Frozen semantics.** On-disk axis order `(Nrx, Nchirps, Nfast, Nframes)` → reversed to
`(Nframes, Nfast, Nchirps, Nrx)`, matching `chirpavg_and_fuse_batch.m`. The two size-256 axes
(fast-time vs chirps) are **indistinguishable by shape** — the loader records the assumed
mapping; the **semantic axis check (§2.4) certifies it per file at QC time**, not in the
loader (which must stay cheap). `reverse_axes`/`to_numeric` are **moved** from
`experiments/audit_77ghz.py`; the audit script imports them back (single copy; existing
`tests/test_audit_77ghz.py` keeps guarding them).

**Acceptance.** T-L77 (see §3): filename parse; **one full-shape `(16,256,256,1–2)` fixture
accepted** (the exact-shape success path — small-dim fixtures would be rejected by the
contract, so they serve only helpers/rejection, C5-23); wrong shape / dtype (incl. float32 &
endianness, C5-18) / compound / missing-var rejected; `reverse_axes` round-trip on small dims;
loaded values equal the written fixture bit-for-bit.

### 2.3 `src/dehyd/data/manifest_77.py`

**Responsibility.** The 77 GHz frame inventory, QC join, and session-eligibility columns.
**Decision: mirror `manifest.py`, do not parameterize it.** Injecting a loader/QC/path-attr
triple into `build_manifest`/`apply_qc` is exactly the factory-style indirection CLAUDE.md
rules out, and it would touch the frozen, artifact-pinned 10 GHz path for zero 10 GHz
benefit. The divergent-copy risk is contained by **sharing the genuinely subtle pieces via
import** rather than copying them.

**Reused by import from `manifest.py`** (column-generic — they only touch
`qc_pass`/`session_eligible`/`subject`): `_join_qc` (the fail-closed one-to-one QC join —
where a silent bug would drop/duplicate frames), `eligible_frames`, `evaluable_subjects`.

**Mirrored with explicit code** (shallow bookkeeping whose columns differ anyway; each
commented as mirroring its `manifest.py` sibling):

```python
QC77_COLUMN_DTYPES = {... qc_nan_inf, qc_flatline, qc_low_in_band, qc_pass, qc_fail_any,
                      qc_in_band_ratio, qc_n_flatline_traces, qc_rx_max_flatline,
                      session_n_pass, session_min_pass, session_eligible ...}

def build_manifest_77(paths, gt) -> DataFrame     # C1–C6 inventory + bijection vs ground
    # truth against paths.data_77ghz_dir via inspect_77ghz_file; actual per-file n_frames.
def resolve_path_77(paths, rel_path) -> Path
def apply_qc_77(manifest, paths, config) -> DataFrame   # load each file once -> run_qc_cube_77
    # -> _join_qc (imported) -> session_n_pass, session_min_pass = ceil(min_frame_fraction *
    # n_frames_in_file), session_eligible.
def session_qc_report_77(manifest_qc) -> DataFrame       # 77 reason columns; no RMS
```

**Acceptance.** T-M77: bijection failures raise (missing / duplicate / stray / unmatched
files); frame counts read from files, never assumed 125; the `ceil(0.5 × n_frames)`
eligibility rule; `session_n_pass + n_fail_any == n_frames` identity; the imported `_join_qc`
still fails closed on dup/drop/misattach.

### 2.4 `src/dehyd/qc/screens_77.py` + `src/dehyd/qc/axis_check_77.py`

**Responsibility.** The frozen 77 GHz per-frame QC screens, and the semantic axis check.
Promoted from `audit_77ghz.py` (`qc_smoke_frame`, `qc_in_band_mask_77`, `axis_metrics`,
`axis_verdict`, `range_gate_bins`); the audit imports them back.

**`screens_77.py` API.**

```python
@dataclass(frozen=True)
class FrameQC77:
    nan_inf: bool; flatline: bool; low_in_band: bool
    in_band_ratio: float; n_flatline_traces: int; per_rx_flatline: tuple[int, ...]
    @property
    def passed(self) -> bool:   # not (nan_inf or flatline or low_in_band) -- can't be
        ...                     # violated by construction (the 10 GHz FrameQC pattern)

def run_qc_frame_77(frame, qc77, pre77) -> FrameQC77
    # frame [256 fast, 256 chirp, 16 rx]. Non-finite short-circuits.
    # FLATLINE (frozen, per-(Rx,chirp) trace): 128-bin magnitude histogram over the
    # trace's own [min,max]; degenerate spread (edges[:-1] >= edges[1:]) counts as
    # flatline; else flag if max bin count >= 0.25 * 256 = 64. Frame's n_flatline_traces =
    # sum over all 16*256 = 4096 traces; frame FAILS if ANY trace flags (the 10 GHz
    # any-chirp rule at ~205x multiplicity: (16*256)/20 = 204.8).
    # IN-BAND: periodic-Hann range FFT, half-spectrum power averaged over all chirps & Rx,
    # ratio = power[mask].sum()/power.sum(); mask from screens.in_band_mask REUSED AS-IS
    # (already parameterized): in_band_mask(256, 500e3, 2e9, 512e-6, (2,4), 1953.125)
    # -> bins 26..54. low_in_band = ratio < 0.30.

def run_qc_cube_77(cube, qc77, pre77) -> list[FrameQC77]   # plain per-frame loop (no cross-
    # frame statistic; structurally leak-proof; stated in the docstring)
```

**`axis_check_77.py` API.** `range_gate_bins(n_fast, bandwidth_hz, gate_m)`,
`axis_metrics(cube, fast_axis, chirp_axis, ...)`, `axis_verdict(metrics, ...)` — promoted.
Placement in the promoted pipeline: the check **runs once per file inside the cohort QC run**
(`run_qc77.py`), on the **raw pre-MTI cube that run has already fully decompressed** (MTI
would strip the near-zero-Doppler energy the check keys on). Not in the loader (it would tax
every downstream load, incl. 80 IBEX array tasks, with ~15 s of whole-cube FFTs) and not a
separate audit pass (it would double the dominant 80-file decompress cost). The survival CSV
gains per-session `axis_G_fast`, `axis_D_chirp`, `axis_verdict`, the raw-file `sha256`, and
the `axis_spec_hash` (below); the run **fails closed on any verdict other than `ACCEPTED`** —
`REJECTED` **and**
`INCONCLUSIVE` both fail (Exp G: "the assignment is accepted only if this structure appears …
otherwise the loader fails"; an inconclusive shape-indistinguishable mapping is not
certification). `range_gate_bins` has its single home here; `pipeline_77` imports it.

**The axis check is a hard guard on every extraction/preprocessing entrypoint, not just the
QC run** (C5-08). A file may reach `run_wst77.py`/`run_preprocess77.py` or the local smoke
before or without the cohort QC artifact. So each such entrypoint, per file, either **(i)
verifies an `ACCEPTED` axis-check record keyed to that file's exact raw `sha256` + an
`axis_spec_hash`** (read from `qc_survival_77ghz.csv`), or **(ii) runs the raw semantic check
inline** on that file (one whole-cube FFT, ~15 s) and requires `ACCEPTED`. Any non-`ACCEPTED`
verdict, or a record whose `sha256`/`axis_spec_hash` does not match, **aborts before any
feature is written**. The local smoke follows the identical guard — so features can never be
produced from an uncertified axis mapping, and step 10 running before step 7 is safe.

**`axis_spec_hash` (the stable certification key, C5-16).** A hash over exactly the inputs
that change the axis verdict — the axis-check **algorithm version**, its thresholds
(`AXIS_MIN_DC_FRACTION`, `AXIS_MIN_GATE_FRACTION`, `AXIS_DOMINANCE_FACTOR`, `AXIS_DC_HALFWIDTH`),
the expected shape + representation, and the gate/bandwidth/fs used to place the range-gate
bins — and **excluding environment-specific fields** (`paths.*`, `results_dir`, device). So a
**path-only overlay (e.g. `ibex.yaml`) leaves the certificate valid** after rsync/local merge,
while **changing any axis-relevant constant invalidates it**. T-Q77/T-A77 assert both:
a path-only overlay is accepted, a changed axis-relevant field is rejected.

**Frozen parameters** (audited): `AXIS_MIN_DC_FRACTION 0.5`, `AXIS_MIN_GATE_FRACTION 0.05`,
`AXIS_DOMINANCE_FACTOR 10.0`, `AXIS_DC_HALFWIDTH 3`; gate bins 27..53, QC mask bins 26..54.

**Acceptance.** T-Q77: flatline fires on a quantised trace and on a degenerate-spread trace;
the any-trace rule (one bad trace of 4096 fails the frame); per-Rx counts correct; in-band
ratio separates in-gate vs out-of-gate tones; NaN/Inf short-circuit contract; mask bins
26..54 pinned; per-frame independence (permuting other frames changes nothing); axis verdict
ACCEPTED/REJECTED/INCONCLUSIVE on synthetic tone cubes (inherited from `test_audit_77ghz`
where the code moved).

### 2.5 `src/dehyd/preprocess/pipeline_77.py`

**Responsibility.** Chain steps 1–5: the executable per-frame front of the Exp G primary
chain. Reuses the fs-agnostic `filters.py`. No EdgeTrim, no reduction (the Doppler chain has
neither; `wst_extract77.m`'s EdgeTrim=8 belongs to the *fast-time secondary* branch, which is
deferred — A-M5-7).

**Public API.**

```python
def preprocess_frame_77(frame, pre77) -> np.ndarray   # complex128 [27 gate_bins, 256 chirps, 16 rx]
    # 1. MTI:      frame - frame.mean(axis=1, keepdims=True)   # per-fast-bin mean over chirps
    # 2. bandpass: design_bandpass_sos(f_lo, f_hi, 500e3, 4) + bandpass_filtfilt(x, sos,
    #              axis=0) with beat_band_hz(gate_m, 2e9, 512e-6)   [REUSED from filters.py]
    # 3. Hann:     hann(256, sym=True) on the fast-time axis (the CHAIN window -- distinct
    #              from QC's periodic Hann, as the audit recorded)
    # 4. range FFT: np.fft.fft(..., axis=0)   (256-pt)
    # 5. crop:     range_gate_bins_77(256, 2e9, (2,4)) -> bins 27..53

def preprocess_cube_77(cube, pre77) -> np.ndarray   # [N, 27, 256, 16], plain per-frame loop
```

**Frozen semantics.** MTI is a within-frame operation (subtract this frame's own per-fast-bin
chirp mean) — no cross-frame statistic. The complex range-FFT output (step 4) is the **first
point I/Q exists** (the M2 real-float finding: nothing before step 4 may assume complex
input). filtfilt is zero-phase (no range-peak shift). Energy accounting uses the audit's
Parseval convention (spectral energies ÷ transform length).

**Acceptance.** T-P77: MTI exactly kills a chirp-constant (static) target and preserves a
Doppler tone; zero-phase (no peak shift); axis correctness on synthetic moving-target cubes (a
tone at gate range along fast-time survives the crop; the fast↔chirp-swapped cube does not);
gate-crop bins 27..53; per-frame loop determinism; stage energies match the audit convention.

### 2.6 `src/dehyd/features/extraction_77.py`

**Responsibility.** Chain steps 6–10 as a linear, followable composition of the fs/shape-
agnostic `wst.py`/`pooling.py` functions and `standardize.to_channels`. No fitted state, no
selection.

**End-to-end shapes (per session):**

```
load          [N~125, 256 fast, 256 chirp, 16 rx]  float64     ~1.05 GB
steps 1-5     [N, 27 bins, 256 chirp, 16 rx]        complex128  ~221 MB
step 6 (per frame): 16 rx x 27 bins = 432 COMPLEX slow-time series (len 256), each
              -> to_channels(series, "iq", pre77.standardize) -> [2, 256] float64
              -> batch [432, 2, 256]  (N=432 series, C=2 channels; rx-major, bin-minor
                 -- a FROZEN fold order. kymatio folds the 432x2 = 864 REAL signals into
                 its internal batch dim, but the scatter_frames API tensor is [432, 2, .])
scatter       scatter_frames([432,2,256], sc) -> S [432, 2, P, t]   (P, t MEASURED)
step 7        reshape [16, 27, 2, P, t] -> mean over bins -> per-Rx [16, 2, P, t]
step 8        Rx fusion: mean (primary) / median (secondary) over rx -> [2, P, t]
log           apply_order_log on the FUSED tensor (A-M5-4); log branch off/on+frozen-ε
              a call arg; the on+tuned-ε branch is applied in the M7 harness (train-only)
step 9        pool_stats -> per-frame vector [D]     (or flatten_series for "flat"; diag/DL)
step 10       aggregate_session([N, D]) -> [2D]  (concat frame-mean + frame-median)
```

**Public API.**

```python
def slow_time_signal_batch(gated_frame, standardize) -> np.ndarray   # [432, 2, 256]
    # 432 = 16 rx x 27 bins complex series; split into 2 real channels {real, imag} ->
    # [432, 2, 256], then VECTORIZED robust-standardize along the time axis (median & MAD
    # over axis=-1 for all 864 channels at once) -- NOT a 432-iteration to_channels loop
    # (C5-19). The batch form is bit-equivalent to stacked to_channels(series,"iq",...)
    # calls (same median/MAD/1.4826/eps), pinned by a bit-equivalence test (T-W77); this
    # is why the "no Python loops" claim holds for standardization too. RAISES if any
    # standardized channel is all-zero (constant slow-time -> MAD 0 -> zeros); the plan's
    # nonzero-energy assertion (implementation_plan.md line ~884).
def extract_frame_features_77(gated_frame, scattering, meta, wst77, *, log_branch, fusion,
                              family, epsilon_by_order=None)
def extract_session_features_77(cube, pre77, wst77, *, tiling, log_branch, fusion, family,
                                epsilon_by_order=None) -> np.ndarray
    # single-variant reference: preprocess_cube_77 -> per-frame scatter/fuse/log/pool ->
    # aggregate_session. Correct by construction. log_branch in {off, on_frozen_eps,
    # on_tuned_eps}. For on_tuned_eps the caller (the M7 harness) supplies epsilon_by_order
    # (order -> ε), applied at the log step BEFORE pooling via re-extraction; M5 never
    # computes ε itself (C5-15). off/on_frozen_eps ignore epsilon_by_order.

@dataclass(frozen=True)
class SessionVariant77Result:
    vectors: dict       # {(tiling_idx, log_branch, fusion, family): session_vector [2D]}
                        # log_branch in {off, on_frozen_eps}; fusion in {mean, median};
                        # family in {pooled, flat}
    prelog_scale: dict  # {(tiling_idx, fusion): (v0, v1, v2)} -- pre-log per-order (0,1,2)
                        # coefficient scale on the FUSED tensor. KEYED BY fusion because
                        # mean vs median fusion give different pre-log tensors (C5-10).
                        # Statistic (frozen, as M4 §2.6): per frame, order o in {1,2} ->
                        # mean over time -> per-path scalar -> mean over order-o paths ->
                        # per-channel -> mean over channels = v_o; order 0 = the signed
                        # order-0 time-mean, mean over channels; session value = MEDIAN over
                        # the session's frames. This is the train-fold-computable scale the
                        # M7 on+tuned-ε log branch consumes (A-M4-7); M5 records it, never
                        # applies a data-dependent ε.
    shapes: dict        # {tiling_idx: (n_paths, n_time)}
    all_finite: bool

def extract_session_variants_77(cube, pre77, wst77) -> SessionVariant77Result
    # COHORT-LOOP form: preprocess once, scatter once per tiling, derive all
    # (log_branch{off,on_frozen_eps} x fusion{mean,median} x family{pooled,flat}) vectors
    # + the (tiling,fusion) pre-log scales from each shared raw tensor. IN-RUN reuse of a
    # deterministic intermediate, NOT a persistent cache (the M4 distinction). Computing
    # both fusions/families here is for diagnostics + the DL path; only (mean, pooled) is
    # the primary classical modeling path (median/flat are labeled secondary/diagnostic).

def canonical_spec_guard_77(config) -> None
    # Raises (naming deviating fields) unless config.preprocess77 == Preprocess77Config()
    # AND config.qc77 == QC77Config() AND config.wst77 == WST77Config() (incl backend=="numpy").
def wst77_spec(wst77, pre77) -> dict   # measured geometry at n_in=256, fs=PRF
```

**Frozen semantics & the two spec-gap resolutions.**

- **Batch unit = one frame = 432 complex slow-time series** (16 rx × 27 bins), scattered as
  `scatter_frames([432, 2, 256])` (N=432 series × C=2 real channels; kymatio internally folds
  the 432×2 = 864 real signals into its batch dim). Deep in the M4 batching-saturation regime;
  peak scattering output ~10–25 MB/call (measured), vs >2 GB if a whole session × tiling were
  scattered at once. The **27-bin and 16-Rx loops are folded into the batch dim, never Python
  loops** (the M4 `pool_stats` hotspot lesson — a per-bin loop is the 54× trap). Per-frame S
  is immediately reduced to per-Rx means and the raw tensor discarded.
- **Standardization (A-M5-3, spec gap closed).** Exp G step 6 is silent on standardization;
  `wst_extract77.m` robust-standardizes every signal before WST, and the 10 GHz chain does so
  per channel via `to_channels`. Resolution: **each slow-time series' real and imag channels
  are robust-standardized separately** (own median/MAD) — the 10 GHz-consistent, coherent
  reading of the reference. To keep the step loop-free it uses a **vectorized batch
  standardization** (median/MAD over the time axis for all 864 channels), proven
  **bit-equivalent to stacked `to_channels(series, "iq", pre77.standardize)`** by a T-W77
  test (C5-19) — so the definition still equals the 10 GHz `to_channels` semantics.
- **Log placement (A-M5-4, spec clarification).** The order-aware log applies to the **fused**
  per-frame tensor (after bin-averaging and Rx fusion, before pooling) — the exact analog of
  a 10 GHz per-frame S; log-then-average would change the declared "average the scattering
  matrices" semantics.
- **Log axis = three branches, mirroring 10 GHz (A-M4-7), with an executable tuned-ε handoff
  (C5-15).** The log axis is **off / on+frozen-ε (1e-6) / on+tuned-ε**. Because
  `log(S + ε)` happens **before** path pooling and frame aggregation, a fold-local ε **cannot**
  be applied to an already-pooled session vector — so `prelog_scale` (the per-`(tiling,fusion)`
  scale summaries) is only what M7 uses to **compute** ε on training subjects; **applying** it
  requires **re-extraction**. M5 therefore ships the handoff, not just the scale:
  `apply_order_log` / the extraction API accept an optional **`epsilon_by_order`** mapping
  (order→ε), and when `log_branch == "on_tuned_eps"` the per-order ε is applied at the log
  step before pooling. The M7 harness then: fits `epsilon_by_order` from **training subjects
  only** (via their `prelog_scale`), and **re-extracts every train/val/test session with that
  one frozen fold ε** — the same ε to all roles. M5 realizes the two data-independent branches
  (off, on+frozen-ε) end-to-end, and provides + tests the tuned-ε application path; it **never
  computes a data-dependent ε itself**. The third branch is gated by the order-2-usefulness
  pre-check confirmed/frozen at M6. A T-W77 test asserts a given `epsilon_by_order` enters
  before pooling and is applied identically regardless of role (no leakage in the mechanism).
- **Rx fusion (mean primary/frozen) and family (pooled classical / flat diagnostic).** The
  extraction computes mean **and** median fusion, and pooled **and** flat families, but the
  **primary classical modeling path is (mean, pooled)**; median is a labeled secondary fusion
  variant and flat is diagnostic/DL-only (§0). Neither is an inner-CV search axis.
- **Cross-backend precondition (C5-13).** `wst77.backend` defaults to `numpy` (canonical for
  all reported features). The torch frontend may be used only after a **77 GHz cross-backend
  agreement test passes** (§3, T-W77) — the precondition rule of implementation_plan.md
  §"WST parameterization", here over the 77 GHz input length and Doppler tilings, both log
  states, on the raw scattering tensors and the pooled vectors.
- **Measured geometry** at n_in=256, fs=PRF=1953.125: expected realized T ≈ 39/78/117 samples
  and J ≈ 6/7/7 for the three tilings, pad expected 512 — all **measured from the
  instantiated bank, never assumed** (the M4 off-by-one lesson); the border warning will fire
  and must be asserted present, never silenced.

**Acceptance.** T-W77: realized T/J pinned as regression values; measured (P,t) finite and
consistent; border warning asserted; batched ≡ single bit-identical; the `[432, 2, 256]`
scatter input and `[16, 27, 2, P, t]` reshape shapes pinned (a `[864, …]` input would fail);
rx-major/bin-minor fold order pinned via a distinct-constant-per-Rx construction; range-bin
averaging equals the hand-computed mean of per-bin scatters; fusion mean vs median
distinguishable; fused-then-log ≠ log-then-fused (correct one pinned); per-channel
standardization applied (a signal with distinct real/imag scales); zero-energy assertion
fires; `prelog_scale` keyed by `(tiling, fusion)` matches a hand-computed per-order scale for
both fusions; pooled/session dims match the layout helpers; `extract_session_variants_77`
equals the single-variant API for every combination; **a 77 GHz numpy-vs-torch cross-backend
agreement test** (reusing `backend_agreement`) passes over both log states on the raw
scattering tensors and the pooled vectors — the precondition for `backend: torch`.

### 2.7 `experiments/{run_qc77,run_preprocess77,run_wst77}.py`

**Responsibility.** Thin CLIs over the library modules (the `run_qc`/`run_wst` pattern) — arg
parsing, guard call, session loop, CSV/provenance writing; no extraction logic of their own.
`--config` repeatable (later wins, for the `ibex.yaml` overlay). Config is the sole
output-path authority; every CSV is re-read and reconciled after writing.

- **`run_qc77.py`** — two modes selected by the step-6 outcome (C5-22):
  - **Outcomes (a)/(b) — authoritative survival/eligibility.** Full-cohort QC with the
    **frozen** flatline rule + per-file semantic axis check. Writes
    `results/qc/qc_survival_77ghz.csv` (session-level `qc_pass`/`session_eligible` + axis
    columns + raw-file `sha256` + `axis_spec_hash`), `results/qc/qc_frames_77ghz.csv`
    (per-frame diagnostics), and a **label-blind flatline diagnostics** section (distribution
    of `n_flatline_traces` per frame, per-Rx totals, flatline-only vs in-band-only failures,
    survival, eligibility). This is a recorded finding/consequence of the **already-frozen**
    rule, not an input to choosing it (C5-03). `record_run(..., stage="milestone-5-qc77")`.
  - **Outcome (c) — threshold-INDEPENDENT characterization only.** A data-adaptive flatline
    rule has **no cohort-wide threshold**, so this run **cannot** emit authoritative pass/
    eligibility. It writes a distinctly named `results/qc/qc_characterization_77ghz.csv`
    (stage `milestone-5-qc77-characterization`, provenance flag `eligibility_authoritative =
    false`) carrying only **threshold-independent raw diagnostics** — per-(Rx,chirp) flatline
    **trace/histogram statistics** (not a flag), the frozen NaN/Inf and in-band screen
    outcomes, and the per-file **axis certification** — and **omits `qc_pass`/
    `session_eligible`** entirely. The distinct name + role marker make it impossible for an
    M7 entrypoint to mistake it for frozen eligibility; extraction is prohibited from
    consuming it and defers all flatline QC + eligibility to the fold-local M7 harness.
  - Both modes are label-blind (no Δm%) and **fail closed on any non-`ACCEPTED` axis
    verdict**.
- **`run_preprocess77.py`** — cohort chain diagnostics (per-session stage energies à la the
  audit's `chain_stages`, MTI removal fraction, gate-crop energy ratio) →
  `results/preprocess/preprocess_diagnostics_77ghz.csv`. **A single cohort job, not sharded**
  (the chain steps 1–5 are cheap vs WST; one CSV, no shard/merge protocol — avoids the
  array-race hazard, C5-11). Runs the per-file axis guard (below) before touching a file.
- **`run_wst77.py`** — cohort feature diagnostics → `results/wst/wst_diagnostics_77ghz.csv`
  (dims per tiling, finiteness across ALL variants — pooled and flat, `(tiling,fusion)`
  pre-log scales, timing). Supports `--subject N [--session 8am]` for local smoke AND IBEX
  array sharding (deterministic shard CSVs `results/wst/shards/wst77_s<subj>_<sess>.csv`; an
  ineligible cell writes a skip-marker shard). **Every shard carries a fingerprint sidecar**
  — run/config fingerprint, git revision, the `qc_survival_77ghz.csv` fingerprint, the
  `axis_spec_hash`, and the raw-file `sha256` (C5-17; implementation_plan.md §"Reproducibility
  / run provenance"). `--merge-shards` verifies exactly 80 shards **and rejects any duplicate
  or any shard whose fingerprints disagree** (stale retries, a different QC rule/config/code
  revision, or a changed raw file) **before** writing the curated CSV, plus the eligible-count
  consistency vs `qc_survival_77ghz.csv`. Count-matching alone is insufficient — stale shards
  can mix without changing the count. **Each per-session task recomputes QC for its own file
  only** (per-frame rule, per-session eligibility — no cross-file state; identical
  `run_qc_cube_77` code guarantees consistency). `canonical_spec_guard_77(config)` first;
  `analysis_role="primary"`. **This curated cohort/array mode requires authoritative
  eligibility, so under step-6 outcome (c) it FAILS CLOSED** (extraction is deferred to the
  M7 fold-local QC harness, which does not exist yet) — it never falls back to the
  non-authoritative any-trace rule (C5-24).
- **`run_wst77.py --smoke` — a non-curated functional smoke, outcome-independent (C5-24).**
  A separate mode for D5/T-R77 that must run under **every** step-6 outcome. It selects a
  **fixed set of frame indices** (e.g. the first K finite frames of one file), applies only
  the **NaN/Inf and axis-certification guards** — **never** the flatline rule / eligibility —
  extracts one tiling to confirm finite features + expected dims, and **writes no shard or
  curated artifact** (diagnostic to stdout/temp only, `analysis_role="smoke"`). So the smoke
  can never reintroduce the non-authoritative any-trace eligibility through the CLI, yet still
  exercises loader → preprocess → WST end to end.
- **Axis-certification guard (all three CLIs + the smoke, C5-08).** Before any file is
  preprocessed/extracted, the entrypoint **verifies an `ACCEPTED` axis record keyed to that
  file's raw `sha256` + `axis_spec_hash`** (§2.4; from `qc_survival_77ghz.csv`), or **runs the
  raw semantic axis check inline** and requires `ACCEPTED`. A missing/mismatched record (wrong
  `sha256` or `axis_spec_hash`) or any non-`ACCEPTED` verdict aborts before writing features.

**Acceptance.** Each runs end-to-end (locally on a subset / on the real cohort); CSVs written
and re-read; a non-canonical `preprocess77`/`qc77`/`wst77` spec (or `backend: torch`) fails
loudly before any I/O naming the deviating fields; a file lacking an `ACCEPTED` axis record
(and failing the inline check) aborts before any feature write; the preprocessing pass is a
single job with one CSV (no shard race); under step-6 outcome (c) `run_qc77.py` emits the
`qc_characterization_77ghz.csv` with `eligibility_authoritative = false` and **no**
`qc_pass`/`session_eligible` columns, and the feature CLIs refuse to consume it as eligibility
(C5-22); distributions recorded in HISTORY.md; committed only on explicit owner request.

### 2.8 `src/dehyd/provenance.py`

**Responsibility.** One minimal, explicit extension. `_hash_inputs` currently reads
`config.paths.data_10ghz_dir` literally. Add an optional `data_dir` parameter:
`record_run(config, manifest, folds=None, extra=None, data_dir=None)` defaulting to the 10 GHz
dir (all existing call sites unchanged); 77 GHz entrypoints pass
`require_77ghz_dir(config)`. **Array tasks pass their single-session manifest slice** so only
that one file is hashed — hashing 22 GB in each of 80 tasks would be pure waste (§5 risk 9).
`TRACKED_PACKAGES` already includes h5py/torch. **Acceptance:** existing `test_provenance.py`
green unchanged; a new case asserts a 77 GHz run hashes the 77 GHz dir and a sliced manifest
hashes only its file.

### 2.9 IBEX scaffolding — `configs/ibex.yaml`, `scripts/ibex/`

**Responsibility.** Run the heavy cohort passes on IBEX as Slurm jobs, same code, config-only
differences (CLAUDE.md §Compute). **`configs/ibex.yaml` is a paths overlay ONLY** (later-wins,
exactly the `load_config` multi-YAML use case):

```yaml
paths:
  data_10ghz_dir: /ibex/user/<user>/dehyd/data/10ghz     # owner fills the literal path
  data_77ghz_dir: /ibex/user/<user>/dehyd/data/77ghz
  weight_xlsx:   /ibex/user/<user>/dehyd/data/weight/metadata_subjects_info.xlsx
  results_dir:   /ibex/user/<user>/dehyd/results
```

That is the **entire** diff — **device stays `cpu`** (frozen numpy-backend policy; GPU/torch
WST for reported artifacts would need an explicit owner policy revision and is not assumed).
Worker/thread counts are sbatch/env concerns (`OMP_NUM_THREADS`), not config. Subsetting is a
CLI axis (`--subject/--session`) used identically by the local smoke and the array — same
code path, satisfying the config-only rule.

**`scripts/ibex/`:**

- `qc77.sbatch` — single job (cohort QC + axis check), ~8 GB mem, ~3–5 h.
- `wst77.sbatch` — **array over the 80 (subject, session) cells** (`--array=0-79`, task id →
  sorted (subject,session) list); ~16 GB, 4 CPUs, `02:00:00`/task (calibrated by the step-10
  smoke). Each task: load its one file → per-file QC → if ineligible, write a skip-marker
  shard and exit 0 → else extract → shard CSV + per-task provenance.
- `preprocess77.sbatch` — the chain-diagnostics cohort pass as **one single job** (not an
  array): chain steps 1–5 are light, it writes exactly one CSV, so no shard/skip/merge
  protocol and no array-race on the output (C5-11). ~8 GB, a few hours.
- `README.md` — every command written literally (the owner runs them; Claude has no ssh):
  install uv; `git clone` + `uv sync --frozen` (uv.lock pins scipy `<1.17`; torch CPU wheel
  rides along); `uv run python -c "from kymatio.numpy import Scattering1D"` smoke;
  `rsync -av --progress data/77ghz/ <user>@ilogin.ibex.kaust.edu.sa:/ibex/user/<user>/dehyd/data/77ghz/`
  (22 GB) + 10 GHz + weight dirs; `sbatch scripts/ibex/qc77.sbatch`; **after the gate**
  `sbatch scripts/ibex/wst77.sbatch`; `rsync` `results/` back; `run_wst77.py --merge-shards`
  locally. `DEHYD_REALDATA=1` (conftest already supports it) enables an optional on-IBEX
  realdata pytest. `HDF5_USE_FILE_LOCKING=FALSE` in the sbatch env (GPFS locking, risk 10).

Array unit = 80 cells (not 16 subjects): ~20–30 min/task schedules well under QoS, retry
granularity is one session, mapping is trivial. sbatch scripts invoke
`uv run python experiments/run_wst77.py --config configs/exp_77ghz.yaml --config configs/ibex.yaml --subject ... --session ...`.

**`.gitattributes`** (repo root): `scripts/ibex/* text eol=lf` — CRLF breaks sbatch shebangs
(`/bin/bash^M: bad interpreter`); developed on Windows, run on Linux (risk 7).

---

## §3 Tests

All constants read from the config object (a test that re-hardcodes 128/0.30/(8,4) passes
vacuously when config and code drift) — with the M4 exception of exactly one literal-pinning
test per frozen-default group. **Two fixture sizes (C5-23):** because `loader_77ghz` enforces
the exact on-disk contract `(16, 256, 256, n_frames)`, its **accept path is proven by at
least one full-shape fixture** — `(16, 256, 256, 1–2)`, gzip-chunked, near-zero/compressible
content so it stays a few MB on disk. **Small-dim fixtures** (`N_FAST=32, N_CHIRP=32, N_RX=2`,
the `test_audit_77ghz.py` pattern) cover the pure loader helpers (`parse_77ghz_filename`,
`reverse_axes`) and every **rejection** path, plus the **shape-generic** `screens_77` /
`pipeline_77` / `extraction_77` (written to read their dims from the array — like the audit
helpers and `wst.py` — so they run at any size; only the loader hard-asserts the real
dimensions). Seeded RNG (`numpy.random.default_rng(<seed>)`); a noiseless tone is not a valid
fixture. Groups and rough counts (≈85–90 new tests; M4 added 66):

| Group | File | Count | What it proves |
|-------|------|:---:|----------------|
| T-C77 | test_config.py | ~12 | each new section validates/rejects; `wst77.tilings` override rejected; `data_77ghz_dir` optional + existence-when-present; `require_77ghz_dir` raises; qc77 band cross-check; canonical guard; 10 GHz configs still load byte-identically |
| T-L77 | test_loader77.py | ~12 | filename parse; **one full-shape `(16,256,256,1–2)` fixture accepted** (proves the exact-shape success path; C5-23); wrong shape/compound/missing rejected; **float32 and unexpected-endianness fixtures rejected** (only 8-byte real float passes; C5-18); `reverse_axes` round-trip (small dims); loaded values == fixture |
| T-M77 | test_manifest77.py | ~8 | bijection failures; frame counts from file; imported `_join_qc` fail-closed; `ceil` eligibility; report identities |
| T-Q77 | test_qc77.py | ~13 | flatline fires (quantised / degenerate spread); any-trace rule; per-Rx counts; in-band ratio in vs out of gate; NaN/Inf short-circuit; mask bins 26..54; per-frame independence; axis verdicts on tone cubes; **fail-closed: INCONCLUSIVE and REJECTED both abort** (C5-07) |
| T-P77 | test_preprocess77.py | ~14 | MTI kills static / preserves Doppler; zero-phase; axis correctness on moving-target cubes (+ swapped-cube negative); gate-crop bins; energy accounting; determinism |
| T-W77 | test_wst77.py | ~20 | realized T/J pinned; (P,t) finite; border warning asserted; batched≡single; **`[432,2,256]` scatter input + `[16,27,2,P,t]` reshape pinned** (C5-09); fold order pinned; bin-averaging vs hand-computed; fusion mean≠median; log placement pinned; `(tiling,fusion)` pre-log scale hand-checked; **batch-standardize bit-equivalent to stacked `to_channels`** (C5-19); **tuned-ε `epsilon_by_order` applied before pooling, identical across roles** (C5-15); zero-energy assertion; dims match layouts; variants==single-variant; **numpy-vs-torch cross-backend agreement** over both log states, raw + pooled (C5-13) |
| T-A77 | test_wst77.py / test_preprocess77.py | ~6 | axis-certification guard: entrypoint aborts when no `ACCEPTED` record matches the file's `sha256`+`axis_spec_hash` and the inline check is non-`ACCEPTED`; passes with a matching `ACCEPTED` record; **`axis_spec_hash` accepts a path-only overlay, rejects a changed axis-relevant field** (C5-16); smoke follows the same guard (C5-08); **`--merge-shards` rejects a duplicate or fingerprint-mismatched shard** (C5-17) |
| T-R77 | test_wst77.py (realdata) | 1–2 | one real file (`subject_1_8am.mat`) via the **non-curated smoke** — load→(NaN/Inf + axis guards)→preprocess→one-tiling extract on **fixed frame indices** (no flatline eligibility, outcome-independent; C5-24) → finite, expected dims; writes no curated artifact; marked `realdata` (hard-fail if data absent) |

**tests/test_no_leakage.py: zero changes** (M5 is per-frame, unfitted; no CV loop touched).
DoD asserts `git diff --exit-code f3fbade -- tests/test_no_leakage.py` is clean — the
**working-tree-aware** form (diff f3fbade against the working tree, no `HEAD`), so staged
**and** unstaged edits are both caught; `… f3fbade HEAD …` would miss uncommitted changes,
which matters here because commits happen only on owner request (C5-12).

---

## §4 Definition of done

| ID | Criterion |
|----|-----------|
| D0 | **Prerequisite — ✅ SATISFIED 2026-07-23 (owner-approved).** A-M5-1 (scope) + A-M5-2 (renumber) written into `implementation_plan.md` **and** `ROADMAP.md` before implementation (C5-01/C5-02) |
| D1 | `uv run pytest` green on a checkout with no private data (all new synthetic tests incl.) |
| D2 | `uv run pytest --realdata` green (T-R77 runs the **non-curated fixed-frame smoke** on a real file — outcome-independent, no flatline eligibility; missing data is a hard fail, not a skip) |
| D3 | Cohort artifacts written and re-read/reconciled — **conditional on the step-6 outcome (C5-20/C5-22):** under **(a)/(b)** the full set `results/qc/qc_survival_77ghz.csv` (authoritative eligibility; + frames CSV + flatline report) **and** `results/preprocess/preprocess_diagnostics_77ghz.csv` **and** `results/wst/wst_diagnostics_77ghz.csv`; under **(c)** only the threshold-independent `results/qc/qc_characterization_77ghz.csv` (raw diagnostics + axis cert, `eligibility_authoritative = false`, **no** `qc_pass`/`session_eligible`) plus the front-end **code deliverables** — no cohort preprocessing/WST feature artifacts (extraction defers to M7) |
| D4 | `tests/test_no_leakage.py` byte-for-byte unmodified since M1 and green (`git diff --exit-code f3fbade -- tests/test_no_leakage.py` clean — working-tree-aware, catches uncommitted edits; C5-12) |
| D5 | Local smoke `run_wst77.py --smoke --subject 1 --session 8am` (non-curated fixed-frame mode, all outcomes) produces finite features with recorded timing; the same entrypoint runs the curated cohort/array mode on IBEX differing only by `--config configs/ibex.yaml` + shard args (curated mode fails closed under outcome (c)) |
| D6 | The flatline-rule decision is made at step 6 on M2-audit **mechanism** grounds (independent of cohort survival), recorded (HISTORY + A-M5-6), and applied uniformly. Under **(b)**: the `QC77Config`/`canonical_spec_guard_77`/`screens_77`/YAML/provenance were revised and T-C77/T-Q77 rerun green **before** the step-7 cohort QC run (no stale rule executed). Under **(c)**: M5 stopped before the cohort feature artifacts and extraction deferred to M7 |
| D7 | Remaining A-M5-3..8 applied to `plans/implementation_plan.md`; the two documents consistent; SECOND_CHAPTER.md §4 written |

Run commands (D3/D5): `uv run python experiments/run_qc77.py --config configs/exp_77ghz.yaml`;
`… run_wst77.py --config configs/exp_77ghz.yaml [--subject N --session S]`.

---

## §5 What could go wrong (known traps, pre-paid)

1. **Chunk layout `(16,4,1,125)` spans all frames** — any frame-subset HDF5 read
   decompresses the whole file. Always load whole-file once and slice in memory; a per-frame
   read loop costs ~125× the I/O.
2. **Flatline outcome can swing eligibility from ~90% to ~10%** (7/10 audited frames failed on
   benign ADC quantisation). The rule is fixed at step 6 on **mechanism grounds from the M2
   single-file audit**, never tuned to cohort survival (that would be cohort-level leakage,
   C5-03); the cohort eligibility is a recorded consequence, and nothing downstream hardcodes
   either outcome — eligibility flows from config + QC columns only.
3. **Memory under parallelism**: 1.05 GB raw + 221 MB gated cube per task → IBEX tasks request
   ≥ ~8–16 GB; local parallel experimentation multiplies the 1 GB/worker. Per-frame scatter
   chunking (§2.6) keeps the WST stage bounded — resist "scatter the whole session at once"
   (~2 GB/tiling).
4. **kymatio pad at n_in=256 is MEASURED, never assumed** (M4's off-by-one at 470→1024):
   expect 512 but let `scattering_shape` say so; the border warning fires again and must be
   asserted present, not silenced by touching a frozen tiling.
5. **The 27-bin loop is the hotspot in disguise** — folded into the batch dim it is free; as a
   Python loop it is a 54× slowdown (the M4 `pool_stats` history).
6. **YAML 1.1 float trap**: bare `2e9` loads as a string; every 77 GHz exponent needs a signed
   exponent (`2.0e+9`, `512.0e-6`) — already bitten at 10 GHz.
7. **CRLF kills sbatch** (`/bin/bash^M: bad interpreter`). `.gitattributes`: `scripts/ibex/*
   text eol=lf`.
8. **Cohort QC wall time is histogram-bound**: ~10⁴ frames × 4096 traces ≈ 41 M
   `np.histogram` calls + ~80 GB gzip decompress ≈ hours. Acceptable once; give it an IBEX
   path; don't re-run casually. Per-session extraction tasks recompute QC for their own file
   only (minutes).
9. **Provenance hashing cost**: sha256 over 22 GB per run is minutes; array tasks hash only
   their own file (the `record_run(data_dir=...)` + sliced-manifest design), or 80 tasks each
   burn the full-cohort hash.
10. **Windows/IBEX path semantics**: config paths resolve against repo root (both OSes);
    `ibex.yaml` uses absolute POSIX paths; `rel_path` stays POSIX (`as_posix()`). GPFS HDF5
    locking can spuriously fail read-opens → `HDF5_USE_FILE_LOCKING=FALSE` in the sbatch env.
11. **h5py/scipy pins**: nothing new (h5py since M2), but `uv sync` on IBEX must not resolve
    scipy ≥ 1.17 — `uv sync --frozen` (the lockfile prevents it).
12. **The two 256 axes are shape-indistinguishable** — a fast↔chirp interchange passes the
    shape assertion. The semantic axis check (§2.4) is the guard; the QC run **fails closed on
    any non-`ACCEPTED` verdict** (REJECTED *and* INCONCLUSIVE), and every extraction/
    preprocessing entrypoint + the smoke re-verifies an `ACCEPTED` record (by `sha256`+config)
    or runs the check inline (C5-07/C5-08), so a bad or uncertified mapping cannot enter
    extraction.

---

## §6 Flagged gaps in implementation_plan.md + proposed amendments

Convention (the A-M4-* precedent): amendments that **correct an ambiguity/gap in the main
plan itself** are applied as inline `*(A-M5-N, date)*` markers at each edited spot in
`implementation_plan.md`, with the before/after recorded here. **A-M5-1 and A-M5-2 rescope and
renumber the authoritative schedules, so — unlike the execution-detail amendments — they are
NOT applied at close-out: they are an owner-approved prerequisite (step 0) written into
`implementation_plan.md` AND `ROADMAP.md` before any implementation** (C5-01/C5-02; a task
plan cannot renumber/rescope the authorities by assertion and repair them afterward). `ROADMAP`
edits require explicit owner approval per CLAUDE.md. The rest (A-M5-3..8) apply at close.

- **A-M5-1 — 77 GHz promoted to a full parallel primary arm.** The main plan frames 77 GHz as
  fusion-only (`implementation_plan.md` §Context "77 GHz is used only for a cross-band fusion
  section"; ROADMAP §§0/2 "solely" for fusion). Amend §Context / §"Experiments" **and**
  ROADMAP §§0/2/4: Exps A–F each gain a 77 GHz arm with band-appropriate parameters;
  **10 GHz remains the sole headline**; Exp G keeps the fusion contrast on the matched
  population. **APPLIED 2026-07-23 (owner-approved, step-0 prerequisite satisfied):**
  `implementation_plan.md` §Context and `ROADMAP.md` §0.
- **A-M5-2 — Build order renumbered.** New §5 = 77 GHz front-end (this milestone); config-
  freeze gate → §6; harness/Exp A → §7; Exp B → §8; Exp C/D → §9; fusion(G)/interp(E)/
  confound(F)/stats(H) → §10; figures → §11 — in `implementation_plan.md` §Build order **and**
  ROADMAP §7 (whose own list, lacking the config-freeze gate, becomes …5 = 77 GHz front-end,
  6 = Exp A, … 10 = figures). All "milestone 5" freeze cross-references → "milestone 6", and
  the harness/torch-mutation "milestone 6" refs → "milestone 7", and the 77 GHz "fusion
  milestone"/"milestone 9" build refs → "milestone 5". **APPLIED 2026-07-23 (owner-approved,
  step-0 prerequisite satisfied):** both documents renumbered consistently.
- **A-M5-3 — Standardization spec gap closed. APPLIED 2026-07-23 (owner-approved).** Exp G
  step 6 was silent on standardization; amended so each slow-time series' real and imag
  channels are robust-standardized separately (own median/MAD, `to_channels('iq')`) before
  WST — consistent with the 10 GHz chain and `wst_extract77.m`'s `standardize_robust`.
  Written into `implementation_plan.md` Exp G step (6).
- **A-M5-4 — Log-placement clarification. APPLIED 2026-07-23 (owner-approved).** The Exp G
  chain listed no log step; amended so the order-aware log applies to the **fused** per-frame
  scattering tensor (after range-bin averaging and Rx fusion, before pooling); the log axis
  keeps its three 10 GHz branches (off / on+frozen-ε / on+tuned-ε, A-M4-7), with the tuned-ε
  branch train-only in the M7 harness and gated by the order-2-usefulness pre-check. Written
  into `implementation_plan.md` Exp G step (8).
- **A-M5-5 — IBEX scope extended.** The main plan scopes IBEX to GPU/DL baselines only. Amend
  §"Compute / IBEX": IBEX also runs the 77 GHz cohort QC and WST as **CPU** batch jobs
  (`scripts/ibex/`, `configs/ibex.yaml`); numpy stays the canonical reported-feature backend.
- **A-M5-6 — Flatline rule finalized on mechanism grounds (leakage-safe). Owner pre-selected
  outcome (b) on 2026-07-23 (for now).** Replaces the "parked since M2" status (lines
  ~44/271/553). **The decision is made from the M2 single-file audit + ADC-quantisation
  physics, independent of full-cohort survival** — a cohort-derived threshold would violate
  §"Frozen 77 GHz pipeline" ("never selected from audited subject data") and the config-freeze
  principle (C5-03). Admissible outcomes: (a) keep the frozen rule; **(b) a mechanism-corrected
  exact replacement** fixed a priori; (c) declare it data-adaptive → move it inside inner CV
  per §"QC screens & thresholds" (fit per training fold at M7). **Owner choice: (b)** — the
  exact corrected rule is still to be specified at step 6 from the quantisation mechanism (not
  from cohort survival), and only the **77 GHz** screen changes (the 10 GHz screen stays frozen
  since M2). So M5 proceeds to the cohort feature runs (step 11); (a)/(c) remain documented
  fallbacks if the step-6 mechanism analysis overturns the preference. Applied uniformly; the
  cohort survival is a reported consequence, never the basis.
- **A-M5-7 — Secondary 77 GHz variants deferred.** The Doppler-FFT-spectrum WST branch and the
  fast-time WST branch (with the nonzero-statistic fix) are explicitly deferred past M5; only
  the primary slow-time I/Q chain is built now.
- **A-M5-8 — Repo-tree / notes update.** `loader_77ghz.py` reannotated from "fusion
  milestone" to M5; new modules listed (`screens_77`, `axis_check_77`, `manifest_77`,
  `pipeline_77`, `extraction_77`, the three `*77` CLIs, `scripts/ibex/`); the semantic axis
  check's production home (once per file in the cohort QC run) recorded.

---

## §7 Open items this milestone resolves or carries

**Resolves:** the 77 GHz loader/QC/preprocessing/WST front-end code exists; the flatline
owner decision (A-M5-6); the standardization and log-placement spec gaps (A-M5-3/4); the
first IBEX path (A-M5-5). **Conditional on the step-6 outcome (C5-20):** under **(a)/(b)** the
front-end has met the **full cohort** and produced the real cohort QC + feature artifacts the
**M6 freeze** rests on; under **(c)** it delivers the code + a **characterization** QC run
only, and QC-dependent cohort feature extraction defers to the M7 harness (the M6 freeze then
rests on the frozen search space + the characterization QC, not on a cohort feature table).

**Carries to M6 (the freeze):** the full A–F 77 GHz design (targets shared with 10 GHz; band-
appropriate search spaces); confirmation of the on+tuned-ε log branch via the
order-2-usefulness pre-check; the frozen 77 GHz protocol-constant whitelist that modelling
entrypoints will validate. **Fixed here, NOT open at M6:** Rx fusion = mean is the primary
(frozen) modeling path and median a labeled secondary variant; pooled is the classical family
and flat is diagnostic/DL-only — promoting median or flat to a search axis would need a prior
authoritative amendment (C5-04/C5-05). **Carries to M7+:** the band-agnostic harness, Exp
A/B/C on both bands, the on+tuned-ε log branch (fold-local ε), and Exp G fusion
(10-only / 77-only / fused; the α combiner) — none built here.

COMMENTS OF CODEX

NO MORE COMMENTS

The plan is now executable, leakage-aware, and consistent with its stated prerequisite
amendments to the authoritative documents. All raised issues were applied and verified; no
items were disputed or escalated. The owner should address the explicit step-0 authority
updates and the step-6 flatline-rule decision before implementation begins.

END OF COMMENTS

DEBATE COMMENTS

_(No disputed items. All of Codex's round-1 comments were conceded and applied to the body,
so nothing sits here. Any comment I dispute on a later pass will appear here with my response
and a STATUS line.)_

END OF DEBATE
