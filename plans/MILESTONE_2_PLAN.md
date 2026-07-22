# MILESTONE 2 PLAN — 10 GHz QC screens + minimal 77 GHz audit

_Task-level execution plan for milestone 2 **only** (ROADMAP §7.2; implementation_plan.md
"Build order" §2)._

_**Status: APPROVED AND IMPLEMENTED (2026-07-21, branch `v1_milestone_2`).** All seven
build steps executed; definition of done §4 D1–D7 met in full — `uv run pytest` →
260 passed / 10 skipped, `--realdata` → 269 passed / 1 skipped (T18), and
`tests/test_no_leakage.py` is byte-for-byte unmodified since M1 and green. This
document is now a **record of what was built and why**, not a proposal; see HISTORY.md
for the per-step log and the deviations discovered during the build (the flatline
degenerate-span case, the YAML exponent bug, and the 77 GHz real-storage finding)._

_Review rounds 2026-07-21: **round 1** — all 17 comments (9 required corrections + 8
test/specification refinements) applied, none disputed; cross-document amendments
§6-A1..A5 applied to `plans/implementation_plan.md`. **Round 2** — all 10 comments
(6 required corrections + 4 consistency cleanups) applied, none disputed; the audit
gained a synthetic test module, Parseval-normalized energy accounting, fully frozen
spectral conventions, a fail-closed manifest join, cross-field band validation, and
the ≈205× multiplicity correction. **Round 3** — all 5 comments (3 required
corrections + 2 hardening items) applied, none disputed: an `H1-storage` verdict with
a frozen accepted complex representation, a predeclared non-finite-slab rule, an
executable bounded-read (spy-dataset) test, cumulative two-axis Parseval coverage, a
corrected degeneracy example, and the 77 GHz QC periodic-Hann window promoted from
audit-only convention to main-plan amendment §6-A6. **Round 4** — all 3 comments
applied, none disputed: the finite-frame floor now scales with `--n-frames`
(`ceil(0.5 × requested)`), the 77 GHz QC smoke short-circuits non-finite frames and
the findings JSON is strict (`null` for unavailable floats, `allow_nan=False`), and
the Parseval/crop and bounded-read tests were reformulated to claim exactly what
holds._

This document adds the execution detail the main plan intentionally omits. It does not
restate design decisions; where a decision is needed it cites
`plans/implementation_plan.md` (the source of truth — for M2 chiefly §"QC screens &
thresholds", §"Confirmed data facts" and Build order §2), `CLAUDE.md`, or `ROADMAP.md`.
Anything here that goes beyond those documents is flagged in §6.

Milestone 1 is done and committed (`f3fbade`): config system, 10 GHz loader,
ground-truth parser + cross-checks, frame manifest, nested-LOSO splitter, provenance,
159 tests green. M2 builds on those components and duplicates none of them. In
particular, the "deterministic xlsx parse + cross-check" listed under Build order §2
was already delivered in M1 — M2 inherits it complete.

**Owner decisions already made (2026-07-21, recorded here so they are not re-litigated):**

1. **In-band energy band margin (10 GHz) = ±1000 Hz (frozen).** The MATLAB reference
   (`wst_integrity_check_dataset.m`, invoked with all defaults in `mainProgram.m`)
   computes the in-band ratio over the gate band widened by `BandMarginHz = 1000` per
   side (the docstring's "500" is stale; the code default 1000 is what ran). At
   df = 520834/534 ≈ 975.3 Hz this is ≈ ±1 FFT bin — it tolerates Hann spectral-leakage
   skirts at the band edges, and the 0.30 threshold was defined together with it. The
   margin becomes a frozen constant `qc.in_band_margin_hz: 1000.0` (§2.3), and the
   main plan's QC table states it (§6-A1).
2. **M2 work happens on a new branch `v1_milestone_2`** (branched from
   `v1_milestone_1`, which is pushed).
3. **In-band energy band margin (77 GHz) = one FFT bin = fs/256 = 1953.125 Hz
   (frozen a priori).** No reference value exists for 77 GHz
   (`wst_integrity_check_dataset77.m` was never delivered); the margin is derived by
   generalizing the *rationale* behind the 10 GHz value — one bin of Hann leakage
   tolerance at the axis's own bin width — not by copying the raw 1000 Hz number
   (≈ half a bin at 77 GHz). Frozen before the audit so no QC constant is selected
   from audited subject data (§2.6 step 4; §6-A4).
4. **77 GHz flatline Rx aggregation = any-trace-fails-frame (frozen a priori).**
   Flatline is evaluated per (Rx, chirp) 256-sample trace; the frame fails if any of
   the 4096 traces flags — the exact structural analog of the 10 GHz any-chirp rule.
   The 4096-vs-20 (≈205×) multiplicity is a recorded property; the audit reports
   per-Rx incidence so a dead channel surfaces explicitly. Revisable before the M5
   freeze only as an explicit owner decision, never from audited data (§2.6 step 4;
   §6-A5).

---

## §0 Scope and ground rules

**In scope:**

- `src/dehyd/qc/__init__.py` + `src/dehyd/qc/screens.py` — the four frozen 10 GHz QC
  screens (§2.1).
- QC / session-eligibility columns and helpers in `src/dehyd/data/manifest.py`,
  with a fail-closed join (§2.2).
- `qc.in_band_margin_hz` + field-level and cross-field validation of every QC-consumed
  config field in `src/dehyd/config.py` + `configs/preprocess.yaml` (§2.3).
- `experiments/run_qc.py` — full-cohort QC pass; writes the curated survival report
  (§2.4).
- `experiments/run_regression.py` — evaluability hookup: folds built from evaluable
  subjects after QC (§2.5).
- `h5py` as a locked dependency; `experiments/audit_77ghz.py` — the minimal one-file
  77 GHz audit, written as an importable module with pure helpers (§2.6, §2.7).
- `tests/test_qc.py` + **`tests/test_audit_77ghz.py`** (synthetic, private-data-free);
  extensions to `tests/test_manifest.py` and `tests/test_config.py`; a `realdata`
  QC-survival test (§3).
- Journal upkeep: HISTORY.md as steps and attempts resolve; SECOND_CHAPTER.md §1
  "Data & ground truth" at milestone close.

**Explicitly out of scope (deferred to their milestones):**

- `src/dehyd/data/loader_77ghz.py` and any 77 GHz extraction — **M9** (the main plan's
  repo tree assigns it to the fusion milestone; the M2 audit is deliberately
  self-contained, §2.6 — its importable helpers are audit-local, not the production
  loader).
- Preprocessing/filters (M3), WST (M4), torch (M4), `harness.py`/`metrics.py`/any
  modeling (M6+).
- **`configs/ibex.yaml` and `scripts/ibex/` stay deferred.** M2 is entirely CPU-local:
  the full-cohort QC pass costs ≈ 15–30 s (≈1.4 GB of zlib MAT decompression dominates;
  the math is seconds) and the 77 GHz audit reads bounded slabs of one file. There is
  no GPU job to script, and a committed IBEX config would have to name paths that do
  not exist yet and would fail input-path validation (same reasoning as M1). They
  arrive at the first IBEX milestone (DL baselines).

**The milestone-2 invariant, protected above all (CLAUDE.md §Hard invariants;
implementation_plan.md §QC screens):**

> **QC is a fixed, frozen, per-frame measurable function applied once before any CV.**

Concretely: every screen consumes exactly one frame plus frozen config constants.
No QC quantity may cross frames, sessions, subjects, or CV roles. The screens are
data-*normalized* only within their own frame (histogram bin edges from that chirp's
own magnitude range; RMS median/MAD across that frame's own 20 chirps; in-band ratio
self-normalized against that frame's own total power) — none of that is population
statistics, so QC cannot leak and does not enter CV. Where a screen *could* silently
become data-dependent, and is forbidden to:

- computing the robust-RMS z across the session's or dataset's frames instead of the
  frame's own 20 chirps (§2.1 pins the population; the main plan's table states it,
  §6-A2);
- deriving histogram bin count/edges from population percentiles (bins are frozen at
  200; edges are the chirp's own min/max);
- revising any threshold because of the survival rate observed on the real cohort.
  Per the main plan: "if we ever decide a threshold should be data-adaptive, it moves
  inside inner CV; until then it is frozen." M2 runs pre-modeling, so a re-freeze with
  documented *non-performance* justification is still possible before M5 — but it must
  be logged as such in HISTORY.md, and after the M5 freeze, never;
- **selecting any 77 GHz QC constant from the audited file's data.** The audited
  subject is a future outer-test subject, so every 77 GHz QC rule the audit exercises
  (margin, Rx aggregation, spectral conventions) is **frozen a priori in this plan
  from physics/reference structure** (§2.6 steps 3–5, §6-A4/A5); the audit *validates*
  the frozen choices and characterizes the data — it never chooses between candidates,
  and a degenerate result is a stop-and-report, never a silent revision.

`tests/test_no_leakage.py` is untouched and must stay green throughout (T-QC7 and the
per-frame API shape make the data-independence executable, §3).

**Ground rules:** work on the new branch `v1_milestone_2` (owner decision 2026-07-21;
`v1_milestone_1` is pushed); commits only when the owner asks — build steps *write and
verify* artifacts, they never commit them; HISTORY.md written continuously as attempts
resolve; superseded material to `archive/` with a note.

---

## §1 Build sequence — exact order and why

Tests land in the same step as their module (keeps everything green as it grows).
HISTORY.md gets **at least** one entry per resolved step — a minimum, not a limit:
every failed or superseded attempt inside a step receives its own entry (CLAUDE.md
§Project journal files), and failures stay in the log.

| # | Step | Why this position |
|---|------|-------------------|
| 1 | Config: `QCConfig.in_band_margin_hz` + **field-level and cross-field validation of every QC-activated field** (bounds + types + band-vs-Nyquist checks, §2.3) + gate tuple normalization + `configs/preprocess.yaml` mirror + `tests/test_config.py` extension | The screens read it; schema and validation first, exactly as M1 did |
| 2 | `src/dehyd/qc/screens.py` + `tests/test_qc.py` (synthetic) | Pure functions, no I/O — testable in isolation before any bookkeeping exists |
| 3 | `src/dehyd/data/manifest.py`: `apply_qc` (fail-closed join) / eligibility / report / helpers + `tests/test_manifest.py` additions | Consumes step 2; the bookkeeping layer between screens and everything downstream |
| 4 | `experiments/run_qc.py` + `realdata` survival test; **run on the real 80 files**; record actual survival in HISTORY.md; **write and verify** the survival CSV (committed only when the owner asks) | First contact of the frozen screens with the real cohort; produces the reportable removal counts the main plan requires |
| 5 | `experiments/run_regression.py`: apply QC → `evaluable_subjects` drive `nested_loso_splits` | The data spine now ends at the *post-QC* population; M1's "every subject is evaluable" placeholder comment is retired |
| 6 | `uv add h5py`; `experiments/audit_77ghz.py` (importable module, pure helpers) + **`tests/test_audit_77ghz.py`** (synthetic HDF5 fixtures); **run on one real file**; write the findings JSON; HISTORY.md findings entry | Independent of steps 1–5; last so a QC-side surprise never blocks the audit or vice versa; the synthetic tests validate the audit logic before the single real run |
| 7 | Journal close-out: SECOND_CHAPTER.md §1 "Data & ground truth"; final HISTORY.md entry | CLAUDE.md write-cadence rules; closing the milestone requires the distilled account |

---

## §2 Per-file specifications

Format per file: **Responsibility** (single) · **Public API** · **Frozen parameters** ·
**Acceptance criteria**.

### 2.1 `src/dehyd/qc/screens.py`

**Responsibility.** The four frozen per-frame QC screens on the **raw** complex cube —
pure functions of (one frame, frozen config), no I/O, no pandas, no state.

**Public API.**

```python
@dataclass(frozen=True)
class FrameQC:
    nan_inf: bool            # any non-finite sample
    flatline: bool           # any chirp's histogram over-concentrated
    low_in_band: bool        # in-band/total power ratio below threshold
    rms_flag: bool           # DIAGNOSTIC ONLY — never a reject criterion
    in_band_ratio: float     # diagnostic, in [0, 1] — or NaN when unavailable
    n_flatline_chirps: int   # diagnostics for the report
    n_rms_outlier_chirps: int
    max_rms_z: float         # NaN when unavailable
    passed: bool             # = not (nan_inf or flatline or low_in_band), always

def run_qc_frame(frame, qc: QCConfig, pre: PreprocessConfig) -> FrameQC
    # frame: complex128 [534 fast-time x 20 chirps] — shape validated exactly
    # against the loader constants (N_FAST_TIME, N_CHIRPS); anything else raises.

def run_qc_cube(cube, qc: QCConfig, pre: PreprocessConfig) -> list[FrameQC]
    # cube: complex128 [534 x 20 x n_frames] (validated); a plain loop over
    # run_qc_frame — documented as an invariant: nothing batch-computed across
    # frames (T-QC7)

def in_band_mask(n_fast, fs_hz, bandwidth_hz, chirp_time_s,
                 gate_m, margin_hz) -> np.ndarray
    # bool over the non-negative half-spectrum bins (see screen 3). RAISES if the
    # resulting mask selects zero bins (band has no FFT-bin support, e.g. wholly
    # above Nyquist) or selects EVERY denominator bin (the screen would be
    # vacuously satisfied, ratio ≡ 1 — a silently disabled screen is a config
    # error, not a valid configuration).
```

**Non-finite-frame semantics (defined before any later screen runs).** If the frame
contains any non-finite sample, `nan_inf = True` and the remaining screens are
**skipped**, not computed on garbage (NumPy histogram range inference raises on
non-finite input; FFT/RMS diagnostics would be non-finite): the skipped screens return
`flatline = False`, `low_in_band = False`, `rms_flag = False`, count diagnostics `0`,
and float diagnostics `NaN` (`in_band_ratio`, `max_rms_z`). `passed` is `False` via
`nan_inf` alone. The `FrameQC` docstring states this contract; T-QC2 exercises an
isolated NaN, an isolated Inf, and a wholly non-finite frame.

**Frozen semantics — exactly the reference (`wst_integrity_check_dataset.m`), stated
in the module docstring with the two deliberate departures called out:**

1. **NaN/Inf:** reject if any sample of the frame is non-finite (and skip the rest,
   per the contract above).
2. **Flatline/saturation:** per chirp, a **200-bin** histogram of `abs(chirp)` over
   **that chirp's own [min, max] magnitude range**; the chirp flags if any bin count
   `>= flatline_max_bin_fraction × 534` (0.25 × 534 = 133.5 → integer counts ≥ 134,
   matching MATLAB's `max(counts) >= FlatlineFrac*Nsamples`). The frame is rejected if
   **any** chirp flags. (The reference's `nbins = max(10, min(200, round(N/2)))`
   evaluates to exactly 200 at N=534; we freeze 200 directly.) Degenerate case: a
   constant chirp gives a single populated bin (numpy expands the zero-width range) →
   fires, as intended; test-covered.
3. **In-band energy ratio (computed BEFORE any filtering, on the raw frame):** per
   chirp, Hann (**periodic**, `scipy.signal.windows.hann(534, sym=False)`, matching
   `hann(N,'periodic')`) window × 534-pt FFT → power on the **non-negative
   half-spectrum: bins 0..266 (DC included; the Nyquist bin 267 excluded** — MATLAB's
   `half = 1:floor(N/2)`); average the 20 per-chirp power spectra; ratio =
   P(band) / max(P(total half-spectrum), eps) — the guarded denominator means an
   all-zero frame yields ratio 0 and **fires**, never a division error. Band = QC gate
   0.9–3.0 m mapped via `HzPerM = 2·(B/Tchirp)/c ≈ 3257.5 Hz/m` → **2931.7–9772.4 Hz**,
   widened by the frozen margin to `lo = max(0, fmin − 1000) = 1931.7 Hz`,
   `hi = min(fs/2, fmax + 1000) = 10772.4 Hz` (mask bins 2..11 at df ≈ 975.3 Hz).
   Reject if ratio `< 0.30`. The 0.30 threshold has no meaning under any other
   denominator or band definition.
4. **Robust-RMS outlier (diagnostic only):** per-chirp RMS over the 534 magnitudes;
   median and MAD **across the frame's own 20 chirps** (never across frames);
   `z = |rms − median| / (1.4826 · (MAD + eps_float64))` (eps placement mirrors the
   reference); a chirp flags if `z > 4.5`. Sets `rms_flag` / diagnostics — **never
   contributes to `passed`**.

**Rejection rule** (implementation_plan.md QC table): `NaN/Inf ∨ flatline ∨ low
in-band`. The reason flags are **independent, non-exclusive booleans** — a frame can
legitimately fail several screens at once (e.g. a constant all-zero frame is both
flatline and low-in-band), which is why session bookkeeping reconciles through
`n_fail_any`, not a sum of reasons (§2.2). Note the reference's own `frame_pass` was
NaN/Inf ∧ flatline only — low in-band was *logged, not rejected* — and the reference
ran its check on **filtered** cubes. Rejecting on low in-band and screening the **raw**
cube are deliberate strengthenings already settled in the main plan; both get
HISTORY.md departure entries (§6-A3).

**The one fixed QC gate — rationale made local and executable.** The mask is built
from `qc.qc_gate_m = (0.9, 3.0)` and **never** from `preprocess.model_gate_m`. Reason
(main plan §QC): the QC-passing frame/session population must be **identical for every
model-gate candidate** later searched in inner CV (1–2 m default vs 0.9–3.0 m
alternative) — if QC used the model gate, changing a model hyperparameter would change
which frames exist, and the "no test-set tuning" chronology would silently break. The
wider band is used so a frame is not rejected for energy that a wider candidate model
gate would legitimately use. A comment at the mask construction says exactly this, and
T-QC9 (§3) enforces it: a ~2.5 m tone (inside the QC gate, outside the model gate)
must pass QC.

**Acceptance.** All §3 `test_qc.py` tests green; every threshold read from the config
object (no re-hardcoded numbers); `run_qc_frame` deterministic and identical whether a
frame is screened alone or inside any cube; `passed == not (nan_inf or flatline or
low_in_band)` holds for every returned result (T-QC14).

### 2.2 `src/dehyd/data/manifest.py` — QC columns, session eligibility, reports

**Responsibility.** Bookkeeping only: join per-frame `FrameQC` results to the manifest,
compute session eligibility, expose the analysis population and the removal report.
The screens stay in `qc/screens.py`; the manifest never computes a screen itself.

**Public API (additions).**

```python
QC_COLUMN_DTYPES = {
    "qc_nan_inf": "bool", "qc_flatline": "bool", "qc_low_in_band": "bool",
    "qc_pass": "bool", "qc_rms_flag": "bool",
    "qc_in_band_ratio": "float64", "qc_max_rms_z": "float64",   # NaN when unavailable
    "qc_n_flatline_chirps": "int64", "qc_n_rms_outlier_chirps": "int64",
    "session_n_pass": "int64", "session_min_pass": "int64",
    "session_eligible": "bool",
}

def apply_qc(manifest, paths, config) -> pd.DataFrame
    # Loads each file once (resolve_path + load_10ghz_file), runs run_qc_cube,
    # joins results by (rel_path, frame_idx) — NEVER by row index (M1 trap:
    # rel_path string order != session order). FAIL-CLOSED join, see below.

def session_qc_report(manifest_qc) -> pd.DataFrame
    # One row per (subject, session): n_frames, n_pass, n_fail_any, per-reason
    # fail counts (nan_inf / flatline / low_in_band — NON-ADDITIVE, see below),
    # n_rms_flagged, min_pass, eligible.
    # THE reportable per-subject/session removal table the main plan requires.

def eligible_frames(manifest_qc) -> pd.DataFrame
    # The analysis population: qc_pass frames of session_eligible sessions.

def evaluable_subjects(manifest_qc) -> tuple[int, ...]
    # Subjects with >= 1 eligible session (Exp A rule; drives N_eval).
```

**Fail-closed join.** Joining by the right key is necessary but not sufficient; the
join must be unable to silently duplicate, drop, or misattach a frame:

- `(rel_path, frame_idx)` is asserted **unique in both** the input manifest and the
  QC result table before merging;
- the merge is a pandas **one-to-one validated merge** (`validate="one_to_one"`)
  with an indicator check: **no left-only or right-only keys** — any unmatched key
  raises `ManifestError` listing the offenders;
- the output row count **equals** the input row count exactly;
- the final frame is restored to the deterministic `SORT_KEYS` order
  (`subject, session_idx, frame_idx`) before return.

Tests inject duplicate, missing, and extra QC keys and assert each fails loudly
(§3 manifest tests).

**Count reconciliation — the only identity that holds.** Rejection reasons are
non-exclusive booleans (§2.1), so per-reason counts **do not sum** to the number of
rejected frames. The report carries `n_fail_any` = frames failing at least one
rejecting screen, and the invariant asserted everywhere (tests, realdata, survival
CSV) is exactly

```
n_pass + n_fail_any == n_frames
```

with the per-reason columns documented as non-additive incidence counts (a frame
failing two screens appears in both columns but once in `n_fail_any`). If a mutually
exclusive breakdown is ever wanted for reporting, it is derived at report time by a
documented priority order — it never replaces the raw incidence counts.

**Frozen parameters.** `session_min_pass = ceil(qc.min_frame_fraction ×
n_frames_in_file)` — `min_frame_fraction = 0.5`, and the count is the file's **actual**
`n_frames_in_file` already carried by the manifest, never a hard-coded 100.
`session_eligible = session_n_pass >= session_min_pass`.

**Absent, never imputed — where it is enforced.** Ineligible sessions (and failing
frames) are **absent from `eligible_frames`** — that is the only view modeling may
consume — but **present in the full manifest and in `session_qc_report`**, so
missingness stays visible and reportable (main plan §"Evaluability after QC": QC
failure may itself correlate with hydration or acquisition quality and must be seen,
not silently absorbed). Nothing anywhere fills a dropped session from other subjects
or sessions.

**Acceptance.** §3 manifest tests green (including the fail-closed injections); on the
real cohort every (subject, session) appears in the report and
`n_pass + n_fail_any == n_frames` reconciles in every row.

### 2.3 Config: `src/dehyd/config.py` + `configs/preprocess.yaml`

**Responsibility.** The new frozen margin constant, plus **field-level and
cross-field validation for every config field M2 activates** — the M1 generic
frozen-section loader constructs the dataclass without type/range checks, and a
YAML-supplied gate arrives as a mutable list; both are fixed now that the fields are
actually consumed.

- `QCConfig.in_band_margin_hz: float = 1000.0`.
- `configs/preprocess.yaml` gains `in_band_margin_hz: 1000.0` under `qc:` with a
  comment stating provenance (reference `BandMarginHz` code default; ≈ ±1 FFT bin at
  df ≈ 975.3 Hz; tolerates Hann leakage at band edges).
- **Normalization:** `qc.qc_gate_m` and `preprocess.model_gate_m` supplied via YAML
  are converted to tuples before dataclass construction (frozen dataclasses must not
  carry mutable lists).
- **Field-level validation (build-time, `ConfigError` on violation):**
  - `histogram_bins`: positive integer (not bool);
  - `flatline_max_bin_fraction` ∈ (0, 1]; `min_in_band_energy_ratio` ∈ [0, 1];
    `min_frame_fraction` ∈ (0, 1];
  - `rms_robust_z_threshold` > 0, finite;
  - `in_band_margin_hz` ≥ 0, finite;
  - each gate (`qc_gate_m`, `model_gate_m`): exactly two finite, positive,
    strictly increasing values;
  - `preprocess.fs_hz`, `bandwidth_hz`, `chirp_time_s` (consumed by the band
    mapping): positive, finite.
- **Cross-field validation (positive-and-increasing is not enough — a syntactically
  valid gate can still disable the screen).** Using `fs_hz`, `bandwidth_hz`, and
  `chirp_time_s`, the config build asserts the QC band is *physically representable*:
  `fmin(gate) < fs/2` (the band is not wholly above Nyquist) and the margin-widened,
  Nyquist-clamped band is non-empty (`lo < hi`). The **bin-level** guards — at least
  one FFT bin of support, and not covering every denominator bin — need the fast-time
  length and therefore live in `in_band_mask`, which raises on an empty or
  all-covering mask (§2.1); together the two layers make a gate/margin combination
  that would produce a vacuous or unsatisfiable screen a hard error, never a silent
  no-op. Boundary tests cover wholly-above-Nyquist gates, zero-bin-support bands, and
  whole-spectrum masks (§3).
- Provenance output needs no change — `config_to_dict` serializes dataclasses
  generically, so the new field is recorded automatically.

**Acceptance.** `tests/test_config.py`: defaults present; YAML overrides honored;
every bound above rejected when violated; **wrong types rejected as well as
out-of-range numbers**; the cross-field Nyquist checks fire; YAML-list gates arrive
as tuples; unknown-key rejection still intact.

### 2.4 `experiments/run_qc.py`

**Responsibility.** The one-command full-cohort QC pass and the curated survival
artifact. Thin CLI, same `--config` pattern (repeatable, later files win) as
`run_regression.py`.

**Behavior.** config → `load_ground_truth` → `build_manifest` → `apply_qc` →
`session_qc_report` → print a summary (frames pass / `n_fail_any` / per-reason
incidence; sessions eligible/dropped; evaluable subjects; N_eval) → write the report
to **`config.paths.results_dir / "qc" / "qc_survival_10ghz.csv"`** (the config is the
single output-path authority — never a literal repo-relative `results/`) →
`record_run(config, manifest_qc, folds=None, extra={"stage": "milestone-2-qc",
<headline counts>})` (the `folds=None` path already exists in `provenance.py`).

**Acceptance.** Runs end-to-end on the real cohort in ~½–1 min; the CSV is **written
and verified** (re-read, reconciliation identity checked) — it is a curated,
regenerable artifact intended for the repo, but **committing it happens only on
explicit owner request** (CLAUDE.md ground rule; step 4 of §1 ends at "written and
verified"). The actual survival numbers are recorded in HISTORY.md. No caching at M2 —
QC recomputes each invocation (≈ 30 s is cheap enough; a cache with hash invalidation
is complexity we defer until it is actual friction).

### 2.5 `experiments/run_regression.py` (evaluability hookup)

**Responsibility (change only).** The data spine now ends at the post-QC population:
after `build_manifest`, call `apply_qc`; folds come from
`nested_loso_splits(evaluable_subjects(manifest_qc), ...)` instead of all manifest
subjects; print one QC summary line (frames surviving, sessions eligible, N_eval).
The M1 comment "Every subject is evaluable at M1; QC-driven evaluability arrives at
M2" is retired. Frame-level population for later milestones is `eligible_frames`.

**Acceptance.** Smoke run on the real data completes; fold count equals N_eval (the
number of subjects with ≥ 1 eligible session — expected 16 on this cohort unless QC
says otherwise, which is precisely what we are about to find out); provenance carries
the QC config and the same folds; `test_no_leakage.py` untouched and green.

### 2.6 `experiments/audit_77ghz.py` — the minimal 77 GHz audit

**Responsibility.** One self-contained, one-shot script that confirms (or refutes) the
77 GHz facts the milestone-5 freeze depends on, on **one real file**, and records
everything. It deliberately does **not** create `loader_77ghz.py` (M9 builds the
loader against this audit's confirmed facts), does not extract features, and touches
exactly one of the 80 files. **Every 77 GHz QC rule and spectral convention it
exercises is frozen a priori in this plan** (steps 3–5 below; §6-A4/A5): the audited
subject is a future outer-test subject, so the audit validates frozen choices and
characterizes the data — it never selects a threshold, rule, or convention from what
it sees (§0 invariant).

**Structure — importable module, pure helpers, thin `main()`.** The
correctness-critical logic (compound→complex conversion, axis reversal, semantic
metrics and the three-way verdict, Parseval-normalized energy accounting, the QC-smoke
screens, JSON assembly) lives in **pure, parameterized helper functions** — gate bins,
DC half-width, thresholds, and shapes are arguments, with the frozen 77 GHz constants
applied only in `main()`. `tests/test_audit_77ghz.py` imports the script as a module
(a `sys.path` insertion of `experiments/`, the same pattern `tests/` already uses for
`reference_procedure`) and drives the helpers on **small synthetic HDF5 fixtures** —
so the audit logic is validated by the mandatory suite *before* the single real-file
run, without creating the deferred production loader.

**CLI.** `--config` (repeatable, as elsewhere — used for `paths.results_dir`; the
77 GHz physical constants stay audit-local until a 77 GHz config section exists at
M9), `--file data/77ghz/subject_1_8am.mat` (default: first subject, baseline session),
`--n-frames` (**default 10 — a CLI default, not a constant**: validated
`1 <= n_frames <= dataset_frame_count` before any read, and every downstream count
derives from the requested value — step 2), `--out` — **a bare filename only**
(default `audit_77ghz.json`),
always resolved beneath `config.paths.results_dir / "qc"`; a value containing path
separators is rejected, so the resolved config remains the **single output-path
authority** (no second authority via an arbitrary path, mirroring `provenance.py`).

**The hypotheses under test (from implementation_plan.md §Confirmed data facts, 77 GHz
bullet — the axis decision is fixed there; the audit is where it is confirmed on a
real file). Three separate verdicts, never merged:**

> **H1-shape:** h5py presents `framesRadar` as shape `(16, 256, 256, 125)`.
> **H1-storage:** the dataset uses one of two accepted representations, fixed in
> advance: **(a) a plain real float dtype** (float32/float64), or **(b) an HDF5
> compound with exactly two float fields `real` and `imag`** of equal width (the
> MAT v7.3 convention for a complex array, as the 10 GHz files use). The width is a
> recorded observation, not an assumption. Anything else — integer fields, other
> field names, native HDF5 complex — is REJECTED, never silently coerced
> (native-complex support would be a deliberate implementation, and it is
> deliberately not implemented here).
>
> _**Revised 2026-07-21 during implementation.** This originally admitted (b) only.
> The real files turned out to be **plain real `float64`** — a real-sampled capture,
> consistent with the reference never calling `real()`/`imag()` on 77 GHz raw data.
> Admitting (a) is a correction of a wrong a-priori assumption about the file format,
> not a threshold chosen from the data: it is a structural fact visible in the HDF5
> metadata alone, it does not change which frames pass any screen, and the observed
> dtype is recorded in full either way. Consequence recorded in
> implementation_plan.md: the Exp G "I/Q" arises at the **range FFT**, not in the raw
> cube._
> **H1-axes:** under the full axis reversal → `(Nframes, Nfast, Nchirps, Nrx)`, the
> two 256 axes are correctly assigned (fast-time ↔ range structure, chirps ↔
> near-zero Doppler).

**Verdict domains (fixed JSON schema):** `H1-shape ∈ {ACCEPTED, REJECTED}`;
`H1-storage ∈ {ACCEPTED, REJECTED}`;
`H1-axes ∈ {ACCEPTED, INCONCLUSIVE, REJECTED, NOT_RUN}`;
`qc_smoke ∈ {NON_DEGENERATE, DEGENERATE, NOT_RUN}`;
`chain ∈ {NON_DEGENERATE, DEGENERATE, NOT_RUN}`.
`NOT_RUN` appears **iff** a gating precondition failed: a REJECTED `H1-shape` or
`H1-storage` gates everything downstream; the finite-frame floor (step 2) gates
`H1-axes` and `chain` only. On an ungated path each verdict must take one of its
substantive values.

**Frozen spectral conventions (fixed here so no fixed threshold rests on an
unstated convention; all of these are recorded in the JSON alongside the scalars):**

- **Hann windows:** the QC smoke (step 4) and the semantic metrics (step 3) use
  **periodic** Hann (`sym=False`) — the convention of the QC reference
  (`hann(N,'periodic')` in `wst_integrity_check_dataset.m`) and of the 10 GHz screens.
  For the QC screen this window is **production-QC material, not an audit-local
  convention** — it is frozen in the main plan (§6-A6).
  The reference-matching chain products (step 5) use **symmetric** Hann
  (`sym=True`) — MATLAB's `hann(N)` default, which is what
  `chirpavg_and_fuse_batch.m` actually calls on both axes. The two conventions are
  deliberate and must not be mixed.
- **Semantic metrics `G(X)`, `D(X)`:** computed on the **full 256-bin spectrum**
  (the slow-time and fast-time signals are complex — no Hermitian symmetry, and
  Doppler needs its negative frequencies; a shared full-spectrum denominator keeps
  `G` and `D` comparable across the two axes). `G(X)` sums the **unshifted** FFT
  bins 27..53; `D(X)` sums the **fftshifted** spectrum over the **seven bins
  centered on zero** (offsets −3..+3); each divided by the full-spectrum total.
- **77 GHz QC in-band ratio:** follows the 10 GHz convention at N=256 —
  **non-negative half-spectrum bins 0..127 (DC included, Nyquist bin 128
  excluded)**; guarded denominator. With df = fs/256 = 1953.125 Hz, gate
  2–4 m → 52.12–104.24 kHz, widened by the frozen one-bin margin →
  **mask bins 26..54** (29 bins) — derived, recorded, asserted by test.
- **Energy accounting (step 5) is Parseval-normalized.** NumPy's default FFT is
  unnormalized (a length-N FFT scales total spectral energy by N), so raw
  spectral sums are not comparable across stages. All audit energy ratios use
  **normalized total energy**: spectral-domain energy divided by the transform
  length (equivalently `norm="ortho"`), leaving the operational/reference FFT
  representation itself unchanged. The 1e-9 floor applies **only** to these
  normalized ratios; a synthetic Parseval test asserts time-domain and
  normalized spectral-domain energies agree.

**Steps — each writes its observations into the findings JSON:**

1. **Structure (H1-shape + H1-storage).** Open with h5py; locate `framesRadar` (the
   raw variable name, verified against `matlab/77ghz_code/mainProgram.m`:
   `load(file).framesRadar`). Record the full observed dtype descriptor (**field
   names, field dtypes, byte order, item size** — so any surprise is actionable),
   dataset shape as presented, **chunk layout** and compression. Evaluate both
   structure verdicts independently, from metadata alone: **H1-shape** — the shape is
   `(16, 256, 256, 125)`; **H1-storage** — the dtype matches the accepted
   representation defined above (compound `real`/`imag` numeric fields of equal float
   width; the width itself is a recorded observation). If **either is REJECTED**:
   write the provenance-complete JSON with the observed descriptor, set `H1-axes`,
   `qc_smoke`, and `chain` to `NOT_RUN`, and exit nonzero — a rejected structure
   verdict is a **stop-and-report**; the owner decides before anything depends on it.
2. **Slab read — bounded, always.** Read the **requested `n_frames`** via bounded
   slabs on the last h5py axis (at the default request of 10: `dset[..., :10]` →
   `(16, 256, 256, 10)`, ~170 MB as complex128 plus the compound source slab
   transiently — the estimate scales with the request). **The whole dataset is never
   read at once**: the compound source plus the converted complex128 copy would
   already exceed ~4 GB before FFT temporaries. If the chunk layout recorded in
   step 1 makes last-axis slicing pathologically slow, iterate **smaller
   chunk-aligned slabs** (e.g. one frame at a time) and concatenate — bounded memory
   is the invariant, not a particular slab size. The slab reader is a helper that
   accepts any dataset-like object and whose every request carries a **bounded
   frame-axis slice** — never an ellipsis/whole-dataset read, never a request
   spanning the entire frame axis; the non-frame axes (Rx, chirp, fast-time) **are**
   read in full within each bounded slab, as they must be — a contract the
   spy-dataset test makes executable (§3). Convert compound → complex128; apply the
   full axis reversal → `(n_frames, 256, 256, 16) = (frame, fast, chirp, rx)`.
   Record: final dtype, |value| min/median/max, NaN/Inf count, slab indices actually
   read.
   **Non-finite-slab rule (frozen).** A slab frame containing any non-finite sample
   would poison every averaged FFT metric downstream, so the **semantic check
   (step 3) and chain products (step 5) use only the finite frames**: a frame is
   excluded iff it contains any non-finite sample (the same frame-level granularity
   as the QC NaN/Inf screen). The predeclared floor **scales with the request**:
   `min_finite_frames = ceil(0.5 × requested_frame_count)` (5 at the default 10);
   below it, `H1-axes` and `chain` are set to `NOT_RUN` and the audit is a
   stop-and-report. The JSON records `requested_frame_count`, `min_finite_frames`,
   and the **selected, excluded, and effective frame indices**. The **QC smoke
   (step 4) runs on all slab frames** — reporting non-finite frames is precisely its
   NaN/Inf screen's job, so it is not gated by the floor.
3. **Raw-data axis semantic check (H1-axes) — BEFORE any clutter subtraction.** This
   is the correctness-critical step, treated with the same care as the leakage
   guarantees. *Why raw:* the reference MTI (`filter_gpt_butterworth77.m`) subtracts
   the per-fast-time-bin **mean over chirps**; a seated subject is quasi-static, so
   nearly all subject energy sits at near-zero Doppler and MTI removes it — running
   the check after MTI would erase exactly the structure the check looks for on
   *both* axes and render it uninformative. Enforced structurally: the semantic-check
   helper takes the raw slab; MTI exists only inside step 5's helper; the script
   computes step 3 before step 5 in straight-line code.
   *Physical constants* (from `filter_gpt_butterworth77.m` / `chirpavg_and_fuse_batch.m`,
   NOT the 10 GHz values): fs = 500 kHz, B = 2 GHz, Tchirp = 512 µs,
   PRF = 1/Tchirp ≈ 1953.125 Hz, range-bin spacing dr = c/(2B) ≈ 0.075 m → the
   reference 2–4 m gate = **bins 27..53** (0-indexed; MATLAB's `find` picks the same
   physical bins 28..54 1-indexed), 27 bins of 256.
   *Symmetric metrics — computed identically for BOTH candidate assignments,* per the
   frozen conventions above (periodic Hann on the transformed axis; power averaged
   over the **effective (finite) frames** × 16 Rx × all 256 lines of the orthogonal
   axis): for each of
   the two 256-sample axes `X`, `G(X)` (gate-bin fraction) and `D(X)` (near-DC
   fraction). Under the proposed assignment (fast = axis 1, chirp = axis 2 after
   reversal), the expected pattern is `G(fast)` material, `D(chirp)` dominant (static
   scene), and `G(chirp)` ≈ 0 (bins 27..53 of a Doppler spectrum ≈ 206–404 Hz —
   nearly empty for a static scene). **`D(fast)` is recorded but is not a standalone
   discriminator under the proposed mapping** (close-in TX leakage can legitimately
   concentrate range power near bin 0); it participates only as part of the
   **mirrored swapped-axis hypothesis** below.
   *Predeclared criteria and three-way verdict:*
   - **ACCEPTED** iff **A1** `D(chirp) ≥ 0.5` ∧ **A2** `G(fast) ≥ 0.05` ∧
     **A3** `G(fast) ≥ 10 × G(chirp)` — i.e. the proposed assignment shows its
     expected structure *and* is materially stronger than the swap (A3 is the
     symmetric comparison).
   - **REJECTED** iff the mirrored criteria hold for the swapped assignment:
     `D(fast) ≥ 0.5` ∧ `G(chirp) ≥ 0.05` ∧ `G(chirp) ≥ 10 × G(fast)` — positive
     evidence the axes are interchanged.
   - **INCONCLUSIVE** otherwise — neither pattern is supported (low SNR, an
     unrepresentative file, or a wrong range assumption are all possible); this is a
     **stop-and-report**, not a failure verdict: the owner decides the follow-up
     (e.g. a second file or session, as a documented decision), and nothing
     downstream proceeds on an unconfirmed axis order.
   All four metric values are recorded regardless of verdict. The thresholds are
   **audit-only diagnostics** with ~an order of magnitude of headroom, chosen to
   detect the qualitative asymmetry, not pipeline QC constants: used once, never
   entering CV; the *observed values* (not just the verdict) feed the M5 freeze.
4. **Proposed 77 GHz QC smoke — frozen rules, validated not chosen.** The rule
   structure and numbers come from implementation_plan.md §Exp G (note
   `wst_integrity_check_dataset77.m` is absent from `matlab/`, so the main plan is the
   only source), completed a priori by this plan where Exp G was silent (§6-A4/A5):
   - NaN/Inf: any non-finite sample **anywhere in the frame's (fast × chirp × Rx)
     cube** rejects the frame and — mirroring the frozen 10 GHz per-frame contract
     (§2.1) — **short-circuits that frame's remaining screens**: flatline and
     in-band are skipped (histogram range inference and FFT diagnostics never run on
     non-finite data), skipped Boolean/count diagnostics return False/zero, and
     float diagnostics take the unavailable sentinel (JSON `null`, step 6).
   - Flatline: evaluated per **(Rx, chirp)** 256-sample fast-time trace — 128-bin
     histogram over the trace's own magnitude range, trace flags if any bin ≥ 25% of
     256 (≥ 64); the **frame fails if any trace flags** (the structural analog of the
     10 GHz any-chirp rule). The **≈205× multiplicity vs 10 GHz** (16 Rx × 256 chirps
     = 4096 traces vs 20 — the receiver factor alone is 16×, the chirp-count increase
     supplies the rest) is a recorded property of the rule, not a tunable — the audit
     reports flatline incidence **per Rx** and per frame so the M5 freeze can *see*
     whether the multiplicity bites (e.g. a systematically dead Rx is an owner
     decision, never a silent threshold change).
   - In-band: per-(Rx, chirp) periodic-Hann-windowed 256-pt fast-time spectra,
     averaged across **all chirps and all Rx** → one ratio per frame (the analog of
     the 10 GHz 20-chirp average), on the **half-spectrum bins 0..127 with mask bins
     26..54** per the frozen conventions above; reject if < 0.30.
   Run on the slab; report pass counts, per-Rx and frame-aggregate distributions of
   the in-band ratio and flatline incidence. **Executable non-degeneracy verdict
   (predeclared):** the QC smoke is `NON_DEGENERATE` iff `n_pass ≥ 1` (not
   all-reject) **and** the median finite-frame in-band ratio over the slab
   `≥ 0.01` (a functioning radar with a subject in the gate puts far more than 1% of
   half-spectrum energy in a 29-of-128-bin band; the floor has an order of magnitude
   of headroom). Otherwise `DEGENERATE` → **stop-and-report** — the frozen rules are
   never revised from the audited subject's data.
5. **Proposed-chain non-degeneracy — the operations later code will actually use**
   (main plan Exp G "exact primary chain" steps 1–5 and the reference RD map;
   symmetric Hann per the frozen conventions — matching `chirpavg_and_fuse_batch.m`),
   two explicit products:
   - **(a) Primary pre-WST gated range cube:** MTI (subtract per-fast-bin mean over
     the 256 chirps) → **fast-time Butterworth bandpass** (`butter(4, Wn,
     'bandpass')`-equivalent SOS, zero-phase, on the 2–4 m beat band ≈ 52.1–104.2 kHz
     at fs = 500 kHz — the reference `filter_gpt_butterworth77.m` step) →
     Hann(fast, symmetric) → 256-pt range FFT → crop to gate bins 27..53.
   - **(b) Range-Doppler diagnostic:** product (a) additionally → **Hann(slow,
     symmetric)** → 256-pt Doppler FFT → fftshift (matching
     `chirpavg_and_fuse_batch.m`, which windows both axes).
   *Non-degeneracy is numeric, not `> 0`:* record the **Parseval-normalized
   energy-retention ratio at every stage** (post-stage / pre-stage, plus
   final / raw-slab — spectral energies divided by the transform length per the
   frozen convention; for the two-axis range-Doppler product the normalization is
   **cumulative, N_fast × N_slow**, never just the last transform's length). The
   **`chain` verdict is `NON_DEGENERATE`** iff the final normalized energy of each
   product is finite and **≥ 1e-9 × the raw slab energy** — a predeclared floor that
   catches true degeneracy (an all-zero post-MTI result from chirp-constant data, a
   wrong-axis transform, or a filtering/cropping bug that removes all supported
   energy) while never firing on legitimate physical attenuation (MTI removing
   dominant static clutter can legitimately leave ≪ 1% of raw energy); otherwise
   `DEGENERATE` → stop-and-report.
6. **Write findings — provenance-complete.** The JSON records: the input file's
   logical path, byte size, and SHA-256 (`dehyd.provenance.sha256_file`); UTC
   timestamp; git commit + dirty state; Python/NumPy/SciPy/h5py versions; **all**
   physical constants, audit thresholds, **and the frozen spectral conventions**
   (window types per stage, spectrum/denominator definitions, exact gate/mask bins,
   normalization convention) — not only the scalars; the frame indices actually read;
   observed HDF5 metadata (dtype, shape, chunks, compression); the axis mapping
   applied; `requested_frame_count`, `min_finite_frames`, and the selected,
   excluded, and effective frame indices under the non-finite-slab rule;
   all recorded metrics and distributions; and the explicit **H1-shape**,
   **H1-storage** (with the observed dtype descriptor), **H1-axes**, `qc_smoke`, and
   `chain` verdicts, each drawn from its fixed domain (incl. `NOT_RUN` on gated
   paths). **Serialization is standards-compliant:** every unavailable or
   non-finite float is encoded as JSON `null` (never the non-standard `NaN` token)
   and the file is written with `allow_nan=False`, so any unhandled NaN/Inf fails
   loudly at write time instead of producing an unparseable artifact. A fixed output filename is
   fine for the curated artifact; the *content* must be enough to reproduce and
   attribute the audit. Findings go to HISTORY.md and SECOND_CHAPTER.md §1; they are
   the factual basis for the milestone-5 freeze of 77 GHz QC / tilings / input
   domain / fusion (Build order §5, Exp G).

**Acceptance.** `tests/test_audit_77ghz.py` green in the mandatory suite (§3); script
runs end-to-end on `subject_1_8am.mat` within bounded memory; JSON written with every
provenance field above; every verdict (`H1-shape`, `H1-storage`, `H1-axes`,
`qc_smoke`, `chain`) takes exactly one value from its fixed domain, with `NOT_RUN`
appearing only on its defined gated path; all four semantic metrics recorded; every
number the M5 freeze needs is in the JSON, not just verdicts.

### 2.7 Environment

`uv add h5py` (ordinary locked-dependency addition at exactly the milestone
MILESTONE_1_PLAN §2.1 anticipated; from M2 on, h5py is required by the mandatory
suite — `tests/test_audit_77ghz.py` builds synthetic HDF5 fixtures). No torch (M4).
No other changes; scipy stays pinned `<1.17` (kymatio).

---

## §3 Tests

All thresholds are read from the config object — a test that re-hardcodes 0.25/0.30/
4.5/200 would pass vacuously when config and code drift apart.

**`tests/test_qc.py` (synthetic; no real data):**

| ID | Test | What it proves |
|----|------|----------------|
| T-QC1 | A clean frame (in-gate tone + small noise) passes all screens | Screens don't fire spuriously |
| T-QC2 | Isolated NaN → `nan_inf`; isolated Inf → `nan_inf`; wholly non-finite frame → `nan_inf`; in all three cases the skipped screens return False/0/NaN exactly per the §2.1 contract and `passed` is False | Screen 1 + the non-finite contract |
| T-QC3 | One constant-magnitude chirp → `flatline` (any-chirp rule); boundary crafted so counts of 134 fire and 133 don't (directly or via a config-fraction override) | Screen 2 + its exact ≥ boundary |
| T-QC4 | A far out-of-band tone (~50 kHz) → `low_in_band`; the in-gate tone doesn't; `in_band_ratio` ∈ [0, 1] for finite frames | Screen 3 fires; value sane |
| T-QC5 | One chirp scaled ×100 → `rms_flag` set **and `passed` still true** (other screens clean) | Diagnostic-only status made executable |
| T-QC6 | Loosening/tightening each threshold via a modified `QCConfig` flips the corresponding verdict | Thresholds genuinely come from config |
| T-QC7 | `run_qc_frame(frame)` ≡ `run_qc_cube(cube)[i]` for the same frame regardless of companion frames in the cube | **Data-independence guard** — no batch statistics can sneak in |
| T-QC8 | Two identical runs → bit-identical results | Determinism |
| T-QC9 | A ~2.5 m tone (inside the 0.9–3.0 m QC gate, outside the 1–2 m model gate) **passes** QC; and the mask is built from `qc_gate_m`, not `model_gate_m` | The fixed-wider-gate design as a test |
| T-QC10 | A tone just **outside the physical gate but inside the ±1000 Hz margin** (e.g. ~10.3 kHz; gate top 9772.4, margin top 10772.4) counts as in-band and passes; a tone **beyond the margin** (e.g. ~12 kHz) fails | The margin is real, on the correct side of the boundary — an in-gate vs 50 kHz pair alone cannot show this |
| T-QC11 | `in_band_mask` bin membership equals independently computed indices: covers the non-negative half-spectrum bins 0..266 only (DC in, **Nyquist bin 267 out**), edges at `lo/hi` land in the correct bins (expected 10 GHz mask: bins 2..11) | Exact mask arithmetic |
| T-QC12 | The window equals `scipy.signal.windows.hann(534, sym=False)` (periodic, not symmetric); an all-zero frame yields ratio 0 (guarded denominator — no division error) and fires `low_in_band` **and** `flatline` (the documented overlap case) | Periodic-Hann definition + zero-power behavior + reason overlap |
| T-QC13 | `run_qc_frame` raises on any shape ≠ (534, 20); `run_qc_cube` raises on any shape ≠ (534, 20, N) | Exact input-shape validation |
| T-QC14 | Over a randomized battery of frames (clean, degenerate, mixed) generated with a **fixed, recorded RNG seed** (`numpy.random.default_rng(<seed in the test>)` — deterministic and reproducible): `passed == not (nan_inf or flatline or low_in_band)` for every result | The rejection rule is the invariant it claims to be |
| T-QC15 | `in_band_mask` **raises** on a band with zero FFT-bin support (e.g. a gate wholly above Nyquist after clamping) and on a margin so large the mask covers every denominator bin (screen would be vacuous) | The §2.3 bin-level guards: a disabled screen is an error, not a configuration |

**`tests/test_manifest.py` additions** (extending the existing synthetic-savemat
fixture pattern, crafting bad frames — e.g. NaN-filled, all-zero — per file):

- `apply_qc` adds exactly the `QC_COLUMN_DTYPES` columns with those dtypes; existing
  M1 columns untouched.
- Ceil boundary with a non-100, odd frame count: n=3 → `min_pass = 2`; 2 passing →
  eligible, 1 passing → ineligible (the real per-file count, never an assumed 100).
- **Reconciliation via `n_fail_any`:** a crafted all-zero frame fails both flatline
  and low-in-band, appears in both incidence columns, and is counted **once** in
  `n_fail_any`; `n_pass + n_fail_any == n_frames` holds in every report row; no test
  asserts a sum of per-reason counts.
- **Fail-closed join:** injected **duplicate** QC keys, **missing** QC keys, and
  **extra** QC keys each raise `ManifestError` loudly (no silent duplication,
  dropping, or misattachment); the output row count equals the input row count; the
  returned frame is in `SORT_KEYS` order.
- `eligible_frames` excludes (a) failing frames of eligible sessions and (b) **all**
  frames of ineligible sessions; `session_qc_report` still shows the dropped session
  with its reason counts (**absent-never-imputed, missingness-visible** as behavior).
- `evaluable_subjects` drops a subject only when **all five** sessions are ineligible.
- Join correctness: results attach by `(rel_path, frame_idx)`, robust to the
  `subject_1_10am < subject_1_8am` string-sort trap (M1 lesson).

**`tests/test_config.py` additions:** margin default 1000.0; YAML override; every
§2.3 bound rejected when violated; wrong types rejected (e.g. string bins, bool,
three-element gate, decreasing gate); **cross-field checks fire** (gate wholly above
Nyquist rejected; margin-widened band that clamps to empty rejected); YAML-list gates
normalized to tuples.

**`tests/test_audit_77ghz.py` (synthetic, private-data-free — mandatory suite).**
Imports `experiments/audit_77ghz.py` as a module and drives its pure helpers on small
synthetic HDF5 fixtures (written with h5py, compound real/imag dtype). Coverage:

- **Compound→complex conversion + exact axis mapping:** a fixture with distinct
  values per (rx, chirp, fast, frame) coordinate round-trips through read + reversal
  to the documented `(frame, fast, chirp, rx)` layout, element-exactly.
- **Verdict paths (helpers parameterized by gate bins / DC half-width / thresholds,
  so small shapes work):** a constructed proposed-axis signal (in-gate tone along
  fast, static along chirp) → `ACCEPTED`; the same cube with the two candidate axes
  transposed → `REJECTED`; a flat-noise cube (fixed seed) → `INCONCLUSIVE`.
- **Shape-failure path:** a wrong-shape fixture makes the audit write a
  provenance-bearing JSON with `H1-shape = REJECTED`, `H1-axes = NOT_RUN`, and exit
  nonzero — asserted on the written file, not just the return value.
- **Provenance keys:** the JSON from a fixture run contains every §2.6 step-6 field
  (path/size/hash, timestamp, git, versions, constants, conventions, frame indices,
  verdicts).
- **Parseval and crop accounting, separated (the convention the 1e-9 floor depends
  on):** (i) **full-spectrum Parseval** — normalized spectral energy equals
  time-domain energy (within float tolerance) through one FFT axis **and** through
  the cumulative fast-then-slow two-axis pipeline (normalization by
  `N_fast × N_slow`, so an implementation dividing only by the last transform length
  fails); (ii) **crop accounting** — cropped energy equals the explicit sum over the
  retained bins and is `<=` the pre-crop normalized total (cropping removes energy
  in general — no equality-to-time-domain claim is made for a cropped spectrum);
  (iii) an **in-crop-support equality** check only with a signal explicitly
  constructed so all spectral support lies inside the retained bins — there, and
  only there, cropped energy equals time-domain energy up to float tolerance.
- **Frozen-constant derivation:** the script-level constants match independent
  computation — gate bins 27..53 from dr ≈ 0.075 m, QC mask bins 26..54 from
  df = 1953.125 Hz + one-bin margin, PRF ≈ 1953.125 Hz.
- **Unsupported-storage path:** a plain-float (non-compound) fixture and a
  wrong-field-names fixture each yield `H1-storage = REJECTED`, downstream verdicts
  `NOT_RUN`, a provenance-complete JSON carrying the observed dtype descriptor, and
  a nonzero exit — asserted on the written file, not just the return value.
- **Non-finite-slab rule:** a fixture with some non-finite frames → exactly those
  frames excluded, selected/excluded/effective indices recorded, semantic + chain
  metrics computed on the finite frames only; a fixture below the
  `ceil(0.5 × requested_frame_count)` floor → `H1-axes = chain = NOT_RUN` and the
  stop-and-report path. **Run with a non-default `--n-frames`** so a hidden
  hard-coded five or ten fails the test; `requested_frame_count` and
  `min_finite_frames` are asserted in the JSON.
- **Non-finite QC-smoke frame + JSON encoding:** a non-finite frame in the slab gets
  the exact short-circuit result (NaN/Inf flag set; flatline/in-band skipped;
  False/zero Booleans and counts; unavailable floats), and the written JSON encodes
  every unavailable float as `null` — parseable by a strict JSON parser; an
  implementation leaking a raw `NaN` token fails (`allow_nan=False`).
- **Bounded-read invariant (spy dataset):** the slab reader is driven with a fake
  dataset that records every `__getitem__` and **raises on an ellipsis/whole-dataset
  request or any request spanning the entire frame axis**; every recorded request
  carries a bounded frame-axis slice no larger than the audit slab/chunk, while
  **full slices on the non-frame axes (Rx, chirp, fast-time) are required and
  allowed** — each bounded frame slab must read those axes completely; the requested
  frame indices come back exactly once, in order — the memory-safety invariant as an
  executable test, not a hope.

**`realdata` (in `tests/test_qc.py`):** full-cohort survival — `apply_qc` over the
real 80 files completes without error, then asserts **structural properties that do
not tune survival**: all 80 (subject, session) cells present; every finite-frame
`qc_in_band_ratio` ∈ [0, 1]; `n_pass + n_fail_any == n_frames` in every report row;
every `session_min_pass` equals `ceil(0.5 × n_frames_in_file)` from the file's actual
count; re-running `apply_qc` on one file reproduces identical results (determinism on
real data, bounded cost); prints the survival table. **No expected-survival-rate
assertion** — rates are unknown until this runs, and asserting them would be
threshold-tuning by the back door. (The curated CSV comes from `run_qc.py`, step 4.)

**`tests/test_no_leakage.py`:** zero changes. QC is frozen, per-frame, applied before
any CV; the reference procedure and all T1–T19 remain as they are, green.

---

## §4 Definition of done

| ID | Criterion |
|----|-----------|
| D1 | `uv run pytest` green on a checkout with no private data (all new synthetic tests included — `test_qc.py`, `test_audit_77ghz.py`, config/manifest extensions) |
| D2 | `uv run pytest --realdata` green, including the QC survival test |
| D3 | `uv run python experiments/run_qc.py --config configs/exp_a_regression.yaml` **writes and verifies** `<results_dir>/qc/qc_survival_10ghz.csv` (curated, regenerable; committed only on explicit owner request); actual per-subject/session survival recorded in HISTORY.md |
| D4 | `run_regression.py` smoke: folds over evaluable subjects post-QC; provenance carries the QC config (incl. the new margin) |
| D5 | 77 GHz audit executed on one real file within bounded memory; provenance-complete `audit_77ghz.json`; **H1-shape and H1-storage each get ACCEPTED/REJECTED; H1-axes gets exactly one of ACCEPTED / INCONCLUSIVE / REJECTED on the ungated path, with NOT_RUN appearing only on its defined gated paths (structure rejection; finite-frame floor)**; `qc_smoke` and `chain` verdicts recorded with all metrics; anything other than across-the-board ACCEPTED/NON_DEGENERATE is a stop-and-report to the owner before downstream planning depends on it |
| D6 | Journals: HISTORY.md entries as attempts resolve (failures kept; at least one per step); SECOND_CHAPTER.md §1 "Data & ground truth" written at close |
| D7 | The §6 amendments A1–A7 are live in `plans/implementation_plan.md` (applied during the 2026-07-21 review rounds); the two documents stay consistent through any further M2 changes |

---

## §5 What could go wrong (known traps, pre-paid)

- **Join by path, never index** — `rel_path` string order ≠ session order (M1 trap;
  §2.2 pins the join key, and the join is fail-closed).
- **numpy histogram on non-finite data raises** — which is why the non-finite
  contract (§2.1) short-circuits before flatline/in-band/RMS ever run.
- **numpy histogram on a constant chirp**: zero-width range is auto-expanded → single
  populated bin → flatline fires. Intended; T-QC3 covers it.
- **Reason codes overlap.** Never assert per-reason counts sum to rejections; the only
  identity is `n_pass + n_fail_any == n_frames` (§2.2).
- **MAD = 0** (all chirp RMS identical): the reference adds eps *inside* the
  denominator — mirrored, so z = 0 when the value equals the median and large when it
  doesn't, never NaN.
- **Unnormalized FFTs scale energy by N.** Raw spectral sums are not comparable
  across FFT stages; every audit energy ratio uses the Parseval-normalized convention
  (§2.6), and a synthetic test pins it.
- **MATLAB `hann(N)` defaults to symmetric; the QC reference used `'periodic'`.**
  The two conventions coexist deliberately (§2.6): periodic for QC/semantic checks,
  symmetric for the reference-matching chain — never mixed silently.
- **77 GHz compound dtype**: MAT v7.3 complex arrives as a structured (real, imag)
  array of unknown float width — the audit records what it finds and converts
  explicitly; nothing assumes complex128 on disk.
- **77 GHz memory**: the compound source slab + complex128 copy + FFT temporaries
  multiply fast — bounded slab reads are the invariant (§2.6 step 2); the whole
  dataset is never materialized.
- **Do not "fix" survival rates.** If the real-cohort survival looks surprising, that
  is a *finding* for HISTORY.md and the owner — not a license to nudge a frozen
  threshold (§0 invariant).

---

## §6 Flagged gaps in `implementation_plan.md` + amendments (A1–A7 APPLIED 2026-07-21)

Found while specifying against the MATLAB reference and during the review rounds; each
keeps the main plan the single source of truth. **A1–A7 have been applied to
`plans/implementation_plan.md`** (review rounds 2026-07-21); F1–F2 are recorded
resolutions needing no main-plan change.

- **A1 — QC table, in-band row (gap: band margin + denominator unspecified).** The
  reference (run with defaults) widened the gate band by ±1000 Hz and used a
  half-spectrum power denominator; the 0.30 threshold is only meaningful under that
  definition. Amended the row to: "in-band(gate **± 1000 Hz margin**) / total
  **non-negative half-spectrum** power < 0.30, per-chirp Hann-windowed 534-pt spectra
  averaged across the 20 chirps", and added `in_band_margin_hz = 1000` to the
  frozen-values note. (Owner decision 2026-07-21; wording "non-negative half-spectrum"
  per review — bins 0..266, DC in, Nyquist out.)
- **A2 — QC table, RMS row (ambiguity: z-score population).** The reference computes
  median/MAD across the frame's own 20 chirps; a population-level reading (across
  frames) would make QC data-dependent. Amended to: "robust-z of per-chirp RMS
  **across the frame's own 20 chirps** (never across frames)".
- **A3 — §Deliberate departures (missing entry).** Added: QC runs on the **raw** cube
  and low in-band energy is a **rejection** criterion; the reference ran its integrity
  check on already-filtered cubes and only *logged* low in-band without rejecting.
- **A4 — Exp G 77 GHz QC (gap: no margin defined).** Frozen a priori: 77 GHz in-band
  margin = **one FFT bin = fs/256 = 1953.125 Hz**, by the same leakage rationale as
  the 10 GHz value (whose 1000 Hz ≈ one bin at *its* df; the rationale generalizes,
  the raw Hz number does not). Chosen from physics/reference convention **before** the
  audit so no QC constant is selected from audited subject data. (Owner decision
  2026-07-21.) The half-spectrum convention (bins 0..127, DC in, Nyquist out — the
  10 GHz convention at N=256) is stated alongside it.
- **A5 — Exp G 77 GHz QC (gap: Rx aggregation undefined).** With 16 receivers the
  rule structure needed completing: flatline per **(Rx, chirp)** trace with frame-fail
  on any flagged trace; in-band spectra averaged across **all chirps and Rx** → one
  ratio per frame; NaN/Inf anywhere in the frame's cube rejects. Frozen a priori as
  the structural analog of the 10 GHz rules; the any-trace multiplicity
  (4096 vs 20 ≈ **205×**) is recorded, the M2 audit reports per-Rx incidence, and any
  revision is an owner decision before the M5 freeze — never a silent change in code.
  (Owner decision 2026-07-21.)
- **A6 — Exp G 77 GHz QC (gap: window convention unstated).** The in-band screen's
  window shapes the production QC population at M9, so it cannot live as an
  audit-local convention: amended Exp G to state the screen uses a **periodic Hann**
  (`hann(256,'periodic')` / `scipy.signal.windows.hann(256, sym=False)`) — the QC
  reference's convention, matching the 10 GHz screens — alongside the half-spectrum
  bins 0..127 and mask bins 26..54. The 10 GHz QC table row now also states its
  window is periodic (always the frozen §2.1 semantics; made explicit for symmetry).
  The **symmetric** Hann remains correct only for the reference-matching
  range-Doppler chain (`chirpavg_and_fuse_batch.m` uses MATLAB's symmetric default),
  which is not QC.
- **A7 — §Confirmed data facts, 77 GHz bullet (wrong assumption, corrected against the
  real file).** Added the milestone-2 audit's confirmed facts: shape `(16,256,256,125)`
  and gzip chunks `(16,4,1,125)` as predicted, but `framesRadar` is stored as **plain
  real `float64`, not a complex/compound array** (ADC-like values quantised to 1/16,
  |x| ≤ 2560) — consistent with the reference never calling `real()`/`imag()` on 77 GHz
  raw data. Consequence recorded there: the Exp G chain's "I/Q" arises at the **complex
  range-FFT output** (chain step 4), not in the raw cube; the chain is unaffected, but
  no stage before step 4 may assume complex input. Total corrected to ~21.5 GB and the
  frame count to an exact 125.
- **F1 — audit thresholds A1–A3, the verdict schema (incl. `H1-storage`'s
  accepted-representation definition and the NOT_RUN gating), the semantic-check and
  range-Doppler-diagnostic conventions, the non-finite-slab rule, and the numeric
  non-degeneracy floors are audit-only** (the main plan specifies the semantic check
  qualitatively; this plan adds the symmetric metrics, predeclared criteria, verdict
  domains, and floors as execution detail). Recorded with observed values; no
  main-plan change. The 77 GHz **production-QC window** is deliberately NOT in this
  audit-only scope — it is main-plan material (A6).
- **F2 — `run_qc.py` / `audit_77ghz.py` (+ its synthetic test module)** are new
  `experiments/` / `tests/` entries not in the main plan's repo tree; they follow its
  thin-CLI and test patterns. No change needed.
- *(A former F3 — letting the M5 freeze choose the 77 GHz margin from audit-reported
  variants — was withdrawn during review round 1 as test-data-informed selection and
  is superseded by A4.)*

---

## §7 Open items this milestone resolves or carries

| Item | Status after M2 |
|------|-----------------|
| `h5py` not installed | **Resolved** — locked dependency (step 6) |
| 77 GHz axis order unconfirmed on a real file | **Resolved by the audit** — H1-shape, H1-storage, and H1-axes each get an explicit verdict with observed values (downstream verdicts NOT_RUN iff the structure verdicts fail or the finite-frame floor is unmet); ACCEPTED feeds the M5 freeze, anything else stops and reports |
| `configs/ibex.yaml`, `scripts/ibex/` | **Still deferred** — no GPU work in M2 (§0); first IBEX milestone |
| QC caching | **Deliberately not built** — recompute ≈ 30 s; revisit only on real friction |
| Branch for M2 work | **Resolved** — new branch `v1_milestone_2` (owner decision 2026-07-21; `v1_milestone_1` pushed) |

