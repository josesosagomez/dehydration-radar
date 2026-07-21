# MILESTONE 1 PLAN — Scaffold + config + `tests/test_no_leakage.py`

_Task-level execution plan for milestone 1 **only** (ROADMAP §7.1; implementation_plan.md
"Build order" §1)._

_**Status: IMPLEMENTED AND COMMITTED (2026-07-21, `f3fbade`).** All ten build steps
executed, all definition-of-done items (§5 D1–D7) met: `uv run pytest` → 151 passed /
8 skipped; `uv run pytest --realdata` → 158 passed / 1 skipped (T18). This document is
now a **record of what was built and why**, not a proposal — see HISTORY.md for the
per-step implementation log and the deviations discovered during the build._

_Planned 2026-07-21 and revised over four rounds of independent (Codex) review: all 25
review comments (10 + 8 + 5 + 2) verified against the real data, the installed
libraries, and the main plan, and incorporated (see §7)._

This document adds the execution detail the main plan intentionally omits. It does not
restate design decisions; where a decision is needed it cites
`plans/implementation_plan.md` (the source of truth), `CLAUDE.md`, or `ROADMAP.md`.
Anything here that goes beyond those documents is flagged in §7.

**Decisions already made by the project owner (2026-07-21, recorded here so they are not
re-litigated):**
1. **Mutation-test staging** — at M1 the strong mutation test runs against a test-local
   sklearn reference procedure (the contract `harness.py` must later satisfy); the torch
   leg is written at M1 but stays skip-marked until the torch fit path exists in
   `harness.py` (M6), at which point the test rebinds to the real harness. torch itself
   enters the environment earlier, at M4, for WST cross-backend validation. (§7-A1; the
   matching amendment is applied in implementation_plan.md.)
2. **`environment.yml`** (root; conda export from planning-phase inspections) — moved to
   `archive/code/` during M1, noted in HISTORY.md. `uv` is the sole local env manager.
3. **T18 (torch mutation leg) activates at M6, not M4** — gated on the torch *fit path*
   existing in `harness.py`, not merely on `import torch` succeeding. torch is
   importable from M4 (WST cross-backend), but there is no torch training procedure to
   test until M6, and a throwaway M4 training fixture would test itself rather than the
   harness. The contract text is written at M1; only its execution waits. (§4 T18.)
4. **`min_train_subjects` constrains the outer-training *pool*, not every inner fit** —
   the main plan's literal rule. At the `n_train = 3` boundary `GroupKFold(3)` fits each
   inner model on 2 subjects; this is accepted rather than adding a second threshold,
   because reaching it would require 12 of 16 subjects to lose every session to QC, and
   the alternative there is discarding the fold entirely. (§3.)

---

## §0 Scope and ground rules

**In scope (HANDOFF.md deliverables 1–9):** pinned uv env; `src/dehyd/config.py`;
`src/dehyd/data/{sessions,loader_10ghz(minimal),ground_truth,manifest}.py`;
`src/dehyd/eval/splits.py`; `src/dehyd/provenance.py`; `configs/` scaffolding;
one `experiments/` stub; `tests/` for all of the above with `test_no_leakage.py` as the
capstone; journal upkeep.

**Explicitly out of scope (deferred to their milestones):** QC screens and reason codes
(M2), the 77 GHz audit + h5py (M2), preprocessing (M3), WST (M4), `harness.py` /
`metrics.py` / any modeling (M6+), sbatch scripts. **torch:** not in the M1 env; it
enters at **M4** (the main plan's WST validation requires numpy/torch cross-backend
checks — Build order §4, Verification), while the torch *fit path* arrives with
`harness.py` at M6.

**Ground rules:** the five hard invariants in CLAUDE.md §Hard invariants govern
acceptance; every fold consumed anywhere comes from `eval/splits.py`; work happens on
branch `v1_milestone_1`; commits only when the owner asks; HISTORY.md is written
continuously as steps resolve (CLAUDE.md §Project journal files).

---

## §1 Build sequence — exact order and why

Tests land in the same step as their module (keeps everything green as it grows).

| # | Step | Why this position |
|---|------|-------------------|
| 1 | Env & skeleton: `pyproject.toml`, `.python-version`, `uv lock && uv sync`, package skeleton, `.gitignore` additions, archive `environment.yml`, **`tests/test_env.py`** (minimal import/version smoke so pytest collects ≥1 test — pytest exits 5 on empty collection) | Everything depends on it; the kymatio/numpy pin (§2.1 risk) is the only env unknown and must be resolved before any code |
| 2 | `configs/data.yaml` + `src/dehyd/config.py` + `tests/test_config.py` | Config is the root dependency of every other module; schema fixed early |
| 3 | `src/dehyd/data/sessions.py` + minimal `src/dehyd/data/loader_10ghz.py` + `tests/test_loader.py` | Manifest needs file enumeration/inspection; verifying real-file facts early de-risks everything downstream |
| 4 | `src/dehyd/data/ground_truth.py` + `tests/test_ground_truth.py` | Manifest needs labels; the cross-checks validate the data before anything is built on it |
| 5 | `src/dehyd/data/manifest.py` + `tests/test_manifest.py` | Joins steps 3+4; the structural-integrity gate on the real 80 files |
| 6 | `src/dehyd/eval/splits.py` + `tests/test_splits.py` | Pure function of subject ids; the leakage test needs it |
| 7 | `src/dehyd/provenance.py` + `tests/test_provenance.py` | Needs config + manifest; standalone otherwise |
| 8 | `tests/test_no_leakage.py` | Capstone; consumes steps 2–7 |
| 9 | `experiments/run_regression.py` stub + `configs/{preprocess,wst,exp_a_regression}.yaml` | The M1 end-to-end smoke: config → ground truth → manifest → splits → provenance JSON on real data |
| 10 | Journal close-out: final HISTORY.md entries; SECOND_CHAPTER.md milestone-1 section | CLAUDE.md write-cadence rules; closing the milestone requires the distilled account |

---

## §2 Per-file specifications

Format per file: **Responsibility** (single) · **Public API** · **Acceptance criteria**.

### 2.1 Environment (`pyproject.toml`, `uv.lock`, `.python-version`)

**Responsibility.** One pinned, reproducible local env (implementation_plan.md §Library
choices); the same env definition later drives IBEX CPU-side work.

**Contents.**
- `pyproject.toml`: package `dehyd`, `requires-python = ">=3.11"`, src layout
  (`src/dehyd/`). Dependencies: `numpy`, `scipy`, `pandas`, `openpyxl`, `PyYAML`,
  `scikit-learn`, `kymatio`, `pytest`. **No torch, no h5py at M1** (h5py enters at M2
  for the 77 GHz audit; torch at M4 for WST cross-backend validation — both as ordinary
  locked-dependency additions at those milestones). Pytest configured here
  (`[tool.pytest.ini_options]`, `testpaths = ["tests"]`, and the `realdata` marker
  registered — registration only silences the unknown-marker warning; the actual
  gating lives in `tests/conftest.py`, see §4 Part B).
- `tests/conftest.py`: adds the `--realdata` option and the gating fixture that makes
  the marker mean something (§4 Part B).
- `.python-version` = `3.11` — matches the plan's "3.11+" and the archived conda env;
  uv fetches the interpreter.
- `.gitignore` additions: `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info/`.
- Move `environment.yml` → `archive/code/environment.yml` (decision 2), HISTORY note.
- `tests/test_env.py`: asserts the pinned imports succeed and `sys.version_info >=
  (3, 11)` — exists so step 1's pytest run collects at least one test (pytest exits
  with status 5 on an empty collection, so "collects zero tests, exits cleanly" is
  impossible).

**Known risk (resolved at this step, recorded in HISTORY.md).** kymatio's latest
release may require `numpy<2`. If `uv lock` or the kymatio import smoke fails on
numpy 2.x, pin `numpy>=1.26,<2` and record the exact conflict and chosen pins. The
resolution is an env fact, not a design change.

**Acceptance.** `uv sync` from a clean checkout succeeds; `uv run python -c "import
numpy, scipy, pandas, openpyxl, yaml, sklearn, kymatio"` succeeds; `uv run pytest`
is green (collects and passes `test_env.py`).

### 2.2 `src/dehyd/config.py` (+ `configs/data.yaml`)

**Responsibility.** Load one or more YAML files into a single validated, frozen config
object; the resolved config is what `provenance.py` records. (CLAUDE.md §Style:
config-driven, seeded.)

**Public API.**
```python
def load_config(*yaml_paths: str | Path) -> Config
```
- Deep-merges the files in order (later wins), then validates. No CLI-override
  mechanism at M1 (subset/device changes are config-file edits; an override flag can be
  added when an experiment actually needs it).
- **Composition (`include:`).** A YAML may declare a reserved top-level
  `include: [path, ...]`; the listed files are merged first, in order, then the
  declaring file's own keys are merged last (declaring file wins). Includes may **not**
  nest — an included file containing `include:` is an error (keeps composition flat and
  followable). This is how `exp_a_regression.yaml` composes
  `data.yaml`/`preprocess.yaml`/`wst.yaml` without duplicating any scientific constant,
  and why the smoke command needs only one `--config` flag.
- **Merge semantics.** Mappings merge recursively; **scalars and lists/tuples are
  replaced wholesale by the later file** (never concatenated) — a later config states
  the entire intended value.
- **Two distinct path rules (deliberately different — they answer different
  questions):**
  - **`include:` entries resolve relative to the declaring YAML file's own directory**
    (the ordinary "import a sibling file" semantics), so `configs/exp_a_regression.yaml`
    writes `include: [data.yaml, preprocess.yaml, wst.yaml]` and finds its siblings in
    `configs/`. A repo-root rule here would be wrong — it would look for
    `./data.yaml`.
  - **Path *values* inside a config** (`data_10ghz_dir`, `weight_xlsx`, `results_dir`)
    resolve against the **repository root** (the directory containing `pyproject.toml`),
    regardless of which YAML declared them or the current working directory — so a
    data root means the same thing no matter which file states it or where the run is
    launched from.
  - **Required-input paths** (`data_10ghz_dir`, `weight_xlsx`) must exist at load time;
    **output directories** (`results_dir`) are *not* required to exist — they must be
    safely creatable and are created on demand by their writers
    (`mkdir(parents=True, exist_ok=True)`).
- **Per-machine roots (local vs IBEX).** The canonical run config is never edited in
  place: a committed overlay YAML (e.g. `configs/ibex.yaml`, containing only the IBEX
  data/results roots) is passed as an additional `--config`; `--config` is repeatable
  and later files win. Same code, config-only difference (CLAUDE.md §Compute).
- `Config` is a frozen dataclass of frozen sections (plain dataclasses — no pydantic,
  per CLAUDE.md §Code style):
  - `PathsConfig`: `data_10ghz_dir`, `weight_xlsx`, `results_dir`.
  - `RunConfig`: `seed: int`, `seed_set: tuple[int, ...]` (exactly 5, per
    implementation_plan.md §nested-CV protocol), `device: Literal["cpu","cuda"]`
    (validated string only — **no torch import**).
  - `SplitConfig`: `n_inner_max: int = 5`, `min_train_subjects: int = 3`.
  - `QCConfig`, `PreprocessConfig`, `WSTConfig`: schema + frozen reference values (used
    from M2/M3/M4 but pinned now because they are frozen constants of the design):
    QC = histogram bins 200, flatline fraction 0.25, in-band ratio min 0.30, RMS
    robust-z 4.5, QC gate 0.9–3.0 m (implementation_plan.md §QC screens);
    preprocess = Butterworth order 4, model gate default 1–2 m, EdgeTrim 32
    (§Preprocessing); WST tilings Q=(10,4)@0.20 ms, (8,2)@0.30 ms, (6,2)@0.40 ms,
    `max_order 2`, log-ε 1e-6 (§WST parameterization).

**Validation (each failure raises with a message naming the offending key/file).**
Structural: unknown keys rejected (typo guard); type checks per field; device in the
allowed set; required-input paths exist; nested `include:` rejected. **Numeric
constraints (a syntactically valid config must not fail later inside `GroupKFold` or
silently weaken the protocol):**
- `seed_set` has **exactly 5 entries and they are distinct** — duplicates would give
  fewer than 5 effective stochastic repeats while looking correct
  (implementation_plan.md §nested-CV protocol: "a fixed seed set (5 seeds)").
- `n_inner_max >= 2` — `GroupKFold` requires `n_splits >= 2`.
- **`min_train_subjects >= 3`** — not the mechanical `GroupKFold` floor of 2, but the
  **protocol** floor: implementation_plan.md §Inner loop states "Inner CV **requires ≥3
  training subjects**; below that the fold cannot select." Validating at 2 would let an
  overlay YAML silently weaken the approved nested-CV rule while remaining
  syntactically valid, so the config layer enforces the protocol, not just the library
  requirement.

**Acceptance — mandatory tests must not require the private data.** The canonical
`configs/data.yaml` points at the real `data/` tree, and validation *requires* input
paths to exist, so loading the canonical config unmodified would raise on a clean
checkout — which would contradict D1. The mandatory tests therefore always append a
**final temporary overlay** whose `data_10ghz_dir` / `weight_xlsx` point into pytest's
`tmp_path` (created empty; existence is all the config layer checks). That still
exercises everything the config layer does — include composition, merge precedence,
both path rules, and every validation branch:
- Composition/merge/validation: loading `configs/exp_a_regression.yaml` **+ overlay**
  succeeds and yields the expected merged values; unknown key, wrong type, missing
  required key, **missing input path**, nested include, **duplicate seeds,
  `n_inner_max < 2`, and `min_train_subjects < 3`** each raise distinct, clear errors;
  a later overlay replaces (never concatenates) the data root.
- **Outside-CWD path test:** `monkeypatch.chdir(tmp_path)`, then load
  `configs/exp_a_regression.yaml` **by absolute path** (a relative top-level config
  path would itself be unresolvable from the new CWD — that is not what this test is
  probing) with the same overlay, asserting the includes resolve via the declaring
  file's directory and the data paths resolve to the same absolute locations as from
  the repo root.
- **`realdata`-marked integration test:** the canonical `configs/exp_a_regression.yaml`
  loads **without any overlay** — i.e. the committed run config really does point at
  the real data. Skipped by default, hard-failing under `--realdata` (§4 Part B).

### 2.3 `src/dehyd/data/sessions.py`

**Responsibility.** The single definition of session order — nothing else.
```python
SESSION_NAMES = ("8am", "10am", "12pm", "2pm", "4pm")   # index 0..4 == S0..S4
SESSION_INDEX = {name: i for i, name in enumerate(SESSION_NAMES)}
```
S0..S4 = paper's DeHydL0..L4 (implementation_plan.md §Confirmed data facts).
`ground_truth.py`, `loader_10ghz.py`, `manifest.py` all import from here. **No YAML
duplicates this order** — `data.yaml` holds paths and expected subject ids only; session
names/order come from this module alone.

**Acceptance.** Imported by all three consumers; no other module or config defines
session names.

### 2.4 `src/dehyd/data/loader_10ghz.py` — minimal M1 scope

**Responsibility.** Filename parsing and file inspection/loading with hard assertions.
QC screens, frame-level filtering, and everything else belong to M2 (HANDOFF item 9).

**Public API.**
```python
def parse_10ghz_filename(path: str | Path) -> tuple[int, int]   # (subject, session_idx)
@dataclass(frozen=True)
class FileInfo: path: Path; subject: int; session_idx: int; n_frames: int; shape: tuple
def inspect_10ghz_file(path: str | Path) -> FileInfo
def load_10ghz_file(path: str | Path) -> np.ndarray   # complex128 [534, 20, n_frames]
```
- `parse_10ghz_filename`: strict regex `subject_(\d+)_(8am|10am|12pm|2pm|4pm)\.mat`;
  raises on non-match (an unparseable name is an *unmatched file* — manifest fails).
- `inspect_10ghz_file`: header-only via `scipy.io.whosmat` — no full 17 MB load, so the
  80-file manifest build stays fast. Asserts `framesRadar` present, **MATLAB class
  `double`** (whosmat's class string — catches a wrong-typed variable without loading),
  ndim 3, dims `[534, 20, N]` with **N > 0**; records the **actual** N (never assumes
  100 — implementation_plan.md §Session eligibility). *Contingency:* if `whosmat`
  proves slow on these zlib-compressed v5 files, fall back to one full `loadmat` per
  file at manifest-build time (≈1.4 GB total I/O, acceptable) — recorded in HISTORY.md
  if taken.
- `load_10ghz_file`: `scipy.io.loadmat(path, variable_names=["framesRadar"])` — the
  large unused `framesRadarIQ` array is never decompressed. Asserts dtype
  `complex128`, shape `(534, 20, N)` consistent with `inspect`.

**Acceptance.** Parses all 80 real filenames; `inspect` matches `(534, 20, 100)` on the
real files; full `load` of ≥1 real file passes dtype/shape assertions. Synthetic-file
unit tests (savemat into tmp dirs) cover each failure mode: bad filename; missing
`framesRadar`; **wrong MATLAB class** (e.g. int16 array); **wrong first-two-axis
shape**; **zero frames**; **non-complex/wrong loaded dtype**.

### 2.5 `src/dehyd/data/ground_truth.py`

**Responsibility.** Deterministic fixed-cell parse of the weight workbook → signed Δm%,
class labels, covariates — with the two sign-aware cross-checks. Fails loudly; never
guesses. (implementation_plan.md §Confirmed data facts, ground-truth bullet.)

**Public API.**
```python
@dataclass(frozen=True)
class GroundTruth:
    sessions: pd.DataFrame  # 80 rows: subject, session_idx, session_name,
                            #          mass_kg, delta_m_kg, delta_m_pct (signed, neg = loss)
    subjects: pd.DataFrame  # 16 rows: subject, age, height_cm, baseline_mass_kg, bmi

def load_ground_truth(xlsx_path: str | Path) -> GroundTruth
```

**Parse & layout validation (all verified against the real workbook, 2026-07-21):**
- openpyxl, sheet `MetaData`, **fixed cell addresses** rows 3–18, cols B/C/D/E–I/J/K.
  **Never** header-name inference, and **never** `max_row`/`max_column` for cohort size
  — the real sheet reports 1000 rows × 113 columns from formatting artifacts.
- **Subject identity from column B, not row position:** each B-cell must match the
  strict pattern `^Subject (\d+)$`; the parsed ids must be **unique and exactly
  {1..16}**. Rows are keyed by the parsed id (the radar `subject_N` ↔ workbook
  "Subject N" identity is owner-confirmed; parsing makes it checked, not assumed).
  **Beyond-range guard:** scan **every non-empty cell in column B outside rows 3–18**
  (the sheet reports ~1000 rows, so a full column scan is trivial) and fail on any
  further `Subject <id>` record — a subject added anywhere in the formatted worksheet
  is caught, not just one immediately below the block.
- **Header validation before session assignment** (guards against column drift; exact
  observed values): row 1 — B `Name`, C `Age`, D `Height (cm)`, E `Weight (kg)`,
  J `Weight lost (kg)`, K `Observations`; row 2 — E `time(8:00)`, F `time(10:00)`,
  G the literal string `'12 Noon'`, H `time(14:00)`, I `time(16:00)` (E/F/H/I are
  `datetime.time` cells; G is text — asserted as such).
- **Column J is a formula** (`=I<row>-E<row>`): the workbook is opened **twice** —
  once with `data_only=True` to read J's cached numeric value for the cross-check, and
  once in formula view to assert the formula is literally `=I{row}-E{row}` for every
  subject row (structure check; catches a hand-edited cell).
- **Covariate validation before BMI:** age and height must be numeric and in plausible
  bands (age 15–80 yr, height 120–220 cm) *before* `bmi = m0 / (height_m)²` is
  computed.
- Targets: `delta_m_pct = (m(s) − m(S0)) / m(S0) × 100` (signed; S0 = col E);
  `delta_m_kg = m(s) − m(S0)`. The 5-class label is the session index (S0..S4) and
  lives in the manifest, not here.

**Cross-checks (both run on every load; failure = `ValueError` listing every offending
subject with computed vs. stated values):**
1. **Col J (cached signed kg):** `|computed (m(S4) − m(S0)) − J| ≤ 0.05 kg`.
2. **Col K (positive % text, e.g. "Loss of 1.74%…"):** parse the first float; compare
   `|parsed − abs(delta_m_pct(S4))| ≤ 0.05` percentage points.

**Tolerance rationale (corrected to the observed workbook, not an assumption):** most
weights are recorded to 0.1 kg but **Subject 15 uses 0.05-kg increments** (69.05,
68.35, 68.05, 67.95), and **column K is not always conventional two-decimal rounding**
(Subject 13: computed 0.5997 % appears as "0.59" — truncation). The observed maximum
K-deviation across all 16 subjects is ≈0.0097 pct-points; J's cached values match the
computed differences to float precision. Both tolerances are therefore **conservative
bounds justified by observation** (≈5× the worst observed deviation), not statements
about recording precision.

**Loud sanity checks (guard against wrong-cell parsing bugs, not against the data):**
exactly 16 subject rows parsed; no missing/NaN weight in E–I; masses within a generous
plausible band (30–200 kg); Δm% within generous bounds (−10 % … +5 %). A genuine
cross-check failure on the real workbook **stops M1 and is investigated** — the
tolerance is never widened to make it pass.

**Test strategy — openpyxl cannot fake a formula cache (verified 2026-07-21).**
openpyxl writes `=I3-E3` but never evaluates it, so reopening an openpyxl-written
fixture with `data_only=True` returns **`None`** for J. The two views are therefore
mutually exclusive in any synthetic fixture: a J cell holding a formula has no cached
value, and a J cell holding a literal number fails formula-view validation. **No
synthetic workbook can exercise the complete public `load_ground_truth()` path** — so
the module is decomposed so that each view is independently testable:

```python
# internal helpers — each takes ONE worksheet view, so each is unit-testable alone
def _validate_layout(ws_formula) -> None      # headers, col-B identity+uniqueness,
                                              # extra-Subject scan, J formula structure
def _read_values(ws_data_only) -> dict        # masses E–I, age, height, cached J, K text
def check_targets(masses, j_kg, k_text) -> list[Discrepancy]   # pure math, no I/O
# public composition — the only place both views meet
def load_ground_truth(xlsx_path) -> GroundTruth
```
- **`check_targets` — pure-function unit tests (no workbook):** tolerance behaviour,
  sign handling, and every cross-check failure mode, called directly with arrays
  (fast, exhaustive, no I/O).
- **`_validate_layout` — formula-view fixtures:** openpyxl temp files with `=I{r}-E{r}`
  written as strings; covers header-cell validation, the column-B identity pattern,
  duplicate/out-of-range ids, an extra `Subject` record anywhere in column B, and a
  hand-edited (non-formula) J cell. Never reads cached values, so the missing cache is
  irrelevant.
- **`_read_values` — literal-value fixtures:** openpyxl temp files with J written as a
  plain number; covers mass/age/height extraction, NaN weights, malformed K text.
  Never asserts formula structure.
- **`load_ground_truth` (both views together) — `realdata` integration only:** the
  composition is exercised on the real workbook, which genuinely carries formulas *and*
  Excel-written caches. (A mock-based unit test of the composition, patching the two
  `load_workbook` calls to return independently controlled views, is optional and not
  required for M1.)

**Acceptance.** Real workbook parses and both cross-checks pass on real data
(integration); every failure mode above raises correctly in the helper-level unit tests
per the split just described.

### 2.6 `src/dehyd/data/manifest.py`

**Responsibility.** The frame index table joining files ↔ ground truth, and the
structural gate that **fails on any missing / duplicate / unmatched record**
(implementation_plan.md §Confirmed data facts, subject-identity bullet).

**Public API.**
```python
def build_manifest(paths: PathsConfig, gt: GroundTruth) -> pd.DataFrame
```
One row per frame; columns: `subject`, `session_idx`, `session_name`, `rel_path`,
`n_frames_in_file`, `frame_idx` (0-based), `delta_m_pct`, `class_label`
(= `session_idx`, the ordinal S0–S4 label). QC columns (reason codes, eligibility) are
**added at M2**, not stubbed now.

**File identity is a logical path, not a physical one (cross-machine portability).**
The canonical identity stored in the manifest is `rel_path` — the path **relative to
the configured `data_10ghz_dir`**, e.g. `subject_1_8am.mat`. It is resolved against
that root (`data_10ghz_dir / rel_path`) whenever a file is loaded or hashed. A
repository-relative physical path would **not** be canonical: on IBEX the data root
lives outside the repo (e.g. `/scratch/...`), so such a path would carry
machine-specific `..` segments and the same file would acquire a different identity on
each machine. The resolved absolute path may be kept in memory for I/O, but it is never
the stored identity.

**Deterministic ordering and dtypes (filesystem enumeration order must never reach
training order, hashes, or saved artifacts).** After all structural checks pass, the
frame table is **sorted by `(subject, session_idx, frame_idx)` and its index reset**,
so a `glob` that returns files in a different order on another machine yields a
frame-for-frame identical manifest. Column dtypes are fixed and asserted:
`subject`, `session_idx`, `n_frames_in_file`, `frame_idx`, `class_label` → `int64`;
`delta_m_pct` → `float64`; `session_name` → `string`; `rel_path` → `string` holding the
**POSIX-style logical path** described above (identical on Windows and IBEX, and what
`provenance.py` records alongside each hash).

**Hard-fail checks (each raises listing all offenders, not just the first):**
- **C1 completeness** — every expected subject×session cell (16×5) has exactly one file;
  missing cells enumerated.
- **C2 no duplicates** — no (subject, session) claimed by two files.
- **C3 no unmatched files** — every `.mat` in `data/10ghz/` parses to a valid
  (subject 1–16, session) pair; extra or unparseable files enumerated.
- **C4 bijection with ground truth** — every file matches exactly one `gt.sessions`
  row and every row has a file.
- **C5 per-file structure** — `inspect_10ghz_file` assertions (framesRadar, class
  `double`, ndim 3, 534×20×N, N>0) hold for all 80 files.
- **C6 actual frame counts** — `n_frames_in_file` recorded from the file header, never
  assumed 100.

**Acceptance.** The **mandatory** unit tests run entirely on synthetic inventories
(tmp dirs of small `savemat` files, through the same `build_manifest` code path), with
C1–C5 each exercised by a dedicated broken inventory. These fixtures construct the
`GroundTruth` object **directly in memory** (it is just two DataFrames) rather than
round-tripping a synthetic workbook — which both sidesteps the openpyxl formula-cache
limitation (§2.5) and keeps manifest tests testing the manifest. Plus:
**reproducibility test** — two builds from the same inventory (with the file listing
deliberately shuffled between them) are frame-for-frame identical, including dtypes.
The **real-data build** (80 files → 16×5×100 = 8000 rows, all counts recorded) is the
`realdata`-marked integration test R1 (§4 Part B) — skipped under the default `pytest`
run, and under `--realdata` a hard failure if the data are absent or incomplete.

### 2.7 `src/dehyd/eval/splits.py`

See §3 for the full contract (kept as its own section per the task brief).

### 2.8 `src/dehyd/provenance.py`

**Responsibility.** Per-run provenance artifact (implementation_plan.md
§Reproducibility / run provenance).

**Public API.**
```python
def record_run(config: Config, manifest: pd.DataFrame,
               folds: list[OuterFold] | None,
               extra: dict | None = None) -> Path   # path to the written provenance.json
```
**`config.paths.results_dir` is the single output authority** — there is no `out_dir`
parameter, so the destination cannot be specified two ways and disagree. Tests that
need a scratch destination construct a `Config` whose `results_dir` is pytest's
`tmp_path`.

Writes `provenance.json` containing: SHA-256 of every unique radar file in the manifest
**and of the ground-truth workbook** (labels must not be able to change without
provenance detecting it) — ~1.4 GB, a few seconds, hashed fresh each run, no cache
(readable-code rule); the fully-resolved config as a dict; the fold manifest (each
subject's role per fold); package versions via `importlib.metadata` for the pinned
deps; git commit + dirty flag (`git rev-parse HEAD` / `git status --porcelain`); device
string; seed and seed_set; `SLURM_JOB_ID` env var if present; platform; ISO timestamp.

**Radar files are recorded by logical path + hash.** Each entry is
`{rel_path, sha256}` where `rel_path` is the manifest's logical identity (§2.6) and the
hash is computed on the **resolved physical file** (`data_10ghz_dir / rel_path`). So a
run on IBEX and a run on Windows over the same data produce the same entries, while
the hash still proves the actual bytes read. The workbook is recorded the same way
(logical name + hash of the resolved file).

**Canonical serialization.** File entries sorted by `rel_path`; subject sets serialized
as sorted lists; folds ordered by test subject — so byte-identical inputs give
byte-identical JSON (modulo the timestamp) and diffs are meaningful.

**Collision policy and Windows-safe naming.** Each run writes into its own directory
`results_dir/runs/<stamp>_<git-shortrev>/provenance.json`, where `<stamp>` is a
**filesystem-safe UTC** format `YYYYMMDDTHHMMSSffffffZ` — no colons (invalid in Windows
paths; this project develops locally on Windows) and microsecond precision so two runs
started in the same second cannot collide. A normal **ISO-8601 timestamp is kept inside
`provenance.json`** for human/tooling use. If the target file already exists,
`record_run` **raises** rather than overwriting — repeated runs never silently clobber
provenance.

**Acceptance.** JSON written for a manifest; two runs on unchanged inputs produce
identical content except timestamp and run-directory name; hash of a deliberately
modified copy of a data file differs (tested on small temp files, never by mutating
real data); recorded radar entries are logical `rel_path`s with no absolute or `..`
segments. The repeated-run test sets `results_dir` to pytest's `tmp_path` —
**outside the repo** — so the first run's output cannot itself flip the recorded
git-dirty state of the second run.

### 2.9 `configs/` scaffolding + `experiments/run_regression.py` stub

- `configs/data.yaml` — real input paths + expected subject ids (1–16). **No session
  names/order** (that lives solely in `sessions.py`, §2.3).
- `configs/preprocess.yaml`, `configs/wst.yaml` — the frozen values listed in §2.2,
  consumed from M3/M4; present now so the schema is validated end to end.
- `configs/exp_a_regression.yaml` — `include: [data.yaml, preprocess.yaml, wst.yaml]`
  (§2.2 composition) + run/split sections; no scientific constant is duplicated. The
  M6 search space is **not** defined here yet (that is the milestone-5 freeze's job).
- `experiments/run_regression.py` — argparse `--config` (repeatable, later files win;
  IBEX runs add an overlay file per §2.2), then: `load_config` → `load_ground_truth` →
  `build_manifest` → `nested_loso_splits` → `record_run`, printing a one-screen
  summary. This **is the M1 smoke** (§5-D5). It clearly states modeling is
  unimplemented (M6). Other experiment entry points are created at their own
  milestones, not stubbed now (§7-A4).

**Acceptance.** `uv run python experiments/run_regression.py --config
configs/exp_a_regression.yaml` completes on real data and writes the provenance JSON
into a fresh run directory.

### 2.10 Journal files & hygiene (existing files — M1 duties only)

- **HISTORY.md** (exists): one entry per resolved step of §1 — including the
  `environment.yml` move, the kymatio/numpy resolution, and any real-data surprise
  (e.g. a file whose frame count ≠ 100). Newest-first, failures kept.
- **SECOND_CHAPTER.md** (exists): at milestone close, the distilled M1 account —
  environment provenance, data-integrity checks and their outcomes on the real cohort
  (including the workbook-precision facts in §2.5), the split protocol and why nested
  LOSO, the leakage-test design as a methods artifact.
- **archive/{code,results}/** (exist): receives `environment.yml`; nothing else at M1.

---

## §3 `eval/splits.py` — the nested-LOSO splitter contract

**Responsibility.** The **single source of folds** in the entire codebase
(implementation_plan.md §LOSO harness — "Single fold source"). No other module
constructs splits; all consumers import from here; the module docstring states this
rule.

**Inputs.**
- `subject_ids: Sequence[int]` — the **evaluable** subjects. Evaluability (QC/session
  eligibility, N_eval) is the *caller's* concern from M2 onward; `splits.py` never
  computes eligibility. At M1 the caller passes all manifest subjects.
- `n_inner_max: int = 5`, `min_train_subjects: int = 3` (from `SplitConfig`).

**Public API.**
```python
@dataclass(frozen=True)
class InnerFold:
    train_subjects: frozenset[int]
    val_subjects: frozenset[int]

@dataclass(frozen=True)
class OuterFold:
    test_subject: int
    train_subjects: frozenset[int]      # all evaluable subjects except test
    selectable: bool                    # False when n_train < min_train_subjects
    inner_folds: tuple[InnerFold, ...]  # empty when not selectable

def nested_loso_splits(subject_ids, *, n_inner_max=5,
                       min_train_subjects=3) -> list[OuterFold]

def iter_triples(folds) -> Iterator[tuple[frozenset, frozenset, int]]
    # (inner_train, inner_val, test) — flat view; reconciles the plan's
    # "(train, val, test)" phrasing with multiple inner folds per outer fold
```

**Behavior.**
- **Outer:** leave-one-subject-out over `sorted(set(subject_ids))` → one fold per
  subject; the held-out subject is touched only for final scoring. Duplicate ids in
  the input raise (a subject cannot appear twice).
- **Inner:** `sklearn.model_selection.GroupKFold(n_splits=min(n_inner_max, n_train))`
  over the outer-training subjects, groups = subject ids (full run: 15 train → 5 inner
  folds of 3 val subjects; smoke with 6 subjects: 5 train → 5 folds of 1).
- **Small-n rule — semantics stated precisely:** `min_train_subjects` constrains the
  **outer-training pool**, exactly as the main plan words it ("Inner CV requires ≥3
  training subjects"): selection is permitted iff `n_train ≥ min_train_subjects`;
  below that the outer fold is yielded with `selectable=False` and **no** inner folds —
  reported non-selectable, never run with a degenerate split. **Per-inner-fit minimums
  are deliberately *not* separately constrained:** at the `n_train = 3` boundary,
  `GroupKFold(3)` fits each inner model on only 2 subjects. This is accepted because
  (a) the real cohort operates at `n_train = 15` and the boundary arises only under
  catastrophic QC loss, where the alternative is discarding the fold entirely, and
  (b) the rule's purpose is a non-degenerate *grouped split*, which 3 training
  subjects is the minimum to provide. **Owner-confirmed 2026-07-21** (header decision 4)
  — the stricter "every inner fit sees ≥3 subjects" reading (which would require
  `n_train ≥ 4`) was considered and declined. If a later milestone revisits this, it is
  a change to `SplitConfig` semantics and must be made there — not silently inside a
  consumer.
- **Determinism:** no RNG anywhere; sorted normalization + GroupKFold's deterministic
  assignment ⇒ two calls with the same input yield identical folds.

**Guaranteed invariants (each is a unit test in `test_splits.py`; the leakage test
re-asserts S1–S4 end to end):**
- S1 `test_subject ∉ train_subjects` for every outer fold.
- S2 per inner fold: `train ∩ val = ∅`; `train ∪ val ⊆ outer.train_subjects`;
  `test_subject` in neither.
- S3 inner val sets **partition** the outer-training subjects (each training subject
  validates exactly once — GroupKFold property, asserted not assumed).
- S4 every subject is the outer test exactly once; union of test subjects = input set.
- S5 all sets non-empty whenever `selectable`.
- S6 adaptive count correct: n_subjects 16→5 inner folds; 6→5; 4→3; 3→non-selectable
  (n_train = 2 < 3).
- S7 call-twice determinism (identical output object-by-object).

---

## §4 `tests/test_no_leakage.py` — the exact assertions

Green from M1 and kept green forever (CLAUDE.md invariant 5). Four parts. The
**mandatory** suite requires no private data (the raw `data/` tree is gitignored, so a
clean checkout must still be green); real-data checks are `realdata`-marked integration
tests.

**`realdata` activation and failure semantics (explicit — a registered marker alone
neither skips nor deselects anything).** `tests/conftest.py` adds a `--realdata`
command-line option (honoured equally via a `DEHYD_REALDATA=1` environment variable, so
IBEX batch jobs need no argv plumbing):
- **Default (`uv run pytest`)** — `pytest_collection_modifyitems` attaches a skip mark
  to every `realdata`-marked test: they are **skipped cleanly**, and a clean checkout
  with no `data/` tree is fully green.
- **Acceptance (`uv run pytest --realdata`)** — the marked tests run, and the shared
  fixture that resolves the data paths **fails (does not skip)** if the 80 radar files
  or the workbook are missing or miscounted. Absent data under `--realdata` is an
  error, because that command's entire purpose is to prove the real cohort validates.

**Part A — split structure** (runs on ids 1–16 and reduced sets {6, 4, 3}):
- T1 pairwise disjointness of {inner-train, inner-val, {test}} for every yielded fold.
- T2 inner sets ⊆ outer-train; test subject appears in no inner set.
- T3 inner val sets partition outer-train.
- T4 each subject held out exactly once; no subject in two roles within any fold.
- T5 adaptive-n and non-selectable behavior per §3-S6.
- T6 determinism: two calls → identical folds.

**Part B — frame mapping** (mandatory tests on a **synthetic manifest** built through
the real `build_manifest` code path — tmp-dir `savemat` files plus a `GroundTruth`
constructed **directly in memory**, per §2.6; no synthetic workbook is involved. The
same assertions re-run on the real data as R1):
- T7 `(subject, session_idx, frame_idx)` unique across all rows.
- T8 `rel_path → subject` is a function (no file under two subjects).
- T9 for every outer fold: selecting manifest rows by `train_subjects` yields **zero**
  rows of the test subject — for any session. (The executable form of CLAUDE.md
  invariant 1.)
- **R1 (`@pytest.mark.realdata`, integration):** `build_manifest` on the real 80 files
  succeeds (C1–C6), yields 16×5×100 = 8000 rows with actual counts recorded, and
  T7–T9 hold on it. **Activation:** skipped only under the default `uv run pytest`;
  under `uv run pytest --realdata` it runs, and absent or incomplete data is a **hard
  failure**, never a skip.

**Part C — strong mutation property test** (implementation_plan.md §no-leakage, item c).
*Fixture:* deterministic CPU — synthetic session-level data, 8 subjects × 5 sessions
(n_train = 7 ⇒ a real 5-fold inner GroupKFold), small feature dim, labels = linear
function + subject offset + seeded noise via one `np.random.SeedSequence`.
**Determinism mechanism (corrected — `Ridge` has no `n_jobs`, verified 2026-07-21, and
BLAS/OpenMP env vars set inside a test arrive too late once NumPy/SciPy are imported at
collection):** the reference procedure runs inside
`threadpoolctl.threadpool_limits(1)` (threadpoolctl ships with scikit-learn — verified
importable) and pins an explicitly deterministic solver, `Ridge(solver="cholesky")`,
rather than leaving `solver="auto"` free to switch algorithms. The fixture **asserts
the achieved limit** via `threadpool_info()` inside the context, so the test verifies
single-threaded execution instead of merely documenting an intent.
*Procedure under test at M1:* a **test-local reference procedure** implementing the
harness contract — nested selection over a small enumerated grid
(`StandardScaler → Ridge`, α ∈ {0.1, 1, 10}) driven **only** by the §3 API, selection
metric = **subject-balanced session-level MAE** (per inner-val subject, the mean
absolute error over that subject's sessions; then the unweighted mean across inner-val
subjects — the main plan's "aggregate to session, mean over inner-val subjects"),
tie-break to simpler config, refit on all outer-train, predict held-out. The procedure
returns a per-fold artifact bundle: selected config, full inner score table,
**per-inner-fold fitted parameters**, final-refit parameters, training-set predictions,
held-out predictions, and the fit-audit (Part D). At **M6** the procedure under test
becomes the real `harness.py` (same assertions; the helper is deleted).

- T10 **determinism precondition:** two unmutated runs are bit-identical in every
  captured artifact (otherwise bit-comparisons below are meaningless).

*Outer-test mutations* (held-out subject only; eligibility-preserving — same
rows/shapes/membership, values only): (i) features replaced, (ii) labels replaced,
(iii) both. Everything determined before scoring must be unchanged:
- T11 selected config identical under every mutation.
- T12 full inner-CV score table bit-identical (`ndarray.tobytes()` equality).
- T13 every fitted parameter bit-identical — per-inner-fold **and** final-refit scaler
  `mean_`/`scale_`, model `coef_`/`intercept_`.
- T14 training-set predictions bit-identical.
- T15 **power checks:** feature mutation ⇒ held-out predictions change; label-only
  mutation ⇒ held-out predictions identical but held-out score changes. (A mutation
  test that cannot fail proves nothing.)

*Inner-validation mutation* (catches fitting on `inner_train + inner_val`, which the
outer-test mutation cannot detect — inner-val subjects are outer-training subjects):
- T16 mutate one subject in the fixture and, for the inner fold(s) where that subject
  is **validation**, assert that fold's fitted preprocessing and model parameters are
  bit-identical (they are functions of `inner_train` only), while the mutated
  subject's validation predictions change under feature mutation (power), and under
  label-only mutation its validation predictions are identical but its validation
  score changes. Explicitly **not** asserted invariant: inner folds where the subject
  is inner-*train* legitimately change, and the *selected config* may legitimately
  change (selection consumes validation scores) — the assertion is scoped to the
  validation-role folds' fits only.

*Selection-objective definition* (guards the subject-balanced metric itself):
- T17 a second fixture with deliberately **unequal session counts per subject**
  (e.g. 5/5/4/2 sessions across inner-val subjects, membership fixed at fixture
  construction — eligibility is already frozen at M1's level of the design): assert the
  procedure's reported inner objective equals the **hand-calculated subject-balanced
  value** exactly. A pooled-session MAE implementation (which overweights subjects
  with more sessions) fails this test; with the equal-count fixture of T10–T16 alone
  the two are numerically identical and the error would pass unnoticed.

*Torch leg:*
- T18 **written now, activated when the torch fit path exists:** same mutation
  protocol asserting bit-identical epoch budget (median-of-inner-folds rule),
  input-normalization statistics, class/sampler weights, early-stopping selection,
  every `state_dict` tensor, and training-set predictions — only the held-out
  prediction/score may change. **Activation rule (explicit; owner-confirmed
  2026-07-21 — header decision 3):** guarded by
  `pytest.importorskip("torch")` *and* a static skip marker
  ("torch fit path lands with harness.py at M6") — torch's arrival in the env at M4
  (for WST cross-backend validation) does **not** activate it, because there is no
  torch training procedure to test until `harness.py`; the marker is removed at M6
  when the test rebinds to the real harness, and T18 must be green before any torch
  result is reported.
  **Skip scope — function-level only, never module-level.** Both guards live **inside
  the T18 test function** (`pytest.importorskip("torch")` as its first statement; the
  static skip as a decorator on that function alone). A module-scope `importorskip` in
  `test_no_leakage.py` would skip **T1–T17 and T19 as well**, letting the mandatory
  leakage suite report green while none of its core assertions ran — precisely the
  silent failure this file exists to prevent. If T18 ever needs module-level torch
  imports, it moves to a separate `tests/test_no_leakage_torch.py` rather than
  widening the guard. **Guarded by acceptance:** D1 runs
  `pytest tests/test_no_leakage.py -m "not realdata"` and checks the summary shows
  T1–T17 and T19 *passed* with T18 the *only* skip — a count, not merely "no failures".
  (The `-m` filter is what makes the count unambiguous: R1 below is `realdata`-marked
  and is also skipped in a default run, so T18 is the only skipped **non-`realdata`**
  test.)

**Part D — fit-audit structure** (fixes the contract for `harness.py`'s fit-audit
artifact, implementation_plan.md §Fit-on-train-only):
- T19 the procedure emits an audit mapping every fitted quantity → the subject set it
  was estimated from, **distinguishing roles**: every inner-selection fit (scaler,
  model, and later PCA / feature selection / normalization / class or sampler weights)
  must be estimated from exactly that inner fold's `inner_train` subjects; the final
  refit must be estimated from exactly the complete `outer_train` set; **no audited
  set ever contains the test subject**, and no inner-selection fit's set contains its
  own inner-val subjects.

---

## §5 Definition of done — mapped to implementation_plan.md §Verification

- **D1 — mandatory suite, no private data.** `uv sync` reproduces the env from
  `uv.lock`; **`uv run pytest`** is green on `test_env`, `test_config`, `test_loader`,
  `test_ground_truth`, `test_manifest`, `test_splits`, `test_provenance`,
  `test_no_leakage`, with `realdata` tests **skipped** and T18 **skipped** (never
  failed). This must hold on a clean checkout with no `data/` tree. **Verified by
  count, not by absence of failures**, via an explicit check that removes the
  `realdata` tests from the picture — otherwise R1 (also in `test_no_leakage.py`, also
  skipped by default) muddies the count:

  ```
  uv run pytest tests/test_no_leakage.py -m "not realdata"
  ```

  Within that selection the summary must show **T1–T17 and T19 passed with T18 as the
  only skip** — i.e. T18 is the only skipped non-`realdata` leakage test — so a
  mis-scoped skip cannot make the suite look green while the core assertions never ran
  (§4 T18).
  ↔ Verification bullet 1 (its `test_preprocess`/`test_wst`/`test_metrics` items belong
  to M3/M4/M6).
- **D2 — real-cohort acceptance.** **`uv run pytest --realdata`** is green on this
  machine, running everything in D1 *plus* the integration tests, which **fail rather
  than skip** if the data are absent: manifest builds and validates on the real 80
  files (8000 frames; actual counts recorded; R1 green). ↔ HANDOFF DoD.
- **D3** Both ground-truth cross-checks pass on the real workbook — the dual-open
  (cached-J + formula-view) path exercised only in that `realdata` test, per the §2.5
  test split. ↔ Verification bullet 1 (ground-truth tests).
- **D4** Every fold anywhere originates from `eval/splits.py`. ↔ §Single fold source.
- **D5** M1 smoke: the §2.9 stub runs config → ground truth → manifest → splits →
  provenance end to end on real data and writes a provenance JSON (80 radar hashes +
  workbook hash + git rev) into a fresh run directory. (The plan's ≥6-subject
  *modeling* smoke is M6's DoD, cited not claimed.)
- **D6** HISTORY.md has per-step entries (env pins, `environment.yml` move, any
  real-data surprises); SECOND_CHAPTER.md has the milestone-1 section.
- **D7** `plans/implementation_plan.md` and this document agree: the staging amendment
  and the inner-validation-mutation extension (§7-A1) and the tolerance-rationale
  correction (§7-A7) are **already applied** to the main plan (done during the review
  round, 2026-07-21).

---

## §6 Local environment facts (verified 2026-07-21, planning session)

- `uv 0.11.21` installed; system Python 3.12.11 (uv fetches 3.11 per
  `.python-version`).
- `data/10ghz/`: exactly 80 `.mat` files matching the expected naming;
  `data/weight/metadata_subjects_info.xlsx` present; `archive/{code,results}/` exist.
- Workbook facts verified directly (drive §2.5): sheet reports 1000×113 from
  formatting; col B = `Subject 1..16`, row 19+ empty; col J = `=I-E` formulas with
  cached values; header row 2 mixes `datetime.time` cells with the literal string
  `'12 Noon'` (G2); Subject 15 weights use 0.05-kg increments; max K-deviation from
  computed Δm% ≈ 0.0097 pct-points (Subject 13 shows truncation, not rounding).
- Library behaviours verified directly (drive §2.5 and §4 Part C): an
  **openpyxl-written formula has no cached value** — reopening with `data_only=True`
  returns `None`; **`Ridge` exposes no `n_jobs`** parameter; **`threadpoolctl` is
  importable** via the scikit-learn install.
- No `src/`, `tests/`, `configs/`, `experiments/` yet. Branch `v1_milestone_1`, clean.
- `.gitignore` already excludes `data*/` (raw data never committed).

---

## §7 Ambiguities found while detailing, and how they are resolved

- **A1 — Mutation test vs. missing harness/torch at M1.** The plan requires
  `test_no_leakage.py` green from M1 and describes part (c) over "sklearn and torch",
  but `harness.py` arrives at M6 and torch is absent from the M1 env (entering at M4
  for WST cross-backend validation — Build order §4). **Resolution (owner-approved;
  torch timing corrected in review):** §4 Part C staging and the T18 activation rule.
  **Amendment applied to `implementation_plan.md`** (test_no_leakage paragraph):
  staging text + the inner-validation mutation assertion (review comment 1), so the
  main plan and this document state the same test.
- **A2 — `environment.yml` at the repo root** is a conda export from planning-phase
  inspections, not the project env. **Resolution (owner-approved):** archived at M1
  step 1 per the file-hygiene rule; no plan text change needed.
- **A3 — Splitter yield shape.** The plan says the API yields "(train_subjects,
  val_subjects, test_subject)" per outer fold, but one outer fold has several inner
  folds. **Resolution:** the §3 `OuterFold`/`InnerFold` structure plus `iter_triples`
  is the concrete reading; no plan edit required (recorded here).
- **A4 — `experiments/` stubs.** The repo-structure listing names several entry
  points; creating empty files for future experiments conflicts with the file-hygiene
  spirit. **Resolution:** only `run_regression.py` exists at M1 (it doubles as the
  smoke); others are created at their own milestones. No plan edit.
- **A5 — `metrics.py` not in M1.** HANDOFF's "metrics tests as they land" refers to
  later milestones; `eval/metrics.py` arrives with the harness (M6). Recorded so its
  absence from M1 is deliberate.
- **A6 — kymatio / numpy-2 compatibility** is unverified until `uv lock` runs.
  **Resolution procedure** in §2.1; outcome recorded in HISTORY.md. An env fact, not a
  design change.
- **A7 — Workbook precision facts vs. the main plan's tolerance rationale.** The main
  plan justified the 0.05 kg tolerance with "weights recorded to 0.1 kg" and the 0.05 %
  tolerance with "two-decimal rounding"; direct inspection (§6) shows Subject 15 uses
  0.05-kg increments and column K truncates at least once. **Resolution: the
  tolerances stand** (both are ≥5× the worst observed deviation) **but their
  justification is corrected to the observed workbook** — here in §2.5 and **applied
  to `implementation_plan.md`** (ground-truth bullet parentheticals), so neither
  document claims an unverified recording precision.
- **A8 — the mandatory test suite must not require the private dataset.** `data/` is
  gitignored, so "pytest green" would otherwise be unachievable on a clean checkout.
  **Resolution:** the mandatory/`realdata` split (§4 Part B) with the explicit
  `--realdata` gate and the two DoD commands (§5-D1/D2). Execution detail only — the
  main plan's Verification bullet is satisfied by the `--realdata` command; no plan
  edit needed.
- **A9 — bitwise determinism needs a mechanism, not a note.** The first draft said
  "sklearn `n_jobs=1`, single-threaded env vars set in the test"; `Ridge` has no
  `n_jobs` and in-test env vars land too late (§6). **Resolution:**
  `threadpool_limits(1)` + `Ridge(solver="cholesky")` + an asserted thread limit
  (§4 Part C). This is how the main plan's "deterministic CPU fixture (fixed seeds,
  single-threaded)" is actually achieved; no plan edit needed.
- **A10 — the config floor for `min_train_subjects` must be the protocol floor, not
  the library floor.** An earlier draft validated `>= 2` (the `GroupKFold` minimum),
  which would let an overlay YAML weaken the main plan's "Inner CV requires ≥3 training
  subjects" while staying syntactically valid. **Resolution:** validate `>= 3` (§2.2).
  The main plan already states the rule; the config layer now enforces it. No plan edit
  needed.
- **A11 — file identity must be logical, not repository-relative.** A repo-relative
  physical path is not portable: IBEX's data root sits outside the repository, so the
  same file would acquire `..`-laden, machine-specific identities. **Resolution:** the
  manifest's canonical identity is `rel_path` relative to `data_10ghz_dir`, resolved
  against that root for I/O and hashing; provenance records logical path + hash of the
  resolved file (§2.6, §2.8). Execution detail consistent with the main plan's
  provenance requirements; no plan edit needed.
- **A12 — the canonical config cannot be loaded by a mandatory test.** Required-input
  path validation and "mandatory suite green on a clean checkout" are in direct
  tension, because `configs/data.yaml` points at the private `data/` tree.
  **Resolution:** mandatory config tests always append a `tmp_path` overlay; loading
  the canonical config unmodified is a `realdata` integration test (§2.2). Execution
  detail; no plan edit needed.
- **A13 — a mis-scoped torch skip could hide the entire leakage suite.** A module-level
  `pytest.importorskip("torch")` in `test_no_leakage.py` would skip T1–T17 and T19 too,
  reporting green with nothing verified. **Resolution:** both T18 guards are
  function-scoped, with D1 verifying the pass/skip *counts* rather than just the absence
  of failures (§4 T18, §5-D1). This protects CLAUDE.md invariant 5 — a test that cannot
  silently stop testing — so it is worth the explicit acceptance check.

---

## Outcome (closed 2026-07-21)

Approved and implemented in full; the implementation_plan.md amendments (A1, A7) were
applied during review. Commit `f3fbade` — 34 files, 5783 insertions, 159 tests.

**What the build changed relative to this plan** (details in HISTORY.md):
- **§2.1 env risk resolved differently than predicted.** The flagged kymatio↔numpy-2
  conflict does not exist; the real one is **kymatio 0.3.0 ↔ scipy ≥1.17**
  (`scipy.special.sph_harm` removed). Pinned `scipy>=1.11,<1.17`. Subtle because
  `import kymatio` succeeds — only `from kymatio.numpy import Scattering1D` fails.
- **§2.4 loader fallback not needed.** `whosmat` measured at 0.017 s/file, so
  header-only inspection stands; no full-`loadmat` fallback was taken.
- **§4 Part C contract extended.** `FoldResult` gained `train_subjects` — the fit audit
  cannot verify roles without it, so `harness.py` must expose it too at M6.
- **Not created:** `configs/ibex.yaml` (would name paths that fail input validation
  locally; written at the first IBEX milestone). `results/runs/` is gitignored.
- **Post-plan fix:** `.gitignore`'s unanchored `data*/` was excluding `src/dehyd/data/`;
  anchored to `/data*/`.

