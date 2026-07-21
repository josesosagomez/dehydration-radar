# HISTORY — implementation log

Running record of every attempt, newest-first. Each entry: what was tried, whether it
succeeded/failed **and why**, and the concrete parameter values + reasoning. Failures
stay in the log. A new session reads only the most recent entries to orient.

---

## 2026-07-21 — **MILESTONE 1 COMPLETE.** Definition of done met in full.

**D1 — mandatory suite, no private data.** `uv run pytest` → **151 passed, 8 skipped**
on a checkout with no `data/` tree required (the 8 skips are the 7 `realdata` tests plus
T18).
**D1 count check.** `uv run pytest tests/test_no_leakage.py -m "not realdata"` →
**24 passed, 1 skipped, 1 deselected** — T18 is the *only* skipped non-`realdata`
leakage test, verified as a count so a mis-scoped skip cannot hide the suite.
**D2 — real-cohort acceptance.** `uv run pytest --realdata` → **158 passed, 1 skipped**
(T18). Manifest builds and validates on the real 80 files: 8000 frames, 16 subjects,
80 sessions.
**D3 —** both ground-truth cross-checks pass on the real workbook for all 16 subjects.
**D4 —** every fold anywhere originates from `eval/splits.py`; nothing else constructs
splits.
**D5 —** the smoke runs end to end on real data and writes provenance with 80 radar
hashes + workbook hash + git rev.
**D6 —** HISTORY.md has a per-step entry (steps 1–9); SECOND_CHAPTER.md §0.1 written.
**D7 —** `plans/implementation_plan.md` and `plans/MILESTONE_1_PLAN.md` agree (the A1
and A7 amendments were applied during the review rounds).

**Milestone-1 scoreboard.** 8 source modules, 8 test modules, **159 tests**. Two
genuine environment/dependency facts discovered empirically rather than assumed
(kymatio↔scipy `sph_harm`; openpyxl formula caches). Four test-scaffolding bugs found
and fixed by the tests themselves during the build (openpyxl `value=None` no-op;
`rel_path` string ordering vs session ordering; `tests/` not a package; `FoldResult`
missing `train_subjects`). The leakage suite was validated **adversarially** — a
deliberately leaky procedure fails it — so its green state carries evidence, not just
absence of failure.

**Open for M2:** install `h5py` and run the minimal 77 GHz audit; QC screens with frozen
thresholds and per-frame reason codes; add QC/eligibility columns to the manifest;
`configs/ibex.yaml` when the cluster roots are known.

---

## 2026-07-21 — M1 step 9: entry-point stub + end-to-end smoke on real data.
## **Success.**

**What was built.** `experiments/run_regression.py`. Implements MILESTONE_1_PLAN §2.9.

**Smoke result (real data, this machine).**
```
config       : configs/exp_a_regression.yaml
device       : cpu   seed: 20260721
ground truth : 16 subjects, 80 sessions
               Delta m% range -2.02 .. 0.00
manifest     : 8000 frames, 16 subjects, 80 sessions
folds        : 16 outer (16 selectable), 5 inner each
provenance   : results/runs/20260721T094017375792Z_2a26fff2/provenance.json
```
The Δm% range **−2.02 … 0.00** matches the expected ≈0 to ≈−2% from the workbook
inspection, and N_eval = 16 with 15 training subjects and 5 inner folds per outer fold
is exactly the full-cohort case the protocol describes.

**Provenance artifact verified:** 80 radar hashes + the workbook hash, all as logical
`rel_path`s; resolved config; 16 fold records with roles; git commit/branch/dirty;
package versions (with `torch: null`, `h5py: null` positively recorded as absent);
platform; `slurm_job_id: null` locally.

**Decisions.**
- The stub deliberately **stops after the data spine**; modeling lands at M6 on top of
  exactly these folds. It prints what it built rather than pretending to model.
- `sys.path` insertion of `src/` so the script runs without an editable install.
- **`results/runs/` is gitignored.** Every invocation writes a new timestamped run
  directory, so committing them all would be noise; `results/` itself stays available
  for curated, reported artifacts added deliberately. Reversible if the owner prefers
  full provenance history in git.
- **`configs/ibex.yaml` not created.** The overlay *mechanism* is implemented and
  tested (`--config` is repeatable, later files win, and a test asserts an overlay
  replaces the data root), but a committed IBEX file would have to name paths that do
  not exist yet and would fail input-path validation locally. It is written at the
  first IBEX milestone, when the real roots are known.

**Next:** step 10 — journal close-out (SECOND_CHAPTER.md milestone-1 section).

---

## 2026-07-21 — M1 step 8: `tests/test_no_leakage.py` (T1–T19) + reference procedure.
## **Success**, and verified adversarially.

**What was built.** `tests/reference_procedure.py` (the contract `harness.py` must
satisfy) and `tests/test_no_leakage.py` (25 tests covering T1–T19). Implements
MILESTONE_1_PLAN §4.

**The reference procedure.** Deterministic nested select-and-refit over
`StandardScaler → Ridge`, α ∈ {0.1, 1, 10}, folds taken **only** from
`eval/splits.py`. Selection metric is the **subject-balanced** session-level MAE. It
returns an auditable bundle per fold — selected config, full inner score table,
per-inner-fold *and* final fitted parameters, training/val/test predictions, and the
fit audit — so tests verify **roles**, not implementation trust. At M6 the leakage
tests rebind to the real harness and this module is deleted.

**Determinism mechanism (the corrected one).** `Ridge` has **no `n_jobs`**, and BLAS
thread env vars set inside a test arrive too late (NumPy/SciPy are already imported at
collection). So the numeric work runs inside `threadpool_limits(1)` with an explicitly
deterministic `solver="cholesky"` instead of leaving `solver="auto"` free to switch
algorithm. A test **asserts the achieved limit** via `threadpool_info()` rather than
documenting an intention. T10 (two unmutated runs bit-identical) is the precondition
that makes every later bit-for-bit comparison non-vacuous.

**Both CV levels are tested, which is the point.**
- **T11–T15 (outer):** mutating the held-out subject's features/labels/both leaves
  selected α, the inner score table, every fitted parameter (inner *and* final), and
  the training predictions **bit-identical** (`.tobytes()`); power checks confirm
  feature mutation moves the held-out prediction and label-only mutation moves the
  score but not the prediction.
- **T16 (inner):** the outer mutation **cannot** detect fitting on
  `inner_train + inner_val`, because inner-val subjects *are* outer-training subjects.
  So a separate test mutates a training subject and asserts that, for the folds where
  it is **validation**, that fold's fits are bit-identical — while its own validation
  predictions do move. Scope is deliberate and documented: folds where it is
  inner-train legitimately change, as may the selected config.
- **T17:** with equal session counts, subject-balanced and pooled MAE are numerically
  identical, so a pooled implementation would pass unnoticed. Tested against a
  **hand-calculated** value on a deliberately unequal fixture (5/2 sessions → 5.5, not
  25/7), plus an end-to-end run with counts {5,5,4,2,5,3,5,4}.
- **T19:** the audit must map every fitted quantity to the subject set it came from —
  inner fits from exactly that fold's `inner_train`, the final refit from exactly the
  full `outer_train`, no audited set ever containing the test subject — plus a test
  that the audit **covers every fitted quantity** (an audit with silent omissions
  would be worthless).

**T18 skip scope (the subtle one).** Both guards — `pytest.importorskip("torch")` and
the static marker — are **inside the T18 function**. A module-level `importorskip`
would skip T1–T17 and T19 too, letting the file report green with nothing verified.
Verified by the acceptance command:
`uv run pytest tests/test_no_leakage.py -m "not realdata"` → **24 passed, 1 skipped
(T18 only), 1 deselected (R1)** — checked as a *count*, not as "no failures".

**Adversarial verification (does the test actually catch a leak?).** A passing test
proves nothing on its own, so the reference procedure was monkeypatched to fit the
scaler on **train + held-out** — the classic leak — and T13 was re-run. Result: clean
procedure passes, **leaky procedure fails**. The test has teeth.

**Two failures during the step, both mine, both in test scaffolding.**
1. `from .reference_procedure import ...` → `ImportError: attempted relative import
   with no known parent package`. `tests/` is not a package; switched to an absolute
   import (pytest's default import mode puts the test dir on `sys.path`).
2. T19 → `AttributeError: 'FoldResult' has no attribute 'train_subjects'`. The audit
   needs the outer-training set to check against; added `train_subjects` to
   `FoldResult`. This is a genuine improvement to the harness contract, not a
   workaround — the real `harness.py` must expose it too.

**Verification.** `uv run pytest` → 151 passed, 8 skipped.
`uv run pytest --realdata` → 158 passed, 1 skipped (T18).

**Next:** step 9 — `experiments/run_regression.py` stub and the M1 end-to-end smoke.

---

## 2026-07-21 — M1 step 7: provenance recorder. **Success.**

**What was built.** `src/dehyd/provenance.py`, `tests/test_provenance.py` (14 tests).
Implements MILESTONE_1_PLAN §2.8.

**Concrete decisions and why.**
- **`results_dir` is the single output authority.** `record_run(config, manifest,
  folds, extra)` has **no `out_dir` parameter**, so the destination cannot be given two
  ways and disagree. Tests supply a `Config` whose `results_dir` is `tmp_path`, which
  also keeps every test write outside the repo — a test run therefore cannot alter the
  git-dirty flag that a later assertion reads.
- **The ground-truth workbook is hashed alongside the 80 radar files.** Hashing only
  radar data would let the labels change without provenance noticing.
- **Logical identity + physical hash.** Entries are `{rel_path, sha256}`: `rel_path`
  from the manifest (portable across Windows/IBEX), hash computed on the resolved file.
  Tests assert no absolute paths and no `..` segments.
- **Canonical serialization** — radar entries sorted by `rel_path`, folds sorted by
  test subject, subject sets as sorted lists, `json.dumps(sort_keys=True)`. A test
  asserts two runs on unchanged inputs differ **only** in `timestamp_utc`.
- **Windows-safe run directories.** `results_dir/runs/<YYYYMMDDTHHMMSSffffffZ>_<rev>/`
  — no colons (invalid in Windows paths) and microsecond precision so two runs in the
  same second cannot collide; a real ISO-8601 timestamp is kept *inside* the JSON.
  Existing `provenance.json` → raises rather than overwriting (tested by pinning the
  stamp format so two runs collide on purpose).
- Package versions include `torch` and `h5py` as `None` — a positive record that they
  were absent, rather than silence.

**One failure during the step, and what it was.** `test_hash_changes_when_data_changes`
failed with two identical hashes. Cause was in the **test**: it compared
`radar_files[0]`, but that list is sorted by `rel_path`, where
`"subject_1_10am.mat"` sorts **before** `"subject_1_8am.mat"` — so index 0 was not the
file the test had modified. Fixed by looking entries up **by path**, and strengthened
to also assert every untouched file's hash is unchanged. Worth recording because the
same string-vs-session ordering trap will recur wherever `rel_path` order is mistaken
for session order.

**Verification.** `uv run pytest` → 127 passed, 6 skipped.

**Next:** step 8 — `tests/test_no_leakage.py` (T1–T19), the milestone capstone.

---

## 2026-07-21 — M1 step 6: nested-LOSO splitter (the single fold source). **Success.**

**What was built.** `src/dehyd/eval/splits.py`, `tests/test_splits.py` (23 tests).
Implements MILESTONE_1_PLAN §3.

**Concrete decisions and why.**
- **`GroupKFold` over one row per subject, not per frame.** The inner splitter is fed
  a `(n_train_subjects, 1)` array with `groups = subject_ids`, so it is literally
  splitting *subjects*; frame-level selection happens downstream by filtering on the
  returned subject sets. This makes it structurally impossible for the splitter to
  emit a frame-level split, which is the invariant it exists to protect.
- **Adaptive inner count `min(n_inner_max, n_train)`** — 5 folds at the full cohort
  (15 training subjects), 5 at the 6-subject smoke subset, 3 at n_train=3.
- **`min_train_subjects` constrains the outer-training pool** (owner decision 4). At
  the boundary `n_train == 3`, `GroupKFold(3)` fits each inner model on 2 subjects;
  a test asserts exactly this so the accepted consequence is visible in the suite
  rather than buried in prose. Below the floor the fold is returned with
  `selectable=False` and **no** inner folds — reported as non-selectable, never run
  degenerate.
- **Frozen dataclasses** (`OuterFold`, `InnerFold`, `frozenset` members) so a consumer
  cannot mutate a fold in place; tested.
- **No RNG anywhere.** Subjects are sorted on entry and GroupKFold's assignment is
  deterministic, so `nested_loso_splits(x) == nested_loso_splits(x)` and input order is
  irrelevant — both tested (S7).
- **Duplicate subject ids raise.** A subject appearing twice would let one copy train
  while another is held out — the exact failure LOSO exists to prevent.
- `iter_triples()` provides the flat `(inner_train, inner_val, test)` view, reconciling
  the main plan's "(train, val, test)" phrasing with several inner folds per outer fold.

**Verification.** All seven documented invariants S1–S7 are unit tests:
S1 test∉train; S2 inner disjoint and ⊆ outer-train with test in neither; S3 inner val
sets **partition** outer-train (asserted, not assumed from the GroupKFold docs); S4
each subject held out exactly once; S5 non-empty when selectable; S6 adaptive count at
n∈{16,6,4,3}; S7 determinism. Full suite: 113 passed, 6 skipped.

**Next:** step 7 — `provenance.py` + `tests/test_provenance.py`.

---

## 2026-07-21 — M1 step 5: frame manifest + structural gate. **Success.**

**What was built.** `src/dehyd/data/manifest.py`, `tests/test_manifest.py` (17 tests).
Implements MILESTONE_1_PLAN §2.6.

**Concrete decisions and why.**
- **Logical file identity (`rel_path`), not repo-relative.** The manifest stores the
  path **relative to `data_10ghz_dir`** (`subject_1_8am.mat`), resolved against that
  root for I/O and hashing via `resolve_path()`. A repo-relative path would break on
  IBEX, whose data root lives outside the repository — the same file would then carry
  machine-specific `..` segments and a different identity per machine. A test asserts
  no `..`, no leading `/`, no drive letters.
- **Deterministic ordering + fixed dtypes.** Sorted by `(subject, session_idx,
  frame_idx)` with the index reset, and every column dtype asserted. Verified by a test
  that **monkeypatches `Path.glob` to return reversed order** and asserts the two builds
  are frame-for-frame identical — so filesystem enumeration order can never reach
  training order, hashes, or saved artifacts.
- **All six checks fail loudly and name every offender**, not just the first: C1
  completeness, C2 duplicates, C3 unparseable/stray files, C4 bijection **in both
  directions** (file with no workbook row *and* workbook row with no file), C5 per-file
  structure (shape and MATLAB class), C6 actual frame counts. A test with two missing
  cells asserts both are named.
- **Mandatory tests build `GroundTruth` directly in memory** (it is two DataFrames)
  rather than round-tripping a synthetic workbook — sidesteps the openpyxl
  formula-cache limitation from step 4 and keeps these tests about the manifest.
- **Frame counts come from the file.** A synthetic session with counts {3,7,2,5,4}
  confirms per-file `n_frames_in_file` and contiguous `frame_idx` — the M2 eligibility
  rule `ceil(0.5 × actual_frame_count)` depends on this not being a hard-coded 100.
- QC columns (reason codes, eligibility) deliberately **not stubbed**; they arrive at M2.

**Verification.** `uv run pytest` → 90 passed, 6 skipped. `uv run pytest --realdata`
→ 96 passed. On the real data the manifest builds to **8000 rows** (16×5×100), subjects
exactly {1..16}, every session exactly 100 frames, all dtypes as specified.

**Next:** step 6 — `eval/splits.py` + `tests/test_splits.py`.

---

## 2026-07-21 — M1 step 4: ground truth (fixed-cell parse + cross-checks). **Success.**

**What was built.** `src/dehyd/data/ground_truth.py`, `tests/test_ground_truth.py`
(31 tests). Implements MILESTONE_1_PLAN §2.5.

**Module split forced by an openpyxl limitation (verified, not assumed).** openpyxl
writes formulas but never evaluates them, so a synthetic workbook can hold **either**
an `=I-E` formula in column J **or** a cached number — never both. No synthetic fixture
can therefore exercise the full dual-view load. The module is split so each view is
independently testable:
- `_validate_layout(ws_formula)` — headers, column-B subject identity, J formula
  structure. Formula-view fixtures.
- `_read_values(ws_data_only)` — masses, covariates, cached J, K text. Literal-value
  fixtures.
- `check_targets(...)` — pure math + both cross-checks, no I/O. Array-level tests,
  including tolerance boundary behaviour and the sign convention.
- `load_ground_truth()` — the only place the two views meet; exercised on the **real**
  workbook (which genuinely has formulas *and* Excel-written caches) in a `realdata`
  test.

**Concrete decisions and why.**
- **Identity parsed from column B** (`^Subject (\d+)$`), asserted unique and exactly
  {1..16} — not inferred from row position. The owner-confirmed radar↔workbook identity
  is thus *checked*, not assumed.
- **Extra-subject guard scans all of column B**, not just the rows below the block; a
  `Subject 17` planted at row 400 is caught (tested).
- **Covariates validated before BMI** is computed from them (age 15–80, height
  120–220 cm) — a metres-instead-of-cm height is caught before it silently produces a
  BMI of ~24000.
- **Tolerances kept at 0.05 kg / 0.05 pct-points but re-justified from the observed
  workbook**, not from an assumed recording precision: Subject 15 uses 0.05-kg
  increments and column K truncates (Subject 13: 0.5997% → "0.59"), worst observed
  deviation ≈0.0097 pct-points, so 0.05 is ≈5× the worst case. Recorded in the module
  docstring so the number is never mistaken for a claim about the instrument.
- **All problems are reported at once**, not just the first — a test asserts two
  corrupted J cells both appear in the error.

**One failure during the step, and what it was.** `test_missing_weight_detected`
initially did not raise. Cause was in the **test**, not the parser:
`ws.cell(row, col, value=None)` is a **no-op** in openpyxl (None is the sentinel for
"don't set"), so the cell was never blanked. Fixed by assigning `.value = None`
directly, and a `"n/a"` string case was added alongside. The mass check was factored
into `_is_plausible_mass()` while fixing it (also rejects `bool`, which is an `int`
subclass).

**Verification.** `uv run pytest` → 74 passed, 5 skipped. `uv run pytest --realdata`
→ 79 passed. On the real workbook: 80 session rows, 16 subjects, **both cross-checks
pass for all 16**; S0 is identically zero, all S4 deltas negative and > −3%, all BMIs
in 15–45.

**Next:** step 5 — `manifest.py` + `tests/test_manifest.py`.

---

## 2026-07-21 — M1 step 3: sessions + minimal 10 GHz loader. **Success.**

**What was built.** `src/dehyd/data/sessions.py` (the single definition of session
order), `src/dehyd/data/loader_10ghz.py` (filename parse, header inspect, full load),
`tests/test_loader.py` (22 tests). Implements MILESTONE_1_PLAN §2.3–2.4.

**Concrete decisions and why.**
- **Header-only inspection via `scipy.io.whosmat`.** Measured on a real file:
  **0.017 s**, so all 80 files cost ≈1.4 s instead of decompressing ≈1.4 GB. The
  planned fallback (full `loadmat` per file) was therefore **not** needed.
- **`whosmat` returns the MATLAB class**, confirmed `('framesRadar', (534, 20, 100),
  'double')` on the real data — so the class assertion the plan asked for is checkable
  without loading. An `int16` array of the correct shape is rejected (tested).
- **`loadmat(..., variable_names=["framesRadar"])`** so the unused
  `framesRadarIQ` [20834×2×100] is never decompressed. A test writes a file containing
  both and confirms loading succeeds regardless.
- **Frame count is read from the file, never assumed 100** — session eligibility at M2
  is `ceil(0.5 × actual_frame_count)`, so an assumed constant would silently corrupt it.
  Tested with a 42-frame synthetic file.
- **Strict filename regex.** An unparseable name raises rather than being skipped,
  because "unmatched file" is a manifest failure condition, not a benign case. Seven
  malformed-name variants tested, including `.MAT` case and trailing junk.
- Complex-dtype check lives in `load_10ghz_file` (whosmat cannot report complexity);
  a real-valued double cube of the right shape is rejected.

**Verification.** `uv run pytest` → 43 passed, 3 skipped. `uv run pytest --realdata`
→ 46 passed: **all 80 real files** inspect as `(534, 20, 100)` MATLAB-class `double`,
subjects exactly {1..16}, and a full real load is complex128 and all-finite.

**Next:** step 4 — `ground_truth.py` + `tests/test_ground_truth.py`.

---

## 2026-07-21 — M1 step 2: config system. **Success.**

**What was built.** `src/dehyd/config.py` (frozen dataclass schema + `load_config`),
`configs/{data,preprocess,wst,exp_a_regression}.yaml`, `tests/test_config.py`
(21 tests). Implements MILESTONE_1_PLAN §2.2.

**Concrete decisions and why.**
- **Two path rules, deliberately different.** `include:` entries resolve against the
  **declaring YAML's directory** (so `exp_a_regression.yaml` can say `data.yaml` and
  find its sibling); path **values** resolve against the **repo root** (so a data root
  means the same thing from any CWD or declaring file). Both are covered by a test that
  loads from an unrelated CWD via `monkeypatch.chdir` and compares the fully resolved
  configs.
- **Merge = later wins, lists replaced wholesale**, never concatenated — a later config
  states the entire intended value. Tested directly (`seed_set` replacement).
- **`include:` may not nest.** Flat composition keeps the resolution order followable;
  nesting raises.
- **Numeric floors enforced at the config layer, not just documented:** `seed_set` must
  be exactly 5 **distinct** seeds (duplicates would silently reduce effective repeats);
  `n_inner_max >= 2` (GroupKFold); **`min_train_subjects >= 3`** — deliberately stricter
  than GroupKFold's mechanical floor of 2, because the approved protocol requires ≥3
  training subjects before an outer fold is selectable. A permissive floor here would
  let an overlay YAML weaken the nested-CV rule while staying syntactically valid.
- **`wst.tilings` cannot be overridden in YAML** — the three tilings are frozen design
  constants; J and output shape are derived/measured at M4, never hard-coded.
- **`results_dir` is not required to exist** (output, created on demand) while
  `data_10ghz_dir` / `weight_xlsx` are (required inputs). This distinction is what makes
  the mandatory suite runnable on a clean checkout.
- **Mandatory tests never touch the private data:** each appends a final `tmp_path`
  overlay redirecting the input paths, so composition/merge/path-rules/validation are
  all exercised without `data/`. That the *canonical* config resolves to the real
  dataset is a separate `realdata` test.

**Verification.** `uv run pytest` → 21 passed, 1 skipped (the `realdata` test).
`uv run pytest --realdata` → 22 passed. Both gate directions confirmed working.

**Next:** step 3 — `sessions.py` + minimal `loader_10ghz.py` + `tests/test_loader.py`.

---

## 2026-07-21 — M1 step 1: environment + repo skeleton. **Success**, with one real
## dependency conflict found (scipy, not numpy).

**What was tried.** Created the pinned uv environment and package skeleton per
`plans/MILESTONE_1_PLAN.md` §1 step 1 / §2.1: `pyproject.toml` (package `dehyd`, src
layout, `requires-python >=3.11`), `.python-version` = 3.11 (uv fetched CPython
3.11.15), `uv lock` + `uv sync`, `src/dehyd/{data,eval}/` skeleton, `tests/test_env.py`,
`tests/conftest.py` (the `--realdata` gate), `.gitignore` additions.

**The env unknown resolved — but it was not the anticipated one.** The plan flagged a
possible **kymatio vs numpy 2.x** conflict, with the contingency "pin numpy<2". That
conflict does **not** exist: kymatio 0.3.0 imports and runs fine on numpy 2.4.6.
The actual conflict is **kymatio 0.3.0 vs scipy ≥1.17**: `kymatio/scattering3d/
filter_bank.py` imports `scipy.special.sph_harm`, which scipy **removed in 1.17**
(superseded by `sph_harm_y`). Symptom is subtle and would have surfaced at M4, not
here: top-level `import kymatio` **succeeds** (so a naive import smoke passes), but
`from kymatio.numpy import Scattering1D` raises `ImportError` because the 1-D entry
point pulls in the 3-D filter bank.

- **Resolution:** pin **`scipy>=1.11,<1.17`** in `pyproject.toml` with the reason in a
  comment; revisit when kymatio ships a release using `sph_harm_y`.
- **Resolved versions:** python 3.11.15, numpy 2.4.6, scipy 1.16.3, kymatio 0.3.0,
  scikit-learn 1.9.0, pandas 2.3.3, openpyxl 3.1.5, PyYAML 6.0.3, pytest 9.1.1,
  threadpoolctl 3.6.0 (arrives via scikit-learn — needed for the M1 determinism
  fixture, §4 Part C).
- **Verified after the pin:** `Scattering1D(J=7, shape=(470,), Q=(10,4), T=104,
  max_order=2)` instantiates and transforms, output shape `(742, 7)`. kymatio emits
  `UserWarning: Signal support is too small to avoid border effects` for J=7 on 470
  samples — **noted for M4**, where the plan already requires padding/output shape to
  be *measured* from the instantiated filter bank rather than assumed. Not an M1 issue.

**Why the plan's ordering paid off.** §2.1 put env resolution first precisely so an
unknown like this fails before any code depends on it. It did — and it was a different
unknown than predicted, which is the argument for resolving it empirically rather than
assuming the documented risk was the only one.

**Incidental.** `environment.yml` (planning-phase conda export) moved to
`archive/code/` per the file-hygiene rule (owner decision 2) — `uv` is now the sole
local env manager. A stale `.pytest_cache/` at the repo root has an unreadable ACL on
this machine (cannot be read, `takeown`'d, or removed without elevation) and made
pytest warn on every run; worked around by setting `cache_dir = ".cache/pytest"` in
`[tool.pytest.ini_options]` rather than leaving permanent noise in the test output.

**Outcome:** `uv run pytest` green (2 passed, no warnings).
**Next:** step 2 — `configs/data.yaml` + `src/dehyd/config.py` + `tests/test_config.py`.

---

## 2026-07-21 — Planning phase complete; plan approved and hardened. Pre-implementation.

**State:** No implementation code written yet. The design is locked in
`plans/implementation_plan.md` and is the spec milestone 1 builds against.

**What was done.** Read CLAUDE.md/AGENTS.md + ROADMAP.md in full, the paper
(`paper/`), and the MATLAB reference (`matlab/`). Inspected a real 10 GHz file
byte-for-byte (not assumed from the paper) and parsed the weight workbook. Produced the
implementation plan, then hardened it across **7 rounds of independent (Codex) review**
— every comment resolved; reviewer's final verdict was "no further comments,
implementation-ready."

**Verified data facts (not assumed).**
- 10 GHz: `data/10ghz/subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat`, 80 files, MAT v5,
  little-endian, zlib. Var `framesRadar` = MATLAB **double** `[534 fast-time × 20 chirps
  × 100 frames]`, complex; on disk the elements are `miINT16` (space optimization) so
  `scipy.io.loadmat` returns **complex128**. Also `framesRadarIQ` [20834×2×100] (raw IQ,
  unused). One file = one subject/session = 100 frames.
- 77 GHz: MAT **v7.3/HDF5**, ~285 MB each (~23 GB), needs `h5py` (not yet installed —
  milestone-2 audit installs it). h5py-reported shape (reviewer-sampled)
  `(16,256,256,125)=(Nrx,Nchirps,Nfast,Nframes)`; full axis reversal →
  `(Nframes,Nfast,Nchirps,Nrx)`. Fast-time↔chirp (both 256) disambiguated by a raw-data
  signal-domain check, not shape alone.
- Ground truth: `data/weight/metadata_subjects_info.xlsx`, sheet `MetaData`, rows 3–18.
  Two-row merged header → parse by fixed cell addresses. Cols E–I = weights 8am→4pm.
  Signed target `Δm% = (m(s) − m(S0))/m(S0)×100` (negative = loss), ≈0 to ≈−2%.
- Subject identity: radar `subject_N` = workbook "Subject N" (owner-confirmed; old
  MATLAB 5–20 numbering was renumbered to 1–16 for cleanliness, same subjects/order).

**Key locked decisions & why (see plan for full detail).**
- MATLAB is a **design reference only** — Python is the sole source of all reported
  numbers; correctness via Python-native self-consistency checks, not numeric diffs.
- Headline = **fluid-loss (Δm%) regression under LOSO**; 5-class demoted to secondary
  **ordinal**. Analysis unit is **session-level** (aggregate per-frame WST features to
  one vector/session) to kill pseudo-replication; per-frame is diagnostic only.
- Deliberate departures from the reference (logged here as they're implemented):
  robust standardize = median/MAD (not the reference's mean/MAD mix); range gate is a
  config parameter (default 1–2 m); WST log transform = order-aware
  (`log(S+ε)` on orders 1–2, ε=1e-6; order 0 left linear); EdgeTrim=32 **after**
  reduction.
- Scoring counts use **N_eval** (evaluable subjects), never a hard 16; session
  eligibility = `≥ ceil(0.5 × actual_frame_count)` QC-passing frames, no imputation.
- 77 GHz primary feature = **slow-time (Doppler) I/Q WST, per-Rx, feature-space fused**
  (magnitude discards Doppler phase; coherent Rx averaging risks phase cancellation).
- Stats: subject-cluster bootstrap (B=10000), seeds collapsed (metric-type-aware),
  all CIs/p-values labeled **conditional/exploratory**; effect sizes + per-subject
  spread carry interpretation.

**Outcome:** success (planning). **Next:** milestone 1 — repo scaffold, config system,
manifest + nested-LOSO splitter + provenance, and `tests/test_no_leakage.py` green
before any modeling.
