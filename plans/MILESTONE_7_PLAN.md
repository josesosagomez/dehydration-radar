# MILESTONE 7 PLAN — the LOSO harness + fluid-loss regression (Exp A), session-level, both bands

## §0 Scope and ground rules

**Why this milestone exists (implementation_plan.md §Build order step 7).** M1–M6 built the
data spine, preprocessing, the WST front-end (both bands), and the frozen A–G protocol. None
of it has yet fit a single model against this cohort. M7 is the **first modelling milestone**:
it builds the fit-on-train-only nested-LOSO harness and runs **Experiment A** — session-level
LOSO regression of signed Δm% from WST features, on **both 10 GHz and 77 GHz**, against the
session-index-only (time-of-day) baseline the radar must beat given the fasting/hydration
confound. This is where the `config-freeze-v1` guarantee starts to be *spent*: the moment an
outer-fold result is inspected, every later design change is exploratory by definition, so the
milestone is deliberately staged to reach "everything green, nothing inspected" and **stop for
owner sign-off** before the first full-cohort outer result exists.

**Review.** This plan is authored before implementation and reviewed through the Codex⇄Claude
loop (`plans/review_prompt_codex.md` / `plans/review_prompt_claude.md`); the resolved-comment
log and any owner-deferred items live in the **Plan review** section appended at the end of
this file. Nothing in §1–§7 is implemented until that review closes and the owner clears any
deferred items.

## Step 0 — owner decisions (RESOLVED 2026-07-25)

Three decisions were settled with the owner before authoring (via the planning session's
clarifying questions). Nothing downstream is conditional on them being reopened.

1. **T18 activation is the ONE sanctioned edit to the frozen `tests/test_no_leakage.py`.**
   The torch mutation leg (`test_t18_torch_mutation_property`) cannot go green while the file
   is literally byte-for-byte frozen: **both** its `@pytest.mark.skip` decorator **and** its
   placeholder body (`raise AssertionError("unreachable until M6")`) live inside the frozen
   file. MILESTONE_1_PLAN.md pre-registered exactly this edit ("the marker is removed at [the
   harness milestone] when the test rebinds to the real harness"). Decision: **activate T18**
   — remove the skip decorator, write the real torch-mutation body per the M1 spec — as a
   pre-registered, documented amendment (**A-M7-1**, §6). The change is bounded to the T18
   hunk; HISTORY.md records a `git diff` proving T1–T17 and T19 are byte-identical.
2. **The 77 GHz feature store is built on IBEX**, as a job array cloning the proven
   `scripts/ibex/wst77.sbatch` pattern (per-session shards + fingerprint sidecars, merged and
   validated locally). The 10 GHz store is built locally (~25–50 min). The owner submits the
   IBEX job mid-milestone, **after** the git-provenance fix (step 1) lands.
3. **An explicit owner checkpoint precedes the first full-cohort Exp A run.** M7 proceeds
   through build → all tests green → both smoke runs, then **STOPS and reports**. The
   full-cohort run — which produces the first outer-fold results and spends the freeze — runs
   only on explicit owner go-ahead, enforced structurally (see §2, `run_regression.py`: the
   run is refused unless `--full-cohort` is passed).

## Step 0b — owner-approval items (RESOLVED 2026-07-25)

Three protocol-completion choices this plan raised were **not** treated as settled until the
owner decided them — consistent with the M6 review's process rule (a technically-sensible
completion of an authoritative-protocol gap must clear an explicit owner gate *before*
implementation, not be applied silently — the C6-16 precedent). All three are now decided (the
full decision threads, with options, are in the Plan-review "Deferred to owner" section at the
end of this file, C17). The summaries below record the outcomes.

- **O1 (A-M7-2) — `inner_fold_variance` estimator: population std `np.std(ddof=0)`.** Tertiary
  tie-break key (reached only after MAE, `simplicity_rank`, and `feature_dimension` all tie) —
  low stakes; the observed inner folds are treated as the whole population. Used by §2.3.
- **O2 — session-index baseline, absent-index behaviour: global training-fold mean.** So every
  test session stays scored and radar-vs-baseline share the identical session set (the paired
  Wilcoxon needs this). Cannot occur in the full 15-train-subject cohort (all five indices
  present); matters only for the 6-subject smoke and sparse boundaries. Used by §2.4.
- **O3 — K=1 baseline protocol-guard path: config-level `protocol_freeze_guard(config)`,
  `active=None`.** The baseline uses none of the WST search axes, so there is nothing for a
  per-fit `active` record to validate; frozen M6 guard code stays untouched. Used by §2.4.

---

**In scope:**

- **`src/dehyd/eval/harness.py`** — one generic nested-LOSO engine (sklearn path) plus the
  Exp A staged-search driver. Fit-on-train-only at both CV levels; consumes folds **only**
  from `eval/splits.py`; routes every tie-break through `eval/selection.py::select_candidate`;
  calls `protocol_freeze_guard(config, active=...)` immediately before every fit and every
  result write; emits a per-fold **fit-audit artifact** (every fitted quantity + the subject
  set it was estimated from). The **tuned-ε** WST branch — the one genuinely fitted WST
  quantity — is computed fold-locally, train-only, at both CV levels.
- **`src/dehyd/eval/metrics.py`** — `subject_balanced_mae`, session-level MAE/RMSE, pooled and
  per-subject predicted-vs-actual r, and the subject-cluster bootstrap (own BCa
  implementation, `B=10000`, percentile fallback recorded) with the metric-type-aware
  seed-collapse rules, plus the Wilcoxon signed-rank baseline comparison.
- **`src/dehyd/models/`** (new package) — `regressors.py` (the five classical families +
  grid enumeration), `baselines.py` (the session-index-only baseline), and `torch_fit.py`
  (the deterministic torch trainer + `TinyMLP` that makes **T18** real).
- **`src/dehyd/features/store.py`** + **`experiments/extract_features.py`** — a persistent
  per-session feature store for both bands (the 77 GHz session feature vectors do not exist
  yet), with fail-closed fingerprint validation.
- **The `tests/reference_procedure.py` rebind** — rewritten from a self-contained sklearn
  reference into a thin adapter over `harness.py`, so the frozen `test_no_leakage.py` exercises
  the real harness. **T18 activated** (Step 0 item 1).
- **Extending `experiments/run_regression.py`** (currently M2-scope) into the Exp A entrypoint:
  staged search, both bands, the owner-gated full run, and regenerable outputs (metrics JSON,
  predicted-vs-actual scatter, predictions CSV, selection-frequency table, fit-audit JSON,
  provenance).
- **The IBEX 77 GHz store job array** (`scripts/ibex/extract77.sbatch` + a submit wrapper) and
  the **git-provenance fix** (`git.commit = None` on compute nodes).
- **Small additive library changes** required by the above: `wst.apply_order_log` gains an
  `epsilon_by_order` parameter (mirroring `apply_order_log_77`); `pooling.pool_stats_batch`;
  `extract_session_variants{,_77}` gain `keep_raw=False`.
- **Tests** for every new module + the store round-trip/equivalence + the frozen-suite rebind.
- **Journal**: HISTORY.md entries per resolved step; SECOND_CHAPTER.md §6 (Exp A) at close.

**Explicitly out of scope (deferred to M8+):**

- **Experiments B, C, D, E, F, G, H.** M7 builds the harness they will all consume and runs
  **only Exp A**. Exp B (clock-decoupling), Exp C/D (ordinal + baselines), and the fusion /
  interpretability / confound / full-stats / figures milestones come later, on this harness.
- **Any torch Exp A / DL baseline *result*.** The torch fit path enters `harness.py` at M7
  **only** to make T18 green on a synthetic fixture; no DL baseline is trained or reported here.
- **Re-opening any M6-frozen decision** (search spaces, budget K, grids, the staged-search
  order, the tuned-ε mechanics, the baseline/stats specs). M7 *consumes* them.
- **The `protocol_freeze.py` shared-helper refactor** flagged at M6 §6 — stays deferred.

**The milestone-7 invariant, protected above all:**

> **No real-subject outer-fold score is made visible before the owner checkpoint.** The harness
> is verified entirely on **synthetic fixtures** and the frozen leakage suite; the feature
> stores are validated structurally; the functional end-to-end check of the real staged search
> is the **synthetic-store** test (§2.3/§3), where "performance" is meaningless. The real-data
> smoke runs (§2.8) run in a **mechanism-only mode**: they exercise the pipeline on a fixed
> 6-subject real subset and assert *mechanism* (loops ran, no subject in two roles, outputs are
> structurally present) but **suppress every performance value** — no MAE/RMSE/r is printed or
> written, no scatter is rendered — precisely because those six subjects are members of the
> final cohort and inspecting their LOSO scores would begin to spend the freeze. **The freeze
> is recorded as spent at the first moment a real-subject outer score is made visible**, which
> by construction is the owner-cleared full run, not the smoke. Every hard invariant from CLAUDE.md
> (LOSO at subject level; fit-on-train-only; no test-set tuning; continuous primary target;
> session-level headline; numpy backs all reported features; folds only from `splits.py`)
> holds at all times, and `tests/test_no_leakage.py` stays byte-for-byte frozen except for the
> single pre-registered T18 hunk (Step 0 item 1).

**Not reopened at M7** (fixed at M6, would each need a prior authoritative amendment):
the two 10 GHz gates and their whitelist; the log axis's three unconditional branches (A-M6-1);
`budget_k = 12`; `tuned_eps_k = 0.1`; every model grid; the staged-search order (Stage 1
feature axes at the fixed ridge anchor α=1.0 → Stage 2 family × grid); the `simplicity_rank`
tie-break ordering; the full statistical protocol (`B=10000`, BCa/percentile, subject
resample unit, the seed-collapse rules); the session-index-only baseline as Exp A's
pre-registered primary comparison; the 77 GHz fixed chain (reduction/channel/gate) and Doppler
tilings.

**Ground rules.** Implementation runs on a new **`v1_milestone_7`** branch, created from
`v1_milestone_6` (`357f734`) when implementation starts — this plan is authored on the current
branch and committed only when the owner asks. HISTORY.md is written continuously, newest-first,
one entry per resolved step (failures kept). Superseded material → `archive/` with a note. The
build is **test-first**: each step's tests are green before the next begins, and the frozen
leakage suite is the gate on the rebind.

---

## §1 Build sequence — exact order and why

| # | Step | Why this position |
|---|------|-------------------|
| 0 | **Add `matplotlib` to the pinned environment** (`pyproject.toml` dependency + `uv.lock` refresh; it must not disturb the `scipy <1.17` pin — matplotlib has no scipy dependency); add it to `provenance.TRACKED_PACKAGES` | The Exp A scatter (§2.8) needs a plotting backend and none is pinned today; land the env change first so every later step runs in the final environment |
| 1 | **Provenance git fix** (`provenance._git_info` gains a `DEHYD_GIT_COMMIT/_BRANCH/_DIRTY` env fallback) + the IBEX **submit wrapper** that captures them; plus the stale-"milestone 6" string cleanups in `run_regression.py` and `extraction.py` | Tiny, independent, and a **prerequisite for any IBEX submission** (step 11). `safe.directory` was tried at M5 and does not take on the compute nodes; submit-time capture is the robust fix |
| 2 | **`eval/metrics.py`** + tests | Pure functions, zero project deps; `subject_balanced_mae` must exist before both the harness and the frozen-suite rebind (which imports it) |
| 3 | **`models/regressors.py`** + tests | The engine's estimator source; grid enumeration gives the literal `budget_k` count check |
| 4 | **`harness.py` core engine** (Dataset, records, `run_candidate_stage`, `select_stage_winner`, `final_refit`, `run_nested_candidates`, `fit_audit`) + harness unit tests on synthetic data | The heart. Verified against T10–T19-style properties **before** the frozen file is touched; this is where `test_selection.py`'s deferred behavioural claim (fit on inner-train, score on inner-val, over real GroupKFold folds) is discharged |
| 5 | **`reference_procedure.py` rebind** (thin adapter) → the frozen `test_no_leakage.py` T1–T17 + T19 go green through real engine code | Only safe once step 4's properties hold; the frozen suite is the acceptance gate on the rebind |
| 6 | **`models/torch_fit.py`** + torch unit tests, then the **T18 activation** (the one sanctioned frozen-file edit, A-M7-1) + the HISTORY diff audit | T18 must be green **before any torch result is ever reported**; the activation is pre-registered (Step 0 item 1) |
| 7 | **`wst.apply_order_log` ε extension** + `pooling.pool_stats_batch` + `harness.tuned_epsilons`, with bit-equivalence tests | Needed by both the store consumer and the Exp A provider; pure math, independent of I/O |
| 8 | **`features/store.py`** + `keep_raw` extraction params + **`experiments/extract_features.py`** + store tests (round-trip, store-vs-direct equivalence, staleness rejection) + **`scripts/ibex/extract77.sbatch`** | The I/O layer lands after the math it persists is pinned |
| 9 | **`models/baselines.py`** + tests | Small; needed by the metrics/report step |
| 10 | **Exp A driver** (`run_exp_a_fold`/`run_exp_a`, `StoreBackedFeatures`, guard wiring) + **`run_regression.py` CLI** + a synthetic-store end-to-end test + the guard-before-write ordering test | Composes everything; the synthetic-store test runs the full staged search **without** private data |
| 10.5 | **Owner-triggered clean M7 implementation commit** (all code green, working tree clean) on `v1_milestone_7`, before any producer runs | Reproducible provenance: the stores must be built from a *committed* revision. Without this, `DEHYD_GIT_COMMIT` would record the M6 base with `dirty=true` while uncommitted M7 code actually produced the features (C7). Owner-triggered per the commit ground rule |
| 11 | **Build the stores** (from the committed revision): 10 GHz locally (~25–50 min); 77 GHz on IBEX via the wrapper, rsync back, `extract_features.py --validate` | Compute only after all code is green **and committed** (step 10.5) |
| 12 | **Mechanism-only smoke runs**, both bands (`--subset 6subjects`): assert loops ran, train/val/test disjointness, outputs structurally present — **performance values suppressed** (C9). **STOP — owner checkpoint.** | Step 0 item 3 + the milestone invariant: the six smoke subjects are cohort members, so their outer scores are blinded until the owner clears the freeze |
| 13 | **On owner go-ahead only:** full-cohort Exp A (`--full-cohort`), both bands — the first visible real outer scores (spends the freeze); then metrics/figures/stats, journal + SECOND_CHAPTER §6 from the **full** results | A separate, owner-triggered invocation; the milestone stays *paused* (not complete) until this runs (C8) |

---

## §2 Per-file specifications

Format per file: **Responsibility** · **Public API / content** · **Frozen values** ·
**Acceptance criteria**. Signatures below are the contract; the exact bodies are written at
implementation time. Test-group IDs (e.g. `T-M7-harness`) map to §3.

### 2.1 `src/dehyd/eval/metrics.py`

**Responsibility.** All scoring and uncertainty for Exp A, as pure functions over
already-computed predictions. No fitting, no fold construction, no I/O.

**Public API.**
```python
def subject_balanced_mae(subjects, y_true, y_pred) -> float
    # mean over subjects of each subject's mean |error| (NOT pooled). The selection metric
    # AND the per-subject headline. Pinned by the frozen T17 fixture to 5.5.
def session_rmse(y_true, y_pred) -> float
def pooled_pearson_r(y_true, y_pred) -> float          # NaN on zero-variance input (caller skips-and-counts)
def per_subject_pearson_r(subjects, y_true, y_pred, *, min_sessions=3) -> dict[int, float]

@dataclass(frozen=True)
class BootstrapCI:
    point: float; low: float; high: float; method: str        # "bca" | "percentile"
    n_eval: int; n_skipped: int; unreliable: bool             # unreliable = >5% resamples skipped

def subject_cluster_bootstrap(
    per_subject_values,            # additive metrics: one value per subject (seeds pre-averaged)
    *, b=10000, level=0.95, rng_seed, method="bca") -> BootstrapCI
def subject_cluster_bootstrap_pooled(
    subjects, y_true_by_seed, y_pred_by_seed,  # pooled metrics: recompute per seed within each resample, then average
    metric_fn, *, b=10000, level=0.95, rng_seed, method="bca") -> BootstrapCI
def wilcoxon_signed_rank(differences) -> tuple[float, float]   # (statistic, p); scipy.stats.wilcoxon
def mean_difference_ci(per_subject_differences, *, b, level, rng_seed) -> BootstrapCI
```

**Frozen values** (all from `StatsConfig`, not re-decided here): `bootstrap_b = 10000`;
`ci_method = "bca"`, `ci_fallback = "percentile"` (recorded when it fires);
`confidence_level = 0.95`; `resample_unit = "subject"`; `undefined_metric_skip_threshold_pct
= 5.0`; `per_subject_pearson_r_min_sessions = 3`; the two `seed_collapse_*` rules (additive
metrics average each subject's 5 per-seed values *before* resampling; pooled metrics recompute
per seed *within* each resample then average). Own BCa implementation (no
`scipy.stats.bootstrap` — scipy pinned `<1.17`).

**Acceptance** (`T-M7-metrics`). `subject_balanced_mae` returns exactly 5.5 on the frozen
fixture (`subjects=[1,1,1,1,1,2,2]`, `y_pred=[1,1,1,1,1,10,10]`, `y_true=0`) and **not** 25/7;
RMSE and pooled r hand-checked; per-subject r honours the ≥3-session rule; the bootstrap is
bit-deterministic under a fixed `rng_seed`; BCa matches a hand-computed small case; the
percentile fallback fires and is recorded on a degenerate (zero-jackknife-variance) input;
skip-and-count works and the >5% flag flips `unreliable`; both seed-collapse paths tested;
Wilcoxon wrapper matches scipy on a known vector.

### 2.2 `src/dehyd/models/regressors.py`

**Responsibility.** Construct and enumerate the five classical families with a **single**
definition of each family's estimator and hyperparameter grid, so the harness never inlines a
model. Deterministic given a seed.

**Public API.**
```python
def build_estimator(family: str, params: dict, *, seed: int)
    # ridge  -> Ridge(alpha=…, solver="cholesky")          (deterministic; solver pinned)
    # svr    -> SVR(C=…, epsilon=…)                         (deterministic)
    # knn    -> KNeighborsRegressor(n_neighbors=…)          (deterministic)
    # rf     -> RandomForestRegressor(…, random_state=seed) (seed-sensitive)
    # gbm    -> GradientBoostingRegressor(…, random_state=seed) (seed-sensitive)
    # each wrapped in a Pipeline([("scaler", StandardScaler()), ("model", …)]) fit inside the fold

def enumerate_grid(family: str, grid: ModelGridConfig) -> list[dict]   # the ≤ budget_k param dicts
SEED_SENSITIVE = frozenset({"rf", "gbm"})
SIMPLICITY_RANK = selection.SIMPLICITY_RANK    # re-exported, single source
```

**Frozen values.** Grids and their combination counts come verbatim from `ModelGridConfig`
(ridge 8, svr 4×3=12, rf 2×3=6, gbm 2×2×2=8, knn 7). Ridge `solver="cholesky"` (determinism,
matching the frozen reference). The `StandardScaler` inside every pipeline is the fit-on-train
transform the fit-audit records as `quantity="scaler"`.

**Acceptance** (`T-M7-regressors`). Each family builds and fits on a tiny synthetic set; ridge
is bit-deterministic; rf/gbm are bit-deterministic for a fixed seed and *differ* across seeds;
`enumerate_grid` returns exactly the documented counts and each family's count is `≤ budget_k`.

### 2.3 `src/dehyd/eval/harness.py`

**Responsibility.** The single fit-on-train-only nested-LOSO engine. One generic candidate
engine serves (a) the Exp A staged search over store-backed features, (b) the frozen-suite
reference shim (scaler + ridge over `ALPHA_GRID`), and (c) — via `torch_fit` — the T18 path.
No other module constructs folds; no other module defines the tie-break.

**Public API / content.**
```python
@dataclass
class Dataset:                     # the frozen ctor shape (shim re-exports this)
    subjects: np.ndarray; features: np.ndarray; targets: np.ndarray
    def rows_for(self, subject_set) -> np.ndarray      # boolean mask via np.isin
    def subject_ids(self) -> list[int]                 # sorted(set(...))

@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    model_params: tuple[tuple[str, float | int | None], ...]     # hashable, sorted
    feature_key: FeatureKey | None      # None => the Dataset's fixed features (shim / torch fixture)
    active: tuple[tuple[str, object], ...] | None   # the protocol_freeze_guard `active` record, or None

@dataclass
class FitRecord:    quantity: str; role: str; subjects: frozenset[int]; params: dict[str, np.ndarray]
@dataclass
class InnerResult:  inner_train: frozenset[int]; inner_val: frozenset[int]; candidate_id: str
                    score: float; val_predictions: dict[int, np.ndarray]; fits: list[FitRecord]
@dataclass
class FoldResult:   test_subject: int; train_subjects: frozenset[int]
                    selected: Candidate | None
                    inner_scores: np.ndarray                 # (n_candidates, n_inner_folds)
                    inner_results: list[InnerResult]          # FLAT, fold-major / candidate-minor
                    final_fits: list[FitRecord]
                    train_predictions: np.ndarray; test_predictions: np.ndarray; test_score: float
                    seed_outcomes: list[SeedOutcome]          # len 1 for deterministic families

def run_candidate_stage(data_for, candidates, fold, *, seeds, before_fit=None) -> StageOutcome
def select_stage_winner(stage: StageOutcome) -> Candidate    # builds CandidateScore, calls selection.select_candidate
def final_refit(data_for, candidate, fold, *, seeds, before_fit) -> list[SeedOutcome]
def run_nested_candidates(dataset, candidates, *, seeds=(0,), before_fit=None, **split_kwargs) -> list[FoldResult]
def fit_audit(results) -> list[dict]                          # duck-typed over engine results, shim views, torch results
def tuned_epsilons(prelog_by_session, train_subjects, *, k=0.1, fallback=1e-6) -> dict[int, float]
def run_exp_a(config, band, store, gt, manifest_qc) -> ExpARunResult
```

**Key mechanics (each has an acceptance test):**
- **Execution vs. assembly order.** Candidates are *executed* candidate-major (so the tuned-ε
  raw-tensor cache is reused within a candidate), but `inner_results` and the
  `(n_candidates, n_inner_folds)` `inner_scores` matrix are *assembled* **fold-major /
  candidate-minor** — the flat ordering the frozen `zip(strict=True)` comparison requires. All
  numeric work runs under `threadpoolctl.threadpool_limits(1)`.
- **Selection.** `select_stage_winner` builds `CandidateScore(candidate_id,
  inner_val_mae = mean-over-folds of the seed-mean MAE, simplicity_rank = SIMPLICITY_RANK[family],
  feature_dimension = X.shape[1], inner_fold_variance = np.std(per_fold_seed_mean, ddof=0))`
  and calls `eval.selection.select_candidate` — **never** an inline tie-break. `ddof=0` is a
  new frozen choice (M6 said only "lower inner-fold variance"); recorded as **A-M7-2** (§6).
- **`before_fit` guard.** Invoked immediately before **every** estimator `.fit()`. Exp A passes
  `lambda c: protocol_freeze_guard(config, active=dict(c.active))`, with `active` built from the
  same enumeration loop that produced the candidate. The shim/T18 pass `before_fit=None`.
  **Completeness is enforced fail-closed** (C5): `protocol_freeze_guard`'s `_check_active`
  validates only the keys *present* in the record, so an under-populated `active` (e.g.
  `{"band": "10ghz"}`) would pass. Before the first fit the harness therefore asserts the
  `active` record carries the **exact required band-specific key set** — 10 GHz: `{band,
  reduction, channel, tiling, log_branch, range_gate_m, model_family}`; 77 GHz: `{band, reduction,
  channel, gate_m, tiling, log_branch, model_family}` — and raises if any is missing or extra.
  (Implemented harness-side so the frozen M6 guard code stays untouched; a shared
  `REQUIRED_ACTIVE_KEYS` constant documents both sets.)
- **Fold viability** (C6, C21). Non-evaluability is decided by an **explicit, enumerated,
  pre-fit predicate set** — *not* by catching estimator exceptions (a broad try/except would
  silently recast a real bug, a malformed feature matrix, or a numerical failure as
  "non-evaluable" and could even drop an outer fold). The enumerated conditions are, at minimum,
  KNN `n_neighbors > n_inner_train_rows` (the frozen grid reaches `k=15`, while a boundary inner
  fit can train on as few as 2 subjects / a handful of sessions); each is checked **before**
  fitting, marks the candidate **non-evaluable for that fold** with a **recorded reason code**
  (in the selection table), assigns a **non-finite score**, and never clips or retunes the frozen
  grid. **Any unexpected exception during `.fit()`/`.predict()` propagates and fails the run
  loudly** — it is never swallowed. `select_candidate` already filters non-finite MAE, so a
  candidate non-evaluable in *any* inner fold of an outer fold drops out of that fold's
  selection; if **every** candidate in a stage is non-evaluable, `select_candidate` raises
  `SelectionError` (the fold contributes no result), consistent with the Exp-C fold-viability
  doctrine `selection.py` cites. Adding any exception class to the non-evaluable set later
  requires enumerating it narrowly with its own test and reason code.
- **Per-family auditable fitted state** (C11, C15). **Every `FitRecord.params` value is an
  `np.ndarray`** (the frozen comparator calls `.tobytes()`), and each captures the complete
  *prediction-determining* state of the selected family, per seed: `scaler` → `{mean_, scale_}`;
  `ridge` → `{coef_, intercept_}`; `svr` → `{support_vectors_, dual_coef_, intercept_,
  _gamma}` (includes `support_vectors_`, which predictions depend on); `knn` → `{_fit_X, _y}`
  (KNN stores its training set — invariant under held-out mutation by construction);
  `rf`/`gbm` → `{ensemble_digest}` where `ensemble_digest` is a **`uint8` array**
  (`np.frombuffer(sha256(canonical_bytes).digest(), np.uint8)`) over a canonical serialization of
  the **complete prediction-determining state** (C20): every fitted tree's node arrays **plus**
  the fitted initializer (`gbm.init_`'s constant prediction — it contributes to every output)
  **plus** the exact combining parameters needed to interpret the trees (`learning_rate`,
  `n_estimators`, `max_depth`; RF has no `init_`/`learning_rate` but the same fields are hashed
  for a uniform record). An `np.ndarray`, hence `.tobytes()`-comparable; the full trees are too
  large to store verbatim but are fully bound by the digest. A test asserts that perturbing
  **any** component — a tree node value, `init_`, or `learning_rate` — changes `params_repr`.
  **In-memory vs. serialized** (C15): the in-memory `FitRecord.params` holds these arrays for
  the bit-identity mutation tests; the emitted `fit_audit_{band}.json` (§2.8) is a **separate,
  JSON-safe** artifact carrying, per fitted quantity, the frozen audit keys `{test_subject,
  quantity, role, fitted_on, inner_val}` plus a `params_repr` = **hex sha256 of each param
  array** (and, for `tuned_epsilon`, the literal `[ε1, ε2]` values, since D2/§2.8 promise the ε
  values in JSON). The §3 tests assert bit-identity on the in-memory records **and** validate
  the serialized JSON artifact (stable hashes across two unmutated runs, ε values present).
- **Staged Exp A** (`run_exp_a_fold`): Stage 1 enumerates the feature axes (10 GHz:
  reduction{A,B}×channel{mag,iq}×tiling{T1,T2,T3}×log{off,frozen,tuned}×gate{(1,2),(0.9,3)} =
  **72**; 77 GHz: 1×1×3×3×1 = **9**) each fit with the fixed ridge anchor (α=1.0), scored on
  inner-val; `select_stage_winner` carries the winner to Stage 2 (families × grids, each
  `≤ budget_k`). Winner refit per seed on all outer-training subjects.
- **Seeds.** `SEED_SENSITIVE = {rf, gbm}` fit once per seed at both CV levels; deterministic
  families fit once (seed-mean of identical values is the value). Outer scoring keeps each
  seed's prediction **separate** (reported mean±sd, never ensembled).
- **Tuned-ε.** `tuned_epsilons` is called **inside** `data_for(candidate, train_subjects)` from
  the fit's own training subjects — structurally train-only at both CV levels. ε for order
  o∈{1,2} is `k · scale_o`, `scale_o` = median-over-training-subjects of (mean over that
  subject's eligible training sessions of the stored per-session `_prelog_scale[o]`);
  non-finite/non-positive → `fallback = 1e-6`. Every tuned-ε fit appends
  `FitRecord("tuned_epsilon", role, train_set, {"epsilon": np.array([ε1, ε2])})` into the audit.
- **Fit-audit.** `fit_audit` emits, per fitted quantity, `{test_subject, quantity, role,
  fitted_on, inner_val}` with `fitted_on ⊆ fold.train_subjects`, `test_subject ∉ fitted_on`,
  and `inner_val` disjoint from `fitted_on` for inner-train fits.

**Frozen values.** Folds only from `nested_loso_splits`; tie-break only from `select_candidate`;
`tuned_eps_k = 0.1`, `fallback = 1e-6`; Ridge `solver="cholesky"`.

**Acceptance** (`T-M7-harness`, plus the frozen suite via §2.7). Flat-assembly order pinned;
score-matrix shape `(n_candidates, n_inner)`; two unmutated runs bit-identical; non-selectable
outer folds excluded; `before_fit` called exactly once per fit (counter) **and rejects an
incomplete `active` record — one negative test per omitted required key** (C5); the
fold-viability rule marks a `k>rows` KNN candidate non-evaluable and still selects among the
rest, and raises when all are non-evaluable, on a sparse-session boundary fixture (C6); per-seed
outer outcomes kept separate; audit covers **every** fitted quantity incl. `tuned_epsilon`; a
tuned-ε mutation micro-test (mutating an inner-val subject leaves that fold's ε and its fits
bit-identical — the T16 analog at the ε level); staged winner carry-forward correct.
**End-to-end headline-path mutation property** (C1): an eligibility-preserving
**synthetic-store** test drives the real two-stage `run_exp_a_fold` (StoreBackedFeatures +
Stage-1/Stage-2 selection + fold-local tuned-ε reconstruction from stored `raw`/`prelog`) and
mutates the outer-test subject's stored `vec`, `raw`, `prelog`, and target, asserting that the
Stage-1 and Stage-2 winners, all inner scores, the tuned epsilons, every fitted transform/model
state (per §2.3 per-family capture), and the training predictions are **bit-identical** — only
the held-out subject's prediction/score may move. This closes the gap that the frozen suite
exercises only the `feature_key=None` Ridge shim, so the actual reported path is proven leak-free.

### 2.4 `src/dehyd/models/baselines.py`

**Responsibility.** The Exp A pre-registered primary comparison: predict Δm% from time of day
(session index) alone. `K = 1`, fit on outer-training subjects only, inside the same outer
folds; audited like any fit.

**Public API.**
```python
def fit_session_index_baseline(subjects, session_idx, targets, train_subjects) -> BaselineFitOutcome
    # {session_idx: mean Δm% over training rows at that index} for indices present in training.
    # ABSENT-index rule (O2, owner-decided 2026-07-25): fall back to the GLOBAL training-fold
    # mean Δm%, so every test session stays scored. The BaselineFitOutcome carries a FitRecord
    # so the audit sees it (C12).
def predict_session_index(model, session_idx) -> np.ndarray

@dataclass
class BaselineFitOutcome:
    model: dict; fit_record: FitRecord    # quantity="session_index_means", role, subjects, params
```

**Guard path** (O3, C12 — owner-decided 2026-07-25: **config-level**). The baseline uses **none**
of the WST search axes, and `protocol_freeze_guard`'s `active` whitelist has no
`session-index-only` `model_family` — so it is guarded at the **config level**
(`protocol_freeze_guard(config)`, `active=None`) before the fit, never with a per-fit WST `active`
record. Frozen M6 guard code stays untouched.

**Frozen values.** No hyperparameters (`K = 1`). Absent-index rule = **global training-fold mean**
(O2, owner-decided).

**Acceptance** (`T-M7-baselines`). Per-index means correct; mutating a *test* subject changes
nothing fitted (train-only); the fit emits a `FitRecord` and the audit records it; a
guard-before-baseline-fit **ordering** test (the `test_protocol_freeze.py` pattern) covers the
actual baseline fit. The absent-index test asserts the **global-training-mean fallback** (O2,
owner-decided): a fold missing a time index predicts the global train mean there, and every test
session stays scored.

### 2.5 `src/dehyd/models/torch_fit.py`

**Responsibility.** A deterministic, single-threaded CPU torch training path — the T18 target.
Structurally distinct from the sklearn engine (inner folds select an **epoch budget**, not a
candidate), so it lives in its own module and shares only the harness's contracts (folds,
`Dataset`, `subject_balanced_mae`, `FitRecord`, `fit_audit`).

**Public API / content.**
```python
class TinyMLP(nn.Module)                 # small fixed arch (e.g. 6->8->1), seeded-Generator init
@dataclass(frozen=True)
class TorchFitSpec:  max_epochs; patience; min_delta; lr; weight_decay; batch_policy
def fit_torch(x_tr, y_tr, subj_tr, x_val, y_val, subj_val, spec, *, seed) -> TorchFitOutcome
    # torch.set_num_threads(1); use_deterministic_algorithms(True); no shuffle.
    # train-only input-norm stats            -> FitRecord "input_norm"
    # train-only inverse session-count weights-> FitRecord "sampler_weights"
    # TRUE early stopping on inner-val subject_balanced_mae (frozen patience/min_delta);
    # records per-epoch train state + inner-val metric; best-so-far checkpoint at stop
    #                                         -> FitRecord "mlp_state"
def run_torch_nested(data, spec, *, seed, **split_kwargs) -> list[TorchFoldResult]
    # inner folds early-stop -> epoch_budget = int(median of the early-stop-selected epochs)
    # outer refit on all outer-training subjects at exactly that budget, NO early stop
```

**Training design — the frozen early-stopping algorithm** (C19). `fit_torch` uses **true
patience/min-delta early stopping** monitored on inner-val `subject_balanced_mae`, exactly as
`implementation_plan.md`'s early-stopped-NN refit rule and M6's frozen
`early_stopping_patience`/`early_stopping_min_delta` specify — **not** a run-to-fixed-max
substitute (that would diverge from the frozen protocol and, worse, make T18 protect a
*different* algorithm than the one Exp D's DL baselines later run). The training weights at each
epoch are a pure function of the *train* data and seed (the optimizer never sees val); the val
metric only drives the **stop time** and the best-so-far checkpoint. Under an inner-val mutation
the two runs may stop at different epochs, so the invariant is asserted over the **common prefix**
of executed epochs (see §2.5 mutation contracts) — the leakage-relevant claim (training never
touched val) is exactly "identical train state over every epoch both runs ran."

**Frozen values.** Single-threaded deterministic CPU; **true early stopping** with the frozen
`patience`/`min_delta`; median-of-inner-folds epoch budget; no shuffle/nondeterministic kernels.
T18's spec uses a small `max_epochs` (≈20–30) so the leg stays fast (<90 s target); the *shared*
torch fit path is the same algorithm Exp D's `BaselineConfig` DL constants map onto in a later
milestone (only the constants differ), so T18 protects the real path.

**Acceptance** (`T-M7-torch`). Two runs bit-identical (state_dicts, predictions); **true early
stopping** halts at the correct epoch under the frozen patience/min-delta on a crafted val curve;
refit uses exactly the median budget; normalization and sampler weights are train-only
(mutation); different seeds → different weights; runtime budget respected.

**The two mutation contracts must be stated separately** (C13), because they are genuinely
different:
- **Outer-test mutation** (T18 proper, the T11–T15 analog): the held-out test subject is in
  *neither* train nor inner-val, so **everything** determined before scoring is invariant —
  epoch budget, input-norm, sampler weights, the inner-val-selected epoch/checkpoint per fold,
  every `state_dict` tensor, and training predictions; only the held-out prediction/score moves.
- **Inner-validation mutation** (a *separate* test, the T16 analog): mutating a subject that
  serves as inner-validation must leave the **training-only** quantities bit-identical — input
  normalization and sampler weights (computed from inner-train, so **fully** invariant) and the
  **per-epoch training trajectory/gradients over the common prefix of executed epochs** (C19).
  Because true early stopping is monitored on val, the two runs may **stop at different epochs**;
  the invariant is therefore "identical train state at every epoch that *both* runs executed"
  (proving the optimizer never touched val), while the **stop time and best-so-far checkpoint may
  legitimately differ** and the mutated subject's own val predictions must move. The test compares
  the common prefix and asserts the val-driven movement **without** demanding an equal-length
  trajectory or an invariant checkpoint (either would be wrong).

### 2.6 `src/dehyd/features/store.py` + `experiments/extract_features.py`

**Responsibility.** A persistent, fingerprinted per-session feature store so Exp A does not
re-scatter per fold — and so the 77 GHz session vectors (which do not exist yet) exist. The
tuned-ε branch cannot be precomputed as session vectors, so the store also holds the **raw
pre-log per-frame scattering tensors** the fold-local ε consumes.

**Store layout.** One `.npz` + one `.fingerprint.json` per (band, subject, session) under
`results/features/{10ghz,77ghz}/`.
- **10 GHz** keys, per gate g∈{0,1}, reduction R∈{A,B}, channel C∈{mag,iq}, tiling t∈{0,1,2}:
  `vec/g{g}/{R}/{C}/t{t}/{off,frozen}` (pooled session vectors, log off / on_frozen_eps),
  `raw/g{g}/{R}/{C}/t{t}` (pre-log tensor `[N,C,P,t]`, the tuned-ε input),
  `prelog/g{g}/{R}/{C}/t{t}` (`(3,)`), `order/t{t}` (`(P,)`).
- **77 GHz** keys, per tiling t: `vec/t{t}/{off,frozen}/fus_{mean,median}`, `raw/t{t}/fus_mean`
  (raw stored for the frozen primary fusion only — fusion is not a search axis),
  `prelog/t{t}/fus_{mean,median}`, `order/t{t}`.
- **Fingerprint sidecar**: `{git, spec_hash, frame_selection:
  "qc_pass_frames_of_eligible_sessions", frame_ids_sha256, n_frames, raw_sha256,
  qc_config_hash, session_eligible: bool, store_version: 1}`. `frame_ids_sha256` is a canonical
  hash of the **ordered QC-pass frame indices** actually selected for this session (C4) — so a
  changed/stale QC manifest that selects a *different* frame set with the same raw-file hash and
  the same count no longer validates (a count check alone cannot catch a substitution).
  `spec_hash` = sha256 of sorted-JSON(band canonical preprocess/wst config + gate whitelist +
  a pooling-contract version string); `qc_config_hash` = sha256 of the resolved `qc`/`qc77`
  config plus the eligibility contract (`min_frame_fraction`), binding the store to the exact
  QC decision that produced it. 77 GHz reuses `axis_spec_hash` as a `spec_hash` component.
  `validate_store` compares all of these against what THIS config/QC/code would produce.

**Public API.**
```python
def write_session_store(band, subject, session, npz_dict, fingerprint, out_dir) -> Path
def read_session_store(band, subject, session, store_dir) -> LazyStore     # lazy per-key npz access
def validate_store(band, store_dir, config, expected_sessions, *, analysis_commit) -> None  # fail-closed
```

**Producer changes** (additive, default-compatible): `extract_session_variants` /
`extract_session_variants_77` gain `keep_raw: bool = False`; their result dataclasses gain a
`raw: dict | None = None` field. So `store.py` persists exactly what the M4/M5-tested extraction
pass computes — no parallel extraction loop. `extract_features.py` modes: **default** = local
loop over the band's eligible sessions (10 GHz reuses `run_wst.py`'s groupby/eligible-frames
pattern); **`--subject/--session`** = one IBEX shard; **`--validate`** = cohort completeness +
fingerprint check + `store_manifest.json`.

**Clean-commit binding, both producers** (C16). The dirty-tree refusal is **not** IBEX-only: the
**local** `extract_features.py` producer also **refuses to build from a dirty tree** (the same
hard check as the submit wrapper), so a `dirty=true` 10 GHz sidecar can never be written.
Every sidecar records the building `git.commit`, and `validate_store` requires **the store's
commit to equal the analysis run's commit** (`analysis_commit`) — so a store built from one
revision cannot silently back an analysis run at another. Tested with a **local** dirty-tree
rejection **and** a store/analysis commit-mismatch rejection, alongside the wrapper test (§2.9).

**Harness consumption.** `StoreBackedFeatures` (plain class in `harness.py`): canonical row
order = sorted `(subject, session_idx)` over eligible sessions. `data_for` reads one `vec/…`
key per session for the data-independent branches (matrices cached per feature key); the
`on_tuned_eps` branch computes ε from stored `prelog` scales, then streams session raw tensors
through the extended `apply_order_log(..., epsilon_by_order)` → `pool_stats_batch` →
`aggregate_session`, caching one raw set per candidate (candidate-major execution; peak
≈600 MB).

**Frozen values.** `store_version = 1`; frame selection = QC-pass frames of eligible sessions
(the M4/M5 population); numpy backend only.

**Acceptance** (`T-M7-store`). npz round-trip bit-identical; lazy key read; `keep_raw=True`
leaves `vectors`/`prelog_scale` bit-identical (additive-param safety); **store-vs-direct
equivalence** — reconstructing the frozen-ε session vector from the stored raw tensor equals
the stored `vec/.../frozen` byte-for-byte; `pool_stats_batch` == the looped `pool_stats`
(`.tobytes()`); `apply_order_log(epsilon_by_order={1:1e-6,2:1e-6})` ≡ the frozen path
byte-for-byte; a stale fingerprint (changed spec_hash **or** `qc_config_hash`) is rejected; a
**same-count frame-membership change fails closed** (swap one selected frame id for another,
keeping `n_frames` and `raw_sha256` fixed — `frame_ids_sha256` catches it) (C4); a missing
eligible session fails closed.

### 2.7 `tests/reference_procedure.py` rebind + `tests/test_no_leakage.py` (T18)

**Responsibility.** Rebind the frozen leakage suite onto the real harness without editing the
frozen test (except the pre-registered T18 hunk). `reference_procedure.py` becomes a **thin
adapter** — zero fitting code of its own.

**The adapter.**
```python
from dehyd.eval import harness
from dehyd.eval.harness import Dataset            # re-export: the frozen ctor shape
from dehyd.eval.metrics import subject_balanced_mae   # re-export: the 5.5 pin lives on
ALPHA_GRID = (0.1, 1.0, 10.0)

@dataclass
class FoldResult:    # the 9-field VIEW; every field IS the engine object (arrays/lists by reference)
    test_subject; train_subjects; selected_alpha; inner_scores; inner_results
    final_fits; train_predictions; test_predictions; test_score

def run_nested_loso(data, **split_kwargs):
    candidates = [harness.Candidate(f"ridge_a{a}", "ridge", (("alpha", a),), None, None)
                  for a in ALPHA_GRID]
    return [_view(r) for r in harness.run_nested_candidates(data, candidates, **split_kwargs)]

def fit_audit(results):    return harness.fit_audit(results)     # duck-typed over the views
```
`_view` maps `selected_alpha = dict(r.selected.model_params)["alpha"]` and passes
`inner_scores`, `inner_results`, `final_fits`, and the prediction arrays through **by
reference**, so every `.tobytes()` bit-identity assertion in the frozen suite exercises engine
output. The reference's old max-alpha tie-break (not pinned by the frozen tests — only
`ALPHA_GRID` membership and bit-identity are) is replaced by `select_candidate`'s
stable-first-on-tie, so ties route through the one frozen tie-break everywhere.

**T18 activation (A-M7-1).** In `test_no_leakage.py`, remove the `@pytest.mark.skip` decorator
and replace the placeholder body with the real leg: after `pytest.importorskip("torch")` (kept
function-local, preserving the skip-scope doctrine that must not skip T1–T17/T19), import
`run_torch_nested`, reuse `make_dataset`/`mutate_subject`, and assert the **outer-test mutation
contract** (per §2.5: epoch budget, input-norm, sampler weights, per-fold selected checkpoint,
every `state_dict` tensor, and train predictions all bit-identical; only the held-out
prediction/score moves). The **inner-validation** torch contract (§2.5, second bullet — train-
only invariants hold, val-driven selection may change) is a **separate** test in
`test_torch_fit.py`, not inside the frozen file, so the one frozen-file edit stays minimal.
T18 is the **only** edit to `test_no_leakage.py`.

**Frozen values.** `ALPHA_GRID = (0.1, 1.0, 10.0)`; single-threaded determinism; Ridge cholesky.

**Acceptance** (`T-M7-rebind`). The full `test_no_leakage.py` passes with **T18 green** and no
other test changed; a shim-purity test asserts `reference_procedure` imports no sklearn symbol
and that view fields are the engine's own arrays (identity check); HISTORY.md records the
`git diff` of `test_no_leakage.py` showing only the T18 hunk changed (T1–T17, T19 byte-identical).

### 2.8 `experiments/run_regression.py` (extend) + `configs/exp_a_regression_77ghz.yaml`

**Responsibility.** The Exp A entrypoint. Extends the existing M2 spine
(config → ground truth → manifest → QC → folds → provenance) with feature-store validation,
the staged search, metrics/stats, and regenerable outputs — the same code path for the local
smoke and the (later) full run, differing only by flags/config.

**CLI.**
```
--config PATH        (repeatable; per-machine overlay last, existing semantics)
--band {10ghz,77ghz} (default 10ghz)
--subset 6subjects   XOR  --full-cohort     (mutually exclusive; one is REQUIRED)
```
`--subset 6subjects` = the 6 lowest evaluable subject ids (deterministic), genuine nested loops,
run in **mechanism-only mode** (C9, C14). A genuine nested search *must* compute inner-validation
scores (to select candidates) and outer `test_score`s — so the mode does **not** claim "no
performance value is computed"; it runs the **identical** search/scoring code path and instead
guarantees that **no performance value leaves the process**: every score is transient and is
**not printed, logged, persisted, plotted, or returned** — the run branches only at the final
**reporting boundary**, where in smoke mode it discards all performance-derived artifacts
(metrics JSON, predictions CSV, scatter, and the selection table) and writes only provenance +
a structural pass/fail run-log. This keeps the CLAUDE.md "same code path, config-only
local-vs-full" invariant (no separate smoke path) while ensuring those six cohort-member outer
scores are never surfaced before the freeze is cleared. Functional correctness of the staged
search is proven on the synthetic store (§2.3, T-M7-harness). **`--full-cohort` is the owner
gate and the only mode that surfaces performance values**: without `--subset`, the script
refuses to run unless `--full-cohort` is explicitly passed, printing "full-cohort Exp A spends
the config freeze — owner go-ahead required." A new `configs/exp_a_regression_77ghz.yaml` composes
`data.yaml`/`data77.yaml`/`preprocess77.yaml`/`wst77.yaml`/`search_77ghz.yaml` (mirroring
`exp_77ghz.yaml`) + `run`/`split`.

**Flow.** `load_config` → `protocol_freeze_guard(config)` pre-flight → data spine (unchanged) →
`store.validate_store` (fail-closed) → `record_run(... extra={"stage": "exp-a-smoke"|"exp-a-full",
"band":…, "n_eval":…, "store_fingerprint":…})` → `harness.run_exp_a` (identical in both modes) →
**reporting boundary**: full-cohort → metrics → guard → write all artifacts; smoke → discard all
performance-derived values, write only provenance + the structural run-log. The branch is at the
reporting boundary only; the search/scoring path above it is shared.

**Outputs** (all under `results/runs/<stamp>_<rev>/`): `provenance.json`;
`metrics_exp_a_{band}.json` (subject-balanced + pooled session MAE, RMSE, pooled r, S1–S4-only
pooled r, per-subject table incl. per-subject r where ≥3 sessions, per-seed mean±sd, every CI
with method-used + skipped-resample count + unreliable flag, the baseline comparison:
per-subject differences, Wilcoxon p, cluster-bootstrap CI on the mean difference — **all labeled
conditional/exploratory**); `predictions_{band}.csv` (subject, session, seed, y_true, y_pred,
fold); `selection_table_{band}.csv` (per-fold selected axes/family/params + a
selection-frequency summary); `fit_audit_{band}.json` (incl. tuned-ε records and ε values);
`scatter_{band}.png` (predicted-vs-actual, rendered via matplotlib's headless **Agg** backend —
the dependency added in step 0). **These performance artifacts are written only in
`--full-cohort` mode**; the mechanism-only smoke (C9) emits none of them (it writes only
provenance + a structural run-log). The smoke additionally **asserts in-code** that no subject
appears in more than one of train/val/test in any recorded fold, and that `fit_audit` covers
every fitted quantity — structural checks, not performance ones.

**Acceptance** (`T-M7-entrypoint`). A synthetic-store end-to-end run drives the full staged
search on 6 fabricated subjects with no private data; the guard runs before any result file is
written (the `test_protocol_freeze.py` entrypoint-order pattern); the full run is refused
without `--full-cohort`; subset selection is deterministic; the provenance env-fallback records
a git commit under a monkeypatched git failure. **Mechanism-only proof** (C14): a smoke run
writes **no** metrics JSON, predictions CSV, scatter, or selection table (only `provenance.json`
+ the run-log), and its stdout contains **no** MAE/RMSE/r value — asserted by scanning the run
directory and captured stdout, so the test proves no performance value *leaves the process*
rather than that none is computed.

### 2.9 `scripts/ibex/extract77.sbatch` + submit wrapper + `provenance.py` fix

**Responsibility.** Build the 77 GHz store on the cluster and self-attest the run's revision.

**Content.** `extract77.sbatch` clones `wst77.sbatch` (same PATH/venv preflight, same
task→(subject,session) map, `--array=0-79`, `02:00:00`, `16G`) but calls
`experiments/extract_features.py --band 77ghz --subject … --session …`. A `submit_extract77.sh`
wrapper captures `git rev-parse HEAD` / branch / dirty **at submit time** into
`DEHYD_GIT_COMMIT/_BRANCH/_DIRTY` (exported into the job env). `provenance._git_info` gains a
fallback: when the in-process `git` call returns `None` (the compute-node case), read those env
vars. `safe.directory` is **not** used — it did not take at M5 (git ignores it outside
protected config), and submit-time capture is robust. **Because the stores are built from the
step-10.5 clean commit** (C7), `DEHYD_GIT_COMMIT` identifies the exact M7 revision that produced
the features and `dirty` is `false`; the submit wrapper **refuses to submit from a dirty tree**
(a hard check), so a store can never be attributed to the M6 base while uncommitted M7 code
actually produced it. A `dirty=true` boolean is treated as non-reproducible provenance, not an
acceptable state for a store build.

**Acceptance** (`T-M7-provenance`). With the in-process git subprocess monkeypatched to fail
and `DEHYD_GIT_COMMIT` set, `record_run` writes that commit (not `None`); with neither, it
still degrades to `nogit` as today; the submit wrapper's dirty-tree refusal is exercised.

---

## §3 Tests

| Group | File | What it proves |
|-------|------|-----------------|
| T-M7-metrics | test_metrics.py | 5.5-pinned `subject_balanced_mae` + unequal-count fixture; RMSE; pooled r; per-subject-r ≥3-session rule; S1–S4 rule; bootstrap determinism; BCa vs hand case; percentile fallback recorded on degenerate input; skip-and-count + >5% flag; both seed-collapse rules; Wilcoxon wrapper |
| T-M7-regressors | test_regressors.py | each family builds/fits; ridge pinned cholesky; per-seed bit-determinism; rf/gbm differ across seeds; grid sizes = documented counts and ≤ budget_k; **per-family auditable fitted-state capture** (scaler/ridge/svr incl. `support_vectors_`/knn/rf/gbm) is bit-comparable (C11); **perturbing a tree node, `gbm.init_`, or `learning_rate` changes the ensemble digest** (C20) |
| T-M7-harness | test_harness.py | flat inner_results assembly order (fold-major/candidate-minor); score-matrix shape; **selection routed through `select_candidate`** (discharges test_selection's deferred behavioural claim: fits on inner-train, scores on inner-val, real GroupKFold); two-run bit-identity; non-selectable folds excluded; before_fit called once per fit; **`active`-completeness fail-closed — one negative test per omitted required key** (C5); **fold-viability**: pre-fit `k>rows` KNN marked non-evaluable with a reason code, selection proceeds among the rest, all-non-evaluable raises, on a sparse-session boundary — and an **unexpected fit/predict exception propagates** rather than being swallowed (C6, C21); per-seed outer outcomes separate; audit covers tuned_epsilon; tuned-ε inner-val mutation micro-test; **end-to-end synthetic-store outer-mutation property** over the real two-stage path (C1); **per-family held-out mutation** (parametrized ridge/svr/knn/rf/gbm: each family's fitted state + train predictions bit-identical under outer-test mutation) (C11); staged winner carry-forward; **shim-purity** (reference_procedure imports no sklearn; view fields identity-equal engine arrays) |
| T-M7-rebind | test_no_leakage.py (frozen) | T1–T17, T19 now exercise harness code via the shim; **T18 activated and green** (outer-test torch mutation contract) |
| T-M7-torch | test_torch_fit.py | two-run bit-identical state_dicts/predictions; **true early stopping** halts at the right epoch under frozen patience/min-delta on a crafted val curve; median epoch budget at refit; norm + sampler weights train-only (mutation); different seeds differ; runtime budget guard; **separate inner-validation mutation test** — train state bit-identical over the **common prefix** of executed epochs, stop-time/checkpoint may differ, mutated val subject's predictions move (C13, C19) |
| T-M7-store | test_store.py | npz round-trip; lazy read; keep_raw leaves vectors/prelog bit-identical; store-vs-direct frozen-ε equivalence; pool_stats_batch == looped; apply_order_log epsilon_by_order ≡ frozen at 1e-6; fingerprint staleness (spec_hash **or** qc_config_hash) rejected; **same-count frame-membership change fails closed** (C4); **store/analysis commit-mismatch rejected** (C16); missing session fails closed |
| T-M7-fitaudit | test_harness.py (or test_fitaudit.py) | **serialized `fit_audit_{band}.json` validated** (C15): every `params_repr` is a stable hex sha256 across two unmutated runs, `tuned_epsilon` carries literal `[ε1, ε2]`, and the frozen `{test_subject,quantity,role,fitted_on,inner_val}` keys are present |
| T-M7-baselines | test_baselines.py | per-index train means; train-only (test-subject mutation inert); **fit emits a FitRecord, audit records it, guard-before-baseline-fit ordering** (C12); absent-index → **global-training-mean fallback** (O2, owner-decided) |
| T-M7-entrypoint | test_run_regression.py | synthetic-store end-to-end staged search (6 fabricated subjects); guard-before-write ordering; full-cohort refused without `--full-cohort`; deterministic subset selection; **mechanism-only smoke leaks no performance value** — no metrics/predictions/scatter/selection-table file and no MAE/RMSE/r in stdout (C9, C14) |
| T-M7-provenance | test_provenance.py | `DEHYD_GIT_COMMIT` env fallback records commit under monkeypatched git failure; degrades to `nogit` with neither; **both producers refuse a dirty tree** — submit wrapper *and* local `extract_features.py` (C7, C16) |
| T-M7-plotting | test_plotting.py | headless **Agg** scatter renders a PNG without a display; matplotlib importable within the pinned env (scipy `<1.17` intact) (C10) |

`tests/test_no_leakage.py`: **exactly one hunk changes** (T18, A-M7-1); T1–T17 and T19 stay
byte-identical, audited by a recorded `git diff`. Expected total ≈ **84 new tests**
(576 → ~660), with T18 flipping skip → pass.

---

## §4 Definition of done

| ID | Criterion |
|----|-----------|
The DoD has two phases. **D0–D9 are the pre-checkpoint gate**; **D10–D13 are the
post-approval completion criteria.** The milestone is *paused* (not complete) between them, and
if the owner withholds go-ahead it stays paused rather than closing (C8).

_Pre-checkpoint (build + verify, nothing real inspected):_

| ID | Criterion |
|----|-----------|
| D0 | This plan reviewed through the Codex⇄Claude loop; all comments applied/withdrawn or owner-deferred; deferred items + the Step 0b owner-approval items (O1–O3) cleared |
| D1 | `matplotlib` pinned in `pyproject.toml`/`uv.lock` with `scipy <1.17` intact; `uv run python -m pytest` green, including every new T-M7-* group |
| D2 | `harness.py` builds folds only from `splits.py`, tie-breaks only via `select_candidate`, calls `protocol_freeze_guard(config, active=…)` (with the fail-closed `active`-completeness check) before every fit/write, and emits a per-fold fit-audit incl. tuned-ε and per-family fitted state |
| D3 | `tests/reference_procedure.py` is a thin adapter over `harness.py` (no fitting code); the frozen `test_no_leakage.py` passes with **T18 green** |
| D4 | `tests/test_no_leakage.py` changed **only** in the pre-registered T18 hunk (A-M7-1); the `git diff` proving T1–T17/T19 byte-identical is in HISTORY.md |
| D5 | The **end-to-end synthetic-store outer-mutation** property (C1) is green: the real two-stage headline path leaks nothing under held-out mutation |
| D6 | A clean owner-triggered M7 commit exists on `v1_milestone_7` (step 10.5); feature stores built **from it** and `--validate`-clean for both bands; store-vs-direct + same-count frame-membership tests green |
| D7 | The git-provenance fix verified; the IBEX 77 GHz store self-attests the **clean M7 commit** (`dirty=false`), and the submit wrapper refuses a dirty tree |
| D8 | Both **mechanism-only** smoke runs (`--subset 6subjects`, both bands) complete on CPU, assert train/val/test disjointness + full fit-audit coverage, and emit **no performance value** |
| D9 | **Owner checkpoint reached and reported**; the milestone pauses here for explicit go-ahead |

_Post-approval (only after the owner clears the freeze):_

| ID | Criterion |
|----|-----------|
| D10 | Both **full-cohort** Exp A runs (`--full-cohort`, 10 GHz and 77 GHz) complete; the freeze is recorded as spent at the first visible real outer score |
| D11 | All promised artifacts exist and are regenerable by one command: metrics JSON (MAE/RMSE/pooled+per-subject r, S1–S4 r, per-seed mean±sd, subject-cluster CIs with method/skip/unreliable flags), predictions CSV, selection-frequency table, fit-audit JSON, and the predicted-vs-actual scatter — vs the session-index-only baseline (Wilcoxon + bootstrap-CI), all labeled conditional/exploratory |
| D12 | `implementation_plan.md` amendments recorded owner-approved: A-M7-1 (T18) plus the Step 0b items **as the owner actually decided them** (O1 inner-fold-variance estimator, O2 baseline absent-index rule, O3 baseline guard path) — the plan does not presuppose which option is chosen |
| D13 | HISTORY.md carries the full-run entry and the review-close summary; SECOND_CHAPTER.md §6 (Exp A) written **from the full results** |

---

## §5 What could go wrong (known traps)

1. **`inner_results` ordering drift breaks `zip(strict=True)`** in the frozen T10/T16.
   Execution is candidate-major (raw-cache reuse), but assembly must stay fold-major /
   candidate-minor. Pinned by a dedicated unit test; T10 catches any regression at the frozen
   level.
2. **Tuned-ε computed at the wrong CV level** — reusing an outer ε inside inner folds would be
   val leakage. Structural defense: ε lives inside `data_for(candidate, train_subjects)` and
   receives the fit's own train set; the ε FitRecord's subject set is audited; an explicit
   inner-val mutation micro-test asserts invariance.
3. **Store staleness** — code/config drift after extraction. Semantic-fingerprint validation at
   load, fail-closed (the proven `--merge-shards` doctrine); the git-dirty flag is recorded.
4. **Ridge cholesky on wide matrices** (pooled D up to ~18k ≫ n≈68). sklearn should take an
   efficient path under `solver="cholesky"`; verified for timing + determinism on a D≈5000
   synthetic in the harness tests and again in the smoke. A pathological cost is an
   owner-visible finding, never a silent solver swap.
5. **RF/GBM nondeterminism.** `random_state=seed`, `threadpool_limits(1)`, no `n_jobs`;
   bit-identity is claimed **per-machine only** (platform recorded in provenance) — cross-machine
   identity is not promised.
6. **BCa degeneracy at N_eval=16** (infinite z0 / zero jackknife variance, e.g. on baseline
   differences). Detected per metric; percentile fallback recorded in the JSON; skip-and-count
   with the >5% unreliable flag; unit-tested on degenerate fixtures.
7. **10 GHz store size (~8–12 GB local).** Acceptable but flagged; the fallback is dropping the
   gate-2 raw tensors and streaming re-extraction for that gate (slower, identical numbers) — a
   labeled operational choice, not a protocol change.
8. **T18 runtime.** A full 8-fold × 5-inner torch run could be slow; the spec caps `max_epochs`
   (~20–30) and `TinyMLP` is tiny; target <90 s, measured in `test_torch_fit` first.
9. **scipy `<1.17` pin.** No dependence on `scipy.stats.bootstrap` (own BCa); `scipy.stats.wilcoxon`
   is stable within the pin.
10. **`selected_alpha` type.** The engine stores Python floats in `model_params`, so the frozen
    `in ALPHA_GRID` membership check holds trivially.
11. **77 GHz sbatch memory/time.** `keep_raw` adds ~20 MB/session output and negligible RAM over
    M5's measured profile; `02:00:00`/`16G` retained; preflight unchanged.
12. **Guard cost.** `protocol_freeze_guard` runs ~10⁴ times (once per fit) but is dict
    comparisons — microseconds; measured in the smoke.

---

## §6 Flagged gaps in implementation_plan.md + proposed amendments

- **A-M7-1 (T18 activation — owner-approved, Step 0 item 1).** `tests/test_no_leakage.py` and
  `tests/reference_procedure.py` carry stale "M6" wording (a pre-A-M5-2 renumber): the rebind
  and torch-leg comments mean the **current M7**. The frozen file cannot be edited to fix the
  stale number, so "M6" there reads as "M7". The **one** sanctioned edit is the pre-registered
  T18 activation (remove the skip, write the real body). Proposed as an APPLIED amendment with
  the HISTORY diff audit as evidence.
- **A-M7-2 (`inner_fold_variance` uses `np.std(..., ddof=0)`) — OWNER-APPROVED (O1, 2026-07-25).**
  `implementation_plan.md`'s tie-break says only "lower inner-fold variance"; M6 never pinned the
  estimator. Per the M6 process rule (a protocol-gap completion clears an explicit owner gate
  before it is baked in — the C6-16 precedent) it went through the Step 0b gate; the owner chose
  **population std (`ddof=0`)**. (Tertiary tie-break key, reached only after MAE,
  `simplicity_rank`, and `feature_dimension` all tie — low stakes, but pinned for reproducibility.)
- **Two further protocol-gap completions cleared the owner gate (Step 0b, 2026-07-25):**
  **O2** — the session-index baseline's absent-index behaviour = **global training-fold mean**;
  **O3** — the K=1 baseline's protocol-guard path = **config-level `protocol_freeze_guard(config)`,
  `active=None`** (frozen M6 guard code untouched). Both recorded before step 9.
- **Git-provenance fix (carried from M6 §7).** `git.commit = None` on IBEX compute nodes,
  resolved via submit-time `DEHYD_GIT_COMMIT` capture + `_git_info` fallback; `safe.directory`
  rejected (git ignores it outside protected config). **Paired with the step-10.5 clean commit**
  so a store is always attributed to the exact committed M7 revision (`dirty=false`), never the
  M6 base with a dirty tree; the submit wrapper refuses a dirty tree. Not an amendment to the
  plan text — a code + sequencing fix the plan tracks.
- **`matplotlib` added to the pinned environment** (step 0) — no plotting backend is pinned
  today, and the Exp A scatter needs one. Additive dependency; must not disturb the `scipy
  <1.17` pin (matplotlib has no scipy dependency). Recorded in `TRACKED_PACKAGES`.
- **Stale "milestone 6" strings** in `experiments/run_regression.py` ("Modeling lands at
  milestone 6", the `"milestone-2-smoke"` stage string) and the `extraction.py` header ("the M6
  harness") — updated to M7 reality in the same steps that touch those files. Hygiene, not a
  protocol change.

No new gaps in the **scientific** protocol are proposed; every modelling/stats constant is
consumed as frozen at M6.

---

## §7 Open items this milestone resolves or carries

**Resolves:** the LOSO harness (sklearn + torch fit paths); Exp A (session-level Δm% regression,
both bands, vs the session-index-only baseline) end to end — the full-cohort result completed
**post-checkpoint on owner go-ahead** (D10–D13), not at the checkpoint; the subject-cluster
bootstrap and Wilcoxon machinery Exp A reports; the persistent feature stores for both bands;
`test_selection.py`'s deferred behavioural claim (fit/score over real folds, via the harness);
the frozen-suite rebind and **T18** (torch mutation property) green; the `git.commit = None`
IBEX provenance gap.

**Fixed here, NOT open at M8+:** the harness's fold source (`splits.py` only), tie-break source
(`select_candidate` only), guard-before-every-fit contract, fit-audit shape, the tuned-ε
fold-local computation, the store schema + fingerprint doctrine, and the T18 real body — each
would need its own prior authoritative amendment to revisit.

**Carries to M8+:** Exp B (clock-decoupling) reuses this harness with its own residual-MAE
objective (A-M6-3); Exp C/D (ordinal + baselines, incl. the Frank-Hall decomposition and the
77 GHz baselines A-M6-2/A-M6-5) add the classification and DL fit paths onto `torch_fit` +
`baselines`; Exp E/F/G/H (interpretability, confound, fusion, full stats, figures) consume the
same engine, metrics, and stores. The DL baseline results run on IBEX under the frozen
`BaselineConfig` constants mapped onto `TorchFitSpec`.

---

## Plan review (Codex ⇄ Claude)

Status: REVIEW_COMPLETE
Codex: NO MORE COMMENTS (2026-07-25)

### Open comments

_(none)_

### Debate comments

_(none)_

### Deferred to owner

These three protocol-gap completions are owner decisions (raised by C2/C3/C12, cross-referenced
in the plan body as Step 0b O1–O3). They are frozen here per the loop protocol: the owner writes
an `Owner decision:` line under each, and Claude then implements that outcome and logs it
resolved. Each must be decided before the build step that depends on it.

#### O1 — `inner_fold_variance` estimator (blocks step 4; raised by C3)
The tie-break's tertiary key. `implementation_plan.md` says only "lower inner-fold variance"; M6
never pinned the estimator. Options: **(a)** population std `np.std(ddof=0)` (plan's proposal);
**(b)** sample std `np.std(ddof=1)`; **(c)** variance rather than std. Low stakes (reached only
after MAE, `simplicity_rank`, `feature_dimension` all tie) but must be fixed, not assumed.
Owner decision: **(a) population std `np.std(ddof=0)` — 2026-07-25.** Implemented in the plan body
(§2.3, §6/A-M7-2); the tie-break treats the observed inner folds as the whole population.

#### O2 — session-index baseline behaviour for a training-absent time index (blocks step 9; raised by C2)
A new outcome-affecting rule absent from `implementation_plan.md` §D(iv)/`BaselineConfig`.
Cannot occur in the full 15-train-subject cohort (all five indices present) but can in the
6-subject smoke / sparse boundaries. Options: **(a)** global-training-mean fallback (plan's
sketch); **(b)** drop that prediction (and define the metric's treatment of the gap);
**(c)** another owner-specified rule. Owner decision: **(a) global-training-mean fallback —
2026-07-25**, so every test session stays scored and radar-vs-baseline share the identical
session set (the paired Wilcoxon needs this). Implemented in the plan body (§2.4).

#### O3 — the K=1 baseline's protocol-guard path (blocks step 9; raised by C12)
The baseline uses no WST search axes and `protocol_freeze_guard`'s `active` whitelist has no
`session-index-only` `model_family`. Options: **(a)** guard it at the config level
(`protocol_freeze_guard(config, active=None)`) before the fit — no per-fit WST record (plan's
proposal); **(b)** approve a guard amendment adding a baseline `active` record to the whitelist.
Owner decision: **(a) config-level guard, `active=None` — 2026-07-25** (the baseline has no WST
axes to validate; frozen M6 guard code stays untouched). Implemented in the plan body (§2.4).

### Resolved review comments

- C1 (applied): headline path only shim-tested → §2.3 acceptance + D5 now require an end-to-end **synthetic-store outer-mutation** test over the real two-stage `run_exp_a_fold` (mutate stored vec/raw/prelog/target; winners, inner scores, tuned-ε, per-family fitted state, train preds bit-identical; only held-out moves).
- C2 (applied): unapproved baseline global-mean fallback → de-frozen; elevated to **owner-open O2** (Step 0b); §2.4 no longer asserts it; its test is written against whatever behaviour the owner fixes.
- C3 (applied): `ddof=0` baked in without an owner gate → moved to **owner-open O1** (Step 0b); §2.3/§6/DoD now treat it as pending, per the C6-16 precedent.
- C4 (applied): fingerprint didn't bind frame membership → added `frame_ids_sha256` (ordered selected frame IDs) + `qc_config_hash` + `session_eligible` to the sidecar, and a same-count membership-change fail-closed test (§2.6).
- C5 (applied): guard passes an under-populated `active` → added a harness-side **fail-closed `active`-completeness check** (exact band key set) + one negative test per omitted key (§2.3, T-M7-harness).
- C6 (applied): KNN `k>rows` had no viability rule → added a deterministic **fold-viability** rule (non-evaluable → non-finite → `select_candidate` filters; all-non-evaluable raises) + a sparse-boundary test (§2.3).
- C7 (applied): stores built from uncommitted dirty M7 code → added **step 10.5** (owner-triggered clean M7 commit) before any producer; submit wrapper refuses a dirty tree; §2.9/D6/D7 now attribute stores to the clean commit.
- C8 (applied): DoD ended at the checkpoint → **split DoD** into pre-checkpoint D0–D9 and post-approval D10–D13 (both full runs, all artifacts/stats, journal + SECOND_CHAPTER §6 from full results); milestone stays paused if go-ahead is withheld.
- C9 (applied): real smokes exposed outer scores → real-data smoke is now **mechanism-only** (performance-blinded, no metrics/scatter); invariant/§2.8/§1 revised; freeze recorded as spent at the first visible real outer score; synthetic-store test carries functional correctness.
- C10 (applied): no plotting dependency pinned → added **step 0** pinning `matplotlib` (scipy `<1.17` intact) + `TRACKED_PACKAGES`; headless **Agg** render smoke (T-M7-plotting); scatter written only in `--full-cohort`.
- C11 (applied): non-Ridge fitted state uncaptured → defined **per-family auditable `FitRecord.params`** (scaler/ridge/svr/knn/rf/gbm, per seed) + parametrized per-family held-out mutation tests (§2.3, T-M7-harness/regressors).
- C12 (applied): baseline unaudited + guard whitelist gap → `fit_session_index_baseline` returns a `BaselineFitOutcome` carrying a `FitRecord`; guard path is **owner-open O3** (config-level `active=None` proposed); ordering test added (§2.4).
- C13 (applied): T18 "both CV levels" conflated two contracts → separated **outer-test** (all invariant, incl. selected checkpoint) from **inner-val** (train-only invariant; val-driven selected epoch may change); T18 asserts the outer-test contract, inner-val is a separate `test_torch_fit.py` test (§2.5/§2.7).
- C14 (applied, re: C9): "no performance value computed" was infeasible (a nested search must compute scores) → reworded to **"no performance value leaves the process"**; §2.8 flow branches only at the reporting boundary (shared search/scoring path), smoke discards all performance artifacts; acceptance scans the run dir + stdout to prove nothing surfaced.
- C15 (applied, re: C11): fitted-state contract not executable as typed → **every `FitRecord.params` value is an `np.ndarray`** (ensemble digest as a `uint8` sha256 array), SVR gains `support_vectors_`, and the serialized `fit_audit` JSON is defined separately (hex-hash `params_repr` + literal ε) with its own test (§2.3, new T-M7-fitaudit).
- C16 (applied, re: C7): dirty-tree refusal was IBEX-only → applied to **both** producers (local `extract_features.py` also refuses a dirty tree); `validate_store` requires the store commit == the analysis commit; local-dirty + commit-mismatch tests added (§2.6/§2.9).
- C17 (applied, re: C2/C3/C12): O1–O3 were only in the body → **moved as decision threads (with options + `Owner decision: pending`) into "Deferred to owner"**; Step 0b now cross-references them; D12 reworded to not presuppose the chosen option.
- C18 (applied, re: C13): early stopping made the inner-val trajectory variable-length → *(superseded by C19 — the run-to-fixed-max approach was reverted)*.
- C19 (applied, re: C18): the run-to-fixed-max design diverged from the frozen early-stopping protocol and would make T18 test a different algorithm than Exp D's DL path → **reverted to true patience/min-delta early stopping** in the shared torch fit path; the inner-val mutation test now compares train state over the **common prefix** of executed epochs (stop-time/checkpoint may differ) — no protocol amendment needed (§2.5/§2.7, T-M7-torch).
- C20 (applied, re: C15): GBM digest was incomplete → the ensemble digest now binds the fitted `init_` and the combining hyperparameters (`learning_rate`/`n_estimators`/`max_depth`), not only tree node arrays; a test asserts perturbing any component changes `params_repr` (§2.3, T-M7-regressors).
- C21 (applied): "cannot fit/predict" was too broad → fold-viability is now an **explicit enumerated pre-fit predicate set** (min: `knn.n_neighbors ≤ n_train_rows`) with recorded reason codes; **unexpected fit/predict exceptions propagate loudly**, never swallowed (§2.3, T-M7-harness).
