# MILESTONE 6 PLAN — the config-freeze gate (both bands, all of A–G, before any outer-fold result exists)

## §0 Scope and ground rules

**Why this milestone exists (implementation_plan.md §Build order step 6).** Experiments
B–G reuse the same 16 subjects as Experiment A. If any protocol choice — a search space, a
baseline spec, a statistical test, an ordinal family — is decided *after* Exp A's outer-fold
results are visible, that choice is indirectly informed by subjects who will later serve as
"test" subjects in B–G. The freeze makes "no config tuned on outer-test" true at the
**cohort** level, not merely within Exp A's own nested CV. Concretely: commit the **complete**
A–G protocol design for **both bands** to versioned `configs/` + git, **before any outer-fold
result — from any experiment — is inspected.**

**Revision history.**
- *Round 1 (Codex, 2026-07-25):* the first draft's "order-2-usefulness pre-check" ran a
  full-cohort LOSO comparison to decide the log axis's third branch — a genuine leakage
  conflict (C6-01), plus 14 other gaps (band-keyed schemas, unfrozen grids, missing 77 GHz
  baselines, an under-scoped whitelist guard, etc.). All 15 comments were applied.
- *Round 2 (Codex, 2026-07-25):* a follow-up pass found that round 1's fix, while
  technically sound, (a) **unilaterally treated a correction to an owner-approved mechanism
  (A-M4-7) as already settled** rather than flagging it for explicit approval (C6-16) — a
  real process error, since `implementation_plan.md` is this task's authoritative base and
  a milestone plan may not silently override it; (b) left several of round 1's fixes
  under-specified in ways that matter (the tuned-ε cross-session aggregation rule, C6-17;
  the staged-selection objective wrongly scored on training data instead of inner-validation
  data, C6-18; budget parity between WST and baselines was not actually uniform, C6-19; the
  whitelist guard would reject an approved search candidate by naively reusing the strict
  artifact guard, C6-24); and (c) found the StatsConfig/BaselineConfig/ExpCConfig schemas
  were still too thin to actually constrain M7 (C6-20/C6-21/C6-22/C6-23). All 10 round-2
  items were addressed; C6-16/C6-19 escalated to an explicit **Step 0 owner-approval gate**,
  consistent with the A-M5-1/A-M5-2 precedent (owner approval *before* implementation).
- *Round 3 (Codex, 2026-07-25):* nine comments, all addressed. **C6-26**: the 77 GHz
  baseline proposal (A-M6-2) was dimensionally broken — its "raw" input averaged away the
  chirp/slow-time axis its own physics baseline needed, and its "matched" input claimed a
  pre-WST Rx fusion Exp G forbids — corrected in both documents. **C6-27**: the
  staged-selection section implied a real fitting algorithm existed to test, when M6 builds
  none; narrowed to one small, genuinely non-modeling helper (`eval/selection.py`).
  **C6-28 → C6-31/C6-32 (a self-correction within this round):** a first fix to the
  proportional-odds library risk silently substituted a different statistical model
  (Frank-Hall decomposition) for the approved one; a follow-up comment caught this and it
  was un-applied and escalated to Step 0 instead. **C6-29**: the whitelist guard couldn't
  see call-time values at all; added a required `active=` parameter. **C6-30**: the prose
  tie-break and the code's tie-break used two different, inconsistent definitions of
  "simpler model"; unified into `simplicity_rank` + `feature_dimension`. **C6-33**: the
  proposed 2 Hz Doppler cutoff for the 77 GHz physics baseline was below the system's actual
  ≈7.63 Hz frequency resolution and physically unrealizable; redefined honestly as a
  DC-bin-vs-any-motion split. **C6-34**: `select_candidate` had no defined behavior for
  non-finite scores; added an explicit finite-value filter.
- *Round 4 (Codex, 2026-07-25):* 4 comments, all applied — consistency lapses from round
  3's fast pace of fixes (a stale Step 0 item description, a verification deferred too late,
  a stale item count, an invented statistical correction not in `implementation_plan.md`).
- **Review closed** (Codex, 2026-07-25): "NO MORE COMMENTS" after 38 comments across 5
  rounds; none disputed. Every comment ended either applied-and-fixed or escalated to the
  Step 0 gate below.
- **Step 0 — RESOLVED (owner, 2026-07-25).** All five items decided; see the record below.
  Item 5's verification was carried out as part of clearing Step 0 (not deferred to M7, per
  C6-36): `statsmodels.miscmodels.ordinal_model.OrderedModel` was checked directly — its
  `__init__`/`.fit()` signatures carry no `sample_weight`/`freq_weights` parameter, and it
  inherits from `GenericLikelihoodModel` (a generic MLE optimizer with no observation-
  weighting mechanism) — so candidate A cannot implement the required class weighting.
  Per the owner's instruction ("go with A, if wrong then B"), **candidate B (the Frank-Hall
  decomposition) is the approved implementation**, recorded as `implementation_plan.md`
  A-M6-5.

## Step 0 — owner decisions (RESOLVED 2026-07-25)

All five items below were open questions through 5 rounds of review; each is now decided.
Nothing past this point is conditional — the rest of this document reflects the resolved
design directly, with no more "pending"/"[provisional]" framing.

1. **A-M6-1 — the order-2-usefulness pre-check is retracted.** Decision: **(a)** — the
   log axis's third branch (`on+tuned-ε`) is an **unconditional** inner-CV candidate for
   both bands, selected fold-locally like every other axis, with no separate cohort-wide
   predictive check. Applied in `implementation_plan.md` (A-M6-1, status APPLIED).
2. **A-M6-2 — the corrected 77 GHz Exp D baseline design is approved as drafted**,
   including the DC-bin-vs-any-resolvable-motion physics baseline: the owner confirmed that
   coarser contrast is still worth reporting despite being unable to distinguish breathing
   from gross movement. Applied in `implementation_plan.md` (A-M6-2, status APPLIED).
3. **Budget-parity interpretation — approved as proposed.** Representation-level choices
   (WST's feature axes; a baseline's raw-vs-matched input) are symmetric across families and
   exempt from the budget K; model/training-hyperparameter grids are capped at K = 12
   uniformly for every family, WST or baseline (§2.1).
4. **Every §3 provisional constant — accepted as proposed**, no changes requested.
5. **Exp C's proportional-odds implementation — candidate B (Frank-Hall decomposition)**,
   after verifying candidate A (`statsmodels.OrderedModel`) lacks the required weighting
   support (see the revision-history note above). Applied in `implementation_plan.md`
   (A-M6-5, status APPLIED) as an explicit, documented substitution — not a silent one.

---

**In scope:**

- **The three owner decisions this session settles** (recorded here and in HISTORY.md):
  1. **77 GHz in-band QC threshold (`qc77.min_in_band_energy_ratio = 0.30`) stays frozen.**
     Not moved inside inner CV; never re-tuned to a new fixed value. A labeled-**exploratory**
     sensitivity re-run at 0.28/0.32 is pre-registered for **after** primary results exist.
  2. **The order-2 log branch is unconditional for both bands** (A-M6-1).
  3. **Exp C's proportional-odds family is the Frank-Hall decomposition** (A-M6-5).
- **Every experiment's complete, executable design**, frozen as validated config (not
  prose): the Exp A/WST search space (both bands, band-keyed, tiling as an explicit axis, a
  literal staged-selection algorithm scoring on inner-validation subjects); the baseline
  specs (Exp D i–iv, both bands); per-family budget K with a uniform counting rule and the
  seed set; the full statistical protocol (Exp H, transcribed completely into config); Exp B
  (search-space reuse + residualization + equal-session objective, A-M6-3); Exp C (ordinal
  cutpoint source, tie handling, class-weight formula, fold-viability, the Frank-Hall
  implementation); Exp E (the named, concrete pre-registered interpretability config); Exp F
  (the nested ridge model, the representation-selection rule, A-M6-4, and the shared λ
  grid); Exp G (recorded as validated config, matching implementation_plan.md exactly).
- **The frozen protocol-constant whitelist validator** — composes the *existing*
  `canonical_spec_guard_77` feature guard unchanged, and independently re-validates the 10 GHz
  `preprocess`/`wst` fields with the one necessary exception (the range-gate axis, which the
  strict artifact guard cannot be reused for without rejecting an approved search candidate),
  plus every new frozen section from this milestone, plus the call-time `active=` record.
- **Small hygiene fixes**: the stale "fixed at the milestone-5 config-freeze gate" comment in
  `configs/exp_a_regression.yaml:22` (milestone renumbered to 6 at A-M5-2; never updated).
- **Tests** for every new frozen config section, cross-band structural rejection, the
  whitelist guard's ordering and `active=` validation.
- **Journal**: HISTORY.md entries per resolved step; SECOND_CHAPTER.md §5 "The config
  freeze" at close.
- **The freeze commit + an annotated git tag** (e.g. `config-freeze-v1`), owner-triggered.

**Explicitly out of scope (deferred to M7+):**

- **Any modelling, model selection, or outer-fold evaluation of Exp A/B/C/D/E/F/G.** M6
  *specifies* the search spaces and protocols; it never *runs* them, and performs **no
  computation on this cohort's data at all**. `eval/harness.py` does not exist yet and is
  not built here (the one exception, `eval/selection.py`, is pure tie-break logic over
  already-computed numbers — see §2.1a).
- **Exp G's α fusion combiner fitting** — the selection rule is already frozen in
  `implementation_plan.md` and recorded as config here; the α value itself is only ever fit
  inside M7's CV.
- **Secondary 77 GHz variants** (Doppler-FFT-spectrum WST, fast-time WST) — deferred past M5
  by A-M5-7; not reopened here.
- **10 GHz and 77 GHz frozen front-end code** (`loader_*`, `manifest*`, `qc/screens*`,
  `preprocess/pipeline*`, `features/extraction*`) — untouched, with the exception noted above
  (§2.3: the whitelist guard reimplements a small field-comparison locally rather than
  modifying `extraction.py`, precisely so that file stays untouched).
- **`tests/test_no_leakage.py`** — byte-for-byte unmodified since M1.

**The milestone-6 invariant, protected above all:**

> **M6 performs no computation that touches predictive signal.** Every value in this
> document is either (a) already fully specified in `implementation_plan.md` and transcribed
> into validated config, or (b) a concrete value this plan proposed on non-performance
> grounds, or (c) a change to previously-approved design — all of (b) and (c) went through
> the explicit Step 0 owner-approval gate above before being treated as settled, rather than
> being applied silently. Nothing here inspects an outer-fold result, and nothing here may
> be adjusted after M7 produces one.

**Not reopened at M6** (fixed at M5): Rx fusion = mean primary / median secondary; feature
family = pooled classical / flat diagnostic-DL-only; the 77 GHz frozen protocol constants
(`flatline_skip_leading_bins=1`, the Doppler tilings, `max_order=2`, `log_epsilon=1e-6`, the
2–4 m gate, the QC screen structure).

**Also not reopened, now that Step 0 has cleared:** the log-axis gating mechanism (A-M6-1),
the 77 GHz baseline design (A-M6-2), the budget-parity rule, and the Exp C proportional-odds
implementation (A-M6-5) — each would need its own prior authoritative amendment to revisit.

**Ground rules.** Work on `v1_milestone_6`; commits only when the owner asks. HISTORY.md
written continuously. Superseded material → `archive/` with a note.

---

## §1 Build sequence — exact order and why

| # | Step | Why this position |
|---|------|-------------------|
| 0 | ✅ **Owner approval on all five Step 0 items — DONE 2026-07-25** (incl. the `statsmodels.OrderedModel` weighting verification) | Prerequisite cleared; the rest of this milestone can now proceed |
| 1 | Plan doc (this file) + HISTORY.md opening entry for M6, recording Step 0's outcome | Establishes the approved design before any config is written |
| 2 | **Write every frozen config section** (§2) | The bulk of the milestone |
| 3 | **Whitelist validator**: `protocol_freeze_guard(config, active=...)` and its tests | Needs the frozen sections from step 2 |
| 4 | **Cleanups + journal**: stale milestone comment; HISTORY entries; SECOND_CHAPTER.md §5 | Standard close-out hygiene |
| 5 | **Freeze commit + annotated tag** (owner-triggered) | Nothing after this tag may be treated as still open for tuning |

---

## §2 Per-file specifications

Format per file: **Responsibility** · **Public API / content** · **Frozen values** ·
**Acceptance criteria**. Every value below is either transcribed from
`implementation_plan.md` or was an owner-confirmed proposal from §3 (below) — never a
placeholder, since `config.py`'s `_reject_unknown` rejects unknown YAML keys and ignores
comments.

### 2.1 `src/dehyd/config.py` — new frozen sections

Following the exact `QC77Config`/`Preprocess77Config`/`WST77Config` pattern
([config.py:174-243](src/dehyd/config.py#L174)).

**Band-keyed search spaces:**

```python
@dataclass(frozen=True)
class SearchSpace10GHzConfig:
    reduction: tuple[str, ...] = ("A", "B")
    channel: tuple[str, ...] = ("mag", "iq")
    tiling: tuple[str, ...] = ("T1", "T2", "T3")
    log_branches: tuple[str, ...] = ("off", "on_frozen_eps", "on_tuned_eps")  # unconditional, A-M6-1
    range_gate_m: tuple[tuple[float, float], ...] = ((1.0, 2.0), (0.9, 3.0))
    model_families: tuple[str, ...] = ("ridge", "svr", "rf", "gbm", "knn")
    budget_k: int = 12
    stage1_anchor_model: str = "ridge"
    stage1_anchor_ridge_alpha: float = 1.0
    tuned_eps_k: float = 0.1

@dataclass(frozen=True)
class SearchSpace77GHzConfig:
    reduction: str = "slow_time_iq_primary"      # fixed, not a candidate set (Exp G)
    channel: str = "iq"                          # fixed
    gate_m: tuple[float, float] = (2.0, 4.0)     # fixed (matches preprocess77.gate_m)
    tiling: tuple[str, ...] = ("T1_77", "T2_77", "T3_77")
    log_branches: tuple[str, ...] = ("off", "on_frozen_eps", "on_tuned_eps")  # unconditional, A-M6-1
    model_families: tuple[str, ...] = ("ridge", "svr", "rf", "gbm", "knn")
    budget_k: int = 12
    stage1_anchor_model: str = "ridge"
    stage1_anchor_ridge_alpha: float = 1.0
    tuned_eps_k: float = 0.1

@dataclass(frozen=True)
class ModelGridConfig:
    ridge_alphas: tuple[float, ...] = (0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)   # 8
    svr_c: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    svr_epsilon: tuple[float, ...] = (0.01, 0.1, 0.3)                                 # 4x3=12
    rf_n_estimators: tuple[int, ...] = (100, 300)
    rf_max_depth: tuple[int | None, ...] = (3, 5, None)                               # 2x3=6
    gbm_n_estimators: tuple[int, ...] = (100, 300)
    gbm_learning_rate: tuple[float, ...] = (0.01, 0.1)
    gbm_max_depth: tuple[int, ...] = (2, 3)                                           # 2x2x2=8
    knn_n_neighbors: tuple[int, ...] = (3, 5, 7, 9, 11, 13, 15)                       # 7
    # Baseline (CNN/spectrogram) training grid -- gives the budget-parity rule (below) a
    # real candidate set to count on the baseline side, not just WST-classical.
    baseline_learning_rate: tuple[float, ...] = (3e-4, 1e-3, 3e-3)
    baseline_weight_decay: tuple[float, ...] = (0.0, 1e-4)                            # 3x2=6
```

**Staged selection algorithm** (fits on inner-training subjects, scores on
inner-validation subjects — matching `implementation_plan.md`'s "session-level MAE ...
mean over inner-val subjects" selection rule exactly):

1. **Stage 1 (feature axes).** For each candidate in
   `reduction × channel × tiling × log_branches × range_gate_m` (10 GHz: 2×2×3×3×2 =
   72 combos; 77 GHz: 1×1×3×3×1 = 9 combos), **fit** the `stage1_anchor_model` (ridge,
   `alpha = 1.0`, non-tuned) on each inner fold's **inner-training** subjects, then
   **score** on that fold's **inner-validation** subjects' session-level MAE. Average the
   per-fold inner-validation MAE across the adaptive `GroupKFold` folds. The candidate with
   the lowest averaged inner-validation MAE carries forward.
2. **Stage 2 (model family + grid).** Fix the Stage-1 winning feature configuration. For
   each `model_families × ModelGridConfig[family]` combination (≤ `budget_k` = 12 combos per
   family), **fit** on inner-training, **score** on inner-validation, same averaging.
3. **Tie-break** (implementation_plan.md §LOSO harness, applied literally, via the ONE
   executable definition in §2.1a's `select_candidate`): lower inner-validation session-level
   MAE → lower `simplicity_rank` (the frozen ridge=0 < knn=1 < svr=2 < rf=3 < gbm=4 ranking)
   → lower `feature_dimension` (the candidate's pooled-WST vector length) → lower
   `inner_fold_variance` (std of the per-fold inner-validation MAE across the `GroupKFold`
   folds). Non-finite MAE/variance candidates are filtered before this runs (§2.1a).
4. **What M6 actually implements vs. what stays prose until M7.** No file in this milestone
   defines a real fitting algorithm — `eval/harness.py` is explicitly M7's. **M6 adds exactly
   one narrowly-scoped, non-modeling helper — `src/dehyd/eval/selection.py`** (§2.1a below) —
   a pure function over **already-computed** candidate scores (no data, no fitting, no
   cohort access), implementing only the tie-break comparison. The **behavioral** claim —
   "fit on inner-training, score on inner-validation, across real `GroupKFold` folds, with a
   fit-audit" — is verified at **M7**, where `eval/harness.py` will call this same
   `selection.py` function after doing the actual fitting.

### 2.1a `src/dehyd/eval/selection.py` — pure tie-break logic

**Responsibility.** The ONLY executable code this milestone adds beyond config and the
whitelist guard: a pure, stateless function that decides a winner among **already-scored**
candidates. It never fits a model, never sees a frame or a subject, and never reads cohort
data — it operates purely on numbers the (future) M7 harness will have already computed.
This keeps it inside M6's "no predictive computation" invariant while still being real,
testable code rather than a prose promise.

**Tie-break definition.** The prose tie-break (implementation_plan.md §LOSO harness:
"simpler model — fewer effective parameters / smaller feature dim") is made fully
executable via two deterministic components, matching the prose's two nouns as two
sequential tie-break levels:
1. **`simplicity_rank`** — a **frozen, config-level mapping**, not a fold-dependent count:
   `{"ridge": 0, "knn": 1, "svr": 2, "rf": 3, "gbm": 4}` (lower = simpler = preferred). A
   literal effective-parameter count isn't comparable across these families (KNN has no
   real parametric complexity in the same sense as ridge/svr/rf/gbm), so this is an ordinal
   ranking, not a raw count.
2. **`feature_dimension`** — the candidate's pooled-WST feature vector length, a
   **deterministic function of the reduction/channel/tiling/log point alone** (already
   measured and recorded in `wst_diagnostics_{10,77}ghz.csv`'s geometry columns — not
   something fit at scoring time), used as the secondary simplicity criterion.

**NaN/Inf handling.** A candidate with a non-finite `inner_val_mae` or a
non-finite/negative `inner_fold_variance` is **not comparable** (mirrors "non-evaluable
configs are skipped in selection, recorded" from Exp C's fold-viability rule) — it is
filtered out **before** the tie-break runs, deterministically (a finite-value filter, never
a NaN-involving sort). If **every** candidate is filtered out, `select_candidate` raises
rather than silently returning an arbitrary one.

```python
@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str          # opaque key identifying the config point
    inner_val_mae: float       # already computed by the CALLER (M7), not by this function
    simplicity_rank: int       # frozen family ranking (ridge=0 < knn=1 < svr=2 < rf=3 < gbm=4)
    feature_dimension: int     # pooled-WST vector length for this candidate's feature-axis point
    inner_fold_variance: float # std of per-fold inner-val MAE, already computed by the caller

def select_candidate(scores: list[CandidateScore]) -> CandidateScore:
    """Filters out any candidate with non-finite inner_val_mae or non-finite/negative
    inner_fold_variance (raises ValueError if nothing remains), then implements
    implementation_plan.md's LOSO-harness tie-break literally: lower inner_val_mae ->
    lower simplicity_rank -> lower feature_dimension -> lower inner_fold_variance.
    Deterministic; no randomness, no I/O, no model object anywhere in scope.
    """
```

**Acceptance (T-C6-stage).** Unit tests construct synthetic `CandidateScore` lists by hand
(no real features, no real model) and assert `select_candidate` picks the right winner
under: a clear MAE winner; an MAE tie broken by `simplicity_rank`; an MAE+`simplicity_rank`
tie broken by `feature_dimension`; a full tie broken by `inner_fold_variance`; a fully-tied
input (stable, deterministic output, e.g. first-in-list); an empty list (raises); a
candidate with NaN/Inf `inner_val_mae` or a non-finite/negative `inner_fold_variance` is
excluded from consideration, not compared; all-candidates-non-finite raises. **This is the
entire T-C6-stage test group for M6.** The claim that M7's real harness fits on
inner-training and scores on inner-validation is verified at **M7** against the real
`eval/harness.py`, not here.

**Budget parity — the owner-approved counting rule (Step 0 item 3).**
`implementation_plan.md` §D requires the same "≤K configs each" for WST-classical and every
learned-baseline family. Every family's degrees of freedom split into two kinds:
**representation-level** choices (WST: reduction/channel/tiling/log/gate; baseline: raw vs.
matched-preprocessing input) are **symmetric across families and exempt from K**, since both
sides get exactly one such choice-structure by design (WST's is richer because the WST
representation itself is the thing under study — ROADMAP explicitly frames the baselines as
deliberately simple, minimally-processed contrast points, not competing feature-engineering
efforts); **model/training-hyperparameter** choices (WST: `ModelGridConfig`'s per-family
grids; baseline: the `baseline_learning_rate × baseline_weight_decay` grid, 6 combos) are
**capped at ≤K = 12 for every family, WST or baseline, uniformly**. The physics baseline
(Exp D iii) has no hyperparameters — its scalar-to-target mapping is a per-fold closed-form
linear fit, not a searched configuration — so it is `K = 1` by construction, not by
exemption.

**Tuned-ε mechanics** (fully specified in `implementation_plan.md`'s WST-parameterization
section — transcribed here, not re-derived): `tuned_eps_k = 0.1`; `scale_o` =
median-over-training-subjects of (mean-over-that-subject's-eligible-training-sessions of the
existing `_prelog_scale` per-session value,
[extraction.py:71](src/dehyd/features/extraction.py#L71)) — subject-balanced, train-only,
applied as a fixed constant to every frame in the fold regardless of role; non-finite/
non-positive aggregate falls back to `log_epsilon = 1e-6`. See `implementation_plan.md`
§"WST parameterization" for the full statement.

**Both-band baseline config:**

```python
@dataclass(frozen=True)
class BaselineConfig:
    # Shared CNN/2D-CNN architecture (implementation_plan.md §D, both bands)
    cnn_channels: tuple[int, int, int] = (16, 32, 64)
    cnn_kernel: int = 7
    cnn_pool: int = 4
    cnn2d_channels: tuple[int, int] = (16, 32)
    cnn2d_kernel: int = 3
    cnn2d_pool: int = 2
    optimizer: str = "adam"
    lr: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)   # named: PyTorch/paper default, not tuned
    adam_weight_decay_default: float = 0.0           # overridden per-config by ModelGridConfig.baseline_weight_decay
    weight_init: str = "framework_default"           # PyTorch Conv1d/Conv2d default (Kaiming-uniform); not customized
    loss: str = "mse"                                # owner-confirmed (§3)
    batch_size: int = 16                             # owner-confirmed (§3)
    max_epochs: int = 200                            # owner-confirmed (§3)
    early_stopping_patience: int = 15                # owner-confirmed (§3)
    early_stopping_min_delta: float = 1e-4           # owner-confirmed (§3)
    checkpoint_metric: str = "inner_val_session_mae"
    checkpoint_direction: str = "minimize"
    frame_to_session_aggregation: str = "median"     # frozen, implementation_plan.md §D(i)
    # Raw/matched-preprocessing 1D inputs: robust per-channel standardization (median/MAD),
    # reusing preprocess/standardize.py -- matches the rest of the pipeline.
    raw_matched_standardize: str = "robust_per_channel"
    # Spectrogram inputs: a DIFFERENT rule -- train-only per-frequency mean/std (fit on
    # inner-training frames' spectrogram bins, applied to inner-val/outer-test unchanged),
    # NOT the robust per-channel rule above. Recorded in the M7 fit-audit like every other
    # fitted quantity (implementation_plan.md "Fit-on-train-only").
    spectrogram_standardize: str = "train_only_per_frequency_mean_std"
    spectrogram_hann: int = 64
    spectrogram_hop: int = 16
    spectrogram_nfft: int = 128
    # 10 GHz physics baseline (implementation_plan.md §D(iii))
    physics_target_range_m_10ghz: tuple[float, float] = (0.9, 1.5)
    physics_background_range_m_10ghz: tuple[float, float] = (1.5, 3.0)
    # 77 GHz -- raw/matched/spectrogram tensor definitions (A-M6-2, owner-approved)
    raw_reduction_77ghz: str = "mean_over_fast_time_and_rx"     # retains the CHIRP/slow-time axis
    raw_channels_77ghz: int = 1                                 # real, pre-range-FFT
    matched_input_77ghz: str = "chain_steps_1_5_single_rx_range_bin_mean"  # single fixed Rx=0, NOT Rx-fused
    matched_reference_rx_index_77ghz: int = 0                   # deterministic, non-tuned choice
    matched_channels_77ghz: int = 2                             # {real, imag}, post-range-FFT
    spectrogram_primary_channels_77ghz: int = 1            # STFT of the raw reduced real slow-time signal
    spectrogram_ablation_channels_77ghz: int = 2           # STFT(real), STFT(imag) of the single-Rx matched signal, stacked
    # Physics baseline, 77 GHz -- a bin-partition (DC vs any resolvable nonzero Doppler),
    # not a physiological rate split (the ~7.63 Hz FFT resolution can't resolve one).
    # Owner-confirmed worth reporting despite the coarseness (A-M6-2, Step 0 item 2).
    physics_static_band_bins_77ghz: tuple[int, int] = (0, 0)     # DC bin only
    physics_motion_band_bins_77ghz: tuple[int, int] = (1, 127)   # all other non-negative half-spectrum bins
    physics_prf_hz_77ghz: float = 1953.125                 # = 1/chirp_time_s, for the Doppler FFT bin width
```

Session-balanced sampling weight: per-frame weight = `1 / n_qc_passing_frames_in_that_session`
(both bands).

**New experiment schemas:**

```python
@dataclass(frozen=True)
class ExpBConfig:
    reuse_exp_a_search_space: bool = True        # A-M6-3
    objective: str = "equal_session_residual_mae"
    session_specific_variant_enabled: bool = True

@dataclass(frozen=True)
class ExpCConfig:
    # Cutpoint source: quantiles of family (a)'s OWN continuous L-predictor's IN-SAMPLE
    # predictions on inner-training sessions (i.e. the fitted regressor -- the same
    # search-space/selection procedure as Exp A, retargeted to L = -Delta_m%, evaluated
    # on the sessions it was fit on) -- NOT the true label quantiles (the cutpoints
    # threshold the *prediction*, per implementation_plan.md's "threshold the continuous
    # L predictor"), and NOT inner-validation/outer-test predictions (would be circular /
    # leak validation information into the cutpoints).
    cutpoint_source: str = "family_a_regressor_in_sample_predictions_inner_train"
    cutpoint_quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    cutpoint_min_separation: float = 1e-9        # repeated nudge on ties until strictly increasing
    # Inverse-frequency class weights (implementation_plan.md's existing rule, now an
    # executable formula): weight_c = n_inner_train_sessions / (n_classes * count_c),
    # computed per fold on inner-training data after QC. Passed as `sample_weight` to
    # any family-(a) model family that supports it (ridge/svr/rf/gbm); KNN does not
    # support sample_weight and is EXCLUDED from class-weighted fitting for Exp C
    # specifically (mirrors the Statistics section's own documented KNN exclusion
    # pattern for weight-incompatible estimators) -- stated explicitly, not silently
    # dropped.
    class_weight_formula: str = "inverse_frequency_inner_train"
    class_weight_unsupported_families: tuple[str, ...] = ("knn",)
    # Family (b) -- RESOLVED at Step 0 item 5 (A-M6-5, owner-approved 2026-07-25).
    # `statsmodels.miscmodels.ordinal_model.OrderedModel` (the literal proportional-odds/
    # cumulative-link model implementation_plan.md §C specifies) was verified directly:
    # its __init__/.fit() signatures carry no sample_weight/freq_weights parameter, and
    # it inherits from GenericLikelihoodModel (a generic MLE optimizer with no
    # observation-weighting mechanism) -- it cannot implement the required class
    # weighting. Per the owner's decision, family (b) is instead the **Frank-Hall
    # ordinal decomposition**: K-1 = 4 independent binary classifiers, each
    # `sklearn.linear_model.LogisticRegression` (already in the pinned stack -- no new
    # dependency), the k-th predicting `P(class > k)` for k in {0,1,2,3}, each fit with
    # `sample_weight` = the SAME inverse-frequency weights as family (a) (sklearn's
    # LogisticRegression.fit(..., sample_weight=...) is standard, verified API). Class
    # probabilities recovered by successive differences of the 4 cumulative
    # probabilities (P(class=k) = P(>k-1) - P(>k), clipped at 0 and renormalized if a
    # monotonicity violation occurs from independent per-threshold fits); prediction =
    # argmax class probability. This is a genuinely different statistical model from a
    # literal cumulative-link fit (separate per-threshold coefficients rather than one
    # shared slope with ordered cutpoints) -- an explicit, documented substitution
    # (implementation_plan.md A-M6-5), not a silent one.
    proportional_odds_impl: str = "frank_hall_ordinal_decomposition_sklearn_logisticregression"
    proportional_odds_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)   # sklearn LogisticRegression's C (inverse reg. strength)
    selection_metric_primary: str = "class_unit_mae"
    selection_metric_secondary: str = "qwk"

@dataclass(frozen=True)
class ExpEConfig:
    reduction_10ghz: str = "A"
    channel_10ghz: str = "mag"
    tiling_10ghz: str = "T1"
    log_10ghz: str = "off"
    gate_10ghz_m: tuple[float, float] = (1.0, 2.0)
    tiling_77ghz: str = "T1_77"                  # first/smallest Doppler tiling, Q=(8,4)@20ms
    log_77ghz: str = "off"
    model: str = "ridge"
    ridge_alpha: float = 1.0
    n_folds: int = 4
    fold_assignment: str = "sorted_subject_id_array_split"  # deterministic, no RNG

@dataclass(frozen=True)
class ExpFConfig:
    radar_representation_rule: str = "exp_a_selected_feature_config_per_fold"  # A-M6-4
    ridge_alphas: tuple[float, ...] = (0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)  # == ModelGridConfig.ridge_alphas
    clock_encoding: str = "session_index_one_hot"
    covariates_primary: tuple[str, ...] = ("age", "height", "baseline_mass", "bmi")
    covariates_sensitivity: tuple[str, ...] = ("age", "height")
    target_sensitivity: str = "signed_kg_change"

@dataclass(frozen=True)
class ExpGConfig:
    alpha_grid: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(21))
    alpha_tie_break: str = "closest_to_one"
    seed_pairing: bool = True
    objective: str = "subject_balanced_oof_mae"

@dataclass(frozen=True)
class StatsConfig:
    # Transcribes implementation_plan.md's ALREADY-approved §Statistics protocol in full
    # -- no redesign, every field names a choice that section already makes.
    confidence_level: float = 0.95
    bootstrap_b: int = 10000
    ci_method: str = "bca"
    ci_fallback: str = "percentile"               # when BCa is unstable at small N
    resample_unit: str = "subject"                # cluster bootstrap over N_eval subjects
    seed_collapse_additive: str = "average_per_subject_before_resample"     # per-subject MAE etc.
    seed_collapse_pooled: str = "average_metric_across_seeds_within_resample"  # pooled r/RMSE/QWK/adj-acc
    undefined_metric_skip_threshold_pct: float = 5.0   # >5% skipped -> CI flagged unreliable
    per_subject_pearson_r_min_sessions: int = 3   # descriptive only below this
    expb_aggregate_estimand: str = "session_weighted_equal_weight_per_session"
    expb_paired_test_estimand: str = "subject_weighted_complete_case_s1_s4"
    paired_test: str = "wilcoxon_signed_rank"
    holm_family_expb_per_session: int = 4          # S1-S4, exploratory
    holm_family_expf_primary: int = 2              # 3v1, 4v2 -- the ONLY Exp F Holm family implementation_plan.md defines
    expf_exploratory_correction: str = "none_reported_individually"  # 2v1, 4v3 -- "exploratory" per implementation_plan.md, with NO Holm-of-2 specified for them
    holm_family_baseline_per_family: int = 3        # CNN/spectrogram/physics, exploratory
    composite_baseline_comparison: str = "single_uncorrected"  # secondary #1, one comparison
    robustness_replicates_r: int = 200
    robustness_min_distinct_subjects: int = 4
    robustness_min_successful_replicates: int = 100
    robustness_ordinal_min_classes: int = 5

@dataclass(frozen=True)
class ProtocolFreezeConfig:
    option_b_peak_neighbors: int = 1
    option_b_mask_taper: bool = True
    fft_gate_transition_hz: float = 500.0
    qc77_min_in_band_energy_ratio: float = 0.30
```

**Acceptance.** Literal-pinning tests per new section; bad-value rejection tests;
cross-band structural rejection; a budget-count test proving every family's
model-hyperparameter grid is ≤`budget_k`; a StatsConfig round-trip test through
`config_to_dict`. (The Stage-1/2 inner-validation-scoring **behavior** is prose here,
verified against the real `eval/harness.py` at M7 — see §2.1a for what M6 actually tests.)

### 2.2 `configs/` — new YAML files

`configs/search_10ghz.yaml`, `configs/search_77ghz.yaml`, `configs/baselines.yaml`,
`configs/stats.yaml`, `configs/exp_{b,c,e,f,g_fusion}.yaml`, `configs/protocol_freeze.yaml`,
plus the `exp_a_regression.yaml:22` comment fix — all loading the dataclasses above.

### 2.3 Whitelist validator — `protocol_freeze_guard`

**Responsibility.** `canonical_spec_guard` (the existing 10 GHz feature guard,
[extraction.py:146](src/dehyd/features/extraction.py#L146)) requires
`preprocess.model_gate_m == (1.0, 2.0)` exactly (verified against
[config.py:119](src/dehyd/config.py#L119) and
[extraction.py:33](src/dehyd/features/extraction.py#L33)), so it would reject the approved
`(0.9, 3.0)` search candidate whenever a modelling entrypoint tries it. That guard is
correct and stays **exactly as-is** on the curated-artifact write path (`run_wst.py`) — it
is not weakened. For **modelling** entrypoints, `protocol_freeze_guard` instead:
1. Re-implements the same per-field comparison **locally** (a small, deliberately duplicated
   ~10-line loop against the public `CANONICAL_PREPROCESS`/`CANONICAL_WST` constants) for
   every `PreprocessConfig`/`WSTConfig` field **except** `model_gate_m`, which is checked for
   **membership** in `SearchSpace10GHzConfig.range_gate_m` instead of equality to the single
   canonical value. This duplication is a deliberate tradeoff (see §6) made specifically so
   `extraction.py` — explicitly frozen front-end code this milestone does not touch — needs
   no refactor.
2. Calls `canonical_spec_guard_77(config)` **unchanged** — verified that no `Preprocess77Config`/
   `QC77Config`/`WST77Config` field is itself a search axis (77 GHz's tiling/log are call
   arguments to `extraction_77.py`, not stored config, mirroring the 10 GHz reduction/channel
   pattern), so the strict 77 GHz guard is safe to reuse as-is for modelling too.
   (`model_gate_m` is the one axis that genuinely IS a stored `Config` field, so point 1's
   membership check on it is real config validation; reduction/channel/tiling/log branch are
   never stored anywhere in `Config` at all, which is exactly why they need the `active`
   mechanism below rather than a config-field check.)
3. Validates every M6-frozen section's genuinely single-valued fields (K, grids,
   `tuned_eps_k`, `ProtocolFreezeConfig`'s own constants) against their canonical default,
   by equality.
4. Raises with every deviating field named.

**The `active` parameter.** Which tiling, reduction, channel, log branch, or 77 GHz fusion
statistic is **actually used for a given fit** is a **call argument** to
`extraction.py`/`extraction_77.py` (mirroring the existing 10 GHz reduction/channel pattern,
"call arguments rather than config, so one config can serve every variant" —
[config.py:114](src/dehyd/config.py#L114)), never a stored config field. A guard that only
inspects `config` cannot see a call-time value, so an entrypoint that constructed an
out-of-whitelist call argument (e.g. a typo'd tiling name, or a gate never enumerated in
`range_gate_m`) would pass a config-only guard and still reach outer-fold evaluation.
**Fixed:** `protocol_freeze_guard` takes a second, **required-for-any-actual-fit**
parameter:

```python
def protocol_freeze_guard(config: Config, active: dict[str, object] | None = None) -> None:
    """active, when given, is the exact call-time protocol record for ONE fit: e.g.
    {"band": "10ghz", "reduction": "A", "channel": "mag", "tiling": "T1",
     "log_branch": "on_frozen_eps", "range_gate_m": (1.0, 2.0), "model_family": "ridge"}
    (77 GHz keys instead where reduction/channel/gate are fixed constants, not choices).
    Every key's value is checked for MEMBERSHIP in the corresponding
    SearchSpace{10,77}GHzConfig whitelist tuple (or equality to the fixed 77 GHz
    constants). active=None only validates the config-level frozen sections (a
    pre-flight check with no fit in progress yet) -- it is NOT a substitute for passing
    active on every real fit. M7's harness MUST call this with active= populated from
    the SAME candidate-enumeration loop that produced the fit (never from a
    freely-constructed value), immediately before any model fit or result write.
    """
```

This is defense-in-depth, not merely "trust the entrypoint's iteration logic": even if M7's
harness has a bug that lets a stray call-time value through, `protocol_freeze_guard(config,
active=...)` catches it structurally, because `active`'s values are checked against the
whitelist regardless of how the caller produced them.

**Acceptance.** A literal-pinning test on the whitelist; a rejection test per frozen field;
a test that both approved 10 GHz gates `(1.0,2.0)` and `(0.9,3.0)` pass the modelling guard
while an out-of-whitelist gate (e.g. `(0.5,1.5)`) fails, exercised via `active=`; an
entrypoint-level negative test for an invalid CALL-TIME reduction, tiling, log branch, or
77 GHz fusion statistic — not only invalid config fields: the config itself is canonical,
but `active` carries a bad value, and the guard still rejects it; one entrypoint-order test
proving a stub M7-style entrypoint calls `protocol_freeze_guard(config, active=...)` before
any result file is written; a regression test confirming `canonical_spec_guard`'s own
existing test suite is untouched.

---

## §3 Provisional constants (owner-confirmed 2026-07-25, Step 0 item 4)

Every value below was invented because `implementation_plan.md` leaves it unstated. None
were derived from running anything against this cohort's data. **The owner accepted the
whole list as-is** — no changes requested.

| Constant | Value | Basis |
|---|---|---|
| `budget_k` (both bands, all families) | 12 | A modest per-family cap; grids sized to fit under it |
| `tuned_eps_k` | 0.1 | 10% of the training-fold order-o scale, per the un-flooring rationale |
| Ridge/SVR/RF/GBM/KNN grids | see `ModelGridConfig` | Standard small grids, no domain tuning |
| `baseline_learning_rate` / `baseline_weight_decay` grid | see `ModelGridConfig` | Needed for the budget-parity reconciliation |
| `loss = "mse"` | — | Natural given MAE/RMSE are the primary metrics |
| `batch_size` / `max_epochs` / `early_stopping_patience` / `early_stopping_min_delta` | 16 / 200 / 15 / 1e-4 | Conventional small-dataset DL defaults |
| `adam_betas` / `weight_init` | (0.9,0.999) / framework default | Named, not customized |
| `proportional_odds_c_grid` | (0.1, 1.0, 10.0) | The Frank-Hall decomposition's per-threshold `sklearn.LogisticRegression` regularization grid (A-M6-5) |
| `physics_static_band_bins_77ghz` / `physics_motion_band_bins_77ghz` | bin 0 / bins 1–127 | The physically-realizable DC-vs-any-motion split (the ≈7.63 Hz FFT bin resolution rules out a rate-specific cutoff); owner-confirmed worth reporting |
| `stage1_anchor_ridge_alpha` | 1.0 | Arbitrary non-tuned anchor, never itself a reported candidate |
| Budget-parity counting rule | representation-level exempt, model-hyperparameter capped at K | Owner-approved reconciliation (Step 0 item 3) |
| `simplicity_rank` mapping (ridge=0 < knn=1 < svr=2 < rf=3 < gbm=4) | see §2.1a | The frozen, executable form of implementation_plan.md's "simpler model" tie-break criterion |

---

## §4 Tests

| Group | File | What it proves |
|-------|------|-----------------|
| T-C6-search | test_config.py | band-keyed search spaces validate/reject; cross-band structural rejection |
| T-C6-baseline | test_config.py | `BaselineConfig` literal-pins every architecture/training/normalization constant for **both bands**, including the 77 GHz raw/matched/spectrogram fields |
| T-C6-exp | test_config.py | `ExpBConfig`..`ExpGConfig` load, round-trip into `config_to_dict`, reject invalid literals; `StatsConfig` covers the full §Statistics protocol |
| T-C6-stage | test_selection.py | `select_candidate` (the ONLY real code M6 adds for this): full tie-break order incl. `simplicity_rank`/`feature_dimension`, NaN/Inf/negative-variance exclusion and all-non-finite raising — synthetic scores only, no fitting, no data |
| T-C6-budget | test_config.py | every family's model-hyperparameter grid is ≤`budget_k`, counted by the uniform rule |
| T-C6-guard | test_protocol_freeze.py | composes `canonical_spec_guard_77` unchanged; the local 10 GHz field check with the range-gate exception; both approved gates pass, an out-of-whitelist gate fails; entrypoint-order test; `canonical_spec_guard`'s own suite is unaffected; `active=` record validation rejects an out-of-whitelist call-time value even when the config itself is canonical |
| T-C6-ordinal | test_config.py / a small `test_frank_hall.py` | the Frank-Hall decomposition's threshold classifiers accept `sample_weight`; cumulative-probability differencing is clipped/renormalized correctly on a synthetic monotonicity violation |

`tests/test_no_leakage.py`: **zero changes**.

---

## §5 Definition of done

| ID | Criterion |
|----|-----------|
| D0 | ✅ **Step 0 cleared 2026-07-25** — all five items decided and recorded, including the `statsmodels.OrderedModel` verification |
| D1 | `uv run pytest` green, including all new T-C6-* groups |
| D2 | Every experiment family has a committed, validated config section reflecting the Step 0 decisions above |
| D3 | `protocol_freeze_guard` implemented, tested, composes `canonical_spec_guard_77` unchanged, uses the local range-gate-exception check for 10 GHz, and both approved gates demonstrably pass it |
| D4 | `tests/test_no_leakage.py` byte-for-byte unmodified since M1 |
| D5 | SECOND_CHAPTER.md §5 written, including the Step 0 decision record and the `OrderedModel` verification finding |
| D6 | `implementation_plan.md`'s A-M6-1/A-M6-2/A-M6-5 tags all read APPLIED, owner-approved, and this plan is consistent with that state |
| D7 | Freeze commit created and annotated-tagged, only when the owner asks |

---

## §6 What could go wrong (known traps)

1. **The `protocol_freeze_guard` local field-comparison duplicates ~10 lines already in
   `canonical_spec_guard`.** A deliberate tradeoff (§2.3) to keep `extraction.py` untouched;
   if a future session would rather see a small additive refactor there instead (a private
   shared helper both functions call, behavior-preserving for existing callers), that is a
   reasonable alternative to revisit, since it touches a file this milestone's ground rules
   call untouched.
2. **The Frank-Hall decomposition is not literally proportional-odds.** It's the
   owner-approved substitute (A-M6-5) because the genuine cumulative-link implementation
   available (`statsmodels.OrderedModel`) doesn't support the required weighting — but if
   the thesis chapter (SECOND_CHAPTER.md) describes Exp C's ordinal families, this
   distinction needs to be stated plainly there too, not glossed over.
3. **Budget-parity's representation/hyperparameter split (§2.1) is an interpretation the
   owner approved, not a literal restatement of `implementation_plan.md`.** Worth restating
   clearly in the eventual thesis methodology section, since a reader checking the main
   plan's text alone wouldn't derive this split on their own.
4. **The 77 GHz physics baseline is honestly coarse (DC vs any motion), not the
   physiologically-targeted feature originally proposed.** The owner has accepted this, but
   the eventual write-up should state the limitation plainly (it cannot distinguish
   breathing from any other motion) rather than let the "physics baseline" label imply more
   specificity than the ≈7.63 Hz Doppler resolution supports.
5. **`simplicity_rank`/`feature_dimension` must be computed identically wherever the
   tie-break runs.** If M7's harness ever recomputes a "feature dimension" differently from
   the recorded WST diagnostics geometry, the tie-break in `select_candidate` and the
   tie-break implied by `implementation_plan.md`'s prose could silently diverge.

---

## §7 Open items this milestone resolves or carries

**Resolves:** the 77 GHz in-band threshold decision; every A–G search space, baseline spec,
statistical protocol, and experiment design as committed, validated config; the frozen
protocol-constant whitelist; all five Step 0 decisions.

**Fixed here, NOT open at M7+:** the log-axis gating mechanism (A-M6-1), the 77 GHz baseline
design (A-M6-2), the budget-parity rule, and the Exp C proportional-odds implementation
(A-M6-5) — each would need its own prior authoritative amendment to revisit.

**Carries to M7:** `eval/harness.py`/`eval/metrics.py` consume the frozen configs, including
the staged algorithm (calling `eval/selection.py`'s `select_candidate` after real
fitting/scoring) and the tuned-ε mechanics (`k=0.1`, restricted `_prelog_scale`, zero-scale
fallback — all frozen here, computed fold-locally there); the Frank-Hall decomposition's
per-threshold sklearn fits and cumulative-probability recombination, implemented against the
already-verified `sample_weight` API; the `git.commit = None` IBEX provenance gap (M5 close)
before M7 re-extracts anything on the cluster.
