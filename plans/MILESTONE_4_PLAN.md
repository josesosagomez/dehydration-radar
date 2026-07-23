# MILESTONE 4 PLAN — WST feature extraction (kymatio)

_Task-level execution plan for milestone 4 **only** (ROADMAP §7.4; implementation_plan.md
"Build order" §4). Status: **APPROVED AND IMPLEMENTED (2026-07-23, branch
`v1_milestone_4`).** All eight build steps executed; definition of done §4 D1–D7 met —
`uv run pytest` → **396 passed / 12 skipped**, `--realdata` → **407 / 1** (T18 only), and
`tests/test_no_leakage.py` byte-for-byte unmodified since M1 and green. Three review rounds
(19 comments, all applied) preceded implementation; this document is now a record of what
was built. See HISTORY.md for the per-step log and the two facts the build surfaced: the
`pool_stats` cohort hotspot (96 s → 11.7 s per session after vectorising) and the measured
order-2 coefficient scale (~1e-6) that falsifies the "ε on an O(1) scale" rationale — ε left
frozen, recorded as a finding, log on/off resolves it at M6._

This document adds the execution detail the main plan intentionally omits. It does not
restate design decisions; where a decision is needed it cites `plans/implementation_plan.md`
(the source of truth — for M4 chiefly §"WST parameterization", §"Analysis unit —
session-level primary", §"Library choices", and Build order §4), `CLAUDE.md`, or
`ROADMAP.md`. Anything here that goes beyond those documents is flagged in §6.

Milestones 1–3 are done, committed and pushed (`f3fbade`, `395eb62`, `a27d8ce`): config
system, 10 GHz loader, ground truth + cross-checks, manifest, frozen QC screens, session
eligibility, nested-LOSO splitter, provenance, the 77 GHz audit, and the full
preprocessing sequence — **319 tests green (329 with `--realdata`)**. M4 builds on those
and duplicates none of them.

**The input population is fixed by M2/M3:** `eligible_frames(manifest_qc)` = **7168
QC-passed frames across 73 eligible sessions and 16 evaluable subjects**, and
`preprocess_cube(cube, pre, reduction=…, channel=…)` → **float64 [n_frames × C × 470]**
(C = 1 for `mag`, 2 for `iq`) is the M4 input. M4's functions are population-agnostic
pure functions; they touch real data only through their `--realdata` tests and the cohort
diagnostic run (§2.6).

**Owner decisions already made (recorded here so they are not re-litigated):**

1. **M4 work happens on branch `v1_milestone_4`** (already created from `v1_milestone_3`;
   HANDOFF.md is its bootstrap).
2. **The cohort-level diagnostic script `experiments/run_wst.py` is IN scope** (owner
   decision 2026-07-23): the frozen WST meets the full eligible cohort once, findings
   recorded — the same first-contact pattern as `run_qc.py` (M2) and `run_preprocess.py`
   (M3). The reusable manifest→features wiring lives in `src/dehyd/features/extraction.py`
   (§2.5); `run_wst.py` is a thin CLI over it.
3. **The T18 torch mutation leg stays skip-marked until M6**, even though torch enters
   the environment here at M4. torch arrives only for the WST cross-backend check; the
   torch *fit path* does not exist until the harness (M6).

---

## §0 Scope and ground rules

**In scope:**

- **`uv add torch`** — CPU wheel locally (the cross-backend check needs a torch frontend).
  **scipy stays pinned `<1.17`** — kymatio 0.3.0 imports `scipy.special.sph_harm`, removed
  in scipy 1.17 (already pinned since M1; torch must not drag scipy forward). `test_env.py`
  gains a torch import; the pyproject comment for torch is updated from "deferred" to "M4".
- `src/dehyd/features/__init__.py`, `wst.py`, `pooling.py`, `extraction.py` — the WST
  parameterization, pooling, and the reusable manifest→features extraction wiring of
  implementation_plan.md §"WST parameterization" and §"Analysis unit" (§2.2–§2.5).
- `WSTConfig` **field validation** in `src/dehyd/config.py` (the fields are *consumed* for
  the first time at M4 → M2's rule: a bad value must fail at config load, not deep inside
  kymatio) + a new `backend` field; `configs/wst.yaml` mirror (§2.1).
- `experiments/run_wst.py` — one-command cohort diagnostic pass; writes a curated artifact
  under `<results_dir>/wst/` (§2.6).
- `tests/test_wst.py` (synthetic, private-data-free) + extensions to `tests/test_config.py`
  and `tests/test_env.py` + a one-file `realdata` end-to-end test (§3).
- Journal upkeep: HISTORY.md as steps/attempts resolve (departures + measured values
  logged); SECOND_CHAPTER.md §3 "WST features" at milestone close.

**Explicitly out of scope (deferred to their milestones):**

- `eval/harness.py`, `eval/metrics.py`, `models/`, any modeling or model selection — **M6+**.
  M4 *implements* the tiling and log alternatives; it never *selects* among them.
- **The T18 torch mutation leg** stays skip-marked until M6 (owner decision above). torch
  enters the env here, but the torch *fit* path it guards does not exist until the harness.
- 77 GHz WST (`wst_extract77.m` tilings, slow-time Doppler I/Q) — **M9**. `wst.py`'s
  functions are written shape/fs-agnostic so M9 reuses them (different N, fs, tilings only).
- `configs/ibex.yaml`, `scripts/ibex/` — first IBEX milestone. M4 is CPU-local: the cohort
  diagnostic (§2.6) is minutes of CPU, WST on 7168 frames × the enumerated combos.
- **No caching of WST features.** M6 recomputes on the fly per run — the full-cohort cost
  is minutes, and a cache with hash invalidation is complexity deferred until it is actual
  friction (the same reasoning as the M2 no-QC-cache and M3 no-preprocess-cache decisions).
  In-run reuse of a deterministic intermediate *within one session's extraction* (§2.5's
  variant helper) is not caching — nothing persists past the call.

**The milestone-4 invariant, protected above all (CLAUDE.md §Hard invariants;
implementation_plan.md §"Analysis unit", §"Fit-on-train-only"):**

> **WST + pooling is a deterministic per-frame function of (one frame's preprocessed
> channels, frozen tiling constants). It contains NO fitted quantities and NO cross-frame,
> cross-session, cross-subject, or cross-role statistics — except the single *declared*
> session aggregation (concat of per-frame mean + median), which is a fixed pair of
> statistics, not a fitted transform.**

Concretely: the scattering transform is a fixed filter bank (data-independent); the
order-aware log is a fixed pointwise map; pooling is fixed moments over fixed segments;
the session aggregation is two fixed statistics with a frozen concat order. Nothing is
estimated on one set and applied to another, so M4 introduces **no leakage vector** and
nothing here enters the CV loops. Where this could silently break, and is forbidden to:

- fitting any scaler/PCA/whitening on the scattering coefficients (that is a fitted
  transform and must live inside the M6 CV harness — it is not part of M4);
- computing a pooling statistic across frames of a session *before* the declared
  mean+median aggregation, or across sessions/subjects (the per-frame pooled vector is a
  pure function of that one frame);
- selecting a tiling, the log switch, or the backend from data anywhere in M4.

The choice axes that ARE searched later — tiling {T1,T2,T3}, log {on,off} — are **explicit
call arguments**, exactly like reduction {A,B} and channel {mag,iq} at M3, selected only
inside inner CV at M6. M4 implements the alternatives; it never picks between them.

**Every WST alternative is classified NOW, before M6 (and locked at the M5 config
freeze).** Three kinds, and nothing may migrate between them after results exist:

- **Inner-CV search axes** (selected per outer fold on inner folds only): **tiling
  {T1,T2,T3}** and **log {on, off}** — exactly the main plan's search space, no additions.
  Call arguments, not config, so one config serves every variant (the reduction/channel
  precedent).
- **Frozen protocol constants** (neither searched nor ablated): `max_order = 2`,
  `log_epsilon = 1e-6`, the **order-aware log rule** (orders 1–2 logged, order 0 linear),
  the pooling definitions (moments, segment split, element order), the session-aggregation
  concat order (mean then median), and the kymatio instantiation options
  (`out_type='array'`, default averaging/oversampling). Configurable only so a run's YAML
  is a complete record and tests can drive behavior; non-default values are rejected by the
  §2.6 artifact guard and (from M6) by the harness at their frozen values.
- **Implementation choice validated by equivalence** (a fourth category unique to M4):
  **`wst.backend ∈ {numpy, torch}`**, default `numpy`. This is *neither* a search axis
  *nor* an ablation — it must not change any reported number. **The policy is scoped
  precisely to WST FEATURE GENERATION: the WST features in every reported artifact —
  the M4 curated diagnostic and all later reported feature/model results that consume
  WST — are computed with the numpy kymatio frontend** (tolerance-equivalent is not
  bit-identical, so exactly one frontend must own reported feature values or reruns
  become incomparable). The torch kymatio *frontend*'s role is (i) the cross-backend
  validation itself and (ii) unreported feature work — smoke tests, GPU
  experimentation — permitted only after the numpy-vs-torch test passes under the
  frozen agreement policy (§2.2), with the backend recorded in provenance. **This
  constrains `wst.backend` only, NOT PyTorch as a modeling framework**: the ROADMAP's
  reported 1D/2D-CNN baselines are torch-trained (on IBEX) as specified, consuming
  their raw/spectrogram inputs or numpy-generated WST features, and T18 exists
  precisely to protect that reported torch *fit* path at M6. If a later milestone
  (e.g. 77 GHz fusion compute) needs torch-frontend WST features in a reported
  artifact, that is an explicit owner decision revising this policy — never a reading
  of an ambiguity.

`tests/test_no_leakage.py` stays **byte-for-byte unmodified since M1** and green throughout
(preprocessing at M3 did not touch it; WST + pooling are per-frame and unfitted, so neither
does M4).

**Ground rules:** work on `v1_milestone_4`; commits only when the owner asks — build steps
*write and verify* artifacts, they never commit them; HISTORY.md written continuously as
attempts resolve (failures kept, newest-first); superseded material to `archive/` with a note.

---

## §1 Build sequence — exact order and why

Tests land in the same step as their module. HISTORY.md gets **at least** one entry per
resolved step; every failed or superseded attempt inside a step gets its own entry.

| # | Step | Why this position |
|---|------|-------------------|
| 1 | **Env:** `uv add torch` (CPU wheel); confirm scipy still `<1.17` and kymatio still imports; `test_env.py` gains the torch import; update the pyproject torch comment (deferred → M4) | Everything downstream can import torch; the pin interaction (torch must not pull scipy ≥ 1.17) is verified before any feature code |
| 2 | **Config:** `_build_wst` field validation (`max_order`, `log_epsilon`, new `backend`) + `configs/wst.yaml` mirror + `tests/test_config.py` extensions | The modules read these; schema and validation first, exactly as M1–M3 did |
| 3 | `src/dehyd/features/wst.py` — instantiate each tiling, **measure** padding / output shape / `n_paths` / `meta()`; record in HISTORY.md; the (frozen) border-effect handling + the batched transform contract + structural tests | Pure functions; everything downstream consumes the scattering output shape, which must be measured first |
| 4 | Order-aware log + the finiteness battery (mag/iq × orders 0/1/2 × log on/off) | Depends on step 3's coefficient shapes; the plan mandates this exact battery |
| 5 | `src/dehyd/features/pooling.py` + pooling/flatten/aggregate tests (hand-computed, element order pinned, the ≥2-sample segment-std rule and nominal-vs-effective dimensions asserted) | Consumes step 3–4 coefficient tensors; the session aggregation is the analysis-unit primitive |
| 6 | Cross-backend (the frozen `backend_agreement` formula, §2.2), shift stability (the fully frozen T-W7 fixture), determinism tests; the one-file `--realdata` end-to-end test | Only meaningful once 3–5 exist; the cross-backend test validates the torch WST frontend (never used for reported feature generation) against the canonical numpy frontend |
| 7 | `src/dehyd/features/extraction.py` (the reusable session/batch wiring) + `experiments/run_wst.py` (thin CLI over it); **run on the real eligible cohort**; write and verify the diagnostics artifact; record actual numbers in HISTORY.md | First contact of the frozen WST with all 7168 frames; produces the §2.6 findings SECOND_CHAPTER §3 needs |
| 8 | Journal close-out: SECOND_CHAPTER.md §3 "WST features"; final HISTORY.md entry; §6 amendments applied to `plans/implementation_plan.md` | CLAUDE.md write-cadence rules; closing the milestone requires the distilled account |

---

## §2 Per-file specifications

Format per file: **Responsibility** (single) · **Public API** · **Frozen parameters** ·
**Acceptance criteria**.

### 2.1 Config: `src/dehyd/config.py` + `configs/wst.yaml`

**Responsibility.** The M4-consumed WST constants, validated at build time. M2's rule:
once a field is actually consumed, a bad value must fail loudly at config load, not surface
as a confusing error deep inside kymatio's filter-bank construction. Today `_build_wst`
only rejects a `tilings` override and passes the rest through (`WSTConfig(**section)`); M4
adds real field validation because these fields are now consumed.

**`WSTConfig` today (unchanged tilings, frozen constants):**

```python
tilings = (WSTTiling(q=(10,4), invariance_ms=0.20),   # T1
           WSTTiling(q=(8,2),  invariance_ms=0.30),    # T2
           WSTTiling(q=(6,2),  invariance_ms=0.40))    # T3   -- frozen, YAML override rejected
max_order   = 2
log_epsilon = 1e-6
```

**New field:**

```python
backend: str = "numpy"   # "numpy" (primary/default) | "torch" -- implementation choice,
                         # not a search axis or ablation; validated by the cross-backend test
```

**Field-level validation (build-time, `ConfigError` on violation):**

- `max_order` integer ∈ {1, 2} (0 would drop all wavelet paths; > 2 is unsupported by the
  design — the plan fixes `max_order = 2`, and this is a frozen constant, but the bound is
  asserted so a typo fails loudly);
- `log_epsilon` > 0, finite (it is a divide/log guard; ≤ 0 would make `log(S + ε)` non-finite);
- `backend` ∈ {"numpy", "torch"} (string; anything else rejected — the `_choice_field`
  precedent used for `gate_method`/`standardize`);
- `tilings` override still rejected (unchanged from today).

Add `BACKENDS = ("numpy", "torch")` beside the existing `GATE_METHODS`/`STANDARDIZE_METHODS`
tuples. `configs/wst.yaml` mirrors `backend` with a provenance comment stating its §0
classification (implementation choice validated by cross-backend equivalence, default numpy).

**No cross-field validation is needed at config load** — J, T_samples, padding and output
shape are **derived and measured from the instantiated filter bank at build time** (§2.2),
never precomputed in config. (This is deliberate: the main plan forbids hard-coding a padded
length or a `padded_len / 2^J` estimate.)

**Acceptance.** `tests/test_config.py`: `backend` default present and overridable;
`max_order`/`log_epsilon`/`backend` bounds rejected (max_order 0/3/float, log_epsilon 0/−1,
backend typo); `tilings` override still rejected; provenance (`config_to_dict`) carries the
new field automatically.

### 2.2 `src/dehyd/features/wst.py`

**Responsibility.** The kymatio parameterization — implementation_plan.md §"WST
parameterization". Pure, **shape/fs-agnostic** functions (M9 reuses them at fs = 500 kHz
with the 77 GHz tilings): the ms→samples→(J,T) mapping, filter-bank instantiation with
**measured** padding/shape, the forward transform, and the order-aware log. No I/O, no
model, no fitted state.

**Public API.**

```python
def t_samples(invariance_ms, fs_hz) -> int
    # round(invariance_ms * 1e-3 * fs_hz). At fs = 520834: 0.20 ms -> 104,
    # 0.30 ms -> 156, 0.40 ms -> 208. Realized invariance within <0.2% of requested
    # (rounding to the nearest sample); the error is recorded per tiling.

def octaves_j(t_samples_value) -> int
    # ceil(log2(T_samples)) so the largest wavelet scale covers the averaging support.
    # 104 -> 7, 156 -> 8, 208 -> 8   (T1 J=7, T2/T3 J=8).

def build_scattering(tiling, wst_cfg, *, n_in, fs_hz) -> object
    # kymatio Scattering1D(J=octaves_j(t_samples(tiling.invariance_ms, fs_hz)),
    #                      shape=(n_in,), Q=tiling.q, T=t_samples(...),
    #                      max_order=wst_cfg.max_order, out_type='array').
    # The frontend comes from wst_cfg.backend ALONE -- "numpy" ->
    # kymatio.numpy.Scattering1D; "torch" -> kymatio.torch.Scattering1D. There is NO
    # separate backend argument, so a call site can never disagree with the config
    # (provenance records wst_cfg.backend and that IS what ran); the cross-backend
    # test builds its second frontend via dataclasses.replace(wst_cfg, backend=...).
    # n_in defaults to the preprocessed length 470 (534 - 2*edge_trim). NOTHING about
    # padding or output length is passed in or assumed -- kymatio computes it from the
    # filter bank. The border-effect UserWarning (emitted for ALL THREE tilings on the
    # pinned stack -- J=8 as well as J=7 at shape 470) is handled per the frozen
    # decision below (asserted present, never silenced).

def scattering_shape(scattering) -> dict
    # The MEASURED geometry, read back from the instantiated object and scattering.meta():
    #   pad_left, pad_right, padded_len, n_paths, n_time (output time length),
    #   and per-path metadata (order, j, xi) as arrays in kymatio's canonical path order.
    # This is what the tests assert and what wst_spec records -- never a formula.

def wst_spec(wst_cfg, pre_cfg, *, fs_hz=None) -> dict
    # Per tiling: {requested_ms, t_samples, realized_ms, realized_error_frac, J, Q,
    #             **scattering_shape(...)}. Plus backend and max_order. Goes into
    # provenance extras (run_wst.py) and HISTORY.md -- the main plan's "record the
    # (requested ms, realized samples, J) triple and the approximation error" requirement,
    # extended with the measured padding/shape the plan also mandates.

def scatter_frames(frames, scattering) -> np.ndarray
    # frames: float64 [N x C x n_in] (a batch of preprocessed frames; C=1 mag, 2 iq).
    # Returns float64 [N, C, n_paths, n_time]. THE BATCHED CONTRACT IS THE PRIMARY
    # API: the N*C signals are folded into kymatio's leading batch dimension and
    # scattered in ONE call, because per-frame calls are ~9-10x slower (reviewer
    # benchmark on the pinned stack: 0.0976/0.0549/0.0387 s/frame single-frame vs
    # 0.0111/0.0050/0.0039 s/frame at batch 20 for T1/T2/T3 -- ~137 min vs ~14 min
    # projected over the cohort combos; re-confirmed at build). Each channel is still
    # scattered independently (the plan's "real & imag scattered separately") -- the
    # batch dimension changes throughput, never semantics: T-W16 asserts batched
    # output == stacked single-frame output BIT-IDENTICALLY, per backend. Chunk
    # unit = one session (<= 100 frames = <= 200 signals; padded float64 input well
    # under 10 MB, T1 output ~8 MB -- no further chunking needed; the memory math is
    # stated in the docstring). numpy backend: float64 throughout. torch backend:
    # kymatio's torch filter bank is FLOAT32 (float64 input raises), so the torch
    # path runs float32 and returns float64 numpy -- the measured dtype policy (§2.2),
    # pinned by T-W9 (numpy-f64 vs torch-f32 within the strict tolerance).

def scatter_channels(channels, scattering) -> np.ndarray
    # Single-frame convenience: channels [C x n_in] -> [C, n_paths, n_time]; defined
    # as exactly scatter_frames(channels[None])[0]. Kept for tests and diagnostics.

@dataclass(frozen=True)
class AgreementResult:
    passed: bool
    max_elementwise_ratio: float  # max over elements of |a-b| / (atol + rtol*max(|a|,|b|))
    rel_l2: float                 # ||a-b||_2 / max(||a||_2, ||b||_2, 1e-12)
    policy: str                   # which named policy evaluated this

def backend_agreement(a, b, *, policy="float64") -> AgreementResult
    # THE frozen cross-backend criterion -- one helper, imported by T-W9 AND the
    # realdata check, so the formula literally cannot diverge between them:
    #   elementwise  |a - b| <= atol + rtol * max(|a|, |b|)   (symmetric), AND
    #   aggregate    rel_l2 <= rtol.
    # NO free tolerance arguments: `policy` selects one of exactly TWO entries in a
    # frozen table -- "float64" (rtol 1e-4, atol 1e-8; the default and the only
    # policy usable without the owner-approved fallback decision) and
    # "float32-fallback" (rtol 1e-3, atol 1e-5; §2.2 dtype policy, owner-gated) --
    # so the gate cannot be loosened at a call site. Input contract: RAISES on
    # unequal shapes, empty arrays, non-finite values, or dtypes violating the
    # policy (the "float64" policy requires float64 on both sides).
    # Returns the MEASURED components alongside the verdict -- a bare boolean could
    # not feed the HISTORY.md / SECOND_CHAPTER record: tests assert `.passed` (and
    # the policy identity) and RECORD `.max_elementwise_ratio` / `.rel_l2`.
    # The atol floor handles near-zero coefficients (where a pure relative test is
    # undefined or noise-dominated); the aggregate L2 bounds the overall error. The
    # gate applies to the RAW scattering tensors AND to the pooled vectors under BOTH
    # log states -- raw agreement does not by itself bound post-log agreement near
    # zero (|dlog(S+eps)| ~ |dS|/eps for S ~ 0), so the consumed representations are
    # checked directly.

def apply_order_log(S, meta, wst_cfg, *, log_on) -> np.ndarray
    # Order-aware log (implementation_plan.md, "Averaging / log"):
    #   log_on=False -> S unchanged.
    #   log_on=True  -> orders 1 and 2: log(S + wst_cfg.log_epsilon);
    #                   order 0: LEFT LINEAR (never logged -- S0 = x * phi is a signed
    #                   low-pass of the median/MAD-standardized input and can be negative;
    #                   log of a negative is the exact bug this rule prevents).
    # `meta` supplies the per-path order so the mask selects orders 1/2 vs 0 by path;
    # RAISES if S's path count disagrees with `meta` (a metadata reorder or mismatched
    # tiling must fail loudly, never be absorbed into a plausible-looking output).
    # `log_on` is a CALL ARGUMENT (an inner-CV axis at M6), not config.
```

**Frozen semantics.**

- **The three tilings and their (Q, invariance_ms) come from `WSTConfig`** — never
  re-hardcoded in `wst.py`. `t_samples`/`octaves_j` derive T and J from those.
- **Padding and output shape are MEASURED, never assumed** (the main plan's hard rule and
  the HANDOFF's "Known M4 concern"). We do **not** hard-code 512 or estimate `n_time` as
  `padded_len / 2^J`. At build time we instantiate the **pinned** kymatio `Scattering1D`
  for each tiling and read back `pad_left`/`pad_right`, `padded_len`, `n_paths`, `n_time`,
  and per-path `meta()` — those observed values are what the tests assert and `wst_spec`
  records. *(Reviewer-sampled values on the pinned stack, to be re-confirmed at build —
  the M2 77 GHz-shape pattern: T1 → (742 paths, 7 time), T2 → (466, 3), T3 → (349, 3).)*
- **The border-effect decision is FROZEN NOW, before any measurement: kymatio's native
  padding is ACCEPTED for all three tilings.** `Scattering1D(shape=(470,))` warns *"signal
  support is too small to avoid border effects"* — on the pinned stack for **all three
  tilings** (J=8 as well as J=7). There is no post-measurement design choice left open:
  the tilings and the 470-sample input are frozen by the main plan, and no mitigation is
  on the table at M4. The border-energy measurement is **descriptive only**, with its
  formula frozen so it is well-defined: b = the T-W7 distance d between the per-path
  global time-mean vectors of the standardized T-W7 fixture and of the same fixture with
  its first 32 and last 32 samples set to zero (32 = the EdgeTrim constant, a fixed
  reference edge width). b is recorded per tiling in HISTORY.md and SECOND_CHAPTER §3 as
  a characterization of how much edge content reaches the averaged coefficients — it has
  **no threshold and gates nothing**. If the owner finds a recorded value alarming, any
  change is an explicit owner decision before the M5 freeze (the M3 "finding, never a
  retune" doctrine), never a silent mitigation. A test asserts the warning **is** emitted
  for **each** tiling (`pytest.warns`) so a future kymatio change that alters padding
  behavior fails loudly rather than silently changing the features.
- **`out_type='array'`** (frozen) so the output is a dense `[n_paths × n_time]` array with
  a stable path order matching `meta()`, which the pooling element order depends on.
- **dtype policy (frozen, stated in the docstring; resolved against the pinned stack at
  step 1, owner-approved).** MEASURED FACT: kymatio's **torch frontend is float32-only**
  — its filter bank is float32 and float64 input raises `TypeError: Input and filter must
  be of the same dtype`. So the plan's original "torch in float64" is not achievable, and
  the pre-declared fork triggered. **Owner decision (2026-07-23): keep the strict
  tolerances; the float32 fallback is NOT adopted.** The numpy frontend runs float64; the
  torch frontend runs float32 (necessarily); the cross-backend check compares
  numpy-float64 against torch-float32 up-cast to float64 — the only achievable comparison
  — under the strict "float64" policy (rtol 1e-4, atol 1e-8). This passes with margin
  (measured max elementwise ratio 0.044/0.009/0.004 for T1/T2/T3, relative L2 ≈ 1e-7), so
  no loosening is needed: torch's float32 accumulation across the scattering depth stays
  well inside the 1e-4 relative bound. The "float32-fallback" named policy (rtol 1e-3,
  atol 1e-5) remains defined in `backend_agreement` but is **unused** — it would require a
  separate owner decision to invoke, and the data shows it is unnecessary. No tolerance is
  ever widened after seeing a failing comparison.

**Acceptance.** §3 structural tests green: each tiling's measured `n_paths`/`n_time`/padding
pinned as regression values (reviewer-sampled values above re-confirmed); `meta()` contains
orders 0/1/2 with order-1 `xi` monotone; the J/T_samples arithmetic matches independent
computation; the border warning is asserted present **for all three tilings**; batched ≡
stacked single-frame bit-identically (T-W16); `wst_spec` reports every value from
config/measurement, none re-hardcoded.

### 2.3 `src/dehyd/features/pooling.py`

**Responsibility.** implementation_plan.md §"WST parameterization" (Feature families) and
§"Analysis unit — session-level primary": collapse a per-frame scattering tensor to a
per-frame feature vector (two families), and aggregate a session's per-frame vectors to the
one-vector-per-session analysis unit. Pure functions.

**Public API.**

```python
def pool_stats(S, meta) -> np.ndarray
    # S: [C x n_paths x n_time] (post-log). Per channel, per path: statistics over
    #   (a) the global series, (b) the first half [0 : n_time//2],
    #   (c) the second half [n_time//2 : n_time].
    # DEGENERATE-SEGMENT RULE (frozen, metadata-only): a segment contributes its MEAN
    # always, its STD (ddof=0) only if the segment has >= 2 samples. The rule depends
    # on n_time ALONE -- identical for every frame/session/subject, so no
    # data-dependence and no leakage vector. Rationale: with the measured shapes,
    # T2/T3 have n_time = 3, whose 1-sample first half would make the first-half std
    # IDENTICALLY ZERO for every path, frame, and subject -- a structurally dead
    # column that a train-fold transform would otherwise discover silently.
    # Effective stats per (channel, path): 6 when both halves have >= 2 samples
    # (T1, n_time = 7), 5 when one half is a single sample (T2/T3, n_time = 3) --
    # pinned per tiling by test as both NOMINAL (6) and EFFECTIVE counts.
    # Returns a 1-D vector in a FIXED, documented element order:
    #   channel (0..C-1) -> path (kymatio meta() order) -> segment (global, first,
    #   second) -> statistic (mean, then std where defined). ddof=0 frozen (the
    #   meanstd_standardize precedent; no reference constraint exists).
    # Requires n_time >= 2 (each half nonempty); RAISES otherwise, and RAISES on an
    # S-vs-meta path-count mismatch (a metadata reorder must fail loudly).

def flatten_series(S) -> np.ndarray
    # Raw-flattened scattering series, fixed order channel -> path -> time. DIAGNOSTIC /
    # DL family only (implementation_plan.md: "raw-flattened ... diagnostic only ...
    # session-aggregated before any classical session-level metric"). Never a classical
    # session-level feature on its own -- but it IS exercised end to end at M4: routed
    # through aggregate_session in the synthetic tests (T-W11), the realdata test, and
    # the run_wst.py finiteness/dimension diagnostics (§2.6), so the family M4 claims
    # to implement is proven wired, not just laid out.

def feature_layout(meta, n_time, n_channels) -> tuple
    # PER-FRAME pooled-vector metadata: one (channel, path, segment, statistic)
    # tuple per element, generated deterministically from the SAME inputs
    # pool_stats uses -- including the >=2-sample segment-std rule -- in exactly the
    # pooled order. len(feature_layout(...)) == len(pool_stats(S, meta)) is
    # asserted by test (T-W10). The building block of the session layout below.

def session_feature_layout(meta, n_time, n_channels, *, family) -> tuple
    # THE layout downstream code actually consumes: per-element metadata of the
    # aggregate_session OUTPUT -- which is TWICE the per-frame length -- with an
    # `aggregate` field (frame_mean | frame_median) prepended to each per-frame
    # tuple, mean block first then median block, exactly matching the frozen concat
    # order. family "pooled" -> (aggregate, channel, path, segment, statistic);
    # family "flat" -> (aggregate, channel, path, time_index) -- column attribution
    # exists for BOTH families. M6 train-fold transforms and Exp E interpretability
    # consume THIS (a per-frame-only layout would leave half the model columns
    # unmapped); nobody reconstructs the tiling-dependent layout independently.
    # Length == 2D and element-for-element agreement with the session vector are
    # asserted by test (T-W10) for both the 5- and 6-stat cases.

def aggregate_session(frame_vectors) -> np.ndarray
    # frame_vectors: [n_frames x D] per-frame vectors (pooled OR raw-flattened) for one
    # session's QC-passed frames. Returns [2D] = concat(mean over frames, median over
    # frames) -- the frozen session-level analysis unit (implementation_plan.md
    # §"Analysis unit": "concatenating the per-frame mean and the per-frame median ...
    # both statistics, fixed"). Concat order is mean-block THEN median-block, frozen.
    # NOT a fitted transform and NOT a competing aggregation choice.
    # Input contract: RAISES on n_frames = 0, on non-2-D input, and on non-finite
    # values. n_frames = 1 IS allowed by the primitive (mean = median = that row;
    # the population-agnostic function does not encode cohort policy) -- cohort
    # eligibility (>= ceil(0.5 x frames) surviving QC) makes a one-frame session
    # unreachable from the cohort path, and that separation of concerns is stated in
    # the docstring.
```

**Frozen semantics.**

- **Element order is a contract**, derived once from the recorded `meta()` path order and
  documented in the module. A test hand-builds a tiny `S` and asserts the exact vector
  layout, so a reordering (which would silently scramble features across the CV loop) fails.
- **Segment split** is `n_time // 2`: first half `[0 : n_time//2]`, second half
  `[n_time//2 : n_time]`. For odd `n_time` the second half carries the extra sample — frozen
  and stated (not a tunable). The ≥2-sample segment-std rule above is part of the frozen
  pooling definition and is mirrored into the main plan by amendment A-M4-2.
- **`pool_stats` is the classical session-level feature family**; `flatten_series` feeds the
  DL paths and is session-aggregated the same way before any classical metric touches it.
- Session aggregation operates on the **pooled** per-frame vectors, giving up to 80
  observations (16 × 5 minus QC-ineligible sessions), one per (subject, session), matched
  1:1 to the Δm% target — the pseudo-replication fix.

**Acceptance.** §3 pooling tests green: `pool_stats` on a crafted `S` equals the hand
computation with the exact documented layout — including a crafted `n_time = 3` case where
the first-half std is absent by the rule (effective 5 stats/path, no structural-zero column
in the output) and an `n_time = 7` case with all 6; a deliberately permuted-order reference
implementation fails the comparison (the layout is pinned, not incidental); nominal vs
effective dimension both asserted per tiling; `feature_layout` matches the pooled vector
element-for-element in both cases and `session_feature_layout` matches the session
vector (length 2D, mean-then-median, both families); `aggregate_session` equals
concat(mean, median) on a crafted matrix, accepts `n_frames = 1` (returns concat(v, v)),
and rejects empty/non-2-D/non-finite input; odd-`n_time` split verified; `n_time < 2` and
path-count-mismatch raise.

### 2.4 `src/dehyd/features/__init__.py`

Re-export the public API (`wst_spec`, `build_scattering`, `scatter_frames`,
`scatter_channels`, `backend_agreement`, `AgreementResult`, `apply_order_log`,
`pool_stats`, `feature_layout`, `session_feature_layout`, `flatten_series`,
`aggregate_session`, and §2.5's `extract_session_features` / `extract_session_variants` /
`SessionVariantResult` / `canonical_spec_guard`) so M6 imports from `dehyd.features` —
the `preprocess/__init__.py` precedent.

### 2.5 `src/dehyd/features/extraction.py` — the reusable manifest→features wiring

**Responsibility.** The session-level extraction sequence as **library code in `src/`,
not inside a CLI script** — so the M6 harness imports a module and the repository's
dependency direction (library ← scripts, never the reverse) holds. `run_wst.py` and the
M6 harness are both thin consumers of this module, and tests import it directly without
`sys.path` manipulation.

**Public API.**

```python
def extract_session_features(cube, pre, wst_cfg, *, reduction, channel, tiling,
                             log_on, family) -> np.ndarray
    # One eligible session's QC-passed frames -> ONE session vector:
    #   preprocess_cube(cube, pre, reduction=..., channel=...)       [N x C x 470]
    #   -> scatter_frames(..., build_scattering(tiling, wst_cfg))    [N x C x P x t]
    #   -> apply_order_log(..., log_on=log_on)
    #   -> pool_stats / flatten_series per frame (family: "pooled" | "flat")
    #   -> aggregate_session(...)                                    [2D]
    # A linear, followable composition of the public §2.2-§2.3 functions -- no logic
    # of its own beyond sequencing (the pipeline.py precedent), so testing it is
    # composition-testing, not re-testing the pieces. The single-variant REFERENCE
    # API: correct by construction, used by tests and one-off analyses.

@dataclass(frozen=True)
class SessionVariantResult:
    vectors: dict       # {(tiling_index, log_on, family): session_vector [2D]}
    prelog_scale: dict  # {tiling_index: (v0_median, v1_median, v2_median)} -- the
                        # §2.6 frozen PRE-LOG order-0/1/2 scale statistics, computed
                        # from the SAME shared raw tensor in the same pass (they
                        # cannot be recovered from pooled/flattened aggregates)
    shapes: dict        # {tiling_index: (n_paths, n_time)} echoed for auditability
    all_finite: bool    # every vector and statistic finite

def extract_session_variants(cube, pre, wst_cfg, *, reduction, channel)
        -> SessionVariantResult
    # The COHORT-LOOP form: preprocess ONCE per (reduction, channel), scatter ONCE
    # per tiling, then derive all four (log {on,off} x family {pooled,flat}) session
    # vectors AND the pre-log diagnostics from each shared raw scattering tensor.
    # Calling extract_session_features per combo would recompute the same unfitted
    # raw tensor four times (and the preprocessing twelve times), turning the
    # ~14-min batched cohort projection into ~56 min of WST alone -- and computing
    # the §2.6 pre-log statistics OUTSIDE this pass would force run_wst.py to
    # duplicate extraction logic, re-scatter, or reach into undocumented
    # intermediates, breaking the thin-CLI and scatter-once contracts. This is
    # IN-RUN reuse of a deterministic intermediate within one session's extraction
    # -- NOT the deferred persistent feature cache (§0). T-W18 asserts bit-identical
    # equality of `.vectors` with the single-variant API AND of `.prelog_scale`
    # with the manual §2.6 formula chain, for every combination.

def canonical_spec_guard(config) -> None
    # The §2.6 primary-artifact guard as a PURE LIBRARY FUNCTION (M3 kept its guard
    # inside the script; moved to src at M4 so guard tests are plain imports).
    # Raises -- naming every deviating field -- unless config.preprocess equals the
    # complete canonical PreprocessConfig() AND config.wst equals the canonical
    # WSTConfig() INCLUDING backend == "numpy" (§2.6: numpy is the canonical
    # artifact backend).
```

**Acceptance.** A composition test asserts `extract_session_features` equals the manual
chain of the public functions for representative (reduction × channel × tiling × log ×
family) combinations; `extract_session_variants` equals the single-variant API for
**every** combination, computing each preprocessing pass and each scattering exactly once
(T-W18); the guard tests (T-W14) import from `dehyd.features` directly.

### 2.6 `experiments/run_wst.py` — cohort diagnostic (first contact)

**Responsibility.** The one-command pass of the frozen WST over the real eligible cohort,
and the curated diagnostics artifact. **A thin CLI over §2.5's library module** — arg
parsing, the guard call, the session loop, CSV/provenance writing; no extraction logic of
its own. Same repeatable `--config` pattern as `run_qc.py`/`run_preprocess.py`.
**Diagnostic only — it selects nothing**: every constant is frozen before it runs, and a
surprising distribution is a *finding* for HISTORY.md and the owner, never a license to
retune (the M2/M3 doctrine).

**Behavior.** config → `canonical_spec_guard` → `load_ground_truth` → `build_manifest` →
`apply_qc` → `eligible_frames` → for each eligible session, load its file once and call
§2.5's `extract_session_variants` per (reduction {a,b} × channel {mag,iq}) — preprocessing
once per pair, scattering once per tiling {T1,T2,T3}, deriving all (log {on,off} × family
{pooled, flat}) vectors AND the pre-log scale statistics from the shared tensors, batched
per session (§2.2) — → per-session diagnostics read straight off the returned
`SessionVariantResult` (no second extraction path in the CLI) → print a cohort summary →
write
**`<results_dir>/wst/wst_diagnostics_10ghz.csv`** (config is the single output-path
authority) → `record_run(config, manifest_qc, folds=None, extra={"stage":
"milestone-4-wst", "analysis_role": "primary", "wst_spec": wst_spec(wst, pre),
"backend": wst.backend, <headline stats>})`.

**Canonical-spec guard (reused from M3, extended to WST).** The curated CSV is the
**primary** first-contact artifact. The guard — a pure helper, testable without a cohort
run — **refuses to run** (a loud error before any I/O) unless:

- the consumed `config.preprocess` equals the complete canonical `PreprocessConfig()` (the
  M3 guard, unchanged: butterworth + robust, gate 1–2 m, order 4, trim 32, etc.); **and**
- the consumed `config.wst` equals the canonical `WSTConfig()` frozen constants
  (`max_order = 2`, `log_epsilon = 1e-6`, the frozen tilings) **including
  `backend == "numpy"`**. Tolerance-passing torch output is equivalent to within the
  frozen policy, not bit-identical, so a second frontend writing any reported artifact
  would make reruns incomparable. This is the **single** policy stated in §0, scoped to
  WST feature generation — the WST features in every reported artifact, this CSV
  included, come from the numpy kymatio frontend; the torch *frontend* serves
  validation and unreported feature work, while PyTorch as a modeling framework is
  unaffected (the reported CNN baselines remain torch-trained, per ROADMAP Exp D). The
  same scoped wording appears in the amended main plan (A-M4-1), so "reported run"
  cannot mean two things across documents.

Diagnostics are computed for **all three tilings and both log states**, because the tiling
and log switch are inner-CV *axes* (not a single primary), so the first-contact artifact
characterizes the whole search space rather than pre-selecting within it. The reduction ×
channel variants are likewise all recorded (the M3 precedent). No modeling, no selection.

**Per-session columns (one row per subject × eligible session), each with a frozen
definition:**

- `n_eligible_frames`.
- **Feature dimensionality** per (tiling, channel): `n_paths`, `n_time`, pooled dimension
  — **nominal `C · n_paths · 6` AND effective `C · n_paths · s(n_time)`** (the ≥2-sample
  segment-std rule, §2.3) — raw-flattened dimension `C · n_paths · n_time`, and the
  session-vector dimensions `2D` for both families — from `wst_spec`/`scattering_shape`,
  not recomputed. (These are constant across sessions; recorded once in the summary and,
  for auditability, echoed per row.)
- **Finiteness confirmation** for every (reduction × channel × tiling × log × family)
  combination — **both** the pooled and the raw-flattened session vectors — the plan's
  "a test must assert every branch is finite" made a cohort-level check too: recorded as
  a boolean (any False is a loud finding). The raw-family session vectors are computed
  and checked but **not written** to the CSV (dimension ~10³–10⁴ per combo; the artifact
  stays curated) — the raw family's cohort evidence is its dimension + finiteness, not
  its values.
- **Coefficient-scale diagnostics (frozen formulas, PRE-LOG)** to characterize the
  standardized scale the ε = 1e-6 assumes (implementation_plan.md: "coefficients live on
  an O(1) standardized scale"). Computed on the **raw scattering tensor before any log**
  — orders 1–2 are non-negative there; measuring logged values would be circular, and a
  "magnitude" of negative logged coefficients answers a different question — and computed
  **once** per (reduction × channel × tiling): the same pre-log statistic serves both log
  states by construction. Exact reduction order, frozen: per frame, for order o ∈ {1, 2}:
  mean over time → per-(channel, path) scalar; mean over the paths of order o →
  per-channel scalar; mean over channels → per-frame scalar `v_o`. Order 0: the single
  order-0 path's signed time-mean, mean over channels. Session value = **median over the
  session's frames** of `v_o`. Exact zeros in the raw coefficients participate as
  ordinary values (this is a scale statistic; no flooring, no special case). **Computed
  inside the §2.5 variant pass and returned as `SessionVariantResult.prelog_scale`** —
  the CLI never recomputes, re-scatters, or reaches into intermediates. Pinned by a
  hand-computed test (T-W15). These answer "is ε = 1e-6 negligible against the
  coefficient scale?" — a finding, never a retune.
- **Timing:** wall-clock per session for the WST pass. Two things make the runtime claim
  real rather than aspirational: the batched contract (§2.2 — ~14 min projected vs ~137
  single-frame, reviewer benchmark re-confirmed here) and the §2.5 variant helper
  (scatter once per reduction × channel × tiling; deriving the four log × family variants
  per combo naively would quadruple the WST work to ~56 min). This column measures it and
  gives the M6 full-run cost and the IBEX-vs-local decision a data basis.

These are the concrete cohort numbers SECOND_CHAPTER §3 needs (feature dimensionality per
tiling, the realized invariance/J triple, the measured padding, coefficient scale, cohort
finiteness, and timing). Each formula gets a synthetic test before the cohort run.

**Acceptance.** Runs end-to-end on the real cohort in minutes; the CSV is written and
verified (re-read, row count = 73 eligible sessions); invoking with any non-canonical
preprocess or WST spec (an ablation switch, the 0.9–3.0 m candidate gate, a non-default
`max_order`/`log_epsilon`, or `backend: "torch"`) fails loudly before any I/O, naming the
deviating fields; actual distributions recorded in HISTORY.md; committed only on explicit
owner request.

---

## §3 Tests

All constants are read from the config object — a test that re-hardcodes 104/7/(10,4)/1e-6
would pass vacuously when config and code drift apart — with **one deliberate exception**:
T-W17 pins the frozen defaults as **literals**, because if both implementation and
expectation read `WSTConfig`, an accidental edit to a frozen default passes every test
(the self-reference trap). Exactly one test hard-codes the contract on purpose; every
other test derives its expectations from config or from explicit inputs. Synthetic
fixtures use seeded RNG
noise (`numpy.random.default_rng(<seed in the test>)`); **a noiseless tone is not a valid
fixture** (M2/M3 lesson: degenerate MAD, degenerate histograms — here also a WST with
near-zero higher-order paths).

**`tests/test_wst.py` (synthetic; no real data):**

| ID | Test | What it proves |
|----|------|----------------|
| T-W1 | `t_samples` and `octaves_j` from **explicit literal inputs** (not config): `t_samples(0.20, 520834) = 104`, `(0.30, 520834) = 156`, `(0.40, 520834) = 208`; `octaves_j` → 7/8/8; realized-error fraction < 0.2% each | The ms→samples→J conversions are exact, tested independently of the config they will consume (paired with T-W17, which pins the config defaults themselves) |
| T-W2 | `build_scattering` + `scattering_shape` per tiling: `n_paths`, `n_time`, `pad_left`/`pad_right`, `padded_len` MEASURED and pinned as regression values (reviewer-sampled on the pinned stack: T1 (742, 7), T2 (466, 3), T3 (349, 3) — re-confirmed at build); `meta()` has orders {0,1,2}; order-1 `xi` strictly decreasing (kymatio convention) | The output geometry is measured, not assumed; a kymatio change to padding/paths fails loudly |
| T-W3 | The border-effect `UserWarning` is emitted for **all three tilings** at shape (470,) — J=8 emits it as well as J=7 (`pytest.warns` per tiling); the recorded `pad_left`/`pad_right` are the measured values §2.2 documents; the descriptive border metric b equals its hand computation on the frozen fixture | The known M4 concern is surfaced for every tiling; padding pinned; the border metric is a frozen formula, not a judgment call |
| T-W4 | `scatter_channels`: mag → `[1, n_paths, n_time]`, iq → `[2, n_paths, n_time]` with real and imag scattered SEPARATELY (channel 0 == scatter(real), channel 1 == scatter(imag)); output float64 | The channel contract; I/Q are two independent scattering passes |
| T-W5 | **Finiteness battery** — every branch finite: mag & iq × orders {0,1,2} present in the output × log {on, off}, on seeded standardized input. No NaN/Inf anywhere | The plan's mandated "every branch produces finite values" |
| T-W6 | Order-aware log: with `log_on=True`, order-0 coefficients are UNCHANGED (linear) while order-1/2 equal `log(S + 1e-6)`; a crafted NEGATIVE order-0 coefficient survives (not logged, stays negative — the exact bug the rule prevents); `log_on=False` is identity | The order-aware rule is correct and order 0 is never logged |
| T-W7 | Shift stability, frozen before first run — **the gate is relative to the input, not an absolute cutoff**. Fixture: `sin(2π·f·n/fs + 0.7)` with `f = 4·fs/470` ≈ 4432.6 Hz — inside the 1–2 m band, an **integer** number of cycles in 470 samples, so a circular shift is an exact translation of the tone — plus Gaussian noise σ = 0.1 from `default_rng(0)` (**seed frozen: 0**), then `robust_standardize`. Shift: **circular** (`np.roll`; commutes with median/MAD standardization; zero-fill is deliberately NOT used — it conflates translation with border-content change, which the descriptive metric b measures separately) by **s = 8 samples**. Exercised: all three tilings × one real channel, linear (pre-log) domain. Distance: `d(v, w) = ‖v − w‖₂ / max(‖v‖₂, 1e-12)`. **Gate (a priori): `d(m(x_s), m(x)) ≤ 0.5 · d(x_s, x)` for every tiling**, with `m(·)` = the per-path global time-means — an averaging operator with support T ≥ 104 samples that fails to even HALVE the effect of an 8-sample shift is broken outright, whatever the border effects; plus the fixture-sanity anchor `d(x_s, x) > 0.2`, justified analytically before any WST runs (the tone's shift sensitivity is `2·sin(π·f·s/fs)` ≈ 0.42). **The absolute per-tiling d values are DESCRIPTIVE, never gates** — reviewer-measured on the pinned stack at seed 0: 0.05444 (T1), 0.13884 (T2), 0.15771 (T3); recorded in HISTORY.md, pinned as regression drift pins only; and the T2/T3 > T1 ordering is recorded as a border-effect **finding** (at J = 8 the padding transient dominates the larger nominal invariance scale — feeds the §2.2 border characterization and SECOND_CHAPTER §3). *(The round-1 absolute 0.10 gate is withdrawn: its constant C was never bounded and the measured 0.158 would fail it. The cutoff was NOT raised to clear the observation — the criterion was replaced by one justifiable independently of it.)* | A falsifiable stability claim that finite-record border effects cannot invalidate, an analytic fixture anchor, and honest descriptive characterization of the absolute drift |
| T-W8 | Determinism: two runs on the same seeded fixture (same backend) are bit-identical | No hidden RNG in the transform |
| T-W9 | **Cross-backend**: `backend_agreement(..., policy="float64")` — the frozen two-entry policy table, no free tolerances — passes numpy-vs-torch on a shared standardized fixture: on the RAW tensors AND on the pooled vectors under BOTH log states, all three tilings, both channels, float64 both sides. The test asserts `.passed` AND `.policy == "float64"`, and records `.max_elementwise_ratio`/`.rel_l2` (the HISTORY/SECOND_CHAPTER equivalence numbers); unequal shapes, empty arrays, non-finite values, and dtype-policy violations raise; the torch frontend is built via `dataclasses.replace(wst_cfg, backend="torch")` — the config is the only backend authority | The main plan's gate with an unambiguous, unloosenable formula whose measured margins are recorded, shared verbatim with the realdata check |
| T-W10 | `pool_stats` on a crafted `S`: equals the hand-computed vector in the EXACT documented element order (channel → path → segment → stat); ddof = 0 pinned (a ddof = 1 impl fails); a permuted-order reference fails the comparison; a crafted `n_time = 3` case yields **5 stats/path with no structural-zero column** (the ≥2-sample segment-std rule) and `n_time = 7` yields 6; nominal vs effective dimension both asserted; `feature_layout` has exactly the pooled vector's length and matches the documented (channel, path, segment, statistic) enumeration in BOTH the n_time = 3 and n_time = 7 cases; `session_feature_layout` matches the `aggregate_session` output element-for-element — length 2D, mean block then median block, both families, both the 5- and 6-stat cases; `n_time < 2` and S-vs-meta path-count mismatch raise | The per-frame AND session-level layouts are pinned contracts with machine-readable metadata, not incidental behavior |
| T-W11 | `flatten_series`: fixed channel → path → time order on a crafted `S`; length = C · n_paths · n_time; routed through `aggregate_session` → a finite `[2·C·n_paths·n_time]` session vector | Raw-flatten layout pinned AND the raw family wired end to end, not just laid out |
| T-W12 | `aggregate_session`: concat(mean, median) over a crafted `[n_frames × D]` matrix, mean-block then median-block, length 2D | The session analysis unit is exactly the frozen concat |
| T-W13 | Input-contract raises: wrong channel count, wrong length (≠ config-derived n_in), non-finite input, wrong ndim for `scatter_frames` batches; `aggregate_session` rejects empty (0-frame), non-2-D, and non-finite input and ACCEPTS `n_frames = 1` (returns concat(v, v) — the primitive is population-agnostic; cohort eligibility, not the primitive, forbids tiny sessions); `apply_order_log`/`pool_stats` reject an S-vs-meta path-count mismatch | The boundaries that would otherwise turn a missing session or a metadata reorder into a plausible-looking vector |
| T-W14 | `canonical_spec_guard` (imported from `dehyd.features`, §2.5): accepts canonical `PreprocessConfig()` + `WSTConfig()` (backend numpy); raises — naming the deviating field — on a preprocess ablation, the 0.9–3.0 m candidate gate, `max_order: 1`, `log_epsilon: 1e-3`, **and `backend: "torch"`** (numpy is the canonical artifact backend) (pure-function test, no cohort run) | A non-primary or non-canonical-backend run can never overwrite the primary curated CSV |
| T-W15 | Diagnostic formulas (§2.6) on synthetic frames: nominal/effective/raw dimensions equal their §2.6 definitions; the **pre-log** coefficient-scale statistic matches a hand computation with the frozen reduction order (time → paths-of-order → channels; frames-median), is identical under log on and off, and treats exact zeros as ordinary values; finiteness booleans behave as documented | The cohort numbers are the frozen formulas, testably — and the ε-scale question is answered pre-log, not circularly |
| T-W16 | Batched ≡ single-frame: `scatter_frames` on a seeded `[N × C × 470]` batch equals `np.stack` of per-frame `scatter_channels` calls **bit-identically**, numpy AND torch backends, all three tilings; a session-sized batch (100 frames) stays within the documented memory envelope | A batch dimension changes throughput, never semantics (the T-PP5 pattern); the §2.6 runtime claim rests on tested equivalence |
| T-W17 | **Frozen-defaults contract — the one deliberate literal test:** `WSTConfig()` equals, as hard-coded literals, tilings ((10,4), 0.20 ms), ((8,2), 0.30 ms), ((6,2), 0.40 ms), `max_order = 2`, `log_epsilon = 1e-6`, `backend = "numpy"`; and `PreprocessConfig().fs_hz = 520834.0` (the value the WST spec consumes) | An accidental edit to a frozen default cannot pass silently just because implementation and expectation read the same config (the self-reference trap) |
| T-W18 | `extract_session_variants().vectors` ≡ `extract_session_features` for **every** (reduction × channel × tiling × log × family) combination on a seeded synthetic cube, bit-identically; `.prelog_scale` equals the manual §2.6 formula chain on the same cube (consistent with T-W15's hand computation); and it computes each (reduction, channel) preprocessing pass exactly once and each tiling's scattering exactly once (call counts asserted via a counting monkeypatch) | In-run reuse changes cost, never values — and the CLI needs no second extraction path for its diagnostics |

**`tests/test_config.py` additions:** `backend` default and override; `max_order` (0, 3,
float, bool rejected), `log_epsilon` (0, −1, non-finite rejected), `backend` typo rejected;
`tilings` override still rejected; provenance carries `backend`.

**`tests/test_env.py` addition:** `import torch` in `test_pinned_imports`; the scipy pin
asserted on **parsed versions** — `packaging.version.Version(scipy.__version__) <
Version("1.17")` — never string comparison (lexicographic comparison calls `"1.9" ≥
"1.17"` even though 1.9 < 1.17 as versions; exactly the bug class parsed versions
prevent).

**`realdata` (in `tests/test_wst.py`):** one real file (`subject_1_8am.mat`): run QC
(`run_qc_cube`) → passing frames → `preprocess_cube` (a representative variant, e.g.
Option A / iq) → `scatter_frames` (batched) through all three tilings → order-log both
states → **both families** (`pool_stats` AND `flatten_series`) → `aggregate_session`.
Every output finite; shapes match `wst_spec` (nominal, effective, and raw dimensions);
re-run bit-identical; numpy and torch agree via `backend_agreement(policy="float64")` —
the **identical helper** T-W9 imports, so the formula cannot diverge between the
synthetic and real checks — with the returned `.max_elementwise_ratio`/`.rel_l2` printed
and recorded (the real-data equivalence numbers for SECOND_CHAPTER §3). Print the
feature dimensionality and pre-log coefficient-scale distributions.
**No expected-distribution assertion** — distributions are unknown until this runs, and
asserting them would be tuning by the back door (M2/M3 doctrine).

**`tests/test_no_leakage.py`: zero changes.** WST + pooling are per-frame and unfitted (§0);
T1–T19 and the reference procedure remain untouched and green; the T18 torch mutation leg
stays skip-marked (owner decision).

---

## §4 Definition of done

| ID | Criterion |
|----|-----------|
| D1 | `uv run pytest` green on a checkout with no private data (all new synthetic tests included, torch installed) |
| D2 | `uv run pytest --realdata` green, including the one-file WST end-to-end test; the T18 torch mutation leg still the only intentional skip |
| D3 | `uv run python experiments/run_wst.py --config configs/exp_a_regression.yaml` runs over the full eligible cohort, **writes and verifies** `<results_dir>/wst/wst_diagnostics_10ghz.csv` (73 rows); actual distributions recorded in HISTORY.md; committed only on explicit owner request |
| D4 | `tests/test_no_leakage.py` **byte-for-byte unmodified since M1** and green (`git diff f3fbade HEAD -- tests/test_no_leakage.py` empty) |
| D5 | HISTORY.md entries per resolved step, including: the measured padding/output shape per tiling, the frozen border-effect decision with the descriptive b values, the (requested ms, realized samples, J, error) triple, the measured T-W7 stability values (gate ratios + descriptive absolutes, with the T2/T3 > T1 border finding), the cross-backend `AgreementResult` components, the dtype/backend policy, and every logged departure — including the ROADMAP §3.3 pooling departure (A-M4-6) — each with its reason |
| D6 | SECOND_CHAPTER.md §3 "WST features" written at close: provenance of every parameter (the ms→(J,T) mapping, Q per tiling, max_order, ε, the order-aware log rule, pooling and session-aggregation definitions), the measured padding/border decision, the shift-stability characterization, the ROADMAP §3.3 pooling departure with its rationale, the cross-backend equivalence result (measured error components), and the cohort diagnostic findings |
| D7 | §6 amendments applied to `plans/implementation_plan.md`; the two documents consistent |

---

## §5 What could go wrong (known traps, pre-paid)

- **scipy pin interaction:** `uv add torch` must not pull scipy ≥ 1.17 (kymatio 0.3.0
  breaks — `sph_harm` removed). Verify at step 1 and pin the torch version in `uv.lock`;
  T-env asserts `scipy.__version__ < 1.17` after the add.
- **kymatio's torch frontend is float32-only** (MEASURED at step 1: float64 input raises
  `TypeError`). The cross-backend check therefore compares numpy-float64 vs torch-float32
  (§2.2) — it passes the strict tolerance with margin (max ratio ≤ 0.044), so the
  owner-declined "float32-fallback" named policy stays defined but unused. No tolerance is
  widened post-hoc; invoking the fallback would need a fresh owner decision.
- **No padding-formula assumptions:** never hard-code 512 or `padded_len / 2^J` — read
  `pad_left`/`pad_right` and `meta()` from the instantiated object (the main plan's rule;
  the HANDOFF's named M4 concern). T-W2/T-W3 pin the measured values.
- **`log(0)` / `log(negative)`:** the ε = 1e-6 guards orders 1–2 (non-negative moduli);
  **order 0 is never logged** because it is signed — T-W6 pins both. A test with a crafted
  negative order-0 coefficient proves the rule.
- **kymatio path order:** the pooling element order depends on `meta()`'s canonical order;
  `out_type='array'` is frozen and T-W10 pins the layout so a kymatio reordering fails
  loudly instead of silently scrambling features across folds.
- **Border-effect warning:** the decision is frozen BEFORE measurement — native padding
  accepted for all three tilings, border metric descriptive-only (§2.2) — never silenced
  by changing a frozen tiling. T-W3 asserts the warning per tiling (J=8 emits it too, not
  only J=7 — don't test only T1).
- **Short-tiling half segments are 1 sample** (measured n_time = 3 for T2/T3): the
  ≥2-sample segment-std rule (§2.3) exists precisely so no structurally-zero column
  ships; do not "restore the missing std" later without reopening A-M4-2 as an explicit
  owner decision.
- **Noiseless-tone fixtures are degenerate** (M2/M3 lesson): always add small seeded noise;
  a pure tone can give near-zero second-order paths and a misleading invariance measurement.
- **New package dir `src/dehyd/features/`:** check the staged list with `git add -An` before
  any commit — the M1 `.gitignore`-matches-at-any-depth trap (a stray `features/` ignore
  pattern would silently drop the package).
- **YAML 1.1 signed exponents (M1 trap):** any new float written in exponent form in
  `wst.yaml` needs a signed exponent (`1.0e-6`, not `1.0e6`); `log_epsilon: 1.0e-6` is
  already signed, but the rule holds for any future edit.
- **`tests/` is not a package** — absolute imports. The M4 guard and extraction wiring
  live in `src/dehyd/features/` (importable normally); only if a test must touch a
  CLI-side helper does `experiments/` go on `sys.path` (the M3 pattern — avoided here by
  design). The `.pytest_cache` redirect to `.cache/pytest` stays untouched.
- **Do not "fix" surprising cohort distributions.** If `run_wst.py` shows an odd coefficient
  scale, feature dimensionality, or invariance behavior, that is a finding for HISTORY.md and
  the owner — not a license to nudge a tiling, ε, or the pooling (§0 invariant; M2/M3 doctrine).

---

## §6 Flagged gaps in `implementation_plan.md` + proposed amendments

A-M4-1, A-M4-2 and A-M4-5 correct ambiguities in the main plan itself and were **applied
immediately** (2026-07-23, review round 1 — the A-M3-6/A-M3-7 precedent); A-M4-3 and
A-M4-4 add execution detail and apply at milestone close, as M3's A-M3-1..5 did:

- **A-M4-1 — Name the `wst.backend` config field and the canonical reporting backend
  (APPLIED to implementation_plan.md 2026-07-23, review round 1).** The main plan said
  "torch backend ... selectable by config" without naming a field, and "either may back
  a reported run" without fixing which backend produces curated artifacts —
  tolerance-equivalent is not bit-identical, so mixed-backend artifacts would be
  incomparable across reruns. Amended §"Library choices": field `wst.backend ∈
  {numpy, torch}` (default numpy), an implementation choice validated by the
  cross-backend test — neither a search axis nor an ablation; **numpy is the canonical
  backend for curated/reported artifacts**, torch a validated execution alternative
  recorded in provenance.
- **A-M4-2 — Pin the pooling and session-aggregation exact definitions (APPLIED to
  implementation_plan.md 2026-07-23, review round 1).** "Mean/std over global +
  first/second half" left ddof, the half split, element order, and concat order open —
  and with the measured n_time = 3 for T2/T3, a literal reading ships a
  structurally-zero first-half-std column for every frame, path, and subject. Amended
  §"WST parameterization" (Feature families): 6 nominal stats/path (mean, std ddof = 0
  over global/first/second half, split at `n_time // 2`), the **≥2-sample segment-std
  rule** (metadata-only, no data-dependence), element order channel → path(meta) →
  segment → stat, raw-flatten order channel → path → time, and session concat order
  mean-block then median-block.
- **A-M4-5 — Freeze the cross-backend agreement formula (APPLIED to
  implementation_plan.md 2026-07-23, review round 1).** "Agree to ≤1e-4 relative" was
  ambiguous — undefined or noise-dominated near zero, and silent about the log domain.
  Amended §"WST parameterization" (Cross-backend test): elementwise
  `|a−b| ≤ 1e-8 + 1e-4·max(|a|,|b|)` AND relative L2 ≤ 1e-4 with denominator floor
  1e-12, applied to the raw tensors and the pooled vectors under both log states,
  float64 both backends; the pre-declared float32 fallback (rtol 1e-3, atol 1e-5)
  additionally requires owner approval.
- **A-M4-3 — Repo-tree additions (F-item, like M2's F2 and M3's A-M3-4; apply at
  milestone close).** `experiments/run_wst.py` (+ its diagnostics artifact under
  `results/wst/`), `src/dehyd/features/__init__.py`, and
  `src/dehyd/features/extraction.py` (the reusable manifest→features wiring — in `src/`
  so the M6 harness never imports from a CLI script) are not spelled out in the main
  plan's tree; they follow its thin-CLI and package patterns. Note in the tree.
- **A-M4-4 — WSTConfig field validation (apply at milestone close; gap: the plan
  validates preprocess/QC fields at load but says nothing about validating the
  now-consumed WST fields).** Note that `max_order`, `log_epsilon`, and `backend` are
  validated at config load (the M2 "validate once consumed" rule), and the measured
  padding/shape is recorded, never validated as a precomputed constant.
- **A-M4-6 — The ≥2-sample segment-std rule is a deliberate ROADMAP departure, not
  only an ambiguity fix.** ROADMAP §3.3 literally specifies "pooled statistics
  (mean/std over global + two halves)" — six per path. For T2/T3 (measured
  n_time = 3) the rule yields **five**, dropping a 1-sample half's std that is
  identically zero and carries no information — only a structurally dead column.
  Handled exactly like the M3 no-window departure from ROADMAP §3.2: logged as a
  **deliberate departure with this rationale in HISTORY.md and SECOND_CHAPTER.md**
  at implementation; `ROADMAP.md` itself is amended only on explicit owner approval,
  never unilaterally. The `feature_layout`/`session_feature_layout` metadata (§2.3) is
  the departure's executable record — downstream code consumes the actual layout, not
  the nominal six.
- **A-M4-7 — Pre-registered third log branch `on+tuned-ε` (APPLIED to
  implementation_plan.md 2026-07-23, post-M4 owner discussion).** The cohort measured
  the ε rationale false (ε is 12–64 % of the order-2 scale, §HISTORY finding) and a LOSO
  diagnostic found the fold-to-fold order-2 scale stable to < 1 %, so a data-derived ε
  would be near-leakage-free here. Amended §"WST parameterization" (the log bullet) and
  the §"LOSO harness" inner-CV search-space enumeration to add a **third, mutually
  exclusive log branch** — `log on with a fold-local, scale-relative per-order ε rule`,
  computed train-only and selected on inner-validation — alongside `log off` and
  `log on + frozen ε`. Kept to one extra rule (not an ε grid) and **gated** on an
  order-2-usefulness pre-check so it cannot widen the N = 16 search for no reason.
  Pre-registered now, confirmed/frozen at the M5 config-freeze gate, decided at M6;
  **ε is unchanged at M4** (this is a committed experiment, not a retune). Applied to
  the main plan immediately (it corrects the main plan's false ε rationale), the
  A-M3-6/A-M3-7 precedent.
- Any further gap found while drafting the code is added here and (if it corrects the main
  plan itself) applied immediately per the cross-document consistency rule, as M3 did with
  A-M3-6/A-M3-7.

---

## §7 Open items this milestone resolves or carries

| Item | Status after M4 (planned) |
|------|---------------------------|
| WST feature extraction (Build order §4) | **Resolved** — the kymatio parameterization implemented and validated by path-structure, shift-stability, finiteness, and numpy/torch cross-backend checks |
| Cohort-level WST characterization | **Resolved** — `run_wst.py` diagnostics feed SECOND_CHAPTER §3 |
| torch in the environment | **Resolved** — added at M4 for the cross-backend check; the T18 torch mutation leg stays skip-marked until M6 (owner decision, unchanged) |
| kymatio border-effect warning at shape 470 (all three tilings) | **Resolved** — decision frozen pre-measurement (native padding accepted); descriptive border metric recorded in HISTORY.md; no frozen tiling changed |
| Config freeze (Build order §5) | **Next (M5)** — the complete A–G protocol freeze, including the WST search space (tiling × log) and the 77 GHz decisions |
| Adaptive / scale-relative ε | **Carried to M5** — the cohort measured ε = 1e-6 at 12–64 % of the order-2 scale (not the assumed O(1)), and a LOSO diagnostic showed the fold-to-fold order-2 scale stable to < 1 % (HISTORY.md). Pre-registered as a **third log branch** (`on+tuned-ε`, fold-local per-order rule) in implementation_plan.md §"WST parameterization"/§"LOSO harness search space" (A-M4-7), gated on an order-2-usefulness pre-check, confirmed/frozen at M5 and decided at M6. **ε unchanged at M4.** |
| `configs/ibex.yaml`, `scripts/ibex/` | **Still deferred** — no GPU work in M4; first IBEX milestone |
| 77 GHz flatline multiplicity (7/10 audited frames) | **Still parked** — owner decision at M5, never retuned from data |
| WST-feature caching | **Deliberately not built** — M6 recomputes on the fly; revisit only on real friction |
| Branch | M4 work on `v1_milestone_4` (pushed branches `v1_milestone_1..3`; nothing merged to `main`) |

---

_Review round 1 (2026-07-23): all 11 Codex comments accepted and applied — (1) T-W7
frozen before first run (fixture construction, circular shift s = 8, distance formula
with denominator floor, a-priori 0.10 threshold; measured values recorded as drift pins
only, never the acceptance criterion); (2) the border-effect decision made
non-discretionary (kymatio native padding accepted for all three tilings, the border
metric descriptive-only with a frozen formula, the warning asserted for J=8 as well as
J=7); (3) a batched extraction contract (`scatter_frames`, session-chunked, bit-identical
to stacked single-frame calls; the reviewer benchmark ~137 → ~14 min adopted as a
testable claim); (4) the ≥2-sample segment-std rule resolving T2/T3's structurally-zero
first-half std at the measured n_time = 3, with nominal AND effective dimensions pinned;
(5) reusable wiring moved to `src/dehyd/features/extraction.py` with `run_wst.py` reduced
to a thin CLI (dependency direction restored); (6) backend authority = config alone
(`build_scattering` takes no backend argument) and numpy fixed as the canonical
artifact/reporting backend; (7) the frozen `backend_agreement` formula (elementwise +
aggregate L2, atol floor, both log states, float64) with a pre-declared float32 fallback
requiring owner approval; (8) the raw-flattened family exercised end to end (synthetic,
realdata, and cohort dimension/finiteness diagnostics); (9) the ε-scale diagnostic fixed
pre-log with a frozen reduction order and zeros as ordinary values; (10) the
frozen-defaults literal contract test (T-W17) + parsed-version scipy pin; (11) the
completed input-contract battery including the n_frames = 1 policy. Amendments
A-M4-1/2/5 were applied to `plans/implementation_plan.md` immediately (they correct the
main plan itself — the A-M3-6/A-M3-7 precedent); A-M4-3/4 apply at milestone close._

---

_Review round 2 (2026-07-23): all 5 Codex comments accepted and applied — (1) the
round-1 absolute T-W7 gate (d ≤ 0.10) was **withdrawn, not raised**: Codex reproduced the
declared fixture (seed now frozen at 0) and measured d = 0.05444/0.13884/0.15771 for
T1/T2/T3, failing the cutoff whose constant C was never actually bounded; the gate is now
the a-priori-justifiable **relative-stability criterion** d(S) ≤ 0.5·d(input) with an
analytic fixture anchor (input shift sensitivity ≈ 0.42), the absolute values demoted to
descriptive record + drift pins, and the T2/T3 > T1 ordering logged as a border-effect
finding; (2) `extract_session_variants` added so the cohort loop preprocesses once per
(reduction, channel) and scatters once per tiling instead of recomputing the unfitted
tensor 4× (~56 → ~14 min), with bit-identical equality to the single-variant API and
call counts asserted (T-W18); (3) the canonical-backend policy made single and precise —
**every reported artifact across the study is numpy-backed**; torch = validation +
unreported work, revisable only by explicit owner decision — with identical wording in
both plan documents; (4) `backend_agreement` hardened: two named policies in a frozen
table (no free tolerances), full input contract, and an `AgreementResult` carrying the
measured error components that HISTORY/SECOND_CHAPTER record; (5) the ≥2-sample
segment-std rule re-classified as a **deliberate ROADMAP §3.3 departure** (A-M4-6, the
M3 no-window precedent — ROADMAP amended only on owner approval), with `feature_layout`
added as the machine-readable (channel, path, segment, statistic) metadata downstream
code consumes. Main-plan A-M4-1/A-M4-5 wording updated accordingly in the same round._

---

_Review round 3 (2026-07-23): all 3 Codex comments accepted and applied — (1)
`extract_session_variants` now returns a structured `SessionVariantResult` carrying the
variant vectors AND the frozen pre-log scale statistics (plus shapes/finiteness),
computed from the shared raw tensor inside the same pass, so the CLI reads its
diagnostics off the result instead of duplicating extraction, re-scattering, or reaching
into intermediates (pinned in T-W18 against the manual chain); (2) `session_feature_layout`
added — the per-element metadata of the actual `aggregate_session` output (aggregate ∈
{frame_mean, frame_median} prepended, mean block then median block, both the pooled and
flat families), tested element-for-element for the 5- and 6-stat cases, with the
per-frame `feature_layout` demoted to its building block; (3) the numpy-only policy
scoped explicitly to **WST feature generation with the numpy kymatio frontend** — it
constrains `wst.backend` only, not PyTorch as a modeling framework: the ROADMAP's
reported torch-trained CNN baselines (Exp D) and the T18-protected torch fit path at M6
are unaffected; the stale "lets torch back a reported run" build-step wording was
removed and the scoped wording mirrored in `implementation_plan.md` (A-M4-1/A-M4-5)._
