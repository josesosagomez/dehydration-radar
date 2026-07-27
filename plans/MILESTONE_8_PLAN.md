# MILESTONE 8 PLAN — Exp B, clock-decoupling: session-mean-residualized fluid-loss regression, both bands

## §0 Scope and ground rules

**Why this milestone exists (implementation_plan.md §Build order step 8).** Experiment A (M7) is
complete on the full cohort. Its headline result is **negative**: in both bands the radar
regressor lost to the session-index-only (time-of-day) baseline — 10 GHz mean difference +0.200
[0.145, 0.260], Wilcoxon p=3.05e-5; 77 GHz +0.216 [0.127, 0.296], p=7.6e-4 — with pooled
predicted-vs-actual r centred near zero in both bands. Exp A alone cannot distinguish "no signal"
from "signal present but swamped by the fasting-clock confound", because Δm% is structurally
confounded with time of day (subjects fast and dehydrate progressively across the measurement
day). Experiment B is the analysis designed to separate those two explanations: within a *fixed*
session every subject was measured at the same clock time but had lost different amounts of
fluid, so predicting the **session-mean-residualized** target tests whether radar tracks
between-subject fluid-loss variation rather than decoding the clock. ROADMAP.md:93-99 calls this
"the crucial evidence… a headline analysis, not a footnote."

Its **core design** — the residualization rule, the search-space reuse, the objective, the
baseline, the existence of both estimands, the Holm-4 family — was locked as part of
`config-freeze-v1` (357f734) **before any Exp A result was examined** (implementation_plan.md:232;
A-M6-3). That pre-registration is the entire scientific value of this milestone and is now the
thing most at risk, precisely because we have since seen a negative Exp A. M8's job is to
implement that core design exactly as written, **plus two narrow, explicitly disclosed
completions of it (A-M8-1, A-M8-2 — see the invariant below) that were decided after Exp A's
results were visible**, and report whatever the result gives — nothing beyond those two named,
disclosed completions may be chosen, tuned, or re-scoped in response to Exp A's outcome.

**Review.** This plan goes through the Codex⇄Claude review loop (`plans/review_prompt_claude.md`,
`plans/review_prompt_codex.md`) before any source is written, per owner decision (Step 0 item 4).
The review block is appended verbatim at the end of this file with `Status: AWAITING_CODEX`.

## Step 0 — owner decisions (RESOLVED 2026-07-27)

Four decisions were settled with the owner before authoring (via the planning session's
clarifying questions). Nothing downstream is conditional on them being reopened.

1. **The frozen protocol is internally ambiguous about what "the primary test" means, and this
   plan resolves it — but that resolution happened on 2026-07-27, after Exp A's full-cohort
   results were already visible, and the chronology stays explicit rather than being folded into
   "frozen before Exp A."** `implementation_plan.md:1218-1219` calls the equal-session aggregate
   "the single pre-specified primary test (radar vs baseline)", but that aggregate is
   *session-weighted*, while the Statistics section's test form is Wilcoxon (`:1213-1217`), which
   needs per-subject pairs — the *subject-weighted, complete-case* estimand. Both cannot be the
   primary at once, and the frozen text never says which wins. **Decision (A-M8-1): primary = the
   bootstrap CI on `aggregate(radar) − aggregate(baseline)`, session-weighted, matching the
   aggregate's own words; the subject-weighted complete-case Wilcoxon is reported alongside as a
   companion answering a different question, never conflated with the primary.** What A-M8-1
   actually decides is only which of two **already-fully-defined** quantities gets the "primary"
   label in the write-up — both were completely specified by the frozen text before this decision,
   and **both are computed and reported in every run regardless of the choice** (see
   `summarize_exp_b`'s output shape, §2.4): no computation, model, search, or data use changes
   based on it. That is the mitigating fact; it is not the same claim as "decided before Exp A",
   and SECOND_CHAPTER.md must state the chronology plainly (D11/D12), not elide it.
2. **Gating for the full-cohort run.** M7's owner-checkpoint pause existed to protect the
   `config-freeze-v1` guarantee — no outer score visible until explicit go-ahead, because seeing
   one would make every later design choice indirectly test-informed. Exp A's full run already
   spent that freeze; there is nothing left to blind for M8's compute step: Exp B's **core**
   design (what runs, what data it uses, how candidates are selected) was itself frozen before
   Exp A was seen, so seeing Exp A's numbers cannot leak into any of that (C13). A-M8-1/A-M8-2 —
   decided after Exp A's numbers were visible — are statistics-labelling and edge-case-reporting
   completions, not choices about what runs or what data is used, so they do not reopen the
   blinding question this decision is about. **Decision: keep the mechanism-only smoke as a
   cheap pre-flight (it catches bugs before a 16-fold IBEX job) and keep `--full-cohort` as an
   explicit flag, but drop the owner pause between them — the DoD is single-phase, not the M7
   two-phase split.**
3. **Scope of the session-specific variant.** `configs/exp_b.yaml` freezes
   `session_specific_variant_enabled: true` — the four session-specific models,
   `implementation_plan.md:722-724` describes them as "a thin secondary robustness variant only."
   **Decision: build it, as the last implementation step**, so it cannot delay or complicate the
   primary pooled analysis, and so the frozen config flag is honoured rather than deviated from.
4. **Review process.** **Decision: the full Codex⇄Claude loop**, not a direct owner approval and
   not a scoped-to-statistics-only review — Exp B is the crucial-evidence experiment, so the
   independent adversarial check against the frozen protocol matters most here, exactly as it did
   for M7 (21 comments, 3 escalated to the owner).

## Step 0b — owner-approval items (RESOLVED 2026-07-27)

One protocol-completion choice this plan raises was not settled by existing text and needed an
explicit decision before authoring the statistics spec, consistent with the M6/M7 process rule
(a technically-sensible completion of an authoritative-protocol gap clears an owner gate before
implementation — the C6-16 precedent).

- **O-B2 (A-M8-2) — a bootstrap replicate that empties a session: skipped-and-counted, not
  averaged over the survivors.** Also decided 2026-07-27, after Exp A's results were visible —
  named explicitly so the chronology in Step 0 item 1 is not read as applying only to A-M8-1. The
  plan's aggregate presupposes all four S1–S4 sessions ("recompute the four per-session residual
  MAEs and average them", `:1210-1212`); it does not say what happens if a subject-resample
  happens to omit every subject holding one session's only remaining rows. Silently falling back
  to a 3-session mean would change the estimand mid-bootstrap without recording it. **Decision:
  use the existing pre-registered skip-and-count machinery** (`n_skipped`, `unreliable`,
  `undefined_metric_skip_threshold_pct = 5.0`) — that replicate is dropped from the bootstrap
  distribution and counted, exactly as an already-non-finite metric value is handled elsewhere in
  `metrics.py`. **This is not computation-neutral (C10):** on the replicates where the edge
  fires, skip-and-count genuinely changes which replicates enter the bootstrap distribution
  compared with a silent 3-session fallback, and so can move the CI — it is a real, if narrow and
  rare-firing, protocol completion, not a relabelling. It is adopted because it is the only
  treatment consistent with how every other undefined-metric value is already handled in
  `metrics.py`, not because it was chosen to favour any particular result; at ~16 subjects with
  ≥12 eligible per session it is expected to be ~0% of replicates, and if it is not, that is
  itself information worth surfacing rather than hiding.

---

**In scope:**
- `src/dehyd/eval/metrics.py` — `per_session_residual_mae`, `equal_session_residual_mae`,
  `holm_adjusted`, an extracted `_cluster_bootstrap_over_rows` shared bootstrap loop, and
  `session_weighted_bootstrap` built on it.
- `src/dehyd/models/baselines.py` — `session_means` (the single train-only μ_s computation),
  `fit_session_mean_baseline`, `predict_session_mean`.
- `src/dehyd/eval/harness.py` — the **one** structural edit: a keyword-only `score_fn=None`
  threaded through `_fit_score_inner`, `_score_candidates_on_fold`, `_final_refit`,
  `run_nested_candidates`, plus an optional `session_idx` field on `FeatureBundle`.
- `src/dehyd/eval/exp_b.py` (new) — the full Exp B composition, mirroring `exp_a.py`'s shape:
  session-spine construction restricted to S1–S4, the residualizing feature provider, the
  picklable per-fold worker, fold-parallel orchestration, out-of-fold assembly, reporting, and
  the session-specific secondary variant.
- `experiments/run_clock_decoupling.py` (new) — the CLI entrypoint, mirroring
  `run_regression.py` exactly.
- `scripts/ibex/run_exp_b.sbatch` (new) — cloned from `run_exp_a.sbatch`.
- `tests/test_exp_b.py`, `tests/test_run_clock_decoupling.py` (new), plus additions to
  `tests/test_metrics.py`, `tests/test_baselines.py`, `tests/test_harness.py`.
- Both full-cohort Exp B runs (10 GHz, 77 GHz) on IBEX.
- HISTORY.md entries per resolved step; SECOND_CHAPTER.md §7 written from the full results.

**Explicitly out of scope (deferred to M9+):**
- Experiment C (ordinal classification), Experiment D (baselines: 1D-CNN, spectrogram+2D-CNN,
  physics power-ratio) — specs already frozen at M6, built on this same harness later.
- Experiment E (interpretability) — depends on the Exp B model as its primary target
  (`implementation_plan.md:927-934`) and therefore must follow M8, not be folded into it.
- Experiment F (confound check), Experiment G (cross-band fusion), Experiment H (the full
  cross-experiment statistics chapter).
- Any change to the frozen search space, the frozen statistical constants
  (`bootstrap_b=10000`, `ci_method="bca"`, `confidence_level=0.95`, …), or the M6 protocol-freeze
  guard's whitelist.
- The optional run-startup QC caching flagged (not done) at the end of M7's HANDOFF.

**The milestone-8 invariant, protected above all:**

> Exp B's **core design** — the residualization rule, the search-space reuse, the objective, the
> baseline, the existence and definitions of both estimands, the Holm-4 family — is frozen and was
> fixed before Exp A's results were ever examined (`config-freeze-v1`, before M7 ran). Two narrow
> completions of that design — **A-M8-1** (which already-defined estimand is billed as "primary")
> and **A-M8-2** (bootstrap behaviour on an empty-session replicate) — were decided on 2026-07-27,
> **after** Exp A's negative full-cohort result was visible, and that chronology is stated plainly
> rather than folded into "frozen before Exp A" (§0 Step 0 item 1, Step 0b). These two completions
> are not the same kind of thing (C10): **A-M8-1 is genuinely computation-neutral** — both
> quantities it might label "primary" are computed and reported in every run regardless of the
> label, so the decision changes prose, not numbers. **A-M8-2 is not computation-neutral** —
> choosing skip-and-count rather than a three-session fallback changes which bootstrap replicates
> enter the distribution on the (rare) occasions the edge fires, and can therefore move the CI. It
> is a post-Exp-A, outcome-affecting edge-case completion, retained under the pre-existing global
> conditional/exploratory label, justified because it is the only treatment consistent with how
> every other undefined-metric case is already handled in `metrics.py` — not because it is
> neutral. Beyond these two named, disclosed completions, M8 does not otherwise adjust, tune, or
> re-scope anything in response to Exp A's outcome, and does not treat a disappointing Exp B
> number as a reason to revisit a frozen choice.
> A result obtained by silently changing the protocol after seeing Exp A would be worthless
> regardless of its sign — the honest alternative is to say exactly what was fixed when, which is
> what §0, §6, and SECOND_CHAPTER.md (D11/D12) must do.

**Not reopened at M8** (fixed at M6/M7, would each need a prior authoritative amendment):
the harness's fold source (`eval/splits.py` only), tie-break source (`eval/selection.py::select_candidate`
only), the guard-before-every-fit contract (`protocol_freeze_guard`), the fit-audit shape, the
tuned-ε fold-local computation and its `k=0.1`/fallback `1e-6`, the store schema + fingerprint
doctrine, `tests/test_no_leakage.py`'s frozen body (T18's real body from M7 stands; no further
edits), the five model families + grids, `bootstrap_b=10000`/`ci_method="bca"`/`confidence_level=0.95`,
and Exp B's own frozen config (`reuse_exp_a_search_space=true`, `objective=equal_session_residual_mae`,
`session_specific_variant_enabled=true`).

**Ground rules.** Branch `v1_milestone_8` off `v1_milestone_7` @ `bda8e45`. HISTORY.md entries
written as each step resolves, not batched at the end — per CLAUDE.md. Superseded material moves
to `archive/`, noted in HISTORY.md. Tests are written alongside each per-file spec in §2, not
after the fact; a step is not "done" until its acceptance tests are green.

---

## §1 Build sequence — exact order and why

| # | Step | Why this position |
|---|------|--------------------|
| 0 | Write this plan; run the Codex⇄Claude review loop to closure | Owner decision (Step 0 item 4); no source is written until the protocol reading is adversarially checked — Exp B is the crucial-evidence experiment |
| 0.5 | **Propagate A-M8-1/A-M8-2 into `plans/implementation_plan.md` itself** (rewrite/annotate `:1218-1219`'s contradiction; document the empty-session bootstrap rule next to `:1210-1212`), stating the post-Exp-A chronology plainly, and commit that doc-only change | (C9, moved from the previous draft's step 11) Both amendments are already fully decided at Step 0/0b, before any M8 source is written, so there is no reason to defer this edit — and a real reason not to: doing it here means it is already part of every later commit, including the step-8.5 clean commit, so no *second* doc-only commit ever lands after the stores have been rebuilt/validated against that commit, which would otherwise invalidate the commit-match doctrine (C1) for the final milestone state |
| 1 | Pin current behaviour: capture `run_nested_candidates`'s `inner_scores.tobytes()` + selected candidate + `test_score` on a fixed synthetic-store fixture, and `subject_cluster_bootstrap_pooled`'s CI on a fixed fixture | Steps 4 and 2 both claim "no behaviour change to Exp A's path" — that claim is unverifiable without a byte-for-byte pin captured *before* either edit |
| 2 | `metrics.py` — `per_session_residual_mae`, `equal_session_residual_mae`, `holm_adjusted`, extract `_cluster_bootstrap_over_rows`, add `session_weighted_bootstrap` | Pure functions, no project dependencies; they *define* the objective everything downstream is built around. Re-assert the step-1 pooled-bootstrap pin immediately after the extraction |
| 3 | `models/baselines.py` — `session_means`, `fit_session_mean_baseline`, `predict_session_mean` | The provider (step 5) consumes `session_means`; this depends only on numpy + `FitRecord`, no harness change needed |
| 4 | `eval/harness.py` — the `score_fn` thread + `FeatureBundle.session_idx` + `_score` helper | **The one risky edit in the milestone.** Isolated in its own commit, between two green states, with a real non-default scorer already available (step 2) to prove the hook actually changes behaviour when given one. Immediately re-run the step-1 pin plus the full `test_harness.py` and `test_no_leakage.py` |
| 5 | `eval/exp_b.py`, run half — `build_sessions_b`, `evaluable_subjects_b`, `SessionResidualFeatures`, `equal_session_objective` (module-level, picklable), `ExpBFoldResult`, `_run_single_fold_b`, `run_exp_b` | Needs steps 2–4. Imports `stage1_candidates`/`stage2_candidates`/`StoreBackedFeatures`/`expected_fingerprints`/`_selection_frequency` from `exp_a.py` — never copies them (A-M6-3 requires one enumeration of the frozen search space) |
| 6 | `eval/exp_b.py`, report half — `_oof_matrix`, `summarize_exp_b`, `write_exp_b_reports`, `run_and_report_b`, `_assert_mechanism_ok_b` | Needs the run half to produce `ExpBFoldResult`s to summarize |
| 7 | The session-specific secondary variant: `run_exp_b_one_session` (the real, per-session unit of work), `run_exp_b_session_specific` (sequential test-only wrapper), `merge_session_specific_reports`, `config_fingerprint`; plus a minimal, zero-behaviour-change rename in `src/dehyd/provenance.py` (`_fold_manifest` → public `fold_manifest`, one call-site update) since M8 gives it a second legitimate caller (C21) | Owner decision (Step 0 item 3): last, so it cannot delay or complicate the primary pooled result. Reuses the same `equal_session_residual_mae` objective (degenerates to plain residual MAE over one session) — no new objective needed. Designed from the start as 4 independent units of work (C11), not a sequential loop, so real concurrency is possible without nested in-process parallelism |
| 8 | `experiments/run_clock_decoupling.py` (incl. `--session-specific`/`--init-run-group`/`--session`/`--run-dir`/`--merge-sessions`) + `scripts/ibex/run_exp_b.sbatch` + `scripts/ibex/run_exp_b_variant.sbatch` (one file, `STAGE=init|array|merge`) + `scripts/ibex/submit_exp_b_variant.sh` | Needs everything above; the primary sbatch mirrors `run_exp_a.sbatch` exactly (justified: comparable single-search wall-time), but the variant is a **SLURM array**, not a single job looping over sessions (§2.5, C11) — a single job iterating 4 full searches sequentially would have wall-time ≈ their sum, which is exactly what C11 flagged as unjustified in the previous draft. The `init` stage (heavy raw-file hashing for `record_run`) is its own properly-sized batch stage, not a login-node command (C23), and the wrapper mirrors `submit_extract77.sh`'s actual git-capture idiom rather than an invented one |
| 8.5 | **Owner-triggered clean M8 implementation commit** (all code green, working tree clean) on `v1_milestone_8`, before any producer/store rebuild — already includes step 0.5's `implementation_plan.md` fix | `validate_store` requires the store's recorded commit == the analysis commit; *any* code change in steps 0–8 moves the commit, so the store cannot be validated against the M8 analysis revision until it is rebuilt from a commit that includes all of M8's code. Mirrors M7's step 10.5 precedent |
| 8.6 | Confirm/rebuild both 10 GHz and 77 GHz feature stores **from the step-8.5 commit** (`extract10.sbatch`/`extract77.sbatch` if needed) and `--validate` both | Local/CPU synthetic-store tests exercise the mechanism, not real data; the real smokes and full runs need stores whose recorded commit matches the M8 revision, or `validate_store` fails closed |
| 9 | Local/CPU synthetic-store tests green end to end; then mechanism-only smoke, both bands (`--subset 6subjects`), against the step-8.6 stores | Cheap correctness gate before spending IBEX time on a 16-fold job (and, separately, before the 4x session-specific searches) |
| 10 | Full-cohort Exp B, both bands, on IBEX (`--full-cohort`) — **no owner pause** (Step 0 item 2) | Nothing left to blind; Exp B's core design (what runs, what data it uses) was frozen before Exp A was seen, so there is no freeze left to spend here — A-M8-1/A-M8-2 are reporting/labelling completions, not data-use choices (C13) |
| 10.5 | Session-specific secondary variant, both bands: run `scripts/ibex/submit_exp_b_variant.sh`, which (1) submits `STAGE=init` and blocks on it (`sbatch --wait`), refusing to continue if it fails (C23); (2) submits `STAGE=array` (`--array=1-4`, one task per session) with the captured `run_dir`, each task sized from step 10's measured per-search wall-time (C8/C11); (3) submits `STAGE=merge` with `--dependency=afterany:<array-job-id>` | The array gives REAL cross-session concurrency (wall-time ≈ one session's search, if the cluster schedules the 4 tasks concurrently) instead of the previous draft's unjustified sequential-loop claim (C11); an unexpected failure in one task exits that task nonzero and simply leaves its output file absent — never caught and downgraded in-process (C12) — so the merge step's `completed_sessions` reflects real scheduler outcomes, not swallowed exceptions; the heavy raw-file-hashing `init` stage runs in its own sized batch allocation, never on a login node, and a failed `init` never reaches the array (C23) |
| 11 | HISTORY.md per-step entries (continuously, already written by this point); SECOND_CHAPTER.md §7 from the full results, including the A-M8-1/A-M8-2 chronology | CLAUDE.md journal rules — the chapter section is written from real numbers, not drafted in advance, and states plainly what was decided when |

---

## §2 Per-file specifications

Format per file: **Responsibility** · **Public API / content** · **Frozen values** ·
**Acceptance criteria**. Signatures below are the contract; exact bodies are written at
implementation time. Test-group IDs (e.g. `T-M8-objective`) map to §3.

### 2.1 `src/dehyd/eval/metrics.py` (additions)

**Responsibility.** Exp B's objective and its statistical reporting, as pure functions over
already-computed predictions — no fitting, no fold construction, no I/O. Extends the existing
frozen M7 surface (`subject_balanced_mae`, `subject_cluster_bootstrap`,
`subject_cluster_bootstrap_pooled`, `wilcoxon_signed_rank`, `mean_difference_ci`), which is
**unchanged** apart from one internal extraction (below).

**Public API.**
```python
def per_session_residual_mae(session_idx, y_true, y_pred) -> dict[int, float]
    # {session index: residual MAE over that session's rows}. The shared building block of
    # BOTH the inner-CV objective and the reported per-session breakdown — one definition, so
    # selection and reporting cannot drift. A session with no rows is simply absent.

def equal_session_residual_mae(subjects, y_true, y_pred, session_idx) -> float
    # Exp B's inner-CV selection objective AND the per-fold building block of its reported
    # primary aggregate (StatsConfig.expb_aggregate_estimand =
    # "session_weighted_equal_weight_per_session"): the mean of per_session_residual_mae's
    # values, EQUAL WEIGHT PER SESSION over sessions present in THIS call's rows — deliberately
    # NOT subject-weighted (that is the separate paired-test estimand). `subjects` is accepted
    # and unused, so the signature matches the harness score_fn hook uniformly across any future
    # objective. NaN on empty input.
    #
    # RUN-LEVEL VIABILITY (C4): this function silently averages over whatever sessions are
    # present in its arguments — correct for scoring one fold's rows, where per-fold session
    # drops are expected and already logged (implementation_plan.md's degenerate-fold rule).
    # It must NOT be used, uncritically, to compute the run-level primary point estimate: if a
    # session is absent from the GLOBAL out-of-fold matrix (every fold happened to drop it, or
    # no subject was ever eligible for it), a naive call over the whole matrix would silently
    # report a three-session mean labelled as the four-session primary. `summarize_exp_b` (§2.4)
    # is the sole caller responsible for checking global per-session N_eval > 0 for all of
    # S1-S4 BEFORE treating its output as the primary aggregate — see the viability field in the
    # output shape below.

def holm_adjusted(p_values, *, family_size: int | None = None) -> list[float]
    # Holm-Bonferroni step-down adjustment, RETURNED IN INPUT ORDER. family_size defaults to
    # len(p_values) but is pinned by the caller to StatsConfig.holm_family_expb_per_session = 4
    # so a session missing from a given run cannot weaken the pre-registered correction. NaN
    # inputs pass through as NaN and still occupy a family slot (conservative, not dropped).

def _cluster_bootstrap_over_rows(subjects, theta_of_rows, *, b, level, rng_seed, method,
                                  skip_threshold_pct) -> BootstrapCI
    # The shared subject-cluster machinery extracted from the existing
    # subject_cluster_bootstrap_pooled: resample SUBJECTS with replacement (b draws of n
    # subjects, all of a subject's rows travel together with multiplicity), evaluate
    # theta_of_rows(row_index) per resample, jackknife leave-one-subject-out for BCa,
    # percentile fallback, skip-and-count. Internal helper — not exported.

def session_weighted_bootstrap(
    subjects, session_idx, y_true, y_pred_by_seed, *,
    y_pred_reference=None, b=10000, level=0.95, rng_seed, method="bca",
    skip_threshold_pct=5.0,
) -> BootstrapCI
    # Exp B's PRIMARY CI (A-M8-1). Within each subject-resample, recompute the per-session
    # residual MAEs on the resampled rows and average with EQUAL WEIGHT per session, per seed,
    # then average across seeds -> one scalar per resample (the POOLED/nonlinear seed-collapse
    # rule, since this is an average of averages, not an average of per-subject values). A
    # resample that empties a session is skipped-and-counted (O-B2/A-M8-2), never silently
    # averaged over the survivors. `y_pred_reference` (the session-mean baseline; zeros on the
    # residual scale), when given, makes the bootstrapped quantity
    # aggregate(radar) - aggregate(reference) directly.
```

**Key mechanics (each has an acceptance test).**
- **Estimand divergence is provable, not assumed.** `equal_session_residual_mae` (session-weight)
  and a plain per-subject mean over eligible sessions (subject-weight) are asserted to differ on
  a fixture where subjects have unequal eligible-session counts — the exact condition under which
  `implementation_plan.md:1208-1217` says the two estimands diverge.
- **Bootstrap extraction is bit-identity-preserving.** `subject_cluster_bootstrap_pooled`'s RNG
  draw order (`rng.integers(0, n, size=n)` per replicate, `b` replicates) and output must be
  bytewise unchanged after being rewritten as a thin wrapper over `_cluster_bootstrap_over_rows` —
  proven against the step-1 pin, not merely "should be equivalent."
- **Seed-collapse rule assignment is explicit in code comments**, because it is easy to get
  backwards: the aggregate is pooled/nonlinear (recompute-per-seed-then-average); per-session MAE
  and the paired per-subject scalar are additive (seed-average-per-subject-first).

**Frozen values.** `bootstrap_b=10000`, `ci_method="bca"` with `"percentile"` fallback,
`confidence_level=0.95`, `resample_unit="subject"`, `undefined_metric_skip_threshold_pct=5.0`
(all from `StatsConfig`, not re-decided here); `holm_family_expb_per_session=4`
(`StatsConfig`, config.py:461).

**Acceptance** (`T-M8-objective`, `T-M8-holm`, `T-M8-bootstrap`). `equal_session_residual_mae` on
a deliberately unequal fixture ≠ `subject_balanced_mae` and ≠ a naive pooled MAE, matched against
a hand-computed value; averages only over sessions present; NaN on empty. Holm: hand-computed
4-p family, step-down monotonicity enforced, clipped at 1.0, input order preserved, NaN occupies
a slot, `family_size=4` gives a strictly stronger correction than `len(p)=3` on the same p-values.
`session_weighted_bootstrap`: deterministic given `rng_seed`; provably different from
subject-weighted on the unequal fixture; `y_pred_reference` yields the difference form; an
empty-session replicate is skipped-and-counted and can trip `unreliable`;
`subject_cluster_bootstrap_pooled` reproduces the step-1 pin exactly post-extraction.

### 2.2 `src/dehyd/models/baselines.py` (additions)

**Responsibility.** The single train-only computation of per-session means μ_s, shared by three
consumers: the residualizing feature provider (subtracts μ_s from the target), the Exp B baseline
(predicts μ_s, i.e. residual 0), and the fit audit (μ_s must appear as an audited fitted
quantity at both CV levels). One function, three callers — never three computations.

**Public API.**
```python
def session_means(subjects, session_idx, targets, train_subjects, *, min_train_subjects=2)
    -> tuple[dict[int, float], tuple[int, ...]]
    # Train-only per-session mean mu_s = mean of `targets` over rows whose subject is in
    # train_subjects and whose session is s. A session with fewer than min_train_subjects
    # DISTINCT eligible training subjects has an undefined/unstable mean and is DROPPED for
    # this fold (implementation_plan.md:736-740) -- excluded from residualization, the
    # objective, and reporting for that fold. NEVER filled from validation/test labels, from
    # other subjects, or from a global fallback (the deliberate contrast with Exp A's O2).
    # Iterates sorted sessions and sorted subject membership for bit-identical float
    # accumulation serial vs parallel. Returns ({s: mu_s} over kept sessions, sorted dropped
    # session indices).

def fit_session_mean_baseline(subjects, session_idx, targets, train_subjects, *,
                               role="outer_train", min_train_subjects=2) -> BaselineFitOutcome
    # Exp B's pre-registered baseline: predict each session's train-only mean Delta-m% --
    # i.e. residual 0. Mirrors fit_session_index_baseline's shape but deliberately differs:
    # NO global-mean fallback (Exp A's O2 does not apply here) -- a degenerate session is
    # DROPPED, matching session_means. Emits quantity="session_means" with all-ndarray
    # FitRecord params (indices, means, dropped).

def predict_session_mean(model: dict, session_idx) -> np.ndarray
    # Per-row mu_s. RAISES on an index absent from the model (a dropped or unseen session) --
    # by construction a dropped session's rows never reach here; silently imputing would
    # reintroduce exactly the leak the drop rule forbids. This is the one place Exp B
    # deliberately does NOT copy Exp A's O2 fallback.
```

**Frozen values.** `min_train_subjects = 2` (implementation_plan.md's degenerate-fold rule, "a
session s with < 2 eligible training subjects"). No hyperparameters.

**Acceptance** (`T-M8-mu`). Counts **distinct training subjects**, not rows; `<2` distinct →
session dropped, never imputed; mutating a validation/test subject's target leaves μ_s bytewise
identical (train-only); dropping session 3 leaves μ_1/μ_2/μ_4 bytewise unchanged (independence
across sessions); `fit_session_mean_baseline` emits a `FitRecord` with all-ndarray params;
`predict_session_mean` raises `KeyError`/a named error on a dropped/unknown index rather than
falling back.

### 2.3 `src/dehyd/eval/harness.py` (the one structural edit)

**Responsibility.** Remains the single generic fit-on-train-only nested-LOSO engine serving Exp
A, the frozen leakage-suite shim, the T18 torch path, **and now Exp B** — via one new, optional
parameter, not a second engine. No other module constructs folds or defines the tie-break; that
does not change.

**Public API / content (diff only — everything else in this file is unchanged).**
```python
@dataclass
class FeatureBundle:
    subjects: np.ndarray
    X: np.ndarray
    y: np.ndarray
    extra_fits: tuple = ()
    session_idx: np.ndarray | None = None
        # Per-row session index, aligned to `subjects`/`X`/`y`. Only objectives that group BY
        # SESSION (Exp B's equal-session residual MAE) need it. Appended AFTER extra_fits with
        # a None default so every existing positional construction (exp_a.py, test_harness.py,
        # fixed_feature_provider) stays valid unchanged.

def _score(score_fn, bundle, rows, y_pred) -> float:
    # The one scoring choke point. score_fn=None -> the CURRENT, UNCHANGED call to
    # subject_balanced_mae(bundle.subjects[rows], bundle.y[rows], y_pred). A supplied score_fn
    # additionally receives bundle.session_idx[rows] (or None if the bundle carries none).
    # Signature: score_fn(subjects, y_true, y_pred, session_idx) -> float.

def _fit_score_inner(candidate, bundle, inner, seeds, before_fit, *, score_fn=None) -> tuple
def _score_candidates_on_fold(candidates, fold, seeds, before_fit, data_for, *,
                               score_fn=None) -> StageOutcome
def _final_refit(candidate, fold, seeds, before_fit, data_for, *, score_fn=None) -> tuple
def run_nested_candidates(dataset, candidates, *, seeds=(0,), before_fit=None, data_for=None,
                           score_fn=None, **split_kwargs) -> list[FoldResult]
```

**Key mechanics (each has an acceptance test).**
- **`score_fn` is keyword-only on every signature it touches.** `exp_a.py:260/264/269` calls
  `_score_candidates_on_fold`/`_final_refit` **positionally** and must not need editing — this is
  the load-bearing reason the parameter is keyword-only with a `None` default, not inserted
  positionally.
- **`score_fn=None` reaches the byte-identical call with byte-identical arguments** it reached
  before this edit — proven against the step-1 pin, not asserted.
- **Row-subset safety, verified by reading the code, not assumed.** `train_rows`/`val_rows`/
  `test_rows` are recomputed from `bundle.subjects` on every call (existing lines, unchanged);
  `inner_scores` holds scalars only, independent of row count; flat `inner_results` assembly is
  index-based. So a provider that returns a **row subset** (Exp B's degenerate-session drop) is
  safe without further harness changes — this was verified against the current source, not
  inferred.
- **A zero-row selection is a hard crash inside `pipe.predict`**, not a NaN (confirmed:
  `ValueError: Found array with 0 sample(s)… required by StandardScaler`). This is a caller
  responsibility (§2.4/`_run_single_fold_b`), not something `harness.py` itself needs to guard —
  documented here so the contract is explicit at the boundary.

**Frozen values.** Everything already frozen in this file (folds only from `splits.py`, tie-break
only via `select_candidate`, `tuned_eps_k=0.1`/fallback `1e-6`, Ridge `solver="cholesky"`) is
untouched. `score_fn` has no default behaviour of its own; `None` is not "a" scoring choice, it is
"defer to the existing one."

**Acceptance** (`T-M8-harness-hook`, plus full `T-M7-harness`/`test_no_leakage.py` re-run).
`score_fn=None` reproduces the step-1 pin bytewise (`inner_scores.tobytes()`, selected candidate,
`test_score`). A supplied non-default `score_fn` demonstrably changes the selected candidate on a
fixture built so the two objectives disagree (proves the hook has power, not just wiring).
`session_idx=None` bundles work unchanged with any scorer that tolerates `None`. A row-subset
provider: masks recompute correctly post-drop, a val subject left with zero surviving rows yields
an empty (not crashing) `val_predictions` entry. `exp_a.py`, `reference_procedure.py`, and
`test_no_leakage.py` are **git-diff-clean** after this step.

### 2.4 `src/dehyd/eval/exp_b.py` (new)

**Responsibility.** The full Exp B composition — the residualized-target analogue of `exp_a.py`,
reusing (never copying) its search-space enumeration and feature path.

**Public API / content.**
```python
def build_sessions_b(config, band) -> list[dict]
    # Exp A's build_sessions(), filtered to session_idx in {1,2,3,4} -- S0 EXCLUDED AT THE
    # SOURCE (its Delta-m% is identically 0, which would give every fold a free,
    # perfectly-"predicted" session that deflates every MAE). Same record shape as Exp A.

def evaluable_subjects_b(sessions) -> list[int]
    # Subjects with >=1 eligible S1-S4 session (implementation_plan.md:611-613) -- NOT Exp A's
    # ">=1 eligible session" rule, which would admit an S0-only subject into nested_loso_splits
    # with zero rows once S0 is filtered, crashing downstream. Computed and applied BEFORE
    # nested_loso_splits is called.

class SessionResidualFeatures:
    # Wraps (does not subclass) exp_a.StoreBackedFeatures -- the X path, including its tuned-e
    # cache keyed by (feature_key, frozenset(train_subjects)), is reused byte-for-byte.
    def __init__(self, band, sessions, store_dir, config): ...
    def data_for(self, candidate, train_subjects) -> FeatureBundle
        # Calls self.base.data_for(...) for X; calls baselines.session_means(...) for mu_s and
        # the fold's drop set; drops those sessions' rows from subjects/X/y/session_idx in
        # lockstep; residualizes y = raw_y - mu_s on the KEPT rows; emits mu_s via
        # extra_fits=(("session_means", {...}),) alongside any inherited tuned_epsilon entry,
        # so it is audited exactly like any other fitted quantity. Records the drop for this
        # train_subjects set for later logging.

def equal_session_objective(subjects, y_true, y_pred, session_idx) -> float
    # Module-level (picklable under spawn) wrapper: metrics.equal_session_residual_mae. This
    # exact function object is the score_fn passed into harness calls -- never a lambda or
    # closure, so it survives multiprocessing pickling.

@dataclass
class ExpBFoldResult:
    test_subject: int
    selected_feature_key: tuple
    selected_family: str
    selected_params: dict
    test_predictions: np.ndarray
    test_targets: np.ndarray            # residual scale
    test_session_idx: np.ndarray
    seed_outcomes: list
    baseline_predictions: np.ndarray    # == np.zeros(...) on the residual scale, by construction
    final_fits: list
    dropped_sessions_outer: tuple
    dropped_sessions_inner: tuple        # ((sorted(train_subjects), dropped), ...)
    reason: str | None = None            # non-None: this fold contributes no out-of-fold rows

def _run_single_fold_b(config, band, sessions, store_dir, fold, seeds) -> ExpBFoldResult
    # Top-level, picklable. Builds its OWN SessionResidualFeatures (open npz handles are not
    # shareable across processes) and pins threadpool_limits(1), mirroring
    # exp_a._run_single_fold exactly. Before calling _final_refit, checks whether the held-out
    # subject has ANY surviving (non-dropped-session) row for this fold; if none, returns an
    # ExpBFoldResult with empty prediction arrays and reason="no_surviving_test_rows" rather
    # than letting sklearn's zero-row ValueError propagate. Uses the SAME
    # baselines.session_means(...) call (same train_subjects) that the provider used, so the
    # baseline and the residualization never compute two different mu_s (T-M8-provider proves
    # this bytewise).

def run_exp_b(config, band, sessions, store_dir, *, seeds, n_workers=1) -> list[ExpBFoldResult]
    # Mirrors exp_a.run_exp_a's fold-parallel structure and spawn-context Pool exactly
    # (results.sort by test_subject for deterministic reassembly).

def _oof_matrix(results) -> tuple
    # Concatenates (subjects, session_idx, residual y_true, per-seed y_pred, baseline zeros)
    # across folds in canonical order. Asserts (subject, session) pairs are UNIQUE across the
    # whole matrix (each subject appears in exactly one outer fold) -- a one-line check that
    # catches a whole class of assembly bugs.

def summarize_exp_b(results, config) -> dict
    # See the output shape below. PRIMARY pooled model only -- the session-specific variant's
    # summary is `summarize_variant_session` (below), a separate function, not a mode of this
    # one (no `variant=` parameter: avoids one function silently branching between two different
    # output shapes for two structurally different models).

def _write_predictions_csv(results, out_path) -> None
def _write_selection_table_csv(results, out_path) -> None
def _write_dropped_folds_csv(results, out_path) -> None
    # (C18) Extracted from write_exp_b_reports's body so BOTH the primary path and the
    # session-specific variant write the SAME formats from the SAME code -- never two
    # implementations of "what a predictions/selection-table/dropped-folds CSV looks like".

def write_exp_b_reports(results, summary, out_dir, band) -> dict[str, Path]
    # metrics_exp_b_{band}.json, predictions_b_{band}.csv, selection_table_b_{band}.csv,
    # dropped_sessions_{band}.csv, scatter_b_{band}.png (residual scale) -- same shapes/naming
    # convention as write_exp_a_reports. Calls the three _write_*_csv helpers above.

def run_and_report_b(config, band, sessions, store_dir, run_dir, *, mode, analysis_commit,
                      n_workers=1) -> dict
    # validate_store -> run_exp_b -> _assert_mechanism_ok_b -> smoke (structural run-log only,
    # NO performance value, matching exp_a's C9/C14 doctrine) or full reporting for the PRIMARY
    # pooled model ONLY. There is no variant flag here at all (C6/C11 revision): the
    # session-specific variant is invoked through an entirely separate call tree (below, and
    # §2.5's `--session-specific` CLI path), so "smoke never touches the variant" is structural
    # -- there is no shared parameter that could be mis-set -- not a default that could be
    # overridden.

def _assert_mechanism_ok_b(results, sessions) -> None
    # Fold-role disjointness (as exp_a._assert_mechanism_ok) PLUS: no session_idx == 0 row
    # anywhere in any result; every emitted session_means FitRecord's subject set excludes the
    # held-out subject at the outer level. Reused UNCHANGED by the session-specific variant
    # (C18) -- the same mechanism/audit assertion runs before writing variant outputs too, not
    # a weaker or skipped check.

# --- session-specific secondary variant (step 7/10.5; implementation_plan.md:722-724) ---
def eligible_subjects_for_session(sessions, session) -> list[int]
    # (C17) The subjects eligible for session `session` -- sorted subject IDs. Used BOTH inside
    # run_exp_b_one_session (to build that session's nested_loso folds) AND by --init-run-group
    # (to populate the group provenance's authoritative expected_subjects_by_session, below) --
    # ONE definition, so the fold-construction logic and the validation-reference logic cannot
    # silently diverge.

def config_fingerprint(config) -> str
    # (C20) sha256(json.dumps(config_to_dict(config), sort_keys=True)) -- config_to_dict is
    # the SAME function provenance.record_run imports from ..config and uses to populate
    # payload["config"], so this hash is guaranteed byte-identical-content-equivalent to what
    # provenance.json's own "config" field holds. ONE named helper, called identically by
    # --init-run-group and every --session task, so the two can never independently invent
    # subtly different hashing recipes that silently fail to match.

def run_exp_b_one_session(config, band, sessions, store_dir, session, *, seeds,
                           n_workers=1) -> list[ExpBFoldResult]
    # THE REAL UNIT OF WORK (C11): one session s's fully independent nested-LOSO search, over a
    # provider that keeps ONLY session s's rows (residualized by that session's own train-only
    # mu_s, via the same baselines.session_means single-source-of-truth) and the SAME
    # equal_session_objective, which degenerates to plain single-session residual MAE when only
    # one session is present. Only eligible_subjects_for_session(sessions, s) enters its own
    # nested_loso folds -- a variable count <=15, per the frozen spec -- so this session's outer
    # folds are distinct from the pooled model's and from every other session's. This is what a
    # single SLURM array task runs directly (§2.5) -- cross-session concurrency comes from the
    # array, not from anything in-process. ANY UNEXPECTED EXCEPTION PROPAGATES (C12): this
    # function never catches a generic exception to produce a placeholder/partial result; only
    # the harness's own pre-defined non-evaluability doctrine (`InnerResult.reason`, e.g. a fold
    # with too few training subjects for any candidate to be viable) may degrade gracefully, and
    # that is the harness's existing behaviour, not new swallowing introduced here. A real bug
    # must crash this call and make the calling process/array-task exit nonzero. **The CALLER
    # (the entrypoint's --session task, §2.5) is responsible for calling `validate_store` with
    # this band's `exp_a.expected_fingerprints(config, band, sessions)` BEFORE invoking this
    # function (C17) -- exactly as `run_and_report_b` already does for the primary path -- and
    # for running `_assert_mechanism_ok_b(results, sessions)` and writing the full artifact set
    # (predictions/selection-table/dropped-folds CSVs, via the shared `_write_*_csv` helpers,
    # C18) before writing the JSON shard.**

def run_exp_b_session_specific(config, band, sessions, store_dir, *, seeds,
                                n_workers=1) -> dict[int, list[ExpBFoldResult]]
    # Sequential convenience wrapper: {s: run_exp_b_one_session(..., s, ...) for s in (1,2,3,4)}.
    # Used ONLY by the synthetic-store test (T-M8-variant), where running four tiny searches
    # sequentially is fine -- NOT the real-IBEX path (C11). The real path is four separate array
    # tasks each calling run_exp_b_one_session directly (§2.5).

def summarize_variant_session(results_s, session, config) -> dict
    # (C6/C16/C18) The per-session summary for ONE session-specific model: {n_eval, radar_mae:
    # BootstrapCI, baseline_mae: BootstrapCI, mean_difference: BootstrapCI, selection_frequency:
    # {...}}. `selection_frequency` reuses exp_a._selection_frequency(results_s) UNCHANGED --
    # implementation_plan.md's mandatory selection-frequency/stability table applies to every
    # experiment, this variant included (C18); it is not optional polish. Still DESCRIPTIVE ONLY
    # on the inferential side -- deliberately NO p-value, and therefore no multiplicity-
    # correction question to answer (C16): the frozen protocol defines Holm-4 for the PRIMARY
    # model's own per-session breakdown only; it says nothing about a family size for these four
    # independently-fitted secondary models. Choosing "uncorrected" (or inventing a family size)
    # would be a THIRD undisclosed post-Exp-A protocol completion, on top of A-M8-1/A-M8-2 --
    # exactly what C10 flagged Exp B must avoid. If a later milestone wants an inferential claim
    # from this variant, that needs its own owner-approved amendment and implementation_plan.md
    # propagation (the A-M8-1/A-M8-2 treatment), not a default buried in this function.

def merge_session_specific_reports(band, run_dir) -> dict
    # (C11/C12/C15/C17/C19/C20) Reads run_dir/provenance.json (written ONCE by --init-run-group,
    # §2.5) for the run-group's authoritative lineage, reading each field from its PRECISE,
    # source-verified location (C19/C20 -- record_run's real payload construction does
    # `if extra: payload["extra"] = extra`, so extra content NESTS, it is not flattened):
    # `analysis_commit` <- provenance["git"]["commit"] (the schema's own native field, never
    # duplicated); `config_hash`/`expected_subjects_by_session` <-
    # provenance["extra"]["config_hash"]/provenance["extra"]["expected_subjects_by_session"]
    # (both `extra`-nested fields --init-run-group supplied, since record_run has no native
    # config_hash field). For each session s in 1..4 whose session_specific_{band}_s{s}.json is
    # PRESENT under
    # run_dir: parses it and FAIL-CLOSED validates -- mirroring store.validate_store's
    # _check_match precedent, not a new pattern -- that the shard's embedded
    # band/session/run_group_id match what is expected (band argument, s, run_dir.name), that
    # its analysis_commit/config_hash/seed_set match the group provenance's values EXACTLY, AND
    # that its n_eval_subjects matches expected_subjects_by_session[s] EXACTLY (C17 -- an
    # evaluated-cohort mismatch is validated against the group's own authoritative snapshot, not
    # merely recorded and trusted). A missing file (task crashed, still running, never submitted)
    # is simply absent from `completed_sessions` -- NOT an error. A PRESENT but malformed or
    # mismatched shard (wrong commit, wrong config, wrong session, wrong cohort, stale leftover
    # file) IS an error: raises `ExpBError` naming the session, the field, and both the expected
    # and found values, rather than silently excluding it or counting it. Counting file
    # existence alone would let a stale or wrong-provenance shard silently pass as complete
    # (C15) -- this function never does that. Store validity itself is NOT re-checked here (it
    # was already enforced, per-task, before that task was allowed to fit anything, C17) -- this
    # function validates the SHARDS' lineage, not the store directly.
```

**`session_specific_{band}_s{S}.json` per-shard schema** (C15/C17/C18) — written by one array
task (`--session-specific --session S`, §2.5) after `validate_store` and
`_assert_mechanism_ok_b` have both passed, directly from `summarize_variant_session`'s output
plus embedded lineage, so the merger can fail-closed validate it without trusting file placement
or raw file counts alone:
```
{
  "run_group_id": "<run_dir.name, e.g. 20260728T120000123456Z_abcd1234>",
  "band": "10ghz" | "77ghz",
  "session": 1,          # must equal the {S} in this shard's own filename
  "analysis_commit": "<git commit this array task ran at -- validate_store already enforced
                       this equals the store's own recorded commit before any fit happened>",
  "config_hash": "<exp_b.config_fingerprint(config) -- the ONE named helper (C20), called
                  identically here and by --init-run-group, never independently invented per
                  task; compares against provenance['extra']['config_hash'] (C19/C20)>",
  "seed_set": [1, 2, 3, 4, 5],
  "n_eval_subjects": [...],   # subject IDs evaluated for this session's search -- validated
                              # against the group provenance's expected_subjects_by_session[S]
                              # (C17), not merely recorded
  "summary": {<summarize_variant_session output, incl. selection_frequency (C18)>}
}
```
Alongside each shard, the same array task also writes (C18) `session_specific_predictions_
{band}_s{S}.csv`, `session_specific_selection_table_{band}_s{S}.csv`, and
`session_specific_dropped_folds_{band}_s{S}.csv` — the per-fold intermediate artifacts (OOF
predictions, selected feature_key/family/params per fold, and any non-evaluable-fold records) —
via the `_write_*_csv` helpers shared with the primary path, satisfying the regenerable-from-
saved-intermediate-artifacts requirement (`ROADMAP.md`'s engineering standards;
`implementation_plan.md:1273-1276`) that a summary-only shard could not.

**Group provenance — matching `provenance.record_run`'s REAL contract, read directly from
`src/dehyd/provenance.py` (C19/C20/C21/C22, not re-guessed a second time).**
`record_run(config, manifest, folds=None, extra: dict | None = None, data_dir=None) -> Path`
returns the **`provenance.json` FILE path**, not a directory (`run_dir = run_path.parent`, as
`run_regression.py` already does). Its payload construction is, verbatim:
```python
payload = {..., "folds": _fold_manifest(folds), "git": git, ...}
if extra:
    payload["extra"] = extra
```
Two consequences, both missed in the previous draft:
- **`extra` nests under `payload["extra"]`, it is NOT merged into the top level** (C20). Every
  field supplied via `extra` is read back at `provenance["extra"][key]`, never
  `provenance[key]`. `analysis_commit` still reads from the **native, non-`extra`** field
  `provenance["git"]["commit"]`; `config_hash` and `expected_subjects_by_session` (both
  `extra`-only content, since neither exists natively) read from `provenance["extra"]
  ["config_hash"]` / `provenance["extra"]["expected_subjects_by_session"]`.
- **`folds=None` produces `"folds": []`** (`_fold_manifest(None) -> []`), and
  `implementation_plan.md:1273-1276` requires **every run** to write "the fold manifest (subject
  role per fold)" — not conditionally, not only when a single fold list exists. An empty `[]`
  for a run-group that represents four real, independently-evaluated searches would violate that
  frozen requirement outright (C21) — "reconstructable in principle" does not satisfy a rule
  that says provenance *writes* it. `provenance.py`'s `_fold_manifest` becomes **`fold_manifest`
  (dropped leading underscore, zero behaviour change)** — a minimal, justified rename since M8
  gives it a second legitimate caller, and duplicating its exact serialization logic elsewhere
  would risk the two silently drifting.
- **`data_dir` defaults to the 10 GHz root** (`_hash_inputs`'s own logic:
  `config.paths.data_10ghz_dir if data_dir is None else data_dir`); the module's own docstring
  states "77 GHz entrypoints pass `require_77ghz_dir(config)`" (C22). Every `record_run` call in
  this plan — the **primary** Exp B entrypoint's call (§2.5's primary Flow) AND `--init-run-group`
  — must pass `data_dir=require_77ghz_dir(config) if band == "77ghz" else None`, or a 77 GHz run
  would try to hash 77 GHz `rel_path`s against the 10 GHz directory and crash inside
  `sha256_file` (file not found), not merely produce wrong provenance.

`--init-run-group`'s corrected call:
```
manifest_qc = apply_qc(build_manifest(config.paths, gt), config.paths, config)          # 10ghz
# or apply_qc_77(build_manifest_77(...), ...) for 77ghz -- SAME QC step run_regression.py
# already performs; record_run's `manifest` argument is REQUIRED, not optional to skip.
config_hash = exp_b.config_fingerprint(config)   # sha256(json.dumps(config_to_dict(config),
    # sort_keys=True)) -- config_to_dict is the SAME function record_run itself imports from
    # ..config and uses to populate payload["config"] (C20), so the hash is guaranteed to be
    # computed over byte-identical content to what provenance.json's own "config" field holds.
    # ONE named helper (exp_b.config_fingerprint), called identically by --init-run-group and
    # every --session task -- never two independently-written hashing recipes.
folds_by_session = {str(s): provenance.fold_manifest(
        nested_loso_splits(eligible_subjects_for_session(sessions, s)))
    for s in (1, 2, 3, 4)}          # satisfies implementation_plan.md:1273-1276 for real (C21)
run_path = provenance.record_run(
    config, manifest_qc, folds=None,
    data_dir=require_77ghz_dir(config) if band == "77ghz" else None,        # C22
    extra={
        "stage": "exp-b-session-specific-group", "band": band, "config_hash": config_hash,
        "expected_subjects_by_session": {"1": eligible_subjects_for_session(sessions, 1), ...,
                                          "4": eligible_subjects_for_session(sessions, 4)},
        "folds_by_session": folds_by_session,
    },
)
run_dir = run_path.parent
print(run_dir)
```
`folds=None` (the top-level parameter) stays None — it is a single-fold-list slot and the
variant has four independent ones — but `extra["folds_by_session"]` now carries the REAL,
canonically-serialized fold-role manifest for all four searches (via the same `fold_manifest`
`record_run` itself uses), so the frozen "every run" requirement is actually met, not argued
around (C21). `expected_subjects_by_session` remains the simpler field
`merge_session_specific_reports` validates shards against; `folds_by_session` exists for the
frozen provenance requirement and full regenerability, not because the merge needs a second,
redundant check on top of it.

**Which fields are authoritative, precisely (C19/C20).** `analysis_commit` compares against
`provenance["git"]["commit"]` (native). `config_hash` and `expected_subjects_by_session` compare
against `provenance["extra"]["config_hash"]` / `provenance["extra"]["expected_subjects_by_session"]`
— **nested under `"extra"`, confirmed against the real payload-construction code above, not
assumed a second time.** `merge_session_specific_reports` reads `provenance.json` at exactly
these paths — no field is redundantly duplicated or ambiguous between two possible
interpretations.

**`session_specific_{band}.json` schema** (C6/C11/C15/C16/C18) — produced by
`merge_session_specific_reports` via the `--session-specific --merge-sessions` CLI path (§2.5);
never written by the primary path, never in smoke mode:
```
{
  "conditional_exploratory": true,
  "note": "Secondary robustness variant (implementation_plan.md:722-724): four INDEPENDENTLY
           fitted single-session models, run as separate array tasks, not the pooled model's
           per-session breakdown. Never elevated to primary. DESCRIPTIVE ONLY -- no p-values, by
           design (C16): the frozen protocol's Holm-4 applies to the primary model's per-session
           breakdown only and does not define a family size for these four secondary models, so
           this variant reports effect sizes + conditional/exploratory CIs and no significance
           claim, rather than deciding an undisclosed multiplicity rule after Exp A.",
  "completed_sessions": [1, 2, 3, 4],   # (C8/C11/C12/C15/C17) EXACTLY the sessions whose shard
                                        # was found AND passed fail-closed lineage validation
                                        # (incl. the evaluated-cohort check, C17) -- never from
                                        # catching an exception, never from counting a
                                        # malformed/mismatched shard. Anything shorter than
                                        # [1,2,3,4] means partial completion and must read as
                                        # incomplete, not as if the milestone's D7 criterion
                                        # were met.
  "1": {<summarize_variant_session output, incl. selection_frequency>}, "2": {...}, "3": {...},
  "4": {...}
}
```

**`summarize_exp_b` output shape** (mirrors `exp_a.summarize_exp_a`'s `ci_dict(...)` convention):
```
{
  "conditional_exploratory": true,
  "estimand_primary": "session_weighted_equal_weight_per_session",
  "estimand_paired": "subject_weighted_complete_case_s1_s4",
  "n_eval_subjects_aggregate": int,
  "n_eval_by_session": {"1": n, "2": n, "3": n, "4": n},
  "n_rows": int, "n_seeds": int,
  "dropped_sessions": {"outer_by_fold": {...}, "inner": [...]},
  "primary_viable": bool,
  # (C4) false iff n_eval_by_session[s] == 0 for any s in 1..4 GLOBALLY (every fold dropped it,
  # or no subject was ever eligible) -- checked BEFORE computing primary_aggregate, never after.
  "primary_unavailable_reason": str | null,
  # e.g. "session 3 has zero out-of-fold rows across the full cohort" -- null when primary_viable.
  "primary_aggregate": {
    "radar": <BootstrapCI>, "baseline": <BootstrapCI>,
    "difference_radar_minus_baseline": <BootstrapCI>          # THE pre-specified primary (A-M8-1)
  } | null,          # null when primary_viable is false -- NEVER a silently-degraded 3-session
                     # mean reported as if it were the four-session primary
  "paired_subject_weighted_complete_case": {
    "n_complete_case": int, "wilcoxon_statistic": float, "wilcoxon_p": float,
    "mean_difference_radar_minus_baseline": <BootstrapCI>
  },
  "per_session_exploratory": {
    "holm_family_size": 4,
    "1": {"n_eval": int, "radar_mae": <BootstrapCI>, "baseline_mae": <BootstrapCI>,
          "mean_difference": <BootstrapCI>, "wilcoxon_p": float, "holm_p": float},
    "2": {...}, "3": {...}, "4": {...}
  },
  # (C5) baseline_mae is a BootstrapCI, not a bare float -- implementation_plan.md:1187-1189
  # requires a subject-cluster CI for Exp B's per-session AND aggregate MAE, not qualified to
  # radar only. Computed via subject_cluster_bootstrap on that session's per-subject baseline
  # residual MAE (the zero-prediction baseline), same seed/subject/skip rules as radar's.
  "selection_frequency": {...},
  "session_specific_variant": {...} | null   # see §2.4's orchestration spec below (C6)
}
```

The primary-viability check (`primary_viable`/`primary_unavailable_reason`) is distinct from, and
sits above, A-M8-2's bootstrap-replicate skip-and-count rule (§2.1): A-M8-2 handles a session
missing from one *resample*; this handles a session missing from the *whole run's* out-of-fold
data, which the aggregate function alone cannot detect (§2.1, C4).

**Frozen values.** `min_train_subjects=2` for all μ_s computations (from baselines.py). Search
space, budget K, seed set, and staged inner-CV mechanism identical to Exp A's (A-M6-3) — nothing
here re-enumerates or re-tunes them. RNG offsets: `config.run.seed + 100..132` per the named table
in §2.1's caller (`exp_b.py`), never a running counter (see §5 trap 10 for the rationale).

**Acceptance** (`T-M8-provider`, `T-M8-residual-leak`, `T-M8-degenerate`, `T-M8-report`). X path
bytewise identical to `StoreBackedFeatures` on kept rows; `y == raw − μ_s` on kept rows;
`session_idx` stays aligned after any drop; `extra_fits` carries `session_means` and (on the
tuned branch) still carries `tuned_epsilon`; the drop set for a given `train_subjects` is
identical across different candidates (**candidate-independence**, asserted directly); the outer
bundle's `session_means` params equal `fit_session_mean_baseline`'s bytewise (single source of
truth, asserted directly). **Residual-leak property**: mutating one inner-*validation* subject's
`delta_m_pct` (eligibility-preserving) leaves every `inner_train`-role fit — including
`session_means` — bytewise identical across the two runs; mutating an inner-*train* subject's
label **does** move `session_means` (power companion); mutating the held-out subject's label
leaves every `outer_train` fit identical. **This mutates labels and checks fitted records; it
does not by itself exercise the full new composition end to end** — see the separate
`T-M8-outer-mutation` property below (C2), which mutates the held-out subject's stored
features/target on a synthetic store and re-runs the entire `SessionResidualFeatures` + drop-row
+ `equal_session_objective` + two-stage-search path, asserting every fold-level quantity except
the held-out subject's own predictions/score is unchanged. Degenerate-fold fixture: a session with exactly one
eligible training subject is absent from the bundle, the objective, and out-of-fold rows, present
in `dropped_sessions`, and does not perturb the surviving μ_s. `summarize_exp_b`: correct
per-session `N_eval`; correct `n_complete_case`; aggregate provably ≠ naive per-subject mean on
an unequal fixture; Holm applied with `family_size=4`; empty complete-case set yields NaNs, not
an exception; **a fixture where one S1–S4 session has zero out-of-fold rows across the ENTIRE
run** sets `primary_viable=false` with a named reason and `primary_aggregate=null` — never a
silently-degraded three-session mean reported as the four-session primary (C4); **per-session
`baseline_mae` is a `BootstrapCI`**, computed by the same subject-cluster bootstrap machinery and
seed/subject rules as `radar_mae`, not a bare float (C5). **Session-specific variant** (`T-M8-
variant`, C6/C11/C12/C14/C15/C16/C17/C18): `run_and_report_b` has no variant parameter at all, so
smoke mode cannot reach the variant structurally, not by a default; each of the four sessions'
`run_exp_b_one_session` calls builds its own independent nested-LOSO search over its own eligible
training subjects (distinct outer folds from the pooled model **and from each other**, verified
directly); **`validate_store` runs before any fit, at both `--init-run-group` and every
`--session` task independently (C17)** — a fixture with a mismatched store/commit raises at
either point, before the search is ever attempted; **`_assert_mechanism_ok_b` runs on every
session's results before any output is written, unchanged from the primary path (C18)**; an
UNEXPECTED exception raised inside `run_exp_b_one_session` (a fixture that forces one) propagates
out uncaught — the test asserts it is NOT swallowed into a placeholder result — while a
pre-defined non-evaluability condition (the harness's existing `InnerResult.reason` doctrine)
still degrades gracefully as it always has; `summarize_variant_session` returns descriptive CIs
**plus `selection_frequency` (C18)** — no `wilcoxon_p`/`holm_p` field anywhere in its output
(C16), since the frozen protocol defines no multiplicity rule for these four secondary models and
inventing one here would be an undisclosed post-Exp-A completion; **the per-session predictions/
selection-table/dropped-folds CSVs are written via the SAME `_write_*_csv` helpers the primary
path uses (C18)** — regenerable from saved intermediate artifacts, not just a summary; `--init-
run-group` issues exactly one `record_run` call producing one shared directory (C14) and embeds
`expected_subjects_by_session` in its `provenance.json` (C17), and every array task's shard
embeds that directory's name as `run_group_id` plus `analysis_commit`/`config_hash`/`seed_set`/
`n_eval_subjects` (C15); `merge_session_specific_reports` on a fixture where only 2 of 4
per-session shards exist on disk yields `completed_sessions=[..]` with fewer than 4 entries,
computed purely from which shards are present **and valid**, which a downstream DoD-D7 check
must read as incomplete, not as satisfying D7; a fixture where a present shard's `analysis_
commit`/`config_hash`/`session`/`run_group_id`/`n_eval_subjects` disagrees with the group's
`provenance.json` (incl. its `expected_subjects_by_session`, C17) or its own filename makes the
merge **raise**, never silently exclude or count it (C15/C17). **`config_hash`/
`expected_subjects_by_session`/`folds_by_session` are read at `provenance["extra"][...]`, never
the top level (C20); `folds_by_session` genuinely satisfies `implementation_plan.md:1273-1276`'s
every-run fold-manifest requirement via the renamed public `provenance.fold_manifest` (C21); a
77 GHz `--init-run-group`/primary run passes `data_dir=require_77ghz_dir(config)` and a
two-distinct-roots fixture proves the hashes come from the 77 GHz files (C22)** — full detail in
§2.4's "Group provenance" note and §2.5.

### 2.5 `experiments/run_clock_decoupling.py` + `scripts/ibex/run_exp_b.sbatch` +
`scripts/ibex/run_exp_b_variant.sbatch` + `scripts/ibex/submit_exp_b_variant.sh` (new)

**Responsibility.** The Exp B CLI entrypoint and its IBEX batch templates — direct analogues of
`run_regression.py`/`run_exp_a.sbatch` for the primary pooled model, **plus a genuinely
concurrent, properly-staged, separately schedulable path for the session-specific variant**
(C8/C11/C23) rather than folding a 5x-larger sequential search into one job's allocation or
running its I/O-heavy setup step somewhere it doesn't belong.

**CLI.** `--config PATH` (repeatable, required), `--band {10ghz,77ghz}` (default `10ghz`),
`--subset 6subjects` XOR `--full-cohort` (mutually exclusive, one required — reuses the existing
`_validate_flags` pattern), `--session-specific` (new: switches to the variant's ENTIRELY
SEPARATE code path — never combined with `--subset`/`--full-cohort` — and requires exactly one
of the three sub-flags below):
  - `--init-run-group` — **run once, on a COMPUTE node via its own sbatch stage, never on a
    login node and never as an array task** (C14/C23). Builds `sessions` and the band's QC
    manifest exactly as the primary path does, computes `config_hash`, then calls the EXISTING
    `provenance.record_run(config, manifest_qc, folds=None, data_dir=..., extra={...})` exactly
    once, takes `run_dir = run_path.parent` (`record_run` returns the **`provenance.json` file
    path**, not a directory — C19), and prints `run_dir` to stdout. This is the fix for the
    run-group problem: `record_run` creates a fresh timestamped directory and refuses to
    overwrite an existing `provenance.json`, so if each of the four array tasks called it
    independently they would land in four *different* directories with no common one for the
    merger to read — and any two tasks racing to write the same `provenance.json` would conflict.
    Calling it exactly once, before the array is submitted, gives one shared `run_dir` (whose
    directory name doubles as `run_group_id`, C15) with no write race, since every array task
    below only *reads* that `provenance.json` and writes its own uniquely-named shard. **This is
    not a lightweight step (C23):** `record_run`'s `_hash_inputs` sha256-hashes every raw radar
    file in the cohort (tens of GB for 77 GHz) — genuine I/O work that belongs in its own sized
    batch allocation, not "a login step," exactly as the codebase's existing IBEX jobs never do
    real I/O on a login node either. Full call detail, incl. why `folds=None` and where
    `config_hash` comes from, is in §2.4's "Group provenance" note (C19); the batch staging and
    submission wrapper that runs this correctly are below (C23).
  - `--session {1,2,3,4} --run-dir PATH` — run **only** that session's independent search
    (`exp_b.run_exp_b_one_session`), then write `session_specific_{band}_s{session}.json` into
    the **given** `run_dir` (never calling `record_run` itself, C14), embedding the per-shard
    lineage fields (§2.4, C15). This is what one SLURM array task runs (C11) — real
    cross-session concurrency comes from four separate task invocations, never from anything
    inside one process.
  - `--merge-sessions --run-dir PATH` — skip the search entirely; call
    `exp_b.merge_session_specific_reports(band, run_dir)` to fail-closed validate and combine
    whichever of the four per-session shards exist under `run_dir` into
    `session_specific_{band}.json` (schema in §2.4). Mirrors the existing 77 GHz store's
    array-then-merge pattern (`extract77.sbatch`/`submit_extract77.sh`) already established in
    this codebase — including its own use of a shared directory + per-shard fingerprints for
    exactly this kind of fail-closed reassembly.

The variant has no mechanism-only smoke mode of its own; its mechanism is proven by
`T-M8-variant` on the synthetic store, not by an IBEX smoke run.

**Flow (primary, `--subset`/`--full-cohort`).** `load_config` → `protocol_freeze_guard(config)`
pre-flight → `exp_b.build_sessions_b` → (smoke: restrict to
`exp_a.select_subset_subjects(evaluable_subjects_b(sessions), k=6)`) → `load_ground_truth` / QC
manifest (as `run_regression.py`) → `nested_loso_splits` over `evaluable_subjects_b(sessions)` →
`record_run(config, manifest_qc, folds, data_dir=require_77ghz_dir(config) if band == "77ghz"
else None, extra={"stage": f"exp-b-{mode}", "band": ..., "n_eval": ..., "n_sessions": ...})`
(the `data_dir` argument — verified against `provenance.py`'s real signature, C22 — defaults to
the 10 GHz root and MUST be supplied for 77 GHz, exactly as `run_regression.py` already does for
Exp A, or a 77 GHz run would try to hash 77 GHz files against the 10 GHz directory and crash) →
`n_workers` from `SLURM_CPUS_PER_TASK` → `exp_b.run_and_report_b(..., mode=mode)`. This call tree
never touches the variant (§2.4, C6) — there is no shared flag to mis-set.

**Flow (`--session-specific --init-run-group`, driver, once).** `protocol_freeze_guard` →
`build_sessions_b` → `load_ground_truth`/`apply_qc` or `apply_qc_77` to build `manifest_qc` for
the band (`record_run`'s `manifest` argument is REQUIRED, not optional to skip — C19, the same
QC step `run_regression.py` already performs for the primary path) →
`store_mod.validate_store(band, store_dir, exp_a.expected_fingerprints(config, band, sessions),
analysis_commit=analysis_commit)` **before anything else** (C17 — fail fast, before the array is
even submitted, on a stale/wrong store) → `config_hash = exp_b.config_fingerprint(config)` (C20;
no such field exists natively in `record_run`'s schema) → `folds_by_session = {str(s):
provenance.fold_manifest(nested_loso_splits(exp_b.eligible_subjects_for_session(sessions, s)))
for s in (1,2,3,4)}` (satisfies `implementation_plan.md:1273-1276`'s every-run fold-manifest
requirement for real, C21 — not argued around) → **one** `run_path = record_run(config,
manifest_qc, folds=None, data_dir=require_77ghz_dir(config) if band == "77ghz" else None`
(C22) `, extra={"stage": "exp-b-session-specific-group", "band": ..., "config_hash": ...,
"expected_subjects_by_session": {str(s): exp_b.eligible_subjects_for_session(sessions, s) for s
in (1,2,3,4)}, "folds_by_session": folds_by_session})` (`folds=None` for the top-level parameter
because the variant has four independent fold structures, not one — C19/C21;
`expected_subjects_by_session` is the authoritative cohort snapshot the merge validates against,
C17) → `run_dir = run_path.parent` (`record_run` returns the provenance **file** path, not a
directory — C19) → print `run_dir` to stdout for the submission script to capture.

**Flow (`--session-specific --session S --run-dir PATH`, one array task).**
`protocol_freeze_guard` → `build_sessions_b` → `store_mod.validate_store(band, store_dir,
exp_a.expected_fingerprints(config, band, sessions), analysis_commit=analysis_commit)` **before
any fit** (C17 — defense in depth: re-validated independently per task, not assumed still true
from `--init-run-group`'s earlier check) → `exp_b.run_exp_b_one_session(..., S, ...,
n_workers=...)` → `exp_b._assert_mechanism_ok_b(results_s, sessions)` (C18 — the same structural
audit the primary path runs, not skipped for the variant) → write `session_specific_predictions_
{band}_s{S}.csv` / `session_specific_selection_table_{band}_s{S}.csv` /
`session_specific_dropped_folds_{band}_s{S}.csv` via the shared `_write_*_csv` helpers (C18) →
`summarize_variant_session(results_s, S, config)` (now including `selection_frequency`, C18) →
`config_hash = exp_b.config_fingerprint(config)` — the SAME named helper `--init-run-group` used
(C20 — never a second, independently-invented hashing scheme that could silently diverge) →
assemble the per-shard lineage fields (`run_group_id` from
`Path(run_dir).name`, `band`,
`session=S`, `analysis_commit`, `config_hash`, `seed_set`, `n_eval_subjects`) → write
`session_specific_{band}_s{S}.json` into `run_dir`. **Does not call `record_run`** (C14 — avoids
the multi-directory/race problem). **An unexpected exception anywhere in this flow propagates
and exits the process nonzero** (C12) — it is not caught to produce a partial or placeholder
shard, and a `validate_store` failure is exactly such a propagating exception, not a soft warning.

**Flow (`--session-specific --merge-sessions --run-dir PATH`, run once after the array).**
`exp_b.merge_session_specific_reports(band, run_dir)` reads `run_dir/provenance.json` for the
group's authoritative lineage (incl. `expected_subjects_by_session`, C17), fail-closed validates
whichever `session_specific_{band}_s{1..4}.json` shards exist against it (C15/C17), and writes
the combined `session_specific_{band}.json` with `completed_sessions` computed from exactly
which shards are present *and valid*. Does not re-check store validity itself — that was already
enforced per-task before any fit happened (C17); this step validates shard *lineage*.

**Outputs.** Primary path: `metrics_exp_b_{band}.json`, `predictions_b_{band}.csv`,
`selection_table_b_{band}.csv`, `dropped_sessions_{band}.csv`, `scatter_b_{band}.png`, under its
own `results/runs/<stamp>_<commit>/` (one `record_run` call per invocation, as today). Variant
path: **one shared** `results/runs/<stamp>_<commit>/` created by `--init-run-group`, containing
per session (C18) `session_specific_predictions_{band}_s{S}.csv`,
`session_specific_selection_table_{band}_s{S}.csv`, `session_specific_dropped_folds_{band}_s{S}
.csv`, and `session_specific_{band}_s{S}.json`, then the merged `session_specific_{band}.json`
— all four tasks and the merge step write into that **same** directory (C14), never into four
separate ones.

**sbatch — two files, three STAGES, one wrapper (C8/C11/C14/C23).** `run_exp_b.sbatch` clones
`scripts/ibex/run_exp_a.sbatch` verbatim for the **primary** path only (16 cores / 64 G,
`BAND`/`MODE` env, `configs/exp_b.yaml` added to the `--config` chain) — Exp A's measured
wall-time is a reasonable prior for one pooled search over a comparable row count, so cloning it
unmodified is justified here.

`run_exp_b_variant.sbatch` handles all three variant stages through one file, dispatching on a
`STAGE` env var (`init` | `array` | `merge`) at the SHELL level — mirroring `run_exp_a.sbatch`'s
existing `MODE` env-var convention rather than inventing three near-duplicate sbatch files. **The
file's own `#SBATCH` header carries NO resource directives at all** — no `--cpus-per-task`,
`--mem`, `--time`, `--array`, or `--output` (only a generic `--job-name`) (C24). This is
deliberate, not an oversight: `#SBATCH` lines are parsed by `sbatch` directly from the script
text at submission time, **before the shell — and therefore `STAGE` — is ever evaluated**, so a
single file's header cannot vary per stage no matter what the shell body does afterward. Putting
one fixed set of resource numbers in the header would silently apply them to ALL THREE stages
regardless of `STAGE`, which is exactly the bug C24 caught in the previous draft. Every
resource-affecting flag is instead supplied **exclusively via the `sbatch` CLI**, once per stage,
by the wrapper below — the single source of truth for sizing, with no competing in-file default
to accidentally fall back to.
- **`STAGE=init`** — a **single-task job**, sized for I/O (hashing every raw radar file,
  potentially tens of GB for 77 GHz — C23), NOT the array tasks' 16-core/64G compute allocation:
  a modest core count (hashing is sequential per file, not internally parallel) and a wall-time
  budget sized from a measured run, analogous to how the array's own allocation is sized from the
  primary run (not guessed) — supplied via `sbatch --cpus-per-task=1 --mem=8G --time=$INIT_TIME`
  (default `01:00:00`, revised after one measured run, exactly like the array's own wall-time).
  Runs `run_clock_decoupling.py --session-specific --init-run-group` and prints `run_dir` as the
  LAST line of its stdout log.
- **`STAGE=array`** (`sbatch --array=1-4 --cpus-per-task=16 --mem=64G --time=$ARRAY_TIME`, the
  same allocation shape as the primary path) — each task runs `run_clock_decoupling.py
  --session-specific --session $SLURM_ARRAY_TASK_ID --run-dir "$RUN_DIR"` with its own full
  per-session allocation — this is what actually gives cross-session concurrency (wall-time ≈ one
  session's search, if the cluster schedules the 4 tasks concurrently), correcting the previous
  draft's unsupported claim that a single sequential job would "scale with the slowest session"
  (C11: without real concurrency, one job iterating four searches has wall-time ≈ their sum, and
  M7 already demonstrated that even a *single* such search could hit the 4-hour wall before fold
  parallelism existed). `$ARRAY_TIME` is set from the primary full-cohort run's measured
  wall-time (step 10), not assumed.
- **`STAGE=merge`** (`sbatch --cpus-per-task=1 --mem=4G --time=00:15:00`) — a single tiny task
  running `run_clock_decoupling.py --session-specific --merge-sessions --run-dir "$RUN_DIR"`,
  submitted with a SLURM dependency on the array (`--dependency=afterany:<array-job-id>`) so it
  only runs once every task has finished or failed.

**`scripts/ibex/submit_exp_b_variant.sh` (new, named artifact — C23)** — the orchestration
wrapper, mirroring `submit_extract77.sh`'s git-capture/dirty-tree-refusal idiom (verified against
its actual source, not assumed) rather than inventing a new convention. Also owns the per-stage
`--output` naming (C25 — a static in-file `#SBATCH --output` cannot bake in a `STAGE`-derived
name any more than resource directives can, for the identical reason) and normalizes
`sbatch --parsable`'s return value before using it in a path or a `--dependency` (C25 — on a
multi-cluster SLURM setup `--parsable` can return `jobid;cluster`, which is not a valid `%j`
substitution or a bare job ID):
```bash
set -euo pipefail; cd "$(git rev-parse --show-toplevel)"
[ -z "$(git status --porcelain)" ] || { echo "dirty tree, refusing"; exit 1; }
export DEHYD_GIT_COMMIT="$(git rev-parse HEAD)" DEHYD_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" DEHYD_GIT_DIRTY="false"
mkdir -p logs

init_raw=$(sbatch --wait --parsable --cpus-per-task=1 --mem=8G --time="${INIT_TIME:-01:00:00}" \
    --output=logs/exp_b_variant_init_%j.out --error=logs/exp_b_variant_init_%j.err \
    --export=ALL,STAGE=init,BAND,DEHYD_GIT_COMMIT,DEHYD_GIT_BRANCH,DEHYD_GIT_DIRTY \
    scripts/ibex/run_exp_b_variant.sbatch)
init_job_id="${init_raw%%;*}"   # strip a possible ";<cluster>" suffix (C25) -- %j in the
                                 # --output pattern above expands to this SAME numeric ID,
                                 # so the two are guaranteed to agree; never parsed apart.
# `--wait` blocks here until the init stage finishes; a nonzero sbatch/job exit makes this
# script exit nonzero too (set -e), so the array is NEVER submitted after a failed init (C23).
run_dir=$(tail -n1 "logs/exp_b_variant_init_${init_job_id}.out")
[ -d "$run_dir" ] || { echo "init did not produce a valid run_dir: $run_dir"; exit 1; }

array_raw=$(sbatch --parsable --array=1-4 --cpus-per-task=16 --mem=64G --time="${ARRAY_TIME:?set from step 10's measurement}" \
    --output=logs/exp_b_variant_array_%A_%a.out --error=logs/exp_b_variant_array_%A_%a.err \
    --export=ALL,STAGE=array,BAND,RUN_DIR="$run_dir" scripts/ibex/run_exp_b_variant.sbatch)
array_job_id="${array_raw%%;*}"   # same normalization (C25)

sbatch --cpus-per-task=1 --mem=4G --time=00:15:00 \
    --output=logs/exp_b_variant_merge_%j.out --error=logs/exp_b_variant_merge_%j.err \
    --dependency=afterany:"$array_job_id" \
    --export=ALL,STAGE=merge,BAND,RUN_DIR="$run_dir" scripts/ibex/run_exp_b_variant.sbatch
```
Machine-readable `run_dir` capture (C23/C25): the init stage's own `%j`-named `.out` log file is
the handoff — `run_dir` is that file's stdout LAST line, read back with `sbatch --wait` blocking
the wrapper until the file is complete, not a fragile background poll, and located using the
`init_job_id` normalized from `--parsable`'s output (never the raw, possibly `;cluster`-suffixed
string). Fails before submitting the array if initialization fails (`set -e` on `sbatch --wait`'s
exit code, plus an explicit directory check); obtains one `run_dir` value and threads it
unchanged into both the array's and the merge job's `--export`; the same normalization pattern
is applied to `array_raw` before it is used in `--dependency=afterany:...`.

**Acceptance** (`T-M8-entrypoint`, `T-M8-variant`). Synthetic-store end-to-end on 6 fabricated
subjects: no `session_idx == 0` row anywhere; subjects with 0 eligible S1–S4 sessions excluded
*before* `nested_loso_splits` is called; `--subset`/`--full-cohort` XOR enforced,
`--session-specific` mutually exclusive with both and requires exactly one of
`--init-run-group`/`--session`/`--merge-sessions`; smoke surfaces **no** performance value on the
primary path (no metrics/predictions/scatter file written, no MAE/RMSE/r token in stdout — the
C9/C14 precedent from Exp A); serial vs `n_workers=2` bit-identical. **`--init-run-group`, against
the REAL `record_run` contract, not a stub (C19/C20/C21/C22):** exactly one
`record_run(config, manifest_qc, folds=None, data_dir=..., extra={...})` call; the printed value
is a **directory** containing `provenance.json` (i.e. `run_path.parent`, verified by
`Path(printed).is_dir()` and `(Path(printed) / "provenance.json").exists()` — catching the
earlier draft's file-vs-directory bug); that `provenance.json`'s `git.commit` is populated (the
native field the merge reads for `analysis_commit`); its **`extra`-nested** `config_hash`/
`expected_subjects_by_session`/`folds_by_session` are present at `provenance["extra"][...]`,
**not** at the top level (C20 — a fixture asserting `"config_hash" not in provenance` and
`provenance["extra"]["config_hash"] == exp_b.config_fingerprint(config)` catches the earlier
draft's exact mistake); `folds_by_session` for all four sessions is non-empty and matches
`provenance.fold_manifest(nested_loso_splits(expected_subjects_by_session[s]))` exactly (C21 —
the frozen every-run fold-manifest requirement, `implementation_plan.md:1273-1276`, actually
satisfied, not argued around); **a two-distinct-data-roots fixture (C22)** — separate synthetic
10 GHz and 77 GHz data directories with deliberately different file sets — proves a `band="77ghz"`
`--init-run-group` call hashes `inputs.radar_files` from the 77 GHz root (via
`data_dir=require_77ghz_dir(config)`), not the 10 GHz default, and that a 77 GHz call **omitting**
`data_dir` fails closed (file-not-found) rather than silently hashing the wrong files;
`merge_session_specific_reports` is shown reading each field from exactly its verified location,
not an invented alternative schema. **Orchestration (C23), tested where it is genuinely testable
in Python and documented as an operational check where it is inherently SLURM-specific:** a
fixture where `--init-run-group` itself raises (e.g. a `validate_store` failure) exits nonzero,
proving `submit_exp_b_variant.sh`'s `sbatch --wait` + `set -e` chain would never reach the array
submission line — this is testable directly against `run_clock_decoupling.py`'s exit code, no
SLURM required; the `run_dir` capture logic (last line of stdout) is tested against a fixture
stdout stream containing preflight/log noise before the final printed path, proving the parse is
robust to extra output, not fragile to it. The wrapper script's actual `--array`/`--dependency`
chaining is verified by code review against the exact syntax `submit_extract77.sh`/
`submit_ibex.sh` already use in this repo (not invented), plus a real dry run on IBEX before the
full run — SLURM job dependencies are not something a local pytest run can execute, and this plan
does not claim otherwise. **Per-stage resource sizing (C24) and the log/job-ID handoff (C25) are
verified by dry-run inspection, since `#SBATCH` parsing and `sbatch --parsable`'s return format
are scheduler behaviour, not application code:** submit all three stages against a real (or a
`sacct`-queryable) SLURM instance and confirm via `scontrol show job`/`sacct --format=...`
that the **effective** allocation recorded for each of the three job records matches its own CLI
override (1 core/8 G for `init`, 16 cores/64 G × 4 tasks for `array`, 1 core/4 G for `merge`) —
proving the shared file's empty `#SBATCH` header never silently supplies a stale default for any
stage; and confirm the `init` job's actual log file is named `exp_b_variant_init_<numeric-job-id
>.out` with no `;cluster` suffix anywhere in the path, matching what `submit_exp_b_variant.sh`'s
normalized `init_job_id` computed. `--session-specific
--session S --run-dir PATH`: each of the four sessions produces its own outer folds (distinct
from the pooled model's and from each other, verified directly against `run_exp_b`'s folds on
the same fixture); **`validate_store` is called, and a fixture with a mismatched store/commit at
THIS step also raises before any fit happens** (C17 — defense in depth, independent of
`--init-run-group`'s own check); no `record_run` call happens on this path (C14); a fixture that
forces an unexpected exception inside `run_exp_b_one_session` propagates all the way to a
nonzero process exit — the test asserts it is **not** caught anywhere in this call chain;
`_assert_mechanism_ok_b` runs on this session's results before any output is written, identically
to the primary path (C18); the three per-session CSVs (`predictions`/`selection_table`/
`dropped_folds`) are written using the SAME `_write_*_csv` helpers the primary path uses, and are
non-empty on a fixture where the session has ≥1 completed fold (C18); the written shard carries
`run_group_id`/`band`/`session`/`analysis_commit`/`config_hash`/`seed_set`/`n_eval_subjects`,
and its `summary` includes `selection_frequency` (C18). `--merge-sessions`: a fixture with only 2
of the 4 per-session shards present yields `completed_sessions` with exactly those 2 entries,
computed purely by reading the filesystem and validating what is found; **negative merge tests
(C15/C17)**: a shard with a mismatched `analysis_commit`, a mismatched `config_hash`, a `session`
field that disagrees with its own filename, a `run_group_id` that doesn't match `run_dir`'s
name, **or an `n_eval_subjects` that disagrees with the group provenance's
`expected_subjects_by_session[S]`** each cause `merge_session_specific_reports` to raise, naming
the session and the mismatched field — never silently excluded or silently counted.

---

## §3 Tests

| Group | File | What it proves |
|-------|------|-----------------|
| T-M8-objective | test_metrics.py | `equal_session_residual_mae` ≠ `subject_balanced_mae` ≠ naive pooled MAE on an unequal-eligibility fixture, matched to a hand-computed value; averages over sessions present only; NaN on empty |
| T-M8-holm | test_metrics.py | hand-computed 4-p family; step-down monotonicity; clipped at 1.0; input order preserved; NaN occupies a slot; `family_size=4` strictly stronger than `len(p)=3` on identical p-values |
| T-M8-bootstrap | test_metrics.py | `session_weighted_bootstrap` determinism given `rng_seed`; **provably ≠ subject-weighted** on the unequal fixture (the estimand-divergence proof underlying A-M8-1); `y_pred_reference` gives the difference form; an empty-session replicate is skipped-and-counted and can trip `unreliable`; `subject_cluster_bootstrap_pooled` bytewise-unchanged against the step-1 pin after the `_cluster_bootstrap_over_rows` extraction |
| T-M8-mu | test_baselines.py | `session_means` counts distinct **subjects**; `<2` distinct → dropped, never imputed; mutating a val/test target leaves μ_s bytewise identical; dropping session 3 leaves μ_1/μ_2/μ_4 bytewise identical; `fit_session_mean_baseline` emits all-ndarray `FitRecord` params; `predict_session_mean` raises on a dropped/unknown index |
| T-M8-harness-hook | test_harness.py | `score_fn=None` reproduces the step-1 pin bytewise (`inner_scores.tobytes()`, selected candidate, `test_score`); a non-default `score_fn` demonstrably changes the selected candidate on a disagreement fixture; `session_idx=None` bundles unaffected; row-subset provider: masks recompute correctly, zero-row val subject → empty (not crashing) `val_predictions` entry |
| T-M8-provider | test_exp_b.py | X path bytewise identical to `StoreBackedFeatures` on kept rows; `y == raw − μ_s`; `session_idx` aligned post-drop; `extra_fits` carries `session_means` and (tuned branch) `tuned_epsilon`; **drop set candidate-independent**; outer bundle's `session_means` bytewise == the baseline's |
| T-M8-residual-leak | test_exp_b.py | **The Exp B fit-record leakage property** (near-verbatim T16 pattern over Exp B results): mutating an inner-*validation* subject's label (eligibility-preserving) leaves every `inner_train` fit incl. `session_means` bytewise identical; mutating an inner-*train* label **does** move μ_s (power companion); mutating the held-out subject's label leaves all `outer_train` fits identical |
| T-M8-outer-mutation | test_exp_b.py | **(C2) End-to-end synthetic-store outer-mutation property for the REAL Exp B composition** — not just fit-record mutation. On a synthetic store, mutate the held-out subject's stored `vec`/`raw`/`prelog` arrays and its target; re-run the full `SessionResidualFeatures` + drop-row + `equal_session_objective` + Stage-1/Stage-2 search path; assert the fold's drop set, both stages' inner scores and selected winners, μ_s, tuned-ε, and every fitted transform/model parameter are bytewise unchanged, and outer-training predictions are unchanged — only the held-out subject's own predictions/score may move. Reuses the synthetic-store fixture already established for Exp A's equivalent property (M7 DoD D5) |
| T-M8-degenerate | test_exp_b.py | fabricated fold where session 3 has exactly one eligible training subject: absent from the bundle/objective/out-of-fold rows, present in `dropped_sessions`, μ_3 never computed, surviving μ_s unchanged |
| T-M8-report | test_exp_b.py | correct per-session `N_eval`; correct `n_complete_case`; aggregate provably ≠ naive per-subject mean on an unequal fixture; Holm applied with `family_size=4`; `conditional_exploratory` set true; empty complete-case set → NaNs, not an exception; **(C4)** a session with zero out-of-fold rows across the whole run sets `primary_viable=false`/`primary_aggregate=null` with a named reason, never a silently-degraded 3-session mean; **(C5)** per-session `baseline_mae` is a `BootstrapCI`, not a float |
| T-M8-variant | test_exp_b.py, test_run_clock_decoupling.py | **(C6/C8/C11/C12/C14/C15/C16/C17/C18/C19/C20/C21/C22/C23/C24/C25)** `run_exp_b_variant.sbatch`'s `#SBATCH` header carries no resource directives at all — verified by inspecting the file itself, since `#SBATCH` is parsed before `STAGE` is known and cannot vary by it (C24); `submit_exp_b_variant.sh`'s `init_job_id`/`array_job_id` normalization strips a `;cluster` suffix from `sbatch --parsable`'s output before any path or `--dependency` use, on a fixture supplying `"12345;ibex"` as the raw value (C25); a fixture where `--init-run-group` itself raises exits nonzero (proving the wrapper's `sbatch --wait`+`set -e` chain would never reach array submission); `run_dir` parsing from a stdout stream with preflight/log noise before the final printed line is robust, not fragile (C23); each of the four session-specific searches (`run_exp_b_one_session`) builds its own outer folds over its own eligible training subjects, provably distinct from the pooled model's folds **and from each other** on the same fixture; `run_and_report_b` has no variant parameter at all, so smoke mode structurally cannot reach it; a fixture forcing an unexpected exception inside `run_exp_b_one_session` propagates uncaught (never downgraded to a placeholder result — only the harness's pre-existing, pre-defined `InnerResult.reason` doctrine may degrade gracefully); `summarize_variant_session`'s output has no `wilcoxon_p`/`holm_p` field, only descriptive CIs, **plus `selection_frequency` (C18)**; **`validate_store` is called and enforced at BOTH `--init-run-group` and every `--session` task, and a mismatched-store fixture raises before any fit at either point (C17)**; **`_assert_mechanism_ok_b` runs on every session-specific result set before any output is written (C18)**; **the three per-session CSVs are written via the same `_write_*_csv` helpers the primary path uses, never a second implementation (C18)**; **against the REAL `record_run` contract (C19/C20): `--init-run-group` calls it with a genuine QC manifest, `folds=None`, and `data_dir` set for 77 GHz (C22); the printed value is a directory containing `provenance.json`; `config_hash`/`expected_subjects_by_session`/`folds_by_session` are readable at `provenance["extra"][...]`, asserted `not in provenance` at the top level (C20); `folds_by_session` matches `provenance.fold_manifest(nested_loso_splits(...))` for all four sessions (C21)**; a shard written by `--session --run-dir` never triggers its own `record_run`, and computes `config_hash` via the same `exp_b.config_fingerprint` helper `--init-run-group` used; **a two-distinct-data-roots fixture (C22)** proves a 77 GHz run hashes from the 77 GHz root, and fails closed if `data_dir` is omitted; `merge_session_specific_reports` on a directory with only 2 of 4 valid per-session shards computes `completed_sessions=[..]` purely from what is present **and passes lineage validation, including the evaluated-cohort check against `expected_subjects_by_session` (C17)**; **negative fixtures (C15/C17)**: a shard with a mismatched `analysis_commit`, `config_hash`, `session`, `run_group_id`, **or `n_eval_subjects`** makes the merge raise, naming the field, rather than being silently excluded or counted |
| T-M8-entrypoint | test_run_clock_decoupling.py | synthetic-store end-to-end; no `session_idx == 0` row anywhere; Exp-B-specific evaluable-subject filtering applied before `nested_loso_splits`; `--subset`/`--full-cohort`/`--session-specific` mutual exclusivity; smoke surfaces no performance value; serial vs `n_workers=2` bit-identical |

`tests/test_no_leakage.py`: **zero changes** — verified by `git diff --exit-code
tests/test_no_leakage.py` as an acceptance step, not merely asserted. Expected total test count:
the M7 baseline (~660) plus the T-M8-* additions above.

---

## §4 Definition of done

Single-phase, per owner decision (Step 0 item 2) — no owner-checkpoint pause, since Exp B's
**core** design was frozen before Exp A was seen (the two named, disclosed completions A-M8-1/
A-M8-2 notwithstanding — §0's invariant) and there is no freeze left to spend on the compute step.

| ID | Criterion |
|----|-----------|
| D0 | This plan reviewed through the Codex⇄Claude loop; all comments applied/withdrawn/owner-deferred; the Step 0b O-B2 item cleared |
| D1 | Full suite green (`uv run python -m pytest`), including every new `T-M8-*` group; `git diff --exit-code tests/test_no_leakage.py` clean |
| D2 | `harness.py`'s `score_fn=None` path is bytewise-identical to the step-1 pre-edit pin (`inner_scores`, selected candidate, `test_score`); a supplied `score_fn` is proven to change the selected candidate on a disagreement fixture |
| D3 | μ_s is audited as a fitted quantity at **both** inner and outer CV levels via `extra_fits`/`FitRecord`; the fit-record leakage property (T-M8-residual-leak) **and** the end-to-end synthetic-store outer-mutation property over the real Exp B composition (T-M8-outer-mutation, C2) are both green |
| D4 | S0 is excluded at the session-spine level; Exp B's own evaluable-subject rule (≥1 eligible S1–S4 session) is applied before `nested_loso_splits`, not Exp A's rule |
| D5 | The degenerate-fold drop rule is implemented, logged (`dropped_sessions_{band}.csv` + metrics JSON), and proven candidate-independent |
| D6 | Both estimands implemented per A-M8-1 (session-weighted aggregate CI as primary, subject-weighted complete-case Wilcoxon as companion) and A-M8-2 (empty-session bootstrap replicates skipped-and-counted); a session globally absent from the out-of-fold data sets `primary_viable=false` rather than a silently-degraded aggregate (C4); per-session `baseline_mae` carries a subject-cluster CI, not a bare float (C5); the Holm-4 exploratory family implemented; all output labelled `conditional_exploratory: true` |
| D7 | The session-specific secondary variant runs as a genuine 4-task SLURM array (`run_exp_b_variant.sbatch STAGE=array`, one task per session calling `run_exp_b_one_session` directly, C11) submitted via `scripts/ibex/submit_exp_b_variant.sh`, with ALL resource sizing (cores/memory/time/array/output) supplied per-stage via explicit `sbatch` CLI flags — the shared file's own `#SBATCH` header carries none, since it cannot vary by a `STAGE` the scheduler parses before the shell runs (C24) — and `sbatch --parsable`'s job-ID output normalized (stripping a possible `;cluster` suffix) before any path or `--dependency` use (C25); the wrapper runs the heavy raw-file-hashing `STAGE=init` stage in its own sized batch allocation — never on a login node — and blocks on it before ever submitting the array (C23), against a single shared run-group directory created by that `--init-run-group` call using `record_run`'s REAL, source-verified contract (genuine QC manifest, `folds=None`, `data_dir` set for 77 GHz, `run_dir = run_path.parent`, C19/C22) (C14); the group's `provenance.json` carries `config_hash`/`expected_subjects_by_session`/`folds_by_session` at their VERIFIED location `provenance["extra"][...]`, not the top level (C20), and `folds_by_session` is a real fold-role manifest via the now-public `provenance.fold_manifest`, satisfying `implementation_plan.md:1273-1276`'s every-run requirement (C21); `validate_store` is called and enforced before any fit, at both `--init-run-group` and every array task independently (C17); each task's `--time` set from the primary run's measured wall-time, not cloned from Exp A's unjustified; an unexpected exception in any task propagates and exits nonzero rather than being caught (C12); `_assert_mechanism_ok_b` runs on every session's results, and per-session predictions/selection-table/dropped-folds CSVs plus `selection_frequency` are written via the same helpers the primary path uses (C18); `summarize_variant_session` reports descriptive CIs only, no p-values (C16); `merge_session_specific_reports`'s `completed_sessions == [1, 2, 3, 4]` for both bands, computed from which per-session shards exist AND pass fail-closed lineage validation — including the evaluated-cohort check against the group provenance's `expected_subjects_by_session` (C17), reading `analysis_commit`/`config_hash` from their verified real locations (C19/C20) — a shorter list, or any lineage mismatch, is NOT this criterion met (C8) |
| D8 | Clean M8 implementation commit exists on `v1_milestone_8` (step 8.5, already including step 0.5's `implementation_plan.md` fix); both 10 GHz and 77 GHz feature stores rebuilt from it and `--validate`-clean (step 8.6) |
| D9 | Both mechanism-only smoke runs (`--subset 6subjects`) complete on CPU, both bands, against the step-D8 stores, assert fold-role disjointness + audit coverage, and emit **no** performance value |
| D10 | Both full-cohort Exp B runs (`--full-cohort`, 10 GHz and 77 GHz) complete on IBEX; all promised artifacts exist and are regenerable by one command |
| D11 | `plans/implementation_plan.md` amended at step 0.5 — before any other M8 source was written (C9) — to reflect A-M8-1 (removing the `:1218-1219` primary-test contradiction) and A-M8-2 (documenting the empty-session bootstrap rule), with the post-Exp-A chronology of both stated plainly, not elided (C3, C7) |
| D12 | HISTORY.md carries a per-step entry (written as work happens) plus the review-close summary; SECOND_CHAPTER.md §7 written **from the full results**, including the two-estimand rationale, the degenerate-drop rule, and the A-M8-1/A-M8-2 decisions **with their actual chronology** (decided 2026-07-27, after Exp A's results were visible) and why that does not compromise them (C3) |

---

## §5 What could go wrong (known traps)

1. **Zero-row `predict` is a hard crash, not a NaN.** Verified directly: `ValueError: Found
   array with 0 sample(s) (shape=(0, 3)) while a minimum of 1 is required by StandardScaler.`
   Reachable if every session the held-out subject has was dropped for its fold (harness.py's
   `_final_refit`), or if every inner-val subject has zero surviving rows. Mitigation: in
   `_run_single_fold_b`, check for surviving test-subject rows **before** calling
   `_final_refit`; if none, return an `ExpBFoldResult` with empty arrays and a `reason` code
   (mirroring the `InnerResult.reason`/C6/C21 doctrine) rather than letting the traceback
   propagate.
2. **Exp B's evaluable subject set is not Exp A's.** A subject with only S0 eligible passes Exp
   A's "≥1 eligible session" rule but fails Exp B's "≥1 eligible S1–S4 session"
   (implementation_plan.md:611-613). Feeding Exp A's subject list into `nested_loso_splits` for
   Exp B gives that subject an outer fold with zero rows once S0 is filtered → trap 1. **This is
   the most likely first bug** — `evaluable_subjects_b` must be computed and applied before
   splitting, not derived from Exp A's helper.
3. **S0 must be excluded at the session-spine level, not at scoring time.** Δm%(S0) ≡ 0 ⇒ μ_0 = 0
   ⇒ every S0 residual is trivially 0 — a free, perfectly-"predicted" session that would deflate
   every MAE and inflate the apparent aggregate skill. Filter in `build_sessions_b`; assert it in
   `SessionResidualFeatures.__init__` and in `_assert_mechanism_ok_b`. Do **not** reuse Exp A's
   `abs(y_true) > 1e-9` heuristic (`exp_a.py:380`) — meaningless once targets are residualized.
4. **A candidate-dependent drop set would silently corrupt selection.** The drop must depend only
   on `(train_subjects, session eligibility)`, never on `feature_key`/`family`/`seed` — otherwise
   `select_candidate` would compare candidates scored on different rows without any signal that
   it happened. Guaranteed by construction today; asserted directly in `T-M8-provider` so a
   future edit cannot silently break it.
5. **A NaN objective poisons the whole candidate with an unhelpful error.**
   `CandidateScore.inner_val_mae = float(np.mean(per_fold))` — one NaN inner fold makes a
   candidate incomparable, and if the cause is candidate-independent (e.g. "no session had any
   inner-val rows this fold"), *every* candidate goes NaN and `select_candidate` raises
   `SelectionError` about non-finite MAE with no hint about session drops. Mitigation:
   pre-check in `_run_single_fold_b` and raise a named `ExpBError` naming the fold, its dropped
   sessions, and the per-session eligible-training-subject counts.
6. **`spawn` pickling discipline.** Build `SessionResidualFeatures` **inside**
   `_run_single_fold_b` (never pass a bound method holding open npz handles across processes);
   `equal_session_objective` must be a **module-level function**, never a lambda or closure
   (Exp A's `before_fit` closure pattern — defined inside the worker — is the model to follow);
   keep `before_fit` itself defined inside the worker exactly as `exp_a.py:255` does.
7. **Determinism of μ_s under parallelism.** `session_means` must iterate sorted sessions and
   sorted subject membership so float accumulation is bit-identical serial vs `n_workers>1` —
   proven by the serial-vs-parallel bit-identity test already established as the Exp A pattern
   (`test_exp_a.py`'s equivalent check).
8. **Never cache the residualized bundle by feature key alone.** Both μ_s and the drop set depend
   on `train_subjects`; any cache added to the wrapper must key on `frozenset(train_subjects)`
   too (matching `StoreBackedFeatures._tuned_cache`'s existing convention), or one fold's μ_s
   would leak into another fold's candidate scoring.
9. **`FeatureBundle.session_idx` must be appended after `extra_fits`, with a `None` default.**
   Verified all four current construction sites (`exp_a.py:211`, `:223`,
   `fixed_feature_provider` in harness.py, `test_harness.py`) pass exactly three positional
   arguments plus `extra_fits=` as a keyword — appending a defaulted field afterward is safe;
   inserting one before it, or making it non-optional, is not.
10. **RNG offset collisions.** Exp A occupies `config.run.seed + 0..3`; Exp B must use its own
    fixed, **named** offsets (starting at +100, per-session offsets as `BASE + session index`
    so they survive a dropped session) — never a running counter, which silently re-maps every
    downstream CI the moment one is added, reordered, or made conditional, changing
    already-reported intervals with no visible diff at the call site. One test asserts all
    resolved offsets, including per-session expansions, are pairwise distinct.
11. **The Stage-1 ridge anchor α was chosen on the raw Δm% scale.** Residual targets have
    materially smaller variance, so the identical anchor α is a *relatively* stronger penalty
    under Exp B. This is **pre-registered** (A-M6-3 mandates reusing the identical search
    space) and must **not** be adjusted — record it plainly in SECOND_CHAPTER.md as a known,
    deliberate consequence of the reuse decision, not silently patched.
12. **Smoke mode must surface no performance value**, matching the C9/C14 doctrine from Exp A.
    Drop counts, `N_eval`, and fold counts are structural and safe to emit even in smoke mode;
    per-session MAEs and any CI are not. Tested by asserting no metrics/predictions file exists
    and no `MAE`/`RMSE`/`r`/`p` token appears in smoke-mode stdout.
13. **Empty complete-case set raises inside `mean_difference_ci`.** Guard before calling it;
    emit `n_complete_case: 0` with NaN fields instead of letting the `ValueError` propagate.
14. **A single job looping over the four session-specific searches has wall-time ≈ their SUM,
    not the slowest one** (C11). `n_workers`/fold-parallelism only parallelizes outer folds
    *within* one session's search; it does nothing across sessions. Claiming "scales with the
    slowest session" without an actual concurrency mechanism is exactly the mistake the previous
    draft made. Mitigation: real concurrency comes only from running the four sessions as
    separate SLURM array tasks (`run_exp_b_variant.sbatch`, `--array=1-4`, each task calling
    `run_exp_b_one_session` for exactly one session with its own full allocation), sized from the
    primary run's measured wall-time — not from a sequential loop inside one job.
15. **Do not let "partial completion should be visible" turn into "catch exceptions and report
    them as partial completion"** (C12). Those are different things: an unexpected exception
    (a real bug — a shape mismatch, an unhandled NaN) must propagate and fail the array task
    loudly, exactly like every other unexpected exception in this codebase; only the harness's
    own pre-defined, enumerated non-evaluability conditions (`InnerResult.reason`) may degrade
    gracefully, and that mechanism already exists and is unchanged by Exp B. Visibility of partial
    completion comes for free from the array/merge design (a crashed task simply never writes its
    per-session file, so `merge_session_specific_reports` sees it absent) — no in-process
    exception handling should be added to `run_exp_b_one_session` to produce that visibility.
16. **Four independent `record_run` calls would land in four different directories with no
    common one to merge from** (C14). `provenance.record_run` creates a fresh, microsecond-stamped
    directory every call and refuses to overwrite an existing `provenance.json` — so if each
    array task called it, not only would there be no shared place for `merge_session_specific_
    reports` to look, but nothing about the four calls would even correlate them as one run. This
    is arguably the most likely first bug in the variant path (analogous to trap 2's "most likely
    first bug" for the primary path), because it looks correct in isolation — each task runs, each
    task calls the existing entrypoint machinery, each task succeeds — and only fails at the merge
    step, possibly silently (an empty merge with no shards found, if no one notices). Mitigation:
    the driver-only `--init-run-group` call, run once before array submission, with the resulting
    `run_dir` threaded to every task and the merger via `--run-dir` — never re-derived per task.
17. **Counting shard file existence is not the same as validating the shard belongs there**
    (C15). A stale file left over from a previous, differently-configured attempt at the same
    `run_dir` (or one manually copied in) would satisfy a naive `os.path.exists` check and silently
    corrupt `completed_sessions` and the merged report. This is exactly the failure mode
    `store.validate_store`'s fail-closed `_check_match` already exists to prevent for feature-store
    shards — `merge_session_specific_reports` must apply the same discipline (commit/config/
    session/run-group identity checks against the group's own `provenance.json`) rather than
    inventing a weaker, ad hoc check, or trusting that a file's mere presence means it is correct.
18. **Redesigning the run-group/merge mechanism (C14/C15) is not the same as re-establishing the
    store-fingerprint discipline the primary path already has** (C17, blocking). It is easy to
    ship a variant execution path that fits models against whatever store happens to be on disk
    without ever calling `validate_store` — nothing in the harness itself enforces this; only the
    entrypoint calling it, as `run_and_report_b` already does for the primary path, does. A stale
    or wrong-commit store would then produce silently invalid variant results that no amount of
    shard-lineage checking (C15) can catch after the fact, because lineage checking only compares
    shards to each other/to the group's own claims — it cannot detect that the group's claims
    themselves were fit against the wrong data. Mitigation: `validate_store` is called
    independently at BOTH `--init-run-group` (fail fast before the array is submitted) and every
    `--session` task (defense in depth, since array tasks may run at different wall-clock times
    on a shared cluster) — never assumed to still hold from an earlier check.
19. **A per-session summary shard is not a substitute for the per-fold artifacts every other
    experiment in this project produces** (C18). `implementation_plan.md`'s mandatory selection-
    frequency/stability table and the project's regenerable-from-saved-intermediate-artifacts
    standard (ROADMAP.md) apply to every experiment, including a "thin secondary" one — being
    secondary changes how prominently a result is reported, not whether its own intermediate
    artifacts exist. Reusing the primary path's own `_write_*_csv` helpers (rather than writing a
    second, subtly different CSV format for the variant) is what keeps this cheap: it is a call
    to existing code, not a parallel implementation to keep in sync.
20. **Designing against an assumed API instead of the real one** (C19). Three separate mistakes
    fell out of describing `record_run` from memory rather than checking its actual contract:
    treating its return value as a directory when it is the `provenance.json` **file** path;
    omitting its required `manifest` argument entirely; and inventing a `config_hash` field that
    does not exist anywhere in its schema. Each is individually small, but together they would
    have made `--init-run-group` simply fail to run (`TypeError` on the missing argument) or
    silently misbehave (`RUN_DIR` pointing at a file, so every array task's `--run-dir` write
    would fail). Mitigation: every call into an existing module (`record_run`, `validate_store`,
    `expected_fingerprints`, `_write_*_csv`) is verified against that module's actual current
    signature before the plan describes how a caller uses it — not assumed by analogy with how
    a similar-sounding call was described elsewhere in this same document.
21. **Fixing an API-mismatch bug is not the same as re-verifying every field path inside the
    fix** (C20). The first `record_run` correction (trap 20) fixed the return value, the missing
    argument, and named a `config_hash` field — but still asserted that field lived at
    `provenance["config_hash"]` (top level) when the actual body does `payload["extra"] = extra`,
    nesting it under `provenance["extra"]["config_hash"]` instead. A fix that touches an external
    contract needs the SAME source-verification discipline applied to every detail of that fix,
    not just the detail that prompted it — a subsequent round of guessing is exactly as risky as
    the first. Mitigation: `merge_session_specific_reports`'s doc and every acceptance criterion
    now cite the precise JSON path (`provenance["extra"][...]` vs. the native
    `provenance["git"]["commit"]`), copied from the actually-read source, not re-described.
22. **A frozen "every run" requirement is not satisfied by an argument that skips it, however
    well-reasoned the skip** (C21). `folds=None` was individually correct (the variant's four
    independent fold structures genuinely don't fit one `folds` slot) but insufficient alone,
    because `implementation_plan.md:1273-1276` requires every run's provenance to write a fold
    manifest, full stop — "the information is reconstructable" answers a different question
    (data loss) than the one the frozen text asks (was it written). Mitigation: `folds=None` for
    the single-list parameter, but a genuine `folds_by_session` fold-role manifest (built from
    the same `nested_loso_splits` calls the tasks themselves use, serialized via the newly-public
    `provenance.fold_manifest`) goes into `extra`, actually discharging the requirement rather
    than arguing it doesn't apply.
23. **`record_run`'s `data_dir` defaults to the 10 GHz root; nothing about a 77 GHz band makes
    that automatic** (C22). Every new `record_run` call this milestone adds — both the primary
    Exp B entrypoint's and `--init-run-group`'s — must pass `data_dir=require_77ghz_dir(config)`
    when `band == "77ghz"`, mirroring the exact convention `provenance.py`'s own docstring states
    ("77 GHz entrypoints pass `require_77ghz_dir(config)`"). Missing this doesn't silently
    corrupt provenance — it crashes inside `sha256_file` (file not found) — but it would crash on
    the FIRST real 77 GHz session-specific run, discovered late rather than caught by a fixture
    with two distinct synthetic data roots.
24. **Fixing what a call does is not the same as fixing where it runs** (C23). C19/C20/C21/C22
    corrected `--init-run-group`'s arguments and return-value handling, but in doing so gave it
    genuine, heavy I/O work (hashing every raw file for `record_run`) that the earlier "small
    driver step... locally/on a login step" framing never accounted for — login nodes are for
    light interactive work, not sha256-ing tens of GB, and neither `submit_extract77.sh` nor
    `submit_ibex.sh` (the two existing precedents) ever run real I/O there either, on inspection
    of their actual source. A plan section can be internally self-consistent (arguments correct,
    return value handled correctly) while still describing an operation that doesn't belong where
    it's placed. Mitigation: `STAGE=init` is its own sized batch allocation, submitted and waited
    on by a real, named wrapper script (`submit_exp_b_variant.sh`) before the array is ever
    submitted — not an unnamed "submission script" gestured at in prose.
25. **`#SBATCH` directives are parsed from the script text at submission time, before the shell
    — and any runtime dispatch variable like `STAGE` — is ever evaluated** (C24). A single sbatch
    file's header is therefore a single, fixed set of resource numbers no matter how many
    different ways the body branches; "one file, `STAGE`-based dispatch" (a genuinely good idea
    for avoiding near-duplicate files, kept from C23) does NOT extend to "and therefore each
    branch can have different `#SBATCH` resource lines" — that combination silently applies one
    stage's numbers to all three, either wasting allocation or starving one. Mitigation: the file
    carries no resource directives at all; every `--cpus-per-task`/`--mem`/`--time`/`--array` is
    an explicit CLI flag on the specific `sbatch` invocation for that stage, in the wrapper —
    correct SLURM usage (CLI flags override in-file directives) applied deliberately, not
    accidentally relied upon.
26. **`sbatch --parsable` is not guaranteed to return a bare integer** (C25). On a multi-cluster
    SLURM configuration it can return `jobid;cluster`, and code that plugs that string directly
    into a `%j`-style log filename or a `--dependency=afterany:` value would either look for a
    file that doesn't exist (the real file is named with just the numeric ID, since `%j` is
    expanded by SLURM itself) or pass SLURM a malformed dependency spec. Mitigation: normalize
    with `${var%%;*}` (strip everything from the first `;` onward) immediately after every
    `sbatch --parsable` call, before the result is used anywhere — never pass the raw value
    through.

---

## §6 Flagged gaps in implementation_plan.md + proposed amendments

- **A-M8-1 (primary-test resolution) — OWNER-APPROVED, Step 0 item 1, decided 2026-07-27, AFTER
  Exp A's full-cohort results were visible (C3).** `implementation_plan.md:1218-1219` names the
  equal-session aggregate "the single pre-specified primary test", but that aggregate is
  session-weighted while the Statistics section's test form (Wilcoxon, `:1213-1217`) requires the
  subject-weighted complete-case estimand — the two cannot both be primary, and the frozen text
  itself never says which wins; this is a genuine textual contradiction in the source of truth,
  not an open scientific choice. Resolved: **primary = the session-weighted aggregate difference
  CI (`session_weighted_bootstrap` with `y_pred_reference`)**; the subject-weighted complete-case
  Wilcoxon is a companion answering a different question, never conflated with the primary. Both
  remain labelled conditional/exploratory per the pre-existing global rule. **Computation-neutral
  by construction:** both quantities are computed and reported in every run regardless of which
  is labelled primary (§2.4 output shape) — the decision only assigns a label, and the label was
  assigned knowing Exp A's outcome, so the chronology is stated here and in SECOND_CHAPTER.md
  (D11/D12) rather than folded into "frozen before Exp A."
- **A-M8-2 (empty-session bootstrap replicates) — OWNER-APPROVED, Step 0b, decided 2026-07-27,
  also AFTER Exp A's results were visible (C3).** The plan's aggregate definition presupposes all
  four S1–S4 sessions are present in every bootstrap replicate but never states what happens when
  a resample happens to omit every subject holding a given session's remaining rows. Resolved:
  **skip-and-count that replicate**, using the existing pre-registered
  `n_skipped`/`unreliable`/`undefined_metric_skip_threshold_pct=5.0` machinery — never silently
  falling back to a 3-session mean. **Unlike A-M8-1, this is NOT computation-neutral (C10):** it
  changes which replicates enter the bootstrap distribution on the (rare) occasions the edge
  fires, and can therefore move the CI. It is adopted because it is the only treatment consistent
  with how every other undefined-metric value is already handled in `metrics.py`, not because it
  was chosen to favour a particular outcome, and it is expected to essentially never fire at this
  cohort size — but it is a real completion, not a relabelling, and is described that way.
- **Required source-of-truth edit (C7).** A-M8-1 and A-M8-2 resolve ambiguities in
  `plans/implementation_plan.md`, the declared source of truth — that document itself must be
  amended (not just this milestone plan), per the M6/M7 precedent of propagating owner-approved
  amendments back into it. Concretely: rewrite/annotate `:1218-1219` so it no longer contradicts
  `:1213-1217` about which estimand is primary, and add the empty-session bootstrap rule next to
  the aggregate's definition (`:1210-1212`). Both edits state the 2026-07-27, post-Exp-A
  chronology plainly. Tracked as build step 0.5 — **before** any other M8 source is written, so
  the doc fix is already part of the step-8.5 clean commit and no later doc-only commit can
  invalidate the store commit-match lineage that C1 established (C9) — and DoD D11.
- **A gap deliberately NOT filled (C16).** The frozen protocol defines Holm-4 for the primary
  model's own per-session S1–S4 breakdown; it says nothing about a multiplicity rule — or even
  whether a p-value belongs at all — for the four **independently-fitted** session-specific
  secondary models. Filling that gap now (picking "uncorrected", or inventing a four-test family)
  would be a **third** undisclosed post-Exp-A protocol completion, alongside A-M8-1/A-M8-2, and
  this one would have no textual anchor in `implementation_plan.md` to justify it as a completion
  of an existing ambiguity — it would be a bare new invention. Resolution: `summarize_variant_
  session` reports effect sizes and conditional/exploratory bootstrap CIs only, no p-value at
  all (§2.4) — sidestepping the multiplicity question rather than deciding it quietly. This is
  not an amendment (no A-M8-3): it changes nothing about the frozen protocol, only about what
  this thin, explicitly-secondary robustness check chooses to report. A future milestone wanting
  an inferential claim from this variant needs its own owner-approved amendment and
  `implementation_plan.md` propagation, exactly like A-M8-1/A-M8-2.
- No new gaps in the **scientific** protocol are proposed; every modelling/statistics constant
  Exp B consumes (search space, budget K, seed set, bootstrap parameters, Holm family size) is
  taken as frozen at M6, exactly as A-M6-3 specifies.

---

## §7 Open items this milestone resolves or carries

**Resolves:** the pluggable-objective harness hook (Exp A's path proven bit-for-bit unaffected);
the train-only session-mean residualization at both CV levels, audited exactly like any other
fitted quantity; the degenerate-fold drop rule, implemented and logged; the two Exp B statistical
estimands and their relationship (A-M8-1); the empty-session bootstrap edge case (A-M8-2); the
session-specific secondary variant honouring the frozen config flag; full-cohort Exp B results on
both bands.

**Fixed here, NOT open at M9+:** the `score_fn`/`FeatureBundle.session_idx` harness contract (a
future experiment needing a different grouped objective reuses this hook, not a new one); the
`session_means`/`fit_session_mean_baseline` shared-computation pattern for any future
session-structured baseline; the A-M8-1/A-M8-2 resolutions themselves — reopening either needs a
prior authoritative amendment, per the standing M7 precedent (§7:846-849) for what stays fixed
across milestones.

**Carries to M9+:** Experiment E's interpretability analysis, which runs on the Exp B (not Exp A)
model as primary (`implementation_plan.md:927-934`) and therefore depends on this milestone's
output; Experiments C/D/F/G/H, which reuse this same harness and the `score_fn` hook it now
exposes.

---

## Plan review (Codex ⇄ Claude)

Status: REVIEW_COMPLETE
Codex: NO MORE COMMENTS (2026-07-27)

### Open comments

_(none)_

### Debate comments

_(none)_

### Deferred to owner

_(none)_

### Resolved review comments

- C1 (applied): store commit/rebuild gate missing before smokes/full runs → added build steps 8.5 (clean M8 commit) and 8.6 (rebuild + `--validate` both bands from that commit) before step 9; added DoD D8 requiring both.
- C2 (applied): residual-leak test only covers fit-record mutation, not the real new composition → added `T-M8-outer-mutation`, an end-to-end synthetic-store outer-mutation property over `SessionResidualFeatures` + drop rows + `equal_session_objective` + both search stages, asserting drop sets/scores/winners/μ_s/tuned-ε/fitted state/training predictions unchanged; only held-out predictions/score may move. Referenced in §2.4 acceptance and DoD D3.
- C3 (applied): chronology of A-M8-1/A-M8-2 was folded into "frozen before Exp A" language → rewrote the milestone-8 invariant blockquote, Step 0 item 1, and Step 0b to state plainly that both were decided 2026-07-27 *after* Exp A's results were visible. (C10 subsequently corrected the "computation-neutral" framing this entry originally used for both — see below.)
- C4 (applied): `equal_session_residual_mae` could silently degrade to a 3-session primary if a session is globally absent from OOF data → added a run-level viability check (`primary_viable`/`primary_unavailable_reason`/`primary_aggregate: null`) to `summarize_exp_b`'s output shape, distinct from A-M8-2's per-replicate rule; added the doc clause on `equal_session_residual_mae` itself and an acceptance/test-table fixture (T-M8-report).
- C5 (applied): per-session `baseline_mae` was a bare float while the frozen stats spec requires a CI for Exp B's per-session MAEs → changed it to a `BootstrapCI` computed by the same subject-cluster bootstrap/seed rules as `radar_mae`; updated the output shape, §2.4 acceptance, and the T-M8-report row.
- C6 (applied): session-specific variant orchestration, output schema, and smoke-mode boundary were unspecified → originally added `include_variant` to `run_and_report_b`; **superseded by the C11/C12 redesign** below (dropped `include_variant` entirely — the variant is now a structurally separate call tree, never a flag on the primary path). The output schema (`summarize_variant_session`, `session_specific_{band}.json` with `completed_sessions`) and `T-M8-variant` stand, updated for that redesign.
- C7 (applied): A-M8-1/A-M8-2 resolve ambiguities in `implementation_plan.md` but no step propagated that back into the source of truth → added a propagation step and DoD D11 requiring `implementation_plan.md` itself be amended (removing the `:1218-1219` contradiction, documenting the empty-session bootstrap rule). **Resequenced by C9** below.
- C8 (applied): cloning Exp A's 4-hour sbatch allocation for a 5x-larger search (primary + 4 session-specific searches) was unjustified, and partial completion could masquerade as success → gave the variant its own sbatch; **the concurrency mechanism was subsequently corrected by C11** below — the wall-time-sizing and visible-completion intent stands, the execution model changed.
- C9 (applied): the `implementation_plan.md` propagation was sequenced at the old step 11, after the clean commit/store rebuild — a doc commit there would move HEAD again and invalidate the commit-match lineage the clean commit/store rebuild (C1) established → moved to a new step 0.5, right after the review loop closes and before any other M8 source is written, so the doc fix is already part of the step-8.5 clean commit and no later doc-only commit ever lands. Updated §1's step table, DoD D11, and §6's C7 bullet to reference step 0.5.
- C10 (applied): the claim that A-M8-2 is "computation-neutral" like A-M8-1 is false — skip-and-count changes which bootstrap replicates enter the distribution when the edge fires, which can move the CI, unlike A-M8-1 which only relabels an already-computed number → rewrote the milestone-8 invariant blockquote, Step 0b's O-B2 entry, and §6's A-M8-2 bullet to state plainly that A-M8-2 is a real, outcome-affecting (if narrow and rare-firing) completion, reserving "computation-neutral" for A-M8-1 only.
- C11 (applied): the claim that the session-specific variant's wall-time "scales with the slowest session" had no supporting concurrency mechanism — `run_exp_b_session_specific` was one call iterating four searches sequentially (wall-time ≈ their sum), and `n_workers` only parallelizes folds within one session, not across sessions → redesigned around a new `run_exp_b_one_session` (the real unit of work) run as a 4-task SLURM array (`run_exp_b_variant.sbatch --array=1-4`, one task per session via `--session-specific --session S`), with a separate `--merge-sessions` step combining per-session output files — mirroring the existing 77 GHz store's array+merge pattern. `run_exp_b_session_specific` is now explicitly a sequential test-only convenience wrapper, not the real-run path. Updated §1 (steps 7, 8, 10.5), §2.4, §2.5, §3's T-M8-variant, DoD D7, and trap 14.
- C12 (applied): the acceptance criterion let a session search "raise" and be caught into a partial `completed_sessions` result, conflicting with the frozen C6/C21 doctrine that only pre-defined non-evaluability may degrade gracefully and unexpected exceptions must propagate → removed all in-process exception-catching language; `run_exp_b_one_session` now explicitly propagates unexpected exceptions (a real bug crashes the array task, which SLURM records as failed), and `merge_session_specific_reports` determines `completed_sessions` purely by reading which per-session files exist on disk, never by catching anything itself. Updated §2.4's API doc, §2.4/§2.5 acceptance text, T-M8-variant, and added trap 15.
- C13 (applied): the §0 opening paragraph and DoD preamble still said Exp B's design without qualification was frozen before Exp A and would be implemented "exactly as written," contradicting A-M8-1/A-M8-2's own post-Exp-A chronology stated elsewhere → qualified both to "core design" plus the two named, disclosed completions; also fixed the same unqualified phrasing in Step 0 item 2 (the gating rationale) and §1 step 10's rationale column, adding a one-line explanation that A-M8-1/A-M8-2 are reporting/labelling completions, not data-use choices, so they don't reopen the blinding question.
- C14 (applied): each array task independently calling `record_run` would create four separate timestamped directories with no shared one for the merger to read, and would risk a write race on `provenance.json` → introduced a driver-only `--session-specific --init-run-group` CLI mode that calls `record_run` exactly once before the array is submitted, threading the resulting `run_dir` to every array task and the merge step via `--run-dir`; array tasks never call `record_run` themselves. Updated §2.4, §2.5 (CLI/Flow/sbatch/Acceptance), §1 step 10.5, DoD D7, and added trap 16.
- C15 (applied): `merge_session_specific_reports` counted file existence alone, which would let a stale or mismatched shard silently pass as complete → added a per-shard schema (`session_specific_{band}_s{S}.json`) embedding `run_group_id`/`band`/`session`/`analysis_commit`/`config_hash`/`store_commit`/`seed_set`/`n_eval_subjects`, and made the merge fail-closed validate every present shard against the run-group's own `provenance.json` (mirroring `store.validate_store`'s `_check_match` precedent), raising on any mismatch rather than excluding or counting it. Added negative-merge acceptance criteria and trap 17.
- C16 (applied): reporting four unadjusted session-specific Wilcoxon p-values with an undiscussed "no Holm correction" was itself a third undisclosed post-Exp-A protocol completion, since the frozen protocol defines Holm-4 only for the primary model's own per-session breakdown and authorizes no multiplicity rule (or even a p-value) for these four independently-fitted secondary models → made `summarize_variant_session` descriptive-only (effect-size CIs, no `wilcoxon_p`/`holm_p` field at all), sidestepping the multiplicity question rather than deciding it quietly; documented as a deliberately-unfilled gap in §6 (no A-M8-3), not a new amendment. Updated §2.4's schemas/docs, §2.5's acceptance text, T-M8-variant, and DoD D7.
- C17 (applied, blocking): the redesigned array/merge flow (C11/C14/C15) never actually called `validate_store` before fitting, so a task could silently consume a stale or wrong-commit store, and the merge validated shard lineage against the group's own claims but had no authoritative check on the evaluated cohort → added `validate_store` calls (with `exp_a.expected_fingerprints`) at BOTH `--init-run-group` (fail fast before the array is submitted) and every `--session` task independently (defense in depth); added `eligible_subjects_for_session` as the one shared definition used both to build folds and to populate a new `expected_subjects_by_session` field in the group's `provenance.json` (via `record_run`'s existing `extra` mechanism, no signature change); `merge_session_specific_reports` now validates each shard's `n_eval_subjects` against that authoritative snapshot exactly, raising on mismatch. Dropped the redundant `store_commit` shard field (now implied by the enforced `validate_store` check). Updated §2.4, §2.5 (all three flows + acceptance), §3's T-M8-variant, DoD D7, and added trap 18.
- C18 (applied): the variant's summary-only shard discarded per-fold OOF predictions, selections, dropped-fold records, and selection-frequency — missing the mandatory selection-stability table every experiment requires and breaking regenerability from saved intermediate artifacts → extracted `_write_predictions_csv`/`_write_selection_table_csv`/`_write_dropped_folds_csv` from `write_exp_b_reports` so the primary path and the variant share one implementation; each `--session` task now also runs `_assert_mechanism_ok_b` and writes `session_specific_{predictions,selection_table,dropped_folds}_{band}_s{S}.csv`; `summarize_variant_session` gained a `selection_frequency` field (reusing `exp_a._selection_frequency` unchanged). Updated §2.4, §2.5 (Flow/Outputs/Acceptance), §3's T-M8-variant, DoD D7, and added trap 19.
- C19 (applied): `--init-run-group` was designed against an assumed `record_run` API rather than its real one — the plan skipped the required `manifest` argument, treated the returned `provenance.json` FILE path as a directory, invented a `config_hash` field that doesn't exist in the schema, and never said what to do with the `folds` parameter for a run-group covering four independent fold structures → fixed all four: `--init-run-group` now builds the band's QC manifest exactly as the primary path does before calling `record_run`; takes `run_dir = run_path.parent` (verified against the actual source, which does the same in `run_regression.py`); computes `config_hash` itself and supplies it (plus `expected_subjects_by_session`) via `record_run`'s existing `extra` mechanism, since no native field exists; and passes `folds=None` with an explicit rationale (four independent per-session fold lists don't fit a single-fold-list parameter, and each is deterministically reconstructable from `expected_subjects_by_session[s]` anyway, so nothing is lost). `merge_session_specific_reports` now names its exact authoritative sources: `git.commit` (native) vs. `config_hash`/`expected_subjects_by_session` (`extra`-derived). Dropped the redundant `store_commit` field. Updated §2.4 ("Group provenance" note + `merge_session_specific_reports` doc), §2.5 (CLI bullet + both flows + acceptance), §3's T-M8-variant, DoD D7, and added trap 20. (C20 subsequently corrected the "extra-derived TOP-LEVEL fields" claim this entry describes — see below.)
- C20 (applied): read the actual `src/dehyd/provenance.py` source (rather than continuing to reason from memory, per trap 20's own lesson) and confirmed the plan's C19 fix still had the schema wrong — `record_run`'s body does `if extra: payload["extra"] = extra`, so `extra` content NESTS under `provenance["extra"]`, it is not flattened to the top level → corrected every reference from `provenance["config_hash"]`/`provenance["expected_subjects_by_session"]` to `provenance["extra"]["config_hash"]`/`provenance["extra"]["expected_subjects_by_session"]`; replaced the vague "canonical JSON hash" placeholder with a named helper `exp_b.config_fingerprint(config)` using `config_to_dict` — the exact function `record_run` itself imports from `..config` and uses to populate `payload["config"]` — called identically by `--init-run-group` and every `--session` task. Updated §2.4's "Group provenance" note, `merge_session_specific_reports` doc, per-shard schema, §2.5's flows/acceptance, §3's T-M8-variant, DoD D7, and added trap 21.
- C21 (applied): `folds=None` was individually well-reasoned (the variant's four independent fold structures don't fit one `folds` slot) but insufficient — verified `implementation_plan.md:1273-1276` really does require every run's provenance to write a fold manifest, not conditionally; "reconstructable" answers a different question than "was it written" → kept `folds=None` for the single-list parameter, but added `extra["folds_by_session"]`, a genuine fold-role manifest for all four sessions built from `nested_loso_splits(eligible_subjects_for_session(sessions, s))` and serialized via `provenance.py`'s own `_fold_manifest` logic — renamed to public `fold_manifest` (zero behaviour change, one call-site update) since M8 gives it a legitimate second caller rather than duplicating its serialization. Updated §1 step 7, §2.4's "Group provenance" note, §2.5's `--init-run-group` flow/acceptance, §3's T-M8-variant, DoD D7, and added trap 22.
- C22 (applied): verified `record_run`'s real signature and confirmed `data_dir` defaults to `config.paths.data_10ghz_dir`, with 77 GHz entrypoints required to pass `require_77ghz_dir(config)` — neither the primary Exp B entrypoint's flow nor `--init-run-group` specified this, so a 77 GHz run would try to hash 77 GHz files against the 10 GHz directory and crash inside `sha256_file` → added `data_dir=require_77ghz_dir(config) if band == "77ghz" else None` to both `record_run` calls, and a two-distinct-synthetic-data-roots acceptance fixture proving a 77 GHz call hashes from the 77 GHz root and fails closed if `data_dir` is omitted. Updated §2.5's primary Flow and `--init-run-group` Flow/Acceptance, §3's T-M8-variant, DoD D7, and added trap 23.
- C23 (applied): `--init-run-group`'s corrected call (C19/C20/C21/C22) now does genuine heavy I/O — `record_run`'s raw-file hashing over the full cohort (tens of GB for 77 GHz) — but the plan still described running it "locally/on a login step," which is not a realistic IBEX workflow, and named no actual submission-wrapper file, only the array sbatch → read the two actual precedents (`submit_extract77.sh`, `submit_ibex.sh`) rather than assuming their shape; redesigned `run_exp_b_variant.sbatch` around a `STAGE=init|array|merge` env var (mirroring `run_exp_a.sbatch`'s existing `MODE` convention) with `init` as its own sized batch allocation, not the array tasks' compute allocation; added a new named wrapper `scripts/ibex/submit_exp_b_variant.sh` mirroring the real git-capture/dirty-refusal idiom, using `sbatch --wait` to block on `init` (failing before the array is ever submitted) and reading `run_dir` from that job's own stdout log as the machine-readable handoff, then chaining the array and a `--dependency=afterany`-gated merge job. Added the Python-testable subset of this to T-M8-variant (init-failure exit code, robust `run_dir` parsing) and documented the SLURM-specific chaining as a code-review + real-dry-run check, not something pytest can execute. Updated §1 (steps 8, 10.5), §2.5 (Responsibility/CLI/sbatch/Acceptance), §3's T-M8-variant, DoD D7, and added trap 24. (C24/C25 subsequently corrected the resource-sizing and log/job-ID mechanics this entry's design still got wrong — see below.)
- C24 (applied): the `STAGE`-dispatch design put resource sizing intent in prose but the wrapper pseudocode passed no stage-specific `sbatch` flags, and `#SBATCH` directives in the one shared file are parsed at submission time BEFORE the shell (and `STAGE`) ever runs — so a single header cannot vary per stage, meaning as drafted either init/merge would waste the array's 16-core/64G allocation or the array would be starved to init's size → removed all resource directives from `run_exp_b_variant.sbatch`'s `#SBATCH` header entirely (job-name only) and made every `--cpus-per-task`/`--mem`/`--time`/`--array` an explicit CLI flag on that stage's specific `sbatch` invocation in the wrapper (1 core/8G/`$INIT_TIME` for init, 16 cores/64G/`$ARRAY_TIME` for array, 1 core/4G/15min for merge) — correct SLURM usage (CLI overrides in-file directives) applied deliberately. Added a dry-run acceptance check via `scontrol`/`sacct` confirming each job's EFFECTIVE allocation matches its own override. Updated §2.5 (sbatch description + wrapper script + acceptance), §3's T-M8-variant, DoD D7, and added trap 25.
- C25 (applied): the `run_dir` handoff read `logs/exp_b_variant_init_${init_job_out}.out`, but no `--output` pattern was ever specified for that name (the same "static header can't vary by STAGE" problem as C24, applied to log filenames), and `sbatch --parsable` can return `jobid;cluster` on multi-cluster SLURM, which doesn't match a `%j`-substituted filename or a bare dependency ID → added explicit `--output=logs/exp_b_variant_init_%j.out` (and array/merge equivalents) as CLI flags per stage, and normalized every `sbatch --parsable` result via `${var%%;*}` before using it in a path or `--dependency=afterany:...`, so the `%j`-expanded log filename and the wrapper's own computed job ID are guaranteed to agree. Updated §2.5 (wrapper script + acceptance), §3's T-M8-variant, DoD D7, and added trap 26.
