# HANDOFF — resume point for a new chat (MILESTONE 9 CLOSED; next job: **plan milestone 10**)

_Written 2026-08-06. The next chat's job: **produce `plans/MILESTONE_10_PLAN.md`** covering
Experiments **G, E, F, H** → `SECOND_CHAPTER.md` §9. Nothing is blocked, nothing is half-done, the
suite is green and pushed. Read this file, then `plans/implementation_plan.md` lines **1056–1298**
(the frozen E/F/G designs) and **1299–1448** (§Statistics = H). Do **not** read all of HISTORY.md —
its top ~3 entries cover milestone 9; search it for anything older._

## TL;DR

- **Milestone 9 is complete** (all 15 steps: Exp C, Exp D, the exploratory frame split, §8) and the
  post-M9 technical-debt queue is **empty**.
- **Branch `v1_milestone_9a`, HEAD `5b5ff06`, pushed. Working tree clean. 1160 passed / 16 skipped.**
  `git diff --exit-code tests/test_no_leakage.py` is clean.
- **The primary result of the whole study is negative and now stands on four independent legs**
  (A, B, C, D — table below). Milestone 10 is not going to rescue it, and must not be planned as
  though it might.
- **First practical fact for M10: both feature stores are stale.** `5b5ff06` and `c523266` changed
  `src/`, and store validation requires **strict git-commit equality**. Any M10 run therefore costs
  a `REVISION` re-stamp + a **full rebuild of both stores** before the first result. Budget it once,
  at the start, for the whole milestone — not per experiment.

## Where everything is

**Git.** Branch `v1_milestone_9a` (main branch is `main`). Milestone-9 tip commits, newest first:

    5b5ff06  store sidecars record `packages` (built WITH, not only built FROM)
    c523266  fix the merge summary crashing on a COMPLETE merge (PosixPath not serializable)
    14def09  version-control the M7 reference artifacts; ignore dcgm/; per-band timing
    951da3b  SECOND_CHAPTER §8 — Experiments C and D
    3f465ab  (analysis commit of the Exp D results)
    f0a46aa  (analysis commit of the Exp A / B / C results)

**Which commit produced which artifact.** Exp A/B/C artifacts carry `f0a46aa`; Exp D carries
`3f465ab`. The diff between them touches only `experiments/run_baselines.py` (Exp D's driver) and
its test — nothing Exp A/B/C uses. That is a legitimate two-commit result set, not an inconsistency;
say so if an examiner asks.

**IBEX.** `/ibex/user/floresge/dehy_radar_new/` — clone of the public GitHub repo, with `data/` and
`results/features/` preserved in place. `configs/ibex.yaml` holds those four paths. GPU work goes
through `scripts/ibex/*.sbatch` + the `submit_*.sh` wrappers (all four are mode 100755 in git now —
that was a real M9 failure).

**Feature stores.** v2, both bands, last built at `3f465ab` (10 GHz: 73 eligible sessions; 77 GHz:
72). Building is bit-reproducible on IBEX and locally. **Stale as of `5b5ff06`** — see TL;DR.

**Run dirs (local `results/runs/`).**

    20260803T143704568296Z_f0a46aa6   Exp A 10 GHz
    20260803T151715023672Z_f0a46aa6   Exp A 77 GHz
    20260803T143705048534Z_f0a46aa6   Exp C 10 GHz
    20260803T160645780475Z_f0a46aa6   Exp C 77 GHz
    20260803T172827484892Z_f0a46aa6   Exp C 10 GHz (cross-vendor determinism control)
    20260806T104207854321Z_3f465abc   Exp D 10 GHz
    20260806T110156650286Z_3f465abc   Exp D 77 GHz
    20260727T111437230187Z_f36c4fb2   M7 reference, 10 GHz  ] now tracked in git,
    20260727T115046533408Z_f36c4fb2   M7 reference, 77 GHz  ] `-text` in .gitattributes

Exp B's artifacts and the frame-split outputs live under `results/` per their own drivers; the frame
split is quarantined in `results/exploratory_frame_split/`.

## The results milestone 10 must be planned around

| experiment | 10 GHz | 77 GHz |
|---|---|---|
| **A** radar MAE vs session-index baseline | +0.200 [0.145, 0.261] **worse**, p=3.05e-5 | +0.216 [0.129, 0.294] **worse**, p=7.6e-4 |
| **A** pooled predicted-vs-actual r | −0.138 [−0.286, 0.075] | −0.153 [−0.407, 0.174] |
| **B** radar − session-mean baseline (primary) | +0.0475 [0.0230, 0.0749] **worse** | +0.0246 [−0.0066, 0.0756] |
| **C** QWK, arm a / arm b | −0.212 / −0.197 | −0.278 / +0.025 |
| **D** best family (all six) | `session_index` 0.269 | `session_index` 0.278 |

Every arm loses to the clock. The frame-split demonstration (10 GHz Frank-Hall QWK −0.197 → **+0.819**,
80.3% accuracy, same features/models/data) is the measured account of why the original analysis
looked strong. It is allowed in §8 **once**, as a labelled leaky-by-construction demonstration, in
the methods discussion, in no results table — that was the owner's explicit decision.

## What milestone 10 is

ROADMAP §7 item 9: **Fusion (G), interpretability (E), confound check (F), statistics (H)** → §9 of
`SECOND_CHAPTER.md` (currently a one-line stub). Then M11 is figures/tables.

**Status of each: designs are FROZEN, code does not exist.**

| | design | config | code | notes |
|---|---|---|---|---|
| E | plan L1056–1109 | `configs/exp_e.yaml` (17 ln) | none | `ExpEConfig` exists in `config.py` |
| F | plan L1110–1158 | `configs/exp_f.yaml` (13 ln) | none | `ExpFConfig` exists |
| G | plan L1159–1296 | `configs/exp_g_fusion.yaml` (10 ln) | none | `ExpGConfig` exists |
| H | plan L1299–1448 | `configs/stats.yaml` | **mostly built** | see below |

There is no `exp_e.py` / `exp_f.py` / `exp_g.py` and no entrypoint in `experiments/`. The three
configs were transcribed and validated at the **milestone-6 config freeze**, before any result
existed — they are pre-registrations. **They are not to be redesigned in the M10 plan.** Any change
needs an explicit owner-approved amendment with the same post-hoc disclosure obligation §8 carries
for the O-M9-5 amendment.

**H is largely already implemented and in use.** `eval/metrics.py` has BCa subject-cluster
bootstrap, session-weighted bootstrap, `holm_adjusted`, `wilcoxon_signed_rank`, `mean_difference_ci`
and the ordinal metrics — all exercised by A/B/C/D. Two pieces are **not** built: the
**selection-variance robustness bootstrap** (`stats.robustness_replicates_r = 200`,
`robustness_min_distinct_subjects = 4`, `robustness_min_successful_replicates = 100`, classical
models only) exists only as config; and the chapter-level assembly (per-subject spread, the
cross-experiment comparison table). Plan H as *assembly + one new component*, not as a rebuild.

## Three planning questions the M10 plan must answer (raise these first)

1. **What is Exp E for, now?** Its pre-registered purpose is *supporting evidence that a signal is
   physical* — "alignment between the informative band and the expected water-driven permittivity
   shift". There is no signal to support. Permutation importance over a model that does not beat a
   constant is measuring noise structure. The options are: run it as pre-registered and report it as
   a null/uninformative attribution (defensible, and it is pre-registered); reframe it as a
   *negative-control* reading (do importances look like noise, as they should?); or drop it with a
   recorded justification. **This is an owner decision, not a planning default** — E is
   pre-registered, and silently dropping a pre-registered analysis after seeing a null is exactly
   the selective-reporting failure this project is built to avoid. Note E runs on the **Exp B**
   model as primary, which means Exp B artifacts must be regenerated at the M10 commit.
2. **Exp F cannot use heart rate.** ROADMAP §4 says F is a heart-rate confound check; the
   implementation plan already records that **heart rate was reportedly collected but is not in the
   delivered data**. The frozen F design is instead four nested ridge models sharing one clock
   encoding (clock / clock+covariates / clock+radar / clock+radar+covariates) plus the
   algebraic-coupling sensitivity analysis (`m0` and BMI sit in Δm%'s denominator). The plan should
   state the ROADMAP–plan divergence explicitly so the chapter does not read as a silent scope cut.
3. **Exp G fuses two null arms.** Both bands lose to the clock, so the pre-specified primary
   contrast (fused vs 10-only) is a comparison between two things that already failed. It is still
   worth running — it is pre-registered, and "fusion does not rescue it" is a real finding — but the
   plan must say up front what a *positive* fused result would mean if one appeared (almost
   certainly selection noise on 21 α grid points), and pre-commit to that reading before seeing it.
   G also needs the **matched subject-session intersection** population built and reported
   (`N_subjects,G` and the number of matched cells are distinct numbers, both reported).

## Process traps that survived milestone 9 — carry them forward

- **A commit move invalidates both stores** (strict commit equality in `store._check_match`). Plan
  M10 so code lands in as few store-invalidating waves as possible: write all of E/F/G, land once,
  stamp once, rebuild once, then run. This is the single biggest schedule lever in the milestone.
- **Every real bug in M9 lived in an untested success path while every component test was green** —
  the comparison stage (fixtures gave six families one shared `config_hash`, which production never
  does), the merge summary (the test pinned the `Path` shape and never serialized), and the M7
  reference CSVs (nothing would have caught line-ending normalization until the gate failed against
  its own reference). **For each new M10 driver, write one end-to-end test on real lineages**, not
  only component tests. This has cost three separate incidents; do not learn it a fourth time.
- **`core.autocrlf=true` on this machine.** Any byte-exact reference artifact committed to the repo
  needs a `-text` entry in `.gitattributes`.
- **Don't wholesale-replace local `results/runs/`** — it is gitignored except the `*_f36c4fb2/`
  references. Pull specific dirs.
- **`provenance.platform` now records `cpu_model` and `slurm_nodelist`**, and store sidecars now
  record `packages`. Both were added because M9 lost two days to a 5.14e-14 divergence that was
  unreconstructible without them. Use them.
- **Sub-ULP numeric differences are expected across CPU vendors.** O-M9-5 is now a conjunction:
  `selection_table_{band}.csv` byte-identical **and** `max |Δy_pred| ≤ 1e-10`. It is a post-hoc
  amendment and is disclosed as one in §8.
- **The M9 CNN sbatch timings are per-band** — 77 GHz needs more wall time than 10 GHz for the same
  stage; `submit_exp_d_cnn.sh`'s header records how to measure `ARRAY_TIME` / `INIT_TIME`.

## Hard invariants (unchanged, never violate)

LOSO at subject level for every reported result; fit-on-train-only at both CV levels (including Exp
B's residualization μ_s and Exp G's α); no test-set tuning; primary target continuous Δm%; ordinal
metrics only for the 5-class task; folds only from `eval/splits.py`; tie-breaks only via
`eval/selection.py`; numpy backs all reported WST features (GPU authorized only for the Exp D DL
baselines); `protocol_freeze_guard` before every fit/write; `tests/test_no_leakage.py` frozen
(`git diff --exit-code` on it is an acceptance step). GPU training is never claimed bit-deterministic.
The frame split is reported **only** as the labelled demonstration described above. Do not report
frame-level accuracy as a headline, do not claim causal isolation of hydration from time of day, do
not overclaim clinical readiness.

## Chapter state

`SECOND_CHAPTER.md` §0–§8 are complete (framing, protocol, data, preprocessing, WST, 77 GHz
front-end, config freeze, Exp A, Exp B, Exp C+D). §9 is the M10 stub. The agreed chapter arc is:
present the original work as an accepted feasibility study first, then the rigorous rebuild and its
null results — that framing decision is the owner's and is already reflected in §8's closing.
