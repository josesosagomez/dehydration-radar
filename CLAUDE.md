# CLAUDE.md — project context

## What this is
A Python rebuild and rigorous extension of a radar-based dehydration-monitoring
study (originally MATLAB). Full spec is in `ROADMAP.md` — read it first, in full,
before planning or writing code. The original paper is in `paper/`. The original
MATLAB code is in `matlab/` and is the reference for preprocessing and WST
parameters.

## Primary modality
10 GHz CN0566 radar. The 77 GHz data is used only for the cross-band fusion
section on the original 16-subject cohort.

## Hard invariants (never violate — a failing check here should stop the build)
1. **Leave-one-subject-out (LOSO) evaluation.** Splits are at the subject level.
   No frame from a held-out subject may appear in training, for any session.
   Frame-level random splitting is not a valid evaluation protocol here.
2. **Fit-on-train-only.** Every fitted transform (scaler, PCA, feature selector,
   class weights) is fit inside the CV loop on training folds only.
3. **No test-set tuning.** Select tilings/hyperparameters via nested CV or a
   held-out subject validation split, never on the test subjects.
4. **Primary target is continuous:** fluid loss as % of baseline body mass.
   5-class S0–S4 is a secondary task and uses ordinal metrics only.
5. Keep `tests/test_no_leakage.py` green at all times.

## Data notes
- Ground truth = body-mass change (Δm) per subject/session. It is the ONLY
  objective hydration reference. There is no temperature or osmolality data.
- Do not assume the raw file format from the paper — inspect a sample file and
  the MATLAB loader before writing loading code.
- Radar data may be large; develop on one subject, then scale to the full set.

## Compute environment
- GPU-based model training runs on **IBEX** (KAUST's Slurm HPC cluster), not
  locally. Any training that needs a GPU must be runnable as a batch job —
  generate `sbatch` submission scripts and keep GPU training out of interactive
  runs.
- Small smoke tests to check the pipeline works end to end (data loading,
  preprocessing, feature extraction, a tiny training run on a subset / on CPU)
  are done **locally**. Make scripts parameterizable so the same code runs as a
  fast local smoke test or as a full GPU job on IBEX — differing only by config
  (data subset size, device, epochs), not by separate code paths.

## Style
- Config-driven runs, seeded, reproducible. Python 3.11+. Pinned environment.
- Prefer clear, tested library code in `src/` over notebook scripts.

## Code style — readable research code, not corporate code
This is a research project. The code will be read, understood, and modified by a
single researcher (and their examiners), not maintained by a large team. Optimize
for a human being able to open a file and follow what it does.
- Write straightforward, linear code. Favor plain functions over deep class
  hierarchies, abstract base classes, factories, or heavy design patterns.
- Don't over-engineer or add layers of indirection "for flexibility" that isn't
  needed. No premature abstraction — solve the actual problem in front of you.
- Keep the signal-processing and ML steps visible and followable, close to the
  math and to how the paper describes them, rather than hidden behind wrappers.
- Comment the *why* (especially the physics/DSP reasoning and any non-obvious
  choices), not the obvious *what*.
- Use clear, descriptive names. A researcher skimming the file should be able to
  tell what each variable and function is without tracing through the codebase.
- Simplicity and readability beat cleverness. If a simpler version is a little
  longer but much easier to understand, prefer it.
- Reproducibility and the no-leakage invariants above still hold — readable does
  not mean sloppy about correctness.

## Project journal files
Maintain three living documents at the repo root. Knowing *when* to write to each
matters as much as what goes in them.

**HISTORY.md — the implementation log (write continuously).**
A running record of every attempt during implementation, so the full history is
followable and can serve as long-term memory across sessions. After each
experiment or attempt resolves, append an entry recording:
- what was tried;
- whether it failed or succeeded, and *why* (the actual reason, not just the
  outcome);
- the concrete parameter values used (filter orders, cutoffs, WST tilings Q and
  invariance scales, model hyperparameters, seeds, data subset sizes) and the
  reasoning for each value.
Write to it as work happens — at the resolution of individual experiments/attempts
— not in one batch at the end. Failures stay in the log; do not delete them.

*Reading HISTORY.md:* a new session should NOT read the whole file by default —
it can be long. On starting up, read only the most recent entries to get oriented
on what's going on. Go further back into HISTORY.md only when you actually need a
specific piece of information from earlier (e.g. why a particular value was chosen,
or whether something was already tried). Structure the file newest-first, or with
clear dated/numbered entry headers, so recent entries are easy to find and older
detail is easy to look up on demand.

**SECOND_CHAPTER.md — thesis chapter material (write at each milestone).**
As each implementation stage/milestone from ROADMAP.md §7 completes, gather the
information needed to write the thesis chapter about this project. This is not a
duplicate of HISTORY.md: it is the distilled, chapter-ready account. Capture the
provenance of everything so the chapter feels complete and nothing is unexplained:
why each parameter value was chosen, why one processing choice was made over an
alternative, what a result means, and how it connects back to the paper's method
and physics. If a reader could ask "where did this number/choice come from?", the
answer should already be here.

**HANDOFF.md — new-session bootstrap (write only when I ask).**
A short summary (max 200 lines) whose only job is to let a fresh Claude Code chat
resume from where the last one stopped: current state, what's done, what's in
progress, next step, and any open decisions. Do **not** update this automatically.
Only write/refresh it when I explicitly ask — that request signals I'm about to
start a new chat.

## File hygiene — keep src clean
Any code or result that is discarded, superseded, proven wrong, or belongs to an
old version of the pipeline must NOT sit alongside the working files. Move it to
an `archive/` folder at the repo root (rename if you prefer a clearer word), with
two subfolders:
- `archive/code/` — retired/superseded Python scripts;
- `archive/results/` — stale or invalidated results and figures.
The working `src/`, `experiments/`, `results/`, and `figures/` folders should only
ever contain current, valid, working material. When you replace or abandon
something, move the old version to `archive/` rather than leaving it in place or
deleting it — and note the move in HISTORY.md.

## Do not
- Report frame-level random-split accuracy as a headline number.
- Claim the study causally isolates hydration from time-of-day.
- Overclaim clinical readiness.
