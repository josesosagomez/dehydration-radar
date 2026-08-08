#!/bin/bash
#SBATCH --job-name=dehyd-robustness
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/robustness_%j.out
#SBATCH --error=logs/robustness_%j.err

# Milestone 10 — Experiment H's selection-variance robustness bootstrap for ONE
# (experiment, band), as a single CPU batch job. E/F/G/H are entirely CPU; no GPU here.
#
# ///////////////////////////////////////////////////////////////////////////////////////
# SIZING — READ BEFORE SUBMITTING A FULL R=200 JOB. Owner decision, 2026-08-08.
#
# A replicate is a COMPLETE run of the chosen experiment (enumeration, inner-CV selection,
# refit) on the drawn cohort. Drawing 16 subjects with replacement leaves ~10 distinct on
# average, so a replicate is ~10 outer folds each training on ~9 subjects instead of 15.
# Measured anchor (HISTORY 2026-07-28): a full 16-fold Exp B run took 01:04:20 on 16 cores
# with all folds in one wave, i.e. ~1 core-hour per fold, ~16 core-hours per run. Scaling by
# fold count and training rows puts one replicate at roughly 6 core-hours, so:
#
#     Exp A / Exp B, R=200 :  ~1,200 core-hours per (experiment, band)
#     Exp C,         R=200 :  ~2,000+ core-hours (two Stage-2 arms per fold)
#
# The header below allocates 32 x 24 h = 768 core-hours, which is NOT enough for a full
# R=200 job on its own. Lowering --replicates to fit is not an option: R=200 is
# pre-registered in §2.4, and min_successful=100 is defined against it. The owner's decision:
#
#   * Exp A and Exp B: raise the allocation at submit time, e.g.
#       `sbatch --cpus-per-task=64 --time=24:00:00 ...`  (1,536 core-hours)
#     subject to what the partition actually offers — check `sinfo` first.
#   * Exp C (and any cell that will not fit one allocation): use the SHARDED path instead,
#       `EXPERIMENT=c BAND=10ghz ARRAY_TIME=08:00:00 bash scripts/ibex/submit_robustness_sharded.sh`
#     which splits 1..R across a SLURM array and merges. That is science-neutral: each
#     replicate's cohort is a pure function of its own seed tuple, so a contiguous replicate
#     range is a complete unit of work, and the merge refuses a set with a gap, an overlap,
#     or a differing commit/config/cohort.
#
# Resources here otherwise: the parallel unit is the REPLICATE, not the fold (200
# independent replicates is the coarser axis, and each runs its own folds serially inside),
# and memory is per replicate-worker — each holds one fold's tuned-ε reconstruction at a
# time, the 77 GHz raw tensors being the big ones — so 32 workers want ~128 G. Override per
# submit if a node is smaller, e.g. `sbatch --cpus-per-task=16 --mem=64G ...`.
# ///////////////////////////////////////////////////////////////////////////////////////
#
# EXPERIMENT, BAND, MODE and REPLICATES are passed via env at submit time; the store for
# BAND must already be built (extract10.sbatch / extract77.sbatch) at THIS commit.
#   EXPERIMENT=a BAND=10ghz MODE=full REPLICATES=200 \
#     sbatch --export=ALL,EXPERIMENT,BAND,MODE,REPLICATES,DEHYD_GIT_COMMIT run_robustness.sbatch
#
# MODE=smoke -> --subset 6subjects (mechanism-only; no estimate surfaced, and the frozen
#               min_successful=100 makes a short run report "inconclusive" ON PURPOSE)
# MODE=full  -> --full-cohort

# RUN_DIR_OUT (optional): after a SUCCESSFUL run the driver atomically writes its absolute run
# directory there, which is how milestone-10 assembly builds its explicit run map without ever
# globbing for a latest directory. Unset -> nothing is written.

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
# Cosmetic: each worker re-pins to 1 thread (threadpool_limits) for determinism + no
# oversubscription, so this job-level value is not what governs the math threads.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

EXPERIMENT=${EXPERIMENT:-a}
BAND=${BAND:-10ghz}
MODE=${MODE:-smoke}
REPLICATES=${REPLICATES:-200}
CFG=configs/exp_a_regression.yaml
[ "$BAND" = "77ghz" ] && CFG=configs/exp_a_regression_77ghz.yaml
if [ "$MODE" = "full" ]; then SEL=(--full-cohort); else SEL=(--subset 6subjects); fi

RUN_DIR_OUT=${RUN_DIR_OUT:-}
if [ -n "$RUN_DIR_OUT" ]; then OUT=(--run-dir-out "$RUN_DIR_OUT"); else OUT=(); fi

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

echo "Exp H robustness: experiment=${EXPERIMENT} band=${BAND} mode=${MODE} R=${REPLICATES} commit=${DEHYD_GIT_COMMIT:-<unset>}"
"${PYTHON[@]}" -u experiments/run_robustness.py \
  --config "$CFG" --config configs/stats.yaml --config configs/ibex.yaml \
  --experiment "$EXPERIMENT" --band "$BAND" --replicates "$REPLICATES" "${SEL[@]}" "${OUT[@]}"
