#!/bin/bash
#SBATCH --job-name=dehyd-exp-e
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/exp_e_%j.out
#SBATCH --error=logs/exp_e_%j.err

# Milestone 10 — Experiment E: leave-one-path-group-out attribution under outer LOSO
# (A-M10-1), one band per job.
#
# SIZED FROM MEASUREMENT, NOT CLONED. This header is deliberately far smaller than
# run_exp_g.sbatch's 32 cores / 24 h: E is the cheap milestone-10 job and the arithmetic is
# not close. There is no inner CV and no search, so one outer fold is exactly `1 + n_paths`
# deterministic scaler+ridge fits on ~50-60 rows. Measured on the pinned stack:
#
#   10 GHz T1      742 paths   16.7 ms per refit   ->  12.4 s per fold  ->  3.3 min serial
#   77 GHz T1_77   424 paths   16.7 ms per refit   ->   7.1 s per fold  ->  1.9 min serial
#
# At --cpus-per-task=16 (one worker per outer fold) the fits are ~15 s of wall clock. The
# hour is for everything else: validate_store re-hashes every raw session file, which is the
# only part of this job that takes real time. 32 G is ~20x the measured footprint (a fold
# holds one 60 x ~10k float64 matrix plus one ablated copy).
#
# BAND and MODE are passed via env at submit time; the store for BAND must already be built
# (extract10.sbatch / extract77.sbatch) at THIS commit.
#   BAND=10ghz MODE=full  sbatch --export=ALL,BAND,MODE,DEHYD_GIT_COMMIT,... run_exp_e.sbatch
#
# MODE=smoke -> --subset 6subjects (mechanism-only; NO importance value surfaced)
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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

BAND=${BAND:-10ghz}
MODE=${MODE:-smoke}
CFG=configs/exp_a_regression.yaml
[ "$BAND" = "77ghz" ] && CFG=configs/exp_a_regression_77ghz.yaml
if [ "$MODE" = "full" ]; then SEL=(--full-cohort); else SEL=(--subset 6subjects); fi

RUN_DIR_OUT=${RUN_DIR_OUT:-}
if [ -n "$RUN_DIR_OUT" ]; then OUT=(--run-dir-out "$RUN_DIR_OUT"); else OUT=(); fi

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

echo "Exp E: band=${BAND} mode=${MODE} commit=${DEHYD_GIT_COMMIT:-<unset>}"
"${PYTHON[@]}" -u experiments/run_interpretability.py \
  --config "$CFG" --config configs/exp_e.yaml --config configs/stats.yaml \
  --config configs/ibex.yaml \
  --band "$BAND" "${SEL[@]}" "${OUT[@]}"
