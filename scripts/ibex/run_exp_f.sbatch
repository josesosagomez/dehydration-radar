#!/bin/bash
#SBATCH --job-name=dehyd-exp-f
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/exp_f_%j.out
#SBATCH --error=logs/exp_f_%j.err

# Milestone 10 — Experiment F: the not-estimable heart-rate record (A-M10-2) plus the
# available clock/static-covariate sensitivity, one band per job.
#
# SIZED FROM MEASUREMENT. Like Exp E, F is cheap and the header says why rather than cloning
# Exp G's. Per outer fold the only expensive cells are the radar models: 8 alphas x 5 inner
# folds + 1 final refit = 41 fits, at ~19.7 ms per fit on a 70 x ~8.9k design, over 5 radar
# model cells (models 3 and 4 across three variants, minus the one pct_reduced reuses):
#
#   41 fits x 19.7 ms = 0.81 s per radar cell  ->  ~4.0 s per fold  ->  ~1.1 min serial
#
# The clock-only and clock+covariate cells are 5-9 column designs and cost microseconds. As
# with E, the hour is for validate_store re-hashing every raw session file, not for the fits.
#
# REQUIRES the approved Exp-A source manifest, which step 12 writes. There is no default and
# no discovery: EXP_A_SOURCES must be an explicit path to an approved exp_a_sources.json, and
# `reference_gate.load_approved_sources` refuses any band it left not_approved.
#
#   BAND=10ghz MODE=full EXP_A_SOURCES=results/milestone10/exp_a_sources.json \
#     sbatch --export=ALL,BAND,MODE,EXP_A_SOURCES,DEHYD_GIT_COMMIT,... run_exp_f.sbatch
#
# MODE=smoke -> --subset 6subjects (mechanism-only; NO contrast value surfaced, but the HR
#               inventory IS written -- it is the registered answer, not a performance value)
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
EXP_A_SOURCES=${EXP_A_SOURCES:?EXP_A_SOURCES must name an approved exp_a_sources.json -- Exp F never discovers it}
CFG=configs/exp_a_regression.yaml
[ "$BAND" = "77ghz" ] && CFG=configs/exp_a_regression_77ghz.yaml
if [ "$MODE" = "full" ]; then SEL=(--full-cohort); else SEL=(--subset 6subjects); fi

RUN_DIR_OUT=${RUN_DIR_OUT:-}
if [ -n "$RUN_DIR_OUT" ]; then OUT=(--run-dir-out "$RUN_DIR_OUT"); else OUT=(); fi

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

echo "Exp F: band=${BAND} mode=${MODE} sources=${EXP_A_SOURCES} commit=${DEHYD_GIT_COMMIT:-<unset>}"
"${PYTHON[@]}" -u experiments/run_confound.py \
  --config "$CFG" --config configs/exp_f.yaml --config configs/stats.yaml \
  --config configs/ibex.yaml \
  --exp-a-sources "$EXP_A_SOURCES" \
  --band "$BAND" "${SEL[@]}" "${OUT[@]}"
