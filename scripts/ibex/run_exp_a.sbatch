#!/bin/bash
#SBATCH --job-name=dehyd-exp-a
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/exp_a_%j.out
#SBATCH --error=logs/exp_a_%j.err

# Milestone 7 — run Experiment A (the staged nested-LOSO search) as a single CPU batch job.
# run_regression parallelises the 16 independent outer folds across SLURM_CPUS_PER_TASK worker
# processes (bit-identical to serial), so 16 cores runs all folds in one wave (~a single fold's
# wall-time). Each worker holds its own fold's tuned-ε reconstruction in memory (the 77 GHz raw
# tensors are the big ones), so 16 workers want ~64 G; bump --mem higher if 77 GHz OOMs. Override
# per submit if a node is smaller, e.g. `sbatch --cpus-per-task=8 --mem=32G ...`.
#
# BAND and MODE are passed via env at submit time; the store for BAND must already be built
# (extract10.sbatch / extract77.sbatch) at THIS commit.
#   BAND=10ghz MODE=smoke  sbatch --export=ALL,BAND,MODE,DEHYD_GIT_COMMIT,... run_exp_a.sbatch
#
# MODE=smoke  -> --subset 6subjects (mechanism-only; no performance value surfaced)
# MODE=full   -> --full-cohort      (THE OWNER GATE: spends the config freeze)

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
IBEX_CONFIG=${IBEX_CONFIG:-configs/ibex.yaml}
[[ -f "$IBEX_CONFIG" ]] || { echo "ERROR: missing $IBEX_CONFIG" >&2; exit 2; }
if [ "$MODE" = "full" ]; then SEL=(--full-cohort); else SEL=(--subset 6subjects); fi

RUN_DIR_OUT=${RUN_DIR_OUT:-}
if [ -n "$RUN_DIR_OUT" ]; then OUT=(--run-dir-out "$RUN_DIR_OUT"); else OUT=(); fi

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

echo "Exp A: band=${BAND} mode=${MODE} commit=${DEHYD_GIT_COMMIT:-<unset>}"
"${PYTHON[@]}" -u experiments/run_regression.py --config "$CFG" --config "$IBEX_CONFIG" \
  --band "$BAND" "${SEL[@]}" "${OUT[@]}"
