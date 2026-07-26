#!/bin/bash
#SBATCH --job-name=dehyd-exp-a
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/exp_a_%j.out
#SBATCH --error=logs/exp_a_%j.err

# Milestone 7 — run Experiment A (the staged nested-LOSO search) as a single CPU batch job.
# The staged search + fold-local tuned-ε reconstruction is memory-heavy (many large raw
# tensors materialised repeatedly); a shared local box can OOM-kill it mid-run. A compute
# node with 32 G + 8 cores runs it comfortably and does not tie up the workstation.
#
# BAND and MODE are passed via env at submit time; the store for BAND must already be built
# (extract10.sbatch / extract77.sbatch) at THIS commit.
#   BAND=10ghz MODE=smoke  sbatch --export=ALL,BAND,MODE,DEHYD_GIT_COMMIT,... run_exp_a.sbatch
#
# MODE=smoke  -> --subset 6subjects (mechanism-only; no performance value surfaced)
# MODE=full   -> --full-cohort      (THE OWNER GATE: spends the config freeze)

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

BAND=${BAND:-10ghz}
MODE=${MODE:-smoke}
CFG=configs/exp_a_regression.yaml
[ "$BAND" = "77ghz" ] && CFG=configs/exp_a_regression_77ghz.yaml
if [ "$MODE" = "full" ]; then SEL=(--full-cohort); else SEL=(--subset 6subjects); fi

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

echo "Exp A: band=${BAND} mode=${MODE} commit=${DEHYD_GIT_COMMIT:-<unset>}"
"${PYTHON[@]}" -u experiments/run_regression.py --config "$CFG" --config configs/ibex.yaml \
  --band "$BAND" "${SEL[@]}"
