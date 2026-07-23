#!/bin/bash
#SBATCH --job-name=dehyd-prep77
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/preprocess77_%j.out
#SBATCH --error=logs/preprocess77_%j.err

# Milestone 5 — 77 GHz preprocessing chain diagnostics as a SINGLE job (not an array):
# the chain steps 1-5 are light, it writes exactly one CSV, so there is no shard/merge
# protocol and no array-race on the output. Optional (diagnostic); not required for the
# feature run.

set -euo pipefail
mkdir -p logs

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

uv run python experiments/run_preprocess77.py \
  --config configs/exp_77ghz.yaml \
  --config configs/ibex.yaml
