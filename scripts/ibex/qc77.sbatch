#!/bin/bash
#SBATCH --job-name=dehyd-qc77
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/qc77_%j.out
#SBATCH --error=logs/qc77_%j.err

# Milestone 5 — cohort 77 GHz QC + semantic axis check (a SINGLE job).
# Writes results/qc/qc_survival_77ghz.csv (authoritative eligibility + axis certificate)
# and qc_frames_77ghz.csv. Histogram-bound (~10^4 frames x 4096 traces) — run once.

set -euo pipefail
mkdir -p logs

# GPFS can spuriously fail HDF5 read-opens without this.
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

uv run python experiments/run_qc77.py \
  --config configs/exp_77ghz.yaml \
  --config configs/ibex.yaml
