#!/bin/bash
#SBATCH --job-name=dehyd-wst77
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-79
#SBATCH --output=logs/wst77_%A_%a.out
#SBATCH --error=logs/wst77_%A_%a.err

# Milestone 5 — 77 GHz WST feature diagnostics as a JOB ARRAY over the 80 (subject,
# session) cells. Each task loads its ONE file, axis-certifies it, and (if the cell is
# eligible per qc_survival_77ghz.csv) writes one deterministic shard + fingerprint sidecar.
# Run AFTER qc77.sbatch has produced the survival CSV. Merge locally:
#   uv run python experiments/run_wst77.py --config configs/exp_77ghz.yaml --merge-shards
# Time limit calibrated by the local `run_wst77.py --smoke` timing (~8 s/tiling/frame set).

set -euo pipefail
mkdir -p logs

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Task id -> (subject, session): subject = id/5 + 1, session = SESSION_NAMES[id % 5].
SESSIONS=(8am 10am 12pm 2pm 4pm)
SUBJECT=$(( SLURM_ARRAY_TASK_ID / 5 + 1 ))
SESSION=${SESSIONS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}

uv run python experiments/run_wst77.py \
  --config configs/exp_77ghz.yaml \
  --config configs/ibex.yaml \
  --subject "${SUBJECT}" --session "${SESSION}"
