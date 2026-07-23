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

# A batch shell is NON-INTERACTIVE: it never sources ~/.bashrc, so the PATH entry the uv
# installer appends there is absent -- this is the `uv: command not found` failure. Add uv's
# install locations explicitly (~/.local/bin for current installers, ~/.cargo/bin for older).
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# sbatch starts in the submission directory; be explicit so relative paths always resolve.
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# GPFS can spuriously fail HDF5 read-opens without this.
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Prefer the venv built by `uv sync --frozen` on the LOGIN node. Compute nodes generally have
# no outbound network, so nothing here may try to resolve or download a package (hence
# --no-sync on the uv fallback).
if [ -x .venv/bin/python ]; then
  PYTHON=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --no-sync python)
else
  echo "ERROR: no .venv/bin/python and no uv on PATH." >&2
  echo "On the LOGIN node, from the repo root, run:  uv sync --frozen" >&2
  exit 1
fi
echo "interpreter  : ${PYTHON[*]}"

"${PYTHON[@]}" experiments/run_qc77.py \
  --config configs/exp_77ghz.yaml \
  --config configs/ibex.yaml
