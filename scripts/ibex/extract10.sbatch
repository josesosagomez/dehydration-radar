#!/bin/bash
#SBATCH --job-name=dehyd-extract10
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-79
#SBATCH --output=logs/extract10_%A_%a.out
#SBATCH --error=logs/extract10_%A_%a.err

# Milestone 7 — build the 10 GHz WST FEATURE STORE as a JOB ARRAY over the 80 (subject,
# session) cells. The store build works fine locally (~30 s/session), but running it as an
# array on IBEX parallelises the whole cohort and keeps the ~5 GB off the local disk. Each
# task builds one session's shard (.npz + .fingerprint.json). Submit via
# scripts/ibex/submit_extract10.sh so DEHYD_GIT_COMMIT/_BRANCH/_DIRTY are captured on the
# login node. Validate locally:
#   uv run python experiments/extract_features.py --config configs/exp_a_regression.yaml --band 10ghz --validate

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

if [ -x .venv/bin/python ]; then
  PYTHON=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --no-sync python)
else
  echo "ERROR: no .venv/bin/python and no uv on PATH. On the LOGIN node run: uv sync --frozen" >&2
  exit 1
fi

SESSIONS=(8am 10am 12pm 2pm 4pm)
SUBJECT=$(( SLURM_ARRAY_TASK_ID / 5 + 1 ))
SESSION=${SESSIONS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}
echo "task ${SLURM_ARRAY_TASK_ID} -> subject ${SUBJECT} ${SESSION}  commit=${DEHYD_GIT_COMMIT:-<unset>}"

"${PYTHON[@]}" experiments/extract_features.py \
  --config configs/exp_a_regression.yaml \
  --config configs/ibex.yaml \
  --band 10ghz --subject "${SUBJECT}" --session "${SESSION}"
