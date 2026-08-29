#!/bin/bash
#SBATCH --job-name=quality_train_10g
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/quality_train_10g_%j.out
#SBATCH --error=logs/quality_train_10g_%j.err

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# CPU-only classical models.  The full nested searches reuse the frozen Experiment-A/C
# code paths; no GPU allocation is needed.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
IBEX_CONFIG=${IBEX_CONFIG:-configs/ibex_sosagojm.yaml}
[[ -f "$IBEX_CONFIG" ]] || { echo "ERROR: missing $IBEX_CONFIG" >&2; exit 2; }

if [ -x .venv/bin/python ]; then
  PYTHON=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --no-sync python)
else
  echo "ERROR: no .venv/bin/python and no uv on PATH. Run: uv sync --frozen" >&2
  exit 1
fi

"${PYTHON[@]}" -u experiments/run_quality_training_sensitivity.py \
  --config configs/quality_training_sensitivity_10ghz.yaml \
  --base-config "$IBEX_CONFIG" \
  --full-cohort
