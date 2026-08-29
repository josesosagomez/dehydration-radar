#!/bin/bash
#SBATCH --job-name=path40-explore
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/path40_exploratory_%j.out
#SBATCH --error=logs/path40_exploratory_%j.err

# One light CPU job: reads the completed diagnostic's compact session table and runs
# (1) subject-level LOSO and (2) a deliberately leaky seeded random-session split.
# No raw radar data and no GPU are needed.

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

DIAGNOSTIC_ROOT=${DIAGNOSTIC_ROOT:-/ibex/user/sosagojm/dehydration_loso_diagnostic}
CONFIG=${CONFIG:-configs/path40_exploratory.yaml}

if [ -x .venv/bin/python ]; then
    PYTHON=(.venv/bin/python)
else
    PYTHON=(uv run --no-sync python)
fi

echo "Path-40 exploratory comparison"
echo "analysis_root=$PWD"
echo "diagnostic_root=$DIAGNOSTIC_ROOT"
echo "config=$CONFIG"

"${PYTHON[@]}" -u experiments/run_path40_exploratory.py \
  --config "$CONFIG" \
  --diagnostic-root "$DIAGNOSTIC_ROOT"

