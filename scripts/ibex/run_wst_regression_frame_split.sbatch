#!/bin/bash
#SBATCH --job-name=wst_reg_leaky
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --constraint=cpu_amd_epyc_9655
#SBATCH --output=logs/wst_regression_frame_split_%j.out
#SBATCH --error=logs/wst_regression_frame_split_%j.err

# Full WST + classical regression under a deliberately leaky five-fold random-frame
# split. This is a protocol demonstration only. It may never replace, tune, or enter the
# valid nested-LOSO Experiment A result.

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"

BAND=${BAND:?submit with BAND=10ghz or BAND=77ghz}
SOURCE_RUN_DIR=${SOURCE_RUN_DIR:?submit with the matching completed Exp A LOSO run directory}
IBEX_CONFIG=${IBEX_CONFIG:-configs/ibex_sosagojm.yaml}

case "$BAND" in
  10ghz) CFG=configs/exp_a_regression.yaml ;;
  77ghz) CFG=configs/exp_a_regression_77ghz.yaml ;;
  *) echo "ERROR: BAND must be 10ghz or 77ghz" >&2; exit 2 ;;
esac

[[ -f "$IBEX_CONFIG" ]] || { echo "ERROR: missing $IBEX_CONFIG" >&2; exit 2; }
[[ -d "$SOURCE_RUN_DIR" ]] || { echo "ERROR: missing source run $SOURCE_RUN_DIR" >&2; exit 2; }

if [[ -x .venv/bin/python ]]; then
  PYTHON=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --no-sync python)
else
  echo "ERROR: no synchronized Python environment; run uv sync --frozen first" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
echo "FULL-WST RANDOM-FRAME REGRESSION (LEAKY, NEVER REPORT AS GENERALIZATION)"
echo "band=$BAND"
echo "source_run=$SOURCE_RUN_DIR"
echo "ibex_config=$IBEX_CONFIG"

"${PYTHON[@]}" -u experiments/run_frame_split_exploratory.py \
  --config "$CFG" \
  --config "$IBEX_CONFIG" \
  --band "$BAND" \
  --task regression \
  --unit radar_wst \
  --source-run-dir "$SOURCE_RUN_DIR"
