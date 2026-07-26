#!/bin/bash
#SBATCH --job-name=dehyd-extract77
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-79
#SBATCH --output=logs/extract77_%A_%a.out
#SBATCH --error=logs/extract77_%A_%a.err

# Milestone 7 — build the 77 GHz WST FEATURE STORE as a JOB ARRAY over the 80 (subject,
# session) cells. Each task loads its ONE file and (if eligible) writes one per-session
# store shard (.npz + .fingerprint.json). Merge is implicit: each shard is standalone.
#   git checkout on IBEX : scripts/ibex/submit_ibex.sh scripts/ibex/extract77.sbatch
#   COPIED tree (no git) : stamp `git rev-parse HEAD > REVISION` before copying, then
#                          `sbatch scripts/ibex/extract77.sbatch` directly (provenance reads REVISION).
# Validate:
#   uv run python experiments/extract_features.py --config configs/exp_a_regression_77ghz.yaml --band 77ghz --validate

set -euo pipefail

# A batch shell is NON-INTERACTIVE: it never sources ~/.bashrc, so the uv PATH entry is absent.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

if [ -x .venv/bin/python ]; then
  PYTHON=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --no-sync python)
else
  echo "ERROR: no .venv/bin/python and no uv on PATH. On the LOGIN node run: uv sync --frozen" >&2
  exit 1
fi

"${PYTHON[@]}" - <<'PYCHECK'
import importlib.util, sys
need = ["numpy", "scipy", "pandas", "openpyxl", "yaml", "sklearn", "kymatio", "h5py", "dehyd"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    print("ERROR: missing modules: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print("preflight    : all runtime modules present")
PYCHECK

# Task id -> (subject, session): subject = id/5 + 1, session = SESSION_NAMES[id % 5].
SESSIONS=(8am 10am 12pm 2pm 4pm)
SUBJECT=$(( SLURM_ARRAY_TASK_ID / 5 + 1 ))
SESSION=${SESSIONS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}
echo "task ${SLURM_ARRAY_TASK_ID} -> subject ${SUBJECT} ${SESSION}  (${PYTHON[*]})  commit=${DEHYD_GIT_COMMIT:-<unset>}"

"${PYTHON[@]}" experiments/extract_features.py \
  --config configs/exp_a_regression_77ghz.yaml \
  --config configs/ibex.yaml \
  --band 77ghz --subject "${SUBJECT}" --session "${SESSION}"
