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

# A batch shell is NON-INTERACTIVE: it never sources ~/.bashrc, so the PATH entry the uv
# installer appends there is absent -- this is the `uv: command not found` failure.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Prefer the venv built by `uv sync --frozen` on the LOGIN node (compute nodes generally have
# no outbound network, so nothing here may resolve or download a package).
if [ -x .venv/bin/python ]; then
  PYTHON=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --no-sync python)
else
  echo "ERROR: no .venv/bin/python and no uv on PATH." >&2
  echo "On the LOGIN node, from the repo root, run:  uv sync --frozen" >&2
  exit 1
fi

# Fast preflight: fail in SECONDS naming every missing module at once, rather than having a
# multi-hour job trip over them one at a time (this is how the openpyxl gap surfaced).
"${PYTHON[@]}" - <<'PYCHECK'
import importlib.util, sys
need = ["numpy", "scipy", "pandas", "openpyxl", "yaml", "sklearn", "kymatio", "h5py", "dehyd"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    print("ERROR: missing modules: " + ", ".join(missing), file=sys.stderr)
    print("On the LOGIN node, from the repo root, run:  uv sync --frozen", file=sys.stderr)
    sys.exit(1)
print("preflight    : all runtime modules present")
PYCHECK

# Task id -> (subject, session): subject = id/5 + 1, session = SESSION_NAMES[id % 5].
SESSIONS=(8am 10am 12pm 2pm 4pm)
SUBJECT=$(( SLURM_ARRAY_TASK_ID / 5 + 1 ))
SESSION=${SESSIONS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}
echo "task ${SLURM_ARRAY_TASK_ID} -> subject ${SUBJECT} ${SESSION}  (${PYTHON[*]})"

"${PYTHON[@]}" experiments/run_wst77.py \
  --config configs/exp_77ghz.yaml \
  --config configs/ibex.yaml \
  --subject "${SUBJECT}" --session "${SESSION}"
