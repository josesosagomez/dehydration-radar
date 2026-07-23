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
echo "interpreter  : ${PYTHON[*]}"

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

"${PYTHON[@]}" experiments/run_preprocess77.py \
  --config configs/exp_77ghz.yaml \
  --config configs/ibex.yaml
