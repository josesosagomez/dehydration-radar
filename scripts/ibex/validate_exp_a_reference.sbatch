#!/bin/bash
#SBATCH --job-name=dehyd-refgate
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/refgate_%j.out
#SBATCH --error=logs/refgate_%j.err

# Milestone 10 step 1 / step 12 -- the Experiment-A reference gate, as a CPU batch job.
#
# WHY THIS RUNS ON IBEX AND NOT LOCALLY. The plan's step-1 command is written as a local
# one, on the assumption that the M9 stores on the owner's machine are the stores that
# backed the M9 reference runs. They are not: the local 77 GHz store holds 1 of 72 sessions,
# and the local 10 GHz store was built at `dab8f708` while both reference runs executed at
# `f0a46aa6` (HISTORY 2026-08-04: "the stores were rebuilt at f0a46aa6"). The authoritative
# M9 stores are the IBEX ones, so the snapshot has to be taken here or it is not evidence
# about the reference runs at all -- which is exactly what the gate refuses to pretend.
#
# It is a batch job rather than a login-shell command for the ordinary reason: it reads both
# whole feature stores and reconstructs one tuned-eps feature matrix per outer fold, which is a
# real Exp-A-scale computation. Single-threaded numpy is enough (no fits happen), so 4 cores
# is plenty; the memory is for one feature key's raw tensors across the cohort. The 8 h is
# for I/O, not arithmetic: a snapshot sha256s both stores' .npz files (~6 GB) AND every raw
# radar file behind them (~27 GB across the two bands, via `expected_fingerprints`), on GPFS.
# The same work on the local 10 GHz store took minutes, so this is slack, not an estimate.
#
# MODE=snapshot (default) -- BEFORE any structural edit or store rebuild:
#   REF10=results/runs/20260803T143704568296Z_f0a46aa6 \
#   REF77=results/runs/20260803T151715023672Z_f0a46aa6 \
#   sbatch --export=ALL,MODE=snapshot,REF10,REF77 scripts/ibex/validate_exp_a_reference.sbatch
#
# MODE=compare -- AFTER the final store rebuild and the final-commit Exp A reruns:
#   sbatch --export=ALL,MODE=compare scripts/ibex/validate_exp_a_reference.sbatch
#
# HASH_NPZ=0 skips hashing the store .npz files (faster, weaker tamper evidence). Leave it
# on for the real snapshot: those file hashes are part of what stops existing at the rebuild.

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results/milestone10/sources
export OMP_NUM_THREADS=1

MODE=${MODE:-snapshot}
OUT_DIR=${OUT_DIR:-results/milestone10}
MANIFEST=${MANIFEST:-${OUT_DIR}/reference_exp_a_manifest.json}
SOURCES=${SOURCES:-${OUT_DIR}/exp_a_sources.json}

EXTRA=()
[ "${HASH_NPZ:-1}" = "0" ] && EXTRA+=(--no-hash-npz)

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

echo "reference gate: mode=${MODE} commit=${DEHYD_GIT_COMMIT:-<unset>}"

if [ "$MODE" = "snapshot" ]; then
  : "${REF10:?set REF10 to the M9 Exp A 10 GHz run directory}"
  : "${REF77:?set REF77 to the M9 Exp A 77 GHz run directory}"
  "${PYTHON[@]}" -u experiments/validate_exp_a_reference.py --snapshot \
    --config configs/ibex.yaml \
    --reference-10 "$REF10" --reference-77 "$REF77" \
    --output "$MANIFEST" "${EXTRA[@]}"
elif [ "$MODE" = "compare" ]; then
  "${PYTHON[@]}" -u experiments/validate_exp_a_reference.py --compare \
    --config configs/ibex.yaml \
    --reference-manifest "$MANIFEST" \
    --final-10-file "${OUT_DIR}/sources/exp_a_10.txt" \
    --final-77-file "${OUT_DIR}/sources/exp_a_77.txt" \
    --output "$SOURCES" "${EXTRA[@]}"
else
  echo "ERROR: MODE must be 'snapshot' or 'compare', got '${MODE}'" >&2
  exit 2
fi
