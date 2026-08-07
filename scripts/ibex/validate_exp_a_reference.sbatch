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
# and the local 10 GHz store was built at `dab8f708`. The authoritative stores are the IBEX
# ones, so the snapshot has to be taken here or it is not evidence about the reference runs
# at all -- which is exactly what the gate refuses to pretend.
#
# WHY THE REFERENCE IS THE 3f465abc PAIR AND NOT THE f0a46aa6 PAIR THE PLAN NAMES. Verified
# on IBEX: both stores are built at `3f465abcd38d2f9f451c2292779b9fcfe1bb5b52` (73 and 72
# sessions). The `f0a46aa6` stores no longer exist -- the commit move to 3f465ab rebuilt
# them, exactly as HISTORY 2026-08-04 records ("the commit moves, so trap 18 applies again:
# both stores rebuilt, Exp A re-run on both bands"). So the f0a46aa6 runs' FEATURE evidence
# is not recomputable anywhere, while the 3f465abc Exp A pair is backed by the live stores.
# Plan section 1.3 named f0a46aa6 because that was current when it was written; its own
# wording is "validates the CURRENT M9 stores/runs", and this is what current now means.
# The f0a46aa6 runs are still passed in as SUPERSEDED so the manifest records, by the
# O-M9-5 criterion, that the substitution changed none of Exp A's answers.
#
# It is a batch job rather than a login-shell command for the ordinary reason: it reads both
# whole feature stores and reconstructs one tuned-eps feature matrix per outer fold, which is a
# real Exp-A-scale computation. Single-threaded numpy is enough (no fits happen), so 4 cores
# is plenty; the memory is for one feature key's raw tensors across the cohort. The 8 h is
# for I/O, not arithmetic: a snapshot sha256s both stores' .npz files (~6 GB) AND every raw
# radar file behind them (~27 GB across the two bands, via `expected_fingerprints`), on GPFS.
# The same work on the local 10 GHz store took minutes, so this is slack, not an estimate.
#
# MODE=snapshot (default) -- BEFORE any structural edit or store rebuild. REF10/REF77
# default to the 3f465abc pair (the runs the live stores back); SUP10/SUP77 default to the
# superseded f0a46aa6 pair. Both are overridable, neither needs setting for the normal run:
#   scripts/ibex/submit_ibex.sh scripts/ibex/validate_exp_a_reference.sbatch
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

# The Exp A runs the LIVE stores back (both built at 3f465abc). Defaults, not hardcodes:
# override REF10/REF77 if the store commit ever moves again.
REF10=${REF10:-results/runs/20260804T150841445054Z_3f465abc}
REF77=${REF77:-results/runs/20260804T171005433711Z_3f465abc}
# The runs the plan named, whose store is already gone: recorded artifact-only.
SUP10=${SUP10:-results/runs/20260803T143704568296Z_f0a46aa6}
SUP77=${SUP77:-results/runs/20260803T151715023672Z_f0a46aa6}

if [ "$MODE" = "snapshot" ]; then
  for d in "$REF10" "$REF77" "$SUP10" "$SUP77"; do
    [ -d "$d" ] || { echo "ERROR: run directory not found: $d" >&2; exit 2; }
  done
  "${PYTHON[@]}" -u experiments/validate_exp_a_reference.py --snapshot \
    --config configs/ibex.yaml \
    --reference-10 "$REF10" --reference-77 "$REF77" \
    --superseded-10 "$SUP10" --superseded-77 "$SUP77" \
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
