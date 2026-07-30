#!/bin/bash
#SBATCH --job-name=dehyd-exp-c
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/exp_c_%j.out
#SBATCH --error=logs/exp_c_%j.err

# Milestone 9 — run Experiment C (the ordinal 5-class secondary task, both arms) as a single
# CPU batch job. Cloned from run_exp_b.sbatch (16 cores / 64 G / 04:00), and the clone is
# justified rather than habitual: Exp C reads the SAME feature store, runs the same Stage-1
# enumeration (72 / 9 candidates) plus a comparable Stage-2 candidate count per fold (41 + 3
# against Exp B's 41), and Exp B measured ~1-1.3 h at these resources.
#
# BAND and MODE are passed via env at submit time; the store for BAND must already be built
# (extract10.sbatch / extract77.sbatch) at THIS commit, and it must be a SCHEMA-v2 store —
# a v1 store fails closed on store_version.
#   BAND=10ghz MODE=full sbatch --export=ALL,BAND,MODE,DEHYD_GIT_COMMIT,... run_exp_c.sbatch
#
# MODE=smoke  -> --subset 6subjects (mechanism-only; no performance value surfaced)
# MODE=full   -> --full-cohort      (no owner pause: the C design was frozen before Exp A)

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
# Cosmetic: each worker re-pins to 1 thread (threadpool_limits) for determinism + no
# oversubscription, so this job-level value is not what governs the math threads.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

BAND=${BAND:-10ghz}
MODE=${MODE:-smoke}
CFG=configs/exp_a_regression.yaml
[ "$BAND" = "77ghz" ] && CFG=configs/exp_a_regression_77ghz.yaml
if [ "$MODE" = "full" ]; then SEL=(--full-cohort); else SEL=(--subset 6subjects); fi

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

echo "Exp C (ordinal): band=${BAND} mode=${MODE} commit=${DEHYD_GIT_COMMIT:-<unset>}"
"${PYTHON[@]}" -u experiments/run_ordinal.py \
  --config "$CFG" --config configs/exp_c.yaml --config configs/ibex.yaml \
  --band "$BAND" "${SEL[@]}"
