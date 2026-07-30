#!/bin/bash
#SBATCH --job-name=dehyd-exp-d-cheap
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/exp_d_cheap_%j.out
#SBATCH --error=logs/exp_d_cheap_%j.err

# Milestone 9 — Experiment D's CHEAP half: the two deterministic baselines (physics power
# ratio, session-index lookup) and, invoked separately once every family has merged, the
# frozen comparison stage. Small CPU job: both baselines are K = 1 with no grid, no seed
# dimension and no network — the physics scalar is one 534-pt (10 GHz) / 256-pt (77 GHz) FFT
# per frame, and the session-index baseline reads no radar data at all.
#
#   BAND=10ghz FAMILY=cheap sbatch ... run_exp_d_cheap.sbatch          # both baselines
#   BAND=77ghz FAMILY=physics sbatch ... run_exp_d_cheap.sbatch        # one of them
#   BAND=10ghz FAMILY=comparisons EXP_A_RUN_DIR=... M7_REFERENCE_DIR=... \
#       FAMILY_RUN_DIRS="physics=A session_index=B cnn1d_raw=C ..." sbatch ...
#
# MODE=smoke -> --subset 6subjects (mechanism-only); MODE=full -> --full-cohort.

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

BAND=${BAND:-10ghz}
MODE=${MODE:-smoke}
FAMILY=${FAMILY:-cheap}
CFG=configs/exp_a_regression.yaml
[ "$BAND" = "77ghz" ] && CFG=configs/exp_a_regression_77ghz.yaml
CONFIG_ARGS=(--config "$CFG" --config configs/baselines.yaml --config configs/ibex.yaml)
if [ "$MODE" = "full" ]; then SEL=(--full-cohort); else SEL=(--subset 6subjects); fi

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

echo "Exp D cheap: band=${BAND} family=${FAMILY} mode=${MODE} commit=${DEHYD_GIT_COMMIT:-<unset>}"

case "$FAMILY" in
  cheap)
    for one in physics session_index; do
      "${PYTHON[@]}" -u experiments/run_baselines.py "${CONFIG_ARGS[@]}" \
        --band "$BAND" --family "$one" "${SEL[@]}"
    done
    ;;
  physics|session_index)
    "${PYTHON[@]}" -u experiments/run_baselines.py "${CONFIG_ARGS[@]}" \
      --band "$BAND" --family "$FAMILY" "${SEL[@]}"
    ;;
  comparisons)
    EXP_A_RUN_DIR=${EXP_A_RUN_DIR:?EXP_A_RUN_DIR must be the M9 Exp A re-run directory}
    M7_REFERENCE_DIR=${M7_REFERENCE_DIR:?M7_REFERENCE_DIR must be the M7 artifact directory}
    FAMILY_RUN_DIRS=${FAMILY_RUN_DIRS:?FAMILY_RUN_DIRS must list every NAME=PATH}
    FAMILY_ARGS=()
    for pair in $FAMILY_RUN_DIRS; do FAMILY_ARGS+=(--family-run-dir "$pair"); done
    "${PYTHON[@]}" -u experiments/run_baselines.py "${CONFIG_ARGS[@]}" \
      --band "$BAND" --family comparisons "${SEL[@]}" \
      --exp-a-run-dir "$EXP_A_RUN_DIR" --m7-reference-dir "$M7_REFERENCE_DIR" \
      "${FAMILY_ARGS[@]}"
    ;;
  *)
    echo "ERROR: unknown FAMILY '${FAMILY}' (expected cheap|physics|session_index|comparisons)" >&2
    exit 1
    ;;
esac
