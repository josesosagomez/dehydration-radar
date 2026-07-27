#!/bin/bash
#SBATCH --job-name=dehyd-exp-b-variant

# Milestone 8 — the session-specific secondary variant (implementation_plan.md:722-724),
# dispatched by STAGE (init|array|merge) at the SHELL level, mirroring run_exp_a.sbatch's
# existing MODE env-var convention rather than three near-duplicate sbatch files.
#
# DELIBERATE: this header carries NO resource directives (no --cpus-per-task/--mem/--time/
# --array/--output) beyond --job-name. #SBATCH lines are parsed by sbatch directly from the
# script TEXT at submission time, BEFORE the shell — and therefore STAGE — is ever evaluated,
# so a single file's header cannot vary per stage no matter what the shell body does
# afterward. Putting one fixed set of numbers here would silently apply them to ALL THREE
# stages regardless of STAGE. Every resource-affecting flag (including --output) is instead
# supplied EXCLUSIVELY via the sbatch CLI, once per stage, by submit_exp_b_variant.sh — the
# single source of truth for sizing, with no competing in-file default to fall back to.
#
# STAGE=init   a SINGLE-TASK job, sized for I/O (record_run's raw-file hashing — tens of GB
#              for 77 GHz — is genuine I/O work, never a login-node step). Runs
#              --session-specific --init-run-group and prints run_dir as the LAST line of
#              its stdout log (the machine-readable handoff submit_exp_b_variant.sh reads).
# STAGE=array  each task (SLURM_ARRAY_TASK_ID in 1..4) runs ONE session's fully independent
#              search via --session-specific --session $SLURM_ARRAY_TASK_ID --run-dir
#              "$RUN_DIR" — real cross-session concurrency comes from the array itself, never
#              from anything in-process.
# STAGE=merge  a single tiny task combining whichever per-session shards exist under
#              "$RUN_DIR" via --session-specific --merge-sessions --run-dir "$RUN_DIR".
#
# Never run this file directly with plain `sbatch` — use scripts/ibex/submit_exp_b_variant.sh,
# which submits all three stages with the correct per-stage CLI resource overrides and chains
# them (init --wait -> array -> merge --dependency=afterany).

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

BAND=${BAND:-10ghz}
CFG=configs/exp_a_regression.yaml
[ "$BAND" = "77ghz" ] && CFG=configs/exp_a_regression_77ghz.yaml
CONFIG_ARGS=(--config "$CFG" --config configs/exp_b.yaml --config configs/ibex.yaml)

STAGE=${STAGE:?STAGE must be one of init|array|merge (set by submit_exp_b_variant.sh)}

case "$STAGE" in
  init)
    echo "Exp B variant STAGE=init: band=${BAND} commit=${DEHYD_GIT_COMMIT:-<unset>}"
    "${PYTHON[@]}" -u experiments/run_clock_decoupling.py "${CONFIG_ARGS[@]}" \
      --band "$BAND" --session-specific --init-run-group
    ;;
  array)
    RUN_DIR=${RUN_DIR:?RUN_DIR must be set (the --init-run-group-printed directory)}
    echo "Exp B variant STAGE=array: band=${BAND} session=${SLURM_ARRAY_TASK_ID} run_dir=${RUN_DIR} commit=${DEHYD_GIT_COMMIT:-<unset>}"
    "${PYTHON[@]}" -u experiments/run_clock_decoupling.py "${CONFIG_ARGS[@]}" \
      --band "$BAND" --session-specific --session "$SLURM_ARRAY_TASK_ID" --run-dir "$RUN_DIR"
    ;;
  merge)
    RUN_DIR=${RUN_DIR:?RUN_DIR must be set (the --init-run-group-printed directory)}
    echo "Exp B variant STAGE=merge: band=${BAND} run_dir=${RUN_DIR}"
    "${PYTHON[@]}" -u experiments/run_clock_decoupling.py "${CONFIG_ARGS[@]}" \
      --band "$BAND" --session-specific --merge-sessions --run-dir "$RUN_DIR"
    ;;
  *)
    echo "ERROR: unknown STAGE '${STAGE}' (expected init|array|merge)" >&2
    exit 1
    ;;
esac
