#!/bin/bash
#SBATCH --job-name=dehyd-robustness-sharded

# Milestone 10 — Experiment H's R=200 robustness bootstrap, SHARDED across a SLURM array and
# then merged, dispatched by STAGE (array|merge) at the SHELL level. The single-allocation
# path is scripts/ibex/run_robustness.sbatch; use this one when R=200 does not fit one wall.
#
# DELIBERATE: this header carries NO resource directives beyond --job-name, for the same
# reason run_exp_b_variant.sbatch does not — #SBATCH lines are parsed by sbatch from the
# script TEXT at submission time, BEFORE the shell (and therefore STAGE) is ever evaluated,
# so one fixed set of numbers here would silently apply to BOTH stages. The array tasks and
# the merge want very different allocations. Every resource flag comes from
# submit_robustness_sharded.sh instead, the single source of truth for sizing.
#
# WHY SHARDING IS SAFE, in one line: each replicate's drawn cohort is a pure function of its
# own seed tuple [robustness_seed, experiment_code, band_code, replicate], so a contiguous
# replicate range is a complete, self-contained unit of work. Splitting 1..R across tasks
# cannot change a drawn cohort, a fit, or an estimate — only which process computed it. What
# it CAN get wrong is bookkeeping, and the merge refuses a shard set with a gap, an overlap,
# or a shard produced at a different commit / config / cohort (robustness.read_shards).
#
# STAGE=array  each task runs ONE contiguous replicate range and writes a self-attesting
#              shard JSON into SHARD_DIR. It writes NO summary and NO percentile range, by
#              design: a range over a sub-range of replicates is not the estimand, and
#              min_successful=100 is defined against the whole R. Tasks deliberately do not
#              call record_run (that hashes every raw file — tens of GB at 77 GHz — and doing
#              it once per task would be N times the I/O for one run's worth of provenance).
# STAGE=merge  one task: validates every shard's lineage, requires exact coverage of 1..R,
#              computes the full-cohort point estimate ONCE, and writes the run directory
#              plus the five §3 artifacts.
#
# The array task's range comes from SLURM_ARRAY_TASK_ID and SHARD_SIZE:
#   START = SLURM_ARRAY_TASK_ID * SHARD_SIZE + 1 ,  STOP = min(START + SHARD_SIZE - 1, R)
# so `--array=0-19` with SHARD_SIZE=10 covers 1..200 exactly. submit_robustness_sharded.sh
# derives the array bound from R and SHARD_SIZE rather than leaving the two to drift.
#
# Never run this file directly with plain `sbatch` — use submit_robustness_sharded.sh.

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

EXPERIMENT=${EXPERIMENT:-a}
BAND=${BAND:-10ghz}
MODE=${MODE:-full}
REPLICATES=${REPLICATES:-200}
SHARD_SIZE=${SHARD_SIZE:-10}
CFG=configs/exp_a_regression.yaml
[ "$BAND" = "77ghz" ] && CFG=configs/exp_a_regression_77ghz.yaml
CONFIG_ARGS=(--config "$CFG" --config configs/stats.yaml --config configs/ibex.yaml)
if [ "$MODE" = "full" ]; then SEL=(--full-cohort); else SEL=(--subset 6subjects); fi

SHARD_DIR=${SHARD_DIR:?SHARD_DIR must be set (submit_robustness_sharded.sh supplies it)}
STAGE=${STAGE:?STAGE must be one of array|merge (set by submit_robustness_sharded.sh)}

case "$STAGE" in
  array)
    TASK=${SLURM_ARRAY_TASK_ID:?STAGE=array must run as a SLURM array task}
    START=$(( TASK * SHARD_SIZE + 1 ))
    STOP=$(( START + SHARD_SIZE - 1 ))
    [ "$STOP" -gt "$REPLICATES" ] && STOP=$REPLICATES
    if [ "$START" -gt "$REPLICATES" ]; then
      echo "ERROR: array task ${TASK} maps to replicate ${START} > R=${REPLICATES}; the array bound and SHARD_SIZE disagree" >&2
      exit 1
    fi
    echo "Exp H robustness STAGE=array: exp=${EXPERIMENT} band=${BAND} replicates ${START}..${STOP} of ${REPLICATES} commit=${DEHYD_GIT_COMMIT:-<unset>}"
    "${PYTHON[@]}" -u experiments/run_robustness.py "${CONFIG_ARGS[@]}" \
      --experiment "$EXPERIMENT" --band "$BAND" --replicates "$REPLICATES" "${SEL[@]}" \
      --replicate-start "$START" --replicate-stop "$STOP" --shard-out "$SHARD_DIR"
    ;;
  merge)
    echo "Exp H robustness STAGE=merge: exp=${EXPERIMENT} band=${BAND} R=${REPLICATES} shard_dir=${SHARD_DIR}"
    "${PYTHON[@]}" -u experiments/run_robustness.py "${CONFIG_ARGS[@]}" \
      --experiment "$EXPERIMENT" --band "$BAND" --replicates "$REPLICATES" "${SEL[@]}" \
      --merge-shards "$SHARD_DIR"
    ;;
  *)
    echo "ERROR: unknown STAGE '${STAGE}' (expected array|merge)" >&2
    exit 1
    ;;
esac
