#!/bin/bash
# Submit ONE (experiment, band) robustness job as a sharded array plus a dependent merge.
# COPIED-TREE workflow, exactly as submit_exp_b_variant.sh: this assumes the repo was copied
# to IBEX (not cloned), so there is no .git here and this script never calls git. Provenance
# comes from a REVISION file at the repo root (`git rev-parse HEAD > REVISION`, stamped on the
# machine that HAS git, then copied over WITH the tree); `provenance.py` reads REVISION
# whenever live git and the DEHYD_GIT_* env vars are both absent, so every stage still
# self-attests its revision. DEHYD_GIT_* is explicitly unset below in case a stale value
# leaked in from the calling shell.
#
#   EXPERIMENT=a BAND=10ghz ARRAY_TIME=06:00:00 bash scripts/ibex/submit_robustness_sharded.sh
#   EXPERIMENT=c BAND=77ghz SHARD_SIZE=5 ARRAY_TIME=08:00:00 bash scripts/ibex/submit_robustness_sharded.sh
#
# ARRAY_TIME is REQUIRED (no safe default — size it from a measured single-shard run, the C8
# rule). Sizing arithmetic, from the Exp B anchor (a full 16-fold run took 01:04:20 on 16
# cores with all folds in one wave, i.e. ~1 core-hour per fold): a replicate draws ~10 distinct
# subjects from 16, each fold training on ~9 subjects instead of 15, so one replicate is
# ~6 core-hours and a SHARD_SIZE=10 shard on 16 cores is ~4 h wall for Exp A/B and ~6.5 h for
# Exp C (two Stage-2 arms). Measure one shard, then set ARRAY_TIME from it.
#
# The array bound is DERIVED from REPLICATES and SHARD_SIZE here rather than passed
# separately, so the two cannot drift apart and leave a gap in 1..R — the merge would refuse
# the set, but it would refuse it hours later, after the whole array had run.
#
# STAGE=merge is chained with --dependency=afterany, NOT afterok, deliberately: a failed array
# task must surface as the merge REFUSING an incomplete shard set with a named missing range,
# not as a silently unsubmitted merge job. Partial coverage is never summarized.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
unset DEHYD_GIT_COMMIT DEHYD_GIT_BRANCH DEHYD_GIT_DIRTY 2>/dev/null || true

[ -f REVISION ] || {
  echo "ERROR: no REVISION file at repo root ($(pwd))." >&2
  echo "       Stamp one with 'git rev-parse HEAD > REVISION' on a machine that has git," >&2
  echo "       then copy it over WITH the rest of the tree, before submitting on IBEX." >&2
  exit 1
}

export EXPERIMENT=${EXPERIMENT:-a}
export BAND=${BAND:-10ghz}
export MODE=${MODE:-full}
export REPLICATES=${REPLICATES:-200}
export SHARD_SIZE=${SHARD_SIZE:-10}
export SHARD_DIR=${SHARD_DIR:-results/milestone10/robustness_shards/${EXPERIMENT}_${BAND}}
mkdir -p logs "$SHARD_DIR"

# Ceiling division, so the last shard covers the remainder rather than dropping it.
n_shards=$(( (REPLICATES + SHARD_SIZE - 1) / SHARD_SIZE ))
array_max=$(( n_shards - 1 ))

echo "robustness sharded: exp=${EXPERIMENT} band=${BAND} R=${REPLICATES} shard_size=${SHARD_SIZE}"
echo "  ${n_shards} shards (--array=0-${array_max}) -> ${SHARD_DIR}, commit $(cat REVISION)"

array_raw=$(sbatch --parsable --array=0-"${array_max}" \
    --cpus-per-task="${ARRAY_CPUS:-16}" --mem="${ARRAY_MEM:-64G}" \
    --time="${ARRAY_TIME:?set ARRAY_TIME from a measured single-shard run}" \
    --output=logs/robustness_array_%A_%a.out --error=logs/robustness_array_%A_%a.err \
    --export=ALL,STAGE=array,EXPERIMENT,BAND,MODE,REPLICATES,SHARD_SIZE,SHARD_DIR \
    scripts/ibex/run_robustness_sharded.sbatch)
array_job_id="${array_raw%%;*}"   # strip a possible ";<cluster>" suffix (C25)
echo "array job ${array_job_id} submitted"

# The merge re-runs the full cohort once for the point estimate, so it needs a real
# allocation — it is one ordinary Exp A/B/C run, not a bookkeeping step.
merge_raw=$(sbatch --parsable \
    --cpus-per-task="${MERGE_CPUS:-16}" --mem="${MERGE_MEM:-64G}" \
    --time="${MERGE_TIME:-04:00:00}" \
    --output=logs/robustness_merge_%j.out --error=logs/robustness_merge_%j.err \
    --dependency=afterany:"$array_job_id" \
    --export=ALL,STAGE=merge,EXPERIMENT,BAND,MODE,REPLICATES,SHARD_SIZE,SHARD_DIR \
    scripts/ibex/run_robustness_sharded.sbatch)
merge_job_id="${merge_raw%%;*}"

echo "done: shard_dir=${SHARD_DIR}  array_job=${array_job_id}  merge_job=${merge_job_id}"
