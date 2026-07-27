#!/bin/bash
# Submit the Exp B session-specific secondary variant's three stages (init -> array -> merge),
# capturing the git revision at SUBMIT time (compute nodes cannot answer git — safe.directory
# does not take there) and exporting it into every job's environment, where
# provenance._git_info falls back to it.
#
# REFUSES to submit from a dirty tree: a run must be attributable to a clean revision
# (C7/C16, mirroring submit_extract77.sh/submit_ibex.sh's precedent).
#
# STAGE=init runs on its own sized allocation (heavy raw-file hashing — genuine I/O, never a
# login-node step, C23) and this script BLOCKS on it (`sbatch --wait`) before ever submitting
# the array — a failed init means the array is never submitted (set -e + the exit-code check
# below). STAGE=array is the REAL cross-session concurrency: 4 tasks, one per session, each
# with its own full 16-core/64G allocation (sized from the primary run's measured wall-time,
# ARRAY_TIME — never guessed). STAGE=merge runs once, gated on the array via
# --dependency=afterany so it runs whether every task succeeded or not (partial completion is
# visible in the merged report's completed_sessions, not swallowed here).
#
# Usage:
#   ARRAY_TIME=01:30:00 scripts/ibex/submit_exp_b_variant.sh              # 10 GHz, default INIT_TIME
#   BAND=77ghz INIT_TIME=02:00:00 ARRAY_TIME=02:00:00 scripts/ibex/submit_exp_b_variant.sh
#
# ARRAY_TIME is REQUIRED (no safe default — size it from step 10's measured primary-run
# wall-time first). INIT_TIME defaults to 01:00:00, revise after one measured run.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: working tree is DIRTY — refusing to submit." >&2
  echo "       A run must be attributable to a clean commit (C7/C16). Commit first." >&2
  exit 1
fi

export DEHYD_GIT_COMMIT="$(git rev-parse HEAD)"
export DEHYD_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
export DEHYD_GIT_DIRTY="false"
export BAND=${BAND:-10ghz}
mkdir -p logs

echo "submitting exp_b_variant STAGE=init at commit ${DEHYD_GIT_COMMIT} (${DEHYD_GIT_BRANCH}), band=${BAND}"

# --wait blocks here until the init stage finishes; a nonzero sbatch/job exit makes this
# script exit nonzero too (set -e), so the array is NEVER submitted after a failed init (C23).
init_raw=$(sbatch --wait --parsable --cpus-per-task=1 --mem=8G --time="${INIT_TIME:-01:00:00}" \
    --output=logs/exp_b_variant_init_%j.out --error=logs/exp_b_variant_init_%j.err \
    --export=ALL,STAGE=init,BAND,DEHYD_GIT_COMMIT,DEHYD_GIT_BRANCH,DEHYD_GIT_DIRTY \
    scripts/ibex/run_exp_b_variant.sbatch)
init_job_id="${init_raw%%;*}"   # strip a possible ";<cluster>" suffix (C25) -- %j in the
                                 # --output pattern above expands to this SAME numeric ID,
                                 # so the two are guaranteed to agree; never parsed apart.

run_dir=$(tail -n1 "logs/exp_b_variant_init_${init_job_id}.out")
[ -d "$run_dir" ] || { echo "ERROR: init did not produce a valid run_dir: '${run_dir}'" >&2; exit 1; }
echo "init OK, run_dir=${run_dir}"

echo "submitting exp_b_variant STAGE=array (--array=1-4), band=${BAND}, time=${ARRAY_TIME:?set ARRAY_TIME from the step-10-measured primary-run wall-time}"
array_raw=$(sbatch --parsable --array=1-4 --cpus-per-task=16 --mem=64G --time="${ARRAY_TIME}" \
    --output=logs/exp_b_variant_array_%A_%a.out --error=logs/exp_b_variant_array_%A_%a.err \
    --export=ALL,STAGE=array,BAND,RUN_DIR="$run_dir",DEHYD_GIT_COMMIT,DEHYD_GIT_BRANCH,DEHYD_GIT_DIRTY \
    scripts/ibex/run_exp_b_variant.sbatch)
array_job_id="${array_raw%%;*}"   # same normalization (C25)
echo "array job ${array_job_id} submitted"

echo "submitting exp_b_variant STAGE=merge, dependency=afterany:${array_job_id}"
merge_raw=$(sbatch --parsable --cpus-per-task=1 --mem=4G --time=00:15:00 \
    --output=logs/exp_b_variant_merge_%j.out --error=logs/exp_b_variant_merge_%j.err \
    --dependency=afterany:"$array_job_id" \
    --export=ALL,STAGE=merge,BAND,RUN_DIR="$run_dir",DEHYD_GIT_COMMIT,DEHYD_GIT_BRANCH,DEHYD_GIT_DIRTY \
    scripts/ibex/run_exp_b_variant.sbatch)
merge_job_id="${merge_raw%%;*}"

echo "done: run_dir=${run_dir}  array_job=${array_job_id}  merge_job=${merge_job_id}"
