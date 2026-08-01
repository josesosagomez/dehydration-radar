#!/bin/bash
# Submit ONE Exp D CNN family x band as its three stages (init -> 16-task GPU array -> merge).
# COPIED-TREE workflow: this assumes the repo was copied to IBEX (not cloned), so there is no
# .git here at all -- this script never calls git. Provenance instead comes from a REVISION
# file at the repo root (`git rev-parse HEAD > REVISION`, stamped on the machine that HAS git,
# then copied over WITH the tree). `provenance.py` reads REVISION whenever live git and the
# DEHYD_GIT_* env vars are both absent, so every stage still self-attests its revision.
# DEHYD_GIT_* is deliberately left unset (and explicitly unset below, in case a stale value
# leaked in from the calling shell) so that fallback always applies. This is the M8 step-10.5
# lesson (commit e88fd33) applied from day one, not retrofitted after a failure.
#
# Locates the repo root from this script's OWN path (not git), so it can be launched anywhere:
#   FAMILY=cnn1d_raw BAND=10ghz ARRAY_TIME=02:00:00 bash scripts/ibex/submit_exp_d_cnn.sh
#
# ARRAY_TIME is REQUIRED and defaults only to the pre-measurement placeholder 08:00:00 -- size
# it from the step-10 single-fold GPU smoke's measured wall-time, then record the measurement
# (the C8 lesson: arrays are sized from measurement, never from a guess). INIT_TIME defaults to
# 01:00:00 (record_run's raw-file hashing plus one full spine build is genuine I/O work, never
# a login-node step, C23).
#
# ARRAY_SPEC is that measurement's own knob: it defaults to the full 1-16, and step 10's
# single-fold GPU smoke sets it to 1-1 so ONE real fold runs on a real GPU and its wall-time
# becomes ARRAY_TIME for step 13. The default stays 1-16 deliberately -- a forgotten
# ARRAY_SPEC must over-run, never silently under-run the cohort.
#   FAMILY=cnn1d_raw BAND=10ghz ARRAY_SPEC=1-1 ARRAY_TIME=02:00:00 bash scripts/ibex/submit_exp_d_cnn.sh
# The merge stage still runs after a 1-task array; it reports the partial `completed_folds`
# as a NAMED non-reportable state (trap 14), which is the correct outcome for a smoke.
#
# STAGE=init runs on its own sized allocation and this script BLOCKS on it (`sbatch --wait`)
# before ever submitting the array -- a failed init means the array is never submitted (set -e
# plus the exit-code check below). STAGE=fold is the REAL cross-fold concurrency: 16 tasks, one
# GPU each. STAGE=merge runs once, gated on the array via --dependency=afterany so it runs
# whether every task succeeded or not (partial completion is a NAMED non-reportable state in
# the merged report, never swallowed here).

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
unset DEHYD_GIT_COMMIT DEHYD_GIT_BRANCH DEHYD_GIT_DIRTY 2>/dev/null || true

[ -f REVISION ] || {
  echo "ERROR: no REVISION file at repo root ($(pwd))." >&2
  echo "       Stamp one with 'git rev-parse HEAD > REVISION' on a machine that has git," >&2
  echo "       then copy it over WITH the rest of the tree, before submitting on IBEX." >&2
  exit 1
}

export BAND=${BAND:-10ghz}
export FAMILY=${FAMILY:?set FAMILY to one of cnn1d_raw cnn1d_matched spec2d_raw spec2d_matched}
mkdir -p logs
tag="${FAMILY}_${BAND}"

echo "submitting exp_d_cnn STAGE=init at commit $(cat REVISION), family=${FAMILY}, band=${BAND}"

# --wait blocks here until init finishes; a nonzero sbatch/job exit makes this script exit
# nonzero too (set -e), so the array is NEVER submitted after a failed init (C23).
init_raw=$(sbatch --wait --parsable --cpus-per-task=4 --mem=32G --time="${INIT_TIME:-01:00:00}" \
    --output="logs/exp_d_cnn_init_${tag}_%j.out" --error="logs/exp_d_cnn_init_${tag}_%j.err" \
    --export=ALL,STAGE=init,BAND,FAMILY \
    scripts/ibex/run_exp_d_cnn.sbatch)
init_job_id="${init_raw%%;*}"   # strip a possible ";<cluster>" suffix (C25) -- %j in the
                                 # --output pattern above expands to this SAME numeric ID,
                                 # so the two are guaranteed to agree; never parsed apart.

# the run dir is the LAST line of init's stdout; `tail -n1` is robust to any preflight noise
# (config echo, worker count, store-validation chatter) printed before it.
run_dir=$(tail -n1 "logs/exp_d_cnn_init_${tag}_${init_job_id}.out")
[ -d "$run_dir" ] || { echo "ERROR: init did not produce a valid run_dir: '${run_dir}'" >&2; exit 1; }
echo "init OK, run_dir=${run_dir}"

echo "submitting exp_d_cnn STAGE=fold (--array=${ARRAY_SPEC:-1-16}), time=${ARRAY_TIME:-08:00:00}"
array_raw=$(sbatch --parsable --array="${ARRAY_SPEC:-1-16}" --gres=gpu:1 --cpus-per-task=4 --mem=32G \
    --time="${ARRAY_TIME:-08:00:00}" \
    --output="logs/exp_d_cnn_fold_${tag}_%A_%a.out" --error="logs/exp_d_cnn_fold_${tag}_%A_%a.err" \
    --export=ALL,STAGE=fold,BAND,FAMILY,RUN_DIR="$run_dir" \
    scripts/ibex/run_exp_d_cnn.sbatch)
array_job_id="${array_raw%%;*}"   # same normalization (C25)
echo "array job ${array_job_id} submitted"

echo "submitting exp_d_cnn STAGE=merge, dependency=afterany:${array_job_id}"
merge_raw=$(sbatch --parsable --cpus-per-task=1 --mem=8G --time=00:20:00 \
    --output="logs/exp_d_cnn_merge_${tag}_%j.out" --error="logs/exp_d_cnn_merge_${tag}_%j.err" \
    --dependency=afterany:"$array_job_id" \
    --export=ALL,STAGE=merge,BAND,FAMILY,RUN_DIR="$run_dir" \
    scripts/ibex/run_exp_d_cnn.sbatch)
merge_job_id="${merge_raw%%;*}"

echo "done: run_dir=${run_dir}  array_job=${array_job_id}  merge_job=${merge_job_id}"
