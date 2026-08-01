#!/bin/bash
#SBATCH --job-name=dehyd-exp-d-cnn

# Milestone 9 — Experiment D's CNN fold array (Step 0 item 3), dispatched by STAGE
# (init|fold|merge) at the SHELL level, mirroring run_exp_b_variant.sbatch rather than three
# near-duplicate files.
#
# DELIBERATE: this header carries NO resource directives (no --cpus-per-task/--mem/--time/
# --gres/--array/--output) beyond --job-name. #SBATCH lines are parsed by sbatch directly from
# the script TEXT at submission time, BEFORE the shell — and therefore STAGE — is ever
# evaluated, so a single file's header cannot vary per stage no matter what the shell body
# does afterward. One fixed set of numbers here would silently apply to ALL THREE stages,
# which differ by more than a factor of ten and by whether a GPU is requested at all. Every
# resource-affecting flag (including --gres and --output) is supplied EXCLUSIVELY via the
# sbatch CLI, once per stage, by submit_exp_d_cnn.sh — the single source of truth for sizing,
# with no competing in-file default to fall back to (C24).
#
# STAGE=init   a SINGLE-TASK CPU job: validates the store, builds the family's frame spine
#              and writes the shared run group (provenance + the authoritative per-fold row
#              census). Prints run_dir as the LAST line of its stdout log — the
#              machine-readable handoff submit_exp_d_cnn.sh reads.
# STAGE=fold   one GPU task per outer fold. SLURM_ARRAY_TASK_ID runs 1..16 while fold ids are
#              POSITIONS in the selectable-fold list (0-based), so the offset is applied here,
#              in one place. A task whose fold index is beyond that list exits 0 having
#              written a named no-op marker (§5 trap 14), never an error.
# STAGE=merge  a single tiny CPU task combining whichever fold shards exist under "$RUN_DIR",
#              fail-closed on lineage and on the per-fold row census.
# STAGE=smoke  the step-10 mechanism-only CPU smoke: --subset 6subjects in ONE process, with
#              the configs/smoke.yaml seed overlay. It exists because step 10 requires a CPU
#              smoke for every Exp D family in BOTH bands and the 77 GHz store lives only
#              here, so that half cannot be smoked locally at all. It takes NO run dir: a
#              smoke must never write mechanism-only artifacts into a real run group, where
#              the merge would find them among the shards.
#
# Never run this file directly with plain `sbatch` — use scripts/ibex/submit_exp_d_cnn.sh.

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

BAND=${BAND:-10ghz}
FAMILY=${FAMILY:?FAMILY must be one of cnn1d_raw|cnn1d_matched|spec2d_raw|spec2d_matched}
CFG=configs/exp_a_regression.yaml
[ "$BAND" = "77ghz" ] && CFG=configs/exp_a_regression_77ghz.yaml
CONFIG_ARGS=(--config "$CFG" --config configs/baselines.yaml --config configs/ibex.yaml)
# The GPU overlay is loaded by ALL THREE run-group stages, not just the one that touches a
# GPU. `run.device` is hashed into `exp_b.config_fingerprint`, so a fold task that loaded it
# while `--init-run-group` did not would abort in `_validate_group_lineage` on config_hash
# after init had already succeeded. init and merge never build a tensor, so naming cuda there
# costs nothing. STAGE=smoke is deliberately excluded: it joins no run group, and step 10
# specifies the mechanism smokes on CPU.
GROUP_ARGS=("${CONFIG_ARGS[@]}" --config configs/gpu.yaml)

# init/fold/merge are submitted by submit_exp_d_cnn.sh. STAGE=smoke is submitted directly,
# since it is one standalone CPU job with no run group and no array:
#   STAGE=smoke BAND=77ghz FAMILY=spec2d_raw sbatch \
#     --cpus-per-task=16 --mem=64G --time=04:00:00 \
#     --output=logs/exp_d_smoke_%j.out --error=logs/exp_d_smoke_%j.err \
#     --export=ALL,STAGE,BAND,FAMILY scripts/ibex/run_exp_d_cnn.sbatch
STAGE=${STAGE:?STAGE must be one of init|fold|merge|smoke}

case "$STAGE" in
  init)
    echo "Exp D CNN STAGE=init: band=${BAND} family=${FAMILY} commit=${DEHYD_GIT_COMMIT:-<unset>}"
    "${PYTHON[@]}" -u experiments/run_baselines.py "${GROUP_ARGS[@]}" \
      --band "$BAND" --family "$FAMILY" --init-run-group
    ;;
  fold)
    RUN_DIR=${RUN_DIR:?RUN_DIR must be set (the --init-run-group-printed directory)}
    FOLD=$((SLURM_ARRAY_TASK_ID - 1))   # array task 1..16 -> fold position 0..15
    echo "Exp D CNN STAGE=fold: band=${BAND} family=${FAMILY} fold=${FOLD} run_dir=${RUN_DIR}"
    "${PYTHON[@]}" -u experiments/run_baselines.py "${GROUP_ARGS[@]}" \
      --band "$BAND" --family "$FAMILY" --fold "$FOLD" --run-dir "$RUN_DIR"
    ;;
  merge)
    RUN_DIR=${RUN_DIR:?RUN_DIR must be set (the --init-run-group-printed directory)}
    echo "Exp D CNN STAGE=merge: band=${BAND} family=${FAMILY} run_dir=${RUN_DIR}"
    "${PYTHON[@]}" -u experiments/run_baselines.py "${GROUP_ARGS[@]}" \
      --band "$BAND" --family "$FAMILY" --merge-folds --run-dir "$RUN_DIR"
    ;;
  smoke)
    echo "Exp D CNN STAGE=smoke: band=${BAND} family=${FAMILY} commit=${DEHYD_GIT_COMMIT:-<unset>}"
    "${PYTHON[@]}" -u experiments/run_baselines.py "${CONFIG_ARGS[@]}" \
      --config configs/smoke.yaml --band "$BAND" --family "$FAMILY" --subset 6subjects
    ;;
  *)
    echo "ERROR: unknown STAGE '${STAGE}' (expected init|fold|merge|smoke)" >&2
    exit 1
    ;;
esac
