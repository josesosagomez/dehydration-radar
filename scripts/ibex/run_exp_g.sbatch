#!/bin/bash
#SBATCH --job-name=dehyd-exp-g
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/exp_g_%j.out
#SBATCH --error=logs/exp_g_%j.err

# Milestone 10 — Experiment G, the matched-session decision-level 10 GHz + 77 GHz fusion, as a
# single CPU batch job. E/F/G/H are entirely CPU; no GPU here. BOTH band stores must already be
# built (extract10.sbatch AND extract77.sbatch) at THIS commit — G fails closed on either.
#
# ///////////////////////////////////////////////////////////////////////////////////////
# SIZING — worked out BEFORE this header was written, because step 3's was not and it cost a
# redesign. Experiment G is the most expensive classical job in the milestone.
#
# G runs a COMPLETE Exp-A staged selection at three levels: once per (outer fold x meta fold x
# band) for the selection-honest meta OOF predictions, and once per (outer fold x band) for the
# outer-final winner. At 16 matched subjects that is
#
#     16 folds x 5 meta folds x 2 bands  +  16 folds x 2 bands  =  192 staged selections
#
# against a full two-band Experiment A's 32 — i.e. about 6x a complete Exp A per band.
#
# Measured anchor (HISTORY 2026-07-28): a full 16-fold Exp B run took 01:04:20 on 16 cores with
# all folds in one wave, i.e. ~1 core-hour per staged selection over ~15 training subjects and
# the 113-candidate 10 GHz space. G's meta levels select over ~12 subjects (T_s minus the
# meta-validation group) with the same candidate count and the same 5 further folds, so ~1
# core-hour per level is a conservative unit. 77 GHz enumerates fewer candidates (9 + 41 = 50 vs
# 113) but reconstructs much larger raw tensors on the tuned branch, so it is budgeted at the
# same unit rather than discounted.
#
#     per (outer fold, band) :  6 levels x ~1 core-hour   = ~6 core-hours
#     per outer fold         :  both bands                = ~12 core-hours
#     whole job              :  16 folds                  = ~192 core-hours
#
# The parallel unit is the (outer fold, BAND) pair, not the fold: the two bands' work is
# completely independent given the fold, and alpha is arithmetic over their saved prediction
# tables afterwards. That gives 2 x 16 = 32 independent units, so 32 cores runs them in ONE wave
# at ~6 h wall-clock. The header therefore allocates 32 x 24 h = 768 core-hours against ~192
# needed — 4x on cores-hours and ~4x on wall-clock.
#
# If the partition cannot give 32 cores, halve them rather than shortening the wall: at
# `--cpus-per-task=16` the 32 units run in two waves, ~12 h, still inside 24 h.
#   sbatch --cpus-per-task=16 --mem=64G ... run_exp_g.sbatch
# Memory is per worker — each holds one band's tuned-e reconstruction for one feature key at a
# time, the 77 GHz raw tensors being the big ones — so 32 workers want ~128 G.
# ///////////////////////////////////////////////////////////////////////////////////////
#
# MODE is passed via env at submit time:
#   MODE=full sbatch --export=ALL,MODE,DEHYD_GIT_COMMIT scripts/ibex/run_exp_g.sbatch
#
# MODE=smoke -> --subset 6subjects (mechanism-only; no performance value surfaced)
# MODE=full  -> --full-cohort
#
# The two band configs are passed SEPARATELY (--config-10 / --config-77) and are never merged
# through the config loader; configs/exp_g_fusion.yaml, configs/stats.yaml and configs/ibex.yaml
# are --shared-config overlays applied to both. run_fusion.py then refuses any pair that
# disagrees on run seeds, split constants, the fusion rule, the statistics or the weight
# workbook.

# RUN_DIR_OUT (optional): after a SUCCESSFUL run the driver atomically writes its absolute run
# directory there, which is how milestone-10 assembly builds its explicit run map without ever
# globbing for a latest directory. Unset -> nothing is written.

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
# Cosmetic: each worker re-pins to 1 thread (threadpool_limits) for determinism + no
# oversubscription, so this job-level value is not what governs the math threads.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

MODE=${MODE:-smoke}
if [ "$MODE" = "full" ]; then SEL=(--full-cohort); else SEL=(--subset 6subjects); fi

RUN_DIR_OUT=${RUN_DIR_OUT:-}
if [ -n "$RUN_DIR_OUT" ]; then OUT=(--run-dir-out "$RUN_DIR_OUT"); else OUT=(); fi

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

echo "Exp G fusion: mode=${MODE} commit=${DEHYD_GIT_COMMIT:-<unset>}"
"${PYTHON[@]}" -u experiments/run_fusion.py \
  --config-10 configs/exp_a_regression.yaml \
  --config-77 configs/exp_a_regression_77ghz.yaml \
  --shared-config configs/exp_g_fusion.yaml \
  --shared-config configs/stats.yaml \
  --shared-config configs/ibex.yaml \
  "${SEL[@]}" "${OUT[@]}"
