#!/bin/bash
#SBATCH --job-name=dehyd-assembly
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/assembly_%j.out
#SBATCH --error=logs/assembly_%j.err

# Milestone 10 — final assembly. Reads the explicit run map, re-hashes every registered
# artifact, and writes the H assembly tables.
#
# This job FITS NOTHING. It reads saved artifacts, checks their SHA-256 against the manifest,
# and writes CSV/JSON — hence the smallest header in the repository. Two cores and half an
# hour is already generous for hashing a few dozen files.
#
# Called TWICE, in this order (plan §6):
#   VALIDATE_ONLY=1 -> schema, SHA-256, config/store/commit and source-lineage checks only;
#                      writes NO numerical summary. Its whole job is to fail before publishing.
#   VALIDATE_ONLY=0 -> the same manifest, this time writing every §3 assembly artifact plus
#                      metrics_milestone10.json.
#
#   RUN_MANIFEST=results/milestone10/run_manifest.json VALIDATE_ONLY=1 \
#     sbatch --export=ALL,RUN_MANIFEST,VALIDATE_ONLY,DEHYD_GIT_COMMIT,... run_stats_assembly.sbatch
#
# RUN_MANIFEST must already exist: it is built by an explicit run of the driver with
# --run/--run-from-file (the pointer files each experiment wrapper wrote via RUN_DIR_OUT).
# Nothing here discovers a run directory, which is the point.

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

RUN_MANIFEST=${RUN_MANIFEST:?RUN_MANIFEST must name the explicit run map -- assembly never discovers runs}
VALIDATE_ONLY=${VALIDATE_ONLY:-1}
OUT_DIR=${OUT_DIR:-results/milestone10}
if [ "$VALIDATE_ONLY" = "1" ]; then MODE=(--validate-only); else MODE=(--out "$OUT_DIR"); fi

if [ -x .venv/bin/python ]; then PYTHON=(.venv/bin/python); else PYTHON=(uv run --no-sync python); fi

echo "Assembly: manifest=${RUN_MANIFEST} validate_only=${VALIDATE_ONLY} commit=${DEHYD_GIT_COMMIT:-<unset>}"
"${PYTHON[@]}" -u experiments/run_stats_assembly.py \
  --run-manifest "$RUN_MANIFEST" "${MODE[@]}"
