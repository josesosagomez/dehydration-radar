#!/bin/bash
# Generic IBEX submit wrapper: capture the git revision at SUBMIT time (compute nodes can't
# answer git — safe.directory does not take there) and export it into the job so
# provenance._git_info falls back to it. REFUSES a dirty tree — a store/result must be
# attributable to a clean commit (C7/C16).
#
#   scripts/ibex/submit_ibex.sh scripts/ibex/extract10.sbatch
#   BAND=10ghz MODE=smoke scripts/ibex/submit_ibex.sh scripts/ibex/run_exp_a.sbatch
#
# Any leading VAR=value env you set (e.g. BAND, MODE) is forwarded via --export=ALL.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SCRIPT=${1:?usage: submit_ibex.sh <sbatch-script>}

if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: working tree is DIRTY — refusing to submit." >&2
  echo "       A store/result must be attributable to a clean commit (C7/C16). Commit first." >&2
  exit 1
fi

export DEHYD_GIT_COMMIT="$(git rev-parse HEAD)"
export DEHYD_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
export DEHYD_GIT_DIRTY="false"

echo "submitting ${SCRIPT} at commit ${DEHYD_GIT_COMMIT} (${DEHYD_GIT_BRANCH})"
sbatch --export=ALL "${SCRIPT}"
