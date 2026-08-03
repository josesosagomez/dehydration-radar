#!/bin/bash
# Submit the 77 GHz feature-store job array, capturing the git revision at SUBMIT time.
#
# The compute nodes cannot answer git (safe.directory does not take there), so provenance
# would record commit=None. This wrapper captures HEAD/branch/dirty here on the LOGIN node
# and exports them as DEHYD_GIT_COMMIT/_BRANCH/_DIRTY into the job environment, where
# provenance._git_info falls back to them.
#
# It REFUSES to submit from a dirty tree: a store must be attributable to a clean revision
# (C7/C16). Commit the milestone-7 code first.
#
#   scripts/ibex/submit_extract77.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: working tree is DIRTY — refusing to submit a store build." >&2
  echo "       A feature store must be attributable to a clean commit (C7/C16). Commit first." >&2
  exit 1
fi

export DEHYD_GIT_COMMIT="$(git rev-parse HEAD)"
export DEHYD_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
export DEHYD_GIT_DIRTY="false"

echo "submitting extract77 at commit ${DEHYD_GIT_COMMIT} (${DEHYD_GIT_BRANCH})"
sbatch --export=ALL,DEHYD_GIT_COMMIT,DEHYD_GIT_BRANCH,DEHYD_GIT_DIRTY scripts/ibex/extract77.sbatch
