#!/usr/bin/env bash
# Policy check: scan workspace documentation using vendored sloplint.
# Fails closed if any document scores at or above the failure threshold (40/100).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${ROOT}"

readonly SLOPLINT="${ROOT}/scripts/ci/sloplint.sh"
readonly THRESHOLD=40
readonly TARGET_DIRS=(docs note website/docs)

status=0
for target in "${TARGET_DIRS[@]}"; do
  if [[ ! -d "${target}" ]]; then
    continue
  fi
  if ! "${SLOPLINT}" scan "${target}" --fail-above "${THRESHOLD}"; then
    echo "ERROR: sloplint found documents exceeding threshold ${THRESHOLD} in ${target}" >&2
    status=1
  fi
done

if ((status == 0)); then
  echo "sloplint scan passed: all documents score below ${THRESHOLD}/100."
fi
exit "${status}"
