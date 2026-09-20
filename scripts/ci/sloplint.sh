#!/usr/bin/env bash
# Run the vendored sloplint scanner from soundshuman.
set -euo pipefail

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: node is required to run sloplint (install Node 18+)" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
exec node "${SCRIPT_DIR}/sloplint/bin/sloplint.js" "$@"
