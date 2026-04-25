#!/usr/bin/env bash
# Compatibility wrapper for the local Iceberg REST + RustFS loader.
#
# Preferred entrypoint:
#   fixture/fixture.sh iceberg rest load --workload tpch --scale sf1
#
# Legacy usage remains supported:
#   fixture/iceberg-local/load.sh tpch 1
#   WORKLOAD=tpcds SF=10 fixture/iceberg-local/load.sh

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/fixture/fixture.sh" \
    iceberg rest load --help
fi

WORKLOAD="${1:-${WORKLOAD:-tpch}}"
SF="${2:-${SF:-1}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

echo "warning: fixture/iceberg-local/load.sh is a wrapper; use fixture/fixture.sh iceberg rest load ..." >&2
exec "${REPO_ROOT}/fixture/fixture.sh" \
  iceberg rest load --workload "${WORKLOAD}" --scale "${SF}" "${@:3}"
