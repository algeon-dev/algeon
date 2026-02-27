#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "warning: extract_tpcds_queries.sh moved to scripts/queries/extract_tpcds_queries.sh" >&2
exec "${SCRIPT_DIR}/queries/extract_tpcds_queries.sh" "$@"
