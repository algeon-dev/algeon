#!/usr/bin/env bash
# Compatibility entrypoint for the old TPC-H fixture generator path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/../../gen_tpch.py" --variant raw --no-verify "$@"
