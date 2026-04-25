#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
echo "warning: generate_tpch.sh moved; use fixture/fixture.sh tpch generate ..." >&2
exec "${REPO_ROOT}/fixture/fixture.sh" tpch generate "$@"
