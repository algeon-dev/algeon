#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
echo "warning: fetch_clickbench.sh moved; use fixture/fixture.sh clickbench fetch ..." >&2
exec "${REPO_ROOT}/fixture/fixture.sh" clickbench fetch "$@"
