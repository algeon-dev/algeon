#!/usr/bin/env bash
# Pure argument-preserving alias for check_all.sh.
#
# Keep the implementation in the canonical scripts/check_all.sh entry point.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scripts_root="$(cd -- "${script_dir}/.." && pwd)"
exec "${scripts_root}/check_all.sh" "$@"
