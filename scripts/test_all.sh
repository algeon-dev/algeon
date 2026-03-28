#!/usr/bin/env bash
# Friendly entrypoint for running the full local test/check surface.
#
# Keep the implementation in check_all.sh so both command names stay in sync.
set -euo pipefail

export GPU_PARALLEL="${GPU_PARALLEL:-1}"
export GPU_TEST_THREADS="${GPU_TEST_THREADS:-4}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/check_all.sh" "$@"
