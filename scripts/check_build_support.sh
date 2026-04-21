#!/usr/bin/env bash
# Compile and run the pure tests colocated with the workspace-local build
# support. The helper is intentionally not a crate, so rustc is the canonical
# test harness and this script is wired into scripts/check_all.sh.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-build-support.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

rustc --edition=2024 --test build-support/rapids.rs -o "$tmp_dir/rapids-tests"
"$tmp_dir/rapids-tests"
