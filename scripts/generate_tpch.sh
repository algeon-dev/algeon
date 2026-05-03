#!/usr/bin/env bash
# Generate TPC-H Parquet datasets under tests/data/tpch/sf<N>/parquet/.
#
# Usage:
#   scripts/generate_tpch.sh [SCALE_FACTOR]
#
# SCALE_FACTOR defaults to 1. Output is snappy-compressed Parquet, one file
# per table. Existing sf<N>/ directory is left in place unless --force is set.

set -euo pipefail

SCALE_FACTOR="${1:-1}"
FORCE="${FORCE:-0}"

if ! command -v tpchgen-cli >/dev/null 2>&1; then
  echo "error: tpchgen-cli not on PATH; run 'cargo install tpchgen-cli'" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/tests/data/tpch/sf${SCALE_FACTOR}/parquet"

if [[ -d "${OUT_DIR}" && "${FORCE}" != "1" ]]; then
  if compgen -G "${OUT_DIR}/*.parquet" >/dev/null; then
    echo "error: ${OUT_DIR} already contains .parquet files; set FORCE=1 to overwrite" >&2
    exit 1
  fi
fi

mkdir -p "${OUT_DIR}"

echo "Generating TPC-H SF${SCALE_FACTOR} (snappy parquet) -> ${OUT_DIR}"
tpchgen-cli \
  --scale-factor "${SCALE_FACTOR}" \
  --output-dir "${OUT_DIR}" \
  --format parquet \
  --parquet-compression SNAPPY \
  --verbose

echo
echo "Done. Files:"
ls -lh "${OUT_DIR}"
