#!/usr/bin/env bash
# Generate TPC-H Parquet datasets under fixture/data/tpch/sf<N>/parquet/.
#
# Usage:
#   fixture/scripts/workloads/tpch/generate.sh --scale sf10
#   fixture/scripts/workloads/tpch/generate.sh --scale 10 --out /tmp/tpch_sf10
#
# Output is snappy-compressed Parquet, one file per table. Existing output is
# left in place unless --force or FORCE=1 is set.

set -euo pipefail

SCALE_FACTOR="1"
OUT_DIR=""
FORCE="${FORCE:-0}"

usage() {
  cat <<'USAGE'
Usage: fixture/scripts/workloads/tpch/generate.sh [options]

Options:
  --scale N|sfN      TPC-H scale factor (default: sf1)
  --out PATH         Output parquet directory
                     (default: fixture/data/tpch/sf<N>/parquet)
  --force            Remove an existing output directory before generating
  -h, --help         Show this help

Compatibility:
  A single positional scale factor is accepted for the old generate_tpch.sh API.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale|-s)
      SCALE_FACTOR="${2:?missing value for $1}"
      shift 2
      ;;
    --out|--output-dir)
      OUT_DIR="${2:?missing value for $1}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      SCALE_FACTOR="$1"
      shift
      ;;
  esac
done

SCALE_FACTOR="${SCALE_FACTOR#sf}"
SCALE_FACTOR="${SCALE_FACTOR#SF}"
if ! [[ "${SCALE_FACTOR}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "error: invalid --scale=${SCALE_FACTOR}; expected N or sfN" >&2
  exit 1
fi

if ! command -v tpchgen-cli >/dev/null 2>&1; then
  echo "error: tpchgen-cli not on PATH; run 'cargo install tpchgen-cli'" >&2
  exit 1
fi

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${REPO_ROOT}/fixture/data/tpch/sf${SCALE_FACTOR}/parquet"
fi

if [[ -e "${OUT_DIR}" && "${FORCE}" == "1" ]]; then
  rm -rf "${OUT_DIR}"
elif [[ -d "${OUT_DIR}" ]]; then
  if compgen -G "${OUT_DIR}/*.parquet" >/dev/null; then
    echo "error: ${OUT_DIR} already contains .parquet files; pass --force or set FORCE=1 to overwrite" >&2
    exit 1
  fi
fi

mkdir -p "${OUT_DIR}"

echo "[tpch-generate] generating TPC-H sf${SCALE_FACTOR} (snappy parquet) -> ${OUT_DIR}"
tpchgen-cli \
  --scale-factor "${SCALE_FACTOR}" \
  --output-dir "${OUT_DIR}" \
  --format parquet \
  --parquet-compression SNAPPY \
  --verbose

echo
echo "[tpch-generate] done. Files:"
ls -lh "${OUT_DIR}"
