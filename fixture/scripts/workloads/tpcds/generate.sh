#!/usr/bin/env bash
# Generate TPC-DS Parquet datasets under fixture/data/tpcds/sf<N>/parquet/.
#
# Usage:
#   fixture/scripts/workloads/tpcds/generate.sh --scale sf10
#   fixture/scripts/workloads/tpcds/generate.sh --scale 10 --out /tmp/tpcds_sf10
#
# DuckDB's tpcds extension performs dsdgen in-process, then this script exports
# one Parquet file per official TPC-DS table. Existing output is left in place
# unless --force or FORCE=1 is set.

set -euo pipefail

SCALE_FACTOR="1"
OUT_DIR=""
FORCE="${FORCE:-0}"
DUCKDB_BIN="${DUCKDB:-}"

TPCDS_TABLES=(
  call_center catalog_page catalog_returns catalog_sales customer
  customer_address customer_demographics date_dim household_demographics
  income_band inventory item promotion reason ship_mode store store_returns
  store_sales time_dim warehouse web_page web_returns web_sales web_site
)

usage() {
  cat <<'USAGE'
Usage: fixture/scripts/workloads/tpcds/generate.sh [options]

Options:
  --scale N|sfN      TPC-DS scale factor (default: sf1)
  --out PATH         Output parquet directory
                     (default: fixture/data/tpcds/sf<N>/parquet)
  --duckdb PATH      DuckDB binary (default: $DUCKDB, ~/bin/duckdb, then PATH)
  --force            Remove an existing output directory before generating
  -h, --help         Show this help

Compatibility:
  A single positional scale factor is accepted.
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
    --duckdb)
      DUCKDB_BIN="${2:?missing value for --duckdb}"
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

if [[ -z "${DUCKDB_BIN}" ]]; then
  if [[ -x "${HOME}/bin/duckdb" ]]; then
    DUCKDB_BIN="${HOME}/bin/duckdb"
  elif command -v duckdb >/dev/null 2>&1; then
    DUCKDB_BIN="$(command -v duckdb)"
  else
    echo "error: duckdb binary not found; pass --duckdb or set DUCKDB" >&2
    exit 1
  fi
fi
if [[ ! -x "${DUCKDB_BIN}" ]]; then
  echo "error: duckdb binary not executable: ${DUCKDB_BIN}" >&2
  exit 1
fi

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${REPO_ROOT}/fixture/data/tpcds/sf${SCALE_FACTOR}/parquet"
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

SQL_FILE="${OUT_DIR}/generation.sql"
TMPDB="${OUT_DIR}/.tpcds_sf${SCALE_FACTOR}.duckdb"
rm -f "${TMPDB}"

{
  echo "INSTALL tpcds;"
  echo "LOAD tpcds;"
  echo "CALL dsdgen(sf=${SCALE_FACTOR});"
  for table in "${TPCDS_TABLES[@]}"; do
    echo "COPY ${table} TO '${OUT_DIR}/${table}.parquet' (FORMAT parquet);"
  done
} > "${SQL_FILE}"

echo "[tpcds-generate] generating TPC-DS sf${SCALE_FACTOR} parquet via DuckDB -> ${OUT_DIR}"
"${DUCKDB_BIN}" "${TMPDB}" < "${SQL_FILE}" >/dev/null
rm -f "${TMPDB}" "${TMPDB}.wal"

echo
echo "[tpcds-generate] done. Files:"
ls -lh "${OUT_DIR}"
