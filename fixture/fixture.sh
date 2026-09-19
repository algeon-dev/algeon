#!/usr/bin/env bash
# Unified public fixture entrypoint for benchmark data, demo datasets, and local Iceberg E2E.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

usage() {
  cat <<'USAGE'
Usage:
  fixture/fixture.sh tpch generate [--scale N...] [--out-root PATH] [--variant raw|optimized|both] [--force]
  fixture/fixture.sh tpcds generate [--scale N...] [--out-root PATH] [--variant raw|optimized|both] [--force]
  fixture/fixture.sh clickbench fetch [--layout single|partitioned|both] [--out PATH]
  fixture/fixture.sh graph dblp-ingest [--source-json PATH] [--out-root PATH] [--overwrite] [DBLP ingest options]
  fixture/fixture.sh graph rmat --scale N --edge-factor K --skew skewed|uniform --seed S --out-root PATH [--force]
  fixture/fixture.sh ember2024 prepare [--split test|train|challenge]... [--rows N] [--force]
  fixture/fixture.sh amazon-reviews prepare [--category NAME] [--out-dir PATH] [--force]
  fixture/fixture.sh kuairand prepare [--variant pure|1k|27k] [--out-dir PATH] [--force] [--download]
  fixture/fixture.sh merrec prepare [--partition NAME] [--input-parquet PATH] [--out-dir PATH] [--force] [--download-sample]
  fixture/fixture.sh iceberg rest load --workload tpch|tpcds|citation_network [--scale sf1] [--load-mode create|add-files]
  fixture/fixture.sh iceberg glue-moto load --workload tpch|tpcds [--scale sf1]

TPC-H / TPC-DS generation uses tpchgen-rs and rewrites the final Parquet layout
with sorted, Snappy-compressed row groups by default. Run a gen_*.py with --help
for the full flag set.
USAGE
}

clickbench_usage() {
  cat <<'USAGE'
Usage: fixture/fixture.sh clickbench fetch [options]

Options:
  --layout single|partitioned|both Dataset layout to download (default: single)
  --mode single|partitioned|both   Backward-compatible alias for --layout
  --out PATH                       Destination root (default: fixture/clickbench/data)
  --output-dir PATH                Backward-compatible alias for --out
  --content-fingerprint            Emit sha256 over file contents after download
  --row-count                      Emit row count using duckdb if available
  --jobs N                         Parallel downloads for partitioned mode (default: 8)
  --dry-run                        Print planned downloads without fetching
  -h, --help                       Show this help
USAGE
}

clickbench_download_one() {
  local url="$1"
  local dest="$2"
  mkdir -p "$(dirname "${dest}")"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'would_download\t%s\t%s\n' "${url}" "${dest}"
    return 0
  fi

  if command -v wget >/dev/null 2>&1; then
    wget --continue --progress=dot:giga --output-document="${dest}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl --fail --location --continue-at - --output "${dest}" "${url}"
  else
    echo "error: wget or curl is required" >&2
    exit 1
  fi
}

clickbench_download_single() {
  clickbench_download_one "${SINGLE_URL}" "${OUTPUT_DIR}/hits.parquet"
}

clickbench_download_partitioned() {
  mkdir -p "${OUTPUT_DIR}/partitioned"
  for i in $(seq 0 99); do
    if [[ "${DRY_RUN}" == "1" ]]; then
      clickbench_download_one "${PARTITIONED_URL_PREFIX}/hits_${i}.parquet" \
        "${OUTPUT_DIR}/partitioned/hits_${i}.parquet"
      continue
    fi
    clickbench_download_one "${PARTITIONED_URL_PREFIX}/hits_${i}.parquet" \
      "${OUTPUT_DIR}/partitioned/hits_${i}.parquet" &
    while [[ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${JOBS}" ]]; do
      wait -n
    done
  done
  wait
}

clickbench_collect_files() {
  case "${MODE}" in
    single)
      find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'hits.parquet' | sort
      ;;
    partitioned)
      find "${OUTPUT_DIR}/partitioned" -maxdepth 1 -type f -name 'hits_*.parquet' | sort
      ;;
    both)
      {
        find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'hits.parquet'
        find "${OUTPUT_DIR}/partitioned" -maxdepth 1 -type f -name 'hits_*.parquet'
      } | sort
      ;;
  esac
}

clickbench_emit_row_count() {
  if ! command -v duckdb >/dev/null 2>&1; then
    printf 'row_count=unknown\n'
    printf 'row_count_reason=duckdb_not_found\n'
    return 0
  fi

  local sql
  case "${MODE}" in
    single)
      sql="SELECT count(*) FROM read_parquet('${OUTPUT_DIR}/hits.parquet');"
      ;;
    partitioned)
      sql="SELECT count(*) FROM read_parquet('${OUTPUT_DIR}/partitioned/hits_*.parquet');"
      ;;
    both)
      sql="SELECT count(*) FROM read_parquet('${OUTPUT_DIR}/hits.parquet');"
      ;;
  esac

  local count
  count="$(duckdb -noheader -csv -c "${sql}" | tr -d '\r')"
  printf 'row_count=%s\n' "${count}"
}

clickbench_emit_summary() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  local files_file
  files_file="$(mktemp)"
  clickbench_collect_files > "${files_file}"

  local file_count total_bytes metadata_fingerprint
  file_count="$(wc -l < "${files_file}" | tr -d ' ')"
  total_bytes="$(
    while IFS= read -r path; do
      stat -c '%s' "${path}"
    done < "${files_file}" | awk '{ total += $1 } END { print total + 0 }'
  )"
  metadata_fingerprint="$(
    while IFS= read -r path; do
      rel="${path#${OUTPUT_DIR}/}"
      printf '%s\t%s\n' "${rel}" "$(stat -c '%s' "${path}")"
    done < "${files_file}" | sha256sum | awk '{ print $1 }'
  )"

  printf 'clickbench_mode=%s\n' "${MODE}"
  printf 'clickbench_root=%s\n' "${OUTPUT_DIR}"
  printf 'file_count=%s\n' "${file_count}"
  printf 'total_bytes=%s\n' "${total_bytes}"
  printf 'expected_official_row_count=%s\n' "${EXPECTED_ROW_COUNT}"
  printf 'metadata_fingerprint=%s\n' "${metadata_fingerprint}"

  if [[ "${CONTENT_FINGERPRINT}" == "1" ]]; then
    local content_fingerprint
    content_fingerprint="$(
      while IFS= read -r path; do
        sha256sum "${path}"
      done < "${files_file}" | sha256sum | awk '{ print $1 }'
    )"
    printf 'content_fingerprint=%s\n' "${content_fingerprint}"
  fi

  rm -f "${files_file}"

  if [[ "${ROW_COUNT}" == "1" ]]; then
    clickbench_emit_row_count
  fi
}

clickbench_fetch() {
  local DEFAULT_OUTPUT_DIR="${REPO_ROOT}/fixture/clickbench/data"
  local SINGLE_URL="https://datasets.clickhouse.com/hits_compatible/athena/hits.parquet"
  local PARTITIONED_URL_PREFIX="https://datasets.clickhouse.com/hits_compatible/athena_partitioned"
  local EXPECTED_ROW_COUNT="99997497"

  local MODE="single"
  local OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
  local CONTENT_FINGERPRINT=0
  local ROW_COUNT=0
  local DRY_RUN=0
  local JOBS=8

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --layout|--mode)
        MODE="${2:?missing value for --mode}"
        shift 2
        ;;
      --out|--output-dir)
        OUTPUT_DIR="${2:?missing value for --output-dir}"
        shift 2
        ;;
      --content-fingerprint)
        CONTENT_FINGERPRINT=1
        shift
        ;;
      --row-count)
        ROW_COUNT=1
        shift
        ;;
      --jobs)
        JOBS="${2:?missing value for --jobs}"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      -h|--help)
        clickbench_usage
        return 0
        ;;
      *)
        echo "error: unknown argument: $1" >&2
        clickbench_usage >&2
        return 1
        ;;
    esac
  done

  case "${MODE}" in
    single|partitioned|both) ;;
    *)
      echo "error: unsupported --mode=${MODE}; expected single, partitioned, or both" >&2
      return 1
      ;;
  esac

  if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: unsupported --jobs=${JOBS}; expected a positive integer" >&2
    return 1
  fi

  case "${MODE}" in
    single)
      clickbench_download_single
      ;;
    partitioned)
      clickbench_download_partitioned
      ;;
    both)
      clickbench_download_single
      clickbench_download_partitioned
      ;;
  esac

  clickbench_emit_summary
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

domain="$1"
command="$2"
shift 2

case "${domain}:${command}" in
  tpch:generate)
    exec python3 "${SCRIPT_DIR}/tpc-h/generate.py" "$@"
    ;;
  tpcds:generate)
    exec python3 "${SCRIPT_DIR}/tpc-ds/generate.py" "$@"
    ;;
  clickbench:fetch)
    clickbench_fetch "$@"
    ;;
  graph:dblp-ingest)
    exec python3 "${SCRIPT_DIR}/demo/citation_network/scripts/prepare.py" "$@"
    ;;
  graph:rmat)
    exec python3 "${SCRIPT_DIR}/graph/rmat_generate.py" "$@"
    ;;
  ember2024:prepare)
    exec "${SCRIPT_DIR}/demo/ember2024/prepare.sh" "$@"
    ;;
  amazon-reviews:prepare)
    exec python3 "${SCRIPT_DIR}/demo/amazon_reviews_2023/scripts/prepare.py" "$@"
    ;;
  kuairand:prepare)
    exec python3 "${SCRIPT_DIR}/demo/kuairand/scripts/prepare.py" "$@"
    ;;
  merrec:prepare)
    exec python3 "${SCRIPT_DIR}/demo/merrec/scripts/prepare.py" "$@"
    ;;
  iceberg:rest)
    subcommand="${1:-}"
    if [[ "${subcommand}" != "load" ]]; then
      echo "error: unsupported iceberg rest command '${subcommand}' (expected load)" >&2
      usage >&2
      exit 2
    fi
    shift
    exec "${SCRIPT_DIR}/iceberg-local/rest_load.sh" "$@"
    ;;
  iceberg:glue-moto)
    subcommand="${1:-}"
    if [[ "${subcommand}" != "load" ]]; then
      echo "error: unsupported iceberg glue-moto command '${subcommand}' (expected load)" >&2
      usage >&2
      exit 2
    fi
    shift
    exec "${SCRIPT_DIR}/iceberg-local/glue_moto_load.sh" "$@"
    ;;
  -h:*|--help:*)
    usage
    ;;
  *)
    echo "error: unsupported fixture command '${domain} ${command}'" >&2
    usage >&2
    exit 2
    ;;
esac
