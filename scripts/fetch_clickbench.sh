#!/usr/bin/env bash
# Download official ClickBench Parquet fixtures.
#
# Usage:
#   scripts/fetch_clickbench.sh --mode single
#   scripts/fetch_clickbench.sh --mode partitioned
#   scripts/fetch_clickbench.sh --mode both --output-dir /opt/dlami/nvme/clickbench
#
# The output layout is:
#   <output-dir>/hits.parquet
#   <output-dir>/athena_partitioned/hits_0.parquet ... hits_99.parquet

set -euo pipefail

DEFAULT_OUTPUT_DIR="/opt/dlami/nvme/clickbench"
SINGLE_URL="https://datasets.clickhouse.com/hits_compatible/athena/hits.parquet"
PARTITIONED_URL_PREFIX="https://datasets.clickhouse.com/hits_compatible/athena_partitioned"
EXPECTED_ROW_COUNT="99997497"

MODE="single"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
CONTENT_FINGERPRINT=0
ROW_COUNT=0
DRY_RUN=0
JOBS=8

usage() {
  cat <<'USAGE'
Usage: scripts/fetch_clickbench.sh [options]

Options:
  --mode single|partitioned|both   Dataset layout to download (default: single)
  --output-dir PATH                Destination root (default: /opt/dlami/nvme/clickbench)
  --content-fingerprint            Emit sha256 over file contents after download
  --row-count                      Emit row count using duckdb if available
  --jobs N                         Parallel downloads for partitioned mode (default: 8)
  --dry-run                        Print planned downloads without fetching
  -h, --help                       Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:?missing value for --mode}"
      shift 2
      ;;
    --output-dir)
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
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${MODE}" in
  single|partitioned|both) ;;
  *)
    echo "error: unsupported --mode=${MODE}; expected single, partitioned, or both" >&2
    exit 1
    ;;
esac

if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: unsupported --jobs=${JOBS}; expected a positive integer" >&2
  exit 1
fi

download_one() {
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

download_single() {
  download_one "${SINGLE_URL}" "${OUTPUT_DIR}/hits.parquet"
}

download_partitioned() {
  mkdir -p "${OUTPUT_DIR}/athena_partitioned"
  for i in $(seq 0 99); do
    if [[ "${DRY_RUN}" == "1" ]]; then
      download_one "${PARTITIONED_URL_PREFIX}/hits_${i}.parquet" \
        "${OUTPUT_DIR}/athena_partitioned/hits_${i}.parquet"
      continue
    fi
    download_one "${PARTITIONED_URL_PREFIX}/hits_${i}.parquet" \
      "${OUTPUT_DIR}/athena_partitioned/hits_${i}.parquet" &
    while [[ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${JOBS}" ]]; do
      wait -n
    done
  done
  wait
}

collect_files() {
  case "${MODE}" in
    single)
      find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'hits.parquet' | sort
      ;;
    partitioned)
      find "${OUTPUT_DIR}/athena_partitioned" -maxdepth 1 -type f -name 'hits_*.parquet' | sort
      ;;
    both)
      {
        find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'hits.parquet'
        find "${OUTPUT_DIR}/athena_partitioned" -maxdepth 1 -type f -name 'hits_*.parquet'
      } | sort
      ;;
  esac
}

emit_summary() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  local files_file
  files_file="$(mktemp)"
  trap 'rm -f "${files_file}"' RETURN
  collect_files > "${files_file}"

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

  if [[ "${ROW_COUNT}" == "1" ]]; then
    emit_row_count
  fi
}

emit_row_count() {
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
      sql="SELECT count(*) FROM read_parquet('${OUTPUT_DIR}/athena_partitioned/hits_*.parquet');"
      ;;
    both)
      sql="SELECT count(*) FROM read_parquet('${OUTPUT_DIR}/hits.parquet');"
      ;;
  esac

  local count
  count="$(duckdb -noheader -csv -c "${sql}" | tr -d '\r')"
  printf 'row_count=%s\n' "${count}"
}

case "${MODE}" in
  single)
    download_single
    ;;
  partitioned)
    download_partitioned
    ;;
  both)
    download_single
    download_partitioned
    ;;
esac

emit_summary
