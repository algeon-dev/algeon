#!/usr/bin/env bash
# Run the P2 QueryService admission matrix:
#   {sf10, sf100} x configured per-device active-attempt limits.
#
# Each GPU workload invocation is wrapped in /tmp/cudf-gpu.lock by default.
# The script validates the stress TSV for native success and parity, and leaves
# one TSV per matrix cell under target/query_service_admission_matrix/.
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU_LOCK="${GPU_LOCK:-/tmp/cudf-gpu.lock}"
OUTPUT_DIR="${NEXUS_ADMISSION_MATRIX_OUTPUT_DIR:-target/query_service_admission_matrix}"
DRY_RUN="${NEXUS_ADMISSION_MATRIX_DRY_RUN:-0}"
SCALES="${NEXUS_ADMISSION_MATRIX_SCALES:-sf10 sf100}"
MAX_ACTIVE_ATTEMPTS_PER_DEVICE="${NEXUS_ADMISSION_MATRIX_MAX_ACTIVE_ATTEMPTS_PER_DEVICE:-2 3 4}"
MAX_QUEUED_ATTEMPTS="${NEXUS_ADMISSION_MATRIX_MAX_QUEUED_ATTEMPTS:-64}"
MAX_QUEUE_WAIT_MS="${NEXUS_ADMISSION_MATRIX_MAX_QUEUE_WAIT_MS:-30000}"
DEVICE_CAPACITY_BYTES="${NEXUS_ADMISSION_MATRIX_DEVICE_CAPACITY_BYTES:-}"

default_tpch_root() {
  local scale="$1"
  local local_root="./fixture/tpc-h/data/${scale}/parquet"
  if [[ "${scale}" == "sf100" ]]; then
    local_root="./fixture/tpc-h/data/sf100_optimized/parquet"
  fi
  printf '%s\n' "${local_root}"
}

SF10_ROOT="${NEXUS_ADMISSION_MATRIX_SF10_PARQUET_DIR:-}"
SF10_ROOT="${SF10_ROOT:-$(default_tpch_root sf10)}"
SF100_ROOT="${NEXUS_ADMISSION_MATRIX_SF100_PARQUET_DIR:-}"
SF100_ROOT="${SF100_ROOT:-$(default_tpch_root sf100)}"

SF10_MAX_SOURCE_CHUNK_BYTES="${NEXUS_ADMISSION_MATRIX_SF10_MAX_SOURCE_CHUNK_BYTES:-4294967296}"
SF10_MAX_ROW_GROUPS_PER_CHUNK="${NEXUS_ADMISSION_MATRIX_SF10_MAX_ROW_GROUPS_PER_CHUNK:-1024}"
SF100_MAX_SOURCE_CHUNK_BYTES="${NEXUS_ADMISSION_MATRIX_SF100_MAX_SOURCE_CHUNK_BYTES:-34359738368}"
SF100_MAX_ROW_GROUPS_PER_CHUNK="${NEXUS_ADMISSION_MATRIX_SF100_MAX_ROW_GROUPS_PER_CHUNK:-1024}"

case "${DRY_RUN}" in
  1 | true | TRUE | yes | YES | on | ON)
    DRY_RUN=1
    ;;
  *)
    DRY_RUN=0
    ;;
esac

mkdir -p "${OUTPUT_DIR}"

root_for_scale() {
  case "$1" in
    sf10) printf '%s\n' "${SF10_ROOT}" ;;
    sf100) printf '%s\n' "${SF100_ROOT}" ;;
    *) echo "ERROR: unsupported scale '$1'; expected sf10 or sf100" >&2; return 2 ;;
  esac
}

expected_rows_for_scale() {
  case "$1" in
    sf10) printf '%s\n' 28 ;;
    sf100) printf '%s\n' 16 ;;
    *) echo "ERROR: unsupported scale '$1'; expected sf10 or sf100" >&2; return 2 ;;
  esac
}

interactive_iterations_for_scale() {
  case "$1" in
    sf10) printf '%s\n' 4 ;;
    sf100) printf '%s\n' 2 ;;
    *) echo "ERROR: unsupported scale '$1'; expected sf10 or sf100" >&2; return 2 ;;
  esac
}

max_source_chunk_bytes_for_scale() {
  case "$1" in
    sf10) printf '%s\n' "${SF10_MAX_SOURCE_CHUNK_BYTES}" ;;
    sf100) printf '%s\n' "${SF100_MAX_SOURCE_CHUNK_BYTES}" ;;
    *) echo "ERROR: unsupported scale '$1'; expected sf10 or sf100" >&2; return 2 ;;
  esac
}

max_row_groups_per_chunk_for_scale() {
  case "$1" in
    sf10) printf '%s\n' "${SF10_MAX_ROW_GROUPS_PER_CHUNK}" ;;
    sf100) printf '%s\n' "${SF100_MAX_ROW_GROUPS_PER_CHUNK}" ;;
    *) echo "ERROR: unsupported scale '$1'; expected sf10 or sf100" >&2; return 2 ;;
  esac
}

validate_report() {
  local report="$1"
  local scale="$2"
  local expected_rows="$3"

  awk \
    -v scale="${scale}" \
    -v expected_rows="${expected_rows}" '
    BEGIN {
      FS = "\t"
      status = 0
      query_rows = 0
      q18 = 0
      q21 = 0
      required["row_type"] = 1
      required["query_id"] = 1
      required["native_runtime_outcome"] = 1
      required["parity_status"] = 1
      required["planning_outcome"] = 1
      required["error_code"] = 1
      required["error_message"] = 1
    }
    /^#/ || NF == 0 { next }
    !seen_header {
      for (i = 1; i <= NF; i++) {
        col[$i] = i
      }
      for (name in required) {
        if (!(name in col)) {
          printf("FAIL %s: missing TSV column %s\n", FILENAME, name) > "/dev/stderr"
          status = 1
          valid_header = 0
        }
      }
      if (status == 0) {
        valid_header = 1
      }
      seen_header = 1
      next
    }
    !valid_header { next }
    $(col["row_type"]) == "query" {
      query_rows++
      query_id = $(col["query_id"])
      if (query_id == "q18") { q18++ }
      if (query_id == "q21") { q21++ }
      if ($(col["native_runtime_outcome"]) != "succeeded") {
        printf("FAIL %s: %s native_runtime_outcome=%s\n", FILENAME, query_id, $(col["native_runtime_outcome"])) > "/dev/stderr"
        status = 1
      }
      if ($(col["parity_status"]) != "matched") {
        printf("FAIL %s: %s parity_status=%s\n", FILENAME, query_id, $(col["parity_status"])) > "/dev/stderr"
        status = 1
      }
      if ($(col["planning_outcome"]) != "selected") {
        printf("FAIL %s: %s planning_outcome=%s\n", FILENAME, query_id, $(col["planning_outcome"])) > "/dev/stderr"
        status = 1
      }
      if ($(col["error_code"]) != "none") {
        printf("FAIL %s: %s error_code=%s error_message=%s\n", FILENAME, query_id, $(col["error_code"]), $(col["error_message"])) > "/dev/stderr"
        status = 1
      }
    }
    END {
      if (!seen_header) {
        printf("FAIL %s: missing TSV header\n", FILENAME) > "/dev/stderr"
        status = 1
      }
      if (valid_header && query_rows != expected_rows) {
        printf("FAIL %s: expected %d query rows, saw %d\n", FILENAME, expected_rows, query_rows) > "/dev/stderr"
        status = 1
      }
      if (valid_header && scale == "sf100" && (q18 == 0 || q21 == 0)) {
        printf("FAIL %s: SF100 matrix must include q18 and q21; saw q18=%d q21=%d\n", FILENAME, q18, q21) > "/dev/stderr"
        status = 1
      }
      exit status
    }
  ' "${report}"
}

query_wall_sum_ms() {
  local report="$1"
  awk '
    BEGIN { FS = "\t"; sum = 0; }
    /^#/ || NF == 0 { next }
    !seen_header {
      for (i = 1; i <= NF; i++) {
        col[$i] = i
      }
      seen_header = 1
      next
    }
    $(col["row_type"]) == "query" {
      sum += $(col["total_wall_ms"])
    }
    END { printf "%.3f\n", sum }
  ' "${report}"
}

summarize_makespan() {
  local scale active_attempts report query_wall_ms
  for scale in ${SCALES}; do
    for active_attempts in ${MAX_ACTIVE_ATTEMPTS_PER_DEVICE}; do
      report="${OUTPUT_DIR}/tpch_${scale}_a${active_attempts}.tsv"
      if [[ -f "${report}" ]]; then
        query_wall_ms="$(query_wall_sum_ms "${report}")"
        printf 'SUMMARY scale=%s max_active_attempts_per_device=%s query_wall_sum_ms=%s\n' \
          "${scale}" "${active_attempts}" "${query_wall_ms}"
      fi
    done
  done
}

print_command() {
  local output="$1"
  shift
  printf 'env'
  printf ' %q' "$@"
  printf ' flock %q cargo run --release -p nexus-bench --bin query_service_tpch_stress > %q\n' \
    "${GPU_LOCK}" "${output}"
}

run_case() {
  local scale="$1"
  local active_attempts="$2"
  local root expected_rows interactive_iterations max_source_chunk_bytes max_row_groups_per_chunk output

  root="$(root_for_scale "${scale}")"
  expected_rows="$(expected_rows_for_scale "${scale}")"
  interactive_iterations="$(interactive_iterations_for_scale "${scale}")"
  max_source_chunk_bytes="$(max_source_chunk_bytes_for_scale "${scale}")"
  max_row_groups_per_chunk="$(max_row_groups_per_chunk_for_scale "${scale}")"
  output="${OUTPUT_DIR}/tpch_${scale}_a${active_attempts}.tsv"

  if [[ "${DRY_RUN}" -eq 0 && ! -d "${root}" ]]; then
    echo "ERROR: TPC-H ${scale} root not found: ${root}" >&2
    echo "Set NEXUS_ADMISSION_MATRIX_SF10_PARQUET_DIR or NEXUS_ADMISSION_MATRIX_SF100_PARQUET_DIR, or populate the default fixture root." >&2
    return 2
  fi

  local env_vars=(
    "NEXUS_PARQUET_DIR=${root}"
    "NEXUS_TPCH_SCALE_FACTOR=${scale}"
    "NEXUS_QUERY_SERVICE_TPCH_MAX_ACTIVE_ATTEMPTS_PER_DEVICE=${active_attempts}"
    "NEXUS_QUERY_SERVICE_TPCH_MAX_QUEUED_ATTEMPTS=${MAX_QUEUED_ATTEMPTS}"
    "NEXUS_QUERY_SERVICE_TPCH_MAX_QUEUE_WAIT_MS=${MAX_QUEUE_WAIT_MS}"
    "NEXUS_QUERY_SERVICE_TPCH_BULK_QUERY_IDS=q01,q09,q18,q21"
    "NEXUS_QUERY_SERVICE_TPCH_INTERACTIVE_QUERY_IDS=q06,q14,q22"
    "NEXUS_QUERY_SERVICE_TPCH_BULK_ITERATIONS=1"
    "NEXUS_QUERY_SERVICE_TPCH_INTERACTIVE_ITERATIONS=${interactive_iterations}"
    "NEXUS_QUERY_SERVICE_TPCH_MAX_SOURCE_CHUNK_BYTES=${max_source_chunk_bytes}"
    "NEXUS_QUERY_SERVICE_TPCH_MAX_ROW_GROUPS_PER_CHUNK=${max_row_groups_per_chunk}"
    "NEXUS_QUERY_SERVICE_TPCH_OVERLOAD_PROBE=0"
  )
  if [[ -n "${DEVICE_CAPACITY_BYTES}" ]]; then
    env_vars+=("NEXUS_QUERY_SERVICE_TPCH_DEVICE_CAPACITY_BYTES=${DEVICE_CAPACITY_BYTES}")
  fi

  echo "==> ${scale} max_active_attempts_per_device=${active_attempts}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_command "${output}" "${env_vars[@]}"
    return 0
  fi

  env "${env_vars[@]}" flock "${GPU_LOCK}" cargo run --release -p nexus-bench --bin query_service_tpch_stress >"${output}"
  validate_report "${output}" "${scale}" "${expected_rows}"
  echo "OK: ${output}"
}

for scale in ${SCALES}; do
  for active_attempts in ${MAX_ACTIVE_ATTEMPTS_PER_DEVICE}; do
    run_case "${scale}" "${active_attempts}"
  done
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "OK: QueryService admission matrix dry run completed."
else
  summarize_makespan
  echo "OK: QueryService admission matrix completed; reports are in ${OUTPUT_DIR}"
fi
