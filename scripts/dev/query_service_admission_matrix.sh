#!/usr/bin/env bash
# Run the Phase C QueryService admission matrix:
#   {guaranteed_minimum, optimistic} x {2, 3, 4} over TPC-H SF10 and SF100.
#
# Each GPU workload invocation is wrapped in /tmp/cudf-gpu.lock by default.
# The script validates the stress TSV for native success and parity, and leaves
# one TSV per matrix cell under target/query_service_admission_matrix/.
# SF100 q18/q21 are required to succeed only when the actual query grant is at
# least the large-query native floor. Lower grants may still succeed; if they
# fail, they must fail cleanly with a temporary structured resource-exhaustion
# error.
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU_LOCK="${GPU_LOCK:-/tmp/cudf-gpu.lock}"
OUTPUT_DIR="${NEXUS_ADMISSION_MATRIX_OUTPUT_DIR:-target/query_service_admission_matrix}"
DRY_RUN="${NEXUS_ADMISSION_MATRIX_DRY_RUN:-0}"
SCALES="${NEXUS_ADMISSION_MATRIX_SCALES:-sf10 sf100}"
SLOT_GUARANTEES="${NEXUS_ADMISSION_MATRIX_SLOT_GUARANTEES:-guaranteed_minimum optimistic}"
CONCURRENCY_LEVELS="${NEXUS_ADMISSION_MATRIX_CONCURRENCY:-2 3 4}"
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
SF100_MAX_SOURCE_CHUNK_BYTES_OVERRIDE="${NEXUS_ADMISSION_MATRIX_SF100_MAX_SOURCE_CHUNK_BYTES:-}"
SF100_MAX_ROW_GROUPS_PER_CHUNK_OVERRIDE="${NEXUS_ADMISSION_MATRIX_SF100_MAX_ROW_GROUPS_PER_CHUNK:-}"
SF100_MAX_SOURCE_CHUNK_BYTES="${SF100_MAX_SOURCE_CHUNK_BYTES_OVERRIDE:-34359738368}"
SF100_MAX_ROW_GROUPS_PER_CHUNK="${SF100_MAX_ROW_GROUPS_PER_CHUNK_OVERRIDE:-1024}"
SF100_MIN_GRANT_MAX_SOURCE_CHUNK_BYTES="${NEXUS_ADMISSION_MATRIX_SF100_MIN_GRANT_MAX_SOURCE_CHUNK_BYTES:-2147483648}"
SF100_MIN_GRANT_MAX_ROW_GROUPS_PER_CHUNK="${NEXUS_ADMISSION_MATRIX_SF100_MIN_GRANT_MAX_ROW_GROUPS_PER_CHUNK:-256}"
SF100_LARGE_QUERY_NATIVE_GRANT_FLOOR_BYTES="${NEXUS_ADMISSION_MATRIX_SF100_LARGE_QUERY_NATIVE_GRANT_FLOOR_BYTES:-44000000000}"

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

is_sf100_min_grant_cell() {
  local scale="$1"
  local slot_guarantee="$2"
  local concurrency="$3"
  [[ "${scale}" == "sf100" && "${slot_guarantee}" == "guaranteed_minimum" && "${concurrency}" -ge 3 ]]
}

max_source_chunk_bytes_for_case() {
  local scale="$1"
  local slot_guarantee="$2"
  local concurrency="$3"
  case "$1" in
    sf10) printf '%s\n' "${SF10_MAX_SOURCE_CHUNK_BYTES}" ;;
    sf100)
      if [[ -z "${SF100_MAX_SOURCE_CHUNK_BYTES_OVERRIDE}" ]] \
        && is_sf100_min_grant_cell "${scale}" "${slot_guarantee}" "${concurrency}"; then
        printf '%s\n' "${SF100_MIN_GRANT_MAX_SOURCE_CHUNK_BYTES}"
      else
        printf '%s\n' "${SF100_MAX_SOURCE_CHUNK_BYTES}"
      fi
      ;;
    *) echo "ERROR: unsupported scale '$1'; expected sf10 or sf100" >&2; return 2 ;;
  esac
}

max_row_groups_per_chunk_for_case() {
  local scale="$1"
  local slot_guarantee="$2"
  local concurrency="$3"
  case "$1" in
    sf10) printf '%s\n' "${SF10_MAX_ROW_GROUPS_PER_CHUNK}" ;;
    sf100)
      if [[ -z "${SF100_MAX_ROW_GROUPS_PER_CHUNK_OVERRIDE}" ]] \
        && is_sf100_min_grant_cell "${scale}" "${slot_guarantee}" "${concurrency}"; then
        printf '%s\n' "${SF100_MIN_GRANT_MAX_ROW_GROUPS_PER_CHUNK}"
      else
        printf '%s\n' "${SF100_MAX_ROW_GROUPS_PER_CHUNK}"
      fi
      ;;
    *) echo "ERROR: unsupported scale '$1'; expected sf10 or sf100" >&2; return 2 ;;
  esac
}

validate_report() {
  local report="$1"
  local scale="$2"
  local expected_rows="$3"

  awk \
    -v scale="${scale}" \
    -v expected_rows="${expected_rows}" \
    -v sf100_large_query_native_grant_floor="${SF100_LARGE_QUERY_NATIVE_GRANT_FLOOR_BYTES}" '
    BEGIN { FS = "\t"; status = 0; query_rows = 0; q18 = 0; q21 = 0; }
    /^#/ || NF == 0 { next }
    !seen_header {
      for (i = 1; i <= NF; i++) {
        col[$i] = i
      }
      seen_header = 1
      next
    }
    $(col["row_type"]) == "query" {
      query_rows++
      query_id = $(col["query_id"])
      if (query_id == "q18") { q18++ }
      if (query_id == "q21") { q21++ }
      grant_bytes = $(col["query_grant_bytes"])
      sf100_large_grant_limited = scale == "sf100" \
        && (query_id == "q18" || query_id == "q21") \
        && grant_bytes != "none" \
        && (grant_bytes + 0) < sf100_large_query_native_grant_floor
      if (sf100_large_grant_limited && $(col["native_runtime_outcome"]) == "failed") {
        if ($(col["planning_outcome"]) != "accepted_replaced") {
          printf("FAIL %s: grant-limited %s planning_outcome=%s\n", FILENAME, query_id, $(col["planning_outcome"])) > "/dev/stderr"
          status = 1
        }
        if ($(col["post_native_execution_path"]) != "native") {
          printf("FAIL %s: grant-limited %s post_native_execution_path=%s\n", FILENAME, query_id, $(col["post_native_execution_path"])) > "/dev/stderr"
          status = 1
        }
        if ($(col["error_kind"]) != "resource_exhausted" || $(col["error_status"]) != "temporary") {
          printf("FAIL %s: grant-limited %s error_kind=%s error_status=%s\n", FILENAME, query_id, $(col["error_kind"]), $(col["error_status"])) > "/dev/stderr"
          status = 1
        }
        if ($(col["error_code"]) != "memory_reservation_failed" && $(col["error_code"]) != "cudf_resource_exhausted") {
          printf("FAIL %s: grant-limited %s error_code=%s\n", FILENAME, query_id, $(col["error_code"])) > "/dev/stderr"
          status = 1
        }
        printf("NOTE %s: grant-limited %s grant_bytes=%s floor_bytes=%s error_code=%s\n", FILENAME, query_id, grant_bytes, sf100_large_query_native_grant_floor, $(col["error_code"])) > "/dev/stderr"
        next
      }
      if ($(col["native_runtime_outcome"]) != "succeeded") {
        printf("FAIL %s: %s native_runtime_outcome=%s\n", FILENAME, query_id, $(col["native_runtime_outcome"])) > "/dev/stderr"
        status = 1
      }
      if ($(col["parity_status"]) != "matched") {
        printf("FAIL %s: %s parity_status=%s\n", FILENAME, query_id, $(col["parity_status"])) > "/dev/stderr"
        status = 1
      }
      if ($(col["planning_outcome"]) != "accepted_replaced") {
        printf("FAIL %s: %s planning_outcome=%s\n", FILENAME, query_id, $(col["planning_outcome"])) > "/dev/stderr"
        status = 1
      }
      if ($(col["post_native_execution_path"]) != "native") {
        printf("FAIL %s: %s post_native_execution_path=%s\n", FILENAME, query_id, $(col["post_native_execution_path"])) > "/dev/stderr"
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
      if (query_rows != expected_rows) {
        printf("FAIL %s: expected %d query rows, saw %d\n", FILENAME, expected_rows, query_rows) > "/dev/stderr"
        status = 1
      }
      if (scale == "sf100" && (q18 == 0 || q21 == 0)) {
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

summarize_pairwise_makespan() {
  local scale concurrency guaranteed_report optimistic_report guaranteed_ms optimistic_ms
  for scale in ${SCALES}; do
    for concurrency in ${CONCURRENCY_LEVELS}; do
      guaranteed_report="${OUTPUT_DIR}/tpch_${scale}_guaranteed_minimum_n${concurrency}.tsv"
      optimistic_report="${OUTPUT_DIR}/tpch_${scale}_optimistic_n${concurrency}.tsv"
      if [[ -f "${guaranteed_report}" && -f "${optimistic_report}" ]]; then
        guaranteed_ms="$(query_wall_sum_ms "${guaranteed_report}")"
        optimistic_ms="$(query_wall_sum_ms "${optimistic_report}")"
        awk -v scale="${scale}" -v n="${concurrency}" -v guaranteed="${guaranteed_ms}" -v optimistic="${optimistic_ms}" '
          BEGIN {
            delta = optimistic - guaranteed
            pct = guaranteed > 0 ? (delta / guaranteed) * 100.0 : 0
            printf("SUMMARY scale=%s max_concurrent_queries=%s guaranteed_minimum_query_wall_sum_ms=%.3f optimistic_query_wall_sum_ms=%.3f optimistic_delta_pct=%.2f\n", scale, n, guaranteed, optimistic, pct)
          }
        '
      fi
    done
  done
}

print_command() {
  local output="$1"
  shift
  printf 'env'
  printf ' %q' "$@"
  printf ' flock %q cargo run --release -p datafusion-nexus-bench --bin query_service_tpch_stress > %q\n' \
    "${GPU_LOCK}" "${output}"
}

run_case() {
  local scale="$1"
  local slot_guarantee="$2"
  local concurrency="$3"
  local root expected_rows interactive_iterations max_source_chunk_bytes max_row_groups_per_chunk output

  root="$(root_for_scale "${scale}")"
  expected_rows="$(expected_rows_for_scale "${scale}")"
  interactive_iterations="$(interactive_iterations_for_scale "${scale}")"
  max_source_chunk_bytes="$(max_source_chunk_bytes_for_case "${scale}" "${slot_guarantee}" "${concurrency}")"
  max_row_groups_per_chunk="$(max_row_groups_per_chunk_for_case "${scale}" "${slot_guarantee}" "${concurrency}")"
  output="${OUTPUT_DIR}/tpch_${scale}_${slot_guarantee}_n${concurrency}.tsv"

  if [[ "${DRY_RUN}" -eq 0 && ! -d "${root}" ]]; then
    echo "ERROR: TPC-H ${scale} root not found: ${root}" >&2
    echo "Set NEXUS_ADMISSION_MATRIX_SF10_PARQUET_DIR or NEXUS_ADMISSION_MATRIX_SF100_PARQUET_DIR, or populate the default fixture root." >&2
    return 2
  fi

  local env_vars=(
    "NEXUS_PARQUET_DIR=${root}"
    "CUDF_TPCH_SCALE_FACTOR=${scale}"
    "CUDF_QUERY_SERVICE_TPCH_SLOT_GUARANTEE=${slot_guarantee}"
    "CUDF_QUERY_SERVICE_TPCH_MAX_CONCURRENT_QUERIES=${concurrency}"
    "CUDF_QUERY_SERVICE_TPCH_BULK_QUERY_IDS=q01,q09,q18,q21"
    "CUDF_QUERY_SERVICE_TPCH_INTERACTIVE_QUERY_IDS=q06,q14,q22"
    "CUDF_QUERY_SERVICE_TPCH_BULK_ITERATIONS=1"
    "CUDF_QUERY_SERVICE_TPCH_INTERACTIVE_ITERATIONS=${interactive_iterations}"
    "CUDF_QUERY_SERVICE_TPCH_MAX_SOURCE_CHUNK_BYTES=${max_source_chunk_bytes}"
    "CUDF_QUERY_SERVICE_TPCH_MAX_ROW_GROUPS_PER_CHUNK=${max_row_groups_per_chunk}"
    "CUDF_QUERY_SERVICE_TPCH_OVERLOAD_PROBE=0"
  )
  if [[ -n "${DEVICE_CAPACITY_BYTES}" ]]; then
    env_vars+=("CUDF_QUERY_SERVICE_TPCH_DEVICE_CAPACITY_BYTES=${DEVICE_CAPACITY_BYTES}")
  fi

  echo "==> ${scale} slot_guarantee=${slot_guarantee} max_concurrent_queries=${concurrency}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_command "${output}" "${env_vars[@]}"
    return 0
  fi

  env "${env_vars[@]}" flock "${GPU_LOCK}" cargo run --release -p datafusion-nexus-bench --bin query_service_tpch_stress >"${output}"
  validate_report "${output}" "${scale}" "${expected_rows}"
  echo "OK: ${output}"
}

for scale in ${SCALES}; do
  for slot_guarantee in ${SLOT_GUARANTEES}; do
    for concurrency in ${CONCURRENCY_LEVELS}; do
      run_case "${scale}" "${slot_guarantee}" "${concurrency}"
    done
  done
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "OK: QueryService admission matrix dry run completed."
else
  summarize_pairwise_makespan
  echo "OK: QueryService admission matrix completed; reports are in ${OUTPUT_DIR}"
fi
