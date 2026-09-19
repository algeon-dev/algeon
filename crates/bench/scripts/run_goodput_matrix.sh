#!/usr/bin/env bash
# Run the goodput comparison matrix.
#
# Each group varies exactly one factor and pins the rest. Every run is a fresh
# process, so the device pool starts cold and the configured warmup segment is
# what makes the observation window hot.
#
# Usage:
#   crates/bench/scripts/run_goodput_matrix.sh <group> [group...]
#
# Groups: calibrate g1 g2a g2b g3 g4 p4b graph-calibrate all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

GPU_LOCK="${GPU_LOCK:-/tmp/cudf-gpu.lock}"
# Timing evidence is only meaningful from an optimized build; a debug build
# inflates host-side planning and Arrow costs enough to hide device effects.
CARGO_PROFILE_FLAG="${ALGEON_GOODPUT_CARGO_PROFILE_FLAG:---release}"
# Point at an already-built binary to keep a long matrix immune to rebuilds.
GOODPUT_BIN="${ALGEON_GOODPUT_BIN:-}"

run_goodput() {
  if [[ -n "${GOODPUT_BIN}" ]]; then
    flock "${GPU_LOCK}" "${GOODPUT_BIN}"
  else
    flock "${GPU_LOCK}" cargo run -q ${CARGO_PROFILE_FLAG} -p bench --all-features --bin query_service_goodput
  fi
}
OUT_ROOT="${ALGEON_GOODPUT_OUT_ROOT:-target/goodput}"
SLO_PATH="${ALGEON_GOODPUT_SLO_PATH:-crates/bench/calibration/goodput_slo.tsv}"
GRAPH_SLO_PATH="${ALGEON_GOODPUT_GRAPH_SLO_PATH:-crates/bench/calibration/goodput_slo_graph.tsv}"
SATURATION_SLO_PATH="${ALGEON_GOODPUT_SATURATION_SLO_PATH:-crates/bench/calibration/goodput_slo_saturation.tsv}"

# Pinned across every group.
DEVICE_ORDINAL="${ALGEON_GOODPUT_DEVICE_ORDINAL:-0}"
CD="${ALGEON_GOODPUT_DEVICE_CAPACITY_BYTES:-68719476736}"   # 64 GiB
WARMUP_SECONDS="${ALGEON_GOODPUT_WARMUP_SECONDS:-120}"
WINDOW_SECONDS="${ALGEON_GOODPUT_WINDOW_SECONDS:-300}"
SEED="${ALGEON_GOODPUT_SEED:-20260910}"
# Independent per-request cap assignment. Calibration reads the same manifest
# as the measured runs, so both arms of a paired comparison are judged against
# deadlines frozen at the caps they actually request.
REQUEST_CAP_MANIFEST="${ALGEON_GOODPUT_REQUEST_CAP_MANIFEST:-}"
# P4b sweeps several seeds and load intensities so one favourable draw cannot
# carry the comparison.
P4B_SEEDS="${ALGEON_GOODPUT_P4B_SEEDS:-${SEED} 20260911 20260912}"
P4B_RATE_MULTIPLIERS="${ALGEON_GOODPUT_P4B_RATE_MULTIPLIERS:-0.7 1.0 1.5}"
P4B_MAX_ACTIVE="${ALGEON_GOODPUT_P4B_MAX_ACTIVE:-3}"
MAX_QUEUED="${ALGEON_GOODPUT_MAX_QUEUED_ATTEMPTS:-32}"
MAX_QUEUE_WAIT_MS="${ALGEON_GOODPUT_MAX_QUEUE_WAIT_MS:-60000}"
SOURCE_CHUNK_BYTES="${ALGEON_GOODPUT_MAX_SOURCE_CHUNK_BYTES:-536870912}"
ROW_GROUPS_PER_CHUNK="${ALGEON_GOODPUT_MAX_ROW_GROUPS_PER_CHUNK:-1024}"
SLO_MULTIPLIER="${ALGEON_GOODPUT_SLO_MULTIPLIER:-3.0}"
QUERY_TIMEOUT_MS="${ALGEON_GOODPUT_QUERY_TIMEOUT_MS:-300000}"

TPCH_DIR="${ALGEON_GOODPUT_TPCH_DIR:-/scratch/data/tpch/sf100/parquet}"
TPCDS_DIR="${ALGEON_GOODPUT_TPCDS_DIR:-/scratch/data/tpcds/sf100/parquet}"
TPCH_SCALE="${ALGEON_GOODPUT_TPCH_SCALE:-sf100}"
TPCDS_SCALE="${ALGEON_GOODPUT_TPCDS_SCALE:-sf100}"
GRAPH_EDGES="${ALGEON_GOODPUT_GRAPH_EDGES:-/scratch/data/graph/rmat_s22_e16_skewed/edges.parquet}"
GRAPH_DATASET="${ALGEON_GOODPUT_GRAPH_DATASET:-rmat_s22_e16_skewed}"

SQL_MIX="${ALGEON_GOODPUT_SQL_MIX:?set ALGEON_GOODPUT_SQL_MIX to the calibrated SQL mix}"
GRAPH_MIX="${ALGEON_GOODPUT_GRAPH_MIX:-graph.pagerank@2,graph.bfs@2,graph.wcc@2,graph.triangles@1,graph.betweenness_k64@1}"
SATURATION_MIX="${ALGEON_GOODPUT_SATURATION_MIX:-tpch.q21@1,adhoc.wide_distinct@1,tpch.q01@1}"

mkdir -p "${OUT_ROOT}"

# Digest of every source difference from HEAD, contents included.
#
# `git status --porcelain` alone names the changed files without reading them,
# so an edit to an already-modified file leaves the digest untouched and two
# different binaries record the same identity. Tracked edits come from the
# diff, untracked ones from their own contents.
worktree_digest() {
  {
    git status --porcelain=v1 --untracked-files=all
    git diff HEAD
    git ls-files --others --exclude-standard -z | xargs -0 -r sha256sum
  } | sha256sum | cut -c1-16
}

# Digest of the executable that will actually run, or `unbuilt` before the
# first build. It is the one fact that distinguishes two runs directly, without
# reasoning about which edits reached the binary.
binary_digest() {
  local path="${GOODPUT_BIN:-target/${CARGO_PROFILE_FLAG#--}/query_service_goodput}"
  [[ -f "${path}" ]] || { echo unbuilt; return; }
  sha256sum "${path}" | cut -c1-16
}

source_identity() {
  local root gitlinks
  root="$(git rev-parse HEAD)"
  gitlinks="$(git submodule status | awk '{printf "%s=%s;", $2, $1}')"
  printf 'root=%s dirty=%s gitlinks=%s profile=%s binary=%s' \
    "${root}" \
    "$(worktree_digest)" \
    "${gitlinks}" \
    "${GOODPUT_BIN:-${CARGO_PROFILE_FLAG#--}}" \
    "$(binary_digest)"
}

IDENTITY="$(source_identity)"

record_environment() {
  local path="${OUT_ROOT}/identity.txt"
  {
    printf 'captured_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'source_identity=%s\n' "${IDENTITY}"
    printf 'device_capacity_bytes=%s\n' "${CD}"
    printf 'request_cap_manifest=%s\n' "${REQUEST_CAP_MANIFEST:-none}"
    [[ -n "${REQUEST_CAP_MANIFEST}" ]] && printf 'request_cap_manifest_sha256=%s\n' "$(sha256sum "${REQUEST_CAP_MANIFEST}" | cut -c1-16)"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
    nvcc --version | tail -1 | sed 's/^/nvcc=/'
    rustc --version | sed 's/^/rustc=/'
    for lib in target/native/cudf-install/lib/libcudf.so target/native/cugraph-build/libcugraph_c.so \
               target/native/cuvs-install/lib/libcuvs.so; do
      [[ -f "${lib}" ]] && printf '%s=%s\n' "$(basename "${lib}")" "$(sha256sum "${lib}" | cut -c1-16)"
    done
  } >"${path}"
  echo "wrote ${path}"
}

run_case() {
  local group="$1" mix="$2" slo="$3" cap="$4" max_active="$5" rate_multiplier="$6" pool_state="$7" warmup="$8"
  echo "==> ${group} cap=${cap} max_active=${max_active} rate_multiplier=${rate_multiplier} pool=${pool_state}"
  ALGEON_GOODPUT_MODE=measure \
  ALGEON_GOODPUT_MIX="${mix}" \
  ALGEON_GOODPUT_TPCH_DIR="${TPCH_DIR}" \
  ALGEON_GOODPUT_TPCDS_DIR="${TPCDS_DIR}" \
  ALGEON_GOODPUT_TPCH_SCALE="${TPCH_SCALE}" \
  ALGEON_GOODPUT_TPCDS_SCALE="${TPCDS_SCALE}" \
  ALGEON_GOODPUT_GRAPH_EDGES="${GRAPH_EDGES}" \
  ALGEON_GOODPUT_GRAPH_DATASET="${GRAPH_DATASET}" \
  ALGEON_GOODPUT_ARRIVAL=poisson \
  ALGEON_GOODPUT_RATE_MULTIPLIER="${rate_multiplier}" \
  ALGEON_GOODPUT_WARMUP_SECONDS="${warmup}" \
  ALGEON_GOODPUT_WINDOW_SECONDS="${WINDOW_SECONDS}" \
  ALGEON_GOODPUT_SEED="${SEED}" \
  ALGEON_GOODPUT_DEVICE_ORDINALS="${DEVICE_ORDINAL}" \
  ALGEON_GOODPUT_DEVICE_CAPACITY_BYTES="${CD}" \
  ALGEON_GOODPUT_QUERY_CAP_BYTES="${cap}" \
  ALGEON_GOODPUT_REQUEST_CAP_MANIFEST="${REQUEST_CAP_MANIFEST}" \
  ALGEON_GOODPUT_MAX_ACTIVE_ATTEMPTS="${max_active}" \
  ALGEON_GOODPUT_MAX_QUEUED_ATTEMPTS="${MAX_QUEUED}" \
  ALGEON_GOODPUT_MAX_QUEUE_WAIT_MS="${MAX_QUEUE_WAIT_MS}" \
  ALGEON_GOODPUT_QUERY_TIMEOUT_MS="${QUERY_TIMEOUT_MS}" \
  ALGEON_GOODPUT_MAX_SOURCE_CHUNK_BYTES="${SOURCE_CHUNK_BYTES}" \
  ALGEON_GOODPUT_MAX_ROW_GROUPS_PER_CHUNK="${ROW_GROUPS_PER_CHUNK}" \
  ALGEON_GOODPUT_SLO_PATH="${slo}" \
  ALGEON_GOODPUT_SLO_MULTIPLIER="${SLO_MULTIPLIER}" \
  ALGEON_GOODPUT_POOL_STATE="${pool_state}" \
  ALGEON_GOODPUT_OUT_DIR="${OUT_ROOT}" \
  ALGEON_GOODPUT_RUN_ID="${group}" \
  ALGEON_GOODPUT_GROUP="${group}" \
  ALGEON_GOODPUT_SOURCE_IDENTITY="${IDENTITY}" \
    run_goodput >/dev/null
  collect_matrix >/dev/null
}

run_calibration() {
  local mix="$1" slo="$2" label="$3"
  echo "==> calibrate ${label}"
  ALGEON_GOODPUT_MODE=calibrate \
  ALGEON_GOODPUT_MIX="${mix}" \
  ALGEON_GOODPUT_TPCH_DIR="${TPCH_DIR}" \
  ALGEON_GOODPUT_TPCDS_DIR="${TPCDS_DIR}" \
  ALGEON_GOODPUT_TPCH_SCALE="${TPCH_SCALE}" \
  ALGEON_GOODPUT_TPCDS_SCALE="${TPCDS_SCALE}" \
  ALGEON_GOODPUT_GRAPH_EDGES="${GRAPH_EDGES}" \
  ALGEON_GOODPUT_GRAPH_DATASET="${GRAPH_DATASET}" \
  ALGEON_GOODPUT_DEVICE_ORDINALS="${DEVICE_ORDINAL}" \
  ALGEON_GOODPUT_DEVICE_CAPACITY_BYTES="${CD}" \
  ALGEON_GOODPUT_QUERY_CAP_BYTES="${CD}" \
  ALGEON_GOODPUT_REQUEST_CAP_MANIFEST="${REQUEST_CAP_MANIFEST}" \
  ALGEON_GOODPUT_MAX_ACTIVE_ATTEMPTS=1 \
  ALGEON_GOODPUT_MAX_SOURCE_CHUNK_BYTES="${SOURCE_CHUNK_BYTES}" \
  ALGEON_GOODPUT_MAX_ROW_GROUPS_PER_CHUNK="${ROW_GROUPS_PER_CHUNK}" \
  ALGEON_GOODPUT_SLO_PATH="${slo}" \
  ALGEON_GOODPUT_OUT_DIR="${OUT_ROOT}" \
  ALGEON_GOODPUT_RUN_ID="calibrate-${label}" \
  ALGEON_GOODPUT_GROUP="calibrate-${label}" \
  ALGEON_GOODPUT_SOURCE_IDENTITY="${IDENTITY}" \
    run_goodput
}

collect_matrix() {
  local out="${OUT_ROOT}/matrix.tsv"
  local first=1
  : >"${out}"
  for summary in "${OUT_ROOT}"/*/summary.tsv; do
    [[ -f "${summary}" ]] || continue
    if [[ "${first}" == 1 ]]; then
      head -1 "${summary}" >>"${out}"
      first=0
    fi
    tail -n +2 "${summary}" >>"${out}"
  done
  echo "wrote ${out}"
}

# Largest declared capacity the device registry accepts, with grants summing to
# nearly all of it. The fixed-cap model forbids a declared capacity above
# physical memory, so the ledger can never over-promise; this group tests
# whether real allocations plus pool overhead can still exceed the device while
# the ledger believes there is room. It is not a goodput comparison: the
# declared capacity differs from every other group, so it is reported alone.
group_saturation() {
  local cd=$((92 * 1024 * 1024 * 1024))
  local cap=$((30 * 1024 * 1024 * 1024))
  CD="${cd}" run_calibration "${SATURATION_MIX}" "${SATURATION_SLO_PATH}" saturation || return 1
  CD="${cd}" WINDOW_SECONDS=180 run_case saturation "${SATURATION_MIX}" "${SATURATION_SLO_PATH}" "${cap}" 3 4.0 hot 60
}

group_g1() {
  for max_active in 1 2 3 4; do
    local cap=$((CD / max_active))
    for rate in 0.7 1.0 1.5; do
      run_case "g1-active${max_active}-rate${rate}" "${SQL_MIX}" "${SLO_PATH}" "${cap}" "${max_active}" "${rate}" hot "${WARMUP_SECONDS}"
    done
  done
}

group_g2a() {
  for divisor in 1 2 4 8; do
    run_case "g2a-cap-div${divisor}" "${SQL_MIX}" "${SLO_PATH}" "$((CD / divisor))" 1 0.7 hot "${WARMUP_SECONDS}"
  done
}

group_g2b() {
  for divisor in 2 4; do
    run_case "g2b-active2-cap-div${divisor}" "${SQL_MIX}" "${SLO_PATH}" "$((CD / divisor))" 2 1.0 hot "${WARMUP_SECONDS}"
  done
}

# One query with a large seed batch versus several queries with small seed
# batches, compared on completed work units rather than completed queries.
# The split configuration runs at max_active=2 because a single betweenness
# query with 64 seeds already peaks above C_d/4.
group_g3() {
  run_case "g3-a-batched"          "graph.betweenness_k256@1" "${GRAPH_SLO_PATH}" "${CD}"        1 1.0 hot "${WARMUP_SECONDS}"
  run_case "g3-b-split-concurrent" "graph.betweenness_k64@1"  "${GRAPH_SLO_PATH}" "$((CD / 2))" 2 1.0 hot "${WARMUP_SECONDS}"
  run_case "g3-c-split-serial"     "graph.betweenness_k64@1"  "${GRAPH_SLO_PATH}" "${CD}"        1 1.0 hot "${WARMUP_SECONDS}"
}

group_g4() {
  run_case "g4-cold" "${SQL_MIX}" "${SLO_PATH}" "${CD}" 1 0.7 cold 0
  run_case "g4-hot"  "${SQL_MIX}" "${SLO_PATH}" "${CD}" 1 0.7 hot "${WARMUP_SECONDS}"
}

# Managed capacity, request caps, arrival trace, seed, window and request set
# are all pinned; concurrency is the only varied factor. Every request carries
# its own cap from the manifest, so the mixed arm overlaps differently sized
# grants rather than repeating one cap.
group_p4b() {
  [[ -n "${REQUEST_CAP_MANIFEST}" ]] || {
    echo "p4b requires ALGEON_GOODPUT_REQUEST_CAP_MANIFEST" >&2
    return 2
  }
  local seed rate
  for seed in ${P4B_SEEDS}; do
    for rate in ${P4B_RATE_MULTIPLIERS}; do
      SEED="${seed}" run_case "p4b-single-seed${seed}-rate${rate}" \
        "${SQL_MIX}" "${SLO_PATH}" "${CD}" 1 "${rate}" hot "${WARMUP_SECONDS}"
      SEED="${seed}" run_case "p4b-mixed-seed${seed}-rate${rate}" \
        "${SQL_MIX}" "${SLO_PATH}" "${CD}" "${P4B_MAX_ACTIVE}" "${rate}" hot "${WARMUP_SECONDS}"
    done
  done
}

[[ $# -gt 0 ]] || { echo "usage: $0 <calibrate|graph-calibrate|g1|g2a|g2b|g3|g4|p4b|saturation|all>..." >&2; exit 2; }

record_environment
for group in "$@"; do
  case "${group}" in
    calibrate) run_calibration "${SQL_MIX}" "${SLO_PATH}" sql ;;
    graph-calibrate) run_calibration "${GRAPH_MIX},graph.betweenness_k256@1" "${GRAPH_SLO_PATH}" graph ;;
    g1) group_g1 ;;
    g2a) group_g2a ;;
    g2b) group_g2b ;;
    g3) group_g3 ;;
    saturation) group_saturation ;;
    g4) group_g4 ;;
    p4b) group_p4b ;;
    all) group_g1; group_g2a; group_g2b; group_g4 ;;
    *) echo "unknown group ${group}" >&2; exit 2 ;;
  esac
done
collect_matrix
