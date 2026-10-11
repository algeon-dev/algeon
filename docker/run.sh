#!/usr/bin/env bash
# Run algeon-server in Docker.
#
# Usage:
#   bash docker/run.sh             # start the server
#   bash docker/run.sh --dry-run   # print the docker run command without executing
#
# AWS credentials (short-lived STS tokens) must be in the host environment:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_SESSION_TOKEN=...
# then run this script.

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# Image
IMAGE="${ALGEON_SERVER_IMAGE:-algeon-server:x86-64-v4}"

# Server core
# ALGEON_SERVER_BIND          host:port to listen on (default: 0.0.0.0:50051)
# ALGEON_SERVER_LOG           tracing filter: "info", "debug", "warn,datafusion=info"
BIND="0.0.0.0:50051"
LOG="info"

# Native GPU execution controls
# Device capacity is owned by the per-device admission profiles; the
# settings below shape bounded source chunking, retries, read scheduling,
# and the statistical aggregate execution mode.
#
# ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES
#   Max bytes read per Parquet source chunk.
#   4 GiB (rg1024/32GiB profile) for decision-grade SF10+ runs.
#
# ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK
#   Max Parquet row groups per source chunk (rg1024 profile).
#
# ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS
#   Bounded retry control.
#
# ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT
#   Bounded source scheduling / prefetch width.
#
# ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE
#   conservative | local_parquet_throughput
#
# ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE
#   gpu_native_tolerant (default) | datafusion_exact_host_welford
NATIVE_MAX_SOURCE_CHUNK_BYTES="${ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES:-}"
NATIVE_MAX_ROW_GROUPS_PER_CHUNK="${ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK:-}"
NATIVE_MAX_RETRY_ATTEMPTS="${ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS:-3}"
NATIVE_MAX_SOURCE_READS_IN_FLIGHT="${ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT:-2}"
NATIVE_SOURCE_CHUNK_PLANNING_MODE="${ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE:-conservative}"
NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE="${ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE:-gpu_native_tolerant}"

# DataFusion session  (single underscores; DataFusion SessionConfig::from_env)
# These are read by DataFusion's SessionConfig::from_env(). algeon-datafusion
# does not parse them directly. DataFusion maps option keys by upper-casing and
# replacing "." with "_"; for example, `datafusion.execution.batch_size`
# becomes `DATAFUSION_EXECUTION_BATCH_SIZE`.
#
# DATAFUSION_EXECUTION_TARGET_PARTITIONS  CPU parallelism for fallback path (no effect on GPU)
# DATAFUSION_EXECUTION_BATCH_SIZE         Arrow record-batch size for DataFusion/fallback operators
DATAFUSION_EXECUTION_TARGET_PARTITIONS="${DATAFUSION_EXECUTION_TARGET_PARTITIONS:-16}"
DATAFUSION_EXECUTION_BATCH_SIZE="${DATAFUSION_EXECUTION_BATCH_SIZE:-8192}"

# S3 region
# AWS_DEFAULT_REGION   REQUIRED for KvikIO S3 (KvikIO reads ONLY this var;
#                      AWS_REGION alone → unsigned requests → 403 on private buckets)
# AWS_REGION           Read by the query engine for S3 source metadata; set both.
#
# Real AWS credentials, when used, must be passed from the host environment.
# Set before running against real AWS:
#   export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_SESSION_TOKEN=...
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# cuGraph SQL DSL
# ALGEON_SERVER_CUGRAPH_ENABLED
#   true exposes the cugraph_* table functions plus metadata UDTFs. The image
#   must be built with cugraph support; set false to ship one artifact but keep
#   the SQL surface hidden at runtime. A lean image built without the cugraph
#   feature must use false.
#
# ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY
#   Optional defaults for graph construction. Policy values:
#   python_cugraph | raw_libcugraph
CUGRAPH_ENABLED="${ALGEON_SERVER_CUGRAPH_ENABLED:-true}"
CUGRAPH_CONSTRUCTION_POLICY="${ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY:-}"

# Build the docker run command
ARGS=(
    docker run --rm
    --gpus all
    --network host

    # Server core
    -e "ALGEON_SERVER_BIND=${BIND}"
    -e "ALGEON_SERVER_LOG=${LOG}"

    # Native GPU execution controls
    -e "ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE=${NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE}"
    -e "ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS=${NATIVE_MAX_RETRY_ATTEMPTS}"
    -e "ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT=${NATIVE_MAX_SOURCE_READS_IN_FLIGHT}"
    -e "ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE=${NATIVE_SOURCE_CHUNK_PLANNING_MODE}"

    # DataFusion session
    -e "DATAFUSION_EXECUTION_TARGET_PARTITIONS=${DATAFUSION_EXECUTION_TARGET_PARTITIONS}"
    -e "DATAFUSION_EXECUTION_BATCH_SIZE=${DATAFUSION_EXECUTION_BATCH_SIZE}"

    # S3 region
    -e "AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}"
    -e "AWS_REGION=${AWS_REGION}"

    # cuGraph SQL DSL
    -e "ALGEON_SERVER_CUGRAPH_ENABLED=${CUGRAPH_ENABLED}"
)

[[ -n "${NATIVE_MAX_SOURCE_CHUNK_BYTES}" ]] &&
    ARGS+=(-e "ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES=${NATIVE_MAX_SOURCE_CHUNK_BYTES}")
[[ -n "${NATIVE_MAX_ROW_GROUPS_PER_CHUNK}" ]] &&
    ARGS+=(-e "ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK=${NATIVE_MAX_ROW_GROUPS_PER_CHUNK}")

[[ -n "${CUGRAPH_CONSTRUCTION_POLICY}" ]] && \
    ARGS+=(-e "ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY=${CUGRAPH_CONSTRUCTION_POLICY}")

# AWS credentials: forward from host environment (short-lived STS tokens).
[[ -n "${AWS_ACCESS_KEY_ID:-}" ]]     && ARGS+=(-e "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}")
[[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] && ARGS+=(-e "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}")
[[ -n "${AWS_SESSION_TOKEN:-}" ]]     && ARGS+=(-e "AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}")
[[ -n "${AWS_PROFILE:-}" ]]           && ARGS+=(-e "AWS_PROFILE=${AWS_PROFILE}")

ARGS+=("${IMAGE}")

# Run or dry-run
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "docker run command (dry run):"
    printf '  %s\n' "${ARGS[@]}"
else
    echo "Starting ${IMAGE} ..."
    exec "${ARGS[@]}"
fi
