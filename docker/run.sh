#!/usr/bin/env bash
# Run datafusion-nexus-server in Docker.
#
# Usage:
#   bash docker/run.sh             # start the server
#   bash docker/run.sh --dry-run   # print the docker run command without executing
#   NEXUS_SERVER_IMAGE=datafusion-nexus-server:x86-64-v3-tuned bash docker/run.sh
#
# AWS credentials (short-lived STS tokens) must be in the host environment:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_SESSION_TOKEN=...
# then run this script.

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
IMAGE="${NEXUS_SERVER_IMAGE:-datafusion-nexus-server:x86-64-v2}"

# ---------------------------------------------------------------------------
# Server core
# NEXUS_SERVER_BIND          host:port to listen on (default: 0.0.0.0:50051)
# NEXUS_SERVER_LOG           tracing filter — "info", "debug", "warn,datafusion=info"
# NEXUS_SERVER_MAX_IN_FLIGHT max concurrent in-flight Flight SQL requests (default: 1)
# ---------------------------------------------------------------------------
BIND="0.0.0.0:50051"
LOG="info"
MAX_IN_FLIGHT=1

# ---------------------------------------------------------------------------
# Native GPU execution controls
# Server default without explicit budget env vars is bounded auto-sized. This
# helper sets an explicit bounded profile for repeatable local/decision-grade
# runs. Set NEXUS_SERVER_NATIVE_MEMORY_POLICY=unbounded to omit bounded controls.
#
# NEXUS_SERVER_NATIVE_MEMORY_POLICY
#   bounded | unbounded
#
# NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES
#   Hard cap on live GPU memory reservations.
#   = nvidia-smi memory.total - 500 MiB overhead
#   RTX PRO 6000 Blackwell (97887 MiB): 97387 MiB = 102117670912 bytes
#
# NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES
#   Max bytes read per Parquet source chunk.
#   4 GiB (rg1024/32GiB profile) for decision-grade SF10+ runs.
#
# NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK
#   Max Parquet row groups per source chunk (rg1024 profile).
#
# NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS
# NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS
#   Bounded retry controls.
#
# NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT
#   Bounded source scheduling / prefetch width.
#
# NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE
#   conservative | local_parquet_throughput
#
# NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE
#   strict_bounded | adaptive
#
# NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE
#   datafusion_exact_host_welford | gpu_native_tolerant
# ---------------------------------------------------------------------------
NATIVE_MEMORY_POLICY="${NEXUS_SERVER_NATIVE_MEMORY_POLICY:-bounded}"
NATIVE_MEMORY_POLICY_NORMALIZED="${NATIVE_MEMORY_POLICY,,}"
NATIVE_DEVICE_BUDGET_BYTES="${NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES:-102117670912}"   # 97387 MiB
NATIVE_MAX_SOURCE_CHUNK_BYTES="${NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES:-4294967296}" # 4 GiB
NATIVE_MAX_ROW_GROUPS_PER_CHUNK="${NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK:-1024}"
NATIVE_MAX_RETRY_ATTEMPTS="${NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS:-3}"
NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS="${NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS:-1}"
NATIVE_MAX_SOURCE_READS_IN_FLIGHT="${NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT:-2}"
NATIVE_SOURCE_CHUNK_PLANNING_MODE="${NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE:-conservative}"
NATIVE_PIPELINE_EXECUTION_MODE="${NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE:-strict_bounded}"
NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE="${NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE:-datafusion_exact_host_welford}"

# ---------------------------------------------------------------------------
# DataFusion session  (single underscores; DataFusion SessionConfig::from_env)
# These are read by DataFusion's SessionConfig::from_env(), not by direct
# datafusion-nexus env parsing. DataFusion maps option keys by upper-casing and
# replacing "." with "_"; for example, `datafusion.execution.batch_size`
# becomes `DATAFUSION_EXECUTION_BATCH_SIZE`.
#
# DATAFUSION_EXECUTION_TARGET_PARTITIONS  CPU parallelism for fallback path (no effect on GPU)
# DATAFUSION_EXECUTION_BATCH_SIZE         Arrow record-batch size for DataFusion/fallback operators
# ---------------------------------------------------------------------------
DATAFUSION_EXECUTION_TARGET_PARTITIONS="${DATAFUSION_EXECUTION_TARGET_PARTITIONS:-16}"
DATAFUSION_EXECUTION_BATCH_SIZE="${DATAFUSION_EXECUTION_BATCH_SIZE:-8192}"

# ---------------------------------------------------------------------------
# Iceberg REST, AWS Glue, or local Glue-compatible catalog
#
# NEXUS_ICEBERG_CATALOG_KIND   Catalog backend: rest or glue
# NEXUS_ICEBERG_CATALOG_NAME   DataFusion catalog name (first segment of catalog.ns.table)
# NEXUS_ICEBERG_NAMESPACE      Iceberg namespace / Glue database name
# NEXUS_ICEBERG_WAREHOUSE      S3 warehouse root
# NEXUS_ICEBERG_TABLES         Pre-load list: "alias=table,table2"
#                                        Omit (empty) → all tables served dynamically
# NEXUS_ICEBERG_GLUE_CATALOG_ID  AWS Glue catalog account ID (omit = caller's account or local mock)
# NEXUS_ICEBERG_REST_URI      Local REST fixture endpoint when kind=rest
# NEXUS_ICEBERG_GLUE_URI      Local Moto endpoint when kind=glue
# NEXUS_ICEBERG_S3_*          RustFS/S3-compatible endpoint and credentials
#
# AWS_DEFAULT_REGION   REQUIRED for KvikIO S3 (KvikIO reads ONLY this var;
#                      AWS_REGION alone → silent unsigned requests → 403 on private buckets)
# AWS_REGION           Used by iceberg-rust catalog; set both.
#
# Real AWS credentials, when used, must be passed from the host environment.
# The defaults below target the public local lakehouse fixture namespace. Set
# before running against real AWS:
#   export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_SESSION_TOKEN=...
# ---------------------------------------------------------------------------
ICEBERG_CATALOG_KIND="${NEXUS_ICEBERG_CATALOG_KIND:-rest}"
ICEBERG_CATALOG_NAME="${NEXUS_ICEBERG_CATALOG_NAME:-lake}"
ICEBERG_NAMESPACE="${NEXUS_ICEBERG_NAMESPACE:-tpch_sf1}"
ICEBERG_WAREHOUSE="${NEXUS_ICEBERG_WAREHOUSE:-s3://lakehouse/warehouse}"
ICEBERG_TABLES="${NEXUS_ICEBERG_TABLES:-}"                 # empty = dynamic; e.g. "lineitem,orders"
ICEBERG_GLUE_CATALOG_ID="${NEXUS_ICEBERG_GLUE_CATALOG_ID:-}" # empty = use caller's own AWS account
ICEBERG_REST_URI="${NEXUS_ICEBERG_REST_URI:-http://localhost:8181}"
ICEBERG_GLUE_URI="${NEXUS_ICEBERG_GLUE_URI:-}"
ICEBERG_S3_ENDPOINT="${NEXUS_ICEBERG_S3_ENDPOINT:-http://localhost:9000}"
ICEBERG_S3_REGION="${NEXUS_ICEBERG_S3_REGION:-us-east-1}"
ICEBERG_S3_PATH_STYLE="${NEXUS_ICEBERG_S3_PATH_STYLE:-true}"
ICEBERG_S3_ACCESS_KEY_ID="${NEXUS_ICEBERG_S3_ACCESS_KEY_ID:-cudfadmin}"
ICEBERG_S3_SECRET_ACCESS_KEY="${NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY:-cudfadminsecret}"
ICEBERG_S3_SESSION_TOKEN="${NEXUS_ICEBERG_S3_SESSION_TOKEN:-}"

AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# ---------------------------------------------------------------------------
# cuGraph SQL DSL
#
# NEXUS_SERVER_CUGRAPH_ENABLED
#   true exposes the cugraph_* table functions plus metadata UDTFs. The image
#   must be built with cugraph support; set false to ship one artifact but keep
#   the SQL surface hidden at runtime. A lean image built without the cugraph
#   feature must use false.
#
# NEXUS_SERVER_CUGRAPH_DIRECTED
# NEXUS_SERVER_CUGRAPH_RENUMBER
# NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY
#   Optional defaults for graph construction. Policy values:
#   python_cugraph | raw_libcugraph
# ---------------------------------------------------------------------------
CUGRAPH_ENABLED="${NEXUS_SERVER_CUGRAPH_ENABLED:-true}"
CUGRAPH_DIRECTED="${NEXUS_SERVER_CUGRAPH_DIRECTED:-}"
CUGRAPH_RENUMBER="${NEXUS_SERVER_CUGRAPH_RENUMBER:-}"
CUGRAPH_CONSTRUCTION_POLICY="${NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY:-}"

# ---------------------------------------------------------------------------
# Build the docker run command
# ---------------------------------------------------------------------------
ARGS=(
    docker run --rm
    --gpus all
    --network host

    # Server core
    -e "NEXUS_SERVER_BIND=${BIND}"
    -e "NEXUS_SERVER_LOG=${LOG}"
    -e "NEXUS_SERVER_MAX_IN_FLIGHT=${MAX_IN_FLIGHT}"

    # Native GPU execution controls
    -e "NEXUS_SERVER_NATIVE_MEMORY_POLICY=${NATIVE_MEMORY_POLICY}"
    -e "NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE=${NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE}"

    # DataFusion session
    -e "DATAFUSION_EXECUTION_TARGET_PARTITIONS=${DATAFUSION_EXECUTION_TARGET_PARTITIONS}"
    -e "DATAFUSION_EXECUTION_BATCH_SIZE=${DATAFUSION_EXECUTION_BATCH_SIZE}"

    # Iceberg / local REST or Glue-compatible catalog
    -e "NEXUS_ICEBERG_CATALOG_KIND=${ICEBERG_CATALOG_KIND}"
    -e "NEXUS_ICEBERG_CATALOG_NAME=${ICEBERG_CATALOG_NAME}"
    -e "NEXUS_ICEBERG_NAMESPACE=${ICEBERG_NAMESPACE}"
    -e "NEXUS_ICEBERG_WAREHOUSE=${ICEBERG_WAREHOUSE}"
    -e "NEXUS_ICEBERG_S3_ENDPOINT=${ICEBERG_S3_ENDPOINT}"
    -e "NEXUS_ICEBERG_S3_REGION=${ICEBERG_S3_REGION}"
    -e "NEXUS_ICEBERG_S3_PATH_STYLE=${ICEBERG_S3_PATH_STYLE}"
    -e "NEXUS_ICEBERG_S3_ACCESS_KEY_ID=${ICEBERG_S3_ACCESS_KEY_ID}"
    -e "NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY=${ICEBERG_S3_SECRET_ACCESS_KEY}"
    -e "AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}"
    -e "AWS_REGION=${AWS_REGION}"

    # cuGraph SQL DSL
    -e "NEXUS_SERVER_CUGRAPH_ENABLED=${CUGRAPH_ENABLED}"
)

if [[ "${NATIVE_MEMORY_POLICY_NORMALIZED}" != "unbounded" ]]; then
    ARGS+=(
        -e "NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES=${NATIVE_DEVICE_BUDGET_BYTES}"
        -e "NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES=${NATIVE_MAX_SOURCE_CHUNK_BYTES}"
        -e "NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK=${NATIVE_MAX_ROW_GROUPS_PER_CHUNK}"
        -e "NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS=${NATIVE_MAX_RETRY_ATTEMPTS}"
        -e "NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS=${NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS}"
        -e "NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT=${NATIVE_MAX_SOURCE_READS_IN_FLIGHT}"
        -e "NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE=${NATIVE_SOURCE_CHUNK_PLANNING_MODE}"
        -e "NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE=${NATIVE_PIPELINE_EXECUTION_MODE}"
    )
fi

# Pre-load specific tables only when ICEBERG_TABLES is non-empty.
[[ -n "${ICEBERG_TABLES}" ]] && ARGS+=(-e "NEXUS_ICEBERG_TABLES=${ICEBERG_TABLES}")

# Glue catalog account ID only when explicitly set.
[[ -n "${ICEBERG_GLUE_CATALOG_ID}" ]] && ARGS+=(-e "NEXUS_ICEBERG_GLUE_CATALOG_ID=${ICEBERG_GLUE_CATALOG_ID}")
[[ -n "${ICEBERG_REST_URI}" ]] && ARGS+=(-e "NEXUS_ICEBERG_REST_URI=${ICEBERG_REST_URI}")
[[ -n "${ICEBERG_GLUE_URI}" ]] && ARGS+=(-e "NEXUS_ICEBERG_GLUE_URI=${ICEBERG_GLUE_URI}")
[[ -n "${ICEBERG_S3_SESSION_TOKEN}" ]] && ARGS+=(-e "NEXUS_ICEBERG_S3_SESSION_TOKEN=${ICEBERG_S3_SESSION_TOKEN}")

# Optional cuGraph defaults.
[[ -n "${CUGRAPH_DIRECTED}" ]] && ARGS+=(-e "NEXUS_SERVER_CUGRAPH_DIRECTED=${CUGRAPH_DIRECTED}")
[[ -n "${CUGRAPH_RENUMBER}" ]] && ARGS+=(-e "NEXUS_SERVER_CUGRAPH_RENUMBER=${CUGRAPH_RENUMBER}")
[[ -n "${CUGRAPH_CONSTRUCTION_POLICY}" ]] && \
    ARGS+=(-e "NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY=${CUGRAPH_CONSTRUCTION_POLICY}")

# AWS credentials — forward from host environment (short-lived STS tokens).
[[ -n "${AWS_ACCESS_KEY_ID:-}" ]]     && ARGS+=(-e "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}")
[[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] && ARGS+=(-e "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}")
[[ -n "${AWS_SESSION_TOKEN:-}" ]]     && ARGS+=(-e "AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}")
[[ -n "${AWS_PROFILE:-}" ]]           && ARGS+=(-e "AWS_PROFILE=${AWS_PROFILE}")

ARGS+=("${IMAGE}")

# ---------------------------------------------------------------------------
# Run or dry-run
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "docker run command (dry run):"
    printf '  %s\n' "${ARGS[@]}"
else
    echo "Starting ${IMAGE} ..."
    exec "${ARGS[@]}"
fi
