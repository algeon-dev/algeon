#!/usr/bin/env bash
# Run datafusion-nexus-server in Docker.
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

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
IMAGE="datafusion-nexus-server:latest"

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
# Native GPU memory budget  (all three required together; omit all → unbounded)
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
# ---------------------------------------------------------------------------
NATIVE_DEVICE_BUDGET_BYTES=102117670912   # 97387 MiB  (97887 - 500 MiB overhead)
NATIVE_MAX_SOURCE_CHUNK_BYTES=4294967296  # 4 GiB
NATIVE_MAX_ROW_GROUPS_PER_CHUNK=1024

# ---------------------------------------------------------------------------
# DataFusion session  (single underscores; DataFusion SessionConfig::from_env)
# These are read by DataFusion's SessionConfig::from_env().
#
# DATAFUSION_EXECUTION_TARGET_PARTITIONS  CPU parallelism for fallback path (no effect on GPU)
# DATAFUSION_EXECUTION_BATCH_SIZE         Arrow record-batch size for fallback path
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
#   the SQL surface hidden at runtime.
#
# NEXUS_SERVER_CUGRAPH_ALGORITHMS
#   Optional allowlist by SQL function name, e.g. cugraph_pagerank,cugraph_bfs.
#
# NEXUS_SERVER_CUGRAPH_DIRECTED
# NEXUS_SERVER_CUGRAPH_RENUMBER
# NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY
#   Optional defaults for graph construction. Policy values:
#   python_cugraph | raw_libcugraph
# ---------------------------------------------------------------------------
CUGRAPH_ENABLED="${NEXUS_SERVER_CUGRAPH_ENABLED:-true}"
CUGRAPH_ALGORITHMS="${NEXUS_SERVER_CUGRAPH_ALGORITHMS:-}"
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

    # Native GPU memory budget
    -e "NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES=${NATIVE_DEVICE_BUDGET_BYTES}"
    -e "NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES=${NATIVE_MAX_SOURCE_CHUNK_BYTES}"
    -e "NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK=${NATIVE_MAX_ROW_GROUPS_PER_CHUNK}"

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

# Pre-load specific tables only when ICEBERG_TABLES is non-empty.
[[ -n "${ICEBERG_TABLES}" ]] && ARGS+=(-e "NEXUS_ICEBERG_TABLES=${ICEBERG_TABLES}")

# Glue catalog account ID only when explicitly set.
[[ -n "${ICEBERG_GLUE_CATALOG_ID}" ]] && ARGS+=(-e "NEXUS_ICEBERG_GLUE_CATALOG_ID=${ICEBERG_GLUE_CATALOG_ID}")
[[ -n "${ICEBERG_REST_URI}" ]] && ARGS+=(-e "NEXUS_ICEBERG_REST_URI=${ICEBERG_REST_URI}")
[[ -n "${ICEBERG_GLUE_URI}" ]] && ARGS+=(-e "NEXUS_ICEBERG_GLUE_URI=${ICEBERG_GLUE_URI}")
[[ -n "${ICEBERG_S3_SESSION_TOKEN}" ]] && ARGS+=(-e "NEXUS_ICEBERG_S3_SESSION_TOKEN=${ICEBERG_S3_SESSION_TOKEN}")

# Optional cuGraph defaults.
[[ -n "${CUGRAPH_ALGORITHMS}" ]] && ARGS+=(-e "NEXUS_SERVER_CUGRAPH_ALGORITHMS=${CUGRAPH_ALGORITHMS}")
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
