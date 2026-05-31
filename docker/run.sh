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
# DATAFUSION_NEXUS_SERVER_BIND          host:port to listen on (default: 0.0.0.0:50051)
# DATAFUSION_NEXUS_SERVER_LOG           tracing filter — "info", "debug", "warn,datafusion=info"
# DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT max concurrent in-flight Flight SQL requests (default: 1)
# ---------------------------------------------------------------------------
BIND="0.0.0.0:50051"
LOG="info"
MAX_IN_FLIGHT=1

# ---------------------------------------------------------------------------
# Native GPU memory budget  (all three required together; omit all → unbounded)
#
# DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES
#   Hard cap on live GPU memory reservations.
#   = nvidia-smi memory.total - 500 MiB overhead
#   RTX PRO 6000 Blackwell (97887 MiB): 97387 MiB = 102117670912 bytes
#
# DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES
#   Max bytes read per Parquet source chunk.
#   4 GiB (rg1024/32GiB profile) for decision-grade SF10+ runs.
#
# DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK
#   Max Parquet row groups per source chunk (rg1024 profile).
# ---------------------------------------------------------------------------
NATIVE_DEVICE_BUDGET_BYTES=102117670912   # 97387 MiB  (97887 - 500 MiB overhead)
NATIVE_MAX_SOURCE_CHUNK_BYTES=4294967296  # 4 GiB
NATIVE_MAX_ROW_GROUPS_PER_CHUNK=1024

# ---------------------------------------------------------------------------
# DataFusion session  (DATAFUSION__ prefix, double-underscore separator)
# These are read by DataFusion's SessionConfig::from_env().
#
# DATAFUSION__EXECUTION__TARGET_PARTITIONS  CPU parallelism for fallback path (no effect on GPU)
# DATAFUSION__EXECUTION__BATCH_SIZE         Arrow record-batch size for fallback path
# ---------------------------------------------------------------------------
DATAFUSION__EXECUTION__TARGET_PARTITIONS=16
DATAFUSION__EXECUTION__BATCH_SIZE=8192

# ---------------------------------------------------------------------------
# Iceberg / AWS Glue
#
# DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME   DataFusion catalog name (first segment of catalog.ns.table)
# DATAFUSION_NEXUS_ICEBERG_NAMESPACE      Iceberg namespace / Glue database name
# DATAFUSION_NEXUS_ICEBERG_WAREHOUSE      S3 warehouse root (required)
# DATAFUSION_NEXUS_ICEBERG_TABLES         Pre-load list: "alias=table,table2"
#                                        Omit (empty) → all tables served dynamically
# DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_ID  AWS Glue catalog account ID (omit = caller's account)
#
# AWS_DEFAULT_REGION   REQUIRED for KvikIO S3 (KvikIO reads ONLY this var;
#                      AWS_REGION alone → silent unsigned requests → 403 on private buckets)
# AWS_REGION           Used by iceberg-rust catalog; set both.
#
# AWS credentials are short-lived STS tokens — pass from the host environment,
# never hardcode here.  Set before running:
#   export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_SESSION_TOKEN=...
# ---------------------------------------------------------------------------
ICEBERG_CATALOG_NAME="glue"
ICEBERG_NAMESPACE="demo_db"
ICEBERG_WAREHOUSE="s3://example-warehouse/demo_db/"
ICEBERG_TABLES=""          # empty = dynamic; e.g. "lineitem=tpch_sf1_lineitem,orders=tpch_sf1_orders"
ICEBERG_GLUE_CATALOG_ID="" # empty = use caller's own AWS account

AWS_DEFAULT_REGION="us-east-1"
AWS_REGION="us-east-1"

# ---------------------------------------------------------------------------
# cuGraph SQL DSL
#
# DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED
#   true exposes the cugraph_* table functions plus metadata UDTFs. The image
#   must be built with cugraph support; set false to ship one artifact but keep
#   the SQL surface hidden at runtime.
#
# DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS
#   Optional allowlist by SQL function name, e.g. cugraph_pagerank,cugraph_bfs.
#
# DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED
# DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER
# DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY
#   Optional defaults for graph construction. Policy values:
#   python_cugraph | raw_libcugraph
# ---------------------------------------------------------------------------
CUGRAPH_ENABLED="${DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED:-true}"
CUGRAPH_ALGORITHMS="${DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS:-}"
CUGRAPH_DIRECTED="${DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED:-}"
CUGRAPH_RENUMBER="${DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER:-}"
CUGRAPH_CONSTRUCTION_POLICY="${DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY:-}"

# ---------------------------------------------------------------------------
# Build the docker run command
# ---------------------------------------------------------------------------
ARGS=(
    docker run --rm
    --gpus all
    --network host

    # Server core
    -e "DATAFUSION_NEXUS_SERVER_BIND=${BIND}"
    -e "DATAFUSION_NEXUS_SERVER_LOG=${LOG}"
    -e "DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT=${MAX_IN_FLIGHT}"

    # Native GPU memory budget
    -e "DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES=${NATIVE_DEVICE_BUDGET_BYTES}"
    -e "DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES=${NATIVE_MAX_SOURCE_CHUNK_BYTES}"
    -e "DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK=${NATIVE_MAX_ROW_GROUPS_PER_CHUNK}"

    # DataFusion session
    -e "DATAFUSION__EXECUTION__TARGET_PARTITIONS=${DATAFUSION__EXECUTION__TARGET_PARTITIONS}"
    -e "DATAFUSION__EXECUTION__BATCH_SIZE=${DATAFUSION__EXECUTION__BATCH_SIZE}"

    # Iceberg / Glue
    -e "DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME=${ICEBERG_CATALOG_NAME}"
    -e "DATAFUSION_NEXUS_ICEBERG_NAMESPACE=${ICEBERG_NAMESPACE}"
    -e "DATAFUSION_NEXUS_ICEBERG_WAREHOUSE=${ICEBERG_WAREHOUSE}"
    -e "AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}"
    -e "AWS_REGION=${AWS_REGION}"

    # cuGraph SQL DSL
    -e "DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED=${CUGRAPH_ENABLED}"
)

# Pre-load specific tables only when ICEBERG_TABLES is non-empty.
[[ -n "${ICEBERG_TABLES}" ]] && ARGS+=(-e "DATAFUSION_NEXUS_ICEBERG_TABLES=${ICEBERG_TABLES}")

# Glue catalog account ID only when explicitly set.
[[ -n "${ICEBERG_GLUE_CATALOG_ID}" ]] && ARGS+=(-e "DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_ID=${ICEBERG_GLUE_CATALOG_ID}")

# Optional cuGraph defaults.
[[ -n "${CUGRAPH_ALGORITHMS}" ]] && ARGS+=(-e "DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS=${CUGRAPH_ALGORITHMS}")
[[ -n "${CUGRAPH_DIRECTED}" ]] && ARGS+=(-e "DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED=${CUGRAPH_DIRECTED}")
[[ -n "${CUGRAPH_RENUMBER}" ]] && ARGS+=(-e "DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER=${CUGRAPH_RENUMBER}")
[[ -n "${CUGRAPH_CONSTRUCTION_POLICY}" ]] && \
    ARGS+=(-e "DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY=${CUGRAPH_CONSTRUCTION_POLICY}")

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
