#!/usr/bin/env bash
# Run datafusion_cudf_server directly (no Docker).
#
# Usage:
#   bash scripts/run_server.sh             # start the server
#   bash scripts/run_server.sh --dry-run   # print the env + command without executing
#
# The release binary must already be built:
#   cargo build --release --features server,iceberg
#
# AWS credentials (short-lived STS tokens) must be in the host environment:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_SESSION_TOKEN=...
# then run this script.

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${REPO_ROOT}/target/release/datafusion_cudf_server"

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found at ${BINARY}" >&2
    echo "       Run: cargo build --release --features server,iceberg" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Server core
# DATAFUSION_CUDF_SERVER_BIND              host:port to listen on (default: 0.0.0.0:50051)
# DATAFUSION_CUDF_SERVER_LOG               tracing filter — "info", "debug", "warn,datafusion=info"
# DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT     max concurrent in-flight Flight SQL requests (default: 1)
# DATAFUSION_CUDF_SERVER_SESSION_PROFILE   "native" or "cpu_fallback" (default: native when
#                                          Iceberg is configured, cpu_fallback otherwise).
#                                          The native profile pins
#                                          target_partitions=1 and disables DataFusion
#                                          repartitioning + dynamic-filter pushdown so plans
#                                          match the native-lowerable shapes the cudf
#                                          optimizer rule expects. Required for Iceberg
#                                          tables that use the GPU-only
#                                          CudfTableFormatScanExec leaf.
# ---------------------------------------------------------------------------
export DATAFUSION_CUDF_SERVER_BIND="0.0.0.0:50051"
export DATAFUSION_CUDF_SERVER_LOG="info"
export DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT=1
# Iceberg presence already selects the native profile by default; leaving this
# unset lets the server pick the right profile from the catalog wiring.
# export DATAFUSION_CUDF_SERVER_SESSION_PROFILE="native"

# ---------------------------------------------------------------------------
# Native GPU memory budget  (all three required together; omit all → unbounded)
#
# DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES
#   Hard cap on live GPU memory reservations.
#   = nvidia-smi memory.total - 500 MiB overhead
#   RTX PRO 6000 Blackwell (97887 MiB): 97387 MiB = 102117670912 bytes
#
# DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES
#   Max bytes read per Parquet source chunk (4 GiB = rg1024/32GiB profile).
#
# DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK
#   Max Parquet row groups per source chunk (rg1024 profile).
# ---------------------------------------------------------------------------
export DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES=102117670912
export DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES=4294967296
export DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK=1024

# ---------------------------------------------------------------------------
# DataFusion session
#
# SessionConfig::from_env() maps each option key to an env var by upper-casing
# and replacing "." with "_", so `datafusion.execution.target_partitions`
# becomes `DATAFUSION_EXECUTION_TARGET_PARTITIONS` (single underscores, NOT
# double).
#
# DATAFUSION_EXECUTION_TARGET_PARTITIONS  CPU parallelism for fallback path.
#                                         IGNORED under the native session
#                                         profile (pinned to 1); only matters
#                                         when DATAFUSION_CUDF_SERVER_SESSION_PROFILE=cpu_fallback.
# DATAFUSION_EXECUTION_BATCH_SIZE         Arrow record-batch size used by both
#                                         profiles.
# ---------------------------------------------------------------------------
# The default Iceberg/native profile pins target_partitions=1 to keep DataFusion
# from inserting CPU-style RepartitionExec / SortPreservingMergeExec nodes over
# the GPU-only CudfTableFormatScanExec leaf. Only override this when explicitly
# running under DATAFUSION_CUDF_SERVER_SESSION_PROFILE=cpu_fallback.
# export DATAFUSION_EXECUTION_TARGET_PARTITIONS=16
export DATAFUSION_EXECUTION_BATCH_SIZE=8192

# ---------------------------------------------------------------------------
# Iceberg / AWS Glue
#
# DATAFUSION_CUDF_ICEBERG_CATALOG_NAME   DataFusion catalog name (first segment of catalog.ns.table)
# DATAFUSION_CUDF_ICEBERG_NAMESPACE      Iceberg namespace / Glue database name; dynamic catalog scope
# DATAFUSION_CUDF_ICEBERG_WAREHOUSE      S3 warehouse root (required)
# DATAFUSION_CUDF_ICEBERG_TABLES         Pre-load list: "alias=table,table2"
#                                        Empty → all tables served dynamically
# DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID  AWS account ID (empty = caller's own account)
#
# AWS_DEFAULT_REGION   REQUIRED for KvikIO S3 (reads ONLY this var;
#                      AWS_REGION alone → unsigned requests → 403 on private buckets)
# AWS_REGION           Used by iceberg-rust catalog; set both.
#
# AWS credentials are short-lived STS tokens — pass from the host environment,
# never hardcode here.
# ---------------------------------------------------------------------------
export DATAFUSION_CUDF_ICEBERG_CATALOG_NAME="glue"
export DATAFUSION_CUDF_ICEBERG_NAMESPACE="demo_db"
export DATAFUSION_CUDF_ICEBERG_WAREHOUSE="s3://example-warehouse/demo_db/"
# export DATAFUSION_CUDF_ICEBERG_TABLES="lineitem=tpch_sf1_lineitem,orders=tpch_sf1_orders"
# export DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID="123456789012"

export AWS_DEFAULT_REGION="us-east-1"
export AWS_REGION="us-east-1"

# AWS credentials forwarded from host environment (short-lived STS tokens).
# Nothing to export here — they are already in the environment when set via
# `export AWS_ACCESS_KEY_ID=...` before calling this script.

# ---------------------------------------------------------------------------
# Run or dry-run
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Binary: ${BINARY}"
    echo ""
    echo "Environment:"
    env | grep -E '^(DATAFUSION|CUDF|AWS)' | sort
else
    echo "Starting datafusion_cudf_server ..."
    exec "${BINARY}"
fi
