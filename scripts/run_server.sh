#!/usr/bin/env bash
# Run datafusion_nexus_server directly (no Docker).
#
# Usage:
#   bash scripts/run_server.sh             # start the server
#   bash scripts/run_server.sh --dry-run   # print the env + command without executing
#
# The release binary must already be built. For the Iceberg+cuGraph production
# path this script configures, use:
#   cargo build --release --features server,iceberg,cugraph --bin datafusion_nexus_server
# or, without Iceberg:
#   cargo build --release --features server-cugraph --bin datafusion_nexus_server
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
TARGET_ROOT="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}"
BINARY="${TARGET_ROOT}/release/datafusion_nexus_server"

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found at ${BINARY}" >&2
    echo "       Run: cargo build --release --features server,iceberg,cugraph --bin datafusion_nexus_server" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Server core
# DATAFUSION_NEXUS_SERVER_BIND              host:port to listen on (default: 0.0.0.0:50051)
# DATAFUSION_NEXUS_SERVER_LOG               tracing filter — "info", "debug", "warn,datafusion=info"
# DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT     max concurrent in-flight Flight SQL requests (default: 1)
# DATAFUSION_NEXUS_SERVER_SESSION_PROFILE   "native" or "cpu_fallback" (default: native when
#                                          Iceberg is configured, cpu_fallback otherwise).
#                                          The native profile pins
#                                          target_partitions=1 and disables DataFusion
#                                          repartitioning + dynamic-filter pushdown so plans
#                                          match the native-lowerable shapes the cudf
#                                          optimizer rule expects. Required for Iceberg
#                                          tables that use the GPU-only
#                                          CudfTableFormatScanExec leaf.
#                                          NOTE: cpu_fallback is **testing-only** while
#                                          Iceberg is configured — the GPU-only leaf has
#                                          no CPU Arrow reader yet (tracked as Milestone 6 of
#                                          docs/plan/2026-05-22-flight-sql-iceberg-cpu-fallback-plan.md),
#                                          so plain ORDER BY queries hit the Milestone 1 leaf
#                                          guard and fail in planning. The server emits a
#                                          startup WARN when this combination is selected.
# ---------------------------------------------------------------------------
export DATAFUSION_NEXUS_SERVER_BIND="${DATAFUSION_NEXUS_SERVER_BIND:-0.0.0.0:50051}"
export DATAFUSION_NEXUS_SERVER_LOG="${DATAFUSION_NEXUS_SERVER_LOG:-info}"
export DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT="${DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT:-1}"
# Iceberg presence already selects the native profile by default; leaving this
# unset lets the server pick the right profile from the catalog wiring.
# export DATAFUSION_NEXUS_SERVER_SESSION_PROFILE="native"

# ---------------------------------------------------------------------------
# cuGraph SQL DSL
#
# DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED              true | false
# DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS           optional allowlist by SQL
#                                                     function name, e.g.
#                                                     cugraph_pagerank,cugraph_bfs
# DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED             default graph direction
# DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER             default cuGraph renumber flag
# DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY  python_cugraph | raw_libcugraph
#
# The binary must be built with the `cugraph` feature. The env gate remains
# explicit so operators can ship one artifact and choose whether the SQL UDTFs
# are exposed at runtime.
# ---------------------------------------------------------------------------
export DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED="${DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED:-true}"
# export DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS="cugraph_pagerank,cugraph_bfs"
# export DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED=true
# export DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER=true
# export DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY=python_cugraph

# ---------------------------------------------------------------------------
# Native GPU memory budget
#
# DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY
#   bounded | unbounded (default: bounded). Use unbounded only for explicitly
#   labelled graph-result smoke tests while the cuGraph bounded memory contract
#   is still missing.
#
# DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES
#   Hard cap on live GPU memory reservations.
#   = nvidia-smi memory.total - 500 MiB overhead
#   RTX PRO 6000 Blackwell (97887 MiB): 97387 MiB = 102117670912 bytes
#
# DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES
#   Max bytes read per Parquet source chunk (4 GiB = rg1024/32GiB profile).
#
# DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK
#   Max Parquet row groups per source chunk (rg1024 profile).
# ---------------------------------------------------------------------------
export DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY="${DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY:-bounded}"
if [[ "${DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY}" != "unbounded" ]]; then
    export DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES="${DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES:-102117670912}"
    export DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES="${DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES:-4294967296}"
    export DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK="${DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK:-1024}"
fi

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
#                                         when DATAFUSION_NEXUS_SERVER_SESSION_PROFILE=cpu_fallback.
# DATAFUSION_EXECUTION_BATCH_SIZE         Arrow record-batch size used by both
#                                         profiles.
# ---------------------------------------------------------------------------
# The default Iceberg/native profile pins target_partitions=1 to keep DataFusion
# from inserting CPU-style RepartitionExec / SortPreservingMergeExec nodes over
# the GPU-only CudfTableFormatScanExec leaf. Only override this when explicitly
# running under DATAFUSION_NEXUS_SERVER_SESSION_PROFILE=cpu_fallback.
# export DATAFUSION_EXECUTION_TARGET_PARTITIONS=16
export DATAFUSION_EXECUTION_BATCH_SIZE="${DATAFUSION_EXECUTION_BATCH_SIZE:-8192}"

# ---------------------------------------------------------------------------
# Iceberg / AWS Glue
#
# DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME   DataFusion catalog name (first segment of catalog.ns.table)
# DATAFUSION_NEXUS_ICEBERG_NAMESPACE      Iceberg namespace / Glue database name; dynamic catalog scope
# DATAFUSION_NEXUS_ICEBERG_WAREHOUSE      S3 warehouse root (required)
# DATAFUSION_NEXUS_ICEBERG_TABLES         Pre-load list: "alias=table,table2"
#                                        Empty → all tables served dynamically
# DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_ID  AWS account ID (empty = caller's own account)
#
# AWS_DEFAULT_REGION   REQUIRED for KvikIO S3 (reads ONLY this var;
#                      AWS_REGION alone → unsigned requests → 403 on private buckets)
# AWS_REGION           Used by iceberg-rust catalog; set both.
#
# AWS credentials are short-lived STS tokens — pass from the host environment,
# never hardcode here.
# ---------------------------------------------------------------------------
if [[ "${DATAFUSION_NEXUS_RUN_SERVER_ICEBERG:-1}" != "0" ]]; then
    export DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME="${DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME:-glue}"
    export DATAFUSION_NEXUS_ICEBERG_NAMESPACE="${DATAFUSION_NEXUS_ICEBERG_NAMESPACE:-demo_db}"
    export DATAFUSION_NEXUS_ICEBERG_WAREHOUSE="${DATAFUSION_NEXUS_ICEBERG_WAREHOUSE:-s3://example-warehouse/demo_db/}"
    # export DATAFUSION_NEXUS_ICEBERG_TABLES="lineitem=tpch_sf1_lineitem,orders=tpch_sf1_orders"
    # export DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_ID="123456789012"

    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
    export AWS_REGION="${AWS_REGION:-us-east-1}"
fi

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
    echo "Starting datafusion_nexus_server ..."
    exec "${BINARY}"
fi
