#!/usr/bin/env bash
# Run datafusion_nexus_server directly (no Docker).
#
# Usage:
#   bash scripts/run_server.sh             # start the server
#   bash scripts/run_server.sh --dry-run   # print the env + command without executing
#
# The release binary must already be built. Either artifact flavor works:
#   cargo build --release --features server,iceberg --bin datafusion_nexus_server
#   cargo build --release --features server,iceberg,cugraph --bin datafusion_nexus_server
# The cuGraph SQL UDTFs stay disabled unless NEXUS_SERVER_CUGRAPH_ENABLED=true
# is exported before calling this script (requires the cugraph build).
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
    echo "       Run: cargo build --release --features server,iceberg --bin datafusion_nexus_server" >&2
    echo "       (add ,cugraph to the feature list for the cuGraph SQL UDTFs)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Server core
# NEXUS_SERVER_BIND              host:port to listen on (default: 0.0.0.0:50051)
# NEXUS_SERVER_LOG               tracing filter — "info", "debug", "warn,datafusion=info"
# NEXUS_SERVER_MAX_IN_FLIGHT     max concurrent in-flight Flight SQL requests (default: 1)
# NEXUS_SERVER_SESSION_PROFILE   "native" or "cpu_fallback" (default: native when
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
export NEXUS_SERVER_BIND="${NEXUS_SERVER_BIND:-0.0.0.0:50051}"
export NEXUS_SERVER_LOG="${NEXUS_SERVER_LOG:-info}"
export NEXUS_SERVER_MAX_IN_FLIGHT="${NEXUS_SERVER_MAX_IN_FLIGHT:-1}"
# Iceberg presence already selects the native profile by default; leaving this
# unset lets the server pick the right profile from the catalog wiring.
# export NEXUS_SERVER_SESSION_PROFILE="native"

# ---------------------------------------------------------------------------
# cuGraph SQL DSL
#
# NEXUS_SERVER_CUGRAPH_ENABLED              true | false (unset = disabled)
# NEXUS_SERVER_CUGRAPH_ALGORITHMS           optional allowlist by SQL
#                                                     function name, e.g.
#                                                     cugraph_pagerank,cugraph_bfs
# NEXUS_SERVER_CUGRAPH_DIRECTED             default graph direction
# NEXUS_SERVER_CUGRAPH_RENUMBER             default cuGraph renumber flag
# NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY  python_cugraph | raw_libcugraph
#
# No default is exported here: unset is valid for every artifact flavor (a
# cugraph build keeps the SQL UDTFs disabled; a no-cugraph build rejects any
# cuGraph env except an explicit `false`). To expose the UDTFs, export
# NEXUS_SERVER_CUGRAPH_ENABLED=true before calling this script — that requires
# a binary built with the `cugraph` feature.
# ---------------------------------------------------------------------------
# export NEXUS_SERVER_CUGRAPH_ENABLED=true
# export NEXUS_SERVER_CUGRAPH_ALGORITHMS="cugraph_pagerank,cugraph_bfs"
# export NEXUS_SERVER_CUGRAPH_DIRECTED=true
# export NEXUS_SERVER_CUGRAPH_RENUMBER=true
# export NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY=python_cugraph

# ---------------------------------------------------------------------------
# Native GPU execution controls
#
# NEXUS_SERVER_NATIVE_MEMORY_POLICY
#   bounded | unbounded (default: bounded). Use unbounded only for explicitly
#   labelled graph-result smoke tests while the cuGraph bounded memory contract
#   is still missing.
#
# NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES
#   Hard cap on live GPU memory reservations. Leave unset with the two chunk
#   knobs below to use the server's cudf-rs inventory based auto-sized bounded
#   profile.
#
# NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES
#   Max bytes read per Parquet source chunk (4 GiB = rg1024/32GiB profile).
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
export NEXUS_SERVER_NATIVE_MEMORY_POLICY="${NEXUS_SERVER_NATIVE_MEMORY_POLICY:-bounded}"
NEXUS_SERVER_NATIVE_MEMORY_POLICY_NORMALIZED="${NEXUS_SERVER_NATIVE_MEMORY_POLICY,,}"
if [[ "${NEXUS_SERVER_NATIVE_MEMORY_POLICY_NORMALIZED}" != "unbounded" ]]; then
    export NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS="${NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS:-3}"
    export NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS="${NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS:-1}"
    export NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT="${NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT:-2}"
    export NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE="${NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE:-conservative}"
    export NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE="${NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE:-strict_bounded}"
fi
export NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE="${NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE:-datafusion_exact_host_welford}"

# ---------------------------------------------------------------------------
# DataFusion session
#
# SessionConfig::from_env() maps each option key to an env var by upper-casing
# and replacing "." with "_", so `datafusion.execution.target_partitions`
# becomes `DATAFUSION_EXECUTION_TARGET_PARTITIONS` (single underscores, NOT
# double).
# These variables are consumed by DataFusion itself; datafusion-nexus only
# starts from the resulting SessionConfig and then applies the server profile.
#
# DATAFUSION_EXECUTION_TARGET_PARTITIONS  CPU parallelism for fallback path.
#                                         IGNORED under the native session
#                                         profile (pinned to 1); only matters
#                                         when NEXUS_SERVER_SESSION_PROFILE=cpu_fallback.
# DATAFUSION_EXECUTION_BATCH_SIZE         Arrow record-batch size for
#                                         DataFusion/fallback operators. Native
#                                         cudf source chunking is controlled by
#                                         NEXUS_SERVER_NATIVE_* budget knobs.
# ---------------------------------------------------------------------------
# The default Iceberg/native profile pins target_partitions=1 to keep DataFusion
# from inserting CPU-style RepartitionExec / SortPreservingMergeExec nodes over
# the GPU-only CudfTableFormatScanExec leaf. Only override this when explicitly
# running under NEXUS_SERVER_SESSION_PROFILE=cpu_fallback.
# export DATAFUSION_EXECUTION_TARGET_PARTITIONS=16
export DATAFUSION_EXECUTION_BATCH_SIZE="${DATAFUSION_EXECUTION_BATCH_SIZE:-8192}"

# ---------------------------------------------------------------------------
# Iceberg / AWS Glue or local Glue-compatible catalog
#
# NEXUS_ICEBERG_CATALOG_KIND   Catalog backend: rest or glue
# NEXUS_ICEBERG_CATALOG_NAME   DataFusion catalog name (first segment of catalog.ns.table)
# NEXUS_ICEBERG_NAMESPACE      Iceberg namespace / Glue database name; dynamic catalog scope
# NEXUS_ICEBERG_WAREHOUSE      S3 warehouse root
# NEXUS_ICEBERG_TABLES         Pre-load list: "alias=table,table2"
#                                        Empty → all tables served dynamically
# NEXUS_ICEBERG_GLUE_CATALOG_ID  AWS account ID (empty = caller's own account or local mock)
# NEXUS_ICEBERG_REST_URI      Local REST fixture endpoint when kind=rest
# NEXUS_ICEBERG_GLUE_URI      Local Moto endpoint when kind=glue
# NEXUS_ICEBERG_S3_*          RustFS/S3-compatible endpoint and credentials
#
# AWS_DEFAULT_REGION   REQUIRED for KvikIO S3 (reads ONLY this var;
#                      AWS_REGION alone → unsigned requests → 403 on private buckets)
# AWS_REGION           Used by iceberg-rust catalog; set both.
#
# Real AWS credentials, when used, must be passed from the host environment.
# The defaults below target the public local lakehouse fixture namespace.
# ---------------------------------------------------------------------------
if [[ "${NEXUS_RUN_SERVER_ICEBERG:-1}" != "0" ]]; then
    export NEXUS_ICEBERG_CATALOG_KIND="${NEXUS_ICEBERG_CATALOG_KIND:-rest}"
    export NEXUS_ICEBERG_CATALOG_NAME="${NEXUS_ICEBERG_CATALOG_NAME:-lake}"
    export NEXUS_ICEBERG_NAMESPACE="${NEXUS_ICEBERG_NAMESPACE:-tpch_sf1}"
    export NEXUS_ICEBERG_WAREHOUSE="${NEXUS_ICEBERG_WAREHOUSE:-s3://lakehouse/warehouse}"
    export NEXUS_ICEBERG_REST_URI="${NEXUS_ICEBERG_REST_URI:-http://localhost:8181}"
    export NEXUS_ICEBERG_S3_ENDPOINT="${NEXUS_ICEBERG_S3_ENDPOINT:-http://localhost:9000}"
    export NEXUS_ICEBERG_S3_REGION="${NEXUS_ICEBERG_S3_REGION:-us-east-1}"
    export NEXUS_ICEBERG_S3_PATH_STYLE="${NEXUS_ICEBERG_S3_PATH_STYLE:-true}"
    export NEXUS_ICEBERG_S3_ACCESS_KEY_ID="${NEXUS_ICEBERG_S3_ACCESS_KEY_ID:-cudfadmin}"
    export NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY="${NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY:-cudfadminsecret}"
    # export NEXUS_ICEBERG_TABLES="lineitem,orders"
    # export NEXUS_ICEBERG_GLUE_CATALOG_ID=""

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
    env | grep -E '^(DATAFUSION|CUDF|AWS|NEXUS)' | sort
else
    echo "Starting datafusion_nexus_server ..."
    exec "${BINARY}"
fi
