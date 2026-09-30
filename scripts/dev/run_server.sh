#!/usr/bin/env bash
# Run algeon_server directly (no Docker).
#
# Usage:
#   bash scripts/dev/run_server.sh             # start the server
#   bash scripts/dev/run_server.sh --dry-run   # print the env + command without executing
#
# The release binary must already be built:
#   cargo build --workspace --all-features --release --bin algeon_server
# The cuGraph SQL frontend is enabled by default. Set
# ALGEON_SERVER_CUGRAPH_ENABLED=false to hide it; execution requires the
# cugraph build. cuVS and cuML SQL functions are installed by the server;
# execution requires the corresponding `cuvs` or `cuml` build feature.
#
# AWS credentials (short-lived STS tokens) must be in the host environment:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_SESSION_TOKEN=...
# then run this script.

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_ROOT="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}"
BINARY="${TARGET_ROOT}/release/algeon_server"

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found at ${BINARY}" >&2
    echo "       Run: cargo build --workspace --all-features --release --bin algeon_server" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Server core
# ALGEON_SERVER_BIND              host:port to listen on (default: 0.0.0.0:50051)
# ALGEON_SERVER_LOG               tracing filter: "info", "debug", "warn,datafusion=info"
# ---------------------------------------------------------------------------
export ALGEON_SERVER_BIND="${ALGEON_SERVER_BIND:-0.0.0.0:50051}"
export ALGEON_SERVER_LOG="${ALGEON_SERVER_LOG:-info}"

# ---------------------------------------------------------------------------
# cuGraph SQL DSL
#
# ALGEON_SERVER_CUGRAPH_ENABLED              true | false (unset = enabled)
# ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY  python_cugraph | raw_libcugraph
#
# No default is exported here: the server configuration defaults to enabled.
# Set ALGEON_SERVER_CUGRAPH_ENABLED=false before calling this script to hide the
# frontend.
# ---------------------------------------------------------------------------
# export ALGEON_SERVER_CUGRAPH_ENABLED=true
# export ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY=python_cugraph

# ---------------------------------------------------------------------------
# Native GPU execution controls
#
# ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES
#   Max bytes read per Parquet source chunk (4 GiB = rg1024/32GiB profile).
#
# ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK
#   Max Parquet row groups per source chunk (rg1024 profile).
#
# ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS
#   Bounded retry count.
#
# ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT
#   Bounded source scheduling / prefetch width.
#
# ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE
#   conservative | local_parquet_throughput
#
# ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE
#   gpu_native_tolerant (default) | datafusion_exact_host_welford
# ---------------------------------------------------------------------------
export ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS="${ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS:-3}"
export ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT="${ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT:-2}"
export ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE="${ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE:-conservative}"
export ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE="${ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE:-gpu_native_tolerant}"

# ---------------------------------------------------------------------------
# DataFusion session
#
# SessionConfig::from_env() maps each option key to an env var by upper-casing
# and replacing "." with "_", so `datafusion.execution.target_partitions`
# becomes `DATAFUSION_EXECUTION_TARGET_PARTITIONS` (single underscores, NOT
# double).
# These variables are consumed by DataFusion itself; algeon-datafusion only
# enables catalog discovery on the resulting SessionConfig.
#
# DATAFUSION_EXECUTION_TARGET_PARTITIONS  CPU parallelism and physical-plan
#                                         partitioning for the server session.
# DATAFUSION_EXECUTION_BATCH_SIZE         Arrow record-batch size for
#                                         DataFusion/fallback operators. Native
#                                         cudf source chunking is controlled by
#                                         ALGEON_SERVER_NATIVE_* budget knobs.
# ---------------------------------------------------------------------------
# export DATAFUSION_EXECUTION_TARGET_PARTITIONS=16
export DATAFUSION_EXECUTION_BATCH_SIZE="${DATAFUSION_EXECUTION_BATCH_SIZE:-8192}"

# ---------------------------------------------------------------------------
# S3 region
#
# AWS_DEFAULT_REGION   REQUIRED for KvikIO S3 (reads ONLY this var;
#                      AWS_REGION alone → unsigned requests → 403 on private buckets)
# AWS_REGION           Read by the query engine for S3 source metadata; set both.
#
# Real AWS credentials, when used, must be passed from the host environment.
# ---------------------------------------------------------------------------
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_REGION="${AWS_REGION:-us-east-1}"

# AWS credentials forwarded from host environment (short-lived STS tokens).
# Nothing to export here: they are already in the environment when set via
# `export AWS_ACCESS_KEY_ID=...` before calling this script.

# ---------------------------------------------------------------------------
# Run or dry-run
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Binary: ${BINARY}"
    echo ""
    echo "Environment:"
    env | grep -E '^(DATAFUSION|CUDF|AWS|ALGEON)' | sort
else
    echo "Starting algeon_server ..."
    exec "${BINARY}"
fi
