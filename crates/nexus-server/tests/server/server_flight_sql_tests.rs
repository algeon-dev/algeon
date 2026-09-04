use std::{
    fs::File,
    net::TcpListener,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use arrow::{
    array::{
        Array, BooleanArray, Decimal128Array, Int32Array, Int64Array, ListArray, StringArray, UInt8Array, UInt32Array,
        UnionArray,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::{
    FlightInfo,
    error::FlightError,
    sql::{
        ActionCancelQueryRequest, ActionCancelQueryResult, ActionCreatePreparedStatementRequest,
        ActionCreatePreparedStatementResult, Any, CommandGetCrossReference, CommandGetDbSchemas,
        CommandGetExportedKeys, CommandGetImportedKeys, CommandGetPrimaryKeys, CommandGetTables,
        CommandPreparedStatementQuery, ProstMessageExt, SqlInfo, SqlNullOrdering, SqlOuterJoinsSupportLevel,
        SqlSupportedGroupBy, SqlSupportedResultSetConcurrency, SqlSupportedResultSetType, SqlSupportedTransaction,
        SqlSupportedUnions, TicketStatementQuery, client::FlightSqlServiceClient,
    },
};
use datafusion::{
    catalog::Session,
    datasource::{TableProvider, TableType},
    execution::SessionStateBuilder,
    logical_expr::{
        Expr, LogicalPlan,
        planner::{PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning},
    },
    physical_plan::ExecutionPlan,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
    sql::sqlparser::ast::TableFactor,
};
use datafusion_nexus::{
    attempt::AttemptOptions,
    backend::{GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend},
    native_exec::NexusNativeExecutionConfig,
};
#[cfg(not(feature = "cugraph"))]
use nexus_server::ERROR_OPERATION_METADATA_KEY;
use nexus_server::{
    ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_STATUS_METADATA_KEY, FlightSqlServer, ServerConfig,
    ServerShutdown, UNSUPPORTED_COLUMN_METADATA_KEY, UNSUPPORTED_COLUMN_ROLE_METADATA_KEY,
    UNSUPPORTED_COLUMN_TYPE_METADATA_KEY, serve_with_listener,
};

fn declare_test_gpu_ownership(config: &mut ServerConfig) {
    config.gpu_memory_ownership =
        config.gpu_devices.iter().copied().map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive)).collect();
}

fn test_server_config() -> ServerConfig {
    test_server_config_with_env(std::collections::BTreeMap::new())
}

fn test_server_config_with_env(mut env: std::collections::BTreeMap<&'static str, String>) -> ServerConfig {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, "[[admission.device_profiles]]\ndevice_ordinal = 0\n").expect("write server config");
    let path = path.to_string_lossy().into_owned();
    env.insert(nexus_server::NEXUS_SERVER_CONFIG_FILE_ENV, path);
    ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("explicit test device profile")
}

// These cases inject an already-built backend, so their server configuration
// must describe the same concrete controller rather than the automatic
// planning placeholder used by ordinary server construction. Keep this local
// to injected-backend coverage: production construction continues to size
// each selected device from its own immutable profile.
const INJECTED_BACKEND_TEST_DEVICE_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

fn build_injected_test_backend(config: &mut ServerConfig) -> NexusGpuBackend {
    config.gpu_device_profiles = config
        .gpu_device_profiles
        .iter()
        .cloned()
        .map(|profile| {
            profile
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit {
                    limit_bytes: INJECTED_BACKEND_TEST_DEVICE_BUDGET_BYTES,
                })
                .with_attempt_floor_bytes(INJECTED_BACKEND_TEST_DEVICE_BUDGET_BYTES)
        })
        .collect();
    config.native_optimizer_config =
        config.native_optimizer_config.clone().with_native_execution_config(NexusNativeExecutionConfig::bounded(
            INJECTED_BACKEND_TEST_DEVICE_BUDGET_BYTES,
            NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
            NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
            NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            NexusNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
        ));
    nexus_server::backend_from_server_config(config).expect("injected backend test profile must build")
}
use futures::TryStreamExt;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use prost::Message as _;
use tokio_util::sync::CancellationToken;
use tonic::{Code, transport::Endpoint};

const QUERY_ID_METADATA_KEY: &str = "x-datafusion-nexus-query-id";
const CANCEL_RESULT_CANCELLING: i32 = 2;

include!("server_flight_sql_tests/support.rs");
include!("server_flight_sql_tests/smoke.rs");
include!("server_flight_sql_tests/query_stats.rs");
include!("server_flight_sql_tests/ddl.rs");
include!("server_flight_sql_tests/errors.rs");
include!("server_flight_sql_tests/tickets.rs");
include!("server_flight_sql_tests/metadata.rs");
include!("server_flight_sql_tests/prepared.rs");
include!("server_flight_sql_tests/explain_gpu.rs");
