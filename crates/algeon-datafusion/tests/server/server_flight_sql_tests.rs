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

#[cfg(not(feature = "cugraph"))]
use algeon_datafusion::server::ERROR_OPERATION_METADATA_KEY;
use algeon_datafusion::{
    attempt::AttemptOptions,
    backend::{AlgeonGpuBackend, GpuDeviceMemoryLimit, GpuMemoryOwnership},
    native_exec::AlgeonNativeExecutionConfig,
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig},
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, ALGEON_SERVER_EXECUTION_MODE_ENV, BLOCKING_KIND_METADATA_KEY,
        BLOCKING_SUBJECT_METADATA_KEY, ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_STATUS_METADATA_KEY,
        FlightSqlServer, REASON_METADATA_KEY, ServerConfig, ServerShutdown, UNSUPPORTED_COLUMN_METADATA_KEY,
        UNSUPPORTED_COLUMN_ROLE_METADATA_KEY, UNSUPPORTED_COLUMN_TYPE_METADATA_KEY, backend_from_server_config,
        build_session_context_with_backend, serve_with_listener,
    },
};
use arrow::{
    array::{
        Array, ArrayRef, BinaryArray, BooleanArray, Decimal128Array, Float64Array, Int32Array, Int64Array, ListArray,
        StringArray, StringViewArray, UInt8Array, UInt32Array, UnionArray,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Schema},
    ipc::convert::try_schema_from_ipc_buffer,
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
    datasource::{TableProvider, TableType, empty::EmptyTable},
    execution::SessionStateBuilder,
    logical_expr::{
        Expr, LogicalPlan,
        planner::{PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning},
    },
    physical_plan::ExecutionPlan,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
    sql::sqlparser::ast::TableFactor,
};
use query_planning::error::bytes_with_gib;
use tokio::time::{sleep, timeout};

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
    env.insert(ALGEON_SERVER_CONFIG_FILE_ENV, path);
    ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("explicit test device profile")
}

// These cases inject an already-built backend, so their server configuration
// must describe the same concrete controller rather than the automatic
// planning placeholder used by ordinary server construction. Keep this local
// to injected-backend coverage: production construction continues to size
// each selected device from its own immutable profile.
const INJECTED_BACKEND_TEST_DEVICE_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

fn build_injected_test_backend(config: &mut ServerConfig) -> AlgeonGpuBackend {
    config.gpu_device_profiles = config
        .gpu_device_profiles
        .iter()
        .cloned()
        .map(|profile| {
            profile
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit {
                    limit_bytes: INJECTED_BACKEND_TEST_DEVICE_BUDGET_BYTES,
                })
                .with_default_query_limit_bytes(INJECTED_BACKEND_TEST_DEVICE_BUDGET_BYTES)
        })
        .collect();
    config.native_optimizer_config =
        config.native_optimizer_config.clone().with_native_execution_config(AlgeonNativeExecutionConfig::default());
    backend_from_server_config(config).expect("injected backend test profile must build")
}
use futures::TryStreamExt;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use prost::{Message as _, bytes::Bytes};
use tokio_util::sync::CancellationToken;
use tonic::{Code, transport::Endpoint};

const QUERY_ID_METADATA_KEY: &str = "x-algeon-query-id";
const CANCEL_RESULT_CANCELLING: i32 = 2;

include!("server_flight_sql_tests/support.rs");
include!("server_flight_sql_tests/smoke.rs");
include!("server_flight_sql_tests/admission.rs");
include!("server_flight_sql_tests/query_stats.rs");
include!("server_flight_sql_tests/ddl.rs");
include!("server_flight_sql_tests/errors.rs");
include!("server_flight_sql_tests/tickets.rs");
include!("server_flight_sql_tests/metadata.rs");
include!("server_flight_sql_tests/prepared.rs");
include!("server_flight_sql_tests/explain_gpu.rs");
