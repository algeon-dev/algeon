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
    attempt::{AlgeonAttempt, AttemptOptions, AttemptStream},
    backend::{AlgeonGpuBackend, GpuDeviceMemoryLimit, GpuMemoryOwnership},
    error::ErrorCode,
    native_exec::AlgeonNativeExecutionConfig,
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig},
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, ALGEON_SERVER_EXECUTION_MODE_ENV, BLOCKING_KIND_METADATA_KEY,
        BLOCKING_SUBJECT_METADATA_KEY, ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_STATUS_METADATA_KEY,
        FlightSqlServer, REASON_METADATA_KEY, Result as ServerResult, ServerConfig, ServerShutdown,
        UNSUPPORTED_COLUMN_METADATA_KEY, UNSUPPORTED_COLUMN_ROLE_METADATA_KEY, UNSUPPORTED_COLUMN_TYPE_METADATA_KEY,
        backend_from_server_config, build_session_context_with_backend, init_cuda_device, serve_with_listener,
    },
};
use arrow::{
    array::{
        Array, ArrayRef, BinaryArray, BooleanArray, Decimal128Array, Float64Array, Int32Array, Int64Array, ListArray,
        StringArray, StringViewArray, UInt8Array, UInt32Array, UnionArray,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Schema, SchemaRef},
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
        CommandPreparedStatementQuery, ProstMessageExt, SqlInfo, TicketStatementQuery, client::FlightSqlServiceClient,
    },
};
use datafusion::{
    catalog::Session,
    common::Result as DataFusionResult,
    datasource::{TableProvider, TableType, empty::EmptyTable},
    error::DataFusionError,
    execution::SessionStateBuilder,
    logical_expr::{
        Expr, LogicalPlan,
        planner::{PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning},
    },
    physical_plan::ExecutionPlan,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
    sql::sqlparser::ast::TableFactor,
};
use explain_gpu::{coverage_candidate_row, coverage_string};
use futures::TryStreamExt;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use prost::{Message as _, bytes::Bytes};
use query_planning::error::bytes_with_gib;
use support::{
    ControlledPlanReplacement, ScanFailingTableProvider, assert_error_metadata, assert_no_diagnostic_fields,
    assert_query_id_metadata, assert_single_sealed_attempt_report, cancel_query_raw, close_prepared_raw,
    collect_metadata_batches, collect_metadata_schema_and_batches, collect_name_and_type, collect_string_column,
    connect_with_retry, create_prepared_raw, execute_batches, execute_flight_info_with_stream_schema, free_local_addr,
    get_query_stats_raw, int_value, query_id_from_statement_flight_info, query_id_from_status,
    query_stats_tsv_single_row, sql_info_string, tsv_value, unwrap_flight_status, wait_for_empty_gpu_ledger,
    wait_for_shared_ledger_queue, write_int64_pairs_parquet, write_basic_list_parquet,
    write_unsupported_decimal_list_parquet, write_update_edges_parquet,
};
use tempfile::{Builder as FlightSqlTempfileBuilder, Builder as AdmissionTempfileBuilder};
use tokio::{
    task::{JoinHandle, spawn_blocking, yield_now},
    time::{sleep, timeout},
};
use tokio_util::sync::CancellationToken;
use tonic::{
    Code,
    transport::{Channel, Endpoint},
};

mod admission;
mod ddl;
mod errors;
mod explain_gpu;
mod metadata;
mod prepared;
mod query_stats;
mod smoke;
mod support;
mod tickets;

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

const QUERY_ID_METADATA_KEY: &str = "x-algeon-query-id";
const CANCEL_RESULT_CANCELLING: i32 = 2;
