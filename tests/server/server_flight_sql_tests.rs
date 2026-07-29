#![cfg(feature = "server")]

use std::{
    fs::File,
    net::TcpListener,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{
    array::{Array, BooleanArray, Int32Array, Int64Array, StringArray, UInt8Array, UInt32Array, UnionArray},
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
    execution::SessionStateBuilder,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
#[cfg(not(feature = "cugraph"))]
use datafusion_nexus::server::ERROR_OPERATION_METADATA_KEY;
use datafusion_nexus::{
    backend::{GpuMemoryOwnership, NexusGpuBackend},
    server::{
        ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_STATUS_METADATA_KEY, FlightSqlServer, ServerConfig,
        ServerShutdown, serve_with_listener,
    },
    session::NexusSessionStateBuilderExt,
};

fn declare_test_gpu_ownership(config: &mut ServerConfig) {
    config.gpu_memory_ownership =
        config.gpu_devices.iter().copied().map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive)).collect();
}
use futures::TryStreamExt;
use nexus_query_engine::runtime::SlotGuarantee;
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
