use std::{
    collections::BTreeSet,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use arrow::{
    array::{Array, Int32Array, StringArray, UInt8Array},
    datatypes::{DataType, Field, Schema},
    ipc::writer::IpcWriteOptions,
};
use arrow_flight::{
    FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest, HandshakeResponse, IpcMessage, SchemaAsIpc, Ticket,
    encode::FlightDataEncoderBuilder,
    error::FlightError,
    flight_service_server::FlightService,
    sql::{
        ActionCancelQueryRequest, ActionCancelQueryResult, ActionClosePreparedStatementRequest,
        ActionCreatePreparedStatementRequest, ActionCreatePreparedStatementResult, CommandGetCatalogs,
        CommandGetCrossReference, CommandGetDbSchemas, CommandGetExportedKeys, CommandGetImportedKeys,
        CommandGetPrimaryKeys, CommandGetSqlInfo, CommandGetTableTypes, CommandGetTables,
        CommandPreparedStatementQuery, CommandPreparedStatementUpdate, CommandStatementQuery, CommandStatementUpdate,
        ProstMessageExt, SqlInfo, SqlNullOrdering, SqlOuterJoinsSupportLevel, SqlSupportedCaseSensitivity,
        SqlSupportedGroupBy, SqlSupportedResultSetConcurrency, SqlSupportedResultSetType, SqlSupportedTransaction,
        SqlSupportedUnions, SqlTransactionIsolationLevel, SupportedSqlGrammar, TicketStatementQuery,
        metadata::{GetDbSchemasBuilder, GetTablesBuilder, SqlInfoDataBuilder},
        server::{FlightSqlService, PeekableFlightDataStream},
    },
};
use dashmap::DashMap;
use datafusion::{
    common::TableReference,
    datasource::TableType,
    error::DataFusionError,
    execution::SessionStateBuilder,
    logical_expr::{DdlStatement, LogicalPlan, Statement},
    physical_plan::execute_stream as execute_physical_stream,
    prelude::SessionContext,
};
use futures::{Stream, StreamExt, TryStreamExt};
use nexus_query_engine::runtime::ExecutionCancellation;
use prost::Message;
use tokio_util::sync::CancellationToken;
use tonic::{Code, Request, Response, Status, Streaming, metadata::MetadataValue};
use tracing::Instrument;
use uuid::Uuid;

use super::{
    admission::QueryAdmission,
    auth::{AuthInterceptor, handle_handshake},
    cancel::QueryCancellation,
    config::ServerSessionProfile,
    datafusion_error::find_adapter_error,
    diagnostics::{log_leaf_guard_rejection, log_native_planning_report_for_tables},
    error::{adapter_error_to_status, datafusion_error_to_captured_error, datafusion_error_to_status_for_operation},
    observability::{CapturedError, QueryObservability},
    table_format_operator_fast_fail::fast_fail_unsupported_table_format_operator_plan,
};
#[cfg(feature = "cugraph")]
use crate::graph_sql::CugraphSqlConfig;
#[cfg(feature = "cugraph")]
use crate::session::CudfSessionStateBuilderExt;
use crate::{
    error::{Error, ErrorCode, ErrorSourceDiagnostics},
    native_exec::{NativeMetricsCapture, NativeMetricsCaptureSink},
    planner::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig},
    session::install_cudf_native_rule_with_session_config,
    session_config::DataFusionNexusSessionConfig,
};

mod correlation;
mod ddl;
mod execution;
mod metadata;
mod prepared;
mod rpc;
mod service;
mod statement;
mod status;
#[cfg(all(test, feature = "iceberg"))]
mod tests;
mod tickets;

use correlation::finish_adapter_error_status_with_correlation_id;
pub(super) use correlation::{attach_query_response_metadata, request_correlation_id};
use ddl::{referenced_table_names, reject_drop_object_type_mismatch};
#[cfg(all(test, feature = "iceberg"))]
use execution::cancel_invalid_argument_status;
use metadata::*;
use prepared::PreparedStatementEntry;
pub use service::DatafusionCudfFlightSqlService;
#[cfg(all(test, feature = "iceberg"))]
use service::ServerCatalogDiagnostics;
use status::*;
use tickets::*;

const FLIGHT_SQL_CANCEL_RESULT_CANCELLING: i32 = 2;
const FLIGHT_SQL_CANCEL_RESULT_NOT_CANCELLABLE: i32 = 3;
const GET_ADMISSION_STATS_ACTION: &str = "GetAdmissionStats";
const GET_QUERY_STATS_ACTION: &str = "GetQueryStats";
const NO_UPDATED_RECORDS_COUNT: i64 = 0;
const SQL_ANSI92_ENTRY_SQL_BIT_INDEX: i32 = 0;
const QUERY_ID_METADATA_KEY: &str = "x-datafusion-nexus-query-id";
const CORRELATION_ID_METADATA_KEY: &str = "x-datafusion-nexus-correlation-id";
const CORRELATION_ID_METADATA_ALIASES: &[&str] = &[CORRELATION_ID_METADATA_KEY, "x-correlation-id"];
const MAX_CORRELATION_ID_LEN: usize = 256;
const READ_ONLY_SCHEMA_PROVIDER_REGISTER_TABLE_MESSAGE: &str = "schema provider does not support registering tables";

/// Hard cap on live prepared-statement handles across all connections.
/// Bounds the worst case for clients that leak handles by disconnecting
/// without calling `ClosePreparedStatement`. Sized for DataGrip-class
/// clients (a few hundred concurrent handles); raise when a real workload
/// demands it. Hitting the cap returns `RESOURCE_EXHAUSTED` rather than
/// silently evicting another connection's prepared statement.
const MAX_PREPARED_STATEMENTS: usize = 1024;
