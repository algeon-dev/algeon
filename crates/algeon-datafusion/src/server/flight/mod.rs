use arrow::{
    array::{Array, Int32Array, StringArray, UInt8Array},
    datatypes::{DataType, Field, Schema},
    ipc::writer::IpcWriteOptions,
};
use arrow_flight::{
    FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest, HandshakeResponse, IpcMessage, SchemaAsIpc, Ticket,
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
use datafusion::{
    common::TableReference,
    datasource::TableType,
    error::DataFusionError,
    execution::SessionStateBuilder,
    logical_expr::{DdlStatement, LogicalPlan, Statement},
    physical_plan::{SendableRecordBatchStream, execute_stream},
    prelude::SessionContext,
};
use futures::{Stream, StreamExt, TryStreamExt, stream::BoxStream};
use prost::Message;
use query_engine::{
    observability::{RuntimeObservationConfig, RuntimeObservationHandle},
    plan::MlForestModelDigest,
    runtime::ExecutionCancellation,
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tonic::{Code, Request, Response, Status, Streaming, metadata::MetadataValue};
use tracing::Instrument;

#[cfg(test)]
use super::auth::{AuthMode, TokenStore};
#[cfg(test)]
use super::coordinator::CoordinatorArtifact;
#[cfg(all(test, feature = "iceberg"))]
use super::coordinator::RetainedUnavailableReason;
use super::{
    admission::QueryAdmission,
    auth::{AuthInterceptor, handle_handshake},
    cancel::QueryCancellation,
    coordinator::{
        CoordinatorTicketPayload, QueryCoordinator, QueryCoordinatorConfig, QueryCoordinatorError,
        QueryCoordinatorErrorKind, QueryExecutionLease, QueryPlanningLease, QueryPlanningReportDeliveryTerminal,
        StatementExecutionDomain,
    },
    diagnostics::log_native_planning_report_for_tables,
    error::{adapter_error_to_status, datafusion_error_to_captured_error, datafusion_error_to_status_for_operation},
    observability::CapturedError,
};
use crate::{
    attempt::server_internal::AlgeonAttemptServerExt,
    server::{
        adapter_error::{DataFusionErrorExt, Error, ErrorCode, ErrorLogDiagnostics},
        adapter_session::{
            ALGEON_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME, ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
            AlgeonSessionStateBuilderExt,
        },
        cuml_sql::forest_model_digests,
        gpu_coverage::{ExplainGpuOutputMode, coverage_batch, coverage_schema, parse_explain_gpu, validate_query},
        native_exec::{
            NativeMetricsCapture, ServerNativeObservationActivityFactory as NativeObservationActivityFactory,
        },
        planner::{
            ALGEON_FINAL_PLAN_REQUIREMENT_RULE_NAME, AlgeonNativeOptimizerConfig, AlgeonNativePhysicalOptimizerRule,
        },
    },
};

mod correlation;
mod ddl;
mod execution;
mod frame_limiter;
mod metadata;
mod ownership;
mod prepared;
mod rpc;
mod service;
mod statement;
mod status;
#[cfg(all(test, feature = "iceberg"))]
mod tests;
mod tickets;

use correlation::tracing_correlation_id;
pub(super) use correlation::{attach_query_response_metadata, request_correlation_id};
use ddl::{referenced_table_names, reject_drop_object_type_mismatch};
#[cfg(all(test, feature = "iceberg"))]
use execution::cancel_invalid_argument_status;
use frame_limiter::encode_flight_data_stream;
use metadata::*;
use ownership::{Owner, QueryOwnership};
use prepared::{PreparedStatementEntry, PreparedStatementStore};
pub use service::DatafusionAlgeonFlightSqlService;
use statement::verify_forest_model_identity;
use status::*;
use tickets::*;

const FLIGHT_SQL_CANCEL_RESULT_CANCELLING: i32 = 2;
const FLIGHT_SQL_CANCEL_RESULT_NOT_CANCELLABLE: i32 = 3;
const GET_ADMISSION_STATS_ACTION: &str = "GetAdmissionStats";
const GET_QUERY_STATS_ACTION: &str = "GetQueryStats";
const GET_QUERY_RUNTIME_ARTIFACT_ACTION: &str = "GetQueryRuntimeArtifact";
const NO_UPDATED_RECORDS_COUNT: i64 = 0;
const SQL_ANSI92_ENTRY_SQL_BIT_INDEX: i32 = 0;
const QUERY_ID_METADATA_KEY: &str = "x-algeon-query-id";
const CORRELATION_ID_METADATA_KEY: &str = "x-algeon-correlation-id";
const CORRELATION_ID_METADATA_ALIASES: &[&str] = &[CORRELATION_ID_METADATA_KEY, "x-correlation-id"];
const MAX_CORRELATION_ID_LEN: usize = 256;
