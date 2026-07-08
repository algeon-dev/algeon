use std::{
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
use nexus_query_engine::ExecutionCancellation;
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
    diagnostics::{log_leaf_guard_rejection, log_native_planning_report},
    error::{adapter_error_to_status, datafusion_error_to_captured_error, datafusion_error_to_status_for_operation},
    observability::{CapturedError, QueryObservability},
    table_format_operator_fast_fail::fast_fail_unsupported_table_format_operator_plan,
};
#[cfg(feature = "cugraph")]
use crate::CudfSessionStateBuilderExt;
#[cfg(feature = "cugraph")]
use crate::graph_sql::CugraphSqlConfig;
use crate::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, DataFusionNexusSessionConfig, Error, ErrorCode,
    error::ErrorSourceDiagnostics,
    native_exec::{NativeMetricsCapture, NativeMetricsCaptureSink},
    session::install_cudf_native_rule_with_session_config,
};

mod metadata;
mod rpc;
mod status;
mod tickets;

use metadata::*;
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
/// A live Flight SQL prepared statement.
///
/// JDBC clients such as DataGrip prepare a statement once and then re-run the
/// held handle. A DDL statement prepares without a result-set schema, so JDBC
/// clients classify it as an update and execute through `executeUpdate` (the
/// update channel). Some clients still re-run that same handle, so the second
/// execution would otherwise hit DataFusion's "already exists" / "not found"
/// catalog error. We deduplicate no-result-set DDL per handle: the first
/// execution runs, later executions of the *same* handle are no-ops.
/// Query / DML prepared statements set `no_result_ddl = false` and are never
/// deduplicated, so they can still be executed repeatedly.
///
/// Contract: a no-result-set DDL handle is applied **at most once**. This is
/// scoped to *one* handle — a fresh `CreatePreparedStatement` (a distinct
/// handle) re-runs the DDL and still surfaces a genuine duplicate, so an
/// independent client running `CREATE` against a pre-existing table is not
/// masked. The dedup targets the sequential double-send a single JDBC client
/// issues for one logical statement; it deliberately does not re-apply the DDL
/// if catalog state changes underneath a reused handle (e.g. the object is
/// dropped elsewhere and the same handle is executed again), and it does not
/// serialize two *concurrent* executions of the same handle.
#[derive(Debug)]
struct PreparedStatementEntry {
    /// The prepared SQL text. Re-planned on each execution (planning is
    /// side-effect-free; only `execute_logical_plan` mutates the catalog).
    sql: String,
    /// True iff the prepared plan is DDL (no result set). Classified once at
    /// prepare time so neither execution channel has to re-plan to decide
    /// whether dedup applies.
    no_result_ddl: bool,
    /// Set after the first successful execution. Only consulted for
    /// `no_result_ddl` handles, to turn a repeat execution into a no-op.
    executed: AtomicBool,
}

impl PreparedStatementEntry {
    fn new(sql: String, no_result_ddl: bool) -> Self {
        Self { sql, no_result_ddl, executed: AtomicBool::new(false) }
    }

    /// True when this is a no-result-set DDL handle that has already executed
    /// once — i.e. a repeat execution that must be treated as a no-op.
    fn ddl_already_executed(&self) -> bool {
        self.no_result_ddl && self.executed.load(Ordering::Relaxed)
    }

    /// Record a successful execution. Only meaningful for `no_result_ddl`
    /// handles; a no-op otherwise, so callers can mark unconditionally.
    fn mark_executed(&self) {
        if self.no_result_ddl {
            self.executed.store(true, Ordering::Relaxed);
        }
    }
}

#[derive(Clone)]
pub struct DatafusionCudfFlightSqlService {
    ctx: Arc<SessionContext>,
    native_optimizer_config: CudfOptimizerConfig,
    /// Static session-profile selection from [`crate::server::ServerConfig`];
    /// per-statement diagnostics use it to add the "switch to native" hint
    /// when the testing-only `cpu_fallback` profile is active.
    session_profile: ServerSessionProfile,
    admission: Arc<QueryAdmission>,
    cancellations: Arc<QueryCancellation>,
    observability: Arc<QueryObservability>,
    tickets: Arc<DashMap<String, StatementTicket>>,
    /// Map of prepared-statement handle → [`PreparedStatementEntry`].
    /// Populated by `do_action_create_prepared_statement` and removed by
    /// `do_action_close_prepared_statement` (or replaced when the same
    /// handle is reused). The handle is an opaque UUID and is distinct
    /// from the per-execution `query_id` issued at `get_flight_info_*`
    /// time; one prepared statement can be executed multiple times,
    /// each producing a fresh `query_id` / Flight ticket.
    prepared_statements: Arc<DashMap<String, PreparedStatementEntry>>,
    statement_ticket_ttl: Duration,
    /// Parent token that every per-statement token is derived from. Cancelled
    /// by [`crate::server::serve`] on SIGINT/SIGTERM so statements already
    /// registered for execution and tickets still waiting for `DoGet` see
    /// cancellation at the first await.
    shutdown_token: CancellationToken,
    /// Bearer-token validator + handshake credential store. Cloned into
    /// the service from [`crate::server::serve`] so the same
    /// [`TokenStore`] is shared between the handshake mint path and the
    /// per-RPC validate path.
    auth: AuthInterceptor,
    catalog_diagnostics: ServerCatalogDiagnostics,
    #[cfg(feature = "cugraph")]
    cugraph_sql_config: Option<CugraphSqlConfig>,
}

#[derive(Clone, Debug, Default)]
struct ServerCatalogDiagnostics {
    iceberg_enabled: bool,
    workspace_catalog: Option<String>,
    workspace_schema: Option<String>,
}

impl ServerCatalogDiagnostics {
    fn from_server_config(config: &super::config::ServerConfig) -> Self {
        Self {
            iceberg_enabled: server_config_iceberg_enabled(config),
            workspace_catalog: config.workspace.as_ref().map(|workspace| workspace.catalog.clone()),
            workspace_schema: config.workspace.as_ref().map(|workspace| workspace.schema.clone()),
        }
    }
}

#[cfg(feature = "iceberg")]
fn server_config_iceberg_enabled(config: &super::config::ServerConfig) -> bool {
    config.iceberg.is_some()
}

#[cfg(not(feature = "iceberg"))]
fn server_config_iceberg_enabled(_config: &super::config::ServerConfig) -> bool {
    false
}

/// Per-statement session context paired with the retained native rule that
/// will plan it. The native rule reference is kept alive for the lifetime of
/// planning so [`CudfNativePhysicalOptimizerRule::last_report`] can be read
/// after physical optimization runs (during `DataFrame::execute_stream`).
struct QueryPlanning {
    ctx: SessionContext,
    native_rule: Arc<CudfNativePhysicalOptimizerRule>,
}

impl DatafusionCudfFlightSqlService {
    pub fn new(
        ctx: Arc<SessionContext>,
        native_optimizer_config: CudfOptimizerConfig,
        session_profile: ServerSessionProfile,
        admission: Arc<QueryAdmission>,
        statement_ticket_ttl: Duration,
        auth: AuthInterceptor,
        #[cfg(feature = "cugraph")] cugraph_sql_config: Option<CugraphSqlConfig>,
    ) -> Self {
        Self {
            ctx,
            native_optimizer_config,
            session_profile,
            admission,
            cancellations: Arc::default(),
            observability: Arc::default(),
            tickets: Arc::default(),
            prepared_statements: Arc::default(),
            statement_ticket_ttl,
            shutdown_token: CancellationToken::new(),
            auth,
            catalog_diagnostics: ServerCatalogDiagnostics::default(),
            #[cfg(feature = "cugraph")]
            cugraph_sql_config,
        }
    }

    pub(super) fn with_server_config_diagnostics(mut self, config: &super::config::ServerConfig) -> Self {
        self.catalog_diagnostics = ServerCatalogDiagnostics::from_server_config(config);
        self
    }

    #[cfg(all(test, feature = "iceberg"))]
    fn with_catalog_diagnostics(mut self, diagnostics: ServerCatalogDiagnostics) -> Self {
        self.catalog_diagnostics = diagnostics;
        self
    }

    /// Validate the bearer token on a tonic request. Returns the
    /// authenticated [`super::auth::Principal`] when auth is enabled,
    /// or `None` when the server is running in
    /// [`AuthMode::Disabled`]. Errors as `Status::unauthenticated`.
    fn check_auth<T>(&self, request: &Request<T>) -> Result<Option<super::auth::Principal>, Status> {
        self.auth.check(request.metadata())
    }

    fn query_span(query_id: &str, correlation_id: Option<&str>, operation: &'static str) -> tracing::Span {
        tracing::info_span!(
            "datafusion_nexus.flight_sql.query",
            query_id,
            correlation_id = correlation_id.unwrap_or(""),
            operation
        )
    }

    fn query_stage_span(
        parent: Option<&tracing::Span>,
        query_id: &str,
        correlation_id: Option<&str>,
        stage: &'static str,
    ) -> tracing::Span {
        match parent {
            Some(parent) => tracing::info_span!(
                parent: parent,
                "datafusion_nexus.flight_sql.stage",
                query_id,
                correlation_id = correlation_id.unwrap_or(""),
                stage
            ),
            None => tracing::info_span!(
                "datafusion_nexus.flight_sql.stage",
                query_id,
                correlation_id = correlation_id.unwrap_or(""),
                stage
            ),
        }
    }

    async fn build_get_tables_response(
        &self,
        query: CommandGetTables,
    ) -> Result<(Arc<Schema>, arrow::record_batch::RecordBatch), Status> {
        let include_schema = query.include_schema;
        let filters = GetTablesPrefilters::from(&query);
        let mut builder = GetTablesBuilder::from(query);
        let empty_schema = Schema::empty();
        for catalog_name in self.ctx.catalog_names() {
            if !filters.catalog_matches(&catalog_name) {
                continue;
            }

            let Some(catalog) = self.ctx.catalog(&catalog_name) else {
                continue;
            };
            for schema_name in catalog.schema_names() {
                if !filters.schema_matches(&schema_name) {
                    continue;
                }

                let Some(schema_provider) = catalog.schema(&schema_name) else {
                    continue;
                };
                for table_name in schema_provider.table_names() {
                    if !filters.table_name_matches(&table_name) {
                        continue;
                    }

                    let (table_schema, table_type_str) = if include_schema && !filters.has_table_type_filter() {
                        match schema_provider.table(&table_name).await {
                            Ok(Some(table)) => (Some(table.schema()), table_type_to_flight_sql(table.table_type())),
                            Ok(None) => continue,
                            Err(error) => {
                                Self::warn_get_tables_metadata_error(
                                    &catalog_name,
                                    &schema_name,
                                    &table_name,
                                    "load_table",
                                    &error,
                                );
                                continue;
                            },
                        }
                    } else {
                        let table_type = match schema_provider.table_type(&table_name).await {
                            Ok(Some(table_type)) => table_type,
                            Ok(None) => continue,
                            Err(error) if include_schema => {
                                Self::warn_get_tables_metadata_error(
                                    &catalog_name,
                                    &schema_name,
                                    &table_name,
                                    "table_type",
                                    &error,
                                );
                                continue;
                            },
                            Err(error) => {
                                return Err(datafusion_error_to_status_for_operation(
                                    error,
                                    "datafusion_nexus.server.flight.do_get_tables.table_type",
                                ));
                            },
                        };

                        let table_type_str = table_type_to_flight_sql(table_type);
                        if !filters.table_type_matches(table_type_str) {
                            continue;
                        }

                        let table_schema = if include_schema {
                            match schema_provider.table(&table_name).await {
                                Ok(Some(table)) => Some(table.schema()),
                                Ok(None) => None,
                                Err(error) => {
                                    Self::warn_get_tables_metadata_error(
                                        &catalog_name,
                                        &schema_name,
                                        &table_name,
                                        "load_table",
                                        &error,
                                    );
                                    None
                                },
                            }
                        } else {
                            None
                        };
                        if include_schema && table_schema.is_none() {
                            continue;
                        }
                        (table_schema, table_type_str)
                    };

                    let table_schema_ref: &Schema = table_schema.as_deref().unwrap_or(&empty_schema);
                    builder
                        .append(&catalog_name, &schema_name, &table_name, table_type_str, table_schema_ref)
                        .map_err(|error| {
                            arrow_flight_error_to_status_for_operation(
                                error,
                                "datafusion_nexus.server.flight.do_get_tables.append",
                                "failed to append Flight SQL GetTables metadata row",
                            )
                        })?;
                }
            }
        }
        let schema = builder.schema();
        let batch = builder.build().map_err(|error| {
            arrow_flight_error_to_status_for_operation(
                error,
                "datafusion_nexus.server.flight.do_get_tables.build",
                "failed to build Flight SQL GetTables metadata batch",
            )
        })?;
        Ok((schema, batch))
    }

    fn warn_get_tables_metadata_error(
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
        failure: &'static str,
        error: &DataFusionError,
    ) {
        let diagnostics = ErrorSourceDiagnostics::from_error(error);
        if let Some(adapter_error) = find_adapter_error(error) {
            tracing::warn!(
                catalog = %catalog_name,
                schema = %schema_name,
                table = %table_name,
                failure,
                error_code = adapter_error.code().as_str(),
                error_kind = adapter_error.kind().as_str(),
                error_status = adapter_error.status().as_str(),
                error_operation = adapter_error.operation(),
                error_message = adapter_error.message(),
                error_source_chain = diagnostics.chain(),
                error_root_cause = diagnostics.root_cause_when_distinct(),
                "failed to inspect table metadata for Flight SQL GetTables"
            );
        } else {
            tracing::warn!(
                catalog = %catalog_name,
                schema = %schema_name,
                table = %table_name,
                failure,
                error_source_chain = diagnostics.chain(),
                error_root_cause = diagnostics.root_cause_when_distinct(),
                "failed to inspect table metadata for Flight SQL GetTables"
            );
        }
    }

    pub(crate) fn admission(&self) -> &Arc<QueryAdmission> {
        &self.admission
    }

    pub(crate) fn cancellations(&self) -> &Arc<QueryCancellation> {
        &self.cancellations
    }

    pub(crate) fn shutdown_token(&self) -> &CancellationToken {
        &self.shutdown_token
    }

    #[cfg(test)]
    pub(crate) fn native_optimizer_config(&self) -> &CudfOptimizerConfig {
        &self.native_optimizer_config
    }

    fn query_planning(
        &self,
        query_id: &str,
        correlation_id: Option<&str>,
        cancellation: ExecutionCancellation,
    ) -> QueryPlanning {
        let observability: Arc<dyn NativeMetricsCaptureSink> = self.observability.clone();
        let metrics_capture = NativeMetricsCapture::new(query_id.to_string(), observability);
        let mut execution_config = self
            .native_optimizer_config
            .native_execution_config()
            .clone()
            .with_cancellation(cancellation)
            .with_metrics_capture(metrics_capture);
        if let Some(correlation_id) = correlation_id.map(str::trim).filter(|value| !value.is_empty()) {
            execution_config = execution_config.with_trace_id(correlation_id.to_string());
        }
        let optimizer_config =
            self.native_optimizer_config.clone().with_native_execution_config(execution_config.clone());
        // Retain the native rule so we can read its `last_report` for the
        // Milestone 4 planning-report log after physical optimization runs.
        let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
        let mut session_extension = DataFusionNexusSessionConfig::from_optimizer_config(&optimizer_config);
        if optimizer_config.native_execution_config().has_query_admission_authority() {
            session_extension = session_extension.frozen_for_backend_admission();
        }
        let builder = install_cudf_native_rule_with_session_config(
            SessionStateBuilder::new_from_existing(self.ctx.state()),
            Arc::clone(&native_rule),
            session_extension,
        )
        .expect("server query planning should install a session config derived from the same optimizer config");
        #[cfg(feature = "cugraph")]
        let builder = if let Some(cugraph_sql_config) = &self.cugraph_sql_config {
            builder.with_cugraph_sql(cugraph_sql_config.clone().with_native_execution_config(execution_config))
        } else {
            builder
        };
        let state = builder.build();
        QueryPlanning { ctx: SessionContext::new_with_state(state), native_rule }
    }

    async fn statement_flight_info(
        &self,
        query_id: String,
        sql: String,
        correlation_id: Option<String>,
        descriptor: FlightDescriptor,
        started_at: Instant,
        ticket_kind: StatementTicketKind,
        prepared_handle: Option<String>,
    ) -> Result<FlightInfo, Status> {
        // Before ticket issuance there is no per-query token, so planning
        // observes the server shutdown token directly. The native flag is still
        // shared with the optimizer rule, but `do_action_cancel_query` cannot
        // reach it until the ticket is registered.
        let native_cancellation = ExecutionCancellation::new();
        let planning = self.query_planning(&query_id, correlation_id.as_deref(), native_cancellation.clone());
        let query_ctx = &planning.ctx;
        let query_state = query_ctx.state();
        let logical_plan = tokio::select! {
            result = query_state.create_logical_plan(&sql)
                .instrument(Self::query_stage_span(None, &query_id, correlation_id.as_deref(), "planning")) => {
                result.map_err(|error| {
                    self.observability.finish_failed(
                        &query_id,
                        "planning",
                        datafusion_error_to_captured_error(&error, ticket_kind.create_logical_plan_operation()),
                    );
                    datafusion_error_to_status_for_operation(error, ticket_kind.create_logical_plan_operation())
                })?
            },
            _ = self.shutdown_token.cancelled() => {
                native_cancellation.cancel();
                let status = finish_adapter_error_status_with_correlation_id(
                    self.observability.as_ref(),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                    server_shutdown_error(),
                );
                return Err(status);
            }
        };
        if let Err(error) = verify_flight_info_plan(&logical_plan, ticket_kind) {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            return Err(status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            ));
        }
        if let Err(error) =
            reject_drop_object_type_mismatch(query_ctx, &logical_plan, ticket_kind.create_logical_plan_operation())
                .await
        {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            return Err(status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            ));
        }
        if let Some(error) = fast_fail_unsupported_table_format_operator_plan(&logical_plan) {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            return Err(status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            ));
        }
        let schema = flight_info_result_schema(&logical_plan);

        // Both ticket variants carry the per-execution `query_id` bytes. For
        // prepared statements this reuses `prepared_statement_handle`; the
        // persistent handle remains in `self.prepared_statements`.
        let ticket_payload = match ticket_kind {
            StatementTicketKind::Statement => {
                let ticket = TicketStatementQuery { statement_handle: query_id.as_bytes().to_vec().into() };
                ticket.as_any().encode_to_vec().into()
            },
            StatementTicketKind::PreparedStatement => {
                let ticket =
                    CommandPreparedStatementQuery { prepared_statement_handle: query_id.as_bytes().to_vec().into() };
                ticket.as_any().encode_to_vec().into()
            },
        };
        let endpoint = FlightEndpoint::new().with_ticket(Ticket { ticket: ticket_payload });
        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|error| {
                arrow_error_to_status_for_operation(
                    error,
                    "datafusion_nexus.server.flight.get_flight_info_query.schema",
                    "failed to build Flight SQL query FlightInfo schema",
                )
            })?
            .with_descriptor(descriptor)
            .with_endpoint(endpoint)
            .with_ordered(false);
        self.tickets.insert(
            query_id.clone(),
            StatementTicket {
                query_id,
                sql,
                correlation_id,
                ticket_kind,
                prepared_handle,
                native_cancellation,
                started_at,
                issued_at: Instant::now(),
            },
        );
        Ok(info)
    }

    fn expire_stale_tickets(&self, now: Instant) {
        let ttl = self.statement_ticket_ttl;
        self.tickets.retain(|query_id, ticket| {
            let age = now.saturating_duration_since(ticket.issued_at);
            if age < ttl {
                return true;
            }
            self.cancellations.remove(query_id);
            self.observability.seal(query_id);
            tracing::warn!(
                query_id = query_id.as_str(),
                ticket_age_ms = age.as_millis(),
                statement_ticket_ttl_ms = ttl.as_millis(),
                "expired abandoned Flight SQL statement ticket"
            );
            false
        });
    }

    async fn execute_ticket(
        &self,
        query_id: String,
        request_correlation_id: Option<String>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.expire_stale_tickets(Instant::now());
        let (_, ticket) = self.tickets.remove(&query_id).ok_or_else(|| {
            let status = status_with_query_id_and_correlation_id(
                adapter_error_status(Error::new(
                    ErrorCode::InvalidRequest,
                    "datafusion_nexus.server.flight.execute_ticket.resolve",
                    "unknown, expired, or already consumed statement ticket",
                )),
                &query_id,
                request_correlation_id.as_deref(),
            );
            log_statement_status(&query_id, "ticket", Instant::now(), &status);
            status
        })?;

        // Derive from `shutdown_token` so cancellation cascades to statements
        // that race in between ticket consumption and cancellation registry insert.
        let token = self.shutdown_token.child_token();
        let StatementTicket {
            query_id,
            sql,
            correlation_id,
            ticket_kind,
            prepared_handle,
            native_cancellation,
            started_at,
            ..
        } = ticket;
        let correlation_id = correlation_id.or(request_correlation_id);
        self.observability.record_correlation_id(&query_id, correlation_id.as_deref());
        let query_span = Self::query_span(&query_id, correlation_id.as_deref(), "do_get_statement");
        {
            let _query_enter = query_span.enter();
            tracing::info!(
                query_id,
                correlation_id = correlation_id.as_deref().unwrap_or(""),
                "executing Flight SQL statement ticket"
            );
        }
        self.cancellations.insert(query_id.clone(), token.clone(), native_cancellation.clone());
        let native_cancellation_for_bridge = native_cancellation.clone();
        let token_for_bridge = token.clone();
        // Native cancellation is cooperative. A cancelled query keeps its
        // worker and native execution admission until the current libcudf/CUDA call
        // reaches the next cancellation boundary; this does not promise kernel
        // preemption.
        let native_cancel_bridge = tokio::spawn(async move {
            token_for_bridge.cancelled().await;
            native_cancellation_for_bridge.cancel();
        });
        let lifecycle = Arc::new(QueryLifecycle::new(query_id.clone(), started_at));
        // Construct the guard before any post-ticket branch so cancellation
        // registration and observability cleanup stay tied together after the
        // ticket leaves `self.tickets`.
        let guard = QueryExecutionGuard {
            query_id: query_id.clone(),
            correlation_id: correlation_id.clone(),
            cancellations: Arc::clone(&self.cancellations),
            observability: Arc::clone(&self.observability),
            flight_token: token.clone(),
            native_cancellation: native_cancellation.clone(),
            native_cancel_bridge,
            shutdown_token: self.shutdown_token.clone(),
            lifecycle: Arc::clone(&lifecycle),
        };
        if token.is_cancelled() {
            native_cancellation.cancel();
            let status = finish_adapter_error_status_with_correlation_id(
                self.observability.as_ref(),
                &query_id,
                correlation_id.as_deref(),
                "planning",
                cancellation_error(&self.shutdown_token),
            );
            lifecycle.finish_status("planning", &status);
            self.observability.seal(&query_id);
            return Err(status);
        }

        // DataGrip-class clients re-run held prepared-statement handles. A
        // no-result-set DDL handle that already executed once is a no-op on
        // re-execution: return an empty result so the client's `execute()`
        // succeeds instead of surfacing DataFusion's "already exists" /
        // "not found" catalog error from the duplicate run.
        if let Some(handle) = &prepared_handle
            && self.prepared_statements.get(handle).is_some_and(|entry| entry.ddl_already_executed())
        {
            tracing::info!(
                query_id,
                prepared_statement_handle = handle.as_str(),
                "deduplicated repeated prepared DDL execution on the query channel; returning an empty result"
            );
            lifecycle.finish_success("stream");
            self.observability.finish_success(&query_id);
            self.observability.seal(&query_id);
            drop(guard);
            let query_id_for_encode = query_id.clone();
            let correlation_id_for_encode = correlation_id.clone();
            let stream = FlightDataEncoderBuilder::new()
                .with_schema(Arc::new(Schema::empty()))
                .build(futures::stream::empty::<Result<arrow::record_batch::RecordBatch, FlightError>>())
                .map_err(move |error| {
                    status_with_query_id_and_correlation_id(
                        arrow_flight_error_to_status_for_operation(
                            error,
                            "datafusion_nexus.server.flight.execute_ticket.empty_ddl_encode",
                            "failed to encode empty Flight SQL DDL result stream",
                        ),
                        &query_id_for_encode,
                        correlation_id_for_encode.as_deref(),
                    )
                });
            let mut response = Response::new(Box::pin(stream) as <Self as FlightService>::DoGetStream);
            attach_query_response_metadata(&mut response, &query_id, correlation_id.as_deref());
            return Ok(response);
        }

        let planning = self.query_planning(&query_id, correlation_id.as_deref(), native_cancellation.clone());
        let query_ctx = &planning.ctx;
        let query_state = query_ctx.state();
        let logical_plan = tokio::select! {
            result = query_state.create_logical_plan(&sql)
                .instrument(Self::query_stage_span(
                    Some(&query_span),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                )) => match result {
                Ok(plan) => plan,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    self.observability.finish_failed(
                        &query_id,
                        "planning",
                        datafusion_error_to_captured_error(&error, ticket_kind.sql_operation()),
                    );
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, ticket_kind.sql_operation()),
                        &query_id,
                    );
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    lifecycle.finish_status("planning", &status);
                    self.observability.seal(&query_id);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
                native_cancellation.cancel();
                let status = finish_adapter_error_status_with_correlation_id(
                    self.observability.as_ref(),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                    cancellation_error(&self.shutdown_token),
                );
                lifecycle.finish_status("planning", &status);
                self.observability.seal(&query_id);
                return Err(status);
            },
        };
        if let Err(error) =
            reject_drop_object_type_mismatch(query_ctx, &logical_plan, ticket_kind.sql_operation()).await
        {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            let status = status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            );
            lifecycle.finish_status("planning", &status);
            self.observability.seal(&query_id);
            return Err(status);
        }
        let df = tokio::select! {
            result = query_ctx.execute_logical_plan(logical_plan)
                .instrument(Self::query_stage_span(
                    Some(&query_span),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                )) => match result {
                Ok(df) => df,
                Err(error) => {
                    // DDL execution and logical-plan wrapping happen before
                    // the physical optimizer runs, so `last_report` is empty
                    // here. The leaf-guard helper no-ops on non-leaf-guard
                    // errors, but we run it for completeness so any
                    // planning-time adapter error that does carry the
                    // leaf-guard facts gets the structured warn line.
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    self.observability.finish_failed(
                        &query_id,
                        "planning",
                        datafusion_error_to_captured_error(&error, ticket_kind.sql_operation()),
                    );
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, ticket_kind.sql_operation()),
                        &query_id,
                    );
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    lifecycle.finish_status("planning", &status);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
                native_cancellation.cancel();
                let status = finish_adapter_error_status_with_correlation_id(
                    self.observability.as_ref(),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                    cancellation_error(&self.shutdown_token),
                );
                lifecycle.finish_status("planning", &status);
                return Err(status);
            },
        };
        // `execute_logical_plan` has run any DDL side effect by now, so record
        // the success: a re-run of this prepared handle short-circuits to an
        // empty result (`mark_executed` self-guards on `no_result_ddl`, so
        // query/DML handles are unaffected).
        if let Some(handle) = &prepared_handle
            && let Some(entry) = self.prepared_statements.get(handle)
        {
            entry.mark_executed();
        }
        if let Some(error) = fast_fail_unsupported_table_format_operator_plan(df.logical_plan()) {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            let status = status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            );
            lifecycle.finish_status("planning", &status);
            return Err(status);
        }
        let schema = df.schema().as_arrow().clone();
        let task_ctx = Arc::new(df.task_ctx());
        // Physical planning runs the retained native optimizer rule. Capture
        // the post-optimizer plan and retained diagnostics before execution so
        // query_stats can report both native rewrites and complete CPU fallback.
        let plan = tokio::select! {
            result = df.create_physical_plan()
                .instrument(Self::query_stage_span(
                    Some(&query_span),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                )) => match result {
                Ok(plan) => plan,
                Err(error) => {
                    let diagnostics = planning.native_rule.last_planning_diagnostics();
                    log_native_planning_report(&query_id, diagnostics.as_ref().map(|diagnostics| &diagnostics.report));
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    self.observability.capture_planning_failure(
                        &query_id,
                        "physical_planning",
                        "physical_planning_failed",
                        error.to_string(),
                    );
                    self.observability.finish_failed(
                        &query_id,
                        "execute_stream",
                        datafusion_error_to_captured_error(&error, ticket_kind.execute_stream_operation()),
                    );
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, ticket_kind.execute_stream_operation()),
                        &query_id,
                    );
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    lifecycle.finish_status("execute_stream", &status);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
                native_cancellation.cancel();
                let status = finish_adapter_error_status_with_correlation_id(
                    self.observability.as_ref(),
                    &query_id,
                    correlation_id.as_deref(),
                    "execute_stream",
                    cancellation_error(&self.shutdown_token),
                );
                lifecycle.finish_status("execute_stream", &status);
                return Err(status);
            },
        };
        let diagnostics = planning.native_rule.last_planning_diagnostics();
        log_native_planning_report(&query_id, diagnostics.as_ref().map(|diagnostics| &diagnostics.report));
        self.observability.capture_planning(&query_id, diagnostics, Arc::clone(&plan));
        let native_execution_span =
            Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "native_execution");
        let stream = match execute_physical_stream(Arc::clone(&plan), task_ctx) {
            Ok(stream) => stream,
            Err(error) => {
                log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                self.observability.finish_failed(
                    &query_id,
                    "execute_stream",
                    datafusion_error_to_captured_error(&error, ticket_kind.execute_stream_operation()),
                );
                let status = status_with_query_id(
                    datafusion_error_to_status_for_operation(error, ticket_kind.execute_stream_operation()),
                    &query_id,
                );
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                lifecycle.finish_status("execute_stream", &status);
                return Err(status);
            },
        };
        let mut native_stream = stream;
        let native_execution_span_for_poll = native_execution_span.clone();
        let stream = futures::stream::poll_fn(move |context| {
            let _native_enter = native_execution_span_for_poll.enter();
            native_stream.as_mut().poll_next(context)
        });
        let cancel_token_for_tail = token.clone();
        let shutdown_token_for_tail = self.shutdown_token.clone();
        let query_id_for_tail = query_id.clone();
        let correlation_id_for_tail = correlation_id.clone();
        let lifecycle_for_body = Arc::clone(&lifecycle);
        let query_id_for_body = query_id.clone();
        let correlation_id_for_body = correlation_id.clone();
        let observability_for_body = Arc::clone(&self.observability);
        let query_id_for_output = query_id.clone();
        let observability_for_output = Arc::clone(&self.observability);
        let streaming_span =
            Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "result_streaming");
        let streaming_span_for_output = streaming_span.clone();
        let stream_operation = ticket_kind.stream_operation();
        // `take_until` ends cleanly on cancellation, so append a one-shot tail
        // that emits CANCELLED as the gRPC trailer when the token fired.
        let body = stream
            .map_ok(move |batch| {
                let _stream_enter = streaming_span_for_output.enter();
                observability_for_output.record_output_batch(&query_id_for_output, &batch);
                batch
            })
            .map_err(move |error| {
                let _stream_enter = DatafusionCudfFlightSqlService::query_stage_span(
                    None,
                    &query_id_for_body,
                    correlation_id_for_body.as_deref(),
                    "result_streaming",
                )
                .entered();
                observability_for_body.finish_failed(
                    &query_id_for_body,
                    "stream",
                    datafusion_error_to_captured_error(&error, stream_operation),
                );
                datafusion_error_to_flight_error(error, &query_id_for_body, &lifecycle_for_body, stream_operation)
            })
            .take_until(token.cancelled_owned());
        let lifecycle_for_tail = Arc::clone(&lifecycle);
        let observability_for_tail = Arc::clone(&self.observability);
        let tail = futures::stream::once(async move {
            cancel_token_for_tail.is_cancelled().then(|| {
                let status = finish_adapter_error_status_with_correlation_id(
                    observability_for_tail.as_ref(),
                    &query_id_for_tail,
                    correlation_id_for_tail.as_deref(),
                    "stream",
                    cancellation_error(&shutdown_token_for_tail),
                );
                lifecycle_for_tail.finish_status("stream", &status);
                Err(FlightError::from(status))
            })
        })
        .filter_map(|opt| async move { opt });
        let lifecycle_for_end = Arc::clone(&lifecycle);
        let observability_for_end = Arc::clone(&self.observability);
        let query_id_for_end = query_id.clone();
        let end = futures::stream::once(async move {
            lifecycle_for_end.finish_success("stream");
            observability_for_end.finish_success(&query_id_for_end);
            None::<Result<arrow::record_batch::RecordBatch, FlightError>>
        })
        .filter_map(|opt| async move { opt });
        let stream = body.chain(tail).chain(end).map(move |batch| {
            let _guard = &guard;
            batch
        });
        let mut stream = Box::pin(stream);
        let result_streaming_span_for_poll = streaming_span.clone();
        let stream = futures::stream::poll_fn(move |context| {
            let _stream_enter = result_streaming_span_for_poll.enter();
            stream.as_mut().poll_next(context)
        });
        let lifecycle_for_encode = Arc::clone(&lifecycle);
        let query_id_for_encode = query_id.clone();
        let correlation_id_for_encode = correlation_id.clone();
        let observability_for_encode = Arc::clone(&self.observability);
        let stream =
            FlightDataEncoderBuilder::new().with_schema(Arc::new(schema)).build(stream).map_err(move |error| {
                let status = status_with_query_id_and_correlation_id(
                    arrow_flight_error_to_status_for_operation(
                        error,
                        "datafusion_nexus.server.flight.execute_ticket.result_encode",
                        "failed to encode Flight SQL result stream",
                    ),
                    &query_id_for_encode,
                    correlation_id_for_encode.as_deref(),
                );
                observability_for_encode.finish_status(&query_id_for_encode, "flight_encode", &status);
                lifecycle_for_encode.finish_status("flight_encode", &status);
                status
            });

        let mut response = Response::new(Box::pin(stream) as <Self as FlightService>::DoGetStream);
        attach_query_response_metadata(&mut response, &query_id, correlation_id.as_deref());
        Ok(response)
    }

    async fn execute_update_sql(
        &self,
        query_id: String,
        sql: String,
        correlation_id: Option<String>,
        statement_kind: UpdateStatementKind,
    ) -> Result<i64, Status> {
        let started_at = Instant::now();
        let query_span = Self::query_span(&query_id, correlation_id.as_deref(), "update_statement");
        let _query_enter = query_span.enter();
        tracing::info!(
            query_id,
            correlation_id = correlation_id.as_deref().unwrap_or(""),
            sql,
            statement_kind = ?statement_kind,
            "executing Flight SQL update statement"
        );

        drop(_query_enter);

        let lifecycle = QueryLifecycle::new(query_id.clone(), started_at);
        let native_cancellation = ExecutionCancellation::new();
        let planning = self.query_planning(&query_id, correlation_id.as_deref(), native_cancellation.clone());
        let query_ctx = &planning.ctx;
        let state = query_ctx.state();
        let plan = tokio::select! {
            result = state.create_logical_plan(&sql)
                .instrument(Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "planning")) => match result {
                Ok(plan) => plan,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    self.observability.finish_failed(
                        &query_id,
                        "planning",
                        datafusion_error_to_captured_error(&error, statement_kind.sql_operation()),
                    );
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                        &query_id,
                    );
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    lifecycle.finish_status("planning", &status);
                    self.observability.seal(&query_id);
                    return Err(status);
                },
            },
            _ = self.shutdown_token.cancelled() => {
                native_cancellation.cancel();
                let status = finish_adapter_error_status_with_correlation_id(
                    self.observability.as_ref(),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                    server_shutdown_error(),
                );
                lifecycle.finish_status("planning", &status);
                self.observability.seal(&query_id);
                return Err(status);
            },
        };
        if let Err(error) = reject_parameterized_plan(&plan, statement_kind.sql_operation()) {
            self.observability.finish_failed(
                &query_id,
                "planning",
                datafusion_error_to_captured_error(&error, statement_kind.sql_operation()),
            );
            let status = status_with_query_id(
                datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                &query_id,
            );
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            lifecycle.finish_status("planning", &status);
            self.observability.seal(&query_id);
            return Err(status);
        }
        if let Err(error) = verify_update_plan(&plan) {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            let status = status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            );
            lifecycle.finish_status("planning", &status);
            self.observability.seal(&query_id);
            return Err(status);
        }
        if let Err(error) = reject_drop_object_type_mismatch(query_ctx, &plan, statement_kind.sql_operation()).await {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            let status = status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            );
            lifecycle.finish_status("planning", &status);
            self.observability.seal(&query_id);
            return Err(status);
        }
        let plan_for_error = plan.clone();
        let df = tokio::select! {
            result = query_ctx.execute_logical_plan(plan)
                .instrument(Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "planning")) => match result {
                Ok(df) => df,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    let status = match self.ddl_read_only_registration_error(
                        query_ctx,
                        &plan_for_error,
                        error,
                        statement_kind.sql_operation(),
                    ) {
                        Ok(error) => finish_adapter_error_status_with_correlation_id(
                            self.observability.as_ref(),
                            &query_id,
                            correlation_id.as_deref(),
                            "planning",
                            error,
                        ),
                        Err(error) => {
                            self.observability.finish_failed(
                                &query_id,
                                "planning",
                                datafusion_error_to_captured_error(&error, statement_kind.sql_operation()),
                            );
                            let status = status_with_query_id(
                                datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                                &query_id,
                            );
                            status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref())
                        },
                    };
                    lifecycle.finish_status("planning", &status);
                    self.observability.seal(&query_id);
                    return Err(status);
                },
            },
            _ = self.shutdown_token.cancelled() => {
                native_cancellation.cancel();
                let status = finish_adapter_error_status_with_correlation_id(
                    self.observability.as_ref(),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                    server_shutdown_error(),
                );
                lifecycle.finish_status("planning", &status);
                self.observability.seal(&query_id);
                return Err(status);
            },
        };

        let _batches = tokio::select! {
            result = df.collect()
                .instrument(Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "native_execution")) => match result {
                Ok(batches) => batches,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    self.observability.finish_failed(
                        &query_id,
                        "collect",
                        datafusion_error_to_captured_error(&error, statement_kind.collect_operation()),
                    );
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, statement_kind.collect_operation()),
                        &query_id,
                    );
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    lifecycle.finish_status("collect", &status);
                    self.observability.seal(&query_id);
                    return Err(status);
                },
            },
            _ = self.shutdown_token.cancelled() => {
                native_cancellation.cancel();
                let status = finish_adapter_error_status_with_correlation_id(
                    self.observability.as_ref(),
                    &query_id,
                    correlation_id.as_deref(),
                    "collect",
                    server_shutdown_error(),
                );
                lifecycle.finish_status("collect", &status);
                self.observability.seal(&query_id);
                return Err(status);
            },
        };

        lifecycle.finish_success("collect");
        self.observability.finish_success(&query_id);
        self.observability.seal(&query_id);
        // Flight SQL update success responses carry only a record count, so
        // there is no portable client-visible query_id. Failure captures are
        // still retained because their Status metadata exposes the query_id.
        self.observability.delete(&query_id);
        Ok(NO_UPDATED_RECORDS_COUNT)
    }

    fn ddl_read_only_registration_error(
        &self,
        ctx: &SessionContext,
        plan: &LogicalPlan,
        source: DataFusionError,
        operation: &'static str,
    ) -> Result<Error, DataFusionError> {
        if !datafusion_error_contains_message(&source, READ_ONLY_SCHEMA_PROVIDER_REGISTER_TABLE_MESSAGE) {
            return Err(source);
        }
        let Some((target, statement_kind)) = ddl_register_table_target(plan) else {
            return Err(source);
        };

        let options = ctx.state();
        let catalog_options = &options.config().options().catalog;
        let resolved = target.clone().resolve(&catalog_options.default_catalog, &catalog_options.default_schema);
        let message = format!(
            "schema provider does not support registering tables for DDL target {}.{}",
            resolved.catalog, resolved.schema
        );
        let mut error = Error::new(ErrorCode::DataFusionPlanning, operation, message)
            .with_fact("target_catalog", resolved.catalog.as_ref())
            .with_fact("target_schema", resolved.schema.as_ref())
            .with_fact("target_table", resolved.table.as_ref())
            .with_fact("statement_kind", statement_kind)
            .with_fact("iceberg_enabled", self.catalog_diagnostics.iceberg_enabled);
        if let Some(workspace_catalog) = &self.catalog_diagnostics.workspace_catalog {
            error = error.with_fact("workspace_catalog", workspace_catalog);
        }
        if let Some(workspace_schema) = &self.catalog_diagnostics.workspace_schema {
            error = error.with_fact("workspace_schema", workspace_schema);
        }
        Ok(error.with_source(source))
    }

    fn dataset_schema_for_prepared_statement(&self, plan: &LogicalPlan) -> crate::Result<Schema> {
        match plan {
            LogicalPlan::Ddl(ddl) => {
                verify_update_ddl(
                    ddl,
                    "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                )?;
                Ok(Schema::empty())
            },
            LogicalPlan::Dml(_) | LogicalPlan::Copy(_) => Err(Error::new(
                ErrorCode::UnsupportedOperator,
                "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                "Flight SQL prepared DML and COPY statements are not supported",
            )),
            LogicalPlan::Statement(statement) => match statement {
                Statement::SetVariable(_) | Statement::ResetVariable(_) => Ok(Schema::empty()),
                Statement::TransactionStart(_) | Statement::TransactionEnd(_) => Err(Error::new(
                    ErrorCode::UnsupportedOperator,
                    "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                    "Flight SQL transactions are not supported",
                )),
                Statement::Prepare(_) | Statement::Execute(_) | Statement::Deallocate(_) => Err(Error::new(
                    ErrorCode::InvalidRequest,
                    "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                    "SQL PREPARE, EXECUTE, and DEALLOCATE are not supported through Flight SQL prepared statements",
                )),
            },
            _ => Ok(plan.schema().as_arrow().clone()),
        }
    }

    fn prepared_statement_has_result_set(&self, plan: &LogicalPlan) -> bool {
        !matches!(
            plan,
            LogicalPlan::Ddl(_) | LogicalPlan::Statement(Statement::SetVariable(_) | Statement::ResetVariable(_))
        )
    }

    fn cancel_from_flight_info(&self, raw_info: &[u8]) -> Result<bool, Status> {
        let info = FlightInfo::decode(raw_info).map_err(|error| {
            cancel_invalid_argument_status(
                "Flight SQL cancel action contained malformed FlightInfo bytes",
                "flight_info_decode",
                error,
            )
        })?;
        let Some(endpoint) = info.endpoint.first() else {
            return Ok(false);
        };
        let Some(ticket) = endpoint.ticket.as_ref() else {
            return Ok(false);
        };
        let any = arrow_flight::sql::Any::decode(&*ticket.ticket).map_err(|error| {
            cancel_invalid_argument_status(
                "Flight SQL cancel action contained a malformed ticket envelope",
                "ticket_any_decode",
                error,
            )
        })?;
        // The ticket may be either a `TicketStatementQuery` (issued by
        // `get_flight_info_statement`) or a `CommandPreparedStatementQuery`
        // (issued by `get_flight_info_prepared_statement`). Both encode the
        // per-execution `query_id` UTF-8 bytes in their handle field; once
        // extracted, the cancellation path is identical.
        let query_id_bytes = if let Some(statement_ticket) = any.unpack::<TicketStatementQuery>().map_err(|error| {
            cancel_invalid_argument_status(
                "Flight SQL cancel action could not unpack statement ticket",
                "statement_ticket_unpack",
                error,
            )
        })? {
            statement_ticket.statement_handle
        } else if let Some(prepared_ticket) = any.unpack::<CommandPreparedStatementQuery>().map_err(|error| {
            cancel_invalid_argument_status(
                "Flight SQL cancel action could not unpack prepared-statement ticket",
                "prepared_ticket_unpack",
                error,
            )
        })? {
            prepared_ticket.prepared_statement_handle
        } else {
            return Ok(false);
        };
        let query_id = std::str::from_utf8(&query_id_bytes).map_err(|error| {
            cancel_invalid_argument_status(
                "Flight SQL cancel action contained a non-UTF-8 query id",
                "query_id_utf8",
                error,
            )
        })?;
        if let Some((_, ticket)) = self.tickets.remove(query_id) {
            ticket.native_cancellation.cancel();
            let error = cancellation_error(&self.shutdown_token);
            self.observability.finish_failed(query_id, "cancel_query", CapturedError::from_adapter_error(&error));
            self.observability.seal(query_id);
            tracing::info!(query_id, outcome = "cancelled", stage = "ticket", "cancelled Flight SQL statement ticket");
            return Ok(true);
        }
        let cancelled = self.cancellations.cancel(query_id);
        if cancelled {
            let error = cancellation_error(&self.shutdown_token);
            self.observability.finish_failed(query_id, "cancel_query", CapturedError::from_adapter_error(&error));
            self.observability.seal(query_id);
        }
        Ok(cancelled)
    }
}

fn cancel_invalid_argument_status(
    message: &'static str,
    stage: &'static str,
    source: impl std::error::Error + Send + Sync + 'static,
) -> Status {
    adapter_error_status(
        Error::new(ErrorCode::InvalidRequest, "datafusion_nexus.server.flight.cancel_query.decode", message)
            .with_fact("cancel_decode_stage", stage)
            .with_source(source),
    )
}

fn ddl_register_table_target(plan: &LogicalPlan) -> Option<(&TableReference, &'static str)> {
    let LogicalPlan::Ddl(ddl) = plan else {
        return None;
    };
    match ddl {
        DdlStatement::CreateExternalTable(statement) => Some((&statement.name, "CreateExternalTable")),
        DdlStatement::CreateMemoryTable(statement) => Some((&statement.name, "CreateMemoryTable")),
        DdlStatement::CreateView(statement) => Some((&statement.name, "CreateView")),
        _ => None,
    }
}

fn datafusion_error_contains_message(error: &DataFusionError, needle: &str) -> bool {
    if error.to_string().contains(needle) {
        return true;
    }
    match error {
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => {
            datafusion_error_contains_message(source, needle)
        },
        DataFusionError::Shared(source) => datafusion_error_contains_message(source, needle),
        DataFusionError::Collection(errors) => {
            errors.iter().any(|error| datafusion_error_contains_message(error, needle))
        },
        _ => false,
    }
}

pub(super) fn request_correlation_id<T>(request: &Request<T>) -> Result<Option<String>, Status> {
    for key in CORRELATION_ID_METADATA_ALIASES {
        let Some(value) = request.metadata().get(*key) else {
            continue;
        };
        let value = value
            .to_str()
            .map_err(|error| Status::invalid_argument(format!("{key} metadata is not valid ASCII: {error}")))?
            .trim();
        if value.is_empty() {
            continue;
        }
        if value.len() > MAX_CORRELATION_ID_LEN {
            return Err(Status::invalid_argument(format!("{key} metadata exceeds {MAX_CORRELATION_ID_LEN} bytes")));
        }
        return Ok(Some(value.to_string()));
    }
    Ok(None)
}

pub(super) fn attach_query_response_metadata<T>(
    response: &mut Response<T>,
    query_id: &str,
    correlation_id: Option<&str>,
) {
    if let Ok(value) = MetadataValue::try_from(query_id) {
        response.metadata_mut().insert(QUERY_ID_METADATA_KEY, value);
    }
    if let Some(correlation_id) = correlation_id
        && let Ok(value) = MetadataValue::try_from(correlation_id)
    {
        response.metadata_mut().insert(CORRELATION_ID_METADATA_KEY, value);
    }
}

fn finish_adapter_error_status_with_correlation_id(
    observability: &QueryObservability,
    query_id: &str,
    correlation_id: Option<&str>,
    stage: &'static str,
    error: Error,
) -> Status {
    observability.finish_failed(query_id, stage, CapturedError::from_adapter_error(&error));
    status_with_query_id_and_correlation_id(adapter_error_status(error), query_id, correlation_id)
}

async fn reject_drop_object_type_mismatch(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    operation: &'static str,
) -> crate::Result<()> {
    let (object_name, expected_table_type) = match plan {
        LogicalPlan::Ddl(DdlStatement::DropTable(drop_table)) => (&drop_table.name, TableType::Base),
        LogicalPlan::Ddl(DdlStatement::DropView(drop_view)) => (&drop_view.name, TableType::View),
        _ => return Ok(()),
    };
    let table_name = object_name.table().to_string();
    let Ok(schema) = ctx.state().schema_for_ref(object_name.clone()) else {
        return Ok(());
    };
    let Ok(Some(actual_table_type)) = schema.table_type(&table_name).await else {
        return Ok(());
    };
    if actual_table_type == expected_table_type {
        return Ok(());
    };

    let object_name = object_name.to_string();
    let actual_label = table_type_client_label(actual_table_type);
    let expected_label = table_type_client_label(expected_table_type);
    let suggested_drop = drop_statement_for_table_type(actual_table_type);
    Err(Error::new(
        ErrorCode::InvalidRequest,
        operation,
        format!(
            "Object '{object_name}' is a {actual_label}, not a {expected_label}; use {suggested_drop} to remove it"
        ),
    )
    .with_fact("object_name", object_name)
    .with_fact("requested_table_type", table_type_to_flight_sql(expected_table_type))
    .with_fact("actual_table_type", table_type_to_flight_sql(actual_table_type)))
}

fn table_type_client_label(table_type: TableType) -> &'static str {
    match table_type {
        TableType::Base => "base table",
        TableType::View => "view",
        TableType::Temporary => "temporary table",
    }
}

fn drop_statement_for_table_type(table_type: TableType) -> &'static str {
    match table_type {
        TableType::Base | TableType::Temporary => "DROP TABLE",
        TableType::View => "DROP VIEW",
    }
}

#[cfg(test)]
mod correlation_tests {
    use super::*;
    use crate::{
        GpuFallbackPolicy,
        server::auth::{AuthMode, TokenStore},
    };

    fn test_service() -> DatafusionCudfFlightSqlService {
        DatafusionCudfFlightSqlService::new(
            Arc::new(SessionContext::new()),
            CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
            ServerSessionProfile::Native,
            Arc::new(QueryAdmission::new(1).expect("test admission limit")),
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
            #[cfg(feature = "cugraph")]
            None,
        )
    }

    #[test]
    fn request_correlation_id_accepts_server_key_and_alias() {
        let mut request = Request::new(());
        request.metadata_mut().insert(CORRELATION_ID_METADATA_KEY, MetadataValue::try_from("corr-123").unwrap());

        assert_eq!(request_correlation_id(&request).unwrap().as_deref(), Some("corr-123"));

        let mut alias_request = Request::new(());
        alias_request.metadata_mut().insert("x-correlation-id", MetadataValue::try_from("corr-alias").unwrap());

        assert_eq!(request_correlation_id(&alias_request).unwrap().as_deref(), Some("corr-alias"));
    }

    #[test]
    fn query_response_metadata_echoes_query_and_correlation_id() {
        let mut response = Response::new(());

        attach_query_response_metadata(&mut response, "query-1", Some("corr-1"));

        assert_eq!(
            response.metadata().get(QUERY_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("query-1")
        );
        assert_eq!(
            response.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("corr-1")
        );
    }

    #[test]
    fn status_metadata_echoes_query_and_correlation_id() {
        let status = status_with_query_id_and_correlation_id(Status::unavailable("busy"), "query-1", Some("corr-1"));

        assert_eq!(status.metadata().get(QUERY_ID_METADATA_KEY).and_then(|value| value.to_str().ok()), Some("query-1"));
        assert_eq!(
            status.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("corr-1")
        );
    }

    #[tokio::test]
    async fn get_flight_info_statement_echoes_correlation_id_and_retains_it_in_stats() {
        let service = test_service();
        let mut request = Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new()));
        request.metadata_mut().insert(CORRELATION_ID_METADATA_KEY, MetadataValue::try_from("client-corr-42").unwrap());

        let response = service
            .get_flight_info_statement(
                CommandStatementQuery { query: "SELECT 1 AS one".to_string(), transaction_id: None },
                request,
            )
            .await
            .expect("statement flight info should succeed");
        let query_id = response
            .metadata()
            .get(QUERY_ID_METADATA_KEY)
            .and_then(|value| value.to_str().ok())
            .expect("query id response metadata")
            .to_string();

        assert_eq!(
            response.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("client-corr-42")
        );
        let ticket = service.tickets.get(&query_id).expect("statement ticket should be retained");
        assert_eq!(ticket.correlation_id.as_deref(), Some("client-corr-42"));
        drop(ticket);
        let payload = service
            .observability
            .get(&query_id)
            .expect("query stats capture should be retained")
            .to_query_stats_payload();
        assert!(payload.contains("# correlation_id=client-corr-42\n"), "{payload}");
    }

    #[tokio::test]
    async fn do_get_only_correlation_id_is_retained_in_stats() {
        let service = test_service();
        let response = service
            .get_flight_info_statement(
                CommandStatementQuery { query: "SELECT 1 AS one".to_string(), transaction_id: None },
                Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
            )
            .await
            .expect("statement flight info should succeed");
        let query_id = response
            .metadata()
            .get(QUERY_ID_METADATA_KEY)
            .and_then(|value| value.to_str().ok())
            .expect("query id response metadata")
            .to_string();
        assert!(
            !service
                .observability
                .get(&query_id)
                .expect("query stats capture should be retained")
                .to_query_stats_payload()
                .contains("# correlation_id=")
        );

        let response = service
            .execute_ticket(query_id.clone(), Some("do-get-corr-7".to_string()))
            .await
            .expect("statement do_get should succeed");

        assert_eq!(
            response.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("do-get-corr-7")
        );
        let payload = service
            .observability
            .get(&query_id)
            .expect("query stats capture should be retained")
            .to_query_stats_payload();
        assert!(payload.contains("# correlation_id=do-get-corr-7\n"), "{payload}");
    }

    #[tokio::test]
    async fn dropping_do_get_stream_cancels_statement() {
        let service = test_service();
        let response = service
            .get_flight_info_statement(
                CommandStatementQuery { query: "SELECT 1 AS one".to_string(), transaction_id: None },
                Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
            )
            .await
            .expect("statement flight info should succeed");
        let query_id = response
            .metadata()
            .get(QUERY_ID_METADATA_KEY)
            .and_then(|value| value.to_str().ok())
            .expect("query id response metadata")
            .to_string();
        let native_cancellation = {
            let ticket = service.tickets.get(&query_id).expect("statement ticket should be retained");
            ticket.native_cancellation.clone()
        };

        let response =
            service.execute_ticket(query_id.clone(), None).await.expect("statement do_get should return a lazy stream");
        assert!(!native_cancellation.is_cancelled(), "stream creation must not cancel the query");

        drop(response);

        assert!(native_cancellation.is_cancelled(), "dropping the DoGet stream must cancel native execution");
        assert_eq!(service.cancellations.len(), 0, "dropped streams must leave no cancellable registry entry");
        let payload = service
            .observability
            .get(&query_id)
            .expect("query stats capture should be retained")
            .to_query_stats_payload();
        assert!(payload.contains("# outcome=failed"), "{payload}");
        assert!(payload.contains("# sealed=true"), "{payload}");
        assert!(payload.contains("[structured_error]"), "{payload}");
        assert!(payload.contains("code=cancelled"), "{payload}");
        assert!(payload.contains("operation=datafusion_nexus.server.flight.cancel_query"), "{payload}");
    }

    #[tokio::test]
    async fn get_admission_stats_action_returns_current_admission_payload() {
        let service = test_service();

        let action_types = service
            .list_custom_actions()
            .await
            .expect("custom actions should be listed")
            .into_iter()
            .map(|action| action.unwrap().r#type)
            .collect::<Vec<_>>();
        assert!(action_types.contains(&GET_QUERY_STATS_ACTION.to_string()), "{action_types:?}");
        assert!(action_types.contains(&GET_ADMISSION_STATS_ACTION.to_string()), "{action_types:?}");

        let response = service
            .do_action_fallback(Request::new(arrow_flight::Action {
                r#type: GET_ADMISSION_STATS_ACTION.to_string(),
                body: Vec::new().into(),
            }))
            .await
            .expect("GetAdmissionStats action should succeed");
        let results = response.into_inner().try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(results.len(), 1);
        let payload = String::from_utf8(results[0].body.to_vec()).unwrap();

        assert!(payload.contains("# datafusion_nexus_admission_stats=1\n"), "{payload}");
        assert!(payload.contains("# backend=semaphore\n"), "{payload}");
        assert!(payload.contains("[admission]\nactive_queries\tqueued_queries\tnext_tenant_id\n"), "{payload}");
        assert!(payload.contains("[semaphore]\nmax_in_flight\tavailable_permits\n1\t1\n"), "{payload}");
    }
}

#[cfg(all(test, feature = "iceberg"))]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use datafusion::{datasource::MemTable, prelude::SessionConfig};
    use futures::future::BoxFuture;

    use super::*;
    use crate::{
        GpuFallbackPolicy,
        server::{
            auth::{AuthMode, TokenStore},
            error::{ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_STATUS_METADATA_KEY},
        },
        table_format::{
            iceberg::CudfIcebergTableProvider,
            scan::{
                TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource,
                TableFormatDataFile, TableFormatDataFileFormat, TableFormatPlanner, TableFormatReadCapability,
                TableFormatScanPlan, TableFormatScanRequest,
            },
        },
    };

    type PlannerFuture<'a> = BoxFuture<'a, crate::Result<TableFormatScanPlan>>;

    #[derive(Debug)]
    struct SleepingPlanner {
        plan: TableFormatScanPlan,
        delay: Duration,
    }

    impl SleepingPlanner {
        fn new(plan: TableFormatScanPlan, delay: Duration) -> Self {
            Self { plan, delay }
        }
    }

    impl TableFormatPlanner for SleepingPlanner {
        fn plan_scan(&self, _request: TableFormatScanRequest) -> PlannerFuture<'_> {
            let plan = self.plan.clone();
            let delay = self.delay;
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                Ok(plan)
            })
        }
    }

    #[test]
    fn cancel_invalid_argument_status_carries_adapter_metadata() {
        let invalid_query_id = vec![0xff];
        let status = cancel_invalid_argument_status(
            "Flight SQL cancel action contained a non-UTF-8 query id",
            "query_id_utf8",
            std::str::from_utf8(&invalid_query_id).unwrap_err(),
        );

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(status.metadata().get(ERROR_CODE_METADATA_KEY).unwrap().to_str().unwrap(), "invalid_request");
        assert_eq!(status.metadata().get(ERROR_KIND_METADATA_KEY).unwrap().to_str().unwrap(), "invalid_input");
        assert_eq!(status.metadata().get(ERROR_STATUS_METADATA_KEY).unwrap().to_str().unwrap(), "permanent");
        assert_eq!(status.message(), "Flight SQL cancel action contained a non-UTF-8 query id");
    }

    #[derive(Debug)]
    struct CountingPlanner {
        plan: TableFormatScanPlan,
        requests: AtomicUsize,
    }

    impl CountingPlanner {
        fn new(plan: TableFormatScanPlan) -> Self {
            Self { plan, requests: AtomicUsize::new(0) }
        }

        fn request_count(&self) -> usize {
            self.requests.load(Ordering::SeqCst)
        }
    }

    impl TableFormatPlanner for CountingPlanner {
        fn plan_scan(&self, _request: TableFormatScanRequest) -> PlannerFuture<'_> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            let plan = self.plan.clone();
            Box::pin(async move { Ok(plan) })
        }
    }

    #[tokio::test]
    async fn unsupported_table_format_distinct_list_fast_fails_before_scan_planning() {
        let schema = list_string_schema();
        let planner = Arc::new(CountingPlanner::new(list_string_scan_plan(Arc::clone(&schema))));
        assert_unsupported_table_format_operator_fast_fails_before_scan_planning(
            Arc::clone(&schema),
            planner,
            "SELECT DISTINCT \"pCodes\" FROM iceberg_t",
            "distinct",
            "pCodes",
            "List(",
        )
        .await;
    }

    #[tokio::test]
    async fn unsupported_table_format_group_by_list_fast_fails_before_scan_planning() {
        let schema = list_string_schema();
        let planner = Arc::new(CountingPlanner::new(list_string_scan_plan(Arc::clone(&schema))));
        assert_unsupported_table_format_operator_fast_fails_before_scan_planning(
            Arc::clone(&schema),
            planner,
            "SELECT \"pCodes\", COUNT(*) AS n FROM iceberg_t GROUP BY \"pCodes\"",
            "group_by",
            "pCodes",
            "List(",
        )
        .await;
    }

    #[tokio::test]
    async fn unsupported_table_format_count_distinct_list_fast_fails_before_scan_planning() {
        let schema = list_string_schema();
        let planner = Arc::new(CountingPlanner::new(list_string_scan_plan(Arc::clone(&schema))));
        assert_unsupported_table_format_operator_fast_fails_before_scan_planning(
            Arc::clone(&schema),
            planner,
            "SELECT COUNT(DISTINCT \"pCodes\") AS n FROM iceberg_t",
            "aggregate_distinct",
            "pCodes",
            "List(",
        )
        .await;
    }

    #[tokio::test]
    async fn unsupported_table_format_distinct_map_fast_fails_before_scan_planning() {
        let schema = map_schema();
        let planner = Arc::new(CountingPlanner::new(map_scan_plan(Arc::clone(&schema))));
        assert_unsupported_table_format_operator_fast_fails_before_scan_planning(
            Arc::clone(&schema),
            planner,
            "SELECT DISTINCT \"countDetail\" FROM iceberg_t",
            "distinct",
            "countDetail",
            "Map(",
        )
        .await;
    }

    #[tokio::test]
    async fn unsupported_non_iceberg_operator_column_in_mixed_plan_does_not_fast_fail() {
        let iceberg_schema = iceberg_test_schema();
        let planner = Arc::new(CountingPlanner::new(native_admissible_scan_plan(Arc::clone(&iceberg_schema))));
        let provider = Arc::new(
            CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&iceberg_schema), planner.clone())
                .with_catalog_type("local"),
        );
        let ctx = SessionContext::new();
        ctx.register_table("iceberg_t", provider).expect("test registers iceberg table");

        let memory_schema = Arc::new(Schema::new(vec![Field::new("memCodes", list_string_type(), true)]));
        let memory_table = Arc::new(MemTable::try_new(Arc::clone(&memory_schema), vec![vec![]]).unwrap());
        ctx.register_table("mem_t", memory_table).expect("test registers memory table");

        let df = ctx
            .sql("SELECT DISTINCT m.\"memCodes\" FROM mem_t AS m CROSS JOIN iceberg_t AS i")
            .await
            .expect("logical plan builds");

        assert!(fast_fail_unsupported_table_format_operator_plan(df.logical_plan()).is_none());
        assert_eq!(planner.request_count(), 0, "logical fast-fail proof must not trigger Iceberg scan planning");
    }

    #[tokio::test]
    async fn unsupported_iceberg_operator_column_in_mixed_plan_fast_fails() {
        let iceberg_schema = map_schema();
        let planner = Arc::new(CountingPlanner::new(map_scan_plan(Arc::clone(&iceberg_schema))));
        let provider = Arc::new(
            CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&iceberg_schema), planner.clone())
                .with_catalog_type("local"),
        );
        let ctx = Arc::new(SessionContext::new());
        ctx.register_table("iceberg_t", provider).expect("test registers iceberg table");

        let memory_schema = Arc::new(Schema::new(vec![Field::new("mem_id", DataType::Int64, false)]));
        let memory_table = Arc::new(MemTable::try_new(Arc::clone(&memory_schema), vec![vec![]]).unwrap());
        ctx.register_table("mem_t", memory_table).expect("test registers memory table");

        let service = DatafusionCudfFlightSqlService::new(
            ctx,
            CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
            ServerSessionProfile::Native,
            Arc::new(QueryAdmission::new(1).expect("test admission limit")),
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
            #[cfg(feature = "cugraph")]
            None,
        );

        let status = issue_statement_ticket(
            &service,
            "SELECT DISTINCT i.\"countDetail\" FROM iceberg_t AS i CROSS JOIN mem_t AS m",
        )
        .await
        .expect_err("unsupported Iceberg operator column should fast-fail through mixed joins");

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_status_metadata(&status, "native_candidate_rejected", "planning", "permanent");
        assert!(status.message().contains("operator=distinct"), "status={status:?}");
        assert!(status.message().contains("column=countDetail"), "status={status:?}");
        assert!(status.message().contains("type=Map("), "status={status:?}");
        assert_eq!(planner.request_count(), 0, "Iceberg scan planning must be skipped");
        assert!(service.tickets.is_empty(), "fast-failed statements should not issue execution tickets");
    }

    async fn assert_unsupported_table_format_operator_fast_fails_before_scan_planning(
        schema: Arc<Schema>,
        planner: Arc<CountingPlanner>,
        sql: &str,
        operator: &str,
        column: &str,
        type_prefix: &str,
    ) {
        let provider = Arc::new(
            CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone())
                .with_catalog_type("local"),
        );
        let ctx = Arc::new(SessionContext::new());
        ctx.register_table("iceberg_t", provider).expect("test registers table once");

        let service = DatafusionCudfFlightSqlService::new(
            ctx,
            CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
            ServerSessionProfile::Native,
            Arc::new(QueryAdmission::new(1).expect("test admission limit")),
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
            #[cfg(feature = "cugraph")]
            None,
        );

        let status = issue_statement_ticket(&service, sql)
            .await
            .expect_err("unsupported table-format operator should fail before table scan planning");

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_status_metadata(&status, "native_candidate_rejected", "planning", "permanent");
        assert!(status.message().contains(&format!("operator={operator}")), "status={status:?}");
        assert!(status.message().contains(&format!("column={column}")), "status={status:?}");
        assert!(status.message().contains(&format!("type={type_prefix}")), "status={status:?}");
        assert_eq!(planner.request_count(), 0, "Iceberg scan planning must be skipped");
        assert!(service.tickets.is_empty(), "fast-failed statements should not issue execution tickets");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn planning_timeout_consumes_statement_ticket_without_holding_admission() {
        let schema = iceberg_test_schema();
        let planner =
            Arc::new(SleepingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema)), Duration::from_millis(50)));
        let provider = Arc::new(
            CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner)
                .with_catalog_type("local")
                .with_scan_planning_timeout(Some(Duration::from_millis(1))),
        );
        let ctx = Arc::new(SessionContext::new());
        ctx.register_table("iceberg_t", provider).expect("test registers table once");

        let service = DatafusionCudfFlightSqlService::new(
            ctx,
            CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
            ServerSessionProfile::Native,
            Arc::new(QueryAdmission::new(1).expect("test admission limit")),
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
            #[cfg(feature = "cugraph")]
            None,
        );

        issue_statement_ticket(&service, "SELECT COUNT(*) AS c FROM iceberg_t")
            .await
            .expect("logical planning should issue a ticket before table scan planning");
        let query_id = only_ticket_query_id(&service);
        let status = match service.execute_ticket(query_id, None).await {
            Ok(_) => panic!("physical scan planning should time out"),
            Err(status) => status,
        };
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_status_metadata(&status, "planning_timeout", "planning", "permanent");
        assert!(service.tickets.is_empty(), "timed-out statement ticket should be consumed");

        tokio::time::timeout(Duration::from_secs(1), issue_statement_ticket(&service, "SELECT 1 AS one"))
            .await
            .expect("second statement should not wait on the timed-out query")
            .expect("second statement should issue a ticket");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_ddl_read_only_schema_failure_captures_catalog_target_facts() {
        let ctx = Arc::new(SessionContext::new_with_config(
            SessionConfig::new().with_default_catalog_and_schema("glue", "nexus_graph"),
        ));
        ctx.register_catalog(
            "glue",
            Arc::new(ReadOnlyCatalog::new("nexus_graph", [("edges_by_src", read_only_edge_table())])),
        );
        let service = DatafusionCudfFlightSqlService::new(
            ctx,
            CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
            ServerSessionProfile::Native,
            Arc::new(QueryAdmission::new(1).expect("test admission limit")),
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
            #[cfg(feature = "cugraph")]
            None,
        )
        .with_catalog_diagnostics(ServerCatalogDiagnostics {
            iceberg_enabled: true,
            workspace_catalog: Some("datafusion".to_string()),
            workspace_schema: Some("public".to_string()),
        });
        let query_id = "read-only-ddl-target".to_string();
        let sql = "CREATE VIEW target_edges AS SELECT src, dst FROM edges_by_src".to_string();
        service.observability.start_query_with_correlation_id(query_id.clone(), sql.clone(), None);

        let status = service
            .execute_update_sql(query_id.clone(), sql, None, UpdateStatementKind::Statement)
            .await
            .expect_err("read-only source schema should reject DDL registration");

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_status_metadata(&status, "datafusion_planning", "planning", "permanent");
        assert!(status.message().contains("glue.nexus_graph"), "status={status:?}");
        let payload = service
            .observability
            .get(&query_id)
            .expect("failed DDL update should retain query stats")
            .to_query_stats_payload();
        assert!(payload.contains("operation=datafusion_nexus.server.flight.do_put_statement_update.sql"), "{payload}");
        assert!(payload.contains("fact.target_catalog=glue"), "{payload}");
        assert!(payload.contains("fact.target_schema=nexus_graph"), "{payload}");
        assert!(payload.contains("fact.target_table=target_edges"), "{payload}");
        assert!(payload.contains("fact.statement_kind=CreateView"), "{payload}");
        assert!(payload.contains("fact.iceberg_enabled=true"), "{payload}");
        assert!(payload.contains("fact.workspace_catalog=datafusion"), "{payload}");
        assert!(payload.contains("fact.workspace_schema=public"), "{payload}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_set_datafusion_nexus_key_is_rejected_loudly() {
        let ctx = Arc::new(SessionContext::new());
        let service = DatafusionCudfFlightSqlService::new(
            ctx,
            CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
            ServerSessionProfile::Native,
            Arc::new(QueryAdmission::new(1).expect("test admission limit")),
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
            #[cfg(feature = "cugraph")]
            None,
        );
        let query_id = "reject-nexus-set".to_string();
        let sql = "SET datafusion_nexus.request.requested_grant_bytes = '1048576'".to_string();
        service.observability.start_query_with_correlation_id(query_id.clone(), sql.clone(), None);

        let status = service
            .execute_update_sql(query_id.clone(), sql, None, UpdateStatementKind::Statement)
            .await
            .expect_err("Flight update SET for datafusion-nexus keys should be rejected");

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_status_metadata(&status, "invalid_request", "invalid_input", "permanent");
        assert!(status.message().contains("server or tenant defaults"), "status={status:?}");
        let payload = service
            .observability
            .get(&query_id)
            .expect("failed SET update should retain query stats")
            .to_query_stats_payload();
        assert!(payload.contains("fact.reason=flight_session_scoped_datafusion_nexus_keys_unsupported"), "{payload}");
        assert!(payload.contains("fact.config_key=datafusion_nexus.request.requested_grant_bytes"), "{payload}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_set_plain_datafusion_key_still_succeeds() {
        let ctx = Arc::new(SessionContext::new());
        let service = DatafusionCudfFlightSqlService::new(
            ctx,
            CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
            ServerSessionProfile::Native,
            Arc::new(QueryAdmission::new(1).expect("test admission limit")),
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
            #[cfg(feature = "cugraph")]
            None,
        );
        let query_id = "plain-datafusion-set".to_string();
        let sql = "SET datafusion.execution.batch_size = '1024'".to_string();
        service.observability.start_query_with_correlation_id(query_id.clone(), sql.clone(), None);

        let count = service
            .execute_update_sql(query_id, sql, None, UpdateStatementKind::Statement)
            .await
            .expect("plain DataFusion SET should keep existing update-lane behavior");

        assert_eq!(count, NO_UPDATED_RECORDS_COUNT);
    }

    async fn issue_statement_ticket(
        service: &DatafusionCudfFlightSqlService,
        sql: &str,
    ) -> Result<Response<FlightInfo>, Status> {
        service
            .get_flight_info_statement(
                CommandStatementQuery { query: sql.to_string(), transaction_id: None },
                Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
            )
            .await
    }

    fn only_ticket_query_id(service: &DatafusionCudfFlightSqlService) -> String {
        let query_ids = service.tickets.iter().map(|entry| entry.key().clone()).collect::<Vec<_>>();
        assert_eq!(query_ids.len(), 1, "expected exactly one live statement ticket, got {query_ids:?}");
        query_ids[0].clone()
    }

    fn assert_status_metadata(status: &Status, code: &str, kind: &str, error_status: &str) {
        assert_eq!(status.metadata().get(ERROR_CODE_METADATA_KEY).and_then(|value| value.to_str().ok()), Some(code));
        assert_eq!(status.metadata().get(ERROR_KIND_METADATA_KEY).and_then(|value| value.to_str().ok()), Some(kind));
        assert_eq!(
            status.metadata().get(ERROR_STATUS_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some(error_status)
        );
    }

    #[derive(Debug)]
    struct ReadOnlyCatalog {
        schema_name: String,
        schema: Arc<ReadOnlySchema>,
    }

    impl ReadOnlyCatalog {
        fn new(
            schema_name: impl Into<String>,
            tables: impl IntoIterator<Item = (&'static str, Arc<dyn datafusion::datasource::TableProvider>)>,
        ) -> Self {
            Self { schema_name: schema_name.into(), schema: Arc::new(ReadOnlySchema::new(tables)) }
        }
    }

    impl datafusion::catalog::CatalogProvider for ReadOnlyCatalog {
        fn schema_names(&self) -> Vec<String> {
            vec![self.schema_name.clone()]
        }

        fn schema(&self, name: &str) -> Option<Arc<dyn datafusion::catalog::SchemaProvider>> {
            (name == self.schema_name).then(|| Arc::clone(&self.schema) as Arc<dyn datafusion::catalog::SchemaProvider>)
        }
    }

    #[derive(Debug)]
    struct ReadOnlySchema {
        tables: BTreeMap<String, Arc<dyn datafusion::datasource::TableProvider>>,
    }

    impl ReadOnlySchema {
        fn new(
            tables: impl IntoIterator<Item = (&'static str, Arc<dyn datafusion::datasource::TableProvider>)>,
        ) -> Self {
            Self { tables: tables.into_iter().map(|(name, table)| (name.to_string(), table)).collect() }
        }
    }

    #[async_trait::async_trait]
    impl datafusion::catalog::SchemaProvider for ReadOnlySchema {
        fn table_names(&self) -> Vec<String> {
            self.tables.keys().cloned().collect()
        }

        async fn table(
            &self,
            name: &str,
        ) -> datafusion::common::Result<Option<Arc<dyn datafusion::datasource::TableProvider>>> {
            Ok(self.tables.get(name).cloned())
        }

        async fn table_type(&self, name: &str) -> datafusion::common::Result<Option<TableType>> {
            Ok(self.tables.get(name).map(|table| table.table_type()))
        }

        fn table_exist(&self, name: &str) -> bool {
            self.tables.contains_key(name)
        }
    }

    fn read_only_edge_table() -> Arc<dyn datafusion::datasource::TableProvider> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![0, 1])),
                Arc::new(arrow::array::Int64Array::from(vec![1, 2])),
            ],
        )
        .unwrap();
        Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())
    }

    fn iceberg_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
    }

    fn native_admissible_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
        TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_snapshot_id("12345")
            .with_catalog_type("local")
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(schema)
            .with_data_file(
                TableFormatDataFile::new(
                    "data-0001",
                    "/tmp/iceberg/data-0001.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::local_file_system(),
                )
                .with_size_bytes(4096)
                .with_row_count(11),
            )
            .with_projected_column(TableFormatColumnMapping::new(
                Some(1),
                "id",
                Some("id"),
                DataType::Int64,
                false,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                TableFormatColumnRole::Projected,
            ))
    }

    fn list_string_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("pCodes", list_string_type(), true)]))
    }

    fn list_string_type() -> DataType {
        DataType::List(Arc::new(Field::new("element", DataType::Utf8, true)))
    }

    fn list_string_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
        let data_type = list_string_type();
        single_column_scan_plan(schema, "pCodes", data_type)
    }

    fn map_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("countDetail", map_type(), true)]))
    }

    fn map_type() -> DataType {
        DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(arrow::datatypes::Fields::from(vec![
                    Arc::new(Field::new("key", DataType::Utf8, false)),
                    Arc::new(Field::new("value", DataType::Int32, true)),
                ])),
                false,
            )),
            false,
        )
    }

    fn map_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
        single_column_scan_plan(schema, "countDetail", map_type())
    }

    fn single_column_scan_plan(
        schema: Arc<Schema>,
        column_name: &'static str,
        data_type: DataType,
    ) -> TableFormatScanPlan {
        TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_snapshot_id("12345")
            .with_catalog_type("local")
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(schema)
            .with_data_file(
                TableFormatDataFile::new(
                    "data-0001",
                    "/tmp/iceberg/data-0001.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::local_file_system(),
                )
                .with_size_bytes(4096)
                .with_row_count(11),
            )
            .with_projected_column(TableFormatColumnMapping::new(
                Some(1),
                column_name,
                Some(column_name),
                data_type,
                true,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                TableFormatColumnRole::Projected,
            ))
    }
}
