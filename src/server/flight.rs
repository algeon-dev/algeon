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
    datasource::TableType,
    error::DataFusionError,
    execution::SessionStateBuilder,
    logical_expr::{DdlStatement, LogicalPlan, Statement},
    prelude::SessionContext,
};
use futures::{Stream, StreamExt, TryStreamExt};
use nexus_query_engine::ExecutionCancellation;
use prost::Message;
use tokio::sync::OwnedSemaphorePermit;
use tokio_util::sync::CancellationToken;
use tonic::{Code, Request, Response, Status, Streaming, metadata::MetadataValue};
use uuid::Uuid;

use super::{
    admission::QueryAdmission,
    auth::{AuthInterceptor, handle_handshake},
    cancel::QueryCancellation,
    config::ServerSessionProfile,
    diagnostics::{log_leaf_guard_rejection, log_native_planning_report},
    error::{ServerError, datafusion_error_to_status_for_operation},
    table_format_operator_fast_fail::fast_fail_unsupported_table_format_operator_plan,
};
use crate::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, Error, ErrorCode};

mod metadata;
mod rpc;
mod status;
mod tickets;

use metadata::*;
use status::*;
use tickets::*;

const FLIGHT_SQL_CANCEL_RESULT_CANCELLING: i32 = 2;
const FLIGHT_SQL_CANCEL_RESULT_NOT_CANCELLABLE: i32 = 3;
const UNKNOWN_UPDATE_RECORD_COUNT: i64 = -1;
const SQL_ANSI92_ENTRY_SQL_BIT_INDEX: i32 = 0;
const QUERY_ID_METADATA_KEY: &str = "x-datafusion-nexus-query-id";

/// Hard cap on live prepared-statement handles across all connections.
/// Bounds the worst case for clients that leak handles by disconnecting
/// without calling `ClosePreparedStatement`. Sized for DataGrip-class
/// clients (a few hundred concurrent handles); raise when a real workload
/// demands it. Hitting the cap returns `RESOURCE_EXHAUSTED` rather than
/// silently evicting another connection's prepared statement.
const MAX_PREPARED_STATEMENTS: usize = 1024;
const ADMISSION_SWEEP_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// A live Flight SQL prepared statement.
///
/// JDBC clients such as DataGrip prepare a statement once and then re-run the
/// held handle. A DDL statement prepares with an empty result schema, which the
/// Arrow Flight SQL JDBC driver classifies as `IS_DML` and executes through
/// `executeUpdate` (the update channel); DataGrip re-runs that same handle, so
/// the second execution would otherwise hit DataFusion's "already exists" /
/// "not found" catalog error. We deduplicate no-result-set DDL per handle: the
/// first execution runs, later executions of the *same* handle are no-ops.
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
    /// by [`crate::server::serve`] on SIGINT/SIGTERM so any statement that
    /// has been admitted but not yet registered (or any statement still to
    /// register) sees a pre-cancelled token and stops at the first await.
    shutdown_token: CancellationToken,
    /// Bearer-token validator + handshake credential store. Cloned into
    /// the service from [`crate::server::serve`] so the same
    /// [`TokenStore`] is shared between the handshake mint path and the
    /// per-RPC validate path.
    auth: AuthInterceptor,
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
        max_in_flight: usize,
        statement_ticket_ttl: Duration,
        auth: AuthInterceptor,
    ) -> Self {
        Self {
            ctx,
            native_optimizer_config,
            session_profile,
            admission: Arc::new(QueryAdmission::new(max_in_flight).expect("validated server admission limit")),
            cancellations: Arc::default(),
            tickets: Arc::default(),
            prepared_statements: Arc::default(),
            statement_ticket_ttl,
            shutdown_token: CancellationToken::new(),
            auth,
        }
    }

    /// Validate the bearer token on a tonic request. Returns the
    /// authenticated [`super::auth::Principal`] when auth is enabled,
    /// or `None` when the server is running in
    /// [`AuthMode::Disabled`]. Errors as `Status::unauthenticated`.
    fn check_auth<T>(&self, request: &Request<T>) -> Result<Option<super::auth::Principal>, Status> {
        self.auth.check(request.metadata())
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
                                tracing::warn!(
                                    catalog = %catalog_name,
                                    schema = %schema_name,
                                    table = %table_name,
                                    error = %error,
                                    "failed to load table for Flight SQL GetTables"
                                );
                                continue;
                            },
                        }
                    } else {
                        let table_type = match schema_provider.table_type(&table_name).await {
                            Ok(Some(table_type)) => table_type,
                            Ok(None) => continue,
                            Err(error) if include_schema => {
                                tracing::warn!(
                                    catalog = %catalog_name,
                                    schema = %schema_name,
                                    table = %table_name,
                                    error = %error,
                                    "failed to resolve table type for Flight SQL GetTables"
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
                                    tracing::warn!(
                                        catalog = %catalog_name,
                                        schema = %schema_name,
                                        table = %table_name,
                                        error = %error,
                                        "failed to load table for Flight SQL GetTables"
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
                        .map_err(arrow_flight_error_to_status)?;
                }
            }
        }
        let schema = builder.schema();
        let batch = builder.build().map_err(arrow_flight_error_to_status)?;
        Ok((schema, batch))
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

    fn query_planning(&self, cancellation: ExecutionCancellation) -> QueryPlanning {
        let execution_config =
            self.native_optimizer_config.native_execution_config().clone().with_cancellation(cancellation);
        let optimizer_config = self.native_optimizer_config.clone().with_native_execution_config(execution_config);
        // Retain the native rule so we can read its `last_report` for the
        // Milestone 4 planning-report log after physical optimization runs.
        let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
        let state = SessionStateBuilder::new_from_existing(self.ctx.state())
            .with_cudf_native_rule(Arc::clone(&native_rule))
            .build();
        QueryPlanning { ctx: SessionContext::new_with_state(state), native_rule }
    }

    async fn statement_flight_info(
        &self,
        query_id: String,
        sql: String,
        descriptor: FlightDescriptor,
        started_at: Instant,
        permit: OwnedSemaphorePermit,
        ticket_kind: StatementTicketKind,
        prepared_handle: Option<String>,
    ) -> Result<FlightInfo, Status> {
        // Before ticket issuance there is no per-query token, so planning
        // observes the server shutdown token directly. The native flag is still
        // shared with the optimizer rule, but `do_action_cancel_query` cannot
        // reach it until the ticket is registered.
        let native_cancellation = ExecutionCancellation::new();
        let planning = self.query_planning(native_cancellation.clone());
        let query_ctx = &planning.ctx;
        let query_state = query_ctx.state();
        let logical_plan = tokio::select! {
            result = query_state.create_logical_plan(&sql) => {
                result.map_err(|error| datafusion_error_to_status_for_operation(
                    error,
                    ticket_kind.create_logical_plan_operation(),
                ))?
            },
            _ = self.shutdown_token.cancelled() => {
                native_cancellation.cancel();
                return Err(server_shutdown_status(&query_id));
            }
        };
        verify_flight_info_plan(&logical_plan, ticket_kind)?;
        reject_drop_object_type_mismatch(query_ctx, &logical_plan, ticket_kind.create_logical_plan_operation()).await?;
        if let Some(error) = fast_fail_unsupported_table_format_operator_plan(&logical_plan) {
            return Err(status_with_query_id(adapter_error_status(error), &query_id));
        }
        let schema = logical_plan.schema().as_arrow().clone();

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
            .map_err(arrow_error_to_status)?
            .with_descriptor(descriptor)
            .with_endpoint(endpoint)
            .with_ordered(false);
        self.tickets.insert(
            query_id.clone(),
            StatementTicket {
                query_id,
                sql,
                ticket_kind,
                prepared_handle,
                native_cancellation,
                started_at,
                issued_at: Instant::now(),
                _permit: permit,
            },
        );
        Ok(info)
    }

    async fn acquire_statement_permit(&self, query_id: &str) -> Result<OwnedSemaphorePermit, Status> {
        loop {
            self.expire_stale_tickets(Instant::now());
            match self.admission.try_acquire() {
                Ok(Some(permit)) => return Ok(permit.into_inner()),
                Ok(None) => {},
                Err(error) => {
                    let status = if self.shutdown_token.is_cancelled() {
                        server_shutdown_status(query_id)
                    } else {
                        server_error_to_status(error)
                    };
                    return Err(status_with_query_id(status, query_id));
                },
            }

            let sleep_for = self.next_ticket_expiration_delay(Instant::now()).unwrap_or(ADMISSION_SWEEP_POLL_INTERVAL);
            let sleep_for = sleep_for.min(ADMISSION_SWEEP_POLL_INTERVAL);
            let acquire = self.admission.acquire();
            tokio::select! {
                result = acquire => {
                    return result
                        .map(|permit| permit.into_inner())
                        .map_err(|error| {
                            let status = if self.shutdown_token.is_cancelled() {
                                server_shutdown_status(query_id)
                            } else {
                                server_error_to_status(error)
                            };
                            status_with_query_id(status, query_id)
                        });
                },
                _ = tokio::time::sleep(sleep_for) => {},
                _ = self.shutdown_token.cancelled() => {
                    return Err(server_shutdown_status(query_id));
                },
            }
        }
    }

    fn expire_stale_tickets(&self, now: Instant) {
        let ttl = self.statement_ticket_ttl;
        self.tickets.retain(|query_id, ticket| {
            let age = now.saturating_duration_since(ticket.issued_at);
            if age < ttl {
                return true;
            }
            self.cancellations.remove(query_id);
            tracing::warn!(
                query_id = query_id.as_str(),
                ticket_age_ms = age.as_millis(),
                statement_ticket_ttl_ms = ttl.as_millis(),
                "expired abandoned Flight SQL statement ticket"
            );
            false
        });
    }

    fn next_ticket_expiration_delay(&self, now: Instant) -> Option<Duration> {
        self.tickets
            .iter()
            .map(|ticket| {
                let age = now.saturating_duration_since(ticket.issued_at);
                self.statement_ticket_ttl.saturating_sub(age)
            })
            .min()
    }

    async fn execute_ticket(&self, query_id: String) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.expire_stale_tickets(Instant::now());
        let (_, ticket) = self.tickets.remove(&query_id).ok_or_else(|| {
            let status = adapter_error_status_with_query_id(
                Error::new(
                    ErrorCode::InvalidRequest,
                    "datafusion_nexus.server.flight.execute_ticket.resolve",
                    "unknown, expired, or already consumed statement ticket",
                ),
                &query_id,
            );
            log_statement_status(&query_id, "ticket", Instant::now(), &status);
            status
        })?;

        // Derive from `shutdown_token` so cancellation cascades to statements
        // that race in between permit acquisition and registry insert.
        let token = self.shutdown_token.child_token();
        let StatementTicket {
            query_id,
            sql,
            ticket_kind,
            prepared_handle,
            native_cancellation,
            started_at,
            _permit: permit,
            ..
        } = ticket;

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
            drop(permit);
            let stream = FlightDataEncoderBuilder::new()
                .with_schema(Arc::new(Schema::empty()))
                .build(futures::stream::empty::<Result<arrow::record_batch::RecordBatch, FlightError>>())
                .map_err(move |error| status_with_query_id(Status::from(error), &query_id));
            return Ok(Response::new(Box::pin(stream)));
        }

        self.cancellations.insert(query_id.clone(), token.clone(), native_cancellation.clone());
        let native_cancellation_for_bridge = native_cancellation.clone();
        let token_for_bridge = token.clone();
        let native_cancel_bridge = tokio::spawn(async move {
            token_for_bridge.cancelled().await;
            native_cancellation_for_bridge.cancel();
        });
        let lifecycle = Arc::new(QueryLifecycle::new(query_id.clone(), started_at));
        // Construct the guard before planning so an early return from the
        // tokio::select! shutdown branches still releases the permit and
        // removes the cancellation entry on drop (avoids leaks when planning
        // or stream creation is abandoned).
        let guard = QueryExecutionGuard {
            query_id: query_id.clone(),
            cancellations: Arc::clone(&self.cancellations),
            native_cancel_bridge,
            _permit: permit,
            lifecycle: Arc::clone(&lifecycle),
        };
        let planning = self.query_planning(native_cancellation.clone());
        let query_ctx = &planning.ctx;
        let query_state = query_ctx.state();
        let logical_plan = tokio::select! {
            result = query_state.create_logical_plan(&sql) => match result {
                Ok(plan) => plan,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, ticket_kind.sql_operation()),
                        &query_id,
                    );
                    lifecycle.finish_status("planning", &status);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
                native_cancellation.cancel();
                let status = cancellation_status(&self.shutdown_token, &query_id);
                lifecycle.finish_status("planning", &status);
                return Err(status);
            },
        };
        if let Err(status) =
            reject_drop_object_type_mismatch(query_ctx, &logical_plan, ticket_kind.sql_operation()).await
        {
            let status = status_with_query_id(status, &query_id);
            lifecycle.finish_status("planning", &status);
            return Err(status);
        }
        let df = tokio::select! {
            result = query_ctx.execute_logical_plan(logical_plan) => match result {
                Ok(df) => df,
                Err(error) => {
                    // DDL execution and logical-plan wrapping happen before
                    // the physical optimizer runs, so `last_report` is empty
                    // here. The leaf-guard helper no-ops on non-leaf-guard
                    // errors, but we run it for completeness so any
                    // planning-time adapter error that does carry the
                    // leaf-guard facts gets the structured warn line.
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, ticket_kind.sql_operation()),
                        &query_id,
                    );
                    lifecycle.finish_status("planning", &status);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
                native_cancellation.cancel();
                let status = cancellation_status(&self.shutdown_token, &query_id);
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
            let status = status_with_query_id(adapter_error_status(error), &query_id);
            lifecycle.finish_status("planning", &status);
            return Err(status);
        }
        let schema = df.schema().as_arrow().clone();
        // `execute_stream()` runs physical planning. Log the retained native
        // report after that point; leaf-guard facts only surface on failure.
        let stream = tokio::select! {
            result = df.execute_stream() => match result {
                Ok(stream) => {
                    log_native_planning_report(&query_id, planning.native_rule.last_report().as_ref());
                    stream
                },
                Err(error) => {
                    log_native_planning_report(&query_id, planning.native_rule.last_report().as_ref());
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, ticket_kind.execute_stream_operation()),
                        &query_id,
                    );
                    lifecycle.finish_status("execute_stream", &status);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
                native_cancellation.cancel();
                let status = cancellation_status(&self.shutdown_token, &query_id);
                lifecycle.finish_status("execute_stream", &status);
                return Err(status);
            },
        };
        let cancel_token_for_tail = token.clone();
        let shutdown_token_for_tail = self.shutdown_token.clone();
        let query_id_for_tail = query_id.clone();
        let lifecycle_for_body = Arc::clone(&lifecycle);
        let query_id_for_body = query_id.clone();
        let stream_operation = ticket_kind.stream_operation();
        // `take_until` ends cleanly on cancellation, so append a one-shot tail
        // that emits CANCELLED as the gRPC trailer when the token fired.
        let body = stream
            .map_err(move |error| {
                datafusion_error_to_flight_error(error, &query_id_for_body, &lifecycle_for_body, stream_operation)
            })
            .take_until(token.cancelled_owned());
        let lifecycle_for_tail = Arc::clone(&lifecycle);
        let tail = futures::stream::once(async move {
            cancel_token_for_tail.is_cancelled().then(|| {
                let status = cancellation_status(&shutdown_token_for_tail, &query_id_for_tail);
                lifecycle_for_tail.finish_status("stream", &status);
                Err(FlightError::from(status))
            })
        })
        .filter_map(|opt| async move { opt });
        let lifecycle_for_end = Arc::clone(&lifecycle);
        let end = futures::stream::once(async move {
            lifecycle_for_end.finish_success("stream");
            None::<Result<arrow::record_batch::RecordBatch, FlightError>>
        })
        .filter_map(|opt| async move { opt });
        let stream = body.chain(tail).chain(end).map(move |batch| {
            let _guard = &guard;
            batch
        });
        let lifecycle_for_encode = Arc::clone(&lifecycle);
        let query_id_for_encode = query_id.clone();
        let stream =
            FlightDataEncoderBuilder::new().with_schema(Arc::new(schema)).build(stream).map_err(move |error| {
                let status = status_with_query_id(Status::from(error), &query_id_for_encode);
                lifecycle_for_encode.finish_status("flight_encode", &status);
                status
            });

        Ok(Response::new(Box::pin(stream)))
    }

    async fn execute_update_sql(
        &self,
        query_id: String,
        sql: String,
        statement_kind: UpdateStatementKind,
    ) -> Result<i64, Status> {
        let started_at = Instant::now();
        tracing::info!(query_id, sql, statement_kind = ?statement_kind, "executing Flight SQL update statement");

        let queued_at = Instant::now();
        let _permit = match self.acquire_statement_permit(&query_id).await {
            Ok(permit) => permit,
            Err(status) => {
                log_statement_status(&query_id, "admission", queued_at, &status);
                return Err(status);
            },
        };

        let lifecycle = QueryLifecycle::new(query_id.clone(), started_at);
        let state = self.ctx.state();
        let plan = tokio::select! {
            result = state.create_logical_plan(&sql) => match result {
                Ok(plan) => plan,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                        &query_id,
                    );
                    lifecycle.finish_status("planning", &status);
                    return Err(status);
                },
            },
            _ = self.shutdown_token.cancelled() => {
                let status = server_shutdown_status(&query_id);
                lifecycle.finish_status("planning", &status);
                return Err(status);
            },
        };
        if let Err(status) = reject_parameterized_plan(&plan, statement_kind.sql_operation()) {
            let status = status_with_query_id(status, &query_id);
            lifecycle.finish_status("planning", &status);
            return Err(status);
        }
        if let Err(status) = verify_update_plan(&plan) {
            let status = status_with_query_id(status, &query_id);
            lifecycle.finish_status("planning", &status);
            return Err(status);
        }
        if let Err(status) = reject_drop_object_type_mismatch(&self.ctx, &plan, statement_kind.sql_operation()).await {
            let status = status_with_query_id(status, &query_id);
            lifecycle.finish_status("planning", &status);
            return Err(status);
        }
        let df = tokio::select! {
            result = self.ctx.execute_logical_plan(plan) => match result {
                Ok(df) => df,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                        &query_id,
                    );
                    lifecycle.finish_status("planning", &status);
                    return Err(status);
                },
            },
            _ = self.shutdown_token.cancelled() => {
                let status = server_shutdown_status(&query_id);
                lifecycle.finish_status("planning", &status);
                return Err(status);
            },
        };

        let _batches = tokio::select! {
            result = df.collect() => match result {
                Ok(batches) => batches,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, statement_kind.collect_operation()),
                        &query_id,
                    );
                    lifecycle.finish_status("collect", &status);
                    return Err(status);
                },
            },
            _ = self.shutdown_token.cancelled() => {
                let status = server_shutdown_status(&query_id);
                lifecycle.finish_status("collect", &status);
                return Err(status);
            },
        };

        lifecycle.finish_success("collect");
        Ok(UNKNOWN_UPDATE_RECORD_COUNT)
    }

    fn dataset_schema_for_prepared_statement(&self, plan: &LogicalPlan) -> Result<Schema, Status> {
        match plan {
            LogicalPlan::Ddl(ddl) => {
                verify_update_ddl(
                    ddl,
                    "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                )?;
                Ok(Schema::empty())
            },
            LogicalPlan::Dml(_) | LogicalPlan::Copy(_) => Err(adapter_error_status(Error::new(
                ErrorCode::UnsupportedOperator,
                "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                "Flight SQL prepared DML and COPY statements are not supported",
            ))),
            LogicalPlan::Statement(statement) => match statement {
                Statement::SetVariable(_) | Statement::ResetVariable(_) => Ok(Schema::empty()),
                Statement::TransactionStart(_) | Statement::TransactionEnd(_) => Err(adapter_error_status(Error::new(
                    ErrorCode::UnsupportedOperator,
                    "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                    "Flight SQL transactions are not supported",
                ))),
                Statement::Prepare(_) | Statement::Execute(_) | Statement::Deallocate(_) => {
                    Err(adapter_error_status(Error::new(
                        ErrorCode::InvalidRequest,
                        "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                        "SQL PREPARE, EXECUTE, and DEALLOCATE are not supported through Flight SQL prepared statements",
                    )))
                },
            },
            _ => Ok(plan.schema().as_arrow().clone()),
        }
    }

    fn cancel_from_flight_info(&self, raw_info: &[u8]) -> Result<bool, Status> {
        let info = FlightInfo::decode(raw_info).map_err(|error| Status::invalid_argument(error.to_string()))?;
        let Some(endpoint) = info.endpoint.first() else {
            return Ok(false);
        };
        let Some(ticket) = endpoint.ticket.as_ref() else {
            return Ok(false);
        };
        let any = arrow_flight::sql::Any::decode(&*ticket.ticket)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        // The ticket may be either a `TicketStatementQuery` (issued by
        // `get_flight_info_statement`) or a `CommandPreparedStatementQuery`
        // (issued by `get_flight_info_prepared_statement`). Both encode the
        // per-execution `query_id` UTF-8 bytes in their handle field; once
        // extracted, the cancellation path is identical.
        let query_id_bytes = if let Some(statement_ticket) =
            any.unpack::<TicketStatementQuery>().map_err(|error| Status::invalid_argument(error.to_string()))?
        {
            statement_ticket.statement_handle
        } else if let Some(prepared_ticket) = any
            .unpack::<CommandPreparedStatementQuery>()
            .map_err(|error| Status::invalid_argument(error.to_string()))?
        {
            prepared_ticket.prepared_statement_handle
        } else {
            return Ok(false);
        };
        let query_id =
            std::str::from_utf8(&query_id_bytes).map_err(|error| Status::invalid_argument(error.to_string()))?;
        if let Some((_, ticket)) = self.tickets.remove(query_id) {
            ticket.native_cancellation.cancel();
            tracing::info!(query_id, outcome = "cancelled", stage = "ticket", "cancelled Flight SQL statement ticket");
            return Ok(true);
        }
        Ok(self.cancellations.cancel(query_id))
    }
}

async fn reject_drop_object_type_mismatch(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    operation: &'static str,
) -> Result<(), Status> {
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
    Err(adapter_error_status(
        Error::new(
            ErrorCode::InvalidRequest,
            operation,
            format!(
                "Object '{object_name}' is a {actual_label}, not a {expected_label}; use {suggested_drop} to remove it"
            ),
        )
        .with_fact("object_name", object_name)
        .with_fact("requested_table_type", table_type_to_flight_sql(expected_table_type))
        .with_fact("actual_table_type", table_type_to_flight_sql(actual_table_type)),
    ))
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

#[cfg(all(test, feature = "iceberg"))]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use datafusion::datasource::MemTable;
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
            1,
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
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
            1,
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
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
    async fn planning_timeout_releases_single_flight_sql_admission_permit() {
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
            1,
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
        );

        issue_statement_ticket(&service, "SELECT COUNT(*) AS c FROM iceberg_t")
            .await
            .expect("logical planning should issue a ticket before table scan planning");
        let query_id = only_ticket_query_id(&service);
        let status = match service.execute_ticket(query_id).await {
            Ok(_) => panic!("physical scan planning should time out"),
            Err(status) => status,
        };
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_status_metadata(&status, "planning_timeout", "planning", "permanent");
        assert!(service.tickets.is_empty(), "timed-out statement ticket should be consumed");

        tokio::time::timeout(Duration::from_secs(1), issue_statement_ticket(&service, "SELECT 1 AS one"))
            .await
            .expect("second statement should not wait on the timed-out query's admission permit")
            .expect("second statement should acquire admission and issue a ticket");
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
