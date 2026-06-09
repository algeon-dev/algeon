use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use arrow::{
    array::{Int32Array, StringArray, UInt8Array},
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
};
use crate::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, Error, ErrorCode};

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
    /// Map of prepared-statement handle → SQL text. Populated by
    /// `do_action_create_prepared_statement` and removed by
    /// `do_action_close_prepared_statement` (or replaced when the same
    /// handle is reused). The handle is an opaque UUID and is distinct
    /// from the per-execution `query_id` issued at `get_flight_info_*`
    /// time; one prepared statement can be executed multiple times,
    /// each producing a fresh `query_id` / Flight ticket.
    prepared_statements: Arc<DashMap<String, String>>,
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

/// Selects which Flight SQL ticket envelope to attach to the
/// `FlightInfo` returned for a statement. `Statement` matches the
/// `get_flight_info_statement` path; `PreparedStatement` matches the
/// `get_flight_info_prepared_statement` path. The payload bytes — the
/// per-execution `query_id` — are identical in both cases; only the
/// outer protobuf `Any` type URL differs, so the right `do_get_*`
/// dispatch branch fires.
#[derive(Debug, Clone, Copy)]
enum StatementTicketKind {
    Statement,
    PreparedStatement,
}

impl StatementTicketKind {
    const fn create_logical_plan_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.get_flight_info_statement.create_logical_plan",
            Self::PreparedStatement => {
                "datafusion_nexus.server.flight.get_flight_info_prepared_statement.create_logical_plan"
            },
        }
    }

    const fn sql_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_get_statement.sql",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_get_prepared_statement.sql",
        }
    }

    const fn execute_stream_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_get_statement.execute_stream",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_get_prepared_statement.execute_stream",
        }
    }

    const fn stream_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_get_statement.stream",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_get_prepared_statement.stream",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum UpdateStatementKind {
    Statement,
    PreparedStatement,
}

impl UpdateStatementKind {
    const fn sql_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_put_statement_update.sql",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_put_prepared_statement_update.sql",
        }
    }

    const fn collect_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_put_statement_update.collect",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_put_prepared_statement_update.collect",
        }
    }
}

#[derive(Debug)]
struct StatementTicket {
    query_id: String,
    sql: String,
    ticket_kind: StatementTicketKind,
    native_cancellation: ExecutionCancellation,
    started_at: Instant,
    issued_at: Instant,
    _permit: OwnedSemaphorePermit,
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
        let StatementTicket { query_id, sql, ticket_kind, native_cancellation, started_at, _permit: permit, .. } =
            ticket;
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
        let df = tokio::select! {
            result = query_ctx.sql(&sql) => match result {
                Ok(df) => df,
                Err(error) => {
                    // Logical-planning failures (e.g. SQL parse, unknown
                    // table) happen before the physical optimizer runs, so
                    // `last_report` is empty here. The leaf-guard helper
                    // still no-ops on non-leaf-guard errors, but we run it
                    // for completeness so any planning-time adapter error
                    // that does carry the leaf-guard facts gets the
                    // structured warn line.
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

#[tonic::async_trait]
impl FlightSqlService for DatafusionCudfFlightSqlService {
    type FlightService = Self;

    async fn do_handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>>, Status> {
        handle_handshake(self.auth.mode(), self.auth.tokens(), request).await
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let sql = query.query;
        let planning_started_at = Instant::now();
        tracing::info!(query_id, sql, "planning Flight SQL statement");
        if query.transaction_id.is_some() {
            let status = flight_sql_transactions_unsupported_status(&query_id);
            log_statement_status(&query_id, "planning", planning_started_at, &status);
            return Err(status);
        }
        let queued_at = Instant::now();
        let permit = match self.acquire_statement_permit(&query_id).await {
            Ok(permit) => permit,
            Err(status) => {
                log_statement_status(&query_id, "admission", queued_at, &status);
                return Err(status);
            },
        };
        let admission_wait_ms = queued_at.elapsed().as_millis();
        if admission_wait_ms > 0 || self.admission.max_in_flight() > 1 {
            tracing::info!(
                query_id,
                admission_wait_ms,
                max_in_flight = self.admission.max_in_flight(),
                "admitted Flight SQL statement for planning"
            );
        }
        let info = match self
            .statement_flight_info(
                query_id.clone(),
                sql,
                request.into_inner(),
                planning_started_at,
                permit,
                StatementTicketKind::Statement,
            )
            .await
        {
            Ok(info) => info,
            Err(status) => {
                let status = status_with_query_id(status, &query_id);
                log_statement_status(&query_id, "planning", planning_started_at, &status);
                return Err(status);
            },
        };
        Ok(Response::new(info))
    }

    async fn get_flight_info_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let handle = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let sql = self.prepared_statements.get(&handle).map(|entry| entry.value().clone()).ok_or_else(|| {
            adapter_error_status_with_query_id(
                Error::new(
                    ErrorCode::ObjectNotFound,
                    "datafusion_nexus.server.flight.get_flight_info_prepared_statement.resolve",
                    format!("unknown prepared statement handle: {handle}"),
                ),
                &query_id,
            )
        })?;

        let planning_started_at = Instant::now();
        tracing::info!(
            query_id,
            sql,
            prepared_statement_handle = handle.as_str(),
            "planning Flight SQL prepared statement"
        );
        let queued_at = Instant::now();
        let permit = match self.acquire_statement_permit(&query_id).await {
            Ok(permit) => permit,
            Err(status) => {
                log_statement_status(&query_id, "admission", queued_at, &status);
                return Err(status);
            },
        };
        let info = match self
            .statement_flight_info(
                query_id.clone(),
                sql,
                request.into_inner(),
                planning_started_at,
                permit,
                StatementTicketKind::PreparedStatement,
            )
            .await
        {
            Ok(info) => info,
            Err(status) => {
                let status = status_with_query_id(status, &query_id);
                log_statement_status(&query_id, "planning", planning_started_at, &status);
                return Err(status);
            },
        };
        Ok(Response::new(info))
    }

    async fn get_flight_info_catalogs(
        &self,
        query: CommandGetCatalogs,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        // `CommandGetCatalogs` is a unit message; the builder produces a
        // static schema, and we echo the (Copy) command back as the ticket.
        let schema = arrow_flight::sql::metadata::GetCatalogsBuilder::new().schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_schemas(
        &self,
        query: CommandGetDbSchemas,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = GetDbSchemasBuilder::from(query.clone()).schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_tables(
        &self,
        query: CommandGetTables,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = GetTablesBuilder::from(query.clone()).schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_table_types(
        &self,
        query: CommandGetTableTypes,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = flight_sql_table_types_schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_sql_info(
        &self,
        query: CommandGetSqlInfo,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let ticket = Ticket { ticket: query.as_any().encode_to_vec().into() };
        let endpoint = FlightEndpoint::new().with_ticket(ticket);
        let sql_info = sql_info_data()?;
        let info = FlightInfo::new()
            .try_with_schema(query.into_builder(&sql_info).schema().as_ref())
            .map_err(arrow_error_to_status)?
            .with_descriptor(descriptor)
            .with_endpoint(endpoint);
        Ok(Response::new(info))
    }

    async fn get_flight_info_primary_keys(
        &self,
        query: CommandGetPrimaryKeys,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = flight_sql_primary_keys_schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_exported_keys(
        &self,
        query: CommandGetExportedKeys,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = flight_sql_foreign_keys_schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_imported_keys(
        &self,
        query: CommandGetImportedKeys,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = flight_sql_foreign_keys_schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_cross_reference(
        &self,
        query: CommandGetCrossReference,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = flight_sql_foreign_keys_schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let query_id = std::str::from_utf8(&ticket.statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
        self.execute_ticket(query_id).await
    }

    async fn do_get_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let query_id = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
        self.execute_ticket(query_id).await
    }

    async fn do_get_catalogs(
        &self,
        query: CommandGetCatalogs,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let mut builder = query.into_builder();
        for name in self.ctx.catalog_names() {
            builder.append(name);
        }
        let schema = builder.schema();
        let batch = builder.build().map_err(arrow_flight_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(schema, batch)))
    }

    async fn do_get_schemas(
        &self,
        query: CommandGetDbSchemas,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let mut builder = GetDbSchemasBuilder::from(query);
        for catalog_name in self.ctx.catalog_names() {
            let Some(catalog) = self.ctx.catalog(&catalog_name) else {
                continue;
            };
            for schema_name in catalog.schema_names() {
                builder.append(&catalog_name, &schema_name);
            }
        }
        let schema = builder.schema();
        let batch = builder.build().map_err(arrow_flight_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(schema, batch)))
    }

    async fn do_get_tables(
        &self,
        query: CommandGetTables,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let include_schema = query.include_schema;
        let mut builder = GetTablesBuilder::from(query);
        let empty_schema = Schema::empty();
        for catalog_name in self.ctx.catalog_names() {
            let Some(catalog) = self.ctx.catalog(&catalog_name) else {
                continue;
            };
            for schema_name in catalog.schema_names() {
                let Some(schema_provider) = catalog.schema(&schema_name) else {
                    continue;
                };
                for table_name in schema_provider.table_names() {
                    // For `include_schema=true` we materialize the provider to
                    // read its Arrow schema; this also exposes the canonical
                    // `TableProvider::table_type()`. For Iceberg/Glue that
                    // round-trip hits Glue + S3, so the cheaper path below
                    // uses `SchemaProvider::table_type()` (which the Iceberg
                    // provider implements without touching the backend).
                    let (table_schema, resolved_type) = if include_schema {
                        match schema_provider.table(&table_name).await {
                            Ok(Some(table)) => (Some(table.schema()), Some(table.table_type())),
                            Ok(None) => (None, None),
                            Err(error) => {
                                tracing::warn!(
                                    catalog = %catalog_name,
                                    schema = %schema_name,
                                    table = %table_name,
                                    error = %error,
                                    "failed to load table for Flight SQL GetTables"
                                );
                                (None, None)
                            },
                        }
                    } else {
                        let table_type = schema_provider.table_type(&table_name).await.map_err(|error| {
                            datafusion_error_to_status_for_operation(
                                error,
                                "datafusion_nexus.server.flight.do_get_tables.table_type",
                            )
                        })?;
                        (None, table_type)
                    };
                    // A `None` here means the table vanished between
                    // `table_names()` and the type lookup (TOCTOU race) or the
                    // provider deliberately rejected it. Skip so the response
                    // does not include a ghost row whose type would have to be
                    // a guess.
                    let Some(table_type_str) = resolved_type.map(table_type_to_flight_sql) else {
                        continue;
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
        Ok(Response::new(metadata_record_batch_stream(schema, batch)))
    }

    async fn do_get_table_types(
        &self,
        _query: CommandGetTableTypes,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = build_table_types_batch().map_err(arrow_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(batch.schema(), batch)))
    }

    async fn do_get_sql_info(
        &self,
        query: CommandGetSqlInfo,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let sql_info = sql_info_data()?;
        let builder = query.into_builder(&sql_info);
        let schema = builder.schema();
        let batch = builder.build();
        let stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(futures::stream::once(async { batch }))
            .map_err(Status::from);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_get_primary_keys(
        &self,
        _query: CommandGetPrimaryKeys,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        // DataFusion's catalog does not carry primary-key constraints, so the
        // response is always an empty batch with the Flight SQL standard
        // schema. This lets clients (DataGrip, the Arrow Flight SQL JDBC
        // driver, ADBC) complete introspection without a UNIMPLEMENTED RPC
        // failure tearing down the session.
        let batch = empty_primary_keys_batch().map_err(arrow_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(batch.schema(), batch)))
    }

    async fn do_get_exported_keys(
        &self,
        _query: CommandGetExportedKeys,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = empty_foreign_keys_batch().map_err(arrow_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(batch.schema(), batch)))
    }

    async fn do_get_imported_keys(
        &self,
        _query: CommandGetImportedKeys,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = empty_foreign_keys_batch().map_err(arrow_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(batch.schema(), batch)))
    }

    async fn do_get_cross_reference(
        &self,
        _query: CommandGetCrossReference,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = empty_foreign_keys_batch().map_err(arrow_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(batch.schema(), batch)))
    }

    async fn do_put_statement_update(
        &self,
        query: CommandStatementUpdate,
        request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        self.check_auth(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        if query.transaction_id.is_some() {
            return Err(flight_sql_transactions_unsupported_status(&query_id));
        }
        self.execute_update_sql(query_id, query.query, UpdateStatementKind::Statement).await
    }

    async fn do_put_prepared_statement_update(
        &self,
        query: CommandPreparedStatementUpdate,
        request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        self.check_auth(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let handle = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| {
                adapter_error_status_with_query_id(
                    Error::new(
                        ErrorCode::InvalidRequest,
                        "datafusion_nexus.server.flight.do_put_prepared_statement_update.handle",
                        format!("prepared statement handle is not valid UTF-8: {error}"),
                    ),
                    &query_id,
                )
            })?
            .to_string();
        let sql = self.prepared_statements.get(&handle).map(|entry| entry.value().clone()).ok_or_else(|| {
            adapter_error_status_with_query_id(
                Error::new(
                    ErrorCode::ObjectNotFound,
                    "datafusion_nexus.server.flight.do_put_prepared_statement_update.resolve",
                    format!("unknown prepared statement handle: {handle}"),
                ),
                &query_id,
            )
        })?;

        self.execute_update_sql(query_id, sql, UpdateStatementKind::PreparedStatement).await
    }

    async fn do_action_create_prepared_statement(
        &self,
        query: ActionCreatePreparedStatementRequest,
        request: Request<arrow_flight::Action>,
    ) -> Result<ActionCreatePreparedStatementResult, Status> {
        self.check_auth(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        if query.transaction_id.is_some() {
            return Err(flight_sql_transactions_unsupported_status(&query_id));
        }
        if self.prepared_statements.len() >= MAX_PREPARED_STATEMENTS {
            let error = Error::new(
                ErrorCode::ServiceOverloaded,
                "datafusion_nexus.server.flight.do_action_create_prepared_statement.admission",
                format!("too many live prepared statements (limit {MAX_PREPARED_STATEMENTS})"),
            )
            .with_fact("limit", MAX_PREPARED_STATEMENTS);
            let operation = error.operation();
            return Err(status_with_query_id(
                datafusion_error_to_status_for_operation(error.into(), operation),
                &query_id,
            ));
        }
        let sql = query.query;
        // Plan logically to derive the result-set schema the client needs
        // up-front. This does *not* run the physical optimizer rule and does
        // not consume an admission permit — the real plan / admission /
        // ticket happen later in `get_flight_info_prepared_statement` when
        // the client actually executes this prepared statement.
        let state = self.ctx.state();
        let logical_plan = tokio::select! {
            result = state.create_logical_plan(&sql) => {
                result.map_err(|error| {
                    status_with_query_id(
                        datafusion_error_to_status_for_operation(
                            error,
                            "datafusion_nexus.server.flight.do_action_create_prepared_statement.create_logical_plan",
                        ),
                        &query_id,
                    )
                })?
            },
            _ = self.shutdown_token.cancelled() => {
                return Err(status_with_query_id(Status::cancelled("server shutting down"), &query_id));
            }
        };
        if let Err(status) = reject_parameterized_plan(
            &logical_plan,
            "datafusion_nexus.server.flight.do_action_create_prepared_statement.parameter_binding",
        ) {
            return Err(status_with_query_id(status, &query_id));
        }
        let dataset_schema = self
            .dataset_schema_for_prepared_statement(&logical_plan)
            .map_err(|status| status_with_query_id(status, &query_id))?;
        let dataset_schema_ipc =
            schema_to_ipc_bytes(&dataset_schema).map_err(|error| Status::internal(error.to_string()))?;

        let handle = Uuid::new_v4().hyphenated().to_string();
        self.prepared_statements.insert(handle.clone(), sql);
        tracing::info!(prepared_statement_handle = handle.as_str(), "created Flight SQL prepared statement");

        Ok(ActionCreatePreparedStatementResult {
            prepared_statement_handle: handle.into_bytes().into(),
            dataset_schema: dataset_schema_ipc,
            parameter_schema: prost::bytes::Bytes::new(),
        })
    }

    async fn do_action_close_prepared_statement(
        &self,
        query: ActionClosePreparedStatementRequest,
        request: Request<arrow_flight::Action>,
    ) -> Result<(), Status> {
        self.check_auth(&request)?;
        let handle = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        // Closing an unknown handle is not an error: arrow-flight clients
        // may retry close, and the handle may have been swept by a prior
        // close on the same connection. Trace-log the no-op so client
        // lifecycle bugs (e.g. double-close, close-after-eviction) are
        // observable without breaking the idempotent contract.
        if self.prepared_statements.remove(handle).is_none() {
            tracing::trace!(
                prepared_statement_handle = handle,
                "close of unknown or already-closed prepared statement (no-op)"
            );
        }
        Ok(())
    }

    async fn do_action_cancel_query(
        &self,
        query: ActionCancelQueryRequest,
        request: Request<arrow_flight::Action>,
    ) -> Result<ActionCancelQueryResult, Status> {
        self.check_auth(&request)?;
        let result = if self.cancel_from_flight_info(&query.info)? {
            FLIGHT_SQL_CANCEL_RESULT_CANCELLING
        } else {
            FLIGHT_SQL_CANCEL_RESULT_NOT_CANCELLABLE
        };
        Ok(ActionCancelQueryResult { result })
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}

struct QueryExecutionGuard {
    query_id: String,
    cancellations: Arc<QueryCancellation>,
    native_cancel_bridge: tokio::task::JoinHandle<()>,
    _permit: OwnedSemaphorePermit,
    lifecycle: Arc<QueryLifecycle>,
}

impl Drop for QueryExecutionGuard {
    fn drop(&mut self) {
        self.native_cancel_bridge.abort();
        self.cancellations.remove(&self.query_id);
        self.lifecycle.finish_dropped();
    }
}

struct QueryLifecycle {
    query_id: String,
    started_at: Instant,
    terminal: Mutex<Option<&'static str>>,
}

impl QueryLifecycle {
    fn new(query_id: String, started_at: Instant) -> Self {
        Self { query_id, started_at, terminal: Mutex::new(None) }
    }

    fn finish_success(&self, stage: &'static str) {
        if self.try_finish("succeeded") {
            tracing::info!(
                query_id = self.query_id,
                outcome = "succeeded",
                stage,
                elapsed_ms = self.started_at.elapsed().as_millis(),
                "finished Flight SQL statement"
            );
        }
    }

    fn finish_dropped(&self) {
        if self.try_finish("dropped") {
            tracing::warn!(
                query_id = self.query_id,
                outcome = "dropped",
                stage = "stream",
                elapsed_ms = self.started_at.elapsed().as_millis(),
                "finished Flight SQL statement"
            );
        }
    }

    fn finish_status(&self, stage: &'static str, status: &Status) {
        let outcome = status_outcome(status.code());
        if self.try_finish(outcome) {
            log_statement_status(&self.query_id, stage, self.started_at, status);
        }
    }

    fn try_finish(&self, outcome: &'static str) -> bool {
        let mut terminal = self.terminal.lock().expect("query lifecycle mutex poisoned");
        if terminal.is_some() {
            return false;
        }
        *terminal = Some(outcome);
        true
    }
}

fn datafusion_error_to_flight_error(
    error: DataFusionError,
    query_id: &str,
    lifecycle: &QueryLifecycle,
    operation: &'static str,
) -> FlightError {
    let status = status_with_query_id(datafusion_error_to_status_for_operation(error, operation), query_id);
    lifecycle.finish_status("stream", &status);
    FlightError::from(status)
}

fn verify_flight_info_plan(plan: &LogicalPlan, ticket_kind: StatementTicketKind) -> Result<(), Status> {
    match ticket_kind {
        StatementTicketKind::Statement => verify_statement_query_plan(plan),
        StatementTicketKind::PreparedStatement => verify_prepared_query_plan(plan),
    }
}

fn verify_statement_query_plan(plan: &LogicalPlan) -> Result<(), Status> {
    match plan {
        LogicalPlan::Dml(_) | LogicalPlan::Copy(_) => Err(adapter_error_status(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_nexus.server.flight.get_flight_info_statement.verify_plan",
            "Flight SQL statement query does not support DML or COPY statements",
        ))),
        LogicalPlan::Statement(statement) => verify_sql_statement_namespace(
            statement,
            "datafusion_nexus.server.flight.get_flight_info_statement.verify_plan",
            "Flight SQL statement query does not support SQL session statements; use the update endpoint for session updates",
        ),
        _ => Ok(()),
    }
}

fn verify_prepared_query_plan(plan: &LogicalPlan) -> Result<(), Status> {
    match plan {
        LogicalPlan::Ddl(_) => Err(adapter_error_status(Error::new(
            ErrorCode::InvalidRequest,
            "datafusion_nexus.server.flight.get_flight_info_prepared_statement.verify_plan",
            "Flight SQL prepared DDL statements must be executed with execute_update",
        ))),
        LogicalPlan::Dml(_) | LogicalPlan::Copy(_) => Err(adapter_error_status(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_nexus.server.flight.get_flight_info_prepared_statement.verify_plan",
            "Flight SQL prepared DML and COPY statements are not supported",
        ))),
        LogicalPlan::Statement(statement) => verify_sql_statement_namespace(
            statement,
            "datafusion_nexus.server.flight.get_flight_info_prepared_statement.verify_plan",
            "Flight SQL prepared session-update statements must be executed with execute_update",
        ),
        _ => Ok(()),
    }
}

fn verify_update_plan(plan: &LogicalPlan) -> Result<(), Status> {
    match plan {
        LogicalPlan::Ddl(ddl) => verify_update_ddl(ddl, "datafusion_nexus.server.flight.do_put_update.verify_plan"),
        LogicalPlan::Dml(_) | LogicalPlan::Copy(_) => Err(adapter_error_status(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_nexus.server.flight.do_put_update.verify_plan",
            "Flight SQL update DoPut does not yet support DML or COPY statements",
        ))),
        LogicalPlan::Statement(statement) => verify_sql_statement_namespace(
            statement,
            "datafusion_nexus.server.flight.do_put_update.verify_plan",
            "SQL PREPARE, EXECUTE, and DEALLOCATE are not supported through Flight SQL update DoPut",
        ),
        _ => Err(adapter_error_status(Error::new(
            ErrorCode::InvalidRequest,
            "datafusion_nexus.server.flight.do_put_update.verify_plan",
            "Flight SQL update DoPut only supports DDL and session-update statements; use the statement query endpoint for SELECT",
        ))),
    }
}

fn verify_sql_statement_namespace(
    statement: &Statement,
    operation: &'static str,
    session_update_message: &'static str,
) -> Result<(), Status> {
    match statement {
        Statement::SetVariable(_) | Statement::ResetVariable(_) => {
            if operation == "datafusion_nexus.server.flight.do_put_update.verify_plan" {
                Ok(())
            } else {
                Err(adapter_error_status(Error::new(ErrorCode::InvalidRequest, operation, session_update_message)))
            }
        },
        Statement::TransactionStart(_) | Statement::TransactionEnd(_) => Err(adapter_error_status(Error::new(
            ErrorCode::UnsupportedOperator,
            operation,
            "Flight SQL transactions are not supported",
        ))),
        Statement::Prepare(_) | Statement::Execute(_) | Statement::Deallocate(_) => {
            Err(adapter_error_status(Error::new(
                ErrorCode::InvalidRequest,
                operation,
                "SQL PREPARE, EXECUTE, and DEALLOCATE are not supported through Flight SQL",
            )))
        },
    }
}

fn verify_update_ddl(ddl: &DdlStatement, operation: &'static str) -> Result<(), Status> {
    match ddl {
        DdlStatement::CreateMemoryTable(_) | DdlStatement::CreateIndex(_) => Err(adapter_error_status(Error::new(
            ErrorCode::UnsupportedOperator,
            operation,
            "Flight SQL update DoPut does not yet support query-producing DDL such as CREATE TABLE AS SELECT",
        ))),
        _ => Ok(()),
    }
}

fn reject_parameterized_plan(plan: &LogicalPlan, operation: &'static str) -> Result<(), Status> {
    let parameter_names =
        plan.get_parameter_names().map_err(|error| datafusion_error_to_status_for_operation(error, operation))?;
    if parameter_names.is_empty() {
        return Ok(());
    }
    Err(adapter_error_status(Error::new(
        ErrorCode::InvalidRequest,
        operation,
        "Flight SQL parameter binding is not supported; prepare statements without placeholders",
    )))
}

fn arrow_error_to_status(error: arrow::error::ArrowError) -> Status {
    Status::internal(error.to_string())
}

/// Serialize an Arrow `Schema` into the IPC-encapsulated form that Flight SQL
/// uses for `ActionCreatePreparedStatementResult.dataset_schema` and
/// `parameter_schema`. Returns the `prost::bytes::Bytes` payload ready to drop
/// into the proto message.
fn schema_to_ipc_bytes(schema: &Schema) -> Result<prost::bytes::Bytes, arrow::error::ArrowError> {
    let options = IpcWriteOptions::default();
    let IpcMessage(bytes) = IpcMessage::try_from(SchemaAsIpc::new(schema, &options))?;
    Ok(bytes)
}

fn arrow_flight_error_to_status(error: arrow_flight::error::FlightError) -> Status {
    Status::from(error)
}

/// Flight SQL / ODBC / JDBC standard catalog `table_type` string for an ordinary
/// physical table. Matches the upstream `arrow-flight` builder examples and the
/// SQL standard `INFORMATION_SCHEMA.TABLES.TABLE_TYPE` values consumed by
/// arrow_cli, the ADBC driver, and the Flight SQL JDBC driver.
const FLIGHT_SQL_TABLE_TYPE_TABLE: &str = "TABLE";
/// Flight SQL standard `table_type` string for a view.
const FLIGHT_SQL_TABLE_TYPE_VIEW: &str = "VIEW";

/// The set advertised by `CommandGetTableTypes`. Stored sorted lexicographically
/// as required by the Flight SQL protobuf contract for `GetTableTypes`.
const FLIGHT_SQL_TABLE_TYPES: &[&str] = &[FLIGHT_SQL_TABLE_TYPE_TABLE, FLIGHT_SQL_TABLE_TYPE_VIEW];

/// Schema for `CommandGetTableTypes` responses. Pinned to `table_type: utf8 not
/// null` to match the Flight SQL protobuf contract. This duplicates the schema
/// defined inside `arrow_flight::sql::metadata::table_types`, which is
/// crate-private in arrow-rs 58.3 (and on main). Once arrow-rs re-exports
/// `GetTableTypesBuilder`, switch both this helper and `do_get_table_types` to
/// use it directly.
fn flight_sql_table_types_schema() -> Arc<Schema> {
    use arrow::datatypes::{DataType, Field};
    static SCHEMA: std::sync::OnceLock<Arc<Schema>> = std::sync::OnceLock::new();
    SCHEMA.get_or_init(|| Arc::new(Schema::new(vec![Field::new("table_type", DataType::Utf8, false)]))).clone()
}

/// Schema for `CommandGetPrimaryKeys` responses. Pinned to the columns
/// specified by the Flight SQL protobuf documentation
/// (`catalog_name`, `db_schema_name`, `table_name`, `column_name`,
/// `key_name`, `key_sequence`). DataFusion does not model primary-key
/// constraints, so the response is always an empty batch — but it must
/// still carry this exact schema or arrow-flight clients refuse it.
fn flight_sql_primary_keys_schema() -> Arc<Schema> {
    static SCHEMA: std::sync::OnceLock<Arc<Schema>> = std::sync::OnceLock::new();
    SCHEMA
        .get_or_init(|| {
            Arc::new(Schema::new(vec![
                Field::new("catalog_name", DataType::Utf8, true),
                Field::new("db_schema_name", DataType::Utf8, true),
                Field::new("table_name", DataType::Utf8, false),
                Field::new("column_name", DataType::Utf8, false),
                Field::new("key_name", DataType::Utf8, true),
                Field::new("key_sequence", DataType::Int32, false),
            ]))
        })
        .clone()
}

/// Schema for `CommandGetImportedKeys` / `CommandGetExportedKeys` /
/// `CommandGetCrossReference` responses. The three RPCs share this 13-column
/// foreign-key descriptor schema per the Flight SQL protobuf spec. DataFusion
/// does not model foreign-key constraints, so the body is always empty — but
/// the schema must still match exactly or arrow-flight clients refuse it.
fn flight_sql_foreign_keys_schema() -> Arc<Schema> {
    static SCHEMA: std::sync::OnceLock<Arc<Schema>> = std::sync::OnceLock::new();
    SCHEMA
        .get_or_init(|| {
            Arc::new(Schema::new(vec![
                Field::new("pk_catalog_name", DataType::Utf8, true),
                Field::new("pk_db_schema_name", DataType::Utf8, true),
                Field::new("pk_table_name", DataType::Utf8, false),
                Field::new("pk_column_name", DataType::Utf8, false),
                Field::new("fk_catalog_name", DataType::Utf8, true),
                Field::new("fk_db_schema_name", DataType::Utf8, true),
                Field::new("fk_table_name", DataType::Utf8, false),
                Field::new("fk_column_name", DataType::Utf8, false),
                Field::new("key_sequence", DataType::Int32, false),
                Field::new("fk_key_name", DataType::Utf8, true),
                Field::new("pk_key_name", DataType::Utf8, true),
                Field::new("update_rule", DataType::UInt8, false),
                Field::new("delete_rule", DataType::UInt8, false),
            ]))
        })
        .clone()
}

/// Empty-row `RecordBatch` matching [`flight_sql_foreign_keys_schema`].
fn empty_foreign_keys_batch() -> Result<arrow::record_batch::RecordBatch, arrow::error::ArrowError> {
    let schema = flight_sql_foreign_keys_schema();
    let columns: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(Int32Array::from(Vec::<i32>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(UInt8Array::from(Vec::<u8>::new())),
        Arc::new(UInt8Array::from(Vec::<u8>::new())),
    ];
    arrow::record_batch::RecordBatch::try_new(schema, columns)
}

/// Empty-row `RecordBatch` matching [`flight_sql_primary_keys_schema`].
fn empty_primary_keys_batch() -> Result<arrow::record_batch::RecordBatch, arrow::error::ArrowError> {
    let schema = flight_sql_primary_keys_schema();
    let columns: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(Int32Array::from(Vec::<i32>::new())),
    ];
    arrow::record_batch::RecordBatch::try_new(schema, columns)
}

/// Build the single-batch response for `CommandGetTableTypes`.
///
/// `FLIGHT_SQL_TABLE_TYPES` is the source of truth; keep it sorted at authoring
/// time instead of re-sorting here.
fn build_table_types_batch() -> Result<arrow::record_batch::RecordBatch, arrow::error::ArrowError> {
    debug_assert!(FLIGHT_SQL_TABLE_TYPES.windows(2).all(|w| w[0] < w[1]), "FLIGHT_SQL_TABLE_TYPES must be sorted");
    let column = arrow::array::StringArray::from_iter_values(FLIGHT_SQL_TABLE_TYPES.iter().copied());
    arrow::record_batch::RecordBatch::try_new(flight_sql_table_types_schema(), vec![Arc::new(column)])
}

/// Map DataFusion's `TableType` to the Flight SQL wire string. `Temporary` has
/// no dedicated Flight SQL value, so it surfaces as `"TABLE"` (matching how
/// `information_schema.tables` reports temp tables).
fn table_type_to_flight_sql(table_type: TableType) -> &'static str {
    match table_type {
        TableType::View => FLIGHT_SQL_TABLE_TYPE_VIEW,
        TableType::Base | TableType::Temporary => FLIGHT_SQL_TABLE_TYPE_TABLE,
    }
}

/// Build a `FlightInfo` for a Flight SQL metadata command whose `do_get` echoes
/// the original command back as the ticket payload (matching how
/// `get_flight_info_sql_info` is wired).
fn metadata_flight_info<C>(command: C, descriptor: FlightDescriptor, schema: &Schema) -> Result<FlightInfo, Status>
where C: ProstMessageExt {
    let ticket = Ticket { ticket: command.as_any().encode_to_vec().into() };
    let endpoint = FlightEndpoint::new().with_ticket(ticket);
    let info = FlightInfo::new()
        .try_with_schema(schema)
        .map_err(arrow_error_to_status)?
        .with_descriptor(descriptor)
        .with_endpoint(endpoint)
        .with_ordered(false);
    Ok(info)
}

/// Wrap a single pre-built `RecordBatch` as a Flight `DoGetStream`. Used by the
/// metadata RPCs where the entire response fits in one batch produced by the
/// arrow-flight metadata builders.
fn metadata_record_batch_stream(
    schema: Arc<Schema>,
    batch: arrow::record_batch::RecordBatch,
) -> Pin<Box<dyn Stream<Item = Result<arrow_flight::FlightData, Status>> + Send>> {
    let stream = FlightDataEncoderBuilder::new()
        .with_schema(schema)
        .build(futures::stream::once(async move { Ok(batch) }))
        .map_err(Status::from);
    Box::pin(stream)
}

fn server_error_to_status(error: ServerError) -> Status {
    Status::internal(error.to_string())
}

fn adapter_error_status(error: Error) -> Status {
    let operation = error.operation();
    datafusion_error_to_status_for_operation(error.into(), operation)
}

fn adapter_error_status_with_query_id(error: Error, query_id: &str) -> Status {
    status_with_query_id(adapter_error_status(error), query_id)
}

fn flight_sql_transactions_unsupported_status(query_id: &str) -> Status {
    adapter_error_status_with_query_id(
        Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_nexus.server.flight.transaction",
            "Flight SQL transactions are not supported",
        ),
        query_id,
    )
}

fn server_shutdown_status(query_id: &str) -> Status {
    adapter_error_status_with_query_id(
        Error::new(
            ErrorCode::Cancelled,
            "datafusion_nexus.server.flight.shutdown",
            "server shutting down; statement aborted",
        ),
        query_id,
    )
}

/// Disambiguate per-statement cancellation: if the parent shutdown token has
/// been cancelled the whole server is going away; otherwise it was an
/// `ActionCancelQuery` from the client.
fn cancellation_status(shutdown_token: &CancellationToken, query_id: &str) -> Status {
    if shutdown_token.is_cancelled() {
        server_shutdown_status(query_id)
    } else {
        adapter_error_status_with_query_id(
            Error::new(ErrorCode::Cancelled, "datafusion_nexus.server.flight.cancel_query", "statement was cancelled"),
            query_id,
        )
    }
}

fn status_with_query_id(mut status: Status, query_id: &str) -> Status {
    if let Ok(value) = MetadataValue::try_from(query_id) {
        status.metadata_mut().insert(QUERY_ID_METADATA_KEY, value);
    }
    status
}

fn status_outcome(code: Code) -> &'static str {
    if code == Code::Cancelled { "cancelled" } else { "failed" }
}

fn log_statement_status(query_id: &str, stage: &'static str, started_at: Instant, status: &Status) {
    let outcome = status_outcome(status.code());
    tracing::warn!(
        query_id,
        outcome,
        stage,
        grpc_code = ?status.code(),
        elapsed_ms = started_at.elapsed().as_millis(),
        error = status.message(),
        "failed Flight SQL statement"
    );
}

fn sql_info_data() -> Result<arrow_flight::sql::metadata::SqlInfoData, Status> {
    let mut builder = SqlInfoDataBuilder::new();
    builder.append(SqlInfo::FlightSqlServerName, "datafusion-nexus");
    builder.append(SqlInfo::FlightSqlServerVersion, env!("CARGO_PKG_VERSION"));
    builder.append(SqlInfo::FlightSqlServerArrowVersion, "1.3");
    builder.append(SqlInfo::FlightSqlServerReadOnly, false);
    builder.append(SqlInfo::FlightSqlServerSql, true);
    builder.append(SqlInfo::FlightSqlServerSubstrait, false);
    builder.append(SqlInfo::FlightSqlServerTransaction, SqlSupportedTransaction::None as i32);
    builder.append(SqlInfo::FlightSqlServerCancel, true);
    builder.append(SqlInfo::FlightSqlServerBulkIngestion, false);
    builder.append(SqlInfo::FlightSqlServerIngestTransactionsSupported, false);

    builder.append(SqlInfo::SqlDdlCatalog, false);
    builder.append(SqlInfo::SqlDdlSchema, true);
    builder.append(SqlInfo::SqlDdlTable, true);
    builder.append(SqlInfo::SqlIdentifierCase, SqlSupportedCaseSensitivity::SqlCaseSensitivityLowercase as i32);
    builder.append(SqlInfo::SqlIdentifierQuoteChar, r#"""#);
    builder.append(SqlInfo::SqlQuotedIdentifierCase, SqlSupportedCaseSensitivity::SqlCaseSensitivityUnknown as i32);
    builder.append(SqlInfo::SqlAllTablesAreSelectable, true);
    builder.append(SqlInfo::SqlNullOrdering, SqlNullOrdering::SqlNullsSortedHigh as i32);
    builder.append(SqlInfo::SqlSearchStringEscape, "\\");
    builder.append(SqlInfo::SqlExtraNameCharacters, "_");
    builder.append(SqlInfo::SqlSupportsColumnAliasing, true);
    builder.append(SqlInfo::SqlNullPlusNullIsNull, true);
    builder.append(SqlInfo::SqlSupportsTableCorrelationNames, true);
    builder.append(SqlInfo::SqlSupportsDifferentTableCorrelationNames, true);
    builder.append(SqlInfo::SqlSupportsExpressionsInOrderBy, true);
    builder.append(SqlInfo::SqlSupportsOrderByUnrelated, true);
    builder.append(
        SqlInfo::SqlSupportedGroupBy,
        sql_info_bitmask([
            SqlSupportedGroupBy::SqlGroupByUnrelated as i32,
            SqlSupportedGroupBy::SqlGroupByBeyondSelect as i32,
        ]),
    );
    builder.append(SqlInfo::SqlSupportsLikeEscapeClause, true);
    builder.append(SqlInfo::SqlSupportsNonNullableColumns, true);
    builder.append(
        SqlInfo::SqlSupportedGrammar,
        sql_info_bitmask([SupportedSqlGrammar::SqlMinimumGrammar as i32, SupportedSqlGrammar::SqlCoreGrammar as i32]),
    );
    builder.append(SqlInfo::SqlAnsi92SupportedLevel, sql_info_bitmask([SQL_ANSI92_ENTRY_SQL_BIT_INDEX]));
    builder.append(SqlInfo::SqlSupportsIntegrityEnhancementFacility, false);
    builder.append(SqlInfo::SqlOuterJoinsSupportLevel, SqlOuterJoinsSupportLevel::SqlFullOuterJoins as i32);
    builder.append(SqlInfo::SqlSchemaTerm, "schema");
    builder.append(SqlInfo::SqlProcedureTerm, "");
    builder.append(SqlInfo::SqlCatalogTerm, "catalog");
    builder.append(SqlInfo::SqlCatalogAtStart, true);
    builder.append(SqlInfo::SqlSchemasSupportedActions, 0);
    builder.append(SqlInfo::SqlCatalogsSupportedActions, 0);
    builder.append(SqlInfo::SqlSupportedPositionedCommands, 0);
    builder.append(SqlInfo::SqlSelectForUpdateSupported, false);
    builder.append(SqlInfo::SqlStoredProceduresSupported, false);
    builder.append(SqlInfo::SqlSupportedSubqueries, 0);
    builder.append(SqlInfo::SqlCorrelatedSubqueriesSupported, false);
    builder.append(
        SqlInfo::SqlSupportedUnions,
        sql_info_bitmask([SqlSupportedUnions::SqlUnion as i32, SqlSupportedUnions::SqlUnionAll as i32]),
    );
    builder.append(SqlInfo::SqlDefaultTransactionIsolation, SqlTransactionIsolationLevel::SqlTransactionNone as i32);
    builder.append(SqlInfo::SqlTransactionsSupported, false);
    builder.append(SqlInfo::SqlSupportedTransactionsIsolationLevels, 0);
    builder.append(SqlInfo::SqlDataDefinitionCausesTransactionCommit, false);
    builder.append(SqlInfo::SqlDataDefinitionsInTransactionsIgnored, false);
    builder.append(
        SqlInfo::SqlSupportedResultSetTypes,
        sql_info_bitmask([SqlSupportedResultSetType::SqlResultSetTypeForwardOnly as i32]),
    );
    builder.append(SqlInfo::SqlSupportedConcurrenciesForResultSetUnspecified, 0);
    builder.append(
        SqlInfo::SqlSupportedConcurrenciesForResultSetForwardOnly,
        sql_info_bitmask([SqlSupportedResultSetConcurrency::SqlResultSetConcurrencyReadOnly as i32]),
    );
    builder.append(SqlInfo::SqlSupportedConcurrenciesForResultSetScrollSensitive, 0);
    builder.append(SqlInfo::SqlSupportedConcurrenciesForResultSetScrollInsensitive, 0);
    builder.append(SqlInfo::SqlBatchUpdatesSupported, false);
    builder.append(SqlInfo::SqlSavepointsSupported, false);
    builder.append(SqlInfo::SqlNamedParametersSupported, false);
    builder.append(SqlInfo::SqlLocatorsUpdateCopy, false);
    builder.append(SqlInfo::SqlStoredFunctionsUsingCallSyntaxSupported, false);
    builder.build().map_err(Status::from)
}

fn sql_info_bitmask(values: impl IntoIterator<Item = i32>) -> i32 {
    values.into_iter().fold(0, |mask, value| {
        let shift = u32::try_from(value).expect("Flight SQL SqlInfo bit index must be non-negative");
        mask | 1_i32.checked_shl(shift).expect("Flight SQL SqlInfo bit index must fit in i32")
    })
}
