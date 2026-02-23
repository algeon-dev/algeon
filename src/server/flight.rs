use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use arrow_flight::{
    FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest, HandshakeResponse, Ticket,
    encode::FlightDataEncoderBuilder,
    error::FlightError,
    flight_service_server::FlightService,
    sql::{
        ActionCancelQueryRequest, ActionCancelQueryResult, CommandGetSqlInfo, CommandPreparedStatementQuery,
        CommandStatementQuery, ProstMessageExt, SqlInfo, SqlNullOrdering, SqlOuterJoinsSupportLevel,
        SqlSupportedCaseSensitivity, SqlSupportedGroupBy, SqlSupportedResultSetConcurrency, SqlSupportedResultSetType,
        SqlSupportedTransaction, SqlSupportedUnions, SqlTransactionIsolationLevel, SupportedSqlGrammar,
        TicketStatementQuery, metadata::SqlInfoDataBuilder, server::FlightSqlService,
    },
};
use cudf_query_engine::ExecutionCancellation;
use dashmap::DashMap;
use datafusion::{error::DataFusionError, execution::SessionStateBuilder, prelude::SessionContext};
use futures::{Stream, StreamExt, TryStreamExt};
use prost::Message;
use tokio::sync::OwnedSemaphorePermit;
use tokio_util::sync::CancellationToken;
use tonic::{Code, Request, Response, Status, Streaming, metadata::MetadataValue};
use uuid::Uuid;

use super::{
    admission::QueryAdmission,
    cancel::QueryCancellation,
    config::ServerSessionProfile,
    diagnostics::{log_leaf_guard_rejection, log_native_planning_report},
    error::{ServerError, datafusion_error_to_status},
};
use crate::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt};

const FLIGHT_SQL_CANCEL_RESULT_CANCELLING: i32 = 2;
const FLIGHT_SQL_CANCEL_RESULT_NOT_CANCELLABLE: i32 = 3;
const SQL_ANSI92_ENTRY_SQL_BIT_INDEX: i32 = 0;
const QUERY_ID_METADATA_KEY: &str = "x-datafusion-cudf-query-id";
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
    statement_ticket_ttl: Duration,
    /// Parent token that every per-statement token is derived from. Cancelled
    /// by [`crate::server::serve`] on SIGINT/SIGTERM so any statement that
    /// has been admitted but not yet registered (or any statement still to
    /// register) sees a pre-cancelled token and stops at the first await.
    shutdown_token: CancellationToken,
}

/// Per-statement session context paired with the retained native rule that
/// will plan it. The native rule reference is kept alive for the lifetime of
/// planning so [`CudfNativePhysicalOptimizerRule::last_report`] can be read
/// after physical optimization runs (during `DataFrame::execute_stream`).
struct QueryPlanning {
    ctx: SessionContext,
    native_rule: Arc<CudfNativePhysicalOptimizerRule>,
}

#[derive(Debug)]
struct StatementTicket {
    query_id: String,
    sql: String,
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
    ) -> Self {
        Self {
            ctx,
            native_optimizer_config,
            session_profile,
            admission: Arc::new(QueryAdmission::new(max_in_flight).expect("validated server admission limit")),
            cancellations: Arc::default(),
            tickets: Arc::default(),
            statement_ticket_ttl,
            shutdown_token: CancellationToken::new(),
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

    fn query_planning(&self, cancellation: ExecutionCancellation) -> QueryPlanning {
        let execution_config =
            self.native_optimizer_config.native_execution_config().clone().with_cancellation(cancellation);
        let optimizer_config = self.native_optimizer_config.clone().with_native_execution_config(execution_config);
        // Retain the native rule so we can read its `last_report` for the
        // Phase 4 planning-report log after physical optimization runs.
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
    ) -> Result<FlightInfo, Status> {
        // Observe the server-wide shutdown token so a slow planning call here
        // does not keep tonic's graceful drain waiting. There is no per-query
        // child token yet (the ticket has not been issued), so we watch the
        // parent directly.
        //
        // The native cancellation flag is constructed here so the optimizer-rule
        // wiring in `query_context` shares it with planning. Until the ticket is
        // inserted into `self.tickets`, the flag is not reachable via
        // `do_action_cancel_query` — that's the documented pre-ticket window in
        // the Bug 9 audit. Only the parent `shutdown_token` arm below or a
        // failure path can flip the flag during this window, which is sufficient
        // because no native execution has started yet (this method only runs
        // logical/physical planning to derive the result schema).
        let native_cancellation = ExecutionCancellation::new();
        let planning = self.query_planning(native_cancellation.clone());
        let query_ctx = &planning.ctx;
        let df = tokio::select! {
            result = query_ctx.sql(&sql) => result.map_err(datafusion_error_to_status)?,
            _ = self.shutdown_token.cancelled() => {
                native_cancellation.cancel();
                return Err(server_shutdown_status(&query_id));
            }
        };
        let schema = df.schema().as_arrow().clone();

        let ticket = TicketStatementQuery { statement_handle: query_id.as_bytes().to_vec().into() };
        let endpoint = FlightEndpoint::new().with_ticket(Ticket { ticket: ticket.as_any().encode_to_vec().into() });
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
            let status = status_with_query_id(
                Status::invalid_argument(format!("unknown, expired, or already consumed statement ticket: {query_id}")),
                &query_id,
            );
            log_statement_status(&query_id, "ticket", Instant::now(), &status);
            status
        })?;

        // Derive from `shutdown_token` so cancellation cascades to statements
        // that race in between permit acquisition and registry insert.
        let token = self.shutdown_token.child_token();
        let StatementTicket { query_id, sql, native_cancellation, started_at, _permit: permit, .. } = ticket;
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
                    let status = status_with_query_id(datafusion_error_to_status(error), &query_id);
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
        // `df.execute_stream()` calls `create_physical_plan()` internally,
        // which runs the cudf native physical optimizer rule plus the Phase 1
        // GPU-only leaf guard. Both branches below read `last_report` after
        // that point so the planning report log reflects whatever physical
        // planning produced. The leaf-guard warn line only fires on the
        // failure branch because that is where the guard's structured error
        // surfaces.
        let stream = tokio::select! {
            result = df.execute_stream() => match result {
                Ok(stream) => {
                    log_native_planning_report(&query_id, planning.native_rule.last_report().as_ref());
                    stream
                },
                Err(error) => {
                    log_native_planning_report(&query_id, planning.native_rule.last_report().as_ref());
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    let status = status_with_query_id(datafusion_error_to_status(error), &query_id);
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
        // The body stream ends either naturally or because the token was
        // cancelled. `take_until` does not signal an error on the latter, so
        // a client would see OK end-of-stream with only a prefix of rows.
        // Chain a one-shot tail that, *after* the body ends, emits a
        // CANCELLED status if the token fired. This is observed as a gRPC
        // trailer status rather than a clean stream completion.
        let body = stream
            .map_err(move |error| datafusion_error_to_flight_error(error, &query_id_for_body, &lifecycle_for_body))
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
            lifecycle_for_end.finish_success();
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
        let Some(statement_ticket) =
            any.unpack::<TicketStatementQuery>().map_err(|error| Status::invalid_argument(error.to_string()))?
        else {
            return Ok(false);
        };
        let query_id = std::str::from_utf8(&statement_ticket.statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        if let Some((_, ticket)) = self.tickets.remove(query_id) {
            ticket.native_cancellation.cancel();
            tracing::info!(query_id, outcome = "cancelled", phase = "ticket", "cancelled Flight SQL statement ticket");
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
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>>, Status> {
        let output = futures::stream::iter([Ok(HandshakeResponse { protocol_version: 0, payload: vec![].into() })]);
        Ok(Response::new(Box::pin(output)))
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let sql = query.query;
        let planning_started_at = Instant::now();
        tracing::info!(query_id, sql, "planning Flight SQL statement");
        if query.transaction_id.is_some() {
            let status =
                status_with_query_id(Status::unimplemented("Flight SQL transactions are not supported"), &query_id);
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
            .statement_flight_info(query_id.clone(), sql, request.into_inner(), planning_started_at, permit)
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

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let query_id = std::str::from_utf8(&ticket.statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
        self.execute_ticket(query_id).await
    }

    async fn get_flight_info_sql_info(
        &self,
        query: CommandGetSqlInfo,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
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

    async fn do_get_sql_info(
        &self,
        query: CommandGetSqlInfo,
        _request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
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

    async fn get_flight_info_prepared_statement(
        &self,
        _query: CommandPreparedStatementQuery,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("prepared statement execution is not implemented"))
    }

    async fn do_get_prepared_statement(
        &self,
        _query: CommandPreparedStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        Err(Status::unimplemented("prepared statement execution is not implemented"))
    }

    async fn do_action_cancel_query(
        &self,
        query: ActionCancelQueryRequest,
        _request: Request<arrow_flight::Action>,
    ) -> Result<ActionCancelQueryResult, Status> {
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

    fn finish_success(&self) {
        if self.try_finish("succeeded") {
            tracing::info!(
                query_id = self.query_id,
                outcome = "succeeded",
                phase = "stream",
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
                phase = "stream",
                elapsed_ms = self.started_at.elapsed().as_millis(),
                "finished Flight SQL statement"
            );
        }
    }

    fn finish_status(&self, phase: &'static str, status: &Status) {
        let outcome = status_outcome(status.code());
        if self.try_finish(outcome) {
            log_statement_status(&self.query_id, phase, self.started_at, status);
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

fn datafusion_error_to_flight_error(error: DataFusionError, query_id: &str, lifecycle: &QueryLifecycle) -> FlightError {
    let status = status_with_query_id(datafusion_error_to_status(error), query_id);
    lifecycle.finish_status("stream", &status);
    FlightError::from(status)
}

fn arrow_error_to_status(error: arrow::error::ArrowError) -> Status {
    Status::internal(error.to_string())
}

fn server_error_to_status(error: ServerError) -> Status {
    Status::internal(error.to_string())
}

fn server_shutdown_status(query_id: &str) -> Status {
    status_with_query_id(Status::cancelled(format!("server shutting down; statement {query_id} aborted")), query_id)
}

/// Disambiguate per-statement cancellation: if the parent shutdown token has
/// been cancelled the whole server is going away; otherwise it was an
/// `ActionCancelQuery` from the client.
fn cancellation_status(shutdown_token: &CancellationToken, query_id: &str) -> Status {
    if shutdown_token.is_cancelled() {
        server_shutdown_status(query_id)
    } else {
        status_with_query_id(Status::cancelled(format!("statement {query_id} was cancelled")), query_id)
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

fn log_statement_status(query_id: &str, phase: &'static str, started_at: Instant, status: &Status) {
    let outcome = status_outcome(status.code());
    tracing::warn!(
        query_id,
        outcome,
        phase,
        grpc_code = ?status.code(),
        elapsed_ms = started_at.elapsed().as_millis(),
        error = status.message(),
        "failed Flight SQL statement"
    );
}

fn sql_info_data() -> Result<arrow_flight::sql::metadata::SqlInfoData, Status> {
    let mut builder = SqlInfoDataBuilder::new();
    builder.append(SqlInfo::FlightSqlServerName, "datafusion-cudf");
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
