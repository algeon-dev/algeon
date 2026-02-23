use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    time::Instant,
};

use arrow_flight::{
    FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest, HandshakeResponse, Ticket,
    encode::FlightDataEncoderBuilder,
    error::FlightError,
    flight_service_server::FlightService,
    sql::{
        ActionCancelQueryRequest, ActionCancelQueryResult, CommandGetSqlInfo, CommandPreparedStatementQuery,
        CommandStatementQuery, ProstMessageExt, SqlInfo, TicketStatementQuery, metadata::SqlInfoDataBuilder,
        server::FlightSqlService,
    },
};
use dashmap::DashMap;
use datafusion::{error::DataFusionError, prelude::SessionContext};
use futures::{Stream, StreamExt, TryStreamExt};
use prost::Message;
use tokio::sync::OwnedSemaphorePermit;
use tokio_util::sync::CancellationToken;
use tonic::{Code, Request, Response, Status, Streaming, metadata::MetadataValue};
use uuid::Uuid;

use super::{
    admission::QueryAdmission,
    cancel::QueryCancellation,
    error::{ServerError, datafusion_error_to_status},
};

const FLIGHT_SQL_CANCEL_RESULT_CANCELLING: i32 = 2;
const FLIGHT_SQL_CANCEL_RESULT_NOT_CANCELLABLE: i32 = 3;
const QUERY_ID_METADATA_KEY: &str = "x-datafusion-cudf-query-id";

#[derive(Clone)]
pub struct DatafusionCudfFlightSqlService {
    ctx: Arc<SessionContext>,
    admission: Arc<QueryAdmission>,
    cancellations: Arc<QueryCancellation>,
    tickets: Arc<DashMap<String, StatementTicket>>,
    /// Parent token that every per-statement token is derived from. Cancelled
    /// by [`crate::server::serve`] on SIGINT/SIGTERM so any statement that
    /// has been admitted but not yet registered (or any statement still to
    /// register) sees a pre-cancelled token and stops at the first await.
    shutdown_token: CancellationToken,
}

#[derive(Debug, Clone)]
struct StatementTicket {
    query_id: String,
    sql: String,
}

impl DatafusionCudfFlightSqlService {
    pub fn new(ctx: Arc<SessionContext>, max_in_flight: usize) -> Self {
        Self {
            ctx,
            admission: Arc::new(QueryAdmission::new(max_in_flight).expect("validated server admission limit")),
            cancellations: Arc::default(),
            tickets: Arc::default(),
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

    async fn statement_flight_info(
        &self,
        query_id: String,
        sql: String,
        descriptor: FlightDescriptor,
    ) -> Result<FlightInfo, Status> {
        // Observe the server-wide shutdown token so a slow planning call here
        // does not keep tonic's graceful drain waiting. There is no per-query
        // child token yet (the ticket has not been issued), so we watch the
        // parent directly.
        let df = tokio::select! {
            result = self.ctx.sql(&sql) => result.map_err(datafusion_error_to_status)?,
            _ = self.shutdown_token.cancelled() => {
                return Err(server_shutdown_status(&query_id));
            }
        };
        let schema = df.schema().as_arrow().clone();

        let ticket = TicketStatementQuery { statement_handle: query_id.as_bytes().to_vec().into() };
        let endpoint = FlightEndpoint::new().with_ticket(Ticket { ticket: ticket.as_any().encode_to_vec().into() });
        self.tickets.insert(query_id.clone(), StatementTicket { query_id, sql });

        Ok(FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(arrow_error_to_status)?
            .with_descriptor(descriptor)
            .with_endpoint(endpoint)
            .with_ordered(false))
    }

    async fn execute_ticket(&self, query_id: String) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let (_, ticket) = self.tickets.remove(&query_id).ok_or_else(|| {
            let status = status_with_query_id(
                Status::invalid_argument(format!("unknown or already consumed statement ticket: {query_id}")),
                &query_id,
            );
            log_statement_status(&query_id, "ticket", Instant::now(), &status);
            status
        })?;

        let queued_at = Instant::now();
        let permit = self.admission.acquire().await.map_err(|error| {
            // If the semaphore closed because we are shutting down, surface
            // the queued statement as CANCELLED rather than INTERNAL — the
            // client did nothing wrong; the server is going away.
            let status = if self.shutdown_token.is_cancelled() {
                server_shutdown_status(&ticket.query_id)
            } else {
                server_error_to_status(error)
            };
            let status = status_with_query_id(status, &ticket.query_id);
            log_statement_status(&ticket.query_id, "admission", queued_at, &status);
            status
        })?;
        let admission_wait_ms = permit.wait_millis();
        if admission_wait_ms > 0 || self.admission.max_in_flight() > 1 {
            tracing::info!(
                query_id = ticket.query_id,
                admission_wait_ms,
                total_queue_wait_ms = queued_at.elapsed().as_millis(),
                max_in_flight = self.admission.max_in_flight(),
                "admitted Flight SQL statement"
            );
        }

        // Derive from `shutdown_token` so cancellation cascades to statements
        // that race in between permit acquisition and registry insert.
        let token = self.shutdown_token.child_token();
        let StatementTicket { query_id, sql } = ticket;
        self.cancellations.insert(query_id.clone(), token.clone());
        let lifecycle = Arc::new(QueryLifecycle::new(query_id.clone(), Instant::now()));
        // Construct the guard before planning so an early return from the
        // tokio::select! shutdown branches still releases the permit and
        // removes the cancellation entry on drop (avoids leaks when planning
        // or stream creation is abandoned).
        let guard = QueryExecutionGuard {
            query_id: query_id.clone(),
            cancellations: Arc::clone(&self.cancellations),
            _permit: permit.into_inner(),
            lifecycle: Arc::clone(&lifecycle),
        };
        let df = tokio::select! {
            result = self.ctx.sql(&sql) => match result {
                Ok(df) => df,
                Err(error) => {
                    let status = status_with_query_id(datafusion_error_to_status(error), &query_id);
                    lifecycle.finish_status("planning", &status);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
                let status = cancellation_status(&self.shutdown_token, &query_id);
                lifecycle.finish_status("planning", &status);
                return Err(status);
            },
        };
        let schema = df.schema().as_arrow().clone();
        let stream = tokio::select! {
            result = df.execute_stream() => match result {
                Ok(stream) => stream,
                Err(error) => {
                    let status = status_with_query_id(datafusion_error_to_status(error), &query_id);
                    lifecycle.finish_status("execute_stream", &status);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
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
        let info = match self.statement_flight_info(query_id.clone(), sql, request.into_inner()).await {
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
    _permit: OwnedSemaphorePermit,
    lifecycle: Arc<QueryLifecycle>,
}

impl Drop for QueryExecutionGuard {
    fn drop(&mut self) {
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
    builder.build().map_err(Status::from)
}
