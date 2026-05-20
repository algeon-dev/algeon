use std::{pin::Pin, sync::Arc, time::Instant};

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
use tonic::{Request, Response, Status, Streaming};
use uuid::Uuid;

use super::{
    admission::QueryAdmission,
    cancel::QueryCancellation,
    error::{ServerError, datafusion_error_to_status},
};

const FLIGHT_SQL_CANCEL_RESULT_CANCELLING: i32 = 2;
const FLIGHT_SQL_CANCEL_RESULT_NOT_CANCELLABLE: i32 = 3;

#[derive(Clone)]
pub struct DatafusionCudfFlightSqlService {
    ctx: Arc<SessionContext>,
    admission: Arc<QueryAdmission>,
    cancellations: Arc<QueryCancellation>,
    tickets: Arc<DashMap<String, StatementTicket>>,
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
        }
    }

    async fn statement_flight_info(
        &self,
        query_id: String,
        sql: String,
        descriptor: FlightDescriptor,
    ) -> Result<FlightInfo, Status> {
        let df = self.ctx.sql(&sql).await.map_err(datafusion_error_to_status)?;
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
            Status::invalid_argument(format!("unknown or already consumed statement ticket: {query_id}"))
        })?;

        let queued_at = Instant::now();
        tracing::info!(query_id = ticket.query_id, sql = ticket.sql, "queued Flight SQL statement");
        let permit = self.admission.acquire().await.map_err(server_error_to_status)?;
        tracing::info!(
            query_id = ticket.query_id,
            admission_wait_ms = permit.wait_millis(),
            total_queue_wait_ms = queued_at.elapsed().as_millis(),
            "admitted Flight SQL statement"
        );

        let token = CancellationToken::new();
        self.cancellations.insert(ticket.query_id.clone(), token.clone());
        let df = self.ctx.sql(&ticket.sql).await.map_err(datafusion_error_to_status)?;
        let schema = df.schema().as_arrow().clone();
        let stream = df.execute_stream().await.map_err(datafusion_error_to_status)?;
        let guard = QueryExecutionGuard {
            query_id: ticket.query_id,
            sql: ticket.sql,
            cancellations: Arc::clone(&self.cancellations),
            _permit: permit.into_inner(),
            started_at: Instant::now(),
        };
        let stream =
            stream.map_err(datafusion_error_to_flight_error).take_until(token.cancelled_owned()).map(move |batch| {
                let _guard = &guard;
                batch
            });
        let stream = FlightDataEncoderBuilder::new().with_schema(Arc::new(schema)).build(stream).map_err(Status::from);

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
        if query.transaction_id.is_some() {
            return Err(Status::unimplemented("Flight SQL transactions are not supported"));
        }
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let sql = query.query;
        tracing::info!(query_id, sql, "planning Flight SQL statement");
        let info = self.statement_flight_info(query_id, sql, request.into_inner()).await?;
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
    sql: String,
    cancellations: Arc<QueryCancellation>,
    _permit: OwnedSemaphorePermit,
    started_at: Instant,
}

impl Drop for QueryExecutionGuard {
    fn drop(&mut self) {
        self.cancellations.remove(&self.query_id);
        tracing::info!(
            query_id = self.query_id,
            sql = self.sql,
            elapsed_ms = self.started_at.elapsed().as_millis(),
            "finished Flight SQL statement stream"
        );
    }
}

fn datafusion_error_to_flight_error(error: DataFusionError) -> FlightError {
    FlightError::ExternalError(Box::new(error))
}

fn arrow_error_to_status(error: arrow::error::ArrowError) -> Status {
    Status::internal(error.to_string())
}

fn server_error_to_status(error: ServerError) -> Status {
    Status::internal(error.to_string())
}

fn sql_info_data() -> Result<arrow_flight::sql::metadata::SqlInfoData, Status> {
    let mut builder = SqlInfoDataBuilder::new();
    builder.append(SqlInfo::FlightSqlServerName, "datafusion-cudf");
    builder.append(SqlInfo::FlightSqlServerVersion, env!("CARGO_PKG_VERSION"));
    builder.build().map_err(Status::from)
}
