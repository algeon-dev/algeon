use std::time::Instant;

use arrow::error::ArrowError;
use arrow_flight::error::FlightError;
use datafusion::{common::Result as DataFusionResult, execution::SessionState};
use prost::bytes::Bytes;

use super::*;
use crate::{
    error::Result,
    server::{
        error::{ERROR_CODE_METADATA_KEY, REASON_METADATA_KEY},
        observability::RedactedSql,
    },
};

pub(super) fn datafusion_error_to_flight_error(
    error: DataFusionError,
    query_id: &str,
    terminal: &CoordinatorExecutionTerminal,
    operation: &'static str,
) -> FlightError {
    let (status, captured) = datafusion_error_to_status_and_capture(error, operation, FailedStatement::Other);
    let status = status_with_query_id(status, query_id);
    terminal.finish_captured("stream", &status, captured);
    FlightError::from(status)
}

pub(super) fn datafusion_error_to_status_and_capture(
    error: DataFusionError,
    operation: &'static str,
    statement: FailedStatement,
) -> (Status, CapturedError) {
    let captured = datafusion_error_to_captured_error(&error, operation, statement);
    let status = datafusion_error_to_status_for_operation(error, operation, statement);
    (status, captured)
}

pub(super) fn adapter_error_to_status_and_capture(error: Error) -> (Status, CapturedError) {
    let captured = CapturedError::from_adapter_error(&error);
    let status = adapter_error_status(error);
    (status, captured)
}

pub(super) fn arrow_error_to_status_and_capture(
    error: ArrowError,
    operation: &'static str,
    message: &'static str,
) -> (Status, CapturedError) {
    adapter_error_to_status_and_capture(Error::new(ErrorCode::Internal, operation, message).with_source(error))
}

pub(super) fn arrow_flight_error_to_status_and_capture(
    error: FlightError,
    operation: &'static str,
    message: &'static str,
) -> (Status, CapturedError) {
    match error {
        FlightError::Tonic(status) => {
            let status = *status;
            let captured = CapturedError::from_status(&status);
            (status, captured)
        },
        error => {
            adapter_error_to_status_and_capture(Error::new(ErrorCode::Internal, operation, message).with_source(error))
        },
    }
}

pub(super) fn verify_flight_info_plan(plan: &LogicalPlan, ticket_kind: StatementTicketKind) -> Result<()> {
    match ticket_kind {
        StatementTicketKind::Statement => verify_statement_query_plan(plan),
        StatementTicketKind::PreparedStatement => verify_prepared_query_plan(plan),
        StatementTicketKind::ExplainGpuRows | StatementTicketKind::ExplainGpuJson => Err(Error::new(
            ErrorCode::Internal,
            "algeon_datafusion.server.flight.get_flight_info_statement.explain_gpu",
            "EXPLAIN GPU must bypass ordinary Flight SQL logical-plan verification",
        )),
    }
}

/// Result-set schema declared by FlightInfo and prepared-statement metadata,
/// and applied by the DoGet encoder to every batch. SQL planning leaves some
/// output types provisional until the analyzer's `TypeCoercion` pass (a
/// `UNION` keeps its first branch's types), so the schema comes from the
/// analyzed plan; the encoder casts batches to whatever schema it receives.
pub(super) fn flight_result_schema(state: &SessionState, plan: &LogicalPlan) -> DataFusionResult<Schema> {
    match plan {
        LogicalPlan::Ddl(_) | LogicalPlan::Statement(Statement::SetVariable(_) | Statement::ResetVariable(_)) => {
            Ok(Schema::empty())
        },
        _ => {
            let analyzed = state.analyzer().execute_and_check(plan.clone(), state.config_options(), |_, _| {})?;
            Ok(analyzed.schema().as_arrow().clone())
        },
    }
}

pub(super) fn verify_statement_query_plan(plan: &LogicalPlan) -> Result<()> {
    match plan {
        LogicalPlan::Dml(_) | LogicalPlan::Copy(_) => Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "algeon_datafusion.server.flight.get_flight_info_statement.verify_plan",
            "Flight SQL statement query does not support DML or COPY statements",
        )),
        LogicalPlan::Statement(statement) => verify_sql_statement_namespace(
            statement,
            "algeon_datafusion.server.flight.get_flight_info_statement.verify_plan",
            "Flight SQL statement query does not support SQL session statements; use the update endpoint for session updates",
        ),
        _ => Ok(()),
    }
}

pub(super) fn verify_prepared_query_plan(plan: &LogicalPlan) -> Result<()> {
    match plan {
        // DDL carries no result set, but DataFusion executes it eagerly in
        // `execute_logical_plan` and yields an empty stream. The shared do_get
        // path (`execute_ticket`) runs that identically for statement and
        // prepared tickets, so prepared DDL flows through the query channel just
        // like a non-prepared statement (see `verify_statement_query_plan`).
        // JDBC clients such as DataGrip always wrap statements in a server-side
        // prepared statement and call `execute()`, never `execute_update()`, so
        // rejecting DDL here made `CREATE EXTERNAL TABLE` / `DROP` unrunnable
        // from those clients. We still reject query-producing DDL (CTAS / CREATE
        // INDEX) to match `verify_prepared_statement_plan`.
        LogicalPlan::Ddl(ddl) => {
            verify_update_ddl(ddl, "algeon_datafusion.server.flight.get_flight_info_prepared_statement.verify_plan")
        },
        LogicalPlan::Dml(_) | LogicalPlan::Copy(_) => Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "algeon_datafusion.server.flight.get_flight_info_prepared_statement.verify_plan",
            "Flight SQL prepared DML and COPY statements are not supported",
        )),
        LogicalPlan::Statement(statement) => verify_sql_statement_namespace(
            statement,
            "algeon_datafusion.server.flight.get_flight_info_prepared_statement.verify_plan",
            "Flight SQL prepared session-update statements must be executed with execute_update",
        ),
        _ => Ok(()),
    }
}

pub(super) fn verify_update_plan(plan: &LogicalPlan) -> Result<()> {
    match plan {
        LogicalPlan::Ddl(ddl) => verify_update_ddl(ddl, "algeon_datafusion.server.flight.do_put_update.verify_plan"),
        LogicalPlan::Dml(_) | LogicalPlan::Copy(_) => Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "algeon_datafusion.server.flight.do_put_update.verify_plan",
            "Flight SQL update DoPut does not yet support DML or COPY statements",
        )),
        LogicalPlan::Statement(statement) => verify_sql_statement_namespace(
            statement,
            "algeon_datafusion.server.flight.do_put_update.verify_plan",
            "SQL PREPARE, EXECUTE, and DEALLOCATE are not supported through Flight SQL update DoPut",
        ),
        _ => Err(Error::new(
            ErrorCode::InvalidRequest,
            "algeon_datafusion.server.flight.do_put_update.verify_plan",
            "Flight SQL update DoPut only supports DDL and session-update statements; use the statement query endpoint for SELECT",
        )),
    }
}

pub(super) fn verify_sql_statement_namespace(
    statement: &Statement,
    operation: &'static str,
    session_update_message: &'static str,
) -> Result<()> {
    match statement {
        Statement::SetVariable(set) if operation == "algeon_datafusion.server.flight.do_put_update.verify_plan" => {
            reject_flight_algeon_datafusion_session_key(&set.variable, operation)
        },
        Statement::SetVariable(_) | Statement::ResetVariable(_) => {
            if operation == "algeon_datafusion.server.flight.do_put_update.verify_plan" {
                Ok(())
            } else {
                Err(Error::new(ErrorCode::InvalidRequest, operation, session_update_message))
            }
        },
        Statement::TransactionStart(_) | Statement::TransactionEnd(_) => {
            Err(Error::new(ErrorCode::UnsupportedOperator, operation, "Flight SQL transactions are not supported"))
        },
        Statement::Prepare(_) | Statement::Execute(_) | Statement::Deallocate(_) => Err(Error::new(
            ErrorCode::InvalidRequest,
            operation,
            "SQL PREPARE, EXECUTE, and DEALLOCATE are not supported through Flight SQL",
        )),
    }
}

fn reject_flight_algeon_datafusion_session_key(variable: &str, operation: &'static str) -> Result<()> {
    let normalized = variable.trim().to_ascii_lowercase();
    if normalized != "algeon_datafusion" && !normalized.starts_with("algeon_datafusion.") {
        return Ok(());
    }
    Err(Error::new(
        ErrorCode::InvalidRequest,
        operation,
        "Flight SQL does not support session-scoped algeon-datafusion keys; configure algeon-datafusion request policy through server configuration or admission defaults",
    )
    .with_log_safe_fact("reason", "flight_session_scoped_algeon_datafusion_keys_unsupported")
    .with_fact("config_key", variable)
    .with_log_safe_fact("configuration_scope", "server_or_admission_defaults"))
}

pub(super) fn verify_update_ddl(ddl: &DdlStatement, operation: &'static str) -> Result<()> {
    match ddl {
        DdlStatement::CreateMemoryTable(_) | DdlStatement::CreateIndex(_) => Err(Error::new(
            ErrorCode::UnsupportedOperator,
            operation,
            "Flight SQL update DoPut does not yet support query-producing DDL such as CREATE TABLE AS SELECT",
        )),
        _ => Ok(()),
    }
}

pub(super) fn reject_parameterized_plan(
    plan: &LogicalPlan,
    operation: &'static str,
) -> std::result::Result<(), DataFusionError> {
    let parameter_names = plan.get_parameter_names()?;
    if parameter_names.is_empty() {
        return Ok(());
    }
    Err(Error::new(
        ErrorCode::InvalidRequest,
        operation,
        "Flight SQL parameter binding is not supported; prepare statements without placeholders",
    )
    .into())
}

pub(super) fn arrow_error_to_status_for_operation(
    error: ArrowError,
    operation: &'static str,
    message: &'static str,
) -> Status {
    adapter_error_to_status(&Error::new(ErrorCode::Internal, operation, message).with_source(error))
}

/// Serialize an Arrow `Schema` into the IPC-encapsulated form that Flight SQL
/// uses for `ActionCreatePreparedStatementResult.dataset_schema` and
/// `parameter_schema`. Returns the `prost::bytes::Bytes` payload ready to drop
/// into the proto message.
pub(super) fn schema_to_ipc_bytes(schema: &Schema) -> std::result::Result<Bytes, ArrowError> {
    let options = IpcWriteOptions::default();
    let IpcMessage(bytes) = IpcMessage::try_from(SchemaAsIpc::new(schema, &options))?;
    Ok(bytes)
}

pub(super) fn arrow_flight_error_to_status_for_operation(
    error: FlightError,
    operation: &'static str,
    message: &'static str,
) -> Status {
    match error {
        FlightError::Tonic(status) => *status,
        error => adapter_error_to_status(&Error::new(ErrorCode::Internal, operation, message).with_source(error)),
    }
}

pub(super) fn adapter_error_status(error: Error) -> Status {
    let operation = error.operation();
    datafusion_error_to_status_for_operation(error.into(), operation, FailedStatement::Other)
}

pub(super) fn flight_sql_transactions_unsupported_error() -> Error {
    Error::new(
        ErrorCode::UnsupportedOperator,
        "algeon_datafusion.server.flight.transaction",
        "Flight SQL transactions are not supported",
    )
}

pub(super) fn server_shutdown_error() -> Error {
    Error::new(
        ErrorCode::Cancelled,
        "algeon_datafusion.server.flight.shutdown",
        "server shutting down; statement aborted",
    )
}

/// Disambiguate per-statement cancellation: if the parent shutdown token has
/// been cancelled the whole server is going away; otherwise it was an
/// `ActionCancelQuery` or dropped DoGet stream from the client.
pub(super) fn cancellation_error(shutdown_token: &CancellationToken) -> Error {
    if shutdown_token.is_cancelled() {
        server_shutdown_error()
    } else {
        Error::new(ErrorCode::Cancelled, "algeon_datafusion.server.flight.cancel_query", "statement was cancelled")
    }
}

pub(super) fn status_with_query_id(mut status: Status, query_id: &str) -> Status {
    if let Ok(value) = MetadataValue::try_from(query_id) {
        status.metadata_mut().insert(QUERY_ID_METADATA_KEY, value);
    }
    status
}

pub(super) fn status_with_query_id_and_correlation_id(
    mut status: Status,
    query_id: &str,
    correlation_id: Option<&str>,
) -> Status {
    if let Ok(value) = MetadataValue::try_from(query_id) {
        status.metadata_mut().insert(QUERY_ID_METADATA_KEY, value);
    }
    if let Some(correlation_id) = correlation_id
        && let Ok(value) = MetadataValue::try_from(correlation_id)
    {
        status.metadata_mut().insert(CORRELATION_ID_METADATA_KEY, value);
    }
    status
}

pub(super) fn status_outcome(code: Code) -> &'static str {
    if code == Code::Cancelled { "cancelled" } else { "failed" }
}

pub(super) fn log_statement_status(
    query_id: &str,
    sql: Option<&str>,
    stage: &'static str,
    started_at: Instant,
    status: &Status,
) {
    let outcome = status_outcome(status.code());
    // The status message can name the caller's columns, types, and values;
    // the operational log keeps only the bounded code and reason metadata.
    let metadata = |key| status.metadata().get(key).and_then(|value| value.to_str().ok());
    let error_code = metadata(ERROR_CODE_METADATA_KEY);
    let reason = metadata(REASON_METADATA_KEY);
    if let Some(sql) = sql {
        tracing::warn!(
            query_id,
            sql = %RedactedSql(sql),
            outcome,
            stage,
            grpc_code = ?status.code(),
            error_code,
            reason,
            elapsed_ms = started_at.elapsed().as_millis(),
            "failed Flight SQL statement"
        );
    } else {
        tracing::warn!(
            query_id,
            outcome,
            stage,
            grpc_code = ?status.code(),
            error_code,
            reason,
            elapsed_ms = started_at.elapsed().as_millis(),
            "failed Flight SQL statement"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::error::ERROR_CODE_METADATA_KEY;

    #[test]
    fn arrow_flight_tonic_status_is_passed_through() {
        let mut status = Status::permission_denied("source credentials rejected");
        status.metadata_mut().insert(
            ERROR_CODE_METADATA_KEY,
            MetadataValue::try_from("source_access_denied").expect("metadata value should be valid"),
        );

        let projected = arrow_flight_error_to_status_for_operation(
            FlightError::from(status.clone()),
            "algeon_datafusion.server.flight.test.encode",
            "encode failed",
        );

        assert_eq!(projected.code(), status.code());
        assert_eq!(projected.message(), status.message());
        assert_eq!(
            projected.metadata().get(ERROR_CODE_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("source_access_denied")
        );
    }
}
