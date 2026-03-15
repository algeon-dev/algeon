use super::*;

pub(super) fn datafusion_error_to_flight_error(
    error: DataFusionError,
    query_id: &str,
    lifecycle: &QueryLifecycle,
    operation: &'static str,
) -> FlightError {
    let status = status_with_query_id(datafusion_error_to_status_for_operation(error, operation), query_id);
    lifecycle.finish_status("stream", &status);
    FlightError::from(status)
}

pub(super) fn verify_flight_info_plan(plan: &LogicalPlan, ticket_kind: StatementTicketKind) -> Result<(), Status> {
    match ticket_kind {
        StatementTicketKind::Statement => verify_statement_query_plan(plan),
        StatementTicketKind::PreparedStatement => verify_prepared_query_plan(plan),
    }
}

pub(super) fn verify_statement_query_plan(plan: &LogicalPlan) -> Result<(), Status> {
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

pub(super) fn verify_prepared_query_plan(plan: &LogicalPlan) -> Result<(), Status> {
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

pub(super) fn verify_update_plan(plan: &LogicalPlan) -> Result<(), Status> {
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

pub(super) fn verify_sql_statement_namespace(
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

pub(super) fn verify_update_ddl(ddl: &DdlStatement, operation: &'static str) -> Result<(), Status> {
    match ddl {
        DdlStatement::CreateMemoryTable(_) | DdlStatement::CreateIndex(_) => Err(adapter_error_status(Error::new(
            ErrorCode::UnsupportedOperator,
            operation,
            "Flight SQL update DoPut does not yet support query-producing DDL such as CREATE TABLE AS SELECT",
        ))),
        _ => Ok(()),
    }
}

pub(super) fn reject_parameterized_plan(plan: &LogicalPlan, operation: &'static str) -> Result<(), Status> {
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

pub(super) fn arrow_error_to_status(error: arrow::error::ArrowError) -> Status {
    Status::internal(error.to_string())
}

/// Serialize an Arrow `Schema` into the IPC-encapsulated form that Flight SQL
/// uses for `ActionCreatePreparedStatementResult.dataset_schema` and
/// `parameter_schema`. Returns the `prost::bytes::Bytes` payload ready to drop
/// into the proto message.
pub(super) fn schema_to_ipc_bytes(schema: &Schema) -> Result<prost::bytes::Bytes, arrow::error::ArrowError> {
    let options = IpcWriteOptions::default();
    let IpcMessage(bytes) = IpcMessage::try_from(SchemaAsIpc::new(schema, &options))?;
    Ok(bytes)
}

pub(super) fn arrow_flight_error_to_status(error: arrow_flight::error::FlightError) -> Status {
    Status::from(error)
}

pub(super) fn server_error_to_status(error: ServerError) -> Status {
    Status::internal(error.to_string())
}

pub(super) fn adapter_error_status(error: Error) -> Status {
    let operation = error.operation();
    datafusion_error_to_status_for_operation(error.into(), operation)
}

pub(super) fn adapter_error_status_with_query_id(error: Error, query_id: &str) -> Status {
    status_with_query_id(adapter_error_status(error), query_id)
}

pub(super) fn flight_sql_transactions_unsupported_status(query_id: &str) -> Status {
    adapter_error_status_with_query_id(
        Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_nexus.server.flight.transaction",
            "Flight SQL transactions are not supported",
        ),
        query_id,
    )
}

pub(super) fn server_shutdown_status(query_id: &str) -> Status {
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
pub(super) fn cancellation_status(shutdown_token: &CancellationToken, query_id: &str) -> Status {
    if shutdown_token.is_cancelled() {
        server_shutdown_status(query_id)
    } else {
        adapter_error_status_with_query_id(
            Error::new(ErrorCode::Cancelled, "datafusion_nexus.server.flight.cancel_query", "statement was cancelled"),
            query_id,
        )
    }
}

pub(super) fn status_with_query_id(mut status: Status, query_id: &str) -> Status {
    if let Ok(value) = MetadataValue::try_from(query_id) {
        status.metadata_mut().insert(QUERY_ID_METADATA_KEY, value);
    }
    status
}

pub(super) fn status_outcome(code: Code) -> &'static str {
    if code == Code::Cancelled { "cancelled" } else { "failed" }
}

pub(super) fn log_statement_status(query_id: &str, stage: &'static str, started_at: Instant, status: &Status) {
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
