use super::*;

/// Preserves which Flight SQL query path issued a ticket after the ticket bytes
/// have collapsed to a per-execution `query_id`.
///
/// Statement and prepared-statement queries share execution code, but clients
/// see different RPC surfaces and operation names. Keeping this tag in the
/// ticket lets errors, diagnostics, and ticket payloads stay tied to the path
/// the client actually used.
#[derive(Debug, Clone, Copy)]
pub(super) enum StatementTicketKind {
    Statement,
    PreparedStatement,
}

impl StatementTicketKind {
    pub(super) const fn code(self) -> u8 {
        match self {
            Self::Statement => 0,
            Self::PreparedStatement => 1,
        }
    }

    pub(super) fn from_code(code: u8) -> Result<Self, Status> {
        match code {
            0 => Ok(Self::Statement),
            1 => Ok(Self::PreparedStatement),
            _ => Err(Status::internal("query coordinator retained an invalid statement ticket kind")),
        }
    }

    pub(super) const fn create_logical_plan_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.get_flight_info_statement.create_logical_plan",
            Self::PreparedStatement => {
                "datafusion_nexus.server.flight.get_flight_info_prepared_statement.create_logical_plan"
            },
        }
    }

    pub(super) const fn sql_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_get_statement.sql",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_get_prepared_statement.sql",
        }
    }

    pub(super) const fn execute_stream_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_get_statement.execute_stream",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_get_prepared_statement.execute_stream",
        }
    }

    pub(super) const fn stream_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_get_statement.stream",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_get_prepared_statement.stream",
        }
    }
}

/// Keeps the two DoPut update channels on one execution path without losing
/// client-visible operation context.
///
/// Update statements do not pass through the coordinator's statement-ticket
/// handoff, so this is the
/// lightweight equivalent of `StatementTicketKind` for planning/collect errors
/// and tracing labels.
#[derive(Debug, Clone, Copy)]
pub(super) enum UpdateStatementKind {
    Statement,
    PreparedStatement,
}

impl UpdateStatementKind {
    pub(super) const fn into_execution_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_put_statement_update.into_execution",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_put_prepared_statement_update.into_execution",
        }
    }

    pub(super) const fn sql_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_put_statement_update.sql",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_put_prepared_statement_update.sql",
        }
    }

    pub(super) const fn collect_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_put_statement_update.collect",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_put_prepared_statement_update.collect",
        }
    }
}

/// Owns the resources that must live exactly as long as a lazy DoGet stream.
///
/// `DoGet` returns before the stream is fully polled, so ordinary stack cleanup
/// would release admission and cancellation state too early. The stream holds
/// this guard to keep those resources alive and to convert client drop into
/// cooperative native cancellation.
pub(super) struct QueryExecutionGuard {
    pub(super) query_id: String,
    pub(super) correlation_id: Option<String>,
    pub(super) cancellations: Arc<QueryCancellation>,
    pub(super) flight_token: CancellationToken,
    pub(super) native_cancellation: ExecutionCancellation,
    pub(super) native_cancel_bridge: tokio::task::JoinHandle<()>,
    pub(super) shutdown_token: CancellationToken,
    pub(super) terminal: Arc<CoordinatorExecutionTerminal>,
}

pub(super) struct CoordinatorExecutionTerminal {
    lease: Mutex<Option<QueryExecutionLease>>,
    query_id: String,
    started_at: Instant,
}

impl CoordinatorExecutionTerminal {
    pub(super) fn new(lease: QueryExecutionLease, query_id: String, started_at: Instant) -> Arc<Self> {
        Arc::new(Self { lease: Mutex::new(Some(lease)), query_id, started_at })
    }

    pub(super) fn is_finished(&self) -> bool {
        self.lease.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_none()
    }

    pub(super) fn finish_success(&self, stage: &'static str) {
        if let Some(lease) = self.lease.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            lease.succeed();
            tracing::info!(
                query_id = self.query_id,
                outcome = "succeeded",
                stage,
                elapsed_ms = self.started_at.elapsed().as_millis(),
                "finished Flight SQL statement"
            );
        }
    }

    pub(super) fn finish_captured(&self, stage: &'static str, status: &Status, captured: CapturedError) {
        if let Some(mut lease) = self.lease.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            lease.finish_captured(stage, status, captured);
            log_statement_status(&self.query_id, stage, self.started_at, status);
        }
    }

    pub(super) fn finish_panicked(&self, stage: &'static str) {
        if let Some(mut lease) = self.lease.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            lease.finish_panicked();
            tracing::error!(
                query_id = self.query_id,
                outcome = "panicked",
                stage,
                elapsed_ms = self.started_at.elapsed().as_millis(),
                "Flight SQL statement panicked"
            );
        }
    }
}

impl Drop for QueryExecutionGuard {
    fn drop(&mut self) {
        if !self.terminal.is_finished() {
            if std::thread::panicking() {
                self.terminal.finish_panicked("stream");
            } else {
                let (status, captured) = adapter_error_to_status_and_capture(cancellation_error(&self.shutdown_token));
                let status =
                    status_with_query_id_and_correlation_id(status, &self.query_id, self.correlation_id.as_deref());
                self.terminal.finish_captured("stream", &status, captured);
            }
            self.native_cancellation.cancel();
            self.flight_token.cancel();
        }
        self.native_cancel_bridge.abort();
        self.cancellations.remove(&self.query_id);
    }
}
