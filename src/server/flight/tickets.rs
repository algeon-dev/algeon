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
/// Update statements do not pass through `StatementTicket`, so this is the
/// lightweight equivalent of `StatementTicketKind` for planning/collect errors
/// and tracing labels.
#[derive(Debug, Clone, Copy)]
pub(super) enum UpdateStatementKind {
    Statement,
    PreparedStatement,
}

impl UpdateStatementKind {
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

/// Handoff record between `GetFlightInfo` and `DoGet`.
///
/// Flight SQL splits statement execution across RPCs: `GetFlightInfo` plans
/// enough to return schema and ticket bytes, while `DoGet` later consumes the
/// ticket and runs the query. This record keeps the per-execution state that
/// must survive that gap: admission ownership, cancellation handle,
/// correlation id, and prepared-statement dedup context.
#[derive(Debug)]
pub(super) struct StatementTicket {
    pub(super) query_id: String,
    pub(super) sql: String,
    pub(super) correlation_id: Option<String>,
    pub(super) ticket_kind: StatementTicketKind,
    /// The originating prepared-statement handle, when this ticket was issued
    /// from `get_flight_info_prepared_statement`. `None` for non-prepared
    /// statement tickets. Used to deduplicate repeated no-result-set DDL
    /// execution on the query channel (see `PreparedStatementEntry`).
    pub(super) prepared_handle: Option<String>,
    pub(super) native_cancellation: ExecutionCancellation,
    pub(super) started_at: Instant,
    pub(super) issued_at: Instant,
    pub(super) _permit: QueryAdmissionPermit,
}

/// Owns the resources that must live exactly as long as a lazy DoGet stream.
///
/// `DoGet` returns before the stream is fully polled, so ordinary stack cleanup
/// would release admission and cancellation state too early. The stream holds
/// this guard to keep those resources alive and to seal observability on every
/// exit path, including client drop.
pub(super) struct QueryExecutionGuard {
    pub(super) query_id: String,
    pub(super) cancellations: Arc<QueryCancellation>,
    pub(super) observability: Arc<QueryObservability>,
    pub(super) native_cancel_bridge: tokio::task::JoinHandle<()>,
    pub(super) _permit: QueryAdmissionPermit,
    pub(super) lifecycle: Arc<QueryLifecycle>,
}

impl Drop for QueryExecutionGuard {
    fn drop(&mut self) {
        self.native_cancel_bridge.abort();
        self.cancellations.remove(&self.query_id);
        self.observability.seal(&self.query_id);
        self.lifecycle.finish_dropped();
    }
}

/// Serializes terminal accounting for a statement.
///
/// Planning errors, stream errors, cancellation trailers, successful end of
/// stream, and guard drop can race through different async paths. This latch
/// makes the first terminal outcome authoritative so logs and query lifecycle
/// state do not double-finish the same query.
pub(super) struct QueryLifecycle {
    pub(super) query_id: String,
    pub(super) started_at: Instant,
    pub(super) terminal: Mutex<Option<&'static str>>,
}

impl QueryLifecycle {
    pub(super) fn new(query_id: String, started_at: Instant) -> Self {
        Self { query_id, started_at, terminal: Mutex::new(None) }
    }

    pub(super) fn finish_success(&self, stage: &'static str) {
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

    pub(super) fn finish_status(&self, stage: &'static str, status: &Status) {
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
