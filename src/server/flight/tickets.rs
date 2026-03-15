use super::*;

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

#[derive(Debug)]
pub(super) struct StatementTicket {
    pub(super) query_id: String,
    pub(super) sql: String,
    pub(super) ticket_kind: StatementTicketKind,
    pub(super) native_cancellation: ExecutionCancellation,
    pub(super) started_at: Instant,
    pub(super) issued_at: Instant,
    pub(super) _permit: OwnedSemaphorePermit,
}

pub(super) struct QueryExecutionGuard {
    pub(super) query_id: String,
    pub(super) cancellations: Arc<QueryCancellation>,
    pub(super) native_cancel_bridge: tokio::task::JoinHandle<()>,
    pub(super) _permit: OwnedSemaphorePermit,
    pub(super) lifecycle: Arc<QueryLifecycle>,
}

impl Drop for QueryExecutionGuard {
    fn drop(&mut self) {
        self.native_cancel_bridge.abort();
        self.cancellations.remove(&self.query_id);
        self.lifecycle.finish_dropped();
    }
}

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
