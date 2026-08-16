use nexus_query_engine::observability::RuntimeTerminalOutcome;

use super::*;
use crate::coordinator::QueryKey;

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
    ExplainGpuRows,
    ExplainGpuJson,
}

impl StatementTicketKind {
    pub(super) const fn code(self) -> u8 {
        match self {
            Self::Statement => 0,
            Self::PreparedStatement => 1,
            Self::ExplainGpuRows => 2,
            Self::ExplainGpuJson => 3,
        }
    }

    pub(super) fn from_code(code: u8) -> Result<Self, Status> {
        match code {
            0 => Ok(Self::Statement),
            1 => Ok(Self::PreparedStatement),
            2 => Ok(Self::ExplainGpuRows),
            3 => Ok(Self::ExplainGpuJson),
            _ => Err(Status::internal("query coordinator retained an invalid statement ticket kind")),
        }
    }

    pub(super) const fn create_logical_plan_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.get_flight_info_statement.create_logical_plan",
            Self::PreparedStatement => {
                "datafusion_nexus.server.flight.get_flight_info_prepared_statement.create_logical_plan"
            },
            Self::ExplainGpuRows | Self::ExplainGpuJson => {
                "datafusion_nexus.server.flight.get_flight_info_statement.explain_gpu"
            },
        }
    }

    pub(super) const fn sql_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_get_statement.sql",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_get_prepared_statement.sql",
            Self::ExplainGpuRows | Self::ExplainGpuJson => {
                "datafusion_nexus.server.flight.do_get_statement.explain_gpu"
            },
        }
    }

    pub(super) const fn metadata_logical_proof_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.get_flight_info_statement.metadata_logical_proof",
            Self::PreparedStatement => {
                "datafusion_nexus.server.flight.get_flight_info_prepared_statement.metadata_logical_proof"
            },
            Self::ExplainGpuRows | Self::ExplainGpuJson => {
                "datafusion_nexus.server.flight.get_flight_info_statement.explain_gpu"
            },
        }
    }

    pub(super) const fn execute_metadata_logical_proof_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_get_statement.metadata_logical_proof",
            Self::PreparedStatement => {
                "datafusion_nexus.server.flight.do_get_prepared_statement.metadata_logical_proof"
            },
            Self::ExplainGpuRows | Self::ExplainGpuJson => {
                "datafusion_nexus.server.flight.do_get_statement.explain_gpu"
            },
        }
    }

    pub(super) const fn execute_stream_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_get_statement.execute_stream",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_get_prepared_statement.execute_stream",
            Self::ExplainGpuRows | Self::ExplainGpuJson => {
                "datafusion_nexus.server.flight.do_get_statement.explain_gpu"
            },
        }
    }

    pub(super) const fn stream_operation(self) -> &'static str {
        match self {
            Self::Statement => "datafusion_nexus.server.flight.do_get_statement.stream",
            Self::PreparedStatement => "datafusion_nexus.server.flight.do_get_prepared_statement.stream",
            Self::ExplainGpuRows | Self::ExplainGpuJson => {
                "datafusion_nexus.server.flight.do_get_statement.explain_gpu"
            },
        }
    }

    pub(super) const fn explain_gpu_output_mode(self) -> Option<ExplainGpuOutputMode> {
        match self {
            Self::ExplainGpuRows => Some(ExplainGpuOutputMode::Rows),
            Self::ExplainGpuJson => Some(ExplainGpuOutputMode::Json),
            Self::Statement | Self::PreparedStatement => None,
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
    pub(super) attempt_cancellation: Option<nexus_query_engine::attempt::AttemptCancellation>,
    pub(super) shutdown_token: CancellationToken,
    /// Set only after Flight has polled the execution body and therefore
    /// started response delivery beyond its schema frame.
    pub(super) delivery_started: Arc<std::sync::atomic::AtomicBool>,
    pub(super) terminal: Arc<CoordinatorExecutionTerminal>,
}

pub(super) struct CoordinatorExecutionTerminal {
    state: Mutex<CoordinatorExecutionTerminalState>,
    finished: Notify,
    key: QueryKey,
    query_id: String,
    sql: Arc<str>,
    started_at: Instant,
}

/// Completion target retained by a sealed attempt-report observer.
///
/// Planning does not keep an execution lease, but it does retain an exact
/// coordinator generation through `QueryPlanningReportDeliveryTerminal`. That
/// lets a planning-stage handoff failure remove only its own report gate after
/// the planning lease has either terminalized or moved into a ticket.
pub(super) enum CoordinatorReportHandoffTerminal {
    Planning(QueryPlanningReportDeliveryTerminal),
    Execution(Arc<CoordinatorExecutionTerminal>),
}

impl CoordinatorReportHandoffTerminal {
    pub(super) fn planning(lease: &QueryPlanningLease) -> Self {
        Self::Planning(lease.attempt_report_delivery_terminal())
    }

    pub(super) fn execution(terminal: Arc<CoordinatorExecutionTerminal>) -> Self {
        Self::Execution(terminal)
    }

    pub(super) fn finish_attempt_report_delivery_failed(
        &self,
        error: QueryCoordinatorError,
        attempt_outcome: Option<RuntimeTerminalOutcome>,
    ) {
        match self {
            Self::Planning(terminal) => terminal.finish_attempt_report_delivery_failed(attempt_outcome),
            Self::Execution(terminal) => terminal.finish_attempt_report_delivery_failed(error, attempt_outcome),
        }
    }

    pub(super) fn key(&self) -> &QueryKey {
        match self {
            Self::Planning(terminal) => terminal.key(),
            Self::Execution(terminal) => terminal.key(),
        }
    }

    pub(super) fn mark_attempt_report_attached(&self) {
        if let Self::Execution(terminal) = self {
            terminal.mark_attempt_report_attached();
        }
    }
}

struct CoordinatorExecutionTerminalState {
    lease: Option<QueryExecutionLease>,
    requires_attempt_report: bool,
    attempt_report_attached: bool,
    pending: Option<CoordinatorExecutionTerminalOutcome>,
}

enum CoordinatorExecutionTerminalOutcome {
    Succeeded { stage: &'static str },
    Captured { stage: &'static str, status: Status, captured: Box<CapturedError> },
    Dropped { stage: &'static str },
    Panicked { stage: &'static str },
}

impl CoordinatorExecutionTerminal {
    #[cfg(test)]
    pub(super) fn new(lease: QueryExecutionLease, query_id: String, sql: Arc<str>, started_at: Instant) -> Arc<Self> {
        Self::new_with_attempt_report(lease, query_id, sql, started_at, false)
    }

    /// Creates a terminal lease that cannot be released until the attempt
    /// owner has attached its immutable report. The coordinator's report gate
    /// remains the final fail-closed check; this ordering prevents consumers
    /// from observing a completed-but-unsealed Flight record in between.
    pub(super) fn new_with_attempt_report(
        lease: QueryExecutionLease,
        query_id: String,
        sql: Arc<str>,
        started_at: Instant,
        requires_attempt_report: bool,
    ) -> Arc<Self> {
        let key = lease.key.clone();
        Arc::new(Self {
            state: Mutex::new(CoordinatorExecutionTerminalState {
                lease: Some(lease),
                requires_attempt_report,
                attempt_report_attached: !requires_attempt_report,
                pending: None,
            }),
            finished: Notify::new(),
            key,
            query_id,
            sql,
            started_at,
        })
    }

    pub(super) fn is_finished(&self) -> bool {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).lease.is_none()
    }

    pub(super) fn key(&self) -> &QueryKey {
        &self.key
    }

    /// Waits for the coordinator lease to be released after the attempt's
    /// immutable report has been retained. Flight holds this at stream EOF so
    /// a completed DoGet cannot race an immediately following GetQueryStats.
    pub(super) async fn wait_finished(&self) {
        loop {
            let notified = self.finished.notified();
            if self.is_finished() {
                return;
            }
            notified.await;
        }
    }

    /// Releases a terminal outcome deferred until the attempt-owned sealed
    /// snapshot was retained by the coordinator.
    pub(super) fn mark_attempt_report_attached(&self) {
        let completion = {
            let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            state.attempt_report_attached = true;
            // The attempt can seal before the Flight RPC has selected its
            // transport outcome. In that ordering there is no pending
            // outcome yet, so retain the lease for the later `finish_*` call.
            if let Some(outcome) = state.pending.take() {
                state.lease.take().map(|lease| (lease, outcome))
            } else {
                None
            }
        };
        if let Some((lease, outcome)) = completion {
            self.finish_lease(lease, outcome);
            self.finished.notify_waiters();
        }
    }

    /// Completes a terminal whose sealed attempt report could not be retained
    /// by the coordinator. Unlike an execution outcome, this must bypass the
    /// report gate: waiting for a report that was just rejected would retain
    /// the execution lease indefinitely.
    pub(super) fn finish_attempt_report_delivery_failed(
        &self,
        error: QueryCoordinatorError,
        attempt_outcome: Option<RuntimeTerminalOutcome>,
    ) {
        let (status, captured) = adapter_error_to_status_and_capture(
            error.adapter_error("datafusion_nexus.server.flight.attempt_report_handoff"),
        );
        let status = status_with_query_id(status, &self.query_id);
        let completion = {
            let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(lease) = state.lease.take() else { return };
            // Any normal terminal outcome was waiting on the same report gate.
            // Delivery failure is the truthful completion for this coordinator
            // record, so it replaces that deferred outcome atomically.
            state.pending = None;
            (
                lease,
                CoordinatorExecutionTerminalOutcome::Captured {
                    stage: "attempt_report_delivery",
                    status,
                    captured: Box::new(captured),
                },
            )
        };
        self.finish_attempt_report_delivery_lease(completion.0, completion.1, attempt_outcome);
        self.finished.notify_waiters();
    }

    pub(super) fn finish_success(&self, stage: &'static str) {
        self.finish(CoordinatorExecutionTerminalOutcome::Succeeded { stage });
    }

    pub(super) fn finish_captured(&self, stage: &'static str, status: &Status, captured: CapturedError) {
        self.finish(CoordinatorExecutionTerminalOutcome::Captured {
            stage,
            status: status.clone(),
            captured: Box::new(captured),
        });
    }

    /// Records that the client dropped its response stream. This remains
    /// distinct from a cancellation status delivered to the client: the
    /// immutable attempt report can be cancelled while Flight delivery was
    /// simply abandoned by the peer.
    pub(super) fn finish_dropped(&self, stage: &'static str) {
        self.finish(CoordinatorExecutionTerminalOutcome::Dropped { stage });
    }

    pub(super) fn finish_panicked(&self, stage: &'static str) {
        self.finish(CoordinatorExecutionTerminalOutcome::Panicked { stage });
    }

    fn finish(&self, outcome: CoordinatorExecutionTerminalOutcome) {
        let completion = {
            let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.lease.is_none() || state.pending.is_some() {
                return;
            }
            if state.requires_attempt_report && !state.attempt_report_attached {
                state.pending = Some(outcome);
                return;
            }
            let lease = state.lease.take().expect("unfinished Flight terminal retains its coordinator lease");
            (lease, outcome)
        };
        self.finish_lease(completion.0, completion.1);
        self.finished.notify_waiters();
    }

    fn finish_lease(&self, mut lease: QueryExecutionLease, outcome: CoordinatorExecutionTerminalOutcome) {
        match outcome {
            CoordinatorExecutionTerminalOutcome::Succeeded { stage } => {
                lease.succeed();
                tracing::info!(
                    query_id = self.query_id,
                    outcome = "succeeded",
                    stage,
                    elapsed_ms = self.started_at.elapsed().as_millis(),
                    "finished Flight SQL statement"
                );
            },
            CoordinatorExecutionTerminalOutcome::Captured { stage, status, captured } => {
                lease.finish_captured(stage, &status, *captured);
                log_statement_status(&self.query_id, Some(&self.sql), stage, self.started_at, &status);
            },
            CoordinatorExecutionTerminalOutcome::Dropped { stage } => {
                drop(lease);
                tracing::info!(
                    query_id = self.query_id,
                    outcome = "dropped",
                    stage,
                    elapsed_ms = self.started_at.elapsed().as_millis(),
                    "Flight SQL statement response stream was dropped"
                );
            },
            CoordinatorExecutionTerminalOutcome::Panicked { stage } => {
                lease.finish_panicked();
                tracing::error!(
                    query_id = self.query_id,
                    outcome = "panicked",
                    stage,
                    elapsed_ms = self.started_at.elapsed().as_millis(),
                    "Flight SQL statement panicked"
                );
            },
        }
    }

    fn finish_attempt_report_delivery_lease(
        &self,
        mut lease: QueryExecutionLease,
        outcome: CoordinatorExecutionTerminalOutcome,
        attempt_outcome: Option<RuntimeTerminalOutcome>,
    ) {
        match outcome {
            CoordinatorExecutionTerminalOutcome::Captured { stage, status, captured } => {
                lease.finish_attempt_report_delivery_failed(stage, &status, *captured, attempt_outcome);
                log_statement_status(&self.query_id, Some(&self.sql), stage, self.started_at, &status);
            },
            // This private path is constructed only by
            // `finish_attempt_report_delivery_failed` above.
            CoordinatorExecutionTerminalOutcome::Succeeded { .. }
            | CoordinatorExecutionTerminalOutcome::Dropped { .. }
            | CoordinatorExecutionTerminalOutcome::Panicked { .. } => {
                unreachable!("attempt report delivery failure must retain a structured captured error")
            },
        }
    }
}

impl Drop for QueryExecutionGuard {
    fn drop(&mut self) {
        if !self.terminal.is_finished() {
            if std::thread::panicking() {
                self.terminal.finish_panicked("stream");
            } else if self.delivery_started.load(std::sync::atomic::Ordering::Acquire) {
                let (status, captured) = adapter_error_to_status_and_capture(cancellation_error(&self.shutdown_token));
                let status =
                    status_with_query_id_and_correlation_id(status, &self.query_id, self.correlation_id.as_deref());
                self.terminal.finish_captured("stream", &status, captured);
            } else {
                self.terminal.finish_dropped("stream");
            }
            self.native_cancellation.cancel();
            if let Some(attempt_cancellation) = &self.attempt_cancellation {
                attempt_cancellation.cancel();
            }
            self.flight_token.cancel();
        }
        self.cancellations.remove(&self.query_id);
    }
}
