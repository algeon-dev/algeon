//! Ticket issue/consume and planning/execution lease transitions.

use std::{
    cell::Cell,
    sync::{Arc, Weak},
    time::Instant,
};

use query_runtime::observability::{RuntimeObservationConfig, RuntimeTerminalOutcome};

use super::{
    super::observability::CapturedError,
    QueryCoordinator,
    types::{
        CoordinatorTicketPayload, QueryCoordinatorError, QueryCoordinatorErrorKind, QueryHotCounters, QueryKey,
        QueryPhase, QueryTerminal, RetainedUnavailableReason,
    },
};
use crate::server::attempt::AlgeonAttempt;

impl QueryCoordinator {
    pub(crate) fn consume_ticket(
        self: &Arc<Self>,
        query_id: &str,
        now: Instant,
    ) -> Result<(QueryExecutionLease, CoordinatorTicketPayload), QueryCoordinatorError> {
        let mut state = self.lock_state();
        let key = state.current.get(query_id).cloned().ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "unknown query ticket")
        })?;
        let (ticket, hot, expired) = {
            let record = state.records.get_mut(&key).expect("current query key must reference a record");
            if record.phase != QueryPhase::Ticketed {
                return Err(QueryCoordinatorError::new(
                    QueryCoordinatorErrorKind::InvalidTransition,
                    "query is not ticketed",
                ));
            }
            let ticket = record.ticket.take().expect("ticketed query must own a ticket payload");
            let expired = now >= ticket.deadline;
            if expired {
                if record.terminal_outcome.is_none() {
                    let mut terminal = QueryTerminal::Dropped;
                    Self::apply_terminal_to_capture(record, &mut terminal, now);
                    record.terminal_outcome = Some(terminal.compact_outcome());
                    record.terminal = Some(terminal);
                }
                record.unavailable = Some(RetainedUnavailableReason::TicketExpired);
                record.phase = QueryPhase::Finalizing;
            } else {
                record.phase = QueryPhase::Executing;
                record.active_leases = 1;
                record.execution_cancellation = ticket.attempt.as_ref().map(AlgeonAttempt::cancellation);
            }
            (ticket, Arc::clone(&record.hot), expired)
        };
        state.ticketed_count = state.ticketed_count.checked_sub(1).expect("consumed ticket count underflow");
        if expired {
            drop(state);
            self.finalize_ready();
            return Err(QueryCoordinatorError::new(QueryCoordinatorErrorKind::TicketExpired, "query ticket expired"));
        }
        let lease = QueryExecutionLease { coordinator: Arc::downgrade(self), key, hot, released: false };
        Ok((lease, ticket))
    }

    pub(crate) fn issue_ticket(
        &self,
        key: &QueryKey,
        payload: CoordinatorTicketPayload,
    ) -> Result<(), QueryCoordinatorError> {
        let mut state = self.lock_state();
        if state.ticketed_count >= self.config.max_ticketed_queries {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::TicketCapacity,
                "maximum ticketed query count reached",
            ));
        }
        let record = Self::record_mut(&mut state, key)?;
        if record.phase != QueryPhase::Planning || record.active_leases != 1 || record.terminal_outcome.is_some() {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::InvalidTransition,
                "planning query cannot transition to ticketed",
            ));
        }
        record.phase = QueryPhase::Ticketed;
        record.ticket = Some(payload);
        record.active_leases = 0;
        state.ticketed_count = state.ticketed_count.checked_add(1).expect("ticketed count overflow");
        self.notify.notify_waiters();
        Ok(())
    }

    pub(crate) fn promote_execution(&self, key: &QueryKey) -> Result<(), QueryCoordinatorError> {
        let mut state = self.lock_state();
        let record = Self::record_mut(&mut state, key)?;
        if matches!(record.terminal, Some(QueryTerminal::Cancelled { stage: "server_shutdown", .. })) {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::Shutdown,
                "server shutdown cancelled query before execution began",
            ));
        }
        if record.phase != QueryPhase::Planning || record.active_leases != 1 || record.terminal_outcome.is_some() {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::InvalidTransition,
                "planning query cannot transition to execution",
            ));
        }
        record.phase = QueryPhase::Executing;
        Ok(())
    }

    pub(crate) fn set_terminal(&self, key: &QueryKey, mut terminal: QueryTerminal) {
        let should_finalize = {
            let mut state = self.lock_state();
            let Ok(record) = Self::record_mut(&mut state, key) else { return };
            if record.terminal_outcome.is_none() {
                Self::apply_terminal_to_capture(record, &mut terminal, Instant::now());
                record.terminal_outcome = Some(terminal.compact_outcome());
                record.terminal = Some(terminal);
                Self::bump_capture_version(record);
            }
            if record.active_leases > 0 {
                record.active_leases -= 1;
            }
            record.active_leases == 0 && !matches!(record.phase, QueryPhase::Retained | QueryPhase::Tombstone)
        };
        self.notify.notify_waiters();
        if should_finalize {
            self.finalize_ready();
        }
    }

    /// Finishes a planning-only lease for which no native attempt was ever
    /// created. Attempt-owned registration starts with a report gate so a
    /// real attempt cannot race terminal publication; this path is the one
    /// explicit exception before an attempt handoff has been registered.
    ///
    /// This is deliberately distinct from `abandon_attempt_report_delivery`:
    /// there is no sealed report and therefore no delivery failure or
    /// canonical attempt outcome to preserve.
    pub(crate) fn finish_planning_without_attempt_report(&self, key: &QueryKey, mut terminal: QueryTerminal) {
        let should_finalize = {
            let mut state = self.lock_state();
            let Ok(record) = Self::record_mut(&mut state, key) else { return };
            // A `QueryPlanningLease` becomes released on a successful move
            // into a ticket or execution lease. Its unreleased terminal path
            // therefore still owns only the pre-execution planning slot.
            if record.phase == QueryPhase::Planning {
                record.requires_attempt_report = false;
            }
            if record.terminal_outcome.is_none() {
                Self::apply_terminal_to_capture(record, &mut terminal, Instant::now());
                record.terminal_outcome = Some(terminal.compact_outcome());
                record.terminal = Some(terminal);
                Self::bump_capture_version(record);
            }
            if record.active_leases > 0 {
                record.active_leases -= 1;
            }
            record.active_leases == 0 && !matches!(record.phase, QueryPhase::Retained | QueryPhase::Tombstone)
        };
        self.notify.notify_waiters();
        if should_finalize {
            self.finalize_ready();
        }
    }
}

#[derive(Debug)]
pub(crate) struct QueryPlanningLease {
    pub(crate) coordinator: Weak<QueryCoordinator>,
    pub(crate) key: QueryKey,
    pub(crate) hot: Arc<QueryHotCounters>,
    /// The runtime observation this query's attempt records: the configured
    /// capture, or none when the active capture budget could not hold it.
    pub(crate) observation_config: RuntimeObservationConfig,
    /// Set exactly when a native attempt has registered its immutable report
    /// handoff. Until then a planning failure is not allowed to leave the
    /// coordinator waiting for an attempt that was never created.
    pub(crate) attempt_report_handoff_registered: Cell<bool>,
    pub(crate) released: bool,
}

/// A generation-bound completion hook for an attempt report that was sealed
/// while a query was still in Flight planning.
///
/// The planning lease itself may move into a ticket before the attempt seals,
/// so an observer cannot borrow or own that lease. This hook preserves the
/// exact coordinator generation instead. On delivery failure it removes the
/// report gate for that generation; the planning or execution lease remains
/// the sole owner of the query terminal outcome.
#[derive(Clone, Debug)]
pub(crate) struct QueryPlanningReportDeliveryTerminal {
    coordinator: Weak<QueryCoordinator>,
    key: QueryKey,
}

impl QueryPlanningReportDeliveryTerminal {
    pub(crate) fn key(&self) -> &QueryKey {
        &self.key
    }

    pub(crate) fn finish_attempt_report_delivery_failed(&self, attempt_outcome: Option<RuntimeTerminalOutcome>) {
        if let Some(coordinator) = self.coordinator.upgrade() {
            coordinator.abandon_attempt_report_delivery(&self.key, attempt_outcome);
        }
    }
}

impl QueryPlanningLease {
    pub(crate) fn key(&self) -> &QueryKey {
        &self.key
    }

    pub(crate) fn observation_config(&self) -> RuntimeObservationConfig {
        self.observation_config
    }

    /// Creates the generation-bound report-delivery hook that remains valid
    /// across this lease's transition to a ticket or its terminal release.
    pub(crate) fn attempt_report_delivery_terminal(&self) -> QueryPlanningReportDeliveryTerminal {
        self.attempt_report_handoff_registered.set(true);
        QueryPlanningReportDeliveryTerminal { coordinator: Weak::clone(&self.coordinator), key: self.key.clone() }
    }

    #[cfg(test)]
    pub(crate) fn hot_counters(&self) -> Arc<QueryHotCounters> {
        Arc::clone(&self.hot)
    }

    pub(crate) fn issue_ticket(mut self, payload: CoordinatorTicketPayload) -> Result<QueryKey, QueryCoordinatorError> {
        let coordinator = self.coordinator.upgrade().ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "coordinator dropped")
        })?;
        if let Err(error) = coordinator.issue_ticket(&self.key, payload) {
            let captured = CapturedError::from_adapter_error(
                &error.adapter_error("algeon_datafusion.server.query_coordinator.issue_ticket"),
            );
            self.finish(QueryTerminal::failed("ticket", captured));
            return Err(error);
        }
        self.released = true;
        Ok(self.key.clone())
    }

    pub(crate) fn into_execution(mut self) -> Result<QueryExecutionLease, QueryCoordinatorError> {
        let coordinator = self.coordinator.upgrade().ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "coordinator dropped")
        })?;
        if let Err(error) = coordinator.promote_execution(&self.key) {
            let captured = CapturedError::from_adapter_error(
                &error.adapter_error("algeon_datafusion.server.query_coordinator.promote_execution"),
            );
            self.finish(QueryTerminal::failed("planning", captured));
            return Err(error);
        }
        self.released = true;
        Ok(QueryExecutionLease {
            coordinator: Weak::clone(&self.coordinator),
            key: self.key.clone(),
            hot: Arc::clone(&self.hot),
            released: false,
        })
    }

    pub(crate) fn fail(mut self, stage: &'static str, error: CapturedError) {
        self.finish(QueryTerminal::failed(stage, error));
    }

    pub(crate) fn succeed(mut self) {
        self.finish(QueryTerminal::Succeeded);
    }

    pub(crate) fn finish_captured(&mut self, stage: &'static str, status: &tonic::Status, error: CapturedError) {
        let terminal = if status.code() == tonic::Code::Cancelled {
            QueryTerminal::Cancelled { stage, error: Arc::new(error) }
        } else {
            QueryTerminal::failed(stage, error)
        };
        self.finish(terminal);
    }

    pub(crate) fn finish(&mut self, terminal: QueryTerminal) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(coordinator) = self.coordinator.upgrade() {
            if self.attempt_report_handoff_registered.get() {
                coordinator.set_terminal(&self.key, terminal);
            } else {
                coordinator.finish_planning_without_attempt_report(&self.key, terminal);
            }
        }
    }
}

impl Drop for QueryPlanningLease {
    fn drop(&mut self) {
        let terminal = if std::thread::panicking() { QueryTerminal::Panicked } else { QueryTerminal::Dropped };
        self.finish(terminal);
    }
}

#[derive(Debug)]
pub(crate) struct QueryExecutionLease {
    pub(crate) coordinator: Weak<QueryCoordinator>,
    pub(crate) key: QueryKey,
    pub(crate) hot: Arc<QueryHotCounters>,
    pub(crate) released: bool,
}

impl QueryExecutionLease {
    pub(crate) fn hot_counters(&self) -> Arc<QueryHotCounters> {
        Arc::clone(&self.hot)
    }

    pub(crate) fn succeed(mut self) {
        self.finish_success();
    }

    #[cfg(test)]
    pub(crate) fn fail(mut self, stage: &'static str, error: CapturedError) {
        self.finish_failed(stage, error);
    }

    pub(crate) fn finish_success(&mut self) {
        self.finish(QueryTerminal::Succeeded);
    }

    pub(crate) fn finish_failed(&mut self, stage: &'static str, error: CapturedError) {
        self.finish(QueryTerminal::failed(stage, error));
    }

    pub(crate) fn finish_cancelled(&mut self, stage: &'static str, error: CapturedError) {
        self.finish(QueryTerminal::Cancelled { stage, error: Arc::new(error) });
    }

    pub(crate) fn finish_panicked(&mut self) {
        self.finish(QueryTerminal::Panicked);
    }

    #[cfg(test)]
    pub(crate) fn finish_status(&mut self, stage: &'static str, status: &tonic::Status) {
        let error = CapturedError::from_status(status);
        self.finish_captured(stage, status, error);
    }

    pub(crate) fn finish_captured(&mut self, stage: &'static str, status: &tonic::Status, error: CapturedError) {
        if status.code() == tonic::Code::Cancelled {
            self.finish_cancelled(stage, error);
        } else {
            self.finish_failed(stage, error);
        }
    }

    /// Makes a report-handoff failure explicit before releasing this lease.
    /// The native attempt remains the only possible author of its outcome;
    /// the coordinator records only that it could not retain that projection.
    pub(crate) fn finish_attempt_report_delivery_failed(
        &mut self,
        stage: &'static str,
        status: &tonic::Status,
        error: CapturedError,
        attempt_outcome: Option<RuntimeTerminalOutcome>,
    ) {
        if self.released {
            return;
        }
        if let Some(coordinator) = self.coordinator.upgrade() {
            coordinator.abandon_attempt_report_delivery(&self.key, attempt_outcome);
        }
        self.finish_captured(stage, status, error);
    }

    pub(crate) fn finish(&mut self, terminal: QueryTerminal) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(coordinator) = self.coordinator.upgrade() {
            coordinator.set_terminal(&self.key, terminal);
        }
    }
}

impl Drop for QueryExecutionLease {
    fn drop(&mut self) {
        let terminal = if std::thread::panicking() { QueryTerminal::Panicked } else { QueryTerminal::Dropped };
        self.finish(terminal);
    }
}
