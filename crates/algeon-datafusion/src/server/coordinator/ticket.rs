//! Ticket issue/consume and planning/execution lease transitions.

use std::{
    cell::Cell,
    sync::{Arc, Weak},
    time::Instant,
};

use query_engine::{
    observability::{RuntimeObservationHandle, RuntimeObservationOwner, RuntimeTerminalOutcome},
    runtime::QueryActivityLease,
};

use super::{
    super::observability::CapturedError,
    QueryCoordinator,
    types::{
        CoordinatorTicketPayload, ObservationId, QueryCoordinatorError, QueryCoordinatorErrorKind, QueryHotCounters,
        QueryKey, QueryPhase, QueryTerminal, RetainedUnavailableReason,
    },
};
use crate::server::native_exec::ServerNativeObservationActivityFactory as NativeObservationActivityFactory;

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
                record.execution_cancellation = Some(ticket.native_cancellation.clone());
            }
            (ticket, Arc::clone(&record.hot), expired)
        };
        #[cfg(test)]
        let observation_id =
            state.records.get(&key).expect("current query key must reference a record").observation_id.clone();
        state.ticketed_count = state.ticketed_count.checked_sub(1).expect("consumed ticket count underflow");
        if expired {
            drop(state);
            self.finalize_ready();
            return Err(QueryCoordinatorError::new(QueryCoordinatorErrorKind::TicketExpired, "query ticket expired"));
        }
        let lease = QueryExecutionLease {
            coordinator: Arc::downgrade(self),
            key,
            #[cfg(test)]
            observation_id,
            hot,
            released: false,
        };
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

    pub(crate) fn acquire_activity(
        self: &Arc<Self>,
        key: &QueryKey,
        observation_id: &ObservationId,
    ) -> Result<ObservationActivityLease, QueryCoordinatorError> {
        let mut state = self.lock_state();
        let record = Self::record_mut(&mut state, key)?;
        if &record.observation_id != observation_id {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::StaleGeneration,
                "observation identity does not match the query generation",
            ));
        }
        if record.terminal_outcome.is_some()
            || matches!(record.phase, QueryPhase::Finalizing | QueryPhase::Retained | QueryPhase::Tombstone)
        {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::InvalidTransition,
                "terminal query cannot acquire observation activity",
            ));
        }
        record.active_activities = record.active_activities.checked_add(1).ok_or_else(|| {
            QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::InvalidTransition,
                "query observation activity count overflowed",
            )
        })?;
        Ok(ObservationActivityLease { coordinator: Arc::downgrade(self), key: key.clone(), released: false })
    }

    pub(crate) fn observation_handle(
        &self,
        key: &QueryKey,
        observation_id: &ObservationId,
    ) -> Result<RuntimeObservationHandle, QueryCoordinatorError> {
        let state = self.lock_state();
        let record = state.records.get(key).ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "unknown query generation")
        })?;
        if &record.observation_id != observation_id {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::StaleGeneration,
                "observation identity does not match the query generation",
            ));
        }
        Ok(record
            .observation_owner
            .as_ref()
            .map_or_else(RuntimeObservationHandle::disabled, RuntimeObservationOwner::handle))
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
            record.active_leases == 0
                && record.active_activities == 0
                && !matches!(record.phase, QueryPhase::Retained | QueryPhase::Tombstone)
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
            record.active_leases == 0
                && record.active_activities == 0
                && !matches!(record.phase, QueryPhase::Retained | QueryPhase::Tombstone)
        };
        self.notify.notify_waiters();
        if should_finalize {
            self.finalize_ready();
        }
    }

    pub(crate) fn release_activity(&self, key: &QueryKey) {
        let should_finalize = {
            let mut state = self.lock_state();
            let Ok(record) = Self::record_mut(&mut state, key) else { return };
            if matches!(record.phase, QueryPhase::Retained | QueryPhase::Tombstone) {
                return;
            }
            if record.active_activities > 0 {
                record.active_activities -= 1;
            }
            record.terminal_outcome.is_some() && record.active_leases == 0 && record.active_activities == 0
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
    pub(crate) observation_id: ObservationId,
    pub(crate) hot: Arc<QueryHotCounters>,
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

    /// Creates the generation-bound report-delivery hook that remains valid
    /// across this lease's transition to a ticket or its terminal release.
    pub(crate) fn attempt_report_delivery_terminal(&self) -> QueryPlanningReportDeliveryTerminal {
        self.attempt_report_handoff_registered.set(true);
        QueryPlanningReportDeliveryTerminal { coordinator: self.coordinator.clone(), key: self.key.clone() }
    }

    #[cfg(test)]
    pub(crate) fn observation_id(&self) -> &ObservationId {
        &self.observation_id
    }

    #[cfg(test)]
    pub(crate) fn hot_counters(&self) -> Arc<QueryHotCounters> {
        Arc::clone(&self.hot)
    }

    pub(crate) fn observation_handle(&self) -> Result<RuntimeObservationHandle, QueryCoordinatorError> {
        self.coordinator
            .upgrade()
            .ok_or_else(|| QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "coordinator dropped"))?
            .observation_handle(&self.key, &self.observation_id)
    }

    pub(crate) fn activity_factory(&self) -> NativeObservationActivityFactory {
        observation_activity_factory(self.coordinator.clone(), self.key.clone(), self.observation_id.clone())
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
            coordinator: self.coordinator.clone(),
            key: self.key.clone(),
            #[cfg(test)]
            observation_id: self.observation_id.clone(),
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
    #[cfg(test)]
    pub(crate) observation_id: ObservationId,
    pub(crate) hot: Arc<QueryHotCounters>,
    pub(crate) released: bool,
}

impl QueryExecutionLease {
    #[cfg(test)]
    pub(crate) fn observation_id(&self) -> &ObservationId {
        &self.observation_id
    }

    pub(crate) fn hot_counters(&self) -> Arc<QueryHotCounters> {
        Arc::clone(&self.hot)
    }

    #[cfg(test)]
    pub(crate) fn observation_handle(&self) -> Result<RuntimeObservationHandle, QueryCoordinatorError> {
        self.coordinator
            .upgrade()
            .ok_or_else(|| QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "coordinator dropped"))?
            .observation_handle(&self.key, &self.observation_id)
    }

    #[cfg(test)]
    pub(crate) fn activity_factory(&self) -> NativeObservationActivityFactory {
        observation_activity_factory(self.coordinator.clone(), self.key.clone(), self.observation_id.clone())
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

pub(crate) fn observation_activity_factory(
    coordinator: Weak<QueryCoordinator>,
    key: QueryKey,
    observation_id: ObservationId,
) -> NativeObservationActivityFactory {
    Arc::new(move || {
        let coordinator = coordinator.upgrade()?;
        coordinator.acquire_activity(&key, &observation_id).ok().map(QueryActivityLease::new)
    })
}

#[derive(Debug)]
pub(crate) struct ObservationActivityLease {
    pub(crate) coordinator: Weak<QueryCoordinator>,
    pub(crate) key: QueryKey,
    pub(crate) released: bool,
}

impl Drop for ObservationActivityLease {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(coordinator) = self.coordinator.upgrade() {
            coordinator.release_activity(&self.key);
        }
    }
}
