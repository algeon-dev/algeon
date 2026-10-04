//! Admission close and the admission-quiescence drain phase.

use std::{
    mem,
    sync::{PoisonError, TryLockError, atomic::Ordering},
    time::Instant,
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{AdmissionController, query_service_shutdown_timeout};
use crate::runtime::{
    deadline::{deadline_expired, deadline_remaining, mutex_lock_recovering_until},
    query_service::{
        QueryServiceDrainPhase, QueryServiceDrainPhaseOutcome,
        admission::{AdmissionRequest, finish_closed_queued_requests},
    },
};

/// A closed admission gate whose removed queued requests still need their
/// rejection. The service closes its own resources before [`Self::finish`]
/// wakes the waiters.
#[must_use]
pub(in crate::runtime::query_service) struct ClosedAdmission<'a> {
    controller: &'a AdmissionController,
    result: Result<()>,
    closed_requests: Vec<AdmissionRequest>,
}

impl ClosedAdmission<'_> {
    /// Wakes every admission waiter, rejects each request the close removed
    /// from the queue, and returns the close result.
    pub(in crate::runtime::query_service) fn finish(self) -> Result<()> {
        self.controller.notify_admission_available();
        finish_closed_queued_requests(self.closed_requests);
        self.result
    }
}

impl AdmissionController {
    /// Closes the atomic admission gate. Commit paths recheck it while holding
    /// the admission mutex.
    pub(in crate::runtime::query_service) fn close_gate(&self) {
        self.admission_open.store(false, Ordering::Release);
    }

    /// Closes admission and takes every queued request.
    ///
    /// The atomic gate closes before the best-effort mutex update, so this
    /// signal never waits behind an admission transaction. A contended mutex
    /// leaves the queue to the waiters, which observe the closed gate.
    pub(in crate::runtime::query_service) fn close(&self) -> ClosedAdmission<'_> {
        self.close_gate();
        let (result, closed_requests) = match self.state.try_lock() {
            Ok(mut state) => {
                state.closed = true;
                let closed_requests = mem::take(&mut state.queued).into_iter().collect::<Vec<_>>();

                (Ok(()), closed_requests)
            },
            Err(TryLockError::Poisoned(error)) => {
                let mut state = error.into_inner();
                state.closed = true;
                let closed_requests = mem::take(&mut state.queued).into_iter().collect::<Vec<_>>();

                (
                    Err(QueryError::new(
                        QueryErrorCode::Internal,
                        "query_engine.query_service.admission",
                        "query service admission mutex was poisoned",
                    )
                    .with_fact("reason", "admission_mutex_poisoned")),
                    closed_requests,
                )
            },
            Err(TryLockError::WouldBlock) => (Ok(()), Vec::new()),
        };
        ClosedAdmission { controller: self, result, closed_requests }
    }

    /// Marks admission closed, rejects every queued request, and waits until
    /// no request is queued or in preflight and every lifecycle ticket has
    /// left, under one absolute deadline.
    pub(in crate::runtime::query_service) fn wait_for_quiescence_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> QueryServiceDrainPhaseOutcome {
        // A panic under the admission mutex leaves the queue, preflight set,
        // and lifecycle tickets valid: a placement that unwinds before its
        // ticket exists drops its grant. Recovering the guard and waiting for
        // the tickets therefore still proves admission quiescence.
        let Some(mut state) = mutex_lock_recovering_until(&self.state, deadline) else {
            return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                phase: QueryServiceDrainPhase::AdmissionQuiescence,
                error: query_service_shutdown_timeout(operation, None, "admission_mutex"),
            };
        };
        state.closed = true;
        let closed_requests = mem::take(&mut state.queued).into_iter().collect::<Vec<_>>();
        if !closed_requests.is_empty() {
            // Release admission lock before completing queued requests and re-locking.
            drop(state);
            finish_closed_queued_requests(closed_requests);
            state = match self.lock_admission_until(deadline, operation) {
                Ok(state) => state,
                Err(error) => {
                    return QueryServiceDrainPhaseOutcome::from_result(
                        QueryServiceDrainPhase::AdmissionQuiescence,
                        Err(error),
                    );
                },
            };
        }
        let mut waited_for_quiescence = false;
        loop {
            if state.lifecycle.is_quiescent() && state.queued.is_empty() && state.pending_preflight.is_empty() {
                // Quiescence first observed at or after the absolute deadline is not accepted, matching the
                // final-completion phase.
                if waited_for_quiescence && deadline_expired(deadline) {
                    return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                        phase: QueryServiceDrainPhase::AdmissionQuiescence,
                        error: query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"),
                    };
                }
                return QueryServiceDrainPhaseOutcome::Quiescent { phase: QueryServiceDrainPhase::AdmissionQuiescence };
            }
            if deadline_expired(deadline) {
                return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                    phase: QueryServiceDrainPhase::AdmissionQuiescence,
                    error: query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"),
                };
            }
            let Some(remaining) = deadline_remaining(deadline) else {
                return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                    phase: QueryServiceDrainPhase::AdmissionQuiescence,
                    error: query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"),
                };
            };
            waited_for_quiescence = true;
            #[cfg(test)]
            self.shutdown_waiters.fetch_add(1, Ordering::AcqRel);
            let (next, _timeout) =
                self.admission_available.wait_timeout(state, remaining).unwrap_or_else(PoisonError::into_inner);
            #[cfg(test)]
            self.shutdown_waiters.fetch_sub(1, Ordering::AcqRel);
            // A timed-out wait is not terminal by itself: the loop head re-checks quiescence on the reacquired
            // state before it reports the deadline.
            state = next;
        }
    }
}
