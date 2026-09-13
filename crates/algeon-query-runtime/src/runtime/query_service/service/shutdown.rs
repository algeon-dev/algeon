//! Service close and quiescence-drain paths.

#[cfg(any(test, feature = "test-hooks"))]
use std::time::Duration;
use std::{
    mem,
    sync::{TryLockError, atomic::Ordering},
    time::Instant,
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{
    super::{
        QueryServiceDrainOutcome, QueryServiceDrainPhase, QueryServiceDrainPhaseOutcome,
        admission::finish_closed_queued_requests, resources::QueryServiceResources,
    },
    QueryService,
};
use crate::runtime::{
    deadline::{DeadlineMutexLockError, deadline_expired, deadline_remaining, mutex_lock_until},
    query_service::service::query_service_shutdown_timeout,
};

impl QueryService {
    /// Closes admission and wakes every queued waiter.
    ///
    /// The atomic gate closes before the best-effort mutex update, so this
    /// signal never waits behind an admission transaction. Commit paths
    /// recheck the gate while holding the admission mutex. Active query
    /// handles keep their slots until dropped or finished.
    pub fn close(&self) -> Result<()> {
        self.inner.admission_open.store(false, Ordering::Release);
        let (close_result, closed_requests) = match self.inner.admission.try_lock() {
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
        self.inner.service_epoch.invalidate();
        let stream_pool_close_result =
            self.inner.resources.as_ref().map_or(Ok(()), QueryServiceResources::close_stream_pools);
        self.inner.notify_admission_available();
        finish_closed_queued_requests(closed_requests);
        close_result.and(stream_pool_close_result)
    }

    /// Closes admission and waits for every admitted handle and queued waiter
    /// to leave the service.
    ///
    /// This is the synchronous teardown boundary used when an async server
    /// future is cancelled: returning guarantees no query cleanup still owns
    /// service runtime resources, so the owning resource aggregate can be
    /// destroyed. The process-wide backend owner remains attached to that
    /// aggregate until its resources are dropped.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn close_and_wait_for_quiescence(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        self.close_and_wait_for_quiescence_until(deadline)
    }

    /// Closes admission and waits for complete cleanup under one caller-owned
    /// absolute deadline.
    ///
    /// Time spent acquiring the admission lock or closing the service consumes
    /// this same deadline instead of starting a second grace period.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn close_and_wait_for_quiescence_until(&self, deadline: Instant) -> Result<()> {
        self.close_drain_step_until(deadline).into_result()
    }

    #[doc(hidden)]
    pub fn close_drain_step_until(&self, deadline: Instant) -> QueryServiceDrainOutcome {
        let operation = "query_engine.query_service.close_and_wait_for_quiescence";
        self.inner.admission_open.store(false, Ordering::Release);
        self.inner.service_epoch.invalidate();
        let mut phases = Vec::with_capacity(8);
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::StreamPoolClose,
            self.inner.resources.as_ref().map_or(Ok(()), QueryServiceResources::close_stream_pools),
        ));
        self.inner.notify_admission_available();
        let admission_quiescence = self.wait_for_admission_quiescence_until(deadline, operation);
        let admission_is_quiescent = admission_quiescence.is_quiescent();
        phases.push(admission_quiescence);
        if !admission_is_quiescent {
            // Query-handle release schedules ordered CUDA destruction on the
            // device execution pool. A bounded wait may expire while an
            // outward context still owns the final query anchor, so closing
            // that pool here would make the later release impossible to run.
            // Report every dependent phase as pending and let the next drain
            // pass enter them after admission cleanup has finished.
            phases.push(dependent_drain_phase_pending(
                QueryServiceDrainPhase::ExecutionPoolQuiescence,
                QueryServiceDrainPhase::AdmissionQuiescence,
                operation,
            ));
            phases.push(dependent_drain_phase_pending(
                QueryServiceDrainPhase::StreamPoolQuiescence,
                QueryServiceDrainPhase::AdmissionQuiescence,
                operation,
            ));
            phases.push(dependent_drain_phase_pending(
                QueryServiceDrainPhase::ResourceEpoch,
                QueryServiceDrainPhase::AdmissionQuiescence,
                operation,
            ));
            phases.push(dependent_drain_phase_pending(
                QueryServiceDrainPhase::FinalCompletion,
                QueryServiceDrainPhase::AdmissionQuiescence,
                operation,
            ));
            return QueryServiceDrainOutcome::new(phases);
        }
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::ExecutionPoolQuiescence,
            self.inner.resources.as_ref().map_or(Ok(()), |resources| {
                resources.close_and_drain_gpu_execution_pools_until(deadline)?;
                resources.close_and_drain_resident_caches_until(deadline)
            }),
        ));
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::StreamPoolQuiescence,
            self.inner
                .resources
                .as_ref()
                .map_or(Ok(()), |resources| resources.close_and_drain_stream_pools_until(deadline)),
        ));
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::ResourceEpoch,
            self.inner.service_epoch.wait_for_idle_until(deadline, operation),
        ));
        let blocked_by_phase = phases.iter().find(|outcome| !outcome.is_quiescent()).map(|outcome| outcome.phase());
        phases.push(match blocked_by_phase {
            Some(blocked_by_phase) => {
                dependent_drain_phase_pending(QueryServiceDrainPhase::FinalCompletion, blocked_by_phase, operation)
            },
            None if deadline_expired(deadline) => QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                phase: QueryServiceDrainPhase::FinalCompletion,
                error: query_service_shutdown_timeout(operation, None, "final_completion"),
            },
            None => QueryServiceDrainPhaseOutcome::Quiescent { phase: QueryServiceDrainPhase::FinalCompletion },
        });
        QueryServiceDrainOutcome::new(phases)
    }

    fn wait_for_admission_quiescence_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> QueryServiceDrainPhaseOutcome {
        let (mut state, admission_was_poisoned) = match mutex_lock_until(&self.inner.admission, deadline) {
            Ok(state) => (state, false),
            Err(DeadlineMutexLockError::Poisoned) => {
                let state = match self.inner.admission.try_lock() {
                    Err(TryLockError::Poisoned(error)) => error.into_inner(),
                    Ok(_) | Err(TryLockError::WouldBlock) => {
                        return QueryServiceDrainPhaseOutcome::PermanentFailure {
                            phase: QueryServiceDrainPhase::AdmissionQuiescence,
                            error: QueryError::new(
                                QueryErrorCode::Internal,
                                operation,
                                "query service admission mutex poison state changed during shutdown recovery",
                            )
                            .with_fact("reason", "admission_mutex_poison_recovery_raced"),
                        };
                    },
                };
                (state, true)
            },
            Err(DeadlineMutexLockError::TimedOut) => {
                return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                    phase: QueryServiceDrainPhase::AdmissionQuiescence,
                    error: query_service_shutdown_timeout(operation, None, "admission_mutex"),
                };
            },
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
        if admission_was_poisoned {
            return QueryServiceDrainPhaseOutcome::PermanentFailure {
                phase: QueryServiceDrainPhase::AdmissionQuiescence,
                error: QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service admission mutex was poisoned during shutdown",
                )
                .with_fact("reason", "admission_mutex_poisoned"),
            };
        }
        let mut waited_for_quiescence = false;
        loop {
            if state.lifecycle.is_quiescent() && state.queued.is_empty() && state.pending_preflight.is_empty() {
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
            self.inner.shutdown_waiters.fetch_add(1, Ordering::AcqRel);
            let waited = self.inner.admission_available.wait_timeout(state, remaining);
            #[cfg(test)]
            self.inner.shutdown_waiters.fetch_sub(1, Ordering::AcqRel);
            let (next, timeout) = match waited {
                Ok(waited) => waited,
                Err(_error) => {
                    return QueryServiceDrainPhaseOutcome::PermanentFailure {
                        phase: QueryServiceDrainPhase::AdmissionQuiescence,
                        error: QueryError::new(
                            QueryErrorCode::Internal,
                            operation,
                            "query service admission condition variable was poisoned during shutdown",
                        ),
                    };
                },
            };
            state = next;
            if timeout.timed_out() {
                return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                    phase: QueryServiceDrainPhase::AdmissionQuiescence,
                    error: query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"),
                };
            }
        }
    }
}

fn dependent_drain_phase_pending(
    phase: QueryServiceDrainPhase,
    blocked_by_phase: QueryServiceDrainPhase,
    operation: &'static str,
) -> QueryServiceDrainPhaseOutcome {
    QueryServiceDrainPhaseOutcome::DeadlineElapsed {
        phase,
        error: QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            operation,
            "query service cleanup phase is waiting for an earlier drain phase",
        )
        .with_fact("reason", "query_service_cleanup_dependency_pending")
        .with_fact("deadline_stage", phase.as_str())
        .with_fact("blocked_by_phase", blocked_by_phase.as_str())
        .with_cleanup_deadline_elapsed(),
    }
}
