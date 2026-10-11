//! Service close and quiescence-drain paths.

#[cfg(any(test, feature = "test-hooks"))]
use std::time::Duration;
use std::time::Instant;

use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{
    super::{QueryServiceDrainOutcome, QueryServiceDrainPhase, QueryServiceDrainPhaseOutcome},
    QueryService,
};
use crate::runtime::{deadline::deadline_expired, query_service::controller::query_service_shutdown_timeout};

impl QueryService {
    /// Closes admission and wakes every queued waiter.
    ///
    /// The atomic gate closes before the best-effort mutex update, so this
    /// signal never waits behind an admission transaction. Commit paths
    /// recheck the gate while holding the admission mutex. Active query
    /// handles keep their slots until dropped or finished.
    pub fn close(&self) -> Result<()> {
        let closed = self.inner.admission.close();
        self.inner.service_epoch.invalidate();
        let stream_pool_close_result = self.inner.resources.close_stream_pools();
        closed.finish().and(stream_pool_close_result)
    }

    /// Closes admission and waits for every admitted handle and queued waiter
    /// to leave the service.
    ///
    /// This is the synchronous teardown boundary used when an async server
    /// future is cancelled. Returning guarantees that no query cleanup still
    /// owns service runtime resources, so the owning resource aggregate can be
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
    /// Time spent acquiring the admission lock or closing the service counts
    /// against this same deadline. No second grace period starts.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn close_and_wait_for_quiescence_until(&self, deadline: Instant) -> Result<()> {
        self.close_drain_step_until(deadline).into_result()
    }

    pub(crate) fn close_drain_step_until(&self, deadline: Instant) -> QueryServiceDrainOutcome {
        let operation = "query_engine.query_service.close_and_wait_for_quiescence";
        self.inner.admission.close_gate();
        self.inner.service_epoch.invalidate();
        let mut phases = Vec::with_capacity(8);
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::StreamPoolClose,
            self.inner.resources.close_stream_pools(),
        ));
        self.inner.admission.notify_admission_available();
        let admission_quiescence = self.inner.admission.wait_for_quiescence_until(deadline, operation);
        let admission_is_quiescent = admission_quiescence.is_quiescent();
        phases.push(admission_quiescence);
        if !admission_is_quiescent {
            // Query-handle release schedules ordered CUDA destruction on the
            // device execution pool. A bounded wait may expire while an
            // outward context still owns the final query anchor. Closing that
            // pool here would make the later release impossible to run.
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
            self.inner
                .resources
                .close_and_drain_gpu_execution_pools_until(deadline)
                .and_then(|()| self.inner.resources.close_and_drain_resident_caches_until(deadline)),
        ));
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::StreamPoolQuiescence,
            self.inner.resources.close_and_drain_stream_pools_until(deadline),
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
