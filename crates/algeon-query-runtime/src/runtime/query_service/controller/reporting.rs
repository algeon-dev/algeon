//! Admission reports: ledger snapshots and queue state for service reports.

#[cfg(test)]
use std::time::Instant;

use query_planning::error::Result;
#[cfg(test)]
use query_planning::error::{QueryError, QueryErrorCode};

use super::AdmissionController;
#[cfg(test)]
use crate::runtime::{
    deadline::{DeadlineMutexLockError, deadline_remaining, mutex_lock_until},
    query_service::admission::AdmissionState,
};
use crate::runtime::{
    gpu::{DeviceLedgerReport, LedgerSnapshot},
    query_service::{QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot},
};

impl AdmissionController {
    pub(in crate::runtime::query_service) fn device_ledger_reports(&self) -> Result<Vec<DeviceLedgerReport>> {
        self.devices.ledgers().map(|ledger| ledger.report()).collect()
    }

    /// Returns process-wide active-query slot counts for each service device.
    pub(in crate::runtime::query_service) fn device_admission_snapshots(
        &self,
    ) -> Vec<QueryServiceDeviceAdmissionSnapshot> {
        self.devices
            .ledgers()
            .map(|ledger| {
                let snapshot = ledger.snapshot();
                QueryServiceDeviceAdmissionSnapshot {
                    device_ordinal: ledger.device_ordinal(),
                    process_active_query_slots: snapshot.as_ref().map_or(0, |snapshot| snapshot.active_attempts),
                    has_protection: snapshot.as_ref().is_ok_and(|snapshot| snapshot.protected_slot),
                }
            })
            .collect()
    }

    /// Returns a point-in-time admission snapshot for service reports.
    pub(in crate::runtime::query_service) fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        let mut state = self.lock_admission()?;
        let expired = self.sweep_expired_queued_requests_locked(&mut state);
        // A ledger grant becomes provisional before the complete physical
        // envelope can publish, so provisional transactions stay hidden.
        // A published handle in terminal cleanup still owns its immutable
        // grant, however; include it until the report commit and mechanical
        // capacity release complete.
        let lifecycle = state.lifecycle.snapshot();
        let active_queries = lifecycle.active.saturating_add(lifecycle.published_cleanup);
        let snapshot = QueryServiceAdmissionSnapshot {
            active_queries,
            queued_queries: state.queued.len(),
            queue_high_watermark: state.queue_high_watermark(),
            protected_devices: self
                .device_ledger_snapshots()
                .map(|snapshots| {
                    snapshots
                        .iter()
                        .filter(|snapshot| snapshot.protected_slot)
                        .map(|snapshot| snapshot.device_ordinal)
                        .collect()
                })
                .unwrap_or_default(),
        };
        // Queue-sweep completion may reenter service callers, so release the admission lock first.
        drop(state);
        expired.finish();
        Ok(snapshot)
    }

    /// One atomic snapshot per device ledger, for placement, reports, and
    /// metrics.
    pub(in crate::runtime::query_service) fn device_ledger_snapshots(&self) -> Result<Vec<LedgerSnapshot>> {
        self.devices.ledgers().map(|ledger| ledger.snapshot()).collect()
    }

    /// Test hook: closes new admission on one device ledger, driving the
    /// queue's protection retarget path without a real device fault.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn close_device_admission_for_test(
        &self,
        device_ordinal: usize,
        reason: &str,
    ) -> Result<()> {
        self.ledger_for_ordinal(device_ordinal)?.close_admission(reason)?;
        self.notify_admission_available();
        Ok(())
    }

    /// Waits for the release or provisional-rollback callback that follows
    /// stream-pool cleanup.
    ///
    /// A returned root stream can make a pool look idle before its cleanup
    /// completion commits the report and drops the immutable grant. The same
    /// ordering applies when an unpublished envelope rolls back. Capacity
    /// remains unavailable until that callback clears the admission counters,
    /// so the public idle fence must include both phases.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn wait_for_lifecycle_quiescence_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<()> {
        let mut state = match mutex_lock_until(&self.state, deadline) {
            Ok(state) => state,
            Err(DeadlineMutexLockError::Poisoned) => {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service admission mutex was poisoned while waiting for stream cleanup",
                )
                .with_fact("reason", "admission_mutex_poisoned"));
            },
            Err(DeadlineMutexLockError::TimedOut) => {
                return Err(stream_pool_release_commit_timeout(operation, None));
            },
        };
        loop {
            if state.lifecycle.is_quiescent() {
                return Ok(());
            }
            let Some(remaining) = deadline_remaining(deadline) else {
                return Err(stream_pool_release_commit_timeout(operation, Some(&state)));
            };
            let (next, timeout) = self.admission_available.wait_timeout(state, remaining).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service admission condition variable was poisoned while waiting for stream cleanup",
                )
                .with_fact("reason", "admission_condition_poisoned")
            })?;
            state = next;
            if timeout.timed_out() && !state.lifecycle.is_quiescent() {
                return Err(stream_pool_release_commit_timeout(operation, Some(&state)));
            }
        }
    }
}

#[cfg(test)]
fn stream_pool_release_commit_timeout(operation: &'static str, state: Option<&AdmissionState>) -> QueryError {
    let mut error = QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out waiting for query-service stream cleanup to commit capacity release",
    )
    .with_fact("reason", "stream_pool_idle_timeout")
    .with_fact("cleanup_stage", "release_commit");
    if let Some(state) = state {
        let lifecycle = state.lifecycle.snapshot();
        error = error
            .with_fact("active_queries", lifecycle.active)
            .with_fact("admitting_queries", lifecycle.provisional)
            .with_fact("cleanup_in_progress", lifecycle.cleanup);
    }
    error
}
