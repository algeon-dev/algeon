//! Attempt-scoped cooperative cancellation and deadline helpers.
use std::time::{Duration, Instant};

use query_planning::error::Result;

use crate::runtime::ExecutionCancellation;

#[derive(Debug, Clone)]
pub struct AttemptCancellation {
    execution: ExecutionCancellation,
    deadline: Option<Instant>,
}

impl AttemptCancellation {
    #[doc(hidden)]
    pub fn new(deadline: Option<Instant>) -> Self {
        Self { execution: ExecutionCancellation::with_deadline(deadline), deadline }
    }

    /// Requests cooperative cancellation; cleanup still completes.
    pub fn cancel(&self) {
        self.execution.cancel()
    }
    pub fn is_cancelled(&self) -> bool {
        self.execution.is_cancelled()
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Returns whether the optional absolute deadline has expired.
    pub fn is_expired(&self) -> bool {
        self.execution.is_expired()
    }

    pub fn execution_cancellation(&self) -> ExecutionCancellation {
        self.execution.clone()
    }

    /// Cooperative cancellation check at adapter-owned boundaries (planning,
    /// pre-execution, and result drain). Caller cancellation wins first; an
    /// expired deadline reports the distinct `deadline_expired` reason.
    pub fn check(&self, operation: &'static str) -> Result<()> {
        self.execution.check(operation)
    }
}

/// Returns the earlier caller deadline or configured queue expiry.
pub fn effective_queue_expiry(
    submitted: Instant,
    caller_deadline: Option<Instant>,
    configured_max_wait: Duration,
) -> Instant {
    let configured = submitted.checked_add(configured_max_wait).unwrap_or(submitted);
    caller_deadline.map_or(configured, |deadline| deadline.min(configured))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptCancellationBoundary {
    Planning,
    AdmissionQueue,
    AdmissionTransaction,
    StreamAcquisition,
    WorkerJob,
    GpuExecutionSafeBoundary,
    ResultDrain,
    Cleanup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCancellationState {
    NotRequested,
    CancellationRequested,
    DeadlineExpired { effective_deadline: Instant },
}

/// Pure propagation-state model for one later boundary.
///
/// - `AdmissionQueue` is the only boundary whose expiry folds in the configured
///   maximum wait: `min(caller_deadline, submitted + configured_max_wait)`.
/// - `Cleanup` never observes cancellation or expiry: cleanup always runs to
///   completion so capacity and health stay truthful, even after the caller
///   deadline has expired. Algeon never claims CUDA kernel preemption; GPU
///   execution boundaries stop cooperatively at safe points.
pub fn cancellation_state_at(
    boundary: AttemptCancellationBoundary,
    cancelled: bool,
    caller_deadline: Option<Instant>,
    submitted: Instant,
    configured_max_wait: Duration,
    now: Instant,
) -> BoundaryCancellationState {
    if matches!(boundary, AttemptCancellationBoundary::Cleanup) {
        return BoundaryCancellationState::NotRequested;
    }
    if cancelled {
        return BoundaryCancellationState::CancellationRequested;
    }
    let deadline = match boundary {
        AttemptCancellationBoundary::AdmissionQueue => {
            Some(effective_queue_expiry(submitted, caller_deadline, configured_max_wait))
        },
        _ => caller_deadline,
    };
    match deadline {
        Some(deadline) if now >= deadline => {
            BoundaryCancellationState::DeadlineExpired { effective_deadline: deadline }
        },
        _ => BoundaryCancellationState::NotRequested,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_queue_expiry_is_the_earlier_of_caller_deadline_and_configured_wait() {
        let submitted = Instant::now();
        let configured = Duration::from_secs(10);
        // No caller deadline: the configured wait bounds the queue.
        assert_eq!(effective_queue_expiry(submitted, None, configured), submitted + configured);
        // Earlier caller deadline wins.
        let earlier = submitted + Duration::from_secs(1);
        assert_eq!(effective_queue_expiry(submitted, Some(earlier), configured), earlier);
        // Later caller deadline loses to the configured wait.
        let later = submitted + Duration::from_secs(60);
        assert_eq!(effective_queue_expiry(submitted, Some(later), configured), submitted + configured);
    }

    #[test]
    fn cancellation_state_models_every_boundary() {
        let submitted = Instant::now();
        let max_wait = Duration::from_secs(10);
        let caller_deadline = submitted + Duration::from_secs(1);
        let before_deadline = submitted;
        let after_deadline = submitted + Duration::from_secs(2);
        let after_max_wait = submitted + Duration::from_secs(11);

        for boundary in [
            AttemptCancellationBoundary::Planning,
            AttemptCancellationBoundary::AdmissionQueue,
            AttemptCancellationBoundary::AdmissionTransaction,
            AttemptCancellationBoundary::StreamAcquisition,
            AttemptCancellationBoundary::WorkerJob,
            AttemptCancellationBoundary::GpuExecutionSafeBoundary,
            AttemptCancellationBoundary::ResultDrain,
            AttemptCancellationBoundary::Cleanup,
        ] {
            // An explicit caller cancel wins at every non-cleanup boundary.
            let state = cancellation_state_at(boundary, true, None, submitted, max_wait, before_deadline);
            if matches!(boundary, AttemptCancellationBoundary::Cleanup) {
                assert_eq!(state, BoundaryCancellationState::NotRequested, "cleanup never cancels: {boundary:?}");
            } else {
                assert_eq!(state, BoundaryCancellationState::CancellationRequested, "{boundary:?}");
            }
        }

        // Queue expiry folds in the configured maximum wait even without a
        // caller deadline; other boundaries only observe the caller deadline.
        assert!(matches!(
            cancellation_state_at(
                AttemptCancellationBoundary::AdmissionQueue,
                false,
                None,
                submitted,
                max_wait,
                after_max_wait
            ),
            BoundaryCancellationState::DeadlineExpired { .. }
        ));
        assert_eq!(
            cancellation_state_at(
                AttemptCancellationBoundary::WorkerJob,
                false,
                None,
                submitted,
                max_wait,
                after_max_wait
            ),
            BoundaryCancellationState::NotRequested
        );

        // The caller deadline applies at non-queue boundaries.
        assert!(matches!(
            cancellation_state_at(
                AttemptCancellationBoundary::ResultDrain,
                false,
                Some(caller_deadline),
                submitted,
                max_wait,
                after_deadline
            ),
            BoundaryCancellationState::DeadlineExpired { effective_deadline } if effective_deadline == caller_deadline
        ));
        // Within the deadline nothing is requested.
        assert_eq!(
            cancellation_state_at(
                AttemptCancellationBoundary::AdmissionQueue,
                false,
                Some(caller_deadline),
                submitted,
                max_wait,
                before_deadline
            ),
            BoundaryCancellationState::NotRequested
        );
        assert_eq!(
            cancellation_state_at(
                AttemptCancellationBoundary::Cleanup,
                false,
                Some(caller_deadline),
                submitted,
                max_wait,
                after_deadline
            ),
            BoundaryCancellationState::NotRequested
        );
    }
}
