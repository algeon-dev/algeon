//! Cooperative cancellation flag for native query execution.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use query_planning::error::{QueryError, Result};

use super::ExecutionContext;

/// Shared cooperative cancellation flag for native query execution.
///
/// Cancellation is observed only at native-engine boundaries. It does not
/// preempt a libcudf/CUDA call already in flight, but it prevents the next
/// source chunk, pipeline task, operator, or materialization step from starting
/// once the flag is set.
#[derive(Debug, Clone)]
pub struct ExecutionCancellation {
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl ExecutionCancellation {
    /// Creates a non-cancelled execution flag.
    pub fn new() -> Self {
        Self::with_deadline(None)
    }

    /// Creates a cancellation token with an optional absolute deadline.
    ///
    /// The deadline is checked at the same cooperative native-engine
    /// boundaries as caller cancellation. It cannot preempt a CUDA call
    /// already in flight, but it prevents the next safe boundary from
    /// starting more native work after the deadline has elapsed.
    pub fn with_deadline(deadline: Option<Instant>) -> Self {
        Self { cancelled: Arc::new(AtomicBool::new(false)), deadline }
    }

    /// Marks the execution as cancelled.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns whether this execution must stop at its next cooperative safe
    /// boundary.
    ///
    /// An elapsed deadline is intentionally an effective cancellation here.
    /// Callers that need to distinguish an explicit caller action from expiry
    /// must use [`Self::check`], which preserves explicit cancellation as the
    /// primary reason.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_requested() || self.is_expired()
    }

    /// Returns the optional absolute deadline shared by every clone.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Returns whether the optional deadline has elapsed.
    pub fn is_expired(&self) -> bool {
        self.deadline.is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// Returns a structured native error if cancellation has been requested.
    pub fn check(&self, operation: &'static str) -> Result<()> {
        if self.cancellation_requested() {
            return Err(QueryError::new(
                query_planning::error::QueryErrorCode::Cancelled,
                operation,
                "native query execution was cancelled",
            )
            .with_fact("cancellation_requested", true));
        }
        if self.is_expired() {
            return Err(QueryError::new(
                query_planning::error::QueryErrorCode::Cancelled,
                operation,
                "native query execution deadline expired",
            )
            .with_fact("cancellation_requested", true)
            .with_fact("reason", "deadline_expired"));
        }
        Ok(())
    }

    fn cancellation_requested(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

impl Default for ExecutionCancellation {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::observability::RuntimeExecutionStageContext for ExecutionContext {
    fn runtime_operator_stage_context(
        &self,
    ) -> (
        crate::observability::RuntimeObservationHandle,
        Option<crate::observability::PlanExecutionId>,
        Option<crate::observability::RuntimeSpanId>,
    ) {
        let observation = self.runtime_observation().clone();
        let (plan_execution_id, parent_span_id) = self
            .runtime_execution_scope()
            .map(|scope| (Some(scope.plan_execution_id().clone()), Some(scope.plan_span_id())))
            .unwrap_or((None, None));
        (observation, plan_execution_id, parent_span_id)
    }
}
