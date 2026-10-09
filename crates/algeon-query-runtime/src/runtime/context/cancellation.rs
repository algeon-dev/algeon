//! Cooperative cancellation token for native query execution.

use std::time::{Duration, Instant};

use query_planning::error::{QueryError, QueryErrorCode, Result};
use tokio::time::sleep_until;
use tokio_util::sync::{CancellationToken, WaitForCancellationFuture};

use super::ExecutionContext;
use crate::{
    observability::{PlanExecutionId, RuntimeExecutionStageContext, RuntimeObservationHandle, RuntimeSpanId},
    runtime::deadline::block_until,
};

/// Shared cooperative cancellation token for native query execution.
///
/// Cancellation is observed only at native-engine boundaries. It does not
/// preempt a libcudf/CUDA call already in flight, but it prevents the next
/// source chunk, pipeline task, operator, or materialization step from starting
/// once it is requested. A wait can also resolve on it through
/// [`Self::cancelled_or_expired`].
#[derive(Debug, Clone)]
pub struct ExecutionCancellation {
    token: CancellationToken,
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
        Self { token: CancellationToken::new(), deadline }
    }

    /// Marks the execution as cancelled and wakes every
    /// [`Self::cancelled_or_expired`] waiter.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Resolves once the execution is cancelled or its deadline elapses, so a
    /// wait with no other wakeup for either can race it. [`Self::check`]
    /// still tells the two apart. Must run inside a Tokio runtime with the
    /// time driver enabled when a deadline is set.
    pub async fn cancelled_or_expired(&self) {
        match self.deadline {
            Some(deadline) => tokio::select! {
                () = self.token.cancelled() => {},
                () = sleep_until(deadline.into()) => {},
            },
            None => self.token.cancelled().await,
        }
    }

    /// Resolves to the typed error that ends a wait: the cancellation, or the
    /// deadline once it elapses. A waiter returns it directly and does not
    /// read the clock again, since a paused Tokio clock can disagree with a
    /// fresh reading.
    pub async fn terminated(&self, operation: &'static str) -> QueryError {
        self.cancelled_or_expired().await;
        match self.check(operation) {
            Err(error) => error,
            Ok(()) => deadline_expired_error(operation),
        }
    }

    /// Resolves once cancellation is requested, ignoring the deadline. It
    /// needs no Tokio timer, so a blocking wait races it and bounds itself by
    /// [`Self::deadline`].
    pub(crate) fn cancel_requested(&self) -> WaitForCancellationFuture<'_> {
        self.token.cancelled()
    }

    /// Blocks the calling thread for `duration`. Cancellation or an elapsed
    /// deadline ends the wait early with the typed error [`Self::check`]
    /// reports.
    pub fn sleep(&self, duration: Duration, operation: &'static str) -> Result<()> {
        let until = Instant::now() + duration;
        let until = self.deadline.map_or(until, |deadline| deadline.min(until));
        block_until(self.cancel_requested(), Some(until));
        self.check(operation)
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

    /// Returns a structured native error if cancellation has been requested or the deadline has
    /// elapsed. Explicit cancellation takes precedence over expiry.
    pub fn check(&self, operation: &'static str) -> Result<()> {
        if self.cancellation_requested() {
            return Err(QueryError::new(QueryErrorCode::Cancelled, operation, "native query execution was cancelled")
                .with_fact("cancellation_requested", true));
        }
        if self.is_expired() {
            return Err(deadline_expired_error(operation));
        }
        Ok(())
    }

    fn cancellation_requested(&self) -> bool {
        self.token.is_cancelled()
    }
}

fn deadline_expired_error(operation: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, operation, "native query execution deadline expired")
        .with_fact("cancellation_requested", true)
        .with_fact("reason", "deadline_expired")
}

impl Default for ExecutionCancellation {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeExecutionStageContext for ExecutionContext {
    fn runtime_operator_stage_context(
        &self,
    ) -> (RuntimeObservationHandle, Option<PlanExecutionId>, Option<RuntimeSpanId>) {
        let observation = self.runtime_observation().clone();
        let (plan_execution_id, parent_span_id) = self
            .runtime_execution_scope()
            .map(|scope| (Some(scope.plan_execution_id().clone()), Some(scope.plan_span_id())))
            .unwrap_or((None, None));
        (observation, plan_execution_id, parent_span_id)
    }
}
