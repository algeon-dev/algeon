//! Attempt-scoped cooperative cancellation and deadline helpers.
use std::time::Instant;

use query_planning::error::Result;

use crate::runtime::ExecutionCancellation;

/// The attempt's cooperative cancellation signal and optional deadline.
///
/// All clones, and the `ExecutionCancellation` that native execution receives
/// through `execution_cancellation`, share one token. `cancel` is
/// irreversible: it stops the next native boundary from starting work and
/// leaves already submitted GPU work to settle, so cleanup and the terminal
/// report still complete.
#[derive(Debug, Clone)]
pub struct AttemptCancellation {
    execution: ExecutionCancellation,
    deadline: Option<Instant>,
}

impl AttemptCancellation {
    pub(crate) fn new(deadline: Option<Instant>) -> Self {
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

    /// The same token for the native engine's boundary checks; cancelling
    /// either side cancels both.
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
