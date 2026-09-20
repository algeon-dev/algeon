//! Execution-attempt lifecycle contracts.
mod cancellation;
mod execution;
mod lifecycle;
mod owner;
mod queue;
pub use cancellation::AttemptCancellation;
#[doc(hidden)]
pub use execution::{AttemptExecutionRound, AttemptHostOccurrence};
pub(crate) use lifecycle::{AttemptLifecycle, AttemptLifecycleError};
pub use owner::{
    AttemptAdmissionBinding, AttemptAdmissionRejection, AttemptBegin, AttemptCompletion, AttemptCompletionFactory,
    AttemptHandle, AttemptOwner, BeginAttemptRequest,
};
pub(crate) use owner::{AttemptFinalizer, AttemptFinalizerSlot};
pub(crate) use queue::AttemptCompletionQueue;
pub use queue::{AttemptBackendCounters, AttemptBackendCountersSnapshot};
