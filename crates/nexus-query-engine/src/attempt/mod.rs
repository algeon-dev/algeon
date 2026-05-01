//! Execution-attempt lifecycle contracts.
mod cancellation;
mod lifecycle;
mod owner;
mod queue;
pub use cancellation::{
    AttemptCancellation, AttemptCancellationBoundary, BoundaryCancellationState, cancellation_state_at,
    effective_queue_expiry,
};
pub(crate) use lifecycle::AttemptLifecycle;
pub use lifecycle::AttemptLifecycleError;
pub(crate) use owner::AttemptReleaseCommit;
pub use owner::{
    AttemptAdmissionBinding, AttemptAdmissionRejection, AttemptBegin, AttemptCompletion, AttemptHandle, AttemptOwner,
    BeginAttemptRequest,
};
pub use queue::{AttemptBackendCounters, AttemptBackendCountersSnapshot, AttemptCompletionQueue};
