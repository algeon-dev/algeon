//! Execution-attempt lifecycle contracts.
mod cancellation;
mod lifecycle;
mod owner;
mod queue;
pub use cancellation::{
    AttemptCancellation, AttemptCancellationBoundary, BoundaryCancellationState, cancellation_state_at,
    effective_queue_expiry,
};
#[doc(hidden)]
pub use lifecycle::AttemptLifecycle;
pub use lifecycle::AttemptLifecycleError;
pub use owner::{
    AttemptAdmissionBinding, AttemptAdmissionRejection, AttemptBegin, AttemptCompletion, AttemptCompletionFactory,
    AttemptHandle, AttemptOwner, BeginAttemptRequest,
};
#[doc(hidden)]
pub use owner::{AttemptFinalizer, AttemptFinalizerSlot};
pub use queue::{AttemptBackendCounters, AttemptBackendCountersSnapshot, AttemptCompletionQueue};
