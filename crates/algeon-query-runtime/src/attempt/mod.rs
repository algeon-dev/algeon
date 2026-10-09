//! Execution-attempt lifecycle contracts.
//!
//! One attempt is one execution lifecycle; the contract is in
//! `website/docs/design/admission.mdx#from-capability-validation-to-admission`.
//! `lifecycle` is the pure state machine, `owner` wraps it in the capabilities
//! allowed to drive it, `queue` holds reports that no receiver collected, and
//! `execution` tracks occurrence claims within one invocation round. What the
//! query service needs to finish an admitted attempt is `pub(crate)`; adapters
//! see only the owner, handle, binding, and completion types.
mod cancellation;
mod execution;
mod lifecycle;
mod observation;
mod owner;
mod queue;
pub use cancellation::AttemptCancellation;
#[doc(hidden)]
pub use execution::{AttemptExecutionRound, AttemptHostOccurrence};
pub(crate) use lifecycle::{AttemptLifecycle, AttemptLifecycleError};
pub(crate) use owner::{AdmittedGrantRelease, AttemptAdmission, AttemptReportPublisher};
pub use owner::{
    AttemptAdmissionBinding, AttemptAdmissionRejection, AttemptBegin, AttemptCompletion, AttemptCompletionFactory,
    AttemptHandle, AttemptOwner, BeginAttemptRequest,
};
pub(crate) use queue::AttemptCompletionQueue;
pub use queue::{AttemptBackendCounters, AttemptBackendCountersSnapshot};
