//! Execution-attempt lifecycle contracts.
//!
//! An attempt represents an isolated execution lifecycle, with contracts documented in
//! `website/docs/design/admission.mdx#from-capability-validation-to-admission`.
//! `lifecycle` contains the underlying state machine, `owner` supplies drive capabilities,
//! `queue` buffers uncollected reports, and `execution` tracks occurrence claims within
//! an invocation round. Query service completion interfaces remain `pub(crate)`, exposing
//! only owner, handle, binding, and completion types to external adapters.
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
