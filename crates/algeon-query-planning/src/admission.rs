//! Analysis-owned bounded-pipeline admission.
//!
//! Admission composes capability analysis with compiled-pipeline structural
//! support. It does not execute GPU work.

mod bounded;
mod diagnostics;
mod evidence;
mod scope;
mod shared;
mod support;

pub use bounded::{AdmissionDecision, AdmissionRejection, AdmittedPlan, admit};
pub use diagnostics::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts};
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub use evidence::CUGRAPH_GRAPH_EXECUTION_CAPABILITY;
#[doc(hidden)]
pub use evidence::{AttemptCapabilityEvidence, ExecutionCapability, capability_labels};
#[doc(hidden)]
pub use scope::{
    AttemptExecutionManifest, AttemptExecutionOccurrence, AttemptExecutionScope, AttemptProducerOccurrence,
};
#[doc(hidden)]
pub use support::pipeline_tasks_within_migrated_contract;
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub use support::{
    CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
};
