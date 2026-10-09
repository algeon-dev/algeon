//! Analysis-owned bounded-pipeline admission.
//!
//! Admission composes capability analysis with compiled-pipeline structural
//! support. It does not execute GPU work.
//!
//! [`admit`] decides once, from the plan alone, whether a native plan runs as
//! a compiled pipeline. Validation, capability analysis, pipeline compilation
//! and the compiled task, resource and join contracts run in that order, and
//! the first unmet contract becomes the rejection. The result is an
//! [`AdmittedPlan`]; execution and reporting read it and do not decide again.
//! Admission claims no device memory and no attempt capacity, and a
//! missing size estimate cannot reject a plan; the query domain's allocator
//! enforces allocation limits during execution
//! (`website/docs/design/admission.mdx#allocation-enforcement`). The `scope`
//! module carries the plan and schema identity that lets execution check it
//! is running the plan that was admitted.

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
