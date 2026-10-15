//! Analysis-owned bounded-pipeline admission.
//!
//! Admission evaluates capability analysis against compiled-pipeline structural
//! support without executing GPU work.
//!
//! [`admit`] determines from the plan alone whether a native plan can execute
//! as a compiled pipeline. The verification pipeline runs validation, capability
//! analysis, pipeline compilation, and compiled task/resource/join contract
//! checks in sequence, terminating with a rejection at the first unmet contract.
//! The resulting [`AdmittedPlan`] is consumed by downstream execution and reporting
//! without repeating admission checks.
//! Admission allocates neither device memory nor attempt capacity, and missing
//! size estimates do not trigger plan rejections; memory limits are enforced
//! by the query domain allocator during execution
//! (`website/docs/design/admission.mdx#allocation-enforcement`).
//! The `scope` module provides the plan and schema identities required by execution
//! to verify conformity with the admitted plan.

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
