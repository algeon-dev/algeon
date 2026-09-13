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

pub use bounded::{BoundedPipelineAdmission, BoundedPipelineDecision, bounded_pipeline_decision};
#[doc(hidden)]
pub use bounded::{artifact_binding_error, statistical_aggregate_execution_mode_rejection};
pub use diagnostics::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts};
#[doc(hidden)]
pub use diagnostics::{
    bounded_join_unsupported_shape_facts_with_projection, insert_bounded_join_rejection_projection_error_facts,
};
#[doc(hidden)]
pub use evidence::{
    AttemptCapabilityEvidence, AttemptCapabilityEvidence as AttemptRequirement, AttestedCugraphGraphExecutionScope,
    CUGRAPH_GRAPH_EXECUTION_CAPABILITY, ExecutionCapability, capability_labels,
};
#[doc(hidden)]
pub use scope::{
    AttemptExecutionBoundary, AttemptExecutionManifest, AttemptExecutionOccurrence, AttemptExecutionScope,
    AttemptProducerOccurrence,
};
#[doc(hidden)]
pub use support::{
    CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
    pipeline_tasks_within_migrated_contract,
};

#[doc(hidden)]
pub use crate::capability::{CompiledJoinExecutionStrategy, CompiledJoinStrategySelection};
#[doc(hidden)]
pub use crate::{
    plan::{DeferredGraphExecutionScopeIdentity, canonical_execution_scope_fingerprint},
    util::log_safe_fingerprint,
};
