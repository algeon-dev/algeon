//! Analysis-owned bounded-pipeline admission.
//!
//! Admission composes capability analysis with compiled-pipeline structural
//! support. It does not execute GPU work.

mod bounded;
mod diagnostics;
mod evidence;
mod join_strategy;
mod scope;
mod shared;
mod support;

#[doc(hidden)]
pub use bounded::bounded_cugraph_pipeline_admission;
pub use bounded::{
    BoundedPipelineAdmission, bounded_pipeline_admission, bounded_pipeline_admission_with_statistical_aggregate_mode,
    bounded_pipeline_rejection_projection, bounded_pipeline_rejection_projection_with_statistical_aggregate_mode,
};
pub use diagnostics::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts};
#[doc(hidden)]
pub use diagnostics::{
    bounded_join_unsupported_shape_facts_with_projection, insert_bounded_join_rejection_projection_error_facts,
};
#[doc(hidden)]
pub use evidence::{
    AttemptCapabilityEvidence, AttemptCapabilityEvidence as AttemptRequirement, AttestedCugraphGraphExecutionScope,
    CUGRAPH_GRAPH_EXECUTION_CAPABILITY, ExecutionCapability,
};
#[doc(hidden)]
pub use join_strategy::{CompiledJoinExecutionStrategy, CompiledJoinStrategySelection};
#[doc(hidden)]
pub use scope::{AttemptExecutionScope, AttemptProducerOccurrence};
#[doc(hidden)]
pub use support::{
    CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
};

#[doc(hidden)]
pub use crate::{plan::canonical_execution_scope_fingerprint, util::log_safe_fingerprint};
