//! Analysis-owned bounded-pipeline admission.
//!
//! Admission composes capability analysis with compiled-pipeline structural
//! support. It does not execute GPU work.

mod bounded;
mod diagnostics;
mod envelope;
mod join_strategy;
mod requirement;
mod shared;
mod support;

pub use bounded::{
    BoundedPipelineAdmission, bounded_pipeline_admission, bounded_pipeline_admission_with_statistical_aggregate_mode,
    bounded_pipeline_rejection_projection, bounded_pipeline_rejection_projection_with_statistical_aggregate_mode,
};
pub use diagnostics::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts};
pub(crate) use diagnostics::{
    bounded_join_unsupported_shape_facts_with_projection, insert_bounded_join_rejection_projection_error_facts,
};
pub(crate) use envelope::{AttemptReservationEnvelope, AttemptReservationPolicy};
pub(crate) use join_strategy::{CompiledJoinExecutionStrategy, CompiledJoinStrategySelection};
pub(crate) use requirement::AttemptCapabilityEvidence as AttemptRequirement;
pub use requirement::QueryPlanCapabilityProof;
#[cfg(test)]
pub(crate) use requirement::{AttemptCapabilityEvidence, EstimateQuality};
#[doc(hidden)]
pub use support::{
    CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
};
