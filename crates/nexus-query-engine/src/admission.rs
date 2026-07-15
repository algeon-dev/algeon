//! Analysis-owned strict and adaptive bounded-pipeline admission.
//!
//! Admission composes capability analysis with compiled-pipeline structural
//! support. It does not execute GPU work. Strict and adaptive evidence and
//! reports remain separate public contracts.

mod adaptive;
mod diagnostics;
mod join_strategy;
mod shared;
mod strict;
mod support;

pub use adaptive::{
    AdaptivePipelineAdmission, adaptive_bounded_pipeline_admission,
    adaptive_bounded_pipeline_admission_with_memory_history,
};
pub use diagnostics::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts};
pub(crate) use diagnostics::{
    bounded_join_unsupported_shape_facts_with_projection, insert_bounded_join_rejection_projection_error_facts,
};
pub(crate) use join_strategy::{CompiledJoinExecutionStrategy, CompiledJoinStrategySelection};
pub use strict::{
    StrictBoundedPipelineAdmission, strict_bounded_pipeline_admission,
    strict_bounded_pipeline_admission_with_statistical_aggregate_mode, strict_bounded_pipeline_rejection_projection,
    strict_bounded_pipeline_rejection_projection_with_statistical_aggregate_mode,
};
#[doc(hidden)]
pub use support::{
    CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
};
