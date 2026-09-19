use query_engine::admission::{
    bounded_join_modeled_facts as modeled_facts, bounded_join_unsupported_shape_facts as unsupported_facts,
};

use super::*;

fn bounded_pipeline_admission(
    plan: &query_engine::plan::QueryPlan,
) -> query_engine::error::Result<Option<query_engine::admission::BoundedPipelineAdmission>> {
    query_engine::admission::bounded_pipeline_decision(
        plan,
        query_engine::mode::StatisticalAggregateExecutionMode::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
    )?
    .into_execution_admission()
}

#[path = "bounded_join_admission/aggregate_over_join_subplans.rs"]
mod aggregate_over_join_subplans;
#[path = "bounded_join_admission/mixed_residual.rs"]
mod mixed_residual;
#[path = "bounded_join_admission/plans.rs"]
pub(crate) mod plans;
#[path = "bounded_join_admission/residual_resident_grouped_aggregate.rs"]
mod residual_resident_grouped_aggregate;
use plans::*;

include!("bounded_join_admission/support.rs");
include!("bounded_join_admission/strategy.rs");
include!("bounded_join_admission/resident_subplan.rs");
include!("bounded_join_admission/root_topk.rs");
include!("bounded_join_admission/conditional_subplan.rs");
