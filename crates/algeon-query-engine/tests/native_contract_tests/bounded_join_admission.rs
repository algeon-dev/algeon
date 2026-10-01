use std::collections::BTreeMap;

use query_engine::{
    admission::{AdmittedPlan, admit, bounded_join_modeled_facts, bounded_join_unsupported_shape_facts},
    backend::AnalysisContext,
    error::{QueryError, Result},
    mode::StatisticalAggregateExecutionMode,
    plan::QueryPlan,
    rapids_backend::rapids_analysis_context,
};

use super::*;

fn bounded_pipeline_admission(plan: &QueryPlan) -> Result<Option<Arc<AdmittedPlan>>> {
    Ok(admit(plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default(), rapids_analysis_context())?
        .admission()
        .cloned())
}

/// The error `admit` returns for a plan the validator finds malformed.
fn malformed_plan_error(plan: &QueryPlan) -> QueryError {
    admit(plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default(), rapids_analysis_context())
        .expect_err("a malformed plan has no admission decision")
}

fn modeled_facts(
    plan: &QueryPlan,
    mode: StatisticalAggregateExecutionMode,
    ctx: AnalysisContext<'_>,
) -> Option<BTreeMap<&'static str, String>> {
    let decision = admit(plan, mode, ctx.policy(), ctx).unwrap();
    bounded_join_modeled_facts(plan, &decision, ctx)
}

fn unsupported_facts(
    plan: &QueryPlan,
    mode: StatisticalAggregateExecutionMode,
    ctx: AnalysisContext<'_>,
) -> Option<BTreeMap<&'static str, String>> {
    let decision = admit(plan, mode, ctx.policy(), ctx).unwrap();
    bounded_join_unsupported_shape_facts(plan, &decision, ctx)
}

#[path = "bounded_join_admission/aggregate_over_join_subplans.rs"]
mod aggregate_over_join_subplans;
#[path = "bounded_join_admission/mixed_residual.rs"]
mod mixed_residual;
#[path = "bounded_join_admission/plans.rs"]
#[expect(dead_code, reason = "fixtures shared with the GPU pipeline target are not all used by this target")]
pub(crate) mod plans;
#[path = "bounded_join_admission/residual_resident_grouped_aggregate.rs"]
mod residual_resident_grouped_aggregate;
use plans::*;

include!("bounded_join_admission/support.rs");
include!("bounded_join_admission/strategy.rs");
include!("bounded_join_admission/resident_subplan.rs");
include!("bounded_join_admission/root_topk.rs");
include!("bounded_join_admission/conditional_subplan.rs");
