//! Names the DataFusion operator that keeps a rejected plan from lowering.

use datafusion::physical_plan::{
    ExecutionPlan,
    joins::{HashJoinExec, NestedLoopJoinExec},
};

use crate::report::{CandidateShape, PlanBlocker};

/// The first join operator of `plan`, for a candidate the join admission
/// contract refused.
pub(crate) fn first_join_operator(plan: &dyn ExecutionPlan) -> Option<PlanBlocker> {
    if plan.downcast_ref::<HashJoinExec>().is_some() || plan.downcast_ref::<NestedLoopJoinExec>().is_some() {
        return Some(PlanBlocker::Operator(plan.name().to_string()));
    }
    plan.children().into_iter().find_map(|child| first_join_operator(child.as_ref()))
}

/// The first operator of `plan`, in pre-order, that native lowering has no
/// counterpart for. `candidate_shape` is the shape the plan was tested as;
/// a `RepartitionExec` the shape keeps is part of the candidate.
pub(crate) fn first_operator_without_native_counterpart(
    plan: &dyn ExecutionPlan,
    candidate_shape: &CandidateShape,
) -> Option<PlanBlocker> {
    let name = plan.name();
    if !operator_has_native_counterpart(name, candidate_shape) {
        return Some(PlanBlocker::Operator(name.to_string()));
    }
    plan.children()
        .into_iter()
        .find_map(|child| first_operator_without_native_counterpart(child.as_ref(), candidate_shape))
}

fn operator_has_native_counterpart(name: &str, candidate_shape: &CandidateShape) -> bool {
    if name == "SortExec" || name.starts_with("SortExec(") {
        return true;
    }
    matches!(
        name,
        "DataSourceExec"
            | "FilterExec"
            | "ProjectionExec"
            | "AggregateExec"
            | "LocalLimitExec"
            | "GlobalLimitExec"
            | "CoalescePartitionsExec"
            | "AlgeonNativeExec"
            | "AlgeonNativeComposedExec"
            | "AlgeonNativeGpuProducerExec"
            | "NativeAdmissionExec"
    ) || (name == "RepartitionExec" && candidate_shape.contains_operator("RepartitionExec"))
}
