//! Names the DataFusion join a rejected bounded-join candidate holds.

use datafusion::physical_plan::{
    ExecutionPlan,
    joins::{HashJoinExec, NestedLoopJoinExec},
};

use crate::report::PlanBlocker;

/// The first join operator of `plan`, for a candidate the join admission
/// contract refused.
pub(crate) fn first_join_operator(plan: &dyn ExecutionPlan) -> Option<PlanBlocker> {
    if plan.downcast_ref::<HashJoinExec>().is_some() || plan.downcast_ref::<NestedLoopJoinExec>().is_some() {
        return Some(PlanBlocker::Operator(plan.name().to_string()));
    }
    plan.children().into_iter().find_map(|child| first_join_operator(child.as_ref()))
}
