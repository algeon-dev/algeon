use super::QueryPlan;

fn canonical_execution_scope(plan: &QueryPlan) -> QueryPlan {
    let mut plan = plan.clone_with_deferred_host_materialization_sink();
    plan.advisory_outputs.clear();
    plan
}

#[doc(hidden)]
pub fn canonical_execution_scope_fingerprint(plan: &QueryPlan) -> String {
    format!("{:#?}", canonical_execution_scope(plan))
}
