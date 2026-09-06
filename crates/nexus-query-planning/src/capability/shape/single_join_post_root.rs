use super::{
    CapabilityDecision, QueryPlan, generic_join_node, reachable_join_count_from_sink, root_join_with_post_root_stages,
};

pub(in crate::capability) fn single_join_post_root_pipeline_shape_decision(
    plan: &QueryPlan,
) -> Result<(), CapabilityDecision> {
    let (root_join_id, post_root_stages) = root_join_with_post_root_stages(plan, "single_join_post_root_pipeline")?;
    if post_root_stages.is_empty() {
        return Err(CapabilityDecision::rejected(
            "single_join_post_root_pipeline_requires_post_root_stage",
            [("root_join_id", root_join_id.to_string())],
        ));
    }
    generic_join_node(plan, root_join_id, "single_join_post_root_pipeline_requires_supported_root_join")?;
    let reachable_join_count = reachable_join_count_from_sink(plan)?;
    if reachable_join_count != 1 {
        return Err(CapabilityDecision::rejected(
            "single_join_post_root_pipeline_requires_single_reachable_join",
            [("root_join_id", root_join_id.to_string()), ("reachable_join_count", reachable_join_count.to_string())],
        ));
    }
    Ok(())
}
