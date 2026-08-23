use super::{
    CapabilityDecision, NodeId, PlanNode, ProjectionGlobalLimitShape, QueryPlan, collect_reachable_node_ids_from,
    node_operator_fact,
};
use crate::capability::aggregate::{AggregateStateContract, aggregate_state_contract};

pub(in crate::capability) fn projection_global_limit_shape_decision(
    plan: &QueryPlan,
) -> Result<ProjectionGlobalLimitShape, CapabilityDecision> {
    let sink_input_id = plan.sink().input();
    let Some(sink_input_node) = plan.node(sink_input_id) else {
        return Err(CapabilityDecision::rejected(
            "projection_global_limit_missing_node",
            [("node_id", sink_input_id.to_string()), ("role", "sink_input".to_string())],
        ));
    };
    let (post_limit_projection_id, global_limit_id, global_limit, limit_root) = match &sink_input_node.kind {
        PlanNode::Projection(final_projection) => {
            let global_limit_id = final_projection.input;
            let Some(global_limit_node) = plan.node(global_limit_id) else {
                return Err(CapabilityDecision::rejected(
                    "projection_global_limit_missing_node",
                    [("node_id", global_limit_id.to_string()), ("role", "global_limit".to_string())],
                ));
            };
            let PlanNode::Limit(global_limit) = &global_limit_node.kind else {
                return Err(CapabilityDecision::rejected(
                    "projection_global_limit_requires_projection_input_limit",
                    [
                        ("node_id", global_limit_id.to_string()),
                        ("operator", node_operator_fact(&global_limit_node.kind).to_string()),
                    ],
                ));
            };
            (Some(sink_input_id), global_limit_id, global_limit, false)
        },
        PlanNode::Limit(global_limit) => (None, sink_input_id, global_limit, true),
        _ => {
            return Err(CapabilityDecision::rejected(
                "projection_global_limit_requires_sink_projection",
                [
                    ("node_id", sink_input_id.to_string()),
                    ("operator", node_operator_fact(&sink_input_node.kind).to_string()),
                ],
            ));
        },
    };
    if global_limit.spec.fetch.is_none() {
        return Err(CapabilityDecision::rejected(
            if limit_root {
                "projection_global_limit_limit_root_requires_fetch_limit"
            } else {
                "projection_global_limit_requires_fetch_limit"
            },
            [("node_id", global_limit_id.to_string()), ("skip", global_limit.spec.skip.to_string())],
        ));
    }
    let limit_reachable = collect_reachable_node_ids_from(plan, global_limit.input)?;
    if limit_reachable
        .iter()
        .any(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Sort(_))))
    {
        return Err(CapabilityDecision::rejected(
            if limit_root {
                "projection_global_limit_limit_root_requires_non_sort_limit_input"
            } else {
                "projection_global_limit_requires_non_sort_limit_input"
            },
            [("limit_input", global_limit.input.to_string())],
        ));
    }

    let aggregate_ids = limit_reachable
        .iter()
        .copied()
        .filter(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Aggregate(_))))
        .collect::<Vec<_>>();
    let join_ids = limit_reachable
        .iter()
        .copied()
        .filter(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Join(_))))
        .collect::<Vec<_>>();
    if join_ids.is_empty() {
        return Err(CapabilityDecision::rejected(
            if limit_root {
                "projection_global_limit_limit_root_requires_join_fragment"
            } else {
                "projection_global_limit_requires_join_fragment"
            },
            [("limit_input", global_limit.input.to_string())],
        ));
    }
    validate_projection_global_limit_distinct_or_dedup_aggregates(
        plan,
        post_limit_projection_id,
        global_limit_id,
        &aggregate_ids,
    )?;

    Ok(ProjectionGlobalLimitShape { post_limit_projection_id, global_limit_id, join_ids })
}

fn validate_projection_global_limit_distinct_or_dedup_aggregates(
    plan: &QueryPlan,
    post_limit_projection_id: Option<NodeId>,
    global_limit_id: NodeId,
    aggregate_ids: &[NodeId],
) -> Result<(), CapabilityDecision> {
    for aggregate_id in aggregate_ids {
        let Some(PlanNode::Aggregate(aggregate)) = plan.node(*aggregate_id).map(|node| &node.kind) else {
            return Err(CapabilityDecision::rejected(
                "projection_global_limit_missing_node",
                [("node_id", aggregate_id.to_string()), ("role", "aggregate".to_string())],
            ));
        };
        match aggregate_state_contract(&aggregate.spec) {
            Ok(AggregateStateContract::Plain) => {},
            Ok(AggregateStateContract::DedupGroupOnly | AggregateStateContract::ExactDistinctCount) => {},
            Err(reason) => {
                return Err(CapabilityDecision::rejected(
                    "projection_global_limit_rejects_unsupported_distinct_or_dedup_aggregate",
                    [
                        (
                            "post_limit_projection_id",
                            post_limit_projection_id.map_or_else(|| "none".to_string(), |id| id.to_string()),
                        ),
                        ("global_limit_id", global_limit_id.to_string()),
                        ("aggregate_id", aggregate_id.to_string()),
                        ("aggregate_state_contract_rejection", reason.as_str().to_string()),
                        ("required_remediation", "distinct_dedup_aggregate_contract".to_string()),
                    ],
                ));
            },
        }
    }
    Ok(())
}
