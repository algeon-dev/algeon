use super::{
    CapabilityDecision, EdgeContract, InvariantSelector, NodeCapability, PlanNode, QueryPlan, SubgraphInvariant,
    closed_multi_join_subgraph_structure_decision, collect_reachable_node_ids,
    multi_join_aggregate_over_join_subplan_topk_shape_decision, reachable_local_capability_rejection,
    root_join_shape_decision, single_join_post_root_pipeline_shape_decision, solved_invariant_decision,
};

pub(in crate::capability) fn analyze_root_join_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
) -> CapabilityDecision {
    if let Some(rejection) = reachable_local_capability_rejection(plan, nodes, edges) {
        return rejection;
    }
    match root_join_shape_decision(plan) {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(decision) => decision,
    }
}

pub(in crate::capability) fn analyze_union_pipeline_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
) -> CapabilityDecision {
    if let Some(rejection) = reachable_local_capability_rejection(plan, nodes, edges) {
        return rejection;
    }
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if !reachable.iter().any(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Union(_)))) {
        return CapabilityDecision::rejected(
            "union_pipeline_requires_union_node",
            [("sink_input", plan.sink().input().to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn analyze_single_join_post_root_pipeline_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
) -> CapabilityDecision {
    if let Some(rejection) = reachable_local_capability_rejection(plan, nodes, edges) {
        return rejection;
    }
    match single_join_post_root_pipeline_shape_decision(plan) {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(decision) => decision,
    }
}

pub(in crate::capability) fn analyze_closed_multi_join_subgraph_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    if let Some(rejection) = reachable_local_capability_rejection(plan, nodes, edges) {
        return rejection;
    }
    let structure_decision = closed_multi_join_subgraph_structure_decision(plan);
    if matches!(structure_decision, CapabilityDecision::Rejected { .. }) {
        return structure_decision;
    }
    for kind in [
        InvariantSelector::TopologicalBuildProbeSchedule,
        InvariantSelector::ResidentLifetimeDominatesConsumers,
        InvariantSelector::ClosedJoinSubtree,
    ] {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_closed_multi_join_subgraph_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn analyze_multi_join_aggregate_over_join_subplan_topk_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    if let Some(rejection) = reachable_local_capability_rejection(plan, nodes, edges) {
        return rejection;
    }
    let shape = match multi_join_aggregate_over_join_subplan_topk_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.aggregate_id,
            consumer_id: shape.aggregate_projection_id,
        },
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.final_sort_id },
        InvariantSelector::TopologicalBuildProbeSchedule,
        InvariantSelector::ResidentLifetimeDominatesConsumers,
    ] {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_multi_join_aggregate_over_join_subplan_topk_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}
