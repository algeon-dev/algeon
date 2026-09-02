use super::*;

pub(in crate::capability) fn analyze_root_topk_join_aggregate_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match root_topk_join_aggregate_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    let mut required = shape
        .join_ids
        .iter()
        .copied()
        .map(|join_id| InvariantSelector::BuildBeforeProbe { join_id })
        .collect::<Vec<_>>();
    required.extend(
        shape.shared_producer_ids.iter().copied().map(|producer| InvariantSelector::SharedResidentSubplan { producer }),
    );
    if shape.aggregate_id != shape.aggregate_projection_id {
        required.push(InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.aggregate_id,
            consumer_id: shape.aggregate_projection_id,
        });
    }
    required.extend(
        shape.window_input_sort_ids.iter().copied().map(|sort_id| InvariantSelector::FinalBarrierOrder { sort_id }),
    );
    required.extend([
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.final_sort_id },
    ]);
    if shape.join_ids.len() > 1 {
        required.extend([
            InvariantSelector::TopologicalBuildProbeSchedule,
            InvariantSelector::ResidentLifetimeDominatesConsumers,
        ]);
    }

    for kind in required {
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
                    "missing_root_topk_join_aggregate_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}
