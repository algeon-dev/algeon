use super::*;

pub(in crate::capability) fn analyze_mixed_keyed_residual_join_resident_grouped_aggregate_input_admission(
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
    let shape = match mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::ResidualUsesBothInputs { join_id: shape.root_join_id },
        InvariantSelector::ProjectionTransparentResidency {
            peeled_producer: shape.join_subtree_root_id,
            projection_id: shape.join_subtree_projection_id,
        },
        InvariantSelector::ProjectionTransparentResidency {
            peeled_producer: shape.grouped_aggregate_id,
            projection_id: shape.grouped_aggregate_projection_id,
        },
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
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
                    "missing_mixed_residual_resident_grouped_aggregate_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn analyze_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission(
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
    let shape = match hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.grouped_aggregate_id,
            consumer_id: shape.grouped_aggregate_projection_id,
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
                    "missing_hash_join_resident_grouped_aggregate_topk_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn analyze_rollup_rank_topk_after_aggregate_join_pushdown_admission(
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
    let shape = match rollup_rank_topk_after_aggregate_join_pushdown_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    let mut required = vec![
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.rank_sort_id },
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
    ];
    if let Some(consumer_id) = shape.rank_output_residency_consumer_id {
        required.push(InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.rank_window_id,
            consumer_id,
        });
    }
    if let Some(final_sort_id) = shape.final_sort_id {
        required.push(InvariantSelector::FinalBarrierOrder { sort_id: final_sort_id });
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
                    "missing_rollup_rank_topk_after_aggregate_join_pushdown_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn analyze_keyed_hash_join_resident_subplan_admission(
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
    let shape = match keyed_hash_join_resident_subplan_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.joined_grouped_branch_join_id },
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::SharedResidentSubplan { producer: shape.shared_grouped_aggregate_id },
        InvariantSelector::AggregateDerivesFromSharedProducer {
            shared_producer: shape.shared_grouped_aggregate_id,
            aggregate_id: shape.scalar_aggregate_id,
        },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.final_sort_id },
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
                    "missing_keyed_hash_resident_subplan_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn analyze_keyless_conditional_join_input_subplan_admission(
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
    let shape = match keyless_conditional_join_input_subplan_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::ResidualUsesBothInputs { join_id: shape.root_join_id },
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
            None if matches!(
                kind,
                InvariantSelector::TopologicalBuildProbeSchedule
                    | InvariantSelector::ResidentLifetimeDominatesConsumers
            ) && shape.reachable_join_count <= 1 => {},
            None => {
                return CapabilityDecision::rejected(
                    "missing_keyless_conditional_input_subplan_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn analyze_keyless_conditional_join_aggregate_over_join_subplan_inputs_admission(
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
    let shape = match keyless_conditional_join_aggregate_over_join_subplan_inputs_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    let mut required = vec![
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::ResidualUsesBothInputs { join_id: shape.root_join_id },
        InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.grouped_aggregate_id,
            consumer_id: shape.grouped_aggregate_projection_id,
        },
        InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.scalar_aggregate_id,
            consumer_id: shape.scalar_aggregate_projection_id,
        },
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
        InvariantSelector::TopologicalBuildProbeSchedule,
        InvariantSelector::ResidentLifetimeDominatesConsumers,
    ];
    if let Some(final_sort_id) = shape.final_sort_id {
        required.push(InvariantSelector::FinalBarrierOrder { sort_id: final_sort_id });
    }
    if let Some(shared_producer) = shape.scalar_aggregate_shared_producer_id {
        required.push(InvariantSelector::AggregateDerivesFromSharedProducer {
            shared_producer,
            aggregate_id: shape.scalar_aggregate_id,
        });
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
                    "missing_keyless_conditional_aggregate_over_join_subplan_inputs_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn analyze_multi_join_resident_grouped_aggregate_subquery_filter_admission(
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
    let shape = match multi_join_resident_grouped_aggregate_subquery_filter_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::BuildBeforeProbe { join_id: shape.residual_join_id },
        InvariantSelector::ResidualUsesBothInputs { join_id: shape.residual_join_id },
        InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.grouped_aggregate_id,
            consumer_id: shape.grouped_aggregate_projection_id,
        },
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
                    "missing_multi_join_resident_grouped_aggregate_subquery_filter_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}
