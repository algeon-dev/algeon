use super::{graph::*, model::*, shape::*, *};

pub(super) fn analyze_root_join_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
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
    match root_join_shape_decision(plan) {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(decision) => decision,
    }
}

pub(super) fn analyze_union_pipeline_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
) -> CapabilityDecision {
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
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_single_join_post_root_pipeline_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
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
    match single_join_post_root_pipeline_shape_decision(plan) {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(decision) => decision,
    }
}

pub(super) fn analyze_closed_multi_join_subgraph_admission(
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

pub(super) fn analyze_multi_join_aggregate_over_join_subplan_topk_admission(
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

pub(super) fn analyze_mixed_keyed_residual_join_resident_grouped_aggregate_input_admission(
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

pub(super) fn analyze_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission(
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

pub(super) fn analyze_keyed_hash_join_resident_subplan_admission(
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

pub(super) fn analyze_keyless_conditional_join_input_subplan_admission(
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

pub(super) fn analyze_keyless_conditional_join_aggregate_over_join_subplan_inputs_admission(
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

pub(super) fn analyze_multi_join_resident_grouped_aggregate_subquery_filter_admission(
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

#[derive(Debug, Clone, Copy)]
enum InvariantSelector {
    BuildBeforeProbe {
        join_id: NodeId,
    },
    TopologicalBuildProbeSchedule,
    /// Mirrors `SubgraphInvariantKind::ResidentLifetimeDominatesConsumers`: all
    /// reachable resident producers must outlive every admitted consumer edge.
    ResidentLifetimeDominatesConsumers,
    ClosedJoinSubtree,
    SharedResidentSubplan {
        producer: NodeId,
    },
    /// Mirrors `SubgraphInvariantKind::AggregateDerivesFromSharedProducer` for
    /// a scalar aggregate whose inputs must trace back to one shared producer.
    AggregateDerivesFromSharedProducer {
        shared_producer: NodeId,
        aggregate_id: NodeId,
    },
    FinalBarrierOrder {
        sort_id: NodeId,
    },
    PostRootStageOrder {
        root_join_id: NodeId,
    },
    /// Mirrors `SubgraphInvariantKind::ProjectionTransparentResidency`: the
    /// named projection may be peeled only if it preserves resident output.
    ProjectionTransparentResidency {
        peeled_producer: NodeId,
        projection_id: NodeId,
    },
    /// Mirrors `SubgraphInvariantKind::ResidentUnaryChainPreservesResidency`:
    /// the selected unary chain must stay connected and resident end to end.
    ResidentUnaryChainPreservesResidency {
        peeled_producer: NodeId,
        consumer_id: NodeId,
    },
    ResidualUsesBothInputs {
        join_id: NodeId,
    },
}

impl InvariantSelector {
    const fn as_str(self) -> &'static str {
        match self {
            Self::BuildBeforeProbe { .. } => "build_before_probe",
            Self::TopologicalBuildProbeSchedule => "topological_build_probe_schedule",
            Self::ResidentLifetimeDominatesConsumers => "resident_lifetime_dominates_consumers",
            Self::ClosedJoinSubtree => "closed_join_subtree",
            Self::SharedResidentSubplan { .. } => "shared_resident_subplan",
            Self::AggregateDerivesFromSharedProducer { .. } => "aggregate_derives_from_shared_producer",
            Self::FinalBarrierOrder { .. } => "final_barrier_order",
            Self::PostRootStageOrder { .. } => "post_root_stage_order",
            Self::ProjectionTransparentResidency { .. } => "projection_transparent_residency",
            Self::ResidentUnaryChainPreservesResidency { .. } => "resident_unary_chain_preserves_residency",
            Self::ResidualUsesBothInputs { .. } => "residual_uses_both_inputs",
        }
    }
}

fn solved_invariant_decision(
    invariants: &[SubgraphInvariant],
    selector: InvariantSelector,
) -> Option<&CapabilityDecision> {
    invariants
        .iter()
        .find(|invariant| match (&invariant.kind, selector) {
            (
                SubgraphInvariantKind::BuildBeforeProbe { join_id: left },
                InvariantSelector::BuildBeforeProbe { join_id: right },
            ) => *left == right,
            (
                SubgraphInvariantKind::TopologicalBuildProbeSchedule,
                InvariantSelector::TopologicalBuildProbeSchedule,
            ) => true,
            (
                SubgraphInvariantKind::ResidentLifetimeDominatesConsumers,
                InvariantSelector::ResidentLifetimeDominatesConsumers,
            ) => true,
            (SubgraphInvariantKind::ClosedJoinSubtree { .. }, InvariantSelector::ClosedJoinSubtree) => true,
            (SubgraphInvariantKind::SharedResidentSubplan, InvariantSelector::SharedResidentSubplan { producer }) => {
                invariant.nodes.first().is_some_and(|node| *node == producer)
            },
            (
                SubgraphInvariantKind::AggregateDerivesFromSharedProducer,
                InvariantSelector::AggregateDerivesFromSharedProducer { shared_producer, aggregate_id },
            ) => {
                invariant.nodes.first().is_some_and(|node| *node == shared_producer)
                    && invariant.nodes.last().is_some_and(|node| *node == aggregate_id)
            },
            (SubgraphInvariantKind::FinalBarrierOrder, InvariantSelector::FinalBarrierOrder { sort_id }) => {
                invariant.nodes.last().is_some_and(|node| *node == sort_id)
            },
            (
                SubgraphInvariantKind::PostRootStageOrder { root_join_id: left },
                InvariantSelector::PostRootStageOrder { root_join_id: right },
            ) => *left == right,
            (
                SubgraphInvariantKind::ProjectionTransparentResidency,
                InvariantSelector::ProjectionTransparentResidency { peeled_producer, projection_id },
            ) => invariant.nodes.as_slice() == [peeled_producer, projection_id],
            (
                SubgraphInvariantKind::ResidentUnaryChainPreservesResidency,
                InvariantSelector::ResidentUnaryChainPreservesResidency { peeled_producer, consumer_id },
            ) => {
                invariant.nodes.first().is_some_and(|node| *node == peeled_producer)
                    && invariant.nodes.last().is_some_and(|node| *node == consumer_id)
            },
            (
                SubgraphInvariantKind::ResidualUsesBothInputs { join_id: left },
                InvariantSelector::ResidualUsesBothInputs { join_id: right },
            ) => *left == right,
            _ => false,
        })
        .map(|invariant| &invariant.decision)
}

pub(super) fn first_reachable_node_rejection(
    nodes: &[NodeCapability],
    reachable: &BTreeSet<NodeId>,
) -> Option<CapabilityDecision> {
    nodes.iter().find_map(|node| {
        if !reachable.contains(&node.node_id) {
            return None;
        }
        match &node.executable {
            CapabilityDecision::Rejected { reason, facts } => {
                Some(CapabilityDecision::Rejected { reason, facts: facts.clone() })
            },
            _ => None,
        }
    })
}

pub(super) fn first_reachable_edge_rejection(
    edges: &[EdgeContract],
    reachable: &BTreeSet<NodeId>,
) -> Option<CapabilityDecision> {
    edges.iter().find_map(|edge| {
        if !reachable.contains(&edge.producer) {
            return None;
        }
        if let EdgeConsumer::Node(consumer) = edge.consumer
            && !reachable.contains(&consumer)
        {
            return None;
        }
        match &edge.decision {
            CapabilityDecision::Rejected { reason, facts } => {
                Some(CapabilityDecision::Rejected { reason, facts: facts.clone() })
            },
            _ => None,
        }
    })
}
