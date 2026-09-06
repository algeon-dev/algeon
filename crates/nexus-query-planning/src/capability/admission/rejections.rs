use std::collections::BTreeSet;

use super::{
    CapabilityDecision, EdgeConsumer, EdgeContract, NodeCapability, NodeId, QueryPlan, SubgraphInvariant,
    SubgraphInvariantKind, collect_reachable_node_ids,
};

pub(in crate::capability::admission) fn node_ids_fact(node_ids: &[NodeId]) -> String {
    if node_ids.is_empty() {
        return "none".to_string();
    }
    node_ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

#[derive(Debug, Clone, Copy)]
pub(in crate::capability::admission) enum InvariantSelector {
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
    LimitBoundedProjection {
        limit_id: NodeId,
        projection_id: NodeId,
    },
}

impl InvariantSelector {
    pub(in crate::capability::admission) const fn as_str(self) -> &'static str {
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
            Self::LimitBoundedProjection { .. } => "limit_bounded_projection",
        }
    }
}

pub(in crate::capability::admission) fn solved_invariant_decision(
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
            (
                SubgraphInvariantKind::LimitBoundedProjection,
                InvariantSelector::LimitBoundedProjection { limit_id, projection_id },
            ) => invariant.nodes.as_slice() == [limit_id, projection_id],
            _ => false,
        })
        .map(|invariant| &invariant.decision)
}

fn first_reachable_node_rejection(
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

fn first_reachable_edge_rejection(edges: &[EdgeContract], reachable: &BTreeSet<NodeId>) -> Option<CapabilityDecision> {
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

/// Shared reachable node/edge precondition for a plan-shape analyzer: the
/// sink-reachable subgraph must resolve, and no reachable node or edge may
/// already be rejected, before the analyzer evaluates its own shape.
pub(in crate::capability) fn reachable_local_capability_rejection(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
) -> Option<CapabilityDecision> {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return Some(decision),
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return Some(rejection);
    }
    first_reachable_edge_rejection(edges, &reachable)
}
