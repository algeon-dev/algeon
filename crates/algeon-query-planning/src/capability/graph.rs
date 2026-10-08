//! Plan-graph traversal helpers used by capability analysis.
//!
//! Provides reachability utilities over [`QueryPlan`] nodes that admission
//! and invariant analyses share. Plan construction already guarantees that
//! every node is reachable from the sink and that inputs precede consumers.
use std::collections::{BTreeMap, BTreeSet};

use super::{JoinSide, NodeId, PlanNode, QueryPlan};

/// Each node's consumers, in node order.
pub(super) fn consumer_nodes(plan: &QueryPlan) -> BTreeMap<NodeId, Vec<NodeId>> {
    let mut consumers = BTreeMap::<NodeId, Vec<NodeId>>::new();
    for node in plan.nodes() {
        for input in node.input_ids() {
            consumers.entry(*input).or_default().push(node.id);
        }
    }
    consumers
}

/// The first non-projection node at or below `node_id`, with the number of
/// projections peeled to reach it. `None` only for a missing node or a cycle,
/// neither of which a constructed plan contains.
pub(super) fn peel_projection_chain_for_capability(plan: &QueryPlan, node_id: NodeId) -> Option<(NodeId, usize)> {
    let mut current = node_id;
    let mut projection_count = 0;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        let node = plan.node(current)?;
        let PlanNode::Projection(projection) = &node.kind else {
            return Some((current, projection_count));
        };
        projection_count += 1;
        current = projection.input;
    }
}

/// The first node at or below `node_id` that is neither a filter nor a
/// projection, with the chain of peeled nodes listed consumer-first.
pub(super) fn peel_resident_unary_chain_for_capability(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Option<(NodeId, Vec<NodeId>)> {
    let mut current = node_id;
    let mut chain = Vec::new();
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        let node = plan.node(current)?;
        match &node.kind {
            PlanNode::Projection(projection) => {
                chain.push(current);
                current = projection.input;
            },
            PlanNode::Filter(filter) => {
                chain.push(current);
                current = filter.input;
            },
            PlanNode::Source(_)
            | PlanNode::Join(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_)
            | PlanNode::MlAlgorithm(_) => {
                return Some((current, chain));
            },
        }
    }
}

pub(super) const fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

pub(super) const fn opposite_join_side(side: JoinSide) -> JoinSide {
    match side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
}
