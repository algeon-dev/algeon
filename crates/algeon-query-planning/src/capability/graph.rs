//! Plan-graph traversal helpers used by capability analysis.
//!
//! Provides reachability and topological-walk utilities over [`QueryPlan`]
//! nodes that admission and invariant analyses share.
use std::collections::{BTreeMap, BTreeSet};

use super::{JoinSide, NodeId, PlanNode, QueryPlan, model::CapabilityDecision};

// Proves a dependency walk can be scheduled without cycles or missing nodes.
pub(super) struct GraphScheduleProof<'a> {
    plan: &'a QueryPlan,
    state: Vec<VisitState>,
}

impl<'a> GraphScheduleProof<'a> {
    pub(super) fn new(plan: &'a QueryPlan) -> Self {
        Self { plan, state: vec![VisitState::Unvisited; plan.nodes().len()] }
    }

    pub(super) fn schedule_from(&mut self, node_id: NodeId) -> Result<(), CapabilityDecision> {
        let Some(node) = self.plan.node(node_id) else {
            return Err(CapabilityDecision::rejected(
                "topological_schedule_missing_node",
                [("node_id", node_id.to_string())],
            ));
        };
        match self.state.get(node_id.index()).copied() {
            Some(VisitState::Done) => return Ok(()),
            Some(VisitState::Visiting) => {
                return Err(CapabilityDecision::rejected(
                    "topological_schedule_cycle",
                    [("node_id", node_id.to_string())],
                ));
            },
            Some(VisitState::Unvisited) => {},
            None => {
                return Err(CapabilityDecision::rejected(
                    "topological_schedule_missing_node",
                    [("node_id", node_id.to_string())],
                ));
            },
        }
        self.state[node_id.index()] = VisitState::Visiting;
        match &node.kind {
            PlanNode::Join(join) => {
                let build_input = join.inputs[if join.spec.build_side == JoinSide::Left { 0 } else { 1 }];
                let probe_input = join.inputs[if join.spec.build_side == JoinSide::Left { 1 } else { 0 }];
                self.schedule_from(build_input)?;
                self.schedule_from(probe_input)?;
            },
            _ => {
                for input in node.input_ids() {
                    self.schedule_from(*input)?;
                }
            },
        }
        self.state[node_id.index()] = VisitState::Done;
        Ok(())
    }
}

// Three-color DFS marker shared by graph proofs and reachability collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VisitState {
    Unvisited,
    Visiting,
    Done,
}

pub(super) fn collect_reachable_node_ids(plan: &QueryPlan) -> Result<BTreeSet<NodeId>, CapabilityDecision> {
    collect_reachable_node_ids_from(plan, plan.sink().input())
}

pub(super) fn collect_reachable_node_ids_from(
    plan: &QueryPlan,
    root: NodeId,
) -> Result<BTreeSet<NodeId>, CapabilityDecision> {
    let mut collector = ReachableCollector::new(plan);
    collector.collect(root)?;
    Ok(collector.reachable)
}

// Collects the sink-reachable subgraph while rejecting malformed cycles.
pub(super) struct ReachableCollector<'a> {
    plan: &'a QueryPlan,
    state: Vec<VisitState>,
    reachable: BTreeSet<NodeId>,
}

impl<'a> ReachableCollector<'a> {
    fn new(plan: &'a QueryPlan) -> Self {
        Self { plan, state: vec![VisitState::Unvisited; plan.nodes().len()], reachable: BTreeSet::new() }
    }

    fn collect(&mut self, node_id: NodeId) -> Result<(), CapabilityDecision> {
        let Some(node) = self.plan.node(node_id) else {
            return Err(CapabilityDecision::rejected(
                "reachable_subgraph_missing_node",
                [("node_id", node_id.to_string())],
            ));
        };
        match self.state.get(node_id.index()).copied() {
            Some(VisitState::Done) => return Ok(()),
            Some(VisitState::Visiting) => {
                return Err(CapabilityDecision::rejected(
                    "reachable_subgraph_cycle",
                    [("node_id", node_id.to_string())],
                ));
            },
            Some(VisitState::Unvisited) => {},
            None => {
                return Err(CapabilityDecision::rejected(
                    "reachable_subgraph_missing_node",
                    [("node_id", node_id.to_string())],
                ));
            },
        }
        self.state[node_id.index()] = VisitState::Visiting;
        self.reachable.insert(node_id);
        for input in node.input_ids() {
            self.collect(*input)?;
        }
        self.state[node_id.index()] = VisitState::Done;
        Ok(())
    }
}

pub(super) fn reachable_join_count_from_sink(plan: &QueryPlan) -> Result<usize, CapabilityDecision> {
    let reachable = collect_reachable_node_ids(plan)?;
    Ok(reachable
        .into_iter()
        .filter(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Join(_))))
        .count())
}

pub(super) fn consumer_nodes_for_reachable(
    plan: &QueryPlan,
    reachable: &BTreeSet<NodeId>,
) -> BTreeMap<NodeId, Vec<NodeId>> {
    let mut consumers = BTreeMap::<NodeId, Vec<NodeId>>::new();
    for node_id in reachable {
        let Some(node) = plan.node(*node_id) else {
            continue;
        };
        for input in node.input_ids() {
            if reachable.contains(input) {
                consumers.entry(*input).or_default().push(*node_id);
            }
        }
    }
    consumers
}

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
