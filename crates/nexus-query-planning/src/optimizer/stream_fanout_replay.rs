use std::collections::{BTreeMap, BTreeSet};

use super::{consumers_by_node, rewrite};
use crate::plan::{JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryNode, QueryPlan};

/// Stable rule name used in optimizer diagnostics.
pub const STREAM_FANOUT_REPLAY_RULE_NAME: &str = "stream_fanout_replay";

/// Configuration for replaying shared stream probe-input subtrees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamFanoutReplayOptions {
    /// Whether the pass may duplicate replayable probe-input subtrees.
    pub enabled: bool,
}

impl StreamFanoutReplayOptions {
    /// Returns options with stream-fanout replay disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for StreamFanoutReplayOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Diagnostics for replayable shared stream fanouts considered by the optimizer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamFanoutReplayDiagnostics {
    /// Number of shared fanout candidates considered.
    pub considered_count: usize,
    /// Number of fanout candidates rewritten by replaying the producer subtree.
    pub applied_count: usize,
    /// Per-candidate evidence in discovery order.
    pub candidates: Vec<StreamFanoutReplayCandidateDiagnostics>,
}

impl StreamFanoutReplayDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate stream-fanout replay evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamFanoutReplayCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Shared producer node whose subtree is replayed for each join probe.
    pub fanout_node_id: NodeId,
    /// Number of direct consumers of the shared producer node.
    pub consumer_count: usize,
    /// Join-probe consumer nodes that share this producer.
    pub join_probe_consumer_node_ids: Vec<NodeId>,
    /// True when the returned plan duplicates the producer subtree.
    pub applied: bool,
    /// Rejection reason when no rewrite was applied.
    pub rejected_reason: Option<StreamFanoutReplayRejectedReason>,
}

/// Reasons a stream-fanout replay candidate was not rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFanoutReplayRejectedReason {
    /// Candidate was found outside the sink-reachable subtree.
    NotReachableFromSink,
}

#[doc(hidden)]
pub struct StreamFanoutReplayResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: StreamFanoutReplayDiagnostics,
}

#[doc(hidden)]
pub fn optimize_stream_fanout_replay(plan: QueryPlan, options: &StreamFanoutReplayOptions) -> StreamFanoutReplayResult {
    if !options.enabled {
        return StreamFanoutReplayResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: StreamFanoutReplayDiagnostics::default(),
        };
    }

    let candidates = replay_candidates(&plan);
    if candidates.is_empty() {
        return StreamFanoutReplayResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: StreamFanoutReplayDiagnostics::default(),
        };
    }

    let fanout_nodes = candidates.keys().copied().collect::<BTreeSet<_>>();
    let mut rewriter = StreamFanoutReplayRewriter::new(&plan, fanout_nodes);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    let applied_fanouts = rewriter.applied_fanouts;
    let changed = rewriter.changed;
    let rewritten_plan = rewrite::finish_rewritten_plan(rewriter.nodes, new_root, plan.sink())
        .with_keyed_hash_join_strategy_options_from(&plan);

    let mut diagnostics = StreamFanoutReplayDiagnostics::default();
    let direct_consumers = consumers_by_node(&plan);
    for (fanout_node_id, join_probe_consumer_node_ids) in candidates {
        diagnostics.considered_count += 1;
        let applied = applied_fanouts.contains(&fanout_node_id);
        if applied {
            diagnostics.applied_count += 1;
        }
        diagnostics.candidates.push(StreamFanoutReplayCandidateDiagnostics {
            rule_name: STREAM_FANOUT_REPLAY_RULE_NAME,
            fanout_node_id,
            consumer_count: direct_consumers[&fanout_node_id].len(),
            join_probe_consumer_node_ids,
            applied,
            rejected_reason: (!applied).then_some(StreamFanoutReplayRejectedReason::NotReachableFromSink),
        });
    }

    let change = if changed { super::NativeOptimizerChange::Changed } else { super::NativeOptimizerChange::Unchanged };
    StreamFanoutReplayResult { plan: rewritten_plan, change, diagnostics }
}

struct StreamFanoutReplayRewriter<'a> {
    original: &'a QueryPlan,
    fanout_nodes: BTreeSet<NodeId>,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
    applied_fanouts: BTreeSet<NodeId>,
    changed: bool,
}

impl<'a> StreamFanoutReplayRewriter<'a> {
    fn new(original: &'a QueryPlan, fanout_nodes: BTreeSet<NodeId>) -> Self {
        Self {
            original,
            fanout_nodes,
            nodes: Vec::new(),
            memo: BTreeMap::new(),
            applied_fanouts: BTreeSet::new(),
            changed: false,
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if self.fanout_nodes.contains(&old_id) {
            self.changed = true;
            self.applied_fanouts.insert(old_id);
            return self.clone_subtree_fresh(old_id);
        }
        if let Some(new_id) = self.memo.get(&old_id) {
            return *new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        let new_kind = self.clone_kind_with_rewritten_inputs(&old_node.kind);
        let new_id = self.push_node(new_kind, old_node.facts.clone());
        self.memo.insert(old_id, new_id);
        new_id
    }

    fn clone_subtree_fresh(&mut self, old_id: NodeId) -> NodeId {
        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        let new_kind = self.clone_kind_with_fresh_inputs(&old_node.kind);
        self.push_node(new_kind, old_node.facts.clone())
    }

    fn clone_kind_with_rewritten_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        self.clone_kind_with_input_mapper(kind, CloneMode::Memoized)
    }

    fn clone_kind_with_fresh_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        self.clone_kind_with_input_mapper(kind, CloneMode::Fresh)
    }

    fn clone_kind_with_input_mapper(&mut self, kind: &PlanNode, mode: CloneMode) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| self.rewrite_input(input, mode))
    }

    fn rewrite_input(&mut self, input: NodeId, mode: CloneMode) -> NodeId {
        match mode {
            CloneMode::Memoized => self.rewrite_subtree(input),
            CloneMode::Fresh => self.clone_subtree_fresh(input),
        }
    }

    fn push_node(&mut self, kind: PlanNode, facts: crate::plan::PlanFacts) -> NodeId {
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

#[derive(Debug, Clone, Copy)]
enum CloneMode {
    Memoized,
    Fresh,
}

fn replay_candidates(plan: &QueryPlan) -> BTreeMap<NodeId, Vec<NodeId>> {
    let direct_consumers = consumers_by_node(plan);
    let mut probe_consumers = BTreeMap::<NodeId, Vec<NodeId>>::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        if !join_probe_stream_capable(join) {
            continue;
        }
        let probe_input = join.inputs[probe_input_index(join)];
        probe_consumers.entry(probe_input).or_default().push(node.id);
    }

    probe_consumers
        .into_iter()
        .filter(|(producer, consumers)| {
            consumers.len() > 1
                && direct_consumers.get(producer).is_some_and(|all_consumers| all_consumers.len() == consumers.len())
                && stream_replayable_subtree(plan, *producer)
        })
        .collect()
}

fn join_probe_stream_capable(join: &JoinNode) -> bool {
    join.spec.residual_predicate.is_none()
        && match join.spec.join_type {
            JoinType::Inner | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => true,
            JoinType::Left => join.spec.build_side == JoinSide::Right,
            JoinType::Full | JoinType::Cross => false,
        }
}

fn probe_input_index(join: &JoinNode) -> usize {
    match join.spec.build_side {
        JoinSide::Left => 1,
        JoinSide::Right => 0,
    }
}

fn stream_replayable_subtree(plan: &QueryPlan, root: NodeId) -> bool {
    let mut visiting = BTreeSet::new();
    stream_replayable_subtree_inner(plan, root, &mut visiting)
}

fn stream_replayable_subtree_inner(plan: &QueryPlan, node_id: NodeId, visiting: &mut BTreeSet<NodeId>) -> bool {
    if !visiting.insert(node_id) {
        return false;
    }
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    let replayable = match &node.kind {
        PlanNode::Source(_) => true,
        PlanNode::Filter(filter) => stream_replayable_subtree_inner(plan, filter.input, visiting),
        PlanNode::Projection(projection) => stream_replayable_subtree_inner(plan, projection.input, visiting),
        PlanNode::Join(join) => {
            join.spec.join_type == JoinType::Inner
                && join.spec.residual_predicate.is_none()
                && stream_replayable_subtree_inner(plan, join.inputs[0], visiting)
                && stream_replayable_subtree_inner(plan, join.inputs[1], visiting)
        },
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => false,
    };
    visiting.remove(&node_id);
    replayable
}
