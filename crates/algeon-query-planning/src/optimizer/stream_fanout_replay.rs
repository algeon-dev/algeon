//! Stream fanout replay: a private copy of a shared probe input for each join
//! that streams it.
//!
//! The rule matches a node whose only consumers are the probe inputs of two or
//! more joins, where every strategy that can run each join pulls its probe
//! chunk by chunk, and whose subtree is sources, filters, projections and
//! residual-free inner joins. In the rewrite, the subtree is cloned once per
//! consuming join, so each join streams its own copy. As a precondition, the
//! clones compute the same rows, so results are unchanged; the trade is
//! repeated scan and filter work for streaming. A resident resource with
//! several consumers is collected into one frame (`ResourceSharing`), which would
//! hold the whole probe side in device memory and serialize the joins behind it.
//! When any consuming join has a residual, the node is replayed only when its
//! conservative row or byte estimate reaches the large-build threshold of
//! [`KeyedHashJoinStrategyOptions`]; below it, or without an estimate, it stays
//! shared. Every clone keeps the facts of the node it copies. The clones have
//! no image in the rewrite origin, so routes into the replayed subtree are
//! dropped and rediscovered by the late route discovery pass.

use std::collections::{BTreeMap, BTreeSet};

use super::{consumers_by_node, rewrite};
use crate::{
    capability::{KeyedHashJoinStrategyOptions, classify_join, join_probe_streams_chunks},
    error::Result,
    plan::{
        JoinNode, JoinSide, JoinType, NodeId, PlanFacts, PlanNode, QueryNode, QueryPlan, RewrittenNodes,
        estimated_output_bytes, estimated_output_rows,
    },
};

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

    pub(crate) fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate stream-fanout replay evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamFanoutReplayCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub(crate) rule_name: &'static str,
    /// Shared producer node whose subtree is replayed for each join probe.
    pub(crate) fanout_node_id: NodeId,
    /// Number of direct consumers of the shared producer node.
    pub consumer_count: usize,
    pub(crate) join_probe_consumer_node_ids: Vec<NodeId>,
    /// Conservative row estimate of the shared producer node, if known.
    pub(crate) estimated_rows: Option<u64>,
    /// Conservative byte estimate of the shared producer node, if known.
    pub(crate) estimated_bytes: Option<u64>,
    /// Large-build row threshold the estimate is compared with when a
    /// consuming join has a residual.
    pub(crate) large_build_rows: u64,
    /// Large-build byte threshold the estimate is compared with when a
    /// consuming join has a residual.
    pub(crate) large_build_bytes: u64,
    /// Evidence that selected the producer for replay, when it was selected.
    pub(crate) selection_basis: Option<StreamFanoutReplaySelectionBasis>,
    /// True when the returned plan duplicates the producer subtree.
    pub applied: bool,
    /// Why the returned plan keeps the producer shared, when it does.
    pub rejected_reason: Option<StreamFanoutReplayRejectedReason>,
}

/// Evidence that selected a shared producer for replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamFanoutReplaySelectionBasis {
    /// No consuming join has a residual, so the producer is replayed at any size.
    ResidualFreeConsumers,
    /// The conservative row estimate reaches the large-build row threshold.
    ConservativeRows,
    /// The conservative byte estimate reaches the large-build byte threshold.
    ConservativeBytes,
}

impl StreamFanoutReplaySelectionBasis {
    /// Stable diagnostic string for this basis.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ResidualFreeConsumers => "residual_free_consumers",
            Self::ConservativeRows => "conservative_rows",
            Self::ConservativeBytes => "conservative_bytes",
        }
    }
}

/// Reasons a shared producer stays shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFanoutReplayRejectedReason {
    /// The producer was selected but lies outside the subtree the sink reaches,
    /// so no reader copied it.
    NotReachableFromSink,
    /// A consuming join has a residual, and the producer's conservative row and
    /// byte estimates stay below the large-build thresholds.
    ResidualConsumerBelowLargeBuild,
    /// A consuming join has a residual, and the producer has neither a
    /// conservative row nor a conservative byte estimate.
    ResidualConsumerUnestimated,
}

pub(crate) struct StreamFanoutReplayResult {
    pub(crate) plan: QueryPlan,
    pub(crate) change: super::NativeOptimizerChange,
    pub(crate) diagnostics: StreamFanoutReplayDiagnostics,
}

pub(crate) fn optimize_stream_fanout_replay(
    plan: QueryPlan,
    options: &StreamFanoutReplayOptions,
    join_options: &KeyedHashJoinStrategyOptions,
) -> Result<StreamFanoutReplayResult> {
    if !options.enabled {
        return Ok(StreamFanoutReplayResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: StreamFanoutReplayDiagnostics::default(),
        });
    }

    let candidates = replay_candidates(&plan);
    if candidates.is_empty() {
        return Ok(StreamFanoutReplayResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: StreamFanoutReplayDiagnostics::default(),
        });
    }

    let selections = candidates
        .iter()
        .map(|(fanout_node_id, candidate)| {
            let estimated_rows = estimated_output_rows(&plan, *fanout_node_id);
            let estimated_bytes = estimated_output_bytes(&plan, *fanout_node_id);
            let selection =
                replay_selection(estimated_rows, estimated_bytes, candidate.residual_consumer, join_options);
            (*fanout_node_id, (estimated_rows, estimated_bytes, selection))
        })
        .collect::<BTreeMap<_, _>>();
    let fanout_nodes = selections
        .iter()
        .filter_map(|(fanout_node_id, (_, _, selection))| selection.is_ok().then_some(*fanout_node_id))
        .collect::<BTreeSet<_>>();
    let mut rewriter = StreamFanoutReplayRewriter::new(&plan, fanout_nodes);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    let applied_fanouts = rewriter.applied_fanouts;
    let changed = rewriter.changed;
    let rewritten_plan = plan.rewrite(RewrittenNodes::rebuilt(rewriter.nodes, new_root, rewriter.memo))?;

    let mut diagnostics = StreamFanoutReplayDiagnostics::default();
    let direct_consumers = consumers_by_node(&plan);
    for (fanout_node_id, candidate) in candidates {
        diagnostics.considered_count += 1;
        let (estimated_rows, estimated_bytes, selection) = selections[&fanout_node_id];
        let applied = applied_fanouts.contains(&fanout_node_id);
        if applied {
            diagnostics.applied_count += 1;
        }
        let rejected_reason = match selection {
            Ok(_) if applied => None,
            Ok(_) => Some(StreamFanoutReplayRejectedReason::NotReachableFromSink),
            Err(reason) => Some(reason),
        };
        diagnostics.candidates.push(StreamFanoutReplayCandidateDiagnostics {
            rule_name: STREAM_FANOUT_REPLAY_RULE_NAME,
            fanout_node_id,
            consumer_count: direct_consumers[&fanout_node_id].len(),
            join_probe_consumer_node_ids: candidate.join_probe_consumer_node_ids,
            estimated_rows,
            estimated_bytes,
            large_build_rows: join_options.large_keyed_hash_build_row_threshold(),
            large_build_bytes: join_options.large_keyed_hash_build_bytes_threshold(),
            selection_basis: selection.ok(),
            applied,
            rejected_reason,
        });
    }

    let change = if changed { super::NativeOptimizerChange::Changed } else { super::NativeOptimizerChange::Unchanged };
    Ok(StreamFanoutReplayResult { plan: rewritten_plan, change, diagnostics })
}

/// Rebuilds the plan with each fanout node's subtree copied per reader.
///
/// Nodes outside a fanout subtree are memoized and keep one image; a fanout
/// node and everything below it are cloned fresh on every reach, and those
/// clones are kept out of `memo` so the rewrite origin never maps one source
/// node to two images.
struct StreamFanoutReplayRewriter<'a> {
    original: &'a QueryPlan,
    fanout_nodes: BTreeSet<NodeId>,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
    /// Fanout nodes at least one reader reached; a candidate outside the
    /// sink-reachable subtree is never reached and reports as not applied.
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

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

#[derive(Debug, Clone, Copy)]
enum CloneMode {
    Memoized,
    Fresh,
}

/// A shared node whose consumers all stream it as their probe input.
#[derive(Default)]
struct ReplayCandidate {
    join_probe_consumer_node_ids: Vec<NodeId>,
    /// Whether any consuming join has a residual predicate.
    residual_consumer: bool,
}

/// Fanout nodes the rule considers, each with the joins that probe it. A node
/// with any consumer other than a streaming probe is excluded, because that
/// consumer would still need the shared frame.
fn replay_candidates(plan: &QueryPlan) -> BTreeMap<NodeId, ReplayCandidate> {
    let direct_consumers = consumers_by_node(plan);
    let mut probe_consumers = BTreeMap::<NodeId, ReplayCandidate>::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        if !join_probe_stream_capable(node.id, join) {
            continue;
        }
        let probe_input = join.inputs[probe_input_index(join)];
        let candidate = probe_consumers.entry(probe_input).or_default();
        candidate.join_probe_consumer_node_ids.push(node.id);
        candidate.residual_consumer |= join.spec.residual_predicate.is_some();
    }

    probe_consumers
        .into_iter()
        .filter(|(producer, candidate)| {
            let consumers = &candidate.join_probe_consumer_node_ids;
            consumers.len() > 1
                && direct_consumers.get(producer).is_some_and(|all_consumers| all_consumers.len() == consumers.len())
                && stream_replayable_subtree(plan, *producer)
        })
        .collect()
}

/// Whether the join's probe pulls its input chunk by chunk under every
/// strategy that can execute it.
fn join_probe_stream_capable(node_id: NodeId, join: &JoinNode) -> bool {
    classify_join(node_id, join)
        .is_ok_and(|form| form.strategies().iter().all(|strategy| join_probe_streams_chunks(join, form, *strategy)))
}

/// Decides whether a candidate is replayed.
///
/// A residual consumer keeps a small producer shared: replaying the shared
/// lineitem probe of TPC-H SF10 Q21's residual semi and anti joins read 4656
/// row groups and took about 450 ms, compared with 3189 row groups and 314 ms
/// without replay. A shared producer is collected into one device frame that
/// stays resident until its last consumer finishes, and the concatenation
/// that builds it holds the chunks and the frame at once. The large-build
/// threshold bounds the same single-frame residency for a hash build, so above
/// it the rule trades rescans for that memory. Q21's filtered lineitem at
/// SF1000 is about 3.8 billion rows, beyond cuDF's 2^31 - 1 row limit for one
/// frame.
fn replay_selection(
    estimated_rows: Option<u64>,
    estimated_bytes: Option<u64>,
    residual_consumer: bool,
    join_options: &KeyedHashJoinStrategyOptions,
) -> std::result::Result<StreamFanoutReplaySelectionBasis, StreamFanoutReplayRejectedReason> {
    if !residual_consumer {
        return Ok(StreamFanoutReplaySelectionBasis::ResidualFreeConsumers);
    }
    if estimated_rows.is_some_and(|rows| join_options.rows_reach_large_build(rows)) {
        return Ok(StreamFanoutReplaySelectionBasis::ConservativeRows);
    }
    if estimated_bytes.is_some_and(|bytes| join_options.bytes_reach_large_build(bytes)) {
        return Ok(StreamFanoutReplaySelectionBasis::ConservativeBytes);
    }
    if estimated_rows.is_none() && estimated_bytes.is_none() {
        return Err(StreamFanoutReplayRejectedReason::ResidualConsumerUnestimated);
    }
    Err(StreamFanoutReplayRejectedReason::ResidualConsumerBelowLargeBuild)
}

fn probe_input_index(join: &JoinNode) -> usize {
    match join.spec.build_side {
        JoinSide::Left => 1,
        JoinSide::Right => 0,
    }
}

/// Whether cloning the subtree only repeats streaming work. An aggregate,
/// sort, limit, union or GPU function below the fanout would be recomputed as
/// a whole per copy, so such subtrees stay shared.
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
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => false,
    };
    visiting.remove(&node_id);
    replayable
}
