//! Node id remapping for IR rewrites that renumber nodes.
//!
//! Compaction drops unreachable nodes and reindexes the survivors, so every
//! side structure keyed by [`NodeId`](super::NodeId) must follow the same
//! mapping. [`NodeRemap`] is that mapping: advisory bindings and
//! dynamic-filter routes translate through it, and entries whose node is gone
//! are dropped and are not re-hung on whatever node reuses the old index.

use std::collections::BTreeMap;

use super::{NodeId, PlanNode, QueryNode};

/// Old-to-new node id mapping produced by one renumbering rewrite.
///
/// Only kept nodes have an image; a missing image means the node was removed
/// and any side data bound to it must be dropped.
#[derive(Debug, Clone)]
pub(crate) struct NodeRemap {
    forward: BTreeMap<NodeId, NodeId>,
    /// Inverse of `forward`. Well defined only because `new` requires an
    /// injective image; a merge that maps two sources to one image must be
    /// filtered out by the caller before construction.
    backward: BTreeMap<NodeId, NodeId>,
}

impl NodeRemap {
    /// Builds the two-way mapping from an old-to-new image. The image must be
    /// injective: each kept node reuses exactly one new index.
    pub(crate) fn new(forward: BTreeMap<NodeId, NodeId>) -> Self {
        let backward = forward.iter().map(|(old, new)| (*new, *old)).collect();
        Self { forward, backward }
    }

    /// Returns the new id of a kept node, or `None` when it was removed.
    pub(crate) fn remap(&self, old: NodeId) -> Option<NodeId> {
        self.forward.get(&old).copied()
    }

    /// Returns the old id a new node was renumbered from, or `None` when the
    /// new node has no counterpart in the source plan.
    pub(crate) fn old_id(&self, new: NodeId) -> Option<NodeId> {
        self.backward.get(&new).copied()
    }

    /// Maps every id below `len` to itself.
    pub(crate) fn identity(len: usize) -> Self {
        Self::new((0..len).map(|index| (NodeId(index), NodeId(index))).collect())
    }

    /// Composes this mapping with the one applied after it. A source node
    /// whose image `next` drops has no image in the result, so one lookup
    /// through the composition answers whether a node survived both steps.
    pub(crate) fn then(&self, next: &Self) -> Self {
        Self::new(self.forward.iter().filter_map(|(old, mid)| Some((*old, next.remap(*mid)?))).collect())
    }

    /// Rewrites a source-plan node's input edges through the mapping.
    ///
    /// Returns `None` when any referenced input has no image, so callers fail
    /// closed and do not compare against a partially translated node.
    pub(crate) fn try_translate_inputs(&self, kind: &PlanNode) -> Option<PlanNode> {
        let mut missing = false;
        let translated = clone_plan_node_with_inputs(kind, |input| {
            self.forward.get(&input).copied().unwrap_or_else(|| {
                missing = true;
                input
            })
        });
        (!missing).then_some(translated)
    }
}

pub(crate) fn clone_plan_node_with_inputs(
    kind: &PlanNode,
    mut rewrite_input: impl FnMut(NodeId) -> NodeId,
) -> PlanNode {
    match kind {
        PlanNode::Source(source) => PlanNode::Source(source.clone()),
        PlanNode::Filter(filter) => {
            let mut filter = filter.clone();
            filter.input = rewrite_input(filter.input);
            PlanNode::Filter(filter)
        },
        PlanNode::Projection(projection) => {
            let mut projection = projection.clone();
            projection.input = rewrite_input(projection.input);
            PlanNode::Projection(projection)
        },
        PlanNode::Aggregate(aggregate) => {
            let mut aggregate = aggregate.clone();
            aggregate.input = rewrite_input(aggregate.input);
            PlanNode::Aggregate(aggregate)
        },
        PlanNode::Window(window) => {
            let mut window = window.clone();
            window.input = rewrite_input(window.input);
            PlanNode::Window(window)
        },
        PlanNode::Sort(sort) => {
            let mut sort = sort.clone();
            sort.input = rewrite_input(sort.input);
            PlanNode::Sort(sort)
        },
        PlanNode::Limit(limit) => {
            let mut limit = limit.clone();
            limit.input = rewrite_input(limit.input);
            PlanNode::Limit(limit)
        },
        PlanNode::Join(join) => {
            let mut join = join.clone();
            join.inputs = [rewrite_input(join.inputs[0]), rewrite_input(join.inputs[1])];
            PlanNode::Join(join)
        },
        PlanNode::Union(union) => {
            let mut union = union.clone();
            union.inputs = union.inputs.iter().map(|input| rewrite_input(*input)).collect();
            PlanNode::Union(union)
        },
        PlanNode::EdgeNormalize(edge_normalize) => {
            let mut edge_normalize = edge_normalize.clone();
            edge_normalize.input = rewrite_input(edge_normalize.input);
            PlanNode::EdgeNormalize(edge_normalize)
        },
        PlanNode::GraphAlgorithm(graph) => {
            let mut graph = graph.clone();
            graph.inputs = graph.inputs.iter().map(|input| rewrite_input(*input)).collect();
            PlanNode::GraphAlgorithm(graph)
        },
        PlanNode::VectorAlgorithm(vector) => {
            let mut vector = vector.clone();
            vector.inputs = vector.inputs.iter().map(|input| rewrite_input(*input)).collect();
            PlanNode::VectorAlgorithm(vector)
        },
        PlanNode::MlAlgorithm(ml) => {
            let mut ml = ml.clone();
            ml.inputs = ml.inputs.iter().map(|input| rewrite_input(*input)).collect();
            PlanNode::MlAlgorithm(ml)
        },
    }
}

/// Drops the nodes `root` cannot reach and renumbers the rest in their
/// current order.
///
/// Keeping the relative order is what preserves the input-before-consumer
/// invariant without a topological sort: `check_node_order` has already
/// verified it for the uncompacted list. The caller must have run that check,
/// because this function indexes `nodes` by id and panics on a root or input
/// outside the list.
pub(crate) fn compact_reachable_nodes(nodes: Vec<QueryNode>, root: NodeId) -> (Vec<QueryNode>, NodeId, NodeRemap) {
    let mut reachable = vec![false; nodes.len()];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if reachable[id.index()] {
            continue;
        }
        reachable[id.index()] = true;
        stack.extend(nodes[id.index()].input_ids().iter().copied());
    }
    let mut forward = BTreeMap::new();
    for node in nodes.iter().filter(|node| reachable[node.id.index()]) {
        forward.insert(node.id, NodeId(forward.len()));
    }
    let new_root = *forward.get(&root).expect("rewritten root must be reachable");
    let compacted = nodes
        .into_iter()
        .filter(|node| reachable[node.id.index()])
        .map(|node| {
            let id = *forward.get(&node.id).expect("reachable node must have a remapped id");
            QueryNode { id, kind: remap_plan_node_inputs(node.kind, &forward), facts: node.facts }
        })
        .collect::<Vec<_>>();
    (compacted, new_root, NodeRemap::new(forward))
}

pub(crate) fn remap_plan_node_inputs(kind: PlanNode, remap: &BTreeMap<NodeId, NodeId>) -> PlanNode {
    clone_plan_node_with_inputs(&kind, |input| remap_node_id(input, remap))
}

fn remap_node_id(id: NodeId, remap: &BTreeMap<NodeId, NodeId>) -> NodeId {
    *remap.get(&id).expect("reachable input node must have a remapped id")
}
