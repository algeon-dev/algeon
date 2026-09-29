//! Node id remapping for IR rewrites that renumber nodes.
//!
//! Compaction drops unreachable nodes and reindexes the survivors, so every
//! side structure keyed by [`NodeId`](super::NodeId) must follow the same
//! mapping. [`NodeRemap`] is that mapping: advisory bindings and
//! dynamic-filter routes translate through it, and entries whose node is gone
//! are dropped instead of re-hung on whatever node reuses the old index.

use std::collections::BTreeMap;

use super::{GraphAlgorithmParams, GraphAlgorithmSpec, NodeId, PlanNode};

/// Old-to-new node id mapping produced by one renumbering rewrite.
///
/// Only kept nodes have an image; a missing image means the node was removed
/// and any side data bound to it must be dropped.
#[derive(Debug, Clone)]
pub(crate) struct NodeRemap {
    forward: BTreeMap<NodeId, NodeId>,
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

    /// Rewrites a source-plan node's input edges through the mapping.
    ///
    /// Returns `None` when any referenced input has no image, so callers fail
    /// closed instead of comparing against a partially translated node.
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
            let old_inputs = graph.inputs.clone();
            if old_inputs.is_empty() {
                graph.input = rewrite_input(graph.input);
                graph.inputs = vec![graph.input];
                return PlanNode::GraphAlgorithm(graph);
            }

            let new_inputs = old_inputs.iter().map(|input| rewrite_input(*input)).collect::<Vec<_>>();
            let input_remap = old_inputs.iter().copied().zip(new_inputs.iter().copied()).collect::<BTreeMap<_, _>>();
            graph.input = new_inputs[0];
            graph.inputs = new_inputs;
            remap_graph_algorithm_spec_side_inputs(&mut graph.spec, &input_remap);
            PlanNode::GraphAlgorithm(graph)
        },
        PlanNode::VectorAlgorithm(vector) => {
            let mut vector = vector.clone();
            vector.spec.params.remap_relation_inputs(&mut rewrite_input);
            vector.inputs = vector.spec.params.relation_bindings().map(|binding| binding.input).collect();
            PlanNode::VectorAlgorithm(vector)
        },
        PlanNode::MlAlgorithm(ml) => {
            let mut ml = ml.clone();
            ml.spec.params.remap_relation_inputs(&mut rewrite_input);
            ml.inputs = ml.spec.params.relation_bindings().map(|binding| binding.input).collect();
            PlanNode::MlAlgorithm(ml)
        },
    }
}

pub(crate) fn remap_plan_node_inputs(kind: PlanNode, remap: &BTreeMap<NodeId, NodeId>) -> PlanNode {
    clone_plan_node_with_inputs(&kind, |input| remap_node_id(input, remap))
}

fn remap_node_id(id: NodeId, remap: &BTreeMap<NodeId, NodeId>) -> NodeId {
    *remap.get(&id).expect("reachable input node must have a remapped id")
}

fn remap_graph_algorithm_spec_side_inputs(spec: &mut GraphAlgorithmSpec, input_remap: &BTreeMap<NodeId, NodeId>) {
    let GraphAlgorithmParams::Bfs(params) = &mut spec.params else {
        return;
    };
    if let Some(input) = &mut params.source_vertices {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
    if let Some(input) = &mut params.predicates.include_vertices {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
    if let Some(input) = &mut params.predicates.exclude_vertices {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
    if let Some(input) = &mut params.predicates.target_vertices {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
    if let Some(input) = &mut params.predicates.include_edge_ids {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
    if let Some(input) = &mut params.predicates.include_edges {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
}

fn remap_graph_side_input(input: NodeId, input_remap: &BTreeMap<NodeId, NodeId>) -> NodeId {
    *input_remap.get(&input).expect("graph side-input binding must be present in graph input list")
}
