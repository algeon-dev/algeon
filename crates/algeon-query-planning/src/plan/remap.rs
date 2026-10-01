//! Node id remapping for IR rewrites that renumber nodes.
//!
//! Compaction drops unreachable nodes and reindexes the survivors, so every
//! side structure keyed by [`NodeId`](super::NodeId) must follow the same
//! mapping. [`NodeRemap`] is that mapping: advisory bindings and
//! dynamic-filter routes translate through it, and entries whose node is gone
//! are dropped instead of re-hung on whatever node reuses the old index.

use std::collections::BTreeMap;

use super::{NodeId, PlanNode, QueryNode};

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

    /// Maps every id below `len` to itself.
    pub(crate) fn identity(len: usize) -> Self {
        Self::new((0..len).map(|index| (NodeId(index), NodeId(index))).collect())
    }

    /// Composes this mapping with the one applied after it.
    pub(crate) fn then(&self, next: &Self) -> Self {
        Self::new(self.forward.iter().filter_map(|(old, mid)| Some((*old, next.remap(*mid)?))).collect())
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
            graph.spec.params.remap_side_inputs(|input| {
                *input_remap.get(&input).expect("graph side-input binding must be present in graph input list")
            });
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

/// Drops the nodes `root` cannot reach and renumbers the rest in their
/// current order.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{
        GraphAlgorithm, GraphAlgorithmNode, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBetweennessCentralityParams,
        GraphBetweennessSeeds, GraphConstructionOptions, GraphEdgeBetweennessCentralityParams, GraphEdgeColumnBindings,
        GraphForceAtlas2InitialPositionsInput, GraphForceAtlas2Params, GraphForceAtlas2VertexAttributesInput,
        GraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams, GraphPredicateInput, GraphSimilarityParams,
        GraphSsspParams, GraphSsspSource, GraphVertexPairInput,
    };

    fn graph_node(algorithm: GraphAlgorithm, params: GraphAlgorithmParams) -> PlanNode {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            algorithm,
            GraphEdgeColumnBindings::new("src", "dst", Option::<String>::None),
            GraphConstructionOptions::default(),
            params,
        );
        let side_inputs = spec.params.side_input_node_ids().collect::<Vec<_>>();
        let inputs = std::iter::once(NodeId(0)).chain(side_inputs).collect();
        PlanNode::GraphAlgorithm(GraphAlgorithmNode { input: NodeId(0), inputs, spec })
    }

    fn side_input_cases() -> Vec<(&'static str, PlanNode)> {
        let side = |index| GraphPredicateInput::new(NodeId(index), "vertex");
        vec![
            (
                "personalized_pagerank",
                graph_node(
                    GraphAlgorithm::PersonalizedPageRank,
                    GraphAlgorithmParams::PersonalizedPageRank(GraphPersonalizedPageRankParams {
                        personalization: GraphPageRankPersonalizationInput::new(NodeId(1), "vertex", "value"),
                        ..GraphPersonalizedPageRankParams::default()
                    }),
                ),
            ),
            (
                "similarity",
                graph_node(
                    GraphAlgorithm::Jaccard,
                    GraphAlgorithmParams::Similarity(GraphSimilarityParams::new(GraphVertexPairInput::new(
                        NodeId(1),
                        "first",
                        "second",
                    ))),
                ),
            ),
            (
                "sssp",
                graph_node(
                    GraphAlgorithm::Sssp,
                    GraphAlgorithmParams::Sssp(GraphSsspParams {
                        source_vertex: GraphSsspSource::Input(side(1)),
                        cutoff: None,
                    }),
                ),
            ),
            (
                "betweenness",
                graph_node(
                    GraphAlgorithm::BetweennessCentrality,
                    GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
                        seeds: Some(GraphBetweennessSeeds::Input { input: side(1), row_count_upper_bound: 4 }),
                        ..GraphBetweennessCentralityParams::default()
                    }),
                ),
            ),
            (
                "edge_betweenness",
                graph_node(
                    GraphAlgorithm::EdgeBetweennessCentrality,
                    GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams {
                        seeds: Some(GraphBetweennessSeeds::Input { input: side(1), row_count_upper_bound: 4 }),
                        ..GraphEdgeBetweennessCentralityParams::default()
                    }),
                ),
            ),
            (
                "force_atlas2",
                graph_node(
                    GraphAlgorithm::ForceAtlas2,
                    GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params {
                        vertex_attributes: Some(GraphForceAtlas2VertexAttributesInput::new(
                            NodeId(1),
                            "vertex",
                            Some("radius"),
                            Option::<String>::None,
                            Option::<String>::None,
                        )),
                        initial_positions: Some(GraphForceAtlas2InitialPositionsInput::new(
                            NodeId(2),
                            "vertex",
                            "x",
                            "y",
                        )),
                        ..GraphForceAtlas2Params::default()
                    }),
                ),
            ),
        ]
    }

    /// Renumbering a graph node moves its side-input bindings with its input
    /// list, so a side input never keeps an id of the pre-rewrite plan.
    #[test]
    fn renumbering_moves_every_graph_side_input_with_the_input_list() {
        for (label, node) in side_input_cases() {
            let renumbered = clone_plan_node_with_inputs(&node, |input| NodeId(input.index() + 10));
            let PlanNode::GraphAlgorithm(graph) = &renumbered else { unreachable!("{label} stays a graph node") };
            let side_inputs = graph.spec.params.side_input_node_ids().collect::<Vec<_>>();
            assert!(!side_inputs.is_empty(), "{label} has side inputs");
            assert_eq!(side_inputs, graph.inputs[1..], "{label}");
            assert!(side_inputs.iter().all(|input| input.index() >= 10), "{label}: {side_inputs:?}");
        }
    }
}
