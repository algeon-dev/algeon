//! Explicit vertex-pair similarity parameter contracts.

use super::NodeId;

/// Relation-backed aligned vertex pairs consumed by a similarity algorithm.
///
/// The relation remains a normal native-plan input.  It is neither a catalog
/// name nor an executor-local lookup, so a graph algorithm records it as an
/// explicit dependency alongside its edge-list input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphVertexPairInput {
    /// Native plan node producing the candidate-pair relation.
    pub input: NodeId,
    /// First vertex-id column in the candidate-pair relation.
    pub first_column: String,
    /// Second vertex-id column in the candidate-pair relation.
    pub second_column: String,
}

impl GraphVertexPairInput {
    /// Creates an explicit candidate-pair relation binding.
    pub fn new(input: NodeId, first_column: impl Into<String>, second_column: impl Into<String>) -> Self {
        Self { input, first_column: first_column.into(), second_column: second_column.into() }
    }
}

/// Shared parameters for explicit-pair graph similarity algorithms.
///
/// The metric is deliberately represented by [`crate::plan::GraphAlgorithm`], rather than
/// by four duplicate parameter payloads.  Whether an execution consumes edge
/// weights is likewise derived from the edge weight binding; a second boolean
/// here could contradict the graph construction contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphSimilarityParams {
    /// Required candidate-pair relation.
    pub vertex_pairs: GraphVertexPairInput,
}

impl GraphSimilarityParams {
    /// Creates similarity parameters for an explicit candidate-pair relation.
    pub fn new(vertex_pairs: GraphVertexPairInput) -> Self {
        Self { vertex_pairs }
    }

    /// Visits the required pair-relation input in stable execution order.
    pub fn side_input_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        std::iter::once(self.vertex_pairs.input)
    }
}

impl Default for GraphSimilarityParams {
    fn default() -> Self {
        // This placeholder is used only by registry metadata projections. A
        // real graph plan must bind an existing pair-relation node.
        Self::new(GraphVertexPairInput::new(NodeId::from_index_for_internal_use(0), "first", "second"))
    }
}
