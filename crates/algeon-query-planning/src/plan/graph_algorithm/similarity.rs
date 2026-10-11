//! Explicit vertex-pair similarity parameter contracts.

/// Relation-backed aligned vertex pairs consumed by a similarity algorithm.
///
/// The relation remains a normal native-plan input. It is not a catalog name or
/// an executor-local lookup. A graph algorithm therefore records it as an
/// explicit dependency alongside its edge-list input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphVertexPairInput {
    /// Position of the relation among the algorithm node inputs.
    pub input: usize,
    /// First vertex-id column in the candidate-pair relation.
    pub first_column: String,
    /// Second vertex-id column in the candidate-pair relation.
    pub second_column: String,
}

impl GraphVertexPairInput {
    /// Creates an explicit candidate-pair relation binding.
    pub fn new(input: usize, first_column: impl Into<String>, second_column: impl Into<String>) -> Self {
        Self { input, first_column: first_column.into(), second_column: second_column.into() }
    }
}

/// Shared parameters for explicit-pair graph similarity algorithms.
///
/// The metric is carried by [`crate::plan::GraphAlgorithm`], so the four
/// similarity metrics share this one payload type. Whether an execution consumes
/// edge weights is likewise derived from the edge weight binding; a second
/// boolean here could contradict the graph construction contract.
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
    pub(crate) fn side_input_ordinals(&self) -> impl Iterator<Item = usize> + '_ {
        std::iter::once(self.vertex_pairs.input)
    }
}

impl Default for GraphSimilarityParams {
    fn default() -> Self {
        // This placeholder is used only by registry metadata projections. A
        // real graph plan must bind an existing pair-relation node.
        Self::new(GraphVertexPairInput::new(1, "first", "second"))
    }
}
