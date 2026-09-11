use super::super::*;

/// Reject a graph node before any input lookup or graph preparation occurs.
///
/// Callers on the feature-off path must not first materialize a GPU input
/// merely to learn that the optional cuGraph implementation is unavailable.
pub(in crate::exec) fn graph_algorithm_feature_disabled(
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedOperator,
        "query_engine.graph_algorithm.execute",
        "cuGraph graph algorithm execution requires the cugraph Cargo feature",
    )
    .with_fact("node_id", node_id)
    .with_fact("algorithm", node.spec.algorithm.as_str())
    .with_fact("reason", "required_feature_disabled")
    .with_fact("required_feature", "cugraph")
    .with_fact("host_fallback", "disabled")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionOptions, GraphEdgeColumnBindings,
        GraphPageRankParams,
    };

    #[test]
    fn unavailable_error_is_structured_before_input_lookup() {
        let input = NodeId::from_index_for_internal_use(0);
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));
        let spec = GraphAlgorithmSpec::new(
            GraphAlgorithm::PageRank,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
            schema,
        );
        let node = crate::plan::GraphAlgorithmNode { input, inputs: vec![input], spec };
        let error = graph_algorithm_feature_disabled(NodeId::from_index_for_internal_use(1), &node);

        assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
        assert_eq!(error.operation(), "query_engine.graph_algorithm.execute");
        assert_eq!(error.fact_value("reason"), Some("required_feature_disabled"));
        assert_eq!(error.fact_value("required_feature"), Some("cugraph"));
        assert_eq!(error.fact_value("host_fallback"), Some("disabled"));
    }
}
