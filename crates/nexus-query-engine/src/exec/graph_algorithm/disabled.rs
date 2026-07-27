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
        "nexus_query_engine.graph_algorithm.execute",
        "cuGraph graph algorithm execution requires the cugraph Cargo feature",
    )
    .with_fact("node_id", node_id)
    .with_fact("algorithm", node.spec.algorithm.as_str())
    .with_fact("reason", "required_feature_disabled")
    .with_fact("required_feature", "cugraph")
    .with_fact("host_fallback", "disabled")
}
