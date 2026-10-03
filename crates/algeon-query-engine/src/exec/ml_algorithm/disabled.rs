use query_planning::{
    error::{QueryError, QueryErrorCode},
    plan::{MlAlgorithmNode, NodeId},
};

pub(in crate::exec) fn ml_algorithm_execution_unavailable(node_id: NodeId, node: &MlAlgorithmNode) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedOperator,
        "query_engine.ml_algorithm.execute",
        "cuML execution requires the cuml Cargo feature",
    )
    .with_fact("node_id", node_id)
    .with_fact("operator", "ml_algorithm")
    .with_fact("algorithm", node.spec.algorithm.as_str())
    .with_fact("reason", "required_feature_disabled")
    .with_fact("required_feature", "cuml")
    .with_fact("host_fallback", "disabled")
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::DataType;
    use query_planning::plan::{
        MlAlgorithm, MlAlgorithmParams, MlAlgorithmSpec, MlRelationBinding, MlTsvdParams, VectorRowIdBinding,
        vector_pca_output_schema,
    };

    use super::*;

    #[test]
    fn unavailable_ml_feature_returns_structured_facts() {
        let input = NodeId::from_index_for_test(0);
        let spec = MlAlgorithmSpec::new(
            MlAlgorithm::Tsvd,
            MlAlgorithmParams::Tsvd(MlTsvdParams::new(
                MlRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x", "y"]),
                1,
            )),
            vector_pca_output_schema(DataType::Int64, 1),
        );
        let node = MlAlgorithmNode { inputs: vec![input], spec };
        let error = ml_algorithm_execution_unavailable(NodeId::from_index_for_test(1), &node);
        assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
        assert_eq!(error.fact_value("reason"), Some("required_feature_disabled"));
        assert_eq!(error.fact_value("required_feature"), Some("cuml"));
        assert_eq!(error.fact_value("host_fallback"), Some("disabled"));
    }
}
