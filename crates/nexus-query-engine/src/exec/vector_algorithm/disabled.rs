use super::super::*;

pub(in crate::exec) fn execute_vector_algorithm(
    node_id: NodeId,
    node: &crate::plan::VectorAlgorithmNode,
) -> Result<GpuDataFrame> {
    Err(vector_algorithm_execution_unavailable(node_id, node))
}

pub(in crate::exec) fn vector_algorithm_execution_unavailable(
    node_id: NodeId,
    node: &crate::plan::VectorAlgorithmNode,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedOperator,
        "nexus_query_engine.vector_algorithm.execute",
        "cuVS vector algorithm execution requires the cuvs Cargo feature",
    )
    .with_fact("node_id", node_id)
    .with_fact("operator", "vector_algorithm")
    .with_fact("algorithm", node.spec.algorithm.as_str())
    .with_fact("reason", "cuvs_feature_disabled")
    .with_fact("required_feature", "cuvs")
    .with_fact("host_fallback", "disabled")
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::DataType;

    use super::*;
    use crate::plan::{
        VectorAlgorithm, VectorAlgorithmParams, VectorAlgorithmSpec, VectorPcaParams, VectorRelationBinding,
        VectorRowIdBinding, vector_pca_output_schema,
    };

    #[test]
    fn unavailable_error_is_structured_without_constructing_gpu_inputs() {
        let input = NodeId::from_index_for_internal_use(0);
        let binding = VectorRelationBinding::new(input, VectorRowIdBinding::Column("id".into()), ["x"]);
        let spec = VectorAlgorithmSpec::new(
            VectorAlgorithm::Pca,
            VectorAlgorithmParams::Pca(VectorPcaParams::new(binding, 1)),
            vector_pca_output_schema(DataType::Int64, 1),
        );
        let node = crate::plan::VectorAlgorithmNode { inputs: vec![input], spec };

        let error = vector_algorithm_execution_unavailable(NodeId::from_index_for_internal_use(1), &node);

        assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
        assert_eq!(error.operation(), "nexus_query_engine.vector_algorithm.execute");
        assert_eq!(error.fact_value("operator"), Some("vector_algorithm"));
        assert_eq!(error.fact_value("algorithm"), Some("pca"));
        assert_eq!(error.fact_value("host_fallback"), Some("disabled"));
        assert_eq!(error.fact_value("required_feature"), Some("cuvs"));
    }
}
