use super::super::*;

pub(in crate::exec) fn execute_graph_algorithm(
    _context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: std::collections::BTreeMap<NodeId, GpuDataFrame>,
    _metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows();
    let input_columns = input.num_columns();
    let side_input_count = side_inputs.len();
    Err(QueryError::new(
        QueryErrorCode::UnsupportedOperator,
        "nexus_query_engine.graph_algorithm.execute",
        "cuGraph graph algorithm execution requires the cugraph Cargo feature",
    )
    .with_fact("node_id", node_id)
    .with_fact("algorithm", node.spec.algorithm.as_str())
    .with_fact("input_rows", input_rows)
    .with_fact("input_columns", input_columns)
    .with_fact("side_input_count", side_input_count)
    .with_fact("required_feature", "cugraph"))
}
