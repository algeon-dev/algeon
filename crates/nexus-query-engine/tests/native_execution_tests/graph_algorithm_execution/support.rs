static GRAPH_SEED_CHUNK_RETRY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn int32_graph_spec(
    algorithm: GraphAlgorithm,
    edge_columns: GraphEdgeColumnBindings,
    graph_options: GraphConstructionOptions,
    params: GraphAlgorithmParams,
) -> GraphAlgorithmSpec {
    graph_spec_for_vertex_type(algorithm, edge_columns, graph_options, params, &DataType::Int32)
}

fn graph_spec_for_vertex_type(
    algorithm: GraphAlgorithm,
    edge_columns: GraphEdgeColumnBindings,
    graph_options: GraphConstructionOptions,
    params: GraphAlgorithmParams,
    vertex_type: &DataType,
) -> GraphAlgorithmSpec {
    let output_schema =
        nexus_query_engine::plan::graph_result_schema_for_vertex_type(algorithm, &params, vertex_type);
    GraphAlgorithmSpec::new(algorithm, edge_columns, graph_options, params, output_schema)
}

fn int32_stable_graph_result_schema(algorithm: GraphAlgorithm) -> Arc<Schema> {
    stable_graph_result_schema_for_vertex_type(algorithm, &DataType::Int32)
}

fn stable_graph_result_schema_for_vertex_type(algorithm: GraphAlgorithm, vertex_type: &DataType) -> Arc<Schema> {
    nexus_query_engine::plan::graph_result_schema_for_vertex_type(
        algorithm,
        &algorithm.definition().default_params(),
        vertex_type,
    )
}

struct GraphVertexValues(Vec<i64>);

impl GraphVertexValues {
    fn value(&self, row: usize) -> i64 {
        self.0[row]
    }
}

fn graph_vertex_values(batch: &RecordBatch, column: usize) -> GraphVertexValues {
    let values = match batch.column(column).data_type() {
        DataType::Int32 => int32_array(batch, column).values().iter().map(|value| i64::from(*value)).collect(),
        DataType::Int64 => int64_array(batch, column).values().to_vec(),
        data_type => panic!("expected graph vertex column, got {data_type}"),
    };
    GraphVertexValues(values)
}

fn gpu_resident_source(input_name: &str, schema: Arc<Schema>, batch: &RecordBatch) -> GpuResidentSource {
    GpuResidentSource::new(input_name, schema).with_row_count_upper_bound(
        u64::try_from(batch.num_rows()).expect("native graph fixture row count must fit u64"),
    )
}

fn assert_vertex_betweenness_batches_close(expected: &RecordBatch, actual: &RecordBatch) {
    let expected = vertex_betweenness_scores(expected);
    let actual = vertex_betweenness_scores(actual);
    assert_eq!(actual.len(), expected.len());
    for (vertex, expected_value) in expected {
        let actual_value = actual.get(&vertex).unwrap_or_else(|| panic!("missing vertex {vertex}"));
        assert!(
            (actual_value - expected_value).abs() <= 1.0e-6,
            "vertex {vertex}: expected {expected_value}, got {actual_value}"
        );
    }
}

fn vertex_betweenness_scores(batch: &RecordBatch) -> std::collections::BTreeMap<i64, f64> {
    let vertices = graph_vertex_values(batch, 0);
    let values = float64_array(batch, 1);
    (0..batch.num_rows()).map(|row| (vertices.value(row), values.value(row))).collect()
}

fn assert_edge_betweenness_batches_close(expected: &RecordBatch, actual: &RecordBatch) {
    let expected = edge_betweenness_scores(expected);
    let actual = edge_betweenness_scores(actual);
    assert_eq!(actual.len(), expected.len());
    for (edge, expected_value) in expected {
        let actual_value = actual.get(&edge).unwrap_or_else(|| panic!("missing edge {edge:?}"));
        assert!(
            (actual_value - expected_value).abs() <= 1.0e-6,
            "edge {edge:?}: expected {expected_value}, got {actual_value}"
        );
    }
}

fn edge_betweenness_scores(batch: &RecordBatch) -> std::collections::BTreeMap<(i64, i64), f64> {
    let sources = graph_vertex_values(batch, 0);
    let destinations = graph_vertex_values(batch, 1);
    let scores = float64_array(batch, 2);
    (0..batch.num_rows()).map(|row| ((sources.value(row), destinations.value(row)), scores.value(row))).collect()
}
