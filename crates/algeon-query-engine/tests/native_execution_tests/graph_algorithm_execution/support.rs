use query_planning::plan::graph_result_schema_for_vertex_type;

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
    let output_schema = graph_result_schema_for_vertex_type(algorithm, &params, vertex_type);
    GraphAlgorithmSpec::new(algorithm, edge_columns, graph_options, params, output_schema)
}

fn int32_stable_graph_result_schema(algorithm: GraphAlgorithm) -> Arc<Schema> {
    stable_graph_result_schema_for_vertex_type(algorithm, &DataType::Int32)
}

fn stable_graph_result_schema_for_vertex_type(algorithm: GraphAlgorithm, vertex_type: &DataType) -> Arc<Schema> {
    graph_result_schema_for_vertex_type(
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
