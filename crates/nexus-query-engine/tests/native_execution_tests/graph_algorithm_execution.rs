use super::*;

#[test]
fn bfs_drops_null_edge_rows_before_graph_construction() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let edge_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Int32, true), Field::new("dst", DataType::Int32, true)]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(0), Some(1), None, Some(2), Some(2)])),
            Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(99), None, Some(3)])),
        ],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::Bfs));

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let vertices = int64_array(batch, 0);
    let distances = int64_array(batch, 1);
    let distance_for = |vertex: i64| {
        (0..batch.num_rows())
            .find_map(|row| (vertices.value(row) == vertex).then(|| distances.value(row)))
            .unwrap_or_else(|| panic!("missing vertex {vertex} in BFS output"))
    };
    assert_eq!(distance_for(0), 0);
    assert_eq!(distance_for(1), 1);
    assert_eq!(distance_for(2), 2);
    assert_eq!(distance_for(3), 3);

    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("null_edge_rows_dropped=2"));

    let null_filter = result
        .metrics()
        .operators
        .iter()
        .find(|row| row.operator_kind == "graph_edge_null_filter")
        .expect("null-edge filter metrics should be recorded when dirty rows are dropped");
    assert_eq!(null_filter.input_rows, 5);
    assert_eq!(null_filter.output_rows, 3);
}

#[test]
fn bfs_python_cugraph_reports_null_edges_dropped_before_edge_normalize() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let edge_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Int32, true), Field::new("dst", DataType::Int32, true)]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(0), Some(1), None, Some(2), Some(2)])),
            Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(99), None, Some(3)])),
        ],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::Bfs));

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("null_edge_rows_dropped=2"));
}

#[test]
fn bfs_drops_null_weight_and_edge_id_rows_before_graph_construction() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
        Field::new("weight", DataType::Float64, true),
        Field::new("edge_id", DataType::Int32, true),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 0, 1, 2])),
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 4])),
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(1.0), Some(1.0), Some(1.0)])),
            Arc::new(Int32Array::from(vec![Some(10), Some(11), None, Some(14), Some(24)])),
        ],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")).with_edge_id("edge_id"),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::Bfs));

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("null_edge_rows_dropped=2"));

    let null_filter = result
        .metrics()
        .operators
        .iter()
        .find(|row| row.operator_kind == "graph_edge_null_filter")
        .expect("null-edge filter metrics should be recorded when dirty rows are dropped");
    assert_eq!(null_filter.input_rows, 5);
    assert_eq!(null_filter.output_rows, 3);
}

#[test]
fn bfs_include_edge_ids_filters_edges() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
        Field::new("edge_id", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0, 1, 2])),
            Arc::new(Int32Array::from(vec![1, 2, 3, 3])),
            Arc::new(Int32Array::from(vec![10, 11, 12, 13])),
        ],
    )
    .unwrap();

    let allowed_schema = Arc::new(Schema::new(vec![Field::new("edge_id", DataType::Int32, false)]));
    let allowed_batch =
        RecordBatch::try_new(Arc::clone(&allowed_schema), vec![Arc::new(Int32Array::from(vec![11, 13]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let allowed =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("allowed", Arc::clone(&allowed_schema))));
    let params = GraphBfsParams {
        source_vertex: 0,
        source_vertices: None,
        depth_limit: None,
        predicates: GraphBfsPredicateInputs {
            include_edge_ids: Some(GraphPredicateInput::new(allowed, "edge_id")),
            ..Default::default()
        },
        return_target_info: false,
    };
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>).with_edge_id("edge_id"),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::Bfs));

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "allowed".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &allowed_batch).unwrap()),
            Arc::clone(&allowed_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let vertices = int64_array(batch, 0);
    let distances = int64_array(batch, 1);
    let distance_for = |vertex: i64| {
        (0..batch.num_rows())
            .find_map(|row| (vertices.value(row) == vertex).then(|| distances.value(row)))
            .unwrap_or_else(|| panic!("missing vertex {vertex} in BFS output"))
    };
    assert_eq!(distance_for(0), 0);
    assert!(distance_for(1) > 1_000_000);
    assert_eq!(distance_for(2), 1);
    assert_eq!(distance_for(3), 2);

    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("predicate_include_edge_ids_rows=2"));
}

#[test]
fn bfs_include_edges_relation_filters_by_source_destination() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 0, 1, 2])), Arc::new(Int32Array::from(vec![1, 2, 3, 3]))],
    )
    .unwrap();

    let allowed_schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Int32, false),
        Field::new("allowed_dst", DataType::Int32, false),
    ]));
    let allowed_batch = RecordBatch::try_new(
        Arc::clone(&allowed_schema),
        vec![Arc::new(Int32Array::from(vec![0, 2])), Arc::new(Int32Array::from(vec![2, 3]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let allowed =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("allowed", Arc::clone(&allowed_schema))));
    let params = GraphBfsParams {
        source_vertex: 0,
        source_vertices: None,
        depth_limit: None,
        predicates: GraphBfsPredicateInputs {
            include_edges: Some(GraphEdgePredicateInput::new(allowed, "allowed_src", "allowed_dst", None::<&str>)),
            ..Default::default()
        },
        return_target_info: false,
    };
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::Bfs));

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "allowed".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &allowed_batch).unwrap()),
            Arc::clone(&allowed_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let vertices = int64_array(batch, 0);
    let distances = int64_array(batch, 1);
    let distance_for = |vertex: i64| {
        (0..batch.num_rows())
            .find_map(|row| (vertices.value(row) == vertex).then(|| distances.value(row)))
            .unwrap_or_else(|| panic!("missing vertex {vertex} in BFS output"))
    };
    let contains_vertex = |vertex: i64| (0..batch.num_rows()).any(|row| vertices.value(row) == vertex);

    assert_eq!(distance_for(0), 0);
    assert!(!contains_vertex(1));
    assert_eq!(distance_for(2), 1);
    assert_eq!(distance_for(3), 2);

    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("predicate_include_edges_rows=2"));
    assert!(graph_metrics[0].algorithm_metadata.contains("predicate_include_edges_aligned_rows=2"));
}

#[test]
fn bfs_target_metadata_columns_are_shaped_when_requested() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 0])), Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();

    let target_schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int32, false)]));
    let target_batch =
        RecordBatch::try_new(Arc::clone(&target_schema), vec![Arc::new(Int32Array::from(vec![2]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let targets =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("targets", Arc::clone(&target_schema))));
    let mut params = GraphBfsParams::new(0, None);
    params.predicates.target_vertices = Some(GraphPredicateInput::new(targets, "vertex_id"));
    params.return_target_info = true;
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int64, false),
        Field::new("distance", DataType::Int64, false),
        Field::new("predecessor", DataType::Int64, true),
        Field::new("target_found", DataType::Boolean, false),
        Field::new("target_distance", DataType::Int64, true),
    ]));
    let spec = GraphAlgorithmSpec::new(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
        Arc::clone(&output_schema),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, output_schema);

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "targets".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &target_batch).unwrap()),
            Arc::clone(&target_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let target_found = boolean_array(batch, 3);
    let target_distance = int64_array(batch, 4);
    for row in 0..batch.num_rows() {
        assert!(target_found.value(row));
        assert_eq!(target_distance.value(row), 2);
    }

    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("target_found=true"));
    assert!(graph_metrics[0].algorithm_metadata.contains("target_distance=2"));
}
