use super::*;

#[test]
fn bfs_drops_null_edge_rows_before_graph_construction() {
    let context = backend_context();

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(edges, Vec::new(), spec).unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    // The graph task feeds the host sink through the compiled pipeline.
    assert_eq!(result.metrics().compiled_pipeline_attempts, 1);
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let vertices = graph_vertex_values(batch, 0);
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
    assert!(!graph_metrics[0].algorithm_metadata.contains("graph_cache="));
    assert_eq!(graph_metrics[0].graph_peak_bytes, None);

    let null_filter = result
        .metrics()
        .operators
        .iter()
        .find(|row| row.operator_kind == "graph_edge_null_filter")
        .expect("null-edge filter metrics should be recorded when dirty rows are dropped");
    assert_eq!(null_filter.input_rows, 5);
    assert_eq!(null_filter.output_rows, 3);
    admitted.finish();
}

#[test]
fn bfs_python_cugraph_reports_null_edges_dropped_before_edge_normalize() {
    let context = backend_context();

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(edges, Vec::new(), spec).unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("null_edge_rows_dropped=2"));
    assert_eq!(graph_metrics[0].graph_peak_bytes, None);
    admitted.finish();
}

#[test]
fn bfs_keeps_null_weight_edges_but_drops_null_edge_id_rows() {
    let context = backend_context();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
        Field::new("weight", DataType::Float64, true),
        Field::new("edge_id", DataType::Int32, true),
    ]));
    // Edges: 0->1 (w=1, id=10), 1->2 (w=NULL, id=11), 0->3 (w=1, id=NULL),
    //        1->4 (w=1, id=14), 2->4 (w=1, id=24). BFS from 0.
    // Null weight on 1->2 must be kept (BFS ignores weight) -> 2 reachable.
    // Null edge_id on 0->3 is dropped (edge-id path consumes it) -> 3 unreachable.
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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")).with_edge_id("edge_id"),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(edges, Vec::new(), spec).unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let vertices = graph_vertex_values(batch, 0);
    let distances = int64_array(batch, 1);
    let distance_for = |vertex: i64| {
        (0..batch.num_rows()).find_map(|row| (vertices.value(row) == vertex).then(|| distances.value(row)))
    };
    // Reachability preserved across the null-weight edge 1->2.
    assert_eq!(distance_for(0), Some(0));
    assert_eq!(distance_for(1), Some(1));
    assert_eq!(distance_for(2), Some(2));
    assert_eq!(distance_for(4), Some(2));
    // Vertex 3's only inbound edge had a null edge_id and was dropped.
    let unreachable = i64::from(i32::MAX);
    assert!(distance_for(3).is_none_or(|distance| distance == unreachable));

    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("null_edge_rows_dropped=1"));

    let null_filter = result
        .metrics()
        .operators
        .iter()
        .find(|row| row.operator_kind == "graph_edge_null_filter")
        .expect("null-edge filter metrics should be recorded when dirty rows are dropped");
    assert_eq!(null_filter.input_rows, 5);
    assert_eq!(null_filter.output_rows, 4);
    admitted.finish();
}

// Companion to the above isolates BFS with only a weight binding
// (no edge_id). A null weight must never drop its edge, so every vertex stays
// reachable. Before the fix this returned i32::MAX sentinels for 1 and 2.
#[test]
fn bfs_null_weight_does_not_drop_reachable_edges() {
    let context = backend_context();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
        Field::new("weight", DataType::Float64, true),
    ]));
    // 0->9 (w=1), 0->1 (w=NULL), 1->2 (w=1). BFS from 0 ignores weights.
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0, 1])),
            Arc::new(Int32Array::from(vec![9, 1, 2])),
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(1.0)])),
        ],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(edges, Vec::new(), spec).unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let vertices = graph_vertex_values(batch, 0);
    let distances = int64_array(batch, 1);
    let distance_for = |vertex: i64| {
        (0..batch.num_rows())
            .find_map(|row| (vertices.value(row) == vertex).then(|| distances.value(row)))
            .unwrap_or_else(|| panic!("missing vertex {vertex} in BFS output"))
    };
    assert_eq!(distance_for(0), 0);
    assert_eq!(distance_for(9), 1);
    assert_eq!(distance_for(1), 1);
    assert_eq!(distance_for(2), 2);

    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("null_edge_rows_dropped=0"));
    admitted.finish();
}

#[test]
fn bfs_include_edge_ids_filters_edges() {
    let context = backend_context();

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let allowed = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "allowed",
        Arc::clone(&allowed_schema),
        &allowed_batch,
    )));
    let params = GraphBfsParams {
        source_vertex: 0,
        source_vertices: None,
        depth_limit: None,
        predicates: GraphBfsPredicateInputs {
            include_edge_ids: Some(GraphPredicateInput::new(1, "edge_id")),
            ..Default::default()
        },
        endpoint_policy: GraphBfsEndpointPolicy::Strict,
        path_output: None,
        return_target_info: false,
    };
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>).with_edge_id("edge_id"),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, vec![allowed], spec).unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "allowed".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &allowed_batch).unwrap()),
            Arc::clone(&allowed_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let vertices = graph_vertex_values(batch, 0);
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
    admitted.finish();
}

#[test]
fn bfs_include_edges_relation_filters_by_source_destination() {
    let context = backend_context();

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let allowed = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "allowed",
        Arc::clone(&allowed_schema),
        &allowed_batch,
    )));
    let params = GraphBfsParams {
        source_vertex: 0,
        source_vertices: None,
        depth_limit: None,
        predicates: GraphBfsPredicateInputs {
            include_edges: Some(GraphEdgePredicateInput::new(1, "allowed_src", "allowed_dst", None::<&str>)),
            ..Default::default()
        },
        endpoint_policy: GraphBfsEndpointPolicy::Strict,
        path_output: None,
        return_target_info: false,
    };
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, vec![allowed], spec).unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "allowed".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &allowed_batch).unwrap()),
            Arc::clone(&allowed_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let vertices = graph_vertex_values(batch, 0);
    let distances = int64_array(batch, 1);
    let distance_for = |vertex: i64| {
        (0..batch.num_rows()).find_map(|row| (vertices.value(row) == vertex).then(|| distances.value(row)))
    };
    assert_eq!(distance_for(0), Some(0));
    assert_eq!(distance_for(1), None);
    assert_eq!(distance_for(2), Some(1));
    assert_eq!(distance_for(3), Some(2));

    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("predicate_include_edges_rows=2"));
    admitted.finish();
}

#[test]
fn bfs_source_vertices_side_input_accepts_multiple_rows() {
    let context = backend_context();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 10])), Arc::new(Int32Array::from(vec![1, 11]))],
    )
    .unwrap();
    let source_schema = Arc::new(Schema::new(vec![Field::new("source_vertex", DataType::Int32, false)]));
    let source_batch =
        RecordBatch::try_new(Arc::clone(&source_schema), vec![Arc::new(Int32Array::from(vec![0, 10]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let sources = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "sources",
        Arc::clone(&source_schema),
        &source_batch,
    )));
    let mut params = GraphBfsParams::new(0, None);
    params.source_vertices = Some(GraphPredicateInput::new(1, "source_vertex"));
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, vec![sources], spec).unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "sources".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &source_batch).unwrap()),
            Arc::clone(&source_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let vertices = graph_vertex_values(batch, 0);
    let distances = int64_array(batch, 1);
    let distance_for = |vertex: i64| {
        (0..batch.num_rows())
            .find_map(|row| (vertices.value(row) == vertex).then(|| distances.value(row)))
            .unwrap_or_else(|| panic!("missing vertex {vertex} in BFS output"))
    };
    assert_eq!(distance_for(0), 0);
    assert_eq!(distance_for(1), 1);
    assert_eq!(distance_for(10), 0);
    assert_eq!(distance_for(11), 1);
    assert!(result.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata.contains("source_vertices_rows=2"));
    admitted.finish();
}

#[test]
fn bfs_source_vertices_side_input_rejects_empty_source_set() {
    let context = backend_context();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0])), Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    let source_schema = Arc::new(Schema::new(vec![Field::new("source_vertex", DataType::Int32, false)]));
    let source_batch =
        RecordBatch::try_new(Arc::clone(&source_schema), vec![Arc::new(Int32Array::from(Vec::<i32>::new()))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let sources = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "sources",
        Arc::clone(&source_schema),
        &source_batch,
    )));
    let mut params = GraphBfsParams::new(0, None);
    params.source_vertices = Some(GraphPredicateInput::new(1, "source_vertex"));
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, vec![sources], spec).unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "sources".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &source_batch).unwrap()),
            Arc::clone(&source_schema),
        )
        .unwrap(),
    );

    let error = match execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs) {
        Ok(_) => panic!("empty BFS source-vertices side input should fail"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("source-vertices side input must contain at least one row"), "{error}");
    admitted.finish();
}

#[test]
fn bfs_empty_path_endpoint_policy_returns_empty_after_edge_filter() {
    let context = backend_context();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2])), Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let source_schema = Arc::new(Schema::new(vec![Field::new("source_vertex", DataType::Int32, false)]));
    let source_batch =
        RecordBatch::try_new(Arc::clone(&source_schema), vec![Arc::new(Int32Array::from(vec![0]))]).unwrap();
    let target_schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int32, false)]));
    let target_batch =
        RecordBatch::try_new(Arc::clone(&target_schema), vec![Arc::new(Int32Array::from(vec![3]))]).unwrap();
    let allowed_schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Int32, false),
        Field::new("allowed_dst", DataType::Int32, false),
    ]));
    let allowed_batch = RecordBatch::try_new(
        Arc::clone(&allowed_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int32Array::from(vec![2, 3]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let sources = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "sources",
        Arc::clone(&source_schema),
        &source_batch,
    )));
    let targets = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "targets",
        Arc::clone(&target_schema),
        &target_batch,
    )));
    let allowed = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "allowed",
        Arc::clone(&allowed_schema),
        &allowed_batch,
    )));
    // Ordinals name `side_inputs` positions starting at 1; ordinal 0 is the edge list.
    let mut params = GraphBfsParams::new(0, None);
    params.source_vertices = Some(GraphPredicateInput::new(1, "source_vertex"));
    params.predicates.target_vertices = Some(GraphPredicateInput::new(2, "vertex_id"));
    params.predicates.include_edges = Some(GraphEdgePredicateInput::new(3, "allowed_src", "allowed_dst", None::<&str>));
    params.endpoint_policy = GraphBfsEndpointPolicy::EmptyOnMissingPathEndpoint;
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, vec![sources, targets, allowed], spec).unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    for (name, schema, batch) in [
        ("edges", Arc::clone(&edge_schema), &edge_batch),
        ("sources", Arc::clone(&source_schema), &source_batch),
        ("targets", Arc::clone(&target_schema), &target_batch),
        ("allowed", Arc::clone(&allowed_schema), &allowed_batch),
    ] {
        gpu_inputs.insert(
            name.to_string(),
            GpuResultHandle::new(
                &admitted,
                GpuDataFrame::new(DataFrame::from_record_batch(&runtime, batch).unwrap()),
                schema,
            )
            .unwrap(),
        );
    }

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    assert_eq!(batch.num_rows(), 0);
    admitted.finish();
}

#[test]
fn bfs_target_metadata_columns_are_shaped_when_requested() {
    let context = backend_context();

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let targets = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "targets",
        Arc::clone(&target_schema),
        &target_batch,
    )));
    let mut params = GraphBfsParams::new(0, None);
    params.predicates.target_vertices = Some(GraphPredicateInput::new(1, "vertex_id"));
    params.return_target_info = true;
    let params = GraphAlgorithmParams::Bfs(Box::new(params));
    let output_schema = graph_result_schema_for_vertex_type(GraphAlgorithm::Bfs, &params, &DataType::Int32);
    let spec = GraphAlgorithmSpec::new(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        params,
        Arc::clone(&output_schema),
    );
    let graph = builder.add_graph_algorithm(edges, vec![targets], spec).unwrap();
    let plan = builder.finish_with_host_sink_schema(graph, output_schema).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap()),
            Arc::clone(&edge_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "targets".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &target_batch).unwrap()),
            Arc::clone(&target_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
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
    admitted.finish();
}
