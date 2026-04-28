#[test]
fn bfs_include_edge_ids_filters_edges() {
    let runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let cache = Arc::new(GpuResidentGraphCache::new(64 * 1024 * 1024));
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_resident_graph_cache(Arc::clone(&cache))
        .build()
        .unwrap();

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
        endpoint_policy: GraphBfsEndpointPolicy::Strict,
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
    assert!(graph_metrics[0].algorithm_metadata.contains("graph_cache=bypass_edge_id_graph"));
    assert_eq!(cache.len().unwrap(), 0);
    assert_eq!(cache.checked_out_len().unwrap(), 0);
}
