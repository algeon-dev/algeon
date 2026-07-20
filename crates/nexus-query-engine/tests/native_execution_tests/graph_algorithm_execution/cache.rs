#[test]
fn bfs_resident_graph_cache_reuses_parked_forward_graph() {
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
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2, 2])), Arc::new(Int32Array::from(vec![1, 2, 0, 3]))],
    )
    .unwrap();

    let dir = unique_temp_dir("bfs-resident-graph-cache");
    let edge_path = dir.join("edges.parquet");
    DataFrame::from_record_batch(&runtime, &edge_batch).unwrap().write_parquet(&edge_path).unwrap();
    let edge_file_size = fs::metadata(&edge_path).unwrap().len();
    let local_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&edge_path).with_file_size_bytes(edge_file_size).with_stable_id(format!(
            "local:{}:{}:mtime=test",
            edge_path.display(),
            edge_file_size
        ))],
        Arc::clone(&edge_schema),
        vec!["src".to_string(), "dst".to_string()],
    );

    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );

    let mut descriptor_builder = QueryPlanBuilder::new();
    let descriptor_source = descriptor_builder.add_source(SourceNode::LocalParquet(local_source));
    let descriptor_plan = descriptor_builder.finish_with_host_sink(descriptor_source);
    let GraphResidencyPlanAnalysis::Keyable(descriptor) =
        graph_residency_descriptor_for_edge_plan(&descriptor_plan, &spec)
    else {
        panic!("local parquet edge source should be keyable for graph residency");
    };

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::Bfs));

    let gpu_frame = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap())
        .with_graph_residency(descriptor);
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs
        .insert("edges".to_string(), GpuResultHandle::new(&context, gpu_frame, Arc::clone(&edge_schema)).unwrap());

    let cold = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let cold_metadata = &cold.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata;
    assert!(cold_metadata.contains("graph_cache=miss_inserted"), "{cold_metadata}");
    assert_eq!(cache.len().unwrap(), 1);
    assert_eq!(cache.checked_out_len().unwrap(), 0);
    assert!(cache.retained_device_bytes().unwrap() > 0);
    assert!(cache.retained_device_bytes().unwrap() <= cache.max_retained_device_bytes());

    let warm = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let (QueryOutput::HostArrow(cold_batch), QueryOutput::HostArrow(warm_batch)) = (cold.output(), warm.output())
    else {
        panic!("host sink should materialize BFS output");
    };
    assert_eq!(cold_batch, warm_batch);
    let warm_metadata = &warm.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata;
    assert!(warm_metadata.contains("graph_cache=hit;"), "{warm_metadata}");
    assert_eq!(cache.len().unwrap(), 1);
    assert_eq!(cache.checked_out_len().unwrap(), 0);
}

#[test]
fn strict_query_service_exact_runtime_warm_pricing_executes_bfs_and_pagerank_below_cold_grant() {
    assert_query_service_exact_runtime_warm_pricing(NativePipelineExecutionMode::StrictBounded);
}

#[test]
fn adaptive_query_service_exact_runtime_warm_pricing_executes_bfs_and_pagerank_below_cold_grant() {
    assert_query_service_exact_runtime_warm_pricing(NativePipelineExecutionMode::Adaptive);
}

#[test]
fn bounded_bfs_resident_graph_cache_reuses_parked_forward_graph() {
    let capacity = 64 * 1024 * 1024;
    let runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let governor = MemoryGovernor::new(0, capacity).unwrap();
    let grant = governor.request(GrantRequest::new(AdmissionRequestId::new(99), capacity, capacity)).unwrap();
    let cache = Arc::new(GpuResidentGraphCache::new_governor_backed(capacity, governor.clone()));
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: capacity,
            max_source_chunk_bytes: capacity,
            max_row_groups_per_chunk: 1,
            min_retry_chunk_row_groups: 1,
            max_retry_attempts: 1,
        })
        .with_memory_grant(grant)
        .with_resident_graph_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2, 2])), Arc::new(Int32Array::from(vec![1, 2, 0, 3]))],
    )
    .unwrap();

    let dir = unique_temp_dir("bounded-bfs-resident-graph-cache");
    let edge_path = dir.join("edges.parquet");
    DataFrame::from_record_batch(&runtime, &edge_batch).unwrap().write_parquet(&edge_path).unwrap();
    let edge_file_size = fs::metadata(&edge_path).unwrap().len();
    let local_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&edge_path).with_file_size_bytes(edge_file_size).with_stable_id(format!(
            "local:{}:{}:mtime=test",
            edge_path.display(),
            edge_file_size
        ))],
        Arc::clone(&edge_schema),
        vec!["src".to_string(), "dst".to_string()],
    );

    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::PythonCugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );

    let mut descriptor_builder = QueryPlanBuilder::new();
    let descriptor_source = descriptor_builder.add_source(SourceNode::LocalParquet(local_source));
    let descriptor_plan = descriptor_builder.finish_with_host_sink(descriptor_source);
    let GraphResidencyPlanAnalysis::Keyable(descriptor) =
        graph_residency_descriptor_for_edge_plan(&descriptor_plan, &spec)
    else {
        panic!("local parquet edge source should be keyable for graph residency");
    };

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::Bfs));

    let gpu_frame = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap())
        .with_graph_residency(descriptor);
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs
        .insert("edges".to_string(), GpuResultHandle::new(&context, gpu_frame, Arc::clone(&edge_schema)).unwrap());

    let cold = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let cold_metadata = &cold.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata;
    assert!(cold_metadata.contains("graph_cache=miss_inserted"), "{cold_metadata}");
    let cold_reservation_stages =
        cold.metrics().reservation_events.iter().filter_map(|event| event.stage.as_deref()).collect::<Vec<_>>();
    assert!(cold_reservation_stages.contains(&"bounded_generic_edge_normalize"), "{cold_reservation_stages:?}");
    assert!(governor.evictable_bytes().unwrap() > 0);

    let warm = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let warm_metadata = &warm.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata;
    assert!(warm_metadata.contains("graph_cache=hit;"), "{warm_metadata}");
    let warm_reservation_stages =
        warm.metrics().reservation_events.iter().filter_map(|event| event.stage.as_deref()).collect::<Vec<_>>();
    assert!(!warm_reservation_stages.contains(&"bounded_generic_edge_normalize"));
}

#[test]
fn source_alias_residency_does_not_preserve_graph_residency_descriptor() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2])), Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();

    let dir = unique_temp_dir("shared-residency-graph-descriptor");
    let edge_path = dir.join("edges.parquet");
    DataFrame::from_record_batch(&runtime, &edge_batch).unwrap().write_parquet(&edge_path).unwrap();
    let edge_file_size = fs::metadata(&edge_path).unwrap().len();
    let local_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&edge_path).with_file_size_bytes(edge_file_size).with_stable_id(format!(
            "local:{}:{}:mtime=test",
            edge_path.display(),
            edge_file_size
        ))],
        Arc::clone(&edge_schema),
        vec!["src".to_string(), "dst".to_string()],
    );
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_source));
    let descriptor_plan = builder.finish_with_host_sink(source);
    let GraphResidencyPlanAnalysis::Keyable(descriptor) =
        graph_residency_descriptor_for_edge_plan(&descriptor_plan, &spec)
    else {
        panic!("local parquet edge source should be keyable for graph residency");
    };

    let source = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap())
        .with_graph_residency(descriptor.clone());
    let selected = source.frame().select(["src", "dst"]).unwrap();
    let derived = GpuDataFrame::try_new_with_source_residency(selected, &source).unwrap();

    assert!(
        derived.graph_residency().is_none(),
        "shared-residency derived frames must not blindly preserve graph residency"
    );
}

#[test]
fn bfs_without_graph_residency_reports_cache_bypass() {
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
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2])), Arc::new(Int32Array::from(vec![1, 2, 3]))],
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
    let metadata = &result.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata;
    assert!(metadata.contains("graph_cache=bypass_no_residency_descriptor"), "{metadata}");
    assert_eq!(cache.len().unwrap(), 0);
    assert_eq!(cache.checked_out_len().unwrap(), 0);
}

#[test]
fn bfs_include_edges_bypasses_resident_graph_cache() {
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

    let dir = unique_temp_dir("bfs-include-edges-resident-graph-cache");
    let edge_path = dir.join("edges.parquet");
    DataFrame::from_record_batch(&runtime, &edge_batch).unwrap().write_parquet(&edge_path).unwrap();
    let edge_file_size = fs::metadata(&edge_path).unwrap().len();
    let local_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&edge_path).with_file_size_bytes(edge_file_size).with_stable_id(format!(
            "local:{}:{}:mtime=test",
            edge_path.display(),
            edge_file_size
        ))],
        Arc::clone(&edge_schema),
        vec!["src".to_string(), "dst".to_string()],
    );

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
        endpoint_policy: GraphBfsEndpointPolicy::Strict,
        return_target_info: false,
    };
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );

    let mut descriptor_builder = QueryPlanBuilder::new();
    let descriptor_source = descriptor_builder.add_source(SourceNode::LocalParquet(local_source));
    let descriptor_plan = descriptor_builder.finish_with_host_sink(descriptor_source);
    let GraphResidencyPlanAnalysis::Keyable(descriptor) =
        graph_residency_descriptor_for_edge_plan(&descriptor_plan, &spec)
    else {
        panic!("local parquet edge source should be keyable for graph residency");
    };

    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::Bfs));

    let mut gpu_inputs = GpuResidentInputMap::new();
    let edge_frame = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap())
        .with_graph_residency(descriptor);
    gpu_inputs
        .insert("edges".to_string(), GpuResultHandle::new(&context, edge_frame, Arc::clone(&edge_schema)).unwrap());
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
    let metadata = &result.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata;
    assert!(metadata.contains("predicate_include_edges_rows=2"), "{metadata}");
    assert!(metadata.contains("graph_cache=bypass_predicate_include_edges"), "{metadata}");
    assert_eq!(cache.len().unwrap(), 0);
    assert_eq!(cache.checked_out_len().unwrap(), 0);
}

#[test]
fn pagerank_edge_id_binding_bypasses_resident_graph_cache() {
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
            Arc::new(Int32Array::from(vec![0, 1, 2])),
            Arc::new(Int32Array::from(vec![1, 2, 0])),
            Arc::new(Int32Array::from(vec![10, 11, 12])),
        ],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>).with_edge_id("edge_id"),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::PageRank));

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
    let metadata = &result.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata;
    assert!(metadata.contains("graph_cache=bypass_edge_id_binding"), "{metadata}");
    assert_eq!(cache.len().unwrap(), 0);
    assert_eq!(cache.checked_out_len().unwrap(), 0);
}
