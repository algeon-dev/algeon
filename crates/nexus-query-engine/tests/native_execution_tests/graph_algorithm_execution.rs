use super::*;

static GRAPH_ADAPTIVE_RETRY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn bfs_drops_null_edge_rows_before_graph_construction() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

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
    assert!(!graph_metrics[0].algorithm_metadata.contains("graph_cache="));
    assert!(graph_metrics[0].graph_peak_bytes.is_some());
    assert!(result.metrics().graph_reservation_rows.is_empty());
    assert!(result.metrics().budgeted_execution_events.iter().any(|event| {
        event.operation == "datafusion_nexus_cugraph_bfs"
            && event.boundary == "graph_memory"
            && event.enforcement == "stream_bound_reservation"
            && event.status == "reservation_enforced"
            && event.allocator_high_water_bytes.is_some()
    }));

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
fn bounded_bfs_resident_graph_cache_reuses_parked_forward_graph() {
    let capacity = 64 * 1024 * 1024;
    let runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let governor = MemoryGovernor::new(0, capacity).unwrap();
    let grant = governor.request(GrantRequest::new(99, capacity, capacity)).unwrap();
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
    assert!(governor.evictable_bytes().unwrap() > 0);

    let warm = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let warm_metadata = &warm.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata;
    assert!(warm_metadata.contains("graph_cache=hit;"), "{warm_metadata}");
}

#[test]
fn shared_residency_does_not_preserve_graph_residency_descriptor() {
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
    let derived = GpuDataFrame::try_new_with_shared_residency(selected, &source).unwrap();

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
fn bfs_python_cugraph_reports_null_edges_dropped_before_edge_normalize() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

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
    assert!(graph_metrics[0].graph_peak_bytes.is_some());
    assert!(result.metrics().graph_reservation_rows.is_empty());
    assert!(result.metrics().budgeted_execution_events.iter().any(|event| {
        event.operation == "datafusion_nexus_cugraph_bfs"
            && event.boundary == "graph_memory"
            && event.enforcement == "stream_bound_reservation"
            && event.status == "reservation_enforced"
            && event.allocator_high_water_bytes.is_some()
    }));
}

#[test]
fn pagerank_python_cugraph_reports_edge_normalize_in_graph_reservation() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_graph_memory_calibration_override(GraphMemoryCalibrationOverride::default().with_raw_input_edges(45))
        .build()
        .unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2])), Arc::new(Int32Array::from(vec![1, 2, 0]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::PageRank));
    assert!(plan.contains_edge_normalize());

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
    let reservations = &result.metrics().graph_reservation_rows;
    assert_eq!(reservations.len(), 1);
    assert!(reservations[0].edge_normalize_inserted);
    assert_eq!(reservations[0].construction_policy, GraphConstructionPolicy::RawLibcugraph);
    assert_eq!(reservations[0].raw_input_edges, Some(45));
    assert!(reservations[0].graph_model_error_ratio.is_some());
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

#[test]
fn bounded_bfs_counts_gpu_tables_without_graph_memory_reservation() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 16 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
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
    assert!(result.metrics().graph_reservation_rows.is_empty(), "BFS should not add graph-memory reservation rows");
    assert!(!result.metrics().reservation_events.iter().any(|event| {
        event.memory_boundary == "graph_memory" && event.reservation_kind == ReservationKind::GraphAlgorithm
    }));
    let graph_budget = result
        .metrics()
        .budgeted_execution_events
        .iter()
        .find(|event| event.operation == "datafusion_nexus_cugraph_bfs")
        .expect("bounded BFS should record a graph reservation observation");
    assert_eq!(graph_budget.boundary, "graph_memory");
    assert_eq!(graph_budget.enforcement, "stream_bound_reservation");
    assert_eq!(graph_budget.status, "reservation_enforced");
    assert_eq!(
        graph_budget.budget_bytes,
        u64::try_from(usize::MAX).unwrap(),
        "BFS must not inherit the bounded context budget as a graph execution hard cap"
    );
    assert_eq!(result.metrics().graph_algorithm_runtime_rows.len(), 1);
    assert!(result.metrics().graph_algorithm_runtime_rows[0].graph_peak_bytes.is_some());
    let source_borrow = result
        .metrics()
        .reservation_events
        .iter()
        .find(|event| {
            event.memory_boundary == "bounded_generic_gpu_resident_source"
                && event.reservation_kind == ReservationKind::SourceRead
                && event.action == MemoryReservationEventAction::AliasAcquire
        })
        .expect("bounded BFS should account the borrowed GPU-resident edge table");
    assert!(source_borrow.requested_bytes > 0);
    assert_eq!(source_borrow.charged_delta_bytes, 0);
    let output_reserve = result
        .metrics()
        .reservation_events
        .iter()
        .find(|event| {
            event.memory_boundary == "bounded_generic_graph_algorithm_output"
                && event.reservation_kind == ReservationKind::OperatorOutput
                && event.action == MemoryReservationEventAction::Reserve
        })
        .expect("bounded BFS should account the GPU-resident output table");
    assert!(output_reserve.requested_bytes > 0);
    assert!(
        output_reserve.current_reserved_bytes >= output_reserve.requested_bytes,
        "BFS output reservation should be charged without double-charging the borrowed input table"
    );
    assert!(result.metrics().reservation_events.iter().any(|event| {
        event.memory_boundary == "bounded_generic_graph_algorithm_output"
            && event.reservation_kind == ReservationKind::OperatorOutput
            && event.action == MemoryReservationEventAction::Reserve
    }));
    assert!(!result.metrics().reservation_events.iter().any(|event| {
        event.memory_boundary == "bounded_generic_graph_algorithm_output"
            && event.reservation_kind == ReservationKind::OperatorOutput
            && event.action == MemoryReservationEventAction::Transfer
    }));
    assert!(result.metrics().reservation_live_peak_bytes >= output_reserve.current_reserved_bytes);
}

#[test]
fn graph_budget_uses_grant_arena_member_stream_reservation() {
    let runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 512 * 1024 * 1024).unwrap();
    let grant = governor.request(GrantRequest::new(10_001, 256 * 1024 * 1024, 256 * 1024 * 1024)).unwrap();
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_grant(grant)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 512 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
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
    let graph_budget = result
        .metrics()
        .budgeted_execution_events
        .iter()
        .find(|event| event.operation == "datafusion_nexus_cugraph_bfs")
        .expect("grant-backed graph execution should record a graph reservation observation");
    assert_eq!(graph_budget.boundary, "graph_memory");
    assert_eq!(graph_budget.enforcement, "reservation_arena_member");
    assert_eq!(graph_budget.status, "reservation_enforced");
    assert_eq!(
        graph_budget.budget_bytes,
        u64::try_from(usize::MAX).unwrap(),
        "BFS should keep the unbounded graph execution sentinel while still using the grant arena resource"
    );
    assert!(
        graph_budget.context_facts.contains_key("arena_id"),
        "grant-backed graph reservation should report parent arena usage"
    );
}

#[test]
fn bounded_bfs_counts_side_input_tables_without_graph_memory_reservation() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 16 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

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
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let sources =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("sources", Arc::clone(&source_schema))));
    let mut params = GraphBfsParams::new(0, None);
    params.source_vertices = Some(GraphPredicateInput::new(sources, "source_vertex"));
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
        "sources".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &source_batch).unwrap()),
            Arc::clone(&source_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    assert!(result.metrics().graph_reservation_rows.is_empty(), "BFS should not add graph-memory reservation rows");
    assert!(!result.metrics().reservation_events.iter().any(|event| {
        event.memory_boundary == "graph_memory" && event.reservation_kind == ReservationKind::GraphAlgorithm
    }));
    let graph_budget = result
        .metrics()
        .budgeted_execution_events
        .iter()
        .find(|event| event.operation == "datafusion_nexus_cugraph_bfs")
        .expect("bounded BFS with side inputs should record a graph reservation observation");
    assert_eq!(graph_budget.boundary, "graph_memory");
    assert_eq!(graph_budget.enforcement, "stream_bound_reservation");
    assert_eq!(graph_budget.status, "reservation_enforced");
    assert_eq!(result.metrics().graph_algorithm_runtime_rows.len(), 1);
    assert!(result.metrics().graph_algorithm_runtime_rows[0].graph_peak_bytes.is_some());
    let source_read_bytes = result
        .metrics()
        .reservation_events
        .iter()
        .filter(|event| {
            event.memory_boundary == "bounded_generic_gpu_resident_source"
                && event.reservation_kind == ReservationKind::SourceRead
                && event.action == MemoryReservationEventAction::AliasAcquire
        })
        .map(|event| {
            assert_eq!(event.charged_delta_bytes, 0);
            event.requested_bytes
        })
        .collect::<Vec<_>>();
    assert_eq!(
        source_read_bytes.len(),
        2,
        "BFS edge and source-vertices inputs should both be accounted as borrowed resident inputs"
    );
    assert!(source_read_bytes.iter().all(|bytes| *bytes > 0));
    let output_reserve = result
        .metrics()
        .reservation_events
        .iter()
        .find(|event| {
            event.memory_boundary == "bounded_generic_graph_algorithm_output"
                && event.reservation_kind == ReservationKind::OperatorOutput
                && event.action == MemoryReservationEventAction::Reserve
        })
        .expect("bounded BFS should account the GPU-resident output table");
    assert!(
        output_reserve.current_reserved_bytes >= output_reserve.requested_bytes,
        "BFS output reservation should be charged without double-charging borrowed edge and side-input tables"
    );
    assert!(result.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata.contains("source_vertices_rows=2"));
}

#[test]
fn bounded_betweenness_success_keeps_graph_memory_reservation() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 16 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2, 3])), Arc::new(Int32Array::from(vec![1, 2, 3, 0]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            seeds: Some(GraphBetweennessSeeds::I64(vec![0, 2])),
            ..Default::default()
        }),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));

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
    let graph_reservations = &result.metrics().graph_reservation_rows;
    assert_eq!(graph_reservations.len(), 1);
    assert_eq!(graph_reservations[0].algorithm, GraphAlgorithm::BetweennessCentrality);
    assert_eq!(graph_reservations[0].outcome, GraphReservationOutcome::Admitted);
    assert_eq!(graph_reservations[0].vertex_upper_bound, 8);
    assert_eq!(graph_reservations[0].vertex_estimate, 4);
    assert_eq!(graph_reservations[0].vertex_estimate_source, GraphVertexEstimateSource::RuntimeExactDistinct);
    let graph_statistics =
        graph_reservations[0].statistics.expect("bounded betweenness should record graph statistics");
    assert_eq!(graph_statistics.exact_vertices, 4);
    assert_eq!(graph_statistics.sum_out_degree, Some(2));
    assert_eq!(graph_statistics.degree_max, None);
    assert_eq!(graph_statistics.sum_min_degree_oriented, None);
    let graph_reserve = result
        .metrics()
        .reservation_events
        .iter()
        .find(|event| {
            event.memory_boundary == "graph_memory" && event.reservation_kind == ReservationKind::GraphAlgorithm
        })
        .expect("bounded betweenness should reserve graph memory");
    assert_eq!(graph_reserve.requested_bytes, graph_reservations[0].requested_bytes);
    let probe_reserve = result
        .metrics()
        .reservation_events
        .iter()
        .find(|event| {
            event.memory_boundary == GRAPH_VERTEX_ESTIMATE_BOUNDARY
                && event.reservation_kind == ReservationKind::GraphAlgorithm
                && event.action == MemoryReservationEventAction::Reserve
        })
        .expect("bounded betweenness should reserve a runtime graph-statistics probe");
    assert!(
        probe_reserve.requested_bytes > GRAPH_VERTEX_ESTIMATE_PROBE_MIN_BYTES,
        "betweenness runtime degree evidence should reserve more than the exact-vertex floor"
    );
    let graph_budget = result
        .metrics()
        .budgeted_execution_events
        .iter()
        .find(|event| event.operation == "datafusion_nexus_cugraph_betweenness_centrality")
        .expect("bounded betweenness should enforce a graph memory budget");
    assert_eq!(graph_budget.boundary, "graph_memory");
    assert_eq!(graph_budget.enforcement, "stream_bound_reservation");
    assert_eq!(graph_budget.status, "reservation_enforced");
    assert_eq!(graph_budget.budget_bytes, graph_reserve.requested_bytes);
    assert!(result.metrics().reservation_events.iter().any(|event| {
        event.memory_boundary == "bounded_generic_graph_algorithm_output"
            && event.reservation_kind == ReservationKind::OperatorOutput
            && event.action == MemoryReservationEventAction::Transfer
    }));
}

#[test]
fn bounded_sparse_pagerank_uses_runtime_vertex_estimate_to_admit_graph_memory() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let edge_count = 1_000_000_u64;
    let source = (0..edge_count).map(|row| (row % 4) as i32).collect::<Vec<_>>();
    let destination = (0..edge_count).map(|row| ((row + 1) % 4) as i32).collect::<Vec<_>>();
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(source)), Arc::new(Int32Array::from(destination))],
    )
    .unwrap();
    let edge_frame = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap());
    let input_bytes = edge_frame.device_allocated_size_bytes().unwrap();

    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let model_input = GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: edge_count,
        vertex_width_bytes: 4,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: false,
    };
    let fallback_estimate = GraphMemoryModel.estimate(&spec, model_input).unwrap();
    let exact_estimate = GraphMemoryModel
        .estimate(&spec, model_input.with_vertex_estimate(GraphVertexEstimate::runtime_exact_distinct(4)))
        .unwrap();
    let fallback_request = fallback_estimate.requested_bytes.max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
    let exact_request = exact_estimate.requested_bytes.max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
    assert!(exact_request < fallback_request, "test requires sparse graph pricing to be smaller than the 2E fallback");
    let graph_budget_remaining = exact_request + (fallback_request - exact_request) / 2;
    assert!(graph_budget_remaining < fallback_request);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: input_bytes + graph_budget_remaining,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::PageRank));

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs
        .insert("edges".to_string(), GpuResultHandle::new(&context, edge_frame, Arc::clone(&edge_schema)).unwrap());

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize PageRank output");
    };
    assert_eq!(batch.num_rows(), 4);

    let graph_reservations = &result.metrics().graph_reservation_rows;
    assert_eq!(graph_reservations.len(), 1);
    assert_eq!(graph_reservations[0].algorithm, GraphAlgorithm::PageRank);
    assert_eq!(graph_reservations[0].outcome, GraphReservationOutcome::Admitted);
    assert_eq!(graph_reservations[0].vertex_upper_bound, edge_count * 2);
    assert_eq!(graph_reservations[0].vertex_estimate, 4);
    assert_eq!(graph_reservations[0].vertex_estimate_source, GraphVertexEstimateSource::RuntimeExactDistinct);
    let graph_statistics =
        graph_reservations[0].statistics.expect("bounded PageRank should record exact graph statistics");
    assert_eq!(graph_statistics.exact_vertices, 4);
    assert_eq!(graph_statistics.sum_out_degree, None);
    assert_eq!(graph_statistics.degree_max, None);
    assert_eq!(graph_statistics.sum_min_degree_oriented, None);
    // R6 records the drift ratio from the graph stream reservation high-water
    // even when a tiny graph cannot raise the process pool high-water.
    assert!(
        graph_reservations[0].graph_model_error_ratio.is_some(),
        "graph reservation rows should carry lease-attributed model drift evidence"
    );
    let graph_runtime_rows = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_runtime_rows.len(), 1);
    assert!(
        graph_runtime_rows[0].graph_peak_bytes.is_some(),
        "every successful graph runtime row must carry a non-null peak"
    );
    assert!(
        graph_reservations[0].requested_bytes <= graph_budget_remaining,
        "runtime exact vertex estimate should admit within a budget that the 2E fallback would exceed"
    );
    assert!(fallback_request > graph_budget_remaining);

    let probe_expected_bytes =
        graph_vertex_estimate_probe_bytes(model_input, input_bytes.saturating_add(graph_budget_remaining));
    let probe_reserve = result
        .metrics()
        .reservation_events
        .iter()
        .find(|event| {
            event.memory_boundary == GRAPH_VERTEX_ESTIMATE_BOUNDARY
                && event.reservation_kind == ReservationKind::GraphAlgorithm
                && event.action == MemoryReservationEventAction::Reserve
        })
        .expect("bounded graph admission should reserve a sized exact-vertex probe");
    assert_eq!(probe_reserve.requested_bytes, probe_expected_bytes);
    assert!(result.metrics().reservation_events.iter().any(|event| {
        event.memory_boundary == GRAPH_VERTEX_ESTIMATE_BOUNDARY
            && event.reservation_kind == ReservationKind::GraphAlgorithm
            && event.action == MemoryReservationEventAction::Release
    }));
    assert!(result.metrics().budgeted_execution_events.iter().any(|event| {
        event.operation == "nexus_query_engine.graph_algorithm.graph_memory_vertex_estimate"
            && event.boundary == GRAPH_VERTEX_ESTIMATE_BOUNDARY
            && event.enforcement == "stream_bound_reservation"
            && event.status == "reservation_enforced"
            && event.unresolved_contract.as_deref() == Some("graph_vertex_estimate_probe")
    }));
}

#[test]
fn bounded_triangle_count_reports_runtime_graph_statistics() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 512 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 0, 1, 2])), Arc::new(Int32Array::from(vec![1, 2, 2, 3]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::TriangleCountAll,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
    );
    let expected_model_spec = spec.clone();
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::TriangleCountAll));

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
        panic!("host sink should materialize triangle-count output");
    };
    assert_eq!(batch.num_rows(), 4);

    let graph_reservations = &result.metrics().graph_reservation_rows;
    assert_eq!(graph_reservations.len(), 1);
    let graph_report = &graph_reservations[0];
    assert_eq!(graph_report.algorithm, GraphAlgorithm::TriangleCountAll);
    assert_eq!(graph_report.outcome, GraphReservationOutcome::Admitted);
    let graph_statistics = graph_report.statistics.expect("bounded triangle count should record graph statistics");
    assert_eq!(graph_statistics.exact_vertices, 4);
    assert_eq!(graph_statistics.sum_out_degree, None);
    assert_eq!(graph_statistics.degree_max, Some(3));
    assert_eq!(graph_statistics.degree_p99, Some(3));
    assert_eq!(graph_statistics.sum_min_degree_oriented, Some(1));
    assert_eq!(graph_statistics.oriented_edge_count, Some(4));
    assert_eq!(graph_statistics.peak_chunk_min_degree_sum, Some(1));
    let exact_vertex_no_statistics = GraphMemoryModel
        .estimate(
            &expected_model_spec,
            GraphMemoryModelInput {
                raw_input_edges: graph_report.raw_input_edges,
                input_edges: graph_report.input_edges,
                vertex_width_bytes: graph_report.vertex_width_bytes,
                weight_width_bytes: graph_report.weight_width_bytes,
                has_edge_id: graph_report.has_edge_id,
                device_total_memory_bytes: None,
                vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(graph_statistics.exact_vertices)),
                statistics: None,
                edge_normalize_inserted: graph_report.edge_normalize_inserted,
            },
        )
        .unwrap();
    let exact_vertex_with_statistics = GraphMemoryModel
        .estimate(
            &expected_model_spec,
            GraphMemoryModelInput {
                raw_input_edges: graph_report.raw_input_edges,
                input_edges: graph_report.input_edges,
                vertex_width_bytes: graph_report.vertex_width_bytes,
                weight_width_bytes: graph_report.weight_width_bytes,
                has_edge_id: graph_report.has_edge_id,
                device_total_memory_bytes: None,
                vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(graph_statistics.exact_vertices)),
                statistics: Some(graph_statistics),
                edge_normalize_inserted: graph_report.edge_normalize_inserted,
            },
        )
        .unwrap();
    assert_eq!(graph_report.model_requested_bytes, exact_vertex_with_statistics.requested_bytes);
    assert!(
        graph_report.model_requested_bytes < exact_vertex_no_statistics.requested_bytes,
        "runtime triangle statistics must feed the mechanistic model instead of falling back to exact-vertex no-stats pricing"
    );
}

#[test]
fn bounded_graph_statistics_optional_resource_exhaustion_keeps_exact_vertices() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 512 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 0, 1, 2])), Arc::new(Int32Array::from(vec![1, 2, 2, 3]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::TriangleCountAll,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::TriangleCountAll));

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

    force_next_graph_statistics_optional_resource_exhausted_for_test();
    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();

    let graph_reservations = &result.metrics().graph_reservation_rows;
    assert_eq!(graph_reservations.len(), 1);
    assert_eq!(graph_reservations[0].algorithm, GraphAlgorithm::TriangleCountAll);
    assert_eq!(graph_reservations[0].vertex_estimate, 4);
    assert_eq!(graph_reservations[0].vertex_estimate_source, GraphVertexEstimateSource::RuntimeExactDistinct);
    let graph_statistics =
        graph_reservations[0].statistics.expect("bounded graph admission should keep exact graph statistics");
    assert_eq!(graph_statistics.exact_vertices, 4);
    assert_eq!(graph_statistics.sum_out_degree, None);
    assert_eq!(graph_statistics.degree_max, None);
    assert_eq!(graph_statistics.degree_p99, None);
    assert_eq!(graph_statistics.sum_min_degree_oriented, None);
    assert_eq!(graph_statistics.peak_chunk_min_degree_sum, None);
}

#[test]
fn betweenness_with_explicit_seeds_records_approximate_metadata() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2, 3])), Arc::new(Int32Array::from(vec![1, 2, 3, 0]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            seeds: Some(GraphBetweennessSeeds::I64(vec![0, 2])),
            ..Default::default()
        }),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));

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
        panic!("host sink should materialize betweenness output");
    };
    assert_eq!(batch.num_columns(), 2);
    assert!(batch.num_rows() > 0);
    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("approximate=true"));
    assert!(graph_metrics[0].algorithm_metadata.contains("seed_count=2"));
}

#[test]
fn betweenness_k_selects_deterministic_approximate_seed_count() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 16 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![3, 0, 1, 2])), Arc::new(Int32Array::from(vec![0, 1, 2, 3]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            k: Some(2),
            ..Default::default()
        }),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));

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
    assert!(graph_metrics[0].algorithm_parameters.contains("k=2"));
    assert!(graph_metrics[0].algorithm_metadata.contains("approximate=true"));
    assert!(graph_metrics[0].algorithm_metadata.contains("seed_count=2"));
    let graph_reservations = &result.metrics().graph_reservation_rows;
    assert_eq!(graph_reservations.len(), 1);
    assert_eq!(graph_reservations[0].seed_count, Some(2));
    assert_eq!(graph_reservations[0].seed_mode, Some(GraphSeedMode::RuntimeKSelection));
    let seed_budget = result
        .metrics()
        .budgeted_execution_events
        .iter()
        .find(|event| event.operation == "nexus_query_engine.graph_algorithm.betweenness_seed_policy")
        .unwrap_or_else(|| {
            panic!(
                "runtime k seed selection should run under a graph reservation; observed events: {:#?}",
                result.metrics().budgeted_execution_events
            )
        });
    assert_eq!(seed_budget.boundary, "graph_memory");
    assert_eq!(seed_budget.enforcement, "stream_bound_reservation");
    assert_eq!(seed_budget.status, "reservation_enforced");
    assert_eq!(seed_budget.budget_bytes, graph_reservations[0].requested_bytes);
    assert!(seed_budget.allocator_high_water_bytes.is_some());
    assert!(seed_budget.context_facts.contains_key("reservation_id"));
}

#[test]
fn adaptive_betweenness_oom_retries_once_by_chunking_same_seed_set() {
    let _guard = GRAPH_ADAPTIVE_RETRY_TEST_LOCK.lock().expect("graph adaptive retry test lock poisoned");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let memory_history = Arc::new(AdaptiveMemoryHistory::new());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 16 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive)
        .with_adaptive_memory_history(Arc::clone(&memory_history))
        .build()
        .unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2, 3])), Arc::new(Int32Array::from(vec![1, 2, 3, 0]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            seeds: Some(GraphBetweennessSeeds::I64(vec![0, 1, 2, 3])),
            ..Default::default()
        }),
    );
    let history_spec = spec.clone();
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));

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

    let expected_result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(expected_batch) = expected_result.output() else {
        panic!("host sink should materialize full betweenness output");
    };

    force_next_graph_adaptive_full_attempt_resource_exhausted_for_test();
    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize adaptive chunked betweenness output");
    };
    assert!(batch.num_rows() > 0);
    assert_vertex_betweenness_batches_close(expected_batch, batch);
    assert_eq!(result.metrics().adaptive_retry_count, 1);
    let graph_reservation = result
        .metrics()
        .reservation_events
        .iter()
        .find(|event| {
            event.memory_boundary == "graph_memory" && event.reservation_kind == ReservationKind::GraphAlgorithm
        })
        .expect("adaptive graph retry should reserve graph memory");
    let graph_budget_events = result
        .metrics()
        .budgeted_execution_events
        .iter()
        .filter(|event| event.operation == "datafusion_nexus_cugraph_betweenness_centrality")
        .collect::<Vec<_>>();
    assert_eq!(graph_budget_events.len(), 2, "adaptive retry should budget each seed chunk");
    let max_budget_peak = graph_budget_events.iter().filter_map(|event| event.allocator_high_water_bytes).max();
    for event in &graph_budget_events {
        assert_eq!(event.boundary, "graph_memory");
        assert_eq!(event.enforcement, "stream_bound_reservation");
        assert_eq!(event.status, "reservation_enforced");
        assert_eq!(event.unresolved_contract, None);
        assert_eq!(event.budget_bytes, graph_reservation.requested_bytes);
        assert!(event.allocator_high_water_bytes.is_some());
    }
    assert!(result.metrics().max_task_observed_peak_bytes.is_some());
    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("adaptive_retry=oom_retry"));
    assert!(graph_metrics[0].algorithm_metadata.contains("seed_count=4"));
    assert!(graph_metrics[0].algorithm_metadata.contains("seed_chunk_batches=2"));
    if let Some(max_budget_peak) = max_budget_peak {
        assert!(
            graph_metrics[0].graph_peak_bytes.is_some_and(|peak| peak >= max_budget_peak),
            "runtime graph peak should cover the graph-boundary budget high-water fallback"
        );
    }
    assert!(
        memory_history.lookup_query_plan(&plan).is_none(),
        "graph retry evidence must not enter query-level adaptive history"
    );
    let history = memory_history
        .lookup_graph_algorithm(&history_spec)
        .expect("adaptive graph retry should record graph-tier memory-history floor evidence");
    assert_eq!(history.observation_count, 1);
    assert!(history.max_allocator_high_water_bytes > 0);
}

#[test]
fn adaptive_betweenness_failed_retry_still_records_graph_floor() {
    let _guard = GRAPH_ADAPTIVE_RETRY_TEST_LOCK.lock().expect("graph adaptive retry test lock poisoned");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let memory_history = Arc::new(AdaptiveMemoryHistory::new());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 16 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive)
        .with_adaptive_memory_history(Arc::clone(&memory_history))
        .build()
        .unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2, 3])), Arc::new(Int32Array::from(vec![1, 2, 3, 0]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            seeds: Some(GraphBetweennessSeeds::I64(vec![0, 1, 2, 3])),
            ..Default::default()
        }),
    );
    let history_spec = spec.clone();
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));

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

    force_next_graph_adaptive_full_attempt_resource_exhausted_for_test();
    force_next_graph_adaptive_chunk_attempt_resource_exhausted_for_test();
    let error = match execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs) {
        Ok(_) => panic!("forced failed graph retry should return resource exhausted"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::CudfResourceExhausted);
    assert!(memory_history.lookup_query_plan(&plan).is_none());
    let history = memory_history
        .lookup_graph_algorithm(&history_spec)
        .expect("failed graph retry should still persist graph-tier floor evidence");
    assert!(history.observation_count >= 1);
    assert!(history.max_allocator_high_water_bytes > 0);
}

#[test]
fn adaptive_edge_betweenness_retry_matches_full_seed_result() {
    let _guard = GRAPH_ADAPTIVE_RETRY_TEST_LOCK.lock().expect("graph adaptive retry test lock poisoned");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 16 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive)
        .build()
        .unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2, 3])), Arc::new(Int32Array::from(vec![1, 2, 3, 0]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::EdgeBetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams {
            seeds: Some(GraphBetweennessSeeds::I64(vec![0, 1, 2, 3])),
            ..Default::default()
        }),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder
        .finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::EdgeBetweennessCentrality));

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

    let expected_result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(expected_batch) = expected_result.output() else {
        panic!("host sink should materialize full edge betweenness output");
    };

    force_next_graph_adaptive_full_attempt_resource_exhausted_for_test();
    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize adaptive chunked edge betweenness output");
    };
    assert_edge_betweenness_batches_close(expected_batch, batch);
    assert_eq!(result.metrics().adaptive_retry_count, 1);
    let graph_budget_events = result
        .metrics()
        .budgeted_execution_events
        .iter()
        .filter(|event| event.operation == "datafusion_nexus_cugraph_edge_betweenness_centrality")
        .collect::<Vec<_>>();
    assert_eq!(graph_budget_events.len(), 2, "adaptive retry should budget each edge-betweenness seed chunk");
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
    let vertices = int64_array(batch, 0);
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
    let sources = int64_array(batch, 0);
    let destinations = int64_array(batch, 1);
    let scores = float64_array(batch, 2);
    (0..batch.num_rows()).map(|row| ((sources.value(row), destinations.value(row)), scores.value(row))).collect()
}

#[test]
fn exact_betweenness_above_threshold_is_refused_before_cugraph() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1])), Arc::new(Int32Array::from(vec![1, 2]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            exact_vertex_threshold: 1,
            ..Default::default()
        }),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));

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

    let error = match execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs) {
        Ok(_) => panic!("exact betweenness above threshold should be refused before cuGraph"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("exact_betweenness_vertex_threshold_exceeded"));
    assert_eq!(error.fact_value("actual_vertex_count"), Some("3"));
    assert_eq!(error.fact_value("required_option"), Some("seeds_or_k"));
}

#[test]
fn bounded_exact_betweenness_above_threshold_is_refused_before_graph_memory_denial() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1])), Arc::new(Int32Array::from(vec![1, 2]))],
    )
    .unwrap();
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 512 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            exact_vertex_threshold: 1,
            ..Default::default()
        }),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));

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

    let error = match execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs) {
        Ok(_) => panic!("bounded exact betweenness above threshold should be refused before graph memory admission"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("exact_betweenness_vertex_threshold_exceeded"));
    assert_eq!(error.fact_value("actual_vertex_count"), Some("3"));
    assert_eq!(error.fact_value("required_option"), Some("seeds_or_k"));
}

#[test]
fn exact_betweenness_threshold_uses_actual_vertices_not_edge_upper_bound() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2])), Arc::new(Int32Array::from(vec![1, 2, 0]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            exact_vertex_threshold: 3,
            ..Default::default()
        }),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));

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

    execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
}

// BFS does not consume edge weights, so a null weight must NOT drop its edge
// (that would silently break reachability). A null edge_id, however, is
// consumed by the edge-id graph path here and is correctly dropped. This pins
// the fix for the BFS null-weight reachability P0.
#[test]
fn bfs_keeps_null_weight_edges_but_drops_null_edge_id_rows() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

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
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    let vertices = int64_array(batch, 0);
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
}

// Companion to the above isolating the pure P0: BFS with ONLY a weight binding
// (no edge_id). A null weight must never drop its edge, so every vertex stays
// reachable. Before the fix this returned i32::MAX sentinels for 1 and 2.
#[test]
fn bfs_null_weight_does_not_drop_reachable_edges() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

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
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
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
    assert_eq!(distance_for(9), 1);
    assert_eq!(distance_for(1), 1);
    assert_eq!(distance_for(2), 2);

    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("null_edge_rows_dropped=0"));
}

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

#[test]
fn bfs_include_edges_relation_filters_by_source_destination() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

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
        endpoint_policy: GraphBfsEndpointPolicy::Strict,
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
fn bfs_source_vertices_side_input_accepts_multiple_rows() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

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
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let sources =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("sources", Arc::clone(&source_schema))));
    let mut params = GraphBfsParams::new(0, None);
    params.source_vertices = Some(GraphPredicateInput::new(sources, "source_vertex"));
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
        "sources".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &source_batch).unwrap()),
            Arc::clone(&source_schema),
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
    assert_eq!(distance_for(10), 0);
    assert_eq!(distance_for(11), 1);
    assert!(result.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata.contains("source_vertices_rows=2"));
}

#[test]
fn bfs_source_vertices_side_input_rejects_empty_source_set() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

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
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let sources =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("sources", Arc::clone(&source_schema))));
    let mut params = GraphBfsParams::new(0, None);
    params.source_vertices = Some(GraphPredicateInput::new(sources, "source_vertex"));
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
        "sources".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &source_batch).unwrap()),
            Arc::clone(&source_schema),
        )
        .unwrap(),
    );

    let error = match execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs) {
        Ok(_) => panic!("empty BFS source-vertices side input should fail"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("source-vertices side input must contain at least one row"), "{error}");
}

#[test]
fn bfs_empty_path_endpoint_policy_returns_empty_after_edge_filter() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

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
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let sources =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("sources", Arc::clone(&source_schema))));
    let targets =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("targets", Arc::clone(&target_schema))));
    let allowed =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("allowed", Arc::clone(&allowed_schema))));
    let mut params = GraphBfsParams::new(0, None);
    params.source_vertices = Some(GraphPredicateInput::new(sources, "source_vertex"));
    params.predicates.target_vertices = Some(GraphPredicateInput::new(targets, "vertex_id"));
    params.predicates.include_edges =
        Some(GraphEdgePredicateInput::new(allowed, "allowed_src", "allowed_dst", None::<&str>));
    params.endpoint_policy = GraphBfsEndpointPolicy::EmptyOnMissingPathEndpoint;
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::Bfs));

    let mut gpu_inputs = GpuResidentInputMap::new();
    for (name, schema, batch) in [
        ("edges", Arc::clone(&edge_schema), edge_batch),
        ("sources", Arc::clone(&source_schema), source_batch),
        ("targets", Arc::clone(&target_schema), target_batch),
        ("allowed", Arc::clone(&allowed_schema), allowed_batch),
    ] {
        gpu_inputs.insert(
            name.to_string(),
            GpuResultHandle::new(
                &context,
                GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).unwrap()),
                schema,
            )
            .unwrap(),
        );
    }

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize BFS output");
    };
    assert_eq!(batch.num_rows(), 0);
    let metadata = &result.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata;
    assert!(metadata.contains("endpoint_policy=empty_on_missing_path_endpoint"), "{metadata}");
    assert!(metadata.contains("empty_result_reason=source_vertices_absent"), "{metadata}");
    assert!(metadata.contains("source_rows=1"), "{metadata}");
    assert!(metadata.contains("valid_source_rows=0"), "{metadata}");
    assert!(metadata.contains("target_rows=1"), "{metadata}");
    assert!(metadata.contains("valid_target_rows=1"), "{metadata}");
}

#[test]
fn bfs_target_metadata_columns_are_shaped_when_requested() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

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
    let params = GraphAlgorithmParams::Bfs(Box::new(params));
    let output_schema = GraphAlgorithm::Bfs.definition().result_schema_for_params(&params);
    let spec = GraphAlgorithmSpec::new(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        params,
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
