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
fn graph_budget_uses_grant_arena_member_stream_reservation() {
    let runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let governor = DeviceLedger::for_test(runtime.device().ordinal(), 512 * 1024 * 1024, 2);
    let grant = governor
        .admit_for_test(AdmissionRequestId::new(10_001), 256 * 1024 * 1024, 256 * 1024 * 1024)
        .unwrap()
        .expect("grant within the envelope admits");
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
