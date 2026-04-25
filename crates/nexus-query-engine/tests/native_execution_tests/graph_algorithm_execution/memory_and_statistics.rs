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
    let grant = governor
        .request(GrantRequest::new(AdmissionRequestId::new(10_001), 256 * 1024 * 1024, 256 * 1024 * 1024))
        .unwrap();
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
