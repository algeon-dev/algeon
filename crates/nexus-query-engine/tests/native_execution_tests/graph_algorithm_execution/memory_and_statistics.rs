#[test]
fn bounded_bfs_counts_gpu_tables_without_graph_memory_reservation() {
    let context = backend_context_with_grant(16 * 1024 * 1024);

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs));
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
    assert_eq!(graph_budget.enforcement, "query_domain_member");
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
    admitted.finish();
}

#[test]
fn bounded_bfs_counts_side_input_tables_without_graph_memory_reservation() {
    let context = backend_context_with_grant(16 * 1024 * 1024);

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let sources = builder.add_source(SourceNode::GpuResident(gpu_resident_source("sources", Arc::clone(&source_schema), &source_batch)));
    let mut params = GraphBfsParams::new(0, None);
    params.source_vertices = Some(GraphPredicateInput::new(sources, "source_vertex"));
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs));
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
    assert_eq!(graph_budget.enforcement, "query_domain_member");
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
    admitted.finish();
}

#[test]
fn bounded_betweenness_success_keeps_graph_memory_reservation() {
    let context = backend_context_with_grant(16 * 1024 * 1024);

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let spec = int32_graph_spec(
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
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));
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
    assert_eq!(graph_budget.enforcement, "query_domain_member");
    assert_eq!(graph_budget.status, "reservation_enforced");
    assert_eq!(graph_budget.budget_bytes, graph_reserve.requested_bytes);
    assert!(result.metrics().reservation_events.iter().any(|event| {
        event.memory_boundary == "bounded_generic_graph_algorithm_output"
            && event.reservation_kind == ReservationKind::OperatorOutput
            && event.action == MemoryReservationEventAction::Transfer
    }));
    admitted.finish();
}

#[test]
fn bounded_sparse_pagerank_uses_runtime_vertex_estimate_to_admit_graph_memory() {
    let fixture_runtime = backend_runtime();
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
    let fixture_edge_frame = GpuDataFrame::new(DataFrame::from_record_batch(&fixture_runtime, &edge_batch).unwrap());
    let input_bytes = fixture_edge_frame.device_allocated_size_bytes().unwrap();

    let spec = int32_graph_spec(
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
        construction_workspace_bounds: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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

    drop(fixture_edge_frame);
    drop(fixture_runtime);
    let context = backend_context_with_grant(input_bytes + graph_budget_remaining);

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::PageRank));
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let edge_frame = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs
        .insert("edges".to_string(), GpuResultHandle::new(&admitted, edge_frame, Arc::clone(&edge_schema)).unwrap());

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
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
    let live_before_probe = probe_reserve
        .current_reserved_bytes
        .saturating_sub(u64::try_from(probe_reserve.charged_delta_bytes).expect("granted probe charge fits u64"));
    let probe_expected_bytes =
        graph_vertex_estimate_probe_bytes(model_input, probe_reserve.budget_bytes.saturating_sub(live_before_probe));
    assert_eq!(probe_reserve.requested_bytes, probe_expected_bytes);
    assert!(result.metrics().reservation_events.iter().any(|event| {
        event.memory_boundary == GRAPH_VERTEX_ESTIMATE_BOUNDARY
            && event.reservation_kind == ReservationKind::GraphAlgorithm
            && event.action == MemoryReservationEventAction::Release
    }));
    assert!(result.metrics().budgeted_execution_events.iter().any(|event| {
        event.operation == "nexus_query_engine.graph_algorithm.graph_memory_vertex_estimate"
            && event.boundary == GRAPH_VERTEX_ESTIMATE_BOUNDARY
            && event.enforcement == "query_domain_member"
            && event.status == "reservation_enforced"
            && event.unresolved_contract.as_deref() == Some("graph_vertex_estimate_probe")
    }));
    admitted.finish();
}

#[test]
fn bounded_triangle_count_reports_runtime_graph_statistics() {
    let context = backend_context_with_grant(512 * 1024 * 1024);

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let spec = int32_graph_spec(
        GraphAlgorithm::TriangleCountAll,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
    );
    let expected_model_spec = spec.clone();
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::TriangleCountAll));
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
    let construction_workspace = cugraph::cudf_interop::graph_construction_workspace_preflight(
        &cugraph::cudf_interop::GraphConstructionPreflightInput::new(
            cugraph::types::DataType::Int32,
            cugraph::types::DataType::Int32,
            None,
            graph_report.has_edge_id,
            false,
            true,
            usize::try_from(graph_report.input_edges).unwrap(),
            usize::try_from(graph_statistics.exact_vertices).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
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
                construction_workspace_bounds: Some(nexus_query_engine::graph_memory::GraphConstructionWorkspaceBounds::new(
                    construction_workspace.renumber_workspace_bytes(),
                    construction_workspace.compression_workspace_bytes(),
                )),
                pair_rows: None,
                pair_vertex_width_bytes: None,
                pair_cardinality_source: None,
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
                construction_workspace_bounds: Some(nexus_query_engine::graph_memory::GraphConstructionWorkspaceBounds::new(
                    construction_workspace.renumber_workspace_bytes(),
                    construction_workspace.compression_workspace_bytes(),
                )),
                pair_rows: None,
                pair_vertex_width_bytes: None,
                pair_cardinality_source: None,
            },
        )
        .unwrap();
    assert_eq!(graph_report.model_requested_bytes, exact_vertex_with_statistics.requested_bytes);
    assert!(
        graph_report.model_requested_bytes < exact_vertex_no_statistics.requested_bytes,
        "runtime triangle statistics must feed the mechanistic model instead of falling back to exact-vertex no-stats pricing"
    );
    admitted.finish();
}

#[test]
fn bounded_graph_statistics_optional_resource_exhaustion_keeps_exact_vertices() {
    let context = backend_context_with_grant(512 * 1024 * 1024);

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let spec = int32_graph_spec(
        GraphAlgorithm::TriangleCountAll,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::TriangleCountAll));
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

    force_next_graph_statistics_optional_resource_exhausted_for_test();
    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();

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
    admitted.finish();
}

#[test]
fn pagerank_python_cugraph_reports_edge_normalize_in_graph_reservation() {
    let context = backend_context();
    let calibration_override = GraphMemoryCalibrationOverride::default().with_raw_input_edges(45);

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2])), Arc::new(Int32Array::from(vec![1, 2, 0]))],
    )
    .unwrap();
    let fixture_runtime = backend_runtime();
    let fixture_dir = unique_temp_dir("pagerank-normalization-source-lifetime");
    let edge_path = fixture_dir.join("edges.parquet");
    DataFrame::from_record_batch(&fixture_runtime, &edge_batch).unwrap().write_parquet(&edge_path).unwrap();
    let local_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&edge_path)],
        Arc::clone(&edge_schema),
        vec!["src".to_string(), "dst".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::LocalParquet(local_source));
    let spec = int32_graph_spec(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::PageRank));
    assert!(plan.contains_edge_normalize());
    let admitted = context.admit_plan_with_graph_memory_calibration_override(&plan, calibration_override).unwrap();

    let result = admitted
        .execute_with_graph_memory_calibration_override(&plan, &GpuResidentInputMap::new(), calibration_override)
        .unwrap();
    let reservations = &result.metrics().graph_reservation_rows;
    assert_eq!(reservations.len(), 1);
    assert!(reservations[0].edge_normalize_inserted);
    assert_eq!(reservations[0].construction_policy, GraphConstructionPolicy::RawLibcugraph);
    assert_eq!(reservations[0].raw_input_edges, Some(45));
    assert!(reservations[0].graph_model_error_ratio.is_some());
    let normalized_frame_bytes = result
        .metrics()
        .reservation_events
        .iter()
        .find(|event| {
            event.memory_boundary == "bounded_generic_edge_normalize_output"
                && event.action == MemoryReservationEventAction::Transfer
        })
        .expect("edge normalization must transfer its live output charge")
        .requested_bytes;
    let retained_source_bytes = u64::try_from(edge_batch.num_rows()).expect("fixture rows fit u64") * 2 * 4;
    let semantic_edge_bytes = reservations[0]
        .input_edges
        .checked_mul(reservations[0].vertex_width_bytes.checked_mul(2).expect("edge width fits u64"))
        .expect("semantic edge bytes fit u64");
    let mut expected_incremental_bytes =
        reservations[0].model_requested_bytes.max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
    expected_incremental_bytes -= normalized_frame_bytes.min(expected_incremental_bytes);
    expected_incremental_bytes -= retained_source_bytes.min(semantic_edge_bytes).min(expected_incremental_bytes);
    assert_eq!(
        reservations[0].requested_bytes, expected_incremental_bytes,
        "graph admission must credit both live normalization owners while their grant debits remain retained"
    );
    drop(result);
    admitted.finish();
    fs::remove_dir_all(fixture_dir).unwrap();
}
