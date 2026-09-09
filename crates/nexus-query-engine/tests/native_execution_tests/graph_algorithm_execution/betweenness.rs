#[test]
fn betweenness_with_explicit_seeds_records_approximate_metadata() {
    let context = backend_context();

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
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize betweenness output");
    };
    assert_eq!(batch.num_columns(), 2);
    assert!(batch.num_rows() > 0);
    let graph_metrics = &result.metrics().graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("approximate=true"));
    assert!(graph_metrics[0].algorithm_metadata.contains("seed_count=2"));
    admitted.finish();
}

#[test]
fn betweenness_k_selects_deterministic_approximate_seed_count() {
    let context = backend_context_with_grant(16 * 1024 * 1024);

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let spec = int32_graph_spec(
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
    assert_eq!(seed_budget.enforcement, "query_domain_member");
    assert_eq!(seed_budget.status, "reservation_enforced");
    // cuGraph executes under the query cap. The graph-memory model's own
    // request is calibration evidence and never becomes the native limit.
    assert_eq!(seed_budget.budget_bytes, 16 * 1024 * 1024);
    assert!(seed_budget.allocator_high_water_bytes.is_some());
    assert!(seed_budget.context_facts.contains_key("reservation_id"));
    admitted.finish();
}

#[test]
fn exact_betweenness_above_threshold_is_refused_before_cugraph() {
    let context = backend_context();

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let spec = int32_graph_spec(
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

    let error = match execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs) {
        Ok(_) => panic!("exact betweenness above threshold should be refused before cuGraph"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("exact_betweenness_vertex_threshold_exceeded"));
    assert_eq!(error.fact_value("actual_vertex_count"), Some("3"));
    assert_eq!(error.fact_value("required_option"), Some("seeds_or_k"));
    admitted.finish();
}

#[test]
fn bounded_exact_betweenness_above_threshold_is_refused_before_graph_memory_denial() {
    let context = backend_context_with_grant(512 * 1024);
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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let spec = int32_graph_spec(
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

    let error = match execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs) {
        Ok(_) => panic!("bounded exact betweenness above threshold should be refused before graph memory admission"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("exact_betweenness_vertex_threshold_exceeded"));
    assert_eq!(error.fact_value("actual_vertex_count"), Some("3"));
    assert_eq!(error.fact_value("required_option"), Some("seeds_or_k"));
    admitted.finish();
}

#[test]
fn exact_betweenness_threshold_uses_actual_vertices_not_edge_upper_bound() {
    let context = backend_context();

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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let spec = int32_graph_spec(
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

    execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    admitted.finish();
}

// BFS does not consume edge weights, so a null weight must NOT drop its edge
// (that would silently break reachability). A null edge_id, however, is
// consumed by the edge-id graph path here and is correctly dropped. This pins
// the BFS null-weight reachability fix.

#[test]
fn bounded_betweenness_oom_retries_once_by_chunking_same_seed_set() {
    let _guard = GRAPH_SEED_CHUNK_RETRY_TEST_LOCK.lock().expect("graph seed-chunk retry test lock poisoned");
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 16 * 1024 * 1024,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });

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
            seeds: Some(GraphBetweennessSeeds::I64(vec![0, 1, 2, 3])),
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

    let expected_result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(expected_batch) = expected_result.output() else {
        panic!("host sink should materialize full betweenness output");
    };

    force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test();
    let outcome = execute_query_plan_with_gpu_inputs_observed(
        &admitted,
        &plan,
        &gpu_inputs,
        RuntimeObservationConfig::native_execution(),
    );
    let QueryOutput::HostArrow(batch) = outcome.result().expect("bounded seed-chunk retry should preserve native execution")
    else {
        panic!("host sink should materialize bounded chunked betweenness output");
    };
    let metrics = outcome.metrics();
    assert!(batch.num_rows() > 0);
    assert_vertex_betweenness_batches_close(expected_batch, batch);
    metrics
        .reservation_events
        .iter()
        .find(|event| {
            event.memory_boundary == "graph_memory" && event.reservation_kind == ReservationKind::GraphAlgorithm
        })
        .expect("bounded graph seed-chunk retry should reserve graph memory");
    let graph_budget_events = metrics
        .budgeted_execution_events
        .iter()
        .filter(|event| event.operation == "datafusion_nexus_cugraph_betweenness_centrality")
        .collect::<Vec<_>>();
    assert_eq!(graph_budget_events.len(), 2, "bounded retry should budget each seed chunk");
    let max_budget_peak = graph_budget_events.iter().filter_map(|event| event.allocator_high_water_bytes).max();
    for event in &graph_budget_events {
        assert_eq!(event.boundary, "graph_memory");
        assert_eq!(event.enforcement, "query_domain_member");
        assert_eq!(event.status, "reservation_enforced");
        assert_eq!(event.unresolved_contract, None);
        assert_eq!(event.budget_bytes, 16 * 1024 * 1024);
        assert!(event.allocator_high_water_bytes.is_some());
    }
    assert!(metrics.max_task_observed_peak_bytes.is_some());
    let graph_metrics = &metrics.graph_algorithm_runtime_rows;
    assert_eq!(graph_metrics.len(), 1);
    assert!(graph_metrics[0].algorithm_metadata.contains("bounded_seed_chunk_retry=oom_retry"));
    assert!(graph_metrics[0].algorithm_metadata.contains("seed_count=4"));
    assert!(graph_metrics[0].algorithm_metadata.contains("seed_chunk_batches=2"));
    if let Some(max_budget_peak) = max_budget_peak {
        assert!(
            graph_metrics[0].graph_peak_bytes.is_some_and(|peak| peak >= max_budget_peak),
            "runtime graph peak should cover the graph-boundary budget high-water fallback"
        );
    }
    let snapshot = outcome.observation().expect("bounded retry should preserve one native observation");
    assert_eq!(snapshot.manifest().query_id.as_str(), metrics.query_id().as_str());
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
    let plan_execution_ids = snapshot
        .topology()
        .filter_map(|record| match record {
            nexus_query_engine::observability::RuntimeTopologyRecord::PlanNode(row) => {
                Some(row.plan_execution_id.clone())
            },
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(plan_execution_ids.len(), 1, "internal retry must not start another query or execution identity");
    let plan_boundaries = snapshot
        .events()
        .iter()
        .filter_map(|event| match &event.payload {
            RuntimeObservationPayload::PlanExecutionBoundary { boundary } => Some(*boundary),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        plan_boundaries,
        vec![PlanExecutionBoundaryKind::Started, PlanExecutionBoundaryKind::Finished],
        "internal retry remains inside the initial native execution and never replays through another plan",
    );
    admitted.finish();
}

#[test]
fn bounded_betweenness_failed_retry_still_records_graph_floor() {
    let _guard = GRAPH_SEED_CHUNK_RETRY_TEST_LOCK.lock().expect("graph seed-chunk retry test lock poisoned");
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 16 * 1024 * 1024,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });

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
            seeds: Some(GraphBetweennessSeeds::I64(vec![0, 1, 2, 3])),
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

    force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test();
    force_next_graph_seed_chunk_retry_chunk_attempt_resource_exhausted_for_test();
    let outcome = execute_query_plan_with_gpu_inputs_observed(
        &admitted,
        &plan,
        &gpu_inputs,
        RuntimeObservationConfig::native_execution(),
    );
    let error = match outcome.result() {
        Ok(_) => panic!("forced failed graph retry should return resource exhausted"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded);
    assert_eq!(error.status(), QueryErrorCode::QueryMemoryLimitExceeded.status());
    let snapshot = outcome.observation().expect("failed bounded retry retains its terminal observation");
    assert_eq!(
        snapshot.footer().terminal,
        Some(RuntimeTerminalOutcome::from_query_error(error)),
        "terminal status must come from the terminal retry error, not the recorded retry action",
    );
    admitted.finish();
}

#[test]
fn bounded_edge_betweenness_retry_matches_full_seed_result() {
    let _guard = GRAPH_SEED_CHUNK_RETRY_TEST_LOCK.lock().expect("graph seed-chunk retry test lock poisoned");
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 16 * 1024 * 1024,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });

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
        .finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::EdgeBetweennessCentrality));
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

    let expected_result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(expected_batch) = expected_result.output() else {
        panic!("host sink should materialize full edge betweenness output");
    };

    force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test();
    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize bounded chunked edge betweenness output");
    };
    assert_edge_betweenness_batches_close(expected_batch, batch);
    let graph_budget_events = result
        .metrics()
        .budgeted_execution_events
        .iter()
        .filter(|event| event.operation == "datafusion_nexus_cugraph_edge_betweenness_centrality")
        .collect::<Vec<_>>();
    assert_eq!(graph_budget_events.len(), 2, "bounded retry should budget each edge-betweenness seed chunk");
    admitted.finish();
}
