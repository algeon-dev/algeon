#[test]
fn betweenness_with_explicit_seeds_records_approximate_metadata() {
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());

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
    let context = backend_context_with_grant(16 * 1024 * 1024);
    let runtime = Arc::new(context.runtime().unwrap());

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
    assert_eq!(seed_budget.enforcement, "reservation_arena_member");
    assert_eq!(seed_budget.status, "reservation_enforced");
    assert_eq!(seed_budget.budget_bytes, graph_reservations[0].requested_bytes);
    assert!(seed_budget.allocator_high_water_bytes.is_some());
    assert!(seed_budget.context_facts.contains_key("reservation_id"));
}

#[test]
fn exact_betweenness_above_threshold_is_refused_before_cugraph() {
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());

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
    let context = backend_context_with_grant(512 * 1024);
    let runtime = Arc::new(context.runtime().unwrap());
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
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());

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
