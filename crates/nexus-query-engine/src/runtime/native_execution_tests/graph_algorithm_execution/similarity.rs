fn similarity_spec(
    algorithm: GraphAlgorithm,
    pair_input: nexus_query_engine::plan::NodeId,
    with_weight: bool,
) -> GraphAlgorithmSpec {
    similarity_spec_for_pair_columns(algorithm, pair_input, with_weight, "first", "second")
}

fn similarity_spec_for_pair_columns(
    algorithm: GraphAlgorithm,
    pair_input: nexus_query_engine::plan::NodeId,
    with_weight: bool,
    first_column: &str,
    second_column: &str,
) -> GraphAlgorithmSpec {
    GraphAlgorithmSpec::with_stable_schema(
        algorithm,
        GraphEdgeColumnBindings::new("src", "dst", with_weight.then_some("weight")),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::PythonCugraph },
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::new(GraphVertexPairInput::new(
            pair_input,
            first_column,
            second_column,
        ))),
    )
}

fn similarity_edge_schema(vertex: DataType, with_weight: bool) -> Arc<Schema> {
    let mut fields = vec![Field::new("src", vertex.clone(), false), Field::new("dst", vertex, false)];
    if with_weight {
        fields.push(Field::new("weight", DataType::Float64, false));
    }
    Arc::new(Schema::new(fields))
}

fn similarity_pair_schema(vertex: DataType, nullable: bool) -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("first", vertex.clone(), nullable), Field::new("second", vertex, nullable)]))
}

fn symmetric_similarity_edges_i32(with_weight: bool) -> (Arc<Schema>, RecordBatch) {
    let schema = similarity_edge_schema(DataType::Int32, with_weight);
    let src = Arc::new(Int32Array::from(vec![0, 0, 1, 2]));
    let dst = Arc::new(Int32Array::from(vec![1, 2, 2, 3]));
    let columns: Vec<Arc<dyn Array>> = if with_weight {
        vec![src, dst, Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 5.0]))]
    } else {
        vec![src, dst]
    };
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    (schema, batch)
}

fn assert_bounded_similarity_pair_input_overlap_is_credited() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let (edge_schema, edge_batch) = symmetric_similarity_edges_i32(false);
    let pair_schema = similarity_pair_schema(DataType::Int32, false);
    let pair_rows = 100_000_usize;
    let pair_batch = RecordBatch::try_new(
        Arc::clone(&pair_schema),
        vec![Arc::new(Int32Array::from(vec![0; pair_rows])), Arc::new(Int32Array::from(vec![1; pair_rows]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
    let pairs = builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&pair_schema), vec![pair_batch])));
    let graph = builder.add_graph_algorithm(edges, similarity_spec(GraphAlgorithm::Jaccard, pairs, false));
    let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(GraphAlgorithm::Jaccard));

    let context_with_budget = |device_budget_bytes| {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .build()
            .unwrap()
    };
    let gpu_inputs_for = |context: &ExecutionContext| {
        let mut inputs = GpuResidentInputMap::new();
        inputs
            .insert("edges".to_string(), graph_input_handle(context, &runtime, Arc::clone(&edge_schema), &edge_batch));
        inputs
    };

    // First establish the isolated model price and actual literal-source
    // reservation. The calibrated similarity model includes the pair columns;
    // the second execution narrows the budget to the interval that only the
    // incremental graph request may admit.
    let baseline_context = context_with_budget(64 * 1024 * 1024);
    let baseline_inputs = gpu_inputs_for(&baseline_context);
    let baseline = execute_query_plan_with_gpu_inputs(&baseline_context, &plan, &baseline_inputs).unwrap();
    assert_eq!(baseline.metrics().graph_reservation_rows.len(), 1);
    let baseline_report = baseline.metrics().graph_reservation_rows[0].clone();
    let pair_source_bytes = baseline
        .metrics()
        .reservation_events
        .iter()
        .find(|event| {
            event.memory_boundary == "bounded_generic_literal_batch_source"
                && event.reservation_kind == ReservationKind::SourceRead
                && event.action == MemoryReservationEventAction::Reserve
        })
        .expect("literal pair relation must retain a live source reservation")
        .requested_bytes;
    let pair_input_overlap_bytes = u64::try_from(pair_rows).unwrap() * 2 * 4;
    let credited_bytes = pair_input_overlap_bytes.min(pair_source_bytes);
    assert!(credited_bytes > 0);
    assert_eq!(
        baseline_report.requested_bytes,
        baseline_report
            .model_requested_bytes
            .saturating_sub(credited_bytes)
            .max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES),
        "the graph reservation must be the incremental price after retaining the literal pair relation"
    );
    let graph_capacity = baseline_report.requested_bytes.saturating_add(credited_bytes / 2);
    assert!(
        baseline_report.model_requested_bytes > graph_capacity,
        "the regression budget must lie between the discounted and isolated graph prices"
    );
    drop(baseline);
    drop(baseline_inputs);

    let context = context_with_budget(pair_source_bytes.saturating_add(graph_capacity));
    let gpu_inputs = gpu_inputs_for(&context);
    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap_or_else(|error| {
        panic!("bounded similarity must admit its live pair input exactly once: {error:#?}")
    });
    assert_eq!(result.metrics().graph_reservation_rows.len(), 1);
    let report = &result.metrics().graph_reservation_rows[0];
    assert_eq!(report.outcome, GraphReservationOutcome::Admitted);
    assert_eq!(report.requested_bytes, baseline_report.requested_bytes);
    assert!(report.requested_bytes <= graph_capacity);
}

#[test]
fn bounded_similarity_credits_live_literal_pair_input_once() {
    assert_bounded_similarity_pair_input_overlap_is_credited();
}
