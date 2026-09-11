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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
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
    let plan = builder
        .finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));
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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
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
    let plan = builder
        .finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));
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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
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
    let plan = builder
        .finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));
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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
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
    let plan = builder
        .finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));
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

/// `k` resolves to the first `k` distinct vertices in stable sorted order,
/// independent of the order edges arrive in.
#[test]
fn betweenness_k_resolves_to_sorted_first_vertices() {
    // Chain A (0->..->19, 19 edges) and chain B (1000->..->1009, 9 edges),
    // with B's edges listed first so first-seen order differs from sorted
    // order. k=2 must resolve to sorted seeds {0,1}, giving nonzero scores
    // exactly on chain-A interior vertices {1..=18} (18 vertices). First-seen
    // seeds {1000,1001} would instead score 8 chain-B vertices.
    let mut src_values: Vec<i32> = Vec::new();
    let mut dst_values: Vec<i32> = Vec::new();
    for vertex in 1000..1009 {
        src_values.push(vertex);
        dst_values.push(vertex + 1);
    }
    for vertex in 0..19 {
        src_values.push(vertex);
        dst_values.push(vertex + 1);
    }
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(src_values)), Arc::new(Int32Array::from(dst_values))],
    )
    .unwrap();

    let dir = unique_temp_dir("nexus-query-engine-betweenness-grant-invariant");
    let path = dir.join("edges.parquet");
    write_record_batch_with_row_group_rows(&path, Arc::clone(&edge_schema), edge_batch, 4);

    let run_with_grant = |grant: Option<u64>| {
        let context = backend_context_with_grant(grant);
        let mut builder = QueryPlanBuilder::new();
        let edges = builder.add_source(SourceNode::LocalParquet(fixture_local_parquet_source::from_path(
            &path,
            Arc::clone(&edge_schema),
            vec!["src".to_string(), "dst".to_string()],
        )));
        let spec = int32_graph_spec(
            GraphAlgorithm::BetweennessCentrality,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions {
                directed: true,
                renumber: true,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
                k: Some(2),
                ..Default::default()
            }),
        );
        let graph = builder.add_graph_algorithm(edges, spec);
        let plan = builder.finish_with_host_sink_schema(
            graph,
            int32_stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality),
        );
        let admitted = context.admit_plan(&plan).unwrap();
        let gpu_inputs = GpuResidentInputMap::new();
        let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
        let QueryOutput::HostArrow(batch) = result.output() else {
            panic!("host sink should materialize betweenness output");
        };
        let vertices = graph_vertex_values(batch, 0);
        let values = batch.column(1);
        let mut scored: Vec<i64> = Vec::new();
        for row in 0..batch.num_rows() {
            let positive = match values.data_type() {
                DataType::Float32 => values.as_primitive::<arrow::datatypes::Float32Type>().value(row) > 0.0,
                DataType::Float64 => values.as_primitive::<arrow::datatypes::Float64Type>().value(row) > 0.0,
                data_type => panic!("expected betweenness score column, got {data_type}"),
            };
            if positive {
                scored.push(vertices.value(row));
            }
        }
        scored.sort_unstable();
        admitted.finish();
        scored
    };

    let expected: Vec<i64> = (1..=18).collect();
    for grant in [Some(16 * 1024 * 1024), Some(128 * 1024 * 1024), Some(1024 * 1024 * 1024), None] {
        let scored = run_with_grant(grant);
        assert_eq!(scored, expected, "betweenness k=2 seeds must be the sorted-first vertices for grant {grant:?}");
    }
}

/// Betweenness results must not depend on the device budget.
///
/// The admission contract treats the cap as a resource decision, not a
/// semantic decision: the same seeds over the same graph must produce the
/// same scores whether cuGraph processes them in one batch or several.
/// cuGraph sizes its multisource batches from the remaining query-domain
/// budget, so shrinking the grant exercises different batch counts; every
/// grant that fits the query must still score all 1000 middle vertices.
#[test]
fn betweenness_result_is_independent_of_device_budget() {
    // Two disjoint two-level trees. Star 0: hub 0 -> middles 1..=500 ->
    // leaves. Star 1: hub 100000 -> middles -> leaves. Explicit seeds
    // {0, 100000}: the only nonzero scores are the 1000 middles.
    let mut src_values: Vec<i32> = Vec::new();
    let mut dst_values: Vec<i32> = Vec::new();
    let mut leaf = 501i32;
    for middle in 1..=500i32 {
        src_values.push(0);
        dst_values.push(middle);
        for _ in 0..100 {
            src_values.push(middle);
            dst_values.push(leaf);
            leaf += 1;
        }
    }
    let mut leaf1 = 100501i32;
    for middle in 100001..=100500i32 {
        src_values.push(100000);
        dst_values.push(middle);
        for _ in 0..100 {
            src_values.push(middle);
            dst_values.push(leaf1);
            leaf1 += 1;
        }
    }
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(src_values)), Arc::new(Int32Array::from(dst_values))],
    )
    .unwrap();

    // Grants below ~6MiB cannot fit this query and fail loudly with
    // QueryMemoryLimitExceeded; that is the designed behavior. Every grant
    // at or above fit must produce the identical result.
    for grant_mb in [6u64, 7, 8, 9, 10, 12, 16, 32] {
        let context = backend_context_with_grant(grant_mb * 1024 * 1024);
        let mut builder = QueryPlanBuilder::new();
        let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
            "edges",
            Arc::clone(&edge_schema),
            &edge_batch,
        )));
        let spec = int32_graph_spec(
            GraphAlgorithm::BetweennessCentrality,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions {
                directed: true,
                renumber: true,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
                seeds: Some(GraphBetweennessSeeds::I64(vec![0, 100000])),
                ..Default::default()
            }),
        );
        let graph = builder.add_graph_algorithm(edges, spec);
        let plan = builder.finish_with_host_sink_schema(
            graph,
            int32_stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality),
        );
        let admitted = context
            .admit_plan(&plan)
            .unwrap_or_else(|error| panic!("grant={grant_mb}MiB must admit the two-star plan: {error:?}"));
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
        let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs)
            .unwrap_or_else(|error| panic!("grant={grant_mb}MiB must execute the two-star plan: {error:?}"));
        let QueryOutput::HostArrow(batch) = result.output() else {
            panic!("host sink should materialize betweenness output");
        };
        assert_eq!(batch.num_rows(), 101_002, "vertex count must not depend on the grant");
        let values = batch.column(1);
        let mut scored = 0u64;
        for row in 0..batch.num_rows() {
            let positive = match values.data_type() {
                DataType::Float32 => values.as_primitive::<arrow::datatypes::Float32Type>().value(row) > 0.0,
                DataType::Float64 => values.as_primitive::<arrow::datatypes::Float64Type>().value(row) > 0.0,
                data_type => panic!("expected betweenness score column, got {data_type}"),
            };
            scored += u64::from(positive);
        }
        assert_eq!(scored, 1000, "all middles must be scored at grant={grant_mb}MiB");
        admitted.finish();
    }
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
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
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
    let plan = builder
        .finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::BetweennessCentrality));
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
