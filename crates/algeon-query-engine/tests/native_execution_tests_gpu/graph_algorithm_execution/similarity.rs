use super::*;

// The pair relation is the only side input of a similarity node; ordinal 0 is the edge list.
const SIMILARITY_PAIR_INPUT_ORDINAL: usize = 1;

fn similarity_spec(algorithm: GraphAlgorithm, with_weight: bool) -> GraphAlgorithmSpec {
    similarity_spec_for_pair_columns(algorithm, with_weight, "first", "second")
}

fn similarity_spec_for_pair_columns(
    algorithm: GraphAlgorithm,
    with_weight: bool,
    first_column: &str,
    second_column: &str,
) -> GraphAlgorithmSpec {
    similarity_spec_for_pair_columns_and_vertex_type(
        algorithm,
        &DataType::Int32,
        with_weight,
        first_column,
        second_column,
    )
}

fn similarity_spec_for_pair_columns_and_vertex_type(
    algorithm: GraphAlgorithm,
    vertex_type: &DataType,
    with_weight: bool,
    first_column: &str,
    second_column: &str,
) -> GraphAlgorithmSpec {
    graph_spec_for_vertex_type(
        algorithm,
        GraphEdgeColumnBindings::new("src", "dst", with_weight.then_some("weight")),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::PythonCugraph },
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::new(GraphVertexPairInput::new(
            SIMILARITY_PAIR_INPUT_ORDINAL,
            first_column,
            second_column,
        ))),
        vertex_type,
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

fn similarity_plan(
    algorithm: GraphAlgorithm,
    edge_schema: Arc<Schema>,
    pair_schema: Arc<Schema>,
    edge_row_count_upper_bound: usize,
    pair_row_count_upper_bound: usize,
    with_weight: bool,
) -> QueryPlan {
    let vertex_type = edge_schema.field_with_name("src").expect("similarity edge schema has src").data_type().clone();
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("edges", edge_schema).with_row_count_upper_bound(
            u64::try_from(edge_row_count_upper_bound).expect("test edge cardinality must fit u64"),
        ),
    ));
    let pairs = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("pairs", pair_schema).with_row_count_upper_bound(
            u64::try_from(pair_row_count_upper_bound).expect("test pair cardinality must fit u64"),
        ),
    ));
    let graph = builder
        .add_graph_algorithm(
            edges,
            vec![pairs],
            similarity_spec_for_pair_columns_and_vertex_type(algorithm, &vertex_type, with_weight, "first", "second"),
        )
        .unwrap();
    builder
        .finish_with_host_sink_schema(graph, stable_graph_result_schema_for_vertex_type(algorithm, &vertex_type))
        .unwrap()
}

fn execute_similarity_plan(
    context: &AdmittedPlanContext,
    runtime: &Arc<cudf::Runtime>,
    plan: &QueryPlan,
    edge_schema: Arc<Schema>,
    edge_batch: &RecordBatch,
    pair_schema: Arc<Schema>,
    pair_batch: &RecordBatch,
) -> QueryEngineResult<QueryExecutionResult> {
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert("edges".to_string(), graph_input_handle(context, runtime, edge_schema, edge_batch));
    gpu_inputs.insert("pairs".to_string(), graph_input_handle(context, runtime, pair_schema, pair_batch));
    execute_query_plan_with_gpu_inputs(context, plan, &gpu_inputs)
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

fn assert_similarity_stable_output(
    batch: &RecordBatch,
    expected_vertex_type: &DataType,
    expected_pairs: &[(i64, i64)],
) {
    assert_eq!(
        batch.schema().as_ref(),
        stable_graph_result_schema_for_vertex_type(GraphAlgorithm::Jaccard, expected_vertex_type).as_ref(),
    );
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(batch.num_rows(), expected_pairs.len());
    let mut actual_pairs = (0..batch.num_rows())
        .map(|row| (graph_vertex_values(batch, 0).value(row), graph_vertex_values(batch, 1).value(row)))
        .collect::<Vec<_>>();
    let mut expected_pairs = expected_pairs.to_vec();
    actual_pairs.sort_unstable();
    expected_pairs.sort_unstable();
    assert_eq!(actual_pairs, expected_pairs, "similarity must preserve the pair-relation multiset");
    assert!((0..batch.num_rows()).all(|row| float64_array(batch, 2).value(row).is_finite()));
}

fn similarity_score(batch: &RecordBatch, first: i64, second: i64) -> f64 {
    (0..batch.num_rows())
        .find_map(|row| {
            (graph_vertex_values(batch, 0).value(row) == first && graph_vertex_values(batch, 1).value(row) == second)
                .then(|| float64_array(batch, 2).value(row))
        })
        .unwrap_or_else(|| panic!("missing similarity output pair ({first}, {second})"))
}

fn assert_bounded_similarity_pair_input_is_admitted() {
    let (edge_schema, edge_batch) = symmetric_similarity_edges_i32(false);
    let pair_schema = similarity_pair_schema(DataType::Int32, false);
    let pair_rows = 100_000_usize;
    let pair_batch = RecordBatch::try_new(
        Arc::clone(&pair_schema),
        vec![Arc::new(Int32Array::from(vec![0; pair_rows])), Arc::new(Int32Array::from(vec![1; pair_rows]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let pairs = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("pairs", Arc::clone(&pair_schema))
            .with_row_count_upper_bound(u64::try_from(pair_rows).expect("test pair cardinality must fit u64")),
    ));
    let graph =
        builder.add_graph_algorithm(edges, vec![pairs], similarity_spec(GraphAlgorithm::Jaccard, false)).unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Jaccard)).unwrap();

    // The allocator is the sole arbiter of admission; a generous bounded
    // budget only needs to prove the large resident pair input is admitted
    // and passed to the graph node zero-copy exactly once, with one charge against the grant.
    let context = backend_context_with_grant(64 * 1024 * 1024);
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs
        .insert("edges".to_string(), graph_input_handle(&admitted, &runtime, Arc::clone(&edge_schema), &edge_batch));
    gpu_inputs
        .insert("pairs".to_string(), graph_input_handle(&admitted, &runtime, Arc::clone(&pair_schema), &pair_batch));

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs)
        .unwrap_or_else(|error| panic!("bounded similarity must admit its live pair input exactly once: {error:#?}"));
    assert_eq!(result.metrics().boundary.gpu_resident_input_handoffs, 2);
    drop(result);
    drop(gpu_inputs);
    drop(runtime);
    admitted.finish();
    drop(admitted);
}

pub(super) fn bounded_similarity_accepts_gpu_pair_input_exactly_once() {
    assert_bounded_similarity_pair_input_is_admitted();
}

pub(super) fn pairwise_similarity_executes_all_metrics_with_stable_schema_and_multiset_pairs() {
    let context = backend_context();
    let (edge_schema, edge_batch) = symmetric_similarity_edges_i32(false);
    let pair_schema = similarity_pair_schema(DataType::Int32, false);
    let expected_pairs = vec![(0, 1), (0, 1), (1, 0), (0, 0), (0, 3)];
    let pair_batch = RecordBatch::try_new(
        Arc::clone(&pair_schema),
        vec![Arc::new(Int32Array::from(vec![0, 0, 1, 0, 0])), Arc::new(Int32Array::from(vec![1, 1, 0, 0, 3]))],
    )
    .unwrap();

    for algorithm in
        [GraphAlgorithm::Jaccard, GraphAlgorithm::Sorensen, GraphAlgorithm::Overlap, GraphAlgorithm::Cosine]
    {
        let plan = similarity_plan(
            algorithm,
            Arc::clone(&edge_schema),
            Arc::clone(&pair_schema),
            edge_batch.num_rows(),
            pair_batch.num_rows(),
            false,
        );
        let admitted = context.admit_plan(&plan).unwrap();
        let runtime = Arc::new(admitted.runtime().unwrap());
        {
            let result = execute_similarity_plan(
                &admitted,
                &runtime,
                &plan,
                Arc::clone(&edge_schema),
                &edge_batch,
                Arc::clone(&pair_schema),
                &pair_batch,
            )
            .unwrap_or_else(|error| panic!("{} should execute: {error:#?}", algorithm.as_str()));
            let QueryOutput::HostArrow(batch) = result.output() else {
                panic!("similarity plan must materialize a host Arrow result")
            };
            assert_similarity_stable_output(batch, &DataType::Int32, &expected_pairs);
            assert!((similarity_score(batch, 0, 1) - similarity_score(batch, 1, 0)).abs() <= 1.0e-12);
            assert!(similarity_score(batch, 0, 0) >= 0.0);
            assert_eq!(result.metrics().graph_algorithm_runtime_rows.len(), 1);
            assert!(result.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata.contains("vertex_pair_rows=5"));
        }
        drop(runtime);
        admitted.finish();
        drop(admitted);
    }
}

pub(super) fn pairwise_similarity_weighted_jaccard_uses_the_graph_weight_binding() {
    let context = backend_context();
    let (edge_schema, edge_batch) = symmetric_similarity_edges_i32(true);
    let pair_schema = similarity_pair_schema(DataType::Int32, false);
    let pair_batch = RecordBatch::try_new(
        Arc::clone(&pair_schema),
        vec![Arc::new(Int32Array::from(vec![0])), Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    let plan = similarity_plan(
        GraphAlgorithm::Jaccard,
        Arc::clone(&edge_schema),
        Arc::clone(&pair_schema),
        edge_batch.num_rows(),
        pair_batch.num_rows(),
        true,
    );
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    {
        let result = execute_similarity_plan(
            &admitted,
            &runtime,
            &plan,
            Arc::clone(&edge_schema),
            &edge_batch,
            Arc::clone(&pair_schema),
            &pair_batch,
        )
        .unwrap();
        let QueryOutput::HostArrow(batch) = result.output() else { panic!("host result expected") };
        assert_similarity_stable_output(batch, &DataType::Int32, &[(0, 1)]);
        assert!(
            (similarity_score(batch, 0, 1) - 0.4).abs() <= 1.0e-6,
            "weighted Jaccard must not fall back to unit weights"
        );
        assert!(result.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata.contains("use_weight=true"));
    }
    drop(runtime);
    admitted.finish();
    drop(admitted);
}

pub(super) fn pairwise_similarity_allows_one_pair_column_to_supply_both_self_pair_endpoints() {
    let context = backend_context();
    let (edge_schema, edge_batch) = symmetric_similarity_edges_i32(false);
    let pair_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int32, false)]));
    let pair_batch =
        RecordBatch::try_new(Arc::clone(&pair_schema), vec![Arc::new(Int32Array::from(vec![0, 1, 0]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "edges",
        Arc::clone(&edge_schema),
        &edge_batch,
    )));
    let pairs = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("pairs", Arc::clone(&pair_schema)).with_row_count_upper_bound(
            u64::try_from(pair_batch.num_rows()).expect("test pair cardinality must fit u64"),
        ),
    ));
    let graph = builder
        .add_graph_algorithm(
            edges,
            vec![pairs],
            similarity_spec_for_pair_columns(GraphAlgorithm::Jaccard, false, "vertex", "vertex"),
        )
        .unwrap();
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Jaccard)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    {
        let result = execute_similarity_plan(
            &admitted,
            &runtime,
            &plan,
            Arc::clone(&edge_schema),
            &edge_batch,
            Arc::clone(&pair_schema),
            &pair_batch,
        )
        .unwrap_or_else(|error| panic!("same-column self-pair relation should execute: {error:#?}"));
        let QueryOutput::HostArrow(batch) = result.output() else { panic!("host result expected") };
        assert_similarity_stable_output(batch, &DataType::Int32, &[(0, 0), (1, 1), (0, 0)]);
    }
    drop(runtime);
    admitted.finish();
    drop(admitted);
}

pub(super) fn pairwise_similarity_accepts_int64_pairs_and_short_circuits_empty_pairs() {
    let context = backend_context();
    let edge_schema = similarity_edge_schema(DataType::Int64, false);
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int64Array::from(vec![0, 0, 1])), Arc::new(Int64Array::from(vec![1, 2, 2]))],
    )
    .unwrap();
    let pair_schema = similarity_pair_schema(DataType::Int64, false);
    let nonempty_pairs = RecordBatch::try_new(
        Arc::clone(&pair_schema),
        vec![Arc::new(Int64Array::from(vec![0])), Arc::new(Int64Array::from(vec![1]))],
    )
    .unwrap();
    let plan = similarity_plan(
        GraphAlgorithm::Jaccard,
        Arc::clone(&edge_schema),
        Arc::clone(&pair_schema),
        edge_batch.num_rows(),
        nonempty_pairs.num_rows(),
        false,
    );
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    {
        let nonempty = execute_similarity_plan(
            &admitted,
            &runtime,
            &plan,
            Arc::clone(&edge_schema),
            &edge_batch,
            Arc::clone(&pair_schema),
            &nonempty_pairs,
        )
        .unwrap();
        let QueryOutput::HostArrow(batch) = nonempty.output() else { panic!("host result expected") };
        assert_similarity_stable_output(batch, &DataType::Int64, &[(0, 1)]);
    }

    let empty_pairs = RecordBatch::try_new(
        Arc::clone(&pair_schema),
        vec![Arc::new(Int64Array::from(Vec::<i64>::new())), Arc::new(Int64Array::from(Vec::<i64>::new()))],
    )
    .unwrap();
    {
        let empty = execute_similarity_plan(
            &admitted,
            &runtime,
            &plan,
            Arc::clone(&edge_schema),
            &edge_batch,
            Arc::clone(&pair_schema),
            &empty_pairs,
        )
        .unwrap();
        let QueryOutput::HostArrow(batch) = empty.output() else { panic!("host result expected") };
        assert_eq!(
            batch.schema().as_ref(),
            stable_graph_result_schema_for_vertex_type(GraphAlgorithm::Jaccard, &DataType::Int64).as_ref(),
        );
        assert_eq!(batch.num_rows(), 0);
        assert!(empty.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata.contains("vertex_pairs_empty"));
    }
    drop(runtime);
    admitted.finish();
    drop(admitted);
}

pub(super) fn pairwise_similarity_rejects_null_pairs_absent_endpoints_and_pair_graph_dtype_mismatch() {
    let context = backend_context();
    let (edge_schema, edge_batch) = symmetric_similarity_edges_i32(false);

    let nullable_pair_schema = similarity_pair_schema(DataType::Int32, true);
    let nullable_pairs = RecordBatch::try_new(
        Arc::clone(&nullable_pair_schema),
        vec![Arc::new(Int32Array::from(vec![Some(0), None])), Arc::new(Int32Array::from(vec![Some(1), Some(2)]))],
    )
    .unwrap();
    let nullable_plan = similarity_plan(
        GraphAlgorithm::Jaccard,
        Arc::clone(&edge_schema),
        Arc::clone(&nullable_pair_schema),
        edge_batch.num_rows(),
        nullable_pairs.num_rows(),
        false,
    );
    let nullable_admitted = context.admit_plan(&nullable_plan).unwrap();
    let nullable_runtime = Arc::new(nullable_admitted.runtime().unwrap());
    let null_error = match execute_similarity_plan(
        &nullable_admitted,
        &nullable_runtime,
        &nullable_plan,
        Arc::clone(&edge_schema),
        &edge_batch,
        Arc::clone(&nullable_pair_schema),
        &nullable_pairs,
    ) {
        Err(error) => error,
        Ok(_) => panic!("null vertex-pair value must reject execution"),
    };
    assert_eq!(null_error.code(), QueryErrorCode::GraphInputInvalid, "{null_error:#?}");
    assert_eq!(null_error.fact_value("algorithm"), Some("jaccard"));
    assert_eq!(null_error.fact_value("input_role"), Some("vertex_pairs"));
    drop(nullable_runtime);
    nullable_admitted.finish();
    drop(nullable_admitted);

    let pair_schema = similarity_pair_schema(DataType::Int32, false);
    let absent_pairs = RecordBatch::try_new(
        Arc::clone(&pair_schema),
        vec![Arc::new(Int32Array::from(vec![99])), Arc::new(Int32Array::from(vec![0]))],
    )
    .unwrap();
    let plan = similarity_plan(
        GraphAlgorithm::Jaccard,
        Arc::clone(&edge_schema),
        Arc::clone(&pair_schema),
        edge_batch.num_rows(),
        absent_pairs.num_rows(),
        false,
    );
    let absent_admitted = context.admit_plan(&plan).unwrap();
    let absent_runtime = Arc::new(absent_admitted.runtime().unwrap());
    let absent_error = match execute_similarity_plan(
        &absent_admitted,
        &absent_runtime,
        &plan,
        Arc::clone(&edge_schema),
        &edge_batch,
        Arc::clone(&pair_schema),
        &absent_pairs,
    ) {
        Err(error) => error,
        Ok(_) => panic!("vertex-pair endpoint absent from the graph must reject execution"),
    };
    assert_eq!(absent_error.code(), QueryErrorCode::CudfInvalidInput);
    drop(absent_runtime);
    absent_admitted.finish();
    drop(absent_admitted);

    let mismatched_schema = similarity_pair_schema(DataType::Int64, false);
    let mismatched_pairs = RecordBatch::try_new(
        Arc::clone(&mismatched_schema),
        vec![Arc::new(Int64Array::from(vec![0])), Arc::new(Int64Array::from(vec![1]))],
    )
    .unwrap();
    let mismatched_plan = similarity_plan(
        GraphAlgorithm::Jaccard,
        Arc::clone(&edge_schema),
        Arc::clone(&mismatched_schema),
        edge_batch.num_rows(),
        mismatched_pairs.num_rows(),
        false,
    );
    let mismatch_error = match context.admit_plan(&mismatched_plan) {
        Err(error) => error,
        Ok(_) => panic!("pair and graph vertex domains must reject before GPU execution"),
    };
    assert_eq!(mismatch_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(mismatch_error.fact_value("reason"), Some("vertex_pair_graph_vertex_dtype_mismatch"));
}
