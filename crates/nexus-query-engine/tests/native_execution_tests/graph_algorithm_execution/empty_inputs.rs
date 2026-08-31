fn empty_int32_edges_batch(schema: Arc<Schema>) -> RecordBatch {
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(Vec::<Option<i32>>::new())),
            Arc::new(Int32Array::from(Vec::<Option<i32>>::new())),
        ],
    )
    .unwrap()
}

fn graph_input_handle(
    context: &ExecutionContext,
    runtime: &Arc<cudf::Runtime>,
    schema: Arc<Schema>,
    batch: &RecordBatch,
) -> GpuResultHandle {
    GpuResultHandle::new(context, GpuDataFrame::new(DataFrame::from_record_batch(runtime, batch).unwrap()), schema)
        .unwrap()
}

#[test]
fn pagerank_empty_edges_returns_a_schema_correct_empty_result_after_null_sanitization() {
    let context = backend_context();
    let edge_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Int32, true), Field::new("dst", DataType::Int32, true)]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![None])), Arc::new(Int32Array::from(vec![None]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let spec = int32_graph_spec(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::PageRank));
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs
        .insert("edges".to_string(), graph_input_handle(&admitted, &runtime, Arc::clone(&edge_schema), &edge_batch));

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else { panic!("graph result must materialize to host Arrow") };
    assert_eq!(batch.schema().as_ref(), int32_stable_graph_result_schema(GraphAlgorithm::PageRank).as_ref());
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(result.metrics().graph_algorithm_runtime_rows.len(), 1);
    let runtime_row = &result.metrics().graph_algorithm_runtime_rows[0];
    assert!(runtime_row.algorithm_metadata.contains("empty_result_reason=edge_source_empty"));
    assert!(runtime_row.algorithm_metadata.contains("empty_edge_input_policy=return_empty"));
    assert!(!runtime_row.algorithm_metadata.contains("graph_cache="));
    assert!(result.metrics().graph_reservation_rows.is_empty());
    admitted.finish();
}

#[test]
fn bfs_strict_rejects_empty_edges_as_a_permanent_graph_input_error() {
    let context = backend_context();
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = empty_int32_edges_batch(Arc::clone(&edge_schema));

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs));
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs
        .insert("edges".to_string(), graph_input_handle(&admitted, &runtime, Arc::clone(&edge_schema), &edge_batch));

    let error = match execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs) {
        Ok(_) => panic!("strict BFS must reject an empty graph"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::GraphInputInvalid);
    assert_eq!(error.status(), nexus_query_engine::error::QueryErrorStatus::Permanent);
    assert_eq!(error.fact_value("algorithm"), Some("bfs"));
    assert_eq!(error.fact_value("input_role"), Some("edge"));
    assert_eq!(error.fact_value("reason"), Some("bfs_source_requires_nonempty_graph"));
    admitted.finish();
}

fn assert_personalized_pagerank_rejects_empty_required_side_input(context: &BackendQueryContext) {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let side_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0])), Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    let side_batch = RecordBatch::try_new(
        Arc::clone(&side_schema),
        vec![Arc::new(Int32Array::from(Vec::<i32>::new())), Arc::new(Float64Array::from(Vec::<f64>::new()))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let personalization = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "personalization",
        Arc::clone(&side_schema),
        &side_batch,
    )));
    let spec = int32_graph_spec(
        GraphAlgorithm::PersonalizedPageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
        GraphAlgorithmParams::PersonalizedPageRank(GraphPersonalizedPageRankParams {
            pagerank: GraphPageRankParams::default(),
            personalization: GraphPageRankPersonalizationInput::new(personalization, "vertex", "value"),
        }),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan =
        builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::PersonalizedPageRank));
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert("edges".to_string(), graph_input_handle(&admitted, &runtime, Arc::clone(&edge_schema), &edge_batch));
    gpu_inputs.insert(
        "personalization".to_string(),
        graph_input_handle(&admitted, &runtime, Arc::clone(&side_schema), &side_batch),
    );

    let error = match execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs) {
        Ok(_) => panic!("personalized PageRank must reject an empty personalization input"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::GraphInputInvalid);
    assert_eq!(error.status(), nexus_query_engine::error::QueryErrorStatus::Permanent);
    assert_eq!(error.fact_value("algorithm"), Some("personalized_pagerank"));
    assert_eq!(error.fact_value("input_role"), Some("personalization"));
    assert_eq!(error.fact_value("side_input_rows"), Some("0"));
    assert_eq!(error.fact_value("reason"), Some("required_side_input_empty"));
    admitted.finish();
}

#[test]
fn personalized_pagerank_rejects_an_empty_required_side_input_before_graph_execution() {
    let context = backend_context();

    assert_personalized_pagerank_rejects_empty_required_side_input(&context);
}

#[test]
fn bounded_personalized_pagerank_rejects_empty_personalization_with_nonempty_edges() {
    let context = backend_context_with_grant(16 * 1024 * 1024);

    assert_personalized_pagerank_rejects_empty_required_side_input(&context);
}

#[test]
fn bfs_empty_edges_with_nonempty_vertex_predicate_is_a_graph_input_error() {
    let context = backend_context();
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let predicate_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int32, false)]));
    let edge_batch = empty_int32_edges_batch(Arc::clone(&edge_schema));
    let predicate_batch =
        RecordBatch::try_new(Arc::clone(&predicate_schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(&edge_schema), &edge_batch)));
    let predicate = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
        "include_vertices",
        Arc::clone(&predicate_schema),
        &predicate_batch,
    )));
    let mut params = GraphBfsParams::new(0, None);
    params.predicates.include_vertices = Some(GraphPredicateInput::new(predicate, "vertex"));
    let spec = int32_graph_spec(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(GraphAlgorithm::Bfs));
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs
        .insert("edges".to_string(), graph_input_handle(&admitted, &runtime, Arc::clone(&edge_schema), &edge_batch));
    gpu_inputs.insert(
        "include_vertices".to_string(),
        graph_input_handle(&admitted, &runtime, Arc::clone(&predicate_schema), &predicate_batch),
    );

    let error = match execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs) {
        Ok(_) => panic!("BFS predicate values cannot match an empty graph"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::GraphInputInvalid);
    assert_eq!(error.status(), nexus_query_engine::error::QueryErrorStatus::Permanent);
    assert_eq!(error.fact_value("reason"), Some("predicate_vertices_absent"));
    assert_eq!(error.fact_value("predicate_role"), Some("include_vertices"));
    assert_eq!(error.fact_value("predicate_rows"), Some("1"));
    assert_eq!(error.fact_value("valid_predicate_rows"), Some("0"));
    admitted.finish();
}
