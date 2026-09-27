use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use arrow::{
    array::{Float32Array, Int64Array, UInt32Array},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf::DataFrame;
use query_engine::{
    admission::admit,
    error::{QueryError, QueryErrorCode},
    exec::{QueryExecutionRequest, admit_query_plan, output::QueryOutput, set_vector_after_index_build_hook_for_tests},
    expr::{Expr, SortExpr},
    mode::StatisticalAggregateExecutionMode,
    plan::{
        ProjectionOutput, QueryPlan, QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmParams, VectorAlgorithmSpec,
        VectorBruteForceKnnParams, VectorCagraParams, VectorDistanceMetric, VectorIvfFlatParams, VectorKMeansParams,
        VectorPairwiseDistanceParams, VectorPcaParams, VectorRelationBinding, VectorRowIdBinding, WindowFunction,
        WindowSpec, vector_kmeans_output_schema, vector_knn_output_schema, vector_pairwise_distance_output_schema,
        vector_pca_output_schema,
    },
    policy::PlanningPolicy,
    rapids_backend::rapids_analysis_context,
    runtime::{
        ExecutionContext,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{GpuResidentSource, LocalParquetSource, SourceNode},
};

use super::{
    assert_attempt_domain_released, backend_context, backend_context_with_grant, execute_query_plan_direct,
    execute_query_plan_with_gpu_inputs,
};

#[derive(Clone, Copy)]
enum VectorFixtureShape {
    Columns,
    FixedSizeList,
    UniformList,
}

#[test]
fn exact_knn_executes_gpu_resident_with_deterministic_ids_and_observed_stages() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let output_schema = vector_knn_output_schema(DataType::Int64, DataType::Int64);
    let mut builder = QueryPlanBuilder::new();
    let dataset =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset", Arc::clone(&dataset_schema))));
    let dataset = builder.add_projection(
        dataset,
        dataset_schema
            .fields()
            .iter()
            .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.as_ref().clone()))
            .collect(),
    );
    let queries =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("queries", Arc::clone(&query_schema))));
    let params = VectorBruteForceKnnParams::new(
        VectorRelationBinding::new(dataset, VectorRowIdBinding::column("id"), ["x"]),
        VectorRelationBinding::new(queries, VectorRowIdBinding::column("query_id"), ["x"]),
        1,
        VectorDistanceMetric::L2Expanded,
    );
    let vector = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::BruteForceKnn,
        VectorAlgorithmParams::BruteForceKnn(params),
        output_schema,
    ));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::Int64, false),
        Field::new("neighbor_id", DataType::Int64, false),
        Field::new("rank", DataType::UInt32, false),
    ]));
    let output = builder.add_projection(
        vector,
        vec![
            ProjectionOutput::new(Expr::column("query_id"), output_schema.field(0).clone()),
            ProjectionOutput::new(Expr::column("neighbor_id"), output_schema.field(1).clone()),
            ProjectionOutput::new(Expr::column("rank"), output_schema.field(2).clone()),
        ],
    );
    let plan = builder.finish_with_host_sink(output);

    let dataset_batch = RecordBatch::try_new(
        Arc::clone(&dataset_schema),
        vec![Arc::new(Int64Array::from(vec![10, 20, 30])), Arc::new(Float32Array::from(vec![0.0, 3.0, 8.0]))],
    )
    .expect("dataset fixture matches schema");
    let query_batch = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![Arc::new(Int64Array::from(vec![101, 102])), Arc::new(Float32Array::from(vec![2.0, 7.0]))],
    )
    .expect("query fixture matches schema");
    let context = backend_context();
    let admitted = context.admit_plan(&plan).expect("composed kNN plan is admitted");
    let runtime = Arc::new(admitted.runtime().expect("query runtime is available"));
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "dataset".into(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &dataset_batch).expect("dataset upload")),
            Arc::clone(&dataset_schema),
        )
        .expect("dataset resident owner"),
    );
    gpu_inputs.insert(
        "queries".into(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &query_batch).expect("query upload")),
            Arc::clone(&query_schema),
        )
        .expect("query resident owner"),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs)
        .expect("bounded compiled kNN execution succeeds");
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap().values().as_ref(), &[101, 102]);
    assert_eq!(batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap().values().as_ref(), &[20, 30]);
    assert_eq!(batch.column(2).as_any().downcast_ref::<UInt32Array>().unwrap().values().as_ref(), &[1, 1]);
    assert_eq!(result.metrics().compiled_pipeline_attempts, 1);
    assert!(result.metrics().compiled_pipeline_tasks_executed >= 5);
    let task_mappings = result.metrics().plan_task_mappings();
    assert!(task_mappings.iter().any(|mapping| mapping.operator_kind.as_deref() == Some("vector_algorithm")));
    assert!(task_mappings.iter().any(|mapping| mapping.operator_kind.as_deref() == Some("unary")));
    admitted.finish();
}

#[test]
fn ivf_flat_executes_with_contiguous_ranks_and_structured_cardinality_errors() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let dataset_batch = RecordBatch::try_new(
        Arc::clone(&dataset_schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5])),
            Arc::new(Float32Array::from(vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2])),
        ],
    )
    .unwrap();
    let query_batch = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![Arc::new(Int64Array::from(vec![99])), Arc::new(Float32Array::from(vec![0.0]))],
    )
    .unwrap();
    for (k, n_lists, n_probes, reason) in
        [(5, 2, 1, None), (1, 7, 1, Some("n_lists_exceeds_dataset_rows")), (7, 2, 1, Some("k_exceeds_dataset_rows"))]
    {
        let mut builder = QueryPlanBuilder::new();
        let dataset =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset", Arc::clone(&dataset_schema))));
        let queries =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("queries", Arc::clone(&query_schema))));
        let mut params = VectorIvfFlatParams::new(
            VectorRelationBinding::new(dataset, VectorRowIdBinding::column("id"), ["x"]),
            VectorRelationBinding::new(queries, VectorRowIdBinding::column("query_id"), ["x"]),
            k,
            VectorDistanceMetric::L2Expanded,
            n_lists,
        );
        params.n_probes = n_probes;
        let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
            VectorAlgorithm::IvfFlat,
            VectorAlgorithmParams::IvfFlat(params),
            vector_knn_output_schema(DataType::Int64, DataType::Int64),
        ));
        let plan = builder.finish_with_host_sink(root);
        let context = backend_context();
        let admitted = context.admit_plan(&plan).expect("IVF-Flat plan admission");
        let runtime = Arc::new(admitted.runtime().unwrap());
        let mut gpu_inputs = GpuResidentInputMap::new();
        for (name, schema, batch) in
            [("dataset", &dataset_schema, &dataset_batch), ("queries", &query_schema, &query_batch)]
        {
            gpu_inputs.insert(
                name.into(),
                GpuResultHandle::new(
                    &admitted,
                    GpuDataFrame::new(DataFrame::from_record_batch(&runtime, batch).unwrap()),
                    Arc::clone(schema),
                )
                .unwrap(),
            );
        }
        let admitted_plan =
            admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy())
                .unwrap();
        let (result, _, _) = execute_query_plan_direct(
            admitted.context.context(),
            QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&gpu_inputs),
        )
        .into_parts();
        match reason {
            Some(reason) => {
                let error = match result {
                    Ok(_) => panic!("invalid evaluated dataset cardinality"),
                    Err(error) => error,
                };
                assert_eq!(error.fact_value("reason"), Some(reason));
            },
            None => {
                let QueryOutput::HostArrow(batch) = result.expect("IVF-Flat execution") else { panic!("host result") };
                assert!((1..5).contains(&batch.num_rows()));
                let ranks = batch.column(4).as_any().downcast_ref::<UInt32Array>().unwrap();
                assert_eq!(ranks.values().as_ref(), &(1..=batch.num_rows() as u32).collect::<Vec<_>>());
            },
        }
        drop(gpu_inputs);
        drop(runtime);
        assert_attempt_domain_released(&admitted);
        admitted.finish();
    }
}

static DOMAIN_BYTES_AFTER_INDEX_BUILD: AtomicU64 = AtomicU64::new(0);

/// Installs the after-index-build hook for one test and removes it on drop, so a
/// failed assertion cannot leave the process-wide hook set.
struct AfterIndexBuildHook;

impl AfterIndexBuildHook {
    fn install(hook: fn(&ExecutionContext)) -> Self {
        set_vector_after_index_build_hook_for_tests(Some(hook));
        Self
    }
}

impl Drop for AfterIndexBuildHook {
    fn drop(&mut self) {
        set_vector_after_index_build_hook_for_tests(None);
    }
}

fn record_domain_bytes_and_cancel(context: &ExecutionContext) {
    let usage = context
        .attempt_allocation_domain_usage()
        .expect("allocation-domain usage samples")
        .expect("admitted context owns one allocation domain");
    DOMAIN_BYTES_AFTER_INDEX_BUILD.store(usage.current_bytes, Ordering::SeqCst);
    context.cancellation().cancel();
}

#[test]
fn ivf_flat_cancellation_after_index_build_releases_query_allocations() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let dataset_batch = RecordBatch::try_new(
        Arc::clone(&dataset_schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5])),
            Arc::new(Float32Array::from(vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2])),
        ],
    )
    .unwrap();
    let query_batch = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![Arc::new(Int64Array::from(vec![99])), Arc::new(Float32Array::from(vec![0.0]))],
    )
    .unwrap();
    let mut builder = QueryPlanBuilder::new();
    let dataset =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset", Arc::clone(&dataset_schema))));
    let queries =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("queries", Arc::clone(&query_schema))));
    let mut params = VectorIvfFlatParams::new(
        VectorRelationBinding::new(dataset, VectorRowIdBinding::column("id"), ["x"]),
        VectorRelationBinding::new(queries, VectorRowIdBinding::column("query_id"), ["x"]),
        1,
        VectorDistanceMetric::L2Expanded,
        2,
    );
    params.n_probes = 1;
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::IvfFlat,
        VectorAlgorithmParams::IvfFlat(params),
        vector_knn_output_schema(DataType::Int64, DataType::Int64),
    ));
    let plan = builder.finish_with_host_sink(root);
    let context = backend_context();
    let admitted = context.admit_plan(&plan).expect("IVF-Flat plan admission");
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    for (name, schema, batch) in
        [("dataset", &dataset_schema, &dataset_batch), ("queries", &query_schema, &query_batch)]
    {
        gpu_inputs.insert(
            name.into(),
            GpuResultHandle::new(
                &admitted,
                GpuDataFrame::new(DataFrame::from_record_batch(&runtime, batch).unwrap()),
                Arc::clone(schema),
            )
            .unwrap(),
        );
    }
    let domain_bytes = |context: &ExecutionContext| {
        context
            .attempt_allocation_domain_usage()
            .expect("allocation-domain usage samples")
            .expect("admitted context owns one allocation domain")
            .current_bytes
    };
    let bytes_before = domain_bytes(&admitted);

    let hook = AfterIndexBuildHook::install(record_domain_bytes_and_cancel);
    let admitted_plan =
        admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy()).unwrap();
    let (result, _, _) = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&gpu_inputs),
    )
    .into_parts();
    drop(hook);

    let error = result.err().expect("cancelled after the index build");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.operation(), "query_engine.vector_algorithm.after_index_build");
    assert!(
        DOMAIN_BYTES_AFTER_INDEX_BUILD.load(Ordering::SeqCst) > bytes_before,
        "the index build allocates inside the attempt domain"
    );
    assert_eq!(domain_bytes(&admitted), bytes_before, "the cancelled build releases its index allocations");
    drop(gpu_inputs);
    drop(runtime);
    assert_attempt_domain_released(&admitted);
    admitted.finish();
}

#[test]
fn compiled_knn_input_failure_records_the_vector_task_and_releases_attempt_allocations() {
    let error = execute_failing_columns_knn(vec![1.0], 2);
    assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput);
    assert_eq!(error.operation(), "query_engine.vector_algorithm.execute");
    assert_eq!(error.fact_value("algorithm"), Some("brute_force_knn"));
    assert_eq!(error.fact_value("reason"), Some("k_exceeds_dataset_rows"));
}

#[test]
fn compiled_knn_non_finite_input_projects_cudf_validation_details() {
    let error = execute_failing_columns_knn(vec![1.0, f32::NAN, f32::INFINITY], 1);
    assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput);
    assert_eq!(error.operation(), "query_engine.vector_algorithm.pack");
    assert_eq!(error.fact_value("relation_role"), Some("dataset"));
    assert_eq!(error.fact_value("validation_reason"), Some("non_finite_vector_value"));
    assert_eq!(error.fact_value("non_finite_count"), Some("2"));
    assert!(error.fact_value("remedy").is_some_and(|remedy| remedy.contains("NaN")));
}

/// Runs a single-column kNN whose failure must surface at the vector operator
/// and checks that the attempt records the vector task and releases its domain.
fn execute_failing_columns_knn(dataset_x: Vec<f32>, k: usize) -> QueryError {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let plan = knn_plan(Arc::clone(&dataset_schema), Arc::clone(&query_schema), VectorFixtureShape::Columns, k);
    let dataset_ids = (7..).take(dataset_x.len()).collect::<Vec<i64>>();
    let dataset_batch = RecordBatch::try_new(
        Arc::clone(&dataset_schema),
        vec![Arc::new(Int64Array::from(dataset_ids)), Arc::new(Float32Array::from(dataset_x))],
    )
    .expect("dataset fixture matches schema");
    let query_batch = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![Arc::new(Int64Array::from(vec![9])), Arc::new(Float32Array::from(vec![2.0]))],
    )
    .expect("query fixture matches schema");
    let context = backend_context();
    let admitted = context.admit_plan(&plan).expect("kNN plan with runtime-checked inputs is admitted");
    let runtime = Arc::new(admitted.runtime().expect("query runtime is available"));
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "dataset".into(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &dataset_batch).expect("dataset upload")),
            Arc::clone(&dataset_schema),
        )
        .expect("dataset resident owner"),
    );
    gpu_inputs.insert(
        "queries".into(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &query_batch).expect("query upload")),
            Arc::clone(&query_schema),
        )
        .expect("query resident owner"),
    );

    let admitted_plan =
        admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy()).unwrap();
    let (result, metrics, _observation) = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&gpu_inputs),
    )
    .into_parts();
    let metrics = metrics.snapshot();
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("invalid kNN input must fail at the vector operator"),
    };
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(
        metrics
            .plan_task_mappings()
            .iter()
            .any(|mapping| { mapping.operator_kind.as_deref() == Some("vector_algorithm") })
    );
    drop(gpu_inputs);
    drop(runtime);
    assert_attempt_domain_released(&admitted);
    admitted.finish();
    error
}

#[test]
fn compiled_pairwise_output_allocation_failure_records_task_and_releases_attempt() {
    let schema = vector_schema("id", VectorFixtureShape::Columns);
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("x", Arc::clone(&schema))));
    let y = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("y", Arc::clone(&schema))));
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::PairwiseDistance,
        VectorAlgorithmParams::PairwiseDistance(VectorPairwiseDistanceParams {
            x: VectorRelationBinding::new(x, VectorRowIdBinding::column("id"), ["x"]),
            y: VectorRelationBinding::new(y, VectorRowIdBinding::column("id"), ["x"]),
            metric: VectorDistanceMetric::L2Expanded,
        }),
        vector_pairwise_distance_output_schema(DataType::Int64, DataType::Int64, DataType::Float32),
    ));
    let plan = builder.finish_with_host_sink(root);
    let ids = (0..256_i64).collect::<Vec<_>>();
    let features = (0..256).map(|index| index as f32).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(ids)), Arc::new(Float32Array::from(features))],
    )
    .unwrap();
    let context = backend_context_with_grant(64 * 1024_u64);
    let admitted = context.admit_plan(&plan).expect("pairwise plan admission");
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    for name in ["x", "y"] {
        gpu_inputs.insert(
            name.into(),
            GpuResultHandle::new(
                &admitted,
                GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).unwrap()),
                Arc::clone(&schema),
            )
            .unwrap(),
        );
    }
    let admitted_plan =
        admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy()).unwrap();
    let (result, metrics, _observation) = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&gpu_inputs),
    )
    .into_parts();
    let metrics = metrics.snapshot();
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("pairwise N*M allocation must exceed query cap"),
    };
    assert_eq!(error.code(), QueryErrorCode::DeviceAllocationFailed, "{error:?}");
    assert!(
        metrics.plan_task_mappings().iter().any(|mapping| mapping.operator_kind.as_deref() == Some("vector_algorithm"))
    );
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    drop(gpu_inputs);
    drop(runtime);
    assert_attempt_domain_released(&admitted);
    admitted.finish();
}

#[test]
fn compiled_cagra_build_allocation_failure_records_task_and_releases_attempt() {
    let schema = vector_schema("id", VectorFixtureShape::Columns);
    let mut builder = QueryPlanBuilder::new();
    let dataset = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset", Arc::clone(&schema))));
    let queries = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("queries", Arc::clone(&schema))));
    let mut params = VectorCagraParams::new(
        VectorRelationBinding::new(dataset, VectorRowIdBinding::column("id"), ["x"]),
        VectorRelationBinding::new(queries, VectorRowIdBinding::column("id"), ["x"]),
        1,
        VectorDistanceMetric::L2Expanded,
    );
    params.graph_degree = 16;
    params.intermediate_graph_degree = 32;
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Cagra,
        VectorAlgorithmParams::Cagra(params),
        vector_knn_output_schema(DataType::Int64, DataType::Int64),
    ));
    let plan = builder.finish_with_host_sink(root);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from((0..128_i64).collect::<Vec<_>>())),
            Arc::new(Float32Array::from((0..128).map(|value| value as f32).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    let context = backend_context_with_grant(64 * 1024_u64);
    let admitted = context.admit_plan(&plan).expect("CAGRA plan admission");
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    for name in ["dataset", "queries"] {
        gpu_inputs.insert(
            name.into(),
            GpuResultHandle::new(
                &admitted,
                GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).unwrap()),
                Arc::clone(&schema),
            )
            .unwrap(),
        );
    }
    let admitted_plan =
        admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy()).unwrap();
    let (result, metrics, _observation) = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&gpu_inputs),
    )
    .into_parts();
    let metrics = metrics.snapshot();
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("CAGRA build must exceed the query cap"),
    };
    assert_eq!(error.code(), QueryErrorCode::DeviceAllocationFailed, "{error:?}");
    assert_eq!(error.fact_value("cuvs_operation"), Some("CagraIndex::build"));
    assert!(
        metrics.plan_task_mappings().iter().any(|mapping| mapping.operator_kind.as_deref() == Some("vector_algorithm"))
    );
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    drop(gpu_inputs);
    drop(runtime);
    assert_attempt_domain_released(&admitted);
    admitted.finish();
}

#[test]
fn bounded_vector_rejection_reports_the_contract_it_misses() {
    // No compiled contract covers a row number over PCA output; the error
    // carries that decision's own rejection instead of a generic vector reason.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/algeon-rejected-vector-source-must-not-open.parquet",
        Arc::clone(&schema),
        vec!["id".into(), "x".into(), "y".into()],
    )));
    let pca_schema = vector_pca_output_schema(DataType::Int64, 1);
    let vector = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(VectorPcaParams::new(
            VectorRelationBinding::new(source, VectorRowIdBinding::column("id"), ["x", "y"]),
            1,
        )),
        Arc::clone(&pca_schema),
    ));
    let window_schema = Arc::new(Schema::new(
        pca_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain([Field::new("row_number", DataType::Int64, false)])
            .collect::<Vec<_>>(),
    ));
    let key = pca_schema.field(0).name().clone();
    let window = builder.add_window(
        vector,
        WindowSpec::new(
            Vec::new(),
            vec![SortExpr::new(Expr::column(key), false, false)],
            vec![WindowFunction::row_number_i64("row_number")],
        ),
        window_schema,
    );
    let plan = builder.finish_with_host_sink(window);
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("the plan is decidable");
    let expected = decision.rejection_projection().expect("no compiled contract covers the plan").reason;

    let error = backend_context().admit_plan(&plan).err().expect("a rejected vector plan fails before any work");
    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("compiled_pipeline_rejection_reason"), Some(expected));
}

#[cfg(not(feature = "cuvs"))]
#[test]
fn cuvs_disabled_rejects_before_local_source_staging() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/algeon-cuvs-disabled-source-must-not-open.parquet",
        Arc::clone(&schema),
        vec!["id".into(), "x".into(), "y".into()],
    )));
    let vector = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(VectorPcaParams::new(
            VectorRelationBinding::new(source, VectorRowIdBinding::column("id"), ["x", "y"]),
            1,
        )),
        vector_pca_output_schema(DataType::Int64, 1),
    ));
    let plan = builder.finish_with_host_sink(vector);
    let context = backend_context();
    let admitted = context.admit_plan(&plan).expect("vector plan is structurally admitted without cuVS");

    let admitted_plan =
        admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy()).unwrap();
    let (result, metrics, _observation) =
        execute_query_plan_direct(admitted.context.context(), QueryExecutionRequest::admitted(&admitted_plan))
            .into_parts();
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("disabled cuVS must fail before opening the local Parquet source"),
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.fact_value("reason"), Some("required_feature_disabled"));
    assert_eq!(error.fact_value("required_feature"), Some("cuvs"));
    assert_eq!(metrics.compiled_pipeline_attempts, 0);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, 0);
    assert_eq!(metrics.source_read_time, std::time::Duration::ZERO);
    admitted.finish();
}

#[test]
fn exact_knn_accepts_proof_bearing_fixed_size_list_vectors() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::FixedSizeList);
    let query_schema = vector_schema("query_id", VectorFixtureShape::FixedSizeList);
    let plan = knn_plan(dataset_schema, query_schema, VectorFixtureShape::FixedSizeList, 2);
    assert_vector_capability_admitted(&plan);
}

#[test]
fn exact_knn_accepts_proof_bearing_uniform_list_vectors() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::UniformList);
    let query_schema = vector_schema("query_id", VectorFixtureShape::UniformList);
    let plan = knn_plan(dataset_schema, query_schema, VectorFixtureShape::UniformList, 2);
    assert_vector_capability_admitted(&plan);
}

#[test]
fn exact_knn_breaks_equal_distance_ties_by_physical_ordinal() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let plan = knn_plan(dataset_schema, query_schema, VectorFixtureShape::Columns, 2);
    assert_vector_capability_admitted(&plan);
}

#[test]
fn exact_knn_preserves_an_empty_query_relation_without_native_search() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let plan = knn_plan(dataset_schema, query_schema, VectorFixtureShape::Columns, 1);
    assert_vector_capability_admitted(&plan);
}

#[test]
fn exact_knn_parameter_shapes_admit_before_runtime_value_validation() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    for k in [1, 2] {
        let plan = knn_plan(Arc::clone(&dataset_schema), Arc::clone(&query_schema), VectorFixtureShape::Columns, k);
        assert_vector_capability_admitted(&plan);
    }
}

#[test]
fn kmeans_executes_fixed_size_list_input_and_centroid_relations_with_permutation_invariant_labels() {
    let schema = vector_schema("id", VectorFixtureShape::FixedSizeList);
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let centroids =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("initial_centroids", Arc::clone(&schema))));
    let params = VectorKMeansParams::new(
        VectorRelationBinding::new_vector_column(input, VectorRowIdBinding::column("id"), "embedding"),
        2,
    )
    .with_initial_centroids(VectorRelationBinding::new_vector_column(
        centroids,
        VectorRowIdBinding::column("id"),
        "embedding",
    ));
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::KMeans,
        VectorAlgorithmParams::KMeans(params),
        vector_kmeans_output_schema(DataType::Int64),
    ));
    assert_vector_capability_admitted(&builder.finish_gpu_resident(root));
}

#[test]
fn pca_executes_uniform_list_input_with_sign_aware_component_oracle() {
    let schema = vector_schema("id", VectorFixtureShape::UniformList);
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    let params = VectorPcaParams::new(
        VectorRelationBinding::new_vector_column(input, VectorRowIdBinding::column("id"), "embedding"),
        1,
    );
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(params),
        vector_pca_output_schema(DataType::Int64, 1),
    ));
    assert_vector_capability_admitted(&builder.finish_gpu_resident(root));
}

fn assert_vector_capability_admitted(plan: &QueryPlan) {
    let context = backend_context();
    context.admit_plan(plan).expect("modeled vector capability must admit under the fixed query cap").finish();
}

fn knn_plan(dataset_schema: SchemaRef, query_schema: SchemaRef, shape: VectorFixtureShape, k: usize) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let dataset = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset", dataset_schema)));
    let queries = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("queries", query_schema)));
    let (dataset_binding, query_binding) = match shape {
        VectorFixtureShape::Columns => (
            VectorRelationBinding::new(dataset, VectorRowIdBinding::column("id"), ["x"]),
            VectorRelationBinding::new(queries, VectorRowIdBinding::column("query_id"), ["x"]),
        ),
        VectorFixtureShape::FixedSizeList | VectorFixtureShape::UniformList => (
            VectorRelationBinding::new_vector_column(dataset, VectorRowIdBinding::column("id"), "embedding"),
            VectorRelationBinding::new_vector_column(queries, VectorRowIdBinding::column("query_id"), "embedding"),
        ),
    };
    let params = VectorBruteForceKnnParams::new(dataset_binding, query_binding, k, VectorDistanceMetric::L2Expanded);
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::BruteForceKnn,
        VectorAlgorithmParams::BruteForceKnn(params),
        vector_knn_output_schema(DataType::Int64, DataType::Int64),
    ));
    builder.finish_gpu_resident(root)
}

fn vector_schema(id: &str, shape: VectorFixtureShape) -> SchemaRef {
    let vector_field = match shape {
        VectorFixtureShape::Columns => Field::new("x", DataType::Float32, false),
        VectorFixtureShape::FixedSizeList => Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 1),
            false,
        ),
        VectorFixtureShape::UniformList => {
            Field::new("embedding", DataType::List(Arc::new(Field::new("item", DataType::Float32, false))), false)
        },
    };
    Arc::new(Schema::new(vec![Field::new(id, DataType::Int64, false), vector_field]))
}
