use std::sync::Arc;

use arrow::{
    array::{Float32Array, Int64Array, UInt32Array},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf::DataFrame;
use query_engine::{
    error::QueryErrorCode,
    exec::{QueryExecutionRequest, output::QueryOutput},
    expr::Expr,
    plan::{
        ProjectionOutput, QueryPlan, QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmParams, VectorAlgorithmSpec,
        VectorBruteForceKnnParams, VectorDistanceMetric, VectorKMeansParams, VectorPcaParams, VectorRelationBinding,
        VectorRowIdBinding, vector_kmeans_output_schema, vector_knn_output_schema, vector_pca_output_schema,
    },
    runtime::{
        ExecutionContext, MemoryControlPolicy,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{GpuResidentSource, LocalParquetSource, SourceNode},
};

use super::{
    assert_attempt_domain_released, backend_context, execute_query_plan_direct, execute_query_plan_with_gpu_inputs,
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
    assert_eq!(batch.column(2).as_any().downcast_ref::<UInt32Array>().unwrap().values().as_ref(), &[0, 0]);
    assert_eq!(result.metrics().compiled_pipeline_attempts, 1);
    assert!(result.metrics().compiled_pipeline_tasks_executed >= 5);
    let task_mappings = result.metrics().plan_task_mappings();
    assert!(task_mappings.iter().any(|mapping| mapping.operator_kind.as_deref() == Some("vector_algorithm")));
    assert!(task_mappings.iter().any(|mapping| mapping.operator_kind.as_deref() == Some("unary")));
    admitted.finish();
}

#[test]
fn compiled_knn_input_failure_records_the_vector_task_and_releases_attempt_allocations() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let plan = knn_plan(Arc::clone(&dataset_schema), Arc::clone(&query_schema), VectorFixtureShape::Columns, 2);
    let dataset_batch = RecordBatch::try_new(
        Arc::clone(&dataset_schema),
        vec![Arc::new(Int64Array::from(vec![7])), Arc::new(Float32Array::from(vec![1.0]))],
    )
    .expect("single-row dataset fixture matches schema");
    let query_batch = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![Arc::new(Int64Array::from(vec![9])), Arc::new(Float32Array::from(vec![2.0]))],
    )
    .expect("query fixture matches schema");
    let context = backend_context();
    let admitted = context.admit_plan(&plan).expect("kNN plan with runtime-checked k is admitted");
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

    let (result, metrics, _observation) = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::new(&plan).with_gpu_inputs(&gpu_inputs),
    )
    .into_parts();
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("k larger than the dataset row count must fail at the vector operator"),
    };
    assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput);
    assert_eq!(error.operation(), "query_engine.vector_algorithm.execute");
    assert_eq!(error.fact_value("algorithm"), Some("brute_force_knn"));
    assert_eq!(error.fact_value("reason"), Some("k_exceeds_dataset_rows"));
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
}

#[test]
fn unbounded_vector_request_requires_compiled_pipeline_before_source_work() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/algeon-unbounded-vector-source-must-not-open.parquet",
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
    let context = ExecutionContext::builder()
        .with_runtime(cudf::Runtime::new().expect("GPU runtime for unbounded vector route test"))
        .with_memory_control_policy(MemoryControlPolicy::Unbounded)
        .build()
        .expect("unbounded execution context");

    let (result, metrics, _observation) =
        execute_query_plan_direct(&context, QueryExecutionRequest::new(&plan)).into_parts();
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("unbounded vector execution must require compiled bounded admission"),
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("compiled_vector_pipeline_required"));
    assert_eq!(metrics.compiled_pipeline_attempts, 0);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, 0);
    assert_eq!(metrics.source_read_time, std::time::Duration::ZERO);
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

    let (result, metrics, _observation) =
        execute_query_plan_direct(admitted.context.context(), QueryExecutionRequest::new(&plan)).into_parts();
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
