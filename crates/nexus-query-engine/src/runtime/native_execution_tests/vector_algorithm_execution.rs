use std::sync::Arc;

use arrow::{
    array::{FixedSizeListArray, Float32Array, Int32Array, Int64Array, ListArray, UInt32Array, UInt64Array},
    buffer::{OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf::{DataFrame, Runtime};
use nexus_query_engine::{
    error::{QueryError, QueryErrorCode},
    exec::{
        execute_query_plan_with_gpu_inputs, execute_query_plan_with_gpu_inputs_observed,
        materialize_gpu_result_to_host, output::QueryOutput,
    },
    metrics::QueryMetrics,
    observability::{
        RuntimeObservationConfig, RuntimeObservationPayload, RuntimeOperatorStageKind, RuntimeOperatorStageObservation,
        RuntimeOperatorStageState,
    },
    plan::{
        QueryPlan, QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmParams, VectorAlgorithmSpec,
        VectorBruteForceKnnParams, VectorDistanceMetric, VectorKMeansParams, VectorPcaParams, VectorRelationBinding,
        VectorRowIdBinding, vector_brute_force_knn_output_schema, vector_kmeans_output_schema,
        vector_pca_output_schema,
    },
    runtime::{
        ExecutionContext,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{GpuResidentSource, SourceNode},
};

#[derive(Clone, Copy)]
enum VectorFixtureShape {
    Columns,
    FixedSizeList,
    UniformList,
}

#[test]
fn exact_knn_executes_gpu_resident_with_deterministic_ids_and_observed_stages() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).expect("execution context");
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let plan = knn_plan(Arc::clone(&dataset_schema), Arc::clone(&query_schema), VectorFixtureShape::Columns, 2);
    let inputs = wide_inputs(
        &context,
        &runtime,
        Arc::clone(&dataset_schema),
        Arc::clone(&query_schema),
        &[10, 10, 20],
        &[0.0, 1.0, 2.0],
        &[7, 7],
        &[0.9, 1.8],
    );

    let outcome = execute_query_plan_with_gpu_inputs_observed(
        &context,
        &plan,
        &inputs,
        RuntimeObservationConfig::native_execution(),
    );
    assert_eq!(outcome.metrics().boundary.device_to_host_materializations, 0);
    let snapshot = outcome.observation().expect("observed execution retains a sealed snapshot");
    let finished = finished_vector_stages(snapshot.events());
    assert_vector_stage_contract(&finished, runtime.device().ordinal());

    let (result, metrics, _) = outcome.into_parts();
    assert_eq!(metrics.boundary.device_to_host_materializations, 0);
    let QueryOutput::GpuResident(handle) = result.expect("exact kNN executes") else {
        panic!("GPU-resident sink should preserve the device output");
    };
    let mut materialization_metrics = QueryMetrics::default();
    let host = materialize_gpu_result_to_host(handle, &mut materialization_metrics).expect("test-only inspection");
    assert_eq!(materialization_metrics.boundary.device_to_host_materializations, 1);
    assert_knn_rows(&host, &[1, 0, 2, 1], &[10, 10, 20, 10]);
}

#[test]
fn exact_knn_accepts_proof_bearing_fixed_size_list_vectors() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).expect("execution context");
    let dataset_schema = vector_schema("id", VectorFixtureShape::FixedSizeList);
    let query_schema = vector_schema("query_id", VectorFixtureShape::FixedSizeList);
    let plan = knn_plan(Arc::clone(&dataset_schema), Arc::clone(&query_schema), VectorFixtureShape::FixedSizeList, 2);
    let inputs = fixed_size_list_inputs(
        &context,
        &runtime,
        Arc::clone(&dataset_schema),
        Arc::clone(&query_schema),
        &[10, 10, 20],
        &[0.0, 1.0, 2.0],
        &[7, 7],
        &[0.9, 1.8],
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).expect("FixedSizeList kNN executes");
    assert_eq!(result.metrics().boundary.device_to_host_materializations, 0);
    let (output, _) = result.into_parts();
    let QueryOutput::GpuResident(handle) = output else {
        panic!("GPU-resident sink should preserve the device output");
    };
    let mut materialization_metrics = QueryMetrics::default();
    let host = materialize_gpu_result_to_host(handle, &mut materialization_metrics).expect("test-only inspection");
    assert_knn_rows(&host, &[1, 0, 2, 1], &[10, 10, 20, 10]);
}

#[test]
fn exact_knn_accepts_proof_bearing_uniform_list_vectors() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).expect("execution context");
    let dataset_schema = vector_schema("id", VectorFixtureShape::UniformList);
    let query_schema = vector_schema("query_id", VectorFixtureShape::UniformList);
    let plan = knn_plan(Arc::clone(&dataset_schema), Arc::clone(&query_schema), VectorFixtureShape::UniformList, 2);
    let inputs = uniform_list_inputs(
        &context,
        &runtime,
        Arc::clone(&dataset_schema),
        Arc::clone(&query_schema),
        &[10, 10, 20],
        &[0.0, 1.0, 2.0],
        &[7, 7],
        &[0.9, 1.8],
    );

    let outcome = execute_query_plan_with_gpu_inputs_observed(
        &context,
        &plan,
        &inputs,
        RuntimeObservationConfig::native_execution(),
    );
    let snapshot = outcome.observation().expect("uniform List execution retains a sealed snapshot");
    assert_vector_stage_contract(&finished_vector_stages(snapshot.events()), runtime.device().ordinal());
    let (result, metrics, _) = outcome.into_parts();
    assert_eq!(metrics.boundary.device_to_host_materializations, 0);
    let QueryOutput::GpuResident(handle) = result.expect("uniform List kNN executes") else {
        panic!("GPU-resident sink should preserve the device output");
    };
    let mut materialization_metrics = QueryMetrics::default();
    let host = materialize_gpu_result_to_host(handle, &mut materialization_metrics).expect("test-only inspection");
    assert_knn_rows(&host, &[1, 0, 2, 1], &[10, 10, 20, 10]);
}

#[test]
fn exact_knn_breaks_equal_distance_ties_by_physical_ordinal() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).expect("execution context");
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let plan = knn_plan(Arc::clone(&dataset_schema), Arc::clone(&query_schema), VectorFixtureShape::Columns, 2);
    let inputs = wide_inputs(&context, &runtime, dataset_schema, query_schema, &[10, 20], &[0.0, 1.0], &[7], &[0.5]);

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).expect("tied exact kNN executes");
    assert_eq!(result.metrics().boundary.device_to_host_materializations, 0);
    let (output, _) = result.into_parts();
    let QueryOutput::GpuResident(handle) = output else {
        panic!("GPU-resident sink should preserve the device output");
    };
    let mut materialization_metrics = QueryMetrics::default();
    let host = materialize_gpu_result_to_host(handle, &mut materialization_metrics).expect("test-only inspection");
    let ordinals = host.column(2).as_any().downcast_ref::<UInt64Array>().expect("neighbor ordinal");
    let ids = host.column(3).as_any().downcast_ref::<Int64Array>().expect("neighbor id");
    assert_eq!(ordinals.values().as_ref(), &[0, 1]);
    assert_eq!(ids.values().as_ref(), &[10, 20]);
}

#[test]
fn exact_knn_preserves_an_empty_query_relation_without_native_search() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).expect("execution context");
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let plan = knn_plan(Arc::clone(&dataset_schema), Arc::clone(&query_schema), VectorFixtureShape::Columns, 1);
    let inputs = wide_inputs(&context, &runtime, dataset_schema, query_schema, &[10], &[0.0], &[], &[]);

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).expect("empty queries are valid");
    assert_eq!(result.metrics().boundary.device_to_host_materializations, 0);
    let (output, _) = result.into_parts();
    let QueryOutput::GpuResident(handle) = output else {
        panic!("GPU-resident sink should preserve the empty device output");
    };
    assert_eq!(handle.frame().num_rows(), 0);
    assert_eq!(handle.frame().num_columns(), 6);
}

#[test]
fn exact_knn_rejects_non_finite_features_and_k_above_dataset_rows() {
    let non_finite = execute_wide_error(&[10], &[f32::NAN], &[7], &[0.0], 1);
    assert_eq!(non_finite.code(), QueryErrorCode::CudfInvalidInput);
    assert_eq!(non_finite.operation(), "nexus_query_engine.vector_algorithm.pack");
    assert_eq!(non_finite.fact_value("relation_role"), Some("dataset"));

    let excessive_k = execute_wide_error(&[10], &[0.0], &[7], &[0.0], 2);
    assert_eq!(excessive_k.code(), QueryErrorCode::CudfInvalidInput);
    assert_eq!(excessive_k.fact_value("reason"), Some("k_exceeds_dataset_rows"));
    assert_eq!(excessive_k.fact_value("dataset_rows"), Some("1"));

    let empty_dataset = execute_wide_error(&[], &[], &[7], &[0.0], 1);
    assert_eq!(empty_dataset.code(), QueryErrorCode::CudfInvalidInput);
    assert_eq!(empty_dataset.fact_value("reason"), Some("empty_dataset"));
}

#[test]
fn kmeans_executes_fixed_size_list_input_and_centroid_relations_with_permutation_invariant_labels() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).expect("execution context");
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
    let plan = builder.finish_gpu_resident(root);
    let input = fixed_size_list_batch(&schema, &[10, 11, 20, 21], &[0.0, 0.1, 10.0, 10.1]);
    let centroids = fixed_size_list_batch(&schema, &[100, 200], &[0.0, 10.0]);
    let inputs = gpu_inputs(
        &context,
        &runtime,
        [("input", Arc::clone(&schema), input), ("initial_centroids", schema, centroids)],
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).expect("K-means executes");
    let (output, metrics) = result.into_parts();
    assert_eq!(metrics.boundary.device_to_host_materializations, 0);
    let QueryOutput::GpuResident(handle) = output else {
        panic!("GPU-resident sink should preserve K-means output");
    };
    let mut materialization_metrics = QueryMetrics::default();
    let host = materialize_gpu_result_to_host(handle, &mut materialization_metrics).expect("test-only inspection");
    assert_kmeans_partition(&host, &[10, 11, 20, 21]);
}

#[test]
fn pca_executes_uniform_list_input_with_sign_aware_component_oracle() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).expect("execution context");
    let schema = vector_schema("id", VectorFixtureShape::UniformList);
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let params = VectorPcaParams::new(
        VectorRelationBinding::new_vector_column(input, VectorRowIdBinding::column("id"), "embedding"),
        1,
    );
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(params),
        vector_pca_output_schema(DataType::Int64, 1),
    ));
    let plan = builder.finish_gpu_resident(root);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![10_i64, 11, 12, 13])),
            Arc::new(uniform_list_array(
                Arc::new(Field::new("item", DataType::Float32, false)),
                &[-2.0, 0.0, -1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                2,
            )),
        ],
    )
    .expect("uniform List PCA input fixture");
    let inputs = gpu_inputs(&context, &runtime, [("input", schema, batch)]);

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).expect("PCA executes");
    let (output, metrics) = result.into_parts();
    assert_eq!(metrics.boundary.device_to_host_materializations, 0);
    let QueryOutput::GpuResident(handle) = output else {
        panic!("GPU-resident sink should preserve PCA output");
    };
    let mut materialization_metrics = QueryMetrics::default();
    let host = materialize_gpu_result_to_host(handle, &mut materialization_metrics).expect("test-only inspection");
    assert_pca_component_matches_centered_input_up_to_sign(&host, &[10, 11, 12, 13], &[-2.0, -1.0, 1.0, 2.0]);
}

fn execute_wide_error(
    dataset_ids: &[i64],
    dataset_values: &[f32],
    query_ids: &[i64],
    query_values: &[f32],
    k: usize,
) -> QueryError {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).expect("execution context");
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let plan = knn_plan(Arc::clone(&dataset_schema), Arc::clone(&query_schema), VectorFixtureShape::Columns, k);
    let inputs = wide_inputs(
        &context,
        &runtime,
        dataset_schema,
        query_schema,
        dataset_ids,
        dataset_values,
        query_ids,
        query_values,
    );
    match execute_query_plan_with_gpu_inputs(&context, &plan, &inputs) {
        Ok(_) => panic!("invalid exact-kNN input should fail"),
        Err(error) => error,
    }
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
        VectorFixtureShape::FixedSizeList => (
            VectorRelationBinding::new_vector_column(dataset, VectorRowIdBinding::column("id"), "embedding"),
            VectorRelationBinding::new_vector_column(queries, VectorRowIdBinding::column("query_id"), "embedding"),
        ),
        VectorFixtureShape::UniformList => (
            VectorRelationBinding::new_vector_column(dataset, VectorRowIdBinding::column("id"), "embedding"),
            VectorRelationBinding::new_vector_column(queries, VectorRowIdBinding::column("query_id"), "embedding"),
        ),
    };
    let params = VectorBruteForceKnnParams::new(dataset_binding, query_binding, k, VectorDistanceMetric::L2Expanded);
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::BruteForceKnn,
        VectorAlgorithmParams::BruteForceKnn(params),
        vector_brute_force_knn_output_schema(DataType::Int64, DataType::Int64),
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

#[allow(clippy::too_many_arguments)]
fn wide_inputs(
    context: &ExecutionContext,
    runtime: &Arc<Runtime>,
    dataset_schema: SchemaRef,
    query_schema: SchemaRef,
    dataset_ids: &[i64],
    dataset_values: &[f32],
    query_ids: &[i64],
    query_values: &[f32],
) -> GpuResidentInputMap {
    let dataset = RecordBatch::try_new(
        Arc::clone(&dataset_schema),
        vec![Arc::new(Int64Array::from(dataset_ids.to_vec())), Arc::new(Float32Array::from(dataset_values.to_vec()))],
    )
    .expect("dataset fixture");
    let queries = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![Arc::new(Int64Array::from(query_ids.to_vec())), Arc::new(Float32Array::from(query_values.to_vec()))],
    )
    .expect("query fixture");
    input_map(context, runtime, dataset_schema, query_schema, dataset, queries)
}

#[allow(clippy::too_many_arguments)]
fn fixed_size_list_inputs(
    context: &ExecutionContext,
    runtime: &Arc<Runtime>,
    dataset_schema: SchemaRef,
    query_schema: SchemaRef,
    dataset_ids: &[i64],
    dataset_values: &[f32],
    query_ids: &[i64],
    query_values: &[f32],
) -> GpuResidentInputMap {
    let dataset = fixed_size_list_batch(&dataset_schema, dataset_ids, dataset_values);
    let queries = fixed_size_list_batch(&query_schema, query_ids, query_values);
    input_map(context, runtime, dataset_schema, query_schema, dataset, queries)
}

fn fixed_size_list_batch(schema: &SchemaRef, ids: &[i64], values: &[f32]) -> RecordBatch {
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(FixedSizeListArray::new(item, 1, Arc::new(Float32Array::from(values.to_vec())), None)),
        ],
    )
    .expect("FixedSizeList fixture")
}

#[allow(clippy::too_many_arguments)]
fn uniform_list_inputs(
    context: &ExecutionContext,
    runtime: &Arc<Runtime>,
    dataset_schema: SchemaRef,
    query_schema: SchemaRef,
    dataset_ids: &[i64],
    dataset_values: &[f32],
    query_ids: &[i64],
    query_values: &[f32],
) -> GpuResidentInputMap {
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let dataset = RecordBatch::try_new(
        Arc::clone(&dataset_schema),
        vec![
            Arc::new(Int64Array::from(dataset_ids.to_vec())),
            Arc::new(uniform_list_array(Arc::clone(&item), dataset_values, 1)),
        ],
    )
    .expect("uniform List dataset fixture");
    let queries = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![Arc::new(Int64Array::from(query_ids.to_vec())), Arc::new(uniform_list_array(item, query_values, 1))],
    )
    .expect("uniform List query fixture");
    input_map(context, runtime, dataset_schema, query_schema, dataset, queries)
}

fn uniform_list_array(item: Arc<Field>, values: &[f32], dimension: usize) -> ListArray {
    assert!(dimension > 0 && values.len().is_multiple_of(dimension), "uniform List fixture shape must be valid");
    let rows = values.len() / dimension;
    let offsets = (0..=rows)
        .map(|row| {
            row.checked_mul(dimension).and_then(|offset| i32::try_from(offset).ok()).expect("fixture offsets fit i32")
        })
        .collect::<Vec<_>>();
    ListArray::new(
        item,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(Float32Array::from(values.to_vec())),
        None,
    )
}

fn input_map(
    context: &ExecutionContext,
    runtime: &Arc<Runtime>,
    dataset_schema: SchemaRef,
    query_schema: SchemaRef,
    dataset: RecordBatch,
    queries: RecordBatch,
) -> GpuResidentInputMap {
    gpu_inputs(context, runtime, [("dataset", dataset_schema, dataset), ("queries", query_schema, queries)])
}

fn gpu_inputs<const N: usize>(
    context: &ExecutionContext,
    runtime: &Arc<Runtime>,
    relations: [(&str, SchemaRef, RecordBatch); N],
) -> GpuResidentInputMap {
    let mut inputs = GpuResidentInputMap::new();
    for (name, schema, batch) in relations {
        let frame = DataFrame::from_record_batch(runtime, &batch).expect("upload test fixture");
        inputs.insert(
            name.to_string(),
            GpuResultHandle::new(context, GpuDataFrame::new(frame), schema).expect("GPU-resident fixture"),
        );
    }
    inputs
}

fn finished_vector_stages(
    events: &[nexus_query_engine::observability::RuntimeTimelineEvent],
) -> Vec<&RuntimeOperatorStageObservation> {
    events
        .iter()
        .filter_map(|event| match &event.payload {
            RuntimeObservationPayload::OperatorStage(stage)
                if matches!(stage.lifecycle, RuntimeOperatorStageState::Finished)
                    && matches!(
                        stage.stage_kind,
                        RuntimeOperatorStageKind::VectorPack
                            | RuntimeOperatorStageKind::VectorNative
                            | RuntimeOperatorStageKind::VectorAssemble
                    ) =>
            {
                Some(stage.as_ref())
            },
            _ => None,
        })
        .collect()
}

fn assert_vector_stage_contract(stages: &[&RuntimeOperatorStageObservation], device_ordinal: usize) {
    let packs = stages
        .iter()
        .copied()
        .filter(|stage| stage.stage_kind == RuntimeOperatorStageKind::VectorPack)
        .collect::<Vec<_>>();
    assert_eq!(packs.len(), 2);
    assert_eq!(
        packs.iter().map(|stage| stage.vector_relation_role.as_deref()).collect::<Vec<_>>(),
        [Some("dataset"), Some("queries")]
    );
    assert!(packs.iter().all(|stage| stage.vector_dimension == Some(1)));
    assert!(packs.iter().all(|stage| stage.matrix_layout.as_deref() == Some("row_major")));

    let native = stages
        .iter()
        .copied()
        .filter(|stage| stage.stage_kind == RuntimeOperatorStageKind::VectorNative)
        .collect::<Vec<_>>();
    assert_eq!(native.len(), 2);
    assert_eq!(
        native.iter().map(|stage| stage.strategy_label.as_deref()).collect::<Vec<_>>(),
        [Some("build_index"), Some("search")]
    );
    assert_eq!(stages.iter().filter(|stage| stage.stage_kind == RuntimeOperatorStageKind::VectorAssemble).count(), 1);

    let binding_ids = stages.iter().map(|stage| stage.execution_binding_id).collect::<Vec<_>>();
    assert!(binding_ids.iter().all(Option::is_some));
    assert!(binding_ids.iter().all(|binding| *binding == binding_ids[0]));
    assert!(stages.iter().all(|stage| stage.device_ordinal == Some(device_ordinal)));
    assert!(stages.iter().all(|stage| stage.vector_algorithm.as_deref() == Some("brute_force_knn")));
    assert!(stages.iter().all(|stage| {
        stage.native_version_major.is_some()
            && stage.native_version_minor.is_some()
            && stage.native_version_patch.is_some()
    }));
}

fn assert_knn_rows(batch: &RecordBatch, expected_ordinals: &[u64], expected_ids: &[i64]) {
    let query_ordinals = batch.column(0).as_any().downcast_ref::<UInt64Array>().expect("query ordinal");
    let neighbor_ordinals = batch.column(2).as_any().downcast_ref::<UInt64Array>().expect("neighbor ordinal");
    let neighbor_ids = batch.column(3).as_any().downcast_ref::<Int64Array>().expect("neighbor id");
    let ranks = batch.column(4).as_any().downcast_ref::<UInt32Array>().expect("rank");
    assert_eq!(query_ordinals.values().as_ref(), &[0, 0, 1, 1]);
    assert_eq!(neighbor_ordinals.values().as_ref(), expected_ordinals);
    assert_eq!(neighbor_ids.values().as_ref(), expected_ids);
    assert_eq!(ranks.values().as_ref(), &[0, 1, 0, 1]);
}

fn assert_kmeans_partition(batch: &RecordBatch, expected_ids: &[i64]) {
    let ordinals = batch.column(0).as_any().downcast_ref::<UInt64Array>().expect("row ordinal");
    let ids = batch.column(1).as_any().downcast_ref::<Int64Array>().expect("logical id");
    let labels = batch.column(2).as_any().downcast_ref::<Int32Array>().expect("cluster label");
    assert_eq!(ordinals.values().as_ref(), &[0, 1, 2, 3]);
    assert_eq!(ids.values().as_ref(), expected_ids);
    assert_eq!(labels.len(), 4);
    assert_eq!(labels.value(0), labels.value(1), "the first compact cluster must have one label");
    assert_eq!(labels.value(2), labels.value(3), "the second compact cluster must have one label");
    assert_ne!(labels.value(0), labels.value(2), "distinct compact clusters must not share a label");
    assert!(labels.values().iter().all(|label| *label >= 0 && *label < 2));
}

fn assert_pca_component_matches_centered_input_up_to_sign(
    batch: &RecordBatch,
    expected_ids: &[i64],
    centered_input: &[f32],
) {
    let ordinals = batch.column(0).as_any().downcast_ref::<UInt64Array>().expect("row ordinal");
    let ids = batch.column(1).as_any().downcast_ref::<Int64Array>().expect("logical id");
    let component = batch.column(2).as_any().downcast_ref::<Float32Array>().expect("PCA component");
    assert_eq!(ordinals.values().as_ref(), &[0, 1, 2, 3]);
    assert_eq!(ids.values().as_ref(), expected_ids);
    assert_eq!(component.len(), centered_input.len());
    let values = component.values().as_ref();
    assert!(values.iter().all(|value| value.is_finite()));
    let dot = values.iter().zip(centered_input).map(|(actual, expected)| actual * expected).sum::<f32>();
    let actual_norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let expected_norm = centered_input.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!(actual_norm > 0.0 && expected_norm > 0.0);
    assert!(
        dot.abs() / (actual_norm * expected_norm) > 0.999,
        "PCA component must span the input one-dimensional subspace, up to sign"
    );
}
