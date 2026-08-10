use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use nexus_query_engine::{
    error::QueryErrorCode,
    plan::{
        QueryPlan, QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmParams, VectorAlgorithmSpec,
        VectorBruteForceKnnParams, VectorDistanceMetric, VectorKMeansParams, VectorPcaParams, VectorRelationBinding,
        VectorRowIdBinding, vector_brute_force_knn_output_schema, vector_kmeans_output_schema,
        vector_pca_output_schema,
    },
    source::{GpuResidentSource, SourceNode},
};

use super::backend_context;

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
    let plan = knn_plan(dataset_schema, query_schema, VectorFixtureShape::Columns, 2);
    assert_vector_capability_unavailable(&plan);
}

#[test]
fn exact_knn_accepts_proof_bearing_fixed_size_list_vectors() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::FixedSizeList);
    let query_schema = vector_schema("query_id", VectorFixtureShape::FixedSizeList);
    let plan = knn_plan(dataset_schema, query_schema, VectorFixtureShape::FixedSizeList, 2);
    assert_vector_capability_unavailable(&plan);
}

#[test]
fn exact_knn_accepts_proof_bearing_uniform_list_vectors() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::UniformList);
    let query_schema = vector_schema("query_id", VectorFixtureShape::UniformList);
    let plan = knn_plan(dataset_schema, query_schema, VectorFixtureShape::UniformList, 2);
    assert_vector_capability_unavailable(&plan);
}

#[test]
fn exact_knn_breaks_equal_distance_ties_by_physical_ordinal() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let plan = knn_plan(dataset_schema, query_schema, VectorFixtureShape::Columns, 2);
    assert_vector_capability_unavailable(&plan);
}

#[test]
fn exact_knn_preserves_an_empty_query_relation_without_native_search() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    let plan = knn_plan(dataset_schema, query_schema, VectorFixtureShape::Columns, 1);
    assert_vector_capability_unavailable(&plan);
}

#[test]
fn exact_knn_rejects_non_finite_features_and_k_above_dataset_rows() {
    let dataset_schema = vector_schema("id", VectorFixtureShape::Columns);
    let query_schema = vector_schema("query_id", VectorFixtureShape::Columns);
    for k in [1, 2] {
        let plan = knn_plan(Arc::clone(&dataset_schema), Arc::clone(&query_schema), VectorFixtureShape::Columns, k);
        assert_vector_capability_unavailable(&plan);
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
    assert_vector_capability_unavailable(&builder.finish_gpu_resident(root));
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
    assert_vector_capability_unavailable(&builder.finish_gpu_resident(root));
}

fn assert_vector_capability_unavailable(plan: &QueryPlan) {
    let context = backend_context();
    let error = match context.admit_plan(plan) {
        Ok(admitted) => {
            admitted.finish();
            panic!("vector plan without a bounded cuVS owner contract must fail before GPU execution")
        },
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.operation(), "nexus_query_engine.query_service.admission");
    assert_eq!(error.fact_value("reason"), Some("required_device_capability_unavailable"));
    assert_eq!(
        error.fact_value("required_capabilities"),
        Some("operator_output:cuvs_vector_algorithm_peak_memory_preflight")
    );
    assert_eq!(error.fact_value("configured_device_count"), Some("1"));
    assert_eq!(error.fact_value("compatible_device_count"), Some("0"));
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
