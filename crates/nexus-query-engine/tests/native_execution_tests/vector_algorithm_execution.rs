use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use nexus_query_engine::{
    error::QueryErrorCode,
    exec::execute_query_plan_with_gpu_inputs,
    plan::{
        QueryPlan, QueryPlanBuilder, VECTOR_ALGORITHM_BOUNDED_MEMORY_BOUNDARY,
        VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT, VectorAlgorithm, VectorAlgorithmParams, VectorAlgorithmSpec,
        VectorBruteForceKnnParams, VectorDistanceMetric, VectorKMeansParams, VectorPcaParams, VectorRelationBinding,
        VectorRowIdBinding, vector_brute_force_knn_output_schema, vector_kmeans_output_schema,
        vector_pca_output_schema,
    },
    runtime::resident::GpuResidentInputMap,
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
fn strict_bounded_knn_rejects_supported_shapes_on_missing_cuvs_peak_memory_preflight() {
    for shape in [VectorFixtureShape::Columns, VectorFixtureShape::FixedSizeList, VectorFixtureShape::UniformList] {
        let context = backend_context();
        let dataset_schema = vector_schema("id", shape);
        let query_schema = vector_schema("query_id", shape);
        let plan = knn_plan(dataset_schema, query_schema, shape, 2);

        assert_strict_bounded_vector_rejection(&context, &plan, VectorAlgorithm::BruteForceKnn);
    }
}

#[test]
fn strict_bounded_kmeans_rejects_on_missing_cuvs_peak_memory_preflight() {
    let context = backend_context();
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

    assert_strict_bounded_vector_rejection(&context, &plan, VectorAlgorithm::KMeans);
}

#[test]
fn strict_bounded_pca_rejects_on_missing_cuvs_peak_memory_preflight() {
    let context = backend_context();
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
    let plan = builder.finish_gpu_resident(root);

    assert_strict_bounded_vector_rejection(&context, &plan, VectorAlgorithm::Pca);
}

fn assert_strict_bounded_vector_rejection(
    context: &nexus_query_engine::runtime::ExecutionContext,
    plan: &QueryPlan,
    algorithm: VectorAlgorithm,
) {
    let error = match execute_query_plan_with_gpu_inputs(context, plan, &GpuResidentInputMap::new()) {
        Ok(_) => panic!("strict bounded vector execution must fail closed without cuVS peak-memory preflight"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.operation(), "nexus_query_engine.bounded.vector_algorithm");
    assert_eq!(error.message(), "bounded vector execution requires a lower-level cuVS peak-memory preflight contract");
    assert_eq!(error.fact_value("execution_mode"), Some("bounded"));
    assert_eq!(error.fact_value("node_id"), Some(plan.root_id().to_string().as_str()));
    assert_eq!(error.fact_value("operator"), Some("vector_algorithm"));
    assert_eq!(error.fact_value("algorithm"), Some(algorithm.as_str()));
    assert_eq!(error.fact_value("rejection_reason"), Some("blocked_lower_level_contract"));
    assert_eq!(error.fact_value("contract_status"), Some("blocked_by_lower_level"));
    assert_eq!(error.fact_value("memory_boundary"), Some(VECTOR_ALGORITHM_BOUNDED_MEMORY_BOUNDARY));
    assert_eq!(error.fact_value("required_contract"), Some(VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT));
    assert_eq!(error.fact_value("candidate_status"), None);
    assert!(error.query_id().is_some(), "execution-boundary rejection must retain query correlation");
    assert!(error.source_error().is_none());
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
