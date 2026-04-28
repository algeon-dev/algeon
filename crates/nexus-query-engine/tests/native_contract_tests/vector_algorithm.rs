use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use nexus_query_engine::{
    adaptive_report::{AdaptiveCandidateStatus, AdaptiveStrategy},
    adaptive_runtime::{AdaptiveMemoryHistory, AdaptiveMemoryHistoryKey, AdaptiveMemoryObservation},
    admission::{
        adaptive_bounded_pipeline_admission, adaptive_bounded_pipeline_admission_with_memory_history,
        strict_bounded_pipeline_admission, strict_bounded_pipeline_rejection_projection,
    },
    capability::{
        CapabilityDecision, ContractQuality, ContractStatus, EdgeConsumer, EdgeContractKind, InputRole, NodeOperator,
        VectorInputRole,
    },
    contract_report::StrictBoundedContractReport,
    exec::preflight_query_plan,
    expr::{ColumnRef, Expr},
    mode::{AdaptiveExecutionMode, StrictBoundedExecutionMode},
    optimizer::{NativeOptimizerOptions, optimize_query_plan},
    plan::{
        CUVS_OPERATION_REGISTRY, CuvsOperation, CuvsOutputContract, JoinKey, JoinOutputColumn, JoinSide, JoinSpec,
        JoinType, PlanNode, ProjectionOutput, QueryPlan, QueryPlanBuilder, VECTOR_ALGORITHM_BOUNDED_MEMORY_BOUNDARY,
        VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT, VectorAlgorithm, VectorAlgorithmParams, VectorAlgorithmSpec,
        VectorBruteForceKnnParams, VectorDistanceMetric, VectorKMeansParams, VectorPcaParams, VectorRelationBinding,
        VectorRowIdBinding, canonicalize_cuvs_options, vector_brute_force_knn_output_schema,
        vector_kmeans_output_schema, vector_pca_output_schema,
    },
    source::{GpuResidentSource, SourceNode},
};
#[cfg(not(feature = "cuvs"))]
use nexus_query_engine::{
    error::QueryErrorCode,
    exec::execute_query_plan_with_gpu_inputs,
    runtime::{ExecutionContext, resident::GpuResidentInputMap},
};

#[test]
fn canonical_cuvs_registry_exposes_all_operations_and_exact_output_contracts() {
    assert_eq!(
        CuvsOperation::all().collect::<Vec<_>>(),
        vec![CuvsOperation::BruteForceKnn, CuvsOperation::KMeans, CuvsOperation::Pca]
    );
    assert_eq!(CUVS_OPERATION_REGISTRY.len(), 3);
    assert_eq!(CuvsOperation::BruteForceKnn.descriptor().output_contract, CuvsOutputContract::BruteForceKnn);
    assert_eq!(CuvsOperation::KMeans.descriptor().output_contract, CuvsOutputContract::KMeansAssignments);
    assert_eq!(CuvsOperation::Pca.descriptor().output_contract, CuvsOutputContract::PcaComponents);
    assert_eq!(
        vector_brute_force_knn_output_schema(DataType::Utf8, DataType::Int64)
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<Vec<_>>(),
        vec!["query_ordinal", "query_id", "neighbor_ordinal", "neighbor_id", "rank", "distance"]
    );
    assert_eq!(
        vector_kmeans_output_schema(DataType::Int32).fields().iter().map(|field| field.name()).collect::<Vec<_>>(),
        vec!["row_ordinal", "id", "cluster_id"]
    );
    assert_eq!(
        vector_pca_output_schema(DataType::LargeUtf8, 2).fields().iter().map(|field| field.name()).collect::<Vec<_>>(),
        vec!["row_ordinal", "id", "pc_0", "pc_1"]
    );
}

#[test]
fn canonical_cuvs_registry_validates_relation_binding_shape() {
    let valid_columns = r#"{"input":{"id":"id","vector":{"columns":["x"]}},"n_clusters":1}"#;
    assert!(canonicalize_cuvs_options(CuvsOperation::KMeans, valid_columns).is_ok());

    let valid_vector_column = r#"{"input":{"id":"id","vector":{"column":"embedding"}},"n_clusters":1}"#;
    assert!(canonicalize_cuvs_options(CuvsOperation::KMeans, valid_vector_column).is_ok());

    let invalid_bindings = [
        (r#"{"input":{},"n_clusters":1}"#, "missing_or_invalid_option"),
        (r#"{"input":{"id":7,"vector":{"columns":["x"]}},"n_clusters":1}"#, "missing_or_invalid_option"),
        (r#"{"input":{"id":"id","vector":{}},"n_clusters":1}"#, "invalid_vector_binding"),
        (r#"{"input":{"id":"id","vector":{"columns":"x"}},"n_clusters":1}"#, "invalid_vector_binding"),
        (r#"{"input":{"id":"id","vector":{"columns":[]}},"n_clusters":1}"#, "invalid_vector_binding"),
        (r#"{"input":{"id":"id","vector":{"columns":["x","x"]}},"n_clusters":1}"#, "duplicate_vector_column"),
        (
            r#"{"input":{"id":"id","vector":{"columns":["x"],"column":"embedding"}},"n_clusters":1}"#,
            "invalid_vector_binding",
        ),
        (r#"{"input":{"id":"id","vector":{"column":"embedding","extra":true}},"n_clusters":1}"#, "unknown_option"),
    ];
    for (raw, expected_code) in invalid_bindings {
        let error = canonicalize_cuvs_options(CuvsOperation::KMeans, raw)
            .expect_err("malformed relation binding should be rejected by canonical metadata");
        assert_eq!(error.code(), expected_code, "raw options: {raw}");
    }
}

#[test]
fn typed_vector_builders_preserve_schemas_input_order_and_display() {
    let knn = knn_plan(false);
    preflight_query_plan(&knn).expect("valid kNN IR should preflight");
    let PlanNode::VectorAlgorithm(knn_node) = &knn.root_node().kind else {
        panic!("kNN root should be a vector algorithm");
    };
    assert_eq!(knn_node.spec.algorithm, VectorAlgorithm::BruteForceKnn);
    assert_eq!(knn_node.inputs.len(), 2);
    assert_eq!(
        knn_node.output_schema().as_ref(),
        vector_brute_force_knn_output_schema(DataType::Int64, DataType::Int64).as_ref()
    );
    let knn_display = knn.to_string();
    assert!(knn_display.contains("VectorAlgorithm"));
    assert!(knn_display.contains("algorithm=brute_force_knn"));
    assert!(knn_display.contains("k=3 metric=l2_expanded"));

    let kmeans = kmeans_plan();
    preflight_query_plan(&kmeans).expect("valid K-means IR should preflight");
    let PlanNode::VectorAlgorithm(kmeans_node) = &kmeans.root_node().kind else {
        panic!("K-means root should be a vector algorithm");
    };
    assert_eq!(kmeans_node.spec.algorithm, VectorAlgorithm::KMeans);
    assert_eq!(kmeans_node.inputs.len(), 1);
    assert_eq!(kmeans_node.output_schema().as_ref(), vector_kmeans_output_schema(DataType::Int64).as_ref());
    assert!(kmeans.to_string().contains("n_clusters=4"));

    let pca = pca_plan();
    preflight_query_plan(&pca).expect("valid PCA IR should preflight");
    let PlanNode::VectorAlgorithm(pca_node) = &pca.root_node().kind else {
        panic!("PCA root should be a vector algorithm");
    };
    assert_eq!(pca_node.spec.algorithm, VectorAlgorithm::Pca);
    assert_eq!(pca_node.inputs.len(), 1);
    assert_eq!(pca_node.output_schema().as_ref(), vector_pca_output_schema(DataType::Int64, 2).as_ref());
    assert!(pca.to_string().contains("n_components=2 whiten=false"));
}

#[test]
fn vector_optimizer_rewrite_remaps_relation_bindings_and_ordered_inputs() {
    let schema = relation_schema("id");
    let query_schema = relation_schema("query_id");
    let mut builder = QueryPlanBuilder::new();
    let dataset = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset", Arc::clone(&schema))));
    let dataset_projection = builder.add_projection(
        dataset,
        schema
            .fields()
            .iter()
            .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.as_ref().clone()))
            .collect(),
    );
    let queries = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("queries", query_schema)));
    let spec = knn_spec(dataset_projection, queries);
    let root = builder.add_vector_algorithm(spec);
    let plan = builder.finish_with_host_sink(root);

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default()).plan;
    preflight_query_plan(&optimized).expect("rewritten vector bindings should remain valid");
    let PlanNode::VectorAlgorithm(vector) = &optimized.root_node().kind else {
        panic!("rewritten root should remain a vector algorithm");
    };
    let binding_inputs = vector.spec.params.relation_bindings().map(|binding| binding.input).collect::<Vec<_>>();
    assert_eq!(vector.inputs, binding_inputs);
    assert_eq!(vector.inputs.len(), 2);
    assert!(matches!(optimized.node(vector.inputs[0]).map(|node| &node.kind), Some(PlanNode::Source(_))));
    assert!(matches!(optimized.node(vector.inputs[1]).map(|node| &node.kind), Some(PlanNode::Source(_))));
}

#[test]
fn vector_validation_rejects_invalid_bindings_parameters_and_output_schema() {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", relation_schema("id"))));
    let missing_feature =
        VectorPcaParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["missing"]), 1);
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(missing_feature),
        vector_pca_output_schema(DataType::Int64, 1),
    ));
    let missing_feature = builder.finish_with_host_sink(root);
    assert_preflight_reason(&missing_feature, "missing_feature_column");

    let nullable_schema =
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false), Field::new("x", DataType::Float32, true)]));
    let nullable_feature = pca_plan_with_schema(nullable_schema, 1, vector_pca_output_schema(DataType::Int64, 1));
    assert_preflight_reason(&nullable_feature, "unsupported_feature_type");

    let zero_k = knn_plan_with_k(0);
    assert_preflight_reason(&zero_k, "zero_k");

    let wrong_schema = pca_plan_with_schema(
        relation_schema("id"),
        2,
        Arc::new(Schema::new(vec![Field::new("wrong", DataType::Float32, false)])),
    );
    assert_preflight_reason(&wrong_schema, "unexpected_output_schema");
}

#[test]
fn vector_column_binding_validates_fixed_size_list_element_type() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
            false,
        ),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let params = VectorPcaParams::new(
        VectorRelationBinding::new_vector_column(input, VectorRowIdBinding::column("id"), "embedding"),
        2,
    );
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(params),
        vector_pca_output_schema(DataType::Int64, 2),
    ));
    preflight_query_plan(&builder.finish_with_host_sink(root)).expect("fixed-size Float32 vectors are valid");

    let mut non_vector_builder = QueryPlanBuilder::new();
    let non_vector_source =
        non_vector_builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("non_vector_input", schema)));
    let error = preflight_query_plan(&non_vector_builder.finish_with_host_sink(non_vector_source))
        .expect_err("numeric list schemas remain unsupported outside typed vector ingress");
    assert_eq!(error.operation(), "nexus_query_engine.plan.validate_schema");
}

#[test]
fn variable_list_vector_column_defers_dimension_validation_to_the_device_witness() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("embedding", DataType::List(Arc::new(Field::new("item", DataType::Float32, false))), false),
    ]));
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

    preflight_query_plan(&builder.finish_with_host_sink(root))
        .expect("non-null List<Float32> is admitted subject to a runtime uniform-dimension device witness");
}

#[test]
fn pca_requires_two_feature_columns_when_the_dimension_is_known() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("x", DataType::Float32, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    let params = VectorPcaParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x"]), 1);
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(params),
        vector_pca_output_schema(DataType::Int64, 1),
    ));

    assert_preflight_reason(&builder.finish_with_host_sink(root), "insufficient_feature_dimension");
}

#[test]
fn large_list_vector_column_defers_dimension_validation_to_the_device_witness() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("embedding", DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, false))), false),
    ]));
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

    preflight_query_plan(&builder.finish_with_host_sink(root))
        .expect("non-null LargeList<Float32> is admitted subject to a runtime uniform-dimension device witness");
}

#[test]
fn vector_column_ingress_tracks_safe_projection_lineage_without_global_list_support() {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
            false,
        ),
    ]));
    let projected_vector = Field::new(
        "projected_embedding",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
        false,
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", source_schema)));
    let projected = builder.add_projection(
        source,
        vec![
            ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int64, false)),
            ProjectionOutput::new(Expr::column("embedding"), projected_vector),
        ],
    );
    let params = VectorPcaParams::new(
        VectorRelationBinding::new_vector_column(projected, VectorRowIdBinding::column("id"), "projected_embedding"),
        2,
    );
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(params),
        vector_pca_output_schema(DataType::Int64, 2),
    ));
    preflight_query_plan(&builder.finish_with_host_sink(root))
        .expect("safe direct-column projection lineage should retain typed vector ingress");
}

#[test]
fn direct_ir_enforces_registry_parameter_maxima_and_logical_id_roles() {
    let oversized_k = knn_plan_with_k(u32::MAX as usize + 1);
    assert_preflight_reason(&oversized_k, "parameter_exceeds_registry_maximum");

    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", relation_schema("id"))));
    let mut kmeans =
        VectorKMeansParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x", "y"]), 1);
    kmeans.n_clusters = i32::MAX as usize + 1;
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::KMeans,
        VectorAlgorithmParams::KMeans(kmeans),
        vector_kmeans_output_schema(DataType::Int64),
    ));
    assert_preflight_reason(&builder.finish_with_host_sink(root), "parameter_exceeds_registry_maximum");

    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", relation_schema("id"))));
    let mut kmeans =
        VectorKMeansParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x", "y"]), 1);
    kmeans.max_iterations = i32::MAX as u32 + 1;
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::KMeans,
        VectorAlgorithmParams::KMeans(kmeans),
        vector_kmeans_output_schema(DataType::Int64),
    ));
    assert_preflight_reason(&builder.finish_with_host_sink(root), "parameter_exceeds_registry_maximum");

    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", relation_schema("id"))));
    let mut kmeans =
        VectorKMeansParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x", "y"]), 1);
    kmeans.n_init = i32::MAX as u32 + 1;
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::KMeans,
        VectorAlgorithmParams::KMeans(kmeans),
        vector_kmeans_output_schema(DataType::Int64),
    ));
    assert_preflight_reason(&builder.finish_with_host_sink(root), "parameter_exceeds_registry_maximum");

    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", relation_schema("id"))));
    let pca = VectorPcaParams::new(
        VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x", "y"]),
        i32::MAX as usize + 1,
    );
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(pca),
        vector_pca_output_schema(DataType::Int64, 1),
    ));
    assert_preflight_reason(&builder.finish_with_host_sink(root), "parameter_exceeds_registry_maximum");

    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", relation_schema("id"))));
    let mut pca =
        VectorPcaParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x", "y"]), 1);
    pca.n_iterations = i32::MAX as u32 + 1;
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(pca),
        vector_pca_output_schema(DataType::Int64, 1),
    ));
    assert_preflight_reason(&builder.finish_with_host_sink(root), "parameter_exceeds_registry_maximum");

    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", relation_schema("id"))));
    let params = VectorPcaParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::RowIndex, ["x", "y"]), 1);
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(params),
        vector_pca_output_schema(DataType::Int64, 1),
    ));
    assert_preflight_reason(&builder.finish_with_host_sink(root), "missing_logical_row_id_column");
}

#[test]
fn relation_initialized_kmeans_accepts_row_index_only_for_initial_centroids() {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", relation_schema("id"))));
    let centroids = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "initial_centroids",
        relation_schema("unused_centroid_id"),
    )));
    let params =
        VectorKMeansParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x", "y"]), 4)
            .with_initial_centroids(VectorRelationBinding::new(centroids, VectorRowIdBinding::RowIndex, ["x", "y"]));
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::KMeans,
        VectorAlgorithmParams::KMeans(params),
        vector_kmeans_output_schema(DataType::Int64),
    ));

    preflight_query_plan(&builder.finish_with_host_sink(root))
        .expect("only relation-backed initial centroids may use positional row identity");
}

#[test]
fn vector_capability_has_typed_edges_and_lower_level_bounded_block() {
    let plan = knn_plan(false);
    let analysis = plan.capability_analysis();
    let capability = analysis.nodes.iter().find(|node| node.node_id == plan.root_id()).unwrap();
    assert_eq!(capability.operator, NodeOperator::VectorAlgorithm { algorithm: VectorAlgorithm::BruteForceKnn });
    assert_eq!(capability.executable, CapabilityDecision::Satisfied);
    assert_eq!(capability.local_requirements.len(), 1);
    let requirement = &capability.local_requirements[0];
    assert_eq!(requirement.memory_boundary, VECTOR_ALGORITHM_BOUNDED_MEMORY_BOUNDARY);
    assert_eq!(requirement.required_contract, VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT);
    assert_eq!(requirement.status, ContractStatus::BlockedByLowerLevel);
    assert_eq!(requirement.quality, ContractQuality::ExactPreflight);

    let vector_edges =
        analysis.edges.iter().filter(|edge| edge.consumer == EdgeConsumer::Node(plan.root_id())).collect::<Vec<_>>();
    assert_eq!(vector_edges.len(), 2);
    for (edge, role) in vector_edges.iter().zip([VectorInputRole::Dataset, VectorInputRole::Queries]) {
        assert_eq!(edge.input_role, InputRole::VectorAlgorithmInput { role });
        assert!(edge.contract_kinds.contains(&EdgeContractKind::VectorAlgorithmInput { role }));
    }

    let report = StrictBoundedContractReport::from_capability_analysis(&analysis).unwrap();
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::UnsupportedStrictBounded);
    assert!(report.gate_rows().iter().any(|row| {
        row.memory_boundary == VECTOR_ALGORITHM_BOUNDED_MEMORY_BOUNDARY
            && row.required_contract == VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT
            && row.contract_status == ContractStatus::BlockedByLowerLevel
    }));
}

#[test]
fn invalid_vector_contract_is_rejected_by_capability_before_execution() {
    let plan = pca_plan_with_schema(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Float64, false),
        ])),
        1,
        vector_pca_output_schema(DataType::Int64, 1),
    );
    let capability = plan.capability_analysis().nodes.into_iter().find(|node| node.node_id == plan.root_id()).unwrap();
    let CapabilityDecision::Rejected { reason, facts } = capability.executable else {
        panic!("invalid vector dtype should be rejected locally");
    };
    assert_eq!(reason, "unsupported_vector_algorithm_node_capability");
    assert_eq!(facts.get("reason").map(String::as_str), Some("unsupported_feature_type"));
    assert_eq!(facts.get("validation_code").map(String::as_str), Some("unsupported_type"));
}

#[test]
fn strict_and_adaptive_vector_rejections_keep_independent_evidence() {
    for plan in [knn_plan(false), knn_plan(true)] {
        assert!(plan.contains_vector_algorithm());
        assert!(
            strict_bounded_pipeline_admission(&plan).expect("strict admission should classify vector plan").is_none()
        );
        let strict = strict_bounded_pipeline_rejection_projection(&plan)
            .expect("strict rejection projection should be constructible")
            .expect("vector plan should have a strict rejection");
        assert_eq!(strict.reason, "blocked_lower_level_contract");
        assert_eq!(
            strict.facts.get("required_contract").map(String::as_str),
            Some(VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT)
        );

        let adaptive = adaptive_bounded_pipeline_admission(&plan)
            .expect("adaptive admission should classify vector plan")
            .expect("vector plan should emit adaptive evidence");
        assert!(adaptive.pipeline().is_none());
        let row = adaptive.report().row();
        assert_eq!(row.execution_mode, AdaptiveExecutionMode::UnsupportedAdaptive);
        assert_eq!(row.candidate_status, AdaptiveCandidateStatus::Rejected);
        assert_eq!(row.adaptive_strategy, AdaptiveStrategy::Unsupported);
        assert_eq!(row.runtime_actions_allowed.to_stable_value(), "none");
        assert_eq!(row.memory_history_evidence.to_stable_value(), "not_used");
        assert!(row.missing_strict_contracts.iter().any(|missing| {
            missing.required_contract == VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT
                && missing.contract_status == ContractStatus::BlockedByLowerLevel
        }));
        let rendered = adaptive.report().stable_kv_row();
        assert!(rendered.contains("execution_mode=unsupported_adaptive"));
        assert!(!rendered.contains("execution_mode=strict_bounded"));
        assert!(!rendered.contains("unsupported_strict_bounded"));
    }
}

#[test]
fn adaptive_vector_rejection_does_not_consult_present_memory_history() {
    for plan in [knn_plan(false), knn_plan(true)] {
        let memory_history = AdaptiveMemoryHistory::new();
        memory_history.record_observation(
            AdaptiveMemoryHistoryKey::for_query_plan(&plan),
            AdaptiveMemoryObservation {
                resident_dataframe_peak_bytes: Some(4096),
                reservation_granted_bytes: 8192,
                source_chunks_committed: 3,
            },
        );
        assert!(memory_history.lookup_query_plan(&plan).is_some());
        let adaptive = adaptive_bounded_pipeline_admission_with_memory_history(&plan, Some(&memory_history))
            .expect("adaptive admission should classify vector plan")
            .expect("vector plan should emit adaptive evidence");

        assert!(adaptive.pipeline().is_none());
        let row = adaptive.report().row();
        assert_eq!(row.execution_mode, AdaptiveExecutionMode::UnsupportedAdaptive);
        assert_eq!(row.candidate_status, AdaptiveCandidateStatus::Rejected);
        assert_eq!(row.adaptive_strategy, AdaptiveStrategy::Unsupported);
        assert_eq!(row.runtime_actions_allowed.to_stable_value(), "none");
        assert!(!row.runtime_actions_allowed.memory_history_lookup);
        assert_eq!(row.memory_history_evidence.to_stable_value(), "not_used");

        let rendered = adaptive.report().stable_kv_row();
        assert!(rendered.contains("memory_history_evidence=not_used"));
        assert!(!rendered.contains("memory_history_evidence=lookup_miss"));
        assert!(!rendered.contains("memory_history_evidence=observed"));
        assert!(!rendered.contains("execution_mode=strict_bounded"));
    }
}

fn knn_plan(with_join_input: bool) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let dataset = if with_join_input {
        joined_vector_relation(&mut builder)
    } else {
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset", relation_schema("id"))))
    };
    let queries =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("queries", relation_schema("query_id"))));
    let root = builder.add_vector_algorithm(knn_spec(dataset, queries));
    builder.finish_with_host_sink(root)
}

fn knn_plan_with_k(k: usize) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let dataset = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset", relation_schema("id"))));
    let queries =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("queries", relation_schema("query_id"))));
    let mut params = match knn_spec(dataset, queries).params {
        VectorAlgorithmParams::BruteForceKnn(params) => params,
        _ => unreachable!(),
    };
    params.k = k;
    let spec = VectorAlgorithmSpec::new(
        VectorAlgorithm::BruteForceKnn,
        VectorAlgorithmParams::BruteForceKnn(params),
        vector_brute_force_knn_output_schema(DataType::Int64, DataType::Int64),
    );
    let root = builder.add_vector_algorithm(spec);
    builder.finish_with_host_sink(root)
}

fn knn_spec(
    dataset: nexus_query_engine::plan::NodeId,
    queries: nexus_query_engine::plan::NodeId,
) -> VectorAlgorithmSpec {
    VectorAlgorithmSpec::new(
        VectorAlgorithm::BruteForceKnn,
        VectorAlgorithmParams::BruteForceKnn(VectorBruteForceKnnParams::new(
            VectorRelationBinding::new(dataset, VectorRowIdBinding::column("id"), ["x", "y"]),
            VectorRelationBinding::new(queries, VectorRowIdBinding::column("query_id"), ["x", "y"]),
            3,
            VectorDistanceMetric::L2Expanded,
        )),
        vector_brute_force_knn_output_schema(DataType::Int64, DataType::Int64),
    )
}

fn kmeans_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", relation_schema("id"))));
    let params =
        VectorKMeansParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x", "y"]), 4);
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::KMeans,
        VectorAlgorithmParams::KMeans(params),
        vector_kmeans_output_schema(DataType::Int64),
    ));
    builder.finish_with_host_sink(root)
}

fn pca_plan() -> QueryPlan {
    pca_plan_with_schema(relation_schema("id"), 2, vector_pca_output_schema(DataType::Int64, 2))
}

fn pca_plan_with_schema(schema: SchemaRef, n_components: usize, output_schema: SchemaRef) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    let params = VectorPcaParams::new(
        VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x", "y"]),
        n_components,
    );
    let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(params),
        output_schema,
    ));
    builder.finish_with_host_sink(root)
}

fn joined_vector_relation(builder: &mut QueryPlanBuilder) -> nexus_query_engine::plan::NodeId {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("x", DataType::Float32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int64, false),
        Field::new("y", DataType::Float32, false),
    ]));
    let output_schema = relation_schema("id");
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset_left", left_schema)));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset_right", right_schema)));
    builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(ColumnRef::input(0, "x", "x"), "x"),
            JoinOutputColumn::right(ColumnRef::input(1, "y", "y"), "y"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    )
}

fn relation_schema(id: &str) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(id, DataType::Int64, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
    ]))
}

fn assert_preflight_reason(plan: &QueryPlan, expected: &str) {
    let error = preflight_query_plan(plan).expect_err("invalid vector contract should fail preflight");
    assert_eq!(error.fact_value("reason"), Some(expected));
}
