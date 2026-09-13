use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use query_engine::{
    admission::bounded_pipeline_decision,
    capability::{CapabilityDecision, EdgeConsumer, EdgeContractKind, InputRole, VectorInputRole},
    dimension_pruning::local_dimension_pruning_resolver,
    exec::preflight_query_plan,
    expr::{BinaryOp, ColumnRef, Expr, LiteralValue},
    optimizer::{NativeOptimizerOptions, optimize_query_plan},
    pipeline::{PipelineTaskKind, compile_pipeline},
    plan::{
        CUVS_OPERATION_REGISTRY, CuvsOperation, CuvsOutputContract, JoinKey, JoinOutputColumn, JoinSide, JoinSpec,
        JoinType, NodeId, PlanNode, ProjectionOutput, QueryPlan, QueryPlanBuilder, VectorAlgorithm,
        VectorAlgorithmParams, VectorAlgorithmSpec, VectorBruteForceKnnParams, VectorDistanceMetric,
        VectorKMeansParams, VectorPcaParams, VectorRelationBinding, VectorRowIdBinding, canonicalize_cuvs_options,
        vector_brute_force_knn_output_schema, vector_kmeans_output_schema, vector_pca_output_schema,
    },
    rapids_backend::rapids_analysis_context,
    source::{GpuResidentSource, SourceNode},
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

    let optimized = optimize_query_plan(
        plan,
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    )
    .plan;
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
fn vector_column_binding_validates_fixed_size_list_element_type_after_relational_carry() {
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
    preflight_query_plan(&non_vector_builder.finish_with_host_sink(non_vector_source))
        .expect("numeric FixedSizeList remains valid as a relational carry column");
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
fn vector_column_ingress_tracks_safe_projection_lineage_with_relational_carry_support() {
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
fn invalid_vector_contract_is_rejected_by_capability_before_execution() {
    let plan = pca_plan_with_schema(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Float64, false),
        ])),
        1,
        vector_pca_output_schema(DataType::Int64, 1),
    );
    let capability = plan
        .capability_analysis(rapids_analysis_context())
        .nodes
        .into_iter()
        .find(|node| node.node_id == plan.root_id())
        .unwrap();
    let CapabilityDecision::Rejected { reason, facts } = capability.executable else {
        panic!("invalid vector dtype should be rejected locally");
    };
    assert_eq!(reason, "unsupported_vector_algorithm_node_capability");
    assert_eq!(facts.get("reason").map(String::as_str), Some("unsupported_feature_type"));
    assert_eq!(facts.get("validation_code").map(String::as_str), Some("unsupported_type"));
}

#[test]
fn vector_operations_compile_with_role_ordered_resident_inputs() {
    let dataset_schema = relation_schema("id");
    let query_schema = relation_schema("query_id");
    let centroid_schema = relation_schema("centroid_id");
    let cases = [
        (
            "brute_force_knn",
            knn_plan(false),
            vec![
                (NodeId::from_index_for_internal_use(0), VectorInputRole::Dataset, dataset_schema),
                (NodeId::from_index_for_internal_use(1), VectorInputRole::Queries, query_schema),
            ],
        ),
        (
            "kmeans",
            kmeans_plan_with_initial_centroids(),
            vec![
                (NodeId::from_index_for_internal_use(0), VectorInputRole::Input, relation_schema("id")),
                (NodeId::from_index_for_internal_use(1), VectorInputRole::InitialCentroids, centroid_schema),
            ],
        ),
        (
            "pca",
            pca_plan(),
            vec![(NodeId::from_index_for_internal_use(0), VectorInputRole::Input, relation_schema("id"))],
        ),
    ];

    for (algorithm, plan, expected_inputs) in cases {
        let vector_node_id = plan.root_id();
        let PlanNode::VectorAlgorithm(vector) = &plan.root_node().kind else {
            panic!("{algorithm} root should be a vector algorithm");
        };
        let analysis = plan.capability_analysis(rapids_analysis_context());
        let vector_edges = analysis
            .edges
            .iter()
            .filter(|edge| edge.consumer == EdgeConsumer::Node(vector_node_id))
            .collect::<Vec<_>>();
        assert_eq!(vector_edges.len(), expected_inputs.len(), "{algorithm} edge count");

        for (input_index, (producer, role, expected_schema)) in expected_inputs.iter().enumerate() {
            assert_eq!(vector.inputs[input_index], *producer, "{algorithm} vector input order");
            let edge = vector_edges
                .iter()
                .find(|edge| edge.producer == *producer)
                .unwrap_or_else(|| panic!("{algorithm} is missing the edge for {role:?}"));
            assert_eq!(
                edge.input_role,
                InputRole::VectorAlgorithmInput { role: *role },
                "{algorithm} edge role for input {input_index}"
            );
            assert!(edge.contract_kinds.contains(&EdgeContractKind::VectorAlgorithmInput { role: *role }));
            assert_eq!(
                plan.node(*producer).expect("vector producer exists").output_schema().as_ref(),
                expected_schema.as_ref(),
                "{algorithm} producer schema for {role:?}"
            );
        }

        let decision = bounded_pipeline_decision(&plan, Default::default(), rapids_analysis_context())
            .expect("bounded admission should classify vector plan");
        let admission = decision.admission().unwrap_or_else(|| {
            panic!("{algorithm} should compile without coverage rejection: {:?}", decision.rejection_projection())
        });
        assert!(decision.rejection_projection().is_none());
        let pipeline = admission.pipeline();
        let vector_tasks =
            pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::VectorAlgorithm).collect::<Vec<_>>();
        assert_eq!(vector_tasks.len(), 1, "{algorithm} should have one compiled vector task");
        let vector_task = vector_tasks[0];
        assert_eq!(vector_task.plan_node(), Some(vector_node_id));
        assert_eq!(vector_task.input_resources().len(), expected_inputs.len());
        assert_eq!(vector_task.output_resources().len(), 1);

        for (input_index, (producer, role, _)) in expected_inputs.iter().enumerate() {
            let resource_id = vector_task.input_resources()[input_index];
            let resource = pipeline
                .resources()
                .iter()
                .find(|resource| resource.id() == resource_id)
                .unwrap_or_else(|| panic!("{algorithm} {role:?} input resource is missing"));
            let producer_task = pipeline
                .tasks()
                .iter()
                .find(|task| task.id() == resource.producer())
                .expect("resident resource producer task exists");
            assert_eq!(producer_task.plan_node(), Some(*producer), "{algorithm} {role:?} resource producer");
            assert!(resource.consumers().contains(&vector_task.id()), "{algorithm} {role:?} resource consumer");
        }
    }
}

#[test]
fn vector_algorithm_composes_with_relational_tasks_and_keeps_resident_handoffs() {
    let schema = relation_schema("id");
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let filtered = builder.add_filter(
        source,
        Expr::binary(Expr::column("x"), BinaryOp::Gt, Expr::literal(LiteralValue::Float32(0.0))),
        Arc::clone(&schema),
    );
    let projected = builder.add_projection(
        filtered,
        schema
            .fields()
            .iter()
            .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.as_ref().clone()))
            .collect(),
    );
    let vector = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
        VectorAlgorithm::Pca,
        VectorAlgorithmParams::Pca(VectorPcaParams::new(
            VectorRelationBinding::new(projected, VectorRowIdBinding::column("id"), ["x", "y"]),
            1,
        )),
        vector_pca_output_schema(DataType::Int64, 1),
    ));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("pc_0", DataType::Float32, false),
    ]));
    let output = builder.add_projection(
        vector,
        vec![
            ProjectionOutput::new(Expr::column("id"), output_schema.field(0).clone()),
            ProjectionOutput::new(Expr::column("pc_0"), output_schema.field(1).clone()),
        ],
    );
    let plan = builder.finish_with_host_sink(output);
    let analysis = plan.capability_analysis(rapids_analysis_context());
    let pipeline = compile_pipeline(&plan, &analysis).expect("relational operators can surround a vector task");
    pipeline.validate().expect("compiled vector composition satisfies resident task contracts");

    let task_for_node = |node| {
        pipeline
            .tasks()
            .iter()
            .find(|task| task.plan_node() == Some(node))
            .unwrap_or_else(|| panic!("compiled task for plan node {node} is missing"))
    };
    let vector_task = task_for_node(vector);
    let upstream_task = task_for_node(projected);
    let downstream_task = task_for_node(output);
    assert_eq!(vector_task.kind(), PipelineTaskKind::VectorAlgorithm);
    assert_eq!(upstream_task.kind(), PipelineTaskKind::Unary);
    assert_eq!(downstream_task.kind(), PipelineTaskKind::Unary);

    let input_resource = vector_task.input_resources()[0];
    let input = pipeline.resources().iter().find(|resource| resource.id() == input_resource).unwrap();
    assert_eq!(input.producer(), upstream_task.id());
    assert!(input.consumers().contains(&vector_task.id()));

    let output_resource = vector_task.output_resources()[0];
    let output = pipeline.resources().iter().find(|resource| resource.id() == output_resource).unwrap();
    assert_eq!(output.producer(), vector_task.id());
    assert!(output.consumers().contains(&downstream_task.id()));
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

fn knn_spec(dataset: NodeId, queries: NodeId) -> VectorAlgorithmSpec {
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

fn kmeans_plan_with_initial_centroids() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", relation_schema("id"))));
    let centroids = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "initial_centroids",
        relation_schema("centroid_id"),
    )));
    let params =
        VectorKMeansParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x", "y"]), 2)
            .with_initial_centroids(VectorRelationBinding::new(centroids, VectorRowIdBinding::RowIndex, ["x", "y"]));
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

fn joined_vector_relation(builder: &mut QueryPlanBuilder) -> NodeId {
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
