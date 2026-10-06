use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use query_engine::{
    dimension_pruning::local_dimension_pruning_resolver, exec::preflight_query_plan,
    rapids_backend::rapids_analysis_context,
};
use query_planning::{
    capability::{CapabilityDecision, EdgeConsumer, EdgeContractKind, InputRole, Reason, UnsupportedPart},
    expr::{BinaryOp, Expr, LiteralValue},
    optimizer::{NativeOptimizerOptions, optimize_query_plan},
    pipeline::{PipelineTaskKind, compile_pipeline},
    plan::{
        CUML_OPERATION_REGISTRY, CumlOperation, CumlOutputContract, MlAlgorithm, MlAlgorithmParams, MlAlgorithmSpec,
        MlForestModelFacts, MlForestModelFormat, MlForestModelSnapshot, MlForestOutputContract, MlForestPrecision,
        MlForestPredictParams, MlHdbscanParams, MlLogisticRegressionParams, MlRandomForestParams, MlRelationBinding,
        MlRelationRole, MlTsvdParams, MlUmapParams, PlanNode, ProjectionOutput, QueryPlan, QueryPlanBuilder,
        VectorElementType, VectorRowIdBinding, canonicalize_cuml_options, ml_classification_output_schema,
        ml_embedding_output_schema, ml_forest_output_schema, ml_hdbscan_output_schema, ml_regression_output_schema,
        vector_pca_output_schema,
    },
    policy::PlanningPolicy,
    source::{GpuResidentSource, SourceNode},
};

fn supervised_plan(label: Option<(DataType, bool)>, predict_label: bool, predict_dimensions: usize) -> QueryPlan {
    supervised_plan_for(MlAlgorithm::LogisticRegression, label, predict_label, predict_dimensions)
}

fn supervised_plan_for(
    algorithm: MlAlgorithm,
    label: Option<(DataType, bool)>,
    predict_label: bool,
    predict_dimensions: usize,
) -> QueryPlan {
    let mut training_fields = vec![Field::new("id", DataType::Int64, false)];
    training_fields.extend((0..2).map(|index| Field::new(format!("x{index}"), DataType::Float32, false)));
    if let Some((dtype, nullable)) = &label {
        training_fields.push(Field::new("label", dtype.clone(), *nullable));
    }
    let training_schema = Arc::new(Schema::new(training_fields));
    let mut predict_fields = vec![Field::new("id", DataType::Utf8, false)];
    predict_fields
        .extend((0..predict_dimensions).map(|index| Field::new(format!("x{index}"), DataType::Float32, false)));
    if predict_label {
        predict_fields.push(Field::new("label", DataType::Int32, false));
    }
    let predict_schema = Arc::new(Schema::new(predict_fields));
    let mut builder = QueryPlanBuilder::new();
    let training = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("training", training_schema)));
    let predict = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("predict", predict_schema)));
    let mut training_binding = MlRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x0", "x1"]);
    training_binding.label = label.map(|_| "label".to_owned());
    let predict_binding = MlRelationBinding::new(
        1,
        VectorRowIdBinding::column("id"),
        (0..predict_dimensions).map(|index| format!("x{index}")),
    );
    let params = match algorithm {
        MlAlgorithm::LogisticRegression => {
            MlAlgorithmParams::LogisticRegression(MlLogisticRegressionParams::new(training_binding, predict_binding))
        },
        MlAlgorithm::RandomForestClassifier => {
            MlAlgorithmParams::RandomForestClassifier(MlRandomForestParams::new(training_binding, predict_binding))
        },
        MlAlgorithm::RandomForestRegressor => {
            let mut params = MlRandomForestParams::new(training_binding, predict_binding);
            params.criterion = None;
            MlAlgorithmParams::RandomForestRegressor(params)
        },
        _ => unreachable!("supervised fixture"),
    };
    let output_schema = if algorithm == MlAlgorithm::RandomForestRegressor {
        ml_regression_output_schema(DataType::Utf8)
    } else {
        ml_classification_output_schema(DataType::Utf8)
    };
    let root = builder
        .add_ml_algorithm(vec![training, predict], MlAlgorithmSpec::new(algorithm, params, output_schema))
        .unwrap();
    builder.finish_with_host_sink(root).unwrap()
}

#[test]
fn random_forest_contracts_validate_roles_types_options_and_schemas() {
    for (algorithm, label_type) in [
        (MlAlgorithm::RandomForestClassifier, DataType::Int32),
        (MlAlgorithm::RandomForestRegressor, DataType::Float32),
    ] {
        let plan = supervised_plan_for(algorithm, Some((label_type.clone(), false)), false, 2);
        preflight_query_plan(&plan).unwrap();
        let PlanNode::MlAlgorithm(node) = &plan.root_node().kind else { panic!("RF plan") };
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.output_schema().field(2).data_type(), &label_type);
        let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
        let pipeline = compile_pipeline(&plan, &analysis).unwrap();
        pipeline.validate().unwrap();
        assert_eq!(
            pipeline
                .tasks()
                .iter()
                .find(|task| task.plan_node() == Some(plan.root_id()))
                .unwrap()
                .input_resources()
                .len(),
            2
        );
        let invalid_type = if label_type == DataType::Int32 { DataType::Float32 } else { DataType::Int32 };
        assert_eq!(
            preflight_query_plan(&supervised_plan_for(algorithm, Some((invalid_type, false)), false, 2))
                .unwrap_err()
                .fact_value("reason"),
            Some("unsupported_type")
        );
    }
    let classifier = CumlOperation::RandomForestClassifier;
    let regressor = CumlOperation::RandomForestRegressor;
    assert!(classifier.descriptor().option("split_criterion").is_some());
    assert!(regressor.descriptor().option("split_criterion").is_none());
    for operation in [classifier, regressor] {
        let schema = operation.descriptor().options_json_schema();
        assert!(schema["properties"]["max_leaves"].get("default").is_none());
        let raw = r#"{"training":{"id":"id","vector":{"columns":["x0"]},"label":"label"},"predict":{"id":"id","vector":{"columns":["x0"]}}}"#;
        let canonical = canonicalize_cuml_options(operation, raw).unwrap();
        assert!(canonical.get("max_leaves").is_none());
        assert_eq!(canonical["n_trees"], 100);
        let invalid = serde_json::json!({"training":{"id":"id","vector":{"columns":["x0"]},"label":"label"},
            "predict":{"id":"id","vector":{"columns":["x0"]}}, "max_leaves": 0})
        .to_string();
        assert_eq!(canonicalize_cuml_options(operation, &invalid).unwrap_err().argument(), Some("max_leaves"));
    }
}

#[test]
fn supervised_bindings_validate_both_relations_and_compile_role_order() {
    let plan = supervised_plan(Some((DataType::Int32, false)), false, 2);
    preflight_query_plan(&plan).unwrap();
    let PlanNode::MlAlgorithm(node) = &plan.root_node().kind else { panic!("supervised root") };
    assert_eq!(
        node.inputs,
        node.spec.params.relation_bindings().map(|binding| node.inputs[binding.input]).collect::<Vec<_>>()
    );
    assert_eq!(node.inputs.len(), 2);
    assert!(plan.to_string().contains("training=("));
    assert!(plan.to_string().contains("predict=("));
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let roles = analysis
        .edges
        .iter()
        .filter(|edge| edge.consumer == EdgeConsumer::Node(plan.root_id()))
        .map(|edge| edge.input_role)
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        [
            InputRole::MlAlgorithmInput { role: MlRelationRole::Training },
            InputRole::MlAlgorithmInput { role: MlRelationRole::Predict }
        ]
    );
    let pipeline = compile_pipeline(&plan, &analysis).unwrap();
    pipeline.validate().unwrap();
    let task = pipeline.tasks().iter().find(|task| task.plan_node() == Some(plan.root_id())).unwrap();
    assert_eq!(task.input_resources().len(), 2);
    preflight_query_plan(&supervised_plan(Some((DataType::Int32, true)), false, 2))
        .expect("a nullable label type is admitted; a null label fails at execution");
    for (label, predict_label, dimensions, reason) in [
        (None, false, 2, "missing_label"),
        (Some((DataType::Int64, false)), false, 2, "unsupported_type"),
        (Some((DataType::Int32, false)), true, 2, "predict_relation_has_label"),
        (Some((DataType::Int32, false)), false, 3, "feature_dimension_mismatch"),
    ] {
        let invalid = supervised_plan(label, predict_label, dimensions);
        assert_eq!(preflight_query_plan(&invalid).unwrap_err().fact_value("reason"), Some(reason));
    }
}

fn schema(id: &str, dtype: DataType, dimensions: usize) -> SchemaRef {
    let mut fields = vec![Field::new(id, DataType::Int64, false)];
    fields.extend((0..dimensions).map(|i| Field::new(format!("x{i}"), dtype.clone(), false)));
    Arc::new(Schema::new(fields))
}

fn plan_with_schema(input_schema: SchemaRef, id: &str, components: usize, output_schema: SchemaRef) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let columns = input_schema.fields().iter().skip(1).map(|field| field.name().clone()).collect::<Vec<_>>();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", input_schema)));
    let root = builder
        .add_ml_algorithm(
            vec![input],
            MlAlgorithmSpec::new(
                MlAlgorithm::Tsvd,
                MlAlgorithmParams::Tsvd(MlTsvdParams::new(
                    MlRelationBinding::new(0, VectorRowIdBinding::column(id), columns),
                    components,
                )),
                output_schema,
            ),
        )
        .unwrap();
    builder.finish_with_host_sink(root).unwrap()
}

fn plan(algorithm: MlAlgorithm) -> QueryPlan {
    let input_schema = schema("id", DataType::Float32, 2);
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", input_schema)));
    let binding = MlRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x0", "x1"]);
    let params = match algorithm {
        MlAlgorithm::Tsvd => MlAlgorithmParams::Tsvd(MlTsvdParams::new(binding, 1)),
        MlAlgorithm::Hdbscan
        | MlAlgorithm::Umap
        | MlAlgorithm::LogisticRegression
        | MlAlgorithm::RandomForestClassifier
        | MlAlgorithm::RandomForestRegressor
        | MlAlgorithm::ForestPredict => {
            unreachable!("decomposition plan helper")
        },
    };
    let root = builder
        .add_ml_algorithm(
            vec![input],
            MlAlgorithmSpec::new(algorithm, params, vector_pca_output_schema(DataType::Int64, 1)),
        )
        .unwrap();
    builder.finish_with_host_sink(root).unwrap()
}

#[test]
fn registry_and_options_are_operation_specific() {
    assert_eq!(
        CumlOperation::all().collect::<Vec<_>>(),
        vec![
            CumlOperation::Tsvd,
            CumlOperation::Hdbscan,
            CumlOperation::Umap,
            CumlOperation::LogisticRegression,
            CumlOperation::RandomForestClassifier,
            CumlOperation::RandomForestRegressor,
            CumlOperation::ForestPredict
        ]
    );
    assert_eq!(
        CUML_OPERATION_REGISTRY.iter().map(|entry| entry.sql_function_name).collect::<Vec<_>>(),
        [
            "cuml_tsvd",
            "cuml_hdbscan",
            "cuml_umap",
            "cuml_logistic_regression",
            "cuml_random_forest_classifier",
            "cuml_random_forest_regressor",
            "cuml_forest_predict"
        ]
    );
    for operation in CumlOperation::all() {
        let descriptor = operation.descriptor();
        assert_eq!(
            descriptor.output_contract,
            match operation {
                CumlOperation::Tsvd => CumlOutputContract::Components,
                CumlOperation::Hdbscan => CumlOutputContract::HdbscanClusters,
                CumlOperation::Umap => CumlOutputContract::Embedding,
                CumlOperation::LogisticRegression
                | CumlOperation::RandomForestClassifier
                | CumlOperation::RandomForestRegressor => CumlOutputContract::Predictions,
                CumlOperation::ForestPredict => CumlOutputContract::ForestPrediction,
            }
        );
        assert_eq!(descriptor.vector_element_support, &[VectorElementType::Float32]);
        assert_eq!(
            descriptor.relation_roles,
            if matches!(
                operation,
                CumlOperation::LogisticRegression
                    | CumlOperation::RandomForestClassifier
                    | CumlOperation::RandomForestRegressor
            ) {
                &[MlRelationRole::Training, MlRelationRole::Predict][..]
            } else {
                &[MlRelationRole::Input][..]
            }
        );
        if descriptor.option("n_components").is_some() {
            assert_eq!(descriptor.options_json_schema()["properties"]["n_components"]["maximum"], i32::MAX);
        }
        let raw = if matches!(
            operation,
            CumlOperation::LogisticRegression
                | CumlOperation::RandomForestClassifier
                | CumlOperation::RandomForestRegressor
        ) {
            r#"{"training":{"id":"id","vector":{"columns":["x0","x1"]},"label":"label"},"predict":{"id":"id","vector":{"columns":["x0","x1"]}}}"#
        } else if operation == CumlOperation::ForestPredict {
            r#"{"input":{"id":"id","vector":{"columns":["x0","x1"]}},"model":"m.json","model_format":"xgboost_json"}"#
        } else if operation == CumlOperation::Tsvd {
            r#"{"input":{"id":"id","vector":{"columns":["x0","x1"]}},"n_components":1}"#
        } else {
            r#"{"input":{"id":"id","vector":{"columns":["x0","x1"]}}}"#
        };
        let canonical = canonicalize_cuml_options(operation, raw).unwrap();
        if operation == CumlOperation::Tsvd {
            assert_eq!(canonical["solver"], "cov_eig_dq");
            assert_eq!(canonical["n_iterations"], 15);
        }
        for invalid in [
            r#"{"input":{"id":"id","vector":{"columns":["x0","x1"]}},"extra":0}"#,
            r#"{"input":{"id":"id","vector":{"columns":["x0","x1"]}},"n_components":0}"#,
        ] {
            if invalid.contains("extra") || operation != CumlOperation::Hdbscan {
                assert!(canonicalize_cuml_options(operation, invalid).is_err(), "{invalid}");
            }
        }
    }
    assert!(
        canonicalize_cuml_options(
            CumlOperation::Tsvd,
            r#"{"input":{"id":"id","vector":{"columns":["x0","x1"]}},"n_components":1,"whiten":true}"#
        )
        .is_err()
    );
    let supervised = CumlOperation::LogisticRegression.descriptor().options_json_schema();
    assert_eq!(supervised["properties"]["training"]["required"], serde_json::json!(["id", "vector", "label"]));
    assert!(supervised["properties"]["predict"]["properties"].get("label").is_none());
    let invalid_predict = r#"{"training":{"id":"id","vector":{"columns":["x0"]},"label":"label"},"predict":{"id":"id","vector":{"columns":["x0"]},"label":"label"}}"#;
    assert_eq!(
        canonicalize_cuml_options(CumlOperation::LogisticRegression, invalid_predict).unwrap_err().code(),
        "predict_relation_has_label"
    );
    assert!(CumlOperation::Tsvd.descriptor().option("whiten").is_none());
}

#[test]
fn clustering_and_embedding_options_and_pipeline_contracts() {
    let input = r#"{"input":{"id":"id","vector":{"columns":["x0","x1"]}}"#;
    for (operation, option, value) in [
        (CumlOperation::Hdbscan, "alpha", "0"),
        (CumlOperation::Hdbscan, "alpha", "-1"),
        (CumlOperation::Umap, "learning_rate", "0"),
        (CumlOperation::Umap, "set_op_mix_ratio", "-1"),
        (CumlOperation::Umap, "set_op_mix_ratio", "1.1"),
    ] {
        let raw = format!("{input},\"{option}\":{value}}}");
        assert_eq!(canonicalize_cuml_options(operation, &raw).unwrap_err().argument(), Some(option));
    }
    let raw = format!("{input},\"min_dist\":2,\"spread\":1}}");
    let error = canonicalize_cuml_options(CumlOperation::Umap, &raw).unwrap_err();
    assert_eq!(error.argument(), Some("min_dist"));
    assert!(error.message().contains("spread"));
    assert_eq!(CumlOperation::Umap.descriptor().options_json_schema()["properties"]["set_op_mix_ratio"]["minimum"], 0);
    assert_eq!(CumlOperation::Umap.descriptor().options_json_schema()["properties"]["set_op_mix_ratio"]["maximum"], 1);
    assert_eq!(CumlOperation::Hdbscan.descriptor().options_json_schema()["properties"]["alpha"]["exclusiveMinimum"], 0);
    for algorithm in [MlAlgorithm::Hdbscan, MlAlgorithm::Umap] {
        let mut builder = QueryPlanBuilder::new();
        let source = builder
            .add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema("id", DataType::Float32, 2))));
        let binding = MlRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x0", "x1"]);
        let (params, expected) = match algorithm {
            MlAlgorithm::Hdbscan => {
                (MlAlgorithmParams::Hdbscan(MlHdbscanParams::new(binding)), ml_hdbscan_output_schema(DataType::Int64))
            },
            MlAlgorithm::Umap => {
                (MlAlgorithmParams::Umap(MlUmapParams::new(binding)), ml_embedding_output_schema(DataType::Int64, 2))
            },
            _ => unreachable!(),
        };
        let root = builder
            .add_ml_algorithm(vec![source], MlAlgorithmSpec::new(algorithm, params, Arc::clone(&expected)))
            .unwrap();
        let plan = builder.finish_with_host_sink(root).unwrap();
        preflight_query_plan(&plan).expect("new ML algorithm must be supported");
        let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
        assert!(matches!(
            analysis.nodes.iter().find(|node| node.node_id == root).unwrap().executable,
            CapabilityDecision::Satisfied
        ));
        let pipeline = compile_pipeline(&plan, &analysis).unwrap();
        pipeline.validate().unwrap();
        assert_eq!(
            pipeline.tasks().iter().find(|task| task.plan_node() == Some(root)).unwrap().kind(),
            PipelineTaskKind::MlAlgorithm
        );
        assert_eq!(plan.root_node().output_schema().as_ref(), expected.as_ref());
    }
}

#[test]
fn builder_display_and_pipeline_preserve_ml_contract() {
    let algorithm = MlAlgorithm::Tsvd;
    let plan = plan(algorithm);
    preflight_query_plan(&plan).unwrap();
    let PlanNode::MlAlgorithm(ml) = &plan.root_node().kind else { panic!("ML root") };
    assert_eq!(
        ml.inputs,
        ml.spec.params.relation_bindings().map(|binding| ml.inputs[binding.input]).collect::<Vec<_>>()
    );
    assert_eq!(ml.output_schema().as_ref(), vector_pca_output_schema(DataType::Int64, 1).as_ref());
    assert!(plan.to_string().contains(&format!("algorithm={algorithm}")));
    assert!(plan.to_string().contains("features:x0|x1"));
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let edge = analysis.edges.iter().find(|edge| edge.consumer == EdgeConsumer::Node(plan.root_id())).unwrap();
    assert_eq!(edge.input_role, InputRole::MlAlgorithmInput { role: MlRelationRole::Input });
    assert!(edge.contract_kinds.contains(&EdgeContractKind::MlAlgorithmInput { role: MlRelationRole::Input }));
    let pipeline = compile_pipeline(&plan, &analysis).unwrap();
    pipeline.validate().unwrap();
    let task = pipeline.tasks().iter().find(|task| task.kind() == PipelineTaskKind::MlAlgorithm).unwrap();
    assert_eq!(task.plan_node(), Some(plan.root_id()));
    assert_eq!(task.input_resources().len(), 1);
    let resource = pipeline.resources().iter().find(|resource| resource.id() == task.input_resources()[0]).unwrap();
    assert!(resource.consumers().contains(&task.id()));
}

#[test]
fn validation_and_capability_reject_bad_ml_contracts() {
    let cases = [
        (
            plan_with_schema(
                schema("id", DataType::Float32, 2),
                "id",
                1,
                Arc::new(Schema::new(vec![Field::new("wrong", DataType::Float32, false)])),
            ),
            "unexpected_output_schema",
            None,
        ),
        (
            plan_with_schema(
                schema("other", DataType::Float32, 2),
                "id",
                1,
                vector_pca_output_schema(DataType::Int64, 1),
            ),
            "missing_row_id_column",
            None,
        ),
        (
            plan_with_schema(schema("id", DataType::Float64, 2), "id", 1, vector_pca_output_schema(DataType::Int64, 1)),
            "unsupported_feature_type",
            Some("Float32"),
        ),
        (
            plan_with_schema(schema("id", DataType::Int16, 2), "id", 1, vector_pca_output_schema(DataType::Int64, 1)),
            "unsupported_feature_type",
            Some("Float32"),
        ),
        (
            plan_with_schema(schema("id", DataType::Float32, 2), "id", 3, vector_pca_output_schema(DataType::Int64, 3)),
            "invalid_component_count",
            None,
        ),
        (
            plan_with_schema(schema("id", DataType::Float32, 1), "id", 1, vector_pca_output_schema(DataType::Int64, 1)),
            "insufficient_feature_dimension",
            None,
        ),
    ];
    for (plan, reason, supported) in cases {
        let error = preflight_query_plan(&plan).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some(reason));
        if let Some(supported) = supported {
            assert_eq!(error.fact_value("supported_types"), Some(supported));
        }
        let root = plan
            .capability_analysis(rapids_analysis_context(PlanningPolicy::default()))
            .nodes
            .into_iter()
            .find(|node| node.node_id == plan.root_id())
            .unwrap();
        let CapabilityDecision::Rejected { reason: rejection, facts } = root.executable else { panic!("must reject") };
        assert_eq!(rejection, Reason::UnsupportedMlAlgorithmNodeCapability(UnsupportedPart::from_code(error.code())));
        assert_eq!(facts.get("reason").map(String::as_str), Some(reason));
    }

    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("one_row", schema("id", DataType::Float32, 2)).with_row_count_upper_bound(1),
    ));
    let root = builder
        .add_ml_algorithm(
            vec![input],
            MlAlgorithmSpec::new(
                MlAlgorithm::Tsvd,
                MlAlgorithmParams::Tsvd(MlTsvdParams::new(
                    MlRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x0", "x1"]),
                    1,
                )),
                vector_pca_output_schema(DataType::Int64, 1),
            ),
        )
        .unwrap();
    preflight_query_plan(&builder.finish_with_host_sink(root).unwrap())
        .expect("one-row upper bound can include empty input");
}

#[test]
fn optimizer_remaps_both_supervised_inputs() {
    let training_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("x0", DataType::Float32, false),
        Field::new("x1", DataType::Float32, false),
        Field::new("label", DataType::Int32, false),
    ]));
    let predict_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("x0", DataType::Float32, false),
        Field::new("x1", DataType::Float32, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let mut bindings = Vec::new();
    let mut projections = Vec::new();
    for (name, schema) in [("training", training_schema), ("predict", predict_schema)] {
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(name, Arc::clone(&schema))));
        let projection = builder
            .add_projection(
                source,
                schema
                    .fields()
                    .iter()
                    .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.as_ref().clone()))
                    .collect(),
            )
            .unwrap();
        let mut binding = MlRelationBinding::new(projections.len(), VectorRowIdBinding::column("id"), ["x0", "x1"]);
        projections.push(projection);
        if name == "training" {
            binding.label = Some("label".to_string());
        }
        bindings.push(binding);
    }
    let root = builder
        .add_ml_algorithm(
            projections,
            MlAlgorithmSpec::new(
                MlAlgorithm::LogisticRegression,
                MlAlgorithmParams::LogisticRegression(MlLogisticRegressionParams::new(
                    bindings.remove(0),
                    bindings.remove(0),
                )),
                ml_classification_output_schema(DataType::Utf8),
            ),
        )
        .unwrap();
    let optimized = optimize_query_plan(
        builder.finish_with_host_sink(root).unwrap(),
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap()
    .plan;
    preflight_query_plan(&optimized).unwrap();
    let PlanNode::MlAlgorithm(ml) = &optimized.root_node().kind else { panic!("ML root") };
    assert_eq!(
        ml.inputs,
        ml.spec.params.relation_bindings().map(|binding| ml.inputs[binding.input]).collect::<Vec<_>>()
    );
    assert!(
        ml.inputs
            .iter()
            .all(|input| matches!(optimized.node(*input).map(|node| &node.kind), Some(PlanNode::Source(_))))
    );
}

#[test]
fn rewrite_remaps_ml_relation_inputs() {
    let input_schema = schema("id", DataType::Float32, 2);
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
    let projected = builder
        .add_projection(
            source,
            input_schema
                .fields()
                .iter()
                .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.as_ref().clone()))
                .collect(),
        )
        .unwrap();
    let ml = builder
        .add_ml_algorithm(
            vec![projected],
            MlAlgorithmSpec::new(
                MlAlgorithm::Tsvd,
                MlAlgorithmParams::Tsvd(MlTsvdParams::new(
                    MlRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x0", "x1"]),
                    1,
                )),
                vector_pca_output_schema(DataType::Int64, 1),
            ),
        )
        .unwrap();
    let optimized = optimize_query_plan(
        builder.finish_with_host_sink(ml).unwrap(),
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap()
    .plan;
    preflight_query_plan(&optimized).unwrap();
    let PlanNode::MlAlgorithm(node) = &optimized.root_node().kind else { panic!("ML root") };
    assert_eq!(
        node.inputs,
        node.spec.params.relation_bindings().map(|binding| node.inputs[binding.input]).collect::<Vec<_>>()
    );
    assert!(matches!(optimized.node(node.inputs[0]).map(|node| &node.kind), Some(PlanNode::Source(_))));
}

#[test]
fn ml_pipeline_keeps_relational_handoffs() {
    let input_schema = schema("id", DataType::Float32, 2);
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
    let filtered = builder
        .add_filter(source, Expr::binary(Expr::column("x0"), BinaryOp::Gt, Expr::literal(LiteralValue::Float32(0.0))))
        .unwrap();
    let ml = builder
        .add_ml_algorithm(
            vec![filtered],
            MlAlgorithmSpec::new(
                MlAlgorithm::Tsvd,
                MlAlgorithmParams::Tsvd(MlTsvdParams::new(
                    MlRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x0", "x1"]),
                    1,
                )),
                vector_pca_output_schema(DataType::Int64, 1),
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(ml).unwrap();
    let pipeline =
        compile_pipeline(&plan, &plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()))).unwrap();
    pipeline.validate().unwrap();
    let ml_task = pipeline.tasks().iter().find(|task| task.plan_node() == Some(ml)).unwrap();
    let filter_task = pipeline.tasks().iter().find(|task| task.plan_node() == Some(filtered)).unwrap();
    assert_eq!(ml_task.kind(), PipelineTaskKind::MlAlgorithm);
    let resource = pipeline.resources().iter().find(|resource| resource.id() == ml_task.input_resources()[0]).unwrap();
    assert_eq!(resource.producer(), filter_task.id());
}

const FOREST_FACTS: MlForestModelFacts =
    MlForestModelFacts { num_features: 4, num_trees: 2, native_output_stride: 1, precision: MlForestPrecision::Single };

fn forest_snapshot(source: &[u8], stride: usize, output: MlForestOutputContract) -> MlForestModelSnapshot {
    MlForestModelSnapshot::new(
        source,
        b"checkpoint".to_vec(),
        MlForestModelFormat::XgboostJson,
        MlForestModelFacts { native_output_stride: stride, ..FOREST_FACTS },
        output,
    )
    .unwrap()
}

fn forest_plan(dimensions: usize, model: MlForestModelSnapshot, output_schema: SchemaRef) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "input",
        schema("id", DataType::Float32, dimensions),
    )));
    let binding =
        MlRelationBinding::new(0, VectorRowIdBinding::column("id"), (0..dimensions).map(|index| format!("x{index}")));
    let root = builder
        .add_ml_algorithm(
            vec![input],
            MlAlgorithmSpec::new(
                MlAlgorithm::ForestPredict,
                MlAlgorithmParams::ForestPredict(MlForestPredictParams { input: binding, model }),
                output_schema,
            ),
        )
        .unwrap();
    builder.finish_with_host_sink(root).unwrap()
}

#[test]
fn forest_model_option_is_bounded_text_and_format_is_required() {
    let operation = CumlOperation::ForestPredict;
    let schema = operation.descriptor().options_json_schema();
    assert_eq!(schema["properties"]["model"]["type"], "string");
    assert_eq!(schema["properties"]["model"]["minLength"], 1);
    assert_eq!(schema["properties"]["model"]["maxLength"], 1024);
    assert_eq!(schema["required"], serde_json::json!(["input", "model", "model_format"]));
    let options = |model: &str, format: Option<&str>| {
        let mut value = serde_json::json!({"input": {"id": "id", "vector": {"columns": ["x0"]}}, "model": model});
        if let Some(format) = format {
            value["model_format"] = serde_json::json!(format);
        }
        canonicalize_cuml_options(operation, &value.to_string())
    };
    // 1024 two-byte characters: the bound counts characters, as JSON Schema maxLength does.
    let canonical = options(&"é".repeat(1024), Some("lightgbm")).unwrap();
    assert_eq!(canonical["output"], "scores");
    for (model, format, argument) in [
        ("é".repeat(1025).as_str(), Some("lightgbm"), "model"),
        ("", Some("lightgbm"), "model"),
        ("m.json", None, "model_format"),
        ("m.json", Some("onnx"), "model_format"),
    ] {
        let error = options(model, format).unwrap_err();
        assert_eq!(
            (error.code(), error.argument()),
            ("missing_or_invalid_option", Some(argument)),
            "{model} {format:?}"
        );
    }
}

#[test]
fn forest_output_schemas_follow_the_snapshot_contract() {
    for (stride, output, fields) in [
        (1, MlForestOutputContract::Score, vec![("prediction", DataType::Float32)]),
        (3, MlForestOutputContract::ClassIndex { classes: 3 }, vec![("prediction", DataType::Int32)]),
        (
            3,
            MlForestOutputContract::Vector(3),
            vec![("output_0", DataType::Float32), ("output_1", DataType::Float32), ("output_2", DataType::Float32)],
        ),
    ] {
        let expected = ml_forest_output_schema(DataType::Int64, output);
        assert_eq!(
            expected
                .fields()
                .iter()
                .map(|field| (field.name().as_str(), field.data_type().clone()))
                .collect::<Vec<_>>(),
            [vec![("row_ordinal", DataType::UInt64), ("id", DataType::Int64)], fields].concat()
        );
        assert!(expected.fields().iter().all(|field| !field.is_nullable()));
        let plan = forest_plan(4, forest_snapshot(b"model", stride, output), Arc::clone(&expected));
        preflight_query_plan(&plan).unwrap();
        let text = plan.to_string();
        assert!(text.contains("algorithm=forest_predict"), "{text}");
        assert!(text.contains(&format!("output={output}")), "{text}");
        let mismatched = if stride == 1 {
            ml_forest_output_schema(DataType::Int64, MlForestOutputContract::ClassIndex { classes: 2 })
        } else {
            ml_forest_output_schema(DataType::Int64, MlForestOutputContract::Score)
        };
        let error =
            preflight_query_plan(&forest_plan(4, forest_snapshot(b"model", stride, output), mismatched)).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("unexpected_output_schema"));
    }
}

#[test]
fn forest_feature_count_mismatch_rejects_preflight_and_capability() {
    let output = MlForestOutputContract::Score;
    let plan = forest_plan(3, forest_snapshot(b"model", 1, output), ml_forest_output_schema(DataType::Int64, output));
    let error = preflight_query_plan(&plan).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("forest_feature_count_mismatch"));
    let root = plan
        .capability_analysis(rapids_analysis_context(PlanningPolicy::default()))
        .nodes
        .into_iter()
        .find(|node| node.node_id == plan.root_id())
        .unwrap();
    let CapabilityDecision::Rejected { reason, facts } = root.executable else { panic!("must reject") };
    assert_eq!(reason, Reason::UnsupportedMlAlgorithmNodeCapability(UnsupportedPart::Operator));
    assert_eq!(facts.get("reason").map(String::as_str), Some("forest_feature_count_mismatch"));
    assert_eq!(facts.get("expected").map(String::as_str), Some("4"));
    assert_eq!(facts.get("actual").map(String::as_str), Some("3"));
    assert_eq!(facts.get("validation_code").map(String::as_str), Some("invalid_plan"));
    assert_eq!(facts.get("algorithm").map(String::as_str), Some("forest_predict"));
}

#[test]
fn forest_plan_identity_tracks_model_bytes_without_embedding_them() {
    let output = MlForestOutputContract::Score;
    let plan_for = |source: &[u8]| {
        forest_plan(4, forest_snapshot(source, 1, output), ml_forest_output_schema(DataType::Int64, output))
    };
    let original = plan_for(b"model bytes v1");
    let fingerprint = original.execution_fingerprint();
    assert_eq!(fingerprint, plan_for(b"model bytes v1").execution_fingerprint());
    assert_ne!(fingerprint, plan_for(b"model bytes v2").execution_fingerprint());
    let PlanNode::MlAlgorithm(node) = &original.root_node().kind else { panic!("forest root") };
    let MlAlgorithmParams::ForestPredict(params) = &node.spec.params else { panic!("forest params") };
    assert!(original.to_string().contains(&format!("model=sha256:{} ", params.model.digest().short_hex())));
}
