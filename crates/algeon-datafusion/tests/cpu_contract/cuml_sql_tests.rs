#[cfg(not(feature = "cuml"))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::BTreeSet, fs, sync::Arc};

use algeon_datafusion::{
    cuml_sql::{
        CUML_HDBSCAN_FUNCTION, CUML_LOGISTIC_REGRESSION_FUNCTION, CUML_OPERATION_REGISTRY,
        CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION, CUML_RANDOM_FOREST_REGRESSOR_FUNCTION, CUML_TSVD_FUNCTION,
        CUML_UMAP_FUNCTION, CumlAlgoExec, CumlLogicalNode, CumlOperation, GpuOptionKind, MlRelationRole,
        cuml_options_json_schema,
    },
    error::{DataFusionErrorExt, ErrorCode},
    session::AlgeonSessionStateBuilderExt,
};
#[cfg(feature = "cuml")]
use arrow::array::{BooleanArray, StringArray};
#[cfg(not(feature = "cuml"))]
use arrow::array::{BooleanArray, StringArray};
use arrow::{
    array::Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
#[cfg(not(feature = "cuml"))]
use datafusion::common::tree_node::TreeNodeRecursion;
#[cfg(not(feature = "cuml"))]
use datafusion::execution::context::TaskContext;
use datafusion::{
    catalog::Session,
    common::{DFSchema, Result as DataFusionResult},
    datasource::MemTable,
    execution::{SessionStateBuilder, context::QueryPlanner},
    logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore},
    physical_plan::ExecutionPlan,
    physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner},
    prelude::SessionContext,
};
#[cfg(not(feature = "cuml"))]
use datafusion::{
    physical_expr::PhysicalExpr,
    physical_plan::{
        ChildrenPropertiesMode, DisplayAs, DisplayFormatType, PlanProperties, ReplaceChildrenOptions,
        SendableRecordBatchStream,
    },
};
use query_engine::plan::{
    CumlOutputContract, VectorElementType, ml_embedding_output_schema, ml_hdbscan_output_schema,
    ml_predictions_output_schema, vector_pca_output_schema,
};

fn context() -> SessionContext {
    let ctx = SessionContext::new_with_state(
        SessionStateBuilder::new_with_default_features().try_with_cuml_sql().unwrap().build(),
    );
    for (name, id, feature) in
        [("vectors", DataType::Int64, DataType::Float32), ("double_vectors", DataType::Int64, DataType::Float64)]
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", id, false),
            Field::new("d0", feature.clone(), false),
            Field::new("d1", feature, false),
        ]));
        ctx.register_table(
            name,
            Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![RecordBatch::new_empty(schema)]]).unwrap()),
        )
        .unwrap();
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, true),
        Field::new("d1", DataType::Float32, false),
    ]));
    ctx.register_table(
        "nullable_vectors",
        Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![RecordBatch::new_empty(schema)]]).unwrap()),
    )
    .unwrap();
    for (name, dtype, nullable) in [
        ("training_vectors", DataType::Int32, false),
        ("training_int64_labels", DataType::Int64, false),
        ("training_nullable_labels", DataType::Int32, true),
        ("training_regression", DataType::Float32, false),
    ] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("d0", DataType::Float32, false),
            Field::new("d1", DataType::Float32, false),
            Field::new("label", dtype, nullable),
        ]));
        ctx.register_table(
            name,
            Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![RecordBatch::new_empty(schema)]]).unwrap()),
        )
        .unwrap();
    }
    ctx
}

fn sql(function: &str, table: &str, components: usize) -> String {
    if matches!(
        function,
        CUML_LOGISTIC_REGRESSION_FUNCTION | "cuml_random_forest_classifier" | "cuml_random_forest_regressor"
    ) {
        let label_type = if function == "cuml_random_forest_regressor" { "REAL" } else { "INT" };
        return format!(
            "SELECT * FROM {function}(training => (SELECT id, d0, d1, CAST(0 AS {label_type}) AS label FROM {table}), predict => (SELECT id, d0, d1 FROM {table}))"
        );
    }
    let options = if matches!(function, CUML_HDBSCAN_FUNCTION | CUML_UMAP_FUNCTION) {
        String::new()
    } else {
        format!(", n_components => {components}")
    };
    format!("SELECT * FROM {function}(input => (SELECT id, d0, d1 FROM {table}){options})")
}

#[test]
fn registry_drives_names_roles_options_and_schema() {
    assert_eq!(
        CUML_OPERATION_REGISTRY.iter().map(|entry| entry.sql_function_name).collect::<Vec<_>>(),
        [
            CUML_TSVD_FUNCTION,
            CUML_HDBSCAN_FUNCTION,
            CUML_UMAP_FUNCTION,
            CUML_LOGISTIC_REGRESSION_FUNCTION,
            CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION,
            CUML_RANDOM_FOREST_REGRESSOR_FUNCTION
        ]
    );
    for operation in CumlOperation::all() {
        let entry = operation.descriptor();
        assert_eq!(
            entry.relation_roles,
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
        let schema = cuml_options_json_schema(operation);
        assert_eq!(schema, entry.options_json_schema());
        assert_eq!(
            schema["required"],
            if operation == CumlOperation::Tsvd {
                serde_json::json!(["input", "n_components"])
            } else if matches!(
                operation,
                CumlOperation::LogisticRegression
                    | CumlOperation::RandomForestClassifier
                    | CumlOperation::RandomForestRegressor
            ) {
                serde_json::json!(["training", "predict"])
            } else {
                serde_json::json!(["input"])
            }
        );
        if operation == CumlOperation::Tsvd {
            assert_eq!(schema["properties"]["n_components"]["minimum"], 1);
            assert_eq!(schema["properties"]["solver"]["default"], "cov_eig_dq");
        }
        if operation == CumlOperation::Hdbscan {
            assert_eq!(schema["properties"]["alpha"]["exclusiveMinimum"], 0);
        }
        if operation == CumlOperation::Umap {
            assert_eq!(schema["properties"]["set_op_mix_ratio"]["minimum"], 0);
            assert_eq!(schema["properties"]["set_op_mix_ratio"]["maximum"], 1);
            assert_eq!(schema["properties"]["spread"]["exclusiveMinimum"], 0);
            assert!(schema["properties"].get("transform_queue_size").is_none());
        }
    }
}

#[tokio::test]
async fn new_options_and_exact_row_count_have_structured_validation() {
    let ctx = context();
    for (function, option, value) in [
        (CUML_HDBSCAN_FUNCTION, "alpha", "0"),
        (CUML_HDBSCAN_FUNCTION, "alpha", "-1"),
        (CUML_UMAP_FUNCTION, "learning_rate", "0"),
        (CUML_UMAP_FUNCTION, "set_op_mix_ratio", "-0.1"),
        (CUML_UMAP_FUNCTION, "set_op_mix_ratio", "1.1"),
    ] {
        let sql = format!("SELECT * FROM {function}(input => (SELECT id, d0, d1 FROM vectors), {option} => {value})");
        let error = ctx.state().create_logical_plan(&sql).await.unwrap_err();
        assert_eq!(error.find_algeon_error().unwrap().fact_value("argument"), Some(option), "{sql}");
    }
    let error = ctx
        .state()
        .create_logical_plan(
            "SELECT * FROM cuml_umap(input => (SELECT id, d0, d1 FROM vectors), min_dist => 2.0, spread => 1.0)",
        )
        .await
        .unwrap_err();
    assert_eq!(error.find_algeon_error().unwrap().fact_value("argument"), Some("min_dist"));

    let row = ctx
        .state()
        .create_logical_plan("SELECT CAST(1 AS BIGINT) AS id, CAST(1 AS REAL) AS d0, CAST(2 AS REAL) AS d1")
        .await
        .unwrap();
    let empty = ctx
        .state()
        .create_logical_plan("SELECT CAST(1 AS BIGINT) AS id, CAST(1 AS REAL) AS d0, CAST(2 AS REAL) AS d1 LIMIT 0")
        .await
        .unwrap();
    for operation in CumlOperation::all().filter(|operation| {
        !matches!(
            operation,
            CumlOperation::LogisticRegression
                | CumlOperation::RandomForestClassifier
                | CumlOperation::RandomForestRegressor
        )
    }) {
        let values = if operation == CumlOperation::Tsvd {
            serde_json::json!({"n_components": 1})
        } else if operation == CumlOperation::Umap {
            serde_json::json!({"n_neighbors": 2})
        } else {
            serde_json::json!({"min_samples": 2, "min_cluster_size": 2})
        };
        let bound = |plan: &LogicalPlan| {
            let mut options = values.clone();
            options["input"] = serde_json::json!({"id":"id","vector":{"columns":["d0","d1"]}});
            CumlLogicalNode::try_new(operation, vec![plan.clone()], &options.to_string())
        };
        let error = bound(&row).unwrap_err();
        assert_eq!(error.find_algeon_error().unwrap().fact_value("reason"), Some("insufficient_input_rows"));
        bound(&empty).expect("known empty input is valid");
    }
}

#[test]
fn captured_specs_and_generated_reference_pages_match_the_canonical_registry() {
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("adapter manifest must be two levels below the repository root");
    let spec_directory = repository.join("website/gpu-function-specs/cuml");
    let page_directory = repository.join("website/docs/cuml/functions");
    let expected_names =
        CUML_OPERATION_REGISTRY.iter().map(|descriptor| descriptor.sql_function_name).collect::<BTreeSet<_>>();
    let captured_names = fs::read_dir(&spec_directory)
        .unwrap_or_else(|error| panic!("read captured cuML specs at {}: {error}", spec_directory.display()))
        .map(|entry| entry.expect("read captured cuML spec entry"))
        .filter_map(|entry| {
            let path = entry.path();
            (path.extension().and_then(|extension| extension.to_str()) == Some("json"))
                .then(|| path.file_stem().and_then(|stem| stem.to_str()).expect("UTF-8 spec file name").to_string())
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(captured_names.iter().map(String::as_str).collect::<BTreeSet<_>>(), expected_names);

    for descriptor in CUML_OPERATION_REGISTRY {
        let spec_path = spec_directory.join(format!("{}.json", descriptor.sql_function_name));
        let captured: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(&spec_path)
                .unwrap_or_else(|error| panic!("read captured cuML spec at {}: {error}", spec_path.display())),
        )
        .unwrap_or_else(|error| panic!("parse captured cuML spec at {}: {error}", spec_path.display()));
        assert_eq!(captured["function_name"], descriptor.sql_function_name);
        assert_eq!(captured["provider"], "cuml");
        assert_eq!(captured["summary"], descriptor.summary);
        let options_schema =
            serde_json::from_str::<serde_json::Value>(captured["options_schema_json"].as_str().unwrap()).unwrap();
        let mut expected_options_schema = descriptor.options_json_schema();
        let expected = expected_options_schema.as_object_mut().unwrap();
        let properties = expected["properties"].as_object_mut().unwrap();
        for role in descriptor.relation_roles {
            properties.remove(role.as_str());
        }
        expected["required"].as_array_mut().unwrap().retain(|name| {
            name.as_str().is_none_or(|name| !descriptor.relation_roles.iter().any(|role| role.as_str() == name))
        });
        assert_eq!(
            options_schema, expected_options_schema,
            "captured value options schema must be an exact projection of the canonical registry"
        );
        for option in descriptor.options.iter().filter(|option| !matches!(option.kind, GpuOptionKind::RelationBinding))
        {
            assert_eq!(
                options_schema["properties"][option.name]["description"],
                descriptor.options_json_schema()["properties"][option.name]["description"],
                "{} option {} must retain the canonical description",
                descriptor.sql_function_name,
                option.name
            );
        }
        let roles =
            serde_json::from_str::<serde_json::Value>(captured["relation_roles_json"].as_str().unwrap()).unwrap();
        assert_eq!(
            roles.as_array().unwrap().iter().map(|role| role["name"].as_str().unwrap()).collect::<Vec<_>>(),
            descriptor.relation_roles.iter().map(|role| role.as_str()).collect::<Vec<_>>(),
            "captured relation roles must be an exact projection of the canonical registry"
        );
        for (role, captured_role) in descriptor.relation_roles.iter().zip(roles.as_array().unwrap()) {
            assert_eq!(captured_role["description"], role.description());
            assert_eq!(
                captured_role["fields"]["features"]["element_types"],
                serde_json::json!(
                    descriptor.vector_element_support.iter().map(|element| element.as_str()).collect::<Vec<_>>()
                )
            );
        }
        let result_schema =
            serde_json::from_str::<serde_json::Value>(captured["result_schemas_json"].as_str().unwrap()).unwrap();
        assert_eq!(
            result_schema["output_contract"],
            match descriptor.output_contract {
                CumlOutputContract::Components => "components",
                CumlOutputContract::HdbscanClusters => "hdbscan_clusters",
                CumlOutputContract::Embedding => "embedding",
                CumlOutputContract::Predictions => "predictions",
            }
        );
        let fields = result_schema["fields"].as_array().unwrap();
        assert_eq!(fields.len(), descriptor.output_fields.len());
        for (field, expected) in fields.iter().zip(descriptor.output_fields) {
            assert_eq!(field["name"], expected.name);
            assert_eq!(field["data_type"], expected.data_type);
            assert_eq!(field["nullable"], expected.nullable);
            assert_eq!(field["description"], expected.description);
            assert_eq!(field.get("repeats").and_then(serde_json::Value::as_str), expected.repeats);
        }
        let examples = serde_json::from_str::<serde_json::Value>(captured["examples_json"].as_str().unwrap()).unwrap();
        let quickstart = &examples[0]["quickstart"];
        let limitations =
            serde_json::from_str::<serde_json::Value>(captured["limitations_json"].as_str().unwrap()).unwrap();
        let captured_limits = limitations.as_array().unwrap();
        assert!(
            captured_limits
                .ends_with(&descriptor.limitations.iter().map(|limit| serde_json::json!(limit)).collect::<Vec<_>>())
        );
        assert_eq!(quickstart["display_name"], descriptor.display_name);
        assert_eq!(quickstart["use_this_when"], descriptor.use_this_when);
        assert_eq!(quickstart["sql"], descriptor.quickstart_sql);

        let page_path = page_directory.join(format!("{}.mdx", descriptor.sql_function_name));
        let page = fs::read_to_string(&page_path)
            .unwrap_or_else(|error| panic!("read generated cuML page at {}: {error}", page_path.display()));
        assert!(
            page.contains("GENERATED FROM the unified GPU function catalog by website/scripts/gen-cuml-docs.mjs"),
            "{} must retain its generated-source marker",
            page_path.display()
        );
        assert!(page.contains(&format!("# {}", descriptor.display_name)), "{}", page_path.display());
        assert!(page.contains(&format!("SQL function: `{}`", descriptor.sql_function_name)), "{}", page_path.display());
        assert!(page.contains(descriptor.quickstart_sql), "{}", page_path.display());
        assert!(page.contains(descriptor.summary), "{}", page_path.display());
        for limit in descriptor.limitations {
            assert!(page.contains(&mdx_table_text(limit)), "{}", page_path.display());
        }
        let headings = [
            "## Quickstart",
            "## Inputs",
            "### Vector element types",
            "## Arguments and options",
            "## Output",
            "## Limits",
        ];
        let mut previous_heading = 0;
        for heading in headings {
            let index = page.find(heading).unwrap_or_else(|| panic!("{} is missing {heading}", page_path.display()));
            assert!(index >= previous_heading, "{} puts {heading} out of order", page_path.display());
            previous_heading = index;
        }
        for role in descriptor.relation_roles {
            assert!(page.contains(&format!("`{}`", role.as_str())), "{}", page_path.display());
            assert!(page.contains(role.description()), "{}", page_path.display());
        }
        for element in descriptor.vector_element_support {
            assert!(page.contains(&format!("| `{}` |", element.as_str())), "{}", page_path.display());
        }
        assert_eq!(descriptor.vector_element_support, &[VectorElementType::Float32]);
        for option in descriptor.options.iter().filter(|option| !matches!(option.kind, GpuOptionKind::RelationBinding))
        {
            let row = page
                .lines()
                .rfind(|line| line.starts_with(&format!("| `{}` |", option.name)))
                .unwrap_or_else(|| panic!("{} is missing option {}", page_path.display(), option.name));
            assert!(row.contains(&mdx_table_text(option.description)), "{}", page_path.display());
            let schema = &options_schema["properties"][option.name];
            let default = schema
                .get("default")
                .map_or_else(|| "No default".to_string(), |value| format!("`{}`", json_literal_for_docs(value)));
            assert!(row.contains(&format!(" | {default} |")), "{} option {} default", page_path.display(), option.name);
            for (key, label) in [("minimum", "minimum"), ("maximum", "maximum")] {
                if let Some(bound) = schema.get(key) {
                    assert!(
                        row.contains(&format!("{label} {}", json_literal_for_docs(bound))),
                        "{} option {} {key}",
                        page_path.display(),
                        option.name
                    );
                }
            }
        }
        for field in descriptor.output_fields {
            assert!(page.contains(&format!("`{}`", field.name)), "{}", page_path.display());
            assert!(page.contains(&mdx_table_text(field.description)), "{}", page_path.display());
        }
    }
}

fn mdx_table_text(value: &str) -> String {
    value.replace('<', "&lt;").replace('>', "&gt;")
}

fn json_literal_for_docs(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Number(number) if number.is_f64() => number.as_f64().unwrap().to_string(),
        _ => serde_json::to_string(value).unwrap(),
    }
}

#[tokio::test]
async fn sql_schemas_match_ir_and_physical_nodes() {
    let ctx = context();
    for operation in CumlOperation::all() {
        let logical = ctx
            .state()
            .create_logical_plan(&sql(operation.descriptor().sql_function_name, "vectors", 2))
            .await
            .unwrap();
        let node = find_logical(&logical).expect("cuML logical node");
        assert_eq!(node.operation(), operation);
        assert_eq!(
            node.schema().as_ref(),
            &DFSchema::try_from(
                match operation {
                    CumlOperation::Tsvd => vector_pca_output_schema(DataType::Int64, 2),
                    CumlOperation::Hdbscan => ml_hdbscan_output_schema(DataType::Int64),
                    CumlOperation::Umap => ml_embedding_output_schema(DataType::Int64, 2),
                    CumlOperation::LogisticRegression | CumlOperation::RandomForestClassifier =>
                        ml_predictions_output_schema(DataType::Int64, DataType::Int32),
                    CumlOperation::RandomForestRegressor =>
                        ml_predictions_output_schema(DataType::Int64, DataType::Float32),
                }
                .as_ref()
                .clone()
            )
            .unwrap()
        );
        let physical = ctx.state().create_physical_plan(&logical).await.unwrap();
        assert!(find_exec(physical.as_ref()).is_some(), "{}", operation.descriptor().sql_function_name);
    }
}

#[derive(Debug)]
struct OpaquePlanner;
#[async_trait::async_trait]
impl QueryPlanner for OpaquePlanner {
    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
        session: &dyn Session,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::default().create_physical_plan(plan, session).await
    }
}

#[test]
fn installation_is_idempotent_and_rejects_opaque_planners() {
    let mut builder =
        SessionStateBuilder::new_with_default_features().try_with_cuml_sql().unwrap().try_with_cuml_sql().unwrap();
    assert!(builder.query_planner().is_some());
    let error = SessionStateBuilder::new_with_default_features()
        .with_query_planner(Arc::new(OpaquePlanner))
        .try_with_cuml_sql()
        .unwrap_err();
    assert_eq!(error.find_algeon_error().unwrap().fact_value("reason"), Some("query_planner_conflict"));
}

#[tokio::test]
async fn named_binder_accepts_reordering_and_rejects_bad_arguments() {
    let ctx = context();
    let normal = sql(CUML_TSVD_FUNCTION, "vectors", 1);
    let reordered = "SELECT * FROM cuml_tsvd(n_components => 1, input => (SELECT id, d0, d1 FROM vectors))";
    assert_eq!(
        ctx.state().create_logical_plan(&normal).await.unwrap().display_indent().to_string(),
        ctx.state().create_logical_plan(reordered).await.unwrap().display_indent().to_string()
    );
    for (statement, reason) in [
        (
            "SELECT * FROM cuml_tsvd(input => (SELECT id, d0, d1 FROM vectors), n_components => 1, n_components => 2)",
            "duplicate_argument",
        ),
        ("SELECT * FROM cuml_tsvd(input => (SELECT id, d0, d1 FROM vectors))", "missing_argument"),
        (
            "SELECT * FROM cuml_tsvd(input => (SELECT id, d0, d1 FROM vectors), n_components => 1, bogus => 1)",
            "unknown_argument",
        ),
        ("SELECT * FROM cuml_tsvd((SELECT id, d0, d1 FROM vectors), 1)", "positional_argument_not_supported"),
    ] {
        let error = ctx.state().create_logical_plan(statement).await.unwrap_err();
        assert_eq!(error.find_algeon_error().unwrap().fact_value("reason"), Some(reason), "{statement}");
    }
}

#[tokio::test]
async fn supervised_sql_bindings_check_labels_dimensions_and_predict_id_type() {
    let ctx = context();
    let normal = "SELECT * FROM cuml_logistic_regression(training => (SELECT id, d0, d1, label FROM training_vectors), predict => (SELECT CAST(id AS VARCHAR) AS id, d0, d1 FROM vectors))";
    let reordered = "SELECT * FROM cuml_logistic_regression(predict => (SELECT CAST(id AS VARCHAR) AS id, d0, d1 FROM vectors), training => (SELECT id, d0, d1, label FROM training_vectors))";
    let first = ctx.state().create_logical_plan(normal).await.unwrap();
    assert_eq!(
        first.display_indent().to_string(),
        ctx.state().create_logical_plan(reordered).await.unwrap().display_indent().to_string()
    );
    let node = find_logical(&first).unwrap();
    assert_eq!(
        node.schema().as_ref(),
        &DFSchema::try_from(ml_predictions_output_schema(DataType::Utf8View, DataType::Int32).as_ref().clone())
            .unwrap()
    );
    for (statement, reason) in [
        (
            "SELECT * FROM cuml_logistic_regression(training => (SELECT id, d0, d1, label FROM training_vectors))",
            "missing_argument",
        ),
        (
            "SELECT * FROM cuml_logistic_regression((SELECT id, d0, d1, label FROM training_vectors), (SELECT id, d0, d1 FROM vectors))",
            "positional_argument_not_supported",
        ),
        (
            "SELECT * FROM cuml_logistic_regression(training => (SELECT id, d0, d1 FROM vectors), predict => (SELECT id, d0, d1 FROM vectors))",
            "missing_label",
        ),
        (
            "SELECT * FROM cuml_logistic_regression(training => (SELECT id, d0, d1, label FROM training_vectors), predict => (SELECT id, d0, d1, label FROM training_vectors))",
            "predict_relation_has_label",
        ),
        (
            "SELECT * FROM cuml_logistic_regression(training => (SELECT id, d0, d1, label FROM training_int64_labels), predict => (SELECT id, d0, d1 FROM vectors))",
            "unsupported_type",
        ),
        (
            "SELECT * FROM cuml_logistic_regression(training => (SELECT id, d0, d1, label FROM training_nullable_labels), predict => (SELECT id, d0, d1 FROM vectors))",
            "nullable_label",
        ),
        (
            "SELECT * FROM cuml_logistic_regression(training => (SELECT id, d0, d1, label FROM training_vectors), predict => (SELECT id, d0 FROM vectors))",
            "feature_dimension_mismatch",
        ),
    ] {
        let error = ctx.state().create_logical_plan(statement).await.unwrap_err();
        let typed = error.find_algeon_error().expect(statement);
        if reason == "unsupported_type" {
            assert_eq!(typed.code(), ErrorCode::UnsupportedType, "{statement}: {error}");
        } else {
            assert_eq!(typed.fact_value("reason"), Some(reason), "{statement}: {error}");
        }
    }
}

#[tokio::test]
async fn random_forest_sql_rejects_wrong_labels_and_classifier_only_criterion() {
    let ctx = context();
    for (function, table, reason) in [
        (CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION, "training_regression", "unsupported_type"),
        (CUML_RANDOM_FOREST_REGRESSOR_FUNCTION, "training_vectors", "unsupported_type"),
        (CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION, "training_vectors", "predict_relation_has_label"),
    ] {
        let predict = if reason == "predict_relation_has_label" { "training_vectors" } else { "vectors" };
        let label = if reason == "predict_relation_has_label" { ", label" } else { "" };
        let sql = format!(
            "SELECT * FROM {function}(training => (SELECT id, d0, d1, label FROM {table}), predict => (SELECT id, d0, d1{label} FROM {predict}))"
        );
        let error = ctx.state().create_logical_plan(&sql).await.unwrap_err();
        let typed = error.find_algeon_error().unwrap();
        if reason == "unsupported_type" {
            assert_eq!(typed.code(), ErrorCode::UnsupportedType);
        } else {
            assert_eq!(typed.fact_value("reason"), Some(reason));
        }
    }
    let sql = format!(
        "SELECT * FROM {}(training => (SELECT id, d0, d1, label FROM training_regression), predict => (SELECT id, d0, d1 FROM vectors), split_criterion => 'gini')",
        CUML_RANDOM_FOREST_REGRESSOR_FUNCTION
    );
    assert_eq!(
        ctx.state().create_logical_plan(&sql).await.unwrap_err().find_algeon_error().unwrap().fact_value("reason"),
        Some("unknown_argument")
    );
    let schema = cuml_options_json_schema(CumlOperation::RandomForestRegressor);
    assert!(schema["properties"]["max_leaves"].get("default").is_none());
    assert_eq!(schema["properties"]["training"]["properties"]["label"]["const"], "label");
}

#[tokio::test]
async fn schema_rejections_preserve_structured_reasons_and_float32_support() {
    let ctx = context();
    for (table, reason) in [("double_vectors", None), ("nullable_vectors", Some("nullable_feature_field"))] {
        let error = ctx.state().create_logical_plan(&sql(CUML_TSVD_FUNCTION, table, 1)).await.unwrap_err();
        let typed = error.find_algeon_error().unwrap();
        if let Some(reason) = reason {
            assert_eq!(typed.fact_value("reason"), Some(reason));
        } else {
            assert_eq!(typed.code(), ErrorCode::UnsupportedType);
            assert_eq!(typed.fact_value("expected_type"), Some("Float32"));
        }
    }
    let error = ctx.state().create_logical_plan(&sql(CUML_TSVD_FUNCTION, "vectors", 3)).await.unwrap_err();
    assert_eq!(error.find_algeon_error().unwrap().fact_value("reason"), Some("component_count_exceeds_dimension"));
}

#[cfg(feature = "cuml")]
#[tokio::test]
async fn catalog_reports_validation_paths_for_option_and_relation_fields() {
    let ctx = context();
    let listed = ctx
        .sql("SELECT function_name, available FROM gpu_list_functions() WHERE provider = 'cuml' ORDER BY function_name")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(listed.iter().map(RecordBatch::num_rows).sum::<usize>(), CUML_OPERATION_REGISTRY.len());
    let available = listed[0].column(1).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!((0..CUML_OPERATION_REGISTRY.len()).all(|index| available.value(index)));
    for (relation, components, code, path) in [
        ("vectors", 0, "missing_or_invalid_option", "options.n_components"),
        ("double_vectors", 1, "unsupported_type", "relations.input.fields.d0"),
    ] {
        let call =
            serde_json::json!({"relations": {"input": {"table": relation}}, "options": {"n_components": components}})
                .to_string();
        let result = ctx
            .sql(&format!("SELECT error_code, details_json FROM gpu_validate_call('cuml_tsvd', '{call}')"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let row = &result[0];
        assert_eq!(row.column(0).as_any().downcast_ref::<StringArray>().unwrap().value(0), code);
        let details: serde_json::Value =
            serde_json::from_str(row.column(1).as_any().downcast_ref::<StringArray>().unwrap().value(0)).unwrap();
        assert_eq!(details["path"], path);
    }
}

#[cfg(not(feature = "cuml"))]
#[tokio::test]
async fn execution_rejects_a_disabled_cuml_feature_before_any_child_executes() {
    let ctx = context();
    let listed = ctx
        .sql("SELECT available, unavailable_reason FROM gpu_list_functions() WHERE function_name = 'cuml_tsvd'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert!(!listed[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap().value(0));
    assert_eq!(
        listed[0].column(1).as_any().downcast_ref::<StringArray>().unwrap().value(0),
        "required_feature_disabled"
    );
    let physical = ctx.sql(&sql(CUML_TSVD_FUNCTION, "vectors", 1)).await.unwrap().create_physical_plan().await.unwrap();
    let exec = find_exec(physical.as_ref()).unwrap();
    let counter = Arc::new(AtomicUsize::new(0));
    let child: Arc<dyn ExecutionPlan> =
        Arc::new(CountingChild { inner: Arc::clone(exec.children()[0]), counter: Arc::clone(&counter) });
    let replaced = Arc::new(exec.clone())
        .replace_children(vec![child], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        .unwrap();
    let error = match replaced.execute(0, ctx.task_ctx()) {
        Ok(_) => panic!("disabled cuML feature must reject before child execution"),
        Err(error) => error,
    };
    let typed = error.find_algeon_error().unwrap();
    assert_eq!(typed.fact_value("required_feature"), Some("cuml"));
    assert_eq!(typed.fact_value("host_fallback"), Some("disabled"));
    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

fn find_logical(plan: &LogicalPlan) -> Option<&CumlLogicalNode> {
    if let LogicalPlan::Extension(extension) = plan
        && let Some(node) = extension.node.as_any().downcast_ref::<CumlLogicalNode>()
    {
        return Some(node);
    }
    plan.inputs().into_iter().find_map(find_logical)
}
fn find_exec(plan: &dyn ExecutionPlan) -> Option<&CumlAlgoExec> {
    plan.downcast_ref::<CumlAlgoExec>()
        .or_else(|| plan.children().into_iter().find_map(|child| find_exec(child.as_ref())))
}

#[cfg(not(feature = "cuml"))]
#[derive(Debug)]
struct CountingChild {
    inner: Arc<dyn ExecutionPlan>,
    counter: Arc<AtomicUsize>,
}
#[cfg(not(feature = "cuml"))]
impl DisplayAs for CountingChild {
    fn fmt_as(&self, format: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt_as(format, f)
    }
}
#[cfg(not(feature = "cuml"))]
impl ExecutionPlan for CountingChild {
    fn name(&self) -> &str {
        "CountingChild"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        self.inner.properties()
    }
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.inner]
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            inner: Arc::clone(&self.inner)
                .replace_children(children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))?,
            counter: Arc::clone(&self.counter),
        }))
    }
    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        self.counter.fetch_add(1, Ordering::SeqCst);
        self.inner.execute(partition, context)
    }
}
