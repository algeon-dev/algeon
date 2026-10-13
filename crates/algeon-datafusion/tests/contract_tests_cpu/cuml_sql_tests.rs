#[cfg(not(feature = "cuml"))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::BTreeSet,
    fs, iter,
    path::{Path, PathBuf},
    sync::Arc,
};
#[cfg(feature = "cuml")]
use std::{os::unix::fs::symlink, process::Command, sync::mpsc, thread, time::Duration};

use algeon_datafusion::{
    cuml_sql::{
        CUML_FOREST_PREDICT_FUNCTION, CUML_HDBSCAN_FUNCTION, CUML_LOGISTIC_REGRESSION_FUNCTION,
        CUML_OPERATION_REGISTRY, CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION, CUML_RANDOM_FOREST_REGRESSOR_FUNCTION,
        CUML_TSVD_FUNCTION, CUML_UMAP_FUNCTION, CumlAlgoExec, CumlLogicalNode, CumlModelStoreConfig, CumlOperation,
        CumlSqlConfig, GpuOptionKind, MlRelationRole, cuml_options_json_schema,
    },
    error::{DataFusionErrorExt, ErrorCode},
    gpu_coverage::validate_query,
    planner::AlgeonNativeOptimizerConfig,
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{Array, BooleanArray, StringArray},
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
#[cfg(feature = "cuml")]
use query_planning::plan::MlForestModelDigest;
use query_planning::plan::{
    CumlOutputContract, MlForestModelFacts, MlForestModelFormat, MlForestModelSnapshot, MlForestOutputContract,
    MlForestPrecision, VectorElementType, ml_classification_output_schema, ml_embedding_output_schema,
    ml_forest_output_schema, ml_hdbscan_output_schema, ml_regression_output_schema, vector_pca_output_schema,
};
#[cfg(feature = "cuml")]
use tokio::runtime::Builder;

fn context() -> SessionContext {
    context_with(CumlSqlConfig::default())
}

fn context_with(config: CumlSqlConfig) -> SessionContext {
    register_fixtures(SessionContext::new_with_state(
        SessionStateBuilder::new_with_default_features().try_with_cuml_sql_config(config).unwrap().build(),
    ))
}

fn native_context() -> SessionContext {
    register_fixtures(SessionContext::new_with_state(
        SessionStateBuilder::new_with_default_features()
            .with_algeon_native(AlgeonNativeOptimizerConfig::default())
            .try_with_cuml_sql_config(CumlSqlConfig::default())
            .unwrap()
            .build(),
    ))
}

fn register_fixtures(ctx: SessionContext) -> SessionContext {
    for (name, id, feature) in [
        ("vectors", DataType::Int64, DataType::Float32),
        ("double_vectors", DataType::Int64, DataType::Float64),
        ("integer_vectors", DataType::Int64, DataType::Int16),
    ] {
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
        Field::new("id", DataType::Int64, true),
        Field::new("d0", DataType::Float32, true),
        Field::new("d1", DataType::Float32, true),
    ]));
    ctx.register_table(
        "nullable_vectors",
        Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![RecordBatch::new_empty(schema)]]).unwrap()),
    )
    .unwrap();
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    for (name, vector) in [
        ("nullable_list_vectors", DataType::List(Arc::clone(&item))),
        ("nullable_fixed_vectors", DataType::FixedSizeList(Arc::clone(&item), 2)),
    ] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("vector", vector, true),
            Field::new("label", DataType::Int32, true),
        ]));
        ctx.register_table(
            name,
            Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![RecordBatch::new_empty(schema)]]).unwrap()),
        )
        .unwrap();
    }
    let schema = Arc::new(Schema::new(
        iter::once(Field::new("id", DataType::Int64, false))
            .chain((0..4).map(|index| Field::new(format!("d{index}"), DataType::Float32, false)))
            .collect::<Vec<_>>(),
    ));
    ctx.register_table(
        "forest_vectors",
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
            CUML_RANDOM_FOREST_REGRESSOR_FUNCTION,
            CUML_FOREST_PREDICT_FUNCTION
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
            } else if operation == CumlOperation::ForestPredict {
                serde_json::json!(["input", "model", "model_format"])
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
    let model = Arc::new(
        MlForestModelSnapshot::new(
            b"source",
            b"checkpoint".to_vec(),
            MlForestModelFormat::XgboostJson,
            MlForestModelFacts {
                num_features: 2,
                num_trees: 1,
                native_output_stride: 1,
                precision: MlForestPrecision::Single,
            },
            MlForestOutputContract::Score,
        )
        .unwrap(),
    );
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
        } else if operation == CumlOperation::ForestPredict {
            serde_json::json!({"model": "scorer.json", "model_format": "xgboost_json"})
        } else {
            serde_json::json!({"min_samples": 2, "min_cluster_size": 2})
        };
        let bound = |plan: &LogicalPlan| {
            let mut options = values.clone();
            options["input"] = serde_json::json!({"id":"id","vector":{"columns":["d0","d1"]}});
            let model = (operation == CumlOperation::ForestPredict).then(|| Arc::clone(&model));
            CumlLogicalNode::try_new_with_model(operation, vec![plan.clone()], &options.to_string(), model)
        };
        if operation == CumlOperation::ForestPredict {
            bound(&row).expect("scoring accepts a single row");
        } else {
            let error = bound(&row).unwrap_err();
            assert_eq!(error.find_algeon_error().unwrap().fact_value("reason"), Some("insufficient_input_rows"));
        }
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
    let page_directory = repository.join("website/docs/cuml/udtfs");
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
                CumlOutputContract::ForestPrediction => "forest_prediction",
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
            page.contains("GENERATED FROM the unified GPU UDTF catalog by website/scripts/gen-cuml-docs.mjs"),
            "{} must retain its generated-source marker",
            page_path.display()
        );
        assert!(page.contains(&format!("# {}", descriptor.display_name)), "{}", page_path.display());
        assert!(page.contains(&format!("UDTF: `{}`", descriptor.sql_function_name)), "{}", page_path.display());
        let rendered_sql = page
            .split_once("```sql\n")
            .and_then(|(_, rest)| rest.split_once("\n```").map(|(sql, _)| sql))
            .unwrap_or_else(|| panic!("{} is missing Quickstart SQL", page_path.display()));
        let compact_sql = |sql: &str| sql.chars().filter(|ch| !ch.is_whitespace()).collect::<String>();
        assert_eq!(compact_sql(rendered_sql), compact_sql(descriptor.quickstart_sql), "{}", page_path.display());
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
        let element_types = page_section(&page, "### Vector element types");
        for element in descriptor.vector_element_support {
            assert!(element_types.contains(&format!("`{}`", element.as_str())), "{}", page_path.display());
        }
        // Float64 features are narrowed to Float32; a forest model keeps
        // Float32 input, because its own precision decides the device width.
        let elements: &[VectorElementType] = if descriptor.operation == CumlOperation::ForestPredict {
            &[VectorElementType::Float32]
        } else {
            &[VectorElementType::Float32, VectorElementType::Float64]
        };
        assert_eq!(descriptor.vector_element_support, elements, "{}", descriptor.sql_function_name);
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

/// The body of one generated section, from its heading to the next heading.
fn page_section<'a>(page: &'a str, heading: &str) -> &'a str {
    let (_, rest) = page.split_once(heading).unwrap_or_else(|| panic!("generated page is missing {heading}"));
    rest.split_once("\n#").map_or(rest, |(section, _)| section)
}

fn json_literal_for_docs(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Number(number) if number.is_f64() => number.as_f64().unwrap().to_string(),
        _ => serde_json::to_string(value).unwrap(),
    }
}

/// The relation field contract and the limitations carry the null-value
/// contract; capturing them from a server must reproduce the live descriptor.
#[tokio::test]
async fn captured_relation_roles_and_limitations_equal_the_catalog_descriptors() {
    let ctx = context();
    let spec_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../website/gpu-function-specs/cuml");
    for descriptor in CUML_OPERATION_REGISTRY {
        let name = descriptor.sql_function_name;
        let captured: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(spec_directory.join(format!("{name}.json"))).unwrap()).unwrap();
        let described = ctx
            .sql(&format!("SELECT relation_roles_json, limitations_json FROM gpu_describe_function('{name}')"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        for (column, key) in ["relation_roles_json", "limitations_json"].into_iter().enumerate() {
            let live = described[0].column(column).as_any().downcast_ref::<StringArray>().unwrap().value(0);
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(captured[key].as_str().unwrap()).unwrap(),
                serde_json::from_str::<serde_json::Value>(live).unwrap(),
                "{name}.{key} drifted from its captured spec; run website/scripts/capture-gpu-function-specs.sh"
            );
        }
    }
}

#[tokio::test]
async fn sql_schemas_match_ir_and_physical_nodes() {
    let models = ForestModels::new();
    let ctx = models.context();
    // Without the cuml feature, forest binding fails before any model is read.
    for operation in
        CumlOperation::all().filter(|operation| cfg!(feature = "cuml") || *operation != CumlOperation::ForestPredict)
    {
        let statement = if operation == CumlOperation::ForestPredict {
            forest_sql("xgb_binary_logistic.json", "xgboost_json", "")
        } else {
            sql(operation.descriptor().sql_function_name, "vectors", 2)
        };
        let logical = ctx.state().create_logical_plan(&statement).await.unwrap();
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
                        ml_classification_output_schema(DataType::Int64),
                    CumlOperation::RandomForestRegressor => ml_regression_output_schema(DataType::Int64),
                    CumlOperation::ForestPredict =>
                        ml_forest_output_schema(DataType::Int64, MlForestOutputContract::Score),
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

#[tokio::test]
async fn ordered_input_relations_keep_their_sort_through_physical_optimization() {
    let ctx = context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ]));
    let partitions = (0..3).map(|_| vec![RecordBatch::new_empty(Arc::clone(&schema))]).collect();
    ctx.register_table("partitioned_vectors", Arc::new(MemTable::try_new(schema, partitions).unwrap())).unwrap();
    let statement = format!(
        "SELECT * FROM {CUML_TSVD_FUNCTION}(\
             input => (SELECT id, d0, d1 FROM partitioned_vectors ORDER BY id DESC), n_components => 1)"
    );
    let logical = ctx.state().create_logical_plan(&statement).await.unwrap();
    let physical = ctx.state().create_physical_plan(&logical).await.unwrap();
    let exec = find_exec(physical.as_ref()).expect("CumlAlgoExec");
    let [Some(_)] = &exec.required_input_ordering()[..] else {
        panic!("the input ORDER BY must become a required input ordering")
    };
    let input = exec.children()[0];
    assert_eq!(input.properties().output_partitioning().partition_count(), 1, "{}", input.name());
    let ordering = input.properties().output_ordering().expect("the planned input must keep its ordering");
    assert_eq!(ordering.to_string(), "id@0 DESC");
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
    let mut builder = SessionStateBuilder::new_with_default_features()
        .try_with_cuml_sql()
        .unwrap()
        .try_with_cuml_sql_config(CumlSqlConfig::default())
        .expect("try_with_cuml_sql equals the default config");
    assert!(builder.query_planner().is_some());
    let error = SessionStateBuilder::new_with_default_features()
        .with_query_planner(Arc::new(OpaquePlanner))
        .try_with_cuml_sql()
        .unwrap_err();
    assert_eq!(error.find_algeon_error().unwrap().fact_value("reason"), Some("query_planner_conflict"));
    let missing_root = tempfile::tempdir().unwrap().path().join("missing");
    let error = SessionStateBuilder::new_with_default_features()
        .try_with_cuml_sql_config(CumlSqlConfig { model_store: Some(CumlModelStoreConfig::new(missing_root)) })
        .unwrap_err();
    assert_eq!(error.find_algeon_error().unwrap().fact_value("reason"), Some("invalid_model_store"));

    let roots = [tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap()];
    let store = |root: &tempfile::TempDir| CumlSqlConfig {
        model_store: Some(CumlModelStoreConfig::new(root.path().to_path_buf())),
    };
    let installed =
        SessionStateBuilder::new_with_default_features().try_with_cuml_sql_config(store(&roots[0])).unwrap();
    let installed = installed.try_with_cuml_sql_config(store(&roots[0])).expect("equal model store is a no-op");
    let error = installed.try_with_cuml_sql_config(store(&roots[1])).unwrap_err();
    assert!(error.to_string().contains("conflicting GPU function family installation for 'cuml'"), "{error}");
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
        &DFSchema::try_from(ml_classification_output_schema(DataType::Utf8View).as_ref().clone()).unwrap()
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

/// A statement that fails validation is a rejected `EXPLAIN GPU` summary whose
/// `reason_code` and `detail_json.detail` are the code and the client-visible
/// message that executing the statement returns, including the caller schema
/// detail the Flight status appends.
#[tokio::test]
async fn explain_gpu_rejection_carries_the_code_and_message_execution_returns() {
    let ctx = native_context();
    for (statement, schema_detail) in [
        (
            "SELECT * FROM cuml_random_forest_regressor(training => (SELECT id, d0 AS x FROM vectors ORDER BY id), predict => (SELECT id, d0 AS x FROM vectors ORDER BY id))",
            "",
        ),
        (
            "SELECT * FROM cuml_logistic_regression(training => (SELECT id, d0, d1, label FROM training_int64_labels), predict => (SELECT id, d0, d1 FROM vectors))",
            " (field 'label' has Arrow type `Int64`)",
        ),
    ] {
        let error = ctx.state().create_logical_plan(statement).await.unwrap_err();
        let typed = error.find_algeon_error().expect(statement);
        let message = format!("{}{schema_detail}", typed.message());

        let coverage = validate_query(&ctx, statement).await.unwrap();
        let rejection = coverage.rejection().unwrap_or_else(|| panic!("{statement}: {coverage:?}"));
        assert_eq!(rejection.reason_code, typed.code().as_str(), "{statement}");
        assert_eq!(rejection.detail, message, "{statement}");

        let rows = ctx
            .sql(&format!(
                "SELECT row_kind, gpu_path, outcome, reason_code, detail_json \
                 FROM algeon_explain_coverage('{statement}')"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let [batch] = rows.as_slice() else { panic!("{statement}: {rows:?}") };
        let column = |index: usize| batch.column(index).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(batch.num_rows(), 1, "a rejection has no candidate rows: {batch:?}");
        assert_eq!(column(0).value(0), "summary");
        assert_eq!(column(1).value(0), "rejected");
        assert_eq!(column(2).value(0), "rejected_before_execution");
        assert_eq!(column(3).value(0), typed.code().as_str());
        let detail_json: serde_json::Value = serde_json::from_str(column(4).value(0)).unwrap();
        assert_eq!(detail_json["reason_code"], typed.code().as_str());
        assert_eq!(detail_json["detail"], message);
    }
}

/// Schema nullability is not part of the input contract: null values fail at
/// execution, so nullable-typed IDs, features, list vectors and labels plan,
/// and the result keeps its non-null schema with the input ID type.
#[tokio::test]
async fn nullable_typed_relations_plan_with_the_canonical_non_null_output_schema() {
    let ctx = context();
    for (table, projection) in [
        ("nullable_vectors", "id, d0, d1"),
        ("nullable_list_vectors", "id, vector"),
        ("nullable_fixed_vectors", "id, vector"),
    ] {
        let plan = ctx
            .state()
            .create_logical_plan(&format!(
                "SELECT * FROM cuml_tsvd(input => (SELECT {projection} FROM {table}), n_components => 1)"
            ))
            .await
            .unwrap_or_else(|error| panic!("{table}: {error}"));
        assert_eq!(
            find_logical(&plan).unwrap().schema().as_ref(),
            &DFSchema::try_from(vector_pca_output_schema(DataType::Int64, 1).as_ref().clone()).unwrap(),
            "{table}"
        );
    }
    for (training, predict) in [
        ("SELECT id, d0, d1, label FROM training_nullable_labels", "SELECT id, d0, d1 FROM nullable_vectors"),
        ("SELECT id, vector, label FROM nullable_list_vectors", "SELECT id, vector FROM nullable_fixed_vectors"),
    ] {
        let plan = ctx
            .state()
            .create_logical_plan(&format!(
                "SELECT * FROM cuml_logistic_regression(training => ({training}), predict => ({predict}))"
            ))
            .await
            .unwrap_or_else(|error| panic!("{training}: {error}"));
        assert_eq!(
            find_logical(&plan).unwrap().schema().as_ref(),
            &DFSchema::try_from(ml_classification_output_schema(DataType::Int64).as_ref().clone()).unwrap(),
            "{training}"
        );
    }
}

#[tokio::test]
async fn schema_rejections_preserve_structured_reasons_and_narrowed_float64_support() {
    let ctx = context();
    let item = Arc::new(Field::new("item", DataType::Float64, false));
    for (name, vector) in [
        ("double_list_vectors", DataType::List(Arc::clone(&item))),
        ("double_fixed_vectors", DataType::FixedSizeList(item, 2)),
    ] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("embedding", vector, false),
        ]));
        ctx.register_table(
            name,
            Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![RecordBatch::new_empty(schema)]]).unwrap()),
        )
        .unwrap();
    }
    // Float64 wide columns and list elements pass planning and native
    // preflight; execution narrows them to Float32.
    let list = |table: &str| {
        format!("SELECT * FROM {CUML_TSVD_FUNCTION}(input => (SELECT id, embedding FROM {table}), n_components => 1)")
    };
    for statement in
        [sql(CUML_TSVD_FUNCTION, "double_vectors", 1), list("double_list_vectors"), list("double_fixed_vectors")]
    {
        let logical = ctx.state().create_logical_plan(&statement).await.unwrap();
        assert_eq!(logical.schema().field_with_unqualified_name("pc_0").unwrap().data_type(), &DataType::Float32);
        let physical = ctx.state().create_physical_plan(&logical).await.unwrap();
        assert!(find_exec(physical.as_ref()).is_some(), "{statement}");
    }
    let error = ctx.state().create_logical_plan(&sql(CUML_TSVD_FUNCTION, "integer_vectors", 1)).await.unwrap_err();
    let typed = error.find_algeon_error().unwrap();
    assert_eq!(typed.code(), ErrorCode::UnsupportedType);
    assert_eq!(typed.fact_value("expected_type"), Some("Float32, Float64"));
    let error = ctx.state().create_logical_plan(&sql(CUML_TSVD_FUNCTION, "vectors", 3)).await.unwrap_err();
    assert_eq!(error.find_algeon_error().unwrap().fact_value("reason"), Some("component_count_exceeds_dimension"));
    // The list-shape rule names the same element types as the wide shape.
    for descriptor in CUML_OPERATION_REGISTRY {
        let name = descriptor.sql_function_name;
        let described = ctx
            .sql(&format!("SELECT relation_roles_json FROM gpu_describe_function('{name}')"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let roles: serde_json::Value =
            serde_json::from_str(described[0].column(0).as_any().downcast_ref::<StringArray>().unwrap().value(0))
                .unwrap();
        let features = &roles[0]["fields"]["features"];
        let list_rule = features["one_of"]
            .as_array()
            .unwrap()
            .iter()
            .find(|shape| shape["shape"] == "list")
            .and_then(|shape| shape["rule"].as_str())
            .unwrap();
        let elements = descriptor.vector_element_support.iter().map(|element| element.as_str()).collect::<Vec<_>>();
        assert_eq!(features["element_types"], serde_json::json!(elements), "{name}");
        assert!(list_rule.contains(&format!("of {} elements", elements.join(" or "))), "{name}: {list_rule}");
    }
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
    let names = listed[0].column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let available = listed[0].column(1).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(
        (0..CUML_OPERATION_REGISTRY.len())
            .all(|index| available.value(index) == (names.value(index) != CUML_FOREST_PREDICT_FUNCTION))
    );
    for (relation, components, code, path) in [
        ("vectors", 0, "missing_or_invalid_option", "options.n_components"),
        ("integer_vectors", 1, "unsupported_type", "relations.input.fields.d0"),
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

/// Forest fixtures copied into a temporary models root, next to files the
/// root must not expose.
struct ForestModels {
    directory: tempfile::TempDir,
}

impl ForestModels {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("models");
        fs::create_dir(&root).unwrap();
        for name in [
            "xgb_binary_logistic.json",
            "xgb_multiclass_softmax.json",
            "xgb_multiclass_softprob.json",
            "xgb_multi_target.json",
            "corrupt.json",
        ] {
            fs::copy(forest_fixture(name), root.join(name)).unwrap();
        }
        Self { directory }
    }

    fn root(&self) -> PathBuf {
        self.directory.path().join("models")
    }

    fn config(&self, max_model_bytes: u64) -> CumlSqlConfig {
        CumlSqlConfig { model_store: Some(CumlModelStoreConfig { root: self.root(), max_model_bytes }) }
    }

    fn context(&self) -> SessionContext {
        context_with(self.config(CumlModelStoreConfig::DEFAULT_MAX_MODEL_BYTES))
    }
}

fn forest_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../components/cuml/algeon-cuml/tests/fixtures/forest").join(name)
}

fn forest_sql(model: &str, format: &str, extra: &str) -> String {
    format!(
        "SELECT * FROM cuml_forest_predict(input => (SELECT id, d0, d1, d2, d3 FROM forest_vectors), model => '{model}', model_format => '{format}'{extra})"
    )
}

async fn forest_rejection(ctx: &SessionContext, statement: &str) -> (Option<String>, Option<String>) {
    let error = ctx.state().create_logical_plan(statement).await.unwrap_err();
    let typed = error.find_algeon_error().unwrap_or_else(|| panic!("{statement}: {error}"));
    (typed.fact_value("reason").map(str::to_owned), typed.fact_value("rejection").map(str::to_owned))
}

#[cfg(feature = "cuml")]
#[tokio::test]
async fn forest_predict_binds_the_model_file_digest_and_output_contract() {
    let models = ForestModels::new();
    let ctx = models.context();
    for (model, extra, output) in [
        ("xgb_binary_logistic.json", "", MlForestOutputContract::Score),
        ("xgb_multiclass_softprob.json", "", MlForestOutputContract::Vector(3)),
        ("xgb_multiclass_softmax.json", ", output => 'class_index'", MlForestOutputContract::ClassIndex { classes: 3 }),
    ] {
        let logical = ctx.state().create_logical_plan(&forest_sql(model, "xgboost_json", extra)).await.unwrap();
        let node = find_logical(&logical).unwrap();
        let bound = node.model().expect("forest node carries its model");
        let digest = MlForestModelDigest::of(&fs::read(forest_fixture(model)).unwrap());
        assert_eq!(bound.digest(), digest, "{model}");
        assert_eq!(bound.num_features(), 4);
        assert_eq!(bound.output(), output, "{model}");
        assert_eq!(
            node.schema().as_ref(),
            &DFSchema::try_from(ml_forest_output_schema(DataType::Int64, output).as_ref().clone()).unwrap()
        );
        assert!(
            logical.display_indent().to_string().contains(&format!("model=sha256:{}", digest.short_hex())),
            "{}",
            logical.display_indent()
        );
    }
}

#[cfg(feature = "cuml")]
#[tokio::test]
async fn forest_predict_rewrites_reuse_the_planned_model_without_rereading_the_file() {
    let models = ForestModels::new();
    let ctx = models.context();
    let logical =
        ctx.state().create_logical_plan(&forest_sql("xgb_binary_logistic.json", "xgboost_json", "")).await.unwrap();
    let node = find_logical(&logical).unwrap();
    fs::write(models.root().join("xgb_binary_logistic.json"), b"not a model").unwrap();
    let rewritten = node.with_exprs_and_inputs(Vec::new(), node.inputs().into_iter().cloned().collect()).unwrap();
    assert_eq!(&rewritten, node);
    assert_eq!(rewritten.model().unwrap().digest(), node.model().unwrap().digest());
}

#[cfg(feature = "cuml")]
#[tokio::test]
async fn forest_predict_resolves_models_beneath_the_root_opened_at_install() {
    let models = ForestModels::new();
    let ctx = models.context();
    let installed = models.directory.path().join("installed");
    fs::rename(models.root(), &installed).unwrap();
    fs::create_dir(models.root()).unwrap();
    fs::copy(forest_fixture("xgb_multiclass_softprob.json"), models.root().join("xgb_binary_logistic.json")).unwrap();
    let logical =
        ctx.state().create_logical_plan(&forest_sql("xgb_binary_logistic.json", "xgboost_json", "")).await.unwrap();
    let bound = find_logical(&logical).unwrap().model().unwrap();
    assert_eq!(bound.digest(), MlForestModelDigest::of(&fs::read(installed.join("xgb_binary_logistic.json")).unwrap()));
}

#[cfg(feature = "cuml")]
#[tokio::test]
async fn forest_predict_rejects_model_paths_that_are_not_regular_files_inside_the_root() {
    let models = ForestModels::new();
    let root = models.root();
    fs::copy(forest_fixture("xgb_binary_logistic.json"), models.directory.path().join("outside.json")).unwrap();
    symlink("../outside.json", root.join("escape.json")).unwrap();
    symlink("../absent.json", root.join("dangling.json")).unwrap();
    symlink(root.join("xgb_binary_logistic.json"), root.join("absolute_alias.json")).unwrap();
    symlink("loop.json", root.join("loop.json")).unwrap();
    symlink("xgb_binary_logistic.json", root.join("alias.json")).unwrap();
    fs::create_dir(root.join("nested")).unwrap();
    symlink("../xgb_binary_logistic.json", root.join("nested/up.json")).unwrap();
    assert!(Command::new("mkfifo").arg(root.join("pipe.json")).status().unwrap().success());
    let ctx = models.context();
    let absolute = models.directory.path().join("outside.json").display().to_string();
    let long_name = "m".repeat(256);
    for (model, rejection) in [
        ("../outside.json", "parent_component"),
        ("nested/../xgb_binary_logistic.json", "parent_component"),
        (absolute.as_str(), "absolute_path"),
        ("escape.json", "outside_root"),
        // Escaping symlinks are rejected before the target is looked up, so
        // a missing outside target reports the same rejection.
        ("dangling.json", "outside_root"),
        ("absolute_alias.json", "outside_root"),
        ("loop.json", "symlink_loop"),
        (long_name.as_str(), "name_too_long"),
        ("nested", "not_regular_file"),
        ("missing.json", "not_found"),
        ("xgb_binary_logistic.json/child", "not_found"),
    ] {
        assert_eq!(
            forest_rejection(&ctx, &forest_sql(model, "xgboost_json", "")).await,
            (Some("model_path_rejected".to_string()), Some(rejection.to_string())),
            "{model}"
        );
    }
    // Opening a FIFO without O_NONBLOCK waits for a writer, so a regression
    // blocks the planner thread instead of failing this assertion.
    let (sender, receiver) = mpsc::channel();
    let pipe_ctx = models.context();
    thread::spawn(move || {
        let runtime = Builder::new_current_thread().build().unwrap();
        sender
            .send(runtime.block_on(forest_rejection(&pipe_ctx, &forest_sql("pipe.json", "xgboost_json", ""))))
            .unwrap();
    });
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(30)).expect("FIFO model path must not block planning"),
        (Some("model_path_rejected".to_string()), Some("not_regular_file".to_string()))
    );
    ctx.state().create_logical_plan(&forest_sql("alias.json", "xgboost_json", "")).await.expect("in-root symlink");
    ctx.state()
        .create_logical_plan(&forest_sql("nested/up.json", "xgboost_json", ""))
        .await
        .expect("in-root symlink through '..'");

    let size = fs::metadata(root.join("xgb_binary_logistic.json")).unwrap().len();
    let bounded = context_with(models.config(size - 1));
    let error = bounded
        .state()
        .create_logical_plan(&forest_sql("xgb_binary_logistic.json", "xgboost_json", ""))
        .await
        .unwrap_err();
    let typed = error.find_algeon_error().unwrap();
    assert_eq!(typed.fact_value("reason"), Some("model_too_large"));
    assert_eq!(typed.fact_value("max_model_bytes"), Some((size - 1).to_string().as_str()));
    context_with(models.config(size))
        .state()
        .create_logical_plan(&forest_sql("xgb_binary_logistic.json", "xgboost_json", ""))
        .await
        .expect("a model at max_model_bytes is accepted");
}

#[cfg(feature = "cuml")]
#[tokio::test]
async fn forest_predict_reports_structured_option_model_and_schema_rejections() {
    let models = ForestModels::new();
    let ctx = models.context();
    let long = "m".repeat(1025);
    for (statement, reason) in [
        (
            "SELECT * FROM cuml_forest_predict(input => (SELECT id, d0, d1, d2, d3 FROM forest_vectors), model => 'xgb_binary_logistic.json')".to_string(),
            "missing_argument",
        ),
        (forest_sql(&long, "xgboost_json", ""), "missing_or_invalid_option"),
        (forest_sql("xgb_binary_logistic.json", "onnx", ""), "missing_or_invalid_option"),
        (forest_sql("corrupt.json", "xgboost_json", ""), "model_rejected"),
        (forest_sql("xgb_multi_target.json", "xgboost_json", ""), "model_rejected"),
        (forest_sql("xgb_binary_logistic.json", "lightgbm", ""), "model_rejected"),
        (
            forest_sql("xgb_binary_logistic.json", "xgboost_json", ", output => 'class_index'"),
            "class_index_requires_multiple_outputs",
        ),
    ] {
        assert_eq!(forest_rejection(&ctx, &statement).await.0.as_deref(), Some(reason), "{statement:.200}");
    }
    let error = ctx
        .state()
        .create_logical_plan(
            "SELECT * FROM cuml_forest_predict(input => (SELECT id, d0, d1 FROM vectors), model => 'xgb_binary_logistic.json', model_format => 'xgboost_json')",
        )
        .await
        .unwrap_err();
    let typed = error.find_algeon_error().unwrap();
    assert_eq!(typed.fact_value("reason"), Some("forest_feature_count_mismatch"));
    assert_eq!((typed.fact_value("expected"), typed.fact_value("actual")), (Some("4"), Some("2")));

    assert_eq!(
        forest_rejection(&context(), &forest_sql("xgb_binary_logistic.json", "xgboost_json", "")).await.0.as_deref(),
        Some("model_root_unconfigured")
    );
}

#[cfg(feature = "cuml")]
#[tokio::test]
async fn forest_predict_catalog_validation_binds_models_and_reports_an_unconfigured_root() {
    let models = ForestModels::new();
    let validate = |ctx: SessionContext, options: serde_json::Value| async move {
        let call = serde_json::json!({"relations": {"input": {"table": "forest_vectors"}}, "options": options});
        let batches = ctx
            .sql(&format!(
                "SELECT error_code, output_schema_json, details_json FROM gpu_validate_call('cuml_forest_predict', '{call}')"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let column = |index: usize| {
            let column = batches[0].column(index).as_any().downcast_ref::<StringArray>().unwrap();
            (!column.is_null(0)).then(|| column.value(0).to_string())
        };
        let schema: serde_json::Value = serde_json::from_str(&column(1).unwrap()).unwrap();
        let fields = schema["fields"]
            .as_array()
            .map(|fields| {
                fields
                    .iter()
                    .map(|field| {
                        format!("{}:{}", field["name"].as_str().unwrap(), field["data_type"].as_str().unwrap())
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        (column(0), fields)
    };
    for (model, output, fields) in [
        ("xgb_binary_logistic.json", "scores", &["row_ordinal:UInt64", "id:Int64", "prediction:Float32"][..]),
        ("xgb_multiclass_softmax.json", "class_index", &["row_ordinal:UInt64", "id:Int64", "prediction:Int32"][..]),
        (
            "xgb_multiclass_softprob.json",
            "scores",
            &["row_ordinal:UInt64", "id:Int64", "output_0:Float32", "output_1:Float32", "output_2:Float32"][..],
        ),
    ] {
        let options = serde_json::json!({"model": model, "model_format": "xgboost_json", "output": output});
        assert_eq!(
            validate(models.context(), options).await,
            (None, fields.iter().map(|field| field.to_string()).collect())
        );
    }
    let options = serde_json::json!({"model": "escape/../x.json", "model_format": "xgboost_json"});
    assert_eq!(validate(models.context(), options).await.0.as_deref(), Some("model_path_rejected"));
    let options = serde_json::json!({"model": "xgb_binary_logistic.json"});
    assert_eq!(validate(models.context(), options).await.0.as_deref(), Some("missing_or_invalid_option"));

    let unconfigured = context();
    let listed = unconfigured
        .sql("SELECT available, unavailable_reason FROM gpu_list_functions() WHERE function_name = 'cuml_forest_predict'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert!(!listed[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap().value(0));
    assert_eq!(listed[0].column(1).as_any().downcast_ref::<StringArray>().unwrap().value(0), "model_root_unconfigured");
    let options = serde_json::json!({"model": "xgb_binary_logistic.json", "model_format": "xgboost_json"});
    assert_eq!(validate(unconfigured, options).await.0.as_deref(), Some("model_root_unconfigured"));
}

#[cfg(not(feature = "cuml"))]
#[tokio::test]
async fn forest_predict_rejects_a_disabled_cuml_feature_before_reading_the_model() {
    let models = ForestModels::new();
    assert_eq!(
        forest_rejection(&models.context(), &forest_sql("missing.json", "xgboost_json", "")).await.0.as_deref(),
        Some("required_feature_disabled")
    );
}

/// A function the session cannot run reports why before its arguments are
/// bound, so the missing `model` and `model_format` are not the first error.
#[tokio::test]
async fn forest_predict_reports_its_availability_before_binding_arguments() {
    let error = context()
        .state()
        .create_logical_plan(&format!(
            "SELECT * FROM {CUML_FOREST_PREDICT_FUNCTION}(input => (SELECT id, d0, d1, d2, d3 FROM forest_vectors))"
        ))
        .await
        .unwrap_err();
    let reason = if cfg!(feature = "cuml") { "model_root_unconfigured" } else { "required_feature_disabled" };
    assert_eq!(error.find_algeon_error().unwrap().fact_value("reason"), Some(reason));
}
