use std::{
    collections::BTreeSet,
    fs,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use datafusion::{
    common::{DFSchema, config::Dialect, tree_node::TreeNodeRecursion},
    execution::{SessionStateBuilder, context::TaskContext},
    logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore},
    physical_expr::PhysicalExpr,
    physical_plan::{
        ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, ReplaceChildrenOptions,
        SendableRecordBatchStream,
    },
    prelude::SessionContext,
    sql::{
        parser::Statement as DataFusionStatement,
        sqlparser::ast::{
            Expr as SqlExpr, FunctionArg, FunctionArgExpr, SetExpr, Statement as SqlStatement, TableFactor,
        },
    },
};
use datafusion_nexus::{
    cuvs_sql::{
        CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_KMEANS_FUNCTION, CUVS_OPERATION_REGISTRY, CUVS_PCA_FUNCTION, CuvsAlgoExec,
        CuvsLogicalNode, CuvsOperation, CuvsRelationArgumentSyntax, CuvsRelationRole, cuvs_options_json_schema,
    },
    error::{DataFusionErrorExt, ErrorCode},
    gpu_functions::{GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_LIST_FUNCTIONS_FUNCTION, GPU_VALIDATE_CALL_FUNCTION},
    native_exec::NexusNativeExecutionConfig,
    session::NexusSessionStateBuilderExt,
};
use nexus_query_engine::plan::{
    CUVS_ID_TYPES, CuvsOperation as NativeCuvsOperation, canonicalize_cuvs_options,
    vector_brute_force_knn_output_schema, vector_kmeans_output_schema, vector_pca_output_schema,
};

#[path = "cuvs_sql_fixtures.rs"]
mod fixtures;

use fixtures::{KMEANS_OPTIONS, KNN_OPTIONS, PCA_OPTIONS, cuvs_context};

#[test]
fn canonical_registry_drives_names_relations_and_json_schemas() {
    let names = CUVS_OPERATION_REGISTRY.iter().map(|descriptor| descriptor.sql_function_name).collect::<Vec<_>>();
    assert_eq!(names, [CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_KMEANS_FUNCTION, CUVS_PCA_FUNCTION]);
    assert_eq!(
        CuvsOperation::BruteForceKnn.descriptor().relation_roles,
        [CuvsRelationRole::Dataset, CuvsRelationRole::Queries]
    );
    assert_eq!(CuvsOperation::KMeans.descriptor().relation_roles, [CuvsRelationRole::Input]);
    assert!(CUVS_OPERATION_REGISTRY.iter().all(|descriptor| {
        descriptor.relation_argument_syntax == CuvsRelationArgumentSyntax::ParenthesizedSubquery
    }));

    let knn = cuvs_options_json_schema(CuvsOperation::BruteForceKnn);
    assert_eq!(knn["additionalProperties"], false);
    assert_eq!(knn["required"], serde_json::json!(["dataset", "queries", "k", "metric"]));
    assert_eq!(knn["properties"]["k"]["minimum"], 1);
    assert_eq!(
        knn["properties"]["metric"]["enum"],
        serde_json::json!(["l2_expanded", "l2_sqrt_expanded", "cosine", "inner_product"])
    );
    assert_eq!(
        CUVS_ID_TYPES,
        &[
            nexus_query_engine::plan::CuvsIdType::Int32,
            nexus_query_engine::plan::CuvsIdType::Int64,
            nexus_query_engine::plan::CuvsIdType::Utf8,
            nexus_query_engine::plan::CuvsIdType::LargeUtf8,
            nexus_query_engine::plan::CuvsIdType::Utf8View,
        ]
    );

    let pca = cuvs_options_json_schema(CuvsOperation::Pca);
    assert_eq!(pca["properties"]["solver"]["default"], "cov_eig_dq");
    assert_eq!(pca["properties"]["n_iterations"]["default"], 15);
    assert_eq!(pca["properties"]["flip_signs_based_on_u"]["default"], false);
    for operation in CuvsOperation::all() {
        assert_eq!(
            cuvs_options_json_schema(operation),
            operation.descriptor().options_json_schema(),
            "root SQL metadata must be the canonical engine metadata"
        );
    }
}

#[test]
fn captured_specs_and_generated_reference_pages_match_the_canonical_registry() {
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let spec_directory = repository.join("website/gpu-function-specs/cuvs");
    let page_directory = repository.join("website/docs/cuvs/operations");
    let expected_names =
        CUVS_OPERATION_REGISTRY.iter().map(|descriptor| descriptor.sql_function_name).collect::<BTreeSet<_>>();
    let captured_names = fs::read_dir(&spec_directory)
        .unwrap_or_else(|error| panic!("read captured cuVS specs at {}: {error}", spec_directory.display()))
        .map(|entry| entry.expect("read captured cuVS spec entry"))
        .filter_map(|entry| {
            let path = entry.path();
            (path.extension().and_then(|extension| extension.to_str()) == Some("json"))
                .then(|| path.file_stem().and_then(|stem| stem.to_str()).expect("UTF-8 spec file name").to_string())
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(captured_names.iter().map(String::as_str).collect::<BTreeSet<_>>(), expected_names);

    for descriptor in CUVS_OPERATION_REGISTRY {
        let spec_path = spec_directory.join(format!("{}.json", descriptor.sql_function_name));
        let captured: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(&spec_path)
                .unwrap_or_else(|error| panic!("read captured cuVS spec at {}: {error}", spec_path.display())),
        )
        .unwrap_or_else(|error| panic!("parse captured cuVS spec at {}: {error}", spec_path.display()));
        assert_eq!(captured["function_name"], descriptor.sql_function_name);
        assert_eq!(captured["provider"], "cuvs");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(captured["options_schema_json"].as_str().unwrap()).unwrap(),
            descriptor.options_json_schema(),
            "captured options schema must be an exact projection of the canonical registry"
        );
        let options_schema =
            serde_json::from_str::<serde_json::Value>(captured["options_schema_json"].as_str().unwrap()).unwrap();
        for option in descriptor.options {
            assert_eq!(
                options_schema["properties"][option.name]["description"], option.description,
                "{} option {} must retain the canonical description",
                descriptor.sql_function_name, option.name
            );
        }
        let roles =
            serde_json::from_str::<serde_json::Value>(captured["relation_roles_json"].as_str().unwrap()).unwrap();
        assert_eq!(
            roles.as_array().unwrap().iter().map(|role| role["name"].as_str().unwrap()).collect::<Vec<_>>(),
            descriptor.relation_roles.iter().map(|role| role.as_str()).collect::<Vec<_>>(),
            "captured relation roles must be an exact projection of the canonical registry"
        );
        for role in descriptor.relation_roles {
            let captured_role = roles
                .as_array()
                .unwrap()
                .iter()
                .find(|captured_role| captured_role["name"] == role.as_str())
                .unwrap_or_else(|| {
                    panic!("{} must project relation role {}", descriptor.sql_function_name, role.as_str())
                });
            assert_eq!(captured_role["description"], role.description());
        }
        let result_schema =
            serde_json::from_str::<serde_json::Value>(captured["result_schemas_json"].as_str().unwrap()).unwrap();
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
        assert_eq!(quickstart["display_name"], descriptor.display_name);
        assert_eq!(quickstart["use_this_when"], descriptor.use_this_when);
        assert_eq!(quickstart["sql"], descriptor.quickstart_sql);

        let page_path = page_directory.join(format!("{}.mdx", descriptor.sql_function_name));
        let page = fs::read_to_string(&page_path)
            .unwrap_or_else(|error| panic!("read generated cuVS page at {}: {error}", page_path.display()));
        assert!(
            page.contains("GENERATED FROM the unified GPU function catalog by website/scripts/gen-cuvs-docs.mjs"),
            "{} must retain its generated-source marker",
            page_path.display()
        );
        assert!(page.contains(&format!("# {}", descriptor.display_name)), "{}", page_path.display());
        assert!(page.contains(&format!("SQL function: `{}`", descriptor.sql_function_name)), "{}", page_path.display());
        assert!(page.contains(descriptor.quickstart_sql), "{}", page_path.display());
        let headings = [
            "## Signature",
            "## Quickstart",
            "## Relation inputs",
            "## Scalar arguments & JSON options",
            "## Output schema",
            "## Limitations & lifecycle",
            "## Validate before running",
        ];
        let mut previous_heading = 0;
        for heading in headings {
            let index = page.find(heading).unwrap_or_else(|| panic!("{} is missing {heading}", page_path.display()));
            assert!(index >= previous_heading, "{} puts {heading} out of order", page_path.display());
            previous_heading = index;
        }
        for relation_role in descriptor.relation_roles {
            assert!(page.contains(&format!("`{}`", relation_role.as_str())), "{}", page_path.display());
        }
        for option in descriptor.options {
            assert!(page.contains(&format!("`{}`", option.name)), "{}", page_path.display());
            assert!(page.contains(&mdx_table_text(option.description)), "{}", page_path.display());
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

#[tokio::test]
async fn sql_output_schemas_equal_canonical_native_output_schemas() {
    let ctx = cuvs_context();
    let knn = logical_plan(
        &ctx,
        &format!(
            "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}((SELECT * FROM dataset_vectors), (SELECT * FROM query_vectors), '{}')",
            sql_literal(KNN_OPTIONS)
        ),
    )
    .await;
    let knn = find_cuvs_node(&knn).unwrap();
    let expected_knn =
        vector_brute_force_knn_output_schema(arrow::datatypes::DataType::Utf8, arrow::datatypes::DataType::Int64);
    let expected_knn = DFSchema::try_from(expected_knn.as_ref().clone()).unwrap();
    assert_eq!(knn.schema().as_ref(), &expected_knn);

    let kmeans = logical_plan(
        &ctx,
        &format!(
            "SELECT * FROM {CUVS_KMEANS_FUNCTION}((SELECT * FROM dataset_vectors), '{}')",
            sql_literal(KMEANS_OPTIONS)
        ),
    )
    .await;
    let kmeans = find_cuvs_node(&kmeans).unwrap();
    let expected_kmeans = vector_kmeans_output_schema(arrow::datatypes::DataType::Int64);
    let expected_kmeans = DFSchema::try_from(expected_kmeans.as_ref().clone()).unwrap();
    assert_eq!(kmeans.schema().as_ref(), &expected_kmeans);

    let pca = logical_plan(
        &ctx,
        &format!("SELECT * FROM {CUVS_PCA_FUNCTION}((SELECT * FROM dataset_vectors), '{}')", sql_literal(PCA_OPTIONS)),
    )
    .await;
    let pca = find_cuvs_node(&pca).unwrap();
    let expected_pca = vector_pca_output_schema(arrow::datatypes::DataType::Int64, 2);
    let expected_pca = DFSchema::try_from(expected_pca.as_ref().clone()).unwrap();
    assert_eq!(pca.schema().as_ref(), &expected_pca);
}

#[test]
fn identical_cuvs_installation_is_idempotent_over_its_existing_query_planner() {
    let builder = SessionStateBuilder::new_with_default_features().try_with_cuvs_sql().expect("first cuVS SQL install");
    let mut builder = builder.try_with_cuvs_sql().expect("identical second cuVS SQL install must be a no-op");
    assert!(builder.query_planner().is_some());
}

#[tokio::test]
async fn root_relation_binding_validation_delegates_to_canonical_registry() {
    let raw = r#"{"input":{"id":"item_id","vector":{}},"n_clusters":1}"#;
    let canonical_error = canonicalize_cuvs_options(NativeCuvsOperation::KMeans, raw)
        .expect_err("canonical registry should reject malformed vector binding");
    let ctx = cuvs_context();
    let sql = format!("SELECT * FROM {CUVS_KMEANS_FUNCTION}((SELECT * FROM dataset_vectors), '{}')", sql_literal(raw));
    let error =
        ctx.state().create_logical_plan(&sql).await.expect_err("root should delegate malformed binding validation");
    let typed = error.find_nexus_error().expect("structured root validation error");
    assert_eq!(typed.facts().get("reason"), Some(canonical_error.code()));
    assert_eq!(typed.message(), canonical_error.message());
}

#[tokio::test]
async fn catalog_relations_become_two_real_children_and_alias_resolves() {
    let ctx = cuvs_context();
    let sql = format!(
        "SELECT knn.query_id, knn.neighbor_id, knn.distance \
         FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             (SELECT * FROM datafusion.public.dataset_vectors), \
             (SELECT * FROM query_vectors), \
             '{}') AS knn",
        sql_literal(KNN_OPTIONS)
    );
    let statement = ctx.state().sql_to_statement(&sql, &Dialect::Generic).unwrap();
    let DataFusionStatement::Statement(statement_ast) = &statement else { panic!("expected a standard SQL statement") };
    let SqlStatement::Query(query) = statement_ast.as_ref() else { panic!("expected a query") };
    let SetExpr::Select(select) = query.body.as_ref() else { panic!("expected a select") };
    let TableFactor::Table { args: Some(args), .. } = &select.from[0].relation else {
        panic!("expected cuVS table-function syntax")
    };
    assert!(
        args.args[..2]
            .iter()
            .all(|argument| matches!(argument, FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Subquery(_)))))
    );
    let preload_relations = ctx
        .state()
        .resolve_table_references(&statement)
        .unwrap()
        .into_iter()
        .map(|relation| relation.to_string())
        .collect::<Vec<_>>();
    assert!(preload_relations.iter().any(|relation| relation == "datafusion.public.dataset_vectors"));
    assert!(preload_relations.iter().any(|relation| relation == "query_vectors"));

    let plan = logical_plan(&ctx, &sql).await;
    let node = find_cuvs_node(&plan).expect("cuVS logical extension");
    assert_eq!(node.operation(), CuvsOperation::BruteForceKnn);
    assert_eq!(node.inputs().len(), 2);
    let dataset = node.inputs()[0].display_indent().to_string();
    let queries = node.inputs()[1].display_indent().to_string();
    assert!(dataset.contains("TableScan: datafusion.public.dataset_vectors"), "{dataset}");
    assert!(queries.contains("TableScan: query_vectors"), "{queries}");
    assert!(!dataset.contains("EmptyRelation"), "{dataset}");
    assert!(!queries.contains("EmptyRelation"), "{queries}");
    assert!(plan.display_indent().to_string().contains("SubqueryAlias: knn"));
    assert_schema(node.schema(), &["query_ordinal", "query_id", "neighbor_ordinal", "neighbor_id", "rank", "distance"]);
    assert_eq!(
        node.schema().field_with_unqualified_name("query_id").unwrap().data_type(),
        &arrow::datatypes::DataType::Utf8
    );
    assert_eq!(
        node.schema().field_with_unqualified_name("neighbor_id").unwrap().data_type(),
        &arrow::datatypes::DataType::Int64
    );
}

#[tokio::test]
async fn cte_relations_are_preserved_as_two_logical_children() {
    let ctx = cuvs_context();
    let sql = format!(
        "WITH candidates AS (\
             SELECT item_id, d0, d1 FROM datafusion.public.dataset_vectors WHERE d0 >= 0.0\
         ), queries AS (\
             SELECT query_id, d0, d1 FROM query_vectors\
         ) \
         SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             (SELECT * FROM candidates), \
             (SELECT * FROM queries), \
             '{}')",
        sql_literal(KNN_OPTIONS)
    );
    let statement = ctx.state().sql_to_statement(&sql, &Dialect::Generic).unwrap();
    let preload_relations = ctx
        .state()
        .resolve_table_references(&statement)
        .unwrap()
        .into_iter()
        .map(|relation| relation.to_string())
        .collect::<Vec<_>>();
    assert!(preload_relations.iter().any(|relation| relation == "datafusion.public.dataset_vectors"));
    assert!(preload_relations.iter().any(|relation| relation == "query_vectors"));

    let plan = logical_plan(&ctx, &sql).await;
    let node = find_cuvs_node(&plan).expect("cuVS logical extension");
    assert_eq!(node.inputs().len(), 2);
    let candidates = node.inputs()[0].display_indent().to_string();
    let queries = node.inputs()[1].display_indent().to_string();
    assert!(candidates.contains("TableScan: datafusion.public.dataset_vectors"), "{candidates}");
    assert!(candidates.contains("Filter:"), "{candidates}");
    assert!(queries.contains("TableScan: query_vectors"), "{queries}");
    assert!(!candidates.contains("EmptyRelation"), "{candidates}");
    assert!(!queries.contains("EmptyRelation"), "{queries}");
}

#[tokio::test]
async fn direct_join_filter_projection_subqueries_remain_real_children() {
    let ctx = cuvs_context();
    let sql = format!(
        "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             (SELECT e.item_id, e.d0, e.d1 FROM dataset_vectors AS e JOIN query_vectors AS q ON e.d0 = q.d0 WHERE e.d1 >= 0.0),\
             (SELECT query_id, d0, d1 FROM query_vectors WHERE d0 >= 0.0),\
             '{}')",
        sql_literal(KNN_OPTIONS)
    );
    let statement = ctx.state().sql_to_statement(&sql, &Dialect::Generic).unwrap();
    let preload_relations = ctx
        .state()
        .resolve_table_references(&statement)
        .unwrap()
        .into_iter()
        .map(|relation| relation.to_string())
        .collect::<Vec<_>>();
    assert!(preload_relations.iter().any(|relation| relation == "dataset_vectors"));
    assert!(preload_relations.iter().any(|relation| relation == "query_vectors"));

    let plan = logical_plan(&ctx, &sql).await;
    let node = find_cuvs_node(&plan).expect("cuVS logical extension");
    let dataset = node.inputs()[0].display_indent().to_string();
    let queries = node.inputs()[1].display_indent().to_string();
    assert!(dataset.contains("Projection:"), "{dataset}");
    assert!(dataset.contains("Join:"), "{dataset}");
    assert!(dataset.contains("Filter:"), "{dataset}");
    assert!(queries.contains("Projection:"), "{queries}");
    assert!(queries.contains("Filter:"), "{queries}");
    assert!(!dataset.contains("EmptyRelation"), "{dataset}");
    assert!(!queries.contains("EmptyRelation"), "{queries}");
}

#[tokio::test]
async fn bare_identifier_is_omitted_from_async_table_reference_preload() {
    let ctx = cuvs_context();
    let sql = format!("SELECT * FROM {CUVS_KMEANS_FUNCTION}(dataset_vectors, '{}')", sql_literal(KMEANS_OPTIONS));
    let statement = ctx.state().sql_to_statement(&sql, &Dialect::Generic).unwrap();
    let preloaded = ctx
        .state()
        .resolve_table_references(&statement)
        .unwrap()
        .into_iter()
        .map(|relation| relation.to_string())
        .collect::<Vec<_>>();
    assert!(!preloaded.iter().any(|relation| relation == "dataset_vectors"), "{preloaded:?}");
    let error = ctx.state().create_logical_plan(&sql).await.expect_err("bare relation rejected");
    let typed = error.find_nexus_error().expect("structured bare relation error");
    assert_eq!(typed.facts().get("reason"), Some("relation_argument_not_subquery"));
}

#[tokio::test]
async fn operation_output_schemas_and_defaulted_options_are_stable() {
    let ctx = cuvs_context();
    let kmeans = logical_plan(
        &ctx,
        &format!(
            "SELECT * FROM {CUVS_KMEANS_FUNCTION}((SELECT * FROM dataset_vectors), '{}')",
            sql_literal(KMEANS_OPTIONS)
        ),
    )
    .await;
    let kmeans = find_cuvs_node(&kmeans).unwrap();
    assert_schema(kmeans.schema(), &["row_ordinal", "id", "cluster_id"]);
    let normalized: serde_json::Value = serde_json::from_str(kmeans.validated_options().canonical_json()).unwrap();
    assert_eq!(normalized["metric"], "l2_expanded");
    assert_eq!(normalized["max_iter"], 100);
    assert_eq!(normalized["tol"], 0.0001);
    assert_eq!(normalized["n_init"], 1);
    assert_eq!(normalized["init"], "kmeans++");

    let pca = logical_plan(
        &ctx,
        &format!("SELECT * FROM {CUVS_PCA_FUNCTION}((SELECT * FROM dataset_vectors), '{}')", sql_literal(PCA_OPTIONS)),
    )
    .await;
    let pca = find_cuvs_node(&pca).unwrap();
    assert_schema(pca.schema(), &["row_ordinal", "id", "pc_0", "pc_1"]);
    let normalized: serde_json::Value = serde_json::from_str(pca.validated_options().canonical_json()).unwrap();
    assert_eq!(normalized["whiten"], false);
    assert_eq!(normalized["solver"], "cov_eig_dq");
    assert_eq!(normalized["n_iterations"], 15);
}

#[tokio::test]
async fn unified_gpu_metadata_is_relation_aware_and_projects_the_canonical_registry() {
    let ctx = cuvs_context();
    let listed = ctx
        .sql(&format!("SELECT function_name FROM {GPU_LIST_FUNCTIONS_FUNCTION}() WHERE provider = 'cuvs'"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(listed.iter().map(|batch| batch.num_rows()).sum::<usize>(), CUVS_OPERATION_REGISTRY.len());

    let described = ctx
        .sql(&format!("SELECT * FROM {GPU_DESCRIBE_FUNCTION_FUNCTION}('cuvs_pca')"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(described.iter().map(|batch| batch.num_rows()).sum::<usize>(), 1);
    let described_batch = &described[0];
    let validation_schema: serde_json::Value = serde_json::from_str(
        described_batch.column(10).as_any().downcast_ref::<arrow::array::StringArray>().unwrap().value(0),
    )
    .unwrap();
    assert_eq!(
        validation_schema["properties"]["options"]["description"],
        "cuVS operation options and relation-column bindings. See options_schema_json."
    );
    assert!(validation_schema["properties"]["options"].get("properties").is_none());

    let call = serde_json::json!({
        "schema_version": 1,
        "relations": {"input": {"table": "dataset_vectors"}},
        "options": serde_json::from_str::<serde_json::Value>(KMEANS_OPTIONS).unwrap(),
    });
    let valid = ctx
        .sql(&format!(
            "SELECT valid, error_code, relations_resolved, normalized_call_json, output_schema_json, would_execute_gpu \
             FROM {GPU_VALIDATE_CALL_FUNCTION}('cuvs_kmeans', '{}')",
            sql_literal(&call.to_string())
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let valid_column = valid[0].column(0).as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
    let error_code = valid[0].column(1).as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    let resolved_column = valid[0].column(2).as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
    let would_execute = valid[0].column(5).as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
    #[cfg(feature = "cuvs")]
    {
        assert!(!valid_column.value(0));
        assert_eq!(error_code.value(0), "bounded_execution_unavailable");
        assert!(!resolved_column.value(0));
        assert!(!would_execute.value(0));
    }
    #[cfg(not(feature = "cuvs"))]
    {
        assert!(!valid_column.value(0));
        assert_eq!(error_code.value(0), "required_feature_disabled");
        assert!(!resolved_column.value(0));
        assert!(!would_execute.value(0));
    }
}

#[cfg(feature = "cuvs")]
#[tokio::test]
async fn unified_gpu_metadata_reports_bounded_cuvs_as_unavailable() {
    let state = SessionStateBuilder::new_with_default_features().try_with_cuvs_sql().unwrap().build();
    let ctx = SessionContext::new_with_state(state);
    let listed = ctx
        .sql(&format!(
            "SELECT available, unavailable_reason FROM {GPU_LIST_FUNCTIONS_FUNCTION}() WHERE function_name = 'cuvs_kmeans'"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let available = listed[0].column(0).as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
    let reason = listed[0].column(1).as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    assert!(!available.value(0));
    assert_eq!(reason.value(0), "bounded_execution_unavailable");

    let call = serde_json::json!({
        "schema_version": 1,
        "relations": {"input": {"table": "dataset_vectors"}},
        "options": serde_json::from_str::<serde_json::Value>(KMEANS_OPTIONS).unwrap(),
    });
    let validated = ctx
        .sql(&format!(
            "SELECT valid, error_code, would_execute_gpu FROM {GPU_VALIDATE_CALL_FUNCTION}('cuvs_kmeans', '{}')",
            sql_literal(&call.to_string())
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let valid = validated[0].column(0).as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
    let code = validated[0].column(1).as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    let would_execute = validated[0].column(2).as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
    assert!(!valid.value(0));
    assert_eq!(code.value(0), "bounded_execution_unavailable");
    assert!(!would_execute.value(0));
}

#[tokio::test]
async fn unified_gpu_metadata_reports_cuvs_unavailability_before_relation_validation() {
    let ctx = cuvs_context();
    let missing = serde_json::json!({
        "schema_version": 1,
        "relations": {"input": {"table": "missing_vectors"}},
        "options": serde_json::from_str::<serde_json::Value>(KMEANS_OPTIONS).unwrap(),
    });
    let missing = ctx
        .sql(&format!(
            "SELECT valid, error_code, relations_resolved FROM {GPU_VALIDATE_CALL_FUNCTION}('cuvs_kmeans', '{}')",
            sql_literal(&missing.to_string())
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let missing_valid = missing[0].column(0).as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
    let missing_code = missing[0].column(1).as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    let missing_resolved = missing[0].column(2).as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
    #[cfg(feature = "cuvs")]
    {
        assert!(!missing_valid.value(0));
        assert_eq!(missing_code.value(0), "bounded_execution_unavailable");
        assert!(!missing_resolved.value(0));
    }
    #[cfg(not(feature = "cuvs"))]
    {
        assert!(!missing_valid.value(0));
        assert_eq!(missing_code.value(0), "required_feature_disabled");
        assert!(!missing_resolved.value(0));
    }
}

#[tokio::test]
async fn relation_arguments_require_parenthesized_subqueries() {
    let ctx = cuvs_context();
    for relation_argument in ["dataset_vectors", "'dataset_vectors'"] {
        let sql =
            format!("SELECT * FROM {CUVS_KMEANS_FUNCTION}({relation_argument}, '{}')", sql_literal(KMEANS_OPTIONS));
        let error = ctx.state().create_logical_plan(&sql).await.expect_err(&sql);
        let typed = error.find_nexus_error().unwrap_or_else(|| panic!("missing typed error for {sql}: {error:?}"));
        assert_eq!(typed.code(), ErrorCode::InvalidRequest, "{sql}");
        assert_eq!(typed.facts().get("reason"), Some("relation_argument_not_subquery"), "{sql}");
    }
}

#[tokio::test]
async fn local_validation_rejects_schema_dimension_option_and_empty_input_contracts() {
    let ctx = cuvs_context();
    let cases = [
        (
            format!(
                "SELECT * FROM {CUVS_KMEANS_FUNCTION}((SELECT * FROM dataset_vectors), '{{\"input\":{{\"id\":\"missing\",\"vector\":{{\"columns\":[\"d0\",\"d1\"]}}}},\"n_clusters\":1}}')"
            ),
            ErrorCode::FieldNotFound,
            None,
        ),
        (
            format!(
                "SELECT * FROM {CUVS_KMEANS_FUNCTION}((SELECT * FROM bad_id_vectors), '{{\"input\":{{\"id\":\"item_id\",\"vector\":{{\"columns\":[\"d0\",\"d1\"]}}}},\"n_clusters\":1}}')"
            ),
            ErrorCode::UnsupportedType,
            None,
        ),
        (
            format!(
                "SELECT * FROM {CUVS_KMEANS_FUNCTION}((SELECT * FROM nullable_vectors), '{}')",
                sql_literal(KMEANS_OPTIONS)
            ),
            ErrorCode::InvalidRequest,
            Some("nullable_feature_field"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
                    (SELECT * FROM dataset_vectors), \
                    (SELECT * FROM one_dim_queries), \
                    '{{\"dataset\":{{\"id\":\"item_id\",\"vector\":{{\"columns\":[\"d0\",\"d1\"]}}}},\
                       \"queries\":{{\"id\":\"query_id\",\"vector\":{{\"columns\":[\"d0\"]}}}},\
                       \"k\":1,\"metric\":\"cosine\"}}')"
            ),
            ErrorCode::InvalidRequest,
            Some("vector_dimension_mismatch"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_KMEANS_FUNCTION}((SELECT * FROM dataset_vectors), '{{\"input\":{{\"id\":\"item_id\",\"vector\":{{\"columns\":[\"d0\",\"d1\"]}}}},\"n_clusters\":1,\"unknown\":true}}')"
            ),
            ErrorCode::InvalidRequest,
            Some("unknown_option"),
        ),
        (
            format!(
                "WITH empty_input AS (SELECT CAST(1 AS BIGINT) AS item_id, CAST(1 AS FLOAT) AS d0, CAST(2 AS FLOAT) AS d1 LIMIT 0) \
                 SELECT * FROM {CUVS_KMEANS_FUNCTION}((SELECT * FROM empty_input), '{}')",
                sql_literal(KMEANS_OPTIONS)
            ),
            ErrorCode::InvalidRequest,
            Some("empty_input"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_PCA_FUNCTION}((SELECT * FROM dataset_vectors), '{{\"input\":{{\"id\":\"item_id\",\"vector\":{{\"columns\":[\"d0\",\"d1\"]}}}},\"n_components\":3}}')"
            ),
            ErrorCode::InvalidRequest,
            Some("component_count_exceeds_dimension"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_PCA_FUNCTION}((SELECT * FROM one_dim_queries), '{{\"input\":{{\"id\":\"query_id\",\"vector\":{{\"columns\":[\"d0\"]}}}},\"n_components\":1}}')"
            ),
            ErrorCode::InvalidRequest,
            Some("insufficient_feature_dimension"),
        ),
    ];
    for (sql, expected_code, expected_reason) in cases {
        let error = ctx.state().create_logical_plan(&sql).await.expect_err(&sql);
        let typed = error.find_nexus_error().unwrap_or_else(|| panic!("missing typed error for {sql}: {error:?}"));
        assert_eq!(typed.code(), expected_code, "{sql}");
        if let Some(reason) = expected_reason {
            assert_eq!(typed.facts().get("reason"), Some(reason), "{sql}");
        }
    }
}

#[derive(Debug)]
struct ExecuteCountingExec {
    input: Arc<dyn ExecutionPlan>,
    execute_count: Arc<AtomicUsize>,
}

impl DisplayAs for ExecuteCountingExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.input.fmt_as(t, f)
    }
}

impl ExecutionPlan for ExecuteCountingExec {
    fn name(&self) -> &str {
        "ExecuteCountingExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> datafusion::common::Result<TreeNodeRecursion>,
    ) -> datafusion::common::Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(datafusion::common::DataFusionError::Execution(
                "ExecuteCountingExec expects one child".to_string(),
            ));
        }
        Ok(Arc::new(Self { input: children.remove(0), execute_count: Arc::clone(&self.execute_count) }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        self.execute_count.fetch_add(1, Ordering::SeqCst);
        self.input.execute(partition, context)
    }
}

#[tokio::test]
async fn bounded_execution_rejects_before_any_child_execute() {
    let ctx =
        fixtures::cuvs_context_with_native_execution_config(NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1));
    let sql = format!(
        "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             (SELECT * FROM dataset_vectors), \
             (SELECT * FROM query_vectors), \
             '{}')",
        sql_literal(KNN_OPTIONS)
    );
    let logical = logical_plan(&ctx, &sql).await;
    let physical = ctx.state().create_physical_plan(&logical).await.expect("bounded cuVS physical planning");
    let exec = find_cuvs_exec(physical.as_ref()).expect("CuvsAlgoExec");
    let execute_count = Arc::new(AtomicUsize::new(0));
    let children = exec
        .children()
        .into_iter()
        .map(|child| {
            Arc::new(ExecuteCountingExec { input: Arc::clone(child), execute_count: Arc::clone(&execute_count) })
                as Arc<dyn ExecutionPlan>
        })
        .collect();
    let rebound = Arc::new(exec.clone())
        .replace_children(children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        .expect("rebind counting children");
    let error = match rebound.execute(0, Arc::new(TaskContext::default())) {
        Ok(_) => panic!("bounded cuVS execution must reject before creating a result stream"),
        Err(error) => error,
    };
    let typed = error.find_nexus_error().expect("structured bounded cuVS rejection");
    assert_eq!(typed.code(), ErrorCode::UnsupportedOperator);
    #[cfg(feature = "cuvs")]
    {
        assert_eq!(typed.facts().get("reason"), Some("blocked_lower_level_contract"));
        assert_eq!(typed.facts().get("host_fallback"), Some("disabled"));
        assert_eq!(typed.facts().get("required_contract"), Some("cuvs_vector_algorithm_peak_memory_preflight"));
    }
    #[cfg(not(feature = "cuvs"))]
    {
        assert_eq!(typed.facts().get("reason"), Some("cuvs_feature_disabled"));
        assert_eq!(typed.facts().get("required_feature"), Some("cuvs"));
        assert_eq!(typed.facts().get("host_fallback"), Some("disabled"));
    }
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "bounded rejection must precede child execution");
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn cugraph_reaches_cuvs_as_a_direct_gpu_resident_producer() {
    let ctx = fixtures::cuvs_cugraph_context();
    let graph_knn_options = r#"{
      "dataset":{"id":"vertex","vector":{"columns":["x","y"]}},
      "queries":{"id":"query_id","vector":{"columns":["d0","d1"]}},
      "k":1,
      "metric":"cosine"
    }"#;
    let sql = format!(
        "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             (SELECT * FROM cugraph_force_atlas2('edges')), \
             (SELECT * FROM query_vectors), \
             '{}')",
        sql_literal(graph_knn_options)
    );
    let plan = logical_plan(&ctx, &sql).await;
    let node = find_cuvs_node(&plan).expect("cuVS logical extension");
    let dataset = node.inputs()[0].display_indent().to_string();
    assert!(dataset.contains("TableScan: cugraph_force_atlas2()"), "{dataset}");

    let physical = ctx.state().create_physical_plan(&plan).await.expect("combined cuGraph/cuVS physical plan");
    let cuvs = find_cuvs_exec(physical.as_ref()).expect("CuvsAlgoExec");
    let dataset = cuvs.children()[0];
    assert_eq!(dataset.name(), "CugraphAlgoExec", "cuGraph must be the direct cuVS child with no host wrapper");
    assert!(
        dataset.downcast_ref::<datafusion_nexus::cugraph_sql::CugraphAlgoExec>().is_some(),
        "direct child must retain the typed GPU-resident producer"
    );
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn cugraph_join_filter_projection_reaches_bounded_cuvs_rejection_before_child_execution() {
    let ctx = fixtures::cuvs_cugraph_native_context();
    let sql = format!(
        "WITH candidates AS (\
             SELECT e.item_id, e.d0, e.d1 \
             FROM cugraph_louvain('edges') AS g \
             JOIN dataset_vectors AS e ON e.item_id = g.vertex \
             WHERE e.d0 >= CAST(0 AS REAL)\
         ) \
         SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             (SELECT item_id, d0, d1 FROM candidates), \
             (SELECT * FROM query_vectors), \
             '{}')",
        sql_literal(KNN_OPTIONS)
    );
    let logical = logical_plan(&ctx, &sql).await;
    let physical =
        ctx.state().create_physical_plan(&logical).await.expect("combined relational cuGraph/cuVS physical plan");
    let cuvs = find_cuvs_exec(physical.as_ref()).expect("CuvsAlgoExec");
    let execute_count = Arc::new(AtomicUsize::new(0));
    let children = cuvs
        .children()
        .into_iter()
        .map(|child| {
            Arc::new(ExecuteCountingExec { input: Arc::clone(child), execute_count: Arc::clone(&execute_count) })
                as Arc<dyn ExecutionPlan>
        })
        .collect();
    let rebound = Arc::new(cuvs.clone())
        .replace_children(children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        .expect("rebind counting children");
    let error = match rebound.execute(0, Arc::new(TaskContext::default())) {
        Ok(_) => panic!("bounded cuVS execution must reject before touching relational cuGraph inputs"),
        Err(error) => error,
    };
    let typed = error.find_nexus_error().expect("structured bounded cuVS rejection");
    assert_eq!(typed.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(typed.facts().get("reason"), Some("blocked_lower_level_contract"));
    assert_eq!(typed.facts().get("host_fallback"), Some("disabled"));
    assert_eq!(typed.facts().get("required_contract"), Some("cuvs_vector_algorithm_peak_memory_preflight"));
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "bounded rejection must precede child execution");
}

async fn logical_plan(ctx: &SessionContext, sql: &str) -> LogicalPlan {
    ctx.state()
        .create_logical_plan(sql)
        .await
        .unwrap_or_else(|error| panic!("logical plan failed for {sql}: {error:?}"))
}

fn find_cuvs_node(plan: &LogicalPlan) -> Option<&CuvsLogicalNode> {
    if let LogicalPlan::Extension(extension) = plan
        && let Some(node) = extension.node.as_any().downcast_ref::<CuvsLogicalNode>()
    {
        return Some(node);
    }
    plan.inputs().into_iter().find_map(find_cuvs_node)
}

fn find_cuvs_exec(plan: &dyn ExecutionPlan) -> Option<&CuvsAlgoExec> {
    if let Some(exec) = plan.downcast_ref::<CuvsAlgoExec>() {
        return Some(exec);
    }
    plan.children().into_iter().find_map(|child| find_cuvs_exec(child.as_ref()))
}

fn assert_schema(schema: &DFSchema, names: &[&str]) {
    assert_eq!(schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(), names);
    assert!(schema.fields().iter().all(|field| !field.is_nullable()));
}

#[cfg(not(feature = "cuvs"))]
fn assert_feature_disabled(error: datafusion::common::DataFusionError) {
    let typed = error.find_nexus_error().expect("structured cuVS feature-disabled error");
    assert_eq!(typed.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(typed.facts().get("reason"), Some("cuvs_feature_disabled"));
    assert_eq!(typed.facts().get("host_fallback"), Some("disabled"));
}

fn sql_literal(json: &str) -> String {
    json.replace('\'', "''")
}
