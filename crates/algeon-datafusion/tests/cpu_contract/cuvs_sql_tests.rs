use std::{
    collections::BTreeSet,
    fs,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[cfg(feature = "cugraph")]
use algeon_datafusion::cugraph_sql::CugraphAlgoExec;
use algeon_datafusion::{
    cuvs_sql::{
        CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_CAGRA_FUNCTION, CUVS_IVF_FLAT_FUNCTION, CUVS_IVF_PQ_FUNCTION,
        CUVS_KMEANS_FUNCTION, CUVS_OPERATION_REGISTRY, CUVS_PAIRWISE_DISTANCE_FUNCTION, CUVS_PCA_FUNCTION,
        CUVS_TOP_K_FUNCTION, CuvsAlgoExec, CuvsLogicalNode, CuvsOperation, CuvsOptionKind, CuvsRelationRole,
        cuvs_options_json_schema,
    },
    error::{DataFusionErrorExt, ErrorCode},
    gpu_functions::{GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_LIST_FUNCTIONS_FUNCTION, GPU_VALIDATE_CALL_FUNCTION},
    native_exec::AlgeonNativeExecutionConfig,
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{Array, BooleanArray, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    catalog::Session,
    common::{DFSchema, DataFusionError, Result as DataFusionResult, config::Dialect, tree_node::TreeNodeRecursion},
    datasource::MemTable,
    execution::{
        SessionStateBuilder,
        context::{QueryPlanner, TaskContext},
    },
    logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore},
    physical_expr::PhysicalExpr,
    physical_plan::{
        ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, ReplaceChildrenOptions,
        SendableRecordBatchStream,
    },
    physical_planner::DefaultPhysicalPlanner,
    prelude::SessionContext,
    sql::{
        parser::Statement as DataFusionStatement,
        sqlparser::ast::{
            Expr as SqlExpr, FunctionArg, FunctionArgExpr, FunctionArgOperator, SetExpr, Statement as SqlStatement,
            TableFactor,
        },
    },
};
use query_engine::plan::{
    CUVS_ID_TYPES, CuvsIdType, CuvsOutputContract, CuvsVectorElementType, vector_kmeans_output_schema,
    vector_knn_output_schema, vector_pairwise_distance_output_schema, vector_pca_output_schema,
    vector_top_k_output_schema,
};

#[path = "cuvs_sql_fixtures.rs"]
mod fixtures;

use fixtures::{KMEANS_OPTIONS, cuvs_context};

#[test]
fn canonical_registry_drives_names_relations_and_json_schemas() {
    let names = CUVS_OPERATION_REGISTRY.iter().map(|descriptor| descriptor.sql_function_name).collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            CUVS_BRUTE_FORCE_KNN_FUNCTION,
            CUVS_IVF_FLAT_FUNCTION,
            CUVS_IVF_PQ_FUNCTION,
            CUVS_CAGRA_FUNCTION,
            CUVS_PAIRWISE_DISTANCE_FUNCTION,
            CUVS_TOP_K_FUNCTION,
            CUVS_KMEANS_FUNCTION,
            CUVS_PCA_FUNCTION
        ]
    );
    assert_eq!(
        CuvsOperation::BruteForceKnn.descriptor().relation_roles,
        [CuvsRelationRole::Dataset, CuvsRelationRole::Queries]
    );
    assert_eq!(CuvsOperation::KMeans.descriptor().relation_roles, [CuvsRelationRole::Input]);
    assert_eq!(CuvsRelationRole::X.as_str(), "x");
    assert_eq!(CuvsRelationRole::Y.as_str(), "y");
    let expected = [
        &[CuvsVectorElementType::Float32][..],
        &[CuvsVectorElementType::Float32, CuvsVectorElementType::Int8, CuvsVectorElementType::UInt8],
        &[CuvsVectorElementType::Float32, CuvsVectorElementType::Int8, CuvsVectorElementType::UInt8],
        &[CuvsVectorElementType::Float32, CuvsVectorElementType::Int8, CuvsVectorElementType::UInt8],
        &[CuvsVectorElementType::Float32, CuvsVectorElementType::Float64],
        &[CuvsVectorElementType::Float32],
        &[CuvsVectorElementType::Float32, CuvsVectorElementType::Float64],
        &[CuvsVectorElementType::Float32],
    ];
    for (descriptor, expected) in CUVS_OPERATION_REGISTRY.iter().zip(expected) {
        assert_eq!(
            descriptor.vector_element_support.iter().map(|support| support.element).collect::<Vec<_>>(),
            expected
        );
        let metric_option = descriptor.option("metric");
        for support in descriptor.vector_element_support {
            if let Some(option) = metric_option {
                let CuvsOptionKind::Enum { values, .. } = option.kind else { panic!("metric must be an enum") };
                assert!(!support.metrics.is_empty());
                assert!(support.metrics.iter().all(|metric| values.contains(metric)));
            } else {
                assert!(support.metrics.is_empty());
            }
        }
    }
    assert_eq!(
        CuvsOperation::IvfPq.descriptor().element_support(CuvsVectorElementType::Int8).unwrap().metrics,
        ["l2_expanded", "l2_sqrt_expanded", "inner_product"]
    );

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
        &[CuvsIdType::Int32, CuvsIdType::Int64, CuvsIdType::Utf8, CuvsIdType::LargeUtf8, CuvsIdType::Utf8View,]
    );

    let ivf = cuvs_options_json_schema(CuvsOperation::IvfFlat);
    assert_eq!(ivf["required"], serde_json::json!(["dataset", "queries", "k", "metric", "n_lists"]));
    assert_eq!(ivf["properties"]["n_lists"]["minimum"], 1);
    assert_eq!(ivf["properties"]["n_probes"]["default"], 20);
    assert_eq!(ivf["properties"]["kmeans_n_iters"]["default"], 20);
    assert_eq!(ivf["properties"]["kmeans_trainset_fraction"]["exclusiveMinimum"], 0);
    assert_eq!(ivf["properties"]["kmeans_trainset_fraction"]["maximum"], 1);
    assert_eq!(ivf["properties"]["kmeans_trainset_fraction"]["default"], 0.5);
    assert_eq!(ivf["properties"]["metric"]["enum"], knn["properties"]["metric"]["enum"]);

    let pq = cuvs_options_json_schema(CuvsOperation::IvfPq);
    assert_eq!(pq["required"], ivf["required"]);
    assert_eq!(
        pq["properties"]["metric"]["enum"],
        serde_json::json!(["l2_expanded", "l2_sqrt_expanded", "cosine", "inner_product"])
    );
    assert_eq!(pq["properties"]["pq_dim"]["minimum"], 0);
    assert_eq!(pq["properties"]["pq_dim"]["default"], 0);
    assert_eq!(pq["properties"]["pq_bits"]["minimum"], 4);
    assert_eq!(pq["properties"]["pq_bits"]["maximum"], 8);
    assert_eq!(pq["properties"]["pq_bits"]["default"], 8);
    assert_eq!(pq["properties"]["codebook_kind"]["enum"], serde_json::json!(["per_subspace", "per_cluster"]));
    assert_eq!(pq["properties"]["codebook_kind"]["default"], "per_subspace");

    let cagra = cuvs_options_json_schema(CuvsOperation::Cagra);
    assert_eq!(cagra["required"], serde_json::json!(["dataset", "queries", "k", "metric"]));
    assert_eq!(cagra["properties"]["build_algo"]["enum"], serde_json::json!(["ivf_pq", "nn_descent"]));
    assert_eq!(cagra["properties"]["itopk_size"]["default"], 64);
    assert_eq!(cagra["properties"]["metric"]["enum"], serde_json::json!(["l2_expanded", "inner_product", "cosine"]));

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
    // The adapter manifest is under crates/<name>, two levels below the repository root.
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("adapter manifest must be two levels below the repository root");
    let spec_directory = repository.join("website/gpu-function-specs/cuvs");
    let page_directory = repository.join("website/docs/cuvs/functions");
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
        for option in descriptor.options.iter().filter(|option| !matches!(option.kind, CuvsOptionKind::RelationBinding))
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
        for role in roles.as_array().unwrap() {
            assert_eq!(
                role["fields"]["features"]["element_metric_support"],
                serde_json::json!(
                    descriptor
                        .vector_element_support
                        .iter()
                        .map(|support| serde_json::json!({
                            "element": support.element.as_str(), "metrics": support.metrics,
                        }))
                        .collect::<Vec<_>>()
                ),
            );
            assert_eq!(
                role["fields"]["features"]["element_types"],
                serde_json::json!(
                    descriptor
                        .vector_element_support
                        .iter()
                        .map(|support| support.element.as_str())
                        .collect::<Vec<_>>()
                ),
            );
        }
        let result_schema =
            serde_json::from_str::<serde_json::Value>(captured["result_schemas_json"].as_str().unwrap()).unwrap();
        assert_eq!(
            result_schema["output_contract"],
            match descriptor.output_contract {
                CuvsOutputContract::KnnNeighbors => "knn_neighbors",
                CuvsOutputContract::KMeansAssignments => "kmeans_assignments",
                CuvsOutputContract::PcaComponents => "pca_components",
                CuvsOutputContract::PairwiseDistances => "pairwise_distances",
                CuvsOutputContract::TopKValues => "top_k_values",
            },
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
        for relation_role in descriptor.relation_roles {
            assert!(page.contains(&format!("`{}`", relation_role.as_str())), "{}", page_path.display());
        }
        for support in descriptor.vector_element_support {
            assert!(page.contains(&format!("| `{}` |", support.element.as_str())), "{}", page_path.display());
            if let Some(metric) = support.metrics.first() {
                assert!(page.contains(&format!("`{metric}`")), "{}", page_path.display());
            }
        }
        for option in descriptor.options.iter().filter(|option| !matches!(option.kind, CuvsOptionKind::RelationBinding))
        {
            assert!(page.contains(&format!("`{}`", option.name)), "{}", page_path.display());
            assert!(page.contains(&mdx_table_text(option.description)), "{}", page_path.display());
        }
        for field in descriptor.output_fields {
            assert!(page.contains(&format!("`{}`", field.name)), "{}", page_path.display());
            assert!(page.contains(&mdx_table_text(field.description)), "{}", page_path.display());
        }
    }
    let partial = fs::read_to_string(repository.join("website/docs/cuvs/_vector-element-support.mdx")).unwrap();
    for descriptor in CUVS_OPERATION_REGISTRY {
        let row =
            partial.lines().find(|line| line.starts_with(&format!("| `{}` |", descriptor.sql_function_name))).unwrap();
        let cells = [
            CuvsVectorElementType::Float32,
            CuvsVectorElementType::Float64,
            CuvsVectorElementType::Int8,
            CuvsVectorElementType::UInt8,
        ]
        .map(|element| {
            descriptor.element_support(element).map_or_else(
                || "No".to_string(),
                |support| {
                    if support.metrics.is_empty() {
                        "Yes".to_string()
                    } else {
                        support.metrics.iter().map(|metric| format!("`{metric}`")).collect::<Vec<_>>().join(", ")
                    }
                },
            )
        });
        assert_eq!(row, format!("| `{}` | {} |", descriptor.sql_function_name, cells.join(" | ")));
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
            "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
                 dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                 queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
                 k => 1, metric => 'cosine')"
        ),
    )
    .await;
    let knn = find_cuvs_node(&knn).unwrap();
    let expected_knn = vector_knn_output_schema(DataType::Utf8, DataType::Int64);
    let expected_knn = DFSchema::try_from(expected_knn.as_ref().clone()).unwrap();
    assert_eq!(knn.schema().as_ref(), &expected_knn);

    let pairwise = logical_plan(
        &ctx,
        &format!(
            "SELECT * FROM {CUVS_PAIRWISE_DISTANCE_FUNCTION}(\
                 x => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                 y => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
                 metric => 'inner_product')"
        ),
    )
    .await;
    let pairwise = find_cuvs_node(&pairwise).unwrap();
    let expected = vector_pairwise_distance_output_schema(DataType::Int64, DataType::Utf8, DataType::Float32);
    assert_eq!(pairwise.schema().as_ref(), &DFSchema::try_from(expected.as_ref().clone()).unwrap());
    let pairwise_f64 = logical_plan(&ctx, &format!(
        "SELECT * FROM {CUVS_PAIRWISE_DISTANCE_FUNCTION}(\
         x => (SELECT item_id AS id, CAST(d0 AS DOUBLE) AS d0, CAST(d1 AS DOUBLE) AS d1 FROM dataset_vectors), \
         y => (SELECT query_id AS id, CAST(d0 AS DOUBLE) AS d0, CAST(d1 AS DOUBLE) AS d1 FROM query_vectors), metric => 'l2_expanded')"
    )).await;
    let pairwise_f64 = find_cuvs_node(&pairwise_f64).unwrap();
    let expected_f64 = vector_pairwise_distance_output_schema(DataType::Int64, DataType::Utf8, DataType::Float64);
    assert_eq!(pairwise_f64.schema().as_ref(), &DFSchema::try_from(expected_f64.as_ref().clone()).unwrap());

    let top_k = logical_plan(
        &ctx,
        &format!(
            "SELECT * FROM {CUVS_TOP_K_FUNCTION}(\
                 input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                 k => 2, select => 'max')"
        ),
    )
    .await;
    let top_k = find_cuvs_node(&top_k).unwrap();
    let expected = vector_top_k_output_schema(DataType::Int64);
    assert_eq!(top_k.schema().as_ref(), &DFSchema::try_from(expected.as_ref().clone()).unwrap());

    let kmeans = logical_plan(
        &ctx,
        &format!(
            "SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
                 input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                 n_clusters => 1)"
        ),
    )
    .await;
    let kmeans = find_cuvs_node(&kmeans).unwrap();
    let expected_kmeans = vector_kmeans_output_schema(DataType::Int64);
    let expected_kmeans = DFSchema::try_from(expected_kmeans.as_ref().clone()).unwrap();
    assert_eq!(kmeans.schema().as_ref(), &expected_kmeans);

    let pca = logical_plan(
        &ctx,
        &format!(
            "SELECT * FROM {CUVS_PCA_FUNCTION}(\
                 input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                 n_components => 2)"
        ),
    )
    .await;
    let pca = find_cuvs_node(&pca).unwrap();
    let expected_pca = vector_pca_output_schema(DataType::Int64, 2);
    let expected_pca = DFSchema::try_from(expected_pca.as_ref().clone()).unwrap();
    assert_eq!(pca.schema().as_ref(), &expected_pca);
}

#[test]
fn identical_cuvs_installation_is_idempotent_over_its_existing_query_planner() {
    let builder = SessionStateBuilder::new_with_default_features().try_with_cuvs_sql().expect("first cuVS SQL install");
    let mut builder = builder.try_with_cuvs_sql().expect("identical second cuVS SQL install must be a no-op");
    assert!(builder.query_planner().is_some());
}

#[derive(Debug)]
struct SentinelQueryPlanner;

#[async_trait::async_trait]
impl QueryPlanner for SentinelQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_planner::PhysicalPlanner;

        DefaultPhysicalPlanner::default().create_physical_plan(logical_plan, session).await
    }
}

#[test]
fn try_with_cuvs_sql_rejects_opaque_custom_query_planner() {
    let builder = SessionStateBuilder::new_with_default_features().with_query_planner(Arc::new(SentinelQueryPlanner));
    let error =
        builder.try_with_cuvs_sql().expect_err("cuVS SQL installation must reject existing opaque query planner");
    let typed = error.find_algeon_error().expect("structured error");
    assert_eq!(typed.fact_value("reason"), Some("query_planner_conflict"));
}

#[test]
fn try_with_cuvs_sql_rejects_reinstall_when_planner_is_replaced() {
    let builder = SessionStateBuilder::new_with_default_features().try_with_cuvs_sql().expect("first cuVS install");
    let builder = builder.with_query_planner(Arc::new(SentinelQueryPlanner));
    let error =
        builder.try_with_cuvs_sql().expect_err("cuVS SQL reinstall must not succeed when query planner was replaced");
    let typed = error.find_algeon_error().expect("structured error");
    assert_eq!(typed.fact_value("reason"), Some("query_planner_conflict"));
}

#[tokio::test]
async fn root_relation_binding_validation_delegates_to_canonical_registry() {
    let ctx = cuvs_context();
    let sql = format!(
        "SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
             input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             n_clusters => 1)"
    );
    let plan = logical_plan(&ctx, &sql).await;
    let node = find_cuvs_node(&plan).expect("cuVS logical extension");
    let options: serde_json::Value = serde_json::from_str(node.validated_options().canonical_json()).unwrap();
    assert_eq!(options["input"]["id"], "id");
    assert_eq!(options["input"]["vector"]["columns"], serde_json::json!(["d0", "d1"]));
}

#[tokio::test]
async fn cagra_sql_binds_relations_and_canonical_search_options() {
    let ctx = cuvs_context();
    let sql = format!(
        "SELECT * FROM {CUVS_CAGRA_FUNCTION}(\
             dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'l2_expanded', \
             graph_degree => 2, intermediate_graph_degree => 3, itopk_size => 16)"
    );
    let plan = logical_plan(&ctx, &sql).await;
    let node = find_cuvs_node(&plan).expect("CAGRA logical extension");
    assert_eq!(node.operation(), CuvsOperation::Cagra);
    assert_eq!(node.inputs().len(), 2);
    let options: serde_json::Value = serde_json::from_str(node.validated_options().canonical_json()).unwrap();
    assert_eq!(options["build_algo"], "ivf_pq");
    assert_eq!(options["search_algo"], "auto");
    let physical = ctx.state().create_physical_plan(&plan).await.expect("CAGRA physical plan");
    assert!(find_cuvs_exec(physical.as_ref()).is_some());
}

#[tokio::test]
async fn catalog_relations_become_two_real_children_and_alias_resolves() {
    let ctx = cuvs_context();
    let sql = format!(
        "SELECT knn.query_id, knn.neighbor_id, knn.distance \
         FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             dataset => (SELECT item_id AS id, d0, d1 FROM datafusion.public.dataset_vectors), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'cosine') AS knn"
    );
    let statement = ctx.state().sql_to_statement(&sql, &Dialect::Generic).unwrap();
    let DataFusionStatement::Statement(statement_ast) = &statement else { panic!("expected a standard SQL statement") };
    let SqlStatement::Query(query) = statement_ast.as_ref() else { panic!("expected a query") };
    let SetExpr::Select(select) = query.body.as_ref() else { panic!("expected a select") };
    let TableFactor::Table { args: Some(args), .. } = &select.from[0].relation else {
        panic!("expected cuVS table-function syntax")
    };
    assert!(
        args.args.iter().all(|argument| {
            matches!(argument, FunctionArg::Named { operator: FunctionArgOperator::RightArrow, .. })
        })
    );
    assert!(args.args[..2].iter().all(|argument| {
        matches!(argument, FunctionArg::Named { arg: FunctionArgExpr::Expr(SqlExpr::Subquery(_)), .. })
    }));
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
    assert_eq!(node.schema().field_with_unqualified_name("query_id").unwrap().data_type(), &DataType::Utf8);
    assert_eq!(node.schema().field_with_unqualified_name("neighbor_id").unwrap().data_type(), &DataType::Int64);
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
             dataset => (SELECT item_id AS id, d0, d1 FROM candidates), \
             queries => (SELECT query_id AS id, d0, d1 FROM queries), \
             k => 1, metric => 'cosine')"
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
             dataset => (SELECT e.item_id AS id, e.d0, e.d1 FROM dataset_vectors AS e JOIN query_vectors AS q ON e.d0 = q.d0 WHERE e.d1 >= 0.0),\
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors WHERE d0 >= 0.0),\
             k => 1, metric => 'cosine')"
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
    let sql = format!("SELECT * FROM {CUVS_KMEANS_FUNCTION}(input => dataset_vectors, n_clusters => 1)");
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
    let typed = error.find_algeon_error().expect("structured bare relation error");
    assert_eq!(typed.facts().get("reason"), Some("relation_argument_not_subquery"));
}

#[tokio::test]
async fn operation_output_schemas_and_defaulted_options_are_stable() {
    let ctx = cuvs_context();
    let kmeans = logical_plan(
        &ctx,
        &format!(
            "SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
                 input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                 n_clusters => 1)"
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
        &format!(
            "SELECT * FROM {CUVS_PCA_FUNCTION}(\
                 input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                 n_components => 2)"
        ),
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
    let validation_schema: serde_json::Value =
        serde_json::from_str(described_batch.column(10).as_any().downcast_ref::<StringArray>().unwrap().value(0))
            .unwrap();
    let options_schema: serde_json::Value =
        serde_json::from_str(described_batch.column(8).as_any().downcast_ref::<StringArray>().unwrap().value(0))
            .unwrap();
    assert_eq!(validation_schema["properties"]["options"], options_schema);
    assert_eq!(options_schema["additionalProperties"], false);
    assert!(options_schema["required"].as_array().unwrap().contains(&serde_json::json!("n_components")));
    assert!(options_schema["properties"].get("input").is_none());

    let call = serde_json::json!({
        "relations": {"input": {"table": "canonical_vectors"}},
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
    let valid_column = valid[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
    let error_code = valid[0].column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let resolved_column = valid[0].column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
    let would_execute = valid[0].column(5).as_any().downcast_ref::<BooleanArray>().unwrap();
    #[cfg(feature = "cuvs")]
    {
        assert!(valid_column.value(0));
        assert!(error_code.is_null(0));
        assert!(resolved_column.value(0));
        assert!(would_execute.value(0));
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
async fn unified_gpu_metadata_reports_cuvs_as_available() {
    let ctx = cuvs_context();
    let listed = ctx
        .sql(&format!(
            "SELECT available, unavailable_reason FROM {GPU_LIST_FUNCTIONS_FUNCTION}() WHERE function_name = 'cuvs_kmeans'"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let available = listed[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
    let reason = listed[0].column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(available.value(0));
    assert!(reason.is_null(0));

    let call = serde_json::json!({
        "relations": {"input": {"table": "canonical_vectors"}},
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
    let valid = validated[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
    let code = validated[0].column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let would_execute = validated[0].column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(valid.value(0));
    assert!(code.is_null(0));
    assert!(would_execute.value(0));
}

#[tokio::test]
async fn unified_gpu_metadata_reports_cuvs_unavailability_before_relation_validation() {
    let ctx = cuvs_context();
    let missing = serde_json::json!({
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
    let missing_valid = missing[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
    let missing_code = missing[0].column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let missing_resolved = missing[0].column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
    #[cfg(feature = "cuvs")]
    {
        assert!(!missing_valid.value(0));
        assert_eq!(missing_code.value(0), "table_not_found");
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
        let sql = format!("SELECT * FROM {CUVS_KMEANS_FUNCTION}(input => {relation_argument}, n_clusters => 1)");
        let error = ctx.state().create_logical_plan(&sql).await.expect_err(&sql);
        let typed = error.find_algeon_error().unwrap_or_else(|| panic!("missing typed error for {sql}: {error:?}"));
        assert_eq!(typed.code(), ErrorCode::InvalidRequest, "{sql}");
        assert_eq!(typed.facts().get("reason"), Some("relation_argument_not_subquery"), "{sql}");
    }
}

#[tokio::test]
async fn named_binder_is_order_independent_and_rejects_duplicate_missing_and_positional_arguments() {
    let ctx = cuvs_context();
    let canonical = format!(
        "SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
             input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), n_clusters => 1, max_iter => 5)"
    );
    let reordered = format!(
        "SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
             max_iter => 5, n_clusters => 1, input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors))"
    );
    let canonical = ctx.state().create_logical_plan(&canonical).await.unwrap();
    let reordered = ctx.state().create_logical_plan(&reordered).await.unwrap();
    assert_eq!(canonical.display_indent().to_string(), reordered.display_indent().to_string());

    let cases = [
        (
            format!(
                "SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
                     input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), n_clusters => 1, n_clusters => 2)"
            ),
            "duplicate_argument",
        ),
        (
            format!(
                "SELECT * FROM {CUVS_KMEANS_FUNCTION}(input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors))"
            ),
            "missing_argument",
        ),
        (
            format!("SELECT * FROM {CUVS_KMEANS_FUNCTION}((SELECT item_id AS id, d0, d1 FROM dataset_vectors), 1)"),
            "positional_argument_not_supported",
        ),
    ];
    for (sql, reason) in cases {
        let error = ctx.state().create_logical_plan(&sql).await.expect_err(&sql);
        let typed = error.find_algeon_error().unwrap_or_else(|| panic!("missing typed error for {sql}: {error:?}"));
        assert_eq!(typed.facts().get("reason"), Some(reason), "{sql}");
    }
}

#[cfg(feature = "cuvs")]
#[tokio::test]
async fn unified_gpu_validation_details_locate_the_failing_argument_or_field() {
    let ctx = cuvs_context();
    let cases = [
        (
            r#"{"relations":{"input":{"table":"canonical_vectors"}},"options":{"n_clusters":0}}"#,
            "missing_or_invalid_option",
            serde_json::json!({"path": "options.n_clusters", "argument": "n_clusters"}),
        ),
        (
            r#"{"relations":{"input":{"table":"canonical_vectors"}},"options":{}}"#,
            "missing_or_invalid_option",
            serde_json::json!({"path": "options.n_clusters", "argument": "n_clusters"}),
        ),
        (
            r#"{"relations":{"input":{"table":"dataset_vectors"}},"options":{"n_clusters":1}}"#,
            "column_not_found",
            serde_json::json!({"path": "relations.input.fields.id", "relation_role": "input", "field": "id"}),
        ),
        (
            r#"{"relations":{"input":{"table":"missing_vectors"}},"options":{"n_clusters":1}}"#,
            "table_not_found",
            serde_json::json!({"path": "relations.input", "relation_role": "input"}),
        ),
    ];
    for (call, code, expected) in cases {
        let sql = format!(
            "SELECT error_code, details_json FROM gpu_validate_call('{CUVS_KMEANS_FUNCTION}', '{}')",
            call.replace('\'', "''")
        );
        let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
        let batch = &batches[0];
        let error_code = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap().value(0).to_string();
        let details: serde_json::Value =
            serde_json::from_str(batch.column(1).as_any().downcast_ref::<StringArray>().unwrap().value(0)).unwrap();
        assert_eq!(error_code, code, "{call}: {details}");
        for (key, value) in expected.as_object().unwrap() {
            assert_eq!(&details[key], value, "{call}: {details}");
        }
    }
}

#[tokio::test]
async fn unsupported_vector_elements_report_the_operation_support_subset() {
    let ctx = cuvs_context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("vector", DataType::List(Arc::new(Field::new("item", DataType::Float64, false))), false),
    ]));
    let batch = RecordBatch::new_empty(Arc::clone(&schema));
    let table = MemTable::try_new(schema, vec![vec![batch]]).expect("list fixture");
    ctx.register_table("float64_list_vectors", Arc::new(table)).expect("register list fixture");
    let cases = [
        "SELECT item_id AS id, CAST(d0 AS DOUBLE) AS d0, CAST(d1 AS DOUBLE) AS d1 FROM dataset_vectors",
        "SELECT id, vector FROM float64_list_vectors",
    ];
    for input in cases {
        let sql = format!("SELECT * FROM {CUVS_TOP_K_FUNCTION}(input => ({input}), k => 1, select => 'min')");
        let error = ctx.state().create_logical_plan(&sql).await.expect_err(&sql);
        let typed = error.find_algeon_error().expect("structured unsupported vector element error");
        assert_eq!(typed.code(), ErrorCode::UnsupportedType, "{sql}: {typed:?}");
        assert_eq!(typed.facts().get("expected_type"), Some("Float32"), "{sql}");
        assert!(typed.message().contains("Float32"), "{sql}: {typed}");
    }
}

#[tokio::test]
async fn local_validation_rejects_schema_dimension_option_and_empty_input_contracts() {
    let ctx = cuvs_context();
    let cases = [
        (
            format!(
                "SELECT * FROM {CUVS_PAIRWISE_DISTANCE_FUNCTION}(\
                     x => (SELECT item_id AS id, CAST(d0 AS DOUBLE) AS d0, d1 FROM dataset_vectors), \
                     y => (SELECT query_id AS id, d0, d1 FROM query_vectors), metric => 'l2_expanded')"
            ),
            ErrorCode::InvalidRequest,
            Some("vector_element_type_mismatch"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_PAIRWISE_DISTANCE_FUNCTION}(\
                     x => (SELECT item_id AS id, CAST(d0 AS DOUBLE) AS d0, CAST(d1 AS DOUBLE) AS d1 FROM dataset_vectors), \
                     y => (SELECT query_id AS id, d0, d1 FROM query_vectors), metric => 'l2_expanded')"
            ),
            ErrorCode::InvalidRequest,
            Some("vector_element_type_mismatch"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_IVF_PQ_FUNCTION}(\
                     dataset => (SELECT item_id AS id, CAST(d0 AS TINYINT) AS d0, CAST(d1 AS TINYINT) AS d1 FROM dataset_vectors), \
                     queries => (SELECT query_id AS id, CAST(d0 AS TINYINT) AS d0, CAST(d1 AS TINYINT) AS d1 FROM query_vectors), \
                     k => 1, n_lists => 1, n_probes => 1, metric => 'cosine')"
            ),
            ErrorCode::InvalidRequest,
            Some("unsupported_vector_element_metric"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
                     input => (SELECT item_id AS item, d0, d1 FROM dataset_vectors), \
                     n_clusters => 1)"
            ),
            ErrorCode::FieldNotFound,
            None,
        ),
        (
            format!(
                "SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
                     input => (SELECT item_id AS id, d0, d1 FROM bad_id_vectors), \
                     n_clusters => 1)"
            ),
            ErrorCode::UnsupportedType,
            None,
        ),
        (
            format!(
                "SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
                     input => (SELECT item_id AS id, d0, d1 FROM nullable_vectors), \
                     n_clusters => 1)"
            ),
            ErrorCode::InvalidRequest,
            Some("nullable_feature_field"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
                    dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                    queries => (SELECT query_id AS id, d0 FROM one_dim_queries), \
                    k => 1, metric => 'cosine')"
            ),
            ErrorCode::InvalidRequest,
            Some("vector_dimension_mismatch"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_PAIRWISE_DISTANCE_FUNCTION}(\
                     x => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                     y => (SELECT query_id AS id, d0 FROM one_dim_queries), \
                     metric => 'cosine')"
            ),
            ErrorCode::InvalidRequest,
            Some("vector_dimension_mismatch"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_TOP_K_FUNCTION}(\
                     input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                     k => 3, select => 'max')"
            ),
            ErrorCode::InvalidRequest,
            Some("k_exceeds_dimension"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
                     input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                     n_clusters => 1, unknown => true)"
            ),
            ErrorCode::InvalidRequest,
            Some("unknown_argument"),
        ),
        (
            format!(
                "WITH empty_input AS (SELECT CAST(1 AS BIGINT) AS item_id, CAST(1 AS FLOAT) AS d0, CAST(2 AS FLOAT) AS d1 LIMIT 0) \
                 SELECT * FROM {CUVS_KMEANS_FUNCTION}(\
                     input => (SELECT item_id AS id, d0, d1 FROM empty_input), \
                     n_clusters => 1)"
            ),
            ErrorCode::InvalidRequest,
            Some("empty_input"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_PCA_FUNCTION}(\
                     input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
                     n_components => 3)"
            ),
            ErrorCode::InvalidRequest,
            Some("component_count_exceeds_dimension"),
        ),
        (
            format!(
                "SELECT * FROM {CUVS_PCA_FUNCTION}(\
                     input => (SELECT query_id AS id, d0 FROM one_dim_queries), \
                     n_components => 1)"
            ),
            ErrorCode::InvalidRequest,
            Some("insufficient_feature_dimension"),
        ),
    ];
    for (sql, expected_code, expected_reason) in cases {
        let error = ctx.state().create_logical_plan(&sql).await.expect_err(&sql);
        let typed = error.find_algeon_error().unwrap_or_else(|| panic!("missing typed error for {sql}: {error:?}"));
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
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Execution("ExecuteCountingExec expects one child".to_string()));
        }
        Ok(Arc::new(Self { input: children.remove(0), execute_count: Arc::clone(&self.execute_count) }))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        self.execute_count.fetch_add(1, Ordering::SeqCst);
        self.input.execute(partition, context)
    }
}

#[cfg(feature = "cuvs")]
#[tokio::test]
async fn bounded_execution_defers_child_execution_until_the_stream_is_polled() {
    let ctx =
        fixtures::cuvs_context_with_native_execution_config(AlgeonNativeExecutionConfig::bounded(1024, 512, 1, 3, 1));
    let sql = format!(
        "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'cosine')"
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
    let _stream = rebound.execute(0, Arc::new(TaskContext::default())).expect("create bounded cuVS result stream");
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "creating the result stream must not consume its inputs");
}

#[cfg(not(feature = "cuvs"))]
#[tokio::test]
async fn execution_rejects_a_disabled_cuvs_feature_before_any_child_executes() {
    let ctx =
        fixtures::cuvs_context_with_native_execution_config(AlgeonNativeExecutionConfig::bounded(1024, 512, 1, 3, 1));
    for (function, args) in [
        (
            CUVS_BRUTE_FORCE_KNN_FUNCTION,
            "dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), k => 1, metric => 'cosine'",
        ),
        (
            CUVS_IVF_FLAT_FUNCTION,
            "dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), k => 1, metric => 'l2_expanded', n_lists => 1, n_probes => 1",
        ),
        (
            CUVS_IVF_PQ_FUNCTION,
            "dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), k => 1, metric => 'l2_expanded', n_lists => 1, n_probes => 1, pq_dim => 2",
        ),
        (
            CUVS_CAGRA_FUNCTION,
            "dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), k => 1, metric => 'l2_expanded', graph_degree => 2, intermediate_graph_degree => 3",
        ),
        (
            CUVS_PAIRWISE_DISTANCE_FUNCTION,
            "x => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), y => (SELECT query_id AS id, d0, d1 FROM query_vectors), metric => 'l2_expanded'",
        ),
        (CUVS_TOP_K_FUNCTION, "input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), k => 1, select => 'max'"),
        (CUVS_KMEANS_FUNCTION, "input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), n_clusters => 1"),
        (CUVS_PCA_FUNCTION, "input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), n_components => 1"),
    ] {
        let sql = format!("SELECT * FROM {function}({args})");
        let logical = logical_plan(&ctx, &sql).await;
        let physical = ctx.state().create_physical_plan(&logical).await.expect("cuVS physical planning");
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
            Ok(_) => panic!("disabled cuVS feature must reject"),
            Err(error) => error,
        };
        let typed = error.find_algeon_error().expect("structured cuVS feature-disabled error");
        assert_eq!(typed.code(), ErrorCode::UnsupportedOperator);
        assert_eq!(typed.facts().get("reason"), Some("required_feature_disabled"));
        assert_eq!(typed.facts().get("required_feature"), Some("cuvs"));
        assert_eq!(typed.facts().get("host_fallback"), Some("disabled"));
        assert_eq!(execute_count.load(Ordering::SeqCst), 0, "feature rejection must precede child execution");
    }
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn cugraph_reaches_cuvs_as_a_direct_gpu_resident_producer() {
    let ctx = fixtures::cuvs_cugraph_context();
    let sql = format!(
        "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             dataset => (SELECT vertex AS id, x, y FROM cugraph_force_atlas2(edges => (SELECT src, dst FROM edges))), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'cosine')"
    );
    let plan = logical_plan(&ctx, &sql).await;
    let node = find_cuvs_node(&plan).expect("cuVS logical extension");
    let dataset = node.inputs()[0].display_indent().to_string();
    assert!(dataset.contains("CugraphLogicalNode: algorithm=cugraph_force_atlas2"), "{dataset}");

    let physical = ctx.state().create_physical_plan(&plan).await.expect("combined cuGraph/cuVS physical plan");
    let cuvs = find_cuvs_exec(physical.as_ref()).expect("CuvsAlgoExec");
    let dataset = cuvs.children()[0];
    assert_eq!(dataset.name(), "ProjectionExec", "the canonical id projection must remain device executable");
    let cugraph = dataset.children()[0];
    assert_eq!(cugraph.name(), "CugraphAlgoExec", "the projection must directly consume the cuGraph result");
    assert!(
        cugraph.downcast_ref::<CugraphAlgoExec>().is_some(),
        "the projected child must retain the typed GPU-resident producer"
    );
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn cugraph_join_filter_projection_builds_a_lazy_bounded_cuvs_stream() {
    let ctx = fixtures::cuvs_cugraph_native_context();
    let sql = format!(
        "WITH candidates AS (\
             SELECT e.item_id, e.d0, e.d1 \
             FROM cugraph_louvain(edges => (SELECT src, dst FROM edges)) AS g \
             JOIN dataset_vectors AS e ON e.item_id = g.vertex \
             WHERE e.d0 >= CAST(0 AS REAL)\
         ) \
         SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             dataset => (SELECT item_id AS id, d0, d1 FROM candidates), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'cosine')"
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
    let _stream = rebound.execute(0, Arc::new(TaskContext::default())).expect("create bounded cuVS result stream");
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "creating the result stream must not consume its inputs");
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

fn sql_literal(json: &str) -> String {
    json.replace('\'', "''")
}
