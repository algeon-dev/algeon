use super::{support::*, *};

fn gpu_call(edges: &str, options: serde_json::Value) -> String {
    serde_json::json!({
        "schema_version": 1,
        "relations": { "edges": { "table": edges } },
        "options": options,
    })
    .to_string()
}

async fn validate(ctx: &SessionContext, function: &str, edges: &str, options: serde_json::Value) -> RecordBatch {
    let call = gpu_call(edges, options).replace('\'', "''");
    let sql = format!(
        "SELECT valid, error_code, message, function_name, relations_resolved, normalized_call_json, \
         output_schema_json, would_execute_gpu, details_json \
         FROM gpu_validate_call('{function}', '{call}')"
    );
    single_batch(&ctx.sql(&sql).await.unwrap().collect().await.unwrap()).clone()
}

fn assert_validation_schema_matches(batch: &RecordBatch, actual: &Schema) {
    let reported = json_value(string_value(batch, 6, 0));
    let fields = reported["fields"].as_array().expect("validation output schema must contain fields");
    assert_eq!(fields.len(), actual.fields().len());
    for (reported, actual) in fields.iter().zip(actual.fields()) {
        assert_eq!(reported["name"].as_str(), Some(actual.name().as_str()));
        let data_type = format!("{:?}", actual.data_type());
        assert_eq!(reported["data_type"].as_str(), Some(data_type.as_str()));
        assert_eq!(reported["nullable"].as_bool(), Some(actual.is_nullable()));
    }
}

#[tokio::test]
async fn cugraph_sql_registers_only_execution_udtfs_and_the_common_catalog() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let functions = state.table_functions();

    for definition in CUGRAPH_OPERATION_REGISTRY {
        assert!(functions.contains_key(definition.sql_function_name));
    }
    for name in ["gpu_list_functions", "gpu_describe_function", "gpu_validate_call"] {
        assert!(functions.contains_key(name), "common catalog UDTF {name} must be installed");
    }
    let retired_metadata = [
        concat!("cugraph_", "list_algorithms"),
        concat!("cugraph_", "describe_algorithm"),
        concat!("cugraph_", "validate_call"),
    ];
    for name in retired_metadata {
        assert!(!functions.contains_key(name), "retired cuGraph metadata UDTF {name} must not be installed");
    }

    let registered_cugraph_udtfs: std::collections::BTreeSet<&str> =
        functions.keys().filter_map(|name| name.strip_prefix("cugraph_").map(|_| name.as_str())).collect();
    let registry_function_names: std::collections::BTreeSet<&str> =
        CUGRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.sql_function_name).collect();
    assert_eq!(registered_cugraph_udtfs, registry_function_names);
}

#[tokio::test]
async fn cugraph_registry_projects_into_the_common_catalog() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    let batches = ctx
        .sql(
            "SELECT function_name, provider, available, relation_roles_json, signature, summary \
              FROM gpu_list_functions() WHERE provider = 'cugraph' ORDER BY function_name",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    let mut expected = CUGRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.sql_function_name).collect::<Vec<_>>();
    expected.sort_unstable();
    assert_eq!(batch.num_rows(), expected.len());
    for (row, expected_name) in expected.into_iter().enumerate() {
        assert_eq!(string_value(batch, 0, row), expected_name);
        assert_eq!(string_value(batch, 1, row), "cugraph");
        assert!(bool_value(batch, 2, row));
        assert_eq!(json_value(string_value(batch, 3, row))[0]["name"], "edges");
        assert!(string_value(batch, 4, row).starts_with(expected_name));
        assert!(!string_value(batch, 5, row).trim().is_empty());
    }
}

#[tokio::test]
async fn cugraph_descriptions_retain_graph_schemas_and_use_common_validation_api() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    let batches = ctx
        .sql("SELECT function_name, provider, relation_roles_json, argument_descriptions_json, \
              options_schema_json, result_schemas_json, validation_request_schema_json, examples_json, limitations_json \
              FROM gpu_describe_function('cugraph_bfs')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(string_value(batch, 0, 0), "cugraph_bfs");
    assert_eq!(string_value(batch, 1, 0), "cugraph");
    assert_eq!(json_value(string_value(batch, 2, 0))[0]["name"], "edges");
    let arguments = json_value(string_value(batch, 3, 0));
    assert!(json_array_item_by_name(&arguments, "source_vertex_or_vertices").is_object());
    let options = json_value(string_value(batch, 4, 0));
    assert_eq!(options["graph_options"]["directed"]["default"], true);
    let schemas = json_value(string_value(batch, 5, 0));
    assert_eq!(schemas["call_specific_authority"], "gpu_validate_call");
    assert!(schemas["field_descriptions"].as_array().unwrap().iter().any(|field| field["name"] == "vertex"));
    let request_schema = json_value(string_value(batch, 6, 0));
    assert_eq!(request_schema["properties"]["relations"]["required"], serde_json::json!(["edges"]));
    assert_eq!(request_schema["properties"]["schema_version"]["const"], 1);
    assert!(
        request_schema["description"]
            .as_str()
            .is_some_and(|description| description.contains("no version 0 reader or version negotiation"))
    );
    assert!(
        request_schema["properties"]["schema_version"]["description"]
            .as_str()
            .is_some_and(|description| description.contains("rejected before family dispatch"))
    );
    assert!(
        request_schema["properties"]["options"]["description"]
            .as_str()
            .is_some_and(|description| description.contains("Flat cuGraph option object"))
    );
    assert!(request_schema["properties"]["options"].get("x_cugraph_options_schema").is_none());
    let examples = json_value(string_value(batch, 7, 0));
    assert_eq!(
        examples[0]["call_json"],
        serde_json::json!({
            "schema_version": 1,
            "relations": { "edges": { "table": "target_edges" } },
            "options": {
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "depth_limit": 4,
            },
        })
    );
    assert!(examples[0]["call_json"]["options"].get("options").is_none());
    assert!(examples[0]["execution_sql"].as_str().unwrap().starts_with("SELECT * FROM cugraph_bfs"));
    let limitations = json_value(string_value(batch, 8, 0));
    assert!(
        limitations
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item.as_str().is_some_and(|text| text.contains("does not scan")))
    );

    let error = ctx.sql("SELECT * FROM gpu_describe_function('bfs')").await.unwrap_err().to_string();
    assert!(error.contains("unknown GPU function 'bfs'"));
}

#[tokio::test]
async fn cugraph_examples_use_versioned_common_envelopes_with_flat_options() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);

    for definition in CUGRAPH_OPERATION_REGISTRY {
        let batches = ctx
            .sql(&format!("SELECT examples_json FROM gpu_describe_function('{}')", definition.sql_function_name))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let examples = json_value(string_value(single_batch(&batches), 0, 0));
        assert!(!examples.as_array().unwrap().is_empty(), "{} must have an example", definition.sql_function_name);
        for example in examples.as_array().unwrap() {
            assert_eq!(example["function_name"], definition.sql_function_name);
            let call = &example["call_json"];
            assert_eq!(call["schema_version"], 1);
            assert_eq!(call["relations"]["edges"]["table"], "target_edges");
            assert!(call["options"].is_object());
            assert!(call["options"].get("options").is_none());
        }
    }
}

#[tokio::test]
async fn cugraph_validation_uses_common_envelope_and_preserves_graph_facts() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "raw_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    ctx.sql("CREATE VIEW target_edges AS SELECT src, dst, weight FROM raw_edges")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let batch = validate(
        &ctx,
        "cugraph_bfs",
        "target_edges",
        serde_json::json!({
            "source_vertex": 0,
            "src_col": "src",
            "dst_col": "dst",
            "weight_col": "weight",
            "depth_limit": 2,
            "output_mode": "normalized",
        }),
    )
    .await;
    assert!(bool_value(&batch, 0, 0));
    assert!(is_null_value(&batch, 1, 0));
    assert_eq!(string_value(&batch, 3, 0), "cugraph_bfs");
    assert!(bool_value(&batch, 4, 0));
    assert!(string_value(&batch, 5, 0).contains(r#""schema_version":1"#));
    assert!(string_value(&batch, 6, 0).contains(r#""reachable""#));
    assert!(bool_value(&batch, 7, 0));
    let bfs = ctx
        .sql(
            "SELECT * FROM cugraph_bfs('target_edges', 0, 'src', 'dst', 'weight', \
             '{\"depth_limit\":2,\"output_mode\":\"normalized\"}')",
        )
        .await
        .unwrap();
    assert_validation_schema_matches(&batch, bfs.schema().inner());
    let details = json_value(string_value(&batch, 8, 0));
    assert_eq!(details["edges"]["table"], "target_edges");
    assert_eq!(details["edges"]["src_dtype"], "Int64");
    assert_eq!(details["edges"]["dst_dtype"], "Int64");
    assert_eq!(details["edges"]["weight_dtype"], "Float64");
}

#[tokio::test]
async fn cugraph_validation_output_schema_matches_pagerank_provider_for_each_vertex_domain() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "integer_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_string_pagerank_parity_edges(&ctx, "string_edges", "unused_numeric_edges");

    for (edges, expected_vertex_type) in [("integer_edges", DataType::Int64), ("string_edges", DataType::Utf8)] {
        let batch =
            validate(&ctx, "cugraph_pagerank", edges, serde_json::json!({"src_col":"src","dst_col":"dst"})).await;
        assert!(bool_value(&batch, 0, 0), "{}", string_value(&batch, 2, 0));

        let df = ctx.sql(&format!("SELECT * FROM cugraph_pagerank('{edges}', 'src', 'dst')")).await.unwrap();
        assert_eq!(df.schema().field(0).data_type(), &expected_vertex_type, "{edges} PageRank vertex type");
        assert_validation_schema_matches(&batch, df.schema().inner());
    }
}

#[tokio::test]
async fn cugraph_validation_requires_flat_common_options() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "raw_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let batch = validate(
        &ctx,
        "cugraph_bfs",
        "raw_edges",
        serde_json::json!({
            "source_vertex": 0,
            "src_col": "src",
            "dst_col": "dst",
            "options": { "depth_limit": 2 },
        }),
    )
    .await;

    assert!(!bool_value(&batch, 0, 0));
    assert_eq!(string_value(&batch, 1, 0), "invalid_argument");
    assert!(string_value(&batch, 2, 0).contains("must be flat"));
}

#[tokio::test]
async fn cugraph_validation_keeps_relation_aware_errors_structured() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int32, DataType::Int32, DataType::Float64);
    register_vertex_pair_memtable(&ctx, "candidate_pairs", DataType::Int32, DataType::Int32, false);
    register_vertex_pair_memtable(&ctx, "bad_pairs", DataType::Int32, DataType::Int64, false);

    let valid = validate(
        &ctx,
        "cugraph_jaccard",
        "edges",
        serde_json::json!({
            "src_col": "src", "dst_col": "dst",
            "vertex_pairs_table": "candidate_pairs",
            "first_vertex_col": "first", "second_vertex_col": "second"
        }),
    )
    .await;
    assert!(bool_value(&valid, 0, 0), "{}", string_value(&valid, 2, 0));
    assert!(string_value(&valid, 6, 0).contains(r#""similarity""#));

    let invalid = validate(
        &ctx,
        "cugraph_jaccard",
        "edges",
        serde_json::json!({
            "src_col": "src", "dst_col": "dst",
            "vertex_pairs_table": "bad_pairs",
            "first_vertex_col": "first", "second_vertex_col": "second"
        }),
    )
    .await;
    assert!(!bool_value(&invalid, 0, 0));
    assert_eq!(string_value(&invalid, 1, 0), "vertex_dtype_mismatch");
    let details = json_value(string_value(&invalid, 8, 0));
    assert_eq!(details["edges"]["src_dtype"], "Int32");

    let missing_edges = validate(&ctx, "cugraph_pagerank", "missing", serde_json::json!({})).await;
    assert!(!bool_value(&missing_edges, 0, 0));
    assert_eq!(string_value(&missing_edges, 1, 0), "table_not_found");
    assert!(!bool_value(&missing_edges, 4, 0));
}

#[tokio::test]
async fn cugraph_validation_rejects_invalid_common_envelopes_before_graph_parsing() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    for (call, code) in [
        (r#"{"schema_version":1,"relations":{},"options":{}}"#, "missing_relation"),
        (
            r#"{"schema_version":1,"relations":{"edges":{"table":"edges"},"other":{"table":"other"}},"options":{}}"#,
            "unknown_relation_role",
        ),
        (r#"{"schema_version":2,"relations":{"edges":{"table":"edges"}},"options":{}}"#, "unsupported_schema_version"),
        (
            r#"{"schema_version":1,"relations":{"edges":{"table":"edges"}},"options":{},"extra":true}"#,
            "invalid_call_json",
        ),
    ] {
        let escaped = call.replace('\'', "''");
        let batches = ctx
            .sql(&format!("SELECT valid, error_code FROM gpu_validate_call('cugraph_pagerank', '{escaped}')"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batch = single_batch(&batches);
        assert!(!bool_value(batch, 0, 0));
        assert_eq!(string_value(batch, 1, 0), code);
    }

    let unknown = ctx
        .sql("SELECT valid, error_code, function_name FROM gpu_validate_call('bfs', '{\"schema_version\":1,\"relations\":{},\"options\":{}}')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let unknown = single_batch(&unknown);
    assert!(!bool_value(unknown, 0, 0));
    assert_eq!(string_value(unknown, 1, 0), "unknown_function");
    assert_eq!(string_value(unknown, 2, 0), "bfs");
}

#[tokio::test]
async fn cugraph_vertex_id_descriptors_match_registry_string_contracts() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);

    for definition in CUGRAPH_OPERATION_REGISTRY {
        let batches = ctx
            .sql(&format!(
                "SELECT relation_roles_json, result_schemas_json FROM gpu_describe_function('{}')",
                definition.sql_function_name
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batch = single_batch(&batches);
        let relations = json_value(string_value(batch, 0, 0));
        let contract = &relations[0]["vertex_id_contract"];
        let string_capable = definition.supports_logical_string_vertices();
        assert_eq!(
            contract["logical_string_endpoint_types"],
            if string_capable { serde_json::json!(["Utf8", "LargeUtf8", "Utf8View"]) } else { serde_json::json!([]) },
            "{}",
            definition.sql_function_name
        );
        assert_eq!(contract["call_specific_authority"], "gpu_validate_call");
        assert_eq!(contract["edge_id_supported_for_logical_string"], false);
        assert_eq!(contract["logical_string_output_type"].is_string(), string_capable);
        assert_eq!(contract["native_logical_mapping_type"].as_str(), string_capable.then_some("Int64"));

        let result_schemas = json_value(string_value(batch, 1, 0));
        let has_logical_output = result_schemas["schemas"]
            .as_object()
            .unwrap()
            .values()
            .flat_map(|schema| schema["fields"].as_array().unwrap())
            .any(|field| field["data_type"] == "Int64|Utf8");
        assert_eq!(has_logical_output, string_capable, "{}", definition.sql_function_name);
    }
}

#[tokio::test]
async fn cugraph_string_selector_descriptors_publish_their_logical_types() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);

    for function_name in ["cugraph_sssp", "cugraph_betweenness_centrality", "cugraph_edge_betweenness_centrality"] {
        let batches = ctx
            .sql(&format!(
                "SELECT argument_descriptions_json, options_schema_json FROM gpu_describe_function('{function_name}')"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batch = single_batch(&batches);
        let arguments = json_value(string_value(batch, 0, 0));
        let options = json_value(string_value(batch, 1, 0));
        if function_name == "cugraph_sssp" {
            let source =
                arguments.as_array().unwrap().iter().find(|argument| argument["name"] == "source_vertex").unwrap();
            assert_eq!(source["type"], "Int64|Utf8");
        } else {
            assert_eq!(options["algorithm_options"]["seeds"]["type"], "List<Int64>|List<Utf8>|null");
        }
    }
}

#[tokio::test]
async fn cugraph_m1_algorithms_validate_all_string_endpoint_family_combinations() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    let endpoint_types = [
        ("utf8_large", DataType::Utf8, DataType::LargeUtf8),
        ("large_view", DataType::LargeUtf8, DataType::Utf8View),
        ("view_utf8", DataType::Utf8View, DataType::Utf8),
    ];
    for (table, src_type, dst_type) in &endpoint_types {
        register_edge_memtable(&ctx, table, src_type.clone(), dst_type.clone(), DataType::Float64);
    }

    let algorithms = [
        CugraphAlgorithm::PageRank,
        CugraphAlgorithm::Louvain,
        CugraphAlgorithm::EigenvectorCentrality,
        CugraphAlgorithm::KatzCentrality,
        CugraphAlgorithm::Hits,
        CugraphAlgorithm::DegreesAll,
        CugraphAlgorithm::InDegreesAll,
        CugraphAlgorithm::OutDegreesAll,
        CugraphAlgorithm::WeaklyConnectedComponents,
        CugraphAlgorithm::StronglyConnectedComponents,
        CugraphAlgorithm::TriangleCountAll,
        CugraphAlgorithm::CoreNumber,
        CugraphAlgorithm::KCore,
        CugraphAlgorithm::Leiden,
        CugraphAlgorithm::Ecg,
        CugraphAlgorithm::ForceAtlas2,
    ];
    for algorithm in algorithms {
        for (table, _, _) in &endpoint_types {
            let batch = validate(
                &ctx,
                algorithm.function_name(),
                table,
                serde_json::json!({"src_col": "src", "dst_col": "dst"}),
            )
            .await;
            assert!(
                bool_value(&batch, 0, 0),
                "{} rejected {table}: {}",
                algorithm.function_name(),
                string_value(&batch, 2, 0)
            );
            assert_eq!(json_value(string_value(&batch, 6, 0))["fields"][0]["data_type"], "Utf8");
        }
    }
}

#[tokio::test]
async fn cugraph_m2_relation_side_inputs_validate_and_canonicalize_string_outputs() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "string_edges", DataType::Utf8, DataType::LargeUtf8, DataType::Float64);
    register_personalization_memtable(&ctx, "personalization", DataType::Utf8View);
    register_vertex_pair_memtable(&ctx, "candidate_pairs", DataType::Utf8View, DataType::LargeUtf8, false);

    let personalized = validate(
        &ctx,
        "cugraph_personalized_pagerank",
        "string_edges",
        serde_json::json!({
            "src_col": "src",
            "dst_col": "dst",
            "weight_col": "weight",
            "personalization_table": "personalization",
            "personalization_vertex_col": "vertex",
            "personalization_value_col": "value"
        }),
    )
    .await;
    assert!(bool_value(&personalized, 0, 0), "{}", string_value(&personalized, 2, 0));
    assert_eq!(json_value(string_value(&personalized, 6, 0))["fields"][0]["data_type"], "Utf8");

    for function in ["cugraph_jaccard", "cugraph_sorensen", "cugraph_overlap", "cugraph_cosine"] {
        let similarity = validate(
            &ctx,
            function,
            "string_edges",
            serde_json::json!({
                "src_col": "src",
                "dst_col": "dst",
                "weight_col": "weight",
                "vertex_pairs_table": "candidate_pairs",
                "first_vertex_col": "first",
                "second_vertex_col": "second"
            }),
        )
        .await;
        assert!(bool_value(&similarity, 0, 0), "{function}: {}", string_value(&similarity, 2, 0));
        let fields = json_value(string_value(&similarity, 6, 0));
        assert_eq!(fields["fields"][0]["data_type"], "Utf8");
        assert_eq!(fields["fields"][1]["data_type"], "Utf8");
    }
}

#[tokio::test]
async fn cugraph_m3_string_selectors_validate_with_concrete_utf8_output_schemas() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "string_edges", DataType::Utf8View, DataType::LargeUtf8, DataType::Float64);

    let cases = [
        (
            "cugraph_sssp",
            serde_json::json!({
                "src_col": "src",
                "dst_col": "dst",
                "weight_col": "weight",
                "source_vertex": "alice"
            }),
            vec!["vertex", "predecessor"],
        ),
        (
            "cugraph_betweenness_centrality",
            serde_json::json!({"src_col": "src", "dst_col": "dst", "seeds": ["alice", "carol"]}),
            vec!["vertex"],
        ),
        (
            "cugraph_edge_betweenness_centrality",
            serde_json::json!({"src_col": "src", "dst_col": "dst", "seeds": ["alice"]}),
            vec!["source", "destination"],
        ),
    ];
    for (function, options, identity_columns) in cases {
        let batch = validate(&ctx, function, "string_edges", options).await;
        assert!(bool_value(&batch, 0, 0), "{function}: {}", string_value(&batch, 2, 0));
        let schema = json_value(string_value(&batch, 6, 0));
        for identity in identity_columns {
            let field = schema["fields"]
                .as_array()
                .unwrap()
                .iter()
                .find(|field| field["name"] == identity)
                .unwrap_or_else(|| panic!("{function} missing {identity}"));
            assert_eq!(field["data_type"], "Utf8", "{function}.{identity}");
        }
    }

    let mismatch = validate(
        &ctx,
        "cugraph_sssp",
        "string_edges",
        serde_json::json!({
            "src_col": "src",
            "dst_col": "dst",
            "weight_col": "weight",
            "source_vertex": 1
        }),
    )
    .await;
    assert!(!bool_value(&mismatch, 0, 0));
    assert_eq!(string_value(&mismatch, 1, 0), "vertex_dtype_mismatch");
}

#[tokio::test]
async fn cugraph_reinstall_is_idempotent_and_conflicting_graph_defaults_are_rejected() {
    let builder = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("first cuGraph install succeeds")
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("identical reinstall is idempotent");

    let conflicting = CugraphSqlConfig {
        default_graph_options: CugraphGraphOptions { directed: false, ..Default::default() },
        ..Default::default()
    };
    let error = builder.try_with_cugraph_sql(conflicting).expect_err("conflicting graph defaults must be rejected");
    let nexus = typed_nexus_error(&error);
    assert_eq!(nexus.code(), ErrorCode::InvalidRequest);
    assert_eq!(nexus.operation(), "datafusion_nexus.gpu_functions.install");
    assert!(nexus.message().contains("conflicting GPU function family installation"));
}
