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

#[tokio::test]
async fn cugraph_sql_registers_only_execution_udtfs_and_the_common_catalog() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
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
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    let batches = ctx
        .sql(
            "SELECT function_name, provider, domain, available, relation_roles_json, signature, summary \
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
        assert_eq!(string_value(batch, 2, row), "graph");
        assert!(bool_value(batch, 3, row));
        assert_eq!(json_value(string_value(batch, 4, row))[0]["name"], "edges");
        assert!(string_value(batch, 5, row).starts_with(expected_name));
        assert!(!string_value(batch, 6, row).trim().is_empty());
    }
}

#[tokio::test]
async fn cugraph_descriptions_retain_graph_schemas_and_use_common_validation_api() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    let batches = ctx
        .sql("SELECT function_name, provider, domain, relation_roles_json, argument_descriptions_json, \
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
    assert_eq!(string_value(batch, 2, 0), "graph");
    assert_eq!(json_value(string_value(batch, 3, 0))[0]["name"], "edges");
    let arguments = json_value(string_value(batch, 4, 0));
    assert!(json_array_item_by_name(&arguments, "source_vertex_or_vertices").is_object());
    let options = json_value(string_value(batch, 5, 0));
    assert_eq!(options["graph_options"]["directed"]["default"], true);
    let schemas = json_value(string_value(batch, 6, 0));
    assert_eq!(schemas["call_specific_authority"], "gpu_validate_call");
    assert!(schemas["field_descriptions"].as_array().unwrap().iter().any(|field| field["name"] == "vertex"));
    let request_schema = json_value(string_value(batch, 7, 0));
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
    let examples = json_value(string_value(batch, 8, 0));
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
    let limitations = json_value(string_value(batch, 9, 0));
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
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
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
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
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
    let details = json_value(string_value(&batch, 8, 0));
    assert_eq!(details["edges"]["table"], "target_edges");
    assert_eq!(details["edges"]["src_dtype"], "Int64");
    assert_eq!(details["edges"]["dst_dtype"], "Int64");
    assert_eq!(details["edges"]["weight_dtype"], "Float64");
}

#[tokio::test]
async fn cugraph_validation_requires_flat_common_options() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
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
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
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
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
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
