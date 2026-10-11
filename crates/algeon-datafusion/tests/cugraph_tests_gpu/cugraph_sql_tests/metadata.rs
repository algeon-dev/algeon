use algeon_datafusion::cugraph_sql::GpuIslandHostBoundaryConfig;

use super::{support::*, *};

fn gpu_call(edges: &str, options: serde_json::Value) -> String {
    gpu_call_with_relations(edges, serde_json::json!({}), options)
}

fn gpu_call_with_relations(edges: &str, side_relations: serde_json::Value, options: serde_json::Value) -> String {
    let mut relations = serde_json::Map::new();
    relations.insert("edges".to_string(), serde_json::json!({ "table": edges }));
    if let Some(side_relations) = side_relations.as_object() {
        relations.extend(side_relations.clone());
    }
    serde_json::json!({
        "relations": relations,
        "options": options,
    })
    .to_string()
}

async fn validate(ctx: &SessionContext, function: &str, edges: &str, options: serde_json::Value) -> RecordBatch {
    let call = gpu_call(edges, options).replace('\'', "''");
    validate_call(ctx, function, call).await
}

async fn validate_with_relations(
    ctx: &SessionContext,
    function: &str,
    edges: &str,
    side_relations: serde_json::Value,
    options: serde_json::Value,
) -> RecordBatch {
    let call = gpu_call_with_relations(edges, side_relations, options).replace('\'', "''");
    validate_call(ctx, function, call).await
}

async fn validate_call(ctx: &SessionContext, function: &str, call: String) -> RecordBatch {
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
        assert!(
            !functions.contains_key(definition.sql_function_name),
            "{} is planned by the relation planner, not registered as an execution UDTF",
            definition.sql_function_name
        );
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
    assert!(registered_cugraph_udtfs.is_empty());
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
    assert!(json_array_item_by_name(&arguments, "source_vertex").is_object());
    let options = json_value(string_value(batch, 4, 0));
    assert_eq!(options["properties"]["directed"]["default"], true);
    let schemas = json_value(string_value(batch, 5, 0));
    assert_eq!(schemas["call_specific_authority"], "gpu_validate_call");
    assert!(schemas["field_descriptions"].as_array().unwrap().iter().any(|field| field["name"] == "vertex"));
    let request_schema = json_value(string_value(batch, 6, 0));
    assert_eq!(request_schema["properties"]["relations"]["required"], serde_json::json!(["edges"]));
    assert_eq!(request_schema["required"], serde_json::json!(["relations", "options"]));
    assert!(request_schema.get("description").is_none());
    assert_eq!(request_schema["properties"].as_object().unwrap().len(), 2);
    assert!(
        request_schema["properties"]["options"]["description"]
            .as_str()
            .is_some_and(|description| description.contains("Named cuGraph value arguments"))
    );
    assert!(request_schema["properties"]["options"].get("x_cugraph_options_schema").is_none());
    let examples = json_value(string_value(batch, 7, 0));
    assert_eq!(
        examples[0]["call_json"],
        serde_json::json!({
            "relations": { "edges": { "table": "target_edges" } },
            "options": {
                "source_vertex": 123,
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
    assert!(
        error.contains(
            "unknown GPU function bfs; closest registered name: cugraph_bfs; gpu_list_functions() lists every installed function"
        ),
        "{error}"
    );
}

#[tokio::test]
async fn cugraph_examples_use_common_envelopes_with_flat_options() {
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
            assert_eq!(call.as_object().unwrap().len(), 2);
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
            "depth_limit": 2,
            "output_mode": "normalized",
        }),
    )
    .await;
    assert!(bool_value(&batch, 0, 0));
    assert!(is_null_value(&batch, 1, 0));
    assert_eq!(string_value(&batch, 3, 0), "cugraph_bfs");
    assert!(bool_value(&batch, 4, 0));
    let normalized = json_value(string_value(&batch, 5, 0));
    assert_eq!(normalized.as_object().unwrap().len(), 2);
    assert_eq!(normalized["relations"]["edges"]["table"], "target_edges");
    assert!(string_value(&batch, 6, 0).contains(r#""reachable""#));
    assert!(bool_value(&batch, 7, 0));
    let bfs = ctx
        .sql(
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT src, dst, weight FROM target_edges), source_vertex => 0,\
             depth_limit => 2, output_mode => 'normalized')",
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
        let batch = validate(&ctx, "cugraph_pagerank", edges, serde_json::json!({})).await;
        assert!(bool_value(&batch, 0, 0), "{}", string_value(&batch, 2, 0));

        let df =
            ctx.sql(&format!("SELECT * FROM cugraph_pagerank(edges => (SELECT src, dst FROM {edges}))")).await.unwrap();
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
            "options": { "depth_limit": 2 },
        }),
    )
    .await;

    assert!(!bool_value(&batch, 0, 0));
    assert_eq!(string_value(&batch, 1, 0), "unknown_option");
    assert!(string_value(&batch, 2, 0).contains("unknown option 'options'"));
}

#[tokio::test]
async fn cugraph_validation_rejects_sssp_source_vertices_list() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let batch = validate(&ctx, "cugraph_sssp", "edges", serde_json::json!({ "source_vertices": [1, 2] })).await;

    assert!(!bool_value(&batch, 0, 0));
    assert_eq!(string_value(&batch, 1, 0), "unknown_option");
    let message = string_value(&batch, 2, 0);
    assert!(message.contains("unknown option 'source_vertices'"), "{message}");
    assert!(message.contains("Did you mean 'source_vertex'?"), "{message}");
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

    let valid = validate_with_relations(
        &ctx,
        "cugraph_jaccard",
        "edges",
        serde_json::json!({"vertex_pairs": {"table": "candidate_pairs"}}),
        serde_json::json!({}),
    )
    .await;
    assert!(bool_value(&valid, 0, 0), "{}", string_value(&valid, 2, 0));
    assert!(string_value(&valid, 6, 0).contains(r#""similarity""#));

    let invalid = validate_with_relations(
        &ctx,
        "cugraph_jaccard",
        "edges",
        serde_json::json!({"vertex_pairs": {"table": "bad_pairs"}}),
        serde_json::json!({}),
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
async fn cugraph_validation_details_name_the_offending_argument_or_relation() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "vertex_only", "vertex", DataType::Int64, false);

    let alpha = validate(&ctx, "cugraph_pagerank", "edges", serde_json::json!({"alpha": 2.0})).await;
    assert!(!bool_value(&alpha, 0, 0));
    assert!(bool_value(&alpha, 4, 0), "relation resolution precedes value validation");
    let details = json_value(string_value(&alpha, 8, 0));
    assert_eq!(details["path"], "options.alpha");
    assert_eq!(details["argument"], "alpha");

    let directed =
        validate(&ctx, "cugraph_weakly_connected_components", "edges", serde_json::json!({"directed": true})).await;
    assert_eq!(string_value(&directed, 1, 0), "unsupported_option");
    assert_eq!(json_value(string_value(&directed, 8, 0))["path"], "options.directed");

    let personalization = validate_with_relations(
        &ctx,
        "cugraph_personalized_pagerank",
        "edges",
        serde_json::json!({"personalization": {"table": "vertex_only"}}),
        serde_json::json!({}),
    )
    .await;
    assert!(!bool_value(&personalization, 0, 0));
    assert!(string_value(&personalization, 2, 0).contains("'value'"));
    let details = json_value(string_value(&personalization, 8, 0));
    assert_eq!(details["path"], "relations.personalization");
    assert_eq!(details["relation_role"], "personalization");
    assert!(details["personalization"]["resolved"].as_bool().unwrap());
}

#[tokio::test]
async fn cugraph_force_atlas2_validation_resolves_every_optional_relation_role() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int64, DataType::Int64, DataType::Float64);
    for ddl in [
        "CREATE VIEW attributes AS SELECT CAST(1 AS BIGINT) AS vertex, CAST(0.5 AS FLOAT) AS radius",
        "CREATE VIEW positions AS SELECT CAST(1 AS BIGINT) AS vertex, CAST(0.0 AS FLOAT) AS x, CAST(1.0 AS FLOAT) AS y",
    ] {
        ctx.sql(ddl).await.unwrap().collect().await.unwrap();
    }

    let valid = validate_with_relations(
        &ctx,
        "cugraph_force_atlas2",
        "edges",
        serde_json::json!({"vertex_attributes": {"table": "attributes"}, "initial_positions": {"table": "positions"}}),
        serde_json::json!({"prevent_overlapping": true}),
    )
    .await;
    assert!(bool_value(&valid, 0, 0), "{}", string_value(&valid, 2, 0));
    assert!(bool_value(&valid, 4, 0));
    let details = json_value(string_value(&valid, 8, 0));
    for role in ["edges", "vertex_attributes", "initial_positions"] {
        assert_eq!(details[role]["resolved"], true, "{role}");
    }
    let normalized = json_value(string_value(&valid, 5, 0));
    assert_eq!(normalized["relations"]["initial_positions"]["table"], "positions");

    let missing_x = validate_with_relations(
        &ctx,
        "cugraph_force_atlas2",
        "edges",
        serde_json::json!({"initial_positions": {"table": "attributes"}}),
        serde_json::json!({}),
    )
    .await;
    assert!(!bool_value(&missing_x, 0, 0));
    assert_eq!(json_value(string_value(&missing_x, 8, 0))["path"], "relations.initial_positions");
}

#[tokio::test]
async fn cugraph_normalized_calls_revalidate_and_schemas_project_group_constraints() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int64, DataType::Int64, DataType::Float64);

    for function in ["cugraph_betweenness_centrality", "cugraph_edge_betweenness_centrality"] {
        let first = validate(&ctx, function, "edges", serde_json::json!({})).await;
        assert!(bool_value(&first, 0, 0), "{}", string_value(&first, 2, 0));
        let normalized = json_value(string_value(&first, 5, 0));
        assert!(normalized["options"]["k"].is_null(), "{function}: {normalized}");
        let second = validate(&ctx, function, "edges", normalized["options"].clone()).await;
        assert!(bool_value(&second, 0, 0), "{function}: {}", string_value(&second, 2, 0));
        assert_eq!(string_value(&second, 5, 0), string_value(&first, 5, 0));
    }

    let batches = ctx
        .sql(
            "SELECT options_schema_json, validation_request_schema_json \
             FROM gpu_describe_function('cugraph_betweenness_centrality')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    let options_schema = json_value(string_value(batch, 0, 0));
    assert_eq!(options_schema["properties"]["max_iterations"], serde_json::Value::Null);
    assert_eq!(options_schema["properties"]["exact_vertex_threshold"]["maximum"], u64::MAX);
    let exclusive = options_schema["allOf"].as_array().expect("betweenness selector group projects allOf");
    assert!(
        exclusive.iter().any(|constraint| {
            let pair = &constraint["not"]["allOf"];
            pair.is_array() && pair.to_string().contains("seeds") && pair.to_string().contains("\"k\"")
        }),
        "{exclusive:?}"
    );
    let request_schema = json_value(string_value(batch, 1, 0));
    assert_eq!(request_schema["properties"]["options"], options_schema);

    let bfs = ctx
        .sql("SELECT validation_request_schema_json FROM gpu_describe_function('cugraph_bfs')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let bfs = json_value(string_value(single_batch(&bfs), 0, 0));
    let selector = bfs["allOf"]
        .as_array()
        .expect("BFS source selector projects allOf")
        .iter()
        .find(|constraint| constraint["anyOf"].is_array())
        .expect("BFS requires one source selector");
    let members = selector["anyOf"].as_array().unwrap();
    assert_eq!(members.len(), 3);
    assert!(members.iter().any(|member| member["properties"]["relations"]["required"][0] == "sources"), "{members:?}");
}

#[tokio::test]
async fn cugraph_validation_rejects_invalid_common_envelopes_before_graph_parsing() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    for (call, code) in [
        (r#"{"relations":{},"options":{}}"#, "missing_relation"),
        (
            r#"{"relations":{"edges":{"table":"edges"},"other":{"table":"other"}},"options":{}}"#,
            "unknown_relation_role",
        ),
        (r#"{"relations":{"edges":{"table":"edges"}}}"#, "invalid_call_json"),
        (r#"{"relations":{"edges":{"table":"edges"}},"options":{},"extra":true}"#, "invalid_call_json"),
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
        .sql("SELECT valid, error_code, function_name FROM gpu_validate_call('bfs', '{\"relations\":{},\"options\":{}}')")
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
        let legacy_int32 = matches!(
            definition.algorithm,
            CugraphAlgorithm::MinimumSpanningTree | CugraphAlgorithm::SpectralModularityMaximization
        );
        assert_eq!(
            contract["numeric_endpoint_types"],
            serde_json::json!(["Int32", "Int64"]),
            "{}",
            definition.sql_function_name
        );
        assert_eq!(
            contract["logical_string_endpoint_types"],
            serde_json::json!(["Utf8", "LargeUtf8", "Utf8View"]),
            "{}",
            definition.sql_function_name
        );
        assert_eq!(contract["call_specific_authority"], "gpu_validate_call");
        assert_eq!(contract["edge_id_supported_for_logical_string"], false);
        assert_eq!(contract["logical_string_output_type"], "Utf8");
        assert_eq!(
            contract["native_logical_mapping_type"],
            if legacy_int32 { "Int32" } else { "Int64" },
            "{}",
            definition.sql_function_name
        );
        assert_eq!(
            contract["mapped_numeric_endpoint_types"],
            if legacy_int32 { serde_json::json!(["Int64"]) } else { serde_json::json!([]) },
            "{}",
            definition.sql_function_name
        );

        let result_schemas = json_value(string_value(batch, 1, 0));
        let has_logical_output = result_schemas["schemas"]
            .as_object()
            .unwrap()
            .values()
            .flat_map(|schema| schema["fields"].as_array().unwrap())
            .any(|field| field["data_type"] == "Int64|Utf8");
        assert!(has_logical_output, "{}", definition.sql_function_name);
    }
}

#[tokio::test]
async fn cugraph_descriptor_weight_data_types_match_validation() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    let weights = [
        ("int32_weights", DataType::Int32),
        ("int64_weights", DataType::Int64),
        ("float32_weights", DataType::Float32),
        ("float64_weights", DataType::Float64),
    ];
    for (table, weight_type) in &weights {
        register_edge_memtable(&ctx, table, DataType::Int32, DataType::Int32, weight_type.clone());
    }

    for (function, options) in [
        ("cugraph_pagerank", serde_json::json!({})),
        ("cugraph_sssp", serde_json::json!({"source_vertex": 0})),
        ("cugraph_minimum_spanning_tree", serde_json::json!({})),
    ] {
        let batches = ctx
            .sql(&format!("SELECT relation_roles_json FROM gpu_describe_function('{function}')"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let relations = json_value(string_value(single_batch(&batches), 0, 0));
        let weight = relations[0]["fields"].as_array().unwrap().iter().find(|field| field["name"] == "weight").unwrap();
        let advertised = weight["data_types"].as_array().unwrap();

        for (table, weight_type) in &weights {
            let batch = validate(&ctx, function, table, options.clone()).await;
            let accepted = bool_value(&batch, 0, 0);
            assert_eq!(
                accepted,
                advertised.contains(&serde_json::json!(format!("{weight_type:?}"))),
                "{function} with {weight_type:?} weights: {}",
                string_value(&batch, 2, 0)
            );
            if !accepted {
                assert_eq!(string_value(&batch, 1, 0), "unsupported_weight_dtype");
                assert_eq!(json_value(string_value(&batch, 8, 0))["relation_role"], "edges");
            }
        }
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
            assert_eq!(source["schema"]["type"], serde_json::json!(["integer", "string"]));
        } else {
            let seeds = &options["properties"]["seeds"];
            assert_eq!(seeds["oneOf"][1]["type"], "null");
            assert_eq!(seeds["oneOf"][0]["oneOf"][0]["items"]["type"], "integer");
            assert_eq!(seeds["oneOf"][0]["oneOf"][1]["items"]["type"], "string");
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
            let batch = validate(&ctx, algorithm.function_name(), table, serde_json::json!({})).await;
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

    let personalized = validate_with_relations(
        &ctx,
        "cugraph_personalized_pagerank",
        "string_edges",
        serde_json::json!({"personalization": {"table": "personalization"}}),
        serde_json::json!({}),
    )
    .await;
    assert!(bool_value(&personalized, 0, 0), "{}", string_value(&personalized, 2, 0));
    assert_eq!(json_value(string_value(&personalized, 6, 0))["fields"][0]["data_type"], "Utf8");

    for function in ["cugraph_jaccard", "cugraph_sorensen", "cugraph_overlap", "cugraph_cosine"] {
        let similarity = validate_with_relations(
            &ctx,
            function,
            "string_edges",
            serde_json::json!({"vertex_pairs": {"table": "candidate_pairs"}}),
            serde_json::json!({}),
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
        ("cugraph_sssp", serde_json::json!({"source_vertex": "alice"}), vec!["vertex", "predecessor"]),
        ("cugraph_betweenness_centrality", serde_json::json!({"seeds": ["alice", "carol"]}), vec!["vertex"]),
        ("cugraph_edge_betweenness_centrality", serde_json::json!({"seeds": ["alice"]}), vec!["src", "dst"]),
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

    let mismatch = validate(&ctx, "cugraph_sssp", "string_edges", serde_json::json!({"source_vertex": 1})).await;
    assert!(!bool_value(&mismatch, 0, 0));
    assert_eq!(string_value(&mismatch, 1, 0), "vertex_dtype_mismatch");
}

#[tokio::test]
async fn cugraph_reinstall_is_idempotent_and_conflicting_graph_defaults_are_rejected() {
    for conflicting in [
        CugraphSqlConfig {
            default_graph_options: CugraphGraphOptions { directed: false, ..Default::default() },
            ..Default::default()
        },
        CugraphSqlConfig {
            host_boundary_config: GpuIslandHostBoundaryConfig::try_new(8_192, 4_096).unwrap(),
            ..Default::default()
        },
    ] {
        let builder = SessionStateBuilder::new_with_default_features()
            .try_with_cugraph_sql(CugraphSqlConfig::default())
            .expect("first cuGraph install succeeds")
            .try_with_cugraph_sql(CugraphSqlConfig::default())
            .expect("identical reinstall is idempotent");
        let error = builder.try_with_cugraph_sql(conflicting).expect_err("conflicting settings must be rejected");
        let algeon = typed_algeon_error(&error);
        assert_eq!(algeon.code(), ErrorCode::InvalidRequest);
        assert_eq!(algeon.operation(), "algeon_datafusion.gpu_functions.install");
        assert!(algeon.message().contains("conflicting GPU function family installation"));
    }
}
