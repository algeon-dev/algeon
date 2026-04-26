use super::{support::*, *};

#[tokio::test]
async fn cugraph_sql_planning_registers_udtfs_on_session_state_builder() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let functions = state.table_functions();

    for definition in CUGRAPH_OPERATION_REGISTRY {
        assert!(
            functions.contains_key(definition.sql_function_name),
            "SQL surface must register every cuGraph operation registry entry: {}",
            definition.sql_function_name,
        );
    }

    let registered_cugraph_udtfs: std::collections::BTreeSet<&str> =
        functions.keys().filter_map(|name| name.strip_prefix("cugraph_").map(|_| name.as_str())).collect();
    let mut registry_function_names: std::collections::BTreeSet<&str> =
        CUGRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.sql_function_name).collect();
    registry_function_names.extend([
        CUGRAPH_LIST_ALGORITHMS_FUNCTION,
        CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION,
        CUGRAPH_VALIDATE_CALL_FUNCTION,
    ]);
    assert_eq!(
        registered_cugraph_udtfs, registry_function_names,
        "every `cugraph_*` UDTF on the session must come from the algorithm registry or metadata surface",
    );
}

#[tokio::test]
async fn cugraph_metadata_list_algorithms_reports_algorithm_metadata() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let batches = ctx
        .sql(
            "SELECT function_name, source_vertex_policy, summary, weight_col_policy, \
                    output_schema_kind, signature, allowed_arg_counts_json \
             FROM cugraph_list_algorithms() \
             WHERE function_name IN ('cugraph_pagerank', 'cugraph_bfs', 'cugraph_minimum_spanning_tree', 'cugraph_sssp') \
             ORDER BY function_name",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);

    assert_eq!(string_value(batch, 0, 0), "cugraph_bfs");
    assert_eq!(string_value(batch, 1, 0), "required_scalar_or_list_or_relation");
    assert!(string_value(batch, 2, 0).contains("breadth-first"));
    assert_eq!(string_value(batch, 3, 0), "ignored");
    assert_eq!(string_value(batch, 4, 0), "default_mode");
    assert!(string_value(batch, 5, 0).contains("source_vertex"));
    assert_eq!(json_value(string_value(batch, 6, 0)), serde_json::json!([2, 4, 5, 6]));

    assert_eq!(string_value(batch, 0, 1), "cugraph_minimum_spanning_tree");
    assert_eq!(string_value(batch, 1, 1), "none");
    assert_eq!(string_value(batch, 3, 1), "required");
    assert_eq!(string_value(batch, 4, 1), "fixed");
    assert!(string_value(batch, 5, 1).contains("weight_col"));
    assert_eq!(json_value(string_value(batch, 6, 1)), serde_json::json!([4, 5]));

    assert_eq!(string_value(batch, 0, 2), "cugraph_pagerank");
    assert_eq!(string_value(batch, 1, 2), "none");
    assert_eq!(string_value(batch, 3, 2), "optional");
    assert_eq!(string_value(batch, 4, 2), "fixed");
    assert_eq!(json_value(string_value(batch, 6, 2)), serde_json::json!([1, 3, 4, 5]));

    assert_eq!(string_value(batch, 0, 3), "cugraph_sssp");
    assert_eq!(string_value(batch, 1, 3), "required_scalar");
    assert_eq!(string_value(batch, 3, 3), "optional");

    let force_atlas2 = ctx
        .sql(
            "SELECT algorithm \
             FROM cugraph_list_algorithms() \
             WHERE function_name = 'cugraph_force_atlas2'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(string_value(single_batch(&force_atlas2), 0, 0), "ForceAtlas2");

    let force_atlas2_describe = ctx
        .sql(
            "SELECT options_schema_json, limitations_json \
             FROM cugraph_describe_algorithm('cugraph_force_atlas2', 'verbose')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let force_atlas2_describe = single_batch(&force_atlas2_describe);
    let options_schema = json_value(string_value(force_atlas2_describe, 0, 0));
    assert_eq!(
        options_schema["algorithm_options"]["prevent_overlapping"]["allowed_values"],
        serde_json::json!([false]),
    );
    let limitations = json_value(string_value(force_atlas2_describe, 1, 0));
    assert!(
        limitations
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry.as_str().is_some_and(|text| text.contains("vertex_radius"))),
        "ForceAtlas2 metadata must disclose the current prevent_overlapping limitation",
    );

    let mst_describe = ctx
        .sql(
            "SELECT limitations_json \
             FROM cugraph_describe_algorithm('cugraph_minimum_spanning_tree', 'verbose')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let limitations = json_value(string_value(single_batch(&mst_describe), 0, 0));
    assert!(
        limitations
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry.as_str().is_some_and(|text| text.contains("requires Int32"))),
        "legacy MST metadata must disclose the Int32 vertex dtype limitation",
    );

    let spectral_describe = ctx
        .sql(
            "SELECT options_schema_json \
             FROM cugraph_describe_algorithm('cugraph_spectral_modularity_maximization', 'verbose')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let options_schema = json_value(string_value(single_batch(&spectral_describe), 0, 0));
    assert_eq!(options_schema["algorithm_options"]["evs_max_iterations"]["max"], i32::MAX);
    assert_eq!(options_schema["algorithm_options"]["k_means_max_iterations"]["max"], i32::MAX);
}

#[tokio::test]
async fn cugraph_louvain_metadata_discloses_weight_arg_is_accepted_but_not_consumed() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int32, DataType::Int32, DataType::Float64);

    let batches = ctx
        .sql(
            "SELECT weight_col_policy \
             FROM cugraph_list_algorithms() \
             WHERE function_name = 'cugraph_louvain'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(string_value(batch, 0, 0), "ignored");

    let describe = ctx
        .sql(
            "SELECT argument_descriptions_json, options_schema_json \
             FROM cugraph_describe_algorithm('cugraph_louvain', 'verbose')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let describe = single_batch(&describe);
    let argument_descriptions = json_value(string_value(describe, 0, 0));
    let weight_col = json_array_item_by_name(&argument_descriptions, "weight_col");
    assert_eq!(weight_col["semantic_effect"], "none for this algorithm");
    assert!(weight_col["note"].as_str().unwrap().contains("does not consume weights"));
    let options_schema = json_value(string_value(describe, 1, 0));
    let algorithm_options = &options_schema["algorithm_options"];
    assert_eq!(algorithm_options["max_level"]["default"], 100);
    assert_eq!(algorithm_options["max_level"]["min"], 1);
    assert_eq!(algorithm_options["threshold"]["default"], serde_json::json!(1.0e-7));
    assert_eq!(algorithm_options["threshold"]["min"], 0.0);
    assert_eq!(algorithm_options["resolution"]["default"], 1.0);

    let validate = ctx
        .sql(
            r#"SELECT valid, normalized_options_json
               FROM cugraph_validate_call(
                 'cugraph_louvain',
                 'edges',
                 '{"src_col":"src","dst_col":"dst","options":{"directed":false,"max_level":24,"threshold":0.00001,"resolution":2.5}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let validate = single_batch(&validate);
    assert!(bool_value(validate, 0, 0), "Louvain validate_call should accept all scalar options");
    let normalized_options = json_value(string_value(validate, 1, 0));
    assert_eq!(normalized_options["algorithm_options"]["max_level"], 24);
    assert_eq!(normalized_options["algorithm_options"]["threshold"], serde_json::json!(0.00001));
    assert_eq!(normalized_options["algorithm_options"]["resolution"], 2.5);
}

#[tokio::test]
async fn cugraph_metadata_describe_algorithm_returns_structured_rows() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let batches = ctx
        .sql(
            "SELECT function_name, source_vertex_policy, weight_col_policy, result_schemas_json, \
                    options_schema_json, argument_descriptions_json, result_field_descriptions_json, \
                    limitations_json, summary, primary_example_sql, examples_json \
             FROM cugraph_describe_algorithm('cugraph_bfs', 'verbose')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(string_value(batch, 0, 0), "cugraph_bfs");
    assert_eq!(string_value(batch, 1, 0), "required_scalar_or_list_or_relation");
    assert_eq!(string_value(batch, 2, 0), "ignored");

    let result_schemas = json_value(string_value(batch, 3, 0));
    let default_schema = result_schemas["default_schema"].as_str().unwrap();
    let output_schema = &result_schemas["schemas"][default_schema];
    let default_vertex = json_array_item_by_name(&output_schema["fields"], "vertex");
    assert_eq!(default_vertex["data_type"], "Int64|Utf8", "result_schemas_json raw schema is generic");

    let options_schema = json_value(string_value(batch, 4, 0));
    assert_eq!(
        options_schema["graph_options"]["directed"]["description"],
        "Whether graph construction treats edges as directed."
    );
    assert_eq!(options_schema["graph_options"]["directed"]["default"], true);
    assert_eq!(options_schema["type_aliases"]["vertex_domain"], "same logical vertex domain as src_col and dst_col");
    assert_eq!(
        options_schema["algorithm_options"]["depth_limit"]["description"],
        "Maximum BFS depth in hops; null means unbounded traversal."
    );
    assert!(
        options_schema["algorithm_options"]["output_mode"]["description"].as_str().unwrap().contains("result shape")
    );
    assert_eq!(options_schema["result_schema_authority"]["call_specific_authority"], CUGRAPH_VALIDATE_CALL_FUNCTION);
    assert!(
        options_schema["result_schema_authority"]["note"].as_str().unwrap().contains("result_schemas_json"),
        "BFS describe JSON should tell callers where the generic schemas live"
    );
    let target_table = &options_schema["algorithm_options"]["target_vertices_table"];
    assert_eq!(target_table["side_input"]["role"], "target_vertices");
    assert_eq!(target_table["side_input"]["column_option"], "target_vertex_col");
    assert!(target_table["side_input"]["expected_columns"].is_null());
    let include_edges = &options_schema["algorithm_options"]["include_edges_table"];
    assert_eq!(include_edges["side_input"]["role"], "include_edges");
    assert_eq!(
        include_edges["side_input"]["column_options"],
        serde_json::json!(["include_edge_src_col", "include_edge_dst_col", "include_edge_id_col"]),
    );
    assert!(include_edges["side_input"]["expected_columns"].is_null());
    assert_eq!(
        options_schema["algorithm_options"]["include_edge_ids_col"]["defaults_to"],
        "edge_id_col when omitted and include_edge_ids_table is set"
    );
    assert_eq!(options_schema["algorithm_options"]["include_edge_id_col"]["valid_when"], "edge_id_col is set");
    let source_table = &options_schema["algorithm_options"]["source_vertices_table"];
    assert_eq!(source_table["side_input"]["role"], "source_vertices");
    assert_eq!(source_table["side_input"]["column_option"], "source_vertex_col");
    assert_eq!(options_schema["algorithm_options"]["source_vertex_col"]["column_of"], "source_vertices_table");
    assert_eq!(options_schema["algorithm_options"]["source_vertex_col"]["expected_type_ref"], "vertex_domain");

    let argument_descriptions = json_value(string_value(batch, 5, 0));
    let source_vertex = json_array_item_by_name(&argument_descriptions, "source_vertex_or_vertices");
    assert_eq!(source_vertex["type"], "Int64|Utf8|List<Int64>|List<Utf8>|null-with-source_vertices_table");
    assert_eq!(source_vertex["json_key"], "source_vertex");
    assert!(source_vertex["selector_policy"].as_str().unwrap().contains("source_vertices_table"));
    let weight_col = json_array_item_by_name(&argument_descriptions, "weight_col");
    assert_eq!(weight_col["semantic_effect"], "none for traversal; BFS distance is a hop count");
    assert!(weight_col["note"].as_str().unwrap().contains("ignored by BFS traversal"));

    let result_fields = json_value(string_value(batch, 6, 0));
    let distance = json_array_item_by_name(&result_fields, "distance");
    assert_eq!(distance["description"], "Hop-count distance from the nearest selected BFS source vertex.");
    let predecessor = json_array_item_by_name(&result_fields, "predecessor");
    assert_eq!(
        predecessor["description"],
        "Previous vertex on the discovered BFS tree, null for selected source vertices or unreachable vertices depending on output mode."
    );
    let target_found = json_array_item_by_name(&result_fields, "target_found");
    assert!(target_found["description"].as_str().unwrap().contains("target vertex was reached"));
    let path_index = json_array_item_by_name(&result_fields, "path_index");
    assert!(path_index["description"].as_str().unwrap().contains("reconstructed"));

    let limitations = json_value(string_value(batch, 7, 0));
    assert_json_array_contains_string(
        &limitations,
        "source lists and source_vertices_table are multi-source BFS, not one BFS per source",
    );
    assert_json_array_contains_string(
        &limitations,
        "source lists and source_vertices_table must contain at most one source per connected component",
    );

    assert!(string_value(batch, 8, 0).contains("breadth-first"));
    assert!(string_value(batch, 9, 0).contains("SELECT * FROM cugraph_bfs"));

    let mode_schemas = &result_schemas["schemas"];
    let raw_vertex = json_array_item_by_name(&mode_schemas["raw"]["fields"], "vertex");
    assert_eq!(raw_vertex["data_type"], "Int64|Utf8");
    assert_eq!(&mode_schemas["raw"], output_schema);
    let path_index = json_array_item_by_name(&mode_schemas["path"]["fields"], "path_index");
    assert_eq!(path_index["data_type"], "Int64");

    let examples = json_value(string_value(batch, 10, 0));
    json_array_item_by_name(&examples, "multi_source_bfs_list");
    json_array_item_by_name(&examples, "multi_source_bfs_table");
    let path_output = json_array_item_by_name(&examples, "path_output");
    assert_eq!(path_output["function_name"], "cugraph_bfs");
    assert!(path_output["call_json"]["options"]["output_mode"].as_str().is_some());
    assert!(
        examples.as_array().unwrap().iter().all(|example| example["sql"].is_null()),
        "verbose examples should avoid duplicating SQL; use primary_example_sql for a copyable SQL form"
    );

    let error = ctx.sql("SELECT * FROM cugraph_describe_algorithm('cugraph_missing')").await.unwrap_err().to_string();
    assert!(error.contains("unknown cuGraph algorithm function 'cugraph_missing'"));

    let batches = ctx
        .sql("SELECT function_name FROM cugraph_describe_algorithm('pagerank')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(string_value(batch, 0, 0), "cugraph_pagerank");
}

#[tokio::test]
async fn cugraph_metadata_column_order_is_pinned() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let list_df = ctx.sql("SELECT * FROM cugraph_list_algorithms()").await.unwrap();
    assert_eq!(
        schema_field_names(list_df.schema().inner()),
        vec![
            "function_name",
            "algorithm",
            "source_vertex_policy",
            "weight_col_policy",
            "output_schema_kind",
            "signature",
            "allowed_arg_counts_json",
            "summary",
        ],
    );
    let list_batches = list_df.collect().await.unwrap();
    let list_batch = single_batch(&list_batches);
    assert!(list_batch.num_rows() > 0);
    assert_json_columns_parse_and_have_no_control_chars(list_batch);

    let describe_df = ctx.sql("SELECT * FROM cugraph_describe_algorithm('cugraph_bfs')").await.unwrap();
    assert_eq!(
        schema_field_names(describe_df.schema().inner()),
        vec![
            "function_name",
            "algorithm",
            "source_vertex_policy",
            "weight_col_policy",
            "signature",
            "allowed_arg_counts_json",
            "summary",
            "primary_example_sql",
        ],
    );
    let describe_batches = describe_df.collect().await.unwrap();
    let describe_batch = single_batch(&describe_batches);
    assert_json_columns_parse_and_have_no_control_chars(describe_batch);

    let verbose_describe_df =
        ctx.sql("SELECT * FROM cugraph_describe_algorithm('cugraph_bfs', 'verbose')").await.unwrap();
    assert_eq!(
        schema_field_names(verbose_describe_df.schema().inner()),
        vec![
            "function_name",
            "algorithm",
            "source_vertex_policy",
            "weight_col_policy",
            "output_schema_kind",
            "signature",
            "allowed_arg_counts_json",
            "summary",
            "primary_example_sql",
            "result_schemas_json",
            "options_schema_json",
            "argument_descriptions_json",
            "result_field_descriptions_json",
            "limitations_json",
            "examples_json",
        ],
    );
    let verbose_describe_batches = verbose_describe_df.collect().await.unwrap();
    let verbose_describe_batch = single_batch(&verbose_describe_batches);
    assert_json_columns_parse_and_have_no_control_chars(verbose_describe_batch);

    register_edge_memtable(&ctx, "pinned_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    let validate_df = ctx
        .sql(
            r#"SELECT * FROM cugraph_validate_call(
                 'cugraph_pagerank',
                 'pinned_edges',
                 '{"src_col":"src","dst_col":"dst","weight_col":"weight"}'
               )"#,
        )
        .await
        .unwrap();
    assert_eq!(
        schema_field_names(validate_df.schema().inner()),
        vec![
            "valid",
            "error_code",
            "message",
            "function_name",
            "table_resolved",
            "src_col_resolved",
            "dst_col_resolved",
            "weight_col_resolved",
            "src_dtype",
            "dst_dtype",
            "weight_dtype",
            "output_schema_json",
            "normalized_options_json",
            "would_execute_gpu",
        ],
    );
    let validate_batches = validate_df.collect().await.unwrap();
    let validate_batch = single_batch(&validate_batches);
    assert_json_columns_parse_and_have_no_control_chars(validate_batch);
}

#[tokio::test]
async fn cugraph_metadata_validate_call_checks_temp_view_schema_without_gpu_execution() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "raw_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    ctx.sql("CREATE VIEW target_edges AS SELECT src, dst, weight FROM raw_edges")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, table_resolved, src_dtype, dst_dtype, weight_dtype, would_execute_gpu, normalized_options_json
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'target_edges',
                 '{"source_vertex":0,"src_col":"src","dst_col":"dst","weight_col":"weight","options":{"depth_limit":2}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert!(bool_value(batch, 2, 0));
    assert_eq!(string_value(batch, 3, 0), "Int64");
    assert_eq!(string_value(batch, 4, 0), "Int64");
    assert_eq!(string_value(batch, 5, 0), "Float64");
    assert!(bool_value(batch, 6, 0));
    assert!(string_value(batch, 7, 0).contains(r#""depth_limit":2"#));

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, function_name
               FROM cugraph_validate_call(
                 'bfs',
                 'target_edges',
                 '{"source_vertex":0,"src_col":"src","dst_col":"dst"}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert_eq!(string_value(batch, 2, 0), "cugraph_bfs");
}

#[tokio::test]
async fn cugraph_metadata_validate_call_returns_stable_invalid_rows() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "bad_edges", DataType::Utf8, DataType::Int64, DataType::Float64);
    register_edge_memtable(&ctx, "mismatch_edges", DataType::Int32, DataType::Int64, DataType::Float64);
    register_edge_memtable(&ctx, "bad_weight_edges", DataType::Int64, DataType::Int64, DataType::Utf8);

    assert_validate_error(
        &ctx,
        "cugraph_missing",
        "bad_edges",
        r#"{"src_col":"src","dst_col":"dst"}"#,
        "unknown_algorithm",
    )
    .await;
    assert_validate_error(
        &ctx,
        "cugraph_pagerank",
        "missing_edges",
        r#"{"src_col":"src","dst_col":"dst"}"#,
        "table_not_found",
    )
    .await;
    assert_validate_error(
        &ctx,
        "cugraph_pagerank",
        "bad_edges",
        r#"{"src_col":"missing","dst_col":"dst"}"#,
        "column_not_found",
    )
    .await;
    let unsupported = assert_validate_error(
        &ctx,
        "cugraph_pagerank",
        "bad_edges",
        r#"{"src_col":"src","dst_col":"dst"}"#,
        "unsupported_vertex_dtype",
    )
    .await;
    assert_eq!(string_value(&unsupported, 2, 0), "Utf8");
    assert_validate_error(
        &ctx,
        "cugraph_pagerank",
        "mismatch_edges",
        r#"{"src_col":"src","dst_col":"dst"}"#,
        "vertex_dtype_mismatch",
    )
    .await;
    assert_validate_error(
        &ctx,
        "cugraph_pagerank",
        "bad_weight_edges",
        r#"{"src_col":"src","dst_col":"dst","weight_col":"weight"}"#,
        "unsupported_weight_dtype",
    )
    .await;
    assert_validate_error(&ctx, "cugraph_pagerank", "bad_edges", r#"{"src_col":"#, "invalid_call_json").await;
    assert_validate_error(
        &ctx,
        "cugraph_pagerank",
        "bad_edges",
        r#"{"src_col":"src","dst_col":"dst","options":{"bogus":1}}"#,
        "unsupported_option",
    )
    .await;
}

#[tokio::test]
async fn cugraph_metadata_validate_call_reports_unsupported_force_atlas2_overlap_prevention() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_personalization_memtable(&ctx, "ppr_seeds", DataType::Int64, DataType::Float64);

    let batch = validate_call_batch(
        &ctx,
        "cugraph_force_atlas2",
        r#"{"src_col":"src","dst_col":"dst","options":{"prevent_overlapping":true}}"#,
    )
    .await;
    assert!(!bool_value(&batch, 0, 0));
    assert_eq!(string_value(&batch, 1, 0), "unsupported_option");
    assert!(string_value(&batch, 2, 0).contains("vertex_radius side input"));
}

#[tokio::test]
async fn cugraph_metadata_validate_call_accepts_nullable_edge_schema_for_runtime_drop_policy() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_nullable_edge_memtable(&ctx, "nullable_edges", true, true, true);
    register_nullable_edge_id_memtable(&ctx, "nullable_edge_ids");

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, table_resolved, src_dtype, dst_dtype, weight_dtype, would_execute_gpu
               FROM cugraph_validate_call(
                 'cugraph_pagerank',
                 'nullable_edges',
                 '{"src_col":"src","dst_col":"dst","weight_col":"weight"}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert!(bool_value(batch, 2, 0));
    assert_eq!(string_value(batch, 3, 0), "Int64");
    assert_eq!(string_value(batch, 4, 0), "Int64");
    assert_eq!(string_value(batch, 5, 0), "Float64");
    assert!(bool_value(batch, 6, 0));

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, table_resolved, src_dtype, dst_dtype, weight_dtype, would_execute_gpu
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'nullable_edge_ids',
                 '{"source_vertex":0,"src_col":"src","dst_col":"dst","weight_col":"weight","options":{"edge_id_col":"edge_id"}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert!(bool_value(batch, 2, 0));
    assert_eq!(string_value(batch, 3, 0), "Int64");
    assert_eq!(string_value(batch, 4, 0), "Int64");
    assert_eq!(string_value(batch, 5, 0), "Float64");
    assert!(bool_value(batch, 6, 0));
}

#[test]
fn cugraph_sql_operation_registry_matches_native_engine_inventory() {
    use nexus_query_engine::plan::GRAPH_OPERATION_REGISTRY;

    let sql_algorithms: Vec<_> = CUGRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.native_algorithm).collect();
    let native_algorithms: Vec<_> = GRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.algorithm).collect();
    assert_eq!(
        sql_algorithms, native_algorithms,
        "SQL-side cuGraph registry must mirror the nexus-query-engine graph operation registry, in order",
    );

    for entry in CUGRAPH_OPERATION_REGISTRY {
        let native_name = entry.native_algorithm.definition().name;
        assert_eq!(
            entry.sql_function_name,
            format!("cugraph_{native_name}"),
            "SQL UDTF name must follow the cugraph_<native_name> convention",
        );
    }
}

#[tokio::test]
async fn cugraph_metadata_registry_matches_native_graph_descriptors() {
    use nexus_query_engine::plan::GRAPH_OPERATION_REGISTRY;

    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    let batches = ctx
        .sql(
            "SELECT function_name, source_vertex_policy, weight_col_policy, output_schema_kind, \
                    signature, allowed_arg_counts_json \
             FROM cugraph_list_algorithms() ORDER BY function_name",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);

    let mut expected = CUGRAPH_OPERATION_REGISTRY.iter().collect::<Vec<_>>();
    expected.sort_by_key(|entry| entry.sql_function_name);
    assert_eq!(batch.num_rows(), expected.len());
    for (row, entry) in expected.into_iter().enumerate() {
        let descriptor =
            GRAPH_OPERATION_REGISTRY.iter().find(|definition| definition.algorithm == entry.native_algorithm).unwrap();
        assert_eq!(string_value(batch, 0, row), entry.sql_function_name);
        let expected_source_policy = if entry.algorithm == CugraphAlgorithm::Bfs {
            "required_scalar_or_list_or_relation"
        } else if descriptor.source_vertex_requirement().requires_source_vertex() {
            "required_scalar"
        } else {
            "none"
        };
        assert_eq!(
            string_value(batch, 1, row),
            expected_source_policy,
            "metadata source policy must project native descriptor for {}",
            entry.sql_function_name,
        );
        assert_eq!(
            string_value(batch, 2, row),
            descriptor.weight_policy().as_str(),
            "metadata weight policy must project native descriptor for {}",
            entry.sql_function_name,
        );
        assert_eq!(
            string_value(batch, 3, row),
            descriptor.output_schema_kind().as_str(),
            "metadata output schema kind must project native descriptor for {}",
            entry.sql_function_name,
        );
        assert!(string_value(batch, 4, row).starts_with(entry.sql_function_name));
        let expected_arg_counts = serde_json::json!(entry.algorithm.allowed_arg_counts());
        assert_eq!(json_value(string_value(batch, 5, row)), expected_arg_counts);

        let describe_sql = format!(
            "SELECT result_schemas_json, options_schema_json, result_field_descriptions_json, \
                    examples_json, summary \
             FROM cugraph_describe_algorithm('{}', 'verbose')",
            entry.sql_function_name
        );
        let describe_batches = ctx.sql(&describe_sql).await.unwrap().collect().await.unwrap();
        let describe_batch = single_batch(&describe_batches);

        let result_schemas = json_value(string_value(describe_batch, 0, 0));
        let default_schema = result_schemas["default_schema"].as_str().unwrap();
        let output_schema = &result_schemas["schemas"][default_schema];
        let fields = output_schema["fields"].as_array().unwrap();
        assert_eq!(fields.len(), descriptor.base_result_schema().fields().len());
        if entry.algorithm == CugraphAlgorithm::Bfs {
            let vertex = json_array_item_by_name(&output_schema["fields"], "vertex");
            assert_eq!(vertex["data_type"], "Int64|Utf8");
            let predecessor = json_array_item_by_name(&output_schema["fields"], "predecessor");
            assert_eq!(predecessor["data_type"], "Int64|Utf8");
        } else {
            for (field_json, native_field) in fields.iter().zip(descriptor.base_result_schema().fields()) {
                assert_eq!(field_json["name"], native_field.name().as_str());
                assert_eq!(field_json["data_type"], test_data_type_label(native_field.data_type()));
                assert_eq!(field_json["nullable"], native_field.is_nullable());
            }
        }

        let result_fields = json_value(string_value(describe_batch, 2, 0));
        for field in result_fields.as_array().unwrap() {
            assert!(
                field["description"].as_str().is_some_and(|description| !description.is_empty()),
                "result field descriptions must be semantic for {}: {field}",
                entry.sql_function_name,
            );
        }

        let examples = json_value(string_value(describe_batch, 3, 0));
        assert!(
            !examples.as_array().unwrap().is_empty(),
            "every algorithm should have at least one metadata example: {}",
            entry.sql_function_name,
        );
        let summary = string_value(describe_batch, 4, 0);
        assert!(
            !summary.trim().is_empty() && !summary.contains("Run the named cuGraph algorithm"),
            "summary should carry algorithm-specific UDF knowledge for {}: {summary}",
            entry.sql_function_name,
        );

        let options_schema = json_value(string_value(describe_batch, 1, 0));
        let graph_options = &options_schema["graph_options"];
        let native_defaults = descriptor.default_graph_options(&Default::default());
        assert_eq!(graph_options["directed"]["default"], native_defaults.directed);
        assert_eq!(graph_options["renumber"]["default"], native_defaults.renumber);
        assert_eq!(graph_options["construction_policy"]["default"], native_defaults.policy.as_str());
    }
}

#[tokio::test]
async fn cugraph_validate_call_source_and_weight_policy_match_metadata() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int32, DataType::Int32, DataType::Float64);
    register_personalization_memtable(&ctx, "ppr_seeds", DataType::Int32, DataType::Float64);
    register_vertex_pair_memtable(&ctx, "candidate_pairs", DataType::Int32, DataType::Int32, false);

    for entry in CUGRAPH_OPERATION_REGISTRY {
        let function = entry.sql_function_name;
        if entry.native_algorithm.requires_source_vertex() {
            let batch = validate_call_batch(&ctx, function, r#"{"src_col":"src","dst_col":"dst"}"#).await;
            assert!(!bool_value(&batch, 0, 0), "missing source_vertex must be rejected for {function}");
            assert_eq!(string_value(&batch, 1, 0), "invalid_argument");
            assert!(string_value(&batch, 2, 0).contains("requires source_vertex"));
        } else {
            let call_json = if matches!(
                entry.algorithm,
                CugraphAlgorithm::Jaccard
                    | CugraphAlgorithm::Sorensen
                    | CugraphAlgorithm::Overlap
                    | CugraphAlgorithm::Cosine
            ) {
                r#"{"source_vertex":0,"src_col":"src","dst_col":"dst","options":{"vertex_pairs_table":"candidate_pairs","first_vertex_col":"first","second_vertex_col":"second"}}"#
            } else {
                r#"{"source_vertex":0,"src_col":"src","dst_col":"dst"}"#
            };
            let batch = validate_call_batch(&ctx, function, call_json).await;
            assert!(!bool_value(&batch, 0, 0), "extra source_vertex must be rejected for {function}");
            assert_eq!(string_value(&batch, 1, 0), "invalid_argument");
            assert!(string_value(&batch, 2, 0).contains("does not accept source_vertex"));
        }

        let call_json = if entry.native_algorithm.requires_source_vertex() {
            r#"{"source_vertex":0,"src_col":"src","dst_col":"dst"}"#
        } else if entry.algorithm == CugraphAlgorithm::PersonalizedPageRank {
            r#"{"src_col":"src","dst_col":"dst","options":{"personalization_table":"ppr_seeds","personalization_vertex_col":"vertex","personalization_value_col":"value"}}"#
        } else if matches!(
            entry.algorithm,
            CugraphAlgorithm::Jaccard
                | CugraphAlgorithm::Sorensen
                | CugraphAlgorithm::Overlap
                | CugraphAlgorithm::Cosine
        ) {
            r#"{"src_col":"src","dst_col":"dst","options":{"vertex_pairs_table":"candidate_pairs","first_vertex_col":"first","second_vertex_col":"second"}}"#
        } else {
            r#"{"src_col":"src","dst_col":"dst"}"#
        };
        let batch = validate_call_batch(&ctx, function, call_json).await;
        if entry.native_algorithm.weight_policy().requires_weight() {
            assert!(!bool_value(&batch, 0, 0), "missing required weight column must be rejected for {function}");
            assert_eq!(string_value(&batch, 1, 0), "invalid_argument");
            assert!(string_value(&batch, 2, 0).contains("requires a weight column"));
        } else {
            assert!(
                bool_value(&batch, 0, 0),
                "missing weight column should remain valid for non-required policy on {function}: {}",
                string_value(&batch, 2, 0),
            );
        }
    }
}

#[tokio::test]
async fn cugraph_similarity_metadata_describes_and_validates_the_explicit_pair_contract() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int32, DataType::Int32, DataType::Float64);
    register_vertex_pair_memtable(&ctx, "candidate_pairs", DataType::Int32, DataType::Int32, false);
    register_vertex_pair_memtable(&ctx, "nullable_pairs", DataType::Int32, DataType::Int32, true);
    register_vertex_pair_memtable(&ctx, "mismatched_pairs", DataType::Int32, DataType::Int64, false);
    register_vertex_pair_memtable(&ctx, "int64_pairs", DataType::Int64, DataType::Int64, false);
    register_vertex_pair_memtable(&ctx, "utf8_pairs", DataType::Utf8, DataType::Utf8, false);

    for function in ["cugraph_jaccard", "cugraph_sorensen", "cugraph_overlap", "cugraph_cosine"] {
        let describe_sql = format!(
            "SELECT signature, allowed_arg_counts_json, options_schema_json, argument_descriptions_json, limitations_json \
             FROM cugraph_describe_algorithm('{function}', 'verbose')"
        );
        let describe = ctx.sql(&describe_sql).await.unwrap().collect().await.unwrap();
        let describe = single_batch(&describe);
        assert!(
            string_value(describe, 0, 0).contains("table_name, src_col, dst_col, weight_col, options_json"),
            "{function} must expose the required five-argument signature"
        );
        assert_eq!(json_value(string_value(describe, 1, 0)), serde_json::json!([5]));
        let options = json_value(string_value(describe, 2, 0));
        let pair_table = &options["algorithm_options"]["vertex_pairs_table"];
        assert_eq!(pair_table["required"], true);
        assert_eq!(pair_table["side_input"]["role"], "vertex_pairs");
        assert_eq!(
            pair_table["side_input"]["column_options"],
            serde_json::json!(["first_vertex_col", "second_vertex_col"])
        );
        assert_eq!(options["graph_options"]["directed"]["default"], false);
        assert_eq!(options["graph_options"]["directed"]["allowed_values"], serde_json::json!([false]));

        let arguments = json_value(string_value(describe, 3, 0));
        let options_argument = json_array_item_by_name(&arguments, "options_json");
        assert_eq!(options_argument["required"], true);
        assert!(options_argument["note"].as_str().unwrap().contains("vertex_pairs_table"));
        let limitations = json_value(string_value(describe, 4, 0));
        assert_json_array_contains_string(
            &limitations,
            "candidate-pair columns must share the same Int32 or Int64 dtype and match the graph vertex domain",
        );

        let valid_sql = format!(
            r#"SELECT valid, error_code, table_resolved, src_dtype, dst_dtype, output_schema_json, normalized_options_json, would_execute_gpu
               FROM cugraph_validate_call(
                 '{function}',
                 'edges',
                 '{{"src_col":"src","dst_col":"dst","options":{{"vertex_pairs_table":"candidate_pairs","first_vertex_col":"first","second_vertex_col":"second"}}}}'
               )"#
        );
        let valid = ctx.sql(&valid_sql).await.unwrap().collect().await.unwrap();
        let valid = single_batch(&valid);
        assert!(bool_value(valid, 0, 0), "valid explicit pair call should be admitted for {function}");
        assert!(is_null_value(valid, 1, 0));
        assert!(bool_value(valid, 2, 0));
        assert_eq!(string_value(valid, 3, 0), "Int32");
        assert_eq!(string_value(valid, 4, 0), "Int32");
        assert!(string_value(valid, 5, 0).contains(r#""similarity""#));
        assert!(string_value(valid, 6, 0).contains(r#""vertex_pairs""#));
        assert!(bool_value(valid, 7, 0));

        let nullable_sql = format!(
            r#"SELECT valid, error_code
               FROM cugraph_validate_call(
                 '{function}',
                 'edges',
                 '{{"src_col":"src","dst_col":"dst","options":{{"vertex_pairs_table":"nullable_pairs","first_vertex_col":"first","second_vertex_col":"second"}}}}'
               )"#
        );
        let nullable = ctx.sql(&nullable_sql).await.unwrap().collect().await.unwrap();
        let nullable = single_batch(&nullable);
        assert!(bool_value(nullable, 0, 0), "schema-nullable but null-free pair inputs must validate for {function}");
        assert!(is_null_value(nullable, 1, 0));

        for (pairs, first, second, code) in [
            ("missing_pairs", "first", "second", "table_not_found"),
            ("candidate_pairs", "missing", "second", "column_not_found"),
            ("mismatched_pairs", "first", "second", "vertex_dtype_mismatch"),
            ("int64_pairs", "first", "second", "vertex_dtype_mismatch"),
            ("utf8_pairs", "first", "second", "vertex_dtype_mismatch"),
        ] {
            let call_json = format!(
                r#"{{"src_col":"src","dst_col":"dst","options":{{"vertex_pairs_table":"{pairs}","first_vertex_col":"{first}","second_vertex_col":"{second}"}}}}"#
            );
            assert_validate_error(&ctx, function, "edges", &call_json, code).await;
        }

        let directed_call = r#"{"src_col":"src","dst_col":"dst","options":{"directed":true,"vertex_pairs_table":"candidate_pairs","first_vertex_col":"first","second_vertex_col":"second"}}"#;
        assert_validate_error(&ctx, function, "edges", directed_call, "unsupported_option").await;
    }
}

#[tokio::test]
async fn cugraph_validate_call_rejects_int64_for_legacy_int32_only_algorithms() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges_i64", DataType::Int64, DataType::Int64, DataType::Float64);
    register_edge_memtable(&ctx, "edges_i32", DataType::Int32, DataType::Int32, DataType::Float64);

    for (function, call_json) in [
        ("cugraph_minimum_spanning_tree", r#"{"src_col":"src","dst_col":"dst","weight_col":"weight"}"#),
        ("cugraph_spectral_modularity_maximization", r#"{"src_col":"src","dst_col":"dst"}"#),
    ] {
        let escaped = call_json.replace('\'', "''");
        let rejected = ctx
            .sql(&format!(
                "SELECT valid, error_code, message, would_execute_gpu \
                 FROM cugraph_validate_call('{function}', 'edges_i64', '{escaped}')"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let rejected = single_batch(&rejected);
        assert!(!bool_value(rejected, 0, 0), "{function} must reject Int64 vertices");
        assert_eq!(string_value(rejected, 1, 0), "unsupported_vertex_dtype");
        assert!(string_value(rejected, 2, 0).contains("supported vertex dtypes: Int32"));
        assert!(!bool_value(rejected, 3, 0));

        let accepted = ctx
            .sql(&format!(
                "SELECT valid, error_code, would_execute_gpu \
                 FROM cugraph_validate_call('{function}', 'edges_i32', '{escaped}')"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let accepted = single_batch(&accepted);
        assert!(bool_value(accepted, 0, 0), "{function} must accept Int32 vertices");
        assert!(is_null_value(accepted, 1, 0));
        assert!(bool_value(accepted, 2, 0));
    }
}

#[tokio::test]
async fn cugraph_validate_call_rejects_directed_true_for_undirected_only_algorithms() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int32, DataType::Int32, DataType::Float64);

    for (function, call_json, expected) in [
        (
            "cugraph_minimum_spanning_tree",
            r#"{"src_col":"src","dst_col":"dst","weight_col":"weight","options":{"directed":true}}"#,
            "cugraph_minimum_spanning_tree requires directed=false",
        ),
        (
            "cugraph_spectral_modularity_maximization",
            r#"{"src_col":"src","dst_col":"dst","options":{"directed":true}}"#,
            "cugraph_spectral_modularity_maximization requires directed=false",
        ),
    ] {
        let escaped = call_json.replace('\'', "''");
        let batches = ctx
            .sql(&format!(
                "SELECT valid, error_code, message, would_execute_gpu \
                 FROM cugraph_validate_call('{function}', 'edges', '{escaped}')"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batch = single_batch(&batches);
        assert!(!bool_value(batch, 0, 0), "{function} must reject directed=true");
        assert_eq!(string_value(batch, 1, 0), "unsupported_option");
        assert!(string_value(batch, 2, 0).contains(expected), "{}", string_value(batch, 2, 0));
        assert!(!bool_value(batch, 3, 0));
    }
}

#[tokio::test]
async fn cugraph_validate_call_rejects_spectral_iteration_counts_above_libcugraph_int() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int32, DataType::Int32, DataType::Float64);

    for (function, option, message) in [(
        "cugraph_spectral_modularity_maximization",
        "evs_max_iterations",
        "evs_max_iterations must be at most i32::MAX",
    )] {
        let batch = validate_call_batch(
            &ctx,
            function,
            &format!(r#"{{"src_col":"src","dst_col":"dst","options":{{"{option}":2147483648}}}}"#),
        )
        .await;
        assert!(!bool_value(&batch, 0, 0), "{function} must reject oversized {option}");
        assert_eq!(string_value(&batch, 1, 0), "invalid_argument");
        assert!(string_value(&batch, 2, 0).contains(message), "{}", string_value(&batch, 2, 0));
    }
}

#[tokio::test]
async fn cugraph_validate_call_accepts_utf8_bfs_logical_vertex_domain() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_allowed_bfs_edges(&ctx, "allowed_edges");
    register_string_vertex_memtable(&ctx, "targets", "vertex_id", &["d"]);

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, src_dtype, dst_dtype, output_schema_json, normalized_options_json
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'raw_edges',
                 '{"source_vertex":"a","src_col":"src","dst_col":"dst","options":{"output_mode":"normalized","target_vertices_table":"targets","target_vertex_col":"vertex_id","include_edges_table":"allowed_edges","include_edge_src_col":"allowed_src","include_edge_dst_col":"allowed_dst"}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert_eq!(string_value(batch, 2, 0), "Utf8");
    assert_eq!(string_value(batch, 3, 0), "Utf8");
    assert!(string_value(batch, 4, 0).contains(r#""data_type":"Utf8""#));
    assert!(string_value(batch, 5, 0).contains(r#""source_vertex":"a""#));
}

#[tokio::test]
async fn cugraph_validate_call_accepts_bfs_source_arrays_and_relation_sources() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_edge_memtable(&ctx, "numeric_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "bfs_sources", "vertex_id", DataType::Int64, false);
    register_string_bfs_edges(&ctx, "string_edges");
    register_string_vertex_memtable(&ctx, "string_sources", "vertex_id", &["a", "b"]);

    let batch = validate_call_batch(
        &ctx,
        "cugraph_bfs",
        r#"{"source_vertex":[0,2],"src_col":"src","dst_col":"dst","options":{"output_mode":"normalized"}}"#,
    )
    .await;
    assert!(bool_value(&batch, 0, 0), "numeric source arrays should validate: {}", string_value(&batch, 2, 0));

    let batch = validate_call_batch(
        &ctx,
        "cugraph_bfs",
        r#"{"source_vertex":["a","b"],"src_col":"src","dst_col":"dst","options":{"output_mode":"normalized"}}"#,
    )
    .await;
    // validate_call_batch uses the default numeric edge table. Exercise string arrays through explicit SQL below.
    assert!(!bool_value(&batch, 0, 0));

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, output_schema_json, normalized_options_json
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'string_edges',
                 '{"source_vertex":["a","b"],"src_col":"src","dst_col":"dst","options":{"output_mode":"normalized"}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert!(string_value(batch, 2, 0).contains(r#""data_type":"Utf8""#));
    assert!(string_value(batch, 3, 0).contains(r#""source_vertex":["a","b"]"#));

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, normalized_options_json
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'numeric_edges',
                 '{"source_vertex":null,"src_col":"src","dst_col":"dst","options":{"source_vertices_table":"bfs_sources","source_vertex_col":"vertex_id"}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert!(string_value(batch, 2, 0).contains(r#""table":"bfs_sources""#));

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, output_schema_json
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'string_edges',
                 '{"src_col":"src","dst_col":"dst","options":{"source_vertices_table":"string_sources","source_vertex_col":"vertex_id"}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert!(string_value(batch, 2, 0).contains(r#""data_type":"Utf8""#));
}

#[tokio::test]
async fn cugraph_validate_call_rejects_invalid_bfs_source_arrays_and_relation_pairs() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "bfs_sources", "vertex_id", DataType::Int64, false);

    for call_json in [
        r#"{"source_vertex":[],"src_col":"src","dst_col":"dst"}"#,
        r#"{"source_vertex":[0,null],"src_col":"src","dst_col":"dst"}"#,
        r#"{"source_vertex":[0,"a"],"src_col":"src","dst_col":"dst"}"#,
        r#"{"source_vertex":null,"src_col":"src","dst_col":"dst"}"#,
        r#"{"src_col":"src","dst_col":"dst"}"#,
        r#"{"source_vertex":0,"src_col":"src","dst_col":"dst","options":{"source_vertices_table":"bfs_sources","source_vertex_col":"vertex_id"}}"#,
        r#"{"source_vertex":null,"src_col":"src","dst_col":"dst","options":{"source_vertices_table":"bfs_sources"}}"#,
        r#"{"source_vertex":null,"src_col":"src","dst_col":"dst","options":{"source_vertex_col":"vertex_id"}}"#,
    ] {
        let batch = validate_call_batch(&ctx, "cugraph_bfs", call_json).await;
        assert!(
            !bool_value(&batch, 0, 0),
            "invalid source JSON should be rejected: call={call_json}, message={}",
            string_value(&batch, 2, 0)
        );
        assert!(!is_null_value(&batch, 1, 0));
    }
}

#[tokio::test]
async fn cugraph_validate_call_treats_arrow_string_variants_as_one_bfs_logical_domain() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_large_string_bfs_edges(&ctx, "large_edges");
    register_string_vertex_memtable(&ctx, "targets", "vertex_id", &["d"]);

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, src_dtype, dst_dtype
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'large_edges',
                 '{"source_vertex":"a","src_col":"src","dst_col":"dst","options":{"target_vertices_table":"targets","target_vertex_col":"vertex_id"}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert_eq!(string_value(batch, 2, 0), "LargeUtf8");
    assert_eq!(string_value(batch, 3, 0), "LargeUtf8");
}

#[tokio::test]
async fn cugraph_validate_call_rejects_mixed_bfs_logical_vertex_domains() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "numeric_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_string_bfs_edges(&ctx, "string_edges");
    register_vertex_memtable(&ctx, "numeric_targets", "vertex_id", DataType::Int64, false);
    register_string_bfs_edges_with_edge_id(&ctx, "string_edges_with_ids");

    assert_validate_error(
        &ctx,
        "cugraph_bfs",
        "numeric_edges",
        r#"{"source_vertex":"a","src_col":"src","dst_col":"dst","options":{"output_mode":"normalized"}}"#,
        "vertex_dtype_mismatch",
    )
    .await;
    assert_validate_error(
        &ctx,
        "cugraph_bfs",
        "string_edges",
        r#"{"source_vertex":0,"src_col":"src","dst_col":"dst","options":{"output_mode":"normalized"}}"#,
        "vertex_dtype_mismatch",
    )
    .await;
    assert_validate_error(
        &ctx,
        "cugraph_bfs",
        "string_edges",
        r#"{"source_vertex":"a","src_col":"src","dst_col":"dst","options":{"target_vertices_table":"numeric_targets","target_vertex_col":"vertex_id"}}"#,
        "vertex_dtype_mismatch",
    )
    .await;
    assert_validate_error(
        &ctx,
        "cugraph_bfs",
        "string_edges_with_ids",
        r#"{"source_vertex":"a","src_col":"src","dst_col":"dst","options":{"edge_id_col":"edge_id"}}"#,
        "unsupported_option",
    )
    .await;
}

#[tokio::test]
async fn cugraph_bfs_predicate_options_validate_predicate_relations() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "raw_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "excluded_vertices", "vertex_id", DataType::Int64, false);
    register_vertex_memtable(&ctx, "targets", "vertex_id", DataType::Int64, false);
    register_edge_memtable(&ctx, "allowed_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, output_schema_json, normalized_options_json
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'raw_edges',
                 '{"source_vertex":0,"src_col":"src","dst_col":"dst","options":{"output_mode":"normalized","exclude_vertices_table":"excluded_vertices","exclude_vertex_col":"vertex_id","target_vertices_table":"targets","target_vertex_col":"vertex_id","include_edges_table":"allowed_edges","include_edge_src_col":"src","include_edge_dst_col":"dst"}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert!(string_value(batch, 2, 0).contains(r#""reachable""#));
    let normalized = string_value(batch, 3, 0);
    assert!(normalized.contains(r#""mode":"normalized""#), "{normalized}");
    assert!(normalized.contains(r#""exclude_vertices""#), "{normalized}");
    assert!(normalized.contains(r#""include_edges""#), "{normalized}");

    register_vertex_memtable(&ctx, "bad_targets", "vertex_id", DataType::Int32, false);
    // A late predicate-validation failure must still report the resolved edge
    // dtypes: leaving them NULL would contradict src/dst_col_resolved=true.
    let rejected = assert_validate_error(
        &ctx,
        "cugraph_bfs",
        "raw_edges",
        r#"{"source_vertex":0,"src_col":"src","dst_col":"dst","options":{"target_vertices_table":"bad_targets","target_vertex_col":"vertex_id"}}"#,
        "vertex_dtype_mismatch",
    )
    .await;
    assert_eq!(string_value(&rejected, 2, 0), "Int64");
    assert_eq!(string_value(&rejected, 3, 0), "Int64");
    // No weight_col was requested in this call, so weight_dtype stays NULL.
    assert!(is_null_value(&rejected, 4, 0));
}

#[tokio::test]
async fn cugraph_validate_call_accepts_schema_nullable_predicate_columns() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "raw_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    // Predicate side inputs whose Arrow schema is nullable. After the
    // schema-vs-data split, validate must not reject on schema nullability for
    // any predicate role; it only checks existence and dtype.
    register_vertex_memtable(&ctx, "targets", "vertex_id", DataType::Int64, true);
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Int64, true),
        Field::new("allowed_dst", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![2_i64, 3])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("allowed_edges", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'raw_edges',
                 '{"source_vertex":0,"src_col":"src","dst_col":"dst","options":{"target_vertices_table":"targets","target_vertex_col":"vertex_id","include_edges_table":"allowed_edges","include_edge_src_col":"allowed_src","include_edge_dst_col":"allowed_dst"}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0), "schema-nullable predicate columns must validate");
    assert!(is_null_value(batch, 1, 0));
}
