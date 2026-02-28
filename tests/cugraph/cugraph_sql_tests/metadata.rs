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
async fn cugraph_metadata_list_algorithms_reports_allowlist_state() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cugraph_sql(CugraphSqlConfig::default().with_allowed_algorithms([CugraphAlgorithm::PageRank]))
        .build();
    let ctx = SessionContext::new_with_state(state);

    let batches = ctx
        .sql(
            "SELECT function_name, enabled, requires_source_vertex, output_schema_json, \
                    summary, supports_weight, weight_col_policy, output_schema_kind, \
                    has_mode_schemas, options_schema_json \
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
    assert!(!bool_value(batch, 1, 0), "BFS is registered but disabled by the allowlist");
    assert!(bool_value(batch, 2, 0), "BFS must advertise its source_vertex requirement");
    assert!(string_value(batch, 4, 0).contains("breadth-first"));
    assert!(bool_value(batch, 5, 0), "BFS accepts a weight_col binding even though traversal ignores it");
    assert_eq!(string_value(batch, 6, 0), "ignored");
    assert_eq!(string_value(batch, 7, 0), "default_mode");
    assert!(bool_value(batch, 8, 0));
    let bfs_options = json_value(string_value(batch, 9, 0));
    assert_eq!(bfs_options["algorithm_options"]["output_mode"]["type"], "Utf8");
    assert_eq!(bfs_options["algorithm_options"]["output_mode"]["default"], "raw");
    assert_eq!(
        bfs_options["algorithm_options"]["output_mode"]["allowed_values"],
        serde_json::json!(["raw", "normalized", "path"])
    );
    assert!(bfs_options["algorithm_options"]["output_mode"]["description"].as_str().unwrap().contains("result shape"));
    assert_eq!(bfs_options["output_schema_authority"]["call_specific_authority"], CUGRAPH_VALIDATE_CALL_FUNCTION);

    assert_eq!(string_value(batch, 0, 1), "cugraph_minimum_spanning_tree");
    assert_eq!(string_value(batch, 6, 1), "required");
    assert_eq!(string_value(batch, 7, 1), "fixed");
    assert!(!bool_value(batch, 8, 1));

    assert_eq!(string_value(batch, 0, 2), "cugraph_pagerank");
    assert!(bool_value(batch, 1, 2));
    assert!(string_value(batch, 3, 2).contains(r#""vertex""#));
    assert_eq!(string_value(batch, 6, 2), "ignored");
    assert_eq!(string_value(batch, 7, 2), "fixed");
    assert!(!bool_value(batch, 8, 2));
    let pagerank_options = json_value(string_value(batch, 9, 2));
    assert_eq!(pagerank_options["graph_options"]["directed"]["type"], "Boolean");
    assert_eq!(pagerank_options["graph_options"]["construction_policy"]["allowed_values"][0], "python_cugraph");
    assert_eq!(pagerank_options["algorithm_options"]["alpha"]["scope"], "algorithm");
    assert_eq!(pagerank_options["algorithm_options"]["alpha"]["type"], "Float64");

    assert_eq!(string_value(batch, 0, 3), "cugraph_sssp");
    assert_eq!(string_value(batch, 6, 3), "optional");
}

#[tokio::test]
async fn cugraph_metadata_describe_algorithm_returns_structured_rows() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let batches = ctx
        .sql(
            "SELECT valid, error_code, requires_source_vertex, supports_weight, output_schema_json, \
                    options_schema_json, default_graph_options_json, argument_descriptions_json, \
                    option_defaults_json, result_field_descriptions_json, limitations_json, \
                    summary, purpose, mode_schemas_json, examples_json \
             FROM cugraph_describe_algorithm('cugraph_bfs')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert_eq!(string_value(batch, 1, 0), "ok");
    assert!(bool_value(batch, 2, 0));
    assert!(bool_value(batch, 3, 0), "BFS accepts weight_col even though traversal ignores weights");

    let output_schema = json_value(string_value(batch, 4, 0));
    let default_vertex = json_array_item_by_name(&output_schema["fields"], "vertex");
    assert_eq!(default_vertex["data_type"], "Int64", "output_schema_json stays the concrete default schema");

    let options_schema = json_value(string_value(batch, 5, 0));
    assert_eq!(
        options_schema["graph_options"]["directed"]["description"],
        "Whether graph construction treats edges as directed."
    );
    assert_eq!(
        options_schema["algorithm_options"]["depth_limit"]["description"],
        "Maximum BFS depth in hops; null means unbounded traversal."
    );
    assert!(
        options_schema["algorithm_options"]["output_mode"]["description"].as_str().unwrap().contains("result shape")
    );
    assert_eq!(options_schema["output_schema_authority"]["call_specific_authority"], CUGRAPH_VALIDATE_CALL_FUNCTION);
    assert!(
        options_schema["output_schema_authority"]["note"].as_str().unwrap().contains("mode_schemas_json"),
        "BFS describe JSON should tell callers where the mode-specific schema lives"
    );
    let target_table = &options_schema["algorithm_options"]["target_vertices_table"];
    assert_eq!(target_table["side_input"]["role"], "target_vertices");
    assert_eq!(
        target_table["side_input"]["expected_columns"][0]["type"],
        "same logical vertex domain as src_col and dst_col"
    );
    let include_edges = &options_schema["algorithm_options"]["include_edges_table"];
    assert_eq!(include_edges["side_input"]["role"], "include_edges");
    assert_eq!(include_edges["side_input"]["expected_columns"][2]["optional_when"], "edge_id_col is set");
    assert_eq!(
        options_schema["algorithm_options"]["include_edge_ids_col"]["defaults_to"],
        "edge_id_col when omitted and include_edge_ids_table is set"
    );
    assert_eq!(options_schema["algorithm_options"]["include_edge_id_col"]["valid_when"], "edge_id_col is set");

    let default_graph_options = json_value(string_value(batch, 6, 0));
    assert!(default_graph_options["directed"].is_boolean());

    let argument_descriptions = json_value(string_value(batch, 7, 0));
    let source_vertex = json_array_item_by_name(&argument_descriptions, "source_vertex");
    assert_eq!(source_vertex["type"], "Int64|Utf8");
    let weight_col = json_array_item_by_name(&argument_descriptions, "weight_col");
    assert_eq!(weight_col["weight_col_policy"], "ignored");
    assert_eq!(weight_col["semantic_effect"], "none for traversal; BFS distance is a hop count");
    assert!(weight_col["note"].as_str().unwrap().contains("ignored by BFS traversal"));

    let option_defaults = json_value(string_value(batch, 8, 0));
    assert_eq!(option_defaults["algorithm_options"]["output_mode"], "raw");

    let result_fields = json_value(string_value(batch, 9, 0));
    let predecessor = json_array_item_by_name(&result_fields, "predecessor");
    assert_eq!(predecessor["data_type"], "Int64");

    let limitations = json_value(string_value(batch, 10, 0));
    assert_json_array_contains_string(&limitations, "BFS traversal ignores edge weights; distance is a hop count");

    assert!(string_value(batch, 11, 0).contains("breadth-first"));
    assert!(string_value(batch, 12, 0).contains("reconstructed source-to-target path"));

    let mode_schemas = json_value(string_value(batch, 13, 0));
    let raw_vertex = json_array_item_by_name(&mode_schemas["raw"], "vertex");
    assert_eq!(raw_vertex["type"], "Int64|Utf8");
    let path_index = json_array_item_by_name(&mode_schemas["path"], "path_index");
    assert_eq!(path_index["type"], "Int64");

    let examples = json_value(string_value(batch, 14, 0));
    let dry_run_path = json_array_item_by_name(&examples, "dry_run_path");
    assert!(
        dry_run_path["sql"].as_str().unwrap().contains(CUGRAPH_VALIDATE_CALL_FUNCTION),
        "dry-run example should point to the call-specific validation surface"
    );

    let batches = ctx
        .sql("SELECT valid, error_code, message FROM cugraph_describe_algorithm('cugraph_missing')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(!bool_value(batch, 0, 0));
    assert_eq!(string_value(batch, 1, 0), "unknown_algorithm");
    assert!(string_value(batch, 2, 0).contains("cugraph_missing"));
}

#[tokio::test]
async fn cugraph_metadata_schema_versions_and_column_order_are_pinned() {
    assert_eq!(CUGRAPH_LIST_ALGORITHMS_SCHEMA_VERSION, 4);
    assert_eq!(CUGRAPH_DESCRIBE_ALGORITHM_SCHEMA_VERSION, 4);
    assert_eq!(CUGRAPH_VALIDATE_CALL_SCHEMA_VERSION, 1);

    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let list_df = ctx.sql("SELECT * FROM cugraph_list_algorithms()").await.unwrap();
    assert_eq!(
        schema_field_names(list_df.schema().inner()),
        vec![
            "function_name",
            "algorithm",
            "enabled",
            "experimental",
            "requires_source_vertex",
            "supports_weight",
            "min_args",
            "max_args",
            "output_schema_json",
            "options_schema_json",
            "default_graph_options_json",
            "summary",
            "weight_col_policy",
            "output_schema_kind",
            "has_mode_schemas",
            "metadata_schema_version",
        ],
    );
    let list_batches = list_df.collect().await.unwrap();
    let list_batch = single_batch(&list_batches);
    assert!(list_batch.num_rows() > 0);
    assert_json_columns_parse_and_have_no_control_chars(list_batch);
    let version_column = list_batch.schema().index_of("metadata_schema_version").unwrap();
    assert_eq!(int64_value(list_batch, version_column, 0), i64::from(CUGRAPH_LIST_ALGORITHMS_SCHEMA_VERSION));

    let describe_df = ctx.sql("SELECT * FROM cugraph_describe_algorithm('cugraph_bfs')").await.unwrap();
    assert_eq!(
        schema_field_names(describe_df.schema().inner()),
        vec![
            "valid",
            "error_code",
            "message",
            "function_name",
            "algorithm",
            "enabled",
            "experimental",
            "requires_source_vertex",
            "supports_weight",
            "min_args",
            "max_args",
            "output_schema_json",
            "options_schema_json",
            "default_graph_options_json",
            "argument_descriptions_json",
            "option_defaults_json",
            "result_field_descriptions_json",
            "limitations_json",
            "summary",
            "purpose",
            "mode_schemas_json",
            "examples_json",
            "metadata_schema_version",
        ],
    );
    let describe_batches = describe_df.collect().await.unwrap();
    let describe_batch = single_batch(&describe_batches);
    assert_json_columns_parse_and_have_no_control_chars(describe_batch);
    let version_column = describe_batch.schema().index_of("metadata_schema_version").unwrap();
    assert_eq!(int64_value(describe_batch, version_column, 0), i64::from(CUGRAPH_DESCRIBE_ALGORITHM_SCHEMA_VERSION));
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
    assert_eq!(string_value(batch, 1, 0), "ok");
    assert!(bool_value(batch, 2, 0));
    assert_eq!(string_value(batch, 3, 0), "Int64");
    assert_eq!(string_value(batch, 4, 0), "Int64");
    assert_eq!(string_value(batch, 5, 0), "Float64");
    assert!(bool_value(batch, 6, 0));
    assert!(string_value(batch, 7, 0).contains(r#""depth_limit":2"#));
}

#[tokio::test]
async fn cugraph_metadata_validate_call_returns_stable_invalid_rows() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cugraph_sql(CugraphSqlConfig::default().with_allowed_algorithms([CugraphAlgorithm::PageRank]))
        .build();
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
        "cugraph_bfs",
        "bad_edges",
        r#"{"source_vertex":0,"src_col":"src","dst_col":"dst"}"#,
        "algorithm_disabled",
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
    assert_eq!(string_value(batch, 1, 0), "ok");
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
    assert_eq!(string_value(batch, 1, 0), "ok");
    assert!(bool_value(batch, 2, 0));
    assert_eq!(string_value(batch, 3, 0), "Int64");
    assert_eq!(string_value(batch, 4, 0), "Int64");
    assert_eq!(string_value(batch, 5, 0), "Float64");
    assert!(bool_value(batch, 6, 0));
}

#[test]
fn cugraph_sql_operation_registry_matches_native_engine_inventory() {
    use nexus_query_engine::GRAPH_OPERATION_REGISTRY;

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
    use nexus_query_engine::GRAPH_OPERATION_REGISTRY;

    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    let batches = ctx
        .sql(
            "SELECT function_name, requires_source_vertex, weight_col_policy, output_schema_kind, \
                    has_mode_schemas, output_schema_json, default_graph_options_json \
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
        assert_eq!(
            bool_value(batch, 1, row),
            descriptor.source_vertex_requirement().requires_source_vertex(),
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
        assert_eq!(bool_value(batch, 4, row), descriptor.output_schema_kind().has_mode_schemas());

        let output_schema = json_value(string_value(batch, 5, row));
        let fields = output_schema["fields"].as_array().unwrap();
        assert_eq!(fields.len(), descriptor.base_result_schema().fields().len());
        for (field_json, native_field) in fields.iter().zip(descriptor.base_result_schema().fields()) {
            assert_eq!(field_json["name"], native_field.name().as_str());
            assert_eq!(field_json["data_type"], test_data_type_label(native_field.data_type()));
            assert_eq!(field_json["nullable"], native_field.is_nullable());
        }

        let graph_options = json_value(string_value(batch, 6, row));
        let native_defaults = descriptor.default_graph_options(&Default::default());
        assert_eq!(graph_options["directed"], native_defaults.directed);
        assert_eq!(graph_options["renumber"], native_defaults.renumber);
        assert_eq!(graph_options["construction_policy"], native_defaults.policy.as_str());
    }
}

#[tokio::test]
async fn cugraph_validate_call_source_and_weight_policy_match_metadata() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int64, DataType::Int64, DataType::Float64);

    for entry in CUGRAPH_OPERATION_REGISTRY {
        let function = entry.sql_function_name;
        if entry.native_algorithm.requires_source_vertex() {
            let batch = validate_call_batch(&ctx, function, r#"{"src_col":"src","dst_col":"dst"}"#).await;
            assert!(!bool_value(&batch, 0, 0), "missing source_vertex must be rejected for {function}");
            assert_eq!(string_value(&batch, 1, 0), "invalid_argument");
            assert!(string_value(&batch, 2, 0).contains("requires source_vertex"));
        } else {
            let batch =
                validate_call_batch(&ctx, function, r#"{"source_vertex":0,"src_col":"src","dst_col":"dst"}"#).await;
            assert!(!bool_value(&batch, 0, 0), "extra source_vertex must be rejected for {function}");
            assert_eq!(string_value(&batch, 1, 0), "invalid_argument");
            assert!(string_value(&batch, 2, 0).contains("does not accept source_vertex"));
        }

        let call_json = if entry.native_algorithm.requires_source_vertex() {
            r#"{"source_vertex":0,"src_col":"src","dst_col":"dst"}"#
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
    assert_eq!(string_value(batch, 1, 0), "ok");
    assert_eq!(string_value(batch, 2, 0), "Utf8");
    assert_eq!(string_value(batch, 3, 0), "Utf8");
    assert!(string_value(batch, 4, 0).contains(r#""data_type":"Utf8""#));
    assert!(string_value(batch, 5, 0).contains(r#""source_vertex":"a""#));
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
    assert_eq!(string_value(batch, 1, 0), "ok");
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
    assert_eq!(string_value(batch, 1, 0), "ok");
    assert!(string_value(batch, 2, 0).contains(r#""reachable""#));
    let normalized = string_value(batch, 3, 0);
    assert!(normalized.contains(r#""mode":"normalized""#), "{normalized}");
    assert!(normalized.contains(r#""exclude_vertices""#), "{normalized}");
    assert!(normalized.contains(r#""include_edges""#), "{normalized}");

    register_vertex_memtable(&ctx, "bad_targets", "vertex_id", DataType::Int32, false);
    assert_validate_error(
        &ctx,
        "cugraph_bfs",
        "raw_edges",
        r#"{"source_vertex":0,"src_col":"src","dst_col":"dst","options":{"target_vertices_table":"bad_targets","target_vertex_col":"vertex_id"}}"#,
        "vertex_dtype_mismatch",
    )
    .await;
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
    assert_eq!(string_value(batch, 1, 0), "ok");
}
