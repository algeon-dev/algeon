use std::{
    fs,
    path::Path,
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::ScalarValue,
    datasource::MemTable,
    execution::SessionStateBuilder,
    logical_expr::{col, lit},
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::displayable,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_DESCRIBE_ALGORITHM_SCHEMA_VERSION, CUGRAPH_LIST_ALGORITHMS_FUNCTION,
    CUGRAPH_LIST_ALGORITHMS_SCHEMA_VERSION, CUGRAPH_OPERATION_REGISTRY, CUGRAPH_VALIDATE_CALL_FUNCTION,
    CUGRAPH_VALIDATE_CALL_SCHEMA_VERSION, CudfOptimizerConfig, CudfSessionStateBuilderExt, CugraphAlgorithm,
    CugraphAlgorithmParams, CugraphBfsOutputMode, CugraphBfsSourceVertex, CugraphGraphConstructionPolicy,
    CugraphSqlConfig, GpuFallbackPolicy, NativeCudfComposedExec, parse_cugraph_algorithm_spec, stable_result_schema,
};

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
                    summary, supports_weight, options_schema_json \
             FROM cugraph_list_algorithms() \
             WHERE function_name IN ('cugraph_pagerank', 'cugraph_bfs') \
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
    assert!(string_value(batch, 6, 0).contains(r#""output_mode""#));
    assert!(string_value(batch, 6, 0).contains(r#""allowed_values":["raw","normalized","path"]"#));
    assert_eq!(string_value(batch, 0, 1), "cugraph_pagerank");
    assert!(bool_value(batch, 1, 1));
    assert!(string_value(batch, 3, 1).contains(r#""vertex""#));
}

#[tokio::test]
async fn cugraph_metadata_describe_algorithm_returns_structured_rows() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let batches = ctx
        .sql(
            "SELECT valid, error_code, requires_source_vertex, argument_descriptions_json, \
                    summary, purpose, mode_schemas_json, examples_json, limitations_json \
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
    assert!(string_value(batch, 3, 0).contains("source_vertex"));
    assert!(string_value(batch, 3, 0).contains("Int64|Utf8"));
    assert!(string_value(batch, 4, 0).contains("breadth-first"));
    assert!(string_value(batch, 5, 0).contains("reconstructed source-to-target path"));
    assert!(string_value(batch, 6, 0).contains(r#""path_index""#));
    assert!(string_value(batch, 7, 0).contains("dry_run_path"));
    assert!(string_value(batch, 8, 0).contains("ignores edge weights"));

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
    assert_eq!(CUGRAPH_LIST_ALGORITHMS_SCHEMA_VERSION, 3);
    assert_eq!(CUGRAPH_DESCRIBE_ALGORITHM_SCHEMA_VERSION, 3);
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
        ],
    );

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
        ],
    );
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
async fn cugraph_sql_planning_exposes_stable_output_schemas() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    assert_sql_schema(&ctx, "SELECT * FROM cugraph_pagerank('graph_edges')", CugraphAlgorithm::PageRank).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_bfs('graph_edges', 0)", CugraphAlgorithm::Bfs).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_louvain('graph_edges')", CugraphAlgorithm::Louvain).await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_eigenvector_centrality('graph_edges')",
        CugraphAlgorithm::EigenvectorCentrality,
    )
    .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_katz_centrality('graph_edges')", CugraphAlgorithm::KatzCentrality)
        .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_hits('graph_edges')", CugraphAlgorithm::Hits).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_sssp('graph_edges', 0)", CugraphAlgorithm::Sssp).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_degrees_all('graph_edges')", CugraphAlgorithm::DegreesAll).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_in_degrees_all('graph_edges')", CugraphAlgorithm::InDegreesAll)
        .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_out_degrees_all('graph_edges')", CugraphAlgorithm::OutDegreesAll)
        .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_weakly_connected_components('graph_edges')",
        CugraphAlgorithm::WeaklyConnectedComponents,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_strongly_connected_components('graph_edges')",
        CugraphAlgorithm::StronglyConnectedComponents,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_triangle_count_all('graph_edges')",
        CugraphAlgorithm::TriangleCountAll,
    )
    .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_core_number('graph_edges')", CugraphAlgorithm::CoreNumber).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_k_core('graph_edges')", CugraphAlgorithm::KCore).await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_minimum_spanning_tree('graph_edges', 'src', 'dst', 'weight')",
        CugraphAlgorithm::MinimumSpanningTree,
    )
    .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_pagerank_strict('graph_edges')", CugraphAlgorithm::PageRankStrict)
        .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_leiden('graph_edges')", CugraphAlgorithm::Leiden).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_ecg('graph_edges')", CugraphAlgorithm::Ecg).await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_spectral_modularity_maximization('graph_edges')",
        CugraphAlgorithm::SpectralModularityMaximization,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_balanced_cut_clustering('graph_edges')",
        CugraphAlgorithm::BalancedCutClustering,
    )
    .await;
}

#[tokio::test]
async fn cugraph_bfs_path_mode_exposes_path_schema_through_canonical_function() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let df = ctx
        .sql(
            "SELECT * FROM cugraph_bfs('graph_edges', 0, 'src', 'dst', NULL, \
             '{\"output_mode\":\"path\",\"target_vertices_table\":\"targets\",\"target_vertex_col\":\"vertex\"}')",
        )
        .await
        .unwrap();
    assert_eq!(
        df.schema().inner().fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>(),
        vec![
            Field::new("path_index", DataType::Int64, false),
            Field::new("source", DataType::Int64, false),
            Field::new("target", DataType::Int64, false),
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, false),
        ],
    );
}

#[tokio::test]
async fn cugraph_bfs_path_mode_requires_target_relation() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let error = ctx
        .sql("SELECT * FROM cugraph_bfs('graph_edges', 0, 'src', 'dst', NULL, '{\"output_mode\":\"path\"}')")
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("BFS path output requires target_vertices_table and target_vertex_col options"));
}

#[tokio::test]
async fn cugraph_validate_call_accepts_bfs_path_mode_on_canonical_function() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "raw_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "targets", "vertex_id", DataType::Int64, false);

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, output_schema_json, normalized_options_json
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'raw_edges',
                 '{"source_vertex":0,"src_col":"src","dst_col":"dst","options":{"output_mode":"path","target_vertices_table":"targets","target_vertex_col":"vertex_id"}}'
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
    assert!(string_value(batch, 2, 0).contains(r#""path_index""#));
    assert!(string_value(batch, 3, 0).contains(r#""mode":"path""#));
}

#[tokio::test]
async fn cugraph_validate_call_rejects_path_mode_target_info_as_structured_row() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "raw_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "targets", "vertex_id", DataType::Int64, false);

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, message
               FROM cugraph_validate_call(
                 'cugraph_bfs',
                 'raw_edges',
                 '{"source_vertex":0,"src_col":"src","dst_col":"dst","options":{"output_mode":"path","target_vertices_table":"targets","target_vertex_col":"vertex_id","return_target_info":true}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(!bool_value(batch, 0, 0));
    assert_eq!(string_value(batch, 1, 0), "invalid_argument");
    assert!(string_value(batch, 2, 0).contains("cannot combine with return_target_info=true"));
}

#[tokio::test]
async fn cugraph_bfs_path_name_is_not_registered_or_aliased() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let error = ctx
        .sql("SELECT * FROM cugraph_bfs_path('graph_edges', 0, 'src', 'dst', NULL, '{}')")
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("cugraph_bfs_path"));

    let batches = ctx
        .sql(
            "SELECT valid, error_code, message \
             FROM cugraph_describe_algorithm('cugraph_bfs_path')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(!bool_value(batch, 0, 0));
    assert_eq!(string_value(batch, 1, 0), "unknown_algorithm");
    assert!(string_value(batch, 2, 0).contains("cugraph_bfs_path"));

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, message
               FROM cugraph_validate_call(
                 'cugraph_bfs_path',
                 'raw_edges',
                 '{"source_vertex":0,"src_col":"src","dst_col":"dst"}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(!bool_value(batch, 0, 0));
    assert_eq!(string_value(batch, 1, 0), "unknown_algorithm");
    assert!(string_value(batch, 2, 0).contains("cugraph_bfs_path"));
}

#[tokio::test]
async fn cugraph_bfs_normalized_output_mode_is_opt_in() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    assert_sql_schema(&ctx, "SELECT * FROM cugraph_bfs('graph_edges', 0)", CugraphAlgorithm::Bfs).await;

    let df = ctx
        .sql(
            "SELECT * FROM cugraph_bfs('graph_edges', 0, 'src', 'dst', NULL, \
             '{\"output_mode\":\"normalized\"}')",
        )
        .await
        .unwrap();
    let actual = df.schema().inner();
    assert_eq!(
        actual.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>(),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, true),
            Field::new("predecessor", DataType::Int64, true),
            Field::new("reachable", DataType::Boolean, false),
        ],
    );
}

#[test]
fn cugraph_bfs_string_source_vertex_uses_logical_output_schema() {
    let config = CugraphSqlConfig::default();
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit("alice"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"output_mode":"normalized"}"#),
        ],
        &config,
    )
    .unwrap();
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        panic!("expected BFS params");
    };
    assert_eq!(params.source_vertex, CugraphBfsSourceVertex::Utf8("alice".to_string()));
    assert_eq!(
        spec.output_schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>(),
        vec![
            Field::new("vertex", DataType::Utf8, false),
            Field::new("distance", DataType::Int64, true),
            Field::new("predecessor", DataType::Utf8, true),
            Field::new("reachable", DataType::Boolean, false),
        ],
    );
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
async fn cugraph_bfs_table_function_scan_rejects_mixed_logical_vertex_domains() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "numeric_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let df = ctx
        .sql(
            "SELECT * FROM cugraph_bfs('numeric_edges', 'a', 'src', 'dst', NULL, \
             '{\"output_mode\":\"normalized\"}')",
        )
        .await
        .unwrap();
    let error = df.create_physical_plan().await.unwrap_err().to_string();
    assert!(
        error.contains("string source_vertex requires a string source column"),
        "unexpected planning error: {error}"
    );
}

#[test]
fn cugraph_sql_planning_parses_defaults_explicit_columns_and_null_weight() {
    let config = CugraphSqlConfig::default();
    let spec = parse_cugraph_algorithm_spec(CugraphAlgorithm::PageRank, &[lit("graph_edges")], &config).unwrap();
    assert_eq!(spec.table_name.to_string(), "graph_edges");
    assert_eq!(spec.src_col, "src");
    assert_eq!(spec.dst_col, "dst");
    assert_eq!(spec.weight_col, None);
    assert_eq!(spec.graph_options.directed, true);
    assert_eq!(spec.graph_options.renumber, true);
    assert_eq!(spec.graph_options.construction_policy, CugraphGraphConstructionPolicy::PythonCugraph);

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Louvain,
        &[
            lit("main.graph_edges"),
            lit("from_id"),
            lit("to_id"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"directed":false,"renumber":false}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.table_name.to_string(), "main.graph_edges");
    assert_eq!(spec.src_col, "from_id");
    assert_eq!(spec.dst_col, "to_id");
    assert_eq!(spec.weight_col, None);
    assert_eq!(spec.graph_options.directed, false);
    assert_eq!(spec.graph_options.renumber, false);

    // WCC must default to directed=false since it requires a symmetric graph.
    let spec =
        parse_cugraph_algorithm_spec(CugraphAlgorithm::WeaklyConnectedComponents, &[lit("edges")], &config).unwrap();
    assert_eq!(spec.graph_options.directed, false, "WCC must default to directed=false");

    // triangle_count_all / core_number / k_core must also default to
    // directed=false so their parity oracles (which Python cuGraph forces
    // undirected) are computed under the same semantics as the adapter.
    for algorithm in [CugraphAlgorithm::TriangleCountAll, CugraphAlgorithm::CoreNumber, CugraphAlgorithm::KCore] {
        let spec = parse_cugraph_algorithm_spec(algorithm, &[lit("edges")], &config).unwrap();
        assert_eq!(spec.graph_options.directed, false, "{} must default to directed=false", algorithm.function_name());
    }
}

#[test]
fn cugraph_sql_planning_parses_algorithm_options() {
    let config = CugraphSqlConfig::default();
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::PageRank,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            lit("weight"),
            lit(r#"{"alpha":0.7,"epsilon":0.001,"max_iterations":25}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.weight_col, Some("weight".to_string()));
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::PageRank(datafusion_nexus::CugraphPageRankParams {
            alpha: 0.7,
            epsilon: 0.001,
            max_iterations: 25,
        })
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit(42_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"depth_limit":3}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Bfs(datafusion_nexus::CugraphBfsParams {
            source_vertex: CugraphBfsSourceVertex::Int64(42),
            depth_limit: Some(3),
            ..Default::default()
        })
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit(42_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"output_mode":"normalized"}"#),
        ],
        &config,
    )
    .unwrap();
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        panic!("expected BFS params");
    };
    assert_eq!(params.output.mode, CugraphBfsOutputMode::Normalized);
    assert!(!params.output.return_target_info);
    assert_eq!(spec.output_schema.field_with_name("distance").unwrap().is_nullable(), true);
    assert!(spec.output_schema.field_with_name("reachable").is_ok());

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit(42_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(
                r#"{"output_mode":"normalized","target_vertices_table":"targets","target_vertex_col":"vertex_id","return_target_info":true}"#,
            ),
        ],
        &config,
    )
    .unwrap();
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        panic!("expected BFS params");
    };
    assert_eq!(params.output.mode, CugraphBfsOutputMode::Normalized);
    assert!(params.output.return_target_info);
    assert!(spec.output_schema.field_with_name("target_found").is_ok());
    assert_eq!(spec.output_schema.field_with_name("target_distance").unwrap().is_nullable(), true);

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit(7_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{
                  "edge_id_col":"edge_id",
                  "include_edges_table":"allowed_edges",
                  "include_edge_src_col":"src",
                  "include_edge_dst_col":"dst",
                  "include_edge_id_col":"edge_id",
                  "exclude_vertices_table":"excluded_vertices",
                  "exclude_vertex_col":"vertex_id",
                  "target_vertices_table":"targets",
                  "target_vertex_col":"vertex_id",
                  "return_target_info":true
                }"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.edge_id_col, Some("edge_id".to_string()));
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        panic!("expected BFS params");
    };
    assert_eq!(params.predicates.exclude_vertices.as_ref().unwrap().table_name.to_string(), "excluded_vertices");
    assert_eq!(params.predicates.exclude_vertices.as_ref().unwrap().column, "vertex_id");
    assert_eq!(params.predicates.target_vertices.as_ref().unwrap().table_name.to_string(), "targets");
    let include_edges = params.predicates.include_edges.as_ref().unwrap();
    assert_eq!(include_edges.table_name.to_string(), "allowed_edges");
    assert_eq!(include_edges.source_col, "src");
    assert_eq!(include_edges.destination_col, "dst");
    assert_eq!(include_edges.edge_id_col.as_deref(), Some("edge_id"));

    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(
                    r#"{"include_vertices_table":"included","include_vertex_col":"vertex_id","exclude_vertices_table":"excluded","exclude_vertex_col":"vertex_id"}"#,
                ),
            ],
            &config,
        ),
        "mutually exclusive",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"include_edge_ids_table":"edge_ids","include_edge_ids_col":"edge_id"}"#),
            ],
            &config,
        ),
        "requires primary option 'edge_id_col'",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"return_target_info":true}"#),
            ],
            &config,
        ),
        "return_target_info requires target_vertices_table",
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Louvain,
        &[lit("graph_edges"), lit("src"), lit("dst"), lit("weight"), lit(r#"{"directed":false,"resolution":2.5}"#)],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Louvain(datafusion_nexus::CugraphLouvainParams { resolution: 2.5 })
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::EigenvectorCentrality,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"epsilon":0.0005,"max_iterations":80}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::EigenvectorCentrality(datafusion_nexus::CugraphEigenvectorCentralityParams {
            epsilon: 0.0005,
            max_iterations: 80,
        })
    );

    // Eigenvector centrality must reject zero max_iterations the same way
    // PageRank does, since the underlying cugraph-rs newtype requires >= 1.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::EigenvectorCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_iterations":0}"#),
            ],
            &config,
        ),
        "max_iterations must be at least 1",
    );

    // epsilon=0 must also be rejected at planning. cugraph-rs's
    // `NonNegativeF64` accepts zero, but libcugraph's convergence test
    // `diff_sum < vertex_count * epsilon` never accepts on a non-empty
    // graph when the tolerance is zero, so the solver burns all
    // max_iterations of GPU work before failing.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::EigenvectorCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"epsilon":0}"#),
            ],
            &config,
        ),
        "epsilon must be positive",
    );

    // Katz centrality accepts alpha/beta/epsilon/max_iterations.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::KatzCentrality,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"alpha":0.02,"beta":0.5,"epsilon":1e-7,"max_iterations":250}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::KatzCentrality(datafusion_nexus::CugraphKatzCentralityParams {
            alpha: 0.02,
            beta: 0.5,
            epsilon: 1.0e-7,
            max_iterations: 250,
        }),
    );

    // Same epsilon=0 reasoning as eigenvector centrality: libcugraph
    // never converges when the tolerance is zero on a non-empty graph.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::KatzCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"epsilon":0}"#),
            ],
            &config,
        ),
        "epsilon must be positive",
    );

    // alpha must be strictly positive — alpha=0 makes Katz centrality
    // degenerate to the constant `beta` vector.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::KatzCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"alpha":0}"#),
            ],
            &config,
        ),
        "alpha must be positive",
    );

    // libcugraph caps alpha at 1 (Katz attenuation factor must be
    // smaller than the inverse of the graph's largest eigenvalue);
    // alpha > 1 fails inside the kernel, so reject at planning.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::KatzCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"alpha":2}"#),
            ],
            &config,
        ),
        "alpha must be <= 1",
    );

    // beta=0 with the zero initial vector reaches normalization with
    // all-zero scores. cugraph-rs's `NonNegativeF64` accepts zero, so
    // the only place left to reject it is the SQL planner.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::KatzCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"beta":0}"#),
            ],
            &config,
        ),
        "beta must be positive",
    );

    // HITS accepts epsilon/max_iterations/normalize. Mirrors
    // PageRank's pattern; `normalize` defaults to false matching the
    // cugraph-rs `HitsParams::default()` convention.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Hits,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"epsilon":0.0001,"max_iterations":50,"normalize":true}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Hits(datafusion_nexus::CugraphHitsParams {
            epsilon: 0.0001,
            max_iterations: 50,
            normalize: true,
        }),
    );

    // epsilon=0 is the same trap as the other centrality solvers:
    // libcugraph never converges on a non-empty graph at zero
    // tolerance, so reject at planning.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Hits,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"epsilon":0}"#),
            ],
            &config,
        ),
        "epsilon must be positive",
    );

    // max_iterations=0 is rejected by cugraph-rs's IterationCount
    // newtype at runtime; surface that at planning so the failure
    // path is consistent with PageRank / eigenvector / Katz.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Hits,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_iterations":0}"#),
            ],
            &config,
        ),
        "max_iterations must be at least 1",
    );

    // SSSP: source_vertex as second positional arg, optional cutoff in JSON.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Sssp,
        &[
            lit("graph_edges"),
            lit(5_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"cutoff":10.5}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Sssp(datafusion_nexus::CugraphSsspParams { source_vertex: 5, cutoff: Some(10.5) }),
    );
    // cutoff < 0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Sssp,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"cutoff":-1.0}"#),
            ],
            &config,
        ),
        "cutoff must be non-negative",
    );

    // CoreNumber: degree_type option.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::CoreNumber,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"degree_type":"in"}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::CoreNumber(datafusion_nexus::CugraphCoreNumberParams {
            degree_type: datafusion_nexus::CugraphKCoreDegreeType::In,
        }),
    );

    // KCore: k and degree_type.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::KCore,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"k":3,"degree_type":"out"}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::KCore(datafusion_nexus::CugraphKCoreParams {
            k: 3,
            degree_type: datafusion_nexus::CugraphKCoreDegreeType::Out,
        }),
    );
    // k=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::KCore,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"k":0}"#),
            ],
            &config,
        ),
        "k_core k must be at least 1",
    );
    // Invalid degree_type rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::CoreNumber,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"degree_type":"both"}"#),
            ],
            &config,
        ),
        "unsupported 'degree_type' value 'both'",
    );

    // PageRank strict: same options as PageRank.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::PageRankStrict,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"alpha":0.9,"epsilon":0.0001,"max_iterations":50}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::PageRankStrict(datafusion_nexus::CugraphPageRankParams {
            alpha: 0.9,
            epsilon: 0.0001,
            max_iterations: 50,
        }),
    );
    // PageRank strict: epsilon=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRankStrict,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"epsilon":0}"#),
            ],
            &config,
        ),
        "PageRank strict epsilon must be positive",
    );

    // Leiden: max_level, resolution, theta, seed.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Leiden,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"max_level":50,"resolution":0.8,"theta":0.5,"seed":42}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Leiden(datafusion_nexus::CugraphLeidenParams {
            max_level: 50,
            resolution: 0.8,
            theta: 0.5,
            seed: 42,
        }),
    );
    // Leiden: max_level=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Leiden,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_level":0}"#),
            ],
            &config,
        ),
        "Leiden max_level must be at least 1",
    );
    // Leiden: resolution=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Leiden,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"resolution":0}"#),
            ],
            &config,
        ),
        "resolution must be positive",
    );

    // ECG: min_weight, ensemble_size, max_level, threshold, resolution, seed.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Ecg,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"min_weight":0.01,"ensemble_size":5,"max_level":50,"threshold":1e-5,"resolution":0.9,"seed":7}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Ecg(datafusion_nexus::CugraphEcgParams {
            min_weight: 0.01,
            ensemble_size: 5,
            max_level: 50,
            threshold: 1e-5,
            resolution: 0.9,
            seed: 7,
        }),
    );
    // ECG: ensemble_size=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Ecg,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"ensemble_size":0}"#),
            ],
            &config,
        ),
        "ECG ensemble_size must be at least 1",
    );

    // SpectralModularityMaximization: n_clusters, n_eigenvectors, seed.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::SpectralModularityMaximization,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"n_clusters":4,"n_eigenvectors":3,"seed":99}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::SpectralModularityMaximization(datafusion_nexus::CugraphSpectralClusteringParams {
            n_clusters: 4,
            n_eigenvectors: 3,
            seed: 99,
            ..Default::default()
        }),
    );
    // SpectralModularityMaximization: n_clusters=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::SpectralModularityMaximization,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"n_clusters":0}"#),
            ],
            &config,
        ),
        "n_clusters must be at least 1",
    );

    // BalancedCutClustering: same params struct as SpectralModularityMaximization.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::BalancedCutClustering,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"n_clusters":3,"seed":11}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::BalancedCutClustering(datafusion_nexus::CugraphSpectralClusteringParams {
            n_clusters: 3,
            seed: 11,
            ..Default::default()
        }),
    );
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
async fn cugraph_bfs_predicate_options_plan_as_exec_side_inputs() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "raw_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "excluded_vertices", "vertex_id", DataType::Int64, false);
    register_edge_memtable(&ctx, "allowed_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let df = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
                  '{\"exclude_vertices_table\":\"excluded_vertices\",\"exclude_vertex_col\":\"vertex_id\",\
                     \"include_edges_table\":\"allowed_edges\",\"include_edge_src_col\":\"src\",\"include_edge_dst_col\":\"dst\"}')",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("predicate_source_count=2"),
        "BFS predicate tables should be planned as cuGraph execution side inputs:\n{plan_display}"
    );
}

#[test]
fn cugraph_sql_planning_parses_construction_policy_options() {
    let config = CugraphSqlConfig::default();

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::PageRank,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"construction_policy":"python_cugraph"}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.graph_options.construction_policy, CugraphGraphConstructionPolicy::PythonCugraph);

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::PageRank,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"construction_policy":"raw_libcugraph"}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.graph_options.construction_policy, CugraphGraphConstructionPolicy::RawLibcugraph);

    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"construction_policy":"networkx"}"#),
            ],
            &config,
        ),
        "unsupported cuGraph construction_policy 'networkx'",
    );
}

#[test]
fn cugraph_sql_planning_rejects_invalid_arguments() {
    let config = CugraphSqlConfig::default();
    assert_error_contains(
        parse_cugraph_algorithm_spec(CugraphAlgorithm::PageRank, &[col("graph_edges")], &config),
        "table_name must be a literal",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(CugraphAlgorithm::PageRank, &[lit("SELECT src FROM edges")], &config),
        "must be a simple table",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"unknown":1}"#),
            ],
            &config,
        ),
        "unknown option 'unknown'",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"alpha":1.2}"#),
            ],
            &config,
        ),
        "alpha must be in [0, 1]",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_iterations":0}"#),
            ],
            &config,
        ),
        "max_iterations must be at least 1",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"depth_limit":-1}"#),
            ],
            &config,
        ),
        "depth_limit must be non-negative",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"output_mode":"wide"}"#),
            ],
            &config,
        ),
        "unsupported BFS output_mode 'wide'",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Louvain,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"resolution":0}"#),
            ],
            &config,
        ),
        "resolution must be positive",
    );
    // MST without weight_col is rejected at planning time.
    assert_error_contains(
        parse_cugraph_algorithm_spec(CugraphAlgorithm::MinimumSpanningTree, &[lit("graph_edges")], &config),
        "requires a weight column",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::MinimumSpanningTree,
            &[lit("graph_edges"), lit("src"), lit("dst"), datafusion::logical_expr::lit(ScalarValue::Null)],
            &config,
        ),
        "requires a weight column",
    );
}

async fn assert_sql_schema(ctx: &SessionContext, sql: &str, algorithm: CugraphAlgorithm) {
    let df = ctx.sql(sql).await.unwrap();
    let actual = df.schema().inner();
    let expected = stable_result_schema(algorithm);
    assert_eq!(actual.fields().len(), expected.fields().len());
    for (actual, expected) in actual.fields().iter().zip(expected.fields()) {
        assert_eq!(actual.name(), expected.name());
        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(actual.is_nullable(), expected.is_nullable());
    }
}

fn assert_error_contains(result: datafusion::common::Result<datafusion_nexus::CugraphAlgorithmSpec>, expected: &str) {
    let error = result.unwrap_err().to_string();
    assert!(error.contains(expected), "expected error containing {expected:?}, got {error:?}",);
}

#[test]
fn cugraph_sql_planning_schema_contracts_are_exact() {
    assert_eq!(
        schema_fields(CugraphAlgorithm::PageRank),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false),],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Bfs),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", DataType::Int64, true),
        ],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Louvain),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("partition", DataType::Int64, false),],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::EigenvectorCentrality),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false),],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::KatzCentrality),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false),],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Hits),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("hub_score", DataType::Float64, false),
            Field::new("authority_score", DataType::Float64, false),
        ],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Sssp),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Float64, false),
            Field::new("predecessor", DataType::Int64, true),
        ],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::DegreesAll),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("in_degree", DataType::Int64, false),
            Field::new("out_degree", DataType::Int64, false),
        ],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::InDegreesAll),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("in_degree", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::OutDegreesAll),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("out_degree", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::WeaklyConnectedComponents),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("label", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::StronglyConnectedComponents),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("label", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::TriangleCountAll),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("triangle_count", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::CoreNumber),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("core_number", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::KCore),
        vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::MinimumSpanningTree),
        vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::PageRankStrict),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false)],
    );
    for algorithm in [
        CugraphAlgorithm::Leiden,
        CugraphAlgorithm::Ecg,
        CugraphAlgorithm::SpectralModularityMaximization,
        CugraphAlgorithm::BalancedCutClustering,
    ] {
        assert_eq!(
            schema_fields(algorithm),
            vec![Field::new("vertex", DataType::Int64, false), Field::new("partition", DataType::Int64, false)],
            "community algorithm {algorithm:?} must expose vertex + partition schema",
        );
    }
}

#[tokio::test]
async fn cugraph_sql_planning_direct_table_louvain_physical_plan_preserves_edge_schema_order() {
    let dir = unique_temp_dir("datafusion-nexus-cugraph-louvain-planning");
    let path = dir.join("tiny_communities.parquet");
    write_tiny_communities(&path);

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_communities", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, partition \
             FROM cugraph_louvain('tiny_communities', 'src', 'dst', 'weight', '{\"directed\":false}') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert_eq!(
        plan.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["vertex", "partition"]
    );
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("construction_policy=python_cugraph"),
        "cuGraph SQL EXPLAIN must expose the selected construction policy:\n{plan_display}"
    );
    assert!(
        plan_display.contains("edge_normalize_inserted=true"),
        "default Python-compatible cuGraph SQL planning should surface the inserted EdgeNormalize step:\n{plan_display}"
    );

    let raw_df = ctx
        .sql(
            "SELECT vertex, partition \
             FROM cugraph_louvain('tiny_communities', 'src', 'dst', 'weight', \
                                  '{\"directed\":false,\"construction_policy\":\"raw_libcugraph\"}') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let raw_plan = raw_df.create_physical_plan().await.unwrap();
    let raw_plan_display = displayable(raw_plan.as_ref()).indent(false).to_string();
    assert!(
        raw_plan_display.contains("construction_policy=raw_libcugraph"),
        "raw cuGraph SQL EXPLAIN must expose the selected construction policy:\n{raw_plan_display}"
    );
    assert!(
        raw_plan_display.contains("edge_normalize_inserted=false"),
        "raw cuGraph SQL planning should expose that EdgeNormalize was skipped:\n{raw_plan_display}"
    );

    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn cugraph_sql_planning_skips_edge_normalize_for_precanonicalized_edge_views() {
    let dir = unique_temp_dir("datafusion-nexus-cugraph-normalized-planning");
    let directed_path = dir.join("tiny_directed_edges.parquet");
    let undirected_path = dir.join("tiny_communities.parquet");
    write_tiny_directed_edges(&directed_path);
    write_tiny_communities(&undirected_path);

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_directed_edges", directed_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    ctx.register_parquet("tiny_communities", undirected_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    ctx.sql(
        "CREATE VIEW canonical_directed_edges AS \
         SELECT src, dst \
         FROM tiny_directed_edges \
         WHERE active \
         GROUP BY src, dst",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx.sql(
        "CREATE VIEW canonical_undirected_edges AS \
         SELECT src, dst \
         FROM ( \
           SELECT src, dst FROM tiny_communities \
           UNION ALL \
           SELECT dst AS src, src AS dst FROM tiny_communities \
         ) e \
         GROUP BY src, dst",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    ctx.sql(
        "CREATE VIEW weighted_min_edges AS \
         SELECT src, dst, MIN(weight) AS weight \
         FROM tiny_directed_edges \
         WHERE active \
         GROUP BY src, dst",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let directed_df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('canonical_directed_edges', 'src', 'dst') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let directed_plan = directed_df.create_physical_plan().await.unwrap();
    let directed_display = displayable(directed_plan.as_ref()).indent(false).to_string();
    assert!(
        directed_display.contains("construction_policy=python_cugraph"),
        "default policy label should stay visible for pre-canonical directed view:\n{directed_display}"
    );
    assert!(
        directed_display.contains("edge_normalize_inserted=false"),
        "directed unweighted GROUP BY src,dst should prove edge dedup provenance:\n{directed_display}"
    );

    let undirected_df = ctx
        .sql(
            "SELECT vertex, partition \
             FROM cugraph_louvain('canonical_undirected_edges', 'src', 'dst', NULL, '{\"directed\":false}') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let undirected_plan = undirected_df.create_physical_plan().await.unwrap();
    let undirected_display = displayable(undirected_plan.as_ref()).indent(false).to_string();
    assert!(
        undirected_display.contains("construction_policy=python_cugraph"),
        "default policy label should stay visible for pre-canonical undirected view:\n{undirected_display}"
    );
    assert!(
        undirected_display.contains("edge_normalize_inserted=false"),
        "undirected unweighted UNION ALL reverse expansion + GROUP BY src,dst should prove full provenance:\n{undirected_display}"
    );

    let weighted_df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('weighted_min_edges', 'src', 'dst', 'weight') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let weighted_plan = weighted_df.create_physical_plan().await.unwrap();
    let weighted_display = displayable(weighted_plan.as_ref()).indent(false).to_string();
    assert!(
        weighted_display.contains("edge_normalize_inserted=true"),
        "weighted MIN(weight) aggregate must not prove Python cuGraph keep-last duplicate semantics:\n{weighted_display}"
    );

    fs::remove_dir_all(dir).ok();
}

// Per-algorithm Python-oracle GPU tests for PageRank, BFS, and Louvain
// previously lived here. They have been subsumed by the registry-driven
// `tests/cugraph/cugraph_parity.rs` harness, which reads checked-in oracle JSON
// produced by `scripts/cugraph_oracle.py`. The composition test below
// stays in place and will move into the harness as a Milestone 2
// composition entry.

#[tokio::test]
async fn cugraph_sql_gpu_pagerank_filter_sort_limit_stays_gpu_resident_until_collect() {
    let dir = unique_temp_dir("datafusion-nexus-cugraph-pagerank-compose");
    let path = dir.join("tiny_directed_edges.parquet");
    write_tiny_directed_edges(&path);

    let Some(oracle) = python_pagerank_oracle(&path) else {
        eprintln!("skipping cuGraph PageRank composition e2e because the Python cuGraph oracle is unavailable");
        fs::remove_dir_all(dir).ok();
        return;
    };
    eprintln!("Python cuGraph oracle versions: {}", oracle.versions);

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_directed_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.sql(
        "CREATE VIEW graph_edges AS \
         SELECT src, dst, weight FROM tiny_directed_edges WHERE active",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('graph_edges', 'src', 'dst', 'weight') \
             WHERE value > 0 \
             ORDER BY value DESC, vertex ASC \
             LIMIT 3",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("NativeCudfComposedExec"),
        "PageRank post-algorithm filter/sort/limit should be a composed native GPU handoff:\n{plan_display}"
    );
    assert!(
        plan_display.contains("CugraphAlgoExec"),
        "composed native plan should keep cuGraph as the GPU-resident producer child:\n{plan_display}"
    );
    let composed =
        find_composed_exec(plan.as_ref()).expect("PageRank composition plan should contain NativeCudfComposedExec");
    let native_plan = composed.native_plan().to_string();
    for expected in ["GpuResidentSource", "Filter", "Sort", "Limit"] {
        assert!(
            native_plan.contains(expected),
            "composed native consumer should include {expected} over the cuGraph result:\n{native_plan}"
        );
    }

    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let actual = pagerank_rows(&batches);
    let mut expected = oracle.rows.clone();
    expected.retain(|row| row.value > 0.0);
    expected.sort_by(|left, right| {
        right.value.partial_cmp(&left.value).unwrap().then_with(|| left.vertex.cmp(&right.vertex))
    });
    expected.truncate(3);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.vertex, expected.vertex);
        let delta = (actual.value - expected.value).abs();
        assert!(
            delta <= 1.0e-6,
            "vertex {} composed PageRank mismatch: actual={} expected={} delta={}",
            actual.vertex,
            actual.value,
            expected.value,
            delta
        );
    }
    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_include_edges_predicate_executes_end_to_end() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_allowed_bfs_edges(&ctx, "allowed_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"include_edges_table\":\"allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(bfs_distance_for(batch, 0), Some(0));
    assert_eq!(bfs_distance_for(batch, 1), None);
    assert_eq!(bfs_distance_for(batch, 2), Some(1));
    assert_eq!(bfs_distance_for(batch, 3), Some(2));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_utf8_include_edges_maps_logical_ids_end_to_end() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_allowed_bfs_edges(&ctx, "allowed_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor, reachable \
             FROM cugraph_bfs('raw_edges', 'a', 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"include_edges_table\":\"allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\",\
                     \"output_mode\":\"normalized\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(string_bfs_distance_for(batch, "a"), Some(0));
    assert_eq!(string_bfs_distance_for(batch, "b"), None);
    assert_eq!(string_bfs_distance_for(batch, "c"), Some(1));
    assert_eq!(string_bfs_distance_for(batch, "d"), Some(2));
    assert_eq!(string_bfs_predecessor_for(batch, "d"), Some("c".to_string()));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_utf8_nullable_edges_drop_actual_null_rows() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_nullable_string_bfs_edges_with_nulls(&ctx, "raw_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor, reachable \
             FROM cugraph_bfs('raw_edges', 'a', 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"output_mode\":\"normalized\"}') \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(string_bfs_distance_for(batch, "a"), Some(0));
    assert_eq!(string_bfs_distance_for(batch, "b"), Some(1));
    assert_eq!(string_bfs_distance_for(batch, "c"), Some(2));
    assert_eq!(string_bfs_distance_for(batch, "d"), Some(3));
    assert_eq!(string_bfs_distance_for(batch, "x"), None);
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_reconstructs_utf8_target_path_end_to_end() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_allowed_bfs_edges(&ctx, "allowed_edges");
    register_string_vertex_memtable(&ctx, "targets", "vertex_id", &["d"]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', 'a', 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\",\
                     \"include_edges_table\":\"allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\"}') \
             ORDER BY path_index",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(string_path_column(batch, 1), vec!["a", "a", "a"]);
    assert_eq!(string_path_column(batch, 2), vec!["d", "d", "d"]);
    assert_eq!(string_path_column(batch, 3), vec!["a", "c", "d"]);
    assert_eq!(int64_column(batch, 0), vec![0, 1, 2]);
    assert_eq!(int64_column(batch, 4), vec![0, 1, 2]);
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

#[tokio::test]
async fn cugraph_sql_gpu_bfs_include_edges_drops_actual_null_predicate_rows() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_nullable_allowed_bfs_edges_with_nulls(&ctx, "allowed_edges");

    // The whitelist enumerates 0->2 and 2->3 plus a null row that cannot name
    // any edge. The null row is dropped (not rejected), leaving the same
    // policy as the clean include_edges test: reachable {0,2,3}, 1 unreachable.
    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"include_edges_table\":\"allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(bfs_distance_for(batch, 0), Some(0));
    assert_eq!(bfs_distance_for(batch, 1), None);
    assert_eq!(bfs_distance_for(batch, 2), Some(1));
    assert_eq!(bfs_distance_for(batch, 3), Some(2));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_target_vertices_with_null_values_is_rejected() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_nullable_vertex_memtable_with_nulls(&ctx, "targets", "vertex_id");

    let error = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err()
        .to_string();
    // Vertex-set predicate values that are actually null are rejected at
    // execution with the value-level diagnostic code carrying the role/column.
    assert!(error.contains("predicate_column_contains_nulls"), "unexpected error: {error}");
    assert!(error.contains("target_vertices"), "error should carry predicate_role: {error}");
    assert!(error.contains("vertex_id"), "error should carry the column name: {error}");
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_include_edges_empty_after_drop_is_explicit_error() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_all_null_allowed_bfs_edges(&ctx, "allowed_edges");

    let error = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"include_edges_table\":\"allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err()
        .to_string();
    // A whitelist that collapses to empty after dropping null rows is a
    // deliberate error, never a silent empty BFS.
    assert!(error.contains("include_edges whitelist is empty"), "unexpected error: {error}");
}

fn schema_fields(algorithm: CugraphAlgorithm) -> Vec<Field> {
    stable_result_schema(algorithm).fields().iter().map(|field| field.as_ref().clone()).collect()
}

fn find_composed_exec(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> Option<&NativeCudfComposedExec> {
    if let Some(composed) = plan.as_any().downcast_ref::<NativeCudfComposedExec>() {
        return Some(composed);
    }
    plan.children().into_iter().find_map(|child| find_composed_exec(child.as_ref()))
}

#[derive(Debug, Clone)]
struct PageRankRow {
    vertex: i64,
    value: f64,
}

#[derive(Debug)]
struct PythonPageRankOracle {
    versions: String,
    rows: Vec<PageRankRow>,
}

fn pagerank_rows(batches: &[RecordBatch]) -> Vec<PageRankRow> {
    let mut rows = Vec::new();
    for batch in batches {
        let vertices = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let values = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for index in 0..batch.num_rows() {
            rows.push(PageRankRow { vertex: vertices.value(index), value: values.value(index) });
        }
    }
    rows
}

fn python_pagerank_oracle(path: &Path) -> Option<PythonPageRankOracle> {
    let script = r#"
import json
import sys

try:
    import cudf
    import cugraph
except Exception as exc:
    print(f"Python cuGraph oracle unavailable: {exc}", file=sys.stderr)
    sys.exit(77)

df = cudf.read_parquet(sys.argv[1])
df = df[df["active"]][["src", "dst", "weight"]]
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight", renumber=True)
result = cugraph.pagerank(graph, alpha=0.85, tol=1.0e-6, max_iter=100)
value_col = "pagerank" if "pagerank" in result.columns else [name for name in result.columns if name != "vertex"][0]
result = result.rename(columns={value_col: "value"})
result["vertex"] = result["vertex"].astype("int64")
result["value"] = result["value"].astype("float64")
result = result.sort_values("vertex")
rows = [{"vertex": int(row.vertex), "value": float(row.value)} for row in result.to_pandas().itertuples()]
print(json.dumps({
    "versions": {
        "cudf": getattr(cudf, "__version__", "unknown"),
        "cugraph": getattr(cugraph, "__version__", "unknown"),
    },
    "rows": rows,
}))
"#;
    let output = match Command::new("python3").arg("-c").arg(script).arg(path).output() {
        Ok(output) => output,
        Err(error) => {
            handle_python_oracle_unavailable(format!("Python cuGraph oracle unavailable: {error}"));
            return None;
        },
    };
    if output.status.code() == Some(77) {
        handle_python_oracle_unavailable(String::from_utf8_lossy(&output.stderr));
        return None;
    }
    assert!(
        output.status.success(),
        "Python cuGraph oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let versions = value["versions"].to_string();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| PageRankRow { vertex: row["vertex"].as_i64().unwrap(), value: row["value"].as_f64().unwrap() })
        .collect();
    Some(PythonPageRankOracle { versions, rows })
}

fn handle_python_oracle_unavailable(message: impl AsRef<str>) {
    if python_oracle_required() {
        panic!("{}", message.as_ref().trim());
    }
    eprintln!("{}", message.as_ref().trim());
}

fn python_oracle_required() -> bool {
    matches!(
        std::env::var("CUDF_CUGRAPH_ORACLE_REQUIRED").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES" | "required" | "REQUIRED")
    )
}

fn write_tiny_directed_edges(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
        Field::new("active", DataType::Boolean, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 0, 1, 2, 3, 0, 4])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 3, 3, 4, 1, 5])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![true, true, true, true, true, true, false])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_tiny_communities(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5, 2, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 0, 4, 5, 3, 3, 3])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 4.0])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn register_edge_memtable(
    ctx: &SessionContext,
    table_name: &str,
    src_type: DataType,
    dst_type: DataType,
    weight_type: DataType,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", src_type.clone(), false),
        Field::new("dst", dst_type.clone(), false),
        Field::new("weight", weight_type.clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![edge_column(&src_type), edge_column(&dst_type), edge_column(&weight_type)],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_nullable_edge_memtable(
    ctx: &SessionContext,
    table_name: &str,
    src_nullable: bool,
    dst_nullable: bool,
    weight_nullable: bool,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, src_nullable),
        Field::new("dst", DataType::Int64, dst_nullable),
        Field::new("weight", DataType::Float64, weight_nullable),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 1])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_nullable_edge_id_memtable(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, true),
        Field::new("dst", DataType::Int64, true),
        Field::new("weight", DataType::Float64, true),
        Field::new("edge_id", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 1])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10_i64, 12])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_vertex_memtable(
    ctx: &SessionContext,
    table_name: &str,
    column_name: &str,
    data_type: DataType,
    nullable: bool,
) {
    let schema = Arc::new(Schema::new(vec![Field::new(column_name, data_type.clone(), nullable)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![edge_column(&data_type)]).unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_directed_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 0, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 3, 3])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_allowed_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Int64, false),
        Field::new("allowed_dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![2_i64, 3])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_nullable_allowed_bfs_edges_with_nulls(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Int64, true),
        Field::new("allowed_dst", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![Some(0_i64), Some(2), None])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(2_i64), Some(3), Some(3)])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_all_null_allowed_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Int64, true),
        Field::new("allowed_dst", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![None, None] as Vec<Option<i64>>)) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(2_i64), Some(3)])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_nullable_vertex_memtable_with_nulls(ctx: &SessionContext, table_name: &str, column_name: &str) {
    let schema = Arc::new(Schema::new(vec![Field::new(column_name, DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![Some(3_i64), None])) as ArrayRef],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_string_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, false), Field::new("dst", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "c"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["b", "c", "d", "d"])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_nullable_string_bfs_edges_with_nulls(ctx: &SessionContext, table_name: &str) {
    let schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, true), Field::new("dst", DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![Some("a"), None, Some("b"), Some("c"), Some("a")])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("b"), Some("x"), Some("c"), Some("d"), None])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_large_string_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::LargeUtf8, false),
        Field::new("dst", DataType::LargeUtf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(LargeStringArray::from(vec!["a", "a", "b", "c"])) as ArrayRef,
            Arc::new(LargeStringArray::from(vec!["b", "c", "d", "d"])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_string_bfs_edges_with_edge_id(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", DataType::Utf8, false),
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["e0", "e1", "e2", "e3"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "a", "b", "c"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["b", "c", "d", "d"])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_string_allowed_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Utf8, false),
        Field::new("allowed_dst", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "c"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["c", "d"])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_string_vertex_memtable(ctx: &SessionContext, table_name: &str, column_name: &str, values: &[&str]) {
    let schema = Arc::new(Schema::new(vec![Field::new(column_name, DataType::Utf8, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(values.to_vec())) as ArrayRef])
            .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn bfs_distance_for(batch: &RecordBatch, vertex: i64) -> Option<i64> {
    let vertices = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let distances = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..batch.num_rows()).find_map(|row| (vertices.value(row) == vertex).then(|| distances.value(row)))
}

fn string_bfs_distance_for(batch: &RecordBatch, vertex: &str) -> Option<i64> {
    let vertices = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let distances = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..batch.num_rows())
        .find_map(|row| (vertices.value(row) == vertex && !distances.is_null(row)).then(|| distances.value(row)))
}

fn string_bfs_predecessor_for(batch: &RecordBatch, vertex: &str) -> Option<String> {
    let vertices = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let predecessors = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    (0..batch.num_rows()).find_map(|row| {
        (vertices.value(row) == vertex && !predecessors.is_null(row)).then(|| predecessors.value(row).to_string())
    })
}

fn int64_column(batch: &RecordBatch, index: usize) -> Vec<i64> {
    let values = batch.column(index).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..batch.num_rows()).map(|row| values.value(row)).collect()
}

fn string_path_column(batch: &RecordBatch, index: usize) -> Vec<&str> {
    let values = batch.column(index).as_any().downcast_ref::<StringArray>().unwrap();
    (0..batch.num_rows()).map(|row| values.value(row)).collect()
}

fn edge_column(data_type: &DataType) -> ArrayRef {
    match data_type {
        DataType::Int32 => Arc::new(Int32Array::from(vec![0_i32, 1])) as ArrayRef,
        DataType::Int64 => Arc::new(Int64Array::from(vec![0_i64, 1])) as ArrayRef,
        DataType::Float64 => Arc::new(Float64Array::from(vec![1.0_f64, 2.0])) as ArrayRef,
        DataType::Utf8 => Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        other => panic!("test helper does not support {other:?}"),
    }
}

async fn assert_validate_error(
    ctx: &SessionContext,
    function_name: &str,
    table_name: &str,
    call_json: &str,
    expected_code: &str,
) -> RecordBatch {
    let sql = format!(
        "SELECT valid, error_code, src_dtype, dst_dtype, weight_dtype \
         FROM cugraph_validate_call('{function_name}', '{table_name}', '{call_json}')"
    );
    let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
    let batch = single_batch(&batches).clone();
    assert!(!bool_value(&batch, 0, 0), "validation should fail for {sql}");
    assert_eq!(string_value(&batch, 1, 0), expected_code, "unexpected validation error for {sql}");
    batch
}

fn single_batch(batches: &[RecordBatch]) -> &RecordBatch {
    assert_eq!(batches.len(), 1, "expected one RecordBatch");
    &batches[0]
}

fn schema_field_names(schema: &Schema) -> Vec<&str> {
    schema.fields().iter().map(|field| field.name().as_str()).collect()
}

fn string_value(batch: &RecordBatch, column: usize, row: usize) -> &str {
    batch.column(column).as_any().downcast_ref::<StringArray>().unwrap().value(row)
}

fn bool_value(batch: &RecordBatch, column: usize, row: usize) -> bool {
    batch.column(column).as_any().downcast_ref::<BooleanArray>().unwrap().value(row)
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
