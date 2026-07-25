use super::{support::*, *};

#[tokio::test]
async fn cugraph_sql_planning_preserves_structured_logical_edge_id_capability_errors() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges_with_edge_id(&ctx, "string_edges");
    register_string_vertex_memtable(&ctx, "edge_ids", "edge_id", &["e0"]);

    for (options, expected_include_edge_ids) in [
        (r#"{"edge_id_col":"edge_id"}"#, "false"),
        (r#"{"edge_id_col":"edge_id","include_edge_ids_table":"edge_ids","include_edge_ids_col":"edge_id"}"#, "true"),
    ] {
        let df = ctx
            .sql(&format!("SELECT * FROM cugraph_bfs('string_edges', 'a', 'src', 'dst', NULL, '{options}')"))
            .await
            .expect("logical BFS SQL should parse before physical capability preflight");
        let error =
            df.create_physical_plan().await.expect_err("logical edge-id BFS must fail physical capability preflight");
        let nexus = typed_nexus_error(&error);

        assert_eq!(nexus.code(), ErrorCode::RequiredGpuOperationUnsupported);
        assert_eq!(nexus.fact_value("edge_id_col"), Some("true"));
        assert_eq!(nexus.fact_value("include_edge_ids"), Some(expected_include_edge_ids));
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
    assert_sql_schema(
        &ctx,
        r#"SELECT * FROM cugraph_personalized_pagerank('graph_edges', 'src', 'dst', NULL, '{"personalization_table":"ppr_seeds","personalization_vertex_col":"vertex","personalization_value_col":"value"}')"#,
        CugraphAlgorithm::PersonalizedPageRank,
    )
    .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_leiden('graph_edges')", CugraphAlgorithm::Leiden).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_ecg('graph_edges')", CugraphAlgorithm::Ecg).await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_spectral_modularity_maximization('graph_edges')",
        CugraphAlgorithm::SpectralModularityMaximization,
    )
    .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_force_atlas2('graph_edges')", CugraphAlgorithm::ForceAtlas2).await;
}

#[tokio::test]
async fn cugraph_similarity_sql_resolves_the_pair_relation_as_an_explicit_second_input() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "similarity_edges", DataType::Int32, DataType::Int32, DataType::Float64);
    register_vertex_pair_memtable(&ctx, "candidate_pairs", DataType::Int32, DataType::Int32, false);
    register_vertex_pair_memtable(&ctx, "nullable_pairs", DataType::Int32, DataType::Int32, true);
    register_vertex_pair_memtable(&ctx, "mismatched_pairs", DataType::Int32, DataType::Int64, false);
    register_vertex_pair_memtable(&ctx, "int64_pairs", DataType::Int64, DataType::Int64, false);
    register_vertex_pair_memtable(&ctx, "utf8_pairs", DataType::Utf8, DataType::Utf8, false);

    for (function, algorithm) in [
        ("cugraph_jaccard", CugraphAlgorithm::Jaccard),
        ("cugraph_sorensen", CugraphAlgorithm::Sorensen),
        ("cugraph_overlap", CugraphAlgorithm::Overlap),
        ("cugraph_cosine", CugraphAlgorithm::Cosine),
    ] {
        let sql = format!(
            "SELECT * FROM {function}('similarity_edges', 'src', 'dst', NULL, \
             '{{\"vertex_pairs_table\":\"candidate_pairs\",\"first_vertex_col\":\"first\",\"second_vertex_col\":\"second\"}}')"
        );
        assert_sql_schema(&ctx, &sql, algorithm).await;
        let plan = ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();
        let cugraph =
            find_cugraph_algo_exec(plan.as_ref()).expect("similarity must plan as an explicit cuGraph island");
        assert_eq!(cugraph.children().len(), 2, "{function} must retain independent edge and candidate-pair inputs",);

        let nullable_sql = format!(
            "SELECT * FROM {function}('similarity_edges', 'src', 'dst', NULL, \
             '{{\"vertex_pairs_table\":\"nullable_pairs\",\"first_vertex_col\":\"first\",\"second_vertex_col\":\"second\"}}')"
        );
        ctx.sql(&nullable_sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .expect("nullable but null-free pair schemas must be admitted until execution");

        for (pairs, first, second, expected) in [
            ("missing_pairs", "first", "second", "No table named 'missing_pairs' for cuGraph side-input source"),
            ("candidate_pairs", "missing", "second", "missing first vertex column 'missing'"),
            ("mismatched_pairs", "first", "second", "must have the same dtype"),
            ("int64_pairs", "first", "second", "must have dtype Int32"),
            ("utf8_pairs", "first", "second", "must have dtype Int32"),
        ] {
            let sql = format!(
                "SELECT * FROM {function}('similarity_edges', 'src', 'dst', NULL, \
                 '{{\"vertex_pairs_table\":\"{pairs}\",\"first_vertex_col\":\"{first}\",\"second_vertex_col\":\"{second}\"}}')"
            );
            let error = match ctx.sql(&sql).await {
                Ok(df) => df.create_physical_plan().await.unwrap_err(),
                Err(error) => error,
            };
            assert!(error.to_string().contains(expected), "{function}: {error}");
        }
    }
}

#[tokio::test]
async fn cugraph_strongly_connected_components_rejects_undirected_graph_option() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let error = ctx
        .sql(
            "SELECT * FROM cugraph_strongly_connected_components(\
             'graph_edges', 'src', 'dst', 'weight', \
             '{\"directed\":false,\"renumber\":true,\"construction_policy\":\"python_cugraph\"}')",
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("cugraph_strongly_connected_components requires directed=true"), "{error}");
    assert!(error.contains("cugraph_weakly_connected_components"), "{error}");
}

#[tokio::test]
async fn cugraph_undirected_only_algorithms_reject_directed_graph_option() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    for (sql, expected) in [
        (
            "SELECT * FROM cugraph_minimum_spanning_tree(\
             'graph_edges', 'src', 'dst', 'weight', '{\"directed\":true}')",
            "cugraph_minimum_spanning_tree requires directed=false",
        ),
        (
            "SELECT * FROM cugraph_spectral_modularity_maximization(\
             'graph_edges', 'src', 'dst', NULL, '{\"directed\":true}')",
            "cugraph_spectral_modularity_maximization requires directed=false",
        ),
    ] {
        let error = ctx.sql(sql).await.unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[tokio::test]
async fn cugraph_table_function_scan_rejects_int64_for_legacy_int32_only_algorithms() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges_i64", DataType::Int64, DataType::Int64, DataType::Float64);
    register_edge_memtable(&ctx, "edges_i32", DataType::Int32, DataType::Int32, DataType::Float64);

    for (int64_sql, int32_sql, expected_algorithm) in [
        (
            "SELECT * FROM cugraph_minimum_spanning_tree('edges_i64', 'src', 'dst', 'weight')",
            "SELECT * FROM cugraph_minimum_spanning_tree('edges_i32', 'src', 'dst', 'weight')",
            "cugraph_minimum_spanning_tree",
        ),
        (
            "SELECT * FROM cugraph_spectral_modularity_maximization('edges_i64')",
            "SELECT * FROM cugraph_spectral_modularity_maximization('edges_i32')",
            "cugraph_spectral_modularity_maximization",
        ),
    ] {
        let int64_df = ctx.sql(int64_sql).await.unwrap();
        let error = int64_df.create_physical_plan().await.unwrap_err();
        let nexus = typed_nexus_error(&error);
        assert_eq!(nexus.code(), ErrorCode::RequiredGpuOperationUnsupported);
        assert_eq!(nexus.fact_value("algorithm"), Some(expected_algorithm));
        assert_eq!(nexus.fact_value("vertex_dtype"), Some("Int64"));
        assert_eq!(nexus.fact_value("supported_vertex_dtypes"), Some("Int32"));

        let int32_df = ctx.sql(int32_sql).await.unwrap();
        int32_df.create_physical_plan().await.unwrap();
    }

    let malformed = ctx
        .sql("SELECT * FROM cugraph_minimum_spanning_tree('edges_i64', 'missing_src', 'dst', 'weight')")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap_err();
    assert!(malformed.find_nexus_error().is_none(), "missing columns must remain a planning error: {malformed}");
    assert!(malformed.to_string().contains("missing source column 'missing_src'"));
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
    assert!(is_null_value(batch, 1, 0));
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

    let error = ctx.sql("SELECT * FROM cugraph_describe_algorithm('cugraph_bfs_path')").await.unwrap_err().to_string();
    assert!(error.contains("unknown cuGraph algorithm function 'cugraph_bfs_path'"));

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

#[tokio::test]
async fn cugraph_bfs_table_function_scan_rejects_mixed_logical_vertex_domains() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
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

#[tokio::test]
async fn cugraph_bfs_relation_source_string_domain_exposes_utf8_logical_schema() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_vertex_memtable(&ctx, "bfs_sources", "vertex_id", &["a"]);

    let df = ctx
        .sql(
            "SELECT * FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
             '{\"source_vertices_table\":\"bfs_sources\",\
                \"source_vertex_col\":\"vertex_id\",\
                \"output_mode\":\"normalized\"}')",
        )
        .await
        .unwrap();
    assert_eq!(
        df.schema().inner().fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>(),
        vec![
            Field::new("vertex", DataType::Utf8, false),
            Field::new("distance", DataType::Int64, true),
            Field::new("predecessor", DataType::Utf8, true),
            Field::new("reachable", DataType::Boolean, false),
        ],
    );
}

#[tokio::test]
async fn cugraph_bfs_predicate_options_plan_as_exec_side_inputs() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
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
        schema_fields(CugraphAlgorithm::PersonalizedPageRank),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false)],
    );
    for algorithm in [CugraphAlgorithm::Leiden, CugraphAlgorithm::Ecg, CugraphAlgorithm::SpectralModularityMaximization]
    {
        assert_eq!(
            schema_fields(algorithm),
            vec![Field::new("vertex", DataType::Int64, false), Field::new("partition", DataType::Int64, false)],
            "community algorithm {algorithm:?} must expose vertex + partition schema",
        );
    }
    assert_eq!(
        schema_fields(CugraphAlgorithm::ForceAtlas2),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
            Field::new("y", DataType::Float32, false),
        ],
    );
}

#[tokio::test]
async fn cugraph_sql_planning_direct_table_louvain_physical_plan_preserves_edge_schema_order() {
    let dir = unique_temp_dir("datafusion-nexus-cugraph-louvain-planning");
    let path = dir.join("tiny_communities.parquet");
    write_tiny_communities(&path);

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
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
        .with_nexus_native(NexusNativeOptimizerConfig::default())
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
    ctx.sql(
        "CREATE VIEW grouped_by_extra_edge_attr AS \
         SELECT src, dst \
         FROM tiny_directed_edges \
         WHERE active \
         GROUP BY src, dst, weight",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx.sql(
        "CREATE VIEW projected_duplicate_edges AS \
         SELECT src, src AS dst \
         FROM ( \
           SELECT src, dst \
           FROM tiny_directed_edges \
           WHERE active \
           GROUP BY src, dst \
         ) e",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx.sql(
        "CREATE VIEW aggregate_alias_overwrites_dst_edges AS \
         SELECT src, MAX(1) AS dst \
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
        "CREATE VIEW aggregate_over_canonical_edges AS \
         SELECT src, COUNT(*) AS dst \
         FROM canonical_directed_edges \
         GROUP BY src",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx.sql(
        "CREATE VIEW filtered_reverse_expansion_edges AS \
         SELECT src, dst \
         FROM ( \
           SELECT src, dst FROM tiny_communities WHERE src > 0 \
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
    register_edge_memtable(&ctx, "memory_forward_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_edge_memtable(&ctx, "memory_reverse_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    ctx.sql(
        "CREATE VIEW mixed_memory_reverse_edges AS \
         SELECT src, dst \
         FROM ( \
           SELECT src, dst FROM memory_forward_edges \
           UNION ALL \
           SELECT dst AS src, src AS dst FROM memory_reverse_edges \
         ) e \
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

    let grouped_by_extra_df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('grouped_by_extra_edge_attr', 'src', 'dst') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let grouped_by_extra_plan = grouped_by_extra_df.create_physical_plan().await.unwrap();
    let grouped_by_extra_display = displayable(grouped_by_extra_plan.as_ref()).indent(false).to_string();
    assert!(
        grouped_by_extra_display.contains("edge_normalize_inserted=true"),
        "GROUP BY src,dst plus an extra key must not prove edge dedup provenance:\n{grouped_by_extra_display}"
    );

    let projected_duplicate_df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('projected_duplicate_edges', 'src', 'dst') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let projected_duplicate_plan = projected_duplicate_df.create_physical_plan().await.unwrap();
    let projected_duplicate_display = displayable(projected_duplicate_plan.as_ref()).indent(false).to_string();
    assert!(
        projected_duplicate_display.contains("edge_normalize_inserted=true"),
        "non-bijective projection above GROUP BY src,dst must not preserve dedup provenance:\n{projected_duplicate_display}"
    );

    let aggregate_alias_df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('aggregate_alias_overwrites_dst_edges', 'src', 'dst') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let aggregate_alias_plan = aggregate_alias_df.create_physical_plan().await.unwrap();
    let aggregate_alias_display = displayable(aggregate_alias_plan.as_ref()).indent(false).to_string();
    assert!(
        aggregate_alias_display.contains("edge_normalize_inserted=true"),
        "aggregate alias for dst must not inherit GROUP BY src,dst dedup proof:\n{aggregate_alias_display}"
    );

    let aggregate_over_canonical_df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('aggregate_over_canonical_edges', 'src', 'dst') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let aggregate_over_canonical_plan = aggregate_over_canonical_df.create_physical_plan().await.unwrap();
    let aggregate_over_canonical_display =
        displayable(aggregate_over_canonical_plan.as_ref()).indent(false).to_string();
    assert!(
        aggregate_over_canonical_display.contains("edge_normalize_inserted=true"),
        "non-edge-preserving aggregate must not inherit stale canonical facts:\n{aggregate_over_canonical_display}"
    );

    let filtered_reverse_df = ctx
        .sql(
            "SELECT vertex, partition \
             FROM cugraph_louvain('filtered_reverse_expansion_edges', 'src', 'dst', NULL, '{\"directed\":false}') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let filtered_reverse_plan = filtered_reverse_df.create_physical_plan().await.unwrap();
    let filtered_reverse_display = displayable(filtered_reverse_plan.as_ref()).indent(false).to_string();
    assert!(
        filtered_reverse_display.contains("edge_normalize_inserted=true"),
        "filtered forward branch must not prove full reverse-edge expansion:\n{filtered_reverse_display}"
    );

    let mixed_memory_df = ctx
        .sql(
            "SELECT vertex, partition \
             FROM cugraph_louvain('mixed_memory_reverse_edges', 'src', 'dst', NULL, '{\"directed\":false}') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let mixed_memory_plan = mixed_memory_df.create_physical_plan().await.unwrap();
    let mixed_memory_display = displayable(mixed_memory_plan.as_ref()).indent(false).to_string();
    assert!(
        mixed_memory_display.contains("edge_normalize_inserted=true"),
        "reverse projection from a different memory source must not prove reverse-edge expansion:\n{mixed_memory_display}"
    );

    fs::remove_dir_all(dir).ok();
}
