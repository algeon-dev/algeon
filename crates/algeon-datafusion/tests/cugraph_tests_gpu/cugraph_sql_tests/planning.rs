use std::{fs, sync::Arc};

use algeon_datafusion::{
    error::ClientDetail, gpu_coverage::validate_query, planner::AlgeonGpuExecutionMode, report::CandidateOutcomeKind,
};

use super::{support::*, *};

fn register_exact_columns(ctx: &SessionContext, table_name: &str, columns: &[(&str, DataType)]) {
    let schema = Arc::new(Schema::new(
        columns.iter().map(|(name, data_type)| Field::new(*name, data_type.clone(), false)).collect::<Vec<_>>(),
    ));
    let arrays = columns.iter().map(|(_, data_type)| edge_column(data_type)).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
    ctx.register_table(table_name, Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
}

#[tokio::test]
async fn cugraph_sql_planning_preserves_structured_logical_edge_id_capability_errors() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges_with_edge_id(&ctx, "string_edges");
    register_string_vertex_memtable(&ctx, "edge_ids", "edge_id", &["e0"]);

    for (sql, expected_include_edge_ids) in [
        (
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst, edge_id FROM string_edges), source_vertex => 'a')",
            "false",
        ),
        (
            "SELECT * FROM cugraph_bfs(\
                edges => (SELECT src, dst, edge_id FROM string_edges), source_vertex => 'a',\
                include_edge_ids => (SELECT edge_id FROM edge_ids))",
            "true",
        ),
    ] {
        let error = ctx.sql(sql).await.expect_err("logical edge-id BFS must fail during logical capability preflight");
        let algeon = typed_algeon_error(&error);

        assert_eq!(algeon.code(), ErrorCode::RequiredGpuOperationUnsupported);
        assert_eq!(algeon.fact_value("edge_id_col"), Some("true"));
        assert_eq!(algeon.fact_value("include_edge_ids"), Some(expected_include_edge_ids));
    }
}

#[tokio::test]
async fn cugraph_sql_planning_exposes_stable_output_schemas() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_edge_memtable(&ctx, "graph_edges_i32", DataType::Int32, DataType::Int32, DataType::Float64);
    register_personalization_memtable(&ctx, "ppr_seeds", DataType::Int64);

    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_pagerank(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::PageRank,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM graph_edges), source_vertex => 0)",
        CugraphAlgorithm::Bfs,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_louvain(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::Louvain,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_eigenvector_centrality(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::EigenvectorCentrality,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_katz_centrality(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::KatzCentrality,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_hits(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::Hits,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_sssp(edges => (SELECT src, dst FROM graph_edges), source_vertex => 0)",
        CugraphAlgorithm::Sssp,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_degrees_all(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::DegreesAll,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_in_degrees_all(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::InDegreesAll,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_out_degrees_all(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::OutDegreesAll,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_weakly_connected_components(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::WeaklyConnectedComponents,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_strongly_connected_components(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::StronglyConnectedComponents,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_triangle_count_all(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::TriangleCountAll,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_core_number(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::CoreNumber,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_k_core(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::KCore,
    )
    .await;
    assert_sql_schema_with_vertex_type(
        &ctx,
        "SELECT * FROM cugraph_minimum_spanning_tree(edges => (SELECT src, dst, weight FROM graph_edges_i32))",
        CugraphAlgorithm::MinimumSpanningTree,
        &DataType::Int32,
        &["src", "dst"],
    )
    .await;
    assert_sql_schema(
        &ctx,
        r#"SELECT * FROM cugraph_personalized_pagerank(edges => (SELECT src, dst FROM graph_edges), personalization => (SELECT vertex, value FROM ppr_seeds))"#,
        CugraphAlgorithm::PersonalizedPageRank,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_leiden(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::Leiden,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_ecg(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::Ecg,
    )
    .await;
    assert_sql_schema_with_vertex_type(
        &ctx,
        "SELECT * FROM cugraph_spectral_modularity_maximization(edges => (SELECT src, dst FROM graph_edges_i32))",
        CugraphAlgorithm::SpectralModularityMaximization,
        &DataType::Int32,
        &["vertex"],
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_force_atlas2(edges => (SELECT src, dst FROM graph_edges))",
        CugraphAlgorithm::ForceAtlas2,
    )
    .await;
}

#[tokio::test]
async fn cugraph_similarity_sql_resolves_the_pair_relation_as_an_explicit_second_input() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
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
            "SELECT * FROM {function}(\
             edges => (SELECT src, dst FROM similarity_edges),\
             vertex_pairs => (SELECT first, second FROM candidate_pairs))"
        );
        assert_sql_schema_with_vertex_type(&ctx, &sql, algorithm, &DataType::Int32, &["first", "second"]).await;
        let plan = ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();
        let cugraph =
            find_cugraph_algo_exec(plan.as_ref()).expect("similarity must plan as an explicit cuGraph island");
        assert_eq!(cugraph.children().len(), 2, "{function} must retain independent edge and candidate-pair inputs",);

        let nullable_sql = format!(
            "SELECT * FROM {function}(\
             edges => (SELECT src, dst FROM similarity_edges),\
             vertex_pairs => (SELECT first, second FROM nullable_pairs))"
        );
        ctx.sql(&nullable_sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .expect("nullable but null-free pair schemas must be admitted until execution");

        for (pairs, first, second, expected) in [
            ("missing_pairs", "first", "second", "missing_pairs' not found"),
            ("candidate_pairs", "missing", "second", "No field named missing"),
            ("mismatched_pairs", "first", "second", "must have the same dtype"),
            ("int64_pairs", "first", "second", "must have dtype Int32"),
            ("utf8_pairs", "first", "second", "must have dtype Int32"),
        ] {
            let sql = format!(
                "SELECT * FROM {function}(\
                 edges => (SELECT src, dst FROM similarity_edges),\
                 vertex_pairs => (SELECT {first}, {second} FROM {pairs}))"
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
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let error = ctx
        .sql(
            "SELECT * FROM cugraph_strongly_connected_components(\
             edges => (SELECT src, dst, weight FROM graph_edges),\
             directed => false, renumber => true, construction_policy => 'python_cugraph')",
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("cugraph_strongly_connected_components requires directed=true"), "{error}");
    assert!(error.contains("cugraph_weakly_connected_components"), "{error}");
}

#[tokio::test]
async fn cugraph_undirected_only_algorithms_reject_directed_graph_option() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    for (sql, expected) in [
        (
            "SELECT * FROM cugraph_minimum_spanning_tree(\
             edges => (SELECT src, dst, weight FROM graph_edges), directed => true)",
            "cugraph_minimum_spanning_tree requires directed=false",
        ),
        (
            "SELECT * FROM cugraph_spectral_modularity_maximization(\
             edges => (SELECT src, dst FROM graph_edges), directed => true)",
            "cugraph_spectral_modularity_maximization requires directed=false",
        ),
    ] {
        let error = ctx.sql(sql).await.unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[tokio::test]
async fn cugraph_table_function_scan_plans_int64_and_string_vertices_for_legacy_int32_algorithms() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(AlgeonNativeOptimizerConfig::default())
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges_i32", DataType::Int32, DataType::Int32, DataType::Float64);
    register_edge_memtable(&ctx, "edges_i64", DataType::Int64, DataType::Int64, DataType::Float64);
    register_edge_memtable(&ctx, "edges_utf8", DataType::Utf8, DataType::Utf8, DataType::Float64);

    for (algorithm, columns, vertex_columns) in [
        (CugraphAlgorithm::MinimumSpanningTree, "src, dst, weight", ["src", "dst"].as_slice()),
        (CugraphAlgorithm::SpectralModularityMaximization, "src, dst", ["vertex"].as_slice()),
    ] {
        for (table, vertex_type) in
            [("edges_i32", DataType::Int32), ("edges_i64", DataType::Int64), ("edges_utf8", DataType::Utf8)]
        {
            let sql = format!("SELECT * FROM {}(edges => (SELECT {columns} FROM {table}))", algorithm.function_name());
            assert_sql_schema_with_vertex_type(&ctx, &sql, algorithm, &vertex_type, vertex_columns).await;
            ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap_or_else(|error| {
                panic!("{} over {vertex_type:?} vertices must plan natively: {error}", algorithm.function_name())
            });
        }
    }

    let malformed = ctx
        .sql("SELECT * FROM cugraph_minimum_spanning_tree(edges => (SELECT missing_src AS src, dst, weight FROM edges_i64))")
        .await
        .unwrap_err();
    assert!(malformed.to_string().contains("missing_src"));
}

#[tokio::test]
async fn cugraph_result_columns_follow_the_documented_naming_in_descriptors_and_planned_schemas() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let cases: [(&str, &[&str]); 8] = [
        ("cugraph_louvain", &["vertex", "community"]),
        ("cugraph_leiden", &["vertex", "community"]),
        ("cugraph_ecg", &["vertex", "community"]),
        ("cugraph_spectral_modularity_maximization", &["vertex", "community"]),
        ("cugraph_weakly_connected_components", &["vertex", "component"]),
        ("cugraph_strongly_connected_components", &["vertex", "component"]),
        ("cugraph_edge_betweenness_centrality", &["src", "dst", "value"]),
        ("cugraph_pagerank", &["vertex", "value"]),
    ];
    for (function, expected) in cases {
        let sql = format!("SELECT * FROM {function}(edges => (SELECT src, dst FROM graph_edges))");
        let planned = ctx.sql(&sql).await.unwrap();
        assert_eq!(schema_field_names(planned.schema().inner()), expected, "{function} planned schema");

        let batches = ctx
            .sql(&format!("SELECT result_schemas_json FROM gpu_describe_function('{function}')"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let schemas = json_value(string_value(single_batch(&batches), 0, 0));
        let described = schemas["schemas"]["default"]["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field["name"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(described, expected, "{function} described schema");
    }
}

#[tokio::test]
async fn cugraph_bfs_path_mode_exposes_path_schema_through_canonical_function() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "targets", "vertex", DataType::Int64, false);

    let df = ctx
        .sql(
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT src, dst FROM graph_edges), source_vertex => 0,\
             output_mode => 'path', target_vertices => (SELECT vertex FROM targets))",
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
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let error = ctx
        .sql("SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM graph_edges), source_vertex => 0, output_mode => 'path')")
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("BFS path output requires target_vertices"));
}

#[tokio::test]
async fn cugraph_bfs_path_name_is_not_registered_or_aliased() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let error = ctx
        .sql("SELECT * FROM cugraph_bfs_path(edges => (SELECT src, dst FROM graph_edges), source_vertex => 0)")
        .await
        .unwrap_err()
        .to_string();
    assert!(!ctx.state().table_functions().contains_key("cugraph_bfs_path"));
    assert!(error.contains("unknown GPU function cugraph_bfs_path; closest registered name: cugraph_bfs;"), "{error}");

    let error = ctx.sql("SELECT * FROM gpu_describe_function('cugraph_bfs_path')").await.unwrap_err().to_string();
    assert!(error.contains("unknown GPU function cugraph_bfs_path;"), "{error}");

    let batches = ctx
        .sql(
            r#"SELECT valid, error_code, message
               FROM gpu_validate_call(
                 'cugraph_bfs_path',
                 '{"relations":{"edges":{"table":"raw_edges"}},"options":{"source_vertex":0}}'
               )"#,
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert!(!bool_value(batch, 0, 0));
    assert_eq!(string_value(batch, 1, 0), "unknown_function");
    assert!(string_value(batch, 2, 0).contains("cugraph_bfs_path"));
}

#[tokio::test]
async fn cugraph_bfs_default_and_explicit_vertices_share_the_nullable_schema() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM graph_edges), source_vertex => 0)",
        CugraphAlgorithm::Bfs,
    )
    .await;

    let df = ctx
        .sql(
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT src, dst FROM graph_edges), source_vertex => 0, output_mode => 'vertices')",
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
        .with_algeon_native(AlgeonNativeOptimizerConfig::default())
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "numeric_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let error = ctx
        .sql(
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT src, dst FROM numeric_edges), source_vertex => 'a', output_mode => 'vertices')",
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("string vertex selector requires a string source column"),
        "unexpected planning error: {error}"
    );
}

#[tokio::test]
async fn cugraph_bfs_relation_source_string_domain_exposes_utf8_logical_schema() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_vertex_memtable(&ctx, "bfs_sources", "vertex_id", &["a"]);

    let df = ctx
        .sql(
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT src, dst FROM raw_edges),\
             sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'vertices')",
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
async fn cugraph_relation_arguments_reject_outer_query_correlation() {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "outer_vertices", "vertex", DataType::Int64, false);

    let error = ctx
        .sql(
            "SELECT outer_vertices.vertex, graph.distance \
             FROM outer_vertices \
             JOIN cugraph_bfs(\
                 edges => (SELECT src, dst FROM graph_edges WHERE src = outer_vertices.vertex),\
                 source_vertex => 0) AS graph \
             ON outer_vertices.vertex = graph.vertex",
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("outer_vertices"), "unexpected correlation error: {error}");
}

#[tokio::test]
async fn cugraph_bfs_string_subplans_lower_to_native_gpu_producers() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(AlgeonNativeOptimizerConfig::default())
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_vertex_memtable(&ctx, "bfs_sources", "vertex_id", &["a"]);
    register_string_vertex_memtable(&ctx, "included", "vertex_id", &["a", "b"]);

    let df = ctx
        .sql(
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT src, dst FROM raw_edges),\
             sources => (SELECT vertex_id AS vertex FROM bfs_sources),\
             include_vertices => (SELECT vertex_id AS vertex FROM included), output_mode => 'vertices')",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let cugraph = find_cugraph_algo_exec(plan.as_ref()).expect("logical BFS must retain its graph producer boundary");
    assert_eq!(cugraph.children().len(), 3, "edge, source, and include relations remain graph inputs");
    assert!(
        cugraph.children().iter().all(|child| cugraph_input_is_native_gpu_producer(child.as_ref())),
        "logical BFS edge and side inputs must be planned as native GPU-resident producers"
    );
}

#[tokio::test]
async fn cugraph_bfs_predicate_options_plan_as_exec_side_inputs() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(AlgeonNativeOptimizerConfig::default())
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "raw_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "excluded_vertices", "vertex_id", DataType::Int64, false);
    register_edge_memtable(&ctx, "allowed_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let df = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertex => 0,\
                  exclude_vertices => (SELECT vertex_id AS vertex FROM excluded_vertices),\
                  include_edges => (SELECT src, dst FROM allowed_edges))",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("predicate_source_count=2"),
        "BFS predicate tables should be planned as cuGraph execution side inputs:\n{plan_display}"
    );
    let cugraph = find_cugraph_algo_exec(plan.as_ref()).expect("numeric BFS must retain its graph boundary");
    assert!(
        cugraph.children().iter().all(|child| cugraph_input_is_native_gpu_producer(child.as_ref())),
        "numeric BFS edge and predicate relations must be planned as native GPU-resident producers"
    );
}

#[tokio::test]
async fn cugraph_relation_projections_preserve_exact_camel_case_columns() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(AlgeonNativeOptimizerConfig::default())
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_exact_columns(
        &ctx,
        "camel_edges",
        &[
            ("edgeSrc", DataType::Int64),
            ("edgeDst", DataType::Int64),
            ("edgeWeight", DataType::Float64),
            ("edgeKey", DataType::Int64),
        ],
    );
    register_exact_columns(&ctx, "camel_source", &[("sourceVertex", DataType::Int64)]);
    register_exact_columns(&ctx, "camel_include", &[("includeVertex", DataType::Int64)]);
    register_exact_columns(&ctx, "camel_exclude", &[("excludeVertex", DataType::Int64)]);
    register_exact_columns(&ctx, "camel_target", &[("targetVertex", DataType::Int64)]);
    register_exact_columns(
        &ctx,
        "camel_allowed_edges",
        &[("edgeSource", DataType::Int64), ("edgeDestination", DataType::Int64), ("edgeId", DataType::Int64)],
    );
    register_exact_columns(&ctx, "camel_seeds", &[("seedVertex", DataType::Int64), ("seedValue", DataType::Float64)]);
    register_exact_columns(&ctx, "camel_pairs", &[("pairLeft", DataType::Int64), ("pairRight", DataType::Int64)]);

    let sqls = [
        (
            r#"SELECT * FROM cugraph_bfs(
                edges => (SELECT "edgeSrc" AS src, "edgeDst" AS dst, "edgeWeight" AS weight FROM camel_edges),
                sources => (SELECT "sourceVertex" AS vertex FROM camel_source))"#,
            "BFS source relation",
        ),
        (
            r#"SELECT * FROM cugraph_bfs(
                edges => (SELECT "edgeSrc" AS src, "edgeDst" AS dst, "edgeWeight" AS weight FROM camel_edges),
                source_vertex => 0, include_vertices => (SELECT "includeVertex" AS vertex FROM camel_include))"#,
            "BFS include relation",
        ),
        (
            r#"SELECT * FROM cugraph_bfs(
                edges => (SELECT "edgeSrc" AS src, "edgeDst" AS dst, "edgeWeight" AS weight FROM camel_edges),
                source_vertex => 0, exclude_vertices => (SELECT "excludeVertex" AS vertex FROM camel_exclude))"#,
            "BFS exclude relation",
        ),
        (
            r#"SELECT * FROM cugraph_bfs(
                edges => (SELECT "edgeSrc" AS src, "edgeDst" AS dst, "edgeWeight" AS weight FROM camel_edges),
                source_vertex => 0, output_mode => 'path', target_vertices => (SELECT "targetVertex" AS vertex FROM camel_target))"#,
            "BFS target relation",
        ),
        (
            r#"SELECT * FROM cugraph_bfs(
                edges => (SELECT "edgeSrc" AS src, "edgeDst" AS dst, "edgeWeight" AS weight, "edgeKey" AS edge_id FROM camel_edges),
                source_vertex => 0,
                include_edges => (SELECT "edgeSource" AS src, "edgeDestination" AS dst, "edgeId" AS edge_id FROM camel_allowed_edges))"#,
            "BFS include-edges relation",
        ),
        (
            r#"SELECT * FROM cugraph_personalized_pagerank(
                edges => (SELECT "edgeSrc" AS src, "edgeDst" AS dst, "edgeWeight" AS weight FROM camel_edges),
                personalization => (SELECT "seedVertex" AS vertex, "seedValue" AS value FROM camel_seeds))"#,
            "personalized PageRank relation",
        ),
        (
            r#"SELECT * FROM cugraph_jaccard(
                edges => (SELECT "edgeSrc" AS src, "edgeDst" AS dst FROM camel_edges),
                vertex_pairs => (SELECT "pairLeft" AS first, "pairRight" AS second FROM camel_pairs))"#,
            "similarity pair relation",
        ),
    ];

    for (sql, relation) in sqls {
        let df = ctx.sql(sql).await.unwrap_or_else(|error| panic!("{relation} should parse: {error}"));
        let plan = df
            .create_physical_plan()
            .await
            .unwrap_or_else(|error| panic!("{relation} should build exact projections: {error}"));
        let cugraph =
            find_cugraph_algo_exec(plan.as_ref()).unwrap_or_else(|| panic!("{relation} must retain graph boundary"));
        assert_eq!(cugraph.children().len(), 2, "{relation} must retain edge and side-input relations");
        assert!(
            cugraph.children().iter().all(|child| plan_contains_native_exec(child.as_ref())),
            "{relation} inputs must retain native execution planning"
        );
    }
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
            Field::new("distance", DataType::Int64, true),
            Field::new("predecessor", DataType::Int64, true),
            Field::new("reachable", DataType::Boolean, false),
        ],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Louvain),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("community", DataType::Int64, false),],
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
        vec![Field::new("vertex", DataType::Int64, false), Field::new("component", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::StronglyConnectedComponents),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("component", DataType::Int64, false)],
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
            vec![Field::new("vertex", DataType::Int64, false), Field::new("community", DataType::Int64, false)],
            "community algorithm {algorithm:?} must expose vertex + community schema",
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
    let dir = unique_temp_dir("algeon-datafusion-cugraph-louvain-planning");
    let path = dir.join("tiny_communities.parquet");
    write_tiny_communities(&path);

    let (_backend, ctx) = backend_cugraph_sql_context(SessionStateBuilder::new().with_default_features());
    ctx.register_parquet("tiny_communities", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, community \
             FROM cugraph_louvain(edges => (SELECT src, dst, weight FROM tiny_communities), directed => false) \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert_eq!(
        plan.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["vertex", "community"]
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
            "SELECT vertex, community \
             FROM cugraph_louvain(\
                  edges => (SELECT src, dst, weight FROM tiny_communities),\
                  directed => false, construction_policy => 'raw_libcugraph') \
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
    let dir = unique_temp_dir("algeon-datafusion-cugraph-normalized-planning");
    let directed_path = dir.join("tiny_directed_edges.parquet");
    let undirected_path = dir.join("tiny_communities.parquet");
    write_tiny_directed_edges(&directed_path);
    write_tiny_communities(&undirected_path);

    let (_backend, ctx) = backend_cugraph_sql_context(SessionStateBuilder::new().with_default_features());
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
             FROM cugraph_pagerank(edges => (SELECT src, dst FROM canonical_directed_edges)) \
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
            "SELECT vertex, community \
             FROM cugraph_louvain(edges => (SELECT src, dst FROM canonical_undirected_edges), directed => false) \
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
             FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM weighted_min_edges)) \
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
             FROM cugraph_pagerank(edges => (SELECT src, dst FROM grouped_by_extra_edge_attr)) \
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
             FROM cugraph_pagerank(edges => (SELECT src, dst FROM projected_duplicate_edges)) \
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
             FROM cugraph_pagerank(edges => (SELECT src, dst FROM aggregate_alias_overwrites_dst_edges)) \
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
             FROM cugraph_pagerank(edges => (SELECT src, dst FROM aggregate_over_canonical_edges)) \
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
            "SELECT vertex, community \
             FROM cugraph_louvain(edges => (SELECT src, dst FROM filtered_reverse_expansion_edges), directed => false) \
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
            "SELECT vertex, community \
             FROM cugraph_louvain(edges => (SELECT src, dst FROM mixed_memory_reverse_edges), directed => false) \
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

/// A whole-partition `COUNT(*)` over a cuGraph result plans natively: the
/// community assignment stays on the device, is sorted by its partition key,
/// and is counted by the native window.
#[tokio::test]
async fn cugraph_leiden_result_feeds_a_native_partition_count_window() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(AlgeonNativeOptimizerConfig::default())
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let coverage = validate_query(
        &ctx,
        "SELECT vertex, community, COUNT(*) OVER (PARTITION BY community) AS community_size \
         FROM cugraph_leiden(edges => (SELECT src, dst FROM graph_edges))",
    )
    .await
    .unwrap();

    assert_eq!(coverage.gpu_path(), "native", "{:?}", coverage.candidate_outcomes);
    assert!(
        coverage.candidate_outcomes.iter().all(|outcome| outcome.selection == CandidateOutcomeKind::Selected),
        "{:?}",
        coverage.candidate_outcomes
    );
}

/// An edge relation whose filter has no native counterpart.
const UNSUPPORTED_EDGES_SQL: &str = "SELECT * FROM cugraph_pagerank(edges => (\
     SELECT user_id AS src, item_id AS dst FROM events WHERE random() < 2.0))";

fn register_events(ctx: &SessionContext) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Int64, false),
        Field::new("item_id", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10_i64, 10, 20])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("events", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
}

/// A rejection inside the relation a cuGraph function reads reaches the report
/// of the query root, labelled with the function and the relation, instead of
/// leaving the outer candidate selected with no reason.
#[tokio::test]
async fn cugraph_edge_input_that_cannot_lower_is_reported_with_its_function_and_relation() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(AlgeonNativeOptimizerConfig::default())
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_events(&ctx);

    let coverage = validate_query(&ctx, UNSUPPORTED_EDGES_SQL).await.unwrap();

    let input = coverage
        .candidate_outcomes
        .iter()
        .find(|outcome| outcome.detail.starts_with("cugraph_pagerank.edges("))
        .unwrap_or_else(|| panic!("the edge input has a labelled candidate: {:?}", coverage.candidate_outcomes));
    assert_eq!(input.selection, CandidateOutcomeKind::NotSupported);
    assert_eq!(input.reason_code.as_deref(), Some("unsupported_expression"));
    let blocking = input.blocking.as_ref().expect("the rejection names its subject");
    assert_eq!((blocking.kind(), blocking.text()), ("scalar_function", "scalar function random in `random()`"));
    let [call] = coverage.cugraph_calls.as_slice() else {
        panic!("one cuGraph call: {:?}", coverage.cugraph_calls);
    };
    assert!(!call.inputs_native, "{call:?}");
    assert_eq!(coverage.to_json()["cugraph_calls"][0]["inputs_native"], false);

    let rows = ctx
        .sql(&format!(
            "SELECT candidate_shape, outcome, blocking FROM algeon_explain_coverage('{UNSUPPORTED_EDGES_SQL}') \
             WHERE row_kind = 'candidate' AND candidate_shape LIKE 'cugraph_pagerank.edges(%'"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let [batch] = rows.as_slice() else {
        panic!("EXPLAIN GPU rows: {rows:?}");
    };
    let blocking_column = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(blocking_column.value(0), "scalar function random in `random()`");

    let native = validate_query(
        &ctx,
        "SELECT * FROM cugraph_pagerank(edges => (SELECT user_id AS src, item_id AS dst FROM events))",
    )
    .await
    .unwrap();
    assert!(native.cugraph_calls[0].inputs_native, "{:?}", native.cugraph_calls);
}

/// Under `NativeRequired`, the final-plan error names the blocking subject of
/// the cuGraph input that keeps DataFusion work in the plan.
#[tokio::test]
async fn native_required_names_the_blocking_subject_of_a_cugraph_edge_input() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(
            AlgeonNativeOptimizerConfig::default().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired),
        )
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_events(&ctx);

    let error = ctx
        .sql(UNSUPPORTED_EDGES_SQL)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .expect_err("NativeRequired rejects an edge input that stays on DataFusion");

    let error = typed_algeon_error(&error);
    assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(error.fact_value("capability_reason"), Some("unsupported_expression"));
    assert_eq!(error.fact_value("blocking_kind"), Some("scalar_function"));
    assert!(
        matches!(
            error.client_detail(),
            Some(ClientDetail::BlockingSubject { text }) if text == "scalar function random in `random()`"
        ),
        "{:?}",
        error.client_detail()
    );
}
