use super::{support::*, *};

// Per-algorithm Python-oracle GPU tests for PageRank, BFS, and Louvain
// previously lived here. They have been subsumed by the registry-driven
// `tests/cugraph/cugraph_parity.rs` harness, which reads local oracle JSON
// produced by `fixture/scripts/cugraph_oracle.py`. The composition test below
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
