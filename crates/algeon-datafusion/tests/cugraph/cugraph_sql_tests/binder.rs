use super::{support::*, *};

fn named_sql_context() -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(AlgeonNativeOptimizerConfig::default())
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edge_memtable(&ctx, "edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_vertex_memtable(&ctx, "sources", "vertex_id", DataType::Int64, false);
    register_vertex_memtable(&ctx, "excluded", "vertex_id", DataType::Int64, false);
    ctx
}

#[tokio::test]
async fn cugraph_named_binder_derives_canonical_edge_and_bfs_schema() {
    let ctx = named_sql_context();
    let dataframe = ctx
        .sql(
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT src, dst, weight FROM edges),\
             sources => (SELECT vertex_id AS vertex FROM sources),\
             output_mode => 'normalized')",
        )
        .await
        .unwrap();

    assert_eq!(
        dataframe.schema().inner().fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>(),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, true),
            Field::new("predecessor", DataType::Int64, true),
            Field::new("reachable", DataType::Boolean, false),
        ],
    );
}

#[tokio::test]
async fn cugraph_named_binder_applies_scalar_values() {
    let ctx = named_sql_context();
    let dataframe = ctx
        .sql(
            "SELECT * FROM cugraph_pagerank(\
             edges => (SELECT src, dst, weight FROM edges),\
             alpha => 0.9, epsilon => 0.0001, max_iterations => 7)",
        )
        .await
        .unwrap();

    assert_eq!(dataframe.schema().field(0).name(), "vertex");
    assert_eq!(dataframe.schema().field(0).data_type(), &DataType::Int64);
    assert_eq!(dataframe.schema().field(1).name(), "value");
    assert_eq!(dataframe.schema().field(1).data_type(), &DataType::Float64);
}

#[tokio::test]
async fn cugraph_named_binder_is_order_independent() {
    let ctx = named_sql_context();
    let canonical = ctx
        .state()
        .create_logical_plan(
            "SELECT * FROM cugraph_pagerank(\
             edges => (SELECT src, dst, weight FROM edges),\
             alpha => 0.9, epsilon => 0.0001, max_iterations => 7)",
        )
        .await
        .unwrap();
    let reordered = ctx
        .state()
        .create_logical_plan(
            "SELECT * FROM cugraph_pagerank(\
             max_iterations => 7, epsilon => 0.0001,\
             edges => (SELECT src, dst, weight FROM edges), alpha => 0.9)",
        )
        .await
        .unwrap();

    assert_eq!(canonical.display_indent().to_string(), reordered.display_indent().to_string());
}

#[tokio::test]
async fn cugraph_named_binder_rejects_unknown_duplicate_missing_null_and_range_arguments() {
    let ctx = named_sql_context();

    let unknown = ctx
        .sql("SELECT * FROM cugraph_pagerank(edges => (SELECT src, dst FROM edges), unsupported => 1)")
        .await
        .unwrap_err();
    assert_eq!(typed_algeon_error(&unknown).fact_value("reason"), Some("unknown_argument"));

    let duplicate = ctx
        .sql(
            "SELECT * FROM cugraph_pagerank(\
             edges => (SELECT src, dst FROM edges), alpha => 0.8, alpha => 0.9)",
        )
        .await
        .unwrap_err();
    assert_eq!(typed_algeon_error(&duplicate).fact_value("reason"), Some("duplicate_argument"));

    let missing =
        ctx.sql("SELECT * FROM cugraph_sssp(edges => (SELECT src, dst, weight FROM edges))").await.unwrap_err();
    assert_eq!(typed_algeon_error(&missing).fact_value("reason"), Some("missing_argument"));

    for sql in [
        "SELECT * FROM cugraph_pagerank(edges => (SELECT src, dst FROM edges), alpha => NULL)",
        "SELECT * FROM cugraph_pagerank(edges => (SELECT src, dst FROM edges), alpha => 1.1)",
    ] {
        let error = ctx.sql(sql).await.unwrap_err();
        let typed = typed_algeon_error(&error);
        assert_eq!(typed.code(), ErrorCode::InvalidRequest, "{sql}");
        assert_eq!(typed.fact_value("reason"), Some("missing_or_invalid_option"), "{sql}");
        assert_eq!(typed.fact_value("argument"), Some("alpha"), "{sql}");
    }
}

#[tokio::test]
async fn cugraph_named_relation_binder_requires_canonical_edge_columns() {
    let ctx = named_sql_context();
    let error = ctx
        .sql(
            "SELECT * FROM cugraph_pagerank(\
             edges => (SELECT src AS source, dst AS destination FROM edges))",
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("canonical source column 'src'"), "unexpected error: {error}");
}

#[tokio::test]
async fn cugraph_named_relation_planner_keeps_side_inputs_as_children() {
    let ctx = named_sql_context();
    let dataframe = ctx
        .sql(
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT src, dst FROM edges), source_vertex => 0,\
             exclude_vertices => (SELECT vertex_id AS vertex FROM excluded))",
        )
        .await
        .unwrap();
    let plan = dataframe.create_physical_plan().await.unwrap();
    let cugraph = find_cugraph_algo_exec(plan.as_ref()).expect("named BFS must retain its graph boundary");
    assert_eq!(cugraph.children().len(), 2, "edge and side-input relations remain graph inputs");
}

#[tokio::test]
async fn cugraph_named_binder_rejects_conflicting_relations_and_values() {
    let ctx = named_sql_context();
    let cases = [
        (
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM edges), source_vertex => 0, \
             include_vertices => (SELECT vertex_id AS vertex FROM sources), \
             exclude_vertices => (SELECT vertex_id AS vertex FROM excluded))",
            "include_vertices and exclude_vertices are mutually exclusive",
        ),
        (
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM edges), source_vertex => 0, \
             sources => (SELECT vertex_id AS vertex FROM sources))",
            "exactly one of source_vertex, source_vertices, or sources",
        ),
        (
            "SELECT * FROM cugraph_jaccard(edges => (SELECT src, dst FROM edges), directed => true, \
             vertex_pairs => (SELECT src AS first, dst AS second FROM edges))",
            "requires directed=false",
        ),
        (
            "SELECT * FROM cugraph_betweenness_centrality(edges => (SELECT src, dst FROM edges), \
             seeds => make_array(0, 1), k => 2)",
            "seeds",
        ),
        (
            "SELECT * FROM cugraph_force_atlas2(edges => (SELECT src, dst FROM edges), prevent_overlapping => true, \
             vertex_attributes => (SELECT vertex_id AS vertex, CAST(1.0 AS FLOAT) AS mass FROM sources))",
            "prevent_overlapping=true requires vertex_attributes.radius",
        ),
        (
            "SELECT * FROM cugraph_personalized_pagerank(edges => (SELECT src, dst FROM edges))",
            "required argument 'personalization' is missing",
        ),
        (
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM edges), source_vertex => 0, \
             include_edge_ids => (SELECT vertex_id AS edge_id FROM sources))",
            "include_edge_ids requires edges.edge_id",
        ),
    ];
    for (sql, expected) in cases {
        let error = ctx.sql(sql).await.expect_err(sql).to_string();
        assert!(error.contains(expected), "{sql}\nunexpected error: {error}");
    }
}

#[tokio::test]
async fn cugraph_force_atlas2_binds_both_optional_relations_as_children() {
    let ctx = named_sql_context();
    let plan = ctx
        .state()
        .create_logical_plan(
            "SELECT * FROM cugraph_force_atlas2(\
             edges => (SELECT src, dst FROM edges), prevent_overlapping => true, \
             vertex_attributes => (SELECT vertex_id AS vertex, CAST(0.5 AS FLOAT) AS radius FROM sources), \
             initial_positions => (SELECT vertex_id AS vertex, CAST(0.0 AS FLOAT) AS x, CAST(1.0 AS FLOAT) AS y \
             FROM sources))",
        )
        .await
        .unwrap();
    let rendered = plan.display_indent().to_string();
    assert!(
        rendered.contains("relation_roles=[edges,vertex_attributes,initial_positions]"),
        "unexpected plan: {rendered}"
    );
    assert_eq!(
        plan.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        ["vertex", "x", "y"]
    );
}

#[tokio::test]
async fn cugraph_named_relation_binder_rejects_ambiguous_contract_columns() {
    let ctx = named_sql_context();
    let cases = [
        (
            "SELECT * FROM cugraph_pagerank(\
             edges => (SELECT a.src, a.dst, a.weight, b.weight FROM edges a CROSS JOIN edges b))",
            "column 'weight' is ambiguous",
        ),
        (
            "SELECT * FROM cugraph_pagerank(\
             edges => (SELECT a.src, b.src, a.dst FROM edges a JOIN edges b ON a.dst = b.src))",
            "column 'src' is ambiguous",
        ),
        (
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM edges), source_vertex => 0, \
             exclude_vertices => (SELECT a.vertex, b.vertex FROM (SELECT vertex_id AS vertex FROM excluded) a \
             CROSS JOIN (SELECT vertex_id AS vertex FROM excluded) b))",
            "exclude_vertices relation column 'vertex' is ambiguous",
        ),
    ];
    for (sql, expected) in cases {
        let error = ctx.sql(sql).await.expect_err(sql);
        assert!(error.to_string().contains(expected), "{sql}\nunexpected error: {error}");
        assert!(typed_algeon_error(&error).fact_value("relation_role").is_some(), "{sql}");
    }
}

#[tokio::test]
async fn cugraph_named_binder_accepts_typed_literals_and_null_selectors() {
    let ctx = named_sql_context();
    for sql in [
        "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM edges), source_vertex => CAST(1 AS BIGINT))",
        "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM edges), source_vertex => CAST(1 AS INT))",
        "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM edges), \
         source_vertices => make_array(CAST(1 AS INT), 2))",
        "SELECT * FROM cugraph_sssp(edges => (SELECT src, dst, weight FROM edges), \
         source_vertex => CAST(2 AS SMALLINT), cutoff => -CAST(0 AS DOUBLE))",
        "SELECT * FROM cugraph_betweenness_centrality(edges => (SELECT src, dst FROM edges), k => NULL, \
         seeds => make_array(0, 1))",
    ] {
        ctx.state().create_logical_plan(sql).await.unwrap_or_else(|error| panic!("{sql}: {error}"));
    }
}
