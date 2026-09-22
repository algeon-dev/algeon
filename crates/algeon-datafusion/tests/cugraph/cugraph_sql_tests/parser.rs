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
        assert_eq!(typed_algeon_error(&error).code(), ErrorCode::InvalidExecutionPlan, "{sql}");
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
