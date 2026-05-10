use super::*;

#[test]
fn bounded_native_multi_join_rejects_unsupported_mixed_residual_before_source_read() {
    let runtime = backend_runtime();
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let region_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let second_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let lineitem = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_path(
        "/tmp/bounded-multi-join-invalid-residual-left-does-not-need-to-exist.parquet",
        Arc::clone(&left_schema),
        vec!["order_id".into(), "qty".into()],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_path(
        "/tmp/bounded-multi-join-invalid-residual-orders-does-not-need-to-exist.parquet",
        Arc::clone(&orders_schema),
        vec!["order_id".into(), "category".into()],
    )));
    let regions = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_path(
        "/tmp/bounded-multi-join-invalid-residual-regions-does-not-need-to-exist.parquet",
        Arc::clone(&region_schema),
        vec!["order_id".into()],
    )));
    let first_join = builder.add_join(
        lineitem,
        orders,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(
                    nexus_query_engine::expr::ColumnRef::input(1, "category", "category"),
                    "category",
                ),
            ])
            .with_build_side(JoinSide::Right),
        first_join_schema,
    );
    let second_join = builder.add_join(
        first_join,
        regions,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_residual_predicate(Expr::like(Expr::input_column(0, "category", "category"), "t%"))
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        second_join_schema,
    );
    let plan = builder.finish_with_host_sink(second_join);
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    let analysis = plan.capability_analysis();
    let nexus_query_engine::capability::CapabilityDecision::Rejected { reason, facts } =
        &analysis.closed_multi_join_subgraph_admission
    else {
        panic!("unsupported residual expression should reject multi-join capability admission");
    };
    assert_eq!(*reason, "unsupported_join_residual_expression_capability");
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("like"));
    assert_eq!(facts.get("reason").map(String::as_str), Some("join_residual_expression_family_not_supported"));

    let context = bounded_join_execution_context(&runtime);
    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("unsupported bounded multi-join residual should fail before source reads"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.operation(), "nexus_query_engine.join.validate_residual");
    assert!(
        error.fact_value("expression").is_some_and(|expression| expression.contains("LIKE")),
        "LIKE residual should be visible in expression fact: {:?}",
        error.fact_value("expression")
    );
    assert_eq!(error.fact_value("path"), None);
    assert_eq!(error.fact_value("memory_boundary"), None);
}
