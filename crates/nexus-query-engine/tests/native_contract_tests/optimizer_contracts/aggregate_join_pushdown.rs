#[test]
fn aggregate_join_pushdown_reduces_plain_sum_inner_join_without_semijoining_duplicates() {
    let source_schema = schema(&["zip_prefix", "s_store_name", "ss_net_profit"]);
    let right_schema = schema(&["zip_prefix"]);
    let join_schema = schema(&["ss_net_profit", "s_store_name"]);
    let aggregate_input_schema = schema(&["s_store_name", "__cudf_aggregate_arg_0"]);
    let aggregate_schema = schema(&["s_store_name", "sum_profit"]);

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/q08-class-left.parquet")
                .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
                .with_row_group_rows(vec![10_000_000]),
        ],
        Arc::clone(&source_schema),
        vec!["zip_prefix".into(), "s_store_name".into(), "ss_net_profit".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/q08-class-right.parquet")
                .with_projected_row_group_bytes(vec![1024], vec![1024])
                .with_row_group_rows(vec![100]),
        ],
        Arc::clone(&right_schema),
        vec!["zip_prefix".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "zip_prefix", "zip_prefix"),
                ColumnRef::input(1, "zip_prefix", "zip_prefix"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ss_net_profit", "ss_net_profit"), "ss_net_profit"),
            JoinOutputColumn::left(ColumnRef::input(0, "s_store_name", "s_store_name"), "s_store_name"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let projection = builder.add_projection(
        join,
        projection_outputs(
            vec![Expr::column("s_store_name"), Expr::column("ss_net_profit").alias("__cudf_aggregate_arg_0")],
            aggregate_input_schema,
        ),
    );
    let aggregate = builder.add_aggregate(
        projection,
        AggregateSpec::new(
            vec![Expr::column("s_store_name")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("__cudf_aggregate_arg_0"), "sum_profit")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "plain q08-class aggregate-over-join should be pre-aggregated");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);

    let root = optimized.plan.root_node();
    let PlanNode::Aggregate(final_aggregate) = &root.kind else {
        panic!("optimized root should remain a final aggregate");
    };
    assert!(final_aggregate.spec.grouping_sets.is_none());
    assert_eq!(final_aggregate.spec.group_exprs, vec![Expr::column("s_store_name")]);

    let PlanNode::Join(reduced_join) =
        &optimized.plan.node(final_aggregate.input).expect("final aggregate input should exist").kind
    else {
        panic!("final aggregate should consume the reduced join");
    };
    assert_eq!(
        reduced_join.spec.join_type,
        JoinType::Inner,
        "right-side duplicate keys must retain inner join multiplicity"
    );
    assert!(
        reduced_join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left),
        "q08-class reduced join should only emit pre-aggregated left columns"
    );

    let preaggregate_id = reduced_join.inputs[0];
    let PlanNode::Aggregate(preaggregate) =
        &optimized.plan.node(preaggregate_id).expect("pre-aggregate input should exist").kind
    else {
        panic!("left input to reduced join should be the pre-aggregate");
    };
    assert_eq!(
        preaggregate.spec.group_exprs,
        vec![Expr::column("s_store_name").alias("s_store_name"), Expr::column("zip_prefix").alias("zip_prefix")],
        "pre-aggregate must include the join key so the reduced inner join preserves right-key multiplicity"
    );
}
