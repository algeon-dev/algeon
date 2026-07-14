use super::*;

// Selects which leaf is wrapped in an aggregate for multi-join fixtures.
#[derive(Debug, Clone, Copy)]
pub(super) enum MultiJoinAggregateLeaf {
    FirstLeft,
    FirstRight,
    SecondRight,
}

pub(super) fn hash_multi_join_with_aggregate_leaf_plan(
    aggregate_leaf: MultiJoinAggregateLeaf,
) -> nexus_query_engine::plan::QueryPlan {
    let base_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("left_payload", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("right_payload", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("row_count", DataType::Int64, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let base = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/multi-join-base.parquet",
        base_schema,
        vec!["id".into(), "left_payload".into()],
    )));
    let first_right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/multi-join-first-right.parquet",
        Arc::clone(&right_schema),
        vec!["id".into(), "right_payload".into()],
    )));
    let second_right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/multi-join-second-right.parquet",
        right_schema,
        vec!["id".into(), "right_payload".into()],
    )));

    let base = if matches!(aggregate_leaf, MultiJoinAggregateLeaf::FirstLeft) {
        aggregate_leaf_source(&mut builder, base, Arc::clone(&aggregate_schema))
    } else {
        base
    };
    let first_right = if matches!(aggregate_leaf, MultiJoinAggregateLeaf::FirstRight) {
        aggregate_leaf_source(&mut builder, first_right, Arc::clone(&aggregate_schema))
    } else {
        first_right
    };
    let second_right = if matches!(aggregate_leaf, MultiJoinAggregateLeaf::SecondRight) {
        aggregate_leaf_source(&mut builder, second_right, Arc::clone(&aggregate_schema))
    } else {
        second_right
    };

    let first_join = builder.add_join(
        base,
        first_right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let second_join = builder.add_join(
        first_join,
        second_right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id")])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    builder.finish_with_host_sink(second_join)
}

pub(super) fn aggregate_leaf_source(
    builder: &mut QueryPlanBuilder,
    input: nexus_query_engine::plan::NodeId,
    output_schema: Arc<Schema>,
) -> nexus_query_engine::plan::NodeId {
    builder.add_aggregate(
        input,
        AggregateSpec::new(
            vec![Expr::column("id")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("id"), "row_count")],
        ),
        output_schema,
    )
}

pub(super) fn mixed_residual_join_plan(
    left_schema: Arc<Schema>,
    right_schema: Arc<Schema>,
    output_schema: Arc<Schema>,
    join_type: JoinType,
    build_side: JoinSide,
    output_projection: Vec<JoinOutputColumn>,
) -> nexus_query_engine::plan::QueryPlan {
    let left_source =
        LocalParquetSource::new("/tmp/left.parquet", left_schema, vec!["id".into(), "left_payload".into()]);
    let right_source =
        LocalParquetSource::new("/tmp/right.parquet", right_schema, vec!["id".into(), "right_payload".into()]);

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(join_type, vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))])
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "left_payload", "left_payload"),
                BinaryOp::Gt,
                Expr::input_column(1, "right_payload", "right_payload"),
            ))
            .with_output_projection(output_projection)
            .with_null_equality(JoinNullEquality::NullsDoNotMatch)
            .with_build_side(build_side),
        Arc::clone(&output_schema),
    );
    builder.finish_with_host_sink(join)
}

pub(super) fn mixed_residual_multi_join_plan(
    second_join_type: JoinType,
    second_join_build_side: JoinSide,
    second_join_projection: Vec<JoinOutputColumn>,
) -> nexus_query_engine::plan::QueryPlan {
    let base_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("left_payload", DataType::Int32, false),
    ]));
    let first_right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let second_right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("right_payload", DataType::Int32, false),
    ]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("left_payload", DataType::Int32, false),
    ]));
    let second_join_schema = Arc::new(Schema::new(
        second_join_projection
            .iter()
            .map(|column| Field::new(column.output_name.as_str(), DataType::Int32, false))
            .collect::<Vec<_>>(),
    ));

    let mut builder = QueryPlanBuilder::new();
    let base = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/base.parquet",
        base_schema,
        vec!["id".into(), "left_payload".into()],
    )));
    let first_right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/first_right.parquet",
        first_right_schema,
        vec!["id".into()],
    )));
    let second_right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/second_right.parquet",
        second_right_schema,
        vec!["id".into(), "right_payload".into()],
    )));
    let first_join = builder.add_join(
        base,
        first_right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
        ])
        .with_build_side(JoinSide::Right),
        first_join_schema,
    );
    let second_join = builder.add_join(
        first_join,
        second_right,
        JoinSpec::new(
            second_join_type,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "left_payload", "left_payload"),
            BinaryOp::Gt,
            Expr::input_column(1, "right_payload", "right_payload"),
        ))
        .with_output_projection(second_join_projection)
        .with_build_side(second_join_build_side),
        second_join_schema,
    );
    builder.finish_with_host_sink(second_join)
}

pub(super) fn valid_mixed_residual_semi_anti_then_invalid_mixed_residual_join_plan(
    first_join_type: JoinType,
    second_join_type: JoinType,
) -> nexus_query_engine::plan::QueryPlan {
    let base_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("left_payload", DataType::Int32, false),
    ]));
    let first_right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("right_payload", DataType::Int32, false),
    ]));
    let second_right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("right_payload", DataType::Int32, false),
    ]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("left_payload", DataType::Int32, false),
    ]));
    let second_join_schema = Arc::new(Schema::new(vec![
        Field::new("left_payload", DataType::Int32, false),
        Field::new("right_payload", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let base = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/base.parquet",
        base_schema,
        vec!["id".into(), "left_payload".into()],
    )));
    let first_right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/first_right.parquet",
        first_right_schema,
        vec!["id".into(), "right_payload".into()],
    )));
    let second_right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/second_right.parquet",
        second_right_schema,
        vec!["id".into(), "right_payload".into()],
    )));
    let first_join = builder.add_join(
        base,
        first_right,
        JoinSpec::new(
            first_join_type,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "left_payload", "left_payload"),
            BinaryOp::Gt,
            Expr::input_column(1, "right_payload", "right_payload"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
        ])
        .with_build_side(JoinSide::Right),
        first_join_schema,
    );
    let second_join = builder.add_join(
        first_join,
        second_right,
        JoinSpec::new(
            second_join_type,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "left_payload", "left_payload"),
            BinaryOp::Gt,
            Expr::input_column(1, "right_payload", "right_payload"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload"),
        ])
        .with_build_side(JoinSide::Right),
        second_join_schema,
    );
    builder.finish_with_host_sink(second_join)
}

pub(super) fn mixed_residual_multi_join_with_aggregate_leaf_plan() -> nexus_query_engine::plan::QueryPlan {
    let base_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("left_payload", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_payload", DataType::Int64, true),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("right_payload", DataType::Int32, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_payload", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let base = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/base.parquet",
        base_schema,
        vec!["id".into(), "left_payload".into()],
    )));
    let aggregate = builder.add_aggregate(
        base,
        AggregateSpec::new(
            vec![Expr::column("id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("left_payload"), "sum_payload")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let first_right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/first_right.parquet",
        Arc::clone(&right_schema),
        vec!["id".into(), "right_payload".into()],
    )));
    let second_right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/second_right.parquet",
        right_schema,
        vec!["id".into(), "right_payload".into()],
    )));
    let first_join = builder.add_join(
        aggregate,
        first_right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sum_payload", "sum_payload"), "sum_payload"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let second_join = builder.add_join(
        first_join,
        second_right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "sum_payload", "sum_payload"),
            BinaryOp::Gt,
            Expr::input_column(1, "right_payload", "right_payload"),
        ))
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "sum_payload", "sum_payload"),
            "sum_payload",
        )])
        .with_build_side(JoinSide::Right),
        Arc::new(Schema::new(vec![Field::new("sum_payload", DataType::Int64, true)])),
    );
    builder.finish_with_host_sink(second_join)
}

pub(super) fn keyless_conditional_left_semi_plan(
    left_schema: Arc<Schema>,
    right_schema: Arc<Schema>,
    output_schema: Arc<Schema>,
    build_side: JoinSide,
) -> nexus_query_engine::plan::QueryPlan {
    let left_source = LocalParquetSource::new("/tmp/left.parquet", left_schema, vec!["left_payload".into()]);
    let right_source = LocalParquetSource::new("/tmp/right.parquet", right_schema, vec!["right_payload".into()]);
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::LeftSemi, Vec::new())
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "left_payload", "left_payload"),
                BinaryOp::Gt,
                Expr::input_column(1, "right_payload", "right_payload"),
            ))
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "left_payload", "left_payload"),
                "left_payload",
            )])
            .with_build_side(build_side),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}
