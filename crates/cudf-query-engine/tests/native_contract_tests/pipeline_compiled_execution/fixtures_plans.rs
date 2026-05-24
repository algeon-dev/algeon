use super::*;

pub(super) fn root_join_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-left.parquet",
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-right.parquet",
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

pub(super) fn union_of_ranked_join_branches_plan() -> QueryPlan {
    let branch_input_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("item", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("item", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("item", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
    ]));
    let ranked_schema = Arc::new(Schema::new(vec![
        Field::new("item", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
        Field::new("amount_rank", DataType::UInt64, false),
    ]));
    let union_schema = Arc::new(Schema::new(vec![
        Field::new("item", DataType::Int32, false),
        Field::new("amount_rank", DataType::UInt64, false),
    ]));
    let final_schema = Arc::new(Schema::new(vec![
        Field::new("item", DataType::Int32, false),
        Field::new("best_rank", DataType::UInt64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let first = ranked_join_branch(
        &mut builder,
        "/tmp/compiled-union-window-branch-a.parquet",
        "/tmp/compiled-union-window-dim-a.parquet",
        Arc::clone(&branch_input_schema),
        Arc::clone(&dim_schema),
        Arc::clone(&join_schema),
        Arc::clone(&aggregate_schema),
        Arc::clone(&ranked_schema),
        Arc::clone(&union_schema),
    );
    let second = ranked_join_branch(
        &mut builder,
        "/tmp/compiled-union-window-branch-b.parquet",
        "/tmp/compiled-union-window-dim-b.parquet",
        Arc::clone(&branch_input_schema),
        Arc::clone(&dim_schema),
        Arc::clone(&join_schema),
        Arc::clone(&aggregate_schema),
        Arc::clone(&ranked_schema),
        Arc::clone(&union_schema),
    );
    let union = builder.add_union(vec![first, second], Arc::clone(&union_schema));
    let aggregate = builder.add_aggregate(
        union,
        AggregateSpec::new(
            vec![Expr::column("item")],
            vec![AggregateExpr::new(AggregateFunction::Min, Expr::column("amount_rank"), "best_rank")],
        ),
        Arc::clone(&final_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("best_rank"), false, false)]),
        Arc::clone(&final_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), Arc::clone(&final_schema));
    builder.finish_with_host_sink(limit)
}

#[allow(clippy::too_many_arguments)]
fn ranked_join_branch(
    builder: &mut QueryPlanBuilder,
    fact_path: &str,
    dim_path: &str,
    fact_schema: Arc<Schema>,
    dim_schema: Arc<Schema>,
    join_schema: Arc<Schema>,
    aggregate_schema: Arc<Schema>,
    ranked_schema: Arc<Schema>,
    output_schema: Arc<Schema>,
) -> NodeId {
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        fact_path,
        fact_schema,
        vec!["id".into(), "item".into(), "amount".into()],
    )));
    let dim =
        builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(dim_path, dim_schema, vec!["id".into()])));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "item", "item"), "item"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("item")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), false, false)]),
        Arc::clone(&aggregate_schema),
    );
    let window = builder.add_window(
        sort,
        WindowSpec::new(
            vec![],
            vec![SortExpr::new(Expr::column("amount_sum"), false, false)],
            vec![WindowFunction::rank("amount_rank")],
        ),
        Arc::clone(&ranked_schema),
    );
    let filter = builder.add_filter(
        window,
        Expr::binary(Expr::column("amount_rank"), BinaryOp::LtEq, Expr::literal(LiteralValue::UInt64(10))),
        Arc::clone(&ranked_schema),
    );
    let output_fields = output_schema.fields();
    builder.add_projection(
        filter,
        vec![
            ProjectionOutput::new(Expr::column("item"), output_fields[0].as_ref().clone()),
            ProjectionOutput::new(Expr::column("amount_rank"), output_fields[1].as_ref().clone()),
        ],
    )
}

pub(super) fn keyless_root_join_plan() -> QueryPlan {
    keyless_root_join_data_plan(
        Path::new("/tmp/compiled-d1-keyless-left.parquet"),
        Path::new("/tmp/compiled-d1-keyless-right.parquet"),
    )
}

pub(super) fn keyless_outer_cross_root_join_plan() -> QueryPlan {
    keyless_outer_cross_root_join_data_plan(
        Path::new("/tmp/compiled-d1-keyless-outer-cross-left.parquet"),
        Path::new("/tmp/compiled-d1-keyless-outer-cross-right.parquet"),
    )
}

pub(super) fn keyless_root_join_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![])
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "left_id", "left_id"),
                BinaryOp::Lt,
                Expr::input_column(1, "right_id", "right_id"),
            ))
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
                JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
            ])
            .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

pub(super) fn keyless_outer_cross_root_join_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Left, vec![])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
                JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
            ])
            .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

pub(super) fn aggregate_join_input_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d24-aggregate-join-input-left.parquet",
        Arc::clone(&left_schema),
        vec!["id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d24-aggregate-join-input-right.parquet",
        Arc::clone(&right_schema),
        vec!["id".into()],
    )));
    let aggregate = builder.add_aggregate(
        left,
        AggregateSpec::new(
            vec![Expr::column("id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        aggregate_schema,
    );
    let join = builder.add_join(
        aggregate,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "left_id"), ColumnRef::input(1, "id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sum_amount", "sum_amount"), "sum_amount"),
        ])
        .with_build_side(JoinSide::Left),
        join_schema,
    );
    builder.finish_with_host_sink(join)
}

pub(super) fn aggregate_join_build_input_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(left_path).with_file_size_bytes(1024)],
        Arc::clone(&left_schema),
        vec!["id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(right_path).with_file_size_bytes(1024)],
        Arc::clone(&right_schema),
        vec!["id".into()],
    )));
    let aggregate = builder.add_aggregate(
        left,
        AggregateSpec::new(
            vec![Expr::column("id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        aggregate_schema,
    );
    let join = builder.add_join(
        aggregate,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "left_id"), ColumnRef::input(1, "id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sum_amount", "sum_amount"), "sum_amount"),
        ])
        .with_build_side(JoinSide::Left),
        join_schema,
    );
    builder.finish_with_host_sink(join)
}

pub(super) fn root_filtered_left_only_join_plan(join_type: JoinType) -> QueryPlan {
    assert!(matches!(join_type, JoinType::LeftSemi | JoinType::LeftAnti));
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d31-filtered-left.parquet",
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d31-filtered-right.parquet",
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            join_type,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "left_value", "left_value"),
            BinaryOp::Gt,
            Expr::input_column(1, "right_value", "right_value"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "left_value", "left_value"), "left_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

pub(super) fn root_join_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    root_join_data_plan_with_sink(left_path, right_path, false)
}

pub(super) fn root_join_gpu_sink_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    root_join_data_plan_with_sink(left_path, right_path, true)
}

pub(super) fn root_join_data_plan_with_sink(left_path: &Path, right_path: &Path, gpu_resident_sink: bool) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    if gpu_resident_sink { builder.finish_gpu_resident(join) } else { builder.finish_with_host_sink(join) }
}

pub(super) fn shared_source_self_join_data_plan(path: &Path) -> QueryPlan {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        path,
        source_schema,
        vec!["id".into(), "value".into()],
    )));
    let join = builder.add_join(
        source,
        source,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::right(ColumnRef::input(1, "value", "value"), "value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

pub(super) fn root_join_sort_limit_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)]),
        Arc::clone(&output_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(1), output_schema);
    builder.finish_with_host_sink(limit)
}

pub(super) fn simple_multi_join_plan(a_path: &Path, b_path: &Path, c_path: &Path) -> QueryPlan {
    let a_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
    ]));
    let b_schema = Arc::new(Schema::new(vec![
        Field::new("b_id", DataType::Int32, false),
        Field::new("b_value", DataType::Int32, false),
    ]));
    let c_schema = Arc::new(Schema::new(vec![
        Field::new("c_id", DataType::Int32, false),
        Field::new("c_value", DataType::Int32, false),
    ]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
        Field::new("b_value", DataType::Int32, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
        Field::new("b_value", DataType::Int32, true),
        Field::new("c_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let a = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        a_path,
        a_schema,
        vec!["a_id".into(), "a_value".into()],
    )));
    let b = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        b_path,
        b_schema,
        vec!["b_id".into(), "b_value".into()],
    )));
    let c = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        c_path,
        c_schema,
        vec!["c_id".into(), "c_value".into()],
    )));
    let ab = builder.add_join(
        a,
        b,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "a_id", "a_id"), ColumnRef::input(1, "b_id", "b_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_id", "a_id"), "a_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_value", "a_value"), "a_value"),
            JoinOutputColumn::right(ColumnRef::input(1, "b_value", "b_value"), "b_value"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&first_join_schema),
    );
    let abc = builder.add_join(
        ab,
        c,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "a_id", "a_id"), ColumnRef::input(1, "c_id", "c_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_id", "a_id"), "a_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_value", "a_value"), "a_value"),
            JoinOutputColumn::left(ColumnRef::input(0, "b_value", "b_value"), "b_value"),
            JoinOutputColumn::right(ColumnRef::input(1, "c_value", "c_value"), "c_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(abc)
}

pub(super) fn pipeline_without_reservation_gates(pipeline: &CompiledPipeline) -> CompiledPipeline {
    let tasks = pipeline
        .tasks()
        .iter()
        .map(|task| {
            let base = match task.plan_node() {
                Some(node_id) => PipelineTask::for_plan_node(task.id(), task.kind(), node_id),
                None => PipelineTask::new(task.id(), task.kind()),
            };
            base.with_input_resources(task.input_resources().to_vec())
                .with_output_resources(task.output_resources().to_vec())
                .with_gates(vec![])
        })
        .collect::<Vec<_>>();
    let edges = pipeline
        .edges()
        .iter()
        .map(|edge| {
            PipelineEdge::new(edge.id(), edge.from(), edge.to(), edge.kind())
                .with_resources(edge.resources().to_vec())
                .with_gates(vec![])
        })
        .collect::<Vec<_>>();
    let barriers = pipeline
        .barriers()
        .iter()
        .map(|barrier| {
            PipelineBarrier::new(barrier.id(), barrier.kind())
                .with_upstream_tasks(barrier.upstream_tasks().to_vec())
                .with_downstream_tasks(barrier.downstream_tasks().to_vec())
                .with_gates(vec![])
        })
        .collect::<Vec<_>>();
    let resources = pipeline
        .resources()
        .iter()
        .map(|resource| {
            ResidentResource::new(
                resource.id(),
                resource.producer(),
                resource.consumers().to_vec(),
                resource.lifetime(),
            )
        })
        .collect::<Vec<_>>();
    let sink = SinkBoundary::new(pipeline.sink().kind(), pipeline.sink().input()).with_gates(vec![]);
    CompiledPipeline::new(tasks, edges, barriers, resources, vec![], sink)
}

pub(super) fn pipeline_with_extra_sink_candidate(pipeline: &CompiledPipeline) -> CompiledPipeline {
    let mut tasks = pipeline.tasks().to_vec();
    let mut edges = pipeline.edges().to_vec();
    let sink_edge = pipeline
        .edges()
        .iter()
        .find(|edge| edge.kind() == PipelineEdgeKind::SinkInput)
        .expect("fixture should have a sink input edge");
    let [sink_resource] = sink_edge.resources() else {
        panic!("sink input edge should carry one resource");
    };
    let extra_sink = PipelineTaskId::new(tasks.len());
    tasks.push(
        PipelineTask::new(extra_sink, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![*sink_resource]),
    );
    edges.push(
        PipelineEdge::new(PipelineEdgeId::new(edges.len()), sink_edge.from(), extra_sink, PipelineEdgeKind::SinkInput)
            .with_resources(vec![*sink_resource]),
    );

    let resources = pipeline
        .resources()
        .iter()
        .map(|resource| {
            let mut consumers = resource.consumers().to_vec();
            if resource.id() == *sink_resource {
                consumers.push(extra_sink);
                consumers.sort_unstable();
                consumers.dedup();
            }
            ResidentResource::new(resource.id(), resource.producer(), consumers, resource.lifetime())
        })
        .collect::<Vec<_>>();
    let sink =
        SinkBoundary::new(pipeline.sink().kind(), pipeline.sink().input()).with_gates(pipeline.sink().gates().to_vec());
    CompiledPipeline::new(tasks, edges, pipeline.barriers().to_vec(), resources, pipeline.gates().to_vec(), sink)
}

pub(super) fn pipeline_with_post_root_fanout(pipeline: &CompiledPipeline) -> CompiledPipeline {
    let mut tasks = pipeline.tasks().to_vec();
    let mut edges = pipeline.edges().to_vec();
    let fanout_edge = pipeline
        .edges()
        .iter()
        .find(|edge| {
            edge.kind() == PipelineEdgeKind::ResidentFrame
                && pipeline
                    .tasks()
                    .iter()
                    .find(|task| task.id() == edge.from())
                    .is_some_and(|task| task.kind() == PipelineTaskKind::JoinMaterialize)
        })
        .expect("fixture should have a post-root resident-frame edge");
    let [fanout_resource] = fanout_edge.resources() else {
        panic!("post-root resident-frame edge should carry one resource");
    };
    let extra_task = PipelineTaskId::new(tasks.len());
    let extra_resource = ResidentResourceId::new(pipeline.resources().len());
    tasks.push(
        PipelineTask::new(extra_task, PipelineTaskKind::Unary)
            .with_input_resources(vec![*fanout_resource])
            .with_output_resources(vec![extra_resource]),
    );
    edges.push(
        PipelineEdge::new(
            PipelineEdgeId::new(edges.len()),
            fanout_edge.from(),
            extra_task,
            PipelineEdgeKind::ResidentFrame,
        )
        .with_resources(vec![*fanout_resource]),
    );

    let mut resources = pipeline
        .resources()
        .iter()
        .map(|resource| {
            let mut consumers = resource.consumers().to_vec();
            if resource.id() == *fanout_resource {
                consumers.push(extra_task);
                consumers.sort_unstable();
                consumers.dedup();
            }
            ResidentResource::new(resource.id(), resource.producer(), consumers, resource.lifetime())
        })
        .collect::<Vec<_>>();
    resources.push(ResidentResource::new(
        extra_resource,
        extra_task,
        vec![],
        ResidentLifetime::TaskLocal { task: extra_task },
    ));
    let sink =
        SinkBoundary::new(pipeline.sink().kind(), pipeline.sink().input()).with_gates(pipeline.sink().gates().to_vec());
    CompiledPipeline::new(tasks, edges, pipeline.barriers().to_vec(), resources, pipeline.gates().to_vec(), sink)
}

pub(super) fn root_join_with_unary_inputs_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_source_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_source_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        Arc::clone(&left_source_schema),
        vec!["left_id".into(), "left_value".into()],
    )));
    let left_filter = builder.add_filter(
        left,
        Expr::binary(Expr::column("left_id"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(1))),
        Arc::clone(&left_source_schema),
    );
    let left_projected = builder.add_projection(
        left_filter,
        vec![
            ProjectionOutput::new(Expr::column("left_id"), Field::new("left_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("left_value"), Field::new("left_value", DataType::Int32, false)),
        ],
    );
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        Arc::clone(&right_source_schema),
        vec!["right_id".into(), "right_value".into()],
    )));
    let right_filter = builder.add_filter(
        right,
        Expr::binary(Expr::column("right_id"), BinaryOp::Lt, Expr::literal(LiteralValue::Int32(5))),
        Arc::clone(&right_source_schema),
    );
    let right_projected = builder.add_projection(
        right_filter,
        vec![
            ProjectionOutput::new(Expr::column("right_id"), Field::new("right_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("right_value"), Field::new("right_value", DataType::Int32, false)),
        ],
    );
    let join = builder.add_join(
        left_projected,
        right_projected,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

pub(super) fn single_join_post_root_aggregate_data_plan(customers_path: &Path, orders_path: &Path) -> QueryPlan {
    let customer_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int32, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customers = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        customers_path,
        customer_schema,
        vec!["c_custkey".into()],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        orders_path,
        orders_schema,
        vec!["o_orderkey".into(), "o_custkey".into()],
    )));
    let join = builder.add_join(
        customers,
        orders,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                ColumnRef::input(0, "c_custkey", "c_custkey"),
                ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "c_custkey", "c_custkey"), "c_custkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "o_orderkey", "o_orderkey"), "o_orderkey"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("c_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        aggregate_schema,
    );
    builder.finish_with_host_sink(aggregate)
}

pub(super) fn string_group_join_post_root_aggregate_data_plan(customers_path: &Path, orders_path: &Path) -> QueryPlan {
    let customer_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_name", DataType::Utf8, false),
    ]));
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_name", DataType::Utf8, false),
        Field::new("o_orderkey", DataType::Int32, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_name", DataType::Utf8, false),
        Field::new("c_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customers = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        customers_path,
        customer_schema,
        vec!["c_custkey".into(), "c_name".into()],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        orders_path,
        orders_schema,
        vec!["o_orderkey".into(), "o_custkey".into()],
    )));
    let join = builder.add_join(
        customers,
        orders,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                ColumnRef::input(0, "c_custkey", "c_custkey"),
                ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "c_name", "c_name"), "c_name"),
            JoinOutputColumn::right(ColumnRef::input(1, "o_orderkey", "o_orderkey"), "o_orderkey"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("c_name")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        aggregate_schema,
    );
    builder.finish_with_host_sink(aggregate)
}

pub(super) fn single_join_post_root_aggregate_sort_limit_data_plan(
    customers_path: &Path,
    orders_path: &Path,
) -> QueryPlan {
    let customer_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int32, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customers = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        customers_path,
        customer_schema,
        vec!["c_custkey".into()],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        orders_path,
        orders_schema,
        vec!["o_orderkey".into(), "o_custkey".into()],
    )));
    let join = builder.add_join(
        customers,
        orders,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                ColumnRef::input(0, "c_custkey", "c_custkey"),
                ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "c_custkey", "c_custkey"), "c_custkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "o_orderkey", "o_orderkey"), "o_orderkey"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let projection = builder.add_projection(
        join,
        vec![
            ProjectionOutput::new(Expr::column("c_custkey"), Field::new("c_custkey", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("o_orderkey"), Field::new("o_orderkey", DataType::Int32, true)),
        ],
    );
    let aggregate = builder.add_aggregate(
        projection,
        AggregateSpec::new(
            vec![Expr::column("c_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("c_count"), false, false),
            SortExpr::new(Expr::column("c_custkey"), true, false),
        ]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), aggregate_schema);
    builder.finish_with_host_sink(limit)
}

pub(super) fn simple_unary_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let unary = PipelineTaskId::new(1);
    let frame = ResidentResourceId::new(0);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![frame]),
            PipelineTask::new(unary, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, unary, PipelineEdgeKind::ResidentFrame)
                .with_resources(vec![frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![unary]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![unary]),
        ],
        vec![ResidentResource::new(frame, source, vec![unary], ResidentLifetime::BetweenTasks { producer: source })],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, unary),
    )
}

pub(super) fn independent_two_source_pipeline() -> CompiledPipeline {
    let first_source = PipelineTaskId::new(0);
    let second_source = PipelineTaskId::new(1);
    let first_consumer = PipelineTaskId::new(2);
    let second_consumer = PipelineTaskId::new(3);
    let first_frame = ResidentResourceId::new(0);
    let second_frame = ResidentResourceId::new(1);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(first_source, PipelineTaskKind::SourceRead).with_output_resources(vec![first_frame]),
            PipelineTask::new(second_source, PipelineTaskKind::SourceRead).with_output_resources(vec![second_frame]),
            PipelineTask::new(first_consumer, PipelineTaskKind::Unary).with_input_resources(vec![first_frame]),
            PipelineTask::new(second_consumer, PipelineTaskKind::Unary).with_input_resources(vec![second_frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), first_source, first_consumer, PipelineEdgeKind::ResidentFrame)
                .with_resources(vec![first_frame]),
            PipelineEdge::new(PipelineEdgeId::new(1), second_source, second_consumer, PipelineEdgeKind::ResidentFrame)
                .with_resources(vec![second_frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![first_source])
                .with_downstream_tasks(vec![first_consumer]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![second_source])
                .with_downstream_tasks(vec![second_consumer]),
            PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![first_consumer, second_consumer]),
        ],
        vec![
            ResidentResource::new(
                first_frame,
                first_source,
                vec![first_consumer],
                ResidentLifetime::BetweenTasks { producer: first_source },
            ),
            ResidentResource::new(
                second_frame,
                second_source,
                vec![second_consumer],
                ResidentLifetime::BetweenTasks { producer: second_source },
            ),
        ],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, second_consumer),
    )
}

pub(super) fn out_of_order_join_barrier_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let probe = PipelineTaskId::new(1);
    let build = PipelineTaskId::new(2);
    let materialize = PipelineTaskId::new(3);
    let sink = PipelineTaskId::new(4);
    let source_frame = ResidentResourceId::new(0);
    let join_output = ResidentResourceId::new(1);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![source_frame]),
            PipelineTask::new(probe, PipelineTaskKind::JoinProbe).with_input_resources(vec![source_frame]),
            PipelineTask::new(build, PipelineTaskKind::JoinBuild).with_input_resources(vec![source_frame]),
            PipelineTask::new(materialize, PipelineTaskKind::JoinMaterialize).with_output_resources(vec![join_output]),
            PipelineTask::new(sink, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![join_output]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, probe, PipelineEdgeKind::JoinProbeInput)
                .with_resources(vec![source_frame]),
            PipelineEdge::new(PipelineEdgeId::new(1), source, build, PipelineEdgeKind::JoinBuildInput)
                .with_resources(vec![source_frame]),
            PipelineEdge::new(PipelineEdgeId::new(2), build, probe, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(3), probe, materialize, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(4), materialize, sink, PipelineEdgeKind::SinkInput)
                .with_resources(vec![join_output]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![build, probe]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::JoinBuildBeforeProbe)
                .with_upstream_tasks(vec![build])
                .with_downstream_tasks(vec![probe]),
            PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::JoinMaterialization)
                .with_upstream_tasks(vec![probe])
                .with_downstream_tasks(vec![materialize]),
            PipelineBarrier::new(PipelineBarrierId::new(3), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![sink]),
        ],
        vec![
            ResidentResource::new(
                source_frame,
                source,
                vec![build, probe],
                ResidentLifetime::BetweenTasks { producer: source },
            ),
            ResidentResource::new(
                join_output,
                materialize,
                vec![sink],
                ResidentLifetime::BetweenTasks { producer: materialize },
            ),
        ],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink),
    )
}

pub(super) fn downstream_build_branch_after_completed_upstream_build_pipeline() -> CompiledPipeline {
    let first_build_source = PipelineTaskId::new(0);
    let first_build = PipelineTaskId::new(1);
    let first_probe_source = PipelineTaskId::new(2);
    let second_build_source = PipelineTaskId::new(3);
    let first_probe = PipelineTaskId::new(4);
    let second_build = PipelineTaskId::new(5);
    let first_materialize = PipelineTaskId::new(6);
    let second_probe = PipelineTaskId::new(7);
    let second_materialize = PipelineTaskId::new(8);
    let sink = PipelineTaskId::new(9);
    let first_build_frame = ResidentResourceId::new(0);
    let first_probe_frame = ResidentResourceId::new(1);
    let second_build_frame = ResidentResourceId::new(2);
    let first_join_output = ResidentResourceId::new(3);
    let second_join_output = ResidentResourceId::new(4);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(first_build_source, PipelineTaskKind::SourceRead)
                .with_output_resources(vec![first_build_frame]),
            PipelineTask::new(first_build, PipelineTaskKind::JoinBuild).with_input_resources(vec![first_build_frame]),
            PipelineTask::new(first_probe_source, PipelineTaskKind::SourceRead)
                .with_output_resources(vec![first_probe_frame]),
            PipelineTask::new(second_build_source, PipelineTaskKind::SourceRead)
                .with_output_resources(vec![second_build_frame]),
            PipelineTask::new(first_probe, PipelineTaskKind::JoinProbe).with_input_resources(vec![first_probe_frame]),
            PipelineTask::new(second_build, PipelineTaskKind::JoinBuild).with_input_resources(vec![second_build_frame]),
            PipelineTask::new(first_materialize, PipelineTaskKind::JoinMaterialize)
                .with_output_resources(vec![first_join_output]),
            PipelineTask::new(second_probe, PipelineTaskKind::JoinProbe).with_input_resources(vec![first_join_output]),
            PipelineTask::new(second_materialize, PipelineTaskKind::JoinMaterialize)
                .with_output_resources(vec![second_join_output]),
            PipelineTask::new(sink, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![second_join_output]),
        ],
        vec![
            PipelineEdge::new(
                PipelineEdgeId::new(0),
                first_build_source,
                first_build,
                PipelineEdgeKind::JoinBuildInput,
            )
            .with_resources(vec![first_build_frame]),
            PipelineEdge::new(
                PipelineEdgeId::new(1),
                first_probe_source,
                first_probe,
                PipelineEdgeKind::JoinProbeInput,
            )
            .with_resources(vec![first_probe_frame]),
            PipelineEdge::new(PipelineEdgeId::new(2), first_build, first_probe, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(3), first_probe, first_materialize, PipelineEdgeKind::Control),
            PipelineEdge::new(
                PipelineEdgeId::new(4),
                second_build_source,
                second_build,
                PipelineEdgeKind::JoinBuildInput,
            )
            .with_resources(vec![second_build_frame]),
            PipelineEdge::new(
                PipelineEdgeId::new(5),
                first_materialize,
                second_probe,
                PipelineEdgeKind::JoinProbeInput,
            )
            .with_resources(vec![first_join_output]),
            PipelineEdge::new(PipelineEdgeId::new(6), second_build, second_probe, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(7), second_probe, second_materialize, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(8), second_materialize, sink, PipelineEdgeKind::SinkInput)
                .with_resources(vec![second_join_output]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![first_build_source])
                .with_downstream_tasks(vec![first_build]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![first_probe_source])
                .with_downstream_tasks(vec![first_probe]),
            PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![second_build_source])
                .with_downstream_tasks(vec![second_build]),
            PipelineBarrier::new(PipelineBarrierId::new(3), PipelineBarrierKind::JoinBuildBeforeProbe)
                .with_upstream_tasks(vec![first_build])
                .with_downstream_tasks(vec![first_probe]),
            PipelineBarrier::new(PipelineBarrierId::new(4), PipelineBarrierKind::JoinMaterialization)
                .with_upstream_tasks(vec![first_probe])
                .with_downstream_tasks(vec![first_materialize]),
            PipelineBarrier::new(PipelineBarrierId::new(5), PipelineBarrierKind::JoinBuildBeforeProbe)
                .with_upstream_tasks(vec![second_build])
                .with_downstream_tasks(vec![second_probe]),
            PipelineBarrier::new(PipelineBarrierId::new(6), PipelineBarrierKind::JoinMaterialization)
                .with_upstream_tasks(vec![second_probe])
                .with_downstream_tasks(vec![second_materialize]),
            PipelineBarrier::new(PipelineBarrierId::new(7), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![sink]),
        ],
        vec![
            ResidentResource::new(
                first_build_frame,
                first_build_source,
                vec![first_build],
                ResidentLifetime::BetweenTasks { producer: first_build_source },
            ),
            ResidentResource::new(
                first_probe_frame,
                first_probe_source,
                vec![first_probe],
                ResidentLifetime::BetweenTasks { producer: first_probe_source },
            ),
            ResidentResource::new(
                second_build_frame,
                second_build_source,
                vec![second_build],
                ResidentLifetime::BetweenTasks { producer: second_build_source },
            ),
            ResidentResource::new(
                first_join_output,
                first_materialize,
                vec![second_probe],
                ResidentLifetime::BetweenTasks { producer: first_materialize },
            ),
            ResidentResource::new(
                second_join_output,
                second_materialize,
                vec![sink],
                ResidentLifetime::BetweenTasks { producer: second_materialize },
            ),
        ],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink),
    )
}

pub(super) fn shared_resident_fanout_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let first_consumer = PipelineTaskId::new(1);
    let second_consumer = PipelineTaskId::new(2);
    let frame = ResidentResourceId::new(0);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![frame]),
            PipelineTask::new(first_consumer, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
            PipelineTask::new(second_consumer, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, first_consumer, PipelineEdgeKind::SharedResidentProducer)
                .with_resources(vec![frame]),
            PipelineEdge::new(
                PipelineEdgeId::new(1),
                source,
                second_consumer,
                PipelineEdgeKind::SharedResidentProducer,
            )
            .with_resources(vec![frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![first_consumer, second_consumer]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![second_consumer]),
        ],
        vec![ResidentResource::new(
            frame,
            source,
            vec![first_consumer, second_consumer],
            ResidentLifetime::BetweenTasks { producer: source },
        )],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, second_consumer),
    )
}

pub(super) fn aggregate_sort_limit_plan() -> QueryPlan {
    let source_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-aggregate.parquet",
        source_schema,
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "v_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("v_count"), false, false)]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), aggregate_schema);
    builder.finish_with_host_sink(limit)
}

pub(super) fn adaptive_memory_history_key_probe_plan(
    value_type: DataType,
    aggregate_function: AggregateFunction,
    predicate_op: BinaryOp,
) -> QueryPlan {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("adaptive_key", DataType::Int32, false),
        Field::new("adaptive_value", value_type.clone(), true),
    ]));
    let aggregate_value_type = match aggregate_function {
        AggregateFunction::Count | AggregateFunction::NUnique => DataType::Int64,
        AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max | AggregateFunction::Avg => {
            value_type.clone()
        },
        AggregateFunction::StddevSamp => DataType::Float64,
    };
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("adaptive_key", DataType::Int32, false),
        Field::new("adaptive_metric", aggregate_value_type, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/adaptive-key-secret-literal.parquet",
        source_schema,
        vec!["adaptive_key".into(), "adaptive_value".into()],
    )));
    let filter = builder.add_filter(
        source,
        Expr::binary(Expr::column("adaptive_value"), predicate_op, literal_expr_for_type(&value_type)),
        Arc::new(Schema::new(vec![
            Field::new("adaptive_key", DataType::Int32, false),
            Field::new("adaptive_value", value_type.clone(), true),
        ])),
    );
    let aggregate = builder.add_aggregate(
        filter,
        AggregateSpec::new(
            vec![Expr::column("adaptive_key")],
            vec![AggregateExpr::new(aggregate_function, Expr::column("adaptive_value"), "adaptive_metric")],
        ),
        Arc::clone(&aggregate_schema),
    );
    builder.finish_with_host_sink(aggregate)
}

pub(super) fn aggregate_sort_limit_data_plan(path: &Path, source_schema: Arc<Schema>) -> QueryPlan {
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        path,
        source_schema,
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "v_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("k"), true, false)]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), aggregate_schema);
    builder.finish_with_host_sink(limit)
}

pub(super) fn root_join_shaped_pipeline_with_blocked_gate() -> CompiledPipeline {
    let left_source = PipelineTaskId::new(0);
    let right_source = PipelineTaskId::new(1);
    let build = PipelineTaskId::new(2);
    let probe = PipelineTaskId::new(3);
    let materialize = PipelineTaskId::new(4);
    let sink_task = PipelineTaskId::new(5);

    let left_frame = ResidentResourceId::new(0);
    let right_frame = ResidentResourceId::new(1);
    let join_output = ResidentResourceId::new(2);
    let blocked_gate = ReservationGateId::new(0);

    let tasks = vec![
        PipelineTask::new(left_source, PipelineTaskKind::SourceRead).with_output_resources(vec![left_frame]),
        PipelineTask::new(right_source, PipelineTaskKind::SourceRead).with_output_resources(vec![right_frame]),
        PipelineTask::new(build, PipelineTaskKind::JoinBuild)
            .with_input_resources(vec![right_frame])
            .with_gates(vec![blocked_gate]),
        PipelineTask::new(probe, PipelineTaskKind::JoinProbe).with_input_resources(vec![left_frame]),
        PipelineTask::new(materialize, PipelineTaskKind::JoinMaterialize).with_output_resources(vec![join_output]),
        PipelineTask::new(sink_task, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![join_output]),
    ];

    let edges = vec![
        PipelineEdge::new(PipelineEdgeId::new(0), right_source, build, PipelineEdgeKind::JoinBuildInput)
            .with_resources(vec![right_frame]),
        PipelineEdge::new(PipelineEdgeId::new(1), left_source, probe, PipelineEdgeKind::JoinProbeInput)
            .with_resources(vec![left_frame]),
        PipelineEdge::new(PipelineEdgeId::new(2), build, probe, PipelineEdgeKind::Control),
        PipelineEdge::new(PipelineEdgeId::new(3), probe, materialize, PipelineEdgeKind::Control),
        PipelineEdge::new(PipelineEdgeId::new(4), materialize, sink_task, PipelineEdgeKind::SinkInput)
            .with_resources(vec![join_output]),
    ];

    let barriers = vec![
        PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
            .with_upstream_tasks(vec![left_source])
            .with_downstream_tasks(vec![probe]),
        PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SourceChunkBoundary)
            .with_upstream_tasks(vec![right_source])
            .with_downstream_tasks(vec![build]),
        PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::JoinBuildBeforeProbe)
            .with_upstream_tasks(vec![build])
            .with_downstream_tasks(vec![probe]),
        PipelineBarrier::new(PipelineBarrierId::new(3), PipelineBarrierKind::JoinMaterialization)
            .with_upstream_tasks(vec![probe])
            .with_downstream_tasks(vec![materialize]),
        PipelineBarrier::new(PipelineBarrierId::new(4), PipelineBarrierKind::SinkBoundary)
            .with_upstream_tasks(vec![sink_task]),
    ];

    let resources = vec![
        ResidentResource::new(
            left_frame,
            left_source,
            vec![probe],
            ResidentLifetime::BetweenTasks { producer: left_source },
        ),
        ResidentResource::new(
            right_frame,
            right_source,
            vec![build],
            ResidentLifetime::BetweenTasks { producer: right_source },
        ),
        ResidentResource::new(
            join_output,
            materialize,
            vec![sink_task],
            ResidentLifetime::BetweenTasks { producer: materialize },
        ),
    ];

    let gates = vec![ReservationGate::new(
        blocked_gate,
        ReservationGateOwner::Task(build),
        ReservationKind::HashTable,
        ReservationRequirementScope::Node,
        "join_hash_table",
        "hash_join_pre_build_reservation_size_bytes",
        ContractStatus::BlockedByLowerLevel,
        ContractQuality::ExactPreflight,
    )];

    CompiledPipeline::new(
        tasks,
        edges,
        barriers,
        resources,
        gates,
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink_task),
    )
}
