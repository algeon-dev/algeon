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
        vec![
            LocalParquetFile::new(left_path)
                .with_file_size_bytes(2048)
                .with_row_groups(vec![0, 1])
                .with_row_group_rows(vec![2, 2])
                .with_row_group_bytes(vec![1024, 1024]),
        ],
        Arc::clone(&left_schema),
        vec!["id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(right_path)
                .with_file_size_bytes(2048)
                .with_row_groups(vec![0, 1])
                .with_row_group_rows(vec![2, 1])
                .with_row_group_bytes(vec![1024, 1024]),
        ],
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

pub(super) fn shared_source_join_build_and_aggregate_data_plan(fact_path: &Path, dim_path: &Path) -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let joined_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(fact_path)
                .with_file_size_bytes(2048)
                .with_row_groups(vec![0, 1])
                .with_row_group_rows(vec![2, 2])
                .with_row_group_bytes(vec![1024, 1024]),
        ],
        Arc::clone(&fact_schema),
        vec!["id".into(), "amount".into()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(dim_path)
                .with_file_size_bytes(2048)
                .with_row_groups(vec![0, 1])
                .with_row_group_rows(vec![2, 1])
                .with_row_group_bytes(vec![1024, 1024]),
        ],
        Arc::clone(&dim_schema),
        vec!["id".into()],
    )));
    let joined = builder.add_join(
        dim,
        fact,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "dim_id"), ColumnRef::input(1, "id", "fact_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::right(ColumnRef::input(1, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        joined_schema,
    );
    let aggregate = builder.add_aggregate(
        fact,
        AggregateSpec::new(
            vec![Expr::column("id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        aggregate_schema,
    );
    let joined_with_aggregate = builder.add_join(
        joined,
        aggregate,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "joined_id"), ColumnRef::input(1, "id", "agg_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::right(ColumnRef::input(1, "sum_amount", "sum_amount"), "sum_amount"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(joined_with_aggregate)
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

pub(super) fn residual_one_shot_frame_stream_multi_join_plan(a_path: &Path, b_path: &Path, c_path: &Path) -> QueryPlan {
    let a_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
        Field::new("a_partition", DataType::Int32, false),
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
    let a_source = builder.add_source(SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new(a_path)
                    .with_row_group_rows(vec![2, 2])
                    .with_row_group_bytes(vec![1024, 1024])
                    .with_partition_values(vec![LiteralValue::Int32(0)]),
            ],
            Arc::clone(&a_schema),
            vec!["a_id".into(), "a_value".into()],
        )
        .with_file_column_count(2)
        .with_partition_columns(vec![PartitionColumn::new("a_partition", DataType::Int32, false)]),
    ));
    let a = builder.add_projection(
        a_source,
        vec![
            ProjectionOutput::new(Expr::column("a_id"), Field::new("a_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("a_value"), Field::new("a_value", DataType::Int32, false)),
        ],
    );
    let b = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(b_path).with_row_group_rows(vec![2]).with_row_group_bytes(vec![1024])],
        Arc::clone(&b_schema),
        vec!["b_id".into(), "b_value".into()],
    )));
    let c = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(c_path).with_row_group_rows(vec![50_000]).with_row_group_bytes(vec![1024])],
        Arc::clone(&c_schema),
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
        .with_residual_predicate(Expr::binary(
            Expr::binary(
                Expr::input_column(0, "a_value", "a_value"),
                BinaryOp::Lt,
                Expr::input_column(1, "c_value", "c_value"),
            ),
            BinaryOp::And,
            Expr::is_not_distinct_from(Expr::input_column(0, "a_id", "a_id"), Expr::input_column(1, "c_id", "c_id")),
        ))
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

pub(super) fn residual_one_shot_leaf_stream_join_plan(a_path: &Path, c_path: &Path) -> QueryPlan {
    let a_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
        Field::new("a_partition", DataType::Int32, false),
    ]));
    let c_schema = Arc::new(Schema::new(vec![
        Field::new("c_id", DataType::Int32, false),
        Field::new("c_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
        Field::new("c_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let a_source = builder.add_source(SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new(a_path)
                    .with_row_group_rows(vec![2, 2])
                    .with_row_group_bytes(vec![1024, 1024])
                    .with_partition_values(vec![LiteralValue::Int32(0)]),
            ],
            Arc::clone(&a_schema),
            vec!["a_id".into(), "a_value".into()],
        )
        .with_file_column_count(2)
        .with_partition_columns(vec![PartitionColumn::new("a_partition", DataType::Int32, false)]),
    ));
    let a = builder.add_projection(
        a_source,
        vec![
            ProjectionOutput::new(Expr::column("a_id"), Field::new("a_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("a_value"), Field::new("a_value", DataType::Int32, false)),
        ],
    );
    let c = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(c_path).with_row_group_rows(vec![50_000]).with_row_group_bytes(vec![1024])],
        Arc::clone(&c_schema),
        vec!["c_id".into(), "c_value".into()],
    )));
    let ac = builder.add_join(
        a,
        c,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "a_id", "a_id"), ColumnRef::input(1, "c_id", "c_id"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "a_value", "a_value"),
            BinaryOp::Lt,
            Expr::input_column(1, "c_value", "c_value"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_id", "a_id"), "a_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_value", "a_value"), "a_value"),
            JoinOutputColumn::right(ColumnRef::input(1, "c_value", "c_value"), "c_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(ac)
}

pub(super) fn residual_one_shot_leaf_stream_left_build_join_plan(a_path: &Path, c_path: &Path) -> QueryPlan {
    let a_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
        Field::new("a_partition", DataType::Int32, false),
    ]));
    let c_schema = Arc::new(Schema::new(vec![
        Field::new("c_id", DataType::Int32, false),
        Field::new("c_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
        Field::new("c_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let c = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(c_path).with_row_group_rows(vec![50_000]).with_row_group_bytes(vec![1024])],
        Arc::clone(&c_schema),
        vec!["c_id".into(), "c_value".into()],
    )));
    let a_source = builder.add_source(SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new(a_path)
                    .with_row_group_rows(vec![2, 2])
                    .with_row_group_bytes(vec![1024, 1024])
                    .with_partition_values(vec![LiteralValue::Int32(0)]),
            ],
            Arc::clone(&a_schema),
            vec!["a_id".into(), "a_value".into()],
        )
        .with_file_column_count(2)
        .with_partition_columns(vec![PartitionColumn::new("a_partition", DataType::Int32, false)]),
    ));
    let a = builder.add_projection(
        a_source,
        vec![
            ProjectionOutput::new(Expr::column("a_id"), Field::new("a_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("a_value"), Field::new("a_value", DataType::Int32, false)),
        ],
    );
    let ca = builder.add_join(
        c,
        a,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "c_id", "c_id"), ColumnRef::input(1, "a_id", "a_id"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(1, "a_value", "a_value"),
            BinaryOp::Lt,
            Expr::input_column(0, "c_value", "c_value"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "a_id", "a_id"), "a_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "a_value", "a_value"), "a_value"),
            JoinOutputColumn::left(ColumnRef::input(0, "c_value", "c_value"), "c_value"),
        ])
        .with_build_side(JoinSide::Left),
        output_schema,
    );
    builder.finish_with_host_sink(ca)
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
