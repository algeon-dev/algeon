pub(super) fn projection_root_over_full_sort_join_aggregate_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("right_id", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-root-full-sort-left.parquet",
        left_schema,
        vec!["left_id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-root-full-sort-right.parquet",
        right_schema,
        vec!["right_id".into()],
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
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    ).unwrap();
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("left_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    ).unwrap();
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)])).unwrap();
    let projection = builder.add_projection(
        sort,
        projection_outputs(vec![Expr::column("left_id"), Expr::column("amount_sum")], aggregate_schema),
    ).unwrap();
    builder.finish_with_host_sink(projection).unwrap()
}

pub(super) fn root_full_sort_single_keyed_join_grouped_aggregate_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d1", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![Field::new("d1", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-single-fact.parquet",
        fact_schema,
        vec!["k".into(), "d1".into(), "amount".into()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-single-dim.parquet",
        dim_schema,
        vec!["d1".into()],
    )));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d1", "d1"), ColumnRef::input(1, "d1", "d1"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    ).unwrap();
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    ).unwrap();
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)])).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

pub(super) fn root_full_sort_keyed_join_grouped_aggregate_with_orderless_window_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("sold_date_sk", DataType::Int32, false),
        Field::new("item_sk", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let date_schema = Arc::new(Schema::new(vec![Field::new("date_sk", DataType::Int32, false)]));
    let item_schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("class_id", DataType::Int32, false),
        Field::new("category_id", DataType::Int32, false),
    ]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let pre_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("pre_amount_sum", DataType::Int64, true),
    ]));
    let second_join_schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("class_id", DataType::Int32, false),
        Field::new("category_id", DataType::Int32, false),
        Field::new("pre_amount_sum", DataType::Int64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("class_id", DataType::Int32, false),
        Field::new("category_id", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
    ]));
    let projection_schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("class_id", DataType::Int32, false),
        Field::new("category_id", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
        Field::new("class_amount_sum", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-window-fact.parquet",
        fact_schema,
        vec!["sold_date_sk".into(), "item_sk".into(), "amount".into()],
    )));
    let date = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-window-date.parquet",
        date_schema,
        vec!["date_sk".into()],
    )));
    let item = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-window-item.parquet",
        item_schema,
        vec!["item_sk".into(), "class_id".into(), "category_id".into()],
    )));
    let first_join = builder.add_join(
        fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "sold_date_sk", "sold_date_sk"),
                ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        first_join_schema,
    ).unwrap();
    let pre_aggregate = builder.add_aggregate(
        first_join,
        AggregateSpec::new(
            vec![Expr::column("item_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "pre_amount_sum")],
        ),
        Arc::clone(&pre_aggregate_schema),
    ).unwrap();
    let second_join = builder.add_join(
        pre_aggregate,
        item,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "item_sk", "item_sk"), ColumnRef::input(1, "item_sk", "item_sk"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk"),
            JoinOutputColumn::right(ColumnRef::input(1, "class_id", "class_id"), "class_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "category_id", "category_id"), "category_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "pre_amount_sum", "pre_amount_sum"), "pre_amount_sum"),
        ])
        .with_build_side(JoinSide::Right),
        second_join_schema,
    ).unwrap();
    let aggregate = builder.add_aggregate(
        second_join,
        AggregateSpec::new(
            vec![Expr::column("item_sk"), Expr::column("class_id"), Expr::column("category_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("pre_amount_sum"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    ).unwrap();
    let window_input_sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("class_id"), false, false)])).unwrap();
    let window = builder.add_window(
        window_input_sort,
        WindowSpec::new(
            vec![Expr::column("class_id")],
            Vec::new(),
            vec![WindowFunction::aggregate(AggregateFunction::Sum, Expr::column("amount_sum"), Field::new("class_amount_sum", DataType::Int64, true))],
        )).unwrap();
    let projection = builder.add_projection(
        window,
        projection_outputs(
            vec![
                Expr::column("item_sk"),
                Expr::column("class_id"),
                Expr::column("category_id"),
                Expr::column("amount_sum"),
                Expr::column("class_amount_sum"),
            ],
            Arc::clone(&projection_schema),
        ),
    ).unwrap();
    let final_sort = builder.add_sort(
        projection,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("category_id"), false, true),
            SortExpr::new(Expr::column("class_id"), false, true),
            SortExpr::new(Expr::column("item_sk"), false, true),
            SortExpr::new(Expr::column("class_amount_sum"), false, true),
        ])).unwrap();
    builder.finish_with_host_sink(final_sort).unwrap()
}

pub(super) fn root_full_sort_keyless_join_grouped_aggregate_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d1", DataType::Int32, false),
        Field::new("d2", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let dim_one_schema = Arc::new(Schema::new(vec![Field::new("d1", DataType::Int32, false)]));
    let dim_two_schema = Arc::new(Schema::new(vec![Field::new("d2", DataType::Int32, false)]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d2", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let second_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-fact.parquet",
        fact_schema,
        vec!["k".into(), "d1".into(), "d2".into(), "amount".into()],
    )));
    let dim_one = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-dim-one.parquet",
        dim_one_schema,
        vec!["d1".into()],
    )));
    let dim_two = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-dim-two.parquet",
        dim_two_schema,
        vec!["d2".into()],
    )));
    let first_join = builder.add_join(
        fact,
        dim_one,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d1", "d1"), ColumnRef::input(1, "d1", "d1"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "d2", "d2"), "d2"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        first_join_schema,
    ).unwrap();
    let second_join = builder.add_join(
        first_join,
        dim_two,
        JoinSpec::new(JoinType::Cross, Vec::new())
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&second_join_schema),
    ).unwrap();
    let aggregate = builder.add_aggregate(
        second_join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    ).unwrap();
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)])).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

pub(super) fn root_full_sort_keyless_join_statistical_grouped_aggregate_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d1", DataType::Int32, false),
        Field::new("d2", DataType::Int32, false),
        Field::new("amount", DataType::Float64, true),
    ]));
    let dim_one_schema = Arc::new(Schema::new(vec![Field::new("d1", DataType::Int32, false)]));
    let dim_two_schema = Arc::new(Schema::new(vec![Field::new("d2", DataType::Int32, false)]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d2", DataType::Int32, false),
        Field::new("amount", DataType::Float64, true),
    ]));
    let second_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount", DataType::Float64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount_stddev", DataType::Float64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-statistical-fact.parquet",
        fact_schema,
        vec!["k".into(), "d1".into(), "d2".into(), "amount".into()],
    )));
    let dim_one = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-statistical-dim-one.parquet",
        dim_one_schema,
        vec!["d1".into()],
    )));
    let dim_two = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-statistical-dim-two.parquet",
        dim_two_schema,
        vec!["d2".into()],
    )));
    let first_join = builder.add_join(
        fact,
        dim_one,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d1", "d1"), ColumnRef::input(1, "d1", "d1"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "d2", "d2"), "d2"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        first_join_schema,
    ).unwrap();
    let second_join = builder.add_join(
        first_join,
        dim_two,
        JoinSpec::new(JoinType::Cross, Vec::new())
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&second_join_schema),
    ).unwrap();
    let aggregate = builder.add_aggregate(
        second_join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("amount"), "amount_stddev")],
        ),
        Arc::clone(&aggregate_schema),
    ).unwrap();
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_stddev"), true, false)])).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

pub(super) fn root_full_sort_repeated_projection_fanout_grouped_aggregate_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d1", DataType::Int32, false),
        Field::new("d2", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dim_id", DataType::Int32, false),
        Field::new("tag", DataType::Int32, false),
    ]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d2", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
        Field::new("tag1", DataType::Int32, false),
    ]));
    let second_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
        Field::new("tag1", DataType::Int32, false),
        Field::new("tag2", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-fanout-fact.parquet",
        fact_schema,
        vec!["k".into(), "d1".into(), "d2".into(), "amount".into()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-fanout-dim.parquet",
        dim_schema,
        vec!["dim_id".into(), "tag".into()],
    )));
    let shared_dim = builder.add_projection(
        dim,
        projection_outputs(
            vec![Expr::column("dim_id"), Expr::column("tag")],
            Arc::new(Schema::new(vec![
                Field::new("dim_id", DataType::Int32, false),
                Field::new("tag", DataType::Int32, false),
            ])),
        ),
    ).unwrap();
    let first_join = builder.add_join(
        fact,
        shared_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d1", "d1"), ColumnRef::input(1, "dim_id", "dim_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "d2", "d2"), "d2"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            JoinOutputColumn::right(ColumnRef::input(1, "tag", "tag"), "tag1"),
        ])
        .with_build_side(JoinSide::Right),
        first_join_schema,
    ).unwrap();
    let second_join = builder.add_join(
        first_join,
        shared_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d2", "d2"), ColumnRef::input(1, "dim_id", "dim_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            JoinOutputColumn::left(ColumnRef::input(0, "tag1", "tag1"), "tag1"),
            JoinOutputColumn::right(ColumnRef::input(1, "tag", "tag"), "tag2"),
        ])
        .with_build_side(JoinSide::Right),
        second_join_schema,
    ).unwrap();
    let aggregate = builder.add_aggregate(
        second_join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    ).unwrap();
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)])).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

pub(super) fn root_full_sort_keyless_scalar_projection_fanout_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d1", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("d1", DataType::Int32, false),
        Field::new("bucket", DataType::Int32, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let grouped_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
    ]));
    let scalar_schema = Arc::new(Schema::new(vec![Field::new("amount_avg", DataType::Float64, true)]));
    let root_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-keyless-scalar-fact.parquet",
        fact_schema,
        vec!["k".into(), "d1".into(), "amount".into()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-keyless-scalar-dim.parquet",
        dim_schema,
        vec!["d1".into(), "bucket".into()],
    )));
    let shared_dim = builder.add_projection(
        dim,
        projection_outputs(
            vec![Expr::column("d1"), Expr::column("bucket")],
            Arc::new(Schema::new(vec![
                Field::new("d1", DataType::Int32, false),
                Field::new("bucket", DataType::Int32, false),
            ])),
        ),
    ).unwrap();
    let grouped_join = builder.add_join(
        fact,
        shared_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d1", "d1"), ColumnRef::input(1, "d1", "d1"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    ).unwrap();
    let grouped = builder.add_aggregate(
        grouped_join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&grouped_schema),
    ).unwrap();
    let grouped_projection = builder.add_projection(
        grouped,
        projection_outputs(vec![Expr::column("k"), Expr::column("amount_sum")], grouped_schema),
    ).unwrap();
    let scalar_join = builder.add_join(
        fact,
        shared_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d1", "d1"), ColumnRef::input(1, "d1", "d1"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount")])
        .with_build_side(JoinSide::Right),
        Arc::new(Schema::new(vec![Field::new("amount", DataType::Int64, false)])),
    ).unwrap();
    let scalar = builder.add_aggregate(
        scalar_join,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "amount_avg")],
        ),
        Arc::clone(&scalar_schema),
    ).unwrap();
    let scalar_projection =
        builder.add_projection(scalar, projection_outputs(vec![Expr::column("amount_avg")], scalar_schema)).unwrap();
    let root_join = builder.add_join(
        grouped_projection,
        scalar_projection,
        JoinSpec::new(JoinType::Inner, Vec::new())
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "amount_sum", "amount_sum"),
                BinaryOp::Gt,
                Expr::input_column(1, "amount_avg", "amount_avg"),
            ))
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "amount_sum", "amount_sum"), "amount_sum"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&root_join_schema),
    ).unwrap();
    let sort = builder.add_sort(
        root_join,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)])).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

pub(super) fn root_full_sort_shared_aggregate_fanout_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d1", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![Field::new("d1", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
    ]));
    let self_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("left_amount_sum", DataType::Int64, true),
        Field::new("right_amount_sum", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-shared-aggregate-fact.parquet",
        fact_schema,
        vec!["k".into(), "d1".into(), "amount".into()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-shared-aggregate-dim.parquet",
        dim_schema,
        vec!["d1".into()],
    )));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d1", "d1"), ColumnRef::input(1, "d1", "d1"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    ).unwrap();
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    ).unwrap();
    let self_join = builder.add_join(
        aggregate,
        aggregate,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount_sum", "amount_sum"), "left_amount_sum"),
            JoinOutputColumn::right(ColumnRef::input(1, "amount_sum", "amount_sum"), "right_amount_sum"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&self_join_schema),
    ).unwrap();
    let sort = builder.add_sort(
        self_join,
        SortSpec::new(vec![SortExpr::new(Expr::column("left_amount_sum"), true, false)])).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}
